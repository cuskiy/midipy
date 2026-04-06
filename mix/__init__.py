"""Render pipeline.

Signal chain (per track):
    synth → HP → sympathetic → chorus/leslie → mod_vibrato
    → RMS normalise (DRY) → VOLUME scale → CC7 (post-norm) → peak cap
    → reverb send → HRTF pan → accumulate L/R

Master chain:
    HP 20 Hz → early reflections → reverb (FDN convolution)
    → bass_tight → mud_cut → compress → presence → air
    → (binaural enhance) → peak normalise → dither → FLAC

NOTE on sosfiltfilt: master EQ stages use scipy.sosfiltfilt (zero-phase,
forward+backward filtering).  This doubles the effective magnitude change
in dB compared to the parameter value.  All gain parameters below are
specified as *half* the intended effective value.
"""
import math
import os
import numpy as np
import soundfile as sf
from collections import defaultdict
from scipy.signal import fftconvolve, butter, sosfilt

from synth import SR, A4, TIMBRES, PROGRAM_MAP, PANNING
from synth.gm import VOLUME, HP_FREQ, _CENTER_LOCK, CHORUS_CAP, REVERB_SEND
from synth.timbre import biquad_high_shelf
from synth import synthesize, drum
from .spatial import apply_hrtf, early_reflections, binaural_enhance
from .master import (fdn_reverb_ir, compress, air_eq, reverb_darken,
                     mud_cut, bass_tight, presence_eq)
from .cc import (smooth_cc, smooth_cc_sidechain, interp_cc, make_pb_curve)
from .effects import (apply_sympathetic, apply_chorus, apply_mod_vibrato,
                      apply_leslie, _SYMPA_INSTRUMENTS)

# ── Pipeline defaults ────────────────────────────────────────────────
#
# Gain staging notes:
#   - target_rms 0.15 → 0.13, peak_limit 0.92 → 0.89.
#   - comp_thresh 0.28 → 0.30, ratio 2.0 → 1.8 (more dynamics).
#   - CC floor 0.10 → 0.02 (CC7=0 near-silent).
#   - Master EQ presence/air reduced (see master.py docstring).
#   - CC7 sidechain: up_ms 30 → 50 (smoother pump release transient).

PIPELINE = dict(
    target_rms=0.13, track_peak_cap=2.0,
    reverb_wet=0.055, reverb_rt60=1.6, reverb_room=1.0, reverb_predelay=0.012,
    comp_thresh=0.30, comp_ratio=1.8, comp_att_ms=40, comp_rel_ms=400,
    comp_sc_hp=80.0,
    peak_limit=0.89,
    az_scale=1.0, er_wet=0.035, pinna=False,
)

SPATIAL = dict(
    az_scale=1.15, er_wet=0.10, pinna=True, immersive_er=True,
    reverb_wet=0.065, reverb_rt60=1.8, reverb_predelay=0.018,
)

# CC volume/expression floor: minimum gain when CC=0.
# Low enough to be near-silent, high enough to avoid divide-by-zero.
_CC_FLOOR = 0.02
_CC_SCALE = 1.0 - _CC_FLOOR

_IR_CACHE = {}
_HP_MASTER = None
_HP_CACHE = {}
_BRIGHT_CACHE = {}


def clear_caches():
    """Free all module-level caches across the project.

    Call between renders in batch processing to control memory, or after
    changing SR / pipeline parameters that invalidate cached filters.
    """
    global _HP_MASTER
    _IR_CACHE.clear()
    _HP_MASTER = None
    _HP_CACHE.clear()
    _BRIGHT_CACHE.clear()
    # synth caches
    from synth import _KS_CAL
    from synth.timbre import _BP_CACHE
    from synth.additive import _SE_CACHE, _PEAK_CACHE
    from synth.ks import _AH_CACHE
    from synth.supersaw import _FILTER_CACHE
    from synth.inharmonic import _BP_CACHE as _INH_BP
    from .master import _EQ_CACHE
    for c in [_KS_CAL, _BP_CACHE, _SE_CACHE, _PEAK_CACHE,
              _AH_CACHE, _FILTER_CACHE, _INH_BP, _EQ_CACHE]:
        c.clear()


# ── Utilities ────────────────────────────────────────────────────────

def _get_ir(room, rt60):
    key = (round(room, 4), round(rt60, 4))
    if key not in _IR_CACHE:
        print("  generating reverb IR...", end="", flush=True)
        il, ir = fdn_reverb_ir(room_size=room, rt60=rt60)
        rv_hp = butter(2, 120.0, btype='high', fs=SR, output='sos')
        il, ir = sosfilt(rv_hp, il), sosfilt(rv_hp, ir)
        e = max(np.sqrt(np.sum(il**2)), np.sqrt(np.sum(ir**2)), 1e-10)
        _IR_CACHE[key] = (il / e, ir / e)
        print(" done")
    return _IR_CACHE[key]


def _ensure_hp_master():
    global _HP_MASTER
    if _HP_MASTER is None:
        _HP_MASTER = butter(3, 20.0, btype='high', fs=SR, output='sos')


def _get_hp(freq):
    key = int(freq)
    if key not in _HP_CACHE:
        _HP_CACHE[key] = butter(2, max(freq, 20.0), btype='high',
                                fs=SR, output='sos')
    return _HP_CACHE[key]


def _get_bright_shelf(gain_db: float):
    key = round(gain_db * 2) / 2
    if key not in _BRIGHT_CACHE:
        _BRIGHT_CACHE[key] = biquad_high_shelf(4000.0, key)
    return _BRIGHT_CACHE[key]


def _tail_estimate(tb, dur):
    if dur >= 0.2:
        return min(tb.rel * 1.2 + 0.15, 2.5)
    return min(tb.rel + 0.20, 1.0)


from .routing import split_and_merge_tracks, track_channel


# ── Main render ──────────────────────────────────────────────────────

def render(tracks: list, track_pans: list, out_file: str, spatial: bool = False,
           ch_data: dict = None, stems: bool = False, stems_dir: str = None,
           region: tuple = None) -> None:
    """Render MIDI tracks to FLAC.

    Args:
        stems: if True, export per-instrument FLAC files alongside the mix.
        region: (start_sec, end_sec) tuple; None means full render.
                Either element can be None for open-ended.
    """
    P = dict(PIPELINE)
    if spatial:
        P.update(SPATIAL)
    if ch_data is None:
        ch_data = {}

    irl, irr = _get_ir(P['reverb_room'], P['reverb_rt60'])
    _ensure_hp_master()

    sub_tracks = split_and_merge_tracks(tracks, track_pans)

    # Region filter: drop notes outside the requested region
    reg_start = 0.0
    reg_end = None
    if region is not None:
        reg_start = region[0] if region[0] is not None else 0.0
        reg_end = region[1] if region[1] is not None else None

    if reg_start > 0 or reg_end is not None:
        filtered = []
        for nm, notes, pan_cc in sub_tracks:
            kept = []
            for note in notes:
                st, nn, dur, vel, ch, prog = note
                note_end = st + dur
                if reg_end is not None and st >= reg_end:
                    continue
                if note_end <= reg_start:
                    continue
                kept.append(note)
            if kept:
                filtered.append((nm, kept, pan_cc))
        sub_tracks = filtered

    # Compute global buffer size
    buf_len = 0
    for nm, notes, _ in sub_tracks:
        tb = TIMBRES.get(nm, TIMBRES["default"])
        for st, note, dur, vel, ch, prog in notes:
            e = int((st + dur + _tail_estimate(tb, dur)) * SR) + SR
            if e > buf_len:
                buf_len = e
    buf_len += SR

    # If region, clamp buf_len
    if reg_end is not None:
        buf_len = min(buf_len, int(reg_end * SR) + SR * 2)

    # Density-aware target RMS
    n_sub = max(len(sub_tracks), 1)
    density_scale = 1.0 / math.sqrt(max(n_sub / 4.0, 1.0))
    target_rms = P['target_rms'] * density_scale

    mix_l, mix_r = np.zeros(buf_len), np.zeros(buf_len)
    rev_mono = np.zeros(buf_len)
    used_az = set()
    inst_az = {}          # instrument name → assigned azimuth
    az_sc = P['az_scale']
    use_pinna = P['pinna']

    # For stems export
    stem_data = [] if stems else None

    for ti, (tn, notes, pan_cc) in enumerate(sub_tracks):
        nn = len(notes)
        print(f"  [{ti+1}/{len(sub_tracks)}] {tn:<12s} {nn:>4d} notes",
              flush=True)

        ch_id = track_channel(notes)
        cd = ch_data.get(ch_id, {})
        pb_events = cd.get("pb", [])

        # ── CC curves ───────────────────────────────────────────────
        # CC7 (volume): asymmetric smoothing — fast duck, smooth release.
        # CC11 (expression): moderate smoothing (artistic pedal).
        # Both use a low floor (_CC_FLOOR) so CC=0 is near-silent.
        vol_raw = interp_cc(cd.get("vol", []), buf_len,
                            default=100.0 / 127.0)
        vol_curve = (smooth_cc_sidechain(
                         _CC_FLOOR + _CC_SCALE * np.sqrt(vol_raw),
                         down_ms=5.0, up_ms=50.0)
                     if vol_raw is not None else None)
        expr_raw = interp_cc(cd.get("expr", []), buf_len)
        expr_curve = (smooth_cc(
                          _CC_FLOOR + _CC_SCALE * np.sqrt(expr_raw),
                          tau_ms=50.0)
                      if expr_raw is not None else None)
        at_raw = interp_cc(cd.get("aftertouch", []), buf_len, default=0.0)
        at_curve = smooth_cc(at_raw) if at_raw is not None else None
        bright_curve = interp_cc(cd.get("brightness", []), buf_len,
                                 default=64.0 / 127.0)

        # ── Phase 1: render notes DRY (no CC7) ─────────────────────
        # CC11/AT are artistic expression applied per-note.
        # CC7 is channel fader — applied AFTER normalisation.
        buf = np.zeros(buf_len)
        drng = np.random.RandomState(42 + ti)

        for ni, (st, note, dur, vel, ch, prog) in enumerate(notes):
            if ch == 9:
                tone = drum(note, dur, vel, drng)
            else:
                nm = PROGRAM_MAP.get(prog, "default")
                tb = TIMBRES.get(nm, TIMBRES["default"])
                freq = A4 * 2 ** ((note - 69) / 12.0)
                pbc = make_pb_curve(pb_events, st, dur)
                tone = synthesize(freq, dur, vel, tb, nm, ni, pb_curve=pbc)
            s = int(st * SR)
            e = min(s + len(tone), buf_len)
            seg_len = e - s
            if seg_len <= 0:
                continue
            seg = tone[:seg_len].copy()

            # CC11 continuous (expression — artistic intent)
            if expr_curve is not None:
                seg *= expr_curve[s:e]
            # AT continuous
            if at_curve is not None:
                seg *= 1.0 + at_curve[s:e] * 0.3
            # CC74 onset-sampled brightness
            if bright_curve is not None and ch != 9 and seg_len > 64:
                b74 = bright_curve[min(s, buf_len - 1)]
                gain_db = (b74 - 0.504) / 0.504 * 4.5
                if abs(gain_db) > 0.3:
                    sos = _get_bright_shelf(gain_db)
                    seg = sosfilt(sos, seg)

            buf[s:e] += seg

        if np.max(np.abs(buf)) < 1e-10:
            continue  # empty track

        # ── Phase 2: per-track processing on full buffer ───────────
        if tn in _SYMPA_INSTRUMENTS:
            buf = apply_sympathetic(buf, notes, buf_len)

        hp_f = HP_FREQ.get(tn, 60.0)
        buf = sosfilt(_get_hp(hp_f), buf)

        # Chorus depth capped per instrument
        chorus_raw = cd.get("chorus", 0.0)
        chorus_cap = CHORUS_CAP.get(tn, 1.0)
        chorus_depth = min(chorus_raw, chorus_cap)
        if chorus_depth > 0.01 and tn != "drums":
            buf = apply_chorus(buf, chorus_depth)

        if tn == "organ":
            buf = apply_leslie(buf)

        mod_events = cd.get("mod", [])
        mod_curve = interp_cc(mod_events, buf_len, default=0.0)
        if mod_curve is not None and tn != "drums":
            buf = apply_mod_vibrato(buf, mod_curve)

        # ── Phase 3: RMS normalisation on DRY signal ───────────────
        # Compute RMS BEFORE CC7 so sidechain doesn't dilute it.
        active = np.abs(buf) > 1e-7
        if np.any(active):
            first_a = np.argmax(active)
            last_a = buf_len - np.argmax(active[::-1])
            rms = np.sqrt(np.mean(buf[first_a:last_a] ** 2))
        else:
            rms = np.sqrt(np.mean(buf * buf))
        if rms > 1e-6:
            buf *= target_rms / rms
        vol = VOLUME.get(tn, 1.0)
        if vol != 1.0:
            buf *= vol

        # ── Phase 4: apply CC7 AFTER normalisation ─────────────────
        if vol_curve is not None:
            buf *= vol_curve

        pk = np.max(np.abs(buf))
        if pk > P['track_peak_cap']:
            buf *= P['track_peak_cap'] / pk

        # Per-instrument reverb send scale
        rv_depth = cd.get("reverb", 1.0) * REVERB_SEND.get(tn, 1.0)
        rev_mono += buf * rv_depth

        # Panning — same instrument shares position
        if pan_cc is not None and pan_cc != 64:
            az = (pan_cc - 64) / 64.0 * 55.0
        elif tn == "drums":
            az = 0.0
        elif tn in _CENTER_LOCK:
            az = 0.0
        elif tn in inst_az:
            az = inst_az[tn]
        else:
            az = PANNING.get(tn, 0)
            if az in used_az:
                for off in [8, -8, 16, -16, 24, -24]:
                    c = az + off
                    if abs(c) <= 55 and c not in used_az:
                        az = c
                        break
            used_az.add(az)
            inst_az[tn] = az
        az = np.clip(az * az_sc, -65, 65)
        l, r = apply_hrtf(buf, az, pinna=use_pinna)
        mix_l += l
        mix_r += r

        # Stems collection (save HRTF stereo)
        if stems:
            stem_data.append((tn, l.copy(), r.copy()))

    # ── Master chain ─────────────────────────────────────────────────
    if sub_tracks:
        mix_l = sosfilt(_HP_MASTER, mix_l)
        mix_r = sosfilt(_HP_MASTER, mix_r)

        er_w = P['er_wet']
        if er_w > 0:
            erl, err = early_reflections(
                mix_l, mix_r,
                immersive=P.get('immersive_er', False),
                pinna=use_pinna)
            mix_l += erl * er_w
            mix_r += err * er_w

        # Reverb
        wet = P['reverb_wet']
        dry = 1.0 - wet
        mix_l *= dry
        mix_r *= dry
        pd = int(P['reverb_predelay'] * SR)
        use_irl = np.concatenate([np.zeros(pd), irl]) if pd > 0 else irl
        use_irr = np.concatenate([np.zeros(pd), irr]) if pd > 0 else irr
        rev_hp = _get_hp(120.0)
        rev_filt = reverb_darken(sosfilt(rev_hp, rev_mono))
        rev_l = fftconvolve(rev_filt, use_irl, mode='full')[:buf_len] * wet
        rev_r = fftconvolve(rev_filt, use_irr, mode='full')[:buf_len] * wet
        if P.get('immersive_er'):
            offset = int(0.002 * SR)
            rev_r = np.roll(rev_r, offset)
            rev_r[:offset] = 0
        mix_l += rev_l
        mix_r += rev_r

        # Master EQ + compress
        # Corrective cuts BEFORE compression, sweetening boosts AFTER.
        # sosfiltfilt doubles effective dB — see module docstring.
        mix_l, mix_r = bass_tight(mix_l), bass_tight(mix_r)
        mix_l, mix_r = mud_cut(mix_l), mud_cut(mix_r)
        mix_l, mix_r = compress(mix_l, mix_r, P['comp_thresh'],
                                P['comp_ratio'], P['comp_att_ms'],
                                P['comp_rel_ms'], knee_db=3.0,
                                sc_hp=P.get('comp_sc_hp', 0))
        mix_l, mix_r = presence_eq(mix_l), presence_eq(mix_r)
        mix_l, mix_r = air_eq(mix_l), air_eq(mix_r)
        if P.get('immersive_er'):
            mix_l, mix_r = binaural_enhance(mix_l, mix_r)
        pk = max(np.max(np.abs(mix_l)), np.max(np.abs(mix_r)), 1e-10)
        mix_l *= P['peak_limit'] / pk
        mix_r *= P['peak_limit'] / pk

    # Region crop for output
    out_start = int(reg_start * SR) if reg_start > 0 else 0
    out_end = int(reg_end * SR) if reg_end is not None else buf_len
    out_end = min(out_end, buf_len)
    ml_out = mix_l[out_start:out_end]
    mr_out = mix_r[out_start:out_end]

    # Write mix
    stereo = np.column_stack((ml_out, mr_out))
    lsb = 1.0 / (2**23)
    dither = (np.random.default_rng(42).random(stereo.shape)
              - np.random.default_rng(137).random(stereo.shape)) * lsb
    sf.write(out_file, np.clip(stereo + dither, -1, 1),
             SR, format="FLAC", subtype="PCM_24")
    dur_out = (out_end - out_start) / SR
    print(f"  {out_file}  ({dur_out:.1f}s, {len(sub_tracks)} sub-tracks)")

    # ── Stems export ─────────────────────────────────────────────────
    if stems and stem_data:
        stem_out = stems_dir or (out_file.rsplit(".", 1)[0] + "_stems")
        os.makedirs(stem_out, exist_ok=True)
        stem_groups = defaultdict(lambda: ([], []))
        for tn, sl, sr_ in stem_data:
            stem_groups[tn][0].append(sl)
            stem_groups[tn][1].append(sr_)
        # Apply the same peak-limit gain to all stems so their relative
        # levels are preserved and they approximately sum to the mix.
        master_gain = P['peak_limit'] / pk if pk > 1e-10 else 1.0
        lsb = 1.0 / (2**23)
        for tn, (ls, rs) in stem_groups.items():
            cl = sum(ls)[out_start:out_end] * master_gain
            cr = sum(rs)[out_start:out_end] * master_gain
            # Safety: normalize if a stem still exceeds ±1 after gain
            stem_pk = max(np.max(np.abs(cl)), np.max(np.abs(cr)), 1e-10)
            if stem_pk > 1.0:
                cl *= 0.99 / stem_pk
                cr *= 0.99 / stem_pk
            stem_stereo = np.column_stack((cl, cr))
            stem_dither = (np.random.default_rng(hash(tn) & 0x7FFFFFFF).random(stem_stereo.shape)
                           - np.random.default_rng((hash(tn)+1) & 0x7FFFFFFF).random(stem_stereo.shape)) * lsb
            stem_path = os.path.join(stem_out, f"{tn}.flac")
            sf.write(stem_path, np.clip(stem_stereo + stem_dither, -1, 1),
                     SR, format="FLAC", subtype="PCM_24")
        print(f"  stems → {stem_out}/ ({len(stem_groups)} instruments)")
