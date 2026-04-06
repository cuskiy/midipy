"""Render pipeline: per-track synthesis → mix → master → FLAC."""
import math, os
from dataclasses import replace as dc_replace
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, butter, sosfilt

from schema import Note, ChannelData, PipelineConfig, TrackResult, ParseResult
from synth import SR, A4, TIMBRES, PANNING
from synth.gm import (VOLUME, HP_FREQ, _CENTER_LOCK, CHORUS_CAP, REVERB_SEND,
                       REF_RMS, SYMPA_INSTRUMENTS)
from synth.dsp import BoundedCache
from synth.voice import VoiceManager
from .spatial import apply_hrtf, early_reflections, binaural_enhance
from .master import fdn_reverb_ir, compress, reverb_darken, pre_comp_eq, post_comp_eq
from .cc import smooth_cc, smooth_cc_sidechain, interp_cc, make_pb_curve
from .dsp_module import (FilterModule, BrightnessModule, ChorusModule,
                         LeslieModule, ModVibratoModule, DspChain)
from .routing import split_and_merge_tracks, track_channel

BLOCK_SIZE = 64

_SPATIAL_OVERRIDES = dict(
    az_scale=1.15, er_wet=0.10, pinna=True, immersive_er=True,
    reverb_wet=0.065, reverb_rt60=1.8, reverb_predelay=0.018,
)

_CC_FLOOR = 0.02
_CC_SCALE = 0.98
_IR_CACHE = BoundedCache(4)
_HP_MASTER = None
_HP_CACHE = BoundedCache(32)
_SYMPA_GAIN = 0.0004
_SYMPA_MAX_H = 10
_SYMPA_TOLERANCE = 0.012
_SYMPA_MAX_NOTES = 40


def clear_caches() -> None:
    BoundedCache.clear_all()  # clears _IR_CACHE, _HP_CACHE, _EQ_CACHE, _HRTF_CACHE, etc.
    global _HP_MASTER
    _HP_MASTER = None
    # Reset non-BoundedCache spatial singletons
    from . import spatial as _sp
    _sp._PINNA_SOS = _sp._ER_HP = _sp._XOVER = _sp._XFEED_LP = _sp._CEILING_SOS = None
    # Reset cached master EQ sos matrices
    import mix.master as _m
    _m._PRE_COMP_EQ = _m._POST_COMP_EQ = None


def _get_ir(room: float, rt60: float) -> Tuple[np.ndarray, np.ndarray]:
    key = (round(room, 2), round(rt60, 2))
    if key not in _IR_CACHE:
        print("  generating reverb IR...", end=" ", flush=True)
        _IR_CACHE[key] = fdn_reverb_ir(room_size=room, rt60=rt60)
        print("done")
    return _IR_CACHE[key]


def _ensure_hp_master() -> None:
    global _HP_MASTER
    if _HP_MASTER is None:
        _HP_MASTER = butter(3, 20.0, btype='high', fs=SR, output='sos')


def _get_hp(freq: float) -> np.ndarray:
    key = int(freq)
    if key not in _HP_CACHE:
        _HP_CACHE[key] = butter(2, max(freq, 15), btype='high', fs=SR, output='sos')
    return _HP_CACHE[key]


def _tail_estimate(timbre, dur: float) -> float:
    return min(getattr(timbre, 'rel', 0.3) * 2.5 + getattr(timbre, 'd2', 1.0) * 0.5, 6.0)


# ── Sympathetic resonance (piano-family) ─────────────────────────────

def _apply_sympathetic(buf: np.ndarray, notes: List[Note],
                       buf_len: int) -> np.ndarray:
    piano_notes = [(st, midi_note, dur, vel)
                   for st, midi_note, dur, vel, ch, prog in notes if ch != 9]
    if len(piano_notes) < 2:
        return buf
    if len(piano_notes) > _SYMPA_MAX_NOTES:
        piano_notes = sorted(piano_notes, key=lambda x: -x[0])[:_SYMPA_MAX_NOTES]
    ndata = []
    for st, midi_note, dur, vel in piano_notes:
        f0 = A4 * 2 ** ((midi_note - 69) / 12.0)
        s, e = int(st * SR), min(int((st + dur + 0.8) * SR), buf_len)
        harmonics = [(k, f0 * k) for k in range(1, _SYMPA_MAX_H + 1) if f0 * k <= 8000]
        ndata.append((s, e, vel, harmonics))
    res = np.zeros(buf_len)
    for i in range(len(ndata)):
        si, ei, vi, hi = ndata[i]
        for j in range(i + 1, len(ndata)):
            sj, ej, vj, hj = ndata[j]
            ov_s, ov_e = max(si, sj), min(ei, ej)
            if ov_s >= ov_e:
                continue
            n_samp = ov_e - ov_s
            t = np.arange(n_samp, dtype=np.float64) / SR
            fade = np.clip(t / 0.008, 0, 1)
            decay = np.exp(-4.0 * t)
            for ki, fi in hi:
                for kj, fj in hj:
                    if abs(fi - fj) / min(fi, fj) < _SYMPA_TOLERANCE:
                        a = _SYMPA_GAIN * min(vi, vj) / (ki * kj)
                        res[ov_s:ov_e] += a * fade * decay * np.sin(
                            2 * np.pi * (fi + fj) * 0.5 * t + (ki + kj) * 0.5)
    return buf + res


# ── DSP chain builder ────────────────────────────────────────────────

def _build_dsp_chain(inst_name: str, chan_data: ChannelData,
                     buf_len: int) -> DspChain:
    modules = []
    modules.append(FilterModule(_get_hp(HP_FREQ.get(inst_name, 60.0))))
    if inst_name != "drums":
        bright_curve = interp_cc(chan_data.brightness, buf_len, default=64.0 / 127.0)
        if np.max(np.abs(bright_curve - 0.504)) > 0.02:
            modules.append(BrightnessModule(bright_curve, buf_len))
        chorus_curve = interp_cc(chan_data.chorus, buf_len, default=0.0)
        cap = CHORUS_CAP.get(inst_name, 1.0)
        if cap > 0 and np.max(chorus_curve) > 0.01:
            modules.append(ChorusModule(np.minimum(chorus_curve, cap), buf_len))
        if inst_name == "organ":
            modules.append(LeslieModule())
        mod_curve = interp_cc(chan_data.mod, buf_len, default=0.0)
        if np.max(mod_curve) > 0.01:
            modules.append(ModVibratoModule(mod_curve, buf_len))
    return DspChain(modules)


# ── Track render ─────────────────────────────────────────────────────

def _render_track(inst_name: str, notes: List[Note], pan_cc: Optional[int],
                  chan_data: ChannelData, buf_len: int,
                  track_gain: float, density_scale: float,
                  cfg: PipelineConfig,
                  track_idx: int = 0) -> TrackResult:
    """Block-based track render with Voice management and DspChain."""
    # CC curves (default= ensures interp_cc returns array, not None)
    vol_curve = smooth_cc_sidechain(
        _CC_FLOOR + _CC_SCALE * np.sqrt(
            interp_cc(chan_data.vol, buf_len, default=100.0 / 127.0)),
        down_ms=5.0, up_ms=50.0)
    expr_curve = smooth_cc(
        _CC_FLOOR + _CC_SCALE * np.sqrt(
            interp_cc(chan_data.expr, buf_len, default=1.0)),
        tau_ms=50.0)
    at_curve = smooth_cc(interp_cc(chan_data.aftertouch, buf_len, default=0.0))
    rev_curve = smooth_cc(
        interp_cc(chan_data.reverb, buf_len, default=40.0 / 127.0), tau_ms=50.0)

    voice_mgr = VoiceManager()
    chain = _build_dsp_chain(inst_name, chan_data, buf_len)
    buf = np.zeros(buf_len)
    rev_buf = np.zeros(buf_len)
    note_idx = 0

    for block_start in range(0, buf_len, BLOCK_SIZE):
        bs = min(BLOCK_SIZE, buf_len - block_start)
        block_end_sample = block_start + bs
        sl = slice(block_start, block_start + bs)

        while note_idx < len(notes):
            note = notes[note_idx]
            sample_pos = int(note.start * SR)
            if sample_pos >= block_end_sample:
                break
            note_idx += 1
            pbc = make_pb_curve(chan_data.pb, note.start, note.dur)
            voice_mgr.schedule_note(note, sample_pos, inst_name=inst_name,
                                    pb_curve=pbc)

        block = np.zeros(bs)
        voice_mgr.render_block(block, block_start, bs)

        # Expression + aftertouch
        block *= expr_curve[sl]
        block *= 1.0 + at_curve[sl] * 0.3

        # Reverb send from dry signal (pre-effects)
        rev_buf[sl] = block * rev_curve[sl] * REVERB_SEND.get(inst_name, 1.0)

        # DSP chain (applied after reverb send capture)
        buf[sl] = chain.process(block)

    if np.max(np.abs(buf)) < 1e-10:
        return TrackResult(None, None)

    if inst_name in SYMPA_INSTRUMENTS:
        buf = _apply_sympathetic(buf, notes, buf_len)

    # Fixed-gain scaling
    ref_rms = REF_RMS.get(inst_name, 0.10)
    gain = (track_gain * density_scale / ref_rms) * VOLUME.get(inst_name, 1.0)
    buf *= gain
    rev_buf *= gain

    buf *= vol_curve
    rev_buf *= vol_curve

    pk = np.max(np.abs(buf))
    if pk > cfg.track_peak_cap:
        scale = cfg.track_peak_cap / pk
        buf *= scale
        rev_buf *= scale

    return TrackResult(buf, rev_buf)


# ── Master chain ─────────────────────────────────────────────────────

def _master_chain(mix_l, mix_r, rev_mono, cfg, irl, irr, buf_len):
    mix_l = sosfilt(_HP_MASTER, mix_l)
    mix_r = sosfilt(_HP_MASTER, mix_r)

    if cfg.er_wet > 0:
        erl, err = early_reflections(mix_l, mix_r,
                                     immersive=cfg.immersive_er, pinna=cfg.pinna)
        mix_l += erl * cfg.er_wet
        mix_r += err * cfg.er_wet

    wet = cfg.reverb_wet
    mix_l *= (1.0 - wet)
    mix_r *= (1.0 - wet)
    pd = int(cfg.reverb_predelay * SR)
    use_irl = np.concatenate([np.zeros(pd), irl]) if pd > 0 else irl
    use_irr = np.concatenate([np.zeros(pd), irr]) if pd > 0 else irr
    rev_filt = reverb_darken(sosfilt(_get_hp(120.0), rev_mono))
    mix_l += fftconvolve(rev_filt, use_irl, mode='full')[:buf_len] * wet
    mix_r += fftconvolve(rev_filt, use_irr, mode='full')[:buf_len] * wet
    if cfg.immersive_er:
        offset = int(0.002 * SR)
        mix_r = np.roll(mix_r, offset)
        mix_r[:offset] = 0

    mix_l, mix_r = pre_comp_eq(mix_l), pre_comp_eq(mix_r)
    mix_l, mix_r = compress(mix_l, mix_r, cfg.comp_thresh, cfg.comp_ratio,
                            cfg.comp_att_ms, cfg.comp_rel_ms,
                            knee_db=3.0, sc_hp=cfg.comp_sc_hp)
    mix_l, mix_r = post_comp_eq(mix_l), post_comp_eq(mix_r)
    if cfg.immersive_er:
        mix_l, mix_r = binaural_enhance(mix_l, mix_r)
    pk = max(np.max(np.abs(mix_l)), np.max(np.abs(mix_r)), 1e-10)
    mix_l *= cfg.peak_limit / pk
    mix_r *= cfg.peak_limit / pk
    return mix_l, mix_r, pk


# ── Top-level render ─────────────────────────────────────────────────

def _prepare_tracks(notes, channel_pans, region):
    sub_tracks = split_and_merge_tracks(notes, channel_pans)
    reg_start = region[0] if region and region[0] is not None else 0.0
    reg_end = region[1] if region and region[1] is not None else None

    if reg_start > 0 or reg_end is not None:
        sub_tracks = [(nm, [n for n in ns
                            if n.start + n.dur > reg_start
                            and (reg_end is None or n.start < reg_end)], pc)
                      for nm, ns, pc in sub_tracks]
        sub_tracks = [(nm, ns, pc) for nm, ns, pc in sub_tracks if ns]

    buf_len = SR * 10
    for nm, ns, _ in sub_tracks:
        tb = TIMBRES.get(nm) or TIMBRES["default"]
        for note in ns:
            end = int((note.start + note.dur + _tail_estimate(tb, note.dur)) * SR) + SR
            if end > buf_len:
                buf_len = end
    buf_len += SR
    if reg_end is not None:
        buf_len = min(buf_len, int(reg_end * SR) + SR * 2)

    # Sort notes once per track
    sub_tracks = [(nm, sorted(ns, key=lambda n: n.start), pc)
                  for nm, ns, pc in sub_tracks]
    return sub_tracks, buf_len, reg_start, reg_end


def _mix_tracks(sub_tracks, ch_data, cfg, buf_len, stems=False):
    n_sub = max(len(sub_tracks), 1)
    density_scale = 1.0 / math.sqrt(max(n_sub / 4.0, 1.0))

    mix_l, mix_r = np.zeros(buf_len), np.zeros(buf_len)
    rev_mono = np.zeros(buf_len)
    used_az, inst_az = set(), {}
    stem_data = []

    for track_idx, (inst_name, notes, pan_cc) in enumerate(sub_tracks):
        print(f"  [{track_idx+1}/{len(sub_tracks)}] {inst_name:<14s}{len(notes):>3d} notes",
              flush=True)
        chan_data = ch_data.get(track_channel(notes), ChannelData())

        result = _render_track(inst_name, notes, pan_cc, chan_data,
                               buf_len, cfg.track_gain, density_scale, cfg, track_idx)
        if result.audio is None:
            continue
        if result.reverb_bus is not None:
            rev_mono += result.reverb_bus

        # Panning
        if pan_cc is not None and pan_cc not in (0, 64):
            az = (pan_cc - 64) / 64.0 * 55.0
        elif inst_name == "drums" or inst_name in _CENTER_LOCK:
            az = 0.0
        elif inst_name in inst_az:
            az = inst_az[inst_name]
        else:
            az = PANNING.get(inst_name, 0)
            if az in used_az:
                for off in [8, -8, 16, -16, 24, -24]:
                    cand = az + off
                    if abs(cand) <= 55 and cand not in used_az:
                        az = cand
                        break
            used_az.add(az)
            inst_az[inst_name] = az
        az = np.clip(az * cfg.az_scale, -65, 65)
        left, right = apply_hrtf(result.audio, az, pinna=cfg.pinna)
        mix_l += left
        mix_r += right
        if stems:
            stem_data.append((inst_name, left.copy(), right.copy()))

    return mix_l, mix_r, rev_mono, stem_data


def _finalize(mix_l, mix_r, rev_mono, cfg, irl, irr, buf_len,
              out_file, stems, stems_dir, stem_data,
              out_start, out_end, has_audio):
    if has_audio:
        mix_l, mix_r, pk = _master_chain(mix_l, mix_r, rev_mono, cfg,
                                         irl, irr, buf_len)
    else:
        pk = 1e-10
    sl = slice(out_start, out_end)
    stereo = np.column_stack((mix_l[sl], mix_r[sl]))
    lsb = 1.0 / (2**23)
    dither = (np.random.default_rng(42).random(stereo.shape)
              - np.random.default_rng(137).random(stereo.shape)) * lsb
    sf.write(out_file, np.clip(stereo + dither, -1, 1), SR,
             format="FLAC", subtype="PCM_24")
    print(f"  {out_file}  ({(out_end - out_start) / SR:.1f}s)")

    if stems and stem_data:
        stem_out = stems_dir or (out_file.rsplit(".", 1)[0] + "_stems")
        os.makedirs(stem_out, exist_ok=True)
        groups = defaultdict(lambda: ([], []))
        for nm, sl_, sr_ in stem_data:
            groups[nm][0].append(sl_)
            groups[nm][1].append(sr_)
        master_gain = cfg.peak_limit / pk if pk > 1e-10 else 1.0
        for nm, (ls, rs) in groups.items():
            cl = sum(ls)[out_start:out_end] * master_gain
            cr = sum(rs)[out_start:out_end] * master_gain
            spk = max(np.max(np.abs(cl)), np.max(np.abs(cr)), 1e-10)
            if spk > 1.0:
                cl *= 0.99 / spk
                cr *= 0.99 / spk
            sd = np.column_stack((cl, cr))
            rng_seed = hash(nm) & 0x7FFFFFFF
            sd_dither = (np.random.default_rng(rng_seed).random(sd.shape)
                         - np.random.default_rng(rng_seed + 1).random(sd.shape)) * lsb
            sf.write(os.path.join(stem_out, f"{nm}.flac"),
                     np.clip(sd + sd_dither, -1, 1), SR,
                     format="FLAC", subtype="PCM_24")
        print(f"  stems → {stem_out}/ ({len(groups)} instruments)")


def render(result: ParseResult, out_file: str,
           spatial: bool = False,
           stems: bool = False, stems_dir: Optional[str] = None,
           region: Optional[Tuple[float, float]] = None) -> None:
    """Render MIDI to 24-bit FLAC."""
    cfg = dc_replace(PipelineConfig(), **_SPATIAL_OVERRIDES) if spatial else PipelineConfig()
    irl, irr = _get_ir(cfg.reverb_room, cfg.reverb_rt60)
    _ensure_hp_master()

    sub_tracks, buf_len, reg_start, reg_end = _prepare_tracks(
        result.notes, result.channel_pans, region)
    mix_l, mix_r, rev_mono, stem_data = _mix_tracks(
        sub_tracks, result.ch_data, cfg, buf_len, stems=stems)

    out_start = int(reg_start * SR) if reg_start > 0 else 0
    out_end = min(int(reg_end * SR) if reg_end is not None else buf_len, buf_len)
    has_audio = len(sub_tracks) > 0 and max(
        np.max(np.abs(mix_l)), np.max(np.abs(mix_r))) > 1e-10

    _finalize(mix_l, mix_r, rev_mono, cfg, irl, irr, buf_len,
              out_file, stems, stems_dir, stem_data,
              out_start, out_end, has_audio)
