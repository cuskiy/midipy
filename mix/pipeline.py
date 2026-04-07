"""Top-level render pipeline: track dispatch, master chain, FLAC output."""
from __future__ import annotations

import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace as dc_replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, butter, sosfilt

from config import PARALLEL_MIN_TRACKS
from schema import Note, ChannelData, PipelineConfig, ParseResult
from synth import SR, TIMBRES, PANNING
from synth.dsp import BoundedCache
from synth.gm import _CENTER_LOCK
from .spatial import apply_hrtf, early_reflections, binaural_enhance
from .master import fdn_reverb_ir, compress, reverb_darken, pre_comp_eq, post_comp_eq
from .track_render import render_track, _tail_estimate
from .routing import split_and_merge_tracks, track_channel

_SPATIAL_OVERRIDES: dict = dict(
    az_scale=1.15, er_wet=0.10, pinna=True, immersive_er=True,
    reverb_wet=0.065, reverb_rt60=1.8, reverb_predelay=0.018,
)

_IR_CACHE: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]] = {}
_HP_MASTER: Optional[np.ndarray] = None


def clear_caches() -> None:
    """Reset all per-module caches (except reverb IR)."""
    BoundedCache.clear_all()
    global _HP_MASTER
    _HP_MASTER = None
    from . import spatial as _sp
    _sp._PINNA_SOS = _sp._ER_HP = _sp._XOVER = _sp._XFEED_LP = _sp._CEILING_SOS = None
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


def _get_hp_master() -> np.ndarray:
    _ensure_hp_master()
    assert _HP_MASTER is not None
    return _HP_MASTER


# ── Master chain ─────────────────────────────────────────────────────

def _master_chain(
    mix_l: np.ndarray, mix_r: np.ndarray,
    rev_mono: np.ndarray, cfg: PipelineConfig,
    irl: np.ndarray, irr: np.ndarray, buf_len: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    hp = _get_hp_master()
    mix_l = sosfilt(hp, mix_l)
    mix_r = sosfilt(hp, mix_r)

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

    from .track_render import _get_hp
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


# ── Parallel track-render wrapper ────────────────────────────────────

def _render_track_worker(args: tuple) -> Tuple[int, str, Optional[np.ndarray], Optional[np.ndarray]]:
    """Top-level picklable wrapper for multiprocessing."""
    track_idx, inst_name, notes, pan_cc, chan_data, buf_len, track_gain, density_scale, cfg = args
    result = render_track(inst_name, notes, pan_cc, chan_data,
                          buf_len, track_gain, density_scale, cfg, track_idx)
    return (track_idx, inst_name,
            result.audio if result.audio is not None else None,
            result.reverb_bus if result.reverb_bus is not None else None)


# ── Track preparation ────────────────────────────────────────────────

def _prepare_tracks(
    notes: List[Note], channel_pans: Dict[int, int],
    region: Optional[Tuple[Optional[float], Optional[float]]],
) -> Tuple[list, int, float, Optional[float]]:
    sub_tracks = split_and_merge_tracks(notes, channel_pans)
    reg_start = region[0] if region and region[0] is not None else 0.0
    reg_end = region[1] if region and region[1] is not None else None

    if reg_start > 0 or reg_end is not None:
        clipped_tracks: list = []
        for nm, ns, pc in sub_tracks:
            kept: list = []
            for n in ns:
                if n.start + n.dur <= reg_start:
                    continue
                if reg_end is not None and n.start >= reg_end:
                    continue
                if reg_end is not None and n.start + n.dur > reg_end:
                    new_dur = max(reg_end - n.start, 0.05)
                    kept.append(n._replace(dur=new_dur))
                else:
                    kept.append(n)
            if kept:
                clipped_tracks.append((nm, kept, pc))
        sub_tracks = clipped_tracks

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

    sub_tracks = [(nm, sorted(ns, key=lambda n: n.start), pc)
                  for nm, ns, pc in sub_tracks]
    return sub_tracks, buf_len, reg_start, reg_end


# ── Mix tracks ───────────────────────────────────────────────────────

def _mix_tracks(
    sub_tracks: list, ch_data: Dict[int, ChannelData],
    cfg: PipelineConfig, buf_len: int, stems: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    n_sub = max(len(sub_tracks), 1)
    density_scale = 1.0 / math.sqrt(max(n_sub / 4.0, 1.0))

    mix_l, mix_r = np.zeros(buf_len), np.zeros(buf_len)
    rev_mono = np.zeros(buf_len)
    used_az: set = set()
    inst_az: Dict[str, float] = {}
    stem_data: list = []

    # Build work items
    work_items: list = []
    for track_idx, (inst_name, notes, pan_cc) in enumerate(sub_tracks):
        chan_data = ch_data.get(track_channel(notes), ChannelData())
        work_items.append((track_idx, inst_name, notes, pan_cc,
                           chan_data, buf_len, cfg.track_gain,
                           density_scale, cfg))

    # Render tracks (parallel when beneficial)
    results: List[Tuple[int, str, Optional[np.ndarray], Optional[np.ndarray]]] = []
    use_parallel = len(sub_tracks) >= PARALLEL_MIN_TRACKS
    if use_parallel:
        try:
            n_workers = min(len(sub_tracks), max(os.cpu_count() or 1, 1))
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_render_track_worker, w): w[0]
                           for w in work_items}
                for future in as_completed(futures):
                    results.append(future.result())
            results.sort(key=lambda r: r[0])
        except Exception:
            # Fallback to serial on any multiprocessing error
            use_parallel = False
            results = []

    if not use_parallel:
        for w in work_items:
            results.append(_render_track_worker(w))

    # Print track info and mix results
    for track_idx, inst_name, audio, rev_bus in results:
        notes = sub_tracks[track_idx][1]
        pan_cc = sub_tracks[track_idx][2]
        print(f"  [{track_idx+1}/{len(sub_tracks)}] {inst_name:<14s}{len(notes):>3d} notes",
              flush=True)
        if audio is None:
            continue
        if rev_bus is not None:
            rev_mono += rev_bus

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
        az = float(np.clip(az * cfg.az_scale, -65, 65))
        left, right = apply_hrtf(audio, az, pinna=cfg.pinna)
        mix_l += left
        mix_r += right
        if stems:
            stem_data.append((inst_name, left.copy(), right.copy()))

    return mix_l, mix_r, rev_mono, stem_data


# ── Finalize ─────────────────────────────────────────────────────────

def _finalize(
    mix_l: np.ndarray, mix_r: np.ndarray,
    rev_mono: np.ndarray, cfg: PipelineConfig,
    irl: np.ndarray, irr: np.ndarray, buf_len: int,
    out_file: str, stems: bool, stems_dir: Optional[str],
    stem_data: list, out_start: int, out_end: int, has_audio: bool,
) -> None:
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
        groups: dict = defaultdict(lambda: ([], []))
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


# ── Top-level render ─────────────────────────────────────────────────

def render(
    result: ParseResult, out_file: str,
    spatial: bool = False,
    stems: bool = False, stems_dir: Optional[str] = None,
    region: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> None:
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
