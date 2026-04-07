"""Per-track rendering: DSP chain builder, note synthesis, sympathetic resonance."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter

from config import CC_FLOOR, CC_SCALE, SYMPA_GAIN, SYMPA_MAX_H, SYMPA_TOLERANCE, SYMPA_MAX_NOTES
from schema import Note, ChannelData, PipelineConfig, TrackResult
from synth import SR, A4, TIMBRES, synthesize, drum
from synth.dsp import BoundedCache, biquad_peak
from synth.gm import VOLUME, HP_FREQ, CHORUS_CAP, REVERB_SEND, SYMPA_INSTRUMENTS
from synth.timbre import Timbre
from .cc import smooth_cc, smooth_cc_sidechain, interp_cc, make_pb_curve
from .dsp_module import (FilterModule, BrightnessModule, ChorusModule,
                         LeslieModule, ModVibratoModule, DspChain)

_HP_CACHE: BoundedCache = BoundedCache(32)
_BOOM_EQ_CACHE: BoundedCache = BoundedCache(16)

# Per-instrument low-mid boom cut: (center_Hz, gain_dB, Q)
_BOOM_CUT: Dict[str, Tuple[float, float, float]] = {
    "piano":      (250.0, -2.5, 0.9),
    "contrabass": (200.0, -2.0, 0.8),
    "cello":      (220.0, -1.5, 0.9),
    "organ":      (280.0, -1.5, 1.0),
}


def _get_hp(freq: float) -> np.ndarray:
    key = int(freq)
    if key not in _HP_CACHE:
        _HP_CACHE[key] = butter(6, max(freq, 15), btype='high', fs=SR, output='sos')
    return _HP_CACHE[key]


def _get_peak_eq(fc: float, gain_db: float, q: float) -> Optional[np.ndarray]:
    key = (int(fc), int(gain_db * 10), int(q * 10))
    if key not in _BOOM_EQ_CACHE:
        _BOOM_EQ_CACHE[key] = biquad_peak(fc, gain_db, q)
    return _BOOM_EQ_CACHE[key]


def _tail_estimate(timbre: Timbre, dur: float) -> float:
    return min(getattr(timbre, 'rel', 0.3) * 2.5 + getattr(timbre, 'd2', 1.0) * 0.5, 6.0)


# ── Sympathetic resonance (piano-family) ─────────────────────────────

def _apply_sympathetic(buf: np.ndarray, notes: List[Note],
                       buf_len: int) -> np.ndarray:
    piano_notes = [(st, midi_note, dur, vel)
                   for st, midi_note, dur, vel, ch, prog in notes if ch != 9]
    if len(piano_notes) < 2:
        return buf
    if len(piano_notes) > SYMPA_MAX_NOTES:
        piano_notes = sorted(piano_notes, key=lambda x: -x[0])[:SYMPA_MAX_NOTES]
    ndata: list = []
    for st, midi_note, dur, vel in piano_notes:
        f0 = A4 * 2 ** ((midi_note - 69) / 12.0)
        s, e = int(st * SR), min(int((st + dur + 0.8) * SR), buf_len)
        harmonics = [(k, f0 * k) for k in range(1, SYMPA_MAX_H + 1) if f0 * k <= 8000]
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
                    if abs(fi - fj) / min(fi, fj) < SYMPA_TOLERANCE:
                        a = SYMPA_GAIN * min(vi, vj) / (ki * kj)
                        res[ov_s:ov_e] += a * fade * decay * np.sin(
                            2 * np.pi * (fi + fj) * 0.5 * t + (ki + kj) * 0.5)
    return buf + res


# ── DSP chain builder ────────────────────────────────────────────────

def _build_dsp_chain(inst_name: str, chan_data: ChannelData,
                     buf_len: int) -> DspChain:
    modules: list = []
    modules.append(FilterModule(_get_hp(HP_FREQ.get(inst_name, 60.0))))
    if inst_name in _BOOM_CUT:
        fc, gain_db, q = _BOOM_CUT[inst_name]
        sos = _get_peak_eq(fc, gain_db, q)
        if sos is not None:
            modules.append(FilterModule(sos))
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

def render_track(inst_name: str, notes: List[Note], pan_cc: Optional[int],
                 chan_data: ChannelData, buf_len: int,
                 track_gain: float, density_scale: float,
                 cfg: PipelineConfig,
                 track_idx: int = 0) -> TrackResult:
    """Block-based track render with Voice management and DspChain."""
    vol_curve = smooth_cc_sidechain(
        CC_FLOOR + CC_SCALE * np.sqrt(
            interp_cc(chan_data.vol, buf_len, default=100.0 / 127.0)),
        down_ms=5.0, up_ms=50.0)
    expr_curve = smooth_cc(
        CC_FLOOR + CC_SCALE * np.sqrt(
            interp_cc(chan_data.expr, buf_len, default=1.0)),
        tau_ms=50.0)
    at_curve = smooth_cc(interp_cc(chan_data.aftertouch, buf_len, default=0.0))
    rev_curve = smooth_cc(
        interp_cc(chan_data.reverb, buf_len, default=40.0 / 127.0), tau_ms=50.0)

    chain = _build_dsp_chain(inst_name, chan_data, buf_len)
    buf = np.zeros(buf_len)

    drum_rng = np.random.RandomState(42 + track_idx)
    for ni, note in enumerate(notes):
        sample_pos = int(note.start * SR)
        if sample_pos >= buf_len:
            break
        if note.ch == 9:
            tone = drum(note.midi, note.dur, note.vel, drum_rng)
        else:
            nm = inst_name
            tb = TIMBRES.get(nm, TIMBRES["default"])
            freq = A4 * 2 ** ((note.midi - 69) / 12.0)
            pbc = make_pb_curve(chan_data.pb, note.start, note.dur)
            tone = synthesize(freq, note.dur, note.vel, tb, nm, ni, pb_curve=pbc)
        end = min(sample_pos + len(tone), buf_len)
        seg_len = end - sample_pos
        if seg_len > 0:
            buf[sample_pos:end] += tone[:seg_len]

    buf *= expr_curve
    buf *= 1.0 + at_curve * 0.3

    rev_buf = buf * rev_curve * REVERB_SEND.get(inst_name, 1.0)

    buf = chain.process(buf)

    if np.max(np.abs(buf)) < 1e-10:
        return TrackResult(None, None)

    if inst_name in SYMPA_INSTRUMENTS:
        buf = _apply_sympathetic(buf, notes, buf_len)

    active = np.abs(buf) > 1e-7
    if np.any(active):
        first_a = int(np.argmax(active))
        last_a = buf_len - int(np.argmax(active[::-1]))
        rms = float(np.sqrt(np.mean(buf[first_a:last_a] ** 2)))
    else:
        rms = float(np.sqrt(np.mean(buf * buf)))
    target = track_gain * density_scale
    gain = (target / rms) if rms > 1e-6 else 1.0
    gain *= VOLUME.get(inst_name, 1.0)
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
