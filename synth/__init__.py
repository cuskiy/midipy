"""Synthesis dispatcher: engine selection, crossfade, peak cap."""
from __future__ import annotations

from typing import Optional

from .timbre import SR, A4, Timbre
from .instruments import TIMBRES
from .gm import (PROGRAM_MAP, PANNING, FM_INSTRUMENTS, KS_PLUCKED,
                 KS_DUR_THRESHOLD, KS_ALWAYS, SFX_INSTRUMENTS, KS_GAIN)
from .additive import synthesize as _additive, _apply_formants
from .fm import synthesize_fm as _fm
from .ks import synthesize_plucked as _ks_pluck
from .inharmonic import synthesize_sfx as _sfx
from .drums import drum
import numpy as np

_KS_BLEND_LO = KS_DUR_THRESHOLD - 0.25
_KS_BLEND_HI = KS_DUR_THRESHOLD + 0.15

_SFX_MIN_FREQ = 30.0


def _peak_cap(w):
    pk = np.max(np.abs(w))
    if pk > 1.0:
        w *= 1.0 / pk
    return w


def _apply_pb_shift(w, pb_curve):
    """Apply pitch bend to a pre-rendered waveform via variable-rate read.

    When bending up, the read position advances faster than real time,
    so we may run out of source samples.  The caller should provide a
    waveform long enough to cover the maximum read-ahead; any read
    beyond the source is filled with the last valid sample's decay.
    """
    if pb_curve is None:
        return w
    n = len(w)
    pb = pb_curve[:n] if len(pb_curve) >= n else np.pad(
        pb_curve, (0, n - len(pb_curve)), mode='edge')
    if np.max(np.abs(pb)) < 0.01:
        return w
    ratio = 2.0 ** (pb / 12.0)
    read_pos = np.cumsum(ratio) - ratio[0]
    # Extend source with exponential decay so pitch-up bends don't
    # abruptly silence.  Extra length = max overshoot.
    max_read = int(np.max(read_pos)) + 2
    if max_read > n:
        extra = max_read - n
        # Decay tail: fade the last value to zero over the extra samples
        tail_env = np.exp(-np.arange(extra, dtype=np.float64) * 8.0 / max(extra, 1))
        w = np.concatenate([w, w[-1] * tail_env])
    i0 = np.floor(read_pos).astype(int)
    frac = read_pos - i0
    ok = (i0 >= 0) & (i0 < len(w) - 1)
    out = np.zeros(n)
    out[ok] = w[i0[ok]] * (1 - frac[ok]) + w[i0[ok] + 1] * frac[ok]
    return out


def _ks_with_formants(w: np.ndarray, tim: Timbre, freq: float = 0) -> np.ndarray:
    if tim.formant_freqs:
        w = _apply_formants(w, tim, freq)
    return _peak_cap(w)


def _ks_gain(name):
    return KS_GAIN.get(name, 1.0)


def synthesize(freq: float, dur: float, vel: float, tim: Timbre, name: str = "default",
               nid: int = 0, pb_curve: Optional[np.ndarray] = None) -> np.ndarray:
    if name in SFX_INSTRUMENTS:
        if freq < _SFX_MIN_FREQ:
            return np.zeros(0)
        return _peak_cap(_sfx(freq, dur, vel, tim, name, nid, pb_curve=pb_curve))

    if name in FM_INSTRUMENTS:
        return _peak_cap(_fm(freq, dur, vel, tim, name, nid, pb_curve=pb_curve))

    if name in KS_PLUCKED:
        if dur < _KS_BLEND_LO or name in KS_ALWAYS:
            w = _ks_with_formants(
                _ks_pluck(freq, dur, vel, tim, name, nid), tim, freq)
            return _peak_cap(_apply_pb_shift(w, pb_curve))
        if dur <= _KS_BLEND_HI:
            w = _ks_with_formants(
                _ks_pluck(freq, dur, vel, tim, name, nid), tim, freq)
            w = _apply_pb_shift(w, pb_curve)
            w_add = _additive(freq, dur, vel, tim, name, nid, pb_curve=pb_curve)
            n = min(len(w), len(w_add))
            rms_ks = np.sqrt(np.mean(w[:n] ** 2)) + 1e-10
            rms_add = np.sqrt(np.mean(w_add[:n] ** 2)) + 1e-10
            w_add *= rms_ks / rms_add
            blend = (dur - _KS_BLEND_LO) / (_KS_BLEND_HI - _KS_BLEND_LO)
            out = np.zeros(max(len(w), len(w_add)))
            out[:len(w)] += np.sqrt(1 - blend) * w
            out[:len(w_add)] += np.sqrt(blend) * w_add
            return _peak_cap(out)
        w = _additive(freq, dur, vel, tim, name, nid, pb_curve=pb_curve)
        w *= _ks_gain(name)
        return _peak_cap(w)

    return _peak_cap(_additive(freq, dur, vel, tim, name, nid, pb_curve=pb_curve))
