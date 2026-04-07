"""Filtered-noise SFX engine (program 120-127).

Produces atmospheric texture using bandpass-filtered noise.
Three frequency bands (body/mid/air) with independent decay rates
create natural spectral evolution — high bands fade first, sound
warms over time.
"""
from __future__ import annotations

import math
from typing import Optional
import numpy as np
from scipy.signal import sosfilt, butter
from .dsp import BoundedCache
from .timbre import SR, Timbre, vc
from .envelope import envelope

TWO_PI = 2.0 * math.pi

_BP_CACHE = BoundedCache(32)


def _get_bp(lo, hi, order=2):
    """Cached Butterworth bandpass."""
    lo = max(20.0, lo)
    hi = min(hi, SR * 0.45)
    if hi <= lo + 30:
        return None
    key = (int(lo), int(hi), order)
    if key not in _BP_CACHE:
        _BP_CACHE[key] = butter(order, [lo, hi], btype='band',
                                fs=SR, output='sos')
    return _BP_CACHE[key]


# Band definitions: (center_multiplier, bandwidth_multiplier, amplitude, decay_tau)
_BANDS = [
    (1.0, 0.4, 0.55, 1.2),   # body — narrow, warm
    (2.5, 0.8, 0.35, 0.8),   # mid — presence
    (6.0, 1.5, 0.20, 0.5),   # air — shimmer
]


def synthesize_sfx(freq: float, dur: float, vel: float, tim: Timbre, name: str = "sfx", nid: int = 0, pb_curve: Optional[np.ndarray] = None) -> np.ndarray:
    """Render one SFX note as filtered-noise atmospheric texture."""
    tail = min(tim.rel * 1.5 + 0.25, 2.0)
    total_dur = dur + tail
    n = int(SR * total_dur)
    if n == 0:
        return np.zeros(0)

    tv = np.linspace(0, total_dur, n, endpoint=False)
    v = vc(vel)
    rng = np.random.RandomState(
        (int(freq * 100 + vel * 1000) + nid * 7919) % (2**31))

    # Pitch bend: shift the effective frequency for band center tracking
    eff_freq = freq
    if pb_curve is not None:
        pb = pb_curve[:n] if len(pb_curve) >= n else np.pad(
            pb_curve, (0, n - len(pb_curve)), mode="edge")
        # Use median PB value for static band placement
        # (noise bands can't do per-sample pitch tracking)
        eff_freq = freq * 2.0 ** (np.median(pb) / 12.0)

    wave = np.zeros(n)

    for center_m, bw_m, amp, decay_tau in _BANDS:
        center = eff_freq * center_m
        if center >= SR * 0.4:
            continue

        bw = center * bw_m * (0.5 + 0.5 * v)
        bp_lo = max(center - bw / 2, 20.0)
        bp_hi = min(center + bw / 2, SR * 0.44)

        bp = _get_bp(bp_lo, bp_hi)
        if bp is None:
            continue

        noise = rng.randn(n)
        filtered = sosfilt(bp, noise)
        band_env = np.exp(-tv / max(decay_tau, 0.05))

        # Air band: louder at high velocity
        if center_m > 3.0:
            amp *= 0.4 + 0.6 * v

        wave += amp * filtered * band_env

    # Normalise
    rms = np.sqrt(np.mean(wave**2)) + 1e-10
    if rms > 0.001:
        wave /= rms * 4.0

    wave -= np.mean(wave)

    # Amplitude envelope
    noff = int(SR * dur)
    env = envelope(tim, dur, n, vel, freq, noff)
    out = wave * env * (0.15 + 0.85 * v)

    # Gentle cosine fade-in (25 ms)
    fade_in = min(int(SR * 0.025), n)
    if fade_in > 1:
        out[:fade_in] *= 0.5 * (1 - np.cos(
            np.pi * np.linspace(0, 1, fade_in)))

    return out
