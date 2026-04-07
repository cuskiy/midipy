"""KS plucked string with LP-phase-delay pitch compensation.

Triangle pulse excitation, 2x oversampling, anti-harshness LP.
LP phase delay subtracted from delay line for sub-cent pitch accuracy.
"""
from __future__ import annotations
import math
import numpy as np
from .dsp import BoundedCache
from scipy.signal import resample_poly, butter, sosfilt
from .timbre import SR, Timbre, vc

OS = 2
SR_OS = SR * OS
KS_T60 = 2.0
KS_MUTE = 0.985
KS_LP_BASE = 0.55
KS_LP_VEL = 0.08
_AH_CACHE = BoundedCache(32)


def _lp_phase_delay(coeff, freq):
    """Phase delay of 1-pole LP: y = a*x + (1-a)*y_prev."""
    w0 = 2 * math.pi * freq / SR_OS
    if w0 < 1e-12:
        return (1.0 - coeff) / max(coeff, 1e-12)
    b = 1.0 - coeff
    phase = -math.atan2(b * math.sin(w0), 1.0 - b * math.cos(w0))
    return -phase / w0


def _anti_harshness(out, freq):
    fc = min(12000 + freq * 3, SR * 0.45)
    key = int(fc / 100)
    if key not in _AH_CACHE:
        _AH_CACHE[key] = butter(1, key * 100, btype='low', fs=SR, output='sos')
    return sosfilt(_AH_CACHE[key], out)


def synthesize_plucked(freq: float, dur: float, vel: float, tim: Timbre, name: str = "guitar", nid: int = 0) -> np.ndarray:
    tail = min(tim.rel * 1.2 + 0.35, 2.5)
    n = int(SR * (dur + tail))
    if n == 0:
        return np.zeros(0)
    n_os = n * OS
    v = vc(vel)
    rng = np.random.RandomState((int(freq * 100 + vel * 1000) + nid * 7919) % (2**31))

    # LP coefficient
    lp_base = (tim.ks_lp if tim.ks_lp > 0 else KS_LP_BASE) + KS_LP_VEL * v
    lp = 1.0 - (1.0 - lp_base) ** 0.5

    # v79: LP phase delay compensation for sub-cent pitch accuracy
    period = SR_OS / freq
    lp_delay = _lp_phase_delay(lp, freq)
    adj = period - lp_delay
    dl_len = max(int(math.ceil(adj)), 2)
    frac = dl_len - adj
    if frac < 0:
        dl_len += 1; frac += 1.0
    elif frac >= 1.0:
        dl_len -= 1; frac -= 1.0
    dl_len = max(dl_len, 2)

    loss_base = 10 ** (-3 / (max(freq, 40) * KS_T60))
    loss = math.sqrt(loss_base)

    pos = tim.strike_pos if tim.strike_pos > 0.01 else 0.14
    pk_idx = max(1, min(int(pos * dl_len), dl_len - 2))
    exc = np.zeros(dl_len)
    exc[:pk_idx + 1] = np.linspace(0, 1, pk_idx + 1)
    exc[pk_idx:] = np.linspace(1, 0, dl_len - pk_idx)
    exc -= np.mean(exc)
    exc *= v * 0.8
    click = tim.ks_click if tim.ks_click > 0 else 0.05
    click_noise = rng.randn(dl_len) * click * v * 0.5
    click_noise -= np.mean(click_noise)
    exc += click_noise

    noff_os = int(SR_OS * dur)
    loss_mute = loss * KS_MUTE
    dl = exc.copy()
    out_os = np.zeros(n_os)
    ptr, prev = 0, 0.0
    for i in range(n_os):
        idx1 = (ptr + 1) % dl_len
        val = dl[ptr] + frac * (dl[idx1] - dl[ptr])
        filt = lp * val + (1 - lp) * prev
        prev = filt
        out_os[i] = val
        dl[ptr] = filt * (loss if i < noff_os else loss_mute)
        ptr = (ptr + 1) % dl_len

    out = resample_poly(out_os, 1, OS)[:n]
    # Decorrelate multi-track unison: shift output by a fraction of one
    # period.  KS steady-state converges to a unique waveform shape
    # regardless of excitation, so phase can only be varied by delaying
    # the output.  A sub-period delay is inaudible but decorrelates.
    if nid > 0 and freq > 20:
        period_samples = max(int(SR / freq), 1)
        delay = int(nid * period_samples / 5.3)  # spread ~evenly across period
        delay = max(1, min(delay, n - 1))
        out = np.roll(out, delay)
        fade = min(delay, 32)
        out[:fade] *= np.linspace(0, 1, fade)  # fade in to avoid click
    pk = np.max(np.abs(out))
    if pk > 1e-6:
        out /= pk

    att_n = min(int(SR * max(tim.att, 0.002)), n)
    if att_n > 1:
        out[:att_n] *= np.linspace(0, 1, att_n) ** 0.5

    cn = min(int(SR * 0.003), n)
    if cn > 0:
        cs = rng.randn(cn) * click * v * np.exp(-np.arange(cn) * 12.0 / cn)
        cf = min(int(SR * 0.0004), cn)
        if cf > 1:
            cs[:cf] *= np.linspace(0, 1, cf)
        out[:cn] += cs
    bn = min(int(SR * 0.012), n)
    if bn > 1 and click > 0.03:
        body = rng.randn(bn) * click * v * 0.25
        body *= np.exp(-np.arange(bn) * 5.0 / bn)
        body *= np.sin(2 * np.pi * min(freq * 2.5, SR * 0.4) * np.arange(bn) / SR)
        out[:bn] += body

    if n > SR // 4:
        out -= np.mean(out)
    if n > 256:
        out = _anti_harshness(out, freq)

    pk2 = np.max(np.abs(out))
    if pk2 > 1.0:
        out /= pk2

    ac = min(128, n)
    if ac > 1:
        out[-ac:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, ac)))
    return out * (0.15 + 0.85 * v)
