"""FM synthesis: multi-operator with oversampling."""
from __future__ import annotations
import math
from typing import Optional
import numpy as np
from scipy.signal import resample_poly, sosfilt
from .timbre import SR, Timbre, vc, lf, get_bp
from .envelope import envelope
from .additive import _apply_formants

OS = 2
SR_OS = SR * OS

FM_PRESETS = {
    "epiano": dict(c_ratio=1.0, ops=[
        (1.0,  1.2, 0.8,  6.0),
        (7.0,  0.12, 0.08, 18.0),
    ]),
    "celesta": dict(c_ratio=1.0, ops=[
        (1.001, 0.55, 0.25, 4.0),
        (3.011, 0.12, 0.08, 12.0),
    ]),
    "vibes": dict(c_ratio=1.0, ops=[
        (1.0,   1.1, 0.55, 2.0),
        (3.997, 0.22, 0.14, 5.0),
    ]),
    "marimba": dict(c_ratio=1.0, ops=[
        (4.0,  0.7, 0.4, 30.0),
        (10.0, 0.10, 0.08, 35.0),
    ]),
}


def synthesize_fm(freq: float, dur: float, vel: float, tim: Timbre, name: str = "epiano", nid: int = 0, pb_curve: Optional[np.ndarray] = None) -> np.ndarray:
    p = FM_PRESETS.get(name, FM_PRESETS["epiano"])
    tail = min(tim.rel * 1.2 + 0.15, 2.5) if dur >= 0.2 else min(tim.rel + 0.20, 1.0)
    td = dur + tail
    n = int(SR * td)
    if n == 0:
        return np.zeros(0)
    n_os = n * OS
    tv = np.linspace(0, td, n_os, endpoint=False)
    v = vc(vel)
    short = dur < 0.2
    noff = int(SR * dur)
    _l = lf(freq)
    rng = np.random.RandomState((int(freq * 100 + vel * 1000) + nid * 7919) % (2**31))

    fc = freq * p['c_ratio']
    peak_idx = sum(op[0] * (op[1] + op[2]) for op in p['ops'])
    guard = min(1.0, (SR_OS * 0.4) / (fc * (peak_idx + 1)))
    kt = max(0.15, min(1.0, 1.0 - max(0, math.log2(freq / 500.0)) * 0.25))

    sb = 1.0
    if tim.spec_b > 0:
        sb = 1 + tim.spec_b * v * np.exp(-tv / max(tim.spec_tau, 0.001))

    micro = tim.micro
    if micro > 0.05:
        det_cents = micro * 0.5
        voice_offsets = [-det_cents, 0.0, det_cents]
        voice_weights = [0.30, 0.40, 0.30]
    else:
        voice_offsets = [0.0]
        voice_weights = [1.0]

    w_os = np.zeros(n_os)
    for vi, (v_off, v_wt) in enumerate(zip(voice_offsets, voice_weights)):
        fc_v = fc * 2 ** (v_off / 1200.0)

        mod = np.zeros(n_os)
        for mr, ib, iv, idec in p['ops']:
            fm = freq * mr * 2 ** (v_off / 1200.0)
            if fm >= SR_OS * 0.45:
                continue
            # Per-operator guard: limit index so sidebands stay below Nyquist
            op_guard = min(guard, (SR_OS * 0.45 - fc) / max(fm, 1.0))
            op_guard = max(op_guard, 0.0)
            idx = (ib + iv * v) * op_guard * kt * np.exp(-idec * tv)
            if tim.spec_b > 0:
                idx *= sb
            mod += idx * np.sin(2 * np.pi * fm * tv + rng.random() * 0.3)

        # pitch modulation: vibrato + pitch bend
        pm = None
        if tim.vib_d > 0:
            ve = np.clip((tv - tim.vib_del) / 0.2, 0, 1)
            pm = ve * tim.vib_d * np.sin(2 * np.pi * tim.vib_r * tv)
        if pb_curve is not None:
            pb_rep = np.repeat(pb_curve, OS)
            if len(pb_rep) < n_os:
                pb_os = np.pad(pb_rep, (0, n_os - len(pb_rep)), mode='edge') * 100.0
            else:
                pb_os = pb_rep[:n_os] * 100.0
            pm = (pm + pb_os) if pm is not None else pb_os

        # Phase offset from nid for multi-track decorrelation
        ph_nid = nid * 2.399
        if pm is not None:
            ph = 2 * np.pi * np.cumsum(fc_v * 2 ** (pm / 1200.0)) / SR_OS + ph_nid
        else:
            ph = 2 * np.pi * fc_v * tv + ph_nid

        if micro > 0.05 and v_off != 0:
            r_slow = 0.3 + rng.random() * 1.2
            jit = micro * 0.12 * np.sin(2 * np.pi * r_slow * tv + rng.random() * 6.283)
            ph += jit

        w_os += v_wt * np.sin(ph + mod)

    w = resample_poly(w_os, 1, OS)[:n]

    if micro > 0.05:
        tv_n = np.linspace(0, td, n, endpoint=False)
        am_rate = 0.8 + rng.random() * 2.0
        am_depth = micro * 0.02
        w *= 1 + am_depth * np.sin(2 * np.pi * am_rate * tv_n + rng.random() * 6.283)
    tv_n = np.linspace(0, td, n, endpoint=False)

    if tim.trem_d > 0:
        w *= 1 - tim.trem_d * 0.5 * (1 + np.sin(2 * np.pi * tim.trem_r * tv_n))

    if tim.drive > 0:
        d = tim.drive * (1 + 0.3 * v) * (min(dur / 0.2, 1) if short else 1)
        td_v = np.tanh(d)
        if td_v > 1e-6:
            w = np.tanh(w * d) / td_v

    if tim.formant_freqs:
        w = _apply_formants(w, tim, freq)

    # Remove DC offset from FM modulation asymmetry
    w -= np.mean(w)

    env = envelope(tim, dur, n, vel, freq, noff)
    out = w * env * (0.15 + 0.85 * v)

    if tim.noise > 0:
        nl = tim.noise * (1 + tim.vn * v)
        if tim.noise_hi > 0:
            nl *= 1 + tim.noise_hi * _l
        if short:
            nl *= max(dur / 0.2, 0.2)
        ns = rng.randn(n)
        bp_lo = max(freq*1.5, 120) if tim.noise_d < 10 else max(freq*0.5, 80)
        bp = get_bp(bp_lo, min(freq * tim.noise_peak * (1 + 0.3 * v), SR * 0.45)) if n > 512 else None
        if bp is not None:
            ns = sosfilt(bp, ns)
        out += nl * ns * np.exp(-tim.noise_d * (1 + 0.5 * _l) * tv_n) * env

    if tim.key_click > 0:
        cn = min(int(SR * 0.003), n)
        if cn > 0:
            out[:cn] += tim.key_click * v * rng.randn(cn) * np.exp(-np.arange(cn) * 8.0 / cn)
    if short:
        fade = min(int(SR * 0.004), n)
        if fade > 1:
            out[:fade] *= np.linspace(0, 1, fade) ** 0.5
    else:
        mf = min(int(SR * 0.0015), n)
        if mf > 1: out[:mf] *= 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, mf)))
    # Remove residual DC from envelope/noise/click additions
    out -= np.mean(out)
    return out