"""Master chain: FDN reverb IR, stereo compressor, master EQ stages.

All EQ functions use sosfiltfilt (zero-phase, forward+backward) which
doubles the effective magnitude change.  Parameter values are specified
at half the intended effective dB.
"""

import numpy as np
from scipy.signal import sosfilt, sosfiltfilt, butter as _sc_butter
from synth.timbre import SR, biquad_peak, biquad_low_shelf, biquad_high_shelf

_H8_S = 1.0 / 8.0 ** 0.5
_N = 8
_PRIMES = [743, 829, 911, 1013, 1109, 1213, 1321, 1429]
_AP_PRIMES = [113, 131, 151, 173, 191, 211, 233, 251]
_MOD_DEPTHS = [3.5, 4.2, 5.0, 5.8, 4.5, 5.3, 6.1, 3.8]
_MOD_RATES = [0.7, 0.83, 0.97, 1.13, 0.77, 0.91, 1.05, 1.19]
_MOD_PHASES = [i * 2 * np.pi / _N for i in range(_N)]

_EQ_CACHE = {}


def _get_eq(name):
    if name not in _EQ_CACHE:
        if name == "air":
            # +0.75 param → +1.5 dB effective (sosfiltfilt)
            _EQ_CACHE[name] = biquad_high_shelf(8000.0, 0.75)
        elif name == "rev_dark":
            # -0.6 param → -1.2 dB effective
            _EQ_CACHE[name] = biquad_high_shelf(6000.0, -0.6)
        elif name == "mud_cut":
            # -1.5 param → -3.0 dB effective @ 380 Hz
            _EQ_CACHE[name] = biquad_peak(380.0, -1.5, 0.45)
        elif name == "bass_tight":
            # -0.8 param → -1.6 dB effective @ 120 Hz
            _EQ_CACHE[name] = biquad_low_shelf(120.0, -0.8)
        elif name == "presence":
            # +0.5 param → +1.0 dB effective @ 3.5 kHz
            _EQ_CACHE[name] = biquad_peak(3500.0, 0.5, 0.7)
    return _EQ_CACHE[name]


def air_eq(s):
    return sosfiltfilt(_get_eq("air"), s)


def reverb_darken(s):
    return sosfiltfilt(_get_eq("rev_dark"), s)


def mud_cut(s):
    return sosfiltfilt(_get_eq("mud_cut"), s)


def bass_tight(s):
    return sosfiltfilt(_get_eq("bass_tight"), s)


def presence_eq(s):
    return sosfiltfilt(_get_eq("presence"), s)


def fdn_reverb_ir(room_size=1.0, rt60=1.6, damping=0.3, hf_damp=0.55, dur=None):
    """Generate FDN reverb impulse response (per-sample loop)."""
    if dur is None:
        dur = min(rt60 * 1.5, 3.0)
    n = int(SR * dur)
    delays = [max(int(p * room_size), 1) for p in _PRIMES]
    ap_delays = [max(int(p * room_size), 1) for p in _AP_PRIMES]
    gains = np.array([10 ** (-3 * d / (SR * max(rt60, 0.1))) for d in delays])

    max_dl = max(delays)
    max_ap = max(ap_delays)
    dl_bufs = np.zeros((_N, max_dl))
    dl_lens = np.array(delays, dtype=np.int64)
    ap_bufs = np.zeros((_N, max_ap))
    ap_lens = np.array(ap_delays, dtype=np.int64)
    ptrs = np.zeros(_N, dtype=np.int64)
    ap_ptrs = np.zeros(_N, dtype=np.int64)
    ap_g = 0.5

    f_mid = np.zeros(_N)
    f_hf = np.zeros(_N)
    di_mid = 1.0 - damping
    dm_mid = damping
    di_hf = 1.0 - hf_damp
    dm_hf = hf_damp
    s8 = _H8_S

    # Pre-compute modulation
    tv = np.arange(n, dtype=np.float64) / SR
    mod_all = np.empty((_N, n))
    for k in range(_N):
        mod_all[k] = _MOD_DEPTHS[k] * np.sin(
            2 * np.pi * _MOD_RATES[k] * tv + _MOD_PHASES[k])

    ir_l = np.zeros(n)
    ir_r = np.zeros(n)
    o = np.zeros(_N)

    for i in range(n):
        for k in range(_N):
            rp = ptrs[k] - 1 + mod_all[k, i]
            dl = dl_lens[k]
            idx = int(rp) % dl
            if idx < 0:
                idx += dl
            frac = rp - int(rp)
            if frac < 0:
                frac += 1.0
                idx = (idx - 1) % dl
            idx2 = (idx + 1) % dl
            o[k] = dl_bufs[k, idx] + frac * (dl_bufs[k, idx2] - dl_bufs[k, idx])

        # Hadamard mixing (8×8)
        a0 = o[0]+o[1]; a1 = o[0]-o[1]; a2 = o[2]+o[3]; a3 = o[2]-o[3]
        a4 = o[4]+o[5]; a5 = o[4]-o[5]; a6 = o[6]+o[7]; a7 = o[6]-o[7]
        c0 = a0+a2; c1 = a1+a3; c2 = a0-a2; c3 = a1-a3
        c4 = a4+a6; c5 = a5+a7; c6 = a4-a6; c7 = a5-a7
        had = ((c0+c4)*s8, (c1+c5)*s8, (c2+c6)*s8, (c3+c7)*s8,
              (c0-c4)*s8, (c1-c5)*s8, (c2-c6)*s8, (c3-c7)*s8)

        for k in range(_N):
            v = had[k]
            if i == 0:
                v += s8
            f_mid[k] = di_mid * v + dm_mid * f_mid[k]
            f_hf[k] = di_hf * f_mid[k] + dm_hf * f_hf[k]

            ap_d = ap_lens[k]
            ap_out = ap_bufs[k, ap_ptrs[k]]
            ap_in = f_hf[k] + ap_g * ap_out
            ap_bufs[k, ap_ptrs[k]] = ap_in
            ap_ptrs[k] = (ap_ptrs[k] + 1) % ap_d
            diff = ap_out - ap_g * ap_in

            dl_bufs[k, ptrs[k]] = diff * gains[k]
            ptrs[k] = (ptrs[k] + 1) % dl_lens[k]

        ir_l[i] = o[0] + o[2] + o[4] + o[6]
        ir_r[i] = o[1] + o[3] + o[5] + o[7]

    return ir_l, ir_r


def compress(l, r, thresh=0.55, ratio=2.0, att_ms=40, rel_ms=200,
             knee_db=6.0, sc_hp=0):
    n = len(l)
    lk = np.maximum(np.abs(l), np.abs(r))
    if sc_hp > 0:
        sc_sos = _sc_butter(2, max(sc_hp, 20.0), btype='high',
                            fs=SR, output='sos')
        sc_l = sosfilt(sc_sos, l)
        sc_r = sosfilt(sc_sos, r)
        lk = np.maximum(np.abs(sc_l), np.abs(sc_r))
    ca = np.exp(-1 / max(int(SR * att_ms / 1000), 1))
    cr = np.exp(-1 / max(int(SR * rel_ms / 1000), 1))
    BLK = 64
    nb = (n + BLK - 1) // BLK
    padded = np.pad(lk, (0, nb * BLK - n), constant_values=0)
    blk_peaks = padded.reshape(nb, BLK).max(axis=1)
    env_blk = np.empty(nb)
    env_blk[0] = blk_peaks[0]
    for i in range(1, nb):
        cb = (ca if blk_peaks[i] > env_blk[i - 1] else cr) ** BLK
        env_blk[i] = cb * env_blk[i - 1] + (1 - cb) * blk_peaks[i]
    centres = np.arange(nb) * BLK + BLK // 2
    env = np.interp(np.arange(n), centres, env_blk)
    g = np.ones(n)
    knee_half = knee_db * 0.05
    t_lo, t_hi = max(thresh - knee_half, 0.01), thresh + knee_half
    above = env > t_hi
    knee_zone = (env > t_lo) & ~above
    g[above] = (thresh + (env[above] - thresh) / ratio) / env[above]
    if np.any(knee_zone):
        t = (env[knee_zone] - t_lo) / (t_hi - t_lo)
        r_eff = 1 + (ratio - 1) * t
        g[knee_zone] = (thresh + (env[knee_zone] - thresh) / r_eff) / env[knee_zone]
    return l * g, r * g
