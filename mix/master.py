"""Master chain: FDN reverb IR, stereo compressor, master EQ stages.

All EQ functions use sosfiltfilt (zero-phase, forward+backward) which
doubles the effective magnitude change.  Parameter values are specified
at half the intended effective dB.
"""
import numpy as np
from scipy.signal import sosfilt, sosfiltfilt, butter as _sc_butter
from synth.timbre import SR, biquad_peak, biquad_low_shelf, biquad_high_shelf
from synth.dsp import BoundedCache

_H8_S = 1.0 / 8.0 ** 0.5
_N = 8
_PRIMES = [743, 829, 911, 1013, 1109, 1213, 1321, 1429]
_AP_PRIMES = [113, 131, 151, 173, 191, 211, 233, 251]
_MOD_DEPTHS = [3.5, 4.2, 5.0, 5.8, 4.5, 5.3, 6.1, 3.8]
_MOD_RATES = [0.7, 0.83, 0.97, 1.13, 0.77, 0.91, 1.05, 1.19]
_MOD_PHASES = [i * 2 * np.pi / _N for i in range(_N)]

_EQ_CACHE = BoundedCache(16)


def _get_eq(name: str) -> np.ndarray:
    if name not in _EQ_CACHE:
        if name == "air":
            _EQ_CACHE[name] = biquad_high_shelf(8000.0, 0.75)
        elif name == "rev_dark":
            _EQ_CACHE[name] = biquad_high_shelf(6000.0, -0.6)
        elif name == "mud_cut":
            _EQ_CACHE[name] = biquad_peak(380.0, -1.5, 0.45)
        elif name == "bass_tight":
            _EQ_CACHE[name] = biquad_low_shelf(120.0, -0.8)
        elif name == "presence":
            _EQ_CACHE[name] = biquad_peak(3500.0, 0.5, 0.7)
    return _EQ_CACHE[name]


def reverb_darken(s: np.ndarray) -> np.ndarray:
    return sosfiltfilt(_get_eq("rev_dark"), s)


_PRE_COMP_EQ = None
_POST_COMP_EQ = None


def _ensure_master_eq() -> None:
    global _PRE_COMP_EQ, _POST_COMP_EQ
    if _PRE_COMP_EQ is None:
        _PRE_COMP_EQ = np.vstack([_get_eq("bass_tight"), _get_eq("mud_cut")])
        _POST_COMP_EQ = np.vstack([_get_eq("presence"), _get_eq("air")])


def pre_comp_eq(s: np.ndarray) -> np.ndarray:
    """Cascaded bass_tight + mud_cut (single sosfiltfilt pass)."""
    _ensure_master_eq()
    return sosfiltfilt(_PRE_COMP_EQ, s)


def post_comp_eq(s: np.ndarray) -> np.ndarray:
    """Cascaded presence + air (single sosfiltfilt pass)."""
    _ensure_master_eq()
    return sosfiltfilt(_POST_COMP_EQ, s)


def fdn_reverb_ir(room_size: float = 1.0, rt60: float = 1.6,
                   damping: float = 0.3, hf_damp: float = 0.55,
                   dur: float = None) -> tuple:
    """Generate FDN reverb impulse response.

    Per-sample loop kept in Python (delay-line feedback prevents
    vectorisation), but state is held in plain lists rather than numpy
    arrays — element access is ~10× faster, bringing IR generation
    from ~9s to ~1s for the default 2.4s IR.
    """
    if dur is None:
        dur = min(rt60 * 1.5, 3.0)
    n = int(SR * dur)
    delays = [max(int(p * room_size), 1) for p in _PRIMES]
    ap_delays = [max(int(p * room_size), 1) for p in _AP_PRIMES]
    gains = [10 ** (-3 * d / (SR * max(rt60, 0.1))) for d in delays]

    # Per-channel delay buffers as plain Python lists (fast random access)
    dl_bufs = [[0.0] * d for d in delays]
    ap_bufs = [[0.0] * d for d in ap_delays]
    ptrs = [0] * _N
    ap_ptrs = [0] * _N
    f_mid = [0.0] * _N
    f_hf = [0.0] * _N

    ap_g = 0.5
    di_mid = 1.0 - damping
    dm_mid = damping
    di_hf = 1.0 - hf_damp
    dm_hf = hf_damp
    s8 = _H8_S

    # Pre-compute modulation as list of lists (faster scalar access than 2D numpy)
    import math as _m
    two_pi = 2 * _m.pi
    mod_all = [
        [_MOD_DEPTHS[k] * _m.sin(two_pi * _MOD_RATES[k] * (i / SR) + _MOD_PHASES[k])
         for i in range(n)]
        for k in range(_N)
    ]

    ir_l = [0.0] * n
    ir_r = [0.0] * n
    o = [0.0] * _N

    for i in range(n):
        for k in range(_N):
            buf = dl_bufs[k]
            dl = delays[k]
            rp = ptrs[k] - 1 + mod_all[k][i]
            idx = int(rp) % dl
            if idx < 0:
                idx += dl
            frac = rp - int(rp)
            if frac < 0:
                frac += 1.0
                idx = (idx - 1) % dl
            idx2 = (idx + 1) % dl
            a = buf[idx]
            o[k] = a + frac * (buf[idx2] - a)

        a0 = o[0]+o[1]; a1 = o[0]-o[1]; a2 = o[2]+o[3]; a3 = o[2]-o[3]
        a4 = o[4]+o[5]; a5 = o[4]-o[5]; a6 = o[6]+o[7]; a7 = o[6]-o[7]
        c0 = a0+a2; c1 = a1+a3; c2 = a0-a2; c3 = a1-a3
        c4 = a4+a6; c5 = a5+a7; c6 = a4-a6; c7 = a5-a7
        had0 = (c0+c4)*s8; had1 = (c1+c5)*s8; had2 = (c2+c6)*s8; had3 = (c3+c7)*s8
        had4 = (c0-c4)*s8; had5 = (c1-c5)*s8; had6 = (c2-c6)*s8; had7 = (c3-c7)*s8
        had = (had0, had1, had2, had3, had4, had5, had6, had7)

        for k in range(_N):
            v = had[k]
            if i == 0:
                v += s8
            fm = di_mid * v + dm_mid * f_mid[k]
            f_mid[k] = fm
            fh = di_hf * fm + dm_hf * f_hf[k]
            f_hf[k] = fh

            ap_buf = ap_bufs[k]
            ap_p = ap_ptrs[k]
            ap_d = ap_delays[k]
            ap_out = ap_buf[ap_p]
            ap_in = fh + ap_g * ap_out
            ap_buf[ap_p] = ap_in
            ap_ptrs[k] = (ap_p + 1) % ap_d
            diff = ap_out - ap_g * ap_in

            dl_bufs[k][ptrs[k]] = diff * gains[k]
            ptrs[k] = (ptrs[k] + 1) % delays[k]

        ir_l[i] = o[0] + o[2] + o[4] + o[6]
        ir_r[i] = o[1] + o[3] + o[5] + o[7]

    return np.asarray(ir_l), np.asarray(ir_r)


def compress(l: np.ndarray, r: np.ndarray,
             thresh: float = 0.55, ratio: float = 2.0,
             att_ms: float = 40, rel_ms: float = 200,
             knee_db: float = 6.0, sc_hp: float = 0) -> tuple:
    n = len(l)
    lk = np.maximum(np.abs(l), np.abs(r))
    if sc_hp > 0:
        sc_sos = _sc_butter(2, max(sc_hp, 20.0), btype='high', fs=SR, output='sos')
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
