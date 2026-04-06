"""Audio effects: chorus, Leslie, mod vibrato, sympathetic resonance."""

import numpy as np
from scipy.signal import butter, sosfilt
from synth.timbre import SR
from synth import A4

# ── Sympathetic resonance ────────────────────────────────────────────

_SYMPA_INSTRUMENTS = {"piano", "harpsichord", "harp"}
_SYMPA_GAIN = 0.0004
_SYMPA_MAX_H = 10
_SYMPA_TOLERANCE = 0.012


def apply_sympathetic(buf, notes, buf_len):
    """Add sympathetic string resonance for piano-family instruments."""
    piano_notes = [(st, note, dur, vel)
                   for st, note, dur, vel, ch, prog in notes if ch != 9]
    if len(piano_notes) < 2:
        return buf
    ndata = []
    for st, note, dur, vel in piano_notes:
        f0 = A4 * 2 ** ((note - 69) / 12.0)
        s = int(st * SR)
        e = min(int((st + dur + 0.8) * SR), buf_len)
        harmonics = []
        for k in range(1, _SYMPA_MAX_H + 1):
            fk = f0 * k
            if fk > 8000:
                break
            harmonics.append((k, fk))
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
                        sym_f = (fi + fj) * 0.5
                        a = _SYMPA_GAIN * min(vi, vj) / (ki * kj)
                        res[ov_s:ov_e] += a * fade * decay * np.sin(
                            2 * np.pi * sym_f * t + (ki + kj) * 0.5)
    return buf + res


# ── Chorus ───────────────────────────────────────────────────────────

def apply_chorus(buf, depth):
    """GM CC93 chorus: two modulated-delay voices, RMS-compensated."""
    if depth < 0.01:
        return buf
    n = len(buf)
    tv = np.arange(n, dtype=np.float64) / SR
    out = buf.copy()
    for rate, phase in [(0.7, 0.0), (1.1, 2.1)]:
        d = SR * 0.008 + SR * 0.004 * np.sin(2 * np.pi * rate * tv + phase)
        idx = np.arange(n, dtype=np.float64) - d
        i0 = np.floor(idx).astype(int)
        frac = idx - i0
        ok = (i0 >= 0) & (i0 < n - 1)
        delayed = np.zeros(n)
        delayed[ok] = buf[i0[ok]] * (1 - frac[ok]) + buf[i0[ok] + 1] * frac[ok]
        out += delayed * depth * 0.15
    rms_in = np.sqrt(np.mean(buf ** 2)) + 1e-10
    rms_out = np.sqrt(np.mean(out ** 2)) + 1e-10
    out *= rms_in / rms_out
    return out


# ── Mod vibrato ──────────────────────────────────────────────────────

def apply_mod_vibrato(buf, mod_curve):
    """GM CC1 pitch vibrato via modulated delay."""
    if mod_curve is None:
        return buf
    peak = np.max(mod_curve)
    if peak < 0.01:
        return buf
    n = len(buf)
    tv = np.arange(n, dtype=np.float64) / SR
    max_cents = 50.0
    peak_delay = (max_cents * 0.000578) / (2 * np.pi * 5.5) * SR
    d = mod_curve * peak_delay * np.sin(2 * np.pi * 5.5 * tv)
    idx = np.arange(n, dtype=np.float64) - d
    i0 = np.floor(idx).astype(int)
    frac = idx - i0
    ok = (i0 >= 0) & (i0 < n - 1)
    out = buf.copy()
    out[ok] = buf[i0[ok]] * (1 - frac[ok]) + buf[i0[ok] + 1] * frac[ok]
    return out


# ── Leslie ───────────────────────────────────────────────────────────

_LESLIE_LP = None


def apply_leslie(buf):
    """Leslie speaker simulation for organ."""
    global _LESLIE_LP
    if _LESLIE_LP is None:
        _LESLIE_LP = butter(3, 800.0, btype='low', fs=SR, output='sos')
    n = len(buf)
    tv = np.arange(n, dtype=np.float64) / SR
    lo = sosfilt(_LESLIE_LP, buf)
    hi = buf - lo
    hp = 2 * np.pi * 6.8 * tv
    hi *= 1.0 + 0.30 * np.sin(hp)
    dop = 0.0004 * SR * np.sin(hp + 1.57)
    idx = np.arange(n, dtype=np.float64) - dop
    i0 = np.floor(idx).astype(int)
    fr = idx - i0
    ok = (i0 >= 1) & (i0 < n - 1)
    hd = np.zeros(n)
    hd[ok] = hi[i0[ok]] * (1 - fr[ok]) + hi[i0[ok] + 1] * fr[ok]
    lo *= 1.0 + 0.15 * np.sin(2 * np.pi * 5.5 * tv + 0.8)
    out = lo + hd
    rms_in = np.sqrt(np.mean(buf ** 2)) + 1e-12
    rms_out = np.sqrt(np.mean(out ** 2)) + 1e-12
    return out * (rms_in / rms_out)
