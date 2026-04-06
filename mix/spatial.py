import numpy as np
from scipy.signal import fftconvolve, butter, sosfilt
from synth.timbre import SR, biquad_peak, biquad_high_shelf

HEAD_R = 0.0875
C_SOUND = 343.0

_PINNA_SOS = None
_ER_HP = None

_ER_TAPS = [
    (0.011, 0.25, -48),
    (0.020, 0.25,  50),
    (0.031, 0.18,  35),
    (0.041, 0.18, -40),
    (0.054, 0.10,  -8),
]

# Immersive mode: additional ceiling/floor reflections with spectral tilt
_ER_TAPS_IMMERSIVE = [
    (0.009, 0.22, -45),
    (0.015, 0.22,  52),
    (0.023, 0.16, -30),
    (0.029, 0.20,  38),
    (0.036, 0.15, -42),
    (0.044, 0.12,  15),
    (0.052, 0.08, -10),
    (0.065, 0.06,  25),
]
_CEILING_SOS = None


def _ensure_pinna():
    global _PINNA_SOS
    if _PINNA_SOS is None:
        _PINNA_SOS = biquad_peak(7000.0, -2.8, 2.5)


def _ensure_er_hp():
    global _ER_HP
    if _ER_HP is None:
        _ER_HP = butter(2, 180.0, btype='high', fs=SR, output='sos')


_XOVER = None


def _ensure_xover():
    global _XOVER
    if _XOVER is None:
        _XOVER = (butter(2, 500.0, btype='low', fs=SR, output='sos'),
                  butter(2, 500.0, btype='high', fs=SR, output='sos'))


def apply_hrtf(mono, az_deg, length=64, pinna=False):
    if abs(az_deg) < 0.5:
        return mono.copy(), mono.copy()
    _ensure_xover()
    az = np.radians(az_deg)
    sa = abs(np.sin(az))
    itd = abs(HEAD_R / C_SOUND * (az + np.sin(az)))

    # frequency-dependent ILD: low freqs have ~40% of high-freq ILD
    gn_hi = 10 ** (abs(1.5 * np.sin(az)) / 20)
    gf_hi = 10 ** (-abs(1.5 * np.sin(az)) / 20)
    gn_lo = 1.0 + (gn_hi - 1.0) * 0.4
    gf_lo = 1.0 + (gf_hi - 1.0) * 0.4

    lo = sosfilt(_XOVER[0], mono)
    hi = sosfilt(_XOVER[1], mono)

    def _make_ir(gn, gf):
        near, far = np.zeros(length), np.zeros(length)
        near[0] = 1.0
        ds = itd * SR
        d0 = min(int(ds), length - 2)
        frac = ds - d0
        far[d0] = 1 - frac
        if d0 + 1 < length:
            far[d0 + 1] = frac
        fc = 8000 - 3500 * sa
        if sa > 0.05 and fc < SR * 0.45:
            far = sosfilt(butter(1, fc, btype='low', fs=SR, output='sos'), far)
        near *= gn
        far *= gf
        if pinna and sa > 0.1:
            _ensure_pinna()
            far = sosfilt(_PINNA_SOS, far)
        return (far, near) if az_deg >= 0 else (near, far)

    li_hi, ri_hi = _make_ir(gn_hi, gf_hi)
    li_lo, ri_lo = _make_ir(gn_lo, gf_lo)
    nn = len(mono)
    l = fftconvolve(hi, li_hi, mode='full')[:nn] + fftconvolve(lo, li_lo, mode='full')[:nn]
    r = fftconvolve(hi, ri_hi, mode='full')[:nn] + fftconvolve(lo, ri_lo, mode='full')[:nn]
    return l, r


def early_reflections(ml, mr, immersive=False, pinna=False):
    """Early reflections.  In immersive mode uses more taps and adds
    ceiling reflections with high-shelf roll-off to simulate elevation."""
    global _CEILING_SOS
    _ensure_er_hp()
    taps = _ER_TAPS_IMMERSIVE if immersive else _ER_TAPS
    use_pinna = pinna or immersive  # immersive always uses pinna
    if immersive and _CEILING_SOS is None:
        _CEILING_SOS = biquad_high_shelf(4000.0, -3.0)
    mono = (ml + mr) * 0.5
    n = len(ml)
    ol, or_ = np.zeros(n), np.zeros(n)
    for i, (delay, gain, az) in enumerate(taps):
        l, r = apply_hrtf(mono, az, pinna=use_pinna)
        if immersive and i % 2 == 1:
            l = sosfilt(_CEILING_SOS, l)
            r = sosfilt(_CEILING_SOS, r)
        d = int(delay * SR)
        if d < n:
            ol[d:] += l[:n - d] * gain
            or_[d:] += r[:n - d] * gain
    return sosfilt(_ER_HP, ol), sosfilt(_ER_HP, or_)

_XFEED_LP = None


def binaural_enhance(ml, mr):
    """Cross-feed: feeds a delayed, LP-filtered version of each channel
    into the opposite at -20 dB.  Simulates natural speaker cross-talk that
    headphones lack, reducing extreme L/R separation and listener fatigue
    while preserving the spatial image created by HRTF panning."""
    global _XFEED_LP
    if _XFEED_LP is None:
        _XFEED_LP = butter(2, 1200.0, btype='low', fs=SR, output='sos')
    n = len(ml)
    delay = int(0.0003 * SR)  # 0.3ms ITD-like delay
    gain = 10 ** (-20.0 / 20.0)
    # cross-feed: LP-filtered + delayed opposite channel
    xl = np.zeros(n)
    xr = np.zeros(n)
    xl[delay:] = sosfilt(_XFEED_LP, mr)[:-delay or None] * gain
    xr[delay:] = sosfilt(_XFEED_LP, ml)[:-delay or None] * gain
    # ADD cross-feed — blends a hint of opposite channel for speaker-like
    # imaging, rather than subtracting (which would widen unnaturally).
    ol = ml + xl
    or_ = mr + xr
    # preserve RMS so the cross-feed doesn't change perceived loudness
    rms_in = np.sqrt(np.mean(ml**2 + mr**2))
    rms_out = np.sqrt(np.mean(ol**2 + or_**2)) + 1e-10
    scale = rms_in / rms_out
    return ol * scale, or_ * scale