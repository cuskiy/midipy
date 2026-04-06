"""Supersaw lead synthesizer.

5-voice detuned sawtooth ensemble with deterministic phase,
low-shelf EQ, presence boost, and 2x oversampling.

Multiple slightly detuned saws create rich, smooth ensemble sound.
A single raw saw sounds buzzy because each waveform cycle has a sharp
reset discontinuity.  Multi-voice smooths this.
"""
import math
import numpy as np
from scipy.signal import sosfilt, resample_poly, butter
from .dsp import BoundedCache
from .timbre import SR, vc, biquad_low_shelf, biquad_peak
from .envelope import envelope

TWO_PI = 2.0 * math.pi
_OS_FREQ = 200.0
_OS = 2

# 5-voice unison: detuning in cents and amplitude weights.
# Center voice dominant; outer voices add width and shimmer.
_DETUNE_CENTS = [-6.0, -3.0, 0.0, 3.0, 6.0]
_DETUNE_AMPS  = [0.15, 0.22, 0.26, 0.22, 0.15]
_N_VOICES = len(_DETUNE_CENTS)
_VOICE_NORM = 1.0 / math.sqrt(sum(a*a for a in _DETUNE_AMPS))

_FILTER_CACHE = BoundedCache(32)


def _get_cached(name, build_fn):
    if name not in _FILTER_CACHE:
        _FILTER_CACHE[name] = build_fn()
    return _FILTER_CACHE[name]

def _get_hp30():
    return _get_cached("hp30", lambda: butter(2, 30.0, btype='high',
                                               fs=SR, output='sos'))
def _get_lp(fc):
    fc = max(200.0, min(fc, SR * 0.45))
    key = f"lp_{int(fc)}"
    return _get_cached(key, lambda: butter(2, fc, btype='low',
                                            fs=SR, output='sos'))
def _get_low_shelf():
    return _get_cached("loshelf", lambda: biquad_low_shelf(180.0, -2.0))

def _get_presence():
    return _get_cached("presence", lambda: biquad_peak(3500.0, 1.0, 1.2))


def _saw(freq, n, sr, phase=0.0):
    """Band-limited sawtooth via polyBLEP."""
    dt = freq / sr
    ph = (phase / TWO_PI + dt * np.arange(n, dtype=np.float64)) % 1.0
    out = 2.0 * ph - 1.0
    m = ph < dt
    t = ph[m] / dt
    out[m] -= t + t - t * t - 1.0
    m = ph > 1.0 - dt
    t = (ph[m] - 1.0) / dt
    out[m] -= t * t + t + t + 1.0
    return out


def _saw_pb(freq, n, pb_cents, sr, phase=0.0):
    """Band-limited sawtooth with per-sample pitch bend (cents)."""
    dt = freq * 2.0 ** (pb_cents / 1200.0) / sr
    ph = (phase / TWO_PI + np.cumsum(dt)) % 1.0
    out = 2.0 * ph - 1.0
    m = ph < dt
    t = ph[m] / dt[m]
    out[m] -= t + t - t * t - 1.0
    m = ph > 1.0 - dt
    t = (ph[m] - 1.0) / dt[m]
    out[m] -= t * t + t + t + 1.0
    return out


def synthesize_lead(freq: float, dur: float, vel: float, tim, name: str = "lead", nid: int = 0, pb_curve=None) -> 'np.ndarray':
    """Render one supersaw lead note — 5-voice detuned ensemble."""
    tail = min(tim.rel * 1.5 + 0.1, 2.0)
    td = dur + tail
    n = int(SR * td)
    if n == 0:
        return np.zeros(0)

    do_os = freq >= _OS_FREQ
    osr = _OS if do_os else 1
    sr_w = SR * osr
    n_w = n * osr

    tv = np.linspace(0, td, n_w, endpoint=False)
    v = vc(vel)

    # Pitch modulation (cents) — shared by all voices
    pm = np.zeros(n_w)
    if tim.vib_d > 0:
        onset = np.clip((tv - tim.vib_del) / 0.2, 0, 1)
        pm += onset * tim.vib_d * np.sin(TWO_PI * tim.vib_r * tv)
    if pb_curve is not None:
        pb = pb_curve[:n] * 100.0 if len(pb_curve) >= n else \
             np.pad(pb_curve, (0, n - len(pb_curve)), mode='edge') * 100.0
        if do_os:
            pb = np.repeat(pb, osr)[:n_w]
        pm += pb[:n_w]

    has_pm = np.max(np.abs(pm)) > 0.05

    # ── 5-voice detuned ensemble ─────────────────────────────────────
    wave = np.zeros(n_w)
    for det_cents, amp in zip(_DETUNE_CENTS, _DETUNE_AMPS):
        voice_freq = freq * 2.0 ** (det_cents / 1200.0)
        # Deterministic phase per voice — different nid values (from
        # different sub-tracks) produce different phases so multi-track
        # unison doesn't sum perfectly in-phase (which would create
        # N× peak amplitude and trigger aggressive compression).
        phase0 = ((freq * 137.0 + det_cents * 53.0 + nid * 97.0) % TWO_PI)

        if has_pm:
            # Add voice detuning to the shared pitch modulation
            voice_pm = pm + det_cents
            wave += amp * _saw_pb(freq, n_w, voice_pm, sr_w, phase0)
        else:
            wave += amp * _saw(voice_freq, n_w, sr_w, phase0)

    wave *= _VOICE_NORM

    # Downsample
    if do_os:
        wave = resample_poly(wave, 1, osr)[:n]

    # ── Filter chain ─────────────────────────────────────────────────
    # 1. Subsonic HP (DC removal only)
    wave = sosfilt(_get_hp30(), wave)

    # 2. LP — generous cutoff
    fc = (tim.fc_base if tim.fc_base > 0 else 8000.0)
    fc *= tim.fc_min + (1 - tim.fc_min) * v
    if tim.spec_b > 0:
        fc *= 1.0 + tim.spec_b * v
    fc = max(500.0, min(fc, SR * 0.45))
    wave = sosfilt(_get_lp(fc), wave)

    # 3. Low-shelf -4 dB @ 180 Hz
    shelf = _get_low_shelf()
    if shelf is not None:
        wave = sosfilt(shelf, wave)

    # 4. Presence +2.5 dB @ 3.5 kHz
    pres = _get_presence()
    if pres is not None:
        wave = sosfilt(pres, wave)

    # 5. Soft saturation
    if tim.drive > 0:
        d = tim.drive * (1 + 0.3 * v)
        td_v = math.tanh(d)
        if td_v > 1e-6:
            wave = np.tanh(wave * d) / td_v

    # ── Envelope ─────────────────────────────────────────────────────
    noff = int(SR * dur)
    env = envelope(tim, dur, n, vel, freq, noff)
    out = wave * env * (0.15 + 0.85 * v)
    out -= np.mean(out)

    # Smooth cosine onset (2ms)
    fade = min(int(SR * 0.002), n)
    if fade > 1:
        out[:fade] *= 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, fade)))
    return out
