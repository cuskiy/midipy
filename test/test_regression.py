"""Automated regression tests for synth and mix pipeline.

Run:  python -m pytest test/test_regression.py -v
  or: python test/test_regression.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from synth.timbre import SR, Timbre
from synth.instruments import TIMBRES
from synth.envelope import envelope
from synth import synthesize
from synth.drums import drum, _MAP
from synth.ks import synthesize_plucked
from mix.cc import interp_cc, smooth_cc, smooth_cc_sidechain, make_pb_curve


# ── Helpers ──────────────────────────────────────────────────────────

def freq_of(w, sr=SR):
    """Estimate fundamental frequency via FFT with parabolic peak interpolation."""
    n = len(w)
    if n < sr * 0.01:
        return 0.0
    start, end = n // 4, 3 * n // 4
    seg = w[start:end]
    seg = (seg - np.mean(seg)) * np.hanning(len(seg))
    fft = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
    # Ignore DC and very low bins
    fft[:3] = 0
    peak_bin = np.argmax(fft)
    if peak_bin < 1 or peak_bin >= len(fft) - 1:
        return freqs[peak_bin]
    # Parabolic interpolation for sub-bin accuracy
    a, b, c = fft[peak_bin - 1], fft[peak_bin], fft[peak_bin + 1]
    denom = a - 2 * b + c
    if abs(denom) < 1e-20:
        return freqs[peak_bin]
    offset = 0.5 * (a - c) / denom
    return (peak_bin + offset) * sr / len(seg)


def assert_close(a, b, tol, msg=""):
    diff = abs(a - b)
    assert diff <= tol, f"{msg}: {a} vs {b}, diff={diff} > tol={tol}"


# ── Test: Envelope shape ─────────────────────────────────────────────

def test_envelope_attack_peak():
    """Envelope should reach peak ~1.0 during attack."""
    tim = Timbre(att=0.01, d1=0.1, d1l=0.5, d2=2.0, rel=0.3)
    env = envelope(tim, 1.0, int(SR * 1.3), vel=1.0, freq=440.0, noff=int(SR))
    assert np.max(env) >= 0.95, f"Envelope peak too low: {np.max(env)}"
    assert np.max(env) <= 1.05, f"Envelope peak too high: {np.max(env)}"


def test_envelope_release_at_noff():
    """Envelope should start decaying at noff."""
    tim = Timbre(att=0.005, d1=0.05, d1l=0.8, d2=10.0, rel=0.2)
    noff = int(SR * 0.5)
    n = int(SR * 1.0)
    env = envelope(tim, 0.5, n, vel=0.8, freq=440.0, noff=noff)
    # Envelope just before noff should be higher than 200ms after
    level_before = env[noff - 1]
    level_after = env[min(noff + int(SR * 0.2), n - 1)]
    assert level_after < level_before * 0.5, (
        f"Release not effective: before={level_before:.3f}, after={level_after:.3f}")


def test_envelope_no_nan():
    """Envelope should never contain NaN for any instrument."""
    for name, tim in TIMBRES.items():
        for dur in [0.01, 0.1, 1.0]:
            for vel in [0.1, 0.5, 1.0]:
                n = int(SR * (dur + tim.rel + 0.5))
                env = envelope(tim, dur, n, vel, 440.0, int(SR * dur))
                assert not np.any(np.isnan(env)), f"{name} dur={dur} vel={vel}: NaN"


# ── Test: KS pitch accuracy ─────────────────────────────────────────

def test_ks_pitch_accuracy():
    """KS plucked string should produce correct pitch within 5 cents.

    Uses a clean timbre (no click/body noise) so autocorrelation isn't
    confused by inharmonic transient content.
    """
    # Minimal timbre: just the KS delay line output, no extras
    clean_tim = Timbre(ks_lp=0.60, ks_click=0.0, noise=0.0,
                       att=0.001, d1=0.05, d1l=0.2, d2=2.0, rel=0.2)
    for midi_note in [40, 52, 64, 76]:
        freq = 440.0 * 2 ** ((midi_note - 69) / 12.0)
        w = synthesize_plucked(freq, 0.8, 0.8, clean_tim, "guitar", 0)
        if len(w) < SR * 0.1:
            continue
        detected = freq_of(w)
        if detected < 20:
            continue
        cents_error = 1200 * np.log2(detected / freq) if detected > 0 else 999
        assert abs(cents_error) < 5.0, (
            f"MIDI {midi_note}: expected {freq:.1f}Hz, got {detected:.1f}Hz "
            f"({cents_error:+.1f} cents)")


# ── Test: CC interpolation ───────────────────────────────────────────

def test_interp_cc_empty():
    """No events → None."""
    assert interp_cc([], 44100) is None


def test_interp_cc_step_function():
    """CC values should be held (zero-order hold), not linearly interpolated."""
    events = [(0.0, 0.0), (1.0, 1.0)]
    curve = interp_cc(events, int(SR * 2), default=0.5)
    # Before transition: should be 0.0
    assert curve[int(0.5 * SR)] < 0.01, "Mid-hold should be ~0"
    # Just before transition
    assert curve[int(0.999 * SR)] < 0.01, "Just before step should be ~0"
    # Just after transition
    assert curve[int(1.001 * SR)] > 0.99, "Just after step should be ~1"


def test_interp_cc_late_event_default():
    """Late first event should use GM default, not first event value."""
    events = [(5.0, 0.3)]
    curve = interp_cc(events, int(SR * 10), default=100.0 / 127.0)
    gm_default = 100.0 / 127.0
    assert abs(curve[0] - gm_default) < 0.01, (
        f"t=0 should be GM default {gm_default:.3f}, got {curve[0]:.3f}")
    assert abs(curve[int(6.0 * SR)] - 0.3) < 0.01, (
        f"t=6 should be 0.3, got {curve[int(6.0 * SR)]:.3f}")


def test_interp_cc_early_event_uses_value():
    """Early first event should use the event's own value."""
    events = [(0.01, 0.0), (1.0, 1.0)]
    curve = interp_cc(events, int(SR * 2), default=0.787)
    assert curve[0] < 0.01, f"Should use first event value 0, got {curve[0]:.3f}"


# ── Test: Pitch bend curve ───────────────────────────────────────────

def test_pb_curve_none_when_flat():
    """PB curve should return None when all values near zero."""
    events = [(0.0, 0.0), (1.0, 0.005)]
    result = make_pb_curve(events, start=0.0, dur=1.0)
    assert result is None, "Should return None for near-zero PB"


def test_pb_curve_linear_interp():
    """PB curve should linearly interpolate (unlike CC step function)."""
    events = [(0.0, 0.0), (1.0, 2.0)]
    result = make_pb_curve(events, start=0.0, dur=1.0)
    assert result is not None
    mid_val = result[len(result) // 2]
    assert 0.8 < mid_val < 1.2, f"PB midpoint should be ~1.0, got {mid_val:.3f}"


# ── Test: Synthesize all instruments ─────────────────────────────────

def test_all_instruments_render():
    """Every instrument should produce valid, non-empty output."""
    for name, tim in TIMBRES.items():
        freq = 200.0 if name == "sfx" else 440.0
        w = synthesize(freq, 0.5, 0.8, tim, name, 0)
        assert len(w) > 0, f"{name}: zero-length output"
        assert not np.any(np.isnan(w)), f"{name}: NaN in output"
        assert not np.any(np.isinf(w)), f"{name}: Inf in output"
        pk = np.max(np.abs(w))
        assert pk <= 1.001, f"{name}: peak {pk} exceeds 1.0"
        assert pk > 0.01, f"{name}: peak {pk} too low"


def test_all_drums_render():
    """All mapped drums should produce valid output."""
    rng = np.random.RandomState(42)
    for note in sorted(_MAP.keys()):
        w = drum(note, 0.3, 0.9, rng)
        assert len(w) > 0, f"drum {note}: zero-length"
        assert not np.any(np.isnan(w)), f"drum {note}: NaN"
        pk = np.max(np.abs(w))
        assert pk <= 1.001, f"drum {note}: peak {pk} > 1"


# ── Test: Timbre validation ──────────────────────────────────────────

def test_timbre_rejects_invalid():
    """Timbre should reject invalid field values."""
    for field, val, should_fail in [
        ("n", 0, True),
        ("att", -1, True),
        ("rel", -0.5, True),
        ("rolloff", -1, True),
        ("formant_freqs", "30000", True),
        ("n", 8, False),
        ("att", 0.01, False),
    ]:
        try:
            Timbre(**{field: val})
            if should_fail:
                assert False, f"Timbre({field}={val}) should have raised"
        except (ValueError, TypeError):
            if not should_fail:
                assert False, f"Timbre({field}={val}) should NOT have raised"


# ── Test: Cache clearing ─────────────────────────────────────────────

def test_clear_caches():
    """clear_caches() should not crash and should empty caches."""
    from mix import clear_caches, _IR_CACHE
    # Populate a cache first
    from synth.timbre import _BP_CACHE, get_bp
    get_bp(200, 4000)
    assert len(_BP_CACHE) > 0
    clear_caches()
    assert len(_BP_CACHE) == 0, "Cache not cleared"


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [(name, obj) for name, obj in globals().items()
             if name.startswith("test_") and callable(obj)]
    passed = failed = 0
    for name, fn in sorted(tests):
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
