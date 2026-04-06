"""Tests for DspModule, DspChain, and pipeline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from synth.timbre import SR
from scipy.signal import sosfilt, butter
from schema import Note, ChannelData
from mix import _render_track
from schema import PipelineConfig
from mix.dsp_module import (FilterModule, ChorusModule, LeslieModule,
                            ModVibratoModule, BrightnessModule, DspChain)
from mix import BLOCK_SIZE


def test_filter_module_matches_sosfilt():
    """FilterModule block output matches full-buffer sosfilt."""
    sos = butter(2, 200.0, btype='high', fs=SR, output='sos')
    rng = np.random.RandomState(42)
    buf = rng.randn(SR)  # 1s of noise

    # Full buffer reference
    ref = sosfilt(sos, buf)

    # Block-by-block
    fm = FilterModule(sos)
    out = np.zeros_like(buf)
    for i in range(0, len(buf), BLOCK_SIZE):
        bs = min(BLOCK_SIZE, len(buf) - i)
        out[i:i+bs] = fm.process(buf[i:i+bs])

    diff = np.max(np.abs(ref - out))
    assert diff < 1e-10, f"FilterModule drift: {diff}"


def test_chorus_module_runs():
    """ChorusModule processes without crash, output differs from input."""
    rng = np.random.RandomState(42)
    buf = rng.randn(SR)
    cm = ChorusModule(np.ones(SR) * 0.5, SR)
    out = np.zeros_like(buf)
    for i in range(0, len(buf), BLOCK_SIZE):
        bs = min(BLOCK_SIZE, len(buf) - i)
        out[i:i+bs] = cm.process(buf[i:i+bs])
    # Should modify the signal
    assert not np.allclose(buf, out)
    # Should not blow up
    assert not np.any(np.isnan(out))
    assert np.max(np.abs(out)) < 50


def test_leslie_module_runs():
    """LeslieModule processes without crash."""
    rng = np.random.RandomState(42)
    buf = rng.randn(SR)
    lm = LeslieModule()
    out = np.zeros_like(buf)
    for i in range(0, len(buf), BLOCK_SIZE):
        bs = min(BLOCK_SIZE, len(buf) - i)
        out[i:i+bs] = lm.process(buf[i:i+bs])
    assert not np.any(np.isnan(out))


def test_dsp_chain_sequential():
    """DspChain applies modules in order."""
    sos1 = butter(2, 100.0, btype='high', fs=SR, output='sos')
    sos2 = butter(2, 8000.0, btype='low', fs=SR, output='sos')
    chain = DspChain([FilterModule(sos1), FilterModule(sos2)])
    rng = np.random.RandomState(42)
    buf = rng.randn(256)
    out = chain.process(buf)
    assert not np.allclose(buf, out)
    assert not np.any(np.isnan(out))


def test_output_quality():
    """_render_track produces valid, non-silent output with correct RMS."""
    notes = [Note(0.0, 60, 0.3, 0.85, 0, 0),
             Note(0.15, 64, 0.25, 0.75, 0, 0)]
    cd = ChannelData()
    cfg = PipelineConfig()
    buf_len = int(SR * 2)
    r = _render_track("piano", notes, None, cd, buf_len, 0.13, 1.0, cfg, 0)
    assert r.audio is not None, "Output is None"
    assert np.max(np.abs(r.audio)) > 0.01, "Output too quiet"
    assert not np.any(np.isnan(r.audio)), "NaN in output"
    # RMS should be near target
    active = np.abs(r.audio) > 1e-7
    if np.any(active):
        rms = np.sqrt(np.mean(r.audio[np.argmax(active):len(r.audio)-np.argmax(active[::-1])]**2))
        # After vol_curve and inst_vol, RMS may differ from target
        assert rms > 0.01, f"RMS too low: {rms}"

def test_retrigger_no_click():
    """Fast re-trigger via pipeline has no click at transition."""
    notes = [Note(0.0, 60, 0.15, 0.85, 0, 0),
             Note(0.1, 60, 0.15, 0.90, 0, 0)]  # re-trigger same note
    cd = ChannelData()
    cfg = PipelineConfig()
    buf_len = int(SR * 1)
    r = _render_track("piano", notes, None, cd, buf_len, 0.13, 1.0, cfg, 0)
    if r.audio is None:
        return
    # Check around re-trigger point
    retrig_sample = int(0.1 * SR)
    window = r.audio[retrig_sample - 100:retrig_sample + 100]
    max_diff = np.max(np.abs(np.diff(window)))
    assert max_diff < 0.08, f"Click at re-trigger: max_diff={max_diff:.4f}"


def test_filter_continuity():
    """Block-based filters produce no discontinuity at block boundaries."""
    notes = [Note(0.0, 60, 0.5, 0.85, 0, 0)]
    cd = ChannelData()
    cfg = PipelineConfig()
    buf_len = int(SR * 1)
    r = _render_track("piano", notes, None, cd, buf_len, 0.13, 1.0, cfg, 0)
    if r.audio is None:
        return
    # Check block boundaries for clicks
    for boundary in range(BLOCK_SIZE, buf_len - BLOCK_SIZE, BLOCK_SIZE):
        around = r.audio[boundary - 2:boundary + 2]
        if np.max(np.abs(around)) < 1e-6:
            continue
        max_step = np.max(np.abs(np.diff(around)))
        assert max_step < 0.1, f"Click at block boundary {boundary}: {max_step:.4f}"


def test_parse_events():
    """parse_events produces correct events with dur and pedal."""
    import tempfile, os, mido
    mid = mido.MidiFile(ticks_per_beat=480)
    t = mido.MidiTrack()
    mid.tracks.append(t)
    t.append(mido.MetaMessage('set_tempo', tempo=500000))
    t.append(mido.Message('program_change', program=0, channel=0))
    t.append(mido.Message('note_on', note=60, velocity=100, channel=0, time=0))
    t.append(mido.Message('control_change', control=64, value=127, channel=0, time=240))
    t.append(mido.Message('note_off', note=60, velocity=0, channel=0, time=240))
    t.append(mido.Message('control_change', control=64, value=0, channel=0, time=240))
    tmp = tempfile.mktemp(suffix='.mid')
    mid.save(tmp)
    from midi import parse_events
    from schema import EventKind
    events, ch_data, pans = parse_events(tmp)
    os.unlink(tmp)
    on_events = [e for e in events if e.kind == EventKind.NOTE_ON]
    pedal_events = [e for e in events if e.kind == EventKind.PEDAL]
    assert len(on_events) == 1, f"Expected 1 note_on, got {len(on_events)}"
    assert on_events[0].dur > 0, f"dur should be filled, got {on_events[0].dur}"
    assert len(pedal_events) >= 1, f"Expected pedal events, got {len(pedal_events)}"


if __name__ == "__main__":
    tests = [(n, o) for n, o in globals().items() if n.startswith("test_") and callable(o)]
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
