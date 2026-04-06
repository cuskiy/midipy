"""Tests for Voice and VoiceManager (items 23-24)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from synth.timbre import SR
from synth.instruments import TIMBRES
from synth.voice import Voice, VoiceManager, ACTIVE, RELEASING


def test_voice_lifecycle():
    """Voice: trigger → read → done (deferred rendering)."""
    v = Voice()
    tim = TIMBRES["piano"]
    v.key = (0, 60)
    v.trigger(261.63, 0.3, 0.85, tim, "piano", 0, 0)
    assert v.state == ACTIVE
    # Waveform is deferred until first read
    assert v.waveform is None

    total = 0
    while not v.is_done():
        chunk = v.read(64)
        if chunk is None:
            break
        total += len(chunk)
    assert v.is_done()
    assert v.waveform is not None
    assert total == len(v.waveform)


def test_voice_release_fade():
    """Voice: early release applies fade, no click."""
    v = Voice()
    tim = TIMBRES["strings"]
    v.key = (0, 60)
    v.trigger(261.63, 2.0, 0.8, tim, "strings", 0, 0)
    # Read partway
    v.read(int(SR * 0.5))
    pos_before = v.pos
    v.release()
    assert v.state == RELEASING
    # Check fade applied (waveform should decay smoothly)
    seg = v.waveform[pos_before:pos_before + int(SR * 0.1)]
    diffs = np.abs(np.diff(seg))
    assert np.max(diffs) < 0.1, f"Click at release: max diff = {np.max(diffs)}"


def test_voice_read_returns_none_when_done():
    """Voice: read after done returns None."""
    v = Voice()
    v.trigger(440.0, 0.01, 0.5, TIMBRES["piano"], "piano", 0, 0)
    while not v.is_done():
        v.read(1024)
    assert v.read(64) is None


def test_voicemanager_retrigger():
    """Re-trigger same key: old voice releases, both active."""
    mgr = VoiceManager()
    v1 = mgr.note_on(0, 0, 60, 0.8, 0, 0.5)
    assert v1.state == ACTIVE
    assert len(mgr.active) == 1

    v2 = mgr.note_on(int(SR * 0.1), 0, 60, 0.9, 0, 0.5)
    assert v1.state == RELEASING
    assert v2.state == ACTIVE
    assert len(mgr.active) == 2


def test_voicemanager_pedal():
    """Sustain pedal holds voices, releases on pedal off."""
    mgr = VoiceManager()
    mgr.pedal_change(0, True)
    v1 = mgr.note_on(0, 0, 60, 0.8, 0, 0.5)
    mgr.note_off(0, 60)
    assert v1.state == ACTIVE, "Pedal should prevent release"
    assert 0 in mgr.held and len(mgr.held[0]) == 1

    mgr.pedal_change(0, False)
    assert v1.state == RELEASING, "Pedal off should trigger release"
    assert len(mgr.held.get(0, [])) == 0


def test_voicemanager_stealing():
    """Voice stealing when exceeding max polyphony."""
    mgr = VoiceManager(max_poly=4)
    voices = []
    for i in range(5):
        v = mgr.note_on(i * 100, 0, 60 + i, 0.8, 0, 0.5)
        voices.append(v)
    # Should have stolen the oldest
    alive = [v for v in mgr.active if not v.is_done()]
    assert len(alive) <= 4


def test_voicemanager_render_block():
    """render_block produces correct output."""
    mgr = VoiceManager()
    mgr.note_on(0, 0, 60, 0.85, 0, 0.3)
    buf = np.zeros(256)
    mgr.render_block(buf, 0, 256)
    assert np.max(np.abs(buf)) > 0.001, "render_block should produce audio"


def test_voicemanager_drum():
    """Drums play to completion regardless of note_off."""
    mgr = VoiceManager()
    v = mgr.note_on(0, 9, 36, 0.9, 0, 0.1)  # kick drum
    mgr.note_off(9, 36)
    # Drum voice continues (short waveform plays out)
    buf = np.zeros(int(SR * 0.5))
    mgr.render_block(buf, 0, len(buf))
    assert np.max(np.abs(buf)) > 0.001


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
