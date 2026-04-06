"""Audio quality diagnostics — fast, focused on real issues.

Runs in <15s. Tests:
  1. Every instrument renders without NaN/Inf/clipping
  2. Low-register harmonic content (catches "snoring" timbres)
  3. Multi-track phase correlation (catches N× stacking bug)
  4. CC sidechain smoothness

Usage:  python test/test_audio_quality.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from synth.timbre import SR
from synth.instruments import TIMBRES
from synth import synthesize
from synth.drums import drum, _MAP

ISSUES = []

def issue(sev, inst, detail):
    ISSUES.append((sev, inst, detail))
    mark = {"CRIT": "✗", "WARN": "⚠"}[sev]
    print(f"  {mark} {inst}: {detail}")

# ── 1. Render sanity ─────────────────────────────────────────────────
print("1. Render sanity")
for name, tim in TIMBRES.items():
    freq = 200.0 if name == "sfx" else (65.0 if name in ("contrabass","bass") else 262.0)
    try:
        w = synthesize(freq, 0.3, 0.85, tim, name, 0)
        if np.any(np.isnan(w)) or np.any(np.isinf(w)):
            issue("CRIT", name, "NaN/Inf")
        elif np.max(np.abs(w)) > 1.001:
            issue("WARN", name, f"peak={np.max(np.abs(w)):.3f}>1")
    except Exception as e:
        issue("CRIT", name, f"crash: {e}")
rng = np.random.RandomState(42)
for note in sorted(_MAP.keys()):
    w = drum(note, 0.3, 0.9, rng)
    if np.any(np.isnan(w)):
        issue("CRIT", f"drum/{note}", "NaN")
print(f"   {len(TIMBRES)} instruments + {len(_MAP)} drums ✓")

# ── 2. Low-register harmonics ────────────────────────────────────────
print("2. Low-register harmonics")
for name, midi_note in [("contrabass",24),("contrabass",30),("pad",36),("cello",36)]:
    freq = 440.0 * 2**((midi_note-69)/12)
    w = synthesize(freq, 0.8, 0.88, TIMBRES[name], name, 0)
    seg = w[int(SR*0.05):int(SR*0.5)]
    if len(seg) < 512: continue
    fft = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1.0/SR)
    db = 20*np.log10(fft+1e-15); db -= np.max(db)
    mid = (freqs >= 600) & (freqs < 2000)
    mid_pk = np.max(db[mid]) if np.any(mid) else -999
    ok = mid_pk > -40
    print(f"   {'✓' if ok else '⚠'} {name}/n{midi_note} ({freq:.0f}Hz): mid={mid_pk:+.0f}dB")
    if not ok: issue("WARN", name, f"n{midi_note} mid peak {mid_pk:.0f}dB")

# ── 3. Phase correlation ─────────────────────────────────────────────
print("3. Phase correlation (3x stacking)")
for name in ["lead", "piano", "strings", "pad"]:
    waves = [synthesize(440.0, 0.3, 0.85, TIMBRES[name], name, nid) for nid in range(3)]
    n = min(len(w) for w in waves)
    comb = sum(w[:n] for w in waves)
    ratio = np.max(np.abs(comb)) / (np.mean([np.max(np.abs(w[:n])) for w in waves]) + 1e-10)
    ok = ratio < 2.5
    print(f"   {'✓' if ok else '⚠'} {name:12s}: {ratio:.2f}x")
    if not ok: issue("WARN", name, f"correlated stacking {ratio:.1f}x")

# ── 4. CC sidechain ──────────────────────────────────────────────────
print("4. CC sidechain")
from mix.cc import interp_cc, smooth_cc_sidechain
events = []
for i in range(4):
    events.append((i*0.5, 0.0))
    for j in range(1, 31):
        events.append((i*0.5 + 0.007*j, min(j/30, 1.0)))
vol = interp_cc(events, int(SR*2), default=1.0)
vc = smooth_cc_sidechain(0.02 + 0.98*np.sqrt(vol), down_ms=5.0, up_ms=50.0)
ms = np.max(np.abs(np.diff(vc)))
print(f"   {'✓' if ms < 0.01 else '⚠'} max step/sample: {ms:.5f}")

# ── Summary ──────────────────────────────────────────────────────────
crits = sum(1 for s,*_ in ISSUES if s == "CRIT")
warns = sum(1 for s,*_ in ISSUES if s == "WARN")
print(f"\nResult: {crits} critical, {warns} warnings")
for s, inst, detail in ISSUES:
    print(f"  {inst}: {detail}")
sys.exit(1 if crits > 0 else 0)
