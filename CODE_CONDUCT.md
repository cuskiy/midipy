# Code Conduct

*This document is for reference only — not a mandatory checklist.*

Lessons learned from developing this MIDI synthesizer, distilled into
principles for diagnosing and preventing audio quality issues.

## Root cause thinking

**If adjusting parameters doesn't fix it, the engine is wrong.**
The contrabass at 31–39 Hz was unfixable with additive synthesis (stacked
sines can't produce rich spectra at sub-bass frequencies). Switching to
the KS physical model solved it instantly. Three rounds of rolloff/bright
tuning were wasted effort.

**Measure the interaction, not the individual.**
A single lead track sounded fine. Five stacked lead tracks sounded
terrible — because their phases were identical, creating 5× peak
amplitude that crushed the compressor. The bug was invisible in
single-track testing.

## Multi-track phase coherence

Any synthesizer that may be instantiated multiple times on the same
pitch MUST produce different waveforms per instance. Specifically:

- The waveform's **phase** must depend on `nid` (note/instance ID).
- Failing this, N identical tracks sum to N× peak (not √N×), forcing
  the compressor to destroy the dynamics.

Engines and their decorrelation mechanisms:

| Engine   | Method                                        |
|----------|-----------------------------------------------|
| Additive | `ph_nid = nid * 2.399` added to phase accumulator |
| FM       | Same `nid * 2.399` on carrier phase           |
| Supersaw | `nid * 97.0` in polyBLEP starting phase       |
| KS       | Sub-period output delay (`nid * period / 5.3`) |

## CC processing

- **Zero-order hold** (step function) for all CC values — GM standard
  says values take effect immediately. Linear interpolation between
  sparse CC events creates unintended ramps.
- **GM defaults** when the first CC event is late (> 100 ms from t=0).
- **Smoothing** is applied downstream, not in interpolation.

## Testing methodology

1. **Use real MIDI files**, not just synthetic test patterns. Real
   arrangements expose edge cases (extreme registers, multi-track
   unison, aggressive CC automation) that synthetic tests miss.

2. **Multi-track stacking test**: render the same note with nid=0..4,
   sum the results, check that combined peak < 3.5× individual peak.

3. **Extreme register test**: every instrument at its lowest and highest
   reasonable MIDI notes. Check that harmonic content reaches above
   the fundamental (especially for notes < 100 Hz).

4. **Full-pipeline render**: don't test synth engines in isolation.
   The render pipeline (normalisation → CC → compression → EQ) can
   amplify or mask issues that are invisible at the engine level.

## Common pitfalls

- **`sosfiltfilt` doubles dB**: zero-phase filtering applies the filter
  twice. A "+1 dB" parameter produces +2 dB effective boost.

- **Deterministic phase = correlation**: if two instances produce the
  same waveform, they're correlated. RNG-seeded noise doesn't help
  if the core oscillator phase is identical.

- **KS steady-state convergence**: KS delay lines converge to the same
  waveform regardless of initial excitation. Phase decorrelation
  requires post-synthesis delay shifting, not excitation variation.

## Versioning

Each change set produces a new version file (v140, v141, v142, ...).
Never overwrite a released version — always increment.
