# Technical Reference

## Synthesis Engines

### Additive (`additive.py`)

Generates partials as individual sinusoids with per-partial amplitude,
decay, and spectral envelope.

- **Partials**: `n` base + `n_extra × lf(freq)` additional at low pitch
- **Spectral rolloff**: `amp(k) = (1/k)^rolloff × exp(-bright × (k-1))`
- **Inharmonicity**: `f(k) = freq × k × √(1 + B×k²)` where `B = inh × exp(inh_stretch × pitch_register)`
- **Velocity-dependent brightness**: `hammer_hard` controls nonlinear brightness curve; alternatively `vel_bright` provides linear scaling
- **Formant shaping**: Lorentzian resonances applied per-partial (not post-EQ), parameters: `formant_freqs`, `formant_gains`, `formant_qs`
- **Multi-string unison**: up to 3 strings with micro-detuning from `TUNE` table, pitch-scaled by `detune_scale(freq)`
- **Modulation**: `micro` controls per-partial frequency/amplitude jitter via shared LFOs; `bow_noise` and `breath_noise` add body AM
- **Phase decorrelation**: `ph_nid = nid × 2.399` ensures multi-track unison sums as √N× not N×
- **Nyquist guard**: cosine fade from 0.7×Nyquist to Nyquist prevents inter-sample harshness

Used for: piano, organ, strings, cello, brass, woodwind, flute, choir, pad, pluck, default.

### FM (`fm.py`)

Multi-operator FM with 2× oversampling, per-operator index envelopes.

- **Presets**: epiano (rhodes-like), celesta, vibes, marimba — each with custom operator ratios and index curves
- **Guard**: per-operator index limit to keep sidebands below Nyquist
- **Key tracking**: `kt = 1 - max(0, log2(freq/500)) × 0.25` reduces modulation index at high pitch
- **Micro-detuning**: 3-voice ensemble when `micro > 0.05`, with ±0.5 cents spread
- **Phase decorrelation**: same `nid × 2.399` mechanism as additive

Used for: epiano (prog 4-5), celesta (prog 8-10), vibes (prog 11,14), marimba (prog 12-13).

### Karplus-Strong (`ks.py`)

Physical model: delay line with LP feedback, 2× oversampled.

- **Pitch accuracy**: LP phase delay subtracted from delay line length; fractional delay via linear interpolation → sub-cent accuracy
- **Excitation**: triangle pulse + click noise, position controlled by `strike_pos`
- **Loss**: `loss = 10^(-3 / (freq × T60))` per sample, with `KS_MUTE` factor after note-off
- **Anti-harshness**: 1-pole LP at `12000 + freq×3` Hz post-synthesis
- **Phase decorrelation**: output shifted by `nid × period / 5.3` samples with fade-in

Used for: guitar (prog 24-31), bass (prog 32-37), contrabass (prog 43), nylon (prog 24), harp (prog 46), harpsichord (prog 6-7).

KS/additive crossfade: notes < 0.55s use pure KS; 0.55–0.95s blend KS+additive (equal-power); > 0.95s use additive with KS gain calibration.

### Supersaw (`supersaw.py`)

5-voice band-limited sawtooth (polyBLEP) with detuning.

- **Voices**: 5 at [-6, -3, 0, +3, +6] cents, amplitudes [0.15, 0.22, 0.26, 0.22, 0.15]
- **Oversampling**: 2× when `freq ≥ 200 Hz`
- **Filter chain**: HP 30Hz → LP (fc from timbre) → low-shelf -2dB@180Hz → presence +1dB@3.5kHz
- **Phase decorrelation**: `phase0 = (freq×137 + det_cents×53 + nid×97) % 2π`

Used for: lead (prog 80-87, 100-103).

### Filtered noise (`inharmonic.py`)

Three bandpass-filtered noise bands with independent decay.

- **Bands**: body (1×freq, BW=0.4, τ=1.2s), mid (2.5×freq, BW=0.8, τ=0.8s), air (6×freq, BW=1.5, τ=0.5s)
- **Pitch tracking**: median pitch bend shifts band centers

Used for: SFX (prog 120-127).

### Drums (`drums.py`)

Per-note synthesis: frequency sweep + noise + metallic modes.

- **55 sounds** mapped to GM notes 27–81 (kicks, snares, hi-hats, toms, cymbals, latin percussion)
- **Metallic modes**: inharmonic sine partials for hi-hats, cymbals, bells, triangle
- **Snare wires**: modulated bandpass noise at 4.5 kHz + 7.2 kHz
- **Kick sub-body**: fundamental + 1.5× sub-harmonic + beater click

---

## Envelope (`envelope.py`)

Four-stage ADSR with pitch/velocity scaling:

| Stage | Duration | Level | Curve |
|-------|----------|-------|-------|
| Attack | `att × (1 - va×vc)`, capped at `dur×0.25` | 0 → 1 | cosine+power blend |
| Decay 1 | `d1`, capped at `dur×0.4` | 1 → `d1l` | exponential |
| Sustain/Decay 2 | until note-off | `d1l` → decay | exp or dual-exp (`pr` blend ratio) |
| Release | after note-off | → 0 | exponential, rate from `rel` |

- **Liveness modulation**: `live > 0` adds 2-rate sinusoidal AM (2-4 Hz + 4.5-7.5 Hz) for natural variation
- **Cosine fade-out**: last 128 samples always cosine-faded to prevent clicks

---

## Timbre Parameters

The `Timbre` class holds 47 synthesis parameters. Key groups:

**Spectral shape**: `n`, `rolloff`, `bright`, `bright_lo`, `bright_hi`, `hammer_hard`, `vel_bright`, `even_atten`, `harm_amps`
**Strike/excitation**: `strike_pos`, `strike_depth`, `ks_lp`, `ks_click`
**Inharmonicity**: `inh`, `inh_stretch`
**Decay**: `b1` (linear), `b3` (quadratic), `rel_damp`, `coupling`, `decay_exp`
**Envelope**: `att`, `d1`, `d1l`, `d2`, `d2s`, `pr`, `rel`, `va`, `vd`, `ps`, `pdm`
**Modulation**: `vib_r`, `vib_d`, `vib_del`, `trem_r`, `trem_d`, `micro`, `live`
**Filtering**: `fc_base`, `fc_min`, `fc_alpha`, `hammer_p`, `spec_b`, `spec_tau`
**Voicing**: `n_strings`, `det_c`, `det_m`, `sub`, `sub_third`, `phantom`, `tw_leak`
**Noise**: `noise`, `noise_d`, `noise_peak`, `noise_hi`, `vn`, `bow_noise`, `breath_noise`
**Character**: `drive`, `lip_shape`, `key_click`
**Formants**: `formant_freqs`, `formant_gains`, `formant_qs` (comma-separated)

Validation on construction: `n ≥ 1`, `att ≥ 0`, `rel ≥ 0`, `rolloff ≥ 0`, formant frequencies < Nyquist.

---

## Render Pipeline

### Per-track chain

1. **Synthesize** each note (engine auto-selected by GM program)
2. **CC11 expression** (per-note, applied during render)
3. **Aftertouch** (+30% amplitude boost at max)
4. **CC74 brightness** (onset-sampled high-shelf ±4.5 dB @ 4 kHz)
5. **Accumulate** notes into track buffer
6. **Sympathetic resonance** (piano/harpsichord/harp: harmonic-matched partial excitation)
7. **HP filter** (per-instrument cutoff, 2nd order Butterworth)
8. **Chorus** (GM CC93: 2-voice modulated delay, RMS-compensated, per-instrument cap)
9. **Leslie** (organ only: horn/drum rotation simulation with Doppler)
10. **Mod vibrato** (CC1: pitch vibrato via modulated delay)
11. **RMS normalise** on DRY signal (before CC7)
12. **Instrument volume** scaling
13. **CC7 volume** (post-normalisation, zero-order hold, asymmetric smooth: 5ms attack / 50ms release)
14. **Peak cap** at 2.0
15. **Reverb send** (per-instrument scaling)
16. **HRTF pan** (frequency-dependent ILD + ITD, optional pinna notch)

### CC interpolation

Zero-order hold (step function) per GM standard. GM default value used when first event > 100ms from t=0. Downstream LP smoothing handles temporal response.

**CC7 volume curve**: `floor + scale × √(raw)` where floor=0.02, scale=0.98.

### Master chain

1. **HP** 20 Hz (3rd order Butterworth)
2. **Early reflections** (5 or 8 taps with HRTF, HP filtered)
3. **Reverb**: FDN convolution (8-line feedback delay network with Hadamard mixing, modulated delays, allpass diffusers)
4. **Bass tighten**: low-shelf -0.8 dB @ 120 Hz (eff. -1.6 dB via sosfiltfilt)
5. **Mud cut**: peak -1.5 dB @ 380 Hz, Q=0.45 (eff. -3.0 dB)
6. **Compress**: block-based envelope, soft knee (3 dB)
7. **Presence**: peak +0.5 dB @ 3.5 kHz, Q=0.7 (eff. +1.0 dB)
8. **Air**: high-shelf +0.75 dB @ 8 kHz (eff. +1.5 dB)
9. **Binaural crossfeed** (spatial mode: LP 1.2 kHz, -20 dB, 0.3ms delay)
10. **Peak normalise** to 0.89
11. **TPDF dither** to 24-bit

All master EQ uses `sosfiltfilt` (zero-phase, doubles effective dB).

### Pipeline parameters (stereo mode)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `target_rms` | 0.13 | Per-track RMS target (density-scaled: `÷ √(N/4)`) |
| `track_peak_cap` | 2.0 | Per-track peak limiter |
| `reverb_wet` | 0.055 | Reverb mix (dry = 0.945) |
| `reverb_rt60` | 1.6 s | Reverb decay time |
| `reverb_predelay` | 12 ms | Reverb pre-delay |
| `comp_thresh` | 0.30 | Compressor threshold |
| `comp_ratio` | 1.8:1 | Compression ratio |
| `comp_att_ms` | 40 ms | Compressor attack |
| `comp_rel_ms` | 400 ms | Compressor release |
| `comp_sc_hp` | 80 Hz | Sidechain HP filter |
| `peak_limit` | 0.89 | Final peak ceiling |
| `er_wet` | 0.035 | Early reflections level |

### Spatial mode overrides

| Parameter | Value |
|-----------|-------|
| `az_scale` | 1.15 |
| `er_wet` | 0.10 |
| `pinna` | enabled |
| `immersive_er` | enabled (8 taps + ceiling) |
| `reverb_wet` | 0.065 |
| `reverb_rt60` | 1.8 s |
| `reverb_predelay` | 18 ms |

---

## Per-Instrument Parameters

### Volume scaling

| Instrument | Volume | HP (Hz) | Pan (°) | Reverb send |
|------------|--------|---------|---------|-------------|
| piano | 0.95 | 22 | 0 | 1.0 |
| epiano | 1.00 | 30 | -12 | 1.0 |
| organ | 1.00 | 28 | +5 | 1.0 |
| harpsichord | 1.00 | 30 | -8 | 1.0 |
| guitar | 1.00 | 55 | -25 | 1.0 |
| nylon | 1.00 | 55 | -20 | 1.0 |
| bass | 0.88 | 30 | 0 | 0.40 |
| synbass | 0.65 | 30 | 0 | 0.30 |
| contrabass | 0.92 | 22 | +10 | 1.0 |
| strings | 1.00 | 45 | +22 | 1.0 |
| cello | 1.00 | 25 | +15 | 1.0 |
| brass | 1.00 | 45 | +30 | 1.0 |
| woodwind | 1.00 | 65 | -22 | 1.0 |
| flute | 1.00 | 100 | -18 | 1.0 |
| choir | 1.00 | 55 | 0 | 1.0 |
| lead | 0.90 | 45 | +10 | 0.60 |
| pad | 1.00 | 40 | 0 | 1.0 |
| celesta | 1.00 | 150 | +12 | 1.0 |
| vibes | 1.00 | 80 | -16 | 1.0 |
| marimba | 1.00 | 70 | -12 | 1.0 |
| harp | 1.00 | 22 | +18 | 1.0 |
| pluck | 1.00 | 40 | -14 | 1.0 |
| sfx | 0.55 | 60 | 0 | 0.45 |
| drums | 0.75 | 30 | 0 | 0.50 |

Center-locked (ignore pan CC): bass, synbass, pad, drums, sfx, choir.

### Engine routing (GM program numbers)

| Programs | Engine | Instrument |
|----------|--------|------------|
| 0–3 | Additive | piano |
| 4–5 | FM | epiano |
| 6–7 | KS | harpsichord |
| 8–10 | FM | celesta |
| 11, 14 | FM | vibes |
| 12–13 | FM | marimba |
| 15, 46 | KS | harp |
| 16–23 | Additive | organ |
| 24 | KS | nylon |
| 25–31 | KS | guitar |
| 32–37 | KS | bass |
| 38–39 | Additive | synbass |
| 40–41 | Additive | strings |
| 42 | Additive | cello |
| 43 | KS | contrabass |
| 44–45 | Additive | strings |
| 48–51 | Additive | strings |
| 52–55 | Additive | choir |
| 56–63 | Additive | brass |
| 64–71 | Additive | woodwind |
| 72–75 | Additive | flute |
| 76–79 | Additive | woodwind |
| 80–87 | Supersaw | lead |
| 88–99 | Additive | pad |
| 100–103 | Supersaw | lead |
| 104–107 | KS | guitar |
| 108–109 | FM | celesta |
| 110–111 | FM | vibes |
| 112–115 | FM | marimba |
| 116–119 | Additive | pluck |
| 120–127 | Noise | sfx |

---

## GM Compliance

### Supported CCs

| CC | Name | Implementation |
|----|------|----------------|
| 1 | Modulation | Pitch vibrato via modulated delay (5.5 Hz, max 50 cents) |
| 7 | Volume | Post-normalisation fader, zero-order hold, asymmetric smoothing |
| 10 | Pan | HRTF azimuth mapping (-55° to +55°) |
| 11 | Expression | Per-note amplitude scaling (artistic, pre-normalisation) |
| 64 | Sustain pedal | Note extension, capped at 5.0 s |
| 74 | Brightness | Onset-sampled high-shelf EQ (±4.5 dB @ 4 kHz) |
| 91 | Reverb send | Per-channel reverb depth scaling |
| 93 | Chorus send | 2-voice modulated delay, per-instrument cap |
| 100/101 | RPN LSB/MSB | Pitch bend range (RPN 0,0 → CC6 sets semitones) |
| 121 | Reset all | Clears RPN state |

### Pitch bend

Default range: ±2 semitones (adjustable via RPN 0,0).
Linear interpolation between events (correct for continuous PB).
Supported in all engines: additive, FM, KS (post-process), supersaw, SFX (median shift).

### Channel aftertouch

+30% amplitude boost at maximum pressure.

### Output format

24-bit FLAC, 44100 Hz, stereo. TPDF dither applied before quantisation.

---

## Tests

```
python test/test_regression.py       # 14 unit tests (~3s)
python test/test_audio_quality.py    # render + spectral diagnostics (~4s)
python test/gen_all.py               # full instrument showcase
python test/gen_all.py lead drums    # specific instruments only
```

See `CODE_CONDUCT.md` for development methodology.
