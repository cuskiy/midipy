# MIDI-to-FLAC Renderer

Offline synthesizer that renders Standard MIDI files to 24-bit/44.1 kHz FLAC.
No samples, no soundfonts — all audio is generated from mathematical models.

## Usage

```
python main.py input.mid                       # → input.flac
python main.py input.mid -o out.flac           # custom output
python main.py input.mid --spatial             # binaural HRTF mode
python main.py input.mid --stems              # per-instrument stems
python main.py input.mid --stems-dir ./stems  # custom stems directory
python main.py input.mid --start 30 --end 60  # partial render
```

## Requirements

```
pip install mido soundfile scipy numpy
```

## Architecture

```
main.py            Entry point, CLI
config.py          Centralized constants (pedal, CC, sympathetic, etc.)
schema.py          Shared data types (Note, ChannelData, PipelineConfig)
midi.py            MIDI parser (notes, CCs, pitch bend, RPN)
synth/             Synthesis engines + instrument definitions
  voice.py         Voice lifecycle, polyphony management
  timbre.py        Timbre class (47 parameters), DSP utilities
  instruments.py   24 instrument presets
  gm.py            GM program routing, volume, panning, HP cutoffs
  additive.py      Additive synthesis (inline spectral envelope)
  fm.py            FM synthesis (multi-operator, oversampled)
  ks.py            Karplus-Strong physical model (LP-compensated)
  inharmonic.py    Filtered-noise texture (3-band)
  drums.py         Percussion synthesis (55 GM drum sounds)
  envelope.py      ADSR envelope generator with liveness modulation
  dsp.py           Shared DSP: filters, BoundedCache
  __init__.py      Engine dispatcher + crossfade logic
mix/               Render pipeline + mastering
  track_render.py  Per-track synthesis, DSP chain, sympathetic resonance
  pipeline.py      Master chain, parallel dispatch, FLAC output
  dsp_module.py    Block-based DSP chain (HP, chorus, Leslie, vibrato)
  cc.py            CC interpolation (zero-order hold), smoothing
  routing.py       Track splitting, merging, deduplication
  master.py        FDN reverb IR, compressor, master EQ
  spatial.py       HRTF, early reflections, binaural crossfeed
  __init__.py      Re-exports (render, clear_caches)
test/              Tests
  test_regression.py     18 unit/regression tests
  test_pipeline.py       8 DSP module + pipeline tests
  test_voice.py          8 voice lifecycle tests
  test_audio_quality.py  Spectral, stacking, CC diagnostics
  gen_all.py             Render all instruments to FLAC
```

---


## Technical Reference

See [docs/TECHNICAL.md](docs/TECHNICAL.md) for synthesis engine details,
pipeline parameters, per-instrument settings, and GM compliance notes.

## Development

See `CODE_CONDUCT.md` for methodology and common pitfalls.
