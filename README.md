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
midi.py            MIDI parser (notes, CCs, pitch bend, RPN)
synth/             Synthesis engines + instrument definitions
  voice.py         Voice lifecycle, polyphony management
  timbre.py        Timbre class (47 parameters), DSP utilities
  instruments.py   24 instrument presets
  gm.py            GM program routing, volume, panning, HP cutoffs
  additive.py      Additive synthesis (inline spectral envelope)
  fm.py            FM synthesis (multi-operator, oversampled)
  ks.py            Karplus-Strong physical model (LP-compensated)
  supersaw.py      5-voice detuned sawtooth ensemble
  inharmonic.py    Filtered-noise texture (3-band)
  drums.py         Percussion synthesis (55 GM drum sounds)
  envelope.py      ADSR envelope generator with liveness modulation
  __init__.py      Engine dispatcher + crossfade logic
mix/               Render pipeline + mastering
  dsp_module.py    Block-based DSP chain (HP, chorus, Leslie, vibrato)
  __init__.py      Track render loop, CC application, master chain
  cc.py            CC interpolation (zero-order hold), smoothing
  routing.py       Track splitting, merging, deduplication
  master.py        FDN reverb IR, compressor, master EQ
  spatial.py       HRTF, early reflections, binaural crossfeed
test/              Tests
  test_regression.py     14 unit/regression tests
  test_audio_quality.py  Spectral, stacking, CC diagnostics (~4s)
  gen_all.py             Render all instruments to FLAC
```

---


## Technical Reference

See [docs/TECHNICAL.md](docs/TECHNICAL.md) for synthesis engine details,
pipeline parameters, per-instrument settings, and GM compliance notes.

## Development

See `CODE_CONDUCT.md` for methodology and common pitfalls.
