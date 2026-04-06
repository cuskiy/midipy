# synth/

Synthesis engines and instrument definitions.

- `timbre.py` — Timbre class (with field validation), shared DSP utilities
- `instruments.py` — All instrument presets
- `gm.py` — GM program routing, volume, panning, HP cutoffs
- `additive.py` — Additive synthesis with inline spectral envelope
- `fm.py` — FM synthesis for struck/metallic timbres
- `ks.py` — Karplus-Strong for plucked strings
- `inharmonic.py` — Inharmonic synthesis for SFX (irrational-ratio partials)
- `drums.py` — Percussion synthesis
- `envelope.py` — Amplitude envelope generator
- `__init__.py` — Dispatcher: engine selection, crossfade, peak cap
