# mix/

Rendering pipeline and master processing.

- `__init__.py` — Render loop: per-track synthesis → normalise → CC → HRTF → master chain → FLAC
- `cc.py` — CC interpolation (zero-order hold), smoothing, pitch bend curves
- `effects.py` — Chorus, Leslie, mod vibrato, sympathetic resonance
- `routing.py` — Track splitting, merging, deduplication
- `spatial.py` — Frequency-dependent HRTF, pinna notch, early reflections
- `master.py` — FDN reverb IR, stereo compressor, master EQ stages
