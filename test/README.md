# test/

- `test_regression.py` — Unit and regression tests (pitch accuracy, envelope shape, CC interpolation, Timbre validation). Run with `python test/test_regression.py` or `pytest`.
- `test_audio_quality.py` — Audio quality diagnostics: multi-track phase correlation, extreme register harmonics, dense polyphony, CC sidechain stress, full-pipeline render tests.
- `gen_all.py` — Renders every instrument through the full pipeline as individual FLAC files. Output goes to `test/samples/`.
