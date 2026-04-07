"""Centralized constants shared across modules.

Gathers hard-coded values that were previously scattered in midi.py,
mix/__init__.py, and synth/ so they can be tuned from one place.
"""

# ── MIDI parser ──────────────────────────────────────────────────────
MAX_PEDAL_SUSTAIN: float = 5.0          # seconds; cap pedal-held note dur

# ── Mix pipeline ─────────────────────────────────────────────────────
CC_FLOOR: float = 0.02                  # minimum CC multiplier (avoids silence)
CC_SCALE: float = 0.98                  # = 1 - CC_FLOOR
BLOCK_SIZE: int = 4096                  # used by tests for block-continuity

# ── Sympathetic resonance (piano-family) ─────────────────────────────
SYMPA_GAIN: float = 0.00022
SYMPA_MAX_H: int = 8                    # max harmonic order to consider
SYMPA_TOLERANCE: float = 0.006          # relative freq tolerance for resonance
SYMPA_MAX_NOTES: int = 40               # limit for O(n²) interaction search

# ── Parallel rendering ───────────────────────────────────────────────
PARALLEL_MIN_TRACKS: int = 3            # only parallelise if >= this many tracks
