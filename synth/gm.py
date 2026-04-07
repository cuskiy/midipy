"""GM routing: program map, panning, volume, HP cutoffs, engine routing."""

# ── Program → instrument name mapping ────────────────────────────────
PROGRAM_MAP = {}
for _r, _n in [
    (range(0, 4), "piano"), (range(4, 6), "epiano"),
    (range(6, 8), "harpsichord"),
    (range(8, 11), "celesta"), ([11], "vibes"),
    ([12, 13], "marimba"),
    ([14], "vibes"), ([15], "harp"),
    (range(16, 24), "organ"),
    ([24], "nylon"), ([25], "guitar"),
    (range(26, 32), "guitar"),
    (range(32, 38), "bass"), (range(38, 40), "synbass"),
    (range(40, 42), "strings"),
    ([42], "cello"), ([43], "contrabass"), (range(44, 46), "strings"),
    ([45], "pluck"), ([46], "harp"), ([47], "marimba"),
    (range(48, 52), "strings"),
    (range(52, 56), "choir"), (range(56, 64), "brass"),
    (range(64, 72), "woodwind"), (range(72, 76), "flute"),
    (range(76, 80), "woodwind"), (range(80, 88), "lead"),
    (range(88, 100), "pad"), (range(100, 104), "lead"),
    (range(104, 108), "guitar"),
    (range(108, 110), "celesta"), (range(110, 112), "vibes"),
    (range(112, 116), "marimba"), (range(116, 120), "pluck"),
    (range(120, 128), "sfx"),
]:
    for _p in _r:
        PROGRAM_MAP[_p] = _n

# ── Default stereo placement (degrees, + = right) ───────────────────
PANNING = {
    "piano": 0, "epiano": -12, "organ": 5, "harpsichord": -8,
    "guitar": -25, "nylon": -20, "bass": 0, "synbass": 0,
    "strings": 22, "cello": 15, "contrabass": 10,
    "brass": 30, "woodwind": -22, "flute": -18, "choir": 0, "celesta": 12,
    "vibes": -16, "marimba": -12, "harp": 18, "pad": 0, "lead": 10,
    "pluck": -14, "default": 0, "sfx": 0, "drums": 0,
}
_CENTER_LOCK = {"bass", "synbass", "pad", "drums", "sfx", "choir"}

# ── Per-instrument relative volume ───────────────────────────────────
VOLUME = {
    "piano": 0.70, "epiano": 1.0, "organ": 1.0, "harpsichord": 1.0,
    "guitar": 1.0, "nylon": 1.0, "bass": 0.88, "synbass": 0.65,
    "strings": 1.0, "cello": 1.0, "contrabass": 0.92,
    "brass": 1.0, "woodwind": 1.0, "flute": 1.0, "choir": 1.0, "celesta": 1.0,
    "vibes": 1.0, "marimba": 1.0, "harp": 1.0, "pad": 0.55, "lead": 0.90,
    "pluck": 1.0, "default": 1.0, "sfx": 0.55, "drums": 0.75,
}

# ── High-pass filter cutoff per instrument ───────────────────────────
HP_FREQ = {
    "drums": 30.0, "bass": 50.0, "synbass": 50.0,
    "piano": 35.0, "organ": 55.0,
    "cello": 40.0, "contrabass": 65.0, "pad": 55.0,
    "epiano": 30.0, "guitar": 55.0, "nylon": 55.0, "harp": 22.0,
    "harpsichord": 30.0, "pluck": 40.0, "default": 50.0, "sfx": 60.0,
    "strings": 45.0, "brass": 45.0, "woodwind": 65.0, "flute": 100.0,
    "choir": 55.0, "lead": 45.0,
    "celesta": 150.0, "vibes": 80.0, "marimba": 70.0,
}

# ── Per-instrument chorus depth cap ──────────────────────────────────
CHORUS_CAP = {
    "cello": 0.30, "strings": 0.40,
    "bass": 0.15, "synbass": 0.15,
    "drums": 0.0,
}

# ── Per-instrument reverb send scale ─────────────────────────────────
REVERB_SEND = {
    "bass": 0.40, "synbass": 0.30,
    "drums": 0.50, "lead": 0.60,
    "sfx": 0.45,
}

# ── Engine routing sets ──────────────────────────────────────────────
FM_INSTRUMENTS = {"epiano", "celesta", "vibes", "marimba"}

SFX_INSTRUMENTS = {"sfx"}
KS_PLUCKED = {"guitar", "nylon", "harp", "harpsichord"}
KS_ALWAYS = set()
KS_DUR_THRESHOLD = 0.8
SYMPA_INSTRUMENTS = {"piano", "harpsichord", "harp"}

# Precomputed KS/additive gain calibration
KS_GAIN = {
    "guitar": 1.5587, "harp": 1.6591,
    "harpsichord": 2.4288, "nylon": 0.9074,
}
