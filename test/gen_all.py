"""Generate test audio for all instruments.

Section 1: Low→high ascending sweep (single notes)
Section 2: Short melodic passage (legato + intervals)
Section 3: Rapid-fire burst with chords
Section 4 (lead only): Multi-layer unison test — verifies deterministic
    phase coherence when multiple tracks play the same note.

Usage:
    python test/gen_all.py              # all instruments
    python test/gen_all.py lead piano   # specific instruments
"""
import mido, sys, os

BPM = 120
TPB = 480

def s2t(s):
    return int(s * TPB * BPM / 60)

INSTRUMENTS = [
    ("piano",       0, 36, 96),
    ("epiano",      4, 36, 96),
    ("harpsichord", 6, 36, 96),
    ("organ",      16, 36, 96),
    ("guitar",     27, 40, 88),
    ("nylon",      24, 40, 88),
    ("bass",       33, 28, 55),
    ("synbass",    38, 28, 96),
    ("strings",    48, 36, 84),
    ("cello",      42, 28, 72),
    ("harp",       46, 40, 88),
    ("choir",      52, 48, 84),
    ("brass",      56, 36, 80),
    ("woodwind",   64, 48, 96),
    ("flute",      73, 60, 96),
    ("lead",       81, 36, 96),
    ("pad",        89, 36, 84),
    ("pluck",     117, 40, 88),
    ("celesta",     8, 60, 108),
    ("vibes",      11, 48, 84),
    ("marimba",    12, 48, 84),
    ("sfx",       122, 48, 96),
]

DRUM_HITS = [36, 38, 42, 44, 46, 49, 51, 41, 43, 45, 47, 55, 57, 59]


def _make_instrument_midi(prog, lo, hi, path):
    mid = mido.MidiFile(ticks_per_beat=TPB)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(BPM), time=0))
    tr.append(mido.Message('program_change', program=prog, channel=0, time=0))

    rng = (hi - lo)
    mid_n = (lo + hi) // 2

    # === Section 1: Ascending sweep ===
    step = max(rng // 8, 2)
    gap = 0
    for nn in range(lo, hi + 1, step):
        vel = 70 + int(30 * (nn - lo) / max(rng, 1))
        tr.append(mido.Message('note_on', note=nn, velocity=min(vel, 127),
                               channel=0, time=gap))
        tr.append(mido.Message('note_off', note=nn, velocity=0,
                               channel=0, time=s2t(0.45)))
        gap = s2t(0.08)

    # === Section 2: Melodic passage ===
    base = mid_n - 7
    melody = [
        (base,     0.5,  80), (base+2,   0.3,  85), (base+4,   0.3,  90),
        (base+5,   0.5,  85), (base+7,   0.6,  95), (base+12,  0.4, 100),
        (base+11,  0.25, 90), (base+9,   0.25, 85), (base+7,   0.5,  80),
        (base+5,   0.8,  75),
    ]
    gap = s2t(0.25)
    for nn, dur, vel in melody:
        nn = max(lo, min(nn, hi))
        tr.append(mido.Message('note_on', note=nn, velocity=vel,
                               channel=0, time=gap))
        tr.append(mido.Message('note_off', note=nn, velocity=0,
                               channel=0, time=s2t(dur)))
        gap = s2t(0.02)

    # === Section 3: Rapid-fire + chords ===
    gap = s2t(0.2)
    for i in range(8):
        nn = mid_n + (i % 5) - 2
        nn = max(lo, min(nn, hi))
        tr.append(mido.Message('note_on', note=nn, velocity=95 + (i % 3) * 10,
                               channel=0, time=gap))
        tr.append(mido.Message('note_off', note=nn, velocity=0,
                               channel=0, time=s2t(0.08)))
        gap = s2t(0.02)

    gap = s2t(0.15)
    chord1 = [mid_n - 5, mid_n, mid_n + 4]
    chord2 = [mid_n - 3, mid_n, mid_n + 5]
    for chord in [chord1, chord2]:
        for j, nn in enumerate(chord):
            nn = max(lo, min(nn, hi))
            tr.append(mido.Message('note_on', note=nn, velocity=100,
                                   channel=0, time=gap if j == 0 else 0))
        for j, nn in enumerate(chord):
            nn = max(lo, min(nn, hi))
            tr.append(mido.Message('note_off', note=nn, velocity=0,
                                   channel=0, time=s2t(0.5) if j == 0 else 0))
        gap = s2t(0.12)

    # Final flourish
    gap = s2t(0.1)
    for i in range(6):
        nn = mid_n + 6 - i * 2
        nn = max(lo, min(nn, hi))
        tr.append(mido.Message('note_on', note=nn, velocity=110,
                               channel=0, time=gap))
        tr.append(mido.Message('note_off', note=nn, velocity=0,
                               channel=0, time=s2t(0.06)))
        gap = s2t(0.015)

    mid.save(path)


def _make_lead_unison_midi(path):
    """Multi-track unison test for lead.

    Creates 4 tracks on different channels all playing the same melody
    with CC7 sidechain pumping on 2 of them — exactly the pattern
    that caused the 'ghost howling' bug in early versions.

    Tests:
    - Deterministic phase coherence (no beating on unison)
    - CC7 sidechain interaction with multi-track
    - High-frequency aliasing (notes up to MIDI 96)
    """
    mid = mido.MidiFile(ticks_per_beat=TPB)

    melody = [
        (60, 0.5, 100), (64, 0.5, 100), (67, 0.5, 100), (72, 0.5, 100),
        (76, 0.4, 110), (79, 0.4, 110), (84, 0.6, 120), (88, 0.4, 100),
        (91, 0.3, 100), (96, 0.8, 110),  # high notes for aliasing test
    ]

    for ch in range(4):
        tr = mido.MidiTrack()
        mid.tracks.append(tr)
        tr.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(BPM), time=0))
        tr.append(mido.Message('program_change', program=81, channel=ch, time=0))

        # Add CC7 sidechain pumping on ch 0 and 1
        if ch < 2:
            t_acc = 0
            for cyc in range(12):
                for step in range(4):
                    val = int(127 * step / 3)
                    tr.append(mido.Message('control_change', control=7,
                                          value=val, channel=ch,
                                          time=s2t(0.15) if t_acc > 0 or step > 0 else 0))
                    t_acc += 0.15
            # Reset
            tr.append(mido.Message('control_change', control=7,
                                  value=127, channel=ch, time=s2t(0.1)))

        gap = 0
        for nn, dur, vel in melody:
            tr.append(mido.Message('note_on', note=nn, velocity=vel,
                                   channel=ch, time=gap))
            tr.append(mido.Message('note_off', note=nn, velocity=0,
                                   channel=ch, time=s2t(dur)))
            gap = s2t(0.05)

    mid.save(path)


def _make_drum_midi(path):
    mid = mido.MidiFile(ticks_per_beat=TPB)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(BPM), time=0))

    gap = 0
    for nn in DRUM_HITS:
        tr.append(mido.Message('note_on', note=nn, velocity=100,
                               channel=9, time=gap))
        tr.append(mido.Message('note_off', note=nn, velocity=0,
                               channel=9, time=s2t(0.3)))
        gap = s2t(0.05)

    gap = s2t(0.3)
    for bar in range(3):
        for beat in range(4):
            if beat % 2 == 0:
                tr.append(mido.Message('note_on', note=36, velocity=100,
                                       channel=9, time=gap))
                tr.append(mido.Message('note_off', note=36, velocity=0,
                                       channel=9, time=s2t(0.05)))
                gap = 0
            if beat % 2 == 1:
                tr.append(mido.Message('note_on', note=38, velocity=90,
                                       channel=9, time=gap))
                tr.append(mido.Message('note_off', note=38, velocity=0,
                                       channel=9, time=s2t(0.05)))
                gap = 0
            tr.append(mido.Message('note_on', note=42, velocity=70,
                                   channel=9, time=gap))
            tr.append(mido.Message('note_off', note=42, velocity=0,
                                   channel=9, time=s2t(0.05)))
            gap = s2t(0.4)

    gap = s2t(0.1)
    fill = [47, 45, 43, 41, 43, 41, 38, 36]
    for nn in fill:
        tr.append(mido.Message('note_on', note=nn, velocity=110,
                               channel=9, time=gap))
        tr.append(mido.Message('note_off', note=nn, velocity=0,
                               channel=9, time=s2t(0.06)))
        gap = s2t(0.02)

    tr.append(mido.Message('note_on', note=49, velocity=120,
                           channel=9, time=s2t(0.05)))
    tr.append(mido.Message('note_on', note=36, velocity=110,
                           channel=9, time=0))
    tr.append(mido.Message('note_off', note=49, velocity=0,
                           channel=9, time=s2t(0.8)))
    tr.append(mido.Message('note_off', note=36, velocity=0,
                           channel=9, time=0))

    mid.save(path)


if __name__ == "__main__":
    test_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(test_dir)
    sys.path.insert(0, proj_dir)

    out_dir = os.path.join(test_dir, "samples")
    os.makedirs(out_dir, exist_ok=True)

    from midi import parse
    from mix import render

    targets = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    for label, prog, lo, hi in INSTRUMENTS:
        if targets and label not in targets:
            continue
        mp = os.path.join(out_dir, f"{label}.mid")
        fp = os.path.join(out_dir, f"{label}.flac")
        _make_instrument_midi(prog, lo, hi, mp)
        print(f"\n=== {label} (prog={prog}, {lo}-{hi}) ===")
        result = parse(mp)
        render(result, fp)

    # lead unison test
    if not targets or "lead_unison" in targets or "lead" in targets:
        mp = os.path.join(out_dir, "lead_unison.mid")
        fp = os.path.join(out_dir, "lead_unison.flac")
        _make_lead_unison_midi(mp)
        print(f"\n=== lead_unison (4-track unison + CC7 sidechain) ===")
        result = parse(mp)
        render(result, fp)

    if not targets or "drums" in targets:
        mp = os.path.join(out_dir, "drums.mid")
        fp = os.path.join(out_dir, "drums.flac")
        _make_drum_midi(mp)
        print(f"\n=== drums ===")
        result = parse(mp)
        render(result, fp)

    print(f"\nAll done. Samples in {out_dir}/")
