"""Track splitting, merging, and deduplication."""
from collections import defaultdict
from synth.gm import PROGRAM_MAP


def track_channel(notes: list) -> int:
    channels = [n.ch for n in notes]
    return max(set(channels), key=channels.count)


def split_and_merge_tracks(notes: list, channel_pans: dict) -> list:
    """Group flat note list by (instrument_name, channel).
    Returns list of (name, notes, pan_cc) tuples."""
    by_key = defaultdict(list)
    for note in notes:
        nm = "drums" if note.ch == 9 else PROGRAM_MAP.get(note.prog, "default")
        by_key[(nm, note.ch)].append(note)

    return [(nm, sub, channel_pans.get(ch))
            for (nm, ch), sub in sorted(by_key.items())]
