"""Track splitting, merging, and deduplication for the render pipeline."""

from collections import defaultdict
from synth.gm import PROGRAM_MAP


def track_channel(notes):
    """Return the most common MIDI channel in a note list."""
    channels = [n[4] for n in notes]
    return max(set(channels), key=channels.count)


def _track_sig(notes):
    """Hashable signature for exact-duplicate deduplication.
    Includes channel so identical notes on different channels are kept."""
    return tuple((round(n[0], 3), n[1], round(n[2], 3), round(n[3], 3), n[4])
                 for n in notes)


def split_and_merge_tracks(tracks, pans):
    """Split MIDI tracks by instrument name, then merge sub-tracks that
    share the same (instrument_name, channel).

    This merges drum sub-tracks that were split across multiple MIDI
    tracks (3 tracks on ch9 → 1 sub-track) and lead/other sub-tracks
    on the same channel so they share a single RMS normalisation factor.

    Returns:
        List of (instrument_name, notes_list, pan_cc) tuples.
    """
    # Phase 1: split into (name, notes, pan_cc) with dedup
    raw_subs = []
    seen_sigs = set()
    for ti, notes in enumerate(tracks):
        if not notes:
            continue
        pan_cc = pans[ti] if ti < len(pans) else None
        by_name = defaultdict(list)
        for note in notes:
            nm = ("drums" if note[4] == 9
                  else PROGRAM_MAP.get(note[5], "default"))
            by_name[nm].append(note)
        for nm, sub in by_name.items():
            sig = _track_sig(sub)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            raw_subs.append((nm, sub, pan_cc))

    # Phase 2: merge by (name, channel)
    merge_key = defaultdict(lambda: ([], None))
    order_keys = []
    for nm, sub, pan_cc in raw_subs:
        ch = track_channel(sub)
        key = (nm, ch)
        notes_list, _ = merge_key[key]
        if not notes_list:
            order_keys.append(key)
        notes_list.extend(sub)
        if pan_cc is not None and merge_key[key][1] is None:
            merge_key[key] = (notes_list, pan_cc)
        else:
            merge_key[key] = (notes_list, merge_key[key][1])

    # Build final list preserving original order
    merged = []
    for key in order_keys:
        nm = key[0]
        notes_list, pan_cc = merge_key[key]
        merged.append((nm, notes_list, pan_cc))

    return merged
