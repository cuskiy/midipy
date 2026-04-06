"""MIDI parser. Supports GM standard CCs, aftertouch, pitch bend, RPN."""
import mido
from schema import Note, ChannelData, ParseResult, NoteEvent, EventKind

MAX_PEDAL_SUSTAIN = 5.0


def _tempo_map(mid):
    evts = []
    for tr in mid.tracks:
        tick = 0
        for msg in tr:
            tick += msg.time
            if msg.type == "set_tempo":
                evts.append((tick, msg.tempo))
    evts.sort()
    if not evts or evts[0][0] != 0:
        evts.insert(0, (0, 500000))
    return evts


def _tick2sec(tick, tmap, tpb):
    sec, pt, pp = 0.0, 0, tmap[0][1]
    for tk, tp in tmap[1:]:
        if tk >= tick:
            break
        sec += (tk - pt) / tpb * pp / 1e6
        pt, pp = tk, tp
    return sec + (tick - pt) / tpb * pp / 1e6


def _ensure_ch(ch_data: dict, ch: int) -> ChannelData:
    if ch not in ch_data:
        ch_data[ch] = ChannelData()
    return ch_data[ch]


def _dedup_cc(ch_data: dict):
    for ch, cd in ch_data.items():
        for attr in ("vol", "expr", "mod", "pb", "aftertouch",
                     "brightness", "reverb", "chorus"):
            evts = getattr(cd, attr)
            if not evts:
                continue
            evts.sort(key=lambda x: x[0])
            deduped = []
            for t, v in evts:
                if deduped and abs(t - deduped[-1][0]) < 1e-9:
                    deduped[-1] = (t, v)
                else:
                    deduped.append((t, v))
            setattr(cd, attr, deduped)


def parse_events(path: str):
    """Parse MIDI into NoteEvents (with dur via forward scan) + ChannelData + pan.

    Returns (events, ch_data, channel_pan).
    """
    try:
        mid = mido.MidiFile(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read MIDI file '{path}': {e}") from e
    tmap = _tempo_map(mid)
    ch_data = {}
    raw_events = []
    channel_prog = {}
    channel_pan = {}
    rpn_msb = {}
    rpn_lsb = {}
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            t = _tick2sec(tick, tmap, mid.ticks_per_beat)
            ch = msg.channel if hasattr(msg, "channel") else 0
            if msg.type == "program_change":
                channel_prog[ch] = msg.program
            elif msg.type == "note_on" and msg.velocity > 0:
                raw_events.append((t, EventKind.NOTE_ON, msg.note,
                                   msg.velocity / 127.0, ch,
                                   channel_prog.get(ch, 0)))
            elif msg.type == "note_off" or (msg.type == "note_on"
                                            and msg.velocity == 0):
                raw_events.append((t, EventKind.NOTE_OFF, msg.note,
                                   0.0, ch, channel_prog.get(ch, 0)))
            elif msg.type == "pitchwheel":
                cd = _ensure_ch(ch_data, ch)
                cd.pb.append((t, msg.pitch / 8192.0 * cd.pb_range))
            elif msg.type == "aftertouch":
                _ensure_ch(ch_data, ch).aftertouch.append(
                    (t, msg.value / 127.0))
            elif msg.type == "control_change":
                cc, val = msg.control, msg.value
                if cc == 64:
                    raw_events.append((t, EventKind.PEDAL, 0,
                                       float(val >= 64), ch, 0))
                elif cc == 10:
                    channel_pan[ch] = val
                elif cc == 7:
                    _ensure_ch(ch_data, ch).vol.append((t, val / 127.0))
                elif cc == 11:
                    _ensure_ch(ch_data, ch).expr.append((t, val / 127.0))
                elif cc == 1:
                    _ensure_ch(ch_data, ch).mod.append((t, val / 127.0))
                elif cc == 91:
                    _ensure_ch(ch_data, ch).reverb.append(
                        (t, val / 127.0))
                elif cc == 93:
                    _ensure_ch(ch_data, ch).chorus.append(
                        (t, val / 127.0))
                elif cc == 74:
                    _ensure_ch(ch_data, ch).brightness.append(
                        (t, val / 127.0))
                elif cc == 100:
                    rpn_lsb[ch] = val
                elif cc == 101:
                    rpn_msb[ch] = val
                elif cc == 6:
                    if rpn_msb.get(ch) == 0 and rpn_lsb.get(ch) == 0:
                        _ensure_ch(ch_data, ch).pb_range = float(val)
                elif cc == 121:
                    rpn_msb.pop(ch, None)
                    rpn_lsb.pop(ch, None)
    raw_events.sort(key=lambda e: (e[0], -e[1]))
    _dedup_cc(ch_data)

    # Forward scan: fill dur for NOTE_ON by matching NOTE_OFF
    pending = {}
    events = []
    max_time = raw_events[-1][0] + 2.0 if raw_events else 0.0
    for t, kind, midi, vel, ch, prog in raw_events:
        if kind == EventKind.NOTE_ON:
            key = (ch, midi)
            if key in pending:
                idx = pending[key]
                old = events[idx]
                events[idx] = NoteEvent(old.time, old.kind, old.midi,
                                        old.vel, old.ch, old.prog,
                                        dur=t - old.time)
            events.append(NoteEvent(t, kind, midi, vel, ch, prog, dur=0.0))
            pending[key] = len(events) - 1
        elif kind == EventKind.NOTE_OFF:
            key = (ch, midi)
            if key in pending:
                idx = pending.pop(key)
                old = events[idx]
                events[idx] = NoteEvent(old.time, old.kind, old.midi,
                                        old.vel, old.ch, old.prog,
                                        dur=t - old.time)
            events.append(NoteEvent(t, kind, midi, vel, ch, prog))
        else:
            events.append(NoteEvent(t, kind, midi, vel, ch, prog))

    for key, idx in pending.items():
        old = events[idx]
        events[idx] = NoteEvent(old.time, old.kind, old.midi, old.vel,
                                old.ch, old.prog, dur=max_time - old.time)

    return events, ch_data, channel_pan


def _events_to_notes(events):
    """Resolve NoteEvents into Notes, handling pedal-extended durations."""
    pedal_on = {}
    active = {}   # (ch, midi) → NoteEvent
    held = {}     # (ch, midi) → NoteEvent (note_off received while pedal held)
    notes = []

    for ev in events:
        if ev.kind == EventKind.NOTE_ON:
            key = (ev.ch, ev.midi)
            # Close any prior note on same key
            for store in (active, held):
                if key in store:
                    old = store.pop(key)
                    notes.append(Note(old.time, old.midi,
                                      max(ev.time - old.time, 0.01),
                                      old.vel, old.ch, old.prog))
            active[key] = ev
        elif ev.kind == EventKind.NOTE_OFF:
            key = (ev.ch, ev.midi)
            if key in active:
                if pedal_on.get(ev.ch, False):
                    held[key] = active.pop(key)
                else:
                    old = active.pop(key)
                    dur = old.dur if old.dur > 0 else max(ev.time - old.time, 0.01)
                    notes.append(Note(old.time, old.midi, dur,
                                      old.vel, old.ch, old.prog))
        elif ev.kind == EventKind.PEDAL:
            if ev.vel > 0.5:
                pedal_on[ev.ch] = True
            else:
                pedal_on[ev.ch] = False
                for key in list(held):
                    if key[0] == ev.ch:
                        old = held.pop(key)
                        dur = min(max(ev.time - old.time, 0.01),
                                  MAX_PEDAL_SUSTAIN)
                        notes.append(Note(old.time, old.midi, dur,
                                          old.vel, old.ch, old.prog))

    max_time = events[-1].time + 2.0 if events else 0.0
    for store in (active, held):
        for key, ev in list(store.items()):
            dur = min(max(max_time - ev.time, 0.01), MAX_PEDAL_SUSTAIN)
            notes.append(Note(ev.time, ev.midi, dur, ev.vel, ev.ch, ev.prog))
    return notes


def parse(path: str) -> ParseResult:
    """Parse MIDI file. Built on parse_events()."""
    events, ch_data, channel_pans = parse_events(path)
    notes = _events_to_notes(events)
    return ParseResult(notes=notes, channel_pans=channel_pans, ch_data=ch_data)
