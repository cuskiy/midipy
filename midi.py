"""MIDI parser. Supports GM standard CCs, aftertouch, pitch bend, RPN."""
import mido

MAX_PEDAL_SUSTAIN = 5.0
DEFAULT_PB_RANGE = 2.0


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


def _ch(ch_data, ch):
    if ch not in ch_data:
        ch_data[ch] = {
            "vol": [], "expr": [], "mod": [], "pb": [],
            "aftertouch": [], "reverb": 1.0, "chorus": 0.0,
            "brightness": [], "pb_range": DEFAULT_PB_RANGE,
        }
    return ch_data[ch]


def parse(path: str) -> tuple:
    mid = mido.MidiFile(path)
    tmap = _tempo_map(mid)
    all_notes, ch_data = [], {}
    rpn_msb, rpn_lsb = {}, {}

    for track in mid.tracks:
        on, on_vel = {}, {}
        pedal, held = {}, {}
        tick, progs, cpan = 0, {}, {}
        notes = []

        for msg in track:
            tick += msg.time
            t = _tick2sec(tick, tmap, mid.ticks_per_beat)
            ch = msg.channel if hasattr(msg, "channel") else 0

            if msg.type == "program_change":
                progs[ch] = msg.program

            elif msg.type == "pitchwheel":
                cd = _ch(ch_data, ch)
                cd["pb"].append((t, msg.pitch / 8192.0 * cd["pb_range"]))

            elif msg.type == "aftertouch":
                _ch(ch_data, ch)["aftertouch"].append((t, msg.value / 127.0))

            elif msg.type == "control_change":
                cc, val = msg.control, msg.value
                if cc == 64:
                    if val >= 64:
                        pedal[ch] = True
                    else:
                        pedal[ch] = False
                        for nn, st, vl in held.pop(ch, []):
                            d = min(max(t - st, 0.01), MAX_PEDAL_SUSTAIN)
                            notes.append((st, nn, d, vl, ch, progs.get(ch, 0)))
                elif cc == 10:
                    cpan[ch] = val
                elif cc == 7:
                    _ch(ch_data, ch)["vol"].append((t, val / 127.0))
                elif cc == 11:
                    _ch(ch_data, ch)["expr"].append((t, val / 127.0))
                elif cc == 1:
                    _ch(ch_data, ch)["mod"].append((t, val / 127.0))
                elif cc == 91:
                    _ch(ch_data, ch)["reverb"] = val / 127.0
                elif cc == 93:
                    _ch(ch_data, ch)["chorus"] = val / 127.0
                elif cc == 74:
                    _ch(ch_data, ch)["brightness"].append((t, val / 127.0))
                elif cc == 100:
                    rpn_lsb[ch] = val
                elif cc == 101:
                    rpn_msb[ch] = val
                elif cc == 6:
                    if rpn_msb.get(ch) == 0 and rpn_lsb.get(ch) == 0:
                        _ch(ch_data, ch)["pb_range"] = float(val)
                elif cc == 121:
                    rpn_msb.pop(ch, None)
                    rpn_lsb.pop(ch, None)

            elif msg.type == "note_on" and msg.velocity > 0:
                key = (ch, msg.note)
                if key in on:
                    d = max(t - on[key], 0.01)
                    notes.append((on[key], msg.note, d, on_vel[key], ch, progs.get(ch, 0)))
                on[key] = t
                on_vel[key] = msg.velocity / 127.0

            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (ch, msg.note)
                if key in on:
                    st, vl = on.pop(key), on_vel.pop(key)
                    if pedal.get(ch):
                        held.setdefault(ch, []).append((msg.note, st, vl))
                    else:
                        notes.append((st, msg.note, max(t - st, 0.01), vl, ch, progs.get(ch, 0)))

        end = _tick2sec(tick, tmap, mid.ticks_per_beat)
        for ch_, hl in held.items():
            for nn, st, vl in hl:
                notes.append((st, nn, min(max(end - st, 0.01), MAX_PEDAL_SUSTAIN), vl, ch_, progs.get(ch_, 0)))
        for (ci, nn), st in list(on.items()):
            notes.append((st, nn, max(end - st, 0.01), on_vel[(ci, nn)], ci, progs.get(ci, 0)))

        by_prog_ch = {}
        for note in notes:
            by_prog_ch.setdefault((note[5], note[4]), []).append(note)
        for key in sorted(by_prog_ch):
            pn = by_prog_ch[key]
            all_notes.append((pn, cpan.get(key[1])))

    # Sort and deduplicate per-channel CC/PB events (multiple tracks
    # may share a channel, appending events in track-processing order).
    for ch in ch_data:
        cd = ch_data[ch]
        for key in ("vol", "expr", "mod", "pb", "aftertouch", "brightness"):
            evts = cd.get(key, [])
            if evts:
                evts.sort(key=lambda x: x[0])
                # deduplicate: keep last value at each unique time
                deduped = []
                for t, v in evts:
                    if deduped and abs(t - deduped[-1][0]) < 1e-9:
                        deduped[-1] = (t, v)
                    else:
                        deduped.append((t, v))
                cd[key] = deduped

    return [t for t, _ in all_notes], [p for _, p in all_notes], ch_data
