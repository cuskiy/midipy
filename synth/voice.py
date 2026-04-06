"""Voice lifecycle management for note rendering."""
from typing import Dict, List, Optional, Tuple
import numpy as np
from .timbre import SR, Timbre
from .instruments import TIMBRES
from .gm import PROGRAM_MAP
from . import synthesize, A4, drum
from schema import Note

ACTIVE = 0
RELEASING = 1
DONE = 2


class Voice:
    """Single sounding voice wrapping a pre-rendered waveform.

    Synthesis is deferred until the first read() call, which avoids
    rendering voices that get stolen before producing any output.
    """
    __slots__ = ('waveform', 'start_sample', 'pos', 'state', 'key',
                 'release_time', '_released_at', '_pending')

    def __init__(self) -> None:
        self.waveform: Optional[np.ndarray] = None
        self.start_sample: int = 0
        self.pos: int = 0
        self.state: int = DONE
        self.key: Tuple[int, int] = (0, 0)
        self.release_time: float = 0.3
        self._released_at: int = -1
        self._pending: Optional[tuple] = None

    def trigger(self, freq: float, dur: float, vel: float, tim: Timbre,
                name: str, nid: int, start_sample: int,
                pb_curve: Optional[np.ndarray] = None) -> None:
        self._pending = (freq, dur, vel, tim, name, nid, pb_curve)
        self.waveform = None
        self.start_sample = start_sample
        self.pos = 0
        self.state = ACTIVE
        self.release_time = getattr(tim, 'rel', 0.3)
        self._released_at = -1

    def _realize(self) -> None:
        """Synthesize the waveform from stored parameters."""
        if self._pending is None:
            return
        freq, dur, vel, tim, name, nid, pb_curve = self._pending
        self._pending = None
        if name == "drums":
            rng = np.random.RandomState((42 + self.key[1] * 7 + nid) % (2**31))
            self.waveform = drum(self.key[1], dur, vel, rng)
        else:
            self.waveform = synthesize(freq, dur, vel, tim, name, nid,
                                       pb_curve=pb_curve)

    def release(self) -> None:
        if self.state != ACTIVE:
            return
        self._realize()
        self.state = RELEASING
        self._released_at = self.pos
        if self.waveform is None:
            self.state = DONE
            return
        remaining = len(self.waveform) - self.pos
        if remaining <= 0:
            self.state = DONE
            return
        fade_n = min(max(int(self.release_time * SR), 64), remaining)
        if fade_n > 1:
            fade = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_n)))
            self.waveform[self.pos:self.pos + fade_n] *= fade
        end = self.pos + fade_n
        if end < len(self.waveform):
            self.waveform[end:] = 0

    def read(self, n_samples: int) -> Optional[np.ndarray]:
        if self.state == DONE:
            return None
        self._realize()
        if self.waveform is None:
            return None
        end = min(self.pos + n_samples, len(self.waveform))
        if self.pos >= end:
            self.state = DONE
            return None
        chunk = self.waveform[self.pos:end]
        self.pos = end
        if end >= len(self.waveform):
            self.state = DONE
        return chunk

    def is_done(self) -> bool:
        return self.state == DONE


class VoiceManager:
    """Manages voice allocation, sustain pedal, and stealing."""

    def __init__(self, max_poly: int = 64) -> None:
        self.active: List[Voice] = []
        self.pedal: Dict[int, bool] = {}
        self.held: Dict[int, List[Voice]] = {}
        self.max_poly = max_poly
        self._nid_counter = 0
        self._active_count = 0

    def _steal_if_needed(self) -> None:
        while self._active_count >= self.max_poly:
            releasing = [v for v in self.active if v.state == RELEASING]
            victim = min(releasing, key=lambda v: v.start_sample) if releasing \
                else min(self.active, key=lambda v: v.start_sample)
            victim.state = DONE
            self.active.remove(victim)
            self._active_count -= 1

    def _allocate(self, key: Tuple[int, int], freq: float, dur: float,
                  vel: float, tim: Timbre, name: str,
                  sample_pos: int,
                  pb_curve: Optional[np.ndarray] = None) -> Voice:
        # Re-trigger: release existing voice on same key
        for v in self.active:
            if v.key == key and v.state == ACTIVE:
                v.release()
        self._steal_if_needed()

        voice = Voice()
        voice.key = key
        nid = self._nid_counter
        self._nid_counter += 1
        voice.trigger(freq, dur, vel, tim, name, nid, sample_pos,
                      pb_curve=pb_curve)
        self.active.append(voice)
        self._active_count += 1
        return voice

    def schedule_note(self, note: Note, sample_pos: int,
                      inst_name: str,
                      pb_curve: Optional[np.ndarray] = None) -> Voice:
        """Direct Note-based scheduling (dur already known, inst_name from caller)."""
        key = (note.ch, note.midi)
        freq = A4 * 2 ** ((note.midi - 69) / 12.0)
        tim = TIMBRES.get(inst_name, TIMBRES["default"])
        return self._allocate(key, freq, note.dur, note.vel, tim, inst_name,
                              sample_pos, pb_curve)

    def note_on(self, sample_pos: int, ch: int, midi: int, vel: float,
                prog: int, dur: float,
                pb_curve: Optional[np.ndarray] = None) -> Voice:
        """Event-based note_on (for parse_events path)."""
        key = (ch, midi)
        freq = A4 * 2 ** ((midi - 69) / 12.0)
        name = "drums" if ch == 9 else PROGRAM_MAP.get(prog, "default")
        tim = TIMBRES.get(name, TIMBRES["default"])
        return self._allocate(key, freq, dur, vel, tim, name,
                              sample_pos, pb_curve)

    def note_off(self, ch: int, midi: int) -> None:
        if ch == 9:
            return
        key = (ch, midi)
        for v in self.active:
            if v.key == key and v.state == ACTIVE:
                if self.pedal.get(ch, False):
                    self.held.setdefault(ch, []).append(v)
                else:
                    v.release()
                break

    def pedal_change(self, ch: int, on: bool) -> None:
        self.pedal[ch] = on
        if not on:
            for v in self.held.pop(ch, []):
                v.release()

    def render_block(self, buf: np.ndarray, block_start: int,
                     block_size: int) -> None:
        block_end = block_start + block_size
        i = 0
        while i < len(self.active):
            v = self.active[i]
            if v.is_done():
                self.active.pop(i)
                self._active_count -= 1
                continue
            v_global = v.start_sample + v.pos
            if v_global >= block_end:
                i += 1
                continue
            buf_offset = max(0, v_global - block_start)
            n_read = block_size - buf_offset
            chunk = v.read(n_read)
            if chunk is not None and len(chunk) > 0:
                buf[buf_offset:buf_offset + len(chunk)] += chunk
            if v.is_done():
                self.active.pop(i)
                self._active_count -= 1
                continue
            i += 1
