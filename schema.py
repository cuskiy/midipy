"""Shared data types for midi, synth, and mix modules."""
from __future__ import annotations
from enum import IntEnum
from typing import NamedTuple, Optional, List, Tuple, Dict, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    import numpy as np

class Note(NamedTuple):
    start: float
    midi: int
    dur: float
    vel: float
    ch: int
    prog: int

class EventKind(IntEnum):
    NOTE_OFF = 0
    NOTE_ON = 1
    PEDAL = 2

class NoteEvent(NamedTuple):
    time: float
    kind: EventKind
    midi: int
    vel: float
    ch: int
    prog: int
    dur: float = 0.0

CcEvent = Tuple[float, float]

@dataclass
class ChannelData:
    vol: List[CcEvent] = field(default_factory=list)
    expr: List[CcEvent] = field(default_factory=list)
    mod: List[CcEvent] = field(default_factory=list)
    pb: List[CcEvent] = field(default_factory=list)
    aftertouch: List[CcEvent] = field(default_factory=list)
    brightness: List[CcEvent] = field(default_factory=list)
    reverb: List[CcEvent] = field(default_factory=list)
    chorus: List[CcEvent] = field(default_factory=list)
    pb_range: float = 2.0

class ParseResult(NamedTuple):
    notes: List[Note]
    channel_pans: Dict[int, int]
    ch_data: Dict[int, ChannelData]

@dataclass(frozen=True)
class PipelineConfig:
    track_gain: float = 0.13
    track_peak_cap: float = 2.0
    reverb_wet: float = 0.055
    reverb_rt60: float = 1.6
    reverb_room: float = 1.0
    reverb_predelay: float = 0.012
    comp_thresh: float = 0.30
    comp_ratio: float = 1.8
    comp_att_ms: float = 40.0
    comp_rel_ms: float = 400.0
    comp_sc_hp: float = 80.0
    peak_limit: float = 0.89
    az_scale: float = 1.0
    er_wet: float = 0.035
    pinna: bool = False
    immersive_er: bool = False

class TrackResult(NamedTuple):
    audio: Optional['np.ndarray']
    reverb_bus: Optional['np.ndarray']
