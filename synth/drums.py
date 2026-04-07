"""Percussion synthesis: 55 GM drum sounds."""
import numpy as np
from .timbre import SR
from typing import NamedTuple


class DrumParams(NamedTuple):
    f_start: int        # tone sweep start Hz
    f_end: int          # tone sweep end Hz
    tone_decay: float   # tone decay time (s)
    noise_decay: float  # noise decay time (s)
    tone_mix: float     # tone amplitude (0-1)
    noise_mix: float    # noise amplitude (0-1)


# (f_start, f_end, tone_decay, noise_decay, tone_mix, noise_mix)
_MAP = {
    27: DrumParams(600,600,0.015,0.010,0.15,0.35),   # High Q (electronic tick)
    28: DrumParams(300,180,0.020,0.025,0.25,0.55),    # Slap
    29: DrumParams(0,0,0.01,0.015,0.0,0.45),          # Scratch Push
    30: DrumParams(0,0,0.01,0.018,0.0,0.45),          # Scratch Pull
    31: DrumParams(800,800,0.008,0.008,0.40,0.35),    # Sticks
    32: DrumParams(120,38,0.10,0.04,0.90,0.18),       # Square Click (soft kick)
    33: DrumParams(1200,1200,0.005,0.005,0.55,0.30),  # Metronome Bell
    34: DrumParams(140,40,0.12,0.05,0.92,0.20),       # Acoustic Bass Drum (soft)
    35: DrumParams(150,42,0.12,0.06,0.95,0.22), 36: DrumParams(155,45,0.11,0.055,0.95,0.22),
    37: DrumParams(800,800,0.025,0.025,0.30,0.80), 38: DrumParams(185,125,0.07,0.20,0.45,0.75),
    39: DrumParams(250,200,0.05,0.08,0.30,0.85), 40: DrumParams(175,120,0.065,0.18,0.45,0.75),
    41: DrumParams(95,65,0.20,0.08,0.82,0.35), 42: DrumParams(0,0,0.01,0.030,0.0,0.82),
    43: DrumParams(110,78,0.17,0.07,0.78,0.40), 44: DrumParams(0,0,0.01,0.020,0.0,0.82),
    45: DrumParams(125,88,0.15,0.07,0.72,0.42), 46: DrumParams(0,0,0.01,0.12,0.0,0.75),
    47: DrumParams(145,98,0.13,0.06,0.68,0.42), 48: DrumParams(165,115,0.11,0.055,0.62,0.42),
    49: DrumParams(0,0,0.01,0.40,0.0,0.68), 50: DrumParams(190,190,0.08,0.05,0.50,0.50),
    51: DrumParams(0,0,0.01,0.28,0.08,0.68), 52: DrumParams(0,0,0.01,0.45,0.0,0.65),
    53: DrumParams(0,0,0.01,0.10,0.10,0.55),  # ride bell
    54: DrumParams(0,0,0.01,0.015,0.0,0.28),  # tambourine — short jingles
    55: DrumParams(0,0,0.01,0.06,0.0,0.75), 56: DrumParams(540,540,0.06,0.04,0.70,0.30),  # cowbell
    57: DrumParams(0,0,0.01,0.42,0.0,0.68), 58: DrumParams(180,120,0.12,0.18,0.25,0.60),  # vibraslap
    59: DrumParams(0,0,0.01,0.22,0.0,0.70),
    # Latin percussion (GM notes 60-81)
    60: DrumParams(350,320,0.04,0.03,0.70,0.30),  # Hi Bongo
    61: DrumParams(240,200,0.06,0.04,0.75,0.30),  # Low Bongo
    62: DrumParams(420,380,0.03,0.02,0.55,0.25),  # Mute Hi Conga
    63: DrumParams(320,280,0.05,0.04,0.65,0.35),  # Open Hi Conga
    64: DrumParams(220,180,0.07,0.05,0.75,0.30),  # Low Conga
    65: DrumParams(520,500,0.04,0.03,0.72,0.28),  # High Timbale
    66: DrumParams(350,320,0.06,0.04,0.72,0.32),  # Low Timbale
    67: DrumParams(900,900,0.04,0.02,0.75,0.25),  # High Agogo
    68: DrumParams(680,680,0.05,0.03,0.75,0.25),  # Low Agogo
    69: DrumParams(0,0,0.01,0.025,0.0,0.50),      # Cabasa
    70: DrumParams(0,0,0.01,0.018,0.0,0.55),      # Maracas
    71: DrumParams(2400,1800,0.06,0.04,0.50,0.25),  # Short Whistle
    72: DrumParams(2400,1800,0.20,0.10,0.50,0.20),  # Long Whistle
    73: DrumParams(0,0,0.01,0.015,0.10,0.55),      # Short Guiro
    74: DrumParams(0,0,0.01,0.06,0.10,0.55),       # Long Guiro
    75: DrumParams(2500,2500,0.01,0.008,0.80,0.20), # Claves
    76: DrumParams(1200,1200,0.015,0.010,0.70,0.25), # Hi Wood Block
    77: DrumParams(800,800,0.020,0.012,0.70,0.25),   # Low Wood Block
    78: DrumParams(380,300,0.04,0.035,0.45,0.40),    # Mute Cuica
    79: DrumParams(480,250,0.08,0.06,0.55,0.40),     # Open Cuica
    80: DrumParams(0,0,0.01,0.06,0.15,0.60),         # Mute Triangle
    81: DrumParams(0,0,0.01,0.25,0.15,0.60),         # Open Triangle
}
_METAL = {
    42:[3400,6100,8800,12500], 44:[3200,5800,8400,11800],
    46:[2800,5200,7800,11000], 49:[2200,4100,5800,7200,10000,13500],
    51:[2600,5000,7600,10800], 52:[2400,4600,7000,10000,14000],
    53:[3800,7200,10500,14000],  # ride bell
    54:[3200,5800,8400,11000],  # tambourine
    55:[3600,6400,9200,13000], 56:[1800,3600],  # cowbell
    57:[2300,4400,6500,9300,12500],
    59:[2700,5100,7700,11200],
    67:[4200,7500,11000], 68:[3200,5800,8500],  # agogo bells
    80:[5200,8800,13200], 81:[5200,8800,13200],  # triangle
}
_SNARE = {38, 40}
_KICK = {32, 34, 35, 36}
_TOM = {41, 43, 45, 47, 48}
_HIHAT = {42, 44, 46}
_CYMBAL = {49, 51, 52, 55, 57}
_RIDE_BELL = {53}
_STICK = {31, 37}
_TICK = {27, 33, 56, 75, 76, 77}    # clicks, claves, cowbell, woodblocks
_LATIN_TONAL = {60, 61, 62, 63, 64, 65, 66, 67, 68, 78, 79}  # bongos, congas, timbales, agogo, cuica
_SHAKER = {54, 69, 70}              # tambourine, cabasa, maracas
_GUIRO = {73, 74}
_TRIANGLE = {80, 81}
_WHISTLE = {71, 72}
_VIBRA = {58, 59}                   # vibraslap, mute ride
_SCRATCH = {28, 29, 30}             # slap, scratch push/pull

# Per-class target peak (normalize each drum hit to this for consistent loudness)
_DRUM_TARGET = {
    'kick': 0.95, 'snare': 0.92, 'tom': 0.88,
    'hihat': 0.62, 'cymbal': 0.82, 'ride_bell': 0.75,
    'stick': 0.62, 'tick': 0.65,
    'latin': 0.82, 'shaker': 0.55, 'guiro': 0.62,
    'triangle': 0.65, 'whistle': 0.75, 'vibra': 0.70, 'scratch': 0.55,
    'default': 0.85,
}


def _drum_target(note: int) -> float:
    if note in _KICK:       return _DRUM_TARGET['kick']
    if note in _SNARE:      return _DRUM_TARGET['snare']
    if note in _TOM:        return _DRUM_TARGET['tom']
    if note in _HIHAT:      return _DRUM_TARGET['hihat']
    if note in _CYMBAL:     return _DRUM_TARGET['cymbal']
    if note in _RIDE_BELL:  return _DRUM_TARGET['ride_bell']
    if note in _STICK:      return _DRUM_TARGET['stick']
    if note in _TICK:       return _DRUM_TARGET['tick']
    if note in _LATIN_TONAL:return _DRUM_TARGET['latin']
    if note in _SHAKER:     return _DRUM_TARGET['shaker']
    if note in _GUIRO:      return _DRUM_TARGET['guiro']
    if note in _TRIANGLE:   return _DRUM_TARGET['triangle']
    if note in _WHISTLE:    return _DRUM_TARGET['whistle']
    if note in _VIBRA:      return _DRUM_TARGET['vibra']
    if note in _SCRATCH:    return _DRUM_TARGET['scratch']
    return _DRUM_TARGET['default']


def drum(note: int, dur: float, vel: float, rng) -> 'np.ndarray':
    f0,f1,td,nd,tm,nm = _MAP.get(note, DrumParams(200,150,0.08,0.08,0.50,0.50))
    length = max(dur, max(td,nd)*3.5)
    n = int(SR*length)
    if n == 0: return np.zeros(0)
    tv = np.linspace(0, length, n, endpoint=False)
    out = np.zeros(n)

    if tm > 0 and f0 > 0:
        sweep = f1 + (f0-f1)*np.exp(-tv/0.012)
        phase = 2*np.pi*np.cumsum(sweep)/SR
        env = np.exp(-tv/max(td, 0.005))

        if note in _KICK:
            # kick: fundamental + sub-harmonic body + click transient
            out += np.sin(phase)*env*tm
            out += 0.18*np.sin(phase*1.5+rng.random()*6.28)*env*np.exp(-tv/(td*0.35))*tm
            # beater click
            cn = min(int(SR*0.002), n)
            if cn > 0:
                click = rng.randn(cn)*0.15*np.exp(-np.arange(cn, dtype=np.float64)*15/cn)
                out[:cn] += click*tm
            # sub warmth
            sub_env = np.exp(-tv/max(td*1.2, 0.01))
            out += 0.12*np.sin(phase*0.5)*sub_env*tm
        elif note in _TOM:
            # toms: two shell modes (fundamental + ~1.5x), slight body resonance
            out += np.sin(phase)*env*tm
            out += 0.30*np.sin(phase*1.504+rng.random()*6.28)*env*np.exp(-tv/(td*0.5))*tm
            # body resonance at ~2.3x
            out += 0.10*np.sin(phase*2.295+rng.random()*6.28)*env*np.exp(-tv/(td*0.25))*tm
        else:
            # generic tonal drum
            out += np.sin(phase)*env*tm
            out += 0.25*np.sin(phase*1.59+rng.random()*6.28)*env*np.exp(-tv/(td*0.4))*tm

    if nm > 0:
        ns = rng.randn(n)
        nenv = np.exp(-tv/max(nd, 0.005))
        if note in _METAL:
            modes = _METAL[note]
            metal = np.zeros(n)
            for i, mf in enumerate(modes):
                det = 1+0.02*rng.randn()
                amp = (0.50**i)
                decay = nd*(0.5+0.5*rng.random())
                metal += amp*np.sin(2*np.pi*mf*det*tv+rng.random()*6.28)*np.exp(-tv/max(decay, 0.01))
            # blend noise + metallic modes
            ns_shaped = ns*nenv
            # hi-hats and tambourine: more noise; cymbals: more modes
            if note in (42, 44, 54, 55):
                out += (0.50*ns_shaped + 0.50*metal)*nm
            else:
                out += (0.35*ns_shaped + 0.65*metal)*nm
        else:
            out += ns*nenv*nm*0.5

    if note in _SNARE:
        # snare wires: bandpass-filtered noise centered around 4-5kHz
        wn = min(int(SR*nd*2.5), n)
        wire_ns = rng.randn(wn)
        wire_env = np.exp(-np.arange(wn, dtype=np.float64)/(SR*nd*1.5))
        # resonant wire buzz: modulated bandpass character
        wire_t = np.linspace(0, wn/SR, wn, endpoint=False)
        wire_mod = 0.5 + 0.5*np.sin(2*np.pi*(4500+rng.random()*800)*wire_t)
        wire = wire_ns * wire_mod * wire_env * 0.35
        # add a second wire band at higher freq
        wire2_mod = 0.3 + 0.7*np.sin(2*np.pi*(7200+rng.random()*1200)*wire_t)
        wire += wire_ns * wire2_mod * wire_env * 0.15
        out[:wn] += wire*vel

    pk = np.max(np.abs(out))
    # Per-drum-class target peak: gives consistent loudness within a class
    # while preserving perceptual relationships between classes (kick > shaker).
    # Eliminates per-hit RNG variance and hit-to-hit volume drift.
    if pk > 1e-10:
        out *= _drum_target(note) / pk
    out *= vel
    ac = min(128, n)
    if ac > 1: out[-ac:] *= 0.5*(1+np.cos(np.linspace(0, np.pi, ac)))
    return out
