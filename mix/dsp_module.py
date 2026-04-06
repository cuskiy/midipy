"""Block-based DSP modules with persistent state."""
import math
from typing import List, Optional
import numpy as np
from scipy.signal import sosfilt, butter
from synth.timbre import SR, biquad_high_shelf


class DspModule:
    """Base protocol for block-based audio processing."""
    def process(self, block: np.ndarray) -> np.ndarray:
        return block
    def reset(self) -> None:
        pass


class FilterModule(DspModule):
    """sosfilt wrapper with zi state across blocks."""
    def __init__(self, sos: np.ndarray) -> None:
        self.sos = sos
        self.zi: Optional[np.ndarray] = None

    def process(self, block: np.ndarray) -> np.ndarray:
        if self.zi is None:
            self.zi = np.zeros((self.sos.shape[0], 2))
        out, self.zi = sosfilt(self.sos, block, zi=self.zi)
        return out

    def reset(self) -> None:
        self.zi = None


class BrightnessModule(DspModule):
    """CC74-driven high shelf with zi continuity."""
    def __init__(self, cc_curve: np.ndarray, buf_len: int) -> None:
        # Pre-pad to buf_len to avoid bounds checks in process()
        if len(cc_curve) < buf_len:
            self.cc_curve = np.pad(cc_curve, (0, buf_len - len(cc_curve)), mode='edge')
        else:
            self.cc_curve = cc_curve
        self.zi: Optional[np.ndarray] = None
        self.sample_pos = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        b74 = self.cc_curve[self.sample_pos]
        self.sample_pos += len(block)
        gain_db = (b74 - 0.504) / 0.504 * 4.5
        if abs(gain_db) < 0.3:
            return block
        sos = biquad_high_shelf(4000.0, gain_db)
        if self.zi is None:
            self.zi = np.zeros((sos.shape[0], 2))
        out, self.zi = sosfilt(sos, block, zi=self.zi)
        return out

    def reset(self) -> None:
        self.zi = None
        self.sample_pos = 0


class ChorusModule(DspModule):
    """Two modulated-delay voices with dynamic depth from CC93 curve."""
    _RATES = [(0.7, 0.0), (1.1, 2.1)]
    _BASE_DELAY = 0.008
    _MOD_DEPTH = 0.004
    _MIX = 0.15

    def __init__(self, depth_curve: np.ndarray, buf_len: int) -> None:
        if len(depth_curve) < buf_len:
            self.depth_curve = np.pad(depth_curve, (0, buf_len - len(depth_curve)), mode='edge')
        else:
            self.depth_curve = depth_curve
        max_d = int(SR * (self._BASE_DELAY + self._MOD_DEPTH)) + 4
        self.ring_len = max_d + 256
        self.ring = np.zeros(self.ring_len)
        self.wpos = max_d
        self.sample_count = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        depth = self.depth_curve[self.sample_count]
        if depth < 0.01:
            self.sample_count += len(block)
            return block
        n = len(block)
        # Numpy batch write to ring buffer
        indices = (self.wpos + np.arange(n)) % self.ring_len
        self.ring[indices] = block
        # Read delayed samples
        out = block.copy()
        tv = (np.arange(n) + self.sample_count) / SR
        start_pos = self.wpos  # first written position
        for rate, phase in self._RATES:
            delay = SR * self._BASE_DELAY + SR * self._MOD_DEPTH * np.sin(
                2 * math.pi * rate * tv + phase)
            read_f = start_pos + np.arange(n, dtype=np.float64) - delay
            ri0 = np.floor(read_f).astype(int) % self.ring_len
            ri1 = (ri0 + 1) % self.ring_len
            frac = read_f - np.floor(read_f)
            delayed = self.ring[ri0] * (1 - frac) + self.ring[ri1] * frac
            out += delayed * depth * self._MIX
        # RMS compensation
        rms_in = np.sqrt(np.mean(block ** 2)) + 1e-10
        rms_out = np.sqrt(np.mean(out ** 2)) + 1e-10
        if rms_out > rms_in * 1.01:
            out *= rms_in / rms_out
        self.wpos = (self.wpos + n) % self.ring_len
        self.sample_count += n
        return out

    def reset(self) -> None:
        self.ring[:] = 0
        self.wpos = int(SR * (self._BASE_DELAY + self._MOD_DEPTH)) + 4
        self.sample_count = 0


class LeslieModule(DspModule):
    """Leslie speaker: independent LP/HP split + rotor AM/Doppler."""
    def __init__(self) -> None:
        self.lp_sos = butter(3, 800.0, btype='low', fs=SR, output='sos')
        self.hp_sos = butter(3, 800.0, btype='high', fs=SR, output='sos')
        self.lp_zi: Optional[np.ndarray] = None
        self.hp_zi: Optional[np.ndarray] = None
        max_d = int(0.0004 * SR) + 4
        self.ring_len = max_d + 256
        self.ring = np.zeros(self.ring_len)
        self.wpos = max_d
        self.sample_count = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        n = len(block)
        if self.lp_zi is None:
            self.lp_zi = np.zeros((self.lp_sos.shape[0], 2))
            self.hp_zi = np.zeros((self.hp_sos.shape[0], 2))
        lo, self.lp_zi = sosfilt(self.lp_sos, block, zi=self.lp_zi)
        hi, self.hp_zi = sosfilt(self.hp_sos, block, zi=self.hp_zi)

        tv = (np.arange(n) + self.sample_count) / SR
        hp = 2 * math.pi * 6.8 * tv
        hi *= 1.0 + 0.30 * np.sin(hp)
        # Numpy batch write hi to ring buffer
        indices = (self.wpos + np.arange(n)) % self.ring_len
        self.ring[indices] = hi
        # Doppler read
        dop = 0.0004 * SR * np.sin(hp + 1.57)
        read_f = self.wpos + np.arange(n, dtype=np.float64) - dop
        ri0 = np.floor(read_f).astype(int) % self.ring_len
        ri1 = (ri0 + 1) % self.ring_len
        frac = read_f - np.floor(read_f)
        hd = self.ring[ri0] * (1 - frac) + self.ring[ri1] * frac

        lo *= 1.0 + 0.15 * np.sin(2 * math.pi * 5.5 * tv + 0.8)
        out = lo + hd
        # RMS compensation
        rms_in = np.sqrt(np.mean(block ** 2)) + 1e-10
        rms_out = np.sqrt(np.mean(out ** 2)) + 1e-10
        if rms_out > 1e-10:
            out *= rms_in / rms_out
        self.wpos = (self.wpos + n) % self.ring_len
        self.sample_count += n
        return out

    def reset(self) -> None:
        self.lp_zi = None
        self.hp_zi = None
        self.ring[:] = 0
        self.wpos = int(0.0004 * SR) + 4
        self.sample_count = 0


class ModVibratoModule(DspModule):
    """CC1-driven pitch vibrato via modulated delay."""
    _VIB_RATE = 5.5
    _MAX_CENTS = 50.0

    def __init__(self, mod_curve: np.ndarray, buf_len: int) -> None:
        # Pre-pad to buf_len
        if len(mod_curve) < buf_len:
            self.mod_curve = np.pad(mod_curve, (0, buf_len - len(mod_curve)), mode='edge')
        else:
            self.mod_curve = mod_curve
        peak_delay = int((self._MAX_CENTS * 0.000578) / (2*math.pi*self._VIB_RATE) * SR) + 4
        self.ring_len = peak_delay + 256
        self.ring = np.zeros(self.ring_len)
        self.wpos = peak_delay
        self.sample_count = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        n = len(block)
        mod_vals = self.mod_curve[self.sample_count:self.sample_count + n]
        if np.max(mod_vals) < 0.01:
            self.sample_count += n
            return block
        # Numpy batch write
        indices = (self.wpos + np.arange(n)) % self.ring_len
        self.ring[indices] = block
        # Modulated delay read
        tv = (np.arange(n) + self.sample_count) / SR
        peak_d = (self._MAX_CENTS * 0.000578) / (2*math.pi*self._VIB_RATE) * SR
        delay = mod_vals * peak_d * np.sin(2 * math.pi * self._VIB_RATE * tv)
        read_f = self.wpos + np.arange(n, dtype=np.float64) - delay
        ri0 = np.floor(read_f).astype(int) % self.ring_len
        ri1 = (ri0 + 1) % self.ring_len
        frac = read_f - np.floor(read_f)
        out = self.ring[ri0] * (1 - frac) + self.ring[ri1] * frac
        self.wpos = (self.wpos + n) % self.ring_len
        self.sample_count += n
        return out

    def reset(self) -> None:
        self.ring[:] = 0
        peak_delay = int((self._MAX_CENTS * 0.000578) / (2*math.pi*self._VIB_RATE) * SR) + 4
        self.wpos = peak_delay
        self.sample_count = 0


class DspChain:
    """Ordered list of DspModules processed sequentially."""
    def __init__(self, modules: Optional[List[DspModule]] = None) -> None:
        self.modules: List[DspModule] = modules or []

    def process(self, block: np.ndarray) -> np.ndarray:
        for m in self.modules:
            block = m.process(block)
        return block

    def reset(self) -> None:
        for m in self.modules:
            m.reset()
