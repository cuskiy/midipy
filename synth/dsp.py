"""Shared DSP utilities: filters, waveform helpers, bounded cache."""
import math
from typing import Any, ClassVar, Dict
import weakref
import numpy as np
from scipy.signal import butter as _butter

SR = 44100
A4 = 440.0


class BoundedCache:
    """Fixed-size cache with FIFO eviction and global clear."""
    _registry: ClassVar = weakref.WeakSet()

    def __init__(self, maxsize: int = 64) -> None:
        self._data: Dict = {}
        self._maxsize = maxsize
        BoundedCache._registry.add(self)

    def __contains__(self, key: Any) -> bool:
        return key in self._data

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        if key not in self._data and len(self._data) >= self._maxsize:
            del self._data[next(iter(self._data))]
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: Any, default: Any = None) -> Any:
        return self._data.get(key, default)

    def clear(self) -> None:
        self._data.clear()

    @classmethod
    def clear_all(cls) -> None:
        for c in cls._registry:
            c.clear()


_BP_CACHE = BoundedCache(64)


def csv_parse(s: str) -> list:
    return [float(x) for x in s.split(",")] if s else []


def get_bp(lo: float, hi: float):
    key = (int(lo / 10) * 10, int(hi / 50) * 50)
    if key not in _BP_CACHE:
        lo_c, hi_c = max(key[0], 20), min(key[1], int(SR * 0.45))
        _BP_CACHE[key] = _butter(2, [lo_c, hi_c], btype='band', fs=SR, output='sos') if hi_c > lo_c + 50 else None
    return _BP_CACHE[key]


def biquad_peak(fc: float, gain_db: float, q: float):
    fc = max(fc, 30.0)
    if fc >= SR * 0.45:
        return None
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * fc / SR
    alpha = math.sin(w0) / (2 * max(q, 0.3))
    cw = math.cos(w0)
    b0 = 1 + alpha * A; b1 = -2 * cw; b2 = 1 - alpha * A
    a0 = 1 + alpha / A; a1 = -2 * cw; a2 = 1 - alpha / A
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def biquad_low_shelf(fc: float, gain_db: float, q: float = 0.707):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * fc / SR
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2 * q)
    sqA = math.sqrt(A)
    b0 = A * ((A+1) - (A-1)*cw + 2*sqA*alpha)
    b1 = 2*A * ((A-1) - (A+1)*cw)
    b2 = A * ((A+1) - (A-1)*cw - 2*sqA*alpha)
    a0 = (A+1) + (A-1)*cw + 2*sqA*alpha
    a1 = -2 * ((A-1) + (A+1)*cw)
    a2 = (A+1) + (A-1)*cw - 2*sqA*alpha
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def biquad_high_shelf(fc: float, gain_db: float, q: float = 0.707):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * fc / SR
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2 * q)
    sqA = math.sqrt(A)
    b0 = A * ((A+1) + (A-1)*cw + 2*sqA*alpha)
    b1 = -2*A * ((A-1) + (A+1)*cw)
    b2 = A * ((A+1) + (A-1)*cw - 2*sqA*alpha)
    a0 = (A+1) - (A-1)*cw + 2*sqA*alpha
    a1 = 2 * ((A-1) - (A+1)*cw)
    a2 = (A+1) - (A-1)*cw - 2*sqA*alpha
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])

