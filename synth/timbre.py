import math
import numpy as np

SR = 44100
A4 = 440.0
MAX_DET = 5
TUNE = {1: [1.0], 2: [0.9998, 1.0002], 3: [1.00005, 1.00020, 0.99975]}

def strings_for_freq(freq, max_strings):
    if max_strings <= 1: return 1
    if freq < 100: return 1
    if freq < 220: return min(2, max_strings)
    return max_strings

def detune_scale(freq):
    if freq >= 300: return 1.0
    if freq <= 80: return 0.08
    return 0.08 + 0.92 * ((freq - 80) / 220)

_LOG2_C4 = math.log2(261.63)

def pr(f): return max(0.0, min((math.log2(f) - _LOG2_C4 + 2) / 4, 1.0))
def lf(f): return max(0.0, min(1.0 - (math.log2(f) - _LOG2_C4 + 2) / 4, 1.0))
def vc(v): return v ** 0.65

_FIELDS = dict(
    n=10, rolloff=1.0, bright=0.3, bright_lo=0.0, bright_hi=0.0, hammer_hard=0.0,
    vel_bright=0.0, strike_pos=0.0, strike_depth=0.0,
    noise=0.0, noise_d=30.0, noise_peak=8.0, noise_hi=0.0, vn=0.5,
    inh=0.0, inh_stretch=0.0, b1=0.50, b3=0.08, rel_damp=0.0,
    coupling=0.0, phantom=0.0, n_strings=1, n_extra=0,
    spec_b=0.0, spec_tau=0.015, drive=0.0,
    att=0.005, d1=0.4, d1l=0.35, d2=2.0, d2s=0.0, pr=1.0, rel=0.3,
    va=0.5, vd=0.3, ps=0.0, pdm=1.0, det_c=0.0, det_m=0.0,
    vib_r=5.0, vib_d=0.0, vib_del=0.3, trem_r=0.0, trem_d=0.0,
    sub=0.0, sub_third=0.0, live=0.0,
    fc_base=0.0, fc_min=0.3, hammer_p=2.5, fc_alpha=2.0, decay_exp=0.0,
    micro=0.5, even_atten=0.0, key_click=0.0,
    ks_lp=0.0, ks_click=0.0,
    formant_freqs="", formant_gains="", formant_qs="",
    harm_amps="",
    bow_noise=0.0, breath_noise=0.0, lip_shape=0.0, tw_leak=0.0,
)

class Timbre:
    __slots__ = tuple(_FIELDS.keys()) + ('_ref_amp', '_ha_cache')
    def __init__(self, **kw):
        unknown = set(kw) - set(_FIELDS)
        if unknown:
            raise TypeError(f"Unknown Timbre fields: {unknown}")
        for k, v in _FIELDS.items(): setattr(self, k, kw.get(k, v))
        self._ref_amp = None
        self._ha_cache = None
        self._validate()
    def _validate(self):
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.att < 0:
            raise ValueError(f"att must be >= 0, got {self.att}")
        if self.rel < 0:
            raise ValueError(f"rel must be >= 0, got {self.rel}")
        if self.rolloff < 0:
            raise ValueError(f"rolloff must be >= 0, got {self.rolloff}")
        nyq = SR / 2
        for label, val in [("formant_freqs", self.formant_freqs)]:
            if val:
                for f in csv_parse(val):
                    if f >= nyq:
                        raise ValueError(
                            f"{label} contains {f} Hz >= Nyquist ({nyq} Hz)")
    def copy(self, **ov):
        return Timbre(**{k: ov.get(k, getattr(self, k)) for k in _FIELDS})
    def eff_bright(self, vc):
        if self.hammer_hard > 0:
            t = 1 - math.exp(-self.hammer_hard * vc)
            return self.bright_lo + (self.bright_hi - self.bright_lo) * t
        if self.vel_bright > 0: return self.bright * (1 - self.vel_bright * vc * 0.5)
        return self.bright
    def eff_fc(self, vc):
        if self.fc_base <= 0: return 0.0
        return self.fc_base * (self.fc_min + (1 - self.fc_min) * vc ** self.hammer_p)
    def get_harm_amps(self):
        if self._ha_cache is None:
            self._ha_cache = [float(x) for x in self.harm_amps.split(",")] if self.harm_amps else []
        return self._ha_cache
    def ref_amp(self):
        if self._ref_amp is not None:
            return self._ref_amp
        ha = self.get_harm_amps()
        if ha:
            s = max(sum(ha), 0.01)
        else:
            s = sum((1/k**self.rolloff) * math.exp(-self.bright*(k-1)) *
                    (1 - self.strike_depth*(1 - abs(math.sin(k*math.pi*self.strike_pos)))
                     if self.strike_pos > 0 else 1)
                    for k in range(1, self.n+1))
        s += self.sub + self.sub_third
        self._ref_amp = max(s, 0.01)
        return self._ref_amp


# --- shared utilities (used by multiple engines) ---

from scipy.signal import butter as _butter

_BP_CACHE = {}


def csv_parse(s):
    return [float(x) for x in s.split(",")] if s else []


def get_bp(lo, hi):
    key = (int(lo/10)*10, int(hi/50)*50)
    if key not in _BP_CACHE:
        lo_c, hi_c = max(key[0], 20), min(key[1], int(SR*0.45))
        _BP_CACHE[key] = _butter(2, [lo_c, hi_c], btype='band', fs=SR, output='sos') if hi_c > lo_c+50 else None
    return _BP_CACHE[key]


def biquad_peak(fc, gain_db, q):
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


def biquad_low_shelf(fc, gain_db, q=0.707):
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


def biquad_high_shelf(fc, gain_db, q=0.707):
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
