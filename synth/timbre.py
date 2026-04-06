"""Timbre parameter class and pitch-register helpers."""
import math
from synth.dsp import SR, A4, csv_parse, get_bp, biquad_peak, biquad_low_shelf, biquad_high_shelf

MAX_DET = 5
TUNE = {1: [1.0], 2: [0.9998, 1.0002], 3: [1.00005, 1.00020, 0.99975]}
_LOG2_C4 = math.log2(261.63)


def strings_for_freq(freq: float, max_strings: int) -> int:
    if max_strings <= 1: return 1
    if freq < 100: return 1
    if freq < 220: return min(2, max_strings)
    return max_strings


def detune_scale(freq: float) -> float:
    if freq >= 300: return 1.0
    if freq <= 80: return 0.08
    return 0.08 + 0.92 * ((freq - 80) / 220)


def pr(f: float) -> float:
    return max(0.0, min((math.log2(f) - _LOG2_C4 + 2) / 4, 1.0))

def lf(f: float) -> float:
    return max(0.0, min(1.0 - (math.log2(f) - _LOG2_C4 + 2) / 4, 1.0))

def vc(v: float) -> float:
    return v ** 0.65


_FIELDS = dict(
    # ── Harmonics ──
    n=10,               # partial count (1-30)
    rolloff=1.0,        # spectral rolloff exponent (0-2)
    bright=0.3,         # HF damping per partial (0-1)
    bright_lo=0.0,      # bright at low vel (hammer_hard mode)
    bright_hi=0.0,      # bright at high vel (hammer_hard mode)
    hammer_hard=0.0,    # vel→brightness curve (0=off, 1-3)
    vel_bright=0.0,     # linear vel→brightness (0-1)
    strike_pos=0.0,     # string strike position (0-0.5)
    strike_depth=0.0,   # strike notch depth (0-1)
    # ── Noise ──
    noise=0.0,          # noise amplitude (0-0.05)
    noise_d=30.0,       # noise decay rate (1-200)
    noise_peak=8.0,     # noise BP upper freq multiplier
    noise_hi=0.0,       # extra noise at high pitch (0-2)
    vn=0.5,             # vel→noise scale (0-1)
    # ── Inharmonicity ──
    inh=0.0,            # inharmonicity B (0-0.001)
    inh_stretch=0.0,    # B stretch with register (0-5)
    b1=0.50,            # per-partial decay linear (0-3)
    b3=0.08,            # per-partial decay quadratic (0-0.5)
    rel_damp=0.0,       # extra HF damping in release (0-8)
    coupling=0.0,       # string coupling (0-1)
    phantom=0.0,        # combination tone amp (0-0.1)
    n_strings=1,        # strings per note (1-3)
    n_extra=0,          # extra partials at low pitch (0-15)
    # ── Spectral dynamics ──
    spec_b=0.0,         # brightness transient depth (0-1)
    spec_tau=0.015,     # brightness transient decay (s)
    drive=0.0,          # tanh saturation (0-0.3)
    # ── ADSR ──
    att=0.005,          # attack (s, 0.001-0.5)
    d1=0.4,             # decay1 time (s)
    d1l=0.35,           # decay1 end level (0-1)
    d2=2.0,             # sustain decay tau (s)
    d2s=0.0,            # secondary decay tau (s, 0=off)
    pr=1.0,             # primary/secondary ratio (0-1)
    rel=0.3,            # release (s)
    # ── Velocity / pitch ──
    va=0.5,             # vel→attack shortening (0-1)
    vd=0.3,             # vel→sustain lengthening (0-1)
    ps=0.0,             # pitch→sustain boost (0-1)
    pdm=1.0,            # pitch→decay multiplier (0.5-3)
    det_c=0.0,          # sympathetic detune cents (0-8)
    det_m=0.0,          # sympathetic mix amp (0-0.2)
    # ── Vibrato / tremolo ──
    vib_r=5.0,          # vibrato rate (Hz)
    vib_d=0.0,          # vibrato depth (cents, 0=off)
    vib_del=0.3,        # vibrato onset delay (s)
    trem_r=0.0,         # tremolo rate (Hz)
    trem_d=0.0,         # tremolo depth (0-0.3)
    # ── Sub-oscillators ──
    sub=0.0,            # sub-octave amp (0-1)
    sub_third=0.0,      # 3rd harmonic sub amp (0-1)
    live=0.0,           # envelope liveness (0-0.01)
    # ── Filter ──
    fc_base=0.0,        # vel-sensitive LP base Hz (0=off)
    fc_min=0.3,         # fc ratio at vel=0 (0-1)
    hammer_p=2.5,       # vel→fc exponent
    fc_alpha=2.0,       # LP steepness (1-3)
    decay_exp=0.0,      # partial decay pitch scaling (0-1)
    # ── Character ──
    micro=0.5,          # pitch/amp micro-variation (0-1)
    even_atten=0.0,     # even harmonic atten (0-1)
    key_click=0.0,      # key click amp (0-0.2)
    # ── KS engine ──
    ks_lp=0.0,          # delay line LP coeff (0=default)
    ks_click=0.0,       # excitation click (0-0.5)
    # ── Formants (CSV) ──
    formant_freqs="",   # resonance freqs
    formant_gains="",   # resonance gains
    formant_qs="",      # resonance Q factors
    # ── Harmonic amps (CSV, overrides rolloff/bright) ──
    harm_amps="",
    # ── Extended ──
    bow_noise=0.0,      # bowed string noise (0-0.02)
    breath_noise=0.0,   # breath noise (0-0.02)
    lip_shape=0.0,      # brass lip nonlinearity (0-0.5)
    tw_leak=0.0,        # organ tonewheel leak (0-0.1)
)

class Timbre:
    __slots__ = tuple(_FIELDS.keys()) + ('_ref_amp', '_ha_cache')

    def __init__(self, **kw):
        unknown = set(kw) - set(_FIELDS)
        if unknown:
            raise TypeError(f"Unknown Timbre fields: {unknown}")
        for k, v in _FIELDS.items():
            setattr(self, k, kw.get(k, v))
        self._ref_amp = None
        self._ha_cache = None
        self._validate()

    def _validate(self):
        if self.n < 1: raise ValueError(f"n must be >= 1, got {self.n}")
        if self.att < 0: raise ValueError(f"att must be >= 0, got {self.att}")
        if self.rel < 0: raise ValueError(f"rel must be >= 0, got {self.rel}")
        if self.rolloff < 0: raise ValueError(f"rolloff must be >= 0, got {self.rolloff}")
        nyq = SR / 2
        if self.formant_freqs:
            for f in csv_parse(self.formant_freqs):
                if f >= nyq:
                    raise ValueError(f"formant freq {f} Hz >= Nyquist ({nyq} Hz)")

    def copy(self, **ov) -> 'Timbre':
        return Timbre(**{k: ov.get(k, getattr(self, k)) for k in _FIELDS})

    def eff_bright(self, vel_curve: float) -> float:
        if self.hammer_hard > 0:
            t = 1 - math.exp(-self.hammer_hard * vel_curve)
            return self.bright_lo + (self.bright_hi - self.bright_lo) * t
        if self.vel_bright > 0:
            return self.bright * (1 - self.vel_bright * vel_curve * 0.5)
        return self.bright

    def eff_fc(self, vel_curve: float) -> float:
        if self.fc_base <= 0: return 0.0
        return self.fc_base * (self.fc_min + (1 - self.fc_min) * vel_curve ** self.hammer_p)

    def get_harm_amps(self) -> list:
        if self._ha_cache is None:
            self._ha_cache = csv_parse(self.harm_amps)
        return self._ha_cache

    def ref_amp(self) -> float:
        if self._ref_amp is not None:
            return self._ref_amp
        ha = self.get_harm_amps()
        if ha:
            total = max(sum(ha), 0.01)
        else:
            total = sum(
                (1/k**self.rolloff) * math.exp(-self.bright*(k-1)) *
                (1 - self.strike_depth*(1 - abs(math.sin(k*math.pi*self.strike_pos)))
                 if self.strike_pos > 0 else 1)
                for k in range(1, self.n+1)
            )
        total += self.sub + self.sub_third
        self._ref_amp = max(total, 0.01)
        return self._ref_amp
