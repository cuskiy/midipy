"""Additive synthesis with inline spectral envelope.
Formant shaping is done per-partial via Lorentzian resonance, not post-EQ.
_apply_formants retained for KS/FM paths only."""
import numpy as np
import math
from typing import List, NamedTuple, Tuple
from scipy.signal import sosfilt, resample_poly
from .dsp import BoundedCache
from .timbre import (SR, MAX_DET, TUNE, pr, lf, vc, strings_for_freq,
                     detune_scale, csv_parse, get_bp, biquad_peak)
from .envelope import envelope


class _PartialData(NamedTuple):
    k: int          # harmonic number
    am: float       # stretched harmonic ratio
    fk: float       # frequency Hz
    a: float        # amplitude
    R: float        # decay rate
    phi: float      # initial phase


_SE_CACHE = BoundedCache(64)
_PEAK_CACHE = BoundedCache(64)


def _parse_formants(tim):
    key = (tim.formant_freqs, tim.formant_gains, tim.formant_qs)
    if key in _SE_CACHE:
        return _SE_CACHE[key]
    freqs, gains, qs = csv_parse(tim.formant_freqs), csv_parse(tim.formant_gains), csv_parse(tim.formant_qs)
    result = []
    for i, fc in enumerate(freqs):
        g = gains[i] if i < len(gains) else 0.3
        q = qs[i] if i < len(qs) else 2.0
        bw = fc / (2 * max(q, 0.3))
        g_lin = 10 ** (g * 24.0 / 20.0) - 1.0
        result.append((fc, bw, g_lin))
    _SE_CACHE[key] = result
    return result


def _spectral_gain(f, formants):
    g = 1.0
    for fc, bw, g_lin in formants:
        g += g_lin / (1.0 + ((f - fc) / bw) ** 2)
    return g


def _get_peak_filters(tim):
    key = (tim.formant_freqs, tim.formant_gains, tim.formant_qs)
    if key in _PEAK_CACHE:
        return _PEAK_CACHE[key]
    freqs, gains, qs = csv_parse(tim.formant_freqs), csv_parse(tim.formant_gains), csv_parse(tim.formant_qs)
    filters = []
    for i, f in enumerate(freqs):
        g = gains[i] if i < len(gains) else 0.3
        q = qs[i] if i < len(qs) else 2.0
        sos = biquad_peak(f, g * 24.0, q)
        if sos is not None:
            filters.append(sos)
    _PEAK_CACHE[key] = filters
    return filters


def _apply_formants(w, tim, freq=0):
    """Post-synthesis EQ formants for KS/FM engines only."""
    filters = _get_peak_filters(tim)
    if not filters:
        return w
    out = w.copy()
    for sos in filters:
        out = sosfilt(sos, out)
    rms_in = np.sqrt(np.mean(w*w)) + 1e-10
    rms_out = np.sqrt(np.mean(out*out)) + 1e-10
    return out * (rms_in / rms_out)


def _brass_shape(w, vel, tim):
    if tim.lip_shape <= 0:
        return w
    amount = tim.lip_shape * vel**1.5
    if amount < 0.01:
        return w
    gain = 1 + amount * 4
    shaped = np.tanh(w * gain)
    td = np.tanh(gain)
    if td > 1e-6:
        shaped /= td
    return (1 - amount*0.5) * w + amount*0.5 * shaped


def synthesize(freq: float, dur: float, vel: float, tim, name: str = "default", nid: int = 0, pb_curve=None) -> 'np.ndarray':
    tail = min(tim.rel * 1.2 + 0.15, 2.5) if dur >= 0.2 else min(tim.rel + 0.20, 1.0)
    td = dur + tail
    n = int(SR * td)
    if n == 0:
        return np.zeros(0)
    tv = np.linspace(0, td, n, endpoint=False)
    v = vc(vel)
    short = dur < 0.2
    rng = np.random.RandomState((int(freq*100+vel*1000)+nid*7919) % (2**31))
    noff = int(SR * dur)
    _l = lf(freq)

    eb = tim.eff_bright(v)
    fc = tim.eff_fc(v)
    fc_al = tim.fc_alpha
    n_partials = tim.n + int(tim.n_extra * _l)
    B = (tim.inh * math.exp(tim.inh_stretch * pr(freq)) if tim.inh > 0 and tim.inh_stretch > 0
         else tim.inh * (1+3*_l) if tim.inh > 0 else 0.0)
    pscale = (freq/261.63)**tim.decay_exp if tim.decay_exp > 0 else 1.0
    ha = tim.get_harm_amps()
    formants = _parse_formants(tim)

    eff_strike = tim.strike_pos
    if eff_strike > 0.25 and freq > 300:
        t = min((freq - 300) / 700, 1.0)
        eff_strike = eff_strike * (1 - t) + 0.12 * t
    if fc > 0 and freq > 400:
        fc *= 1.0 + min((freq - 400) / 1000, 1.5) * 0.6

    vb = None
    if tim.vib_d > 0:
        ve = np.clip((tv-tim.vib_del)/0.2, 0, 1)
        vb = ve * tim.vib_d * np.sin(2*np.pi*tim.vib_r*tv)

    # merge pitch bend (semitones → cents) with vibrato
    pitch_mod = vb  # cents, or None
    if pb_curve is not None:
        if len(pb_curve) >= n:
            pb_cents = pb_curve[:n] * 100.0
        else:
            pb_cents = np.pad(pb_curve, (0, n - len(pb_curve)), mode='edge') * 100.0
        pitch_mod = (pitch_mod + pb_cents) if pitch_mod is not None else pb_cents

    n_str = strings_for_freq(freq, tim.n_strings)
    tune_tab = TUNE.get(n_str, TUNE[1])
    n_str = min(n_str, len(tune_tab))
    ds = detune_scale(freq)

    # --- shared modulation sources ---
    mod_slow = np.sin(2*np.pi*(0.5+rng.random()*2.0)*tv + rng.random()*6.283)
    mod_mid = np.sin(2*np.pi*(2.5+rng.random()*3.5)*tv + rng.random()*6.283)
    mod_fast = np.sin(2*np.pi*(5.0+rng.random()*4.0)*tv + rng.random()*6.283)

    # bow/breath as multiplicative modulation
    body_mod = np.ones(n)
    if tim.bow_noise > 0:
        onset = np.clip(tv/0.08, 0, 1)
        body_mod += tim.bow_noise * vel * onset * (0.6*mod_mid + 0.4*mod_fast)
    if tim.breath_noise > 0:
        onset = np.clip(tv/0.05, 0, 1)
        # Push noise above harmonics: lower cutoff at least 4× fundamental,
        # and scale amplitude down at low pitches where overlap is inevitable.
        bp_lo = max(freq * 4, 400)
        bp_hi = min(freq * 12, SR * 0.45)
        bp = get_bp(bp_lo, bp_hi) if n > 256 and bp_hi > bp_lo + 100 else None
        breath = sosfilt(bp, rng.randn(n)) * 0.3 if bp is not None else rng.randn(n) * 0.15
        # Reduce breath at low pitch where it overlaps harmonic content
        pitch_scale = min(1.0, max(0.15, (freq - 180) / 320))
        body_mod += tim.breath_noise * pitch_scale * vel * onset * breath

    # --- build partials with inline spectral envelope ---
    _nyq = SR * 0.5
    _nyq_lo = _nyq * 0.7          # start fade at 70% Nyquist (~15.4 kHz)
    _nyq_rng = _nyq - _nyq_lo     # fade range

    pdata: List[_PartialData] = []
    for k in range(1, n_partials+1):
        am = k * math.sqrt(1+B*k*k) if B > 0 else float(k)
        fk = freq * am
        if fk >= _nyq:
            break
        a = ha[k-1] if ha and k <= len(ha) else (1/k**tim.rolloff)*math.exp(-eb*(k-1))
        # vel_bright for harm_amps instruments: dim upper harmonics at low
        # velocity.  harm_amps encode the forte spectrum; this recreates the
        # natural darkening that eff_bright provides for rolloff instruments.
        if ha and k <= len(ha) and tim.vel_bright > 0:
            a *= math.exp(-tim.vel_bright * (1 - v) * (k - 1) * 0.1)
        if tim.even_atten > 0 and k % 2 == 0:
            a *= 1-tim.even_atten
        if fc > 0:
            a *= math.exp(-(fk/fc)**fc_al)
        # Nyquist soft guard — cosine fade from 0.7×Nyquist to Nyquist
        # prevents near-Nyquist partials from creating inter-sample harshness
        if fk > _nyq_lo:
            a *= 0.5 * (1 + math.cos(math.pi * (fk - _nyq_lo) / _nyq_rng))
        if eff_strike > 0:
            pos = math.sin(k*math.pi*eff_strike)
            a *= 1-tim.strike_depth*(1-max(abs(pos), 0.15))
            phi = math.pi if pos < 0 else 0.0
        else:
            phi = 0.0
        if formants:
            a *= _spectral_gain(fk, formants)
        if a < 0.0003:
            continue   # below audible threshold after normalisation
        a *= 1+0.03*(rng.random()-0.5)
        R = tim.b1*max(k-1, 0.05) + tim.b3*k*k
        R *= pscale*(1+0.05*(rng.random()-0.5))
        pdata.append(_PartialData(k, am, fk, a, R, phi))

    if not pdata:
        return np.zeros(n)

    ref_raw = tim.ref_amp()
    if formants:
        avg_g = sum(_spectral_gain(freq*k, formants) for k in range(1, min(tim.n+1, 6))) / min(tim.n, 5)
        ref = ref_raw * max(avg_g, 0.5)
    else:
        ref = ref_raw

    ph_strings = []
    # Phase offset from nid decorrelates multi-track unison: different
    # sub-tracks playing the same note won't sum perfectly in-phase.
    ph_nid = nid * 2.399
    for si in range(n_str):
        dt = 1.0 + (tune_tab[si] - 1.0) * ds
        fr = freq * dt
        ph_strings.append((2*np.pi*np.cumsum(fr*2**(pitch_mod/1200.0))/SR + ph_nid) if pitch_mod is not None else (2*np.pi*fr*tv + ph_nid))

    sf_mod = tim.spec_b * v * np.exp(-tv/max(tim.spec_tau, 0.001)) if tim.spec_b > 0 else None
    rd_tail = np.arange(max(n-noff, 0), dtype=np.float64)/SR if tim.rel_damp > 0 and noff < n else None
    coup = tim.coupling*(pscale**0.3 if tim.decay_exp > 0 else 1.0) if tim.coupling > 0 else 0
    coup_env = None
    if coup > 0 and n_str > 1:
        ff = 1.0/n_str
        coup_env = ff*np.exp(-(n_str-1)*coup*tv) + (1.0-ff)

    micro = tim.micro
    lf_sc = max(1 - 0.35 * _l, 0.1)

    w = np.zeros(n)
    for (k, am, fk, a, R, phi) in pdata:
        he = np.exp(-R*tv)
        if coup_env is not None:
            he *= coup_env
        if sf_mod is not None:
            kn = (k-1)/max(n_partials-1, 1)
            he *= 1 + sf_mod * kn
        if rd_tail is not None and k > 1:
            he[noff:] *= np.exp(-k*tim.rel_damp*rd_tail)

        if micro > 0.01 and a > 0.02 and k <= 10:
            shared = max(1.0 - (k-1)*0.12, 0.2)
            kf = min(k/5, 1.0)
            d_s = micro*(0.25+0.15*kf)*lf_sc
            d_i = micro*(0.08+0.03*kf)*lf_sc
            jitter = d_s*(0.7*mod_slow + 0.3*mod_mid) + d_i*rng.randn()*mod_fast
            ad = micro*(0.02+0.015*min(k/6, 1.0))*lf_sc
            am_mod = 1 + ad*(shared*mod_mid + (1-shared)*mod_fast)
        else:
            jitter, am_mod = 0.0, 1.0

        s_acc = np.sin(ph_strings[0]*am + phi + jitter)
        for si in range(1, n_str):
            if freq*tune_tab[si]*am < SR/2:
                s_acc += np.sin(ph_strings[si]*am + phi + jitter)
        if not isinstance(am_mod, float):
            s_acc *= am_mod
        s_acc *= he
        w += a * s_acc

    if n_str > 1:
        w /= math.sqrt(n_str)

    if tim.phantom > 0 and B > 0 and len(pdata) >= 2:
        pt = tim.phantom
        for i in range(len(pdata)):
            km, _, fk_m, a_m, R_m, _ = pdata[i]
            if a_m < 0.005:
                continue
            for j in range(i+1, len(pdata)):
                kn, _, fk_n, a_n, R_n, _ = pdata[j]
                if km+kn > 8:
                    break
                if pt*a_m*a_n < 0.0005:
                    continue
                fp = fk_m+fk_n
                if fp >= SR/2:
                    continue
                s = pt*a_m*a_n*np.sin(2*np.pi*fp*tv+rng.random()*6.28)
                if R_m+R_n > 0.005:
                    s *= np.exp(-(R_m+R_n)*tv)
                w += s

    if tim.det_c > 0 and tim.det_m > 0 and pdata:
        ec = tim.det_c*(1+_l)
        em = tim.det_m*(1+0.6*_l)
        dr = 2**(ec/1200)
        det_decay = np.exp(-tim.b1*tv)
        if noff < n:
            det_decay[noff:] *= np.exp(-8.0*np.arange(n-noff, dtype=np.float64)/SR)
        for r in (dr, 1/dr):
            dp = 2*np.pi*freq*r*tv
            for mk, am_k, _, amp, R, _ in pdata[:MAX_DET]:
                if freq*r*am_k >= SR/2:
                    break
                s = np.sin(dp*am_k + rng.random()*6.28) * det_decay
                if R > 0.005:
                    s *= np.exp(-R*tv)
                w += em*0.5*amp*s

    if tim.sub > 0 and freq/2 > 20:
        sub_sc = max(1.0 - max(freq - 300, 0) / 500, 0.0)
        if sub_sc > 0.01:
            sub_s = tim.sub*sub_sc*np.sin(2*np.pi*(freq/2)*tv)
            if tim.d2 < 10:
                sub_s *= np.exp(-2.5*tv)
            w += sub_s
    if tim.sub_third > 0 and freq*1.5 < SR/2:
        sub3_sc = max(1.0 - max(freq - 400, 0) / 600, 0.0)
        if sub3_sc > 0.01:
            st = tim.sub_third*sub3_sc*np.sin(2*np.pi*(freq*1.5)*tv)
            if tim.d2 < 10:
                st *= np.exp(-2.5*tv)
            w += st

    if tim.tw_leak > 0:
        for semi_off in [-1, 1]:
            lf_ = freq * 2**(semi_off/12.0)
            if 20 < lf_ < SR/2:
                w += tim.tw_leak*np.sin(2*np.pi*lf_*tv + rng.random()*6.28)

    if tim.noise > 0:
        nl = tim.noise*(1+tim.vn*v)*(1+tim.noise_hi*pr(freq) - 0.3*_l)
        if short:
            nl *= max(dur/0.2, 0.2)
        ns = rng.randn(n)
        # Sustained noise (noise_d < 10): raise lower cutoff above fundamental
        # to avoid sub-fundamental mud.  Transient noise (>=10): keep low for
        # hammer/pluck impact character.
        bp_lo = max(freq*1.5, 120) if tim.noise_d < 10 else max(freq*0.5, 80)
        bp = get_bp(bp_lo, min(freq*tim.noise_peak*(1+0.3*v), SR*0.45)) if n > 512 else None
        if bp is not None:
            ns = sosfilt(bp, ns)
        w += nl*ns*np.exp(-tim.noise_d*(1+0.3*_l)*tv)

    if ref > 0:
        w /= ref

    # 2x oversample for nonlinear stages to prevent aliasing
    has_nonlinear = (tim.lip_shape > 0 and v > 0.01) or tim.drive > 0
    n_orig = len(w)
    if has_nonlinear:
        w = resample_poly(w, 2, 1)

    w = _brass_shape(w, v, tim)

    if tim.drive > 0:
        d = tim.drive*(1+0.3*v)*(min(dur/0.2, 1) if short else 1)
        td_v = np.tanh(d)
        if td_v > 1e-6:
            w = np.tanh(w*d)/td_v

    if has_nonlinear:
        w = resample_poly(w, 1, 2)[:n_orig]

    # Remove DC introduced by asymmetric saturation / formant shaping
    w -= np.mean(w)

    if tim.trem_d > 0:
        w *= 1-tim.trem_d*0.5*(1+np.sin(2*np.pi*tim.trem_r*tv))

    w *= body_mod

    env_out = envelope(tim, dur, n, vel, freq, noff)
    out = w * env_out * (0.15 + 0.85*v)

    if tim.key_click > 0:
        cn = min(int(SR*0.003), n)
        if cn > 0:
            out[:cn] += tim.key_click*v*rng.randn(cn)*np.exp(-np.arange(cn)*8.0/cn)

    if short:
        fade = min(int(SR*0.004), n)
        if fade > 1:
            out[:fade] *= np.linspace(0, 1, fade)**0.5
    return out