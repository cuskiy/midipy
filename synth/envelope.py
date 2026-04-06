import numpy as np
from .timbre import SR, lf, vc

def envelope(tim, dur, n, vel, freq, noff):
    lo_factor, vel_c = lf(freq), vc(vel)
    att_time = min(tim.att * (1 - tim.va*vel_c), dur*0.25)
    dec_time = min(tim.d1, dur*0.4)
    att_n, dec_n = int(SR*att_time), int(SR*dec_time)
    dec_end = att_n + dec_n
    rel_start = min(noff if noff is not None else n, n)
    env = np.zeros(n)

    if att_n > 0:
        t_att = np.linspace(0, 1, att_n)
        pw = 0.8 + 0.5*(1-vel_c) + 0.3*lo_factor
        env[:att_n] = 0.4*(1 - np.cos(np.pi*t_att))/2 + 0.6*t_att**pw
    if dec_n > 0:
        dec_level = min(tim.d1l + (1-tim.d1l)*vel_c*0.15 + tim.ps*lo_factor, 0.995)
        env[att_n:att_n+dec_n] = dec_level + (1-dec_level)*np.exp(-2.7*(1-0.35*lo_factor)*np.linspace(0, 1, dec_n))

    sus_n = max(0, rel_start-dec_end)
    if sus_n > 0:
        sus_level = env[dec_end-1] if dec_end > 0 else tim.d1l
        tau = max(tim.d2*(1+tim.vd*vel_c), 0.05) * (1+(tim.pdm-1)*lo_factor)
        sus_t = np.linspace(0, sus_n/SR, sus_n)
        if tim.d2s > 0 and tim.pr < 1:
            tau_s = tim.d2s*(1+0.5*lo_factor)
            env[dec_end:rel_start] = sus_level * (tim.pr*np.exp(-sus_t/tau) + (1-tim.pr)*np.exp(-sus_t/tau_s))
        else:
            env[dec_end:rel_start] = sus_level * np.exp(-sus_t/tau)

    if rel_start < n:
        rel_level = env[rel_start-1] if rel_start > 0 else 0
        rate = 5.0 / max(tim.rel, 0.02) if dur >= 0.2 else 10.0
        env[rel_start:] = rel_level * np.exp(-rate*np.arange(n-rel_start, dtype=np.float64)/SR)

    if tim.live > 0 and n > SR*0.08:
        ta = np.linspace(0, n/SR, n, endpoint=False)
        rng = np.random.RandomState(int(freq*100+vel*1000) % (2**31))
        r1, r2 = 2+rng.random()*2, 4.5+rng.random()*3
        onset = np.clip(ta/0.12, 0, 1)
        env *= 1 + tim.live*onset*(0.6*np.sin(2*np.pi*r1*ta+rng.random()*6.28)
                                 + 0.4*np.sin(2*np.pi*r2*ta+rng.random()*6.28))

    ac = min(128, n)
    if ac > 1: env[-ac:] *= 0.5*(1 + np.cos(np.linspace(0, np.pi, ac)))
    return env
