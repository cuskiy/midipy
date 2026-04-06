"""CC interpolation, smoothing, and pitch bend curve generation."""

import numpy as np
from scipy.signal import lfilter, lfilter_zi
from synth.timbre import SR

# Threshold (seconds) for deciding whether to use GM default or first
# event value as the initial CC state.  If the first CC event arrives
# later than this, the GM default is used for the preceding interval.
_LATE_EVENT_THRESH = 0.1


def smooth_cc(curve, tau_ms: float = 50.0):
    """1-pole LP with zero-transient init."""
    alpha = 1.0 - np.exp(-1000.0 / (SR * tau_ms))
    b, a = [alpha], [1.0, -(1.0 - alpha)]
    zi = lfilter_zi(b, a) * curve[0]
    return lfilter(b, a, curve, zi=zi)[0]


def smooth_cc_sidechain(curve, down_ms: float = 5.0, up_ms: float = 30.0):
    """Asymmetric smoother for CC7 sidechain: fast duck, smooth release.

    Uses min(fast, slow) trick — fast-attack/slow-release envelope:
    - Downward: fast curve drops first -> minimum selects it (quick duck).
    - Upward:   slow curve lags -> minimum selects it (smooth release).
    """
    fast = smooth_cc(curve, tau_ms=down_ms)
    slow = smooth_cc(curve, tau_ms=up_ms)
    return np.minimum(fast, slow)


def interp_cc(events: list, mx: int, default: float = 1.0):
    """Interpolate CC events to per-sample step curve (zero-order hold).

    GM standard: CC values take effect immediately and hold until the
    next CC event.  This function converts discrete CC events into a
    per-sample step function (sample-and-hold), NOT a linear ramp.
    Downstream smoothing filters handle temporal response.

    Args:
        events: list of (time, value) tuples, sorted by time.
        mx: buffer length in samples.
        default: GM-standard default value for this CC type.  Used as
                 the initial state when the first event arrives later
                 than _LATE_EVENT_THRESH seconds.

    Returns:
        Per-sample numpy array, or None if events is empty.
    """
    if not events:
        return np.full(mx, default)
    times = [t for t, v in events]
    vals = [v for t, v in events]
    # Use the GM default when the first event is late (> 100 ms);
    # otherwise use the first event's own value (covers sidechain
    # patterns where CC7 starts at 0 intentionally).
    if times[0] > _LATE_EVENT_THRESH:
        init_val = default
    else:
        init_val = vals[0]

    # Build step function: hold each value until just before the next
    # event, then step to the new value.
    _EPS = 0.5 / SR  # half a sample
    step_t = [0.0]
    step_v = [init_val]
    for t, v in zip(times, vals):
        if t <= 0.0:
            # Override initial value
            step_v[0] = v
            continue
        # Hold previous value until just before this event
        if t - _EPS > step_t[-1]:
            step_t.append(t - _EPS)
            step_v.append(step_v[-1])
        # Step to new value
        step_t.append(t)
        step_v.append(v)
    # Hold final value to end of buffer
    step_t.append(mx / SR + 1.0)
    step_v.append(step_v[-1])
    t_arr = np.linspace(0, mx / SR, mx, endpoint=False)
    return np.interp(t_arr, step_t, step_v)


def make_pb_curve(pb_events: list, start: float, dur: float):
    """Interpolate pitch bend events into per-sample curve (semitones)."""
    if not pb_events:
        return None
    end = start + dur
    init_val = 0.0
    for t, v in pb_events:
        if t <= start + 1e-6:
            init_val = v
        else:
            break
    margin = 0.002
    during = [(t, v) for t, v in pb_events if start - 1e-6 <= t <= end + margin]
    final_val = init_val
    for t, v in during:
        final_val = v
    all_vals = [init_val] + [v for _, v in during]
    if all(abs(v) < 0.01 for v in all_vals):
        return None
    n = int(SR * dur)
    if n == 0:
        return None
    ts = [0.0]
    vs = [init_val]
    for t, v in during:
        rt = max(0.0, min(t - start, dur))
        if ts and abs(rt - ts[-1]) < 1e-9:
            vs[-1] = v
        else:
            ts.append(rt)
            vs.append(v)
    if ts[-1] < dur - 1e-6:
        ts.append(dur)
        vs.append(final_val)
    t_arr = np.linspace(0, dur, n, endpoint=False)
    return np.interp(t_arr, ts, vs)
