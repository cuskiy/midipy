"""Auto-calibrate REF_RMS and KS_GAIN tables.

Renders each instrument at A4 (440 Hz), vel=0.85, dur=0.5s and measures
the RMS of the steady-state portion. For KS instruments, also measures
the KS/additive gain ratio for crossfade calibration.

Usage:
    python -m synth.calibrate          # print tables
    python -m synth.calibrate --write  # overwrite synth/gm.py in-place
"""
import sys, os, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synth.timbre import SR
from synth.instruments import TIMBRES
from synth.gm import KS_PLUCKED, KS_ALWAYS
from synth.additive import synthesize as _additive, _apply_formants
from synth.ks import synthesize_plucked as _ks
from synth import synthesize

A4 = 440.0
DUR = 0.5
VEL = 0.85


def _rms(w: np.ndarray) -> float:
    if len(w) < 100:
        return 0.0
    start = min(int(SR * 0.05), len(w) // 4)
    end = min(int(SR * 0.4), len(w))
    seg = w[start:end]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) > 0 else 0.0


def calibrate_ref_rms() -> dict:
    result = {}
    for name, tim in sorted(TIMBRES.items()):
        w = synthesize(A4, DUR, VEL, tim, name, 0)
        result[name] = round(_rms(w), 4)
    return result


def calibrate_ks_gain() -> dict:
    result = {}
    for name in sorted(KS_PLUCKED):
        if name in KS_ALWAYS:
            continue
        tim = TIMBRES[name]
        ratios = []
        for midi in (48, 66, 84):
            freq = A4 * 2 ** ((midi - 69) / 12.0)
            for vel in (0.5, 1.0):
                ks_w = _ks(freq, 0.6, vel, tim, name, 0)
                if tim.formant_freqs:
                    ks_w = _apply_formants(ks_w, tim, freq)
                pk = np.max(np.abs(ks_w))
                if pk > 1.0:
                    ks_w *= 1.0 / pk
                ad_w = _additive(freq, 0.6, vel, tim, name, 0)
                n = min(len(ks_w), len(ad_w))
                rk = np.sqrt(np.mean(ks_w[:n] ** 2)) + 1e-10
                ra = np.sqrt(np.mean(ad_w[:n] ** 2)) + 1e-10
                ratios.append(rk / ra)
        result[name] = round(float(np.median(ratios)), 4)
    return result


def _format_dict(d: dict, name: str) -> str:
    items = [f'    "{k}": {v},' for k, v in sorted(d.items())]
    return f"{name} = {{\n" + "\n".join(items) + "\n}"


def write_to_gm(ref_rms: dict, ks_gain: dict) -> None:
    gm_path = os.path.join(os.path.dirname(__file__), "gm.py")
    with open(gm_path) as f:
        src = f.read()
    src = re.sub(
        r'REF_RMS = \{[^}]+\}',
        _format_dict(ref_rms, "REF_RMS"),
        src)
    src = re.sub(
        r'KS_GAIN = \{[^}]+\}',
        _format_dict(ks_gain, "KS_GAIN"),
        src)
    with open(gm_path, 'w') as f:
        f.write(src)
    print(f"  Written to {gm_path}")


if __name__ == "__main__":
    write_mode = "--write" in sys.argv

    print("Calibrating REF_RMS...")
    ref_rms = calibrate_ref_rms()
    print(_format_dict(ref_rms, "REF_RMS"))

    print("\nCalibrating KS_GAIN...")
    ks_gain = calibrate_ks_gain()
    print(_format_dict(ks_gain, "KS_GAIN"))

    if write_mode:
        write_to_gm(ref_rms, ks_gain)
    else:
        print("\nRun with --write to update synth/gm.py")
