"""Auto-calibrate KS_GAIN table (KS/additive crossover gain ratio).

Usage:
    python -m synth.calibrate          # print table
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

A4 = 440.0


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


def _format_dict(d: dict) -> str:
    items = [f'    "{k}": {v},' for k, v in sorted(d.items())]
    return "KS_GAIN = {\n" + "\n".join(items) + "\n}"


def write_to_gm(ks_gain: dict) -> None:
    gm_path = os.path.join(os.path.dirname(__file__), "gm.py")
    with open(gm_path) as f:
        src = f.read()
    src = re.sub(r'KS_GAIN = \{[^}]+\}', _format_dict(ks_gain), src)
    with open(gm_path, 'w') as f:
        f.write(src)
    print(f"  Written to {gm_path}")


if __name__ == "__main__":
    print("Calibrating KS_GAIN...")
    ks_gain = calibrate_ks_gain()
    print(_format_dict(ks_gain))
    if "--write" in sys.argv:
        write_to_gm(ks_gain)
    else:
        print("\nRun with --write to update synth/gm.py")
