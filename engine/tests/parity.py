"""Check that the C engine matches the Python reference.

usage: python engine/tests/parity.py MODEL.mvw INPUT.wav [path/to/mv_cli]
Exits non-zero when features or scores differ more than the tolerance.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "train"))
from features import logmel  # noqa: E402
from mvw_ref import MvwModel  # noqa: E402


def main():
    model, wav = sys.argv[1], sys.argv[2]
    cli = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "..", "mv_cli")
    x, sr = sf.read(wav, dtype="int16")
    assert sr == 16000
    ref = logmel(x.astype(np.float32) / 32768.0)
    with tempfile.TemporaryDirectory() as td:
        ff, sc = os.path.join(td, "f.bin"), os.path.join(td, "s.txt")
        subprocess.run([cli, model, wav, "--features", ff, "--scores", sc, "--quiet", "-t", "2"], check=True,
                       stdout=subprocess.DEVNULL)
        cf = np.fromfile(ff, dtype=np.float32).reshape(-1, ref.shape[1])
        scores = np.loadtxt(sc, ndmin=2)
    assert cf.shape == ref.shape, (cf.shape, ref.shape)
    # compare where the signal is not near-silent (log of tiny power is ill-conditioned)
    mask = ref > -10
    fdiff = np.abs(cf - ref)[mask].max() if mask.any() else 0.0
    m = MvwModel(model)
    sdiff = 0.0
    for fi, s in scores:
        fi = int(fi)
        r = m.score(cf[fi - m.window + 1: fi + 1].astype(np.float64))
        sdiff = max(sdiff, abs(r - s))
    print(f"frames {len(ref)}  max|feature diff| {fdiff:.2e}  network runs {len(scores)}  max|score diff| {sdiff:.2e}")
    ok = fdiff < 1e-2 and sdiff < 1e-3
    print("PARITY OK" if ok else "PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
