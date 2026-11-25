"""Cut a long recording of repeated wake words into single takes and split train/eval.

Input: a 16 kHz mono WAV and a segments JSON with rough speech regions
[{"start": s, "end": e, "n": words_in_region, "style": "normal"|"whisper", ...}].
Regions with n > 1 are split at the n-1 deepest energy dips. Whole regions go to the
eval split so near-identical repeats never land on both sides.

usage: python train/real_takes.py --wav REC.wav --segments segments.json --out DIR [--eval-frac 0.3]
Writes DIR/{train,eval}/{normal,whisper}/take_XXX.wav and DIR/takes.json.
Recordings stay local; the repository ignores all audio and data folders.
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf

SR = 16000


def split_region(x, n):
    if n <= 1:
        return [(0, len(x))]
    hop = 160
    e = np.array([np.mean(x[i:i + 400] ** 2) for i in range(0, len(x) - 400, hop)])
    e = np.convolve(10 * np.log10(e + 1e-10), np.ones(5) / 5, "same")
    lo, hi = int(0.2 * len(e) / n), len(e) - int(0.2 * len(e) / n)
    cuts = []
    cand = np.argsort(e[lo:hi]) + lo
    for c in cand:  # deepest dips at least 0.2 s apart
        if all(abs(c - k) > 20 for k in cuts):
            cuts.append(int(c))
        if len(cuts) == n - 1:
            break
    b = [0] + sorted(k * hop + 200 for k in cuts) + [len(x)]
    return list(zip(b[:-1], b[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    x, sr = sf.read(a.wav, dtype="float32")
    assert sr == SR
    segs = json.load(open(a.segments))
    rnd = np.random.default_rng(a.seed)
    split = {}
    for style in sorted({s["style"] for s in segs}):
        idx = [i for i, s in enumerate(segs) if s["style"] == style]
        total = sum(segs[i]["n"] for i in idx)
        order = list(rnd.permutation(idx))
        got = 0
        for i in order:  # fill the eval split with whole regions up to eval_frac of the words
            if got + segs[i]["n"] <= round(a.eval_frac * total) or got == 0:
                split[i] = "eval"
                got += segs[i]["n"]
            else:
                split[i] = "train"
    takes, k = [], 0
    for i, s in enumerate(segs):
        a0 = max(0, int((s["start"] - 0.08) * SR))
        b0 = min(len(x), int((s["end"] + 0.08) * SR))
        reg = x[a0:b0]
        for (p, q) in split_region(reg, s["n"]):
            d = os.path.join(a.out, split[i], s["style"])
            os.makedirs(d, exist_ok=True)
            fn = os.path.join(d, f"take_{k:03d}.wav")
            sf.write(fn, reg[p:q], SR, subtype="PCM_16")
            takes.append(dict(file=fn, split=split[i], style=s["style"], region=i, dur=round((q - p) / SR, 3),
                              asr_ok=s.get("asr_ok"), hey=s.get("hey")))
            k += 1
    json.dump(takes, open(os.path.join(a.out, "takes.json"), "w"), indent=1)
    for sp in ("train", "eval"):
        for st in ("normal", "whisper"):
            print(sp, st, sum(1 for t in takes if t["split"] == sp and t["style"] == st))


if __name__ == "__main__":
    main()
