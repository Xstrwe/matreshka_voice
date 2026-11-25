"""Evaluate a model on held-out real takes and a negative stream, through the C engine.

usage: python train/eval_real.py --model M.mvw --cli engine/mv_cli --takes DIR/eval \
           --neg-stream eval/neg_stream.wav --out OUTDIR [--thresholds 0.5,0.7,0.8,0.9]
Conditions per take: clean (as recorded, with a quiet room floor), noisy (coloured noise / hum, 5-20 dB SNR, 5 variants)
and far (robot-microphone simulation, 5 variants). Only aggregate numbers are printed.
"""
import argparse
import glob
import json
import os
import subprocess

import numpy as np
import soundfile as sf

from augment import colored_noise, farfield, rms

SR = 16000


def stream(takes, rnd, cond):
    parts, ends, t = [], [], 0
    for clip in takes:
        gap = np.zeros(int(rnd.uniform(1.2, 2.0) * SR), np.float32)
        c = clip.copy()
        if cond == "far":
            c = farfield(c, rnd)
        parts += [gap, c]
        t += len(gap) + len(c)
        ends.append(t / SR)
    parts.append(np.zeros(2 * SR, np.float32))
    x = np.concatenate(parts)
    if True:   # "clean" still gets a quiet room floor (40 dB below speech): real mics never output digital silence
        lvl = rms(np.concatenate(takes))
        nz = colored_noise(len(x), rnd)
        snr = {"clean": 40.0, "noisy": rnd.uniform(5, 20), "far": rnd.uniform(15, 30)}[cond]
        x = x + nz * lvl * 10 ** (-snr / 20)
    return np.clip(x, -1, 1), ends


def detections(cli, model, wav, thr):
    out = subprocess.run([cli, model, wav, "-t", str(thr)], check=True, capture_output=True, text=True).stdout
    return [float(l.split()[1]) for l in out.splitlines() if l.startswith("detection")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cli", required=True)
    ap.add_argument("--takes", required=True)
    ap.add_argument("--neg-stream", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9,0.95")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rnd = np.random.default_rng(99)
    thrs = [float(t) for t in a.thresholds.split(",")]
    files = []
    for style in ("normal", "whisper"):
        takes = [sf.read(f, dtype="float32")[0] for f in sorted(glob.glob(os.path.join(a.takes, style, "*.wav")))]
        if not takes:
            continue
        for cond in ("clean", "noisy", "far"):
            for r in range(1 if cond == "clean" else a.reps):
                x, ends = stream(takes, rnd, cond)
                fn = os.path.join(a.out, f"{style}_{cond}_{r}.wav")
                sf.write(fn, x, SR, subtype="PCM_16")
                files.append((style, cond, fn, ends))
    neg_h = sf.info(a.neg_stream).duration / 3600
    rows = []
    for thr in thrs:
        row = {"threshold": thr}
        for style in ("normal", "whisper"):
            for cond in ("clean", "noisy", "far"):
                hit = tot = 0
                for s, c, fn, ends in files:
                    if s != style or c != cond:
                        continue
                    d = np.array(detections(a.cli, a.model, fn, thr))
                    hit += sum(1 for e in ends if len(d) and np.any((d > e - 0.4) & (d < e + 1.0)))
                    tot += len(ends)
                if tot:
                    row[f"{style}_{cond}"] = f"{hit}/{tot}"
        fa = len(detections(a.cli, a.model, a.neg_stream, thr))
        row["fa_per_hour"] = round(fa / neg_h, 2)
        rows.append(row)
        print(json.dumps(row), flush=True)
    for _, _, fn, _ in files:   # do not keep derived audio around
        os.remove(fn)
    json.dump(rows, open(os.path.join(a.out, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
