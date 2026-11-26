"""Потоковая проверка модели через движок на C (mv_cli).

Склеивает из отложенных клипов длинные тестовые записи, прогоняет по ним движок
так же, как на роботе, и считает полноту (долю найденных слов) и число ложных
срабатываний в час. Проверяется именно C-движок, а не PyTorch, — то, что будет на роботе.

Запуск: python train/eval_stream.py --model models/vektor_ru.mvw --cli engine/mv_cli \
           --data ../data --out ../eval
"""
import argparse
import glob
import json
import os
import subprocess

import numpy as np
import soundfile as sf

from augment import colored_noise, reverb, rms, trim

SR = 16000


def load(pattern):
    """Прочитать WAV по маске, свести в моно и обрезать тишину."""
    out = []
    for p in sorted(glob.glob(pattern)):
        x, sr = sf.read(p, dtype="float32")
        assert sr == SR
        out.append(trim(x.mean(axis=1) if x.ndim > 1 else x))
    return out


# Привести клип к случайной громкости от lo до hi дБ (по RMS).
def level(x, rnd, lo=-30, hi=-14):
    return x / (rms(x) + 1e-9) * 10 ** (rnd.uniform(lo, hi) / 20)


def build_pos(pos, fillers, rnd, noisy):
    """Запись из слов с паузами 1–2 с; вернуть звук и моменты конца каждого слова (в секундах)."""
    parts, ends, t = [], [], 0
    for clip in pos:
        gap = np.zeros(int(rnd.uniform(1.0, 2.0) * SR), np.float32)
        if fillers and rnd.random() < 0.5:     # иногда перед словом звучит другая речь
            f = fillers[rnd.integers(0, len(fillers))][: int(1.5 * SR)]
            f = level(f, rnd)
            gap[: len(f)] = f[: len(gap)]
        c = reverb(clip, rnd) if noisy and rnd.random() < 0.5 else clip
        c = level(c, rnd)
        parts += [gap, c]
        t += len(gap) + len(c)
        ends.append(t / SR)
    parts.append(np.zeros(2 * SR, np.float32))
    x = np.concatenate(parts)
    if noisy:
        x = add_noise(x, rnd, 5, 20)
    return x, ends


def add_noise(x, rnd, snr_lo, snr_hi):
    """Добавить шум кусками по 10 с, каждый кусок со своим видом и уровнем шума."""
    out = x.copy()
    seg = 10 * SR
    for s in range(0, len(x), seg):
        n = min(seg, len(x) - s)
        nz = colored_noise(n, rnd)
        out[s:s + n] += nz * 10 ** ((-24 - rnd.uniform(snr_lo, snr_hi)) / 20)
    return np.clip(out, -1, 1)


def build_neg(neg, rnd, minutes):
    """Запись длиной minutes минут только из негативов: любое срабатывание на ней — ложное."""
    parts, total = [], 0
    while total < minutes * 60 * SR:
        for i in rnd.permutation(len(neg)):
            c = neg[i]
            if rnd.random() < 0.4:
                c = reverb(c, rnd)
            c = level(c, rnd)
            gap = np.zeros(int(rnd.uniform(0.1, 1.5) * SR), np.float32)
            parts += [c, gap]
            total += len(c) + len(gap)
            if total >= minutes * 60 * SR:
                break
    return add_noise(np.concatenate(parts), rnd, 5, 30)


def run(cli, model, wav, thr):
    """Запустить mv_cli; вернуть времена срабатываний и коэффициент реального времени
    (последнее число итоговой строки)."""
    out = subprocess.run([cli, model, wav, "-t", str(thr)], check=True, capture_output=True, text=True).stdout
    dets = [float(l.split()[1]) for l in out.splitlines() if l.startswith("detection")]
    rtf = float(out.strip().splitlines()[-1].split()[-1])
    return dets, rtf


def recall(dets, ends):
    """Слово найдено, если срабатывание пришло от 0,4 с до конца слова до 1 с после.
    Срабатывания вне этих интервалов считаются ложными."""
    dets = np.array(dets)
    hit = sum(1 for e in ends if len(dets) and np.any((dets > e - 0.4) & (dets < e + 1.0)))
    fa = sum(1 for d in dets if not any(e - 0.4 < d < e + 1.0 for e in ends))
    return hit / len(ends), fa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cli", required=True)
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../eval")
    ap.add_argument("--neg-minutes", type=float, default=120)
    ap.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9,0.95")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rnd = np.random.default_rng(1234)     # фиксированный seed — одинаковые тестовые записи
    sets = {
        "libritts_heldout": (load(f"{a.data}/test/pos/librittstest_*.wav"), load(f"{a.data}/test/neg/librittstest_*.wav")),
        "macos_milena": (load(f"{a.data}/say/pos/*.wav"), load(f"{a.data}/say/neg/*.wav")),
    }
    files = {}
    for name, (pos, neg) in sets.items():
        fill = [c for c in neg if len(c) > SR]
        for noisy in (False, True):
            reps = 1 if len(pos) > 100 else 8      # набор Milena маленький — повторяем его с другим шумом и паузами
            x, ends = build_pos(pos * reps, fill, rnd, noisy)
            fn = os.path.join(a.out, f"pos_{name}_{'noisy' if noisy else 'clean'}.wav")
            sf.write(fn, x, SR, subtype="PCM_16")
            files[fn] = ends
    negs = sets["libritts_heldout"][1] + sets["macos_milena"][1]
    neg_fn = os.path.join(a.out, "neg_stream.wav")
    sf.write(neg_fn, build_neg(negs, rnd, a.neg_minutes), SR, subtype="PCM_16")
    neg_hours = sf.info(neg_fn).duration / 3600

    results = []
    for thr in [float(t) for t in a.thresholds.split(",")]:
        row = {"threshold": thr}
        for fn, ends in files.items():
            dets, rtf = run(a.cli, a.model, fn, thr)
            r, fa = recall(dets, ends)
            key = os.path.basename(fn)[4:-4]
            row[f"recall_{key}"] = round(r, 4)
            row[f"n_{key}"] = len(ends)
        dets, rtf = run(a.cli, a.model, neg_fn, thr)
        row["false_accepts"] = len(dets)
        row["fa_per_hour"] = round(len(dets) / neg_hours, 2)
        row["neg_hours"] = round(neg_hours, 2)
        row["realtime_factor"] = rtf
        results.append(row)
        print(json.dumps(row), flush=True)
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
