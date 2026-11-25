"""Нарезка длинной записи с повторами слова на отдельные дубли и деление на train/eval.

Вход: WAV 16 кГц моно и JSON с примерными границами участков речи
[{"start": s, "end": e, "n": слов_в_участке, "style": "normal"|"whisper", ...}].
Участок с n > 1 режется по n-1 самым глубоким провалам громкости. В eval уходят
участки целиком, чтобы почти одинаковые повторы не оказались сразу в обеих частях
(иначе проверка была бы слишком оптимистичной).

Запуск: python train/real_takes.py --wav REC.wav --segments segments.json --out DIR [--eval-frac 0.3]
Пишет DIR/{train,eval}/{normal,whisper}/take_XXX.wav и DIR/takes.json.
Записи хранятся только локально; репозиторий игнорирует весь звук и папки с данными.
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf

SR = 16000


def split_region(x, n):
    """Разрезать участок x на n слов; вернуть список пар (начало, конец) в отсчётах."""
    if n <= 1:
        return [(0, len(x))]
    hop = 160
    # Энергия по кадрам 25 мс с шагом 10 мс, в дБ и сглаженная по 5 кадрам.
    e = np.array([np.mean(x[i:i + 400] ** 2) for i in range(0, len(x) - 400, hop)])
    e = np.convolve(10 * np.log10(e + 1e-10), np.ones(5) / 5, "same")
    # Не режем у самых краёв участка: там провалы — это тишина до и после речи.
    lo, hi = int(0.2 * len(e) / n), len(e) - int(0.2 * len(e) / n)
    cuts = []
    cand = np.argsort(e[lo:hi]) + lo
    for c in cand:  # самые глубокие провалы, не ближе 0,2 с друг к другу
        if all(abs(c - k) > 20 for k in cuts):
            cuts.append(int(c))
        if len(cuts) == n - 1:
            break
    # Номер кадра -> отсчёт; +200 — середина 25-миллисекундного кадра.
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
    # Деление делается отдельно для обычного голоса и шёпота, чтобы оба были в eval.
    split = {}
    for style in sorted({s["style"] for s in segs}):
        idx = [i for i, s in enumerate(segs) if s["style"] == style]
        total = sum(segs[i]["n"] for i in idx)
        order = list(rnd.permutation(idx))
        got = 0
        for i in order:  # набираем eval целыми участками, пока не наберётся eval_frac всех слов
            if got + segs[i]["n"] <= round(a.eval_frac * total) or got == 0:
                split[i] = "eval"
                got += segs[i]["n"]
            else:
                split[i] = "train"
    takes, k = [], 0
    for i, s in enumerate(segs):
        # Запас 80 мс с каждой стороны, чтобы не срезать начало и конец слова.
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
