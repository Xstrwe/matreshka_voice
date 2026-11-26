"""Превращение клипов в аугментированные окна по 1 с и признаки log-mel.

Запуск: python train/build_features.py --data ../data --split train --out ../feats/train.npz

Каждое окно — 100 кадров × 40 признаков (float16 для экономии памяти) и метка:
1 — слово в конце окна, 0 — всё остальное. Результат: .npz с массивами X и y.
"""
import argparse
import glob
import os

import numpy as np
import soundfile as sf

from augment import WIN_SAMPLES, augment, colored_noise, place, speed, trim
from features import logmel


def load_dir(pattern):
    """Прочитать все WAV по маске, свести в моно и обрезать тишину по краям."""
    out = []
    for p in sorted(glob.glob(pattern, recursive=True)):
        x, sr = sf.read(p, dtype="float32")
        assert sr == 16000, p
        if x.ndim > 1:
            x = x.mean(axis=1)
        out.append(trim(x))
    return out


def windows(pos, neg_hard, neg_sent, rnd, pos_aug, neg_aug, n_noise):
    """Собрать обучающие окна из синтетических клипов.

    pos_aug и neg_aug — сколько разных испорченных копий делать из каждого клипа;
    n_noise — сколько окон чистого шума добавить. Обычные предложения служат и
    негативами, и «банком» фоновых шумов (bank).
    """
    X, y = [], []
    bank = [s for s in neg_sent if len(s) > 8000]

    def add(w, label):
        X.append(logmel(w).astype(np.float16))
        y.append(label)

    for clip in pos:
        for _ in range(pos_aug):
            c = speed(clip, rnd) if rnd.random() < 0.7 else clip
            if bank and rnd.random() < 0.3:           # вставляем речь перед словом, как в «Привет, Вектор»
                lead = bank[rnd.integers(0, len(bank))]
                lead = lead[-int(rnd.uniform(0.2, 0.6) * 16000):]
                c = np.concatenate([lead, np.zeros(int(rnd.uniform(0.05, 0.3) * 16000), np.float32), clip])
            add(augment(place(c, WIN_SAMPLES, 0.55, 0.97, rnd), rnd, bank), 1)
        # Начало слова («Век…») — негатив: срабатывать надо только на целое слово.
        cut = clip[: int(len(clip) * rnd.uniform(0.35, 0.65))]
        add(augment(place(cut, WIN_SAMPLES, 0.6, 1.0, rnd), rnd, bank), 0)
    for clip in neg_hard:
        for _ in range(neg_aug):
            if rnd.random() < 0.7:
                clip = speed(clip, rnd)
            add(augment(place(clip, WIN_SAMPLES, 0.45, 1.0, rnd), rnd, bank), 0)
    for clip in neg_sent:
        for _ in range(neg_aug):
            # Длинное предложение — берём случайный кусок, короткое — кладём в окно целиком.
            if len(clip) > WIN_SAMPLES:
                s = rnd.integers(0, len(clip) - WIN_SAMPLES)
                w = clip[s: s + WIN_SAMPLES]
            else:
                w = place(clip, WIN_SAMPLES, 0.5, 1.0, rnd)
            add(augment(w, rnd, bank), 0)
    # Окна чистого шума от -60 до -15 дБ, без речи: тишина и фон тоже не должны будить робота.
    for _ in range(n_noise):
        w = colored_noise(WIN_SAMPLES, rnd) * 10 ** (rnd.uniform(-60, -15) / 20)
        add(np.clip(w, -1, 1).astype(np.float32), 0)
    return np.stack(X), np.array(y, dtype=np.int64)


def real_windows(takes, rnd, n_aug):
    """Окна из настоящих записей слова: скорость и высота, громкость, шум, эхо, эквалайзер
    и с вероятностью 50% имитация дальнего микрофона робота. На каждые 10 положительных
    окон из того же дубля делается одно отрицательное — обрезанное начало слова."""
    X, y = [], []
    for clip in takes:
        for i in range(n_aug):
            c = speed(clip, rnd, 0.88, 1.15) if rnd.random() < 0.6 else clip
            w = augment(place(c, WIN_SAMPLES, 0.55, 0.97, rnd), rnd, None, p_far=0.5)
            X.append(logmel(w).astype(np.float16)); y.append(1)
            if i % 10 == 0:
                cut = clip[: int(len(clip) * rnd.uniform(0.35, 0.6))]
                X.append(logmel(augment(place(cut, WIN_SAMPLES, 0.6, 1.0, rnd), rnd)).astype(np.float16)); y.append(0)
    return np.stack(X), np.array(y, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pos-aug", type=int, default=4)
    ap.add_argument("--neg-aug", type=int, default=3)
    ap.add_argument("--noise", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real", help="folder with real positive takes (*.wav, searched recursively)")
    ap.add_argument("--real-aug", type=int, default=150, help="augmented windows per real take")
    ap.add_argument("--real-only", action="store_true", help="only build windows from --real takes")
    a = ap.parse_args()
    rnd = np.random.default_rng(a.seed)     # один seed — одинаковые признаки при повторном запуске
    # Настоящие записи лежат только локально (data/real) и в репозиторий не попадают.
    if a.real:
        Xr, yr = real_windows(load_dir(os.path.join(a.real, "**", "*.wav")), rnd, a.real_aug)
        if a.real_only:
            np.savez(a.out, X=Xr, y=yr)
            print("real windows", Xr.shape)
            return
    d = os.path.join(a.data, a.split)
    pos = load_dir(os.path.join(d, "pos", "*.wav"))
    neg_hard = load_dir(os.path.join(d, "neg", "*_negh_*.wav"))
    neg_sent = load_dir(os.path.join(d, "neg", "*_negs_*.wav"))
    print(f"clips: pos {len(pos)} hard {len(neg_hard)} sent {len(neg_sent)}", flush=True)
    X, y = windows(pos, neg_hard, neg_sent, rnd, a.pos_aug, a.neg_aug, a.noise)
    if a.real:
        X, y = np.concatenate([X, Xr]), np.concatenate([y, yr])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, X=X, y=y)
    print("windows", X.shape, "positives", int(y.sum()))


if __name__ == "__main__":
    main()
