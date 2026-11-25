"""Обучение WakeNet на признаках из build_features.py.

Запуск: python train/train.py --train ../feats/train.npz --val ../feats/test.npz --out ../runs/vektor

В папку --out пишутся: norm.npy (mean и std признаков), best.pt (лучшая эпоха по полноте
при 0,1% ложных срабатываний на окно), last.pt (последняя эпоха) и log.json (метрики).
С --init дообучение начинается с готовой модели (так модель подстраивали под живой голос).
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from model import WakeNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, nargs="+", help="one or more .npz feature files")
    ap.add_argument("--init", help="run folder to fine-tune from (uses its last.pt and norm.npy)")
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--neg-weight", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"     # MPS — видеоускоритель Apple

    parts = [np.load(f) for f in a.train]
    Xtr = np.concatenate([p["X"] for p in parts]).astype(np.float32)
    ytr = np.concatenate([p["y"] for p in parts])
    if a.init:   # оставляем прежнюю нормализацию, иначе готовые веса перестанут подходить
        mean, std = np.load(os.path.join(a.init, "norm.npy"))
    else:
        # Среднее и разброс каждого из 40 признаков по всем кадрам; +1e-5 защищает от деления на 0.
        mean = Xtr.reshape(-1, Xtr.shape[2]).mean(0)
        std = Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-5
    np.save(os.path.join(a.out, "norm.npy"), np.stack([mean, std]))
    va = np.load(a.val)
    Xva, yva = va["X"].astype(np.float32), va["y"]
    norm = lambda X: torch.from_numpy((X - mean) / std)
    Xtr_t, Xva_t = norm(Xtr), norm(Xva).to(dev)
    ytr_t, yva_t = torch.from_numpy(ytr), torch.from_numpy(yva).to(dev)

    net = WakeNet()
    if a.init:
        net.load_state_dict(torch.load(os.path.join(a.init, "last.pt"), map_location="cpu"))
    net = net.to(dev)
    # AdamW с небольшим затуханием весов; OneCycle сначала разгоняет шаг обучения, потом плавно снижает.
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-3)
    steps = a.epochs * (len(Xtr_t) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    # Ошибка на негативе стоит в neg_weight раз дороже: ложное срабатывание хуже пропуска.
    cw = torch.tensor([a.neg_weight, 1.0], device=dev)
    best, log = -1.0, []
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(len(Xtr_t) // a.batch):
            idx = perm[i * a.batch:(i + 1) * a.batch]
            xb, yb = Xtr_t[idx].to(dev), ytr_t[idx].to(dev)
            # Маскирование в стиле SpecAugment: обнуляем до 4 соседних мел-полос и до 7 кадров,
            # чтобы сеть не полагалась на одну узкую деталь.
            if True:
                f0 = np.random.randint(0, 36); xb[:, :, f0:f0 + np.random.randint(0, 5)] = 0
                t0 = np.random.randint(0, 90); xb[:, t0:t0 + np.random.randint(0, 8), :] = 0
            # label_smoothing 0.05 не даёт сети быть «абсолютно уверенной» — это помогает обобщению.
            loss = F.cross_entropy(net(xb), yb, weight=cw, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        # Проверка на отложенных данных пачками по 2048 окон.
        net.eval()
        with torch.no_grad():
            p = torch.cat([torch.softmax(net(Xva_t[j:j + 2048]), 1)[:, 1] for j in range(0, len(Xva_t), 2048)])
        pos, neg = p[yva_t == 1], p[yva_t == 0]
        thr = torch.quantile(neg, 0.999).item()     # порог, при котором срабатывает 0,1% негативных окон
        rec = (pos > thr).float().mean().item()
        rec5 = (pos > 0.5).float().mean().item(); fa5 = (neg > 0.5).float().mean().item()
        log.append(dict(epoch=ep, loss=tot / (i + 1), recall_at_0p1pct_fa=rec, thr=thr, recall_0p5=rec5, fa_0p5=fa5))
        print(json.dumps(log[-1]), flush=True)
        if rec > best:
            best = rec
            torch.save(net.state_dict(), os.path.join(a.out, "best.pt"))
    torch.save(net.state_dict(), os.path.join(a.out, "last.pt"))
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
