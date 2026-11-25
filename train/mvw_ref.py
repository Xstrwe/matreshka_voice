"""Pure numpy reference for a .mvw model (same math as engine/src/mv_nn.c)."""
import struct

import numpy as np


class MvwModel:
    def __init__(self, path):
        b = open(path, "rb").read()
        assert b[:4] == b"MVW1", "bad magic"
        w = struct.unpack("<11I", b[4:48])
        assert w[0] == 1
        self.n_mels, self.window, self.k = w[1], w[2], w[3]
        self.c = list(w[4:8])
        self.stride = list(w[8:11])
        self.threshold = struct.unpack("<f", b[48:52])[0]
        d = np.frombuffer(b[68:], dtype="<f4").astype(np.float64)
        o = 0

        def take(*shape):
            nonlocal o
            n = int(np.prod(shape))
            a = d[o:o + n].reshape(shape)
            o += n
            return a
        nm, k, c = self.n_mels, self.k, self.c
        self.mean, self.inv_std = take(nm), take(nm)
        self.w0, self.b0 = take(c[0], nm, k), take(c[0])
        self.blocks = []
        for i in range(3):
            self.blocks.append((take(c[i], k), take(c[i]), take(c[i + 1], c[i]), take(c[i + 1])))
        self.fcw, self.fcb = take(2, c[3]), take(2)
        assert o == len(d), "size mismatch"

    def score(self, feats):
        """feats: (window, n_mels) raw log-mel -> P(keyword)."""
        x = ((feats - self.mean) * self.inv_std).T            # (n_mels, T)
        T = x.shape[1] - self.k + 1
        cols = np.stack([x[:, t:t + self.k] for t in range(T)])   # (T, n_mels, k)
        y = np.maximum(np.einsum("tfk,ofk->ot", cols, self.w0) + self.b0[:, None], 0)
        for (dw, db, pw, pb), s in zip(self.blocks, self.stride):
            T = (y.shape[1] - self.k) // s + 1
            z = np.stack([np.sum(y[:, i * s:i * s + self.k] * dw, axis=1) for i in range(T)], axis=1) + db[:, None]
            y = np.maximum(pw @ z + pb[:, None], 0)
        p = y.mean(axis=1)
        l = self.fcw @ p + self.fcb
        return float(1.0 / (1.0 + np.exp(l[0] - l[1])))
