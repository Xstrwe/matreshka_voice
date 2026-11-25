"""Небольшая свёрточная сеть по времени (temporal CNN) для детекции слова.

На вход — окно из 100 кадров признаков (~1 с), на выход — два числа (логиты):
«не слово» и «слово». engine/src/mv_nn.c выполняет экспортированную версию этой
сети, где BatchNorm уже «вклеен» в веса свёрток (см. fold_bn и export.py).

Строение: обычная свёртка conv0 40→48, затем три блока «depthwise + pointwise»
(как в MobileNet): дешёвая свёртка по времени каждого канала отдельно и смешивание
каналов свёрткой 1×1. Шаг 2 в первых двух блоках уменьшает длину по времени в 4 раза.
В конце — среднее по времени и полносвязный слой.
"""
import torch
import torch.nn as nn

N_MELS = 40
WINDOW = 100        # длина окна в кадрах (~1,0 с)
K = 5               # ширина всех свёрток по времени, в кадрах
C0, C1, C2, C3 = 48, 48, 64, 64     # число каналов после conv0 и каждого блока
STRIDES = (2, 2, 1)                 # шаг по времени в блоках b1, b2, b3


class DwPw(nn.Module):
    """Блок depthwise + pointwise: свёртка по времени отдельно для каждого канала
    (groups=cin), затем свёртка 1×1, смешивающая каналы, BatchNorm и ReLU.
    Вместе это намного дешевле обычной свёртки cin→cout с тем же окном."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, K, stride=stride, groups=cin, bias=True)
        # Смещение у pw не нужно: его роль выполняет BatchNorm сразу после.
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        return torch.relu(self.bn(self.pw(self.dw(x))))


class WakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(N_MELS, C0, K, bias=False)
        self.bn0 = nn.BatchNorm1d(C0)
        self.b1 = DwPw(C0, C1, STRIDES[0])
        self.b2 = DwPw(C1, C2, STRIDES[1])
        self.b3 = DwPw(C2, C3, STRIDES[2])
        self.drop = nn.Dropout(0.2)     # действует только при обучении
        self.fc = nn.Linear(C3, 2)

    def forward(self, x):            # x: (B, T, N_MELS), уже нормализован (mean/std из обучения)
        x = x.transpose(1, 2)        # Conv1d ждёт (B, каналы, время)
        x = torch.relu(self.bn0(self.conv0(x)))
        x = self.b3(self.b2(self.b1(x)))
        # Среднее по времени (global average pooling) делает ответ независимым
        # от того, где именно в окне прозвучало слово.
        return self.fc(self.drop(x.mean(dim=2)))


def fold_bn(w, bn, bias=None):
    """Вернуть (w', b'), такие что bn(conv(x, w) + bias) == conv(x, w') + b'.

В режиме вывода BatchNorm — это просто y = (x - mean) / sqrt(var + eps) * gamma + beta,
то есть умножение на число и сдвиг для каждого канала. Их можно заранее умножить
на веса свёртки, и на роботе BatchNorm считать уже не нужно.
"""
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    b = bias if bias is not None else torch.zeros_like(bn.running_mean)
    return w * s.view(-1, *([1] * (w.dim() - 1))), (b - bn.running_mean) * s + bn.bias
