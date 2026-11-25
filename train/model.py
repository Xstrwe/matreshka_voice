"""Small temporal CNN. engine/src/mv_nn.c runs the exported (BatchNorm-folded) version."""
import torch
import torch.nn as nn

N_MELS = 40
WINDOW = 100        # frames (~1.0 s)
K = 5
C0, C1, C2, C3 = 48, 48, 64, 64
STRIDES = (2, 2, 1)


class DwPw(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, K, stride=stride, groups=cin, bias=True)
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
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(C3, 2)

    def forward(self, x):            # x: (B, T, N_MELS), already normalized
        x = x.transpose(1, 2)
        x = torch.relu(self.bn0(self.conv0(x)))
        x = self.b3(self.b2(self.b1(x)))
        return self.fc(self.drop(x.mean(dim=2)))


def fold_bn(w, bn, bias=None):
    """Return (w', b') so that bn(conv(x, w) + bias) == conv(x, w') + b'."""
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    b = bias if bias is not None else torch.zeros_like(bn.running_mean)
    return w * s.view(-1, *([1] * (w.dim() - 1))), (b - bn.running_mean) * s + bn.bias
