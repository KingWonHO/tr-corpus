# single-layer sequence models (GRU, LSTM, Mamba) with a layer-norm head
import numpy as np
import torch
import torch.nn as nn

D_MODEL = 64


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# GRU over the window, head on the last step
class GRUNet(nn.Module):
    def __init__(self, c_in, d=D_MODEL):
        super().__init__()
        self.rnn = nn.GRU(c_in, d, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


# LSTM over the window, head on the last step
class LSTMNet(nn.Module):
    def __init__(self, c_in, d=D_MODEL):
        super().__init__()
        self.rnn = nn.LSTM(c_in, d, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):
        return self.head(self.rnn(x)[0][:, -1]).squeeze(-1)


# one Mamba block (pure-torch mambapy), head on the last step
class MambaNet(nn.Module):
    def __init__(self, c_in, d=D_MODEL):
        super().__init__()
        from mambapy.mamba import Mamba, MambaConfig
        self.inp = nn.Linear(c_in, d)
        self.body = Mamba(MambaConfig(d_model=d, n_layers=1))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):
        h = self.body(self.inp(x))
        return self.head(h[:, -1]).squeeze(-1)


def build(name, c_in, t_len=None):
    if name == "gru":
        return GRUNet(c_in)
    if name == "lstm":
        return LSTMNet(c_in)
    if name == "mamba":
        return MambaNet(c_in)
    raise ValueError(name)


# standardised values with the availability mask appended as channels
def prep(X, M, mu, sd):
    Xs = np.where(M, (X - mu) / sd, 0.0)
    return np.concatenate([Xs, M.astype(np.float32)], axis=2).astype(np.float32)
