"""The trained deep tier: recurrent, SSM, attention, and a convolutional hybrid.

The protocol (reports/02 section 5) asked for one entry per architecture family
rather than several per family, because errors within a family are correlated
and a second member buys almost no evidence.  So: a GRU for recurrence, a
single-layer Mamba for state-space, iTransformer for attention, and a
conv-front-end transformer for the hybrid the domain literature reports on.

Three things are shared with the rest of the benchmark and must stay shared, or
the comparison stops being about architecture:

  panel        selected by masking columns of the same window tensor, never by
               rebuilding windows
  target       the same discrete-time hazard target and per-window weights the
               tree models get
  output       one risk score per window, so the same calibration, operating
               curve and lead-time accounting apply unchanged

Missingness is carried, not imputed.  The availability mask is concatenated as
extra input channels -- the tree models get it as an `avail` feature, and a
sequence model that cannot tell "absent" from "zero" would be solving an easier
and wronger problem.

Scaling is the one thing these need that the trees did not.  Windows in the
`raw` build are physical units spanning 12-600 degC, which no network trains on
directly.  Statistics come from the TRAINING FOLD ONLY, which keeps it leak-free
and makes it the `global` scheme of section 11 rather than the per-experiment
one that section 12 showed breaks transfer.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

SEQ_MODELS = ("gru", "mamba", "itransformer", "convtransformer")

D_MODEL = 64
EPOCHS = 6
BATCH = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------- architectures
class GRUNet(nn.Module):
    """Recurrence baseline: the family entry the plan carried from v2."""

    def __init__(self, c_in, d=D_MODEL):
        super().__init__()
        self.rnn = nn.GRU(c_in, d, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):                      # x: (B, T, C)
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class MambaNet(nn.Module):
    """Single-layer Mamba.

    Chosen over a deeper stack because the constraint here is 192 training
    experiments, not sequence length, and because a selective SSM scales
    linearly -- which is what makes #12's very long ARC records tractable at all.
    `mamba_ssm` has no Windows wheel, so this is the pure-torch `mambapy` port.
    """

    def __init__(self, c_in, d=D_MODEL):
        super().__init__()
        from mambapy.mamba import Mamba, MambaConfig
        self.inp = nn.Linear(c_in, d)
        self.body = Mamba(MambaConfig(d_model=d, n_layers=1))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):
        h = self.body(self.inp(x))
        return self.head(h[:, -1]).squeeze(-1)


class ITransformer(nn.Module):
    """Attention over CHANNELS, not time.

    The research question is which channels carry the warning, and iTransformer
    makes each variate a token, so the attention map is over exactly that.  A
    time-token transformer would put the question one indirection away.
    """

    def __init__(self, c_in, t_len, d=D_MODEL, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Linear(t_len, d)
        enc = nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True,
                                         dropout=0.1, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):                      # (B, T, C) -> tokens are C
        h = self.enc(self.embed(x.transpose(1, 2)))
        return self.head(h.mean(dim=1)).squeeze(-1)


class ConvTransformer(nn.Module):
    """The hybrid: strided convolution front end, attention over what survives.

    Onset is a local event in a long quiet record.  The convolution compresses
    the quiet part and keeps the rate information; attention then relates the
    few remaining positions.
    """

    def __init__(self, c_in, d=D_MODEL, heads=4, layers=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(c_in, d, kernel_size=5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(d, d, kernel_size=5, stride=2, padding=2), nn.GELU())
        enc = nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True,
                                         dropout=0.1, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):
        h = self.stem(x.transpose(1, 2)).transpose(1, 2)
        return self.head(self.enc(h).mean(dim=1)).squeeze(-1)


def build(name, c_in, t_len):
    if name == "gru":
        return GRUNet(c_in)
    if name == "mamba":
        return MambaNet(c_in)
    if name == "itransformer":
        return ITransformer(c_in, t_len)
    if name == "convtransformer":
        return ConvTransformer(c_in)
    raise ValueError(name)


# ------------------------------------------------------------------ plumbing
def _prep(X, M, mu, sd):
    """Standardised values with the availability mask appended as channels."""
    Xs = np.where(M, (X - mu) / sd, 0.0)
    return np.concatenate([Xs, M.astype(np.float32)], axis=2).astype(np.float32)


def fit_seq(name, Xtr, Mtr, ytr, wtr, seed=0, epochs=EPOCHS, verbose=False):
    """Fit one sequence model on a training fold -> risk-scoring callable.

    Windows with zero weight are post-onset or otherwise unlabelled; they are
    dropped rather than fed with weight 0 so the batch count reflects real work.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _device()

    keep = wtr > 0
    Xtr, Mtr, ytr, wtr = Xtr[keep], Mtr[keep], ytr[keep], wtr[keep]
    if ytr.sum() < 2 or (ytr == 0).sum() < 2:
        return lambda X, M: np.zeros(len(X))

    Mtr = Mtr.astype(bool)
    with np.errstate(invalid="ignore"):
        vals = np.where(Mtr, Xtr, np.nan)
        mu = np.nan_to_num(np.nanmean(vals, axis=(0, 1)))
        sd = np.nan_to_num(np.nanstd(vals, axis=(0, 1)), nan=1.0)
    sd = np.where(sd > 1e-6, sd, 1.0).astype(np.float32)
    mu = mu.astype(np.float32)

    F = torch.from_numpy(_prep(Xtr, Mtr, mu, sd))
    y = torch.from_numpy(ytr.astype(np.float32))
    w = torch.from_numpy(wtr.astype(np.float32))

    net = build(name, F.shape[2], F.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pos = float((ytr == 0).sum()) / max(float(ytr.sum()), 1.0)
    lossf = nn.BCEWithLogitsLoss(reduction="none",
                                 pos_weight=torch.tensor(pos, device=dev))
    n = len(F)
    steps = max(1, n // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=epochs * steps, pct_start=0.3)

    net.train()
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(steps):
            idx = perm[i * BATCH:(i + 1) * BATCH]
            xb, yb, wb = F[idx].to(dev), y[idx].to(dev), w[idx].to(dev)
            opt.zero_grad(set_to_none=True)
            l = (lossf(net(xb), yb) * wb).sum() / wb.sum().clamp(min=1e-6)
            l.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(l)
        if verbose:
            print("      %s ep%d loss %.4f" % (name, ep, tot / steps))

    net.eval()

    @torch.no_grad()
    def predict(X, M):
        out = []
        Fx = torch.from_numpy(_prep(X, M.astype(bool), mu, sd))
        for i in range(0, len(Fx), 1024):
            out.append(torch.sigmoid(net(Fx[i:i + 1024].to(dev))).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    return predict
