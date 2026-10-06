"""Training-adequacy check of the sequence probes (external review of 2026-10-03).

Re-runs the held-out design of ``run_validation.py`` (sensor values, no
observation age) for the four sequence models with the number of training
epochs changed from the stored 6 to 12 and 24, everything else identical.  The
6-epoch run is repeated as a reproduction check of the stored results.  Per
epoch it logs the training loss and, as a diagnostic that never selects
anything, the same weighted hazard loss on the source validation/test windows
(records the model is not fitted on).

Outputs per epoch setting: a full ``run_validation`` result directory
``epochs_<E>/`` and one ``loss_curves.csv``.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deep as DEEP              # noqa: E402
import make_windows as MW        # noqa: E402
import run_e3 as R               # noqa: E402
import run_validation as RV      # noqa: E402
import survival as SV            # noqa: E402

CURVES = []


def monitor_set(windows, horizon):
    d, meta = R.load_windows(windows, splits=("train", "val", "test"))
    names = np.array(d["features"])
    age = np.array([s.startswith("age__") or s == MW.AGE_CH for s in names])
    cols = np.flatnonzero(RV.representation_mask(names, "no_age") & ~age)
    y, w = SV.discrete_hazard_targets(d["y_time"], d["y_event"], horizon)
    obs = np.asarray(d["label_observed"]).astype(bool)
    idx = np.flatnonzero(np.isin(d["split"], ["val", "test"]) & obs & (w > 0))
    return (d["X"][idx][:, :, cols], d["mask"][idx][:, :, cols].astype(bool),
            y[idx].astype(np.float32), w[idx].astype(np.float32), len(cols))


def make_callback(name, seed, epochs, mon):
    import torch
    Xm, Mm, ym, wm, _ = mon

    def cb(ep, train_loss, net, mu, sd, dev):
        net.eval()
        F = torch.from_numpy(DEEP._prep(Xm, Mm, mu, sd))
        y, w = torch.from_numpy(ym), torch.from_numpy(wm)
        pos = float((ym == 0).sum()) / max(float(ym.sum()), 1.0)
        lossf = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos, device=dev))
        num = den = 0.0
        with torch.no_grad():
            for i in range(0, len(F), 1024):
                xb, yb, wb = F[i:i + 1024].to(dev), y[i:i + 1024].to(dev), w[i:i + 1024].to(dev)
                num += float((lossf(net(xb), yb) * wb).sum()); den += float(wb.sum())
        CURVES.append(dict(model=name, seed=seed, epochs=epochs, epoch=ep + 1,
                           train_loss=train_loss, monitor_loss=num / max(den, 1e-9)))
    return cb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", default="6,12,24")
    ap.add_argument("--models", default="gru,mamba,itransformer,convtransformer")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--windows", default="W60_native")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mon = monitor_set(a.windows, R.HORIZON_S)
    print("monitor set: %d windows (source val/test, weight > 0)" % len(mon[0]), flush=True)
    orig = DEEP.fit_seq
    for E in [int(e) for e in a.epochs.split(",")]:
        t0 = time.time()

        def patched(name, Xtr, Mtr, ytr, wtr, seed=0, epochs=None, verbose=False, on_epoch=None, _E=E):
            assert Xtr.shape[2] == mon[4], "monitor columns differ from the fitted representation"
            return orig(name, Xtr, Mtr, ytr, wtr, seed=seed, epochs=_E,
                        on_epoch=make_callback(name, seed, _E, mon))

        DEEP.fit_seq = patched
        sys.argv = ["run_validation.py", "--windows", a.windows, "--models", a.models,
                    "--seeds", a.seeds, "--ages", "no_age", "--out", os.path.join(a.out, "epochs_%d" % E)]
        RV.main()
        pd.DataFrame(CURVES).to_csv(os.path.join(a.out, "loss_curves.csv"), index=False)
        print("epochs %d: %.0fs" % (E, time.time() - t0), flush=True)
    DEEP.fit_seq = orig


if __name__ == "__main__":
    main()
