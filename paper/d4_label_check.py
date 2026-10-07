"""Visual check of the L2 guards on D4, where they move the label the most.

In D4 (multi-modal cells and modules, external heating) the guarded L2@1.0
label sits a median 1,476 s after the bare rate crossing -- far more than in
any other dataset.  A reviewer will reasonably ask whether the guards found
the runaway or threw it away.  This script recomputes both crossings from the
standardized corpus with the registry's own rule (trbench/common.onset_L2),
asserts that they reproduce the registry, records why each earlier crossing
was rejected, and plots every D4 record.

    uv run python paper/d4_label_check.py

Writes paper/figures/SF1_d4_labels.{png,pdf} and
paper/tables/S2_d4_guard_reasons.{csv,md}.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "trbench"), os.path.join(ROOT, "paper")]
import common as C          # noqa: E402
import schema as S          # noqa: E402
from make_figures import save_fig, save_table, GREY, RED, BLUE  # noqa: E402

REG = os.path.join(ROOT, "tr-corpus", "registry", "experiments.csv")
ORANGE = "#dd6b20"
MEMBERS = ["T_surface_neg", "T_surface_pos", "T_surface_mid"]


def rate_trace(t, T):
    """The smoothed gradient onset_L2 thresholds (same kernel, same gradient)."""
    dt = float(np.median(np.diff(t)))
    Ts = C._smooth(T, max(3, int(round(1.0 / dt)) | 1))
    return Ts, np.gradient(Ts, t), dt


def candidates(t, T, ceiling):
    """Every start of a >= 1 degC/s run held 2 s, with the guard outcome."""
    Ts, rate, dt = rate_trace(t, T)
    n = max(1, int(round(S.L2_SUSTAIN_S / dt)))
    run = np.convolve((rate >= S.L2_PRIMARY).astype(int), np.ones(n, int), mode="valid")
    starts = np.flatnonzero(run >= n)
    # collapse consecutive indices into episodes
    episodes = [s for k, s in enumerate(starts) if k == 0 or s != starts[k - 1] + 1]
    rows = []
    for i in episodes:
        j = min(len(Ts) - 1, i + int(round(S.L2_HORIZON_S / dt)))
        peak = float(np.nanmax(Ts[i:j + 1]))
        rise = peak - float(Ts[i])
        floor_ok = Ts[i] >= S.L2_T_FLOOR
        rise_ok = rise >= S.L2_RISE_MIN or (ceiling is not None and peak >= ceiling - 1.0)
        reason = "accepted" if floor_ok and rise_ok else (
            "below 60 °C floor" if not floor_ok else "rise < 50 °C in 60 s")
        rows.append(dict(t=float(t[i]), T=float(Ts[i]), rise_60s=rise, reason=reason))
        if reason == "accepted":
            break
    return pd.DataFrame(rows), Ts, rate


def main():
    reg = pd.read_csv(REG)
    d4 = reg[reg.dataset_id == "ds04_osf"].reset_index(drop=True)
    fig, axes = plt.subplots(3, 3, figsize=(7.4, 6.6))
    table = []
    for ax, (_, r) in zip(axes.flat, d4.iterrows()):
        df = pd.read_parquet(os.path.join(ROOT, "tr-corpus", r.file))
        t = df.time_s.to_numpy(float)
        T = df.T_surface_max.to_numpy(float)
        ceiling = r.T_censored_at if np.isfinite(r.T_censored_at) else None
        guarded = C.onset_L2(t, T, 1.0, ceiling=ceiling)
        bare = C.onset_L2(t, T, 1.0, guarded=False)
        assert guarded == r["t_onset_L2_1.0"], (r.experiment_id, guarded, r["t_onset_L2_1.0"])
        assert bare == r.t_onset_L2_unguarded, (r.experiment_id, bare, r.t_onset_L2_unguarded)
        cand, Ts, rate = candidates(t, T, ceiling)
        rej = cand[cand.reason != "accepted"]
        first = cand.iloc[0]
        table.append({
            "record": r.experiment_id, "cell": "%s %s, %s" % (r.cell_type, r.scale, r.soh),
            "bare crossing (s)": int(bare), "T at bare crossing (°C)": round(first["T"], 1),
            "rise in next 60 s (°C)": round(first.rise_60s, 1), "first rejection reason": first.reason,
            "rejected episodes": len(rej), "L2@1.0 (s)": int(guarded),
            "T at L2@1.0 (°C)": round(float(Ts[np.searchsorted(t, guarded)]), 1),
            "peak T (°C)": round(float(np.nanmax(T)), 1),
            "guard shift (s)": int(guarded - bare),
            "L1 venting (s)": "" if not np.isfinite(r.t_onset_L1) else int(r.t_onset_L1),
        })

        tm = t / 60.0
        for m in MEMBERS:
            if m in df and df[m].notna().any():
                ax.plot(tm, df[m], color="#cbd5e0", lw=0.6)
        ax.plot(tm, T, color="black", lw=0.9)
        ax.axhline(S.L2_T_FLOOR, color=GREY, lw=0.6, ls=":")
        ax.axvline(bare / 60, color=ORANGE, lw=1.0, ls="--")
        ax.axvline(guarded / 60, color=RED, lw=1.2)
        if np.isfinite(r.t_onset_L1):
            ax.axvline(r.t_onset_L1 / 60, color=BLUE, lw=1.0, ls=":")
        ax2 = ax.twinx()
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax2.plot(tm, np.clip(rate, -2, 8), color=GREY, lw=0.4, alpha=0.35)
        ax2.axhline(S.L2_PRIMARY, color=GREY, lw=0.5, ls="--")
        ax2.set_ylim(-2, 8)
        ax2.tick_params(labelsize=6, colors=GREY)
        if ax in axes[:, 2]:
            ax2.set_ylabel("dT/dt (°C/s)", fontsize=6.5, color=GREY)
        else:
            ax2.set_yticklabels([])
        ax2.spines["top"].set_visible(False)
        ax.set_title("%s  (%s %s, %s)" % (r.experiment_id, r.cell_type, r.scale, r.soh), fontsize=7, loc="left")
        ax.text(0.02, 0.95, "shift %+d s\nbare: %.0f °C, +%.0f °C/60 s" % (guarded - bare, first["T"], first.rise_60s),
                transform=ax.transAxes, fontsize=5.8, va="top", color="#2d3748")
        ax.tick_params(labelsize=6.5)
        ax.set_xlim(0, t.max() / 60)
        ax.set_ylim(0, max(120, np.nanmax(T) * 1.05))
    for ax in axes[2]:
        ax.set_xlabel("time (min)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("surface T (°C)", fontsize=7)
    handles = [plt.Line2D([], [], color="black", lw=0.9, label="T_surface_max (label probe)"),
               plt.Line2D([], [], color="#cbd5e0", lw=0.9, label="member thermocouples"),
               plt.Line2D([], [], color=ORANGE, ls="--", label="bare 1 °C/s crossing (no guards)"),
               plt.Line2D([], [], color=RED, lw=1.2, label="L2@1.0 (guarded, registry)"),
               plt.Line2D([], [], color=BLUE, ls=":", label="L1 venting"),
               plt.Line2D([], [], color=GREY, lw=0.5, label="dT/dt, right axis (dashed: 1 °C/s)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_fig(fig, "SF1_d4_labels")

    tab = pd.DataFrame(table)
    save_table(tab, "S2_d4_guard_reasons",
               "D4 L2@1.0 labels recomputed from the standardized corpus with trbench/common.onset_L2 "
               "(asserted equal to the registry). 'bare crossing' is the first 1 °C/s run held 2 s with no guards; "
               "'first rejection reason' is why the guards skipped it; 'rejected episodes' counts separate rate "
               "episodes skipped before the accepted one.")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
