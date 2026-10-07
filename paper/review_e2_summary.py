"""Summary table of experiment E2 (fixed and calibrated temperature-level rules), reports/36."""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, "tr-corpus", "results", "review_20261006")


def main():
    ls = pd.read_csv(os.path.join(ROOT, "tr-corpus", "results", "claims_20260925", "lead_support_cells.csv"))
    ls = ls[ls.event_set == "L2@1.0"]
    span = dict(zip(ls.held, ls.span_s))
    rows = []
    for sub in ["fixed_rules", "fixed_rules_calibrated"]:
        ir = pd.read_csv(os.path.join(B, sub, "internal_rows.csv"))
        pr = pd.read_csv(os.path.join(B, sub, "d6_positive_records.csv"))
        nr = pd.read_csv(os.path.join(B, sub, "negative_records.csv"))
        nr = nr[nr.eligible]
        for m, g in ir.groupby("model"):
            l2 = g[g.label == "L2@1.0"].assign(span=lambda t: t.held.map(span))
            d5 = l2.held.str.startswith("ds09")
            r = {"rule": m, "L2 detected /77": int(l2.detected.sum()),
                 "D1–D3 /16": int(l2[~d5].detected.sum()), "D5 /61": int(l2[d5].detected.sum()),
                 "pre-trigger alarms": int(l2.pre_trigger.sum())}
            for ell in (60, 120):
                o = l2[l2.span >= ell]
                r["ℓ=%d met/obs" % ell] = "%d/%d" % (int((o.detected & (o.lead >= ell)).sum()), len(o))
            for lab in ("L2@0.5", "L2@2.0", "L3 voltage", "ISC 25mV"):
                x = l2[["held", "detected"]].merge(g[g.label == lab][["held", "detected"]], on="held", suffixes=("", "_x"))
                r["flips " + lab] = "%d/%d" % (int((x.detected != x.detected_x).sum()), len(x))
            r["D6 detected /43"] = int(pr[pr.model == m].detected.sum())
            for pool in ("calibration", "source check", "D6"):
                for seg in ("first 300 s", "full"):
                    t = nr[(nr.model == m) & (nr.pool == pool) & (nr.segment == seg)]
                    r["%s %s" % (pool, seg)] = "%d/%d" % (int(t.fired.astype(bool).sum()), len(t))
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(B, "fixed_rules", "E2_summary.csv"), index=False)
    with open(os.path.join(B, "fixed_rules", "E2_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(t.to_markdown(index=False) + "\n")
    print(t.T.to_string())


if __name__ == "__main__":
    main()
