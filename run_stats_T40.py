"""
Paired significance testing at the deployment budget (paper Sec. VII-C).

Identical in method to run_stats.py -- exact McNemar against RAIG-tiered within
each condition, Holm step-down across the family, paired-bootstrap intervals on
the difference -- but reads the T=40 run. Kept as a separate file rather than
parameterising the original, so that the T=20 stress-test numbers remain
reproducible from their own script.

Writes results/stats_T40.json.
"""
import json

import numpy as np

import raig

OUT = "results"
PRIMARY = "heterogeneous_1.0"

meta = json.load(open(f"{OUT}/main_T40.json"))
names = meta["method_names"]
idx = {n: i for i, n in enumerate(names)}

report = {}
for cond in meta["results"]:
    try:
        correct = np.load(f"{OUT}/correctT40_{cond}.npy")
    except FileNotFoundError:
        continue
    ref = "RAIG-tiered"
    rows = []
    for other in [n for n in names if n != ref]:
        a, b = correct[:, idx[other]], correct[:, idx[ref]]
        n10, n01, _, p = raig.mcnemar_exact(a, b)
        d, lo, hi = raig.paired_bootstrap_ci(a, b)
        rows.append({"reference": ref, "comparator": other,
                     "acc_reference": float(b.mean()),
                     "acc_comparator": float(a.mean()),
                     "delta": float(d), "ci_lo": float(lo), "ci_hi": float(hi),
                     "n10_comparator_only": n10, "n01_reference_only": n01,
                     "n_discordant": n10 + n01, "p_raw": float(p)})
    adj, rej = raig.holm_bonferroni([r["p_raw"] for r in rows])
    for r, a_, rj in zip(rows, adj, rej):
        r["p_holm"] = float(a_)
        r["significant_holm_0.05"] = bool(rj)
    report[cond] = rows
    if cond == PRIMARY:
        print(f"=== {cond} at T={meta['T']} (n={len(correct)}), reference = {ref} ===")
        print(f"{'comparator':22s} {'acc':>6s} {'ref':>6s} {'delta':>8s} "
              f"{'95% CI':>16s} {'disc':>5s} {'p_raw':>10s} {'p_holm':>10s}")
        for r in sorted(rows, key=lambda x: x["p_raw"]):
            print(f"{r['comparator']:22s} {r['acc_comparator']*100:5.1f}% "
                  f"{r['acc_reference']*100:5.1f}% {r['delta']*100:+7.2f}pp "
                  f"[{r['ci_lo']*100:+5.2f},{r['ci_hi']*100:+5.2f}] "
                  f"{r['n_discordant']:5d} {r['p_raw']:10.2e} {r['p_holm']:10.2e}"
                  + ("  *" if r["significant_holm_0.05"] else ""))

json.dump({"primary_condition": PRIMARY, "T": meta["T"],
           "n_total": meta["n_total"],
           "correction": "Holm step-down, applied within each condition",
           "results": report}, open(f"{OUT}/stats_T40.json", "w"), indent=2)
print(f"\nwrote {OUT}/stats_T40.json")
