"""
Paired significance testing for the extended comparison (paper Sec. VII-C),
with a multiple-comparison correction the conference version lacked.

Runs the exact McNemar test for every comparison of interest against
RAIG-tiered at the pre-registered primary condition, then applies Holm
step-down across the family. The conference version reported three
uncorrected p-values; with eight methods the family is larger and the
correction is no longer optional.

Also reports, for each comparison, the paired-bootstrap interval on the
accuracy difference and the count of discordant pairs, since with accuracies
in the single digits a p-value alone hides how few trials actually
distinguish the methods.

Writes results/stats_ext.json.
"""
import json
import numpy as np

import raig

OUT = "results"
PRIMARY = "heterogeneous_1.0"

meta = json.load(open(f"{OUT}/main_ext.json"))
names = meta["method_names"]
idx = {n: i for i, n in enumerate(names)}

report = {}
for cond in meta["results"]:
    try:
        correct = np.load(f"{OUT}/correct_{cond}.npy")
    except FileNotFoundError:
        continue
    ref = "RAIG-tiered"
    comps = [n for n in names if n != ref]
    rows = []
    for other in comps:
        a, b = correct[:, idx[other]], correct[:, idx[ref]]
        n10, n01, _, p = raig.mcnemar_exact(a, b)
        d, lo, hi = raig.paired_bootstrap_ci(a, b)
        rows.append({"reference": ref, "comparator": other,
                     "acc_reference": float(b.mean()), "acc_comparator": float(a.mean()),
                     "delta": float(d), "ci_lo": float(lo), "ci_hi": float(hi),
                     "n10_comparator_only": n10, "n01_reference_only": n01,
                     "n_discordant": n10 + n01, "p_raw": float(p)})
    adj, rej = raig.holm_bonferroni([r["p_raw"] for r in rows])
    for r, a_, rj in zip(rows, adj, rej):
        r["p_holm"] = float(a_)
        r["significant_holm_0.05"] = bool(rj)
    report[cond] = rows
    if cond == PRIMARY:
        print(f"=== {cond} (n={len(correct)} paired trials), reference = {ref} ===")
        print(f"{'comparator':22s} {'acc':>6s} {'ref':>6s} {'delta':>8s} "
              f"{'95% CI':>16s} {'disc':>5s} {'p_raw':>10s} {'p_holm':>10s}")
        for r in sorted(rows, key=lambda x: x["p_raw"]):
            print(f"{r['comparator']:22s} {r['acc_comparator']*100:5.1f}% "
                  f"{r['acc_reference']*100:5.1f}% {r['delta']*100:+7.2f}pp "
                  f"[{r['ci_lo']*100:+5.2f},{r['ci_hi']*100:+5.2f}] "
                  f"{r['n_discordant']:5d} {r['p_raw']:10.2e} {r['p_holm']:10.2e}"
                  + ("  *" if r["significant_holm_0.05"] else ""))

json.dump({"primary_condition": PRIMARY, "n_total": meta["n_total"],
           "correction": "Holm step-down, applied within each condition",
           "results": report}, open(f"{OUT}/stats_ext.json", "w"), indent=2)
print(f"\nwrote {OUT}/stats_ext.json")
