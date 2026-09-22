"""
Granularity and component ablation at the deployment budget T=40
(paper Sec. VII-F).

Same design as run_ablation_ext.py -- how many reliability tiers does the
curator actually have to supply, and does the split term of Eq. (6) earn its
place -- re-run at the headline budget so the ablation is comparable with the
main table rather than with the stress test.

Writes results/ablation_T40.json.
"""
import json

import numpy as np

import raig

OUT = "results"
R, T, SEEDS, BATCH = 300, 40, [0, 1, 2], 30      # identical to run_main_T40.py
COND = "heterogeneous_1.0"

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
true_eps = raig.tiered_true_eps(cat, 1.0)

EPS2 = {"reliable": (0.03 + 0.15) / 2, "subjective": 0.30}
k2_vec = cat.eps_vector({a: EPS2["reliable" if t in ("objective", "semi") else "subjective"]
                         for a, t in raig.ATTR_TIER.items()})
tiered_vec = cat.eps_vector({a: raig.TIER_EPS[t] for a, t in raig.ATTR_TIER.items()})

methods = [
    raig.Method("K2", "raig", lambda te: k2_vec),
    raig.Method("CapacityOnly", "capacity", lambda te: tiered_vec),
]
for m in methods:
    m.allowed = cat.allowed_mask(m.allowed_tiers)

allc = []
for s in SEEDS:
    out = raig.run_paired_trials(cat, methods, true_eps, R=R, T=T,
                                 rng=np.random.default_rng(1000 + s),
                                 batch_size=BATCH)
    allc.append(out["correct"])
correct = np.concatenate(allc, axis=0)

main = json.load(open(f"{OUT}/main_T40.json"))["results"][COND]
acc_uniform = main["IG+soft-uniform"]["acc"]
acc_tiered = main["RAIG-tiered"]["acc"]
acc_oracle = main["RAIG-oracle"]["acc"]
gap = acc_oracle - acc_uniform


def pct(a):
    return 100.0 * (a - acc_uniform) / gap if gap else float("nan")


acc_k2 = float(correct[:, 0].mean())
acc_cap = float(correct[:, 1].mean())
lo_k2, hi_k2 = raig.acc_ci(correct[:, 0])[1:]
lo_cap, hi_cap = raig.acc_ci(correct[:, 1])[1:]

rows = [
    {"variant": "K=1 (uniform)", "source": "main_T40", "acc": acc_uniform,
     "lo": main["IG+soft-uniform"]["lo"], "hi": main["IG+soft-uniform"]["hi"],
     "pct_gap": 0.0},
    {"variant": "K=2", "source": "this run", "acc": acc_k2,
     "lo": float(lo_k2), "hi": float(hi_k2), "pct_gap": pct(acc_k2)},
    {"variant": "K=3 (tiered)", "source": "main_T40", "acc": acc_tiered,
     "lo": main["RAIG-tiered"]["lo"], "hi": main["RAIG-tiered"]["hi"],
     "pct_gap": pct(acc_tiered)},
    {"variant": "K=M (oracle)", "source": "main_T40", "acc": acc_oracle,
     "lo": main["RAIG-oracle"]["lo"], "hi": main["RAIG-oracle"]["hi"],
     "pct_gap": 100.0},
    {"variant": "Capacity term only", "source": "this run", "acc": acc_cap,
     "lo": float(lo_cap), "hi": float(hi_cap), "pct_gap": pct(acc_cap)},
]
for r in rows:
    print(f"{r['variant']:22s} acc={r['acc']*100:5.2f}%  "
          f"[{r['lo']*100:4.2f},{r['hi']*100:5.2f}]  "
          f"pct_of_oracle_gap={r['pct_gap']:7.1f}%   ({r['source']})")

json.dump({"condition": COND, "R": R, "T": T, "seeds": SEEDS,
           "batch_size": BATCH, "n_total": R * len(SEEDS),
           "eps2": EPS2, "rows": rows},
          open(f"{OUT}/ablation_T40.json", "w"), indent=2)
print(f"\nwrote {OUT}/ablation_T40.json")
