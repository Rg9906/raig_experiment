"""
Where does reliability-aware selection start to pay? (paper Sec. VII-E)

The conference version evaluated four noise multipliers and reported that
RAIG-tiered trails at lambda <= 0.5 and leads at lambda >= 1.0. That leaves
the crossing point unlocated, which is the number a practitioner actually
needs: it says how noisy your users have to be before the correction is worth
making. This sweeps a finer grid and interpolates the crossing.

Writes results/lambda_sweep.json.
"""
import json
import time
import numpy as np

import raig
import methods_ext

OUT = "results"
LAMBDAS = [0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 2.0]
R, T, SEEDS, BATCH = 150, 20, [0, 1, 2], 30
KEEP = ("IG+soft-uniform", "Learned-noiseaware", "RAIG-tiered", "RAIG-oracle")

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
methods, uniform_eps = methods_ext.build(cat, include=KEEP)
names = [m.name for m in methods]
i_u, i_r = names.index("IG+soft-uniform"), names.index("RAIG-tiered")
print(f"methods={names}", flush=True)

rows = []
t0 = time.time()
for lam in LAMBDAS:
    true_eps = raig.tiered_true_eps(cat, lam)
    allc = []
    for s in SEEDS:
        out = raig.run_paired_trials(cat, methods, true_eps, R=R, T=T,
                                     rng=np.random.default_rng(1000 + s),
                                     batch_size=BATCH)
        allc.append(out["correct"])
    correct = np.concatenate(allc, axis=0)
    n10, n01, _, p = raig.mcnemar_exact(correct[:, i_u], correct[:, i_r])
    row = {"lambda": lam, "n": int(len(correct)),
           "acc": {n: float(correct[:, i].mean()) for i, n in enumerate(names)},
           "delta_raig_minus_ig": float(correct[:, i_r].mean() - correct[:, i_u].mean()),
           "mcnemar_p": float(p), "n10": n10, "n01": n01,
           "mean_true_eps": float(true_eps.mean())}
    rows.append(row)
    print(f"[{time.time()-t0:6.0f}s] lam={lam:<5} " +
          ", ".join(f"{n}={row['acc'][n]*100:.1f}" for n in names) +
          f"  delta={row['delta_raig_minus_ig']*100:+.1f}pp p={p:.2e}", flush=True)
    json.dump({"lambdas": LAMBDAS, "R": R, "T": T, "seeds": SEEDS, "rows": rows},
              open(f"{OUT}/lambda_sweep.json", "w"), indent=2)

# Interpolate the crossing point where the paired difference changes sign.
lams = np.array([r["lambda"] for r in rows])
dels = np.array([r["delta_raig_minus_ig"] for r in rows])
cross = None
for i in range(1, len(lams)):
    if dels[i - 1] < 0 <= dels[i]:
        cross = float(lams[i - 1] + (0 - dels[i - 1]) * (lams[i] - lams[i - 1]) /
                      (dels[i] - dels[i - 1]))
        break
summary = {"crossing_lambda": cross,
           "crossing_mean_true_eps": (None if cross is None
                                      else float(cross * raig.tiered_true_eps(cat, 1.0).mean()))}
json.dump({"lambdas": LAMBDAS, "R": R, "T": T, "seeds": SEEDS,
           "rows": rows, **summary}, open(f"{OUT}/lambda_sweep.json", "w"), indent=2)
print(f"\ncrossing at lambda={cross} (mean true eps="
      f"{summary['crossing_mean_true_eps']})")
print(f"wrote {OUT}/lambda_sweep.json ({time.time()-t0:.0f}s)")
