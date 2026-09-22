"""
Accuracy as a function of query budget (paper Sec. VII-D).

The conference version reported a single number at T=20 -- 7.9% top-1 -- which
invites the reading that the method is weak in absolute terms. T=20 is not an
arbitrary choice: it sits essentially ON the catalogue entropy floor
(log2(77,069) = 16.2 bits, and the capacity-corrected budget bound of
Corollary 1 is 20.15 questions), so it is the hardest budget at which the task
is information-theoretically possible at all. Reporting only that point
conflates "the method is weak" with "the budget is minimal".

This runs ONE T=60 pass per condition and snapshots every checkpoint along the
way, which is both cheaper and cleaner than separate runs per budget because
every curve then shares the same targets and noise stream.

Also reports the metrics that are not floor-bound at T=20 -- top-5, top-10,
median rank and MRR -- and the budget each method needs to reach a fixed
accuracy, which is the practitioner-facing number.

Writes results/budget_sweep.json.
"""
import json
import time
import numpy as np

import raig
import methods_ext

OUT = "results"
R, T, SEEDS, BATCH = 150, 60, [0, 1, 2], 25
CHECKPOINTS = [5, 10, 15, 20, 30, 40, 50, 60]
KEEP = ("IG+hard", "IG+soft-uniform", "Learned-noiseaware",
        "RAIG-empirical", "RAIG-tiered", "RAIG-oracle")

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
methods, uniform_eps = methods_ext.build(cat, include=KEEP)
names = [m.name for m in methods]
print(f"N={cat.N} Q={cat.Q} methods={names}", flush=True)

conds = {"clean": raig.tiered_true_eps(cat, 0.0),
         "heterogeneous_1.0": raig.tiered_true_eps(cat, 1.0)}

res = {}
t0 = time.time()
for cond, true_eps in conds.items():
    acc = {c: {k: [] for k in ("correct", "rank", "top5", "top10", "nll")}
           for c in CHECKPOINTS}
    for s in SEEDS:
        out = raig.run_paired_trials(cat, methods, true_eps, R=R, T=T,
                                     rng=np.random.default_rng(1000 + s),
                                     batch_size=BATCH, metrics=True,
                                     checkpoints=CHECKPOINTS)
        for c in CHECKPOINTS:
            for k in acc[c]:
                acc[c][k].append(out["checkpoints"][c][k])
    cell = {n: {} for n in names}
    for c in CHECKPOINTS:
        arrs = {k: np.concatenate(v, axis=0) for k, v in acc[c].items()}
        for mi, n in enumerate(names):
            mean, lo, hi = raig.acc_ci(arrs["correct"][:, mi])
            cell[n][f"T{c}"] = {
                "acc": float(mean), "lo": float(lo), "hi": float(hi),
                "top5": float(arrs["top5"][:, mi].mean()),
                "top10": float(arrs["top10"][:, mi].mean()),
                "median_rank": float(np.median(arrs["rank"][:, mi])),
                "mrr": float(np.mean(1.0 / arrs["rank"][:, mi])),
                "nll_bits": float(np.mean(arrs["nll"][:, mi])),
            }
    # Budget needed to reach a fixed accuracy, by linear interpolation between
    # the bracketing checkpoints. Reported as the practitioner-facing figure.
    for n in names:
        xs = np.array(CHECKPOINTS, dtype=float)
        ys = np.array([cell[n][f"T{c}"]["acc"] for c in CHECKPOINTS])
        reach = {}
        for target in (0.10, 0.25, 0.50, 0.90):
            hit = np.flatnonzero(ys >= target)
            if len(hit) == 0:
                reach[str(target)] = None
            else:
                i = hit[0]
                if i == 0:
                    reach[str(target)] = float(xs[0])
                else:
                    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
                    reach[str(target)] = float(x0 + (target - y0) * (x1 - x0) / max(y1 - y0, 1e-12))
        cell[n]["budget_to_reach"] = reach
    res[cond] = cell
    json.dump({"R": R, "T": T, "seeds": SEEDS, "n_total": R * len(SEEDS),
               "checkpoints": CHECKPOINTS, "results": res},
              open(f"{OUT}/budget_sweep.json", "w"), indent=2)
    print(f"[{time.time()-t0:6.0f}s] {cond}:", flush=True)
    for n in names:
        row = " ".join(f"T{c}={cell[n][f'T{c}']['acc']*100:.1f}" for c in CHECKPOINTS)
        print(f"    {n:20s} {row}", flush=True)

print(f"\nwrote {OUT}/budget_sweep.json ({time.time()-t0:.0f}s)")
