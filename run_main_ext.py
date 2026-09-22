"""
Primary experiment for the journal version: 9 methods x 7 noise conditions,
R=300 x 3 seeds = 900 paired trials per cell, T=20, common random numbers.

Supersedes run_main.py. Two things changed and both matter:

  * The harness is now pinned to a single BLAS thread, which makes the run
    bitwise reproducible. The stored numbers from the multi-threaded run will
    NOT reproduce exactly; verify_harness.py checks the two agree to within
    sampling error, which is the strongest claim available.
  * Selected questions are logged, so the asked-question reliability table --
    the most direct behavioural evidence of the bias, promised by the
    conference version and never reported -- comes out of the same pass.

Writes results/main_ext.json plus results/correct_<condition>.npy, and
checkpoints after every condition so a partial run is still usable.
"""
import json
import os
import time
import numpy as np

import raig
import methods_ext

OUT = "results"
os.makedirs(OUT, exist_ok=True)

R, T, SEEDS, BATCH = 300, 20, [0, 1, 2], 30
CHECKPOINTS = [5, 10, 15, 20]

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
# Mean of each compiled question column, precomputed once. The obvious
# expression for mean_ig_asked below, Hb(cat.X[:, q.ravel()].mean(axis=0)),
# materialises an (N x |asked|) float32 array -- 11.5 GB at T=40, n=900 --
# to compute means this vector already holds. Indexing it instead is the
# same number for O(|asked|) memory.
col_mean = cat.X.mean(axis=0)
methods, uniform_eps = methods_ext.build(cat)
names = [m.name for m in methods]
conds = methods_ext.conditions(cat, uniform_eps)
print(f"N={cat.N} Q={cat.Q} methods={len(names)} uniform_eps={uniform_eps:.6f}", flush=True)

results = {}
t_start = time.time()

for cond_name, eps_fn in conds.items():
    true_eps = eps_fn()
    parts = {"correct": [], "top5": [], "qlog": [],
             "ckpt": {c: {k: [] for k in ("correct", "rank", "top5", "top10", "nll")}
                      for c in CHECKPOINTS}}
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        out = raig.run_paired_trials(cat, methods, true_eps, R=R, T=T, rng=rng,
                                     batch_size=BATCH, track_top5=True,
                                     log_questions=True, metrics=True,
                                     checkpoints=CHECKPOINTS)
        parts["correct"].append(out["correct"])
        parts["top5"].append(out["top5"])
        parts["qlog"].append(out["qlog"])
        for c in CHECKPOINTS:
            for k in parts["ckpt"][c]:
                parts["ckpt"][c][k].append(out["checkpoints"][c][k])

    correct = np.concatenate(parts["correct"], axis=0)          # (900, M)
    top5 = np.concatenate(parts["top5"], axis=0)
    qlog = np.concatenate(parts["qlog"], axis=0)                # (900, M, T)
    np.save(f"{OUT}/correct_{cond_name}.npy", correct)

    per_method = {}
    for mi, name in enumerate(names):
        mean, lo, hi = raig.acc_ci(correct[:, mi])
        q = qlog[:, mi, :]
        tiers = cat.tier_of_q[q].ravel()
        mix = {t: float((tiers == t).mean()) for t in ("objective", "semi", "subjective")}
        entry = {
            "acc": float(mean), "lo": float(lo), "hi": float(hi),
            "top5_acc": float(top5[:, mi].mean()),
            # behavioural evidence: what the selector actually chose to ask
            "mean_eps_asked": float(true_eps[q].mean()),
            "mean_ig_asked": float(raig.Hb(col_mean[q.ravel()]).mean()),
            "tier_mix": mix,
            "distinct_attrs_asked": int(len(np.unique(cat.attr_of_q[q]))),
        }
        for c in CHECKPOINTS:
            arr = {k: np.concatenate(parts["ckpt"][c][k], axis=0)[:, mi]
                   for k in parts["ckpt"][c]}
            entry[f"T{c}"] = {
                "acc": float(arr["correct"].mean()),
                "top5": float(arr["top5"].mean()),
                "top10": float(arr["top10"].mean()),
                "median_rank": float(np.median(arr["rank"])),
                "mrr": float(np.mean(1.0 / arr["rank"])),
                "nll_bits": float(np.mean(arr["nll"])),
            }
        per_method[name] = entry

    results[cond_name] = per_method
    json.dump({"method_names": names, "R": R, "T": T, "seeds": SEEDS,
               "batch_size": BATCH, "checkpoints": CHECKPOINTS,
               "uniform_eps": uniform_eps, "n_total": R * len(SEEDS),
               "results": results},
              open(f"{OUT}/main_ext.json", "w"), indent=2)

    el = time.time() - t_start
    print(f"[{el:7.1f}s] {cond_name}: " +
          ", ".join(f"{n}={per_method[n]['acc']*100:.1f}" for n in names), flush=True)

print(f"DONE in {time.time()-t_start:.0f}s -> {OUT}/main_ext.json", flush=True)
