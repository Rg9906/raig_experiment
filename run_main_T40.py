"""
Primary experiment at a DEPLOYMENT-REALISTIC budget: 9 methods x 7 conditions,
R=300 x 3 seeds = 900 paired trials per cell, T=40, common random numbers.

Why this supersedes run_main_ext.py as the headline run. T=20 sits essentially
ON the capacity-corrected budget bound of Corollary 1 (20.15 questions), which
makes it the hardest budget at which the task is information-theoretically
possible at all. That is a defensible stress test and a poor headline: every
method is pinned against the entropy floor, so the absolute accuracies say
little about deployment and the differences between methods are compressed. A
real system asks until it is confident, and forty questions is about twice the
floor. The budget sweep already reported that the advantage WIDENS with budget,
so moving the headline from T=20 to T=40 moves it towards realism, not towards
a flattering operating point -- T=20 remains reported as the stress test.

Checkpoints at 10/20/30/40 mean this single run yields the whole budget curve
for all nine methods at n=900. The T=20 checkpoint here will NOT reproduce
results/main_ext.json bitwise -- rng.random((T, Rb)) draws a different-shaped
block when T=40, so the simulated users differ -- but it is the same estimand
and the two agree within sampling error.

RESUMABLE, and deliberately so: this machine has shown order-of-magnitude
slowdowns under contention, and a three-hour monolithic run is fragile.
Conditions are processed in priority order (the pre-registered primary
condition first), each is written to the JSON as soon as it finishes, and a
re-run skips whatever is already stored. Kill it and restart it freely.

  py run_main_T40.py                       # all conditions, priority order
  py run_main_T40.py heterogeneous_1.0     # just these
  py run_main_T40.py --fresh               # ignore stored conditions

Writes results/main_T40.json plus results/correctT40_<condition>.npy.
"""
import json
import os
import sys
import time

import numpy as np

import raig
import methods_ext

OUT = "results"
os.makedirs(OUT, exist_ok=True)

R, T, SEEDS, BATCH = 300, 40, [0, 1, 2], 40
CHECKPOINTS = [10, 20, 30, 40]

# Priority order: the pre-registered primary condition first, so that a run cut
# short still yields the headline number; then the two that bound it (clean and
# homogeneous), then the rest.
PRIORITY = ["heterogeneous_1.0", "clean", "homogeneous", "heterogeneous_0.5",
            "heterogeneous_1.5", "randomised", "adversarial"]

args = [a for a in sys.argv[1:] if not a.startswith("--")]
fresh = "--fresh" in sys.argv

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

wanted = args if args else PRIORITY
wanted = [c for c in wanted if c in conds]

results = {}
path = f"{OUT}/main_T40.json"
if os.path.exists(path) and not fresh:
    results = json.load(open(path)).get("results", {})
    if results:
        print(f"resuming: {len(results)} condition(s) already stored "
              f"({', '.join(results)})", flush=True)
todo = [c for c in wanted if c not in results]

print(f"N={cat.N} Q={cat.Q} methods={len(names)} T={T} batch={BATCH} "
      f"uniform_eps={uniform_eps:.6f}", flush=True)
print(f"to run: {', '.join(todo) if todo else '(nothing)'}", flush=True)

t_start = time.time()

for cond_name in todo:
    t_cond = time.time()
    true_eps = conds[cond_name]()
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
        print(f"    {cond_name} seed {seed} done "
              f"({time.time()-t_cond:.0f}s into condition)", flush=True)

    correct = np.concatenate(parts["correct"], axis=0)
    top5 = np.concatenate(parts["top5"], axis=0)
    qlog = np.concatenate(parts["qlog"], axis=0)
    np.save(f"{OUT}/correctT40_{cond_name}.npy", correct)

    per_method = {}
    for mi, name in enumerate(names):
        mean, lo, hi = raig.acc_ci(correct[:, mi])
        q = qlog[:, mi, :]
        tiers = cat.tier_of_q[q].ravel()
        mix = {t: float((tiers == t).mean()) for t in ("objective", "semi", "subjective")}
        entry = {
            "acc": float(mean), "lo": float(lo), "hi": float(hi),
            "top5_acc": float(top5[:, mi].mean()),
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
              open(path, "w"), indent=2)

    print(f"[{time.time()-t_start:7.1f}s] {cond_name} "
          f"({time.time()-t_cond:.0f}s): " +
          ", ".join(f"{n}={per_method[n]['acc']*100:.1f}" for n in names), flush=True)

print(f"DONE in {time.time()-t_start:.0f}s -> {path} "
      f"({len(results)}/{len(conds)} conditions stored)", flush=True)
