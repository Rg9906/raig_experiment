"""
Primary experiment: Table tab:main (top-1 accuracy at T=20, 6 methods x 7
noise conditions, R=300 trials x 3 seeds, common random numbers) plus the
data needed for Table tab:mcnemar (paired McNemar at heterogeneous lambda=1).

Saves results/main_results.json (per-condition, per-method, per-seed
correctness arrays are NOT retained in full to keep the file small; instead
we save summary stats + the concatenated correct/top5 boolean matrices
needed for exact paired testing at lambda=1.0).
"""
import json
import time
import numpy as np
import raig

OUT = "results"
import os
os.makedirs(OUT, exist_ok=True)

print("Loading catalogue...", flush=True)
df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
print(f"N={cat.N} Q={cat.Q}", flush=True)

uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
print("uniform_eps =", uniform_eps, flush=True)
methods = raig.make_methods(cat, uniform_eps)
method_names = [m.name for m in methods]

R = 300
T = 20
SEEDS = [0, 1, 2]
BATCH = 150

conditions = {
    "clean": lambda: raig.tiered_true_eps(cat, 0.0),
    "homogeneous": lambda: raig.homogeneous_true_eps(cat, uniform_eps),
    "heterogeneous_0.5": lambda: raig.tiered_true_eps(cat, 0.5),
    "heterogeneous_1.0": lambda: raig.tiered_true_eps(cat, 1.0),
    "heterogeneous_1.5": lambda: raig.tiered_true_eps(cat, 1.5),
    "randomised": lambda: raig.randomised_true_eps(cat, 1.0),
    "adversarial": lambda: raig.adversarial_true_eps(cat, 1.0),
}

results = {}
raw_lambda1 = None  # keep full per-trial correctness for McNemar at heterogeneous 1.0

t_start = time.time()
for cond_name, eps_fn in conditions.items():
    true_eps = eps_fn()
    all_correct = []
    all_top5 = []
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        out = raig.run_paired_trials(cat, methods, true_eps, R=R, T=T, rng=rng,
                                      batch_size=BATCH, track_top5=True)
        all_correct.append(out["correct"])
        all_top5.append(out["top5"])
    correct = np.concatenate(all_correct, axis=0)  # (R*3, M)
    top5 = np.concatenate(all_top5, axis=0)
    acc = {}
    for mi, name in enumerate(method_names):
        mean, lo, hi = raig.acc_ci(correct[:, mi])
        acc[name] = {"acc": float(mean), "lo": float(lo), "hi": float(hi),
                     "top5_acc": float(top5[:, mi].mean())}
    results[cond_name] = acc
    if cond_name == "heterogeneous_1.0":
        raw_lambda1 = correct
        np.save(f"{OUT}/correct_lambda1.npy", correct)
    elapsed = time.time() - t_start
    print(f"[{elapsed:7.1f}s] {cond_name}: " +
          ", ".join(f"{n}={acc[n]['acc']:.3f}" for n in method_names), flush=True)

with open(f"{OUT}/main_results.json", "w") as f:
    json.dump({"method_names": method_names, "R": R, "T": T, "seeds": SEEDS,
               "uniform_eps": uniform_eps, "results": results}, f, indent=2)

print("DONE", time.time() - t_start, "s total", flush=True)
