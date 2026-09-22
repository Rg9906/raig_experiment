"""
Push top-1 accuracy past 70% by trading catalogue size for budget: a fresh
random N=5,000-track subset per seed (same subsetting design as
run_catalogue_size.py -- independent uniform draw without replacement,
recompiled from scratch so split masses/degenerate questions/identifiability
are all recomputed for that subset), a budget up to T_MAX=200 with
checkpoints along the way, heterogeneous_1.0 (the paper's primary noise
condition), R=300 x 3 seeds = 900 paired trials.

Methods: IG+soft-uniform (classical reference), RAIG-tiered (the proven
static-tier winner), RAIG-oracle (upper bound, true eps), RAIG-streaming
(the online/dynamic reliability estimate from raig.py's "raig-streaming"
mode, included so the dynamic-vs-static comparison carries over to this
smaller, higher-budget regime too).

Writes results/high_accuracy_N5000.json.
"""
import json
import math
import os
import sys
import time

import numpy as np

import raig
import methods_ext

OUT = "results"
os.makedirs(OUT, exist_ok=True)

SIZE = 5000
T_MAX = 200
CHECKPOINTS = [20, 40, 60, 80, 100, 120, 150, 175, 200]
R, SEEDS, BATCH = 300, [0, 1, 2], 40
EXPLORE_FRAC = 0.20

fresh = "--fresh" in sys.argv

df_full = raig.load_catalogue("data/dataset_final.csv")
N_FULL = len(df_full)
print(f"full catalogue N={N_FULL}, subsetting to {SIZE}", flush=True)

seed_paths = {s: f"{OUT}/high_accuracy_N5000_seed{s}.npz" for s in SEEDS}
if fresh:
    for p in seed_paths.values():
        if os.path.exists(p):
            os.remove(p)
todo = [s for s in SEEDS if not os.path.exists(seed_paths[s])]
print(f"to run: {todo if todo else '(nothing -- all seeds stored)'}", flush=True)

names = None
t0 = time.time()
for seed in todo:
    t_seed = time.time()
    rng_sub = np.random.default_rng(50000 + 97 * seed + SIZE)
    idx = rng_sub.choice(N_FULL, size=SIZE, replace=False)
    df = df_full.iloc[np.sort(idx)].reset_index(drop=True)
    cat = raig.Catalogue(df)

    groups = df.groupby(list(raig.FEATURES), observed=True).size()
    ceiling = float((groups == 1).sum() / len(df))
    floor_bits = math.log2(len(groups))

    uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
    unif = np.full(cat.Q, uniform_eps, dtype=np.float32)
    tiered = cat.eps_vector({a: raig.TIER_EPS[t] for a, t in raig.ATTR_TIER.items()})
    true_eps = raig.tiered_true_eps(cat, 1.0)

    ms = [
        raig.Method("IG+soft-uniform", "ig", lambda te: unif),
        raig.Method("RAIG-tiered", "raig", lambda te: tiered),
        raig.Method("RAIG-oracle", "raig", lambda te: te),
        raig.Method("RAIG-streaming", "raig-streaming",
                    lambda te: np.full(cat.Q, raig.STREAM_EPS_INIT, dtype=np.float32),
                    explore_frac=EXPLORE_FRAC),
    ]
    for m in ms:
        m.allowed = cat.allowed_mask(m.allowed_tiers)
    names = [m.name for m in ms]

    out = raig.run_paired_trials(cat, ms, true_eps, R=R, T=T_MAX,
                                  rng=np.random.default_rng(1000 + seed),
                                  batch_size=BATCH, track_top5=True,
                                  metrics=True, checkpoints=CHECKPOINTS)

    ckpt_correct = np.stack([out["checkpoints"][c]["correct"] for c in CHECKPOINTS], axis=0)
    ckpt_top5 = np.stack([out["checkpoints"][c]["top5"] for c in CHECKPOINTS], axis=0)
    np.savez_compressed(
        seed_paths[seed], correct=out["correct"], top5=out["top5"],
        ckpt_correct=ckpt_correct, ckpt_top5=ckpt_top5,
        checkpoints=np.array(CHECKPOINTS), names=np.array(names),
        floor_bits=floor_bits, ceiling=ceiling, N=cat.N, Q=cat.Q,
    )
    print(f"    seed {seed} done ({time.time()-t_seed:.0f}s) N={cat.N} Q={cat.Q} "
          f"floor={floor_bits:.2f}b ceiling={ceiling*100:.1f}% -> {seed_paths[seed]}", flush=True)

done_seeds = [s for s in SEEDS if os.path.exists(seed_paths[s])]
if not done_seeds:
    print("no seeds stored -- nothing to aggregate", flush=True)
    sys.exit(0)

parts = [np.load(seed_paths[s], allow_pickle=True) for s in done_seeds]
if names is None:
    names = list(parts[0]["names"])
correct = np.concatenate([p["correct"] for p in parts], axis=0)
top5 = np.concatenate([p["top5"] for p in parts], axis=0)
ckpt_correct = np.concatenate([p["ckpt_correct"] for p in parts], axis=1)
ckpt_top5 = np.concatenate([p["ckpt_top5"] for p in parts], axis=1)
n_total = correct.shape[0]
mean_floor = float(np.mean([float(p["floor_bits"]) for p in parts]))
mean_ceiling = float(np.mean([float(p["ceiling"]) for p in parts]))

per_method = {}
for mi, name in enumerate(names):
    entry = {"acc": float(correct[:, mi].mean()), "top5_acc": float(top5[:, mi].mean())}
    for ci, c in enumerate(CHECKPOINTS):
        entry[f"T{c}"] = {"acc": float(ckpt_correct[ci, :, mi].mean()),
                           "top5": float(ckpt_top5[ci, :, mi].mean())}
    per_method[name] = entry

# Paired McNemar, RAIG-streaming vs. RAIG-tiered, at each checkpoint -- the
# comparison the "does the online estimator catch up at smaller scale"
# question in sec_robustness.tex actually rests on.
streaming_vs_tiered = {}
if "RAIG-streaming" in names and "RAIG-tiered" in names:
    si, ti = names.index("RAIG-streaming"), names.index("RAIG-tiered")
    for ci, c in enumerate(CHECKPOINTS):
        n10, n01, delta, p = raig.mcnemar_exact(ckpt_correct[ci, :, si], ckpt_correct[ci, :, ti])
        streaming_vs_tiered[f"T{c}"] = {"delta_tiered_minus_streaming_pp": float(delta * 100),
                                         "n10": n10, "n01": n01, "p": float(p)}

result = {"size": SIZE, "T_max": T_MAX, "checkpoints": CHECKPOINTS, "R": R,
          "seeds": done_seeds, "n_total": n_total, "mean_floor_bits": mean_floor,
          "mean_ceiling": mean_ceiling, "per_method": per_method,
          "streaming_vs_tiered": streaming_vs_tiered}
json.dump(result, open(f"{OUT}/high_accuracy_N5000.json", "w"), indent=2)

print(f"\n[N={SIZE}, floor={mean_floor:.2f}b, ceiling={mean_ceiling*100:.1f}%, n={n_total}]", flush=True)
print("accuracy by checkpoint:")
header = "method".ljust(18) + "".join(f"T{c}".rjust(7) for c in CHECKPOINTS)
print(header, flush=True)
for name in names:
    row = name.ljust(18) + "".join(f"{per_method[name][f'T{c}']['acc']*100:6.1f}%" for c in CHECKPOINTS)
    print(row, flush=True)

print(f"\nDONE in {time.time()-t0:.0f}s -> results/high_accuracy_N5000.json "
      f"({len(done_seeds)}/{len(SEEDS)} seeds)", flush=True)
