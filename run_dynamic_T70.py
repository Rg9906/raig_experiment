"""
Dynamic reliability + raised, confidence-gated budget: does replacing the
static 3-tier epsilon with a live, population-level streaming estimate
(RAIG-streaming, raig.py's "raig-streaming" select mode) beat the static
RAIG-tiered baseline, and does raising the budget from T=40 to T=70 with real
early stopping (Method.early_stop_threshold) help further?

Single condition (heterogeneous_1.0 -- the condition where true epsilon is
tier-congruent by construction; this is a SCOPE LIMIT, not an oversight:
randomised/adversarial are the conditions that would stress-test tier- vs.
attribute-granularity adaptation hardest, and are left for a follow-up if this
result is promising). R=300 x 3 seeds = 900 paired trials, matching the
established protocol. Checkpoints [10,20,30,40,50,60,70]: the T=40 checkpoint
lets "dynamic at the existing headline budget" be read off directly and
compared against the stored results/main_T40.json RAIG-tiered number (NOT
bitwise comparable -- different rng draws, same estimand, compare via CI
overlap, exactly as run_main_T40.py's own docstring already does for T20 vs
T40).

RESUMABLE AT SEED GRANULARITY (not condition granularity -- this run has only
one condition, so main_T40's per-condition save would give zero crash
protection). Each seed's raw arrays are saved to
results/dynamic_T70_seed<seed>.npz as soon as that seed finishes; a re-run
skips any seed whose file already exists. Kill it and restart it freely.

  py run_dynamic_T70.py
  py run_dynamic_T70.py --fresh   # ignore any stored seed files

Writes results/dynamic_T70_seed<seed>.npz (raw, per seed) and
results/dynamic_T70.json (aggregated across whatever seeds are present).
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

R, T, SEEDS, BATCH = 300, 70, [0, 1, 2], 4
CHECKPOINTS = [10, 20, 30, 40, 50, 60, 70]
CONDITION = "heterogeneous_1.0"
STOP_THRESHOLDS = (0.5, 0.7, 0.9)
EXPLORE_FRAC = 0.20

fresh = "--fresh" in sys.argv

t0_all = time.time()
df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
methods, uniform_eps = methods_ext.build_dynamic(
    cat, early_stop_thresholds=STOP_THRESHOLDS, explore_frac=EXPLORE_FRAC)
names = [m.name for m in methods]
true_eps = raig.tiered_true_eps(cat, 1.0)

print(f"N={cat.N} Q={cat.Q} methods={len(names)} T={T} batch={BATCH} "
      f"uniform_eps={uniform_eps:.6f}", flush=True)
print("methods: " + ", ".join(names), flush=True)

seed_paths = {seed: f"{OUT}/dynamic_T70_seed{seed}.npz" for seed in SEEDS}
if fresh:
    for p in seed_paths.values():
        if os.path.exists(p):
            os.remove(p)

todo = [s for s in SEEDS if not os.path.exists(seed_paths[s])]
print(f"to run: {todo if todo else '(nothing -- all seeds already stored)'}", flush=True)

for seed in todo:
    t_seed = time.time()
    rng = np.random.default_rng(1000 + seed)
    out = raig.run_paired_trials(cat, methods, true_eps, R=R, T=T, rng=rng,
                                  batch_size=BATCH, track_top5=True,
                                  log_questions=True, metrics=True,
                                  checkpoints=CHECKPOINTS)
    ckpt_correct = np.stack([out["checkpoints"][c]["correct"] for c in CHECKPOINTS], axis=0)
    ckpt_top5 = np.stack([out["checkpoints"][c]["top5"] for c in CHECKPOINTS], axis=0)
    ckpt_top10 = np.stack([out["checkpoints"][c]["top10"] for c in CHECKPOINTS], axis=0)
    ckpt_rank = np.stack([out["checkpoints"][c]["rank"] for c in CHECKPOINTS], axis=0)
    ckpt_nll = np.stack([out["checkpoints"][c]["nll"] for c in CHECKPOINTS], axis=0)
    np.savez_compressed(
        seed_paths[seed],
        correct=out["correct"], top5=out["top5"], stop_turn=out["stop_turn"],
        qlog=out["qlog"], targets=out["targets"],
        ckpt_correct=ckpt_correct, ckpt_top5=ckpt_top5, ckpt_top10=ckpt_top10,
        ckpt_rank=ckpt_rank, ckpt_nll=ckpt_nll,
        checkpoints=np.array(CHECKPOINTS), names=np.array(names),
    )
    print(f"    seed {seed} done ({time.time()-t_seed:.0f}s) -> {seed_paths[seed]}", flush=True)

# -----------------------------------------------------------------------
# Aggregate whatever seeds are on disk (lets you inspect partial progress
# without waiting for all three).
# -----------------------------------------------------------------------
done_seeds = [s for s in SEEDS if os.path.exists(seed_paths[s])]
if not done_seeds:
    print("no seeds stored yet -- nothing to aggregate", flush=True)
    sys.exit(0)

parts = [np.load(seed_paths[s]) for s in done_seeds]
correct = np.concatenate([p["correct"] for p in parts], axis=0)          # (n, M)
top5 = np.concatenate([p["top5"] for p in parts], axis=0)
stop_turn = np.concatenate([p["stop_turn"] for p in parts], axis=0)      # (n, M)
qlog = np.concatenate([p["qlog"] for p in parts], axis=0)                # (n, M, T)
ckpt_correct = np.concatenate([p["ckpt_correct"] for p in parts], axis=1)  # (C, n, M)
ckpt_top5 = np.concatenate([p["ckpt_top5"] for p in parts], axis=1)
ckpt_top10 = np.concatenate([p["ckpt_top10"] for p in parts], axis=1)
ckpt_rank = np.concatenate([p["ckpt_rank"] for p in parts], axis=1)
ckpt_nll = np.concatenate([p["ckpt_nll"] for p in parts], axis=1)
n_total = correct.shape[0]

col_mean = cat.X.mean(axis=0)
per_method = {}
for mi, name in enumerate(names):
    mean, lo, hi = raig.acc_ci(correct[:, mi])
    q = qlog[:, mi, :]
    tiers = cat.tier_of_q[q].ravel()
    mix = {t: float((tiers == t).mean()) for t in ("objective", "semi", "subjective")}
    st = stop_turn[:, mi]
    entry = {
        "acc": float(mean), "lo": float(lo), "hi": float(hi),
        "top5_acc": float(top5[:, mi].mean()),
        "mean_eps_asked": float(true_eps[q].mean()),
        "mean_ig_asked": float(raig.Hb(col_mean[q.ravel()]).mean()),
        "tier_mix": mix,
        "distinct_attrs_asked": int(len(np.unique(cat.attr_of_q[q]))),
        "mean_questions_asked": float(st.mean()),
        "median_questions_asked": float(np.median(st)),
        "frac_stopped_early": float((st < T).mean()),
    }
    for ci, c in enumerate(CHECKPOINTS):
        entry[f"T{c}"] = {
            "acc": float(ckpt_correct[ci, :, mi].mean()),
            "top5": float(ckpt_top5[ci, :, mi].mean()),
            "top10": float(ckpt_top10[ci, :, mi].mean()),
            "median_rank": float(np.median(ckpt_rank[ci, :, mi])),
            "mrr": float(np.mean(1.0 / ckpt_rank[ci, :, mi])),
            "nll_bits": float(np.mean(ckpt_nll[ci, :, mi])),
        }
    per_method[name] = entry

# -----------------------------------------------------------------------
# Paired statistical comparisons within this run (same targets/noise per
# seed, so exact McNemar is valid), Holm-corrected across the family.
# -----------------------------------------------------------------------
idx = {n: i for i, n in enumerate(names)}
comparisons = [("RAIG-streaming", "RAIG-tiered", "dynamic vs static tiers, both T=70"),
               ("RAIG-streaming", "IG+soft-uniform", "dynamic vs classical, both T=70"),
               ("RAIG-tiered", "IG+soft-uniform", "static tiers vs classical, both T=70 (raise-the-cap-alone check)")]
for thresh in STOP_THRESHOLDS:
    comparisons.append((f"RAIG-streaming+stop@{thresh}", "RAIG-streaming",
                         f"cost of early stopping @ {thresh} vs. no-stop dynamic"))
    comparisons.append((f"RAIG-streaming+stop@{thresh}", "RAIG-tiered",
                         f"combined proposal @ {thresh} vs. static baseline"))

stats_out = []
pvals = []
for a, b, label in comparisons:
    n10, n01, delta, p = raig.mcnemar_exact(correct[:, idx[a]], correct[:, idx[b]])
    stats_out.append({"a": a, "b": b, "label": label,
                       "delta_pp": float(delta * 100), "n10": n10, "n01": n01, "p_raw": p})
    pvals.append(p)
adj, rejected = raig.holm_bonferroni(pvals, alpha=0.05)
for s, pa, r in zip(stats_out, adj, rejected):
    s["p_holm"] = float(pa)
    s["significant_at_05"] = bool(r)

result = {
    "method_names": names, "R": R, "T": T, "seeds": done_seeds,
    "batch_size": BATCH, "checkpoints": CHECKPOINTS, "condition": CONDITION,
    "explore_frac": EXPLORE_FRAC, "stop_thresholds": list(STOP_THRESHOLDS),
    "uniform_eps": uniform_eps, "n_total": n_total,
    "per_method": per_method, "paired_comparisons": stats_out,
}
path = f"{OUT}/dynamic_T70.json"
json.dump(result, open(path, "w"), indent=2)

print(f"\n[T=70, n={n_total}] " +
      ", ".join(f"{n}={per_method[n]['acc']*100:.1f}" for n in names), flush=True)
print(f"[T=40 checkpoint within this run] " +
      ", ".join(f"{n}={per_method[n]['T40']['acc']*100:.1f}" for n in names), flush=True)
print("\npaired comparisons (Holm-corrected):", flush=True)
for s in stats_out:
    print(f"  {s['label']:55s} Delta={s['delta_pp']:+.2f}pp  "
          f"p_holm={s['p_holm']:.2e}  {'*' if s['significant_at_05'] else ' '}", flush=True)

print(f"\nDONE in {time.time()-t0_all:.0f}s -> {path} ({len(done_seeds)}/{len(SEEDS)} seeds)", flush=True)
