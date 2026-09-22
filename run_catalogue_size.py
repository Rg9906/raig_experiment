"""
Accuracy as a function of CATALOGUE SIZE (paper Sec. VII-G).

Motivation. The primary comparison runs on all 79,803 tracks at a budget close
to the capacity-corrected bound, which is the hardest configuration the task
admits and produces single-digit top-1 accuracy. That is an honest stress test
and a misleading advertisement: a reader may conclude the method is weak when
what is weak is the operating point. Catalogue size is a parameter of the task,
not of the method, and many deployed interactive identifiers work over a few
thousand items rather than eighty thousand -- a product finder's SKU list, a
fault taxonomy, a game's character set. This sweep reports what the same
methods do across that range.

WHAT THIS IS NOT. It is not cross-validation, and it must not be described as
such. There is no model being fitted and no train/test split: the catalogue is
the hypothesis space, so drawing a subset does not estimate generalisation, it
shrinks the search problem. The entropy floor falls as log2 N, so accuracy
rises for a purely mechanical reason. Reporting a subset accuracy as if it were
the full-catalogue accuracy would misstate the task's difficulty. What the
sweep legitimately shows is (a) how the difficulty scales, and (b) whether the
RAIG-over-IG advantage survives at every scale, which is the claim the paper
actually makes.

Design. For each seed we draw an independent uniform subset of the catalogue
WITHOUT replacement and compile it from scratch, so split masses, degenerate
questions and identifiability are all recomputed for that subset; the target is
drawn from the subset. Averaging over seeds therefore averages over the draw as
well as over targets, and the spread across seeds is reported. Identifiability
is audited per subset, because a smaller catalogue has fewer colliding attribute
vectors and hence a higher attainable ceiling -- a confound that must be
reported alongside the accuracies rather than left implicit.

Writes results/catalogue_size.json.
"""
import json
import math
import os
import sys
import time

import numpy as np

import raig

OUT = "results"
SIZES = [1000, 2500, 5000, 10000, 20000, 40000, 79803]
BUDGETS = [20, 40]
R, SEEDS, BATCH = 200, [0, 1, 2], 40

# Resumable and subsettable, for the same reason run_main_T40.py is: the large
# sizes are expensive and this machine has shown order-of-magnitude slowdowns
# under contention. Cells already stored are skipped unless --fresh is given.
#   py run_catalogue_size.py                 # every size
#   py run_catalogue_size.py 1000 5000       # just these
#   py run_catalogue_size.py --fresh
#   py run_catalogue_size.py T60             # just this budget, every size
#   py run_catalogue_size.py 20000 T20 T40    # these sizes at these budgets
_tok = [a for a in sys.argv[1:] if not a.startswith("--")]
_sizes = [int(a) for a in _tok if not a.upper().startswith("T")]
_budgets = [int(a[1:]) for a in _tok if a.upper().startswith("T")]
if _sizes:
    SIZES = [s for s in SIZES if s in _sizes] or _sizes
if _budgets:
    BUDGETS = _budgets
FRESH = "--fresh" in sys.argv
KEEP = ("Random+soft", "IG+hard", "IG+soft-uniform", "IG+soft-objective",
        "RAIG-tiered", "RAIG-oracle")

df_full = raig.load_catalogue("data/dataset_final.csv")
N_FULL = len(df_full)
uniform_eps_full = None

print(f"full catalogue N={N_FULL}", flush=True)

rows = []
path = f"{OUT}/catalogue_size.json"
if os.path.exists(path) and not FRESH:
    rows = json.load(open(path)).get("rows", [])
    if rows:
        print(f"resuming: {len(rows)} cell(s) already stored", flush=True)
done = {(r["size"], r["T"]) for r in rows}
t0 = time.time()

for size in SIZES:
    for T in BUDGETS:
        if (size, T) in done:
            continue
        per_seed = []
        for s in SEEDS:
            rng_sub = np.random.default_rng(50000 + 97 * s + size)
            if size >= N_FULL:
                df = df_full
            else:
                idx = rng_sub.choice(N_FULL, size=size, replace=False)
                df = df_full.iloc[np.sort(idx)].reset_index(drop=True)
            cat = raig.Catalogue(df)

            # identifiability of THIS subset: the attainable ceiling
            groups = df.groupby(list(raig.FEATURES), observed=True).size()
            ceiling = float((groups == 1).sum() / len(df))
            floor_bits = math.log2(len(groups))

            uniform_eps = float(np.average(
                [raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
            zero = np.zeros(cat.Q, dtype=np.float32)
            unif = np.full(cat.Q, uniform_eps, dtype=np.float32)
            tiered = cat.eps_vector({a: raig.TIER_EPS[t]
                                     for a, t in raig.ATTR_TIER.items()})
            true_eps = raig.tiered_true_eps(cat, 1.0)

            ms = [
                raig.Method("Random+soft", "random", lambda te: unif),
                raig.Method("IG+hard", "ig", lambda te: zero),
                raig.Method("IG+soft-uniform", "ig", lambda te: unif),
                raig.Method("IG+soft-objective", "ig", lambda te: unif,
                            allowed_tiers=("objective",)),
                raig.Method("RAIG-tiered", "raig", lambda te: tiered),
                raig.Method("RAIG-oracle", "raig", lambda te: te),
            ]
            ms = [m for m in ms if m.name in KEEP]
            for m in ms:
                m.allowed = cat.allowed_mask(m.allowed_tiers)
            names = [m.name for m in ms]

            out = raig.run_paired_trials(cat, ms, true_eps, R=R, T=T,
                                         rng=np.random.default_rng(1000 + s),
                                         batch_size=BATCH, track_top5=True)
            c, t5 = out["correct"], out["top5"]
            per_seed.append({
                "seed": s, "N": int(cat.N), "Q": int(cat.Q),
                "distinct_vectors": int(len(groups)),
                "floor_bits": floor_bits, "ceiling": ceiling,
                "uniform_eps": uniform_eps,
                "acc": {n: float(c[:, i].mean()) for i, n in enumerate(names)},
                "top5": {n: float(t5[:, i].mean()) for i, n in enumerate(names)},
                "correct": {n: c[:, i].astype(int).tolist() for i, n in enumerate(names)},
            })

        acc = {n: float(np.mean([p["acc"][n] for p in per_seed])) for n in names}
        sd = {n: float(np.std([p["acc"][n] for p in per_seed], ddof=1)) for n in names}
        top5 = {n: float(np.mean([p["top5"][n] for p in per_seed])) for n in names}
        ca = np.concatenate([np.array(p["correct"]["RAIG-tiered"], dtype=bool)
                             for p in per_seed])
        cb = np.concatenate([np.array(p["correct"]["IG+soft-uniform"], dtype=bool)
                             for p in per_seed])
        n10, n01, _, p_mc = raig.mcnemar_exact(cb, ca)

        row = {
            "size": size, "T": T,
            "mean_floor_bits": float(np.mean([p["floor_bits"] for p in per_seed])),
            "mean_ceiling": float(np.mean([p["ceiling"] for p in per_seed])),
            "mean_Q": float(np.mean([p["Q"] for p in per_seed])),
            "acc": acc, "acc_sd": sd, "top5": top5,
            "delta_raig_minus_ig": acc["RAIG-tiered"] - acc["IG+soft-uniform"],
            "mcnemar": {"n10_ig_only": n10, "n01_raig_only": n01, "p": float(p_mc)},
            "n_total": R * len(SEEDS),
        }
        rows.append(row)
        print(f"[{time.time()-t0:6.0f}s] N={size:>5d} T={T}: "
              f"floor={row['mean_floor_bits']:5.2f}b ceiling={row['mean_ceiling']*100:5.1f}%  "
              + ", ".join(f"{n}={acc[n]*100:5.1f}" for n in
                          ("IG+soft-uniform", "RAIG-tiered", "RAIG-oracle"))
              + f"  delta={row['delta_raig_minus_ig']*100:+5.1f}pp p={p_mc:.2e}",
              flush=True)
        json.dump({"sizes": sorted({r["size"] for r in rows}), "budgets": sorted({r["T"] for r in rows}), "R": R, "seeds": SEEDS,
                   "n_total_per_cell": R * len(SEEDS), "rows": rows},
                  open(f"{OUT}/catalogue_size.json", "w"), indent=2)

print(f"\nwrote {OUT}/catalogue_size.json ({time.time()-t0:.0f}s)")
