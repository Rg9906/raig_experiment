"""
Component and granularity ablation, re-run under the journal protocol
(paper Sec. VII-D).

The conference-version ablation (run_ablation.py) is not comparable to the
journal main table: it used batch_size=150 and seeds 2000+s, while
run_main_ext.py uses batch_size=30 and seeds 1000+s. Batch size changes how the
shared noise stream is drawn -- rng.random((T, Rb)) per batch -- so the two
runs face different users. The method set does NOT matter (every draw in
run_paired_trials is made before the method loop and is independent of M),
which is why the K=1, K=3 and K=M rows below can be read from main_ext.json
instead of being re-simulated, but the batch size does. This script fixes that.

Two axes:

  Granularity. How many reliability tiers does the annotator actually have to
  supply? K=1 is no correction at all (Prop. uniform), K=2 merges objective and
  semi-objective into one "reliable" tier at eps = 0.09, K=3 is the paper
  annotation, K=M is per-attribute truth. Reported as the percentage of the
  oracle gap recovered, which is the quantity that says how much annotation
  effort buys.

  Components. RAIG = split term + capacity term. "Capacity only" ranks purely
  by 1 - Hb(eps_hat), ignoring how the question divides the belief, and is
  included to confirm that the gain comes from the combination rather than from
  a bare preference for reliable questions.

Writes results/ablation_ext.json.
"""
import json
import numpy as np

import raig
import methods_ext

OUT = "results"
R, T, SEEDS, BATCH = 300, 20, [0, 1, 2], 30      # identical to run_main_ext.py
COND = "heterogeneous_1.0"

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
true_eps = raig.tiered_true_eps(cat, 1.0)

# K=2: the natural coarsening that still separates "answerable" from
# "perceptual judgement". Chosen here, not inherited from the annotation.
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

main = json.load(open(f"{OUT}/main_ext.json"))["results"][COND]
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
    {"variant": "K=1 (uniform)", "source": "main_ext", "acc": acc_uniform,
     "lo": main["IG+soft-uniform"]["lo"], "hi": main["IG+soft-uniform"]["hi"],
     "pct_gap": 0.0},
    {"variant": "K=2", "source": "this run", "acc": acc_k2,
     "lo": float(lo_k2), "hi": float(hi_k2), "pct_gap": pct(acc_k2)},
    {"variant": "K=3 (tiered)", "source": "main_ext", "acc": acc_tiered,
     "lo": main["RAIG-tiered"]["lo"], "hi": main["RAIG-tiered"]["hi"],
     "pct_gap": pct(acc_tiered)},
    {"variant": "K=M (oracle)", "source": "main_ext", "acc": acc_oracle,
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
          open(f"{OUT}/ablation_ext.json", "w"), indent=2)
print(f"\nwrote {OUT}/ablation_ext.json")
