"""
Table ablation (tier granularity K and the capacity-only component ablation),
all at heterogeneous lambda=1.0, matching the main protocol (R=300 x 3 seeds,
T=20).

K=1 (uniform) and K=3 (tiered) and K=M (oracle) are literally the
IG+soft-uniform / RAIG-tiered / RAIG-oracle columns of Table main (uniform
hat-eps gives RAIG the same ranking as classical IG, Sec. III-F), so they are
read from results/main_results.json rather than re-simulated. K=2 and the
capacity-only variant are new methods run here.

K=2 partition (not specified by the paper, chosen here and documented): merge
objective and semi-objective into a single "reliable" tier at eps = mean(0.03,
0.15) = 0.09, keep subjective at 0.30. This is the natural coarsening that
still distinguishes "answerable" from "perceptual judgement" attributes.
"""
import json
import numpy as np
import raig

OUT = "results"
df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))

R, T, SEEDS, BATCH = 300, 20, [0, 1, 2], 150
true_eps = raig.tiered_true_eps(cat, 1.0)

TIER2 = {}
for a, t in raig.ATTR_TIER.items():
    TIER2[a] = "reliable" if t in ("objective", "semi") else "subjective"
EPS2 = {"reliable": (0.03 + 0.15) / 2, "subjective": 0.30}
k2_eps = cat.eps_vector({a: EPS2[TIER2[a]] for a in raig.FEATURES})

tiered_eps = cat.eps_vector({a: raig.TIER_EPS[t] for a, t in raig.ATTR_TIER.items()})

methods = [
    raig.Method("K2", "raig", lambda true_eps: k2_eps),
    raig.Method("CapacityOnly", "capacity", lambda true_eps: tiered_eps),
]
for m in methods:
    m.allowed = cat.allowed_mask(m.allowed_tiers)

all_correct = []
for seed in SEEDS:
    rng = np.random.default_rng(2000 + seed)
    out = raig.run_paired_trials(cat, methods, true_eps, R=R, T=T, rng=rng, batch_size=BATCH)
    all_correct.append(out["correct"])
correct = np.concatenate(all_correct, axis=0)

with open(f"{OUT}/main_results.json") as f:
    main = json.load(f)["results"]["heterogeneous_1.0"]

acc_uniform = main["IG+soft-uniform"]["acc"]
acc_tiered = main["RAIG-tiered"]["acc"]
acc_oracle = main["RAIG-oracle"]["acc"]
oracle_gap = acc_oracle - acc_uniform

acc_k2 = float(correct[:, 0].mean())
acc_cap = float(correct[:, 1].mean())

def pct_gap(acc):
    return 100.0 * (acc - acc_uniform) / oracle_gap if oracle_gap != 0 else float("nan")

rows = [
    {"variant": "K=1 (uniform)", "acc": acc_uniform, "pct_gap": 0.0},
    {"variant": "K=2", "acc": acc_k2, "pct_gap": pct_gap(acc_k2)},
    {"variant": "K=3 (tiered)", "acc": acc_tiered, "pct_gap": pct_gap(acc_tiered)},
    {"variant": "K=M (oracle)", "acc": acc_oracle, "pct_gap": 100.0},
    {"variant": "Capacity term only", "acc": acc_cap, "pct_gap": pct_gap(acc_cap)},
]
for r in rows:
    print(f"{r['variant']:22s} acc={r['acc']*100:5.1f}%  pct_of_oracle_gap={r['pct_gap']:6.1f}%")

with open(f"{OUT}/ablation.json", "w") as f:
    json.dump(rows, f, indent=2)
print("saved results/ablation.json")
