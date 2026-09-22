"""
Per-seed breakdown for the original seeds {0,1,2} under the randomised and
adversarial controls.

b5_stability_newseeds.json stores per-seed accuracies only for the five NEW
seeds {3..7}; seeds {0,1,2} exist only pooled. The draft's "8 of 8 seeds"
claim therefore had no per-seed evidence for a third of the seeds it counted.
This fills that gap so the claim is checkable.

Writes results/seed_breakdown_orig.json.
"""
import json
import time
import numpy as np
import raig

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
methods = raig.make_methods(cat, uniform_eps)
names = [m.name for m in methods]

conditions = {
    "randomised": raig.randomised_true_eps(cat, 1.0),
    "adversarial": raig.adversarial_true_eps(cat, 1.0),
}

R, T, SEEDS, BATCH = 300, 20, [0, 1, 2], 50
out = {"R": R, "T": T, "seeds": SEEDS, "method_names": names, "results": {}}

t0 = time.time()
for cond, true_eps in conditions.items():
    per_seed = []
    for s in SEEDS:
        rng = np.random.default_rng(1000 + s)
        r = raig.run_paired_trials(cat, methods, true_eps, R=R, T=T, rng=rng,
                                   batch_size=BATCH)
        acc = {n: float(r["correct"][:, i].mean()) for i, n in enumerate(names)}
        per_seed.append(acc)
        print("[%7.1fs] %s seed %d: IG+soft-uniform=%.3f RAIG-tiered=%.3f"
              % (time.time() - t0, cond, s, acc["IG+soft-uniform"], acc["RAIG-tiered"]),
              flush=True)
    out["results"][cond] = {"per_seed": per_seed}

with open("results/seed_breakdown_orig.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n--- combined 8-seed tally (IG+soft-uniform vs RAIG-tiered) ---")
new = json.load(open("results/b5_stability_newseeds.json"))
for cond in ["adversarial", "randomised"]:
    rows = [(s, d) for s, d in zip(SEEDS, out["results"][cond]["per_seed"])]
    rows += [(s, d) for s, d in zip(new["new_seeds"], new["results"][cond]["per_seed"])]
    wins = ties = 0
    igs, rgs = [], []
    for s, d in rows:
        a, b = d["IG+soft-uniform"] * 100, d["RAIG-tiered"] * 100
        igs.append(a); rgs.append(b)
        wins += a > b; ties += a == b
        print("  %-12s seed %d: IG=%.1f%% RAIG=%.1f%%  %s"
              % (cond, s, a, b, "IG" if a > b else ("tie" if a == b else "RAIG")))
    print("  => %s: IG wins %d/%d, ties %d ; IG range %.1f-%.1f, RAIG range %.1f-%.1f\n"
          % (cond, wins, len(rows), ties, min(igs), max(igs), min(rgs), max(rgs)))
print("wrote results/seed_breakdown_orig.json")
