"""
Clean-condition query-budget sweep for Section VII-F.

The draft asserted 44.1% / 90.2% / 94.9% at T=20/40/60 with no saved artefact,
and its T=20 figure disagreed with Table tab:main (41.9%) for what should be
the identical condition. This reruns the sweep under exactly the main
protocol -- R=300 x seeds {0,1,2} (rng seed 1000+s), clean noise, RAIG-tiered
-- so T=20 is directly comparable to the main table, and saves the result.

Writes results/clean_budget_sweep.json.
"""
import json
import time
import numpy as np
import raig

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
print("N=%d Q=%d" % (cat.N, cat.Q), flush=True)

uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
methods = raig.make_methods(cat, uniform_eps)
raig_tiered = [m for m in methods if m.name == "RAIG-tiered"]
assert len(raig_tiered) == 1

true_eps_clean = raig.tiered_true_eps(cat, 0.0)

R, SEEDS, BATCH = 300, [0, 1, 2], 50
out = {"R_per_seed": R, "seeds": SEEDS, "condition": "clean",
       "method": "RAIG-tiered", "n_total": R * len(SEEDS), "by_T": {}}

t0 = time.time()
for T in [20, 40, 60]:
    allc = []
    for s in SEEDS:
        rng = np.random.default_rng(1000 + s)
        r = raig.run_paired_trials(cat, raig_tiered, true_eps_clean,
                                   R=R, T=T, rng=rng, batch_size=BATCH,
                                   track_top5=True)
        allc.append(r["correct"])
    correct = np.concatenate(allc, axis=0)[:, 0]
    mean, lo, hi = raig.acc_ci(correct)
    out["by_T"][str(T)] = {"acc": float(mean), "lo": float(lo), "hi": float(hi)}
    print("[%7.1fs] T=%d  acc=%.4f (%.4f-%.4f)" % (time.time() - t0, T, mean, lo, hi), flush=True)

with open("results/clean_budget_sweep.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n--- comparison ---", flush=True)
main = json.load(open("results/main_results.json"))
ref = main["results"]["clean"]["RAIG-tiered"]["acc"]
print("main_results.json clean RAIG-tiered (T=20): %.4f" % ref)
print("this sweep          clean RAIG-tiered (T=20): %.4f" % out["by_T"]["20"]["acc"])
print("agreement:", "YES" if abs(ref - out["by_T"]["20"]["acc"]) < 1e-9 else "NO")
print("\npaper claimed 44.1 / 90.2 / 94.9 at T=20/40/60")
print("measured      %.1f / %.1f / %.1f" % tuple(
    out["by_T"][str(t)]["acc"] * 100 for t in [20, 40, 60]))
print("wrote results/clean_budget_sweep.json")
