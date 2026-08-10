import time
import numpy as np
import raig

t0 = time.time()
df = raig.load_catalogue("data/dataset_final.csv", raw_csv="data/dataset_raw.csv" if False else None)
print("loaded", df.shape, "in", time.time() - t0, "s")

cat = raig.Catalogue(df)
print("N =", cat.N, "Q =", cat.Q)

audit = raig.identifiability_audit(df, raig.FEATURES)
print("audit:", audit)

tab, rho, p = raig.measure_pathology(cat)
print(tab.sort_values("max_ig", ascending=False).head(10))
print("spearman rho:", rho, "p:", p)

uniform_eps = float(np.average(
    [raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]
))
print("uniform_eps (question-weighted mean):", uniform_eps)

methods = raig.make_methods(cat, uniform_eps)
true_eps = raig.tiered_true_eps(cat, 1.0)

rng = np.random.default_rng(42)
t0 = time.time()
out = raig.run_paired_trials(cat, methods, true_eps, R=10, T=20, rng=rng, time_selection=True)
dt = time.time() - t0
print("10 trials took", dt, "s ->", dt / 10, "s/trial")
print("per-turn selection ms:", out["ms_per_selection_call"])
print("accuracy per method:", out["correct"].mean(axis=0))
