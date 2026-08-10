import numpy as np
import raig

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
methods = raig.make_methods(cat, uniform_eps)

# CLEAN condition: zero noise everywhere
true_eps_clean = raig.tiered_true_eps(cat, 0.0)
rng = np.random.default_rng(1)
out = raig.run_paired_trials(cat, methods, true_eps_clean, R=20, T=20, rng=rng)
print("CLEAN accuracy per method:", out["correct"].mean(axis=0))
for m, meth in enumerate(methods):
    print(f"  {meth.name}: {out['correct'][:, m].sum()}/20")
