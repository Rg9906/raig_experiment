import numpy as np
import raig

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
methods = raig.make_methods(cat, uniform_eps)
raig_tiered = [m for m in methods if m.name == "RAIG-tiered"]

true_eps_clean = raig.tiered_true_eps(cat, 0.0)

for T in [20, 30, 40, 60]:
    rng = np.random.default_rng(1)
    out = raig.run_paired_trials(cat, raig_tiered, true_eps_clean, R=20, T=T, rng=rng)
    print(f"T={T}: RAIG-tiered accuracy = {out['correct'].mean():.3f}  ({out['correct'].sum()}/20)")

# also check noisy at tiered lambda=1 with T=60 to see if it's purely a budget issue
print()
true_eps_noisy = raig.tiered_true_eps(cat, 1.0)
for T in [20, 40, 60, 100]:
    rng = np.random.default_rng(2)
    out = raig.run_paired_trials(cat, methods, true_eps_noisy, R=20, T=T, rng=rng)
    accs = out["correct"].mean(axis=0)
    print(f"T={T} (noisy lambda=1): " + ", ".join(f"{m.name}={a:.2f}" for m, a in zip(methods, accs)))
