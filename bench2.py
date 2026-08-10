import time
import numpy as np
import raig

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
methods = raig.make_methods(cat, uniform_eps)

true_eps_clean = raig.tiered_true_eps(cat, 0.0)
true_eps_noisy = raig.tiered_true_eps(cat, 1.0)

for R, T, bs in [(60, 20, 60), (300, 20, 150)]:
    rng = np.random.default_rng(7)
    t0 = time.time()
    out = raig.run_paired_trials(cat, methods, true_eps_noisy, R=R, T=T, rng=rng, batch_size=bs)
    dt = time.time() - t0
    accs = out["correct"].mean(axis=0)
    print(f"R={R} T={T} batch={bs}: {dt:.2f}s total, {dt/R*1000:.2f} ms/trial")
    print("  accuracy:", {m.name: round(a, 3) for m, a in zip(methods, accs)})
