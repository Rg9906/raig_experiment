"""
Fig. 5: top-1 accuracy as a function of assumed vs. true crossover, both
homogeneous across attributes, for the RAIG selector. Diagonal = correctly
specified. Grid and trial count are reduced from the main protocol (R=100
instead of 300, single seed) since this is a 5x5=25-cell sweep; documented
here rather than in the paper body.
"""
import json
import numpy as np
import raig

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)

GRID = [0.02, 0.10, 0.18, 0.26, 0.34]
R, T, BATCH = 100, 20, 100

rows = []
for true_e in GRID:
    true_eps = raig.homogeneous_true_eps(cat, true_e)
    for assumed_e in GRID:
        hat_eps_vec = np.full(cat.Q, assumed_e, dtype=np.float32)
        m = raig.Method("grid", "raig", lambda te, h=hat_eps_vec: h)
        m.allowed = cat.allowed_mask(m.allowed_tiers)
        rng = np.random.default_rng(3000)
        out = raig.run_paired_trials(cat, [m], true_eps, R=R, T=T, rng=rng, batch_size=BATCH)
        acc = float(out["correct"].mean())
        rows.append({"assumed": assumed_e, "true": true_e, "acc": acc})
        print(f"assumed={assumed_e:.2f} true={true_e:.2f} acc={acc*100:5.1f}%")

with open("results/misspec_grid.json", "w") as f:
    json.dump({"grid": GRID, "R": R, "T": T, "rows": rows}, f, indent=2)
print("saved results/misspec_grid.json")
