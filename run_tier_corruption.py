"""
How wrong can the annotation be before the correction stops paying?
(paper Sec. VII-G)

This replaces the misspecification surface of the conference version, which
could not have shown anything. That surface swept a HOMOGENEOUS assumed
crossover against a homogeneous true one -- but a constant assumed crossover
adds the same capacity term to every question and therefore leaves the
selection ranking exactly unchanged (Sec. IV-F). The columns of that grid were
mathematically incapable of affecting which questions got asked, so the
experiment could only ever have measured the belief update. The draft
diagnosed this after the fact; here we run the sweep that bears on selection.

The axis that matters is whether attributes are placed in the RIGHT TIER. We
mis-tier a fraction phi of attributes, drawn at random and reassigned to a
different tier, and sweep phi from 0 (the paper annotation) to 1 (every
attribute mis-tiered). The truth is held fixed throughout, so the only thing
degrading is the annotation.

Writes results/tier_corruption.json.
"""
import json
import time
import numpy as np

import raig
import reliability as rel

OUT = "results"
PHIS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
N_REPS = 3          # independent corruptions per phi, to average over which
                    # attributes happen to be mis-tiered
R, T, SEEDS, BATCH = 100, 20, [0, 1, 2], 30

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
true_eps = raig.tiered_true_eps(cat, 1.0)          # the truth, fixed
unif_vec = np.full(cat.Q, uniform_eps, dtype=np.float32)

rng_corrupt = np.random.default_rng(31337)
rows = []
t0 = time.time()
for phi in PHIS:
    reps = []
    for rep in range(1 if phi == 0.0 else N_REPS):
        tiers = rel.corrupt_tiers(raig.ATTR_TIER, phi, rng_corrupt)
        n_wrong = sum(1 for a in tiers if tiers[a] != raig.ATTR_TIER[a])
        hat = cat.eps_vector({a: raig.TIER_EPS[t] for a, t in tiers.items()})
        ms = [raig.Method("IG+soft-uniform", "ig", lambda te: unif_vec),
              raig.Method("RAIG-corrupted", "raig", lambda te, h=hat: h)]
        for m in ms:
            m.allowed = cat.allowed_mask(m.allowed_tiers)
        allc = []
        for s in SEEDS:
            out = raig.run_paired_trials(cat, ms, true_eps, R=R, T=T,
                                         rng=np.random.default_rng(1000 + s),
                                         batch_size=BATCH, log_questions=True)
            allc.append(out["correct"])
        c = np.concatenate(allc, axis=0)
        reps.append({"rep": rep, "n_mis_tiered": n_wrong,
                     "acc_ig": float(c[:, 0].mean()),
                     "acc_raig": float(c[:, 1].mean()),
                     "delta": float(c[:, 1].mean() - c[:, 0].mean())})
    d = np.array([r["delta"] for r in reps])
    row = {"phi": phi, "n_reps": len(reps),
           "mean_acc_ig": float(np.mean([r["acc_ig"] for r in reps])),
           "mean_acc_raig": float(np.mean([r["acc_raig"] for r in reps])),
           "mean_delta": float(d.mean()), "min_delta": float(d.min()),
           "max_delta": float(d.max()), "reps": reps}
    rows.append(row)
    print(f"[{time.time()-t0:6.0f}s] phi={phi:<4} IG={row['mean_acc_ig']*100:5.2f} "
          f"RAIG={row['mean_acc_raig']*100:5.2f}  delta={row['mean_delta']*100:+.2f}pp",
          flush=True)
    json.dump({"phis": PHIS, "R": R, "T": T, "seeds": SEEDS, "n_reps": N_REPS,
               "rows": rows}, open(f"{OUT}/tier_corruption.json", "w"), indent=2)

# Break-even corruption level: the phi at which the advantage reaches zero.
ph = np.array([r["phi"] for r in rows]); de = np.array([r["mean_delta"] for r in rows])
be = None
for i in range(1, len(ph)):
    if de[i - 1] > 0 >= de[i]:
        be = float(ph[i - 1] + (0 - de[i - 1]) * (ph[i] - ph[i - 1]) / (de[i] - de[i - 1]))
        break
json.dump({"phis": PHIS, "R": R, "T": T, "seeds": SEEDS, "n_reps": N_REPS,
           "break_even_phi": be, "rows": rows},
          open(f"{OUT}/tier_corruption.json", "w"), indent=2)
print(f"\nbreak-even corruption fraction phi = {be}")
print(f"wrote {OUT}/tier_corruption.json ({time.time()-t0:.0f}s)")
