"""
Robustness of the primary result to the assumed crossover values
(paper Sec. VII-F). This is the experiment that answers the strongest
objection to the conference version: that all three epsilon values are
assumed rather than measured, so the headline result could be an artefact of
having picked them.

Design. The DEPLOYED system is held fixed: RAIG-tiered always assumes the
point values (0.03, 0.15, 0.30), because that is what a practitioner who
followed the paper would actually ship. What varies is the TRUTH. For each
draw we sample a per-attribute true crossover from the tier priors
(reliability.sample_attr_eps), which produces within-tier heterogeneity the
tiered specification cannot represent, and run the deployed system against it.

The question is therefore not "does the method work when it knows the answer"
but "does a system built on three guessed numbers still beat current practice
when the world does not match those numbers". Reported as the distribution of
the paired accuracy difference over draws, plus the fraction of draws in which
RAIG-tiered wins.

Writes results/eps_ensemble.json.
"""
import json
import time
import numpy as np

import raig
import reliability as rel
import methods_ext

OUT = "results"
N_DRAWS, R, T, BATCH = 60, 150, 20, 30
KEEP = ("IG+soft-uniform", "RAIG-tiered", "RAIG-oracle")

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
methods, uniform_eps = methods_ext.build(cat, include=KEEP)
names = [m.name for m in methods]
i_u, i_r, i_o = (names.index(n) for n in KEEP)
print(f"methods={names}  draws={N_DRAWS}  R={R}", flush=True)

rng_draw = np.random.default_rng(777)
rows = []
t0 = time.time()
for d in range(N_DRAWS):
    eps_map = rel.sample_attr_eps(raig.ATTR_TIER, rng_draw)
    true_eps = cat.eps_vector(eps_map)
    out = raig.run_paired_trials(cat, methods, true_eps, R=R, T=T,
                                 rng=np.random.default_rng(4000 + d),
                                 batch_size=BATCH)
    c = out["correct"]
    n10, n01, _, p = raig.mcnemar_exact(c[:, i_u], c[:, i_r])
    rows.append({
        "draw": d,
        "acc_ig": float(c[:, i_u].mean()),
        "acc_raig": float(c[:, i_r].mean()),
        "acc_oracle": float(c[:, i_o].mean()),
        "delta": float(c[:, i_r].mean() - c[:, i_u].mean()),
        "n10": n10, "n01": n01, "p": float(p),
        # summary of the drawn world, so the outcome can be regressed on it
        "mean_true_eps": float(np.mean(list(eps_map.values()))),
        "eps_by_tier": {t: float(np.mean([v for a, v in eps_map.items()
                                          if raig.ATTR_TIER[a] == t]))
                        for t in ("objective", "semi", "subjective")},
    })
    if (d + 1) % 10 == 0:
        dl = np.array([r["delta"] for r in rows])
        print(f"[{time.time()-t0:6.0f}s] {d+1}/{N_DRAWS} draws: "
              f"mean delta={dl.mean()*100:+.2f}pp  wins={int((dl>0).sum())}/{len(dl)}",
              flush=True)
        json.dump({"n_draws": len(rows), "R": R, "T": T, "rows": rows},
                  open(f"{OUT}/eps_ensemble.json", "w"), indent=2)

delta = np.array([r["delta"] for r in rows])
summary = {
    "n_draws": len(rows), "R": R, "T": T, "n_per_draw": R,
    "mean_delta": float(delta.mean()),
    "median_delta": float(np.median(delta)),
    "q025": float(np.quantile(delta, 0.025)),
    "q975": float(np.quantile(delta, 0.975)),
    "frac_raig_wins": float((delta > 0).mean()),
    "frac_raig_wins_significant": float(np.mean([(r["delta"] > 0 and r["p"] < 0.05)
                                                 for r in rows])),
    "frac_ig_wins_significant": float(np.mean([(r["delta"] < 0 and r["p"] < 0.05)
                                               for r in rows])),
}
json.dump({**summary, "rows": rows}, open(f"{OUT}/eps_ensemble.json", "w"), indent=2)
print("\n" + json.dumps(summary, indent=2))
print(f"\nwrote {OUT}/eps_ensemble.json ({time.time()-t0:.0f}s)")
