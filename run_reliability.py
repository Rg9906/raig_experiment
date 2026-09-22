"""
Sec. V results: measuring attribute reliability instead of assuming it.

Produces:
  results/reliability_discordance.json   repeated-item discordance per attribute
  results/reliability_priors.json        the tier Beta priors, summarised
  results/reliability_psychophysical.json bin-flip probability vs perceptual noise
  results/empirical_eps.json             the deployable epsilon-hat vector

Pure analysis; no session simulation, so it is cheap and runs first because
the main comparison needs empirical_eps.json.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import raig
import reliability as rel

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)

# ---------------------------------------------------------------------------
# 1. Repeated-item discordance
# ---------------------------------------------------------------------------
print("measuring repeated-item discordance...", flush=True)
disc = rel.discordance_table(df, raig.FEATURES, raig.ATTR_TIER)
disc = disc.sort_values("rate", ascending=False)
print(disc.to_string(index=False))

n_groups = rel.duplicate_groups(df)["_k"].nunique()
n_pairs = int(disc["pairs"].iloc[0])

by_tier = disc.groupby("tier")["rate"].agg(["mean", "median", "count"])
print("\nby tier:\n", by_tier)

# Does the measurement agree with the annotation? Spearman between the
# annotated epsilon and the measured discordance, over attributes.
ann = disc["tier"].map(raig.TIER_EPS).values
rho, p = spearmanr(ann, disc["rate"].values)
print(f"\nSpearman(annotated eps, measured discordance) = {rho:.3f} (p={p:.4f})")

# Binary contrast: objective vs the rest. This is the comparison the K=2
# ablation actually depends on, and it is the one the data supports.
obj = disc[disc.tier == "objective"]["rate"].mean()
rest = disc[disc.tier != "objective"]["rate"].mean()
print(f"objective mean={obj:.4f}  non-objective mean={rest:.4f}  ratio={rest/max(obj,1e-9):.2f}x")

# The two attributes that make the ordering messy, diagnosed rather than dropped.
diag = {a: float(disc.set_index("attribute").loc[a, "rate"])
        for a in ("popularity_level", "genre")}
disc_ex = disc[~disc.attribute.isin(("popularity_level", "genre"))]
rho_ex, p_ex = spearmanr(disc_ex["tier"].map(raig.TIER_EPS).values, disc_ex["rate"].values)
print(f"excluding popularity_level and genre: rho={rho_ex:.3f} (p={p_ex:.4f})")

# ---------------------------------------------------------------------------
# 2. Deployable epsilon-hat from the measurement
# ---------------------------------------------------------------------------
emp_eps, kappa = rel.empirical_eps(disc, raig.ATTR_TIER)
print(f"\nempirical eps-hat (kappa={kappa:.3f}):")
for a in sorted(emp_eps, key=lambda x: -emp_eps[x]):
    print(f"   {a:26s} {emp_eps[a]:.4f}   (tier {raig.ATTR_TIER[a]})")

# ---------------------------------------------------------------------------
# 3. Tier priors
# ---------------------------------------------------------------------------
priors = rel.prior_summary()
print("\ntier priors:")
for t, s in priors.items():
    print(f"   {t:11s} Beta({s['alpha']:.2f},{s['beta']:.2f})  "
          f"mean={s['mean']:.3f}  95%=[{s['q025']:.3f},{s['q975']:.3f}]")

# ---------------------------------------------------------------------------
# 4. Psychophysical bin-flip probability
# ---------------------------------------------------------------------------
rng = np.random.default_rng(11)
psy = []
for attr, src in raig.CONTINUOUS_SOURCE.items():
    if src not in df.columns:
        continue
    for k in (2, 4):
        for sigma in (0.10, 0.25, 0.50):
            psy.append({"attribute": attr, "bins": k, "sigma": sigma,
                        "eps": rel.psychophysical_eps(df[src].values, k, sigma, rng)})
psy_df = pd.DataFrame(psy)
print("\npsychophysical bin-flip probability, mean over attributes:")
print(psy_df.groupby(["bins", "sigma"])["eps"].mean().to_string())

out_dir = "results"
json.dump({"n_groups": int(n_groups), "n_pairs": n_pairs,
           "rows": disc.to_dict(orient="records"),
           "by_tier": by_tier.to_dict(),
           "spearman_annotated_vs_measured": {"rho": float(rho), "p": float(p)},
           "spearman_excluding_popularity_genre": {"rho": float(rho_ex), "p": float(p_ex),
                                                   "excluded": diag},
           "objective_mean": float(obj), "non_objective_mean": float(rest),
           "ratio": float(rest / max(obj, 1e-9))},
          open(f"{out_dir}/reliability_discordance.json", "w"), indent=2)
json.dump({"kappa": float(kappa), "eps": emp_eps},
          open(f"{out_dir}/empirical_eps.json", "w"), indent=2)
json.dump(priors, open(f"{out_dir}/reliability_priors.json", "w"), indent=2)
json.dump(psy_df.to_dict(orient="records"),
          open(f"{out_dir}/reliability_psychophysical.json", "w"), indent=2)
print("\nwrote 4 result files")
