"""
Train the two learned-policy baselines (paper Sec. VII-C) and save their
weights, so the main comparison can load them without retraining.

Writes results/learned_weights.json and reports what each policy learned
about the reliability tiers -- specifically the fitted coefficient on the
(attribute x split-balance) interaction, which is the term through which a
learned policy could in principle discover the reliability weighting.
"""
import json
import time
import numpy as np

import raig
import learned

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
feat = learned.PolicyFeatures(cat)
uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
hat = np.full(cat.Q, uniform_eps, dtype=np.float32)

N_SESS, T = 600, 20
variants = {
    "Learned-noiseblind": raig.tiered_true_eps(cat, 0.0),
    "Learned-noiseaware": raig.tiered_true_eps(cat, 1.0),
}

out = {"n_sessions": N_SESS, "T": T, "D": feat.D, "attrs": feat.attrs,
       "uniform_eps": uniform_eps, "weights": {}, "diagnostics": {}}

for name, true_eps in variants.items():
    t0 = time.time()
    rng = np.random.default_rng(20250 + len(name))
    F, y = learned.collect_rollouts(cat, feat, true_eps, N_SESS, T, rng, hat_eps=hat)
    w = learned.fit_ridge(F, y, lam=1.0)
    pred = F @ w
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot

    # The interaction block: coefficient on (attribute one-hot) x Hb(p).
    k = 4 + feat.n_attr
    w_int = w[k:k + feat.n_attr]
    by_tier = {}
    for tier in ("objective", "semi", "subjective"):
        idx = [i for i, a in enumerate(feat.attrs) if raig.ATTR_TIER[a] == tier]
        by_tier[tier] = float(np.mean(w_int[idx]))

    out["weights"][name] = [float(v) for v in w]
    out["diagnostics"][name] = {"n_samples": int(len(y)), "r2": float(r2),
                                "mean_target_bits": float(y.mean()),
                                "interaction_by_tier": by_tier,
                                "fit_seconds": round(time.time() - t0, 1)}
    print(f"{name}: n={len(y)} R2={r2:.3f} mean bits/turn={y.mean():.3f} "
          f"({time.time()-t0:.0f}s)")
    print(f"   mean (attr x split-balance) coefficient by tier: "
          + ", ".join(f"{t}={v:+.3f}" for t, v in by_tier.items()))

json.dump(out, open("results/learned_weights.json", "w"), indent=2)
print("\nwrote results/learned_weights.json")
print("Interpretation: a policy that discovered the reliability weighting "
      "would show a MORE NEGATIVE interaction coefficient for the subjective "
      "tier than for the objective tier, i.e. it would have learned to "
      "discount balanced splits specifically on unreliable attributes.")
