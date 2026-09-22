"""
The advantage as a continuous function of the IG/reliability correlation
(paper Sec. VII-H).

The conference version probed this with two isolated controls: "randomised"
(destroy the correlation) and "adversarial" (invert it). Those are two points
on an axis, and reporting only the endpoints leaves the shape of the curve --
and the break-even point on it -- unmeasured. A synthetic catalogue family
makes the axis itself the independent variable.

This also supplies the scope condition the cross-domain results demand. The
bias is present in music (rho < 0) and absent in the two UCI catalogues
(rho ~ 0), so the practitioner question is not "does RAIG work" but "at what
correlation does it start working", which is exactly what this measures.

Writes results/synthetic_rho.json.
"""
import json
import time
import numpy as np
from scipy.stats import spearmanr

import raig
import datasets as D

OUT = "results"
# make_synthetic convention: rho is corr(IG, epsilon). The paper axis is
# corr(IG, 1 - epsilon), so rho = +1 here is the music-like regime
# (informative attributes are the unreliable ones) and rho = -1 is its
# inverse. Every number reported below is on the paper axis.
RHOS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
N_ITEMS, N_ATTRS = 6000, 20
R, T, SEEDS, BATCH = 200, 20, [0, 1, 2], 50

rows = []
t0 = time.time()
for rho_target in RHOS:
    per_seed = []
    for s in SEEDS:
        rng = np.random.default_rng(9000 + s)
        df, feats, eps_map, ig_proxy = D.make_synthetic(
            N_ITEMS, N_ATTRS, rng=rng, rho=rho_target)
        tier_map = D.eps_to_pseudo_tiers(eps_map)
        cat = raig.Catalogue(df, feats, tier_map)
        live = [f for f in feats if len(cat.feature_qidx[f])]
        ig = raig.per_attribute_max_ig(cat)
        rho_actual, _ = spearmanr([ig[f] for f in live],
                                  [1 - eps_map[f] for f in live])

        true_eps = cat.eps_vector(eps_map)
        tiered = cat.eps_vector({a: raig.TIER_EPS[t] for a, t in tier_map.items()})
        ueps = float(np.average([raig.TIER_EPS[tier_map[a]] for a in cat.attr_of_q]))
        unif = np.full(cat.Q, ueps, dtype=np.float32)
        ms = [raig.Method("IG+soft-uniform", "ig", lambda te: unif),
              raig.Method("RAIG-tiered", "raig", lambda te, h=tiered: h),
              raig.Method("RAIG-oracle", "raig", lambda te: te)]
        for m in ms:
            m.allowed = cat.allowed_mask(m.allowed_tiers)
        out = raig.run_paired_trials(cat, ms, true_eps, R=R, T=T,
                                     rng=np.random.default_rng(1000 + s),
                                     batch_size=BATCH, log_questions=True)
        c = out["correct"]
        q = out["qlog"]
        per_seed.append({
            "seed": s, "rho_actual": float(rho_actual), "N": int(cat.N), "Q": int(cat.Q),
            "acc_ig": float(c[:, 0].mean()), "acc_raig": float(c[:, 1].mean()),
            "acc_oracle": float(c[:, 2].mean()),
            "delta": float(c[:, 1].mean() - c[:, 0].mean()),
            "eps_asked_ig": float(true_eps[q[:, 0, :]].mean()),
            "eps_asked_raig": float(true_eps[q[:, 1, :]].mean()),
        })
    d = np.array([p["delta"] for p in per_seed])
    row = {"rho_target": rho_target,
           "rho_actual_mean": float(np.mean([p["rho_actual"] for p in per_seed])),
           "acc_ig": float(np.mean([p["acc_ig"] for p in per_seed])),
           "acc_raig": float(np.mean([p["acc_raig"] for p in per_seed])),
           "acc_oracle": float(np.mean([p["acc_oracle"] for p in per_seed])),
           "delta_mean": float(d.mean()), "delta_sd": float(d.std(ddof=1)),
           "eps_asked_ig": float(np.mean([p["eps_asked_ig"] for p in per_seed])),
           "eps_asked_raig": float(np.mean([p["eps_asked_raig"] for p in per_seed])),
           "per_seed": per_seed}
    rows.append(row)
    print(f"[{time.time()-t0:6.0f}s] rho={rho_target:+.2f} (actual {row['rho_actual_mean']:+.2f}) "
          f"IG={row['acc_ig']*100:5.1f} RAIG={row['acc_raig']*100:5.1f} "
          f"oracle={row['acc_oracle']*100:5.1f} delta={row['delta_mean']*100:+.1f}pp",
          flush=True)
    json.dump({"rhos": RHOS, "n_items": N_ITEMS, "n_attrs": N_ATTRS,
               "R": R, "T": T, "seeds": SEEDS, "rows": rows},
              open(f"{OUT}/synthetic_rho.json", "w"), indent=2)

# Break-even correlation. The reported axis is corr(max-IG, 1 - epsilon), the
# quantity Sec. VI measures on the real catalogues; it is the sign INVERSE of
# the `rho` argument handed to make_synthetic, which correlates IG with epsilon
# itself. Sort by the measured correlation and take the first sign change of
# the paired advantage, in whichever direction it occurs, so the interpolation
# does not depend on how RHOS happens to be ordered.
_o = np.argsort([r["rho_actual_mean"] for r in rows])
ra = np.array([rows[i]["rho_actual_mean"] for i in _o])
de = np.array([rows[i]["delta_mean"] for i in _o])
be = None
for i in range(1, len(ra)):
    if (de[i - 1] > 0 >= de[i]) or (de[i - 1] <= 0 < de[i]):
        be = float(ra[i - 1] + (0 - de[i - 1]) * (ra[i] - ra[i - 1]) /
                   (de[i] - de[i - 1]))
        break
json.dump({"rhos": RHOS, "n_items": N_ITEMS, "n_attrs": N_ATTRS, "R": R, "T": T,
           "seeds": SEEDS, "break_even_rho": be, "rows": rows},
          open(f"{OUT}/synthetic_rho.json", "w"), indent=2)
print(f"\nbreak-even corr(max-IG, 1-eps) = {be}")
for i in range(len(ra)):
    print(f"   rho={ra[i]:+.3f}  delta={de[i]*100:+.2f}pp")
print(f"wrote {OUT}/synthetic_rho.json ({time.time()-t0:.0f}s)")
