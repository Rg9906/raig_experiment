"""Reliable-subset ceiling for the synthetic family, to complete the
scope-condition argument of Sec. VIII-E on the controlled axis as well as on
the three real catalogues. Cheap: no sessions are simulated."""
import json
import math
import numpy as np
import datasets as D
import raig

rows = []
for rho in (-1.0, 0.0, 1.0):
    rng = np.random.default_rng(9000)
    df, feats, eps_map, ig = D.make_synthetic(6000, 20, rng=rng, rho=rho)
    tier = D.eps_to_pseudo_tiers(eps_map)
    rel = [f for f in feats if tier[f] in ("objective", "semi")]
    g_all = df.groupby(list(feats), observed=True).size()
    g_rel = df.groupby(rel, observed=True).size() if rel else None
    rows.append({
        "rho_arg": rho,
        "n_reliable_attrs": len(rel),
        "floor_all": math.log2(len(g_all)),
        "ceiling_all": float((g_all == 1).sum() / len(df)),
        "floor_reliable": (math.log2(len(g_rel)) if g_rel is not None else None),
        "ceiling_reliable": (float((g_rel == 1).sum() / len(df)) if g_rel is not None else None),
    })
    r = rows[-1]
    print(f"rho_arg={rho:+.1f}  reliable attrs={r['n_reliable_attrs']:2d}  "
          f"floor {r['floor_all']:.2f} -> {r['floor_reliable']:.2f} bits  "
          f"ceiling {r['ceiling_all']*100:.1f}% -> {r['ceiling_reliable']*100:.1f}%")

json.dump({"rows": rows}, open("results/synthetic_capacity.json", "w"), indent=2)
print("wrote results/synthetic_capacity.json")
