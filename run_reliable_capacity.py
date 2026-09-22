"""
How much of each catalogue's discriminating information sits in attributes a
user can actually answer? (paper Sec. VIII-D)

The cross-domain results have a pattern the bias measurement alone does not
explain: reliability-aware selection is transformative on dermatology, where
the IG/reliability correlation is essentially zero, and inert on mushroom,
where it is also essentially zero. Correlation is therefore not what decides
whether the correction pays.

What decides it is whether the RELIABLE attributes still carry enough
information to finish the job. This script measures that directly: for each
catalogue it reports log2 of the number of distinct attribute vectors formed
by (a) all attributes, (b) objective attributes only, and (c) objective plus
semi-objective, and compares each against the capacity-corrected budget the
system actually has. A catalogue whose reliable subset is already close to
fully identifying is one where a selector can afford to avoid the vague
attributes; a catalogue where it is not is one where the vague attributes must
be asked and only weighted.

Writes results/reliable_capacity.json.
"""
import json
import math

import numpy as np

import raig
import datasets as D

OUT = "results"
T_BUDGET = 20


def analyse(name, df, features, tier_map):
    out = {"catalogue": name, "N": int(len(df)), "M": len(features), "subsets": {}}
    subsets = {
        "all": features,
        "objective": [f for f in features if tier_map[f] == "objective"],
        "objective+semi": [f for f in features if tier_map[f] in ("objective", "semi")],
    }
    for label, feats in subsets.items():
        if not feats:
            out["subsets"][label] = None
            continue
        groups = df.groupby(list(feats), observed=True).size()
        distinct = int(len(groups))
        floor = math.log2(distinct)
        # capacity-corrected budget bound using the best epsilon in the subset
        best_eps = min(raig.TIER_EPS[tier_map[f]] for f in feats)
        cap = 1.0 - float(raig.Hb(np.array([best_eps]))[0])
        out["subsets"][label] = {
            "n_attributes": len(feats),
            "distinct_vectors": distinct,
            "entropy_floor_bits": floor,
            "unique_frac": float((groups == 1).sum() / len(df)),
            "largest_class": int(groups.max()),
            "top1_ceiling": float((groups == 1).sum() / len(df)),
            "T_min": floor / cap,
        }
    a = out["subsets"]["all"]["entropy_floor_bits"]
    rel = out["subsets"]["objective+semi"]["entropy_floor_bits"]
    out["reliable_information_share"] = rel / a
    out["reliable_subset_fits_budget"] = bool(
        out["subsets"]["objective+semi"]["T_min"] <= T_BUDGET)
    return out


rows = []

dfm = raig.load_catalogue("data/dataset_final.csv")
rows.append(analyse("music", dfm, raig.FEATURES, raig.ATTR_TIER))

dfu = D.load_mushroom()
rows.append(analyse("mushroom", dfu, D.MUSHROOM_FEATURES, D.MUSHROOM_TIER))

dfd = D.load_dermatology()
rows.append(analyse("dermatology", dfd, D.DERM_FEATURES, D.DERM_TIER))

for r in rows:
    print(f"\n=== {r['catalogue']} (N={r['N']}, M={r['M']}) ===")
    for label, s in r["subsets"].items():
        if s is None:
            continue
        print(f"  {label:15s} M={s['n_attributes']:2d} distinct={s['distinct_vectors']:6d} "
              f"floor={s['entropy_floor_bits']:6.2f} bits  ceiling={s['top1_ceiling']*100:5.1f}%  "
              f"T_min={s['T_min']:6.2f}")
    print(f"  reliable information share = {r['reliable_information_share']*100:.1f}% "
          f"of the full entropy floor; reliable subset fits T={T_BUDGET}: "
          f"{r['reliable_subset_fits_budget']}")

json.dump({"T_budget": T_BUDGET, "rows": rows},
          open(f"{OUT}/reliable_capacity.json", "w"), indent=2)
print(f"\nwrote {OUT}/reliable_capacity.json")
