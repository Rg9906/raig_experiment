"""
Cross-domain bias characterisation and method comparison (paper Sec. VII-A,
VII-H).

Part A asks whether the selection bias reported for the music catalogue is a
general property of attribute-indexed catalogues or a property of how THIS
catalogue was built. It is the second, and saying so precisely is worth more
than claiming the first.

STATISTICS. The conference version measured the bias with a Spearman
correlation between an attribute maximum information gain and its annotated
reliability. Reliability takes only THREE distinct values (one per tier), so
that correlation is computed over a variable that is almost all ties, which
inflates its apparent resolution and makes the p-value hard to interpret. We
therefore report, as the primary test, a Kruskal-Wallis comparison of maximum
information gain ACROSS the three tiers -- the hypothesis actually of
interest, and one that handles ties natively -- plus Kendall tau-b, which has
a proper tie correction. Spearman is retained for comparability with the
earlier version, not relied upon.

Part B runs the six-method comparison on the two UCI catalogues.

Writes results/domains_bias.json and results/domains_methods.json.
"""
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, kruskal, mannwhitneyu

import raig
import datasets as D
import methods_ext

OUT = "results"
TIER_ORDER = ("objective", "semi", "subjective")


def attribute_table(cat, tier_map, binned=frozenset()):
    ig = raig.per_attribute_max_ig(cat)
    rows = []
    for f in cat.features:
        idx = cat.feature_qidx[f]
        if len(idx) == 0:
            continue                      # degenerate attribute, no question
        share = cat.X[:, idx].mean(axis=0)
        rows.append({
            "attribute": f, "tier": tier_map[f],
            "eps": raig.TIER_EPS[tier_map[f]],
            "reliability": 1 - raig.TIER_EPS[tier_map[f]],
            "max_ig": ig[f],
            "cardinality": int(len(idx)),
            # how lopsided the attribute is: the share of its most common
            # value. A skewed attribute cannot produce a balanced binary
            # question however informative it is as a whole.
            "max_value_share": float(share.max()),
            "binned": f in binned,
        })
    return pd.DataFrame(rows)


def bias_stats(tab, label):
    kw_groups = [tab[tab.tier == t]["max_ig"].values for t in TIER_ORDER]
    kw_groups = [g for g in kw_groups if len(g) > 0]
    H, p_kw = kruskal(*kw_groups) if len(kw_groups) > 1 else (np.nan, np.nan)
    tau, p_tau = kendalltau(tab["max_ig"], tab["reliability"], variant="b")
    rho, p_rho = spearmanr(tab["max_ig"], tab["reliability"])
    obj = tab[tab.tier == "objective"]["max_ig"]
    rest = tab[tab.tier != "objective"]["max_ig"]
    if len(obj) and len(rest):
        U, p_mw = mannwhitneyu(obj, rest, alternative="less")
    else:
        U, p_mw = np.nan, np.nan
    out = {
        "domain": label, "n_attributes": int(len(tab)),
        "kruskal_H": float(H), "kruskal_p": float(p_kw),
        "kendall_tau_b": float(tau), "kendall_p": float(p_tau),
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "mannwhitney_obj_lower_U": float(U), "mannwhitney_p": float(p_mw),
        "mean_max_ig_by_tier": {t: float(tab[tab.tier == t]["max_ig"].mean())
                                for t in TIER_ORDER if (tab.tier == t).any()},
        "mean_skew_by_tier": {t: float(tab[tab.tier == t]["max_value_share"].mean())
                              for t in TIER_ORDER if (tab.tier == t).any()},
        "mean_cardinality_by_tier": {t: float(tab[tab.tier == t]["cardinality"].mean())
                                     for t in TIER_ORDER if (tab.tier == t).any()},
    }
    print(f"\n=== {label} (n={out['n_attributes']} attributes) ===")
    print(f"  Kruskal-Wallis max-IG across tiers : H={H:.2f}  p={p_kw:.4f}   [primary]")
    print(f"  Kendall tau-b(max-IG, reliability) : {tau:+.3f} p={p_tau:.4f}  [tie-robust]")
    print(f"  Spearman rho (as in the conf. ver.): {rho:+.3f} p={p_rho:.4f}")
    print(f"  Mann-Whitney objective max-IG lower: p={p_mw:.4f}")
    print("  mean max IG by tier : " +
          ", ".join(f"{t}={v:.3f}" for t, v in out["mean_max_ig_by_tier"].items()))
    print("  mean skew   by tier : " +
          ", ".join(f"{t}={v:.3f}" for t, v in out["mean_skew_by_tier"].items()))
    return out


# ---------------------------------------------------------------------------
# Part A: bias across domains
# ---------------------------------------------------------------------------
bias = {}

df_m = raig.load_catalogue("data/dataset_final.csv")
cat_m = raig.Catalogue(df_m)
tab_m = attribute_table(cat_m, raig.ATTR_TIER, binned=set(raig.CONTINUOUS_SOURCE))
bias["music"] = bias_stats(tab_m, "music")

# The discretisation mechanism, tested within the music catalogue: do the
# attributes produced by quantile-binning a continuous column carry more
# apparent information than the natively categorical ones?
b, nb = tab_m[tab_m.binned], tab_m[~tab_m.binned]
U, p = mannwhitneyu(b["max_ig"], nb["max_ig"], alternative="greater")
bias["music"]["discretisation"] = {
    "n_binned": int(len(b)), "n_categorical": int(len(nb)),
    "mean_max_ig_binned": float(b["max_ig"].mean()),
    "mean_max_ig_categorical": float(nb["max_ig"].mean()),
    "mean_eps_binned": float(b["eps"].mean()),
    "mean_eps_categorical": float(nb["eps"].mean()),
    "mannwhitney_U": float(U), "mannwhitney_p": float(p),
    "spearman_within_binned": [float(x) for x in spearmanr(b["max_ig"], b["reliability"])],
    "spearman_within_categorical": [float(x) for x in spearmanr(nb["max_ig"], nb["reliability"])],
}
print(f"\n  [discretisation] binned max-IG={b['max_ig'].mean():.3f} vs "
      f"categorical={nb['max_ig'].mean():.3f}, Mann-Whitney p={p:.4f}")
print(f"  [discretisation] binned mean eps={b['eps'].mean():.3f} vs "
      f"categorical={nb['eps'].mean():.3f}")

df_mu = D.load_mushroom()
cat_mu = raig.Catalogue(df_mu, D.MUSHROOM_FEATURES, D.MUSHROOM_TIER)
tab_mu = attribute_table(cat_mu, D.MUSHROOM_TIER)
bias["mushroom"] = bias_stats(tab_mu, "mushroom")

df_de = D.load_dermatology()
cat_de = raig.Catalogue(df_de, D.DERM_FEATURES, D.DERM_TIER)
tab_de = attribute_table(cat_de, D.DERM_TIER)
bias["dermatology"] = bias_stats(tab_de, "dermatology")

for name, cat, feats in (("music", cat_m, raig.FEATURES),
                         ("mushroom", cat_mu, D.MUSHROOM_FEATURES),
                         ("dermatology", cat_de, D.DERM_FEATURES)):
    audit = raig.identifiability_audit(cat.df, feats)
    audit["Q_compiled"] = int(cat.Q)
    bias[name]["identifiability"] = {k: float(v) for k, v in audit.items()}
    print(f"\n  [{name}] N={audit['N']} Q={cat.Q} distinct={audit['distinct_vectors']} "
          f"unique={audit['uniquely_identifiable_frac']*100:.1f}% "
          f"floor={audit['entropy_floor_bits']:.2f} bits")

bias["_attribute_tables"] = {
    "music": tab_m.to_dict(orient="records"),
    "mushroom": tab_mu.to_dict(orient="records"),
    "dermatology": tab_de.to_dict(orient="records"),
}
json.dump(bias, open(f"{OUT}/domains_bias.json", "w"), indent=2, default=float)
print(f"\nwrote {OUT}/domains_bias.json")

# ---------------------------------------------------------------------------
# Part B: method comparison on the two UCI catalogues
# ---------------------------------------------------------------------------
R, T, SEEDS, BATCH = 200, 20, [0, 1, 2], 50
res = {}
t0 = time.time()
for name, cat, tier_map in (("mushroom", cat_mu, D.MUSHROOM_TIER),
                            ("dermatology", cat_de, D.DERM_TIER)):
    methods, ueps, tiered = methods_ext.build_generic(cat, tier_map)
    names = [m.name for m in methods]
    res[name] = {"uniform_eps": ueps, "N": int(cat.N), "Q": int(cat.Q), "conditions": {}}
    for lam in (0.5, 1.0, 1.5):
        true_eps = (lam * tiered).astype(np.float32)
        allc, allq = [], []
        for s in SEEDS:
            out = raig.run_paired_trials(cat, methods, true_eps, R=R, T=T,
                                         rng=np.random.default_rng(1000 + s),
                                         batch_size=BATCH, track_top5=True,
                                         log_questions=True)
            allc.append(out["correct"]); allq.append(out["qlog"])
        correct = np.concatenate(allc, axis=0)
        qlog = np.concatenate(allq, axis=0)
        cell = {}
        for mi, nm in enumerate(names):
            mean, lo, hi = raig.acc_ci(correct[:, mi])
            q = qlog[:, mi, :]
            tiers = cat.tier_of_q[q].ravel()
            cell[nm] = {"acc": float(mean), "lo": float(lo), "hi": float(hi),
                        "mean_eps_asked": float(true_eps[q].mean()),
                        "tier_mix": {t: float((tiers == t).mean()) for t in TIER_ORDER}}
        i_r, i_u = names.index("RAIG-tiered"), names.index("IG+soft-uniform")
        n10, n01, _, p = raig.mcnemar_exact(correct[:, i_u], correct[:, i_r])
        cell["_mcnemar_RAIGtiered_vs_IGsoftuniform"] = {
            "n10": n10, "n01": n01, "p": float(p),
            "delta": float(correct[:, i_r].mean() - correct[:, i_u].mean())}
        res[name]["conditions"][f"lambda_{lam}"] = cell
        print(f"[{time.time()-t0:6.0f}s] {name} lam={lam}: " +
              ", ".join(f"{n}={cell[n]['acc']*100:.1f}" for n in names) +
              f"  | McNemar p={p:.2e}", flush=True)
    json.dump({"R": R, "T": T, "seeds": SEEDS, "results": res},
              open(f"{OUT}/domains_methods.json", "w"), indent=2)

print(f"\nwrote {OUT}/domains_methods.json  ({time.time()-t0:.0f}s)")
