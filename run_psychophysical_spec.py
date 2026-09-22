"""
A crossover specification derived from the catalogue instead of asserted
(paper Sec. V-C, VIII-B).

The tiered specification asks a curator for three numbers. This one asks for
one. For every attribute that was produced by cutting a continuous column, the
crossover is COMPUTED from that column: recover the cut points the pipeline
actually used, put a listener's perception at one z-scored standard deviation
times sigma away from the truth, and measure how often the reported label
changes. The shape of the resulting vector across attributes comes entirely
from the data -- which columns have boundaries in dense regions, and how many
labels each carries. Only the perceptual noise scale sigma is assumed, plus a
single constant for the natively categorical attributes, where no boundary
effect exists and the error is one of recall rather than of perception.

Two assumed scalars instead of three assumed tier values is a modest saving in
itself. What matters more is that the per-attribute ORDERING is no longer an
annotator's judgement: it is a measurement on the catalogue, so a practitioner
with a new catalogue and no curator can produce it.

Sweeps sigma and the categorical constant, and evaluates each resulting vector
in the primary condition against the incumbent and the tiered specification.

Writes results/psychophysical_spec.json.
"""
import json
import time

import numpy as np

import raig
import reliability as rel

OUT = "results"
SIGMAS = [0.10, 0.25, 0.50]
EPS_CAT = [0.01, 0.03, 0.10]
R, T, SEEDS, BATCH = 300, 20, [0, 1, 2], 30

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
true_eps = raig.tiered_true_eps(cat, 1.0)
tier_eps = cat.eps_vector({a: raig.TIER_EPS[t] for a, t in raig.ATTR_TIER.items()})
uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
unif = np.full(cat.Q, uniform_eps, dtype=np.float32)

# Which attributes have a continuous source column present in the shipped CSV?
derived, categorical = {}, []
for a in raig.FEATURES:
    src = raig.CONTINUOUS_SOURCE.get(a)
    if src is not None and src in df.columns:
        derived[a] = src
    else:
        categorical.append(a)
print(f"derived from a continuous column: {len(derived)}  "
      f"natively categorical (or source absent): {len(categorical)}", flush=True)

rng = np.random.default_rng(4242)
flip = {}          # attribute -> {sigma: eps}
overlap = {}
for a, src in derived.items():
    flip[a] = {}
    for s in SIGMAS:
        e, ov = rel.shipped_boundary_eps(df[src].values, df[a].values, s, rng)
        flip[a][s] = e
        overlap[a] = ov
    print(f"  {a:24s} n_labels={df[a].nunique():2d} overlap={overlap[a]:.3f}  " +
          "  ".join(f"sig={s}: {flip[a][s]:.3f}" for s in SIGMAS), flush=True)


def build_vector(sigma, eps_cat):
    m = {}
    for a in raig.FEATURES:
        m[a] = flip[a][sigma] if a in flip else eps_cat
    return cat.eps_vector(m), m


rows = []
t0 = time.time()
for sigma in SIGMAS:
    for ec in EPS_CAT:
        vec, per_attr = build_vector(sigma, ec)
        ms = [
            raig.Method("IG+soft-uniform", "ig", lambda te: unif),
            raig.Method("RAIG-psycho", "raig", lambda te, h=vec: h),
            raig.Method("RAIG-tiered", "raig", lambda te: tier_eps),
        ]
        for m in ms:
            m.allowed = cat.allowed_mask(m.allowed_tiers)
        names = [m.name for m in ms]
        allc = []
        for s in SEEDS:
            out = raig.run_paired_trials(cat, ms, true_eps, R=R, T=T,
                                         rng=np.random.default_rng(1000 + s),
                                         batch_size=BATCH)
            allc.append(out["correct"])
        c = np.concatenate(allc, axis=0)
        acc = {n: float(c[:, i].mean()) for i, n in enumerate(names)}
        n10, n01, _, p = raig.mcnemar_exact(c[:, 0], c[:, 1])
        n10t, n01t, _, pt = raig.mcnemar_exact(c[:, 2], c[:, 1])
        row = {
            "sigma": sigma, "eps_cat": ec,
            "acc": acc,
            "mean_eps_hat": float(vec.mean()),
            "vs_IG": {"delta": acc["RAIG-psycho"] - acc["IG+soft-uniform"],
                      "n10": n10, "n01": n01, "p": float(p)},
            "vs_tiered": {"delta": acc["RAIG-psycho"] - acc["RAIG-tiered"],
                          "n10": n10t, "n01": n01t, "p": float(pt)},
            "eps_hat_by_attr": per_attr,
        }
        rows.append(row)
        print(f"[{time.time()-t0:6.0f}s] sigma={sigma} eps_cat={ec}: " +
              ", ".join(f"{n}={acc[n]*100:.2f}" for n in names) +
              f"   vs IG {row['vs_IG']['delta']*100:+.2f}pp (p={p:.2e})"
              f"   vs tiered {row['vs_tiered']['delta']*100:+.2f}pp (p={pt:.2e})",
              flush=True)
        json.dump({"sigmas": SIGMAS, "eps_cat": EPS_CAT, "R": R, "T": T,
                   "seeds": SEEDS, "n_total": R * len(SEEDS),
                   "overlap_diagnostic": overlap,
                   "flip_by_attr": {a: {str(s): v for s, v in d.items()}
                                    for a, d in flip.items()},
                   "rows": rows},
                  open(f"{OUT}/psychophysical_spec.json", "w"), indent=2)

best = max(rows, key=lambda r: r["acc"]["RAIG-psycho"])
print(f"\nbest cell: sigma={best['sigma']} eps_cat={best['eps_cat']} "
      f"acc={best['acc']['RAIG-psycho']*100:.2f}% "
      f"(tiered {best['acc']['RAIG-tiered']*100:.2f}%, "
      f"IG {best['acc']['IG+soft-uniform']*100:.2f}%)")
print(f"wrote {OUT}/psychophysical_spec.json ({time.time()-t0:.0f}s)")
