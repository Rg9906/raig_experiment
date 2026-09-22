"""
Reliability specification and estimation (paper Sec. V).

Four ways to obtain the per-attribute crossover probability, in decreasing
order of how much the practitioner has to know:

  oracle          the true vector. Not attainable; an upper reference.
  tiered          a K-tier qualitative annotation (the paper proposal).
  empirical       MEASURED from the catalogue itself, with no annotation and
                  no human study, via repeated-item discordance (below).
  uniform         one number for everything, which is current practice and
                  provably leaves the classical IG ranking unchanged.

The empirical estimator is the part that answers the reviewer objection that
every epsilon in the paper is assumed. It is not a measurement of user error
-- only a human study is -- but it is a measurement of something real that
user error is bounded below by, and it is free.

REPEATED-ITEM DISCORDANCE. Many catalogues contain the same underlying entity
more than once under different identifiers: a song released on a single and
again on an album, a product listed by two vendors, a component with two part
numbers. Those duplicates are independent renderings of the same item, so the
rate at which their DISCRETISED attribute values disagree is a direct,
assumption-free measurement of how fragile that attribute boundary is. An
attribute whose two renderings of the same song disagree 9% of the time has a
boundary that genuinely sits inside the data, which is exactly the property
that makes it hard for a person to answer. In the Music-Akenator catalogue
3,178 (title, artist) groups contain 2 or more distinct track_ids, giving
5,073 within-group pairs.

The estimator measures catalogue instability, not human error, and the two
differ: popularity is release-dependent by definition and so scores high
without being perceptually vague. We therefore use the estimator for its
SHAPE and fix its SCALE with a single stated constant (see empirical_eps).
"""
import numpy as np
import pandas as pd

import raig


# ---------------------------------------------------------------------------
# Repeated-item discordance
# ---------------------------------------------------------------------------
def duplicate_groups(df, name_col="track_name", artist_col="artists",
                     id_col="track_id"):
    """Rows belonging to a (title, artist) group holding >=2 distinct ids."""
    key = (df[name_col].astype(str).str.lower().str.strip() + "||" +
           df[artist_col].astype(str).str.lower().str.strip())
    d = df.assign(_k=key)
    nid = d.groupby("_k")[id_col].nunique()
    return d[d["_k"].isin(nid[nid >= 2].index)]


def _wilson(k, n, z=1.96):
    """Wilson score interval; degrades gracefully at k=0, which happens for
    the attributes that never disagree."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def discordance_table(df, features, attr_tier, **kw):
    """Per-attribute rate at which two renderings of the same item disagree.

    All unordered within-group pairs are used, so a group of 3 contributes 3
    pairs. Returns one row per attribute with a Wilson 95% interval.
    """
    dup = duplicate_groups(df, **kw)
    groups = [g for _, g in dup.groupby("_k", sort=False) if len(g) >= 2]
    rows = []
    for f in features:
        n = d = 0
        for g in groups:
            v = g[f].values
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    n += 1
                    d += int(v[i] != v[j])
        lo, hi = _wilson(d, n)
        rows.append({"attribute": f, "tier": attr_tier[f], "discordant": d,
                     "pairs": n, "rate": (d / n) if n else float("nan"),
                     "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def empirical_eps(disc_table, attr_tier, tier_eps=None, floor=0.01, ceil=0.45):
    """Turn measured discordance into a deployable epsilon-hat vector.

    The estimator supplies the shape. Its scale is not directly interpretable
    as a human error rate -- two catalogue entries disagreeing is not the same
    event as a listener being wrong -- so we apply ONE global multiplier,
    chosen to match the mean of the tier annotation. That single constant is
    the only assumed quantity in this specification, against three for the
    tiered one and a full vector for the oracle.
    """
    tier_eps = tier_eps or raig.TIER_EPS
    rate = disc_table.set_index("attribute")["rate"].to_dict()
    target_mean = float(np.mean([tier_eps[attr_tier[a]] for a in rate]))
    raw_mean = float(np.mean(list(rate.values())))
    kappa = target_mean / raw_mean if raw_mean > 0 else 1.0
    return {a: float(np.clip(kappa * r, floor, ceil)) for a, r in rate.items()}, kappa


# ---------------------------------------------------------------------------
# Literature-anchored priors over the tier crossovers
# ---------------------------------------------------------------------------
# One prior per tier, with mean equal to the point value the paper uses and a
# spread wide enough to cover the range of human-agreement figures reported
# for descriptors of that kind. The concentration is deliberately low: the
# ensemble exists to show the conclusion does not depend on the point values,
# so a tight prior would beg the question.
#
# The prior is a SCALED Beta on [0, 1/2), not a Beta on [0, 1]. The channel
# model constrains epsilon < 1/2 -- at 1/2 the answer is independent of the
# item and above it the user is systematically anti-correlated, which is a
# different phenomenon (an inverted convention) and not something a crossover
# probability should be asked to represent. An unscaled Beta with mean 0.30
# and a usefully wide spread puts real mass above 1/2, so we draw
# epsilon = 0.5 * X with X ~ Beta(2*m*c, (1-2*m)*c), giving E[epsilon] = m and
# support exactly where the model allows.
TIER_PRIOR_CONC = {"objective": 12.0, "semi": 10.0, "subjective": 9.0}


def tier_beta_params(tier_eps=None, conc=None):
    """(alpha, beta) of the underlying Beta; epsilon is half the draw."""
    tier_eps = tier_eps or raig.TIER_EPS
    conc = conc or TIER_PRIOR_CONC
    out = {}
    for t, m in tier_eps.items():
        c = conc[t]
        x = 2.0 * m                      # mean of the underlying Beta
        out[t] = (x * c, (1 - x) * c)
    return out


def sample_attr_eps(attr_tier, rng, tier_eps=None, conc=None):
    """One draw of a per-ATTRIBUTE true epsilon vector from the tier priors.

    Sampling per attribute rather than per tier is deliberate: it produces
    within-tier heterogeneity that the tiered specification cannot represent,
    which is the realistic case and the one that stresses the method.
    """
    ab = tier_beta_params(tier_eps, conc)
    return {a: float(np.clip(0.5 * rng.beta(*ab[t]), 1e-3, 0.499))
            for a, t in attr_tier.items()}


def prior_summary(tier_eps=None, conc=None, rng=None, n=200000):
    rng = rng or np.random.default_rng(0)
    ab = tier_beta_params(tier_eps, conc)
    out = {}
    for t, (a, b) in ab.items():
        s = 0.5 * rng.beta(a, b, size=n)
        out[t] = {"alpha": a, "beta": b, "scale": 0.5, "mean": float(s.mean()),
                  "q025": float(np.quantile(s, 0.025)),
                  "q975": float(np.quantile(s, 0.975))}
    return out


# ---------------------------------------------------------------------------
# Annotation corruption: how wrong can the tiering be before RAIG stops paying?
# ---------------------------------------------------------------------------
def corrupt_tiers(attr_tier, phi, rng, tiers=("objective", "semi", "subjective")):
    """Reassign a fraction phi of attributes to a DIFFERENT tier at random.

    phi=0 is the paper annotation, phi=1 mis-tiers every attribute. This is
    the misspecification axis that matters. Perturbing a HOMOGENEOUS assumed
    crossover, as the conference draft did, provably cannot change the
    selection ranking (Sec. IV-F), so that sweep could only ever have
    measured the belief update, not the selector.
    """
    out = dict(attr_tier)
    attrs = sorted(attr_tier)
    k = int(round(phi * len(attrs)))
    if k == 0:
        return out
    for a in rng.choice(attrs, size=k, replace=False):
        alt = [t for t in tiers if t != attr_tier[a]]
        out[a] = alt[int(rng.integers(len(alt)))]
    return out


# ---------------------------------------------------------------------------
# Psychophysical account of why median splits are fragile
# ---------------------------------------------------------------------------
def psychophysical_eps(values, k, sigma, rng, n_mc=200000):
    """P(a listener assigns the item to the wrong bin) under perceptual noise.

    The listener perceives the z-scored feature as v + sigma*zeta with
    zeta ~ N(0,1) and reports the quantile bin of that perceived value. This
    is the mechanism the paper claims: a k-bin quantile split places every
    boundary in the densest part of the distribution, so the mass within one
    perceptual standard deviation of a boundary -- and hence the error rate --
    is maximal exactly for the features whose boundaries are least meaningful.
    Returned as a probability, computed by Monte Carlo over the real column.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    z = (v - v.mean()) / (v.std() + 1e-12)
    edges = np.quantile(z, np.linspace(0, 1, k + 1)[1:-1])
    idx = rng.integers(0, len(z), size=n_mc)
    true_bin = np.searchsorted(edges, z[idx])
    seen_bin = np.searchsorted(edges, z[idx] + sigma * rng.standard_normal(n_mc))
    return float((true_bin != seen_bin).mean())


def shipped_boundary_eps(values, labels, sigma, rng, n_mc=200000):
    """P(a listener reports the wrong label) under the catalogue's OWN cuts.

    psychophysical_eps above re-bins the column into k equal-population
    quantiles, which answers "what would binning do?" but not "what does THIS
    catalogue's binning do". The music catalogue's *_level columns were cut at
    fixed semantic thresholds by the original pipeline, not at quantiles, so
    their boundaries sit wherever the pipeline put them and their bin masses
    are uneven. This estimator recovers those boundaries from the data and
    measures the flip probability against them.

    Boundaries are taken as the midpoints between adjacent labels once the
    labels are ordered by their median z-score. Labels that do not form
    contiguous intervals of the underlying variable (which would mean the
    column is not a threshold cut of it) are detected and reported by the
    caller through the returned overlap diagnostic.

    Returns (eps, overlap_fraction). `overlap_fraction` is the share of items
    whose label disagrees with the label implied by the recovered boundaries;
    a large value means the recovered cut points do not describe the column and
    the estimate should not be trusted.
    """
    v = np.asarray(values, dtype=np.float64)
    lab = np.asarray(labels)
    ok = np.isfinite(v)
    v, lab = v[ok], lab[ok]
    z = (v - v.mean()) / (v.std() + 1e-12)

    uniq = list(dict.fromkeys(lab))
    order = sorted(uniq, key=lambda L: np.median(z[lab == L]))
    code = {L: i for i, L in enumerate(order)}
    y = np.array([code[L] for L in lab])

    # midpoint between the top of one label and the bottom of the next
    edges = []
    for i in range(len(order) - 1):
        hi = np.quantile(z[y == i], 0.99)
        lo = np.quantile(z[y == i + 1], 0.01)
        edges.append(0.5 * (hi + lo))
    edges = np.array(sorted(edges)) if edges else np.array([])

    implied = np.searchsorted(edges, z)
    overlap = float((implied != y).mean())

    idx = rng.integers(0, len(z), size=n_mc)
    true_bin = np.searchsorted(edges, z[idx])
    seen_bin = np.searchsorted(edges, z[idx] + sigma * rng.standard_normal(n_mc))
    return float((true_bin != seen_bin).mean()), overlap
