"""
Reference implementation for "Information Gain Prefers the Questions Users
Cannot Answer: Reliability-Aware Query Selection for Interactive Identification".

Implements Algorithms 1-4 and the analysis functions (measure_pathology,
identifiability audit, mcnemar, paired bootstrap CI) against the real
Music-Akenator catalogue (dataset_final.csv, ~79.8k songs x 21 attributes).
"""
import math
import time
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

RNG_GLOBAL = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 21 attributes, exactly as used by the original entropy/ML engines.
# ---------------------------------------------------------------------------
FEATURES = [
    "genre", "mood", "tempo", "language", "popularity_level",
    "duration_length", "danceability_level", "energy_level", "valence_level",
    "acoustic_level", "instrumental_level", "liveness_level", "speechiness_level",
    "loudness_level", "key_category", "mode_category", "time_signature_category",
    "content_rating", "artist_type", "release_type", "track_version",
]

# Raw continuous column backing each *_level (or bucketed) attribute, used
# only by the discretisation ablation (Sec. VI-F / Table ablation).
CONTINUOUS_SOURCE = {
    "popularity_level": "popularity",
    "duration_length": "duration_ms",
    "danceability_level": "danceability",
    "energy_level": "energy",
    "valence_level": "valence",
    "acoustic_level": "acousticness",
    "instrumental_level": "instrumentalness",
    "liveness_level": "liveness",
    "speechiness_level": "speechiness",
    "loudness_level": "loudness",
    "tempo": "tempo_bpm",
}

# ---------------------------------------------------------------------------
# Three-tier reliability annotation (Sec. III-F / Table tiers).
#
# Curatorial rationale (documented, not measured):
#   objective       -- catalogue/metadata facts a listener knows without
#                       having to judge anything: language, explicit-content
#                       flag, artist configuration, which version/edit it is,
#                       and the catalogued genre label.
#   semi-objective  -- well-defined but only imperfectly recalled or
#                       estimated: popularity, duration, whether it's a live
#                       recording, vocal/instrumental balance, acoustic vs.
#                       electronic production, loudness, speechiness.
#   subjective      -- perceptual judgements (mood, energy, danceability,
#                       valence, perceived tempo -- the paper's running
#                       example of a median-split attribute) together with
#                       music-theoretic attributes (key, mode, time
#                       signature) that are well defined but inaccessible to
#                       a non-musician, hence answered close to chance.
# ---------------------------------------------------------------------------
ATTR_TIER = {
    "language": "objective",
    "content_rating": "objective",
    "artist_type": "objective",
    "track_version": "objective",
    "genre": "objective",
    "release_type": "objective",   # degenerate (single value); tier moot

    "popularity_level": "semi",
    "duration_length": "semi",
    "liveness_level": "semi",
    "instrumental_level": "semi",
    "acoustic_level": "semi",
    "loudness_level": "semi",
    "speechiness_level": "semi",

    "mood": "subjective",
    "tempo": "subjective",
    "danceability_level": "subjective",
    "energy_level": "subjective",
    "valence_level": "subjective",
    "key_category": "subjective",
    "mode_category": "subjective",
    "time_signature_category": "subjective",
}
TIER_EPS = {"objective": 0.03, "semi": 0.15, "subjective": 0.30}

assert set(ATTR_TIER) == set(FEATURES)


# ---------------------------------------------------------------------------
# Data loading / discretisation
# ---------------------------------------------------------------------------
def load_catalogue(final_csv="data/dataset_final.csv", raw_csv=None):
    df = pd.read_csv(final_csv)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    if raw_csv is not None:
        raw = pd.read_csv(raw_csv, usecols=["track_id", "tempo"])
        raw = raw.rename(columns={"tempo": "tempo_bpm"}).drop_duplicates("track_id")
        df = df.merge(raw, on="track_id", how="left")
    return df


def discretize(series: pd.Series, k: int) -> pd.Series:
    """Quantile-bin a continuous series into k labelled bins (k>=2)."""
    ranks = series.rank(method="first")
    bins = pd.qcut(ranks, k, labels=[f"bin{i}" for i in range(k)])
    return bins.astype(str)


def rebin_attribute(df: pd.DataFrame, attr: str, k: int) -> pd.Series:
    src = CONTINUOUS_SOURCE[attr]
    return discretize(df[src], k)


# ---------------------------------------------------------------------------
# Catalogue compilation (Sec. III-B / compile_catalogue)
# ---------------------------------------------------------------------------
class Catalogue:
    def __init__(self, df: pd.DataFrame, features=FEATURES, attr_tier=ATTR_TIER):
        self.df = df
        self.N = len(df)
        self.features = features
        cols, attr_of_q, val_of_q = [], [], []
        for f in features:
            vals = df[f].unique()
            for v in vals:
                mask = (df[f].values == v)
                s = mask.sum()
                if s == 0 or s == self.N:
                    continue  # degenerate, drop
                cols.append(mask.astype(np.float32))
                attr_of_q.append(f)
                val_of_q.append(v)
        self.X = np.stack(cols, axis=1)  # (N, Q) float32
        self.Q = self.X.shape[1]
        self.attr_of_q = np.array(attr_of_q)
        self.val_of_q = np.array(val_of_q, dtype=object)
        self.feature_qidx = {
            f: np.where(self.attr_of_q == f)[0] for f in features
        }

    def eps_vector(self, per_attr_eps: dict) -> np.ndarray:
        return np.array([per_attr_eps[a] for a in self.attr_of_q], dtype=np.float32)

    def allowed_mask(self, tiers_allowed) -> np.ndarray:
        return np.array([ATTR_TIER[a] in tiers_allowed for a in self.attr_of_q])


# ---------------------------------------------------------------------------
# Information measures
# ---------------------------------------------------------------------------
def Hb(p):
    # float32-safe bounds: 1 - 1e-12 rounds to exactly 1.0 in float32, which
    # would leave log2(1-p) = log2(0) = -inf unclipped at the upper end.
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def raig_score(p, eps):
    return Hb(p * (1 - eps) + (1 - p) * eps) - Hb(eps)


# ---------------------------------------------------------------------------
# Method specification: selection rule + (assumed) hat-epsilon + allowed set
# ---------------------------------------------------------------------------
class Method:
    def __init__(self, name, select, hat_eps_fn, allowed_tiers=("objective", "semi", "subjective")):
        self.name = name
        self.select = select          # 'random' | 'ig' | 'raig' | 'capacity'
        self.hat_eps_fn = hat_eps_fn  # cat, true_eps_vec -> hat_eps_vec (Q,)
        self.allowed_tiers = allowed_tiers


def make_methods(cat: Catalogue, uniform_eps: float):
    tiered_eps = cat.eps_vector({a: TIER_EPS[t] for a, t in ATTR_TIER.items()})
    zero_eps = np.zeros(cat.Q, dtype=np.float32)
    uniform_eps_vec = np.full(cat.Q, uniform_eps, dtype=np.float32)

    methods = [
        Method("Random+soft", "random", lambda true_eps: uniform_eps_vec),
        Method("IG+hard", "ig", lambda true_eps: zero_eps),
        Method("IG+soft-uniform", "ig", lambda true_eps: uniform_eps_vec),
        Method("IG+soft-objective", "ig", lambda true_eps: uniform_eps_vec,
               allowed_tiers=("objective",)),
        Method("RAIG-tiered", "raig", lambda true_eps: tiered_eps),
        Method("RAIG-oracle", "raig", lambda true_eps: true_eps),
    ]
    for m in methods:
        m.allowed = cat.allowed_mask(m.allowed_tiers)
    return methods


# ---------------------------------------------------------------------------
# Paired session batch: all methods share target + noise stream (Algo 1-4)
#
# Vectorised across an entire batch of trials at once (not just across the M
# methods): belief is a single (N, M*R_batch) matrix, laid out method-major
# (columns [m*R_batch : (m+1)*R_batch] belong to method m across the batch).
# This turns the per-turn cost into ONE (Q,N)x(N, M*R_batch) matmul plus a
# handful of elementwise ops, instead of R_batch separate small matmuls --
# essential for BLAS to reach useful throughput on this shape.
# ---------------------------------------------------------------------------
def run_paired_trials(cat: Catalogue, methods, true_eps_vec, R, T, rng,
                       track_top5=False, time_selection=False, batch_size=150):
    M = len(methods)
    N, Q = cat.N, cat.Q
    X = cat.X                          # (N, Q) float32
    XT = np.ascontiguousarray(X.T)     # (Q, N) float32, contiguous for fast matmul
    true_eps_vec = true_eps_vec.astype(np.float32)

    HAT = np.stack([m.hat_eps_fn(true_eps_vec) for m in methods], axis=0).astype(np.float32)  # (M,Q)
    ALLOWED = np.stack([m.allowed for m in methods], axis=0)  # (M,Q) bool

    correct = np.zeros((R, M), dtype=bool)
    top5 = np.zeros((R, M), dtype=bool) if track_top5 else None
    sel_time_total = 0.0
    sel_time_calls = 0

    all_targets = rng.integers(0, N, size=R)
    offset = 0
    while offset < R:
        Rb = min(batch_size, R - offset)
        targets = all_targets[offset:offset + Rb]
        MC = M * Rb
        xc_batch = X[targets, :]                      # (Rb, Q)
        r_of_col = np.tile(np.arange(Rb), M)           # (MC,) method-major layout

        HAT_full = np.repeat(HAT, Rb, axis=0).T.copy()        # (Q, MC)  -- see note below
        ALLOWED_full = np.repeat(ALLOWED, Rb, axis=0).T.copy()  # (Q, MC)
        # np.repeat(HAT, Rb, axis=0) gives shape (M*Rb, Q) with method-major
        # row order already matching r_of_col's column order; transpose to (Q,MC).

        B = np.full((N, MC), 1.0 / N, dtype=np.float32)
        asked = np.zeros((Q, MC), dtype=bool)
        u_stream = rng.random((T, Rb)).astype(np.float32)      # shared within a trial's 6 methods
        rand_stream = rng.integers(0, 1 << 31, size=(T, Rb))   # private draws for Random+soft

        for t in range(T):
            t0 = time.perf_counter() if time_selection else None
            P = XT @ B  # (Q, MC)  -- the one big matmul per turn

            q_idx = np.empty(MC, dtype=np.int64)
            for m, meth in enumerate(methods):
                cols = slice(m * Rb, (m + 1) * Rb)
                avail = ALLOWED_full[:, cols] & ~asked[:, cols]           # (Q, Rb)
                empty_cols = ~avail.any(axis=0)
                if empty_cols.any():
                    avail[:, empty_cols] = ~asked[:, cols][:, empty_cols]  # fallback: reuse pool
                if meth.select == "random":
                    # uniform random choice among available questions, per column
                    rnd = np.random.default_rng(int(rand_stream[t, 0]) + m * 7919 + t)
                    scores = rnd.random(avail.shape).astype(np.float32)
                    scores = np.where(avail, scores, -1.0)
                    q_idx[cols] = np.argmax(scores, axis=0)
                else:
                    if meth.select == "ig":
                        s = Hb(P[:, cols])
                    elif meth.select == "raig":
                        s = raig_score(P[:, cols], HAT_full[:, cols])
                    else:  # 'capacity' ablation: reliability only, ignores split mass
                        s = np.broadcast_to((1.0 - Hb(HAT_full[:, cols]))[:, :Rb], avail.shape)
                    s = np.where(avail, s, -np.inf)
                    q_idx[cols] = np.argmax(s, axis=0)
            if time_selection:
                sel_time_total += time.perf_counter() - t0
                sel_time_calls += 1

            Xsel = X[:, q_idx]                          # (N, MC)
            y_true = xc_batch[r_of_col, q_idx]           # (MC,)
            eps_true_sel = true_eps_vec[q_idx]           # (MC,)
            u_t_col = u_stream[t][r_of_col]              # (MC,)
            flip = (u_t_col < eps_true_sel).astype(np.float32)
            y_obs = (y_true.astype(np.float32) + flip) % 2

            hat_sel = HAT_full[q_idx, np.arange(MC)]     # (MC,)
            match = (Xsel == y_obs[None, :])
            lik = np.where(match, 1 - hat_sel[None, :], hat_sel[None, :]).astype(np.float32)
            Bnew = B * lik
            Z = Bnew.sum(axis=0)
            safeZ = np.where(Z > 0, Z, 1.0)
            Bupdated = Bnew / safeZ[None, :]
            B = np.where((Z > 0)[None, :], Bupdated, B)

            asked[q_idx, np.arange(MC)] = True

        pred = np.argmax(B, axis=0)                     # (MC,)
        pred = pred.reshape(M, Rb)
        tgt_row = targets[None, :]                       # (1, Rb)
        correct[offset:offset + Rb, :] = (pred == tgt_row).T

        if track_top5:
            top5idx = np.argpartition(-B, 5, axis=0)[:5, :]   # (5, MC)
            top5idx = top5idx.reshape(5, M, Rb)
            hit = (top5idx == tgt_row[None, :, :]).any(axis=0)  # (M, Rb)
            top5[offset:offset + Rb, :] = hit.T

        offset += Rb

    out = {"correct": correct, "top5": top5}
    if time_selection and sel_time_calls > 0:
        out["ms_per_selection_call"] = 1000.0 * sel_time_total / sel_time_calls
    return out


# ---------------------------------------------------------------------------
# Noise-condition construction
# ---------------------------------------------------------------------------
def tiered_true_eps(cat: Catalogue, lam: float) -> np.ndarray:
    base = cat.eps_vector({a: TIER_EPS[t] for a, t in ATTR_TIER.items()})
    return (lam * base).astype(np.float32)


def homogeneous_true_eps(cat: Catalogue, eta: float) -> np.ndarray:
    return np.full(cat.Q, eta, dtype=np.float32)


def randomised_true_eps(cat: Catalogue, lam: float = 1.0, seed: int = 999) -> np.ndarray:
    """Permute the *attribute-level* true-epsilon assignment (not per question)."""
    rng = np.random.default_rng(seed)
    attrs = [a for a in FEATURES if a != "release_type"]
    vals = np.array([lam * TIER_EPS[ATTR_TIER[a]] for a in attrs])
    perm = rng.permutation(len(attrs))
    shuffled = dict(zip(attrs, vals[perm]))
    shuffled["release_type"] = 0.0
    return cat.eps_vector(shuffled).astype(np.float32)


def adversarial_true_eps(cat: Catalogue, lam: float = 1.0) -> np.ndarray:
    """Invert the natural correlation: highest max-IG attribute -> lowest eps."""
    max_ig = per_attribute_max_ig(cat)
    attrs = [a for a in FEATURES if a != "release_type"]
    order = sorted(attrs, key=lambda a: max_ig[a], reverse=True)  # best-splitting first
    vals = sorted([lam * TIER_EPS[ATTR_TIER[a]] for a in attrs])  # ascending eps
    assign = dict(zip(order, vals))
    assign["release_type"] = 0.0
    return cat.eps_vector(assign).astype(np.float32)


def per_attribute_max_ig(cat: Catalogue) -> dict:
    p = cat.X.mean(axis=0)  # split mass at uniform belief
    ig = Hb(p)
    out = {}
    for f in cat.features:
        idx = cat.feature_qidx[f]
        out[f] = float(ig[idx].max()) if len(idx) else 0.0
    return out


# ---------------------------------------------------------------------------
# measure_pathology: per-attribute max IG vs. annotated reliability
# ---------------------------------------------------------------------------
def measure_pathology(cat: Catalogue):
    max_ig = per_attribute_max_ig(cat)
    rows = []
    for f in cat.features:
        if len(cat.feature_qidx[f]) == 0:
            continue
        tier = ATTR_TIER[f]
        eps = TIER_EPS[tier]
        rows.append({"attribute": f, "tier": tier, "eps": eps,
                      "reliability": 1 - eps, "max_ig": max_ig[f]})
    tab = pd.DataFrame(rows)
    rho, pval = spearmanr(tab["max_ig"], tab["reliability"])
    return tab, rho, pval


# ---------------------------------------------------------------------------
# Identifiability audit
# ---------------------------------------------------------------------------
def identifiability_audit(df, features):
    n_before = len(df)
    dfc = df.dropna(subset=features).reset_index(drop=True)
    dropped = n_before - len(dfc)
    groups = dfc.groupby(features, observed=True).size()
    distinct = len(groups)
    collision_rate = groups[groups > 1].sum() / len(dfc)          # fraction of ITEMS in a non-singleton class
    unique_frac = (groups == 1).sum() / len(dfc)                  # fraction of ITEMS uniquely identifiable
    largest_class = int(groups.max())
    entropy_floor = math.log2(distinct)
    max_capacity = 1 - Hb(np.array([min(TIER_EPS.values())]))[0]
    budget_bound = entropy_floor / max_capacity
    return {
        "N": len(dfc), "dropped": dropped, "M": len(features),
        "distinct_vectors": distinct,
        "collision_rate": collision_rate,
        "uniquely_identifiable_frac": unique_frac,
        "largest_equivalence_class": largest_class,
        "entropy_floor_bits": entropy_floor,
        "budget_bound_T": budget_bound,
    }


# ---------------------------------------------------------------------------
# Paired statistical inference
# ---------------------------------------------------------------------------
def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray):
    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    n = n10 + n01
    if n == 0:
        return n10, n01, 0.0, 1.0
    k = min(n10, n01)
    p = binomtest(k, n, 0.5, alternative="two-sided").pvalue
    delta = (correct_b.mean() - correct_a.mean())
    return n10, n01, delta, p


def paired_bootstrap_ci(correct_a: np.ndarray, correct_b: np.ndarray, resamples=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    diffs = correct_b.astype(float) - correct_a.astype(float)
    idx = rng.integers(0, n, size=(resamples, n))
    boot = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return diffs.mean(), lo, hi


def acc_ci(correct: np.ndarray, resamples=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(correct)
    idx = rng.integers(0, n, size=(resamples, n))
    boot = correct.astype(float)[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return correct.mean(), lo, hi
