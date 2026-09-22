"""
Reference implementation for "Information Gain Prefers the Questions Users
Cannot Answer: Reliability-Aware Query Selection for Interactive Identification".

Implements Algorithms 1-4 and the analysis functions (measure_pathology,
identifiability audit, mcnemar, paired bootstrap CI) against the real
Music-Akenator catalogue (dataset_final.csv, ~79.8k songs x 21 attributes)
and, via datasets.py, against the UCI Mushroom and Dermatology catalogues and
a synthetic catalogue family.

DETERMINISM. The BLAS thread count is pinned to 1 below, BEFORE numpy is
imported. This is not a performance tweak, it is a correctness requirement.
A multi-threaded matmul sums (Q,N)x(N,MC) in a shape-dependent order, so
P = XT @ B differs in the last float32 ulp between runs with different batch
shapes. Selection is an argmax over P, near-ties are common in a catalogue
with many equal-mass questions, and a single different question at turn 1
sends the whole session down a different path. Pinning to one thread makes
run_paired_trials bitwise reproducible and -- verified empirically -- makes
the per-method results of a run with M methods identical to those of a run
with any subset of them, which is what licenses adding a new method to a
comparison without re-running the incumbents. On the reference machine it is
also ~1.6x faster: the matmul is memory-bound and the threads were contending.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
# Streaming (online, population-level) reliability estimation -- see
# run_paired_trials's "raig-streaming" select mode. Same constants
# run_em_eps.py uses for the identical, offline, per-attribute M-step;
# duplicated here (not imported) so raig.py stays the module everything else
# imports FROM rather than acquiring a dependency on a standalone script.
# ---------------------------------------------------------------------------
STREAM_EPS_INIT = 0.10
STREAM_EPS_CLIP = (0.005, 0.45)
# Bayesian shrinkage pseudo-count toward STREAM_EPS_INIT for the M-step below:
# an attribute's live estimate only departs from the neutral prior once it has
# accumulated real evidence, which is what keeps a rarely-asked attribute's
# noisy early reading from swinging to the eps ceiling and getting crowded out
# even further (see run_dynamic_T70's diagnosis of exactly that failure mode).
STREAM_PRIOR_PSEUDO_N = 60.0


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
        self.attr_tier = attr_tier
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
        # tier of the attribute each compiled question probes; used by the
        # allowed-set baselines and by the tier-mix reporting of Sec. VIII-B.
        self.tier_of_q = np.array([attr_tier[a] for a in self.attr_of_q])

    def eps_vector(self, per_attr_eps: dict) -> np.ndarray:
        return np.array([per_attr_eps[a] for a in self.attr_of_q], dtype=np.float32)

    def allowed_mask(self, tiers_allowed) -> np.ndarray:
        return np.isin(self.tier_of_q, list(tiers_allowed))


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
    def __init__(self, name, select, hat_eps_fn, allowed_tiers=("objective", "semi", "subjective"),
                 scorer=None, hat_delta=None, early_stop_threshold=None, explore_frac=0.0):
        self.name = name
        # 'random' | 'ig' | 'raig' | 'raig-streaming' | 'capacity' | 'learned'
        self.select = select
        self.hat_eps_fn = hat_eps_fn  # true_eps_vec -> hat_eps_vec (Q,)
        self.allowed_tiers = allowed_tiers
        # 'learned': scorer(P_cols, hat_cols, t) -> (Q, Rb) scores. Used by the
        # learned-policy baselines of Sec. VII-C (see learned.py).
        self.scorer = scorer
        # assumed per-question erasure probability, for the abstention channel
        # of Sec. V-E. None means the method ignores abstention in selection.
        self.hat_delta = hat_delta
        # once a trial's belief max crosses this, that trial's belief/asked
        # mask stop changing for the rest of the run (a real early exit).
        # None (default) means the method always runs the full T-turn budget,
        # exactly as before this option existed.
        self.early_stop_threshold = early_stop_threshold
        # 'raig-streaming' only: fraction of turns forced to a uniformly
        # random available question, so an attribute the running estimate
        # currently distrusts still keeps accumulating evidence to correct
        # that estimate (mirrors run_em_eps.py's own EXPLORE fix for the
        # identical problem in its offline setting). 0.0 (default) disables
        # it and draws no extra randomness for any method that leaves it off.
        self.explore_frac = explore_frac


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
def _snapshot_metrics(B, tgt_of_col, M, Rb):
    """Rank-based metrics at one checkpoint.

    Returns (M, Rb) arrays. Top-1 uses argmax (ties broken by lowest index,
    matching the legacy behaviour); rank uses the mid-rank convention
    1 + #{b > b*} + (#{b == b*} - 1)/2, so that the many exact ties a belief
    vector carries early in a session are not scored optimistically.
    """
    MC = B.shape[1]
    cols = np.arange(MC)
    bt = B[tgt_of_col, cols]                       # (MC,) belief on the target
    pred = np.argmax(B, axis=0)
    greater = (B > bt[None, :]).sum(axis=0)
    equal = (B == bt[None, :]).sum(axis=0)
    rank = greater + 1.0 + (equal - 1.0) / 2.0
    return {
        "correct": (pred == tgt_of_col).reshape(M, Rb),
        "rank": rank.reshape(M, Rb),
        "top5": (rank <= 5).reshape(M, Rb),
        "top10": (rank <= 10).reshape(M, Rb),
        "nll": (-np.log2(np.clip(bt, 1e-30, None))).reshape(M, Rb),
    }


def run_paired_trials(cat: Catalogue, methods, true_eps_vec, R, T, rng,
                       track_top5=False, time_selection=False, batch_size=150,
                       checkpoints=None, log_questions=False, metrics=False,
                       true_delta_vec=None):
    """Run every method on the same targets and the same noise stream.

    The optional arguments all default to the original behaviour and, when
    left off, consume the random stream in exactly the original order, so
    results stay comparable with earlier runs.

      checkpoints     turn counts at which to snapshot metrics, so that one
                      T=60 pass replaces six separate runs (Sec. VII-D).
      log_questions   also return the selected question index per
                      (trial, method, turn), which is all the asked-question
                      reliability table of Sec. VII-B needs.
      metrics         compute rank / top-5 / top-10 / NLL at each checkpoint.
      true_delta_vec  per-question TRUE abstention probability. When supplied,
                      the user declines to answer with that probability, the
                      question is consumed, and no belief update occurs: the
                      erasure channel of Sec. V-E.

    Two opt-in, additive extensions. Both are no-ops -- provably, see
    verify_harness.py -- for every method that doesn't request them, so every
    existing result in results/*.json remains reproducible unchanged:

      Method.early_stop_threshold  once a trial's belief max crosses this,
                      that trial's belief and asked-mask stop changing for
                      the rest of the run: a real early exit, not merely a
                      logged "would have stopped here" statistic. Reported
                      per (trial, method) as out["stop_turn"] (== T when the
                      method never opts in, or never crosses).
      Method.select == "raig-streaming"  the assumed per-ATTRIBUTE epsilon is
                      not a fixed hat_eps_fn output; it is re-estimated
                      BETWEEN batches from the batches already run in this
                      same call, via the identical attribute-pooled expected-
                      disagreement M-step run_em_eps.py uses offline over 600
                      logged sessions, applied here online across this call's
                      own paired-trial batches instead. Method.explore_frac
                      forces a fraction of turns to a random available
                      question so a currently-distrusted attribute keeps
                      accumulating evidence to correct that estimate.
    """
    M = len(methods)
    N, Q = cat.N, cat.Q
    X = cat.X                          # (N, Q) float32
    XT = np.ascontiguousarray(X.T)     # (Q, N) float32, contiguous for fast matmul
    true_eps_vec = true_eps_vec.astype(np.float32)
    if checkpoints is None:
        checkpoints = [T]
    checkpoints = sorted(set(int(c) for c in checkpoints))
    assert checkpoints[-1] <= T, "checkpoint beyond the budget"

    HAT = np.stack([m.hat_eps_fn(true_eps_vec) for m in methods], axis=0).astype(np.float32)  # (M,Q)
    ALLOWED = np.stack([m.allowed for m in methods], axis=0)  # (M,Q) bool
    THRESH = np.array([m.early_stop_threshold if m.early_stop_threshold is not None
                        else np.inf for m in methods], dtype=np.float64)  # (M,)

    # Streaming-EM state: per-attribute (dis, cnt) accumulators, reset fresh
    # at the top of this call so one call is self-contained and reproducible
    # on its own. Uses cat.attr_of_q directly -- no change to Catalogue needed.
    attrs_present = sorted(set(cat.attr_of_q.tolist()))
    attr_index = {a: i for i, a in enumerate(attrs_present)}
    n_attrs = len(attrs_present)
    q_attr_idx = np.array([attr_index[a] for a in cat.attr_of_q], dtype=np.int64)  # (Q,)
    streaming_idx = [i for i, m in enumerate(methods) if m.select == "raig-streaming"]
    for i in streaming_idx:
        methods[i]._stream_dis = np.zeros(n_attrs, dtype=np.float64)
        methods[i]._stream_cnt = np.zeros(n_attrs, dtype=np.float64)
        methods[i]._stream_eps_attr = np.full(n_attrs, STREAM_EPS_INIT, dtype=np.float64)

    correct = np.zeros((R, M), dtype=bool)
    top5 = np.zeros((R, M), dtype=bool) if track_top5 else None
    qlog = np.zeros((R, M, T), dtype=np.int32) if log_questions else None
    stop_turn_out = np.zeros((R, M), dtype=np.int32)
    ckpt = None
    if metrics:
        ckpt = {c: {k: np.zeros((R, M),
                                dtype=(bool if k in ("correct", "top5", "top10") else np.float64))
                    for k in ("correct", "rank", "top5", "top10", "nll")}
                for c in checkpoints}
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
        tgt_of_col = targets[r_of_col]                 # (MC,)

        HAT_full = np.repeat(HAT, Rb, axis=0).T.copy()        # (Q, MC)  -- see note below
        ALLOWED_full = np.repeat(ALLOWED, Rb, axis=0).T.copy()  # (Q, MC)
        THRESH_full = np.repeat(THRESH, Rb)                     # (MC,)
        # np.repeat(HAT, Rb, axis=0) gives shape (M*Rb, Q) with method-major
        # row order already matching r_of_col column order; transpose to (Q,MC).

        # Streaming methods: overwrite this batch's slice of HAT_full with the
        # population estimate accumulated from EARLIER batches only -- this
        # batch's own turns feed the estimate the NEXT batch will read, not
        # itself. Keeps the update out of the per-turn critical path and
        # keeps the RNG-neutrality argument for every other method trivial.
        for i in streaming_idx:
            cols = slice(i * Rb, (i + 1) * Rb)
            eps_per_q = methods[i]._stream_eps_attr[q_attr_idx]  # (Q,)
            HAT_full[:, cols] = eps_per_q[:, None]

        B = np.full((N, MC), 1.0 / N, dtype=np.float32)
        asked = np.zeros((Q, MC), dtype=bool)
        active = np.ones(MC, dtype=bool)
        stop_turn = np.zeros(MC, dtype=np.int32)
        u_stream = rng.random((T, Rb)).astype(np.float32)      # shared across a trial methods
        rand_stream = rng.integers(0, 1 << 31, size=(T, Rb))   # private draws for Random+soft
        # Abstention draws come from their own request, made only when the
        # erasure channel is active, so the default path consumes the random
        # stream in exactly the order earlier runs did.
        a_stream = rng.random((T, Rb)).astype(np.float32) if true_delta_vec is not None else None

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
                    elif meth.select in ("raig", "raig-streaming"):
                        s = raig_score(P[:, cols], HAT_full[:, cols])
                    elif meth.select == "learned":
                        s = meth.scorer(P[:, cols], HAT_full[:, cols], t)
                    else:  # 'capacity' ablation: reliability only, ignores split mass
                        s = np.broadcast_to((1.0 - Hb(HAT_full[:, cols]))[:, :Rb], avail.shape)
                    if meth.hat_delta is not None:
                        # abstention discounts a question by the odds it is
                        # answered at all (Sec. V-E).
                        s = s * (1.0 - meth.hat_delta)[:, None]
                    s = np.where(avail, s, -np.inf)
                    chosen = np.argmax(s, axis=0)
                    if meth.explore_frac:
                        # Forced exploration so an attribute the running
                        # estimate currently distrusts still keeps
                        # accumulating evidence to correct itself (mirrors
                        # run_em_eps.py's own EXPLORE fix for the identical
                        # problem offline). Seeded from the shared stream's
                        # VALUE into a private generator -- reads it, does not
                        # consume its position -- so no other method's draws
                        # are perturbed by this method being present at all.
                        rnd = np.random.default_rng(int(rand_stream[t, 0]) + m * 104729 + t)
                        force = rnd.random(Rb) < meth.explore_frac
                        if force.any():
                            rnd2 = np.random.default_rng(int(rand_stream[t, 0]) + m * 104729 + t + 999983)
                            rscores = rnd2.random(avail.shape).astype(np.float32)
                            rscores = np.where(avail, rscores, -1.0)
                            chosen = np.where(force, np.argmax(rscores, axis=0), chosen)
                    q_idx[cols] = chosen
            if time_selection:
                sel_time_total += time.perf_counter() - t0
                sel_time_calls += 1

            if log_questions:
                qlog[offset:offset + Rb, :, t] = q_idx.reshape(M, Rb).T

            Xsel = X[:, q_idx]                          # (N, MC)
            y_true = xc_batch[r_of_col, q_idx]           # (MC,)
            eps_true_sel = true_eps_vec[q_idx]           # (MC,)
            u_t_col = u_stream[t][r_of_col]              # (MC,)
            flip = (u_t_col < eps_true_sel).astype(np.float32)
            y_obs = (y_true.astype(np.float32) + flip) % 2

            hat_sel = HAT_full[q_idx, np.arange(MC)]     # (MC,)
            p_sel = P[q_idx, np.arange(MC)]              # (MC,) split mass at the asked question
            match = (Xsel == y_obs[None, :])
            lik = np.where(match, 1 - hat_sel[None, :], hat_sel[None, :]).astype(np.float32)
            if true_delta_vec is not None:
                # An abstained question yields a flat likelihood, i.e. no update.
                abstain = a_stream[t][r_of_col] < true_delta_vec[q_idx]   # (MC,)
                lik = np.where(abstain[None, :], np.float32(1.0), lik)
            Bnew = B * lik
            Z = Bnew.sum(axis=0)
            safeZ = np.where(Z > 0, Z, 1.0)
            Bupdated = Bnew / safeZ[None, :]
            B = np.where(((Z > 0) & active)[None, :], Bupdated, B)

            active_idx = np.where(active)[0]
            asked[q_idx[active_idx], active_idx] = True

            # Streaming-EM bookkeeping: fold this turn's (attribute, expected
            # disagreement) into the running per-attribute accumulators, for
            # still-active columns only -- a frozen (early-stopped) trial is a
            # finished interaction and contributes no further evidence.
            for i in streaming_idx:
                cols = slice(i * Rb, (i + 1) * Rb)
                col_ids = np.arange(i * Rb, (i + 1) * Rb)
                live = active[col_ids]
                if not live.any():
                    continue
                q_sel = q_idx[cols][live]
                y_sel = y_obs[cols][live]
                p_sel_i = p_sel[cols][live]
                d = np.where(y_sel == 1, 1.0 - p_sel_i, p_sel_i).astype(np.float64)
                a_sel = q_attr_idx[q_sel]
                np.add.at(methods[i]._stream_dis, a_sel, d)
                np.add.at(methods[i]._stream_cnt, a_sel, 1.0)

            newly = active & (B.max(axis=0) >= THRESH_full)
            stop_turn = np.where(newly, t + 1, stop_turn)
            active = active & ~newly

            if metrics and (t + 1) in ckpt:
                snap = _snapshot_metrics(B, tgt_of_col, M, Rb)
                for k, v in snap.items():
                    ckpt[t + 1][k][offset:offset + Rb, :] = v.T

        # Refresh each streaming method's population estimate for the NEXT
        # batch from everything accumulated through the end of THIS one.
        for i in streaming_idx:
            m_i = methods[i]
            # Bayesian-shrinkage M-step: blend toward the neutral prior with
            # pseudo-count STREAM_PRIOR_PSEUDO_N, so an attribute's estimate
            # moves only as fast as its OWN evidence justifies -- a handful of
            # noisy early observations can no longer swing it to the ceiling.
            blended = ((STREAM_PRIOR_PSEUDO_N * STREAM_EPS_INIT + m_i._stream_dis) /
                       (STREAM_PRIOR_PSEUDO_N + m_i._stream_cnt))
            m_i._stream_eps_attr = np.clip(blended, STREAM_EPS_CLIP[0], STREAM_EPS_CLIP[1])

        pred = np.argmax(B, axis=0)                     # (MC,)
        pred = pred.reshape(M, Rb)
        tgt_row = targets[None, :]                       # (1, Rb)
        correct[offset:offset + Rb, :] = (pred == tgt_row).T
        stop_turn_reported = np.where(stop_turn == 0, T, stop_turn)  # 0 == "never crossed" -> ran full T
        stop_turn_out[offset:offset + Rb, :] = stop_turn_reported.reshape(M, Rb).T

        if track_top5:
            top5idx = np.argpartition(-B, 5, axis=0)[:5, :]   # (5, MC)
            top5idx = top5idx.reshape(5, M, Rb)
            hit = (top5idx == tgt_row[None, :, :]).any(axis=0)  # (M, Rb)
            top5[offset:offset + Rb, :] = hit.T

        offset += Rb

    out = {"correct": correct, "top5": top5, "targets": all_targets, "stop_turn": stop_turn_out}
    if log_questions:
        out["qlog"] = qlog
    if metrics:
        out["checkpoints"] = ckpt
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


def holm_bonferroni(pvals, alpha=0.05):
    """Holm step-down adjusted p-values, for the McNemar family of Sec. VII.

    Returns (adjusted, rejected). Adjusted values are enforced monotone, so a
    later hypothesis can never carry a smaller adjusted p than an earlier one.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj, adj <= alpha
