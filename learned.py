"""
Learned question-selection policy (paper Sec. VII-C).

The applied literature on conversational recommendation (EAR, CPR, UNICORN)
learns which attribute to ask next from simulated interaction rather than
computing it. The paper claims such a policy inherits the reliability blind
spot from its training signal whenever that signal is generated without an
answer-noise model. That is an empirical claim about a family of systems, so
it needs a member of the family actually run, not just cited.

This is a myopic value-regression policy in the EAR/CPR mould: roll out a
behaviour policy, log (state, question) features against the realised value
of having asked that question, fit a regressor, then select by the learned
score. Two variants, differing ONLY in the noise used to generate training
rollouts:

  Learned-noiseblind   trained on noiseless rollouts. This is the modelling
                       choice the paper attributes to prior work.
  Learned-noiseaware   trained on rollouts under the true heterogeneous
                       noise. It can in principle discover the reliability
                       weighting from data.

FAIRNESS OF THE BASELINE. The feature set deliberately includes the
interaction between split balance and attribute identity (Hb(p) x attribute
one-hot). A linear model over those features CAN represent a per-attribute
discount on balanced splits, which is precisely what RAIG encodes. Omitting
the interaction would have guaranteed the baseline could not compete, and any
resulting win would have been an artefact of the feature design rather than a
finding. Whether the noise-aware variant actually recovers the weighting is
then a real question, and the answer is reported either way.

The regressor is ridge, solved in closed form with NumPy: the point is to
represent the family faithfully, and a deeper model would add capacity the
comparison does not turn on while adding a dependency the project avoids.
"""
import numpy as np

import raig


class PolicyFeatures:
    """Builds the (Q, Rb, D) feature tensor lazily, as a sum of separable parts.

    Scoring must be cheap: it happens once per method per turn. Every feature
    is therefore either question-only, column-only, or a pointwise function of
    the split mass, so the score is assembled with broadcasts instead of
    materialising the full tensor.
    """

    def __init__(self, cat: raig.Catalogue):
        self.cat = cat
        self.attrs = list(cat.features)
        self.A = np.zeros((cat.Q, len(self.attrs)), dtype=np.float32)
        for j, a in enumerate(self.attrs):
            self.A[cat.feature_qidx[a], j] = 1.0
        self.prevalence = cat.X.mean(axis=0).astype(np.float32)   # (Q,)
        self.n_attr = len(self.attrs)
        # 4 split terms + attr one-hot + attr x Hb + prevalence + 3 state + bias
        self.D = 4 + self.n_attr + self.n_attr + 1 + 3 + 1

    def build_rows(self, qidx, p, t_frac, Hb_belief, supp):
        """Feature matrix for a set of (question, column) samples.

        Used at training time, where the samples are the asked questions only.
        """
        n = len(qidx)
        h = raig.Hb(p).astype(np.float32)
        F = np.zeros((n, self.D), dtype=np.float32)
        F[:, 0] = p
        F[:, 1] = np.abs(p - 0.5)
        F[:, 2] = h
        F[:, 3] = h * h
        k = 4
        F[:, k:k + self.n_attr] = self.A[qidx]
        k += self.n_attr
        F[:, k:k + self.n_attr] = self.A[qidx] * h[:, None]
        k += self.n_attr
        F[:, k] = self.prevalence[qidx]; k += 1
        F[:, k] = t_frac; k += 1
        F[:, k] = Hb_belief; k += 1
        F[:, k] = supp; k += 1
        F[:, k] = 1.0
        return F

    def make_scorer(self, w, T):
        """Return scorer(P_cols, hat_cols, t) -> (Q, Rb) scores.

        Assembled by broadcasting rather than by building (Q, Rb, D).
        """
        A, prev, nA = self.A, self.prevalence, self.n_attr
        w = np.asarray(w, dtype=np.float32)
        k = 4
        w_attr = w[k:k + nA]; k += nA
        w_attr_h = w[k:k + nA]; k += nA
        w_prev = w[k]; k += 1
        w_t = w[k]; k += 1
        w_He = w[k]; k += 1
        w_supp = w[k]; k += 1
        w_bias = w[k]
        # question-only part, constant across columns within a turn
        q_const = (A @ w_attr) + w_prev * prev + w_bias          # (Q,)
        q_hcoef = (A @ w_attr_h)                                  # (Q,)

        def scorer(P_cols, hat_cols, t, _cache={}):
            h = raig.Hb(P_cols).astype(np.float32)                # (Q, Rb)
            s = (w[0] * P_cols + w[1] * np.abs(P_cols - 0.5)
                 + w[2] * h + w[3] * h * h
                 + q_const[:, None] + q_hcoef[:, None] * h)
            # column-only state terms are constant down a column and so cannot
            # change an argmax taken over questions; they are omitted here and
            # retained at training time, where they do affect the fit.
            return s

        return scorer


def collect_rollouts(cat, feat, true_eps_vec, n_sessions, T, rng,
                     explore=0.35, hat_eps=None):
    """Roll out an epsilon-greedy-over-IG behaviour policy and log values.

    The regression target is the REALISED reduction in belief entropy
    produced by the answer actually received. It is computable by a deployed
    system from its own logs -- it does not reference the hidden target -- so
    a practitioner could fit this policy from real traffic, which is what
    makes it a fair stand-in for the learned-policy family.
    """
    N, Q = cat.N, cat.Q
    X, XT = cat.X, np.ascontiguousarray(cat.X.T)
    if hat_eps is None:
        hat_eps = np.full(Q, 0.098, dtype=np.float32)
    F_all, y_all = [], []
    for _ in range(n_sessions):
        target = int(rng.integers(0, N))
        b = np.full(N, 1.0 / N, dtype=np.float32)
        asked = np.zeros(Q, dtype=bool)
        for t in range(T):
            p = (XT @ b).astype(np.float32)
            if rng.random() < explore:
                cand = np.flatnonzero(~asked)
                q = int(cand[rng.integers(len(cand))])
            else:
                s = np.where(~asked, raig.Hb(p), -np.inf)
                q = int(np.argmax(s))
            bb = np.clip(b, 1e-30, None)
            H_before = float(-(bb * np.log2(bb)).sum())
            supp = float((b > 1e-9).mean())
            F_all.append(feat.build_rows(np.array([q]), p[q:q + 1],
                                         t / T, H_before / np.log2(N), supp))
            y = X[target, q]
            if rng.random() < true_eps_vec[q]:
                y = 1 - y
            lik = np.where(X[:, q] == y, 1 - hat_eps[q], hat_eps[q]).astype(np.float32)
            bn = b * lik
            z = bn.sum()
            if z > 0:
                b = bn / z
            bb = np.clip(b, 1e-30, None)
            H_after = float(-(bb * np.log2(bb)).sum())
            y_all.append(H_before - H_after)      # realised bits gained
            asked[q] = True
    return np.concatenate(F_all, axis=0), np.asarray(y_all, dtype=np.float32)


def fit_ridge(F, y, lam=1.0):
    """Closed-form ridge with the intercept column left unpenalised."""
    D = F.shape[1]
    A = F.T @ F + lam * np.eye(D, dtype=np.float64)
    A[-1, -1] -= lam
    return np.linalg.solve(A, F.T @ y.astype(np.float64))
