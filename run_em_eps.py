"""
Learning the crossover vector from interaction logs, with no annotation, no
ground truth and no user study (paper Sec. V-D, VIII-B).

This is the experiment that answers the standing objection to the whole paper:
that (0.03, 0.15, 0.30) are three static numbers we assumed. Here nothing is
assumed. A deployed system runs the INCUMBENT selector (classical information
gain at a single global crossover, which is what a system that has never heard
of this paper would ship), logs which questions it asked and what the user
answered, and estimates a per-attribute crossover from those logs alone. The
target item is never observed: it is latent, and the estimator marginalises
over it using the system's own posterior.

Estimator. This is Dawid and Skene's repeated-observation argument with the
ITEM as the latent label rather than the annotator's. Write b_d for the
posterior over items after session d, q_{d,t} for the question asked and
y_{d,t} for the answer given. Then

    eps_f  =  sum_{d,t : attr(q)=f}  P[answer disagrees with the item]
              / #{d,t : attr(q)=f}

where the disagreement probability under the current posterior is

    P = 1 - p_q  if y = 1,     P = p_q  if y = 0,     p_q = sum_c b_d(c) x_q(c)

and b_d is itself recomputed from the logs under the current eps. Alternating
these two steps is EM. The fixed point is a self-consistent explanation of the
logs: "a crossover vector under which the answers I actually received are as
likely as possible, given that I do not know what anyone was thinking of".

EXPLORATION MATTERS, and this is a deployment recommendation rather than an
implementation detail. Classical information gain concentrates its questions on
a handful of attributes (Sec. VIII-B), so logs from a greedy policy contain
almost no observations of the attributes it dislikes -- exactly the reliable
ones whose crossover we most want to know. A small fraction of uniformly random
questions fixes this at negligible cost to the sessions in which it fires. We
sweep that fraction.

Writes results/em_eps.json.
"""
import json
import time

import numpy as np

import raig

OUT = "results"
D_SESSIONS = 600          # logged deployment sessions
T = 20
N_EM_ITERS = 8
EXPLORE = [0.0, 0.10, 0.25]
EPS_INIT = 0.10           # neutral start; the estimator is not told the tiers
EPS_CLIP = (0.005, 0.45)
BATCH = 50

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
X = cat.X
XT = np.ascontiguousarray(X.T)
N, Q = cat.N, cat.Q

true_eps = raig.tiered_true_eps(cat, 1.0)
tier_eps = cat.eps_vector({a: raig.TIER_EPS[t] for a, t in raig.ATTR_TIER.items()})
uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
print(f"N={N} Q={Q} uniform_eps={uniform_eps:.4f}  D={D_SESSIONS} T={T}", flush=True)


# ---------------------------------------------------------------------------
# Phase 1: run the incumbent and log what it asked and what it heard.
# ---------------------------------------------------------------------------
def collect_logs(explore, seed):
    """Deploy IG+soft-uniform (the incumbent) and log (question, answer).

    The logs contain no ground truth. `targets` is returned only so that the
    ORACLE diagnostics below can report how well the estimator did; it is never
    passed to the estimator.
    """
    rng = np.random.default_rng(seed)
    hat = np.full(Q, uniform_eps, dtype=np.float32)
    qlog = np.zeros((D_SESSIONS, T), dtype=np.int32)
    ylog = np.zeros((D_SESSIONS, T), dtype=np.int8)
    targets = rng.integers(0, N, size=D_SESSIONS)
    off = 0
    while off < D_SESSIONS:
        Rb = min(BATCH, D_SESSIONS - off)
        tgt = targets[off:off + Rb]
        xc = X[tgt, :]
        B = np.full((N, Rb), 1.0 / N, dtype=np.float32)
        asked = np.zeros((Q, Rb), dtype=bool)
        u = rng.random((T, Rb)).astype(np.float32)
        e = rng.random((T, Rb)).astype(np.float32)          # explore/exploit coin
        r = rng.random((T, Rb, Q)).astype(np.float32)       # random tie-breaks
        for t in range(T):
            P = XT @ B
            avail = ~asked
            s = raig.Hb(P)
            s = np.where(avail, s, -np.inf)
            q_greedy = np.argmax(s, axis=0)
            q_rand = np.argmax(np.where(avail, r[t].T, -1.0), axis=0)
            q_idx = np.where(e[t] < explore, q_rand, q_greedy)

            y_true = xc[np.arange(Rb), q_idx]
            flip = (u[t] < true_eps[q_idx]).astype(np.float32)
            y_obs = ((y_true.astype(np.float32) + flip) % 2).astype(np.int8)
            qlog[off:off + Rb, t] = q_idx
            ylog[off:off + Rb, t] = y_obs

            hat_sel = hat[q_idx]
            match = (X[:, q_idx] == y_obs[None, :].astype(np.float32))
            lik = np.where(match, 1 - hat_sel[None, :], hat_sel[None, :]).astype(np.float32)
            Bn = B * lik
            Z = Bn.sum(axis=0)
            B = np.where((Z > 0)[None, :], Bn / np.where(Z > 0, Z, 1.0)[None, :], B)
            asked[q_idx, np.arange(Rb)] = True
        off += Rb
    return qlog, ylog, targets


# ---------------------------------------------------------------------------
# Phase 2: EM over the logs. The target item is latent throughout.
# ---------------------------------------------------------------------------
def em_estimate(qlog, ylog, n_iter=N_EM_ITERS):
    eps_hat = np.full(Q, EPS_INIT, dtype=np.float32)
    attr_of_q = cat.attr_of_q
    attrs = list(cat.features)
    q_of_attr = {a: np.where(attr_of_q == a)[0] for a in attrs}
    history = []

    for it in range(n_iter):
        # accumulate, per ATTRIBUTE, the expected number of disagreements and
        # the number of observations
        dis = {a: 0.0 for a in attrs}
        cnt = {a: 0 for a in attrs}
        off = 0
        while off < D_SESSIONS:
            Rb = min(BATCH, D_SESSIONS - off)
            qb = qlog[off:off + Rb]              # (Rb, T)
            yb = ylog[off:off + Rb]
            # E-step: posterior over items for this batch of sessions, under
            # the current eps_hat, recomputed from the logs.
            B = np.full((N, Rb), 1.0 / N, dtype=np.float32)
            for t in range(T):
                q_idx = qb[:, t]
                y = yb[:, t].astype(np.float32)
                h = eps_hat[q_idx]
                match = (X[:, q_idx] == y[None, :])
                lik = np.where(match, 1 - h[None, :], h[None, :]).astype(np.float32)
                Bn = B * lik
                Z = Bn.sum(axis=0)
                B = np.where((Z > 0)[None, :], Bn / np.where(Z > 0, Z, 1.0)[None, :], B)
            # M-step contribution: expected disagreement at each logged turn
            P = XT @ B                            # (Q, Rb) split mass under posterior
            for t in range(T):
                q_idx = qb[:, t]
                y = yb[:, t]
                p_q = P[q_idx, np.arange(Rb)]     # mass on x_q = 1
                d = np.where(y == 1, 1.0 - p_q, p_q)
                for a in attrs:
                    m = (attr_of_q[q_idx] == a)
                    if m.any():
                        dis[a] += float(d[m].sum())
                        cnt[a] += int(m.sum())
            off += Rb

        new = eps_hat.copy()
        for a in attrs:
            if cnt[a] > 0:
                v = np.clip(dis[a] / cnt[a], *EPS_CLIP)
                new[q_of_attr[a]] = v
        shift = float(np.abs(new - eps_hat).max())
        eps_hat = new.astype(np.float32)
        by_attr = {a: float(eps_hat[q_of_attr[a]][0]) if len(q_of_attr[a]) else None
                   for a in attrs}
        history.append({"iter": it, "max_shift": shift,
                        "eps_by_attr": by_attr,
                        "obs_by_attr": {a: cnt[a] for a in attrs}})
        print(f"    EM iter {it}: max shift {shift:.4f}", flush=True)
        if shift < 1e-4:
            break
    return eps_hat, history


# ---------------------------------------------------------------------------
# Phase 3: does a system that ships the learned vector beat the incumbent?
# ---------------------------------------------------------------------------
def evaluate(eps_hat_vec, label):
    unif = np.full(Q, uniform_eps, dtype=np.float32)
    ms = [
        raig.Method("IG+soft-uniform", "ig", lambda te: unif),
        raig.Method("RAIG-EM", "raig", lambda te, h=eps_hat_vec: h),
        raig.Method("RAIG-tiered", "raig", lambda te: tier_eps),
        raig.Method("RAIG-oracle", "raig", lambda te: te),
    ]
    for m in ms:
        m.allowed = cat.allowed_mask(m.allowed_tiers)
    names = [m.name for m in ms]
    allc = []
    for s in (0, 1, 2):
        out = raig.run_paired_trials(cat, ms, true_eps, R=300, T=T,
                                     rng=np.random.default_rng(1000 + s),
                                     batch_size=30)
        allc.append(out["correct"])
    c = np.concatenate(allc, axis=0)
    cell = {}
    for i, n in enumerate(names):
        mean, lo, hi = raig.acc_ci(c[:, i])
        cell[n] = {"acc": float(mean), "lo": float(lo), "hi": float(hi)}
    for a, b in (("IG+soft-uniform", "RAIG-EM"), ("RAIG-tiered", "RAIG-EM")):
        n10, n01, _, p = raig.mcnemar_exact(c[:, names.index(a)], c[:, names.index(b)])
        cell[f"_mcnemar_RAIG-EM_vs_{a}"] = {
            "n10": n10, "n01": n01, "p": float(p),
            "delta": float(c[:, names.index(b)].mean() - c[:, names.index(a)].mean())}
    print(f"  [{label}] " + ", ".join(f"{n}={cell[n]['acc']*100:.2f}" for n in names),
          flush=True)
    return cell


results = {}
t0 = time.time()
for explore in EXPLORE:
    print(f"\n=== exploration fraction {explore} ===", flush=True)
    qlog, ylog, targets = collect_logs(explore, seed=7000 + int(explore * 100))
    eps_hat, history = em_estimate(qlog, ylog)

    # Diagnostics against quantities the estimator never saw.
    attrs = [a for a in cat.features if len(cat.feature_qidx[a])]
    hat_by_attr = np.array([eps_hat[cat.feature_qidx[a]][0] for a in attrs])
    true_by_attr = np.array([true_eps[cat.feature_qidx[a]][0] for a in attrs])
    tier_by_attr = np.array([tier_eps[cat.feature_qidx[a]][0] for a in attrs])
    obs = np.array([history[-1]["obs_by_attr"][a] for a in attrs])
    from scipy.stats import spearmanr
    rho, p_rho = spearmanr(hat_by_attr, true_by_attr)
    mae = float(np.abs(hat_by_attr - true_by_attr).mean())

    print(f"  attributes observed at all: {(obs > 0).sum()}/{len(attrs)}"
          f"   min obs {obs.min()}   Spearman(hat, true) = {rho:+.3f} (p={p_rho:.3g})"
          f"   MAE = {mae:.3f}", flush=True)

    cell = evaluate(eps_hat, f"explore={explore}")
    results[str(explore)] = {
        "explore": explore,
        "eps_hat_by_attr": {a: float(v) for a, v in zip(attrs, hat_by_attr)},
        "true_by_attr": {a: float(v) for a, v in zip(attrs, true_by_attr)},
        "tier_by_attr": {a: float(v) for a, v in zip(attrs, tier_by_attr)},
        "obs_by_attr": {a: int(v) for a, v in zip(attrs, obs)},
        "n_attrs_observed": int((obs > 0).sum()),
        "min_obs": int(obs.min()),
        "spearman_hat_vs_true": {"rho": float(rho), "p": float(p_rho)},
        "mae_vs_true": mae,
        "em_iters": len(history),
        "accuracy": cell,
    }
    json.dump({"D_sessions": D_SESSIONS, "T": T, "em_iters": N_EM_ITERS,
               "eps_init": EPS_INIT, "eps_clip": list(EPS_CLIP),
               "results": results},
              open(f"{OUT}/em_eps.json", "w"), indent=2)

print(f"\nwrote {OUT}/em_eps.json ({time.time()-t0:.0f}s)")
