"""
Are the results an artefact of assuming answer errors are independent across
turns? (paper Sec. VII-G)

The channel model of Sec. IV treats every answer as an independent draw at rate
epsilon. A real user who misjudges "is it energetic?" is likely to misjudge it
the same way when asked again, and worse, is likely to be systematically wrong
about related perceptual attributes. Independence is therefore the modelling
assumption most likely to be flattering, and it should be attacked directly
rather than listed as a limitation.

Correlated variant. The first time a SUBJECTIVE attribute is answered wrongly
in a session, that attribute is marked corrupted for the rest of the session
and every later question about it is forced wrong. Objective and semi-objective
attributes keep independent draws, since simple slips rather than conceptual
confusion are the plausible failure there. This is a deliberately harsh model:
it turns a rate-epsilon channel into a per-session absorbing one.

Both variants are run inside the same loop and consume the same random stream
in the same order, so the independent and correlated columns are exactly paired
trial for trial, and any difference between them is the correlation and nothing
else. This is what the conference-version script could not do: it ran the
correlated variant alone at a different batch size, leaving the independent
comparison to be read from a run that faced different users.

Writes results/correlated_ext.json.
"""
import json
import time
import numpy as np

import raig

OUT = "results"
R, T, SEEDS, BATCH = 300, 20, [0, 1, 2], 30      # identical to run_main_ext.py
NAMES = ["IG+soft-uniform", "RAIG-tiered"]

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
by_name = {m.name: m for m in raig.make_methods(cat, uniform_eps)}
true_eps = raig.tiered_true_eps(cat, 1.0)

SUBJ = sorted([a for a, t in raig.ATTR_TIER.items() if t == "subjective"])
subj_of_attr = {a: i for i, a in enumerate(SUBJ)}
subj_of_q = np.full(cat.Q, -1, dtype=np.int32)
for qi, a in enumerate(cat.attr_of_q):
    if a in subj_of_attr:
        subj_of_q[qi] = subj_of_attr[a]


def run(method, rng, correlate):
    """One method, R trials. `correlate=False` reproduces the independent
    channel exactly, so the two calls differ only in the flip rule."""
    N, Q, X = cat.N, cat.Q, cat.X
    XT = np.ascontiguousarray(X.T)
    hat = method.hat_eps_fn(true_eps).astype(np.float32)
    correct = np.zeros(R, dtype=bool)
    targets = rng.integers(0, N, size=R)
    off = 0
    while off < R:
        Rb = min(BATCH, R - off)
        tgt = targets[off:off + Rb]
        xc = X[tgt, :]
        HAT = np.repeat(hat[:, None], Rb, axis=1)
        ALLOWED = np.repeat(method.allowed[:, None], Rb, axis=1)
        B = np.full((N, Rb), 1.0 / N, dtype=np.float32)
        asked = np.zeros((Q, Rb), dtype=bool)
        u = rng.random((T, Rb)).astype(np.float32)
        corrupted = np.zeros((len(SUBJ), Rb), dtype=bool)

        for t in range(T):
            P = XT @ B
            avail = ALLOWED & ~asked
            empty = ~avail.any(axis=0)
            if empty.any():
                avail[:, empty] = ~asked[:, empty]
            s = raig.Hb(P) if method.select == "ig" else raig.raig_score(P, HAT)
            q_idx = np.argmax(np.where(avail, s, -np.inf), axis=0)

            y_true = xc[np.arange(Rb), q_idx]
            indep = (u[t] < true_eps[q_idx]).astype(np.float32)
            if correlate:
                si = subj_of_q[q_idx]
                is_subj = si >= 0
                already = np.zeros(Rb, dtype=bool)
                cols = np.where(is_subj)[0]
                if cols.size:
                    already[cols] = corrupted[si[cols], cols]
                flip = np.where(already, 1.0, indep).astype(np.float32)
                new = np.where(is_subj & ~already & (indep > 0.5))[0]
                if new.size:
                    corrupted[si[new], new] = True
            else:
                flip = indep
            y_obs = (y_true.astype(np.float32) + flip) % 2

            hat_sel = HAT[q_idx, np.arange(Rb)]
            match = (X[:, q_idx] == y_obs[None, :])
            lik = np.where(match, 1 - hat_sel[None, :], hat_sel[None, :]).astype(np.float32)
            Bn = B * lik
            Z = Bn.sum(axis=0)
            B = np.where((Z > 0)[None, :], Bn / np.where(Z > 0, Z, 1.0)[None, :], B)
            asked[q_idx, np.arange(Rb)] = True

        correct[off:off + Rb] = (np.argmax(B, axis=0) == tgt)
        off += Rb
    return correct


acc = {}
t0 = time.time()
for variant, corr in (("independent", False), ("correlated", True)):
    acc[variant] = {}
    for name in NAMES:
        c = np.concatenate([run(by_name[name], np.random.default_rng(1000 + s), corr)
                            for s in SEEDS])
        m, lo, hi = raig.acc_ci(c)
        acc[variant][name] = {"acc": float(m), "lo": float(lo), "hi": float(hi),
                              "correct": c.astype(int).tolist()}
        print(f"[{time.time()-t0:6.0f}s] {variant:12s} {name:18s} "
              f"{m*100:5.2f}%  [{lo*100:4.2f},{hi*100:5.2f}]", flush=True)

out = {"R": R, "T": T, "seeds": SEEDS, "batch_size": BATCH,
       "n_total": R * len(SEEDS), "subjective_attrs": SUBJ, "variants": {}}
for variant in acc:
    ca = np.array(acc[variant]["RAIG-tiered"]["correct"], dtype=bool)
    cb = np.array(acc[variant]["IG+soft-uniform"]["correct"], dtype=bool)
    n10, n01, _, p = raig.mcnemar_exact(cb, ca)
    out["variants"][variant] = {
        n: {k: v for k, v in acc[variant][n].items() if k != "correct"} for n in NAMES}
    out["variants"][variant]["mcnemar_RAIG_vs_IG"] = {
        "n10_ig_only": n10, "n01_raig_only": n01,
        "delta": float(ca.mean() - cb.mean()), "p": float(p)}
    print(f"{variant:12s} McNemar RAIG vs IG: delta={out['variants'][variant]['mcnemar_RAIG_vs_IG']['delta']*100:+.2f}pp p={p:.3e}")

# Does the correlation change the answer? Paired within method, across variants.
for n in NAMES:
    ci = np.array(acc["independent"][n]["correct"], dtype=bool)
    cc = np.array(acc["correlated"][n]["correct"], dtype=bool)
    n10, n01, _, p = raig.mcnemar_exact(ci, cc)
    out.setdefault("correlated_vs_independent", {})[n] = {
        "delta": float(cc.mean() - ci.mean()), "n10_indep_only": n10,
        "n01_corr_only": n01, "p": float(p)}
    print(f"{n:18s} correlated - independent = {(cc.mean()-ci.mean())*100:+.2f}pp  p={p:.3e}")

json.dump(out, open(f"{OUT}/correlated_ext.json", "w"), indent=2)
print(f"\nwrote {OUT}/correlated_ext.json ({time.time()-t0:.0f}s)")
