"""
Lightweight stress test for Limitation #3 (errors assumed independent across
turns). Correlated-error variant: once a user gets a SUBJECTIVE attribute
wrong for the first time in a session, subsequent questions about that same
attribute within that session are forced wrong too (deterministic flip),
rather than independently re-drawn at rate eps. Objective/semi-objective
attributes are unaffected (matches the task's "once a user gets a subjective
attribute wrong" framing -- these are the attributes where genuine conceptual
confusion, not simple mistakes, is plausible).

Run at heterogeneous lambda=1.0, matching the paper's primary comparison.
"""
import json
import time
import numpy as np
import raig

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
methods_all = raig.make_methods(cat, uniform_eps)
by_name = {m.name: m for m in methods_all}

true_eps = raig.tiered_true_eps(cat, 1.0)

SUBJ_ATTRS = sorted([a for a, t in raig.ATTR_TIER.items() if t == "subjective"])
subj_index_of_attr = {a: i for i, a in enumerate(SUBJ_ATTRS)}
n_subj = len(SUBJ_ATTRS)
subj_idx_of_q = np.full(cat.Q, -1, dtype=np.int32)
for qi, a in enumerate(cat.attr_of_q):
    if a in subj_index_of_attr:
        subj_idx_of_q[qi] = subj_index_of_attr[a]

R, T, BATCH = 300, 20, 50
SEEDS = [0, 1, 2]


def run_method_correlated(method, true_eps_vec, R, T, rng, batch_size):
    N, Q = cat.N, cat.Q
    X = cat.X
    XT = np.ascontiguousarray(X.T)
    hat = method.hat_eps_fn(true_eps_vec).astype(np.float32)
    allowed = method.allowed

    correct = np.zeros(R, dtype=bool)
    targets = rng.integers(0, N, size=R)
    offset = 0
    while offset < R:
        Rb = min(batch_size, R - offset)
        tgt = targets[offset:offset + Rb]
        xc = X[tgt, :]
        HAT = np.repeat(hat[:, None], Rb, axis=1)
        ALLOWED = np.repeat(allowed[:, None], Rb, axis=1)

        B = np.full((N, Rb), 1.0 / N, dtype=np.float32)
        asked = np.zeros((Q, Rb), dtype=bool)
        u_stream = rng.random((T, Rb)).astype(np.float32)
        # per-trial, per-subjective-attribute "already erred this session" state
        corrupted = np.zeros((n_subj, Rb), dtype=bool)

        for t in range(T):
            P = XT @ B
            avail = ALLOWED & ~asked
            empty = ~avail.any(axis=0)
            if empty.any():
                avail[:, empty] = ~asked[:, empty]
            if method.select == "ig":
                s = raig.Hb(P)
            else:
                s = raig.raig_score(P, HAT)
            s = np.where(avail, s, -np.inf)
            q_idx = np.argmax(s, axis=0)

            Xsel = X[:, q_idx]
            y_true = xc[np.arange(Rb), q_idx]
            eps_true_sel = true_eps_vec[q_idx]

            subj_sel = subj_idx_of_q[q_idx]                  # (Rb,) subjective-attr index, -1 if not subjective
            is_subj = subj_sel >= 0
            already_corrupted = np.zeros(Rb, dtype=bool)
            if is_subj.any():
                cols = np.where(is_subj)[0]
                already_corrupted[cols] = corrupted[subj_sel[cols], cols]

            # independent draw for non-subjective questions and first-time subjective questions
            indep_flip = (u_stream[t] < eps_true_sel).astype(np.float32)
            # forced flip for subjective questions whose attribute already erred this session
            flip = np.where(already_corrupted, 1.0, indep_flip).astype(np.float32)
            y_obs = (y_true.astype(np.float32) + flip) % 2

            # update corrupted-state: a subjective attribute becomes corrupted the
            # first time its (independent) draw produces an error
            newly_wrong = is_subj & (~already_corrupted) & (indep_flip > 0.5)
            if newly_wrong.any():
                cols = np.where(newly_wrong)[0]
                corrupted[subj_sel[cols], cols] = True

            hat_sel = HAT[q_idx, np.arange(Rb)]
            match = (Xsel == y_obs[None, :])
            lik = np.where(match, 1 - hat_sel[None, :], hat_sel[None, :]).astype(np.float32)
            Bnew = B * lik
            Z = Bnew.sum(axis=0)
            B = np.where((Z > 0)[None, :], Bnew / np.where(Z > 0, Z, 1.0)[None, :], B)
            asked[q_idx, np.arange(Rb)] = True

        pred = np.argmax(B, axis=0)
        correct[offset:offset + Rb] = (pred == tgt)
        offset += Rb

    return correct


results = {}
t0 = time.time()
for name in ["IG+soft-uniform", "RAIG-tiered"]:
    all_correct = []
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        c = run_method_correlated(by_name[name], true_eps, R=R, T=T, rng=rng, batch_size=BATCH)
        all_correct.append(c)
    correct = np.concatenate(all_correct)
    acc = float(correct.mean())
    results[name] = {"acc": acc, "correct": correct.tolist()}
    print(f"[{time.time()-t0:6.0f}s] {name}: correlated-error acc = {acc*100:.2f}%  (n={len(correct)})", flush=True)

# McNemar: RAIG-tiered vs IG+soft-uniform under correlated errors
ca = np.array(results["RAIG-tiered"]["correct"])
cb = np.array(results["IG+soft-uniform"]["correct"])
n10, n01, delta, p = raig.mcnemar_exact(ca, cb)
print(f"McNemar RAIG-tiered vs IG+soft-uniform (correlated): n10={n10} n01={n01} delta(IG-RAIG)={delta*100:+.2f}pp p={p:.4g}")

with open("results/correlated_error.json", "w") as f:
    json.dump({
        "R": R, "T": T, "seeds": SEEDS, "subjective_attrs": SUBJ_ATTRS,
        "acc": {k: v["acc"] for k, v in results.items()},
        "mcnemar_RAIGtiered_vs_IGsoftuniform": {"n10": n10, "n01": n01, "delta_RAIG_minus_IG": -float(delta), "p": float(p)},
    }, f, indent=2)
print("saved results/correlated_error.json")
