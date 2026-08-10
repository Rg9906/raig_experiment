"""
Table latency: per-turn selection latency (ms), questions-to-confidence-
threshold under threshold stopping, and top-5 accuracy, for IG+soft-uniform
and RAIG-tiered, at heterogeneous lambda=1.0. Confidence threshold fixed at
0.90 (this value was left as a free hyperparameter by the paper; chosen here
and recorded in Table hyper).
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
THRESH = 0.90
R, T, BATCH = 150, 20, 150

def run_with_stopping(method, seed):
    """Mirrors raig.run_paired_trials for a single method, additionally
    tracking the first turn at which max belief crosses THRESH."""
    rng = np.random.default_rng(seed)
    N, Q = cat.N, cat.Q
    X = cat.X
    XT = np.ascontiguousarray(X.T)
    hat = method.hat_eps_fn(true_eps).astype(np.float32)
    allowed = method.allowed

    correct = np.zeros(R, dtype=bool)
    top5 = np.zeros(R, dtype=bool)
    q_to_thresh = np.full(R, T, dtype=np.int32)
    reached = np.zeros(R, dtype=bool)

    sel_time_total, sel_time_calls = 0.0, 0
    targets = rng.integers(0, N, size=R)
    offset = 0
    while offset < R:
        Rb = min(BATCH, R - offset)
        tgt = targets[offset:offset + Rb]
        xc = X[tgt, :]
        HAT = np.repeat(hat[:, None], Rb, axis=1)          # (Q, Rb)
        ALLOWED = np.repeat(allowed[:, None], Rb, axis=1)  # (Q, Rb)

        B = np.full((N, Rb), 1.0 / N, dtype=np.float32)
        asked = np.zeros((Q, Rb), dtype=bool)
        u_stream = rng.random((T, Rb)).astype(np.float32)
        local_reached = np.zeros(Rb, dtype=bool)
        local_qthresh = np.full(Rb, T, dtype=np.int32)

        for t in range(T):
            t0 = time.perf_counter()
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
            sel_time_total += time.perf_counter() - t0
            sel_time_calls += 1

            Xsel = X[:, q_idx]
            y_true = xc[np.arange(Rb), q_idx]
            eps_true_sel = true_eps[q_idx]
            flip = (u_stream[t] < eps_true_sel).astype(np.float32)
            y_obs = (y_true.astype(np.float32) + flip) % 2

            hat_sel = HAT[q_idx, np.arange(Rb)]
            match = (Xsel == y_obs[None, :])
            lik = np.where(match, 1 - hat_sel[None, :], hat_sel[None, :]).astype(np.float32)
            Bnew = B * lik
            Z = Bnew.sum(axis=0)
            safeZ = np.where(Z > 0, Z, 1.0)
            B = np.where((Z > 0)[None, :], Bnew / safeZ[None, :], B)
            asked[q_idx, np.arange(Rb)] = True

            newly = (~local_reached) & (B.max(axis=0) >= THRESH)
            local_qthresh[newly] = t + 1
            local_reached |= newly

        pred = np.argmax(B, axis=0)
        correct[offset:offset + Rb] = (pred == tgt)
        top5idx = np.argpartition(-B, 5, axis=0)[:5, :]
        top5[offset:offset + Rb] = (top5idx == tgt[None, :]).any(axis=0)
        q_to_thresh[offset:offset + Rb] = local_qthresh
        reached[offset:offset + Rb] = local_reached
        offset += Rb

    ms_per_call = 1000.0 * sel_time_total / sel_time_calls
    return {
        "acc": float(correct.mean()),
        "top5": float(top5.mean()),
        "ms_per_turn": ms_per_call,
        "mean_q_to_thresh": float(q_to_thresh.mean()),
        "frac_reached": float(reached.mean()),
    }

rows = {}
for name in ["IG+soft-uniform", "RAIG-tiered"]:
    res = run_with_stopping(by_name[name], seed=4242)
    rows[name] = res
    print(name, res)

with open("results/latency.json", "w") as f:
    json.dump(rows, f, indent=2)
print("saved results/latency.json")
