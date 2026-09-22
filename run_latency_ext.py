"""
Cost of the correction, and behaviour under confidence-threshold stopping
(paper Sec. VII-J).

Two questions a practitioner asks before adopting this. First: what does the
correction cost per turn? Second: under a stopping rule rather than a fixed
budget, does it reach confidence sooner?

On cost, the honest answer is available analytically and the measurement can
only confirm it. Both selectors run the same (|Q|,N) x (N,Rb) matmul; RAIG then
adds one elementwise affine map and one subtraction on a vector of length |Q|,
which is O(|Q|) against the matmul's O(N|Q|), a ratio of 1/N ~ 1.3e-5 here. Any
measured difference larger than that is machine noise, and on a shared laptop
machine noise is large. We therefore report the median over REPEATS independent
runs and state the analytic ratio alongside it, rather than presenting a single
timing as a benchmark.

On stopping: the belief must reach max_i b_i >= THRESH. A selector that spends
its turns on questions the user answers badly accumulates contradictory
evidence and its posterior stays flat, so the fraction of sessions that ever
reach the threshold is itself a measure of the bias, and is reported.

Writes results/latency_ext.json.
"""
import json
import statistics
import time
import numpy as np

import raig

OUT = "results"
R, T, SEEDS, BATCH = 300, 20, [0, 1, 2], 30      # identical to run_main_ext.py
THRESH = 0.90
REPEATS = 5
NAMES = ["IG+soft-uniform", "RAIG-tiered"]

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
by_name = {m.name: m for m in raig.make_methods(cat, uniform_eps)}
true_eps = raig.tiered_true_eps(cat, 1.0)
XT = np.ascontiguousarray(cat.X.T)


def run(method, seed):
    N, Q, X = cat.N, cat.Q, cat.X
    hat = method.hat_eps_fn(true_eps).astype(np.float32)
    rng = np.random.default_rng(seed)
    correct = np.zeros(R, dtype=bool)
    top5 = np.zeros(R, dtype=bool)
    q_thresh = np.full(R, T, dtype=np.int32)
    reached = np.zeros(R, dtype=bool)
    sel_t, sel_n = 0.0, 0
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
        loc_reached = np.zeros(Rb, dtype=bool)
        loc_q = np.full(Rb, T, dtype=np.int32)

        for t in range(T):
            t0 = time.perf_counter()
            P = XT @ B
            avail = ALLOWED & ~asked
            empty = ~avail.any(axis=0)
            if empty.any():
                avail[:, empty] = ~asked[:, empty]
            s = raig.Hb(P) if method.select == "ig" else raig.raig_score(P, HAT)
            q_idx = np.argmax(np.where(avail, s, -np.inf), axis=0)
            sel_t += time.perf_counter() - t0
            sel_n += 1

            y_true = xc[np.arange(Rb), q_idx]
            flip = (u[t] < true_eps[q_idx]).astype(np.float32)
            y_obs = (y_true.astype(np.float32) + flip) % 2
            hat_sel = HAT[q_idx, np.arange(Rb)]
            match = (X[:, q_idx] == y_obs[None, :])
            lik = np.where(match, 1 - hat_sel[None, :], hat_sel[None, :]).astype(np.float32)
            Bn = B * lik
            Z = Bn.sum(axis=0)
            B = np.where((Z > 0)[None, :], Bn / np.where(Z > 0, Z, 1.0)[None, :], B)
            asked[q_idx, np.arange(Rb)] = True

            newly = (~loc_reached) & (B.max(axis=0) >= THRESH)
            loc_q[newly] = t + 1
            loc_reached |= newly

        correct[off:off + Rb] = (np.argmax(B, axis=0) == tgt)
        t5 = np.argpartition(-B, 5, axis=0)[:5, :]
        top5[off:off + Rb] = (t5 == tgt[None, :]).any(axis=0)
        q_thresh[off:off + Rb] = loc_q
        reached[off:off + Rb] = loc_reached
        off += Rb

    return {"correct": correct, "top5": top5, "q_thresh": q_thresh,
            "reached": reached, "ms_per_turn": 1000.0 * sel_t / sel_n}


# Methods are INTERLEAVED within each repetition, not run in blocks. A first
# attempt ran all five repetitions of one method and then all five of the
# other, and produced timings for the second method that rose monotonically
# across identical repetitions (14.9, 51.8, 90.2, 116.1, 145.8 ms/turn). A
# tenfold monotone drift across repeated runs of identical work is the machine
# degrading -- thermal or memory pressure on a shared laptop -- not a property
# of the method, and blocking by method assigns all of that drift to whichever
# method runs last. Interleaving makes the drift common to both, and the
# per-repetition RATIO below is the paired statistic that survives it.
ms_by_rep = {n: [] for n in NAMES}
outcome = {}
for rep in range(REPEATS):
    for name in NAMES:
        ms, acc_c, acc_5, qt, rc = [], [], [], [], []
        for s in SEEDS:
            out = run(by_name[name], 1000 + s)
            ms.append(out["ms_per_turn"])
            if rep == 0:            # accuracy is deterministic given the seed
                acc_c.append(out["correct"]); acc_5.append(out["top5"])
                qt.append(out["q_thresh"]); rc.append(out["reached"])
        ms_by_rep[name].append(float(np.mean(ms)))
        if rep == 0:
            outcome[name] = (np.concatenate(acc_c), np.concatenate(acc_5),
                             np.concatenate(qt), np.concatenate(rc))
    print("  rep %d/%d: " % (rep + 1, REPEATS) +
          "  ".join(f"{n}={ms_by_rep[n][-1]:7.2f}" for n in NAMES) +
          f"   ratio={ms_by_rep[NAMES[1]][-1] / ms_by_rep[NAMES[0]][-1]:.3f}",
          flush=True)

rows = {}
for name in NAMES:
    per_rep_ms = ms_by_rep[name]
    c, t5, qth, rch = outcome[name]
    m, lo, hi = raig.acc_ci(c)
    rows[name] = {
        "ms_per_turn_runs": per_rep_ms,
        "ms_per_turn_median": float(statistics.median(per_rep_ms)),
        "ms_per_turn_min": float(min(per_rep_ms)),
        "ms_per_turn_max": float(max(per_rep_ms)),
        "acc": float(m), "lo": float(lo), "hi": float(hi),
        "top5": float(t5.mean()),
        "frac_reached_thresh": float(rch.mean()),
        "mean_q_to_thresh_given_reached": (float(qth[rch].mean()) if rch.any() else None),
        "n_total": int(len(c)),
    }
    print(f"{name:18s} median={rows[name]['ms_per_turn_median']:.2f} ms/turn  "
          f"acc={m*100:.2f}%  reached={rch.mean()*100:.1f}%", flush=True)

# Paired ratio, repetition by repetition. This is the figure to quote: it is
# invariant to any drift that scales both methods together.
ratios = [ms_by_rep[NAMES[1]][i] / ms_by_rep[NAMES[0]][i] for i in range(REPEATS)]
print("\nper-repetition RAIG/IG time ratio: " +
      ", ".join(f"{r:.3f}" for r in ratios) +
      f"   median={statistics.median(ratios):.3f}")

analytic = {
    "N": int(cat.N), "Q": int(cat.Q),
    "extra_flops_ratio_raig_over_ig": 1.0 / float(cat.N),
    "note": ("RAIG adds O(|Q|) elementwise work to an O(N|Q|) matmul, so the "
             "expected overhead is 1/N. Measured differences above that are "
             "machine noise on a shared machine, not method cost."),
}
json.dump({"R": R, "T": T, "seeds": SEEDS, "batch_size": BATCH,
           "threshold": THRESH, "repeats": REPEATS, "interleaved": True,
           "ratio_runs": ratios,
           "ratio_median": float(statistics.median(ratios)),
           "ratio_min": float(min(ratios)), "ratio_max": float(max(ratios)),
           "analytic": analytic, "methods": rows},
          open(f"{OUT}/latency_ext.json", "w"), indent=2)
print(f"\nwrote {OUT}/latency_ext.json")
