"""
Trace diagnostics for the two qualitative claims in Section VII-F that had no
saved artefact:

  (a) "Ten individual belief traces show monotonically increasing mass on the
       true target with no oscillation."  (Clean-condition collapse)
  (b) "Ten traced trials confirm ... the selector's argmax walks a fixed
       traversal order identically in every trial regardless of belief state,
       and each such question splits only about 1% of remaining mass."
       (Capacity-only ablation)

Re-implements the single-trial loop from raig.run_paired_trials so the
per-turn belief and per-turn selected question can be recorded, which the
batched routine does not expose.

Writes results/trace_diagnostics.json.
"""
import json
import numpy as np
import raig

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
X, XT, N, Q = cat.X, np.ascontiguousarray(cat.X.T), cat.N, cat.Q
tiered_hat = cat.eps_vector({a: raig.TIER_EPS[t] for a, t in raig.ATTR_TIER.items()})

R_TRACE, T = 10, 20


def trace(select, hat_eps, true_eps, seed):
    """One session; returns (belief-on-target per turn, selected q per turn,
    split mass of each selected question at selection time)."""
    rng = np.random.default_rng(seed)
    target = int(rng.integers(0, N))
    b = np.full(N, 1.0 / N, dtype=np.float32)
    asked = np.zeros(Q, dtype=bool)
    mass, qs, splits = [], [], []
    for t in range(T):
        p = XT @ b
        if select == "raig":
            s = raig.raig_score(p, hat_eps)
        else:                                    # capacity-only
            s = 1.0 - raig.Hb(hat_eps)
        s = np.where(~asked, s, -np.inf)
        q = int(np.argmax(s))
        qs.append(q)
        splits.append(float(min(p[q], 1 - p[q])))
        y = X[target, q]
        if rng.random() < true_eps[q]:
            y = 1 - y
        lik = np.where(X[:, q] == y, 1 - hat_eps[q], hat_eps[q]).astype(np.float32)
        bn = b * lik
        z = bn.sum()
        if z > 0:
            b = bn / z
        asked[q] = True
        mass.append(float(b[target]))
    return mass, qs, splits


# ---- (a) clean-condition belief traces, RAIG-tiered ------------------------
clean_eps = raig.tiered_true_eps(cat, 0.0)
traces, mono, drops = [], 0, []
for i in range(R_TRACE):
    mass, _, _ = trace("raig", tiered_hat, clean_eps, seed=5000 + i)
    traces.append(mass)
    d = [j for j in range(1, len(mass)) if mass[j] < mass[j - 1] - 1e-12]
    if not d:
        mono += 1
    else:
        drops.append({"trace": i, "turns_with_decrease": d})
print("(a) clean belief traces, RAIG-tiered")
print("    monotonically non-decreasing: %d/%d" % (mono, R_TRACE))
if drops:
    print("    traces with a decrease:", drops)
print("    final mass on target: min=%.4g median=%.4g max=%.4g"
      % (min(t[-1] for t in traces),
         float(np.median([t[-1] for t in traces])),
         max(t[-1] for t in traces)))

# ---- (b) capacity-only traversal, heterogeneous lambda=1 -------------------
noisy_eps = raig.tiered_true_eps(cat, 1.0)
seqs, all_splits = [], []
for i in range(R_TRACE):
    _, qs, splits = trace("capacity", tiered_hat, noisy_eps, seed=6000 + i)
    seqs.append(qs)
    all_splits.extend(splits)
identical = all(s == seqs[0] for s in seqs)
attrs = [cat.attr_of_q[q] for q in seqs[0]]
vals = [str(cat.val_of_q[q]) for q in seqs[0]]
print("\n(b) capacity-only traversal")
print("    identical question sequence across all %d traces: %s" % (R_TRACE, identical))
print("    mean split mass of selected questions: %.4f (%.2f%% of remaining)"
      % (float(np.mean(all_splits)), 100 * float(np.mean(all_splits))))
print("    median split mass: %.4f" % float(np.median(all_splits)))
print("    first 8 selected (attribute = value):")
for a, v in list(zip(attrs, vals))[:8]:
    print("      %-22s = %s" % (a, v))

out = {
    "n_traces": R_TRACE, "T": T,
    "clean_belief_traces": {
        "monotone_non_decreasing": mono,
        "traces_with_decrease": drops,
        "final_mass": [t[-1] for t in traces],
        "traces": traces,
    },
    "capacity_only": {
        "identical_sequence_across_traces": bool(identical),
        "mean_split_mass": float(np.mean(all_splits)),
        "median_split_mass": float(np.median(all_splits)),
        "sequence_attributes": attrs,
        "sequence_values": vals,
    },
}
with open("results/trace_diagnostics.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote results/trace_diagnostics.json")
