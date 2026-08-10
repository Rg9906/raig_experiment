"""
Table mcnemar: exact paired McNemar tests at heterogeneous lambda=1.0 for the
three comparisons that matter most, using the paired per-trial correctness
matrix saved by run_main.py (results/correct_lambda1.npy, shape (R*3, 6),
columns ordered as in results/main_results.json -> method_names).
"""
import json
import numpy as np
import raig

with open("results/main_results.json") as f:
    meta = json.load(f)
method_names = meta["method_names"]
idx = {n: i for i, n in enumerate(method_names)}

correct = np.load("results/correct_lambda1.npy")  # (R*3, M) bool

pairs = [
    ("RAIG-tiered", "IG+soft-uniform"),
    ("RAIG-tiered", "IG+soft-objective"),
    ("RAIG-tiered", "RAIG-oracle"),
]

rows = []
for a, b in pairs:
    ca = correct[:, idx[a]]
    cb = correct[:, idx[b]]
    n10, n01, _, p = raig.mcnemar_exact(ca, cb)
    # display delta as A - B (first-named minus second-named), matching the
    # paper's "Comparison: A vs B" column order; mcnemar_exact's own delta is
    # B - A, so it is not reused here.
    delta = float(ca.mean() - cb.mean())
    rows.append({"a": a, "b": b, "n10": n10, "n01": n01, "delta": delta, "p": p})
    print(f"{a} vs {b}: n10={n10} n01={n01} delta(A-B)={delta:+.4f} p={p:.4g}")

with open("results/mcnemar_lambda1.json", "w") as f:
    json.dump(rows, f, indent=2)
print("saved results/mcnemar_lambda1.json")
