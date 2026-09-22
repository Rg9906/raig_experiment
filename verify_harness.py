"""
Gate test for the extended harness (Sec. VI-E, reproducibility).

Checks, in order:
  1. run_paired_trials is bitwise reproducible across two identical calls.
  2. It is METHOD-SUBSET INVARIANT: running a subset of the methods with the
     same seed/R/T/batch_size gives those methods bitwise-identical outcomes.
     This is what licenses adding a new method to an existing comparison
     without re-running the incumbents, and it only holds single-threaded.
  3. The new optional arguments (checkpoints / log_questions / metrics) do not
     perturb the random stream: with them on, `correct` is unchanged.
  4. The checkpoint at t=T agrees with the end-of-session `correct`.
  5. Results are statistically consistent with the stored multi-threaded run
     (results/correct_lambda1.npy). They will NOT match bitwise -- that is the
     point of the determinism fix -- so this is a two-proportion sanity band,
     not an equality test.

Exits non-zero if any check fails.
"""
import sys
import numpy as np
import raig

FAIL = []


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(label)


df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
methods = raig.make_methods(cat, uniform_eps)
names = [m.name for m in methods]
true_eps = raig.tiered_true_eps(cat, 1.0)

R, T, BATCH = 150, 20, 30
print(f"N={cat.N} Q={cat.Q} R={R} T={T} batch={BATCH}")

print("\n[1] bitwise reproducibility across identical calls")
a = raig.run_paired_trials(cat, methods, true_eps, R=R, T=T,
                           rng=np.random.default_rng(1000), batch_size=BATCH)
b = raig.run_paired_trials(cat, methods, true_eps, R=R, T=T,
                           rng=np.random.default_rng(1000), batch_size=BATCH)
check("identical correctness matrices", np.array_equal(a["correct"], b["correct"]))

print("\n[2] method-subset invariance")
keep = ("IG+soft-uniform", "RAIG-tiered")
sub = [m for m in methods if m.name in keep]
s = raig.run_paired_trials(cat, sub, true_eps, R=R, T=T,
                           rng=np.random.default_rng(1000), batch_size=BATCH)
for j, nm in enumerate([m.name for m in sub]):
    check(f"{nm} identical when run in a {len(sub)}-method list",
          np.array_equal(s["correct"][:, j], a["correct"][:, names.index(nm)]))

print("\n[3] optional arguments do not perturb the random stream")
c = raig.run_paired_trials(cat, methods, true_eps, R=R, T=T,
                           rng=np.random.default_rng(1000), batch_size=BATCH,
                           checkpoints=[5, 10, 20], log_questions=True, metrics=True,
                           track_top5=True)
check("correct unchanged with checkpoints/log/metrics on",
      np.array_equal(c["correct"], a["correct"]))
check("qlog shape", c["qlog"].shape == (R, len(methods), T), str(c["qlog"].shape))

print("\n[4] checkpoint at t=T agrees with end-of-session result")
check("checkpoint[20].correct == correct",
      np.array_equal(c["checkpoints"][20]["correct"], a["correct"]))
r = c["checkpoints"][20]["rank"]
# rank == 1 under the mid-rank convention means the target is the UNIQUE
# maximum, which forces argmax to pick it. The converse does not hold: argmax
# also lands on the target when it is tied at the top, and mid-rank scores
# those > 1. So the invariant is one-directional, and the gap between the two
# counts is exactly the number of tie-won predictions.
implies = bool(np.all(a["correct"][r == 1.0]))
n_tied_wins = int(a["correct"].sum() - (r == 1.0).sum())
check("rank==1 implies correct", implies,
      f"tie-won top-1 predictions: {n_tied_wins}")

print("\n[5] consistency with the stored multi-threaded run")
ref = np.load("results/correct_lambda1.npy")
for nm in ("IG+soft-uniform", "RAIG-tiered", "RAIG-oracle"):
    new = a["correct"][:, names.index(nm)].mean()
    old = ref[:, names.index(nm)].mean()
    se = np.sqrt(new * (1 - new) / R + old * (1 - old) / len(ref)) + 1e-9
    z = abs(new - old) / se
    check(f"{nm}: new={new:.4f} old={old:.4f}", z < 3.0, f"z={z:.2f}")

print("\n[6] Proposition (uniform specification is inert), empirically")
# The paper asserts that RAIG with a single global hat-epsilon selects exactly
# what classical IG selects, so it must reproduce IG+soft-uniform TRIAL FOR
# TRIAL and not merely in aggregate. Aggregate agreement would be far weaker
# evidence: two selectors can reach the same accuracy by different routes.
unif_vec = np.full(cat.Q, uniform_eps, dtype=np.float32)
probe = [
    raig.Method("IG+soft-uniform", "ig", lambda te: unif_vec),
    raig.Method("RAIG-uniform", "raig", lambda te: unif_vec),
]
for m in probe:
    m.allowed = cat.allowed_mask(m.allowed_tiers)
p = raig.run_paired_trials(cat, probe, true_eps, R=R, T=T,
                           rng=np.random.default_rng(1000), batch_size=BATCH,
                           log_questions=True)
same_q = np.array_equal(p["qlog"][:, 0, :], p["qlog"][:, 1, :])
same_c = np.array_equal(p["correct"][:, 0], p["correct"][:, 1])
check("RAIG-uniform asks the same questions as IG+soft-uniform", same_q,
      f"differing selections: {int((p['qlog'][:, 0, :] != p['qlog'][:, 1, :]).sum())}"
      f" of {R * T}")
check("RAIG-uniform matches IG+soft-uniform trial for trial", same_c,
      f"acc {p['correct'][:, 0].mean():.4f} vs {p['correct'][:, 1].mean():.4f}")

print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} CHECK(S) FAILED: {FAIL}"))
sys.exit(1 if FAIL else 0)
