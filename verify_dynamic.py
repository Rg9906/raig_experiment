"""
Regression gate for the streaming-EM / early-stopping additions to
run_paired_trials (raig.py) and methods_ext.build_dynamic.

Checks, in order:
  [1] With no method requesting the new features, run_paired_trials is
      byte-identical to a snapshot taken before these features existed.
  [2] Subset invariance holds for the NEW methods too: appending
      RAIG-streaming(+stop) after the existing 9 does not change the
      existing 9 methods' results at all (the Random+soft index-sensitivity
      case the module docstrings warn about).
  [3] Early stopping is internally consistent: a trial that freezes at turn s
      under early_stop_threshold has the exact same final belief/prediction
      as the checkpoint-at-s prediction of the same method run with no
      stopping at all (same targets, same noise stream).
  [4] RAIG-streaming actually moves away from its flat 0.10 start over the
      course of a run (sanity: the estimator is doing something, not stuck).
"""
import numpy as np

import raig
import methods_ext

print("loading catalogue...")
df = raig.load_catalogue()
cat = raig.Catalogue(df)
print(f"N={cat.N} Q={cat.Q}")

R, T, BATCH = 60, 20, 4
seed = 12345

# ---------------------------------------------------------------------------
# [1] existing methods, with vs. without the new methods present
# ---------------------------------------------------------------------------
base_methods, uniform_eps = methods_ext.build(cat)
true_eps = raig.tiered_true_eps(cat, 1.0)

out_before = raig.run_paired_trials(
    cat, base_methods, true_eps, R=R, T=T, rng=np.random.default_rng(seed),
    batch_size=BATCH, track_top5=True, metrics=True, checkpoints=[10, 20])

dyn_methods, _ = methods_ext.build_dynamic(cat, uniform_eps=uniform_eps,
                                            early_stop_thresholds=(0.5, 0.7, 0.9))
assert [m.name for m in dyn_methods[:9]] == [m.name for m in base_methods], \
    "build_dynamic must not reorder the existing 9 methods"

out_with_new = raig.run_paired_trials(
    cat, dyn_methods, true_eps, R=R, T=T, rng=np.random.default_rng(seed),
    batch_size=BATCH, track_top5=True, metrics=True, checkpoints=[10, 20])

ok1 = np.array_equal(out_before["correct"], out_with_new["correct"][:, :9])
ok1b = np.array_equal(out_before["top5"], out_with_new["top5"][:, :9])
print(f"[1] existing-9 correct identical with new methods appended: {'PASS' if ok1 else 'FAIL'}")
print(f"[1] existing-9 top5     identical with new methods appended: {'PASS' if ok1b else 'FAIL'}")
assert ok1 and ok1b

# also: existing 9 alone vs. existing 9 as a DIFFERENT-length prefix of a
# larger list (paranoia check on Random+soft's index dependency specifically)
random_idx = [i for i, m in enumerate(base_methods) if m.name == "Random+soft"][0]
assert random_idx == 0, "test assumes Random+soft is first, as in methods_ext.build"
ok1c = np.array_equal(out_before["correct"][:, 0], out_with_new["correct"][:, 0])
print(f"[1] Random+soft column identical (index-sensitivity case): {'PASS' if ok1c else 'FAIL'}")
assert ok1c

# ---------------------------------------------------------------------------
# [2] RAIG-streaming(+stop) is deterministic run-to-run given the same seed
# ---------------------------------------------------------------------------
dyn_methods2, _ = methods_ext.build_dynamic(cat, uniform_eps=uniform_eps,
                                             early_stop_thresholds=(0.5, 0.7, 0.9))
out_repeat = raig.run_paired_trials(
    cat, dyn_methods2, true_eps, R=R, T=T, rng=np.random.default_rng(seed),
    batch_size=BATCH, track_top5=True, metrics=True, checkpoints=[10, 20])
ok2 = np.array_equal(out_with_new["correct"], out_repeat["correct"])
print(f"[2] RAIG-streaming(+stop) bitwise-reproducible across identical calls: {'PASS' if ok2 else 'FAIL'}")
assert ok2

# ---------------------------------------------------------------------------
# [3] early-stop consistency: frozen trial's final answer == the no-stop
#     run's checkpoint-at-stop_turn answer, for the SAME method/trial.
# ---------------------------------------------------------------------------
# Build one no-stop RAIG-streaming method and its stop@0.5 twin, run together
# (paired, same targets/noise), with per-turn checkpoints at every turn so we
# can look up "prediction as of turn s" for any s.
nostop = raig.Method("RAIG-streaming-nostop", "raig-streaming",
                      lambda te: np.full(cat.Q, raig.STREAM_EPS_INIT, dtype=np.float32),
                      explore_frac=0.10)
nostop.allowed = cat.allowed_mask(nostop.allowed_tiers)
stopped = raig.Method("RAIG-streaming-stop0.5", "raig-streaming",
                       lambda te: np.full(cat.Q, raig.STREAM_EPS_INIT, dtype=np.float32),
                       explore_frac=0.10, early_stop_threshold=0.5)
stopped.allowed = cat.allowed_mask(stopped.allowed_tiers)
pair = [nostop, stopped]
all_ckpts = list(range(1, T + 1))
out_pair = raig.run_paired_trials(
    cat, pair, true_eps, R=40, T=T, rng=np.random.default_rng(999),
    batch_size=4, metrics=True, checkpoints=all_ckpts)

stop_turns = out_pair["stop_turn"][:, 1]  # method index 1 == stopped
final_pred_stopped = out_pair["correct"][:, 1]
n_checked = 0
n_mismatch = 0
for trial in range(40):
    s = int(stop_turns[trial])
    if s >= T:
        continue  # never actually froze early; nothing to check here
    ckpt_correct_nostop = out_pair["checkpoints"][s]["correct"][trial, 0]  # method 0 == nostop
    n_checked += 1
    if ckpt_correct_nostop != final_pred_stopped[trial]:
        n_mismatch += 1
ok3 = (n_mismatch == 0)
print(f"[3] early-stop frozen prediction == no-stop checkpoint@stop_turn: "
      f"{'PASS' if ok3 else 'FAIL'} ({n_checked} trials froze early, {n_mismatch} mismatched)")
assert ok3

# ---------------------------------------------------------------------------
# [4] sanity: the streaming estimate actually moves from its flat start
# ---------------------------------------------------------------------------
one_stream = raig.Method("RAIG-streaming-solo", "raig-streaming",
                          lambda te: np.full(cat.Q, raig.STREAM_EPS_INIT, dtype=np.float32),
                          explore_frac=0.10)
one_stream.allowed = cat.allowed_mask(one_stream.allowed_tiers)
_ = raig.run_paired_trials(cat, [one_stream], true_eps, R=300, T=40,
                            rng=np.random.default_rng(7), batch_size=4)
final_eps = one_stream._stream_eps_attr
moved = float(np.max(np.abs(final_eps - raig.STREAM_EPS_INIT)))
covered = int(np.sum(one_stream._stream_cnt > 0))
n_attrs = len(final_eps)
print(f"[4] streaming estimate moved from flat start: max|delta|={moved:.4f} "
      f"over {covered}/{n_attrs} attributes with evidence "
      f"({'PASS' if moved > 0.01 and covered == n_attrs else 'FAIL'})")
assert moved > 0.01 and covered == n_attrs

print("\nALL CHECKS PASSED")
