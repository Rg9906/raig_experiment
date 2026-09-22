"""
The abstention (erasure) channel, implemented and measured (paper Sec. V-E,
VII-I).

The conference version derived the abstention extension analytically and left
it in Future Work, together with the conjecture that because abstention
plausibly correlates with subjectivity in the same direction as error, "the
penalty compounds and the bias should be larger than reported here". That is
a testable prediction sitting in a future-work list, which is the wrong place
for it when the harness can test it in ten minutes.

Model. Beyond flipping an answer with probability epsilon_f, the user declines
to answer at all with probability delta_f. An abstained question is consumed
but produces no belief update. A selector that knows delta discounts each
question by the odds it is answered:

    RAIG_erasure(q) = (1 - delta_f) * [ Hb(p_q * eps_f) - Hb(eps_f) ]

delta is tied to the tier, on the reasoning that a user who cannot judge an
attribute reliably is also the user most likely to decline it.

Compares three selectors under increasing abstention: classical IG (blind to
both), RAIG (models error but not abstention), and RAIG+erasure (models both).

Writes results/erasure.json.
"""
import json
import time
import numpy as np

import raig

OUT = "results"
R, T, SEEDS, BATCH = 150, 20, [0, 1, 2], 30
# abstention rate per tier, at three overall intensities
DELTA_TIERS = {
    "none": {"objective": 0.0, "semi": 0.0, "subjective": 0.0},
    "mild": {"objective": 0.02, "semi": 0.10, "subjective": 0.20},
    "strong": {"objective": 0.05, "semi": 0.20, "subjective": 0.40},
}

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
unif = np.full(cat.Q, uniform_eps, dtype=np.float32)
tiered = cat.eps_vector({a: raig.TIER_EPS[t] for a, t in raig.ATTR_TIER.items()})
true_eps = raig.tiered_true_eps(cat, 1.0)

rows = {}
t0 = time.time()
for level, dmap in DELTA_TIERS.items():
    delta_vec = cat.eps_vector({a: dmap[t] for a, t in raig.ATTR_TIER.items()})
    ms = [
        raig.Method("IG+soft-uniform", "ig", lambda te: unif),
        raig.Method("RAIG-tiered", "raig", lambda te: tiered),
        raig.Method("RAIG+erasure", "raig", lambda te: tiered, hat_delta=delta_vec),
    ]
    for m in ms:
        m.allowed = cat.allowed_mask(m.allowed_tiers)
    names = [m.name for m in ms]
    allc, allq = [], []
    for s in SEEDS:
        out = raig.run_paired_trials(
            cat, ms, true_eps, R=R, T=T, rng=np.random.default_rng(1000 + s),
            batch_size=BATCH, log_questions=True,
            true_delta_vec=(None if level == "none" else delta_vec))
        allc.append(out["correct"]); allq.append(out["qlog"])
    c = np.concatenate(allc, axis=0); q = np.concatenate(allq, axis=0)
    cell = {}
    for i, n in enumerate(names):
        mean, lo, hi = raig.acc_ci(c[:, i])
        qi = q[:, i, :]
        tiers = cat.tier_of_q[qi].ravel()
        cell[n] = {"acc": float(mean), "lo": float(lo), "hi": float(hi),
                   "mean_eps_asked": float(true_eps[qi].mean()),
                   "mean_delta_asked": float(delta_vec[qi].mean()),
                   "expected_answered_frac": float(1 - delta_vec[qi].mean()),
                   "tier_mix": {t: float((tiers == t).mean())
                                for t in ("objective", "semi", "subjective")}}
    for a, b in (("IG+soft-uniform", "RAIG-tiered"),
                 ("RAIG-tiered", "RAIG+erasure")):
        n10, n01, _, p = raig.mcnemar_exact(c[:, names.index(a)], c[:, names.index(b)])
        cell[f"_mcnemar_{b}_vs_{a}"] = {
            "n10": n10, "n01": n01, "p": float(p),
            "delta": float(c[:, names.index(b)].mean() - c[:, names.index(a)].mean())}
    rows[level] = cell
    print(f"[{time.time()-t0:6.0f}s] abstention={level}: " +
          ", ".join(f"{n}={cell[n]['acc']*100:.1f}" for n in names), flush=True)
    for n in names:
        print(f"      {n:18s} eps_asked={cell[n]['mean_eps_asked']:.3f} "
              f"delta_asked={cell[n]['mean_delta_asked']:.3f} "
              f"subj={cell[n]['tier_mix']['subjective']*100:.0f}%", flush=True)
    json.dump({"R": R, "T": T, "seeds": SEEDS, "delta_tiers": DELTA_TIERS,
               "results": rows}, open(f"{OUT}/erasure.json", "w"), indent=2)

print(f"\nwrote {OUT}/erasure.json ({time.time()-t0:.0f}s)")
