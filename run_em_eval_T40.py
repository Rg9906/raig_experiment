"""
Re-evaluate the LEARNED crossover vector at the deployment budget (Sec. VIII-B).

run_em_eps.py both estimates the vector and evaluates it, at T=20. The estimate
is a property of the logs and does not depend on the evaluation budget, so this
script reuses the vector already stored in results/em_eps.json and only redoes
the evaluation at T=40, against the same incumbent, annotation and oracle.

Estimation and evaluation are deliberately kept at different budgets here, and
that is the realistic arrangement rather than a shortcut: the logs come from a
system that was already running at whatever budget it used, and the question is
whether the vector recovered from them is good enough for the budget you intend
to ship.

Writes results/em_eval_T40.json.
"""
import json
import time

import numpy as np

import raig

OUT = "results"
R, T, SEEDS, BATCH = 300, 40, [0, 1, 2], 30
EXPLORE_KEY = "0.0"          # the no-exploration estimate: best at T=20

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)
true_eps = raig.tiered_true_eps(cat, 1.0)
tier_eps = cat.eps_vector({a: raig.TIER_EPS[t] for a, t in raig.ATTR_TIER.items()})
uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
unif = np.full(cat.Q, uniform_eps, dtype=np.float32)

em = json.load(open(f"{OUT}/em_eps.json"))["results"][EXPLORE_KEY]
learned_by_attr = em["eps_hat_by_attr"]
# release_type compiles to no questions and is absent from the estimate
full = {a: learned_by_attr.get(a, uniform_eps) for a in raig.FEATURES}
learned_vec = cat.eps_vector(full)
print(f"learned vector: mean={learned_vec.mean():.4f} "
      f"(tiered mean={tier_eps.mean():.4f}), explore={EXPLORE_KEY}", flush=True)

ms = [
    raig.Method("IG+soft-uniform", "ig", lambda te: unif),
    raig.Method("RAIG-EM", "raig", lambda te: learned_vec),
    raig.Method("RAIG-tiered", "raig", lambda te: tier_eps),
    raig.Method("RAIG-oracle", "raig", lambda te: te),
]
for m in ms:
    m.allowed = cat.allowed_mask(m.allowed_tiers)
names = [m.name for m in ms]

t0 = time.time()
allc = []
for s in SEEDS:
    out = raig.run_paired_trials(cat, ms, true_eps, R=R, T=T,
                                 rng=np.random.default_rng(1000 + s),
                                 batch_size=BATCH, track_top5=True)
    allc.append(out["correct"])
    print(f"  seed {s} done ({time.time()-t0:.0f}s)", flush=True)
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

print("\n" + ", ".join(f"{n}={cell[n]['acc']*100:.2f}" for n in names))
for k, v in cell.items():
    if k.startswith("_"):
        print(f"  {k}: delta={v['delta']*100:+.2f}pp n10={v['n10']} n01={v['n01']} p={v['p']:.3g}")

json.dump({"R": R, "T": T, "seeds": SEEDS, "batch_size": BATCH,
           "n_total": R * len(SEEDS), "explore_key": EXPLORE_KEY,
           "learned_mean_eps": float(learned_vec.mean()),
           "accuracy": cell},
          open(f"{OUT}/em_eval_T40.json", "w"), indent=2)
print(f"\nwrote {OUT}/em_eval_T40.json ({time.time()-t0:.0f}s)")
