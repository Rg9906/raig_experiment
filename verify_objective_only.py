"""
Verification for the IG+soft-objective near-zero-accuracy explanation in
Section VII-F: restricting to the objective tier leaves an identifiability
ceiling, not a selector failure.

Recomputes, from the catalogue itself:
  - number of distinct attribute combinations over the objective tier only
  - collision rate and largest equivalence class
  - the implied top-1 accuracy ceiling (fraction of items in a singleton class)
  - number of compiled objective-only questions

Writes results/objective_only_audit.json.
"""
import json
import numpy as np
import raig

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)

obj_attrs = [a for a in raig.FEATURES if raig.ATTR_TIER[a] == "objective"]
groups = df.groupby(obj_attrs, observed=True).size()

N = len(df)
distinct = int(len(groups))
largest = int(groups.max())
singletons = int((groups == 1).sum())
collision_rate = float(groups[groups > 1].sum() / N)
ceiling = float(singletons / N)
q_obj = int(cat.allowed_mask(("objective",)).sum())

out = {
    "objective_attributes": obj_attrs,
    "n_objective_attributes": len(obj_attrs),
    "N": N,
    "distinct_combinations": distinct,
    "collision_rate": collision_rate,
    "largest_equivalence_class": largest,
    "top1_accuracy_ceiling": ceiling,
    "objective_only_questions": q_obj,
}
print(json.dumps(out, indent=2))

print("\n--- paper claims (Section VII-F) ---")
def cmp(label, got, want, tol):
    flag = "OK" if abs(got - want) <= tol else "*** MISMATCH ***"
    print("  %-34s computed=%-10s paper=%-10s %s" % (label, round(got, 4), want, flag))

cmp("distinct combinations", distinct, 1926, 0)
cmp("collision rate (%)", collision_rate * 100, 99.35, 0.02)
cmp("largest equivalence class", largest, 932, 0)
cmp("top-1 ceiling (%)", ceiling * 100, 0.65, 0.02)
cmp("objective-only questions", q_obj, 126, 0)

with open("results/objective_only_audit.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote results/objective_only_audit.json")
