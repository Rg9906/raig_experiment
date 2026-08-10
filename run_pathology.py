import json
import numpy as np
import raig

df = raig.load_catalogue("data/dataset_final.csv")
cat = raig.Catalogue(df)

audit = raig.identifiability_audit(df, raig.FEATURES)
audit["Q_compiled"] = cat.Q
print("IDENTIFIABILITY AUDIT:", json.dumps(audit, indent=2, default=float))

tab, rho, pval = raig.measure_pathology(cat)
print()
print("PATHOLOGY TABLE (2-bin native levels, i.e. main pipeline binning):")
print(tab.sort_values("max_ig", ascending=False).to_string(index=False))
print("Spearman rho =", rho, " p =", pval)

tier_top5_mean = tab.sort_values("max_ig", ascending=False).head(5)["eps"].mean()
tier_all_mean = tab["eps"].mean()
print("mean eps of top-5 max-IG attrs:", tier_top5_mean, " mean eps overall:", tier_all_mean)

with open("results/pathology_native.json", "w") as f:
    json.dump({"rows": tab.to_dict(orient="records"), "rho": float(rho), "p": float(pval),
               "eps_top5": float(tier_top5_mean), "eps_all": float(tier_all_mean)}, f, indent=2)

with open("results/identifiability.json", "w") as f:
    json.dump(audit, f, indent=2, default=float)

uniform_eps = float(np.average([raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))
print()
print("uniform_eps (question-weighted mean of tiered true eps):", uniform_eps)
with open("results/uniform_eps.json", "w") as f:
    json.dump({"uniform_eps": uniform_eps}, f)
