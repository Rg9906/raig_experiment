"""
4-bin discretisation ablation for Table pathology (row 2) and Fig. 4 data.
Rebins every attribute with a continuous backing column (CONTINUOUS_SOURCE)
at k=4 quantile bins instead of the native (mostly 2-bin) discretisation, and
recomputes measure_pathology on the resulting catalogue.
"""
import json
import numpy as np
import raig

df = raig.load_catalogue("data/dataset_final.csv")

df4 = df.copy()
skipped = []
for attr, src in raig.CONTINUOUS_SOURCE.items():
    if src not in df.columns:
        # tempo_bpm requires the raw (pre-discretisation) catalogue, which is
        # not part of dataset_final.csv; "tempo" is already 5-level categorical
        # there, so it is left at its native binning and excluded from the
        # 4-bin rebin set (documented as such in Table pathology's caption).
        skipped.append(attr)
        continue
    df4[attr] = raig.rebin_attribute(df, attr, k=4)
if skipped:
    print("skipped (no raw source column in dataset_final.csv):", skipped)

cat4 = raig.Catalogue(df4)
print("4-bin catalogue: N =", cat4.N, "Q =", cat4.Q)

tab4, rho4, p4 = raig.measure_pathology(cat4)
print(tab4.sort_values("max_ig", ascending=False).to_string(index=False))
print("Spearman rho =", rho4, " p =", p4)

top5_mean4 = tab4.sort_values("max_ig", ascending=False).head(5)["eps"].mean()
all_mean4 = tab4["eps"].mean()
print("mean eps of top-5 max-IG attrs (4-bin):", top5_mean4, " mean eps overall:", all_mean4)

with open("results/pathology_4bin.json", "w") as f:
    json.dump({"rows": tab4.to_dict(orient="records"), "rho": float(rho4), "p": float(p4),
               "eps_top5": float(top5_mean4), "eps_all": float(all_mean4),
               "Q_compiled": cat4.Q}, f, indent=2)
print("saved results/pathology_4bin.json")
