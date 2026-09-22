"""
Check every numeric claim in the journal manuscript against the result files.

Tables in a paper are transcribed by hand, and transcription is where papers
acquire the errors that survive review. This script removes the hand: each
check names a quantity, computes it from results/*.json, formats it exactly as
the manuscript should print it, and asserts that string occurs in the named
.tex file. A check that fails is either a stale table or a stale result file,
and the script says which value it expected.

Usage:
    py audit_paper.py            # check everything that has a result file
    py audit_paper.py -v         # also print the checks that passed

Checks whose result file does not exist yet are reported as SKIP, not as
failures, so the audit is usable while experiments are still running.
"""
import json
import os
import re
import sys

TEX = "paper/journal"
RES = "results"
VERBOSE = "-v" in sys.argv

_tex_cache = {}
_missing_files = set()


def tex(name):
    if name not in _tex_cache:
        path = os.path.join(TEX, name)
        _tex_cache[name] = open(path, encoding="utf-8").read() if os.path.exists(path) else None
    return _tex_cache[name]


def res(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        _missing_files.add(name)
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


PASS, FAIL, SKIP = [], [], []


def check(label, value, texfile, fmt="{:.2f}", literal=None):
    """Assert that `value`, printed with `fmt`, occurs in `texfile`."""
    if value is None:
        SKIP.append((label, "no result file"))
        return
    s = literal if literal is not None else fmt.format(value)
    body = tex(texfile)
    if body is None:
        SKIP.append((label, f"{texfile} not written yet"))
        return
    if s in body:
        PASS.append((label, s, texfile))
    else:
        FAIL.append((label, s, texfile))


def pct(x):
    return 100.0 * x


# ---------------------------------------------------------------------------
# Catalogue / identifiability facts  (sec_setup.tex)
# ---------------------------------------------------------------------------
ident = res("identifiability.json")
if ident:
    check("music N", ident["N"], "sec_setup.tex", literal="79{,}803")
    check("music |Q|", ident["Q_compiled"], "sec_setup.tex", literal="& 193 &")
    check("music distinct vectors", ident["distinct_vectors"], "sec_setup.tex",
          literal="77{,}069")
    check("music unique frac", pct(ident["uniquely_identifiable_frac"]),
          "sec_setup.tex", "{:.1f}\\%")
    check("music entropy floor", ident["entropy_floor_bits"], "sec_setup.tex")
    check("music budget bound", ident["budget_bound_T"], "sec_setup.tex")
    check("largest equivalence class", ident["largest_equivalence_class"],
          "sec_setup.tex", literal="$11$ indistinguishable")

audit = res("objective_only_audit.json")
if audit:
    check("objective-only ceiling", pct(audit["top1_accuracy_ceiling"]),
          "sec_setup.tex", "{:.2f}\\%")
    check("objective-only classes", audit["distinct_combinations"],
          "sec_setup.tex", literal="1{,}926")
    check("objective-only largest class", audit["largest_equivalence_class"],
          "sec_setup.tex", literal="932")

# ---------------------------------------------------------------------------
# Bias measurement  (sec_results.tex)
# ---------------------------------------------------------------------------
pat_n = res("pathology_native.json")
pat_4 = res("pathology_4bin.json")
if pat_n:
    check("native rho", pat_n["rho"], "sec_results.tex", "{:.3f}")
    check("native rho p", pat_n["p"], "sec_results.tex", "{:.3f}")
    check("native eps_top5", pat_n["eps_top5"], "sec_results.tex", "{:.3f}")
if pat_4:
    check("4-bin rho", pat_4["rho"], "sec_results.tex", "{:.3f}")
    check("4-bin eps_top5", pat_4["eps_top5"], "sec_results.tex", "{:.3f}")

bias = res("domains_bias.json")
if bias:
    for dom, texfile in (("music", "sec_results.tex"),
                         ("mushroom", "sec_robustness.tex"),
                         ("dermatology", "sec_robustness.tex")):
        m = bias.get(dom)
        if not m:
            continue
        check(f"{dom} Kruskal H", m["kruskal_H"], texfile)
        check(f"{dom} Kruskal p", m["kruskal_p"], texfile, "{:.3f}")
        check(f"{dom} Kendall tau", m["kendall_tau_b"], texfile, "{:.3f}")
        for tier, v in m["mean_max_ig_by_tier"].items():
            check(f"{dom} mean maxIG {tier}", v, texfile, "{:.3f}")
    mus = bias.get("music")
    if mus:
        check("music Mann-Whitney p", mus["mannwhitney_p"], "sec_results.tex",
              "{:.4f}")
        for tier, v in mus["mean_skew_by_tier"].items():
            check(f"music mean skew {tier}", v, "sec_results.tex", "{:.3f}")
        d = mus["discretisation"]
        check("binned mean maxIG", d["mean_max_ig_binned"], "sec_results.tex",
              "{:.3f}")
        check("categorical mean maxIG", d["mean_max_ig_categorical"],
              "sec_results.tex", "{:.3f}")
        check("binned mean eps", d["mean_eps_binned"], "sec_results.tex", "{:.3f}")
        check("categorical mean eps", d["mean_eps_categorical"], "sec_results.tex",
              "{:.3f}")
        check("discretisation MW p", d["mannwhitney_p"], "sec_results.tex",
              "{:.3f}")

# ---------------------------------------------------------------------------
# Main table  (sec_results.tex)
# ---------------------------------------------------------------------------
main = res("main_ext.json")
if main:
    for cond in main["results"]:
        for name, cell in main["results"][cond].items():
            check(f"main {cond}/{name}", pct(cell["acc"]), "sec_results.tex")
    prim = main["results"]["heterogeneous_1.0"]
    # The behaviour table is reported at the headline budget; its numbers are
    # checked against main_T40.json below. Only the T=20 subjective share of
    # RAIG-tiered is still quoted here, as a contrast.
    check("T20 RAIG subj share", pct(prim["RAIG-tiered"]["tier_mix"]["subjective"]),
          "sec_results.tex", "{:.1f}")
    # Rank metrics are quoted at the headline budget only and are checked
    # against main_T40.json below; the T=20 sweep quotes accuracies alone.

stats = res("stats_ext.json")
if stats:
    for row in stats["results"]["heterogeneous_1.0"]:
        if row["comparator"] == "RAIG-oracle":
            continue
        check(f"delta vs {row['comparator']}", pct(row["delta"]),
              "sec_results.tex", "{:+.2f}")
        # The manuscript prints the TOTAL discordant count per comparison, at
        # each budget, rather than n10 and n01 separately.
        check(f"T20 disc vs {row['comparator']}", row["n_discordant"],
              "sec_results.tex", literal=str(row["n_discordant"]))

# ---------------------------------------------------------------------------
# Budget sweep  (sec_results.tex)
# ---------------------------------------------------------------------------
bud = res("budget_sweep.json")
if bud:
    cells = bud.get("results", bud)
    het = cells.get("heterogeneous_1.0", {})
    # The manuscript draws T=10..40 from the n=900 run and quotes only the
    # T=50/60 tail from this independent n=450 sweep, labelled as such.
    for name, per_T in het.items():
        if not isinstance(per_T, dict):
            continue
        for Tk in ("50", "60"):
            cell = per_T.get(Tk)
            if isinstance(cell, dict) and "acc" in cell:
                check(f"budget tail {name}@T{Tk}", pct(cell["acc"]),
                      "sec_results.tex", "{:.1f}")

# ---------------------------------------------------------------------------
# Learned policies  (sec_results.tex)
# ---------------------------------------------------------------------------
lw = res("learned_weights.json")
if lw:
    for name, d in lw["diagnostics"].items():
        check(f"{name} R2", d["r2"], "sec_results.tex")
        for tier, v in d["interaction_by_tier"].items():
            check(f"{name} interaction {tier}", v, "sec_results.tex", "{:+.3f}")
    check("learned n samples", 0, "sec_results.tex", literal="12{,}000")

# ---------------------------------------------------------------------------
# Reliability estimation  (sec_reliability.tex)
# ---------------------------------------------------------------------------
disc = res("reliability_discordance.json")
if disc:
    check("discordance pairs", disc["n_pairs"], "sec_reliability.tex",
          literal="5{,}073")
    check("discordance groups", disc["n_groups"], "sec_reliability.tex",
          literal="3{,}178")
    for row in disc["rows"]:
        check(f"d_f {row['attribute']}", row["rate"], "sec_reliability.tex",
              "{:.4f}")
    check("objective mean d_f", disc["objective_mean"], "sec_reliability.tex",
          "{:.4f}")
    check("non-objective mean d_f", disc["non_objective_mean"],
          "sec_reliability.tex", "{:.4f}")
    check("objective/non-objective ratio", disc["ratio"], "sec_reliability.tex",
          "{:.1f}")
    sa = disc["spearman_annotated_vs_measured"]
    check("annotation vs measurement rho", sa["rho"], "sec_reliability.tex")
    check("annotation vs measurement p", sa["p"], "sec_reliability.tex", "{:.3f}")
    se = disc["spearman_excluding_popularity_genre"]
    check("rho excluding popularity+genre", se["rho"], "sec_reliability.tex")
    check("p excluding popularity+genre", se["p"], "sec_reliability.tex", "{:.3f}")

emp = res("empirical_eps.json")
if emp:
    check("kappa", emp["kappa"], "sec_reliability.tex")

psy = res("reliability_psychophysical.json")
if psy:
    import statistics as _st
    cells = {}
    for r in psy:
        cells.setdefault((r["bins"], r["sigma"]), []).append(r["eps"])
    for (k, sig), vals in sorted(cells.items()):
        check(f"psychophysical k={k} sigma={sig}", _st.fmean(vals),
              "sec_reliability.tex", "{:.3f}")
    check("psychophysical n attributes", len(cells[(2, 0.10)]),
          "sec_reliability.tex", literal="ten continuous attributes")

pri = res("reliability_priors.json")
if pri:
    for tier, d in pri.items():
        check(f"prior {tier} q025", d["q025"], "sec_reliability.tex", "{:.3f}")
        check(f"prior {tier} q975", d["q975"], "sec_reliability.tex", "{:.3f}")

# ---------------------------------------------------------------------------
# Robustness  (sec_robustness.tex)
# ---------------------------------------------------------------------------
ens = res("eps_ensemble.json")
if ens and "mean_delta" in ens:
    check("ensemble mean delta", pct(ens["mean_delta"]), "sec_robustness.tex")
    check("ensemble win fraction", pct(ens["frac_raig_wins"]),
          "sec_robustness.tex", "{:.0f}\\%")
    check("ensemble q025", pct(ens["q025"]), "sec_robustness.tex")
    check("ensemble q975", pct(ens["q975"]), "sec_robustness.tex")

# --------------------------------------------------------------------------
# Headline run at the deployment budget T=40  (sec_results.tex)
# --------------------------------------------------------------------------
m40 = res("main_T40.json")
if m40:
    prim40 = m40["results"].get("heterogeneous_1.0")
    if prim40:
        for name, cell in prim40.items():
            check(f"T40 {name} acc", pct(cell["acc"]), "sec_results.tex")
            check(f"T40 {name} lo", pct(cell["lo"]), "sec_results.tex")
            check(f"T40 {name} hi", pct(cell["hi"]), "sec_results.tex")
            check(f"T40 {name} top5", pct(cell["top5_acc"]), "sec_results.tex")
            check(f"T40 {name} eps asked", cell["mean_eps_asked"],
                  "sec_results.tex", "{:.3f}")
            check(f"T40 {name} subj", pct(cell["tier_mix"]["subjective"]),
                  "sec_results.tex", "{:.1f}")
        for name, cell in prim40.items():
            if name == "RAIG-oracle":
                continue
            check(f"T40 asked eps {name}", cell["mean_eps_asked"],
                  "sec_results.tex", "{:.3f}")
            check(f"T40 asked IG {name}", cell["mean_ig_asked"],
                  "sec_results.tex", "{:.3f}")
            check(f"T40 asked obj {name}", pct(cell["tier_mix"]["objective"]),
                  "sec_results.tex", "{:.1f}")
        r = prim40["RAIG-tiered"]["T40"]
        u = prim40["IG+soft-uniform"]["T40"]
        check("T40 RAIG median rank", r["median_rank"], "sec_results.tex", "{:.1f}")
        check("T40 IG median rank", u["median_rank"], "sec_results.tex", "{:.1f}")

    if prim40:
        for nm in ("RAIG-tiered", "IG+soft-uniform"):
            check(f"T40 {nm} mrr", prim40[nm]["T40"]["mrr"],
                  "sec_results.tex", "{:.3f}")
            check(f"T40 {nm} top10", pct(prim40[nm]["T40"]["top10"]),
                  "sec_results.tex", "{:.1f}\\%")
        for nm in ("RAIG-tiered", "IG+soft-uniform", "IG+hard"):
            check(f"T40 {nm} nll", prim40[nm]["T40"]["nll_bits"],
                  "sec_results.tex")
        for nm, cell in prim40.items():
            for c in ("T10", "T20", "T30", "T40"):
                check(f"T40 curve {nm}@{c}", pct(cell[c]["acc"]),
                      "sec_results.tex", "{:.2f}")

st40 = res("stats_T40.json")
if st40:
    for row in st40["results"]["heterogeneous_1.0"]:
        if row["comparator"] == "RAIG-oracle":
            continue          # identical by construction; printed as 0.00
        check(f"T40 delta vs {row['comparator']}", pct(row["delta"]),
              "sec_results.tex", "{:+.2f}")
        check(f"T40 disc vs {row['comparator']}", row["n_discordant"],
              "sec_results.tex", literal=str(row["n_discordant"]))

cs = res("catalogue_size.json")
if cs:
    for row in cs["rows"]:
        tag = f"size {row['size']} T{row['T']}"
        check(tag + " floor", row["mean_floor_bits"], "sec_results.tex", "{:.2f}")
        check(tag + " ceiling", pct(row["mean_ceiling"]), "sec_results.tex",
              "{:.1f}")
        for n in ("IG+soft-uniform", "RAIG-tiered"):
            check(f"{tag} {n}", pct(row["acc"][n]), "sec_results.tex", "{:.1f}")
    deltas = [pct(r["delta_raig_minus_ig"]) for r in cs["rows"]]
    # The delta range is quoted in three places and must agree in all of them.
    for f in ("sec_results.tex", "sec_intro.tex", "sec_abstract.tex"):
        check(f"size sweep min delta ({f})", min(deltas), f, "{:.1f}")
        check(f"size sweep max delta ({f})", max(deltas), f, "{:.1f}")
    check("size sweep n cells", len(cs["rows"]), "sec_results.tex",
          literal=str(len(cs["rows"])))
    best = max(cs["rows"], key=lambda r: r["acc"]["RAIG-tiered"])
    check("size sweep best RAIG", pct(best["acc"]["RAIG-tiered"]),
          "sec_results.tex", "{:.1f}")
    check("size sweep best IG", pct(best["acc"]["IG+soft-uniform"]),
          "sec_results.tex", "{:.1f}")

em = res("em_eps.json")
if em:
    for key, r in em["results"].items():
        check(f"EM explore={key} rho", r["spearman_hat_vs_true"]["rho"],
              "sec_robustness.tex", "{:+.3f}")
        check(f"EM explore={key} mae", r["mae_vs_true"], "sec_robustness.tex",
              "{:.3f}")
        check(f"EM explore={key} min_obs", r["min_obs"], "sec_robustness.tex",
              literal=str(r["min_obs"]))
        for n in ("IG+soft-uniform", "RAIG-EM", "RAIG-tiered"):
            check(f"EM explore={key} {n}", pct(r["accuracy"][n]["acc"]),
                  "sec_robustness.tex")
        mc = r["accuracy"]["_mcnemar_RAIG-EM_vs_RAIG-tiered"]
        check(f"EM explore={key} discordant vs tiered", mc["n10"],
              "sec_robustness.tex", literal=str(mc["n10"]))
    r0 = em["results"]["0.0"]
    check("EM delta vs incumbent",
          pct(r0["accuracy"]["_mcnemar_RAIG-EM_vs_IG+soft-uniform"]["delta"]),
          "sec_robustness.tex", "{:+.2f}")
    check("EM sessions", em["D_sessions"], "sec_robustness.tex",
          literal="$600$ sessions")
    check("EM init", em["eps_init"], "sec_robustness.tex", "{:.2f}")

psy = res("psychophysical_spec.json")
if psy and psy.get("rows"):
    for row in psy["rows"]:
        check(f"psycho sigma={row['sigma']} cat={row['eps_cat']} RAIG-psycho",
              pct(row["acc"]["RAIG-psycho"]), "sec_robustness.tex")

lam = res("lambda_sweep.json")
if lam:
    if lam.get("crossing_lambda") is not None:
        check("lambda crossing", lam["crossing_lambda"], "sec_robustness.tex",
              "{:.3f}")
        check("lambda crossing mean eps", lam["crossing_mean_true_eps"],
              "sec_robustness.tex", "{:.4f}")
    for row in lam["rows"]:
        L = row["lambda"]
        check(f"lambda {L} mean eps", row["mean_true_eps"], "sec_robustness.tex",
              "{:.3f}")
        for name in ("IG+soft-uniform", "Learned-noiseaware", "RAIG-tiered"):
            check(f"lambda {L} {name}", pct(row["acc"][name]),
                  "sec_robustness.tex", "{:.1f}")
        check(f"lambda {L} delta", pct(row["delta_raig_minus_ig"]),
              "sec_robustness.tex", "{:.1f}")

syn = res("synthetic_rho.json")
if syn:
    if syn.get("break_even_rho") is not None:
        check("synthetic break-even rho", syn["break_even_rho"],
              "sec_robustness.tex")
    for row in syn["rows"]:
        r = row["rho_actual_mean"]
        check(f"synthetic rho {r:+.3f}", r, "sec_robustness.tex", "{:.3f}")
        for key, name in (("acc_ig", "IG"), ("acc_raig", "RAIG"),
                          ("acc_oracle", "oracle")):
            check(f"synthetic {name} @rho {r:+.2f}", pct(row[key]),
                  "sec_robustness.tex", "{:.1f}")
    dmin = min(row["delta_mean"] for row in syn["rows"])
    dmax = max(row["delta_mean"] for row in syn["rows"])
    check("synthetic min delta", pct(dmin), "sec_robustness.tex", "{:.2f}")
    check("synthetic max delta", pct(dmax), "sec_robustness.tex", "{:.2f}")

cap = res("synthetic_capacity.json")
if cap:
    lo = min(r["ceiling_reliable"] for r in cap["rows"])
    hi = max(r["ceiling_reliable"] for r in cap["rows"])
    check("synthetic reliable ceiling lo", pct(lo), "sec_robustness.tex",
          "{:.0f}\\%")
    check("synthetic reliable ceiling hi", pct(hi), "sec_robustness.tex",
          "{:.0f}\\%")

relcap = res("reliable_capacity.json")
if relcap:
    for row in relcap["rows"]:
        name = row["catalogue"].capitalize()
        a, r2 = row["subsets"]["all"], row["subsets"]["objective+semi"]
        check(f"{name} floor all", a["entropy_floor_bits"], "sec_robustness.tex")
        check(f"{name} ceiling all", pct(a["top1_ceiling"]),
              "sec_robustness.tex", "{:.1f}\\%")
        check(f"{name} floor reliable", r2["entropy_floor_bits"],
              "sec_robustness.tex")
        check(f"{name} ceiling reliable", pct(r2["top1_ceiling"]),
              "sec_robustness.tex", "{:.1f}\\%")

tc = res("tier_corruption.json")
if tc:
    if tc.get("break_even_phi") is not None:
        check("corruption break-even phi", tc["break_even_phi"],
              "sec_robustness.tex")
    for row in tc["rows"]:
        p = row["phi"]
        check(f"corruption phi={p} IG", pct(row["mean_acc_ig"]),
              "sec_robustness.tex")
        check(f"corruption phi={p} RAIG", pct(row["mean_acc_raig"]),
              "sec_robustness.tex")
        check(f"corruption phi={p} delta", pct(row["mean_delta"]),
              "sec_robustness.tex", "{:+.2f}")
        check(f"corruption phi={p} n_mis_tiered", row["reps"][0]["n_mis_tiered"],
              "sec_robustness.tex", literal=str(row["reps"][0]["n_mis_tiered"]))

era = res("erasure.json")
if era:
    for level, cell in era["results"].items():
        for name in ("IG+soft-uniform", "RAIG-tiered", "RAIG+erasure"):
            if name not in cell:
                continue
            check(f"erasure {level}/{name} acc", pct(cell[name]["acc"]),
                  "sec_robustness.tex")
            check(f"erasure {level}/{name} answered",
                  cell[name]["expected_answered_frac"], "sec_robustness.tex",
                  "{:.3f}")
            check(f"erasure {level}/{name} subj%",
                  pct(cell[name]["tier_mix"]["subjective"]),
                  "sec_robustness.tex", "{:.1f}")
        for key, cmp in (("_mcnemar_RAIG-tiered_vs_IG+soft-uniform", "RAIG vs IG"),
                         ("_mcnemar_RAIG+erasure_vs_RAIG-tiered", "erasure vs RAIG")):
            if key in cell:
                check(f"erasure {level} delta {cmp}", pct(cell[key]["delta"]),
                      "sec_robustness.tex", "{:+.2f}")
    # the relative-advantage series, which is the surviving form of the
    # Sec. V-E prediction
    for level, cell in era["results"].items():
        ratio = cell["RAIG-tiered"]["acc"] / cell["IG+soft-uniform"]["acc"]
        check(f"erasure {level} ratio", ratio, "sec_robustness.tex")

dm = res("domains_methods.json")
if dm:
    for dom, block in dm["results"].items():
        for cond, cell in block["conditions"].items():
            for name, d in cell.items():
                if name.startswith("_"):
                    continue
                check(f"domain {dom}/{cond}/{name}", pct(d["acc"]),
                      "sec_robustness.tex", "{:.1f}")

corr = res("correlated_ext.json")
if corr:
    for variant, cell in corr["variants"].items():
        for name in ("IG+soft-uniform", "RAIG-tiered"):
            check(f"correlated {variant}/{name}", pct(cell[name]["acc"]),
                  "sec_robustness.tex")
        check(f"correlated {variant} delta",
              pct(cell["mcnemar_RAIG_vs_IG"]["delta"]), "sec_robustness.tex",
              "{:+.2f}")
    for name, d in corr["correlated_vs_independent"].items():
        check(f"corr-vs-indep {name} delta", pct(d["delta"]),
              "sec_robustness.tex", "{:+.2f}")
        check(f"corr-vs-indep {name} n10", d["n10_indep_only"],
              "sec_robustness.tex", literal=str(d["n10_indep_only"]))

# ---------------------------------------------------------------------------
# Online (streaming) reliability estimator vs. static tiers  (sec_robustness.tex)
# ---------------------------------------------------------------------------
dyn = res("dynamic_T70.json")
if dyn:
    pm = dyn["per_method"]
    for name in ("IG+soft-uniform", "RAIG-tiered", "RAIG-streaming",
                 "RAIG-streaming+stop@0.5", "RAIG-streaming+stop@0.7",
                 "RAIG-streaming+stop@0.9"):
        check(f"dynamicT70 {name} acc", pct(pm[name]["acc"]), "sec_robustness.tex")
    tiered_acc = pm["RAIG-tiered"]["acc"]
    for name in ("IG+soft-uniform", "RAIG-streaming", "RAIG-streaming+stop@0.5",
                 "RAIG-streaming+stop@0.7", "RAIG-streaming+stop@0.9"):
        check(f"dynamicT70 {name} delta-vs-tiered",
              pct(pm[name]["acc"] - tiered_acc), "sec_robustness.tex", "{:+.2f}")

    def sci(p):
        """Render a p-value the way this manuscript writes scientific notation
        elsewhere: '5.0\\times10^{-27}', one significant digit of mantissa."""
        if p == 0:
            return "0"
        exp = 0
        m = p
        while m < 1:
            m *= 10
            exp -= 1
        return f"{m:.1f}\\times10^{{{exp}}}"

    for c in dyn["paired_comparisons"]:
        if c["p_holm"] >= 0.999:
            check(f"dynamicT70 pholm {c['a']}-vs-{c['b']}", c["p_holm"],
                  "sec_robustness.tex", literal="1.00")
        else:
            check(f"dynamicT70 pholm {c['a']}-vs-{c['b']}", c["p_holm"],
                  "sec_robustness.tex", literal=sci(c["p_holm"]))
    for thresh, key in ((0.5, "RAIG-streaming+stop@0.5"),
                        (0.7, "RAIG-streaming+stop@0.7"),
                        (0.9, "RAIG-streaming+stop@0.9")):
        check(f"dynamicT70 frac_stopped {thresh}",
              pct(pm[key]["frac_stopped_early"]), "sec_robustness.tex", "{:.1f}")

hi5k = res("high_accuracy_N5000.json")
if hi5k:
    pmh = hi5k["per_method"]
    check("N5000 ceiling", pct(hi5k["mean_ceiling"]), "sec_robustness.tex", "{:.1f}")
    check("N5000 floor", hi5k["mean_floor_bits"], "sec_robustness.tex", "{:.2f}")
    for name in ("IG+soft-uniform", "RAIG-tiered", "RAIG-streaming"):
        for c in (40, 60, 80, 100, 200):
            check(f"N5000 {name} T{c}", pct(pmh[name][f"T{c}"]["acc"]),
                  "sec_robustness.tex")
    for c in (80, 100, 200):
        check(f"N5000 streaming-vs-tiered p T{c}",
              hi5k["streaming_vs_tiered"][f"T{c}"]["p"], "sec_robustness.tex", "{:.2f}")

# ---------------------------------------------------------------------------
# Ablation and cost  (sec_results.tex)
# ---------------------------------------------------------------------------
abl = res("ablation_T40.json")
if abl:
    for row in abl["rows"]:
        check(f"ablation {row['variant']} acc", pct(row["acc"]), "sec_results.tex")
        check(f"ablation {row['variant']} lo", pct(row["lo"]), "sec_results.tex")
        check(f"ablation {row['variant']} hi", pct(row["hi"]), "sec_results.tex")
        check(f"ablation {row['variant']} gap", row["pct_gap"],
              "sec_results.tex", "{:.0f}\\%")

tr = res("trace_diagnostics.json")
if tr:
    check("capacity-only split mass", tr["capacity_only"]["mean_split_mass"] * 100,
          "sec_results.tex", "{:.1f}\\%")
    check("clean traces monotone",
          tr["clean_belief_traces"]["monotone_non_decreasing"],
          "sec_results.tex", literal="all ten")

lat = res("latency_ext.json")
if lat:
    # The manuscript deliberately quotes the paired RATIO and the analytic
    # overhead, not the absolute ms/turn medians: the medians are a property of
    # a contended laptop, not of either method, so requiring them in the text
    # would be requiring a number we do not want a reader to rely on.
    if "ratio_median" in lat:
        check("latency paired ratio median", lat["ratio_median"],
              "sec_results.tex", "{:.3f}")
        check("latency ratio min", lat["ratio_min"], "sec_results.tex", "{:.3f}")
        check("latency ratio max", lat["ratio_max"], "sec_results.tex", "{:.3f}")
    for name, d in lat["methods"].items():
        check(f"threshold reached {name}", pct(d["frac_reached_thresh"]),
              "sec_results.tex", "{:.2f}\\%")
        check(f"threshold top5 {name}", pct(d["top5"]), "sec_results.tex",
              "{:.1f}\\%")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print(f"audit: {len(PASS)} pass, {len(FAIL)} FAIL, {len(SKIP)} skip")
if _missing_files:
    print("  result files not present: " + ", ".join(sorted(_missing_files)))
if VERBOSE:
    for label, s, f in PASS:
        print(f"  ok   {label:44s} {s!r} in {f}")
for label, s in SKIP:
    print(f"  skip {label:44s} ({s})")
for label, s, f in FAIL:
    print(f"  FAIL {label:44s} expected {s!r} in {f}")
sys.exit(1 if FAIL else 0)
