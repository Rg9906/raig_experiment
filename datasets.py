"""
Additional catalogues (paper Sec. VI-A), so that the bias is characterised in
more than one domain.

  mushroom      UCI Agaricus-Lepiota, 8,124 items x 22 categorical attributes.
                Field identification of a specimen by a forager. The tiering
                follows how a person actually acquires each attribute: counts
                and shapes are read off directly, surfaces need care, and
                COLOUR NAMES and ODOUR are perceptual judgements. This matters
                because odour and spore-print colour are the most informative
                attributes in the catalogue, so the domain is a natural test
                of whether high information gain concentrates on hard-to-
                answer attributes outside music.

  dermatology   UCI Dermatology, 366 items x 34 attributes. Differential
                narrowing by a triage questioner. Its tiering is the one case
                in this paper where we did NOT invent the reliability
                annotation: the dataset documentation itself partitions the
                attributes into "Clinical" and "Histopathological", and a
                histopathological attribute is one that requires a biopsy and
                a microscope, i.e. one a patient cannot answer at all. The
                tiers below refine that documented split by who can answer
                from what evidence.

  synthetic     A controlled family in which the rank correlation between an
                attribute information gain and its reliability is a knob.
                This is what turns the paper randomised and adversarial
                controls, which are two isolated points, into a continuum, and
                it locates the break-even correlation at which reliability
                awareness stops paying.

The UCI files are fetched by download_uci.sh / run_domains.py into data/uci/.
"""
import numpy as np
import pandas as pd

import raig

# ---------------------------------------------------------------------------
# Mushroom
# ---------------------------------------------------------------------------
MUSHROOM_COLS = [
    "edible", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat",
]

# Tiering rationale: what does the forager have to do to answer?
#   objective   count it, or read a discrete shape / context off the specimen
#   semi        judge a texture or a structural type, with a reference to hand
#   subjective  name a COLOUR out of 9-12 named options, or identify an ODOUR
#               out of 9. Colour naming and odour identification are the two
#               classic unreliable human perceptual judgements, and they are
#               also where this catalogue carries most of its information.
MUSHROOM_TIER = {
    "cap-shape": "objective", "bruises": "objective",
    "gill-attachment": "objective", "gill-spacing": "objective",
    "gill-size": "objective", "stalk-shape": "objective",
    "ring-number": "objective", "habitat": "objective",
    "population": "objective",

    "cap-surface": "semi", "stalk-root": "semi",
    "stalk-surface-above-ring": "semi", "stalk-surface-below-ring": "semi",
    "ring-type": "semi", "veil-type": "semi",

    "odor": "subjective", "cap-color": "subjective", "gill-color": "subjective",
    "stalk-color-above-ring": "subjective", "stalk-color-below-ring": "subjective",
    "veil-color": "subjective", "spore-print-color": "subjective",
}
MUSHROOM_FEATURES = list(MUSHROOM_TIER)


def load_mushroom(path="data/uci/mushroom/agaricus-lepiota.data"):
    df = pd.read_csv(path, header=None, names=MUSHROOM_COLS, dtype=str)
    # stalk-root carries 2,480 '?' values. Missing-ness is itself an
    # observable state for a forager (the root was not recovered), so it is
    # kept as an explicit category rather than dropped, which would discard
    # 30% of the catalogue.
    df = df.replace("?", "unknown")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dermatology
# ---------------------------------------------------------------------------
DERM_CLINICAL = [
    "erythema", "scaling", "definite_borders", "itching", "koebner",
    "polygonal_papules", "follicular_papules", "oral_mucosal",
    "knee_elbow", "scalp_involvement", "family_history",
]
DERM_HISTO = [
    "melanin_incontinence", "eosinophils", "PNL_infiltrate",
    "fibrosis_papillary_dermis", "exocytosis", "acanthosis", "hyperkeratosis",
    "parakeratosis", "clubbing_rete_ridges", "elongation_rete_ridges",
    "thinning_suprapapillary", "spongiform_pustule", "munro_microabcess",
    "focal_hypergranulosis", "disappearance_granular_layer",
    "vacuolisation_basal", "spongiosis", "saw_tooth_retes",
    "follicular_horn_plug", "perifollicular_parakeratosis",
    "inflammatory_infiltrate", "band_like_infiltrate",
]
DERM_COLS = DERM_CLINICAL[:10] + DERM_HISTO + ["family_history_x", "age", "diagnosis"]

# Refines the dataset own Clinical / Histopathological split by asking who can
# answer from what evidence, with a patient as the oracle.
DERM_TIER = {}
for _a in ("family_history", "age_band", "knee_elbow", "scalp_involvement",
           "oral_mucosal"):
    DERM_TIER[_a] = "objective"          # known without judgement
for _a in ("erythema", "scaling", "definite_borders", "itching"):
    DERM_TIER[_a] = "semi"               # observable, but a matter of degree
for _a in ("koebner", "polygonal_papules", "follicular_papules"):
    DERM_TIER[_a] = "subjective"         # needs dermatological vocabulary
for _a in DERM_HISTO:
    DERM_TIER[_a] = "subjective"         # needs a biopsy and a microscope
DERM_FEATURES = list(DERM_TIER)


def load_dermatology(path="data/uci/dermatology/dermatology.data"):
    raw = pd.read_csv(path, header=None, dtype=str)
    cols = DERM_CLINICAL[:10] + DERM_HISTO + ["family_history", "age", "diagnosis"]
    raw.columns = cols
    df = raw.copy()
    # Age is the only continuous column and the only one with missing values
    # (8 of them, marked '?'). Banded into quartiles so that every attribute
    # in this catalogue is categorical, as the formulation requires.
    age = pd.to_numeric(df["age"], errors="coerce")
    df["age_band"] = pd.qcut(age.rank(method="first"), 4,
                             labels=[f"q{i}" for i in range(4)]).astype(str)
    df.loc[age.isna(), "age_band"] = "unknown"
    for c in DERM_CLINICAL[:10] + DERM_HISTO + ["family_history"]:
        df[c] = df[c].astype(str)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic family with a tunable IG / reliability correlation
# ---------------------------------------------------------------------------
def make_synthetic(n_items=5000, n_attrs=20, rng=None, rho=-1.0,
                   eps_lo=0.02, eps_hi=0.35):
    """Catalogue whose attributes span a range of split balance, with epsilon
    assigned at a controlled rank correlation to that balance.

    `rho` is the target rank correlation between an attribute's split balance
    (its information-gain proxy) and its CROSSOVER PROBABILITY epsilon, not its
    reliability. Signs therefore invert relative to the quantity the paper
    reports, which is corr(max-IG, 1 - epsilon):

        rho = +1  ->  corr(IG, reliability) = -1.  The best-splitting
                      attributes are the least reliable. This is the
                      adversarial-for-the-user structure the paper measures in
                      the music catalogue (rho = -0.44), and the regime RAIG
                      is designed for.
        rho =  0  ->  no relationship; the randomised control.
        rho = -1  ->  corr(IG, reliability) = +1. The informative attributes
                      are also the reliable ones, so there is nothing for a
                      reliability-aware selector to buy.

    Callers should report the measured corr(IG, 1 - epsilon) rather than this
    argument. Returns (df, features, eps_map, ig_proxy).
    """
    rng = rng or np.random.default_rng(0)
    # Balance parameter: attribute a splits the catalogue into `card` values
    # with a Dirichlet-drawn distribution. Low concentration gives a skewed,
    # low-information attribute; high gives an even, high-information one.
    conc = np.geomspace(0.15, 12.0, n_attrs)
    rng.shuffle(conc)
    cols, ig_proxy = {}, []
    for i, c in enumerate(conc):
        card = int(rng.integers(2, 6))
        probs = rng.dirichlet(np.full(card, c))
        vals = rng.choice(card, size=n_items, p=probs)
        cols[f"a{i:02d}"] = np.array([f"v{v}" for v in vals])
        # max over values of the binary entropy of the one-vs-rest split
        ig_proxy.append(max(float(raig.Hb(np.array([p]))[0]) for p in probs))
    df = pd.DataFrame(cols)
    features = list(cols)
    ig = np.array(ig_proxy)

    # Assign epsilon at a target rank correlation with ig.
    r_ig = np.argsort(np.argsort(ig)).astype(float)
    r_ig = (r_ig - r_ig.mean()) / (r_ig.std() + 1e-12)
    z = rho * r_ig + np.sqrt(max(0.0, 1 - rho ** 2)) * rng.standard_normal(n_attrs)
    order = np.argsort(np.argsort(z))
    grid = np.linspace(eps_lo, eps_hi, n_attrs)
    eps = {features[i]: float(grid[order[i]]) for i in range(n_attrs)}
    return df, features, eps, ig


def eps_to_pseudo_tiers(eps_map, edges=(0.10, 0.22)):
    """Bucket a continuous epsilon map into the paper three tiers, so the
    synthetic catalogue can be run through exactly the same tiered
    specification as the real ones."""
    out = {}
    for a, e in eps_map.items():
        out[a] = ("objective" if e < edges[0]
                  else "semi" if e < edges[1] else "subjective")
    return out
