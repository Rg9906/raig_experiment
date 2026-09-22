"""
The extended method set used by the journal version (paper Table: baselines).

Beyond the six of the conference version this adds:
  Learned-noiseblind / Learned-noiseaware   the learned-policy family (Sec. VII-C)
  RAIG-empirical                            epsilon-hat MEASURED from the
                                            catalogue by repeated-item
                                            discordance, with no annotation
                                            and no human study (Sec. V-B)

Method ORDER is fixed and must not be permuted: Random+soft derives its
private draws from its position in the list, so moving it changes its
sessions. New methods are therefore always appended before the RAIG block
rather than inserted at the front.
"""
import json
import numpy as np

import raig
import learned

METHOD_ORDER = [
    "Random+soft", "IG+hard", "IG+soft-uniform", "IG+soft-objective",
    "Learned-noiseblind", "Learned-noiseaware",
    "RAIG-empirical", "RAIG-tiered", "RAIG-oracle",
]


def build(cat, uniform_eps=None, weights_path="results/learned_weights.json",
          empirical_path="results/empirical_eps.json", include=None):
    if uniform_eps is None:
        uniform_eps = float(np.average(
            [raig.TIER_EPS[raig.ATTR_TIER[a]] for a in cat.attr_of_q]))

    tiered_eps = cat.eps_vector({a: raig.TIER_EPS[t] for a, t in raig.ATTR_TIER.items()})
    zero_eps = np.zeros(cat.Q, dtype=np.float32)
    uniform_vec = np.full(cat.Q, uniform_eps, dtype=np.float32)
    emp = json.load(open(empirical_path))["eps"]
    empirical_vec = cat.eps_vector(emp)

    feat = learned.PolicyFeatures(cat)
    W = json.load(open(weights_path))["weights"]

    ms = [
        raig.Method("Random+soft", "random", lambda te: uniform_vec),
        raig.Method("IG+hard", "ig", lambda te: zero_eps),
        raig.Method("IG+soft-uniform", "ig", lambda te: uniform_vec),
        raig.Method("IG+soft-objective", "ig", lambda te: uniform_vec,
                    allowed_tiers=("objective",)),
        raig.Method("Learned-noiseblind", "learned", lambda te: uniform_vec,
                    scorer=feat.make_scorer(W["Learned-noiseblind"], 20)),
        raig.Method("Learned-noiseaware", "learned", lambda te: uniform_vec,
                    scorer=feat.make_scorer(W["Learned-noiseaware"], 20)),
        raig.Method("RAIG-empirical", "raig", lambda te: empirical_vec),
        raig.Method("RAIG-tiered", "raig", lambda te: tiered_eps),
        raig.Method("RAIG-oracle", "raig", lambda te: te),
    ]
    if include is not None:
        ms = [m for m in ms if m.name in include]
    for m in ms:
        m.allowed = cat.allowed_mask(m.allowed_tiers)
    return ms, uniform_eps


def build_dynamic(cat, uniform_eps=None, weights_path="results/learned_weights.json",
                   empirical_path="results/empirical_eps.json",
                   early_stop_thresholds=(0.5, 0.7, 0.9), explore_frac=0.20):
    """The 9-method set of build(), plus the online/adaptive extensions of
    Sec. IX's "Dynamic reliability + adaptive budget" follow-up: RAIG-streaming
    (population-level epsilon re-estimated between batches, no tier annotation
    assumed -- see raig.run_paired_trials's "raig-streaming" select mode) and
    the same method with real early stopping layered on top at each threshold
    in early_stop_thresholds.

    ALWAYS appended after the existing 9 (never inserted before them):
    Random+soft's private RNG substream is seeded from its position in the
    method list (see the module docstring), so any new method must come
    after it or every existing stored result silently stops reproducing.
    """
    ms, uniform_eps = build(cat, uniform_eps=uniform_eps, weights_path=weights_path,
                             empirical_path=empirical_path)

    streaming_flat = raig.Method(
        "RAIG-streaming", "raig-streaming",
        lambda te: np.full(cat.Q, raig.STREAM_EPS_INIT, dtype=np.float32),
        explore_frac=explore_frac)
    streaming_flat.allowed = cat.allowed_mask(streaming_flat.allowed_tiers)
    ms.append(streaming_flat)

    for thresh in early_stop_thresholds:
        m = raig.Method(
            f"RAIG-streaming+stop@{thresh}", "raig-streaming",
            lambda te: np.full(cat.Q, raig.STREAM_EPS_INIT, dtype=np.float32),
            explore_frac=explore_frac, early_stop_threshold=thresh)
        m.allowed = cat.allowed_mask(m.allowed_tiers)
        ms.append(m)

    return ms, uniform_eps


def conditions(cat, uniform_eps):
    """The seven noise conditions, as in the conference version."""
    return {
        "clean": lambda: raig.tiered_true_eps(cat, 0.0),
        "homogeneous": lambda: raig.homogeneous_true_eps(cat, uniform_eps),
        "heterogeneous_0.5": lambda: raig.tiered_true_eps(cat, 0.5),
        "heterogeneous_1.0": lambda: raig.tiered_true_eps(cat, 1.0),
        "heterogeneous_1.5": lambda: raig.tiered_true_eps(cat, 1.5),
        "randomised": lambda: raig.randomised_true_eps(cat, 1.0),
        "adversarial": lambda: raig.adversarial_true_eps(cat, 1.0),
    }


def build_generic(cat, tier_map, uniform_eps=None):
    """The six-method comparison for a catalogue other than the music one.

    Omits the learned policies (their weights are fitted per catalogue and
    the cross-domain question is about the bias, not about policy learning)
    and RAIG-empirical (which needs repeated items, and neither UCI catalogue
    has an entity-duplicate structure to exploit).
    """
    import numpy as _np
    if uniform_eps is None:
        uniform_eps = float(_np.average(
            [raig.TIER_EPS[tier_map[a]] for a in cat.attr_of_q]))
    tiered = cat.eps_vector({a: raig.TIER_EPS[t] for a, t in tier_map.items()})
    zero = _np.zeros(cat.Q, dtype=_np.float32)
    unif = _np.full(cat.Q, uniform_eps, dtype=_np.float32)
    ms = [
        raig.Method("Random+soft", "random", lambda te: unif),
        raig.Method("IG+hard", "ig", lambda te: zero),
        raig.Method("IG+soft-uniform", "ig", lambda te: unif),
        raig.Method("IG+soft-objective", "ig", lambda te: unif,
                    allowed_tiers=("objective",)),
        raig.Method("RAIG-tiered", "raig", lambda te: tiered),
        raig.Method("RAIG-oracle", "raig", lambda te: te),
    ]
    for m in ms:
        m.allowed = cat.allowed_mask(m.allowed_tiers)
    return ms, uniform_eps, tiered
