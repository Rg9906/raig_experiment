# Reliability-Aware Information Gain (RAIG)

Reference implementation, catalogues and complete result files for
*Information Gain Prefers the Questions Users Cannot Answer:
Reliability-Aware Query Selection for Interactive Identification*.

Classical information gain picks the question that most evenly splits the
belief. When a person supplies the answers, "most evenly splitting" and "most
reliably answerable" are not independent properties, and in catalogues that mix
quantile-binned perceptual descriptors with naturally skewed categorical
metadata they are *negatively* related. RAIG replaces the entropy of the
induced split with the mutual information between the target and the observed
answer through a per-attribute binary symmetric channel:

    RAIG(q) = Hb(p_q * eps_f(q)) - Hb(eps_f(q))          (one line, see raig.py)

where `p * eps = p(1-eps) + (1-p)eps`.

## Layout

| path | what it is |
|---|---|
| `raig.py` | catalogue compilation, both selectors, the paired-trial harness, statistics |
| `methods_ext.py` | the nine-method comparison set used by the journal version |
| `learned.py` | the EAR/CPR-style learned value-regression policy baseline |
| `reliability.py` | tier priors, repeated-item discordance, tier corruption, psychophysical and shipped-boundary flip probabilities |
| `run_em_eps.py` | learns `eps` per attribute from interaction logs by EM with the target item latent — no annotation, no labels, no ground truth |
| `run_psychophysical_spec.py` | derives `eps` from the catalogue's own cut points; reported as a negative result |
| `run_reliable_capacity.py` | the reliable-subset audit that predicts whether the correction will pay |
| `datasets.py` | UCI Mushroom / Dermatology loaders and the synthetic catalogue family |
| `run_*.py` | one experiment each; every one writes a JSON file to `results/` |
| `results/` | every number in the paper, as produced |
| `paper/journal/` | the manuscript |
| `audit_paper.py` | checks every numeric claim in the manuscript against `results/` |

## Reproducing

```
py run_main_ext.py            # primary comparison: 9 methods x 7 conditions, n=900
bash run_queue2.sh            # robustness suite (long; each script checkpoints)
bash run_queue3.sh            # protocol-matched reruns of the ablation/cost studies
py run_em_eps.py              # learn eps from logs (the annotation-free route)
py run_psychophysical_spec.py # derive eps from the catalogue (does not work)
py run_reliable_capacity.py   # the deployment test
py verify_harness.py          # determinism and subset-invariance gate tests
py audit_paper.py             # verify the manuscript against results/
```

Scripts are strictly sequential by design. The belief matrix is
`(N x M*batch)` float32 and two heavy runs at once thrash rather than
parallelise.

### Determinism

`raig.py` pins the BLAS thread count to 1 *before* importing NumPy. This is a
correctness requirement, not a performance tweak. A multi-threaded matmul sums
`(Q,N) x (N,MC)` in a shape-dependent order, so the split-mass vector differs
in the last float32 ulp between runs with different batch shapes; selection is
an argmax over that vector, near-ties are common, and one different question at
turn 1 sends the whole session down a different path. With the pin, runs are
bitwise reproducible, and each method's results are invariant to which other
methods are run alongside it — which is what allows a new method to be added to
a comparison without re-running the incumbents.

Consequence worth knowing: **batch size is part of the protocol.** The shared
noise stream is drawn per batch as `rng.random((T, Rb))`, so a run at
`batch_size=150` faces different simulated users from one at `batch_size=30`.
Every `*_ext.py` script uses `R=300`, `SEEDS=[0,1,2]`, `batch_size=30`,
`rng=default_rng(1000+seed)`; the older scripts do not, and their results are
not comparable cell-for-cell. That is why the `_ext` reruns exist.

## Data

`data/dataset_final.csv` is the music catalogue (79,803 tracks x 21
attributes). `data/uci/` holds the two UCI catalogues; `run_domains.py` fetches
them if absent.

## Auditing the manuscript

`audit_paper.py` computes each quantity from `results/*.json`, formats it the
way the manuscript should print it, and asserts the string occurs in the named
`.tex` file. A failure means a stale table *or* a stale result file, and the
script says which value it expected. Run it after any experiment re-run and
after any edit to a table.
