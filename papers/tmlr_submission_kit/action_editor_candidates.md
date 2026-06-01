# TMLR action editor candidates — F2 methodology submission

The TMLR submission form on OpenReview asks the author to nominate
suitable action editors. The paper crosses three areas:

1. **Causal inference / causal mediation theory** (§3.2 four-PSE
   decomposition, §3.3 bridge-score envelope, §9.2 related work)
2. **ML methodology + reproducibility** (§3.5 W3C-PROV preambles,
   §8 reproducibility tooling, §10 MLRC track positioning)
3. **Applied statistics** (delta-method SEs valid at N=5, finite-
   sample CIs, sensitivity analysis)

Below are five TMLR action editors with overlapping expertise,
ordered by primary-area fit. Confirm current TMLR roster at
[jmlr.org/tmlr/editorial-board.html](https://jmlr.org/tmlr/editorial-board.html)
before submitting — editorial board changes annually.

## Primary candidates (causal mediation / inference)

### 1. Fredrik D. Johansson (Chalmers University of Technology)
- **Areas**: counterfactual estimation, causal inference, treatment-effect estimation
- **Fit**: Direct match for §3.2 Pearl-CDE framework + §3.3 sensitivity envelope.
  Counterfactual-estimation expertise aligns with the no-XM-interaction
  identification used in F2.

### 2. Sameer Deshpande (University of Wisconsin-Madison)
- **Areas**: Bayesian variable selection, causal inference, graphical modeling
- **Fit**: Strong fit for §3.5 (provenance / model-fingerprint discipline) +
  §6 (sensitivity to choices). Graphical-modeling background relevant
  for the mediator-pinning stratification design.

### 3. Xiaojie Mao (Tsinghua University)
- **Areas**: causal inference, machine learning
- **Fit**: Hybrid causal-ML area matches F2's positioning — Pearl-style
  identification applied to ML training-recipe ablations. Likely
  comfortable evaluating both the methodology and the ML
  demonstration.

## Secondary candidates (ML methodology / reproducibility)

### 4. Junpei Komiyama (Mohamed bin Zayed University of AI)
- **Areas**: reproducibility, hypothesis testing
- **Fit**: Strong fit for §3.5 reproducibility infrastructure + §8.2
  formula-locking regression tests. Hypothesis-testing background
  relevant for §6.2 (Student-t vs permutation vs bootstrap-t).

### 5. Matt J. Kusner (Mila)
- **Areas**: property testing, sequential hypothesis testing
- **Fit**: Hypothesis-testing expertise directly relevant to §6.2.
  Property-testing background may engage with §8.2 lock tests as a
  software-verification analogue.

## Tertiary / fallback

### 6. Devendra Singh Dhami (Eindhoven University of Technology)
- **Areas**: causal machine learning
- **Fit**: Hybrid causal-ML; suitable if primary causal candidates
  are unavailable or conflicted.

### 7. Yingzhen Li (Imperial College London)
- **Areas**: causal representation learning, deep learning
- **Fit**: Adjacent (representation learning vs training-recipe
  ablation); secondary fallback only.

## Conflict-of-interest considerations

Before nominating, verify that none of the candidates above are
on the author's institutional collaboration list, recent
co-author network (last 4 years), or PhD advisor / advisee
relationship. The F2 paper's author block is currently anonymous
for double-blind review; the COI list will be auto-derived from
the OpenReview profile.

## Three-name nomination strategy

If the form asks for **3 nominations** (typical TMLR ask), submit:
- Johansson (primary causal)
- Komiyama (reproducibility methodology)
- Deshpande (statistical methodology)

This spans the three contribution areas without redundancy.

## Audit metadata

- Source: TMLR editorial board page WebFetch, Loop 84 (2026-06-02)
- Candidates verified: 7 editors with stated areas matching at least
  one of F2's three contribution dimensions
- Next refresh: at submission time (editorial board changes annually)
