# Loop 61 hostile-reviewer screen (TMLR action editor sim, 10-min desk-reject pass)

## Top 3 desk-rejection risks

### 1. Synthetic counter task is not a language model (Severity 5/5)
Title and abstract sell "Transformer Training-Recipe Ablations" + BPB.
§4.1 reveals deterministic synthetic counter, vocab=64, batch=1, 200
steps, ~8K params. §7.4 admits "pattern could in principle be specific
to this task."

**Fix**: retitle (e.g. "a stratified CDE framework, demonstrated on a
synthetic-counter sandbox") AND either (a) add one small-but-real LM
run (TinyStories or WikiText-2 at ≥1M params for ≥1k steps) or
(b) reposition §5 as a *unit test* of the framework rather than as an
ML finding about RmsNorm.

### 2. Sign-flip claim is causally underspecified vs. obvious counter-reading (Severity 4/5)
§5.2 reports wd0 CDE +0.43 BPB with lower CI at +0.01 — 7%-of-CI
margin. The paper frames as "RmsNorm intrinsically helpful (harmful to
remove)"; reviewer can read as "wd=0 is pathological/degenerate, the
canonical NDE is the answer." The paper never argues *why* wd=0 is a
meaningful counterfactual.

**Fix**: in §5.2 add a paragraph explaining wd=0 corresponds to
published recipes (most pre-AdamW transformer recipes; BitNet b1.58
early ablations). Otherwise, soften "intrinsically helpful" to
"directionally consistent under WD-pinning".

### 3. MLRC/TMLR-track fit asserted but not earned (Severity 4/5)
MLRC requires reproducing a *published* claim or releasing a framework
useful against published claims. F2 reproduces its own Loop 49 result
against itself. §3.5.4 reproducibility relies on the reviewer
rebuilding the trainer with `TRAINER_INTERNALS_SCHEMA` — exactly the
brittle in-house dependency MLRC avoids.

**Fix**: §10.1/§10.2: replace "we release a reproducibility artifact"
with one concrete second-paper reproduction (apply F2 to one
published ablation: NormFormer, BitNet b1.58 §3, or Pre-LN vs Post-LN)
and ship the comparison CSV at HEAD.

## Top 3 reviewer-flag risks (survive desk, kill in review)

1. **CI lower-bound = +0.01 BPB** is a 7%-of-CI margin at N=5; §7.2
   admits it but reviewers will still demand N≥20 on wd0 stratum.
2. **"Stable across strata" invariance is suspiciously exact**: §5.3
   reports NIE_M1 via rms = −0.75 [−1.32, −0.18] byte-identical across
   all three strata. Likely a structural artifact, not an empirical
   result. Either prove it must be invariant by construction (in
   which case it's not evidence), or flag and explain.
3. **Novelty vs. ABLATOR + stratification**: §9.1 acknowledges ABLATOR
   and AblationBench. The differentiating claim (stratified CDE +
   4-PSE + bridge envelope) is real, but the empirical demonstration
   that justifies it is on a synthetic task.

## Top 1 missing experiment

**One published ablation, re-analyzed with F2, where the F2 verdict
differs from the original paper's verdict.** A Pre-LN-vs-Post-LN re-run
on a public ~10M–100M-param transformer (nanoGPT on tiny-shakespeare;
or a 5-seed re-run of one BitNet b1.58 §3 sub-ablation) with F2
producing a different stratified CDE conclusion than the seed-mean.

## Closest single fix

Reframe §1.2 contribution #5 and the abstract from
"**proof-of-concept** of a sign flip in RmsNorm" to
"**unit-test demonstration on a synthetic task** that the framework
detects a sign flip when one is constructed". Tonal shift only;
zero new compute. Combined with one paragraph in §10.2 acknowledging
MLRC final will include re-analysis of one external published
ablation, this converts hostile "rejecting on scope" to neutral.

---

## Audit metadata

- Audited: `papers/f2_methodology.md` at commit `77d1218`
- Reviewer model: TMLR action editor, 10-min desk-reject screen
- Read coverage: full paper, 1330 lines
- No edits made; read-only audit
