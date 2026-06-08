# Supplementary materials manifest — F2 methodology paper

Per TMLR author guidelines, supplementary materials must be in PDF or
ZIP format and capped at 100 MB. Below is the exact contents that
`pack_supplementary.sh` bundles. All materials are reviewer-readable
without git checkout.

## ZIP contents (target: `f2_methodology_supp.zip`)

```
f2_methodology_supp.zip
├── README.md                              — orientation for reviewers
├── reproducibility/
│   ├── appendix_a_commands.md             — copy of paper §App. A
│   ├── appendix_d_test_inventory.md       — 727 tests by suite (auto-gen)
│   └── data_loop49_checksums.md           — MD5 manifest of empirical CSVs
├── data/
│   ├── loop36_dual.csv                    — canonical mediation (Loop 36)
│   ├── loop47_warmup_stratified.csv       — warmup0 sweep (Loop 47)
│   ├── loop49_wd_stratified.csv           — wd0 sweep (Loop 49)
│   ├── loop49_warmup0_dual.csv            — warmup0 PSEs
│   ├── loop49_wd0_dual.csv                — wd0 PSEs
│   └── loop49_3stratum.csv                — Figure 1 source
├── figures/
│   ├── fig1_rms_nde_signflip.png          — headline sign-flip
│   ├── fig2_stratum_registry.png          — architecture flow
│   ├── fig3_canonical_pse_heatmap.png     — 5×4 heatmap
│   ├── fig4_tipping_curves.png            — Γ_tip(Λ) hyperbolae
│   └── fig_template.py                    — reusable matplotlib helpers
└── audit/
    ├── cross_reference_report.md          — every internal ref resolves
    └── derivation_audit_loop59.md         — Loop 59 audit findings
```

## What is NOT bundled

- Rust source code under `src/` (TMLR allows but does not require;
  reviewers reach the code via the git URL in the camera-ready)
- `target/` build artifacts (irrelevant)
- `.trinity/`, `.claude/`, scratch files

## Size estimate

- 6 CSVs: ~64 KB total
- 4 PNGs + fig_template: ~3 MB combined
- Markdown: < 200 KB
- **Estimated zip size**: ~3 MB (well under 100 MB TMLR cap)

## Anonymization note

The git SHA `583b417` is the empirical anchor commit and is not itself
identifying. The CSV `prov:agent_git_sha` field reveals contributor
history if checked against the repo — see
`anonymization_checklist.md` for the strip-or-not decision.
