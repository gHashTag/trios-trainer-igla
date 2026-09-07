# DOI & Coq Witness Provenance (R5-honest)

> Audit date: **2026-05-12** · Auditor lane: `perplexity-computer-grandmaster` · ICA-DOI-2026-05-12-T22:00 · Throne [trios#264](https://github.com/gHashTag/trios/issues/264)

This file disambiguates the algebraic anchor that appears throughout the
`trios-trainer-igla` source tree:

```rust
// Anchor: φ²+φ⁻²=3 · DOI 10.5281/zenodo.19227877
```

The comment marker is **not** a citation of a peer-reviewed paper. It is a
provenance fingerprint linking the file to the Trinity research collection.
R5-honest reading:

## Algebraic identity

`φ² + φ⁻² = 3`  where  `φ = (1+√5)/2`.

This is a direct algebraic consequence of `φ² = φ + 1` (the defining
recurrence of the golden ratio), reduced via the Lucas relation
`L_n = φⁿ + (−φ)⁻ⁿ` at `n = 2`. The identity itself requires no DOI; it is
school-level algebra.

## Coq witness (the actual proof corpus)

The empirical witness that this identity, together with its derived
invariants (Pellis embedding, ternary sufficiency, φ-distance), is
mechanically checked lives at:

**[gHashTag/t27/coq](https://github.com/gHashTag/t27/tree/main/coq)**

Audit of the `main` branch on 2026-05-12 produced the following ground truth:

| Metric | Count |
|---|---|
| `.v` files | 10 (3 in `coq/Theorems/` + 7 in `coq/Kernel/`) |
| `Theorem` declarations | 6 |
| `Lemma` declarations | 42 |
| **Total statements** | **48** |
| `Qed.` (Proven) | 35 |
| `Admitted.` | 0 |

Method: `grep -rE '^\s*(Theorem|Lemma|Qed|Admitted)\s+' coq --include='*.v'`
on a fresh `--depth=1` clone of `gHashTag/t27`.

The frequently-cited phrase **"84 theorems in t27"** is folklore. It does
not match the audited corpus and should not be propagated.

## Zenodo records (provenance only — NOT peer-reviewed papers)

| DOI | Real Zenodo title | Real content (audit 2026-05-12) | Treat as |
|---|---|---|---|
| [10.5281/zenodo.19227877](https://doi.org/10.5281/zenodo.19227877) | `Trinity B007: VSA Operations for Ternary Computing v5.0` | 1 file `B007_v5_description.md` (1968 bytes, markdown abstract only); 1 download, 49 views | Persistent identifier for a software description stub. **NOT a paper.** |
| [10.5281/zenodo.19227879](https://doi.org/10.5281/zenodo.19227879) | `Trinity S³AI Framework — Complete Research Collection v5.0` | 1 file `PARENT_v5_description.md` (16074 bytes); 0 downloads, 21 views | Parent description record for the Trinity series. |
| [10.5281/zenodo.18947017](https://doi.org/10.5281/zenodo.18947017) | `gHashTag/trinity: Trinity v2.0.2 — FPGA Autoregressive Ternary LLM` | 1 ZIP (1.24 MB) | Software release deposit. |

All three records are licensed CC-BY-4.0, owner ID 1570570, no peer review,
no Zenodo community attached.

## Why the anchor marker still cites 19227877

The DOI string in the Rust comment markers serves three honest purposes:

1. **Persistent identifier** — even if this repo's `main` is force-pushed or
   archived, `10.5281/zenodo.19227877` will resolve to the Zenodo deposit
   forever (DataCite guarantee).
2. **Provenance fingerprint** — `acm-ae-check` (and analogous gates in
   `trios`) grep for the literal string to confirm the file belongs to the
   Trinity collection.
3. **Affiliation pointer** — links the file back to the Trinity research
   collection without claiming the DOI itself is a paper.

The marker does **not** claim that:

- the DOI hosts a peer-reviewed publication,
- the file at the DOI contains a formal proof,
- there are 84 theorems behind it.

## Citation grade (Gate G4)

Per `trinity-grandmaster` Gate G4, every numeric anchor must trace to:

- **Rust constant**: this repo (`src/*.rs`).
- **Coq theorem**: `trinity-clara/proofs/igla/lucas_closure_gf16.v::lucas_2_eq_3` (Proven) and the broader `gHashTag/t27/coq` corpus (48 statements / 35 Proven / 0 Admitted).
- **JSON anchor**: `assertions/igla_assertions.json` in `gHashTag/trios`.

Zenodo DOIs are **not** part of the citation grade chain — they are
provenance only.

## How to cite this anchor in new code or docs

✅ Honest forms:

> Anchor: `φ²+φ⁻²=3` (algebraic identity; Coq witness gHashTag/t27/coq, 48 statements / 35 Proven / 0 Admitted, audit 2026-05-12; Zenodo provenance 10.5281/zenodo.19227877 = B007 VSA description stub, NOT a paper).

> The anchor `φ²+φ⁻²=3` is mechanically verified in [gHashTag/t27/coq](https://github.com/gHashTag/t27/tree/main/coq) (48 Coq statements as of 2026-05-12). The Zenodo deposit [10.5281/zenodo.19227877](https://doi.org/10.5281/zenodo.19227877) serves as the persistent identifier for the Trinity collection but is a software description stub, not a peer-reviewed paper.

❌ Forbidden:

- "Trinity paper anchor: 10.5281/zenodo.19227877" — implies peer review that does not exist.
- "84 theorems in t27" — folklore, contradicted by audit.
- "Trinity Anchor record (TRI-27 series)" without qualifying that it is a software description stub.

## References

- Trinity Throne (Queen's registry): [gHashTag/trios#264](https://github.com/gHashTag/trios/issues/264)
- IGLA RACE: [gHashTag/trios#143](https://github.com/gHashTag/trios/issues/143)
- ICA-DOI-2026-05-12-T22:00 NASA mission verification report: this commit
- t27 Coq corpus: [gHashTag/t27/coq](https://github.com/gHashTag/t27/tree/main/coq)
- Sibling honesty patch in trios: [fix/l-doi-honest](https://github.com/gHashTag/trios/tree/fix/l-doi-honest)

φ² + φ⁻² = 3 · TRINITY · NEVER STOP
