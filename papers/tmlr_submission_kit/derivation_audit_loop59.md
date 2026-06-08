# Loop 59 derivation audit — saved for Loop 60 follow-up

## Verified correct
- Gao-Li-Luo arxiv ID `2007.16031`
- Ohnishi-Li arxiv ID `2605.18724`, title, authors (Yale, stat.ME)
- Owen `2508.10083` author + qualitative BCa-undercoverage motivation
- Student-t critical `t_{0.975,4} ≈ 2.776` at N=5
- Structural form of envelope `Λ·(Γ-1)/Γ` matches Ohnishi-Li algebraically

## Issues to address (Loop 60 fix candidates)

### High-confidence, safe to fix immediately
1. **Miles & Shpitser citation**: Actual authors are
   Miles, Shpitser, Kanki, Meloni & Tchetgen Tchetgen.
   Currently cited as "Miles & Shpitser" — drops three coauthors.
2. **"Eq. 6" of Ohnishi-Li** → should be "Theorem 2 / Eq. (5)".
   Eq. 6 is Proposition 2 about bridge-score tightening.
3. **VanderWeele-Ding threshold attribution**: the 1.25 / 2.0 cutoffs
   are NOT in the original VW-D 2017 paper. The rule of thumb
   originates with Haneuse, VanderWeele & Arterburn 2019, JAMA
   ("Using the E-Value to Assess the Potential Effect of Unmeasured
   Confounders in Observational Studies").
4. **Smoking-cancer E-value**: Cornfield/Hammond benchmark is ~9, not
   ~2. If we cite it, get the number right.

### Higher-stakes, need more research
5. **NIE_chain term name**: Gao-Li-Luo (arXiv:2007.16031) §3.2 has
   a 9-term decomposition. Under no-interaction the 6 interaction
   terms vanish, leaving `CDE + PIE_M1 + PIE_M2` (THREE terms, not
   four). The "NIE_chain" term we compute is real (the M1→M2 chain
   pathway) but its name maps to NatINT_M1M2 in Gao-Li-Luo, which is
   an interaction effect — under no-interaction it should be zero.
   Either:
   - Find the actual reference where `NIE_chain` appears as a named
     PSE under no-interaction (possibly Daniel et al. 2015 Stat Med,
     or VanderWeele 2014), OR
   - Drop the NIE_chain term and verify the lock test
     `dual_mediation_no_interaction_residual_lock` still passes when
     we use the 3-term Gao-Li-Luo identification.
6. **Δ_S set notation in §3.2 lines 257-261**: not from
   Gao-Li-Luo. Either prove equivalence to their nested-
   counterfactual formulas in an appendix, or rewrite using their
   notation.
7. **Γ/Λ symbol vs Ohnishi-Li's γ_a, η_a**: explicitly reduce
   `Γ := sup_{m,b} γ_a(m,b)`, `Λ := sup_{m,b} η_a(m,b)` and cite this
   uniform-scalar reduction step.
8. **"This is the VanderWeele-Ding E-value scale"** on §3.3 — Ohnishi-Li
   γ_a is bridge-conditional, provably ≤ the VW-D E-value Γ. Different
   object.

## Files
- `/Users/playra/trios-railway/crates/trios-trainer-igla/papers/f2_methodology.md` §3.2 (lines 234-298), §3.3 (lines 299-360)
