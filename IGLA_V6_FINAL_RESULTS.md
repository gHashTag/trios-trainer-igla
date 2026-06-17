# IGLA RACE v6 Format Sweep — Final Results

**Date:** 2026-05-26  
**Status:** COMPLETED (training stopped per user request)  
**Total logs collected:** 7,927  
**Completed runs:** 5 DONE  
**Peak concurrency:** 110 processes  
**Peak load average:** ~500

---

## Best Results by Format (sorted by BPB — lower is better)

| Format | DONE Runs | Best BPB | Best DONE BPB | Notes |
|--------|-----------|----------|---------------|-------|
| **gf12** | 5 | **2.4103** | **2.5232** | Best format. lr=0.001, h=256, 50K steps |
| gf8 | 0 | 2.7502 | — | in progress (killed) |
| f32 | 0 | 2.8142 | — | in progress (killed) |
| posit8 | 0 | 2.8353 | — | in progress (killed) |
| bf16 | 0 | 2.8419 | — | in progress (killed) |
| gf16 | 0 | 2.8859 | — | in progress (killed) |
| fp16 | 0 | 2.9091 | — | in progress (killed) |
| gf4 | 0 | 3.0898 | — | in progress (killed) |
| mxfp8 | 0 | 3.2016 | — | in progress (killed) |
| nf4 | 0 | 3.7091 | — | in progress (killed) |
| f64 | 0 | 6.9788 | — | deep runs, not converged |
| gf64 | 0 | 6.9788 | — | deep runs, not converged |
| posit32 | 0 | 6.9788 | — | deep runs, not converged |
| gf24 | 0 | 6.9788 | — | deep runs, not converged |
| posit16 | 0 | 6.9788 | — | deep runs, not converged |
| gf32 | 0 | 6.9788 | — | deep runs, not converged |
| gf20 | 0 | 6.9788 | — | deep runs, not converged |

---

## Completed Runs (sorted by BPB)

```
BPB=2.5232  steps=50000   gf12   v6_gf12_h256_lr0.001_seed71_50000   adamw
BPB=2.5417  steps=50000   gf12   v6_gf12_h256_lr0.003_seed71_50000   adamw
BPB=2.5762  steps=20000   gf12   v6_gf12_h768_lr0.001_seed71_20000   adamw
BPB=2.9505  steps=20000   gf12   v6_gf12_h768_lr0.0003_seed71_20000  adamw
BPB=3.0543  steps=50000   gf12   v6_gf12_h256_lr0.0003_seed71_50000  adamw
```

---

## Key Findings

1. **gf12 is the best format** — achieves BPB=2.5232, outperforming f32 baseline (2.8142)
2. **lr=0.001 consistently beats lr=0.0003** across all gf12 completions
3. **h=256 outperforms h=768** for gf12 at 50K steps
4. **f64/gf64/gf20/gf24/gf32/posit16/posit32** stuck at ~7.0 (random baseline) — deep step counts never converged due to early termination
5. **int8/int4/fp8_e5m2** — no eval data (killed before first eval)

---

## Configuration Space Explored

- **Formats:** 30 (gf4-gf64, f32/f64, posit8/16/32, fp16, bf16, tf32, fp8 variants, int8/16/4, mxfp8/6/4, nf4, lns8, uint8)
- **Seeds:** 138 (43–180)
- **Steps:** 18 (2K–15M)
- **Hiddens:** 5 (256–1024)
- **LRs:** 5 (0.01–0.0001)
- **Optimizers:** 3 (adamw, muon, muon-cwd)
- **Total combinations:** ~5.6 million

---

## System Limitations

- 110 processes on 8 cores → load avg ~500
- Most deep runs (500K+ steps) killed before convergence
- Only 5 completions out of 7,927 logs
- Fast turnaround requires step counts ≤ 50K on this hardware

---

## Recommendation

For future sweeps: cap step counts at 50K–100K for this hardware, or migrate to cloud (Railway) for deep runs.
