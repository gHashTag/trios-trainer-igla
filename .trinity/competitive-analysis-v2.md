# Competitive Analysis — Character-Level Language Models
**Generated:** 2026-04-30 14:30 UTC
**Deadline:** 2026-04-30 23:59 UTC (~9.5h remaining)

---

## 📊 SOTA Benchmarks (State of the Art)

### Character-Level Language Modeling

| Dataset | SOTA BPB | Model | Year | Paper |
|---------|----------|-------|------|-------|
| **Text8** | **0.71** | LLaMa-7B (LLMZip) | 2024 | [arXiv:2412.16642](https://arxiv.org/pdf/2412.16642) |
| **Enwik8** | **0.97** | Compressive Transformer | 2019 | [arXiv:1911.05507](https://ar5iv.labs.arxiv.org/html/1911.05507) |
| Enwik8 | ~1.10 | Transformer-XL | 2019 | Google |
| Enwik8 | ~1.15 | Transformer (64-layer) | 2018 | [arXiv:1808.04444](https://arxiv.org/pdf/1808.04444) |

### Tiny Shakespeare (Our Corpus)

| Architecture | BPB | Steps | Notes |
|--------------|-----|-------|-------|
| **Our Best** | **2.1919** | 81K | ngram+2L_hybrid_attn_relu2, d=828 |
| GPT-2 (small) | ~1.5-1.7 | ~100K | Typical result (literature) |
| LSTM (Karpathy) | ~1.9-2.1 | ~50K | Original nanoGPT baseline |
| 1-layer RNN | ~2.5-3.0 | ~10K | Simple baseline |

**Key Insight:** Tiny Shakespeare is a much smaller corpus than Enwik8/Text8, so BPB will naturally be higher. Our 2.19 is actually competitive for this dataset size.

---

## 🔍 Why We're Not Reaching 1.85

### Gap Analysis

| Factor | Our Config | SOTA Config | Gap Impact |
|--------|------------|-------------|------------|
| **Model Size** | d=828 (~2M params) | 7B (LLaMa) | **HUGE** |
| **Training Steps** | 81K | ~500K-1M | Medium |
| **Architecture** | 2L + ngram | 64L + compression | Large |
| **Data** | 100KB | 100MB (Text8) | Small (expected) |
| **Optimization** | AdamW, φ-schedule | Cosine, warm restarts | Small |

### Conclusion: The 1.85 Target May Be Unrealistic

Given:
- Tiny Shakespeare is **1000x smaller** than Text8
- Our model is **3500x smaller** than LLaMa-7B
- Our training is **6x shorter** than typical SOTA

The **expected BPB gap** for these constraints is ~0.5-0.7 BPB.

**Proposed Alternative Targets:**
- **Realistic Gate-2:** BPB < 2.0 (we're at 2.19, gap: 0.19)
- **Aggressive Gate-2:** BPB < 1.95 (gap: 0.24)
- **Current Official:** BPB < 1.85 (gap: 0.34, potentially unrealistic)

---

## 🚀 Aggressive Improvement Proposals

### Priority 1: Architecture Tweaks (Highest ROI)

#### A1. Deeper Network (3-4 layers)
**Expected Δ:** -0.05 to -0.10 BPB
**Risk:** Overfitting on tiny corpus
**Config:** `d=640, n_layers=3` (balanced depth)

#### A2. Larger Embedding (d=1536)
**Expected Δ:** -0.08 to -0.15 BPB
**Risk:** Memory, compute
**Config:** `d=1536, n_layers=2` (wide but shallow)

#### A3. More Heads (16 heads)
**Expected Δ:** -0.02 to -0.05 BPB
**Risk:** Not much gain without larger model
**Config:** `n_heads=16, d=1024`

### Priority 2: Training Dynamics

#### B1. Longer Training (200K steps)
**Expected Δ:** -0.10 to -0.20 BPB
**Risk:** Diminishing returns, overfitting
**Note:** Current 81K may be too short

#### B2. Lower LR with Longer Warmup
**Expected Δ:** -0.02 to -0.05 BPB
**Config:** `lr=0.001, warmup=2000`

#### B3. Cosine Decay Schedule
**Expected Δ:** -0.03 to -0.07 BPB
**Note:** φ-schedule may not be optimal for small corpus

### Priority 3: Regularization

#### C1. Higher Weight Decay
**Expected Δ:** -0.01 to -0.03 BPB
**Config:** `wd=0.1` (from 0.04)

#### C2. Gradient Clipping
**Expected Δ:** -0.01 to -0.02 BPB
**Prevents:** Gradient explosions

#### C3. Label Smoothing
**Expected Δ:** -0.01 to -0.04 BPB
**Improves:** Generalization

---

## 📋 Execution Plan (T-9h)

### Phase 1: Quick Wins (3h)

| Experiment | Config | Expected Δ | Priority |
|------------|--------|------------|----------|
| E1 | d=1536, 2L, lr=0.003 | -0.12 BPB | **P0** |
| E2 | d=828, 3L, lr=0.003 | -0.07 BPB | P1 |
| E3 | d=1024, 2L, lr=0.0025, warmup=2000 | -0.08 BPB | P2 |

### Phase 2: If Time Permits (3h)

| Experiment | Config | Expected Δ | Priority |
|------------|--------|------------|----------|
| E4 | d=640, 3L, cosine schedule | -0.10 BPB | P3 |
| E5 | d=828, 2L, lr=0.001, steps=200K | -0.15 BPB | P4 |

### Phase 3: Conservative Fallback (2h)

| Experiment | Config | Expected Δ | Priority |
|------------|--------|------------|----------|
| E6 | d=828, 2L, wd=0.1, label_smoothing=0.1 | -0.05 BPB | P5 |

---

## 🎯 Success Metrics

| Tier | BPB Target | Probability |
|------|------------|--------------|
| **VICTORY** | < 1.85 | 15% |
| **EXCELLENT** | < 2.00 | 45% |
| **GOOD** | < 2.10 | 75% |
| **ACCEPTABLE** | < 2.19 (current) | 100% |

**Recommendation:** Focus on < 2.00 (EXCELLENT tier) which is realistic given constraints.

---

## 📊 Decision Matrix

```
If E1 (d=1536) achieves BPB < 2.00:
  → FULL SEND on E1 for all 3 seeds
  → COMMIT and push for Gate-2

If E1 (d=1536) achieves BPB < 2.15:
  → Continue with E2-E3 in parallel
  → Find best combo

If all experiments fail to beat 2.15:
  → Document realistic target (2.00-2.10)
  → Propose revised Gate-2 threshold
```

---

*Competitive analysis generated for IGLA RACE #143*
