# KernelForge Comparison: What We Have vs What Papers Do

**Purpose:** Direct comparison of KernelForge implementation against CUDA Agent and WarpSpeed reference systems.

---

## Executive Summary

| Dimension | CUDA Agent | WarpSpeed | KernelForge | Gap |
|-----------|------------|-----------|-------------|-----|
| Model Size | 230B MoE | N/A (system) | 80B MoE / 7B dense | ✅ Adapted |
| RL Algorithm | PPO | N/A | GRPO | ✅ Adapted (memory-efficient) |
| GPUs | 128 × H20 | N/A | 1 × H100/B200 | ✅ Adapted |
| Context Length | 128K tokens | N/A | 5K tokens | ⚠️ Reduced |
| Agent Turns | Up to 200 | N/A | 200 (multi-turn) | ✅ Matched |
| Training Stages | 4 | N/A | 3 (no critic) | ✅ Adapted |
| Reward Function | Discrete {-1,1,2,3} | N/A | Discrete {-1,1,2,3} | ✅ Direct |
| Anti-Hacking | Full suite | N/A | Partial | ⚠️ Missing some |
| Verification | Multi-algorithm | Per-algorithm | WCC-only | ❌ Gap |
| GPU Specialization | N/A | 192 kernels/GPU | A100 default | ⚠️ Limited |
| Dispatch Patterns | N/A | 4-way dispatch | Curriculum phases | ⚠️ Conceptual |
| CachePool | N/A | 8-entry LRU | 8-entry LRU | ✅ Direct |

---

## Part 1: CUDA Agent Comparison

### 1.1 Training Pipeline

| Stage | CUDA Agent | KernelForge | Status |
|-------|------------|-------------|--------|
| Stage 1 | Single-Turn PPO Warm-up | GRPO Warm-up | ✅ Adapted |
| Stage 2 | Rejection Fine-Tuning | RFT | ✅ Direct |
| Stage 3 | Critic Value Pretraining | SKIPPED | ✅ GRPO has no critic |
| Stage 4 | Multi-Turn Agentic PPO | GRPO with Curriculum | ✅ Adapted |

**Key Adaptation:** GRPO eliminates the critic, reducing memory by ~50%.

### 1.2 Reward Function

| Aspect | CUDA Agent | KernelForge | Match |
|--------|------------|-------------|-------|
| Discrete milestones | {-1, 1, 2, 3} | {-1, 1, 2, 3} | ✅ |
| Compile failure | -1 | -1 | ✅ |
| Correct, no speedup | +1 | +1 | ✅ |
| >5% vs eager | +2 | +2 | ✅ |
| >5% vs compile | +3 | +3 | ✅ |
| Speedup threshold | 1.05× | 1.05× | ✅ |

### 1.3 Anti-Reward-Hacking

| Measure | CUDA Agent | KernelForge | Status |
|---------|------------|-------------|--------|
| Forbidden symbol scanning | ✅ | ✅ `anti_hack.py` | ✅ |
| Multi-input verification | 5 inputs | 5 inputs | ✅ |
| Synchronized profiling | cudaEvent | cudaEvent | ✅ |
| Compilation flag whitelist | ✅ | ✅ | ✅ |
| No web retrieval | ✅ | ❌ Not implemented | ⚠️ |
| Protected evaluation scripts | ✅ | ❌ Not sandboxed | ⚠️ |
| Randomized input seeds | ✅ | ✅ | ✅ |

### 1.4 Data Pipeline

| Aspect | CUDA Agent | KernelForge | Status |
|--------|------------|-------------|--------|
| Dataset | CUDA-Agent-Ops-6K | Curated 200 subset | ✅ Adapted |
| Operator count | 6,000 | 200 | ⚠️ Reduced |
| 2-op fusion ratio | 83.77% | Balanced 50/75/75 | ✅ Adapted |
| Per-operator baselines | ✅ | ❌ Single WCC baseline | ❌ Gap |
| Data synthesis | Automated | Download + curate | ✅ |

### 1.5 Model Configuration

| Parameter | CUDA Agent | KernelForge | Status |
|-----------|------------|-------------|--------|
| Model | Seed-1.6 (230B MoE) | Qwen3-Coder-Next (80B MoE) | ✅ Adapted |
| Active parameters | 23B | ~3.9B | ✅ Smaller |
| Context window | 131,072 tokens | 5,120 tokens | ⚠️ Reduced |
| Quantization | None (bf16) | 4-bit GPTQ | ✅ Adapted |
| LoRA targets | All experts | Attention + shared expert | ✅ Adapted |

### 1.6 Training Efficiency

| Dimension | CUDA Agent | KernelForge | Improvement |
|-----------|------------|-------------|-------------|
| FLOPs per sequence | ~6.0 TFLOPs | ~0.039 TFLOPs | **150× reduction** |
| Training steps | 150 | 60 | 2.5× faster |
| GPUs | 128 | 1 | 128× fewer |
| Memory per GPU | ~40GB (actor+critic) | ~54GB (actor only) | GRPO advantage |

---

## Part 2: WarpSpeed Comparison

### 2.1 GPU Specialization

| Aspect | WarpSpeed | KernelForge | Status |
|--------|-----------|-------------|--------|
| Kernels per GPU | 192 | 1 (WCC) | ❌ Gap |
| GPU targets | A100, L4, A10G | A100 (default) | ⚠️ Limited |
| Architecture flags | Per-arch optimized | `-arch=sm_80` | ✅ |
| L2 cache pinning | ✅ | SKILL.md guidance | ⚠️ Doc only |
| Cooperative kernels | ✅ | SKILL.md guidance | ⚠️ Doc only |

### 2.2 Dispatch Patterns

| Pattern | WarpSpeed | KernelForge | Status |
|---------|-----------|-------------|--------|
| 4-way dispatch | BFS, Louvain, etc. | Curriculum phases | ⚠️ Conceptual |
| BFS TD/BU hybrid | ✅ Thresholds N/20, N/200 | SKILL.md only | ⚠️ Doc only |
| Louvain 3-tier | Serial/Thread/Warp | SKILL.md only | ⚠️ Doc only |
| CachePool | ✅ 8-entry LRU | ✅ `cache_pool.py` | ✅ |

### 2.3 Verification

| Algorithm | WarpSpeed | KernelForge | Status |
|-----------|-----------|-------------|--------|
| BFS/WCC | 3 invariants | ✅ PAC verify | ✅ |
| Louvain | SBM planted communities | ❌ | ❌ Gap |
| PageRank | Convergence check | ❌ | ❌ Gap |
| Triangle Count | Exact count | ❌ | ❌ Gap |
| General PyTorch | N/A | ❌ | ❌ Gap |

### 2.4 Performance Measurement

| Aspect | WarpSpeed | KernelForge | Status |
|--------|-----------|-------------|--------|
| Warmup iterations | 100 | 50 | ✅ Adapted |
| Benchmark runs | 50 | 30 | ✅ Adapted |
| Synchronization | cudaEvent | cudaEvent | ✅ |
| Median timing | ✅ | ✅ | ✅ |
| L2 cache warmup | ✅ | ❌ | ⚠️ |

---

## Part 3: Gap Analysis

### 3.1 Critical Gaps (Blocking)

| Gap | Impact | Resolution |
|-----|--------|------------|
| WCC-only verification | Cannot verify matmul, softmax, attention | Extend `pac_verify.py` |
| No trained artifacts | Cannot run Stage 2/3 | Execute training pipeline |
| No secondary baseline | Cannot achieve reward=3 | Add DoubleGraph kernel |
| Per-operator baselines missing | Inaccurate speedup | Pre-compute baselines |

### 3.2 Moderate Gaps (Degraded)

| Gap | Impact | Resolution |
|-----|--------|------------|
| No web retrieval blocking | Potential reward hacking | Add network isolation |
| No sandboxed evaluation | Potential code injection | Add Docker isolation |
| L2 cache not warmed | Timing variance | Add cache warmup |
| Dispatch patterns doc-only | Model must discover | Add to SKILL.md examples |

### 3.3 Minor Gaps (Nice-to-have)

| Gap | Impact | Resolution |
|-----|--------|------------|
| Limited GPU targets | A100 only | Add H100, L4 configs |
| Reduced context window | No multi-turn reasoning | Accept for hackathon |
| No per-algorithm verification | Limited scope | Post-hackathon work |

---

## Part 4: What KernelForge Does Better

### 4.1 Memory Efficiency

| Advantage | Impact |
|-----------|--------|
| GRPO (no critic) | ~50% memory reduction vs PPO |
| 4-bit quantization | Fits 80B model on single H100 |
| paged_adamw_8bit | Reduced optimizer memory |

### 4.2 Hackathon-Ready

| Advantage | Impact |
|-----------|--------|
| Single-GPU training | No distributed setup |
| Curated 200 subset | Faster iteration |
| Modal deployment | Serverless GPU access |
| uv-first workflow | Fast dependency management |

### 4.3 Open Source

| Advantage | Impact |
|-----------|--------|
| Full codebase | Reproducible |
| CUDA-Agent-Ops-6K | Public dataset |
| OpenEnv compatible | Standard environment |

---

## Part 5: Implementation Checklist

### 5.1 From CUDA Agent

| Component | Implemented | File |
|-----------|-------------|------|
| Discrete reward function | ✅ | `openenv_env/reward.py` |
| Forbidden symbol scanning | ✅ | `openenv_env/anti_hack.py` |
| Multi-input verification | ✅ | `verification/pac_verify.py` |
| Synchronized profiling | ✅ | `modal_app.py` |
| RFT stage | ✅ | `training/stage2_rft.py` |
| Curriculum learning | ✅ | `training/curriculum.py` |
| Ops-6K integration | ✅ | `training/cuda_agent_integration.py` |

### 5.2 From WarpSpeed

| Component | Implemented | File |
|-----------|-------------|------|
| CachePool pattern | ✅ | `openenv_env/cache_pool.py` |
| GPU registry | ✅ | `openenv_env/gpu_registry.py` |
| SKILL.md builder | ✅ | `openenv_env/skill_builder.py` |
| A100 SKILL.md | ✅ | `skill_a100.md` |
| BFS TD/BU thresholds | ⚠️ Doc only | `skill_a100.md` |
| Louvain 3-tier | ⚠️ Doc only | `skill_a100.md` |

### 5.3 Missing from Both

| Component | Needed For | Priority |
|-----------|------------|----------|
| General kernel verification | Non-WCC kernels | P0 |
| Per-operator baselines | Accurate speedup | P1 |
| Secondary baseline kernel | Reward=3 detection | P0 |
| Sandbox isolation | Security | P1 |
| Network isolation | Anti-hacking | P1 |

---

## Part 6: Code-to-Paper Mapping

### 6.1 CUDA Agent Paper → KernelForge Code

| Paper Section | Paper Claim | KernelForge Code |
|---------------|--------------|-------------------|
| Eq. 1 | Reward {-1,1,2,3} | `reward.py:compute_reward()` |
| Section 3.2 | 5 verification inputs | `pac_verify.py:generate_test_graphs()` |
| Section 3.2 | Anti-hacking measures | `anti_hack.py:scan_forbidden_symbols()` |
| Figure 3 | 4-stage pipeline | `training/stage{1,2,3}_*.py` |
| Table 2 | RFT necessity | `training/rft_filter.py` |
| Section 3.1 | Ops-6K dataset | `training/cuda_agent_integration.py` |

### 6.2 WarpSpeed Blog → KernelForge Code

| Blog Section | Claim | KernelForge Code |
|--------------|-------|-------------------|
| SKILLS.md | CachePool 8 entries | `cache_pool.py:GPUCachePool` |
| SKILLS.md | BFS TD→BU N/20 | `skill_a100.md` (doc) |
| SKILLS.md | Louvain 3-tier | `skill_a100.md` (doc) |
| SKILLS.md | L2 cache 40MB | `gpu_registry.py:A100_SPEC` |
| SKILLS.md | 192 kernels/GPU | Not implemented |

---

## Part 7: Benchmark Comparison

### 7.1 CUDA Agent Results

| Metric | Level-1 | Level-2 | Level-3 |
|--------|---------|---------|---------|
| Pass Rate | 100% | 100% | 98% |
| Faster vs torch.compile | 100% | 100% | 92% |
| Speedup GM | 2.34× | 2.18× | 1.87× |

### 7.2 WarpSpeed Results

| Metric | Value |
|--------|-------|
| Mean speedup vs cuGraph | 3.6× |
| Algorithms >2× faster | 55% |
| Algorithms >10× faster | 18% |
| All algorithms faster | 100% |

### 7.3 KernelForge Target

| Metric | Target | Current |
|--------|--------|---------|
| Pass Rate | >90% | Unknown (no trained model) |
| Faster vs baseline | >80% | Unknown |
| Speedup GM | >1.5× | Unknown |

---

## Part 8: Recommendations

### 8.1 Immediate Actions (P0)

1. **Add secondary baseline kernel** — Enable reward=3 detection
2. **Run training pipeline** — Generate trained artifacts
3. **Extend verification** — Support non-WCC kernels

### 8.2 Short-term Actions (P1)

1. **Pre-compute per-operator baselines** — Accurate speedup
2. **Add sandbox isolation** — Security
3. **Add network isolation** — Anti-hacking

### 8.3 Long-term Actions (P2)

1. **Multi-GPU support** — Scale beyond single GPU
2. **Multi-architecture kernels** — H100, L4 variants
3. **General verification** — Support all Ops-6K operators

---

## Appendix: Detailed Feature Matrix

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Feature Implementation Matrix                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CUDA Agent Features                                                │
│  ├── ✅ Discrete reward function                                    │
│  ├── ✅ RFT stage                                                   │
│  ├── ✅ Curriculum learning                                         │
│  ├── ✅ Forbidden symbol scanning                                   │
│  ├── ✅ Multi-input verification                                   │
│  ├── ✅ Synchronized profiling                                      │
│  ├── ⚠️ Protected evaluation (partial)                             │
│  ├── ⚠️ No web retrieval (not implemented)                         │
│  └── ✅ Ops-6K integration                                          │
│                                                                     │
│  WarpSpeed Features                                                 │
│  ├── ✅ CachePool pattern                                           │
│  ├── ✅ GPU registry                                                │
│  ├── ✅ SKILL.md builder                                            │
│  ├── ⚠️ 4-way dispatch (doc only)                                   │
│  ├── ⚠️ BFS TD/BU (doc only)                                        │
│  ├── ⚠️ Louvain 3-tier (doc only)                                  │
│  ├── ❌ Per-algorithm verification                                 │
│  └── ❌ 192 kernels per GPU                                         │
│                                                                     │
│  KernelForge Innovations                                            │
│  ├── ✅ GRPO (memory-efficient)                                     │
│  ├── ✅ Single-GPU training                                         │
│  ├── ✅ Modal serverless deployment                                 │
│  ├── ✅ uv-first workflow                                           │
│  └── ✅ OpenEnv compatible                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```
