# KernelForge Evidence References

**Purpose:** Single canonical source for all claims, numbers, and citations used across KernelForge documentation.

---

## CUDA Agent Paper (arXiv:2602.24286)

| Claim | Source | Location |
|-------|--------|----------|
| Pure RL collapsed at step 17 | CUDA Agent paper | Section 3.3, page 6 |
| CUDA tokens = 0.01% of pretraining data | CUDA Agent paper | Section 3.3 |
| RFT ablation: faster-than-compile drops to 49.8% without RFT | CUDA Agent paper | Table 2, page 8 |
| Discrete rewards +36.4pp over continuous | CUDA Agent paper | Table 2 comparison |
| Full system: 98.8% pass, 96.8% faster, 2.11× GM speedup | CUDA Agent paper | Table 2 |
| 4-stage training pipeline | CUDA Agent paper | Figure 3, page 5 |
| Reward function {-1, 1, 2, 3} | CUDA Agent paper | Equation 1, page 5 |
| 6K operator dataset synthesis | CUDA Agent paper | Section 3.1 |
| 83.77% are 2-op fusion | CUDA Agent paper | Section 3.1 |
| Beats Claude Opus 4.5 by ~40% on Level-3 | CUDA Agent paper | Abstract |
| 100% faster-than-compile on Level-1/2 | CUDA Agent paper | Abstract |
| 92% faster-than-compile on Level-3 | CUDA Agent paper | Abstract |

---

## WarpSpeed / DoubleGraph

| Claim | Source | Location |
|-------|--------|----------|
| 3.6× average speedup over cuGraph | doubleAI blog | March 3, 2026 |
| 55% of algorithms >2× faster | doubleAI blog | Results section |
| 18% of algorithms >10× faster | doubleAI blog | Results section |
| 192 kernel files per GPU target | DoubleGraph SKILLS.md | Section 1, Scale table |
| BFS TD→BU threshold N/20 | DoubleGraph SKILLS.md | Section 4.2 |
| BFS BU→TD threshold N/200 | DoubleGraph SKILLS.md | Section 4.2 |
| Louvain serial threshold N≤200 | DoubleGraph SKILLS.md | Section 5 |
| Louvain thread tier avg_degree < 8 | DoubleGraph SKILLS.md | Section 5 |
| Louvain warp tier avg_degree ≥ 8 | DoubleGraph SKILLS.md | Section 5 |
| CachePool 8 entries | DoubleGraph cache_pool.hpp | Implementation |
| 4-way dispatch pattern | DoubleGraph SKILLS.md | Section 3 |
| A100 L2 cache 40MB | NVIDIA A100 whitepaper | Specifications |
| L2 persistence 30MB (75% of L2) | DoubleGraph pattern | SKILLS.md |

---

## DeepSeek-R1 / GRPO

| Claim | Source | Location |
|-------|--------|----------|
| GRPO eliminates critic (~50% memory) | DeepSeek-R1 paper | Architecture section |
| Group-relative advantages | DeepSeek-R1 paper | Algorithm description |
| No value function needed | DeepSeek-R1 paper | Method comparison |

---

## MARS / Multi-Turn RL

| Claim | Source | Location |
|-------|--------|----------|
| MARS +28.7% on held-out games | MARSHAL (arXiv:2510.15414) | Abstract |
| MARS +10.0% on AIME | MARSHAL (arXiv:2510.15414) | Abstract |
| Turn-level cumulative returns | MARSHAL paper | Algorithm section |
| No critic required (GRPO-compatible) | MARSHAL paper | Method section |

---

## Hardware Specifications

| Claim | Source | Location |
|-------|--------|----------|
| A100: 108 SMs | NVIDIA A100 whitepaper | Specifications |
| A100: 40MB L2 cache | NVIDIA A100 whitepaper | Specifications |
| A100: 164KB shared memory per SM | NVIDIA A100 whitepaper | Specifications |
| A100: 65,536 registers per SM | NVIDIA A100 whitepaper | Specifications |
| A100: 2,039 GB/s HBM2e bandwidth | NVIDIA A100 whitepaper | Specifications |
| H100: 132 SMs | NVIDIA H100 whitepaper | Specifications |
| H100: 50MB L2 cache | NVIDIA H100 whitepaper | Specifications |
| H100: 228KB shared memory per SM | NVIDIA H100 whitepaper | Specifications |

---

## Memory Budgets (KernelForge Calculations)

| Claim | Source | Location |
|-------|--------|----------|
| H100 GPTQ path: ~53GB total | Research transcript | Memory budget calculation |
| B200 bf16 path: ~171GB total | Research transcript | Memory budget calculation |
| Unsloth cannot do GPTQ QLoRA for MoE | Research transcript | BitsAndBytes nn.Parameter limitation |
| 4-bit GPTQ reduces 80B to ~26GB | Calculation | Model size × 0.3 |

---

## Training Pipeline Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| LoRA rank | 16 | Truth.md Part 15 |
| Max sequence length | 8192 | Truth.md Part 15 |
| Warmup iterations | 50 | Truth.md Part 15 |
| Benchmark runs | 30 | Truth.md Part 15 |
| Speedup threshold | 1.05× | CUDA Agent paper |
| RFT filter threshold | reward ≥ 1.0 | Truth.md Part 8.3 |
| Curriculum promotion | 50% hit target | Truth.md Part 8.4 |
| Curriculum demotion | 20% positive | Truth.md Part 8.4 |

---

## Dataset Statistics

| Dataset | Size | Source |
|---------|------|--------|
| CUDA-Agent-Ops-6K | 6,000 operators | HuggingFace |
| KernelForge curated_200 | 200 problems | Truth.md Part 9 |
| Balance: easy | 50 | Truth.md Part 9 |
| Balance: medium | 75 | Truth.md Part 9 |
| Balance: hard | 75 | Truth.md Part 9 |

---

## How to Cite

When adding claims to documentation:

1. Add the claim to this file with source and location
2. Reference this file from other docs: "See [EVIDENCE_REFERENCES.md](EVIDENCE_REFERENCES.md)"
3. Do not duplicate the full table in other files

---

## Cross-References

This file replaces duplicate evidence tables in:
- `docs/KernelForge_Truth.md` Part 16
- `docs/KernelForge_Implementation_Spec.md` Appendix C
- `docs/research/CUDA_AGENT_DEEP_DIVE.md` Part 11
- ~~`docs/KernelForge_ARCH_TASKS.md` Part 6~~ (archived)
