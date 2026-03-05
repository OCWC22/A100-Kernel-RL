# KernelForge: Unified Specification

**Single source of truth. All other docs are archived.**

**Hackathon:** OpenEnv Hackathon SF — March 7-8, 2026 (SHACK15, San Francisco)
**Organizers:** Cerebral Valley + PyTorch/Meta + SHACK15
**Prize Pool:** $100K+ cash | Teams up to 4
**Last Updated:** March 4, 2026

---

## Table of Contents

0. [Scope Decisions (LOCKED)](#section-0-scope-decisions-locked)
1. [Project Overview](#section-1-project-overview)
2. [Evidence Base](#section-2-evidence-base)
3. [A100 Hardware Reference](#section-3-a100-hardware-reference)
4. [Compiler Stack & Inference Runtimes](#section-4-compiler-stack--inference-runtimes)
5. [Model Configuration](#section-5-model-configuration)
6. [OpenEnv Environment](#section-6-openenv-environment)
7. [SKILL.md (Agent Context)](#section-7-skillmd-agent-context)
8. [The Scarce Data Problem](#section-8-the-scarce-data-problem)
9. [3-Stage Training Pipeline](#section-9-3-stage-training-pipeline)
10. [SkyDiscover Integration](#section-10-skydiscover-integration)
11. [Data Pipeline](#section-11-data-pipeline)
12. [Evaluation & Gap Analysis](#section-12-evaluation--gap-analysis)
13. [Implementation Status](#section-13-implementation-status)
14. [Repository Structure](#section-14-repository-structure)
15. [Tech Stack & Dependencies](#section-15-tech-stack--dependencies)
16. [Key Constants & Thresholds](#section-16-key-constants--thresholds)
17. [Risk Register](#section-17-risk-register)
18. [Hackathon Timeline](#section-18-hackathon-timeline)
19. [Links & References](#section-19-links--references)
20. [Version History](#section-20-version-history)

---

## SECTION 0: SCOPE DECISIONS (LOCKED)

These decisions are final. Everything below follows from them.

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Target GPU (kernels)** | A100 (sm_80) only | Kernels generated target A100. Cross-compile with `-arch=sm_80` on whatever training GPU is available. |
| **Training GPU** | H100 80GB (primary) or B200 192GB (fallback) | Qwen3.5-35B-A3B needs >40GB VRAM. |
| **Model (GRPO training)** | Qwen3.5-35B-A3B (MoE, 3B active) or Qwen3.5-9B (dense) | Best bang-for-buck: 3B active params, full Qwen3.5 quality, Apache 2.0. |
| **Model (SkyDiscover mutator)** | GLM-5 744B via API ($1/M input) | Frontier LLM as semantic code mutator. ~$2-8 per evolution run. |
| **Fallback model** | Qwen2.5-Coder-7B-Instruct | If Qwen3.5 fails, instant fallback. 10-12GB QLoRA on any A100. |
| **RL algorithm** | GRPO via TRL GRPOTrainer | No critic needed. ~50% memory savings over PPO. |
| **Evolution algorithm** | SkyDiscover EvoX (meta-evolution) | Adaptive search replaces brute-force GRPO for kernel evolution. Parallel track. |
| **Training approach** | 3-stage: Warm-up → RFT → GRPO | CUDA Agent proved pure RL collapses at step 17. Non-negotiable. |
| **Scarce data strategy** | Curriculum + RFT + trajectory seeding | CUDA = 0.01% of pretraining data. Multiple mitigations required. |
| **Environment** | OpenEnv (openenv-core ≥0.2.1) | Hackathon framework. Gymnasium-style step/reset/state. |
| **Kernel scope** | Multi-algorithm: general CUDA ops from Ops-6K | 200-problem curated subset. |
| **Framework** | Unsloth + TRL ≥0.29.0 | Unsloth 2.5x MoE speedup, TRL rollout_func for multi-turn. |

---

## SECTION 1: PROJECT OVERVIEW

An RL post-training system + evolutionary search system that teaches models to generate optimized CUDA kernels targeting A100 (sm_80). We combine three novel approaches:

### The Three Novel Approaches

**1. CUDA Agent (ByteDance, arXiv:2602.24286):** Trained Seed1.6 (230B MoE) on 128 H20 GPUs to write CUDA kernels achieving 2.11× speedup over torch.compile. We adapt their 4-stage pipeline, discrete milestone rewards {-1,1,2,3}, Ops-6K dataset (6,000 operators), and anti-reward-hacking measures — all at single-GPU scale using GRPO instead of PPO.

**2. DoubleGraph / WarpSpeed (doubleAI, March 2026):** Replaced cuGraph with per-GPU-architecture optimized kernels. 192 CUDA files per GPU, 3.6× average speedup. We use their GPU CachePool, 4-way dispatch as curriculum, per-kernel .cu.flags, and A100-specific optimization techniques (L2 pinning, direction-optimizing BFS, warp primitives) encoded in our SKILL.md.

**3. SkyDiscover (UC Berkeley Sky Computing Lab, March 2026):** LLM-driven program evolution using AdaEvolve (adaptive improvement signals) and EvoX (meta-evolution of search strategies). Instead of brute-force GRPO rollouts, SkyDiscover uses a frontier LLM (GLM-5 API) to adaptively evolve kernels with ~80-120 LLM calls, costing $2-8 per run. Authors: Shu Liu, Mert Cemri, Shubham Agarwal et al., supervised by Ion Stoica, Matei Zaharia, Koushik Sen, Alex Dimakis.

### Combined Pipeline

```
CUDA-Agent Ops-6K → Initial kernel generation (model writes CUDA)
        ↓
SkyDiscover EvoX → Evolve kernels further (10-40% additional gains)
        ↓
DoubleGraph → Expert baselines for calibrated reward signal
        ↓
CUDA Graphs → Launch elimination for zero-overhead repeated execution
        ↓
Result: AI-written kernels that beat torch.compile on A100
```

### What "Success" Looks Like

**Minimum viable:** OpenEnv environment compiles/verifies/benchmarks CUDA kernels. 3-stage pipeline runs end-to-end. Evidence that GRPO changes reward distribution upward.

**Good:** Model consistently generates kernels with reward ≥ 2.0 (beats eager). Clear reward curve. SkyDiscover finds 1.5-2× speedups on specific operators.

**Great:** Model discovers optimization patterns through RL not present in SFT/RFT data. Kernels approach torch.compile performance. Dual-path (GRPO + SkyDiscover) both produce wins.

### Why This Wins the Hackathon

1. **Uses OpenEnv** — the hackathon's framework. Reusable, gym-compatible CUDA environments.
2. **Demonstrates RL post-training** — the hackathon's thesis. Actual policy improvement with validated methodology.
3. **Grounded in published evidence** — every design choice traces to an ablation or architectural pattern.
4. **Multi-algorithm scope** — WCC, operator fusion, general operators. Not a single-kernel demo.
5. **Dual-path: evolutionary + RL** — SkyDiscover for immediate wins, GRPO for policy improvement.

---

## SECTION 2: EVIDENCE BASE

Everything in this document traces to one of these sources. No speculation.

### 2.1 CUDA Agent Paper (arXiv:2602.24286, Feb 27 2026)

**What they did:** Trained Seed1.6 (230B MoE, 23B active) with agentic PPO on 128 H20 GPUs. Result: 2.11× geometric mean speedup over torch.compile, 98.8% pass rate, 96.8% faster-than-compile rate.

**4-stage pipeline (Figure 3, page 5):**

| Stage | What | Time | Purpose |
|-------|------|------|---------|
| 1. Single-Turn PPO | RL on base model | ~20% | Bootstrap CUDA syntax |
| 2. RFT | Filter trajectories, SFT | ~5% | Create behavioral prior. **This IS supervised learning.** |
| 3. Value Pretraining | Train critic | ~15% | **We skip this — GRPO has no critic.** |
| 4. Full Agentic PPO | Multi-turn RL, 200 turns | ~60% | Where optimization discovery happens |

**Pure RL failure (Section 3.3, page 6):** Collapsed at training step 17. Root cause: CUDA = <0.01% of pretraining data → token probability ~10⁻⁹ → importance sampling ratio explodes.

**Ablation results (Table 2, page 8):**

| Ablation | Pass Rate | Faster vs Compile | Speedup GM |
|----------|-----------|-------------------|------------|
| Full CUDA Agent | 98.8% | 96.8% | 2.11× |
| Without RFT | 95.6% | 49.8% | 1.05× |
| Without Value Pretraining | 98.6% | 50.9% | 1.00× |
| Without Agent Loop | 77.1% | 14.1% | 0.69× |
| Continuous rewards (not discrete) | 98.4% | 60.4% | 1.21× |

Key takeaways:
- Discrete rewards {-1,1,2,3} beat continuous by **+36.4pp** on faster-than-compile.
- Agent loop (multi-turn) contributes **+82.7pp** to faster-than-compile rate.
- Without RFT: reward collapses, entropy spikes.

**Reward function (Equation 1, page 5):**
```
r = -1   if compilation fails OR correctness fails
r = +1   if correct but no speedup (speedup ≤ 1.05×)
r = +2   if >5% faster than torch.eager
r = +3   if >5% faster than BOTH torch.eager AND torch.compile
```

**Anti-reward-hacking (Section 3.2):**
- Forbidden symbols: `torch`, `at::Tensor`, `c10::`, `triton`, `torch.compile`, `torch.nn.functional`
- 5 randomized inputs per evaluation (agent cannot memorize outputs)
- Synchronized profiling: `cudaDeviceSynchronize()` before/after benchmark
- 50 warmup iterations to prevent CUDA graph exploitation
- No web search: solutions from model weights + SKILL.md only

**Data synthesis (Section 3.1):**
- CUDA-Agent-Ops-6K: 6,000 operators, Apache 2.0, `BytedTsinghua-SIA/CUDA-Agent-Ops-6K`
- 83.77% are 2-op fusions
- Optimization examples: diagonal matmul 73× speedup, fused reduction 24× speedup

### 2.2 DoubleGraph / WarpSpeed (doubleAI, March 2-3 2026)

**What they did:** Replaced cuGraph's architecture-generic kernels with per-GPU optimized implementations. 192 CUDA kernel files per target GPU. Result: 3.6× average speedup, 55% of algorithms >2×, 18% at 10-100×.

**5-Layer Architecture:**

| Layer | What | Our Use |
|-------|------|---------|
| 1. Universal Graph Abstraction (compact_graph_t) | Fixed CSR/CSC type for all kernels | Fixed `extern "C"` kernel interface contract |
| 2. GPU Resource Management (CachePool) | LRU cache, 8 entries/thread, RAII | GPUCachePool in environment for GPU-resident test data |
| 3. 4-Way Dispatch | base × {segment, mask} variants, compile-time specialization | Curriculum progression: base → segmented → masked → advanced |
| 4. Per-GPU Kernel Implementations | Different algorithms per GPU (459-line BFS on A100 vs bitmap-only on L4) | Evidence that GPU-specific optimization needs different strategies |
| 5. Build System | GPU target at build time, .cu.flags sidecars | Agent learns `// CU_FLAGS:` as optimization dimension |

**A100-Specific Techniques (from DoubleGraph):**
1. Direction-Optimizing BFS: 6 kernels, cost-model switching (TD→BU at frontier > N/20, BU→TD at frontier < N/200)
2. Louvain 3-Tier Dispatch: serial (N≤200), thread+register hash (avg_degree<8), warp+SMEM hash (avg_degree≥8)
3. Hand-Rolled Warp Primitives: `__shfl_down_sync` for reductions (tighter register control than CUB)
4. Warp-Cooperative Output with `__ballot_sync`: parallel bit scanning for BFS frontiers
5. L2 Cache Pinning: 30MB of 40MB L2, drops latency from ~200 to ~30-40 cycles
6. cp.async Pipelining: `cuda::memcpy_async` for global→shared (sm_80 exclusive)
7. Cooperative Kernels: persistent threads with `cg::this_grid().sync()`, eliminates launch overhead
8. SpMV-Based PageRank: cuSPARSE + custom atomic reduction
9. WCC Union-Find: hook sampling, path splitting, non-atomic stores (post-Volta safe with `__syncwarp()`)

**The Meta-Pattern:** Characterize hardware → Profile generic kernels → Choose algorithm strategy → Tune parameters → Validate with per-kernel flags. SKILL.md encodes steps 1-2; the RL agent learns 3-5.

### 2.3 SkyDiscover (UC Berkeley Sky Computing Lab, March 3 2026)

**GitHub:** https://github.com/skydiscover-ai/skydiscover
**Papers:** AdaEvolve (arXiv:2602.20133), EvoX (arXiv:2602.23413)
**Authors:** Shu Liu, Mert Cemri, Shubham Agarwal et al. | Faculty: Ion Stoica, Matei Zaharia, Koushik Sen, Alex Dimakis

SkyDiscover reframes LLM-driven program evolution as **hierarchical zeroth-order optimization** over a discrete, non-differentiable fitness landscape. It treats fitness improvements as proxy gradients and evolves both solutions AND the search policy itself.

**AdaEvolve — "Adam for Code" (arXiv:2602.20133):**

Core equations:
- Normalized improvement: δ^(k)_t = max((f' - f*_k) / f*_k, 0)
- Accumulated signal (EMA): G^(k)_t = ρ · G^(k)_{t-1} + (1-ρ) · (δ^(k)_t)² (ρ ≈ 0.9)
- Adaptive exploration intensity: I^(k)_t = I_min + (I_max - I_min) · 1/(1 + √(G^(k)_t + ε))
  → High G = exploit (low I); low G = explore wildly.
- UCB bandit for island selection: k* = argmax_k (R_k/V_k + C·√(ln N / n_k))
- Meta-guidance trigger (stagnation): G^(k)_t ≤ τ_M = 0.12 for all k

Result: +34% median on 172 Frontier-CS tasks. Beats/matches AlphaEvolve on Circle Packing (2.636 vs human SOTA 2.634).

**EvoX — Meta-Evolution (arXiv:2602.23413):**

Inner loop evolves solutions; outer loop evolves the search strategy itself. The optimizer rewrites its own policy.
- Strategy output: C_S(D_t) → (parent, variation operator π, inspiration set)
- Strategy performance: J(S_t | D_t) = (s_end - s_start)/√W · log(1 + s_start)
- Strategy mutation: LLM conditioned on history + population descriptor

Pseudocode:
```python
while t < T:
    # Phase I: evolve solutions under current strategy S_t for window W
    s_start = max(D_t)
    for i in 1 to W:
        (parent, π, inspires) = C_{S_t}(D_t)
        x' = LLM_generate(parent, π, inspires)
        score, artifacts = evaluate(x')
        D_t.add(x')
    # Phase II: score strategy
    Δ = max(D_t) - s_start
    J = Δ * log(1+s_start) / sqrt(W)
    # Phase III: meta-evolve strategy on stagnation
    if Δ < τ:
        S_new = LLM_mutate_strategy(history, ϕ(D_t))
        if valid(S_new): S_t = S_new
```

Outperforms AlphaEvolve on 6/6 real systems tasks (e.g., 30.52 vs human 21.89 on Prism GPU placement).

**Systems wins:** 14% better MoE load balance, 29% lower KV-cache pressure, 41% lower cross-cloud transfer cost.

### 2.4 MARS Credit Assignment (arXiv:2510.15414, ICLR 2026)

**Problem:** Standard GRPO assigns the same advantage to every token in a trajectory — can't distinguish "Turn 1 fixed compile error" from "Turn 4 added irrelevant comment."

**Solution — Turn-level advantage:**
```
R_{i,k} = Σ_{k'=k}^{K} r_{i,k'}    # Cumulative return from turn k onward
A_{i,k} = (R_{i,k} - mean(R)) / std(R)    # Group-relative normalization
```

**Results:** +28.7% on held-out games, +10.0% on AIME, +6.6% on GPQA-Diamond. No extra model overhead. GRPO-compatible (no critic needed). ~50 lines to implement.

### 2.5 Master Evidence Table

| Claim | Source | Location |
|-------|--------|----------|
| Pure RL collapsed at step 17 | CUDA Agent | Section 3.3, p6 |
| CUDA tokens = 0.01% of pretraining | CUDA Agent | Section 3.3 |
| RFT ablation: drops to 49.8% without | CUDA Agent | Table 2, p8 |
| Discrete rewards +36.4pp over continuous | CUDA Agent | Table 2 |
| Full system: 98.8% pass, 2.11× GM | CUDA Agent | Table 2 |
| 4-stage pipeline | CUDA Agent | Figure 3, p5 |
| Reward {-1,1,2,3} | CUDA Agent | Equation 1, p5 |
| 6K operators, 83.77% 2-op fusion | CUDA Agent | Section 3.1 |
| 3.6× avg speedup over cuGraph | DoubleGraph | March 3 blog |
| 55% of algorithms >2× faster | DoubleGraph | Results |
| 18% of algorithms >10× faster | DoubleGraph | Results |
| 192 kernel files per GPU | DoubleGraph | SKILLS.md §1 |
| CachePool 8 entries LRU | DoubleGraph | cache_pool.hpp |
| BFS TD→BU at N/20, BU→TD at N/200 | DoubleGraph | SKILLS.md §4.2 |
| GRPO eliminates critic (~50% memory) | DeepSeek-R1 | Architecture |
| MARS +28.7% games, +10.0% AIME | MARSHAL | arXiv:2510.15414 |
| SkyDiscover +34% median on 172 tasks | AdaEvolve | arXiv:2602.20133 |
| EvoX beats AlphaEvolve 6/6 systems tasks | EvoX | arXiv:2602.23413 |
| A100: 108 SMs, 40MB L2, 164KB SMEM/SM | NVIDIA | A100 Whitepaper |
| A100: 2,039 GB/s HBM2e bandwidth | NVIDIA | A100 Whitepaper |

---

## SECTION 3: A100 HARDWARE REFERENCE

### 3.1 Core Specifications

| Property | A100 SXM (80GB) | Why It Matters |
|----------|------------------|----------------|
| Compute Capability | sm_80 | nvcc flag: `-arch=sm_80` |
| SMs | 108 | Grid launch target. Max active blocks = 108 × blocks/SM |
| FP32 CUDA Cores | 6,912 (64/SM) | 19.5 TFLOPS FP32 |
| Tensor Cores | 432 (4/SM, 3rd gen) | TF32: 156 TFLOPS. BF16: 312 TFLOPS |
| L2 Cache | **40 MB** | 5× V100. Enables L2 persistence pinning |
| Shared Memory/SM | **164 KB** (configurable) | Up from 96KB (V100). Key for tiling |
| Registers/SM | 65,536 (32-bit) | 256 regs/thread at 256 threads/block |
| HBM2e Bandwidth | **2,039 GB/s** | THE bottleneck for bandwidth-bound kernels |
| HBM Capacity | 80 GB | Model + eval buffers must fit |
| NVLink | 600 GB/s (12 links) | Multi-GPU only |
| TDP | 400W | Thermal throttling risk during sustained benchmarks |

### 3.2 Memory Hierarchy

```
┌──────────────────────────────────────────────────────────────────┐
│  Level          │ Capacity    │ Latency      │ Bandwidth        │
│─────────────────┼─────────────┼──────────────┼──────────────────│
│  Registers      │ 256KB/SM    │ 0 cycles     │ ~20 TB/s (est.)  │
│  Shared Memory  │ 164KB/SM    │ ~20 cycles   │ ~19 TB/s         │
│  L1 Cache       │ 192KB/SM    │ ~30 cycles   │ ~19 TB/s         │
│  L2 Cache       │ 40MB total  │ ~200 cycles  │ ~6 TB/s          │
│  HBM2e          │ 80GB total  │ ~400 cycles  │ 2.039 TB/s       │
└──────────────────────────────────────────────────────────────────┘
```

**Key insight:** A100's 40MB L2 is the game-changer. V100 had 6MB, Turing had 5.5MB. Algorithms with working set ≤40MB run almost entirely from L2 (~200 cycles vs ~400 cycles from HBM).

### 3.3 Roofline Model

```
Performance (TFLOPS)
    │
156 ├─────────────────────────────────── TF32 Tensor Core ceiling
    │                              ╱
 19 ├────────────────────── FP32 ceiling
    │                  ╱
    │               ╱
    │            ╱    Ridge point: ~76 FLOPs/byte (TF32)
    │         ╱                    ~10 FLOPs/byte (FP32)
    │      ╱                       ~153 FLOPs/byte (FP16)
    │   ╱
    │╱
    └──────────────────────────────────── Arithmetic Intensity (FLOPs/byte)
```

- GEMM at large sizes: compute-bound → optimize Tensor Core usage
- Elementwise ops (ReLU, add): memory-bound → optimize memory access
- Reductions (softmax, layernorm): memory-bound → optimize coalescing + shared memory
- Fused ops: increase arithmetic intensity by doing more compute per byte

### 3.4 A100-Exclusive Features with Code

**L2 Cache Persistence (sm_80 exclusive):**
```cuda
// Reserve 30MB of 40MB L2 for persistent caching
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 30 * 1024 * 1024);

cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr  = (void*)device_ptr;
attr.accessPolicyWindow.num_bytes = data_size_bytes;
attr.accessPolicyWindow.hitRatio  = 1.0f;
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```
Pin when: arrays accessed repeatedly, working sets ≤30MB, random-access patterns.
Don't pin: streaming access, working sets >>40MB, compute-bound kernels.

**cp.async Pipelining (sm_80 exclusive):**
```cuda
#include <cuda_pipeline.h>

// Old (sm_75): load through registers (2 instructions, register pressure)
smem[threadIdx.x] = global_data[threadIdx.x];

// New (sm_80): direct global→shared, no register usage
__pipeline_memcpy_async(&smem[threadIdx.x], &global_data[threadIdx.x], sizeof(float));
__pipeline_commit();
__pipeline_wait_prior(0);
```
One warp initiates async copy for NEXT tile while computing on CURRENT tile.

**Tensor Cores (3rd gen) — TF32 sweet spot:**

| Precision | Throughput (108 SMs) |
|-----------|---------------------|
| FP16/BF16 | ~312 TFLOPS |
| TF32 | ~156 TFLOPS |
| FP32 | ~19.5 TFLOPS |

Enable: `cublasGemmEx(..., CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);`

**Warp Primitives:**
```cuda
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
unsigned mask = __ballot_sync(0xffffffff, condition);
int count = __popc(mask);
```

**Cooperative Kernels (persistent threads):**
```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void persistent_kernel(/* args */) {
    cg::grid_group grid = cg::this_grid();
    for (int iter = 0; !converged; iter++) {
        process_phase(/* ... */);
        grid.sync();  // Grid-wide barrier
    }
}
// Launch: cudaLaunchCooperativeKernel(...)
// Requires: // CU_FLAGS: --rdc=true
```

**Shared Memory Configuration:**

| Config | Shared Memory | L1 Cache |
|--------|---------------|----------|
| Default | 48 KB | 144 KB |
| Medium | 100 KB | 92 KB |
| Maximum | **164 KB** | 28 KB |

Set: `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);`
Bank conflicts: pad `float tile[32][33]` to avoid 32-bank conflicts.

### 3.5 A100 vs H100 vs B200

| Feature | A100 (sm_80) | H100 (sm_90) | B200 (sm_100) |
|---------|-------------|-------------|---------------|
| SMs | 108 | 132 | 192 |
| HBM | 80 GB HBM2e | 80 GB HBM3 | 192 GB HBM3e |
| Bandwidth | 2.0 TB/s | 3.35 TB/s | 8.0 TB/s |
| L2 cache | 40 MB | 50 MB | 96 MB |
| Shared mem/SM | 192 KB | 228 KB | 228 KB |
| FP16 Tensor | 312 TFLOPS | 990 TFLOPS | 4,500 TFLOPS |
| TMA (hardware DMA) | NO | YES | YES |
| wgmma | NO | YES | YES |
| Thread Block Clusters | NO | YES | YES |
| sm_80 compatibility | ✓ baseline | ✓ runs sm_80 | ✓ runs sm_80 |

**Critical:** sm_80 kernels run on H100/B200 but don't use TMA/wgmma/clusters. sm_90 kernels do NOT run on A100.

### 3.6 Occupancy

```
Occupancy = Active Warps per SM / 64 (max warps)

Limiters:
1. Block size: 256 threads = 8 warps → need 8 blocks for 100%
2. Registers: 65536/SM. At 64 regs/thread, 256-thread block = 16384 → 4 blocks → 50%
3. Shared mem: 164KB/SM. At 48KB/block → 3 blocks → 37.5%
```

Rule of thumb: target 50-75% occupancy. Profile with `ncu --set full`.

**Register control:**
```cuda
__global__ void __launch_bounds__(256, 2) my_kernel(/* ... */) { }
// Or: // CU_FLAGS: --maxrregcount=64
```

---

## SECTION 4: COMPILER STACK & INFERENCE RUNTIMES

### 4.1 Compilation Pipeline

```
Python (torch.matmul)
    ↓
TorchDynamo (captures computation graph)
    ↓
FX Graph (intermediate representation)
    ↓
TorchInductor (backend compiler)
    ├──► Triton Kernels → Triton IR → LLVM IR → PTX → SASS
    ├──► CUDA C++ → nvcc → PTX → SASS
    └──► cuBLAS/cuDNN calls → Pre-compiled SASS
```

### 4.2 nvcc Flags for A100

```bash
nvcc -O3 \
     -arch=sm_80 \
     -lineinfo \
     -std=c++17 \
     --use_fast_math \
     --ptxas-options=-v \
     -Xptxas -O3 \
     -lcublas -lcudnn \
     kernel.cu -o kernel.out
```

### 4.3 Inference Runtimes

| Runtime | Use Case |
|---------|----------|
| **vLLM** | High-throughput serving with PagedAttention. Default for GRPO training rollouts. |
| **SGLang** | Faster for Qwen3.5, better memory management. Fallback if vLLM OOMs. |
| **TensorRT-LLM** | Maximum tokens/sec. Most complex setup. |

---

## SECTION 5: MODEL CONFIGURATION

### 5.1 Primary: Qwen3.5-35B-A3B (MoE)

| Property | Value |
|----------|-------|
| HuggingFace | `Qwen/Qwen3.5-35B-A3B` |
| Total params | 35B |
| Active per token | 3B |
| Architecture | Gated DeltaNet + Sparse MoE, 262K context |
| License | Apache 2.0 |
| GPQA Diamond | ~68 |

**VRAM (4-bit QLoRA + Unsloth on H100 80GB):**

| Component | Memory |
|-----------|--------|
| Model weights (4-bit) | ~17.5 GB |
| LoRA (r=16) + optimizer | ~2.5 GB |
| KV cache (8K context) | ~4 GB |
| Activations + gradients | ~8 GB |
| **Total** | **~32 GB** |
| **Remaining for CUDA sandbox** | **~48 GB** |

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-35B-A3B",
    max_seq_length=8192,
    dtype=None,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
```

### 5.2 Alternative: Qwen3.5-9B (Dense)

| Property | Value |
|----------|-------|
| HuggingFace | `Qwen/Qwen3.5-9B` |
| GPQA Diamond | **81.7** (beats GPT-OSS-120B) |
| VRAM (4-bit QLoRA) | ~14 GB total, ~66 GB remaining |
| Throughput | ~8-10 gens/min (~30-40 GRPO steps/hour) |

Best quality/size ratio. More GRPO steps per hour. Perfect for "show scaling behavior" comparison.

### 5.3 SkyDiscover Backend: GLM-5 via API

| Property | Value |
|----------|-------|
| Total params | 744B (40B active) |
| License | MIT |
| SWE-bench | 77.8% |
| API | $1.00/M input, $3.20/M output (Z.ai) |
| Use | LLM mutator for SkyDiscover evolution. NOT trained locally. |

```bash
export OPENAI_API_KEY=your-zai-key
export OPENAI_API_BASE=https://api.z.ai/v1

uv run skydiscover-run kernel.cu evaluator.py \
    --search evox --model glm-5 --iterations 100
```

### 5.4 Fallback: Qwen2.5-Coder-7B-Instruct

~10-12 GB QLoRA. Fits trivially on any A100. Instant fallback.

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-7B-Instruct",
    max_seq_length=8192, load_in_4bit=True, dtype=None,
)
```

### 5.5 Multi-Track Hackathon Plan

```
TRACK 1 (Background, overnight): GRPO training Qwen3.5-9B or 35B-A3B
TRACK 2 (Fast, 1-2 hours): SkyDiscover EvoX with GLM-5 API
TRACK 3 (If time): Run both models, compare scaling behavior

Run Track 2 first (uses API, minimal local GPU).
Run Track 1 overnight on H100.
```

---

## SECTION 6: OPENENV ENVIRONMENT

### 6.1 Architecture

```
┌─────────────────────────────────────────────────┐
│               Single A100 GPU                    │
│                                                  │
│  ┌────────────────────────┐                      │
│  │ Model (QLoRA 4-bit)    │  ~14-32 GB           │
│  └──────────┬─────────────┘                      │
│             │ Generate CUDA code                  │
│             ▼                                     │
│  ┌────────────────────────┐                      │
│  │ nvcc -arch=sm_80 -O3   │  CPU-side, ~3-5s     │
│  │ + .cu.flags applied    │                      │
│  └──────────┬─────────────┘                      │
│             ▼                                     │
│  ┌────────────────────────┐                      │
│  │ CachePool evaluation   │  ~1-2 GB GPU memory  │
│  │  • 5 randomized inputs │                      │
│  │  • PAC verification    │                      │
│  │  • 50 warmup + 30 runs │                      │
│  └──────────┬─────────────┘                      │
│             │ Discrete reward {-1,1,2,3}          │
│             ▼                                     │
│  ┌────────────────────────┐                      │
│  │ GRPO gradient step     │                      │
│  └────────────────────────┘                      │
│                                                  │
│  Total per step (4 gens): ~40-65 seconds         │
└─────────────────────────────────────────────────┘
```

### 6.2 OpenEnv API

```python
from openenv.core.env_server import Environment, create_fastapi_app
from openenv.core.types import StepResult

class KernelForgeEnv(Environment):
    def reset(self, problem: dict = None) -> dict:
        """Return initial observation (SKILL.md + problem description)."""

    def step(self, action: str) -> StepResult:
        """Accept CUDA code, return (observation, reward, done, info)."""

    @property
    def state(self) -> dict:
        """Return serializable environment state."""

app = create_fastapi_app(KernelForgeEnv)
```

### 6.3 GPU CachePool (from DoubleGraph cache_pool.hpp)

```python
class GPUCachePool:
    """LRU cache for GPU-resident evaluation data. Max 8 entries."""
    def __init__(self, max_entries: int = 8):
        self._max = max_entries
        self._cache = {}
        self._order = []

    def get_or_create(self, key: str, factory):
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        if len(self._cache) >= self._max:
            evict = self._order.pop(0)
            old = self._cache.pop(evict)
            if hasattr(old, '_gpu_arrays'):
                del old._gpu_arrays
        val = factory()
        self._cache[key] = val
        self._order.append(key)
        return val
```

### 6.4 Anti-Reward-Hacking

```python
ALLOWED_CU_FLAGS = {"--use_fast_math", "--extra-device-vectorization", "--rdc=true"}
ALLOWED_CU_FLAG_PREFIXES = ("--maxrregcount=",)
FORBIDDEN_SYMBOLS = ["torch", "at::Tensor", "c10::", "torch::autograd",
                     "triton", "torch.compile", "torch.nn.functional"]

def extract_cu_flags(cuda_code: str) -> list[str]:
    """Parse // CU_FLAGS: comments. Only whitelisted flags accepted."""
    flags = []
    for line in cuda_code.split('\n'):
        stripped = line.strip()
        if stripped.startswith('// CU_FLAGS:'):
            tokens = stripped.replace('// CU_FLAGS:', '').strip().split()
            for tok in tokens:
                if tok in ALLOWED_CU_FLAGS:
                    flags.append(tok)
                elif any(tok.startswith(p) for p in ALLOWED_CU_FLAG_PREFIXES):
                    try:
                        val = int(tok.split('=')[1])
                        if 16 <= val <= 128:
                            flags.append(tok)
                    except (ValueError, IndexError):
                        pass
    return flags

def scan_forbidden_symbols(so_path: str) -> Optional[str]:
    """Check compiled .so for forbidden library calls via nm -D."""
    result = subprocess.run(["nm", "-D", so_path], capture_output=True, text=True, timeout=5)
    for forbidden in FORBIDDEN_SYMBOLS:
        if forbidden in result.stdout:
            return f"Forbidden symbol detected: {forbidden}"
    return None
```

### 6.5 Reward Function

```python
def compute_reward(compiled, correct, speedup_vs_eager, speedup_vs_compile) -> float:
    """CUDA Agent Equation 1, validated +36.4pp over continuous (Table 2)."""
    if not compiled or not correct:
        return -1.0
    if speedup_vs_compile > 1.05:
        return 3.0   # Beats torch.compile
    if speedup_vs_eager > 1.05:
        return 2.0   # Beats eager
    return 1.0       # Correct but not faster
```

**Continuous variant (for SkyDiscover):**
```python
def compute_reward_continuous(compiled, correct, speedup_vs_compile) -> float:
    """Alternative for SkyDiscover where gradient signal matters less."""
    if not compiled or not correct:
        return -1e9
    return math.log(max(speedup_vs_compile, 0.01))
```

### 6.6 Step Pipeline

The `step()` method executes:
1. Parse `// CU_FLAGS:` from agent's code
2. Compile with `nvcc -arch=sm_80 -O3` + extracted flags
3. Load .so, scan for forbidden symbols via `nm -D`
4. Run on 5 randomized inputs, check correctness (atol=1e-3, rtol=1e-3)
5. Benchmark: 50 warmup + 30 timed runs, take median
6. Compute discrete reward {-1, 1, 2, 3}

Full implementation: `openenv_env/kernel_forge_env.py` (240 lines)

---

## SECTION 7: SKILL.md (AGENT CONTEXT)

Given to the agent as observation on every `reset()`. Encodes DoubleGraph's A100 optimization techniques.

```markdown
# SKILL.md — A100-Optimized CUDA Kernel Generation

## Target Hardware
- GPU: NVIDIA A100 (Ampere, sm_80)
- SMs: 108 | L2 Cache: 40 MB | Shared Memory/SM: 164 KB
- HBM2e: 2.0 TB/s | Registers/SM: 65,536
- nvcc: -arch=sm_80 -O3

## Your Task
Write an optimized CUDA C++ extension that accelerates the given
PyTorch operator. Your code runs on A100.

## Kernel Interface
Your code MUST define:
    extern "C" void optimized_forward(/* problem-specific args */);

## Per-Kernel Compilation Flags
You may specify compilation flags in a comment:
    // CU_FLAGS: --use_fast_math --maxrregcount=48
Allowed: --use_fast_math, --maxrregcount=N (16-128),
         --rdc=true, --extra-device-vectorization

## A100 Optimization Techniques (Priority Order)

### Priority 1: Algorithmic Reduction (>50% impact)
- Algebraic simplification: diag(A) x B = row-wise scaling (73x speedup)
- Kernel fusion: merge sequential ops, eliminate intermediate tensors
- Operator rearrangement: x @ (sum_j w_j^T) / 2 instead of sum_j (x @ w_j^T / 2) (24x)

### Priority 2: A100 Memory Hierarchy (20-50% impact)
- L2 cache pinning: 40MB L2, pin with cudaAccessPropertyPersisting
- Vectorized loads: float4 for 16-byte aligned access
- Memory coalescing: consecutive threads → consecutive addresses
- Shared memory tiling: pad float tile[32][33] to avoid bank conflicts
- cp.async pipelining: cuda::memcpy_async for global→shared (sm_80 exclusive)

### Priority 3: A100 Compute (10-20% impact)
- Warp primitives: __shfl_sync, __ballot_sync for intra-warp ops
- Occupancy tuning: __launch_bounds__(256, 4)
- Cooperative kernels: cg::this_grid().sync() (requires --rdc=true)
- TF32 for matmul: ~3x throughput via CUBLAS_COMPUTE_32F_FAST_TF32

### Priority 4: Library Integration
- cuBLAS for GEMM, cuDNN for Conv, cuSPARSE for SpMV

## Rules
- Do NOT modify verification or profiling scripts
- Do NOT call torch.nn.functional or torch.compile in kernels
- Do NOT use Triton or any kernel generation framework
- Do NOT hardcode outputs for specific inputs
```

---

## SECTION 8: THE SCARCE DATA PROBLEM

### 8.1 Why CUDA RL Is Hard

CUDA C++ = <0.01% of pretraining data. Token probability ~10⁻⁹ in BF16. Importance sampling ratio ρ_t(θ) fluctuates wildly → PPO/GRPO clipping can't contain ratios jumping 10⁻⁵ to 10⁵.

**CUDA Agent confirmed:** Pure RL collapsed at step 17. Reward cratered, entropy spiked.

### 8.2 Why Math RL Works But CUDA Doesn't

Math tokens: ~10⁻³ to 10⁻⁴ probability. Base model already "knows" math syntax. RL just improves reasoning.
CUDA tokens: ~10⁻⁹ probability. Knowledge is near-random noise in BF16.

DeepSeek-R1-Zero hit 71% on AIME 2024 from pure GRPO. No equivalent exists for CUDA.

### 8.3 Every Successful CUDA RL System Uses Warm-Up

| System | Model | Warm-Up | Result |
|--------|-------|---------|--------|
| CUDA Agent | 230B MoE | 4-stage pipeline | 2.11× speedup |
| cudaLLM-8B | Qwen3-8B | SFT → RL via verl | 79.75% Level-1 |
| Kevin-32B | QwQ-32B | SFT → RL | Published by Cognition |
| **Pure RL** | **230B** | **None** | **Collapsed at step 17** |

### 8.4 Our 5-Layer Mitigation Stack

1. **Trajectory Seeding** via cudaLLM-8B: generate candidates, evaluate, seed buffer with reward ≥ 1.0
2. **Stage 1 Warm-Up** on trivially easy kernels: bootstrap compilation rate ~50% → ~85%+
3. **RFT (mandatory)**: Filter for correct trajectories, SFT to anchor policy. Without it: 49.8% vs 96.8%.
4. **Curriculum Learning**: progressive difficulty adapted from DoubleGraph's 4-way dispatch
5. **Discrete Milestone Rewards**: {-1,1,2,3} beats continuous by +36.4pp

### 8.5 Decision Tree on Hackathon Day

```
Hour 0: Load model → Test 20 simple kernels
        ↓
    Compilation rate > 70%? ─── YES → Short Stage 1 (30 steps)
        │                        │
        NO                       ↓
        ↓                  Stage 2 RFT (mandatory) → Stage 3 GRPO
  Full Stage 1 (300 steps)
```

---

## SECTION 9: 3-STAGE TRAINING PIPELINE

### 9.1 Why 3 Stages Not 4

CUDA Agent used 4 stages. We skip Stage 3 (Value Pretraining) because GRPO has no critic.

```
CUDA Agent (230B, 128 GPUs)        KernelForge (3-9B active, 1 GPU)
Stage 1: Single-Turn PPO            Stage 1: Multi-Turn GRPO Warm-up
Stage 2: RFT                        Stage 2: RFT
Stage 3: Critic Pretraining          SKIPPED (GRPO = no critic)
Stage 4: Full Agentic PPO            Stage 3: Multi-Turn GRPO + Curriculum
```

### 9.2 Stage 1: GRPO Warm-up (Multi-Turn)

**Goal:** Bootstrap compilation rate ~50% → ~85%+.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max turns/episode | 3 | Short warm-up; most value in 1st error correction |
| Max steps | 300 | Budget for syntax learning |
| Dataset | 512 from Ops-6K (single-op) | Easy: vector_add, relu, softmax |
| Temperature | 0.9 | High exploration |
| Learning rate | 3e-6 | Prevent catastrophic forgetting |
| Num generations | 4 | Group-relative normalization |
| Total evaluations | ~3,600 | 300 × 4 gens × 3 turns |
| Wall-clock | ~4 hours | |

**Abort:** If <20% compilation after 100 steps, fall back.

### 9.3 Stage 2: RFT (Mandatory)

**Goal:** Filter Stage 1 trajectories, SFT on correct ones. Prevents entropy explosion.

| Parameter | Value |
|-----------|-------|
| Trajectories collected | 100 |
| Filter threshold | reward ≥ 1.0 |
| SFT epochs | 3 |
| Wall-clock | ~30 minutes |

CUDA Agent proved this is non-negotiable: without RFT, faster-than-compile drops 96.8% → 49.8%.

### 9.4 Stage 3: GRPO + Curriculum (Multi-Turn)

**Goal:** Learn optimization strategies through progressive difficulty.

| Parameter | Value |
|-----------|-------|
| Max turns/episode | 5 |
| Max steps | 200 |
| Temperature | 0.7 (exploit > explore) |
| Learning rate | 5e-6 |
| Total evaluations | ~4,000 |
| Wall-clock | ~5 hours |

**Curriculum phases:**

| Phase | Target Reward | Examples |
|-------|--------------|----------|
| single_ops | 1.0 | vector_add, relu, softmax, matmul |
| fusion_2op | 2.0 | LayerNorm+ReLU, MatMul+BiasAdd |
| arch_specific | 2.0 | WCC+L2 pinning, Reduction+cooperative |
| advanced | 3.0 | LayerNorm+GELU+Linear, Flash-Attention |

**Promotion:** >50% of last 10 rewards hit target → advance.
**Demotion:** <20% positive → regress.

### 9.5 MARS Credit Assignment (Multi-Turn)

Standard GRPO assigns same advantage to all tokens. MARS assigns per-turn:

```python
def mars_rollout(prompt, env, model, max_turns=5, gamma=1.0):
    turn_rewards = []
    for turn in range(max_turns):
        completion = model.generate(history)
        result = env.step(extract_cuda(completion))
        turn_rewards.append(result.reward if turn == max_turns-1
                           else compute_partial_reward(result))
        if result.reward >= 2.0: break
        history.append(feedback(result))

    # Cumulative returns (MARS)
    cumulative = []
    current = 0.0
    for r in reversed(turn_rewards):
        current = r + gamma * current
        cumulative.append(current)
    cumulative = cumulative[::-1]

    # Group-relative normalization
    advantages = [(R - mean_R) / std_R for R in cumulative]
    # Apply: every token in turn k gets advantages[k]
```

**Partial rewards (intermediate turns):**

| Outcome | Reward |
|---------|--------|
| Compile fails | -1.0 |
| Compiles, wrong output | -0.5 |
| Correct, no speedup | +0.5 |
| Correct, >5% faster | +1.0 |

Final turn uses canonical {-1, 1, 2, 3}.

### 9.6 Compute Budget

| Stage | Steps | Gens | Turns | Evals | Wall-clock |
|-------|-------|------|-------|-------|------------|
| Stage 1 | 300 | 4 | 3 | 3,600 | ~4 hrs |
| Stage 2 | N/A | N/A | 1 | 100 | ~30 min |
| Stage 3 | 200 | 4 | 5 | 4,000 | ~5 hrs |
| **Total** | | | | **7,700** | **~9-10 hrs** |

---

## SECTION 10: SKYDISCOVER INTEGRATION

### 10.1 Why SkyDiscover Over Brute-Force GRPO

Static evolutionary algorithms waste ~70-90% of LLM calls on stagnation. SkyDiscover treats fitness improvements as proxy gradients and evolves both solutions AND the search policy. Combined with frontier LLMs (GLM-5), this makes ~80-120 calls enough to find significant optimizations.

**Dual-path strategy:**
- **SkyDiscover** = immediate results (no training needed, 1-2 hours, API cost ~$2-8)
- **GRPO training** = internalized policy improvement (9-10 hours, model gets better)

Both paths use the same OpenEnv `env.step()` as the evaluation function.

### 10.2 AdaEvolve: Adaptive Improvement Signal

AdaEvolve uses a single accumulated improvement signal **G** (like Adam optimizer's second moment) to drive three-level adaptation:

**Local (per-island):** G high → exploit (small refinements). G low → explore (radical changes).
```
I(k)_t = I_min + (I_max - I_min) / (1 + √(G(k)_t + ε))
```

**Global (across islands):** UCB bandit allocates resources to most promising populations.
```
k* = argmax_k (R_k/V_k + C·√(ln N / n_k)),  C = √2
```

**Meta (all stagnant):** When G ≤ 0.12 for all islands, ask LLM to invent entirely new optimization tactics (e.g., "use SLSQP optimizer", "introduce Steiner tree heuristic").

### 10.3 EvoX: Self-Evolving Search Strategy

The optimizer literally rewrites its own search policy during execution:

```python
while t < T:
    # Phase I: Evolve solutions with current strategy
    for i in range(window_W):
        parent, π, inspires = strategy(population)
        child = LLM_generate(parent, π, inspires)
        score = evaluate(child)
        population.add(child)

    # Phase II: Score strategy
    improvement = max(population) - previous_best
    J = improvement * log(1 + previous_best) / sqrt(W)

    # Phase III: Meta-evolve strategy on stagnation
    if improvement < threshold:
        new_strategy = LLM_mutate_strategy(history, descriptor)
        if valid(new_strategy): strategy = new_strategy
```

**Discovered behaviors:** Random → greedy → stratified multi-objective → UCB + structural → local refine. On GPU placement, EvoX automatically rewrites as Best-Fit-Decreasing + binary-search threshold.

### 10.4 CUDA Kernel Evaluator

```python
# evaluator.py — connects SkyDiscover to our OpenEnv environment
import subprocess, re

def evaluate(program_path: str) -> dict:
    try:
        subprocess.run(["nvcc", "-O3", "-arch=sm_80", "-o", "kernel.out",
                        program_path], check=True, timeout=45)
        result = subprocess.run(["./kernel.out"], capture_output=True,
                               text=True, timeout=120)
        throughput = float(re.search(r"Throughput:\s*([\d.]+)", result.stdout).group(1))
        latency = float(re.search(r"Latency:\s*([\d.]+)", result.stdout).group(1))
        correctness = "PASS" in result.stdout
        score = throughput / (latency + 1e-9) if correctness else -1e9
        return {"combined_score": score,
                "artifacts": {"feedback": f"TP {throughput:.1f} | Lat {latency:.2f}ms"}}
    except Exception as e:
        return {"combined_score": -1e9, "artifacts": {"feedback": str(e)}}
```

### 10.5 Integration with OpenEnv

The same `env.step(cuda_code)` serves both SkyDiscover and GRPO:

```python
# skydiscover_evaluator.py — wrapper for SkyDiscover
from openenv_env.kernel_forge_env import KernelForgeEnv

env = KernelForgeEnv(target_gpu="a100")

def evaluate(program_path: str) -> dict:
    with open(program_path) as f:
        cuda_code = f.read()
    env.reset(problem=current_problem)
    result = env.step(cuda_code)
    return {
        "combined_score": result.reward,
        "artifacts": {"observation": result.observation},
    }
```

### 10.6 Combined Pipeline

```
Step 1: CUDA-Agent Ops-6K provides initial problems
Step 2: GRPO-trained model generates initial high-quality kernels
Step 3: SkyDiscover EvoX evolves the best kernels further (10-40% gains)
Step 4: DoubleGraph A100 baselines calibrate reward signal
Step 5: CUDA Graphs wrap the final optimized loop for zero-overhead replay

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  my_optimized_kernel<<<...>>>(...)  // from SkyDiscover + GRPO
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph);
for(int ep=0; ep<10000; ep++)
    cudaGraphLaunch(graphExec, stream);  // zero launch overhead
```

### 10.7 Run Commands

```bash
# Install SkyDiscover
pip install skydiscover
# or: uv sync --extra gpu_mode --extra external

# Run EvoX evolution with GLM-5 API
uv run skydiscover-run initial_kernel.cu evaluator.py \
    --search evox \
    --model glm-5 \
    --iterations 100 \
    --output hackathon_kernel_v1

# Alternative: use Claude for mutations
uv run skydiscover-run initial_kernel.cu evaluator.py \
    --search evox \
    --model claude-opus-4-6 \
    --iterations 80
```

### 10.8 Limitations

- Single scalar score (combine multi-objective manually)
- Evaluator must be fast (<few seconds) and deterministic
- Cost: $1-8 per 100-iteration run with GLM-5 API
- Stochastic — best-of-3 runs recommended
- Meta-layers can hallucinate invalid tactics (rare with frontier LLMs; fallback exists)

---

## SECTION 11: DATA PIPELINE

### 11.1 Source

CUDA-Agent-Ops-6K (Apache 2.0, `BytedTsinghua-SIA/CUDA-Agent-Ops-6K`). 6,000 operators. 83.77% are 2-op fusions. Each: PyTorch nn.Module with forward(), get_inputs(), get_init_inputs().

### 11.2 Curation

```python
def curate_subset(ops_6k, n=200):
    easy   = [op for op in ops_6k if op["num_ops"] == 1][:50]    # 50
    medium = [op for op in ops_6k if op["num_ops"] == 2][:75]    # 75
    hard   = [op for op in ops_6k if op["num_ops"] >= 3][:75]    # 75
    return easy + medium + hard
```

### 11.3 Baselines

Per-operator on A100: `eager_time_ms` (torch.eager, median of 30) and `compile_time_ms` (torch.compile, median of 30 after warmup). Files: `datasets/curated_200.jsonl`, `datasets/baselines.jsonl`.

---

## SECTION 12: EVALUATION & GAP ANALYSIS

### 12.1 Hypotheses

**H1:** Multi-stage RL improves kernel quality over base model.
**H2:** RFT is necessary (replicates CUDA Agent ablation).
**H3:** A100-specific SKILL.md improves over generic prompts.

### 12.2 Protocol

- 50 held-out operators not in training set
- 4 generations per operator, take best
- Compare: base → Stage 1 → Stage 2 → Stage 3
- Report: pass rate, reward distribution, speedup histogram

### 12.3 Gap Analysis

| Dimension | CUDA Agent | KernelForge |
|-----------|-----------|-------------|
| Model | 230B MoE, 23B active | 3-9B active |
| GPUs | 128 × H20 | 1 × H100 |
| Pipeline | 4-stage PPO | 3-stage GRPO |
| Turns | 200 per episode | 3-5 per episode |
| Data | 6,000 operators | 200 curated |
| Reward | Discrete {-1,1,2,3} | Same ✓ |
| Anti-hacking | Full suite | Same ✓ |
| Evolution | None | SkyDiscover EvoX (additional) |

---

## SECTION 13: IMPLEMENTATION STATUS

### Fully Implemented (22 components)

| Component | File |
|-----------|------|
| OpenEnv Environment | `openenv_env/kernel_forge_env.py` |
| Discrete Reward | `openenv_env/reward.py` |
| Anti-Reward-Hacking | `openenv_env/anti_hack.py` |
| GPU CachePool | `openenv_env/cache_pool.py` |
| GPU Registry | `openenv_env/gpu_registry.py` |
| Skill Builder | `openenv_env/skill_builder.py` |
| Modal evaluate_kernel | `modal_app.py` |
| Modal profile_baselines | `modal_app.py` |
| Stage 1: GRPO Warm-up | `training/stage1_warmup.py` |
| Stage 2: RFT | `training/stage2_rft.py` |
| Stage 3: GRPO Curriculum | `training/stage3_grpo.py` |
| Model Loader | `training/model_loader.py` |
| Curriculum Manager | `training/curriculum.py` |
| Multi-turn Rollout | `training/multi_turn_rollout.py` |
| RFT Filter | `training/rft_filter.py` |
| CUDA-Agent Integration | `training/cuda_agent_integration.py` |
| PAC Verification | `verification/pac_verify.py` |
| Curated 200 Dataset | `datasets/curated_200.jsonl` |
| Ops-6K Full Dataset | `datasets/ops6k_full.jsonl` |
| 13 test files | `tests/` |
| Baselines Dataset | `datasets/baselines.jsonl` |
| WCC Training Data | `datasets/wcc_training.jsonl` |

### Partially Implemented (6)

| Component | Issue |
|-----------|-------|
| Verification scope | WCC-specific, needs generalization |
| Baseline quality | Some operators missing compile baselines |
| RFT generation | Needs Modal connection for trajectory collection |
| Demo | Streamlit reference only |
| Multi-turn context | Context accumulation needs truncation logic |
| Dataset curation | Difficulty labels need validation |

### Missing (8)

| Component | Priority |
|-----------|----------|
| SFT checkpoint | P0 — needed for Stage 1 |
| RFT dataset (generated) | P0 — needed for Stage 2 |
| GRPO checkpoint | P0 — training output |
| Evaluation history | P1 |
| Secondary baseline comparisons | P1 |
| General kernel verification | P1 |
| Sandbox isolation | P2 |
| Network isolation | P2 |

**Readiness: ~70%**

### P0 Blocking Tasks

1. Test model loading on H100 (Qwen3.5-35B-A3B or 9B)
2. Validate Modal A100 connection end-to-end
3. Run Stage 1 warm-up (first 30 steps minimum)
4. Generate RFT dataset from Stage 1 trajectories
5. SkyDiscover evaluator wrapper (connect env.step to SkyDiscover)

---

## SECTION 14: REPOSITORY STRUCTURE

```
A100-Kernel-RL/
├── README.md
├── skill_a100.md                    # SKILL.md (agent context)
├── pyproject.toml                   # Dependencies & entry points
├── modal_app.py                     # Modal serverless GPU backend
├── Dockerfile
│
├── openenv_env/                     # OpenEnv RL Environment
│   ├── kernel_forge_env.py          # Main environment (240 lines)
│   ├── reward.py                    # Discrete {-1,1,2,3}
│   ├── anti_hack.py                 # Symbol scan + flag whitelist
│   ├── cache_pool.py                # LRU GPU cache (DoubleGraph)
│   ├── gpu_registry.py              # A100/H100/B200 specs
│   └── skill_builder.py             # Per-GPU SKILL.md generation
│
├── training/                        # 3-Stage Pipeline
│   ├── stage1_warmup.py             # Multi-turn GRPO warm-up
│   ├── stage2_rft.py                # Rejection fine-tuning
│   ├── stage3_grpo.py               # GRPO + curriculum
│   ├── model_loader.py              # Qwen3.5 / fallback loading
│   ├── curriculum.py                # 4-phase progression
│   ├── multi_turn_rollout.py        # MARS credit assignment
│   ├── rft_filter.py                # Trajectory collection
│   └── cuda_agent_integration.py    # Ops-6K integration
│
├── kernels/                         # Reference CUDA kernels
│   ├── baseline_wcc.cu
│   ├── ecl_cc_h100.cu
│   └── clustered_wcc_h100.cu
│
├── datasets/
│   ├── curated_200.jsonl            # 200 curated problems
│   ├── ops6k_full.jsonl             # Full CUDA-Agent-Ops-6K
│   ├── baselines.jsonl              # Pre-computed A100 baselines
│   └── wcc_training.jsonl           # WCC training data
│
├── verification/
│   └── pac_verify.py                # 3-invariant PAC verification
│
├── tests/                           # 13 test files (~1200 LOC)
│
├── demo/
│   └── streamlit_demo.py
│
└── docs/
    ├── KERNELFORGE_SPEC.md          # THIS FILE (single source of truth)
    └── archive/                     # All previous docs
```

---

## SECTION 15: TECH STACK & DEPENDENCIES

### Compatible Version Stack

| Package | Version | Notes |
|---------|---------|-------|
| **unsloth** | 2026.3.3 | Install with `--no-deps` (stale `trl<=0.24.0` cap) |
| **trl** | >=0.29.0 | Has `rollout_func` + `generate_rollout_completions` |
| **transformers** | >=4.56.2 | Per trl 0.29.0 requirements |
| **torch** | (determined by vllm) | Don't pin ourselves |
| **vllm** | >=0.10.2, <0.13.0 | Per `trl[vllm]` extra |
| **openenv-core** | >=0.2.1 | Hackathon framework |
| **cupy-cuda12x** | >=14.0 | GPU evaluation |
| **skydiscover** | >=0.1.0 | Evolutionary search |
| **modal** | >=0.70 | Remote GPU evaluation |

### Install

```bash
pip install --no-deps unsloth unsloth_zoo
pip install "trl[vllm]>=0.29.0"
pip install -e ".[train,modal]"
pip install skydiscover
```

---

## SECTION 16: KEY CONSTANTS & THRESHOLDS

| Constant | Value | Source |
|----------|-------|--------|
| nvcc arch flag | `-arch=sm_80` | A100 compute capability |
| A100 L2 cache | 40 MB | NVIDIA whitepaper |
| L2 persistence | 30 MB | DoubleGraph (75% of L2) |
| A100 SMEM/SM | 164 KB | NVIDIA whitepaper |
| A100 SMs | 108 | NVIDIA whitepaper |
| A100 HBM BW | 2,039 GB/s | NVIDIA whitepaper |
| Reward: compile fail | -1 | CUDA Agent Eq. 1 |
| Reward: correct, no speedup | +1 | CUDA Agent Eq. 1 |
| Reward: beats eager >5% | +2 | CUDA Agent Eq. 1 |
| Reward: beats compile >5% | +3 | CUDA Agent Eq. 1 |
| Speedup threshold | 1.05× | CUDA Agent Eq. 1 |
| Verify inputs | 5 | CUDA Agent §3.2 |
| Warmup iterations | 50 | Reduced from CUDA Agent's 100 |
| Benchmark runs | 30 | Reduced from 50 |
| Compile timeout | 30 seconds | Practical limit |
| CachePool entries | 8 | DoubleGraph |
| LoRA rank | 16 | Speed/quality balance |
| GRPO temperature (Stage 1) | 0.9 | Exploration |
| GRPO temperature (Stage 3) | 0.7 | Exploitation |
| GRPO beta (KL) | 0.0 | DAPO: not essential |
| GRPO max_completion | 4,096 tokens | Full CUDA kernel |
| Curriculum promotion | >50% of last 10 | DoubleGraph-inspired |
| Curriculum demotion | <20% positive | Prevent frustration |
| Dataset size | 200 (50/75/75) | Curated from Ops-6K |

---

## SECTION 17: RISK REGISTER

| Risk | Severity | Mitigation |
|------|----------|------------|
| Model won't load on hackathon GPU | HIGH | Pre-test. Instant fallback to Qwen2.5-Coder-7B |
| Pure RL collapse despite mitigations | HIGH | 5-layer stack. Monitor reward curve. Valid negative = ablation confirmation |
| vLLM incompatible with QLoRA | MEDIUM | Test locally. Fallback: transformers.generate |
| Modal cold-start latency | LOW | Batch evaluations, keep container warm |
| SkyDiscover API cost overrun | LOW | Cap at 120 iterations ($3-8). Use GLM-5 (cheapest frontier) |
| WiFi too slow for model download | HIGH | Pre-download all models to local drive |
| Context overflow in multi-turn | MEDIUM | Cap at 8192 tokens, truncate history |

---

## SECTION 18: HACKATHON TIMELINE

### Pre-Hackathon (before March 7)

- [ ] Download Qwen3.5-35B-A3B and Qwen3.5-9B models
- [ ] Download Qwen2.5-Coder-7B (fallback)
- [ ] Download CUDA-Agent-Ops-6K dataset
- [ ] Pre-compute baselines on A100
- [ ] Test model loading on H100
- [ ] Test environment end-to-end
- [ ] Set up SkyDiscover with GLM-5 API key
- [ ] Package as Docker container

### Day 1: March 7 (BUILD)

| Time | Activity |
|------|----------|
| Hour 0-1 | Claim GPU. Load model. Test 20 simple CUDA kernels. Decision: shorten Stage 1 or full warmup? |
| Hour 1-2 | **SkyDiscover Track:** Run EvoX with GLM-5 on 5-10 kernels. Immediate results. |
| Hour 2-6 | **GRPO Track:** Stage 1 warm-up (300 steps) |
| Hour 6-7 | Stage 2: RFT (filter, SFT) |
| Hour 7-12 | Stage 3: GRPO + curriculum |
| Evening | Evaluate: base → warm-up → RFT → GRPO. Compare with SkyDiscover results. |

### Day 2: March 8 (SHIP)

| Time | Activity |
|------|----------|
| Hour 0-2 | Final training + full evaluation suite |
| Hour 2-4 | Build demo: reward curve, kernel comparisons, SkyDiscover dashboard |
| Hour 4-5 | Push to HuggingFace Hub |
| Hour 5 | **Pitch (3 min):** "CUDA Agent + DoubleGraph + SkyDiscover at hackathon scale. Here's the reward curve. Here's a kernel the model discovered. Here's what SkyDiscover evolved. Dual-path: RL trains the model, evolution optimizes the kernels. Open-source OpenEnv environment anyone can use." |

---

## SECTION 19: LINKS & REFERENCES

### Papers

| Paper | Link |
|-------|------|
| CUDA Agent | https://arxiv.org/abs/2602.24286 |
| AdaEvolve | https://arxiv.org/abs/2602.20133 |
| EvoX | https://arxiv.org/abs/2602.23413 |
| DeepSeek-R1 (GRPO) | https://arxiv.org/abs/2501.12948 |
| MARS Credit Assignment | https://arxiv.org/abs/2510.15414 |

### Code & Data

| Resource | Link |
|----------|------|
| CUDA-Agent GitHub | https://github.com/BytedTsinghua-SIA/CUDA-Agent |
| CUDA-Agent Ops-6K | https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K |
| SkyDiscover GitHub | https://github.com/skydiscover-ai/skydiscover |
| SkyDiscover Blog | https://skydiscover-ai.github.io/blog.html |
| DoubleGraph GitHub | https://github.com/double-ai/doubleGraph |
| DoubleGraph Blog | https://doubleai.com/research/doubleais-warpspeed-surpassing-expert-written-kernels-at-scale |
| OpenEnv | https://github.com/meta-pytorch/OpenEnv |
| Unsloth | https://github.com/unslothai/unsloth |
| MARS Code | https://github.com/thu-nics/MARS |

### Models

| Model | Link |
|-------|------|
| Qwen3.5-35B-A3B | https://huggingface.co/Qwen/Qwen3.5-35B-A3B |
| Qwen3.5-9B | https://huggingface.co/Qwen/Qwen3.5-9B |
| GLM-5 | https://huggingface.co/zai-org/GLM-5 |
| GLM-5 API | https://api.z.ai |
| cudaLLM-8B | https://huggingface.co/ByteDance-Seed/cudaLLM-8B |
| Qwen2.5-Coder-7B | https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct |

---

## SECTION 20: VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| v1-v2 | Mar 3 | Initial research. Pure RL. WCC-only. |
| v3 | Mar 3 | H100/Hopper primitives. Overly ambitious. |
| v4 | Mar 3 | Pivot to A100. Practical hackathon scope. |
| v5 (PRD) | Mar 4 | CUDA Agent pipeline. DoubleGraph patterns. Qwen3-Coder-Next. |
| Truth.md | Mar 4 | Single source of truth. Deep A100 reference. Scarce data analysis. |
| **SPEC (this doc)** | **Mar 4** | **Consolidated all docs. Added SkyDiscover integration. Updated models to Qwen3.5. Dual-path strategy.** |

**Key corrections across versions:**
- "Pure RL" → Multi-stage (warm-up → RFT → GRPO). Pure RL collapses at step 17.
- WCC-only → Multi-algorithm from Ops-6K
- Qwen2.5-Coder-7B → Qwen3.5-35B-A3B/9B primary, Qwen2.5 as fallback
- GRPO-only → Dual-path: GRPO training + SkyDiscover EvoX evolution
- No SkyDiscover → Full integration with AdaEvolve + EvoX + GLM-5 API
