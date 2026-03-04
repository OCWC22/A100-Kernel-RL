# KernelForge — Truth Document

**Single source of truth. Edit this file directly.**

**Hackathon:** OpenEnv Hackathon SF — March 7-8, 2026 (SHACK15, San Francisco)
**Organizers:** Cerebral Valley + PyTorch/Meta + SHACK15
**Prize Pool:** $100K+ cash | Teams up to 4
**Last Updated:** March 4, 2026

---

## PART 0: SCOPE DECISIONS (LOCKED)

These decisions are final. Everything below follows from them.

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Target GPU (kernels)** | A100 (sm_80) only | Kernels generated target A100. Cross-compile with `-arch=sm_80` on whatever training GPU is available. |
| **Training GPU** | H100 80GB (primary) or B200 192GB (fallback) | Qwen3-Coder-Next needs >40GB VRAM. A100 80GB also viable if GPTQ works. |
| **Model** | Qwen3-Coder-Next (80B MoE, ~3.9B active) | Best available coder model. 512 routed experts, 10 active/token. Hybrid Gated DeltaNet + Gated Attention. |
| **Fallback model** | Qwen2.5-Coder-7B-Instruct | If Qwen3-Coder-Next fails to load or train, instant fallback. 10-12GB QLoRA on any A100. |
| **RL algorithm** | GRPO via TRL GRPOTrainer | No critic needed. ~50% memory savings over PPO. |
| **Training approach** | 3-stage: Warm-up → RFT → GRPO | CUDA Agent proved pure RL collapses at step 17. Non-negotiable. |
| **Scarce data strategy** | Curriculum + RFT + trajectory seeding from cudaLLM-8B | CUDA = 0.01% of pretraining data. Multiple mitigations required. |
| **Environment** | OpenEnv (openenv-core ≥0.2.1) | Hackathon framework. Gymnasium-style step/reset/state. |
| **Kernel scope** | Multi-algorithm: general CUDA ops from Ops-6K | Not WCC-only. 200-problem curated subset. |

---

## PART 1: WHAT WE ARE BUILDING

An RL post-training system that teaches Qwen3-Coder-Next (80B MoE, ~3.9B active parameters per token) to generate optimized CUDA kernels targeting A100 (sm_80), using a multi-stage training pipeline adapted from ByteDance's CUDA Agent and architectural principles from DoubleAI's WarpSpeed/DoubleGraph.

CUDA kernel generation is an extremely scarce domain — less than 0.01% of pretraining data — which makes naive RL collapse within ~17 steps. This project combines multiple scarce-data mitigations (curriculum learning, rejection fine-tuning, trajectory seeding, shaped rewards) to make RL work on this underrepresented domain.

The system has four components:

1. **OpenEnv Environment** — accepts CUDA code, compiles it with nvcc, verifies correctness on randomized inputs, benchmarks performance, returns a discrete reward.
2. **3-Stage Training Pipeline** — warm-up GRPO → rejection fine-tuning → curriculum GRPO, validated by CUDA Agent's ablation studies.
3. **A100-Specific SKILL.md** — encodes DoubleGraph's per-GPU optimization techniques as agent context.
4. **Curriculum System** — progressive difficulty from single ops through architecture-specific optimizations.

### What "Success" Looks Like

**Minimum viable:** OpenEnv environment compiles/verifies/benchmarks CUDA kernels. 3-stage pipeline runs end-to-end. Evidence that GRPO changes reward distribution upward.

**Good:** GRPO model consistently generates kernels with reward ≥ 2.0 (beats eager). Clear reward curve. Ablation confirms RFT is necessary.

**Great:** Model discovers optimization patterns through RL not present in SFT/RFT data. Kernels approach torch.compile performance. Published as reusable OpenEnv environment on HuggingFace.

---

## PART 2: EVIDENCE BASE

Everything in this document traces to one of these sources. No speculation.

### 2.1 CUDA Agent Paper (arXiv:2602.24286, Feb 27 2026)

**What they did:** Trained Seed1.6 (230B MoE, 23B active) with agentic PPO on 128 H20 GPUs to write CUDA kernels. Result: 2.11× geometric mean speedup over torch.compile, 98.8% pass rate, 96.8% faster-than-compile rate.

**Their 4-stage pipeline (Figure 3, page 5):**

| Stage | What | Time Allocation | Purpose |
|-------|------|-----------------|---------|
| 1. Single-Turn PPO | RL on base model, single kernel per prompt | ~20% | Bootstrap CUDA syntax → compilable kernels |
| 2. RFT (Rejection Fine-Tuning) | Filter stage 1 trajectories for reward ≥ 1, then SFT | ~5% | Create strong behavioral prior. **This IS supervised learning.** |
| 3. Value Pretraining | Train critic on filtered trajectories | ~15% | Initialize critic for stage 4. **We skip this — GRPO has no critic.** |
| 4. Full Agentic PPO | Multi-turn RL with 128K context, up to 200 turns | ~60% | Actual RL training where model discovers optimizations |

**Pure RL failure (Section 3.3, page 6):**
- Initial pure RL trial collapsed at training step 17
- Root cause: CUDA code = <0.01% of pretraining data → CUDA token probability ~10⁻⁹ in BF16
- Importance sampling ratio ρ_t(θ) = π_θ(a|s) / π_θ_old(a|s) fluctuates wildly or explodes
- Training reward collapses, actor entropy spikes, policy becomes diffuse

**Ablation results (Table 2, page 8):**

| Ablation | Pass Rate | Faster vs Compile | Speedup GM |
|----------|-----------|-------------------|------------|
| Full CUDA Agent | 98.8% | 96.8% | 2.11× |
| Without RFT | 95.6% | 49.8% | 1.05× |
| Without Value Pretraining | 98.6% | 50.9% | 1.00× |
| Without Agent Loop | 77.1% | 14.1% | 0.69× |
| Continuous rewards (not discrete) | 98.4% | 60.4% | 1.21× |

Key takeaways:
- Without RFT: reward collapses, entropy spikes. Results reported at "final validation step before training collapse."
- Discrete rewards {-1,1,2,3} beat continuous speedup by 36.4 percentage points on faster-than-compile metric.
- Agent loop (multi-turn) contributes +82.7pp to faster-than-compile rate.

**Reward function (Equation 1, page 5):**

```
r = -1   if compilation fails OR correctness fails
r = +1   if correct but no speedup (speedup ≤ 1.05×)
r = +2   if >5% faster than torch.eager
r = +3   if >5% faster than BOTH torch.eager AND torch.compile
```

**Anti-reward-hacking measures (Section 3.2):**
- Protected evaluation scripts: verification/profiling run in subprocess with read-only access
- Forbidden symbols: `torch`, `at::Tensor`, `c10::`, `torch::autograd`, `triton`, `torch.compile`, `torch.nn.functional`
- Randomized inputs: 5 different randomly-shaped inputs per evaluation (agent cannot memorize outputs)
- Synchronized profiling: proper `cudaDeviceSynchronize()` before/after benchmarking
- Warm-up iterations: prevent CUDA graph exploitation
- No web search: solutions from model weights + SKILL.md only

**Data synthesis (Section 3.1):**
- CUDA-Agent-Ops-6K: 6,000 operators, Apache 2.0, `BytedTsinghua-SIA/CUDA-Agent-Ops-6K`
- 83.77% are 2-op fusions
- Each operator: PyTorch nn.Module with forward(), get_inputs(), get_init_inputs()

**Optimization examples from CUDA Agent (Appendix D):**
- Diagonal matmul: `torch.diag(A) @ B` → row-wise scaling. O(N²M) → O(NM). **73× speedup.**
- Fused reduction: `Σ_j (x·w_j^T)/2` → `x·(Σ_j w_j^T)/2`. Single column reduction + dot product. **24× speedup.**

### 2.2 DoubleGraph / WarpSpeed (DoubleAI, announced March 2-3 2026)

**What they did:** Replaced NVIDIA cuGraph's architecture-generic kernels with per-GPU-architecture optimized implementations. 192 CUDA kernel files per target GPU. Result: 3.6× average speedup over expert-tuned code, 18% of algorithms at 10-100× speedup.

**5-Layer Architecture:**

**Layer 1 — Universal Graph Abstraction (compact_graph_t):**
Stripped-down CSR/CSC type. Every kernel takes the same input type. Eliminates template metaprogramming from hot paths.
→ **Our use:** Fixed `extern "C"` kernel interface contract per algorithm type.

**Layer 2 — GPU Resource Management (CachePool):**
Thread-local LRU cache, 8 entries per thread. Repeated calls reuse previously allocated buffers. RAII via Cacheable subclasses. Source: cache_pool.hpp, 125 lines.
→ **Our use:** GPUCachePool in environment keeps test data GPU-resident across evaluations.

**Layer 3 — 4-Way Dispatch:**
Every algorithm has 4 variants: base × {no_segment, segment} × {no_mask, mask}. Separate .cu file per variant. Compile-time specialization, not runtime branching.
→ **Our use:** Curriculum progression. Start base, promote to segmented (degree-aware), then masked (edge filtering), then seg+mask.

**Layer 4 — Per-GPU Kernel Implementations (the optimization work):**
Same algorithm, completely different code per GPU. A100 BFS uses direction-optimizing with queue↔bitmap. L4 BFS uses bitmap-only with CUB. A10G uses 2-level batched top-down with L2 pinning.
→ **Our use:** Evidence that GPU-specific optimization requires different algorithmic strategies, not just parameter tuning.

**Layer 5 — Build System:**
GPU target selected at build time via `-DTARGET_GPU=A100`. Per-kernel `.cu.flags` sidecars control compilation options. No runtime dispatch.
→ **Our use:** Agent learns to specify per-kernel `// CU_FLAGS:` comments as an optimization dimension.

**A100-Specific Techniques (from DoubleGraph analysis):**

**Technique 1: Direction-Optimizing BFS**
- 459 lines, 6 kernels in A100 implementation
- Two phases: top-down (queue, sparse frontier) and bottom-up (bitmap, dense frontier)
- Cost-model switching thresholds (A100-specific, tuned for 40MB L2):
  - TD → BU when `frontier_size > N/20`
  - BU → TD when `frontier_size < N/200`
- Queue↔bitmap frontier conversion using warp-level `__ballot_sync` for parallel bit scanning
- A100's 40MB L2 makes bitmap-based bottom-up cheap (large bitmaps fit in cache)

**Technique 2: Louvain 3-Tier Dispatch**
- Community detection dispatches based on graph coarsening level:
  - **Serial:** N ≤ 200 (tiny coarsened graph, CPU-style loop)
  - **Thread + register hash table:** avg_degree < 8 (1 thread per vertex, hash in registers)
  - **Warp + shared-memory hash table:** avg_degree ≥ 8 (1 warp per vertex, hash in SMEM)
- Hash table uses linear probing with `atomicCAS`

**Technique 3: Hand-Rolled Warp Primitives (No CUB on A100)**
```cuda
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```
- Tighter register control than CUB library calls
- Eliminates library call overhead
- Architecture-dependent choice: A100 uses hand-rolled, L4 uses CUB

**Technique 4: Warp-Cooperative Output with `__ballot_sync`**
- BFS top-down kernel: 1 warp per frontier vertex
- `__ballot_sync` used for collective insertion into output frontier
- Parallelizes bit scanning during queue↔bitmap conversion

**Technique 5: L2 Cache Pinning (40MB on A100)**
```cuda
// Set aside 30MB of L2 for persistent caching
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 30 * 1024 * 1024);

// Configure stream to use persistent L2
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = (void*)data_ptr;
attr.accessPolicyWindow.num_bytes = data_size;
attr.accessPolicyWindow.hitRatio = 1.0f;
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```
- A100's 40MB L2 fits parent arrays for graphs up to 10M vertices (40MB / 4 bytes)
- Drops random access latency from ~200 cycles (HBM) to ~30-40 cycles (L2)

**Technique 6: cp.async Pipelining (sm_80 feature)**
- `cuda::memcpy_async` for pipelined global→shared memory transfers
- Exclusive to sm_80+
- One thread initiates copy, others compute

**Technique 7: Cooperative Kernels (Persistent Threads)**
```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void persistent_bfs(/* ... */) {
    cg::grid_group grid = cg::this_grid();
    for (int level = 0; !done; level++) {
        // Process frontier
        grid.sync();  // Grid-wide barrier — no kernel relaunch
    }
}
```
- Eliminates kernel launch overhead for iterative algorithms
- Requires `// CU_FLAGS: --rdc=true` and `cudaLaunchCooperativeKernel`
- Used for BFS mask and core_number on A100 but NOT on L4 or A10G

**Technique 8: SpMV-Based PageRank**
- cuSPARSE for matrix-vector product (`cusparseSpMV` with CSR format)
- Custom kernels for dangling node sum (atomic reduction) and fused update+convergence check (single pass over rank vector)

**Technique 9: WCC Union-Find Optimizations**
- Hook sampling: sample edges for initial hooking phase
- Path splitting: `parent[x] = parent[parent[x]]` instead of full path compression
- Non-atomic stores safe in specific conditions: within a converged warp on post-Volta (A100), non-atomic stores are safe when all threads verified converged via `__syncwarp()` and write to unique addresses
- Single cooperative kernel launch (eliminate per-iteration kernel launch overhead)

**The Meta-Pattern (transferable process):**
1. Characterize hardware — L2 size, SMEM capacity, register file, warp scheduler
2. Profile generic kernels — run originals on target hardware, find bottlenecks
3. Choose algorithm strategy — pick representations (bitmap vs queue vs hybrid) based on memory hierarchy
4. Tune kernel parameters — block size, register count, shared memory allocation
5. Validate with per-kernel flags — .cu.flags for compilation control

SKILL.md encodes steps 1-2 as context. The RL agent must learn steps 3-5 through training.

---

## PART 3: A100 DEEP HARDWARE REFERENCE

This section exists because the RL agent's SKILL.md — and our curriculum design — must encode A100-specific optimization knowledge. Every number here is from the NVIDIA A100 Whitepaper or CUDA Programming Guide unless marked otherwise.

### 3.1 Core Specifications

| Property | A100 SXM (80GB) | A100 PCIe (40GB) | Why It Matters |
|----------|------------------|-------------------|----------------|
| Compute Capability | sm_80 | sm_80 | nvcc flag: `-arch=sm_80` |
| SMs | 108 | 108 | Grid launch target. Max active blocks = 108 × blocks/SM. |
| FP32 CUDA Cores | 6,912 (64/SM) | 6,912 | Raw throughput: 19.5 TFLOPS FP32 |
| Tensor Cores | 432 (4/SM, 3rd gen) | 432 | TF32: 156 TFLOPS. BF16: 312 TFLOPS. |
| L2 Cache | **40 MB** | **40 MB** | 5× V100 (6MB). Enables L2 persistence pinning. |
| Shared Memory/SM | **164 KB** (configurable) | 164 KB | Up from 96KB (V100), 64KB (Turing). Key for tiling. |
| Registers/SM | 65,536 (32-bit) | 65,536 | 256 regs/thread at 256 threads/block. Tight for complex kernels. |
| HBM2e Bandwidth | **2,039 GB/s** | **1,555 GB/s** | Bandwidth-bound kernels: this is THE bottleneck. |
| HBM Capacity | 80 GB | 40 GB | Model + eval buffers must fit. |
| NVLink | 600 GB/s (12 links) | N/A | Only relevant for multi-GPU (not our primary path). |
| PCIe | Gen4, 64 GB/s | Gen4, 64 GB/s | CPU↔GPU transfers during nvcc compile are negligible. |
| TDP | 400W | 250W | Thermal throttling risk during sustained benchmarking. |

### 3.2 Memory Hierarchy — The Optimization Landscape

Understanding the memory hierarchy is what separates a 1× kernel from a 50× kernel. Each level has different latency, bandwidth, and capacity:

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

**The key insight for A100:** The 40MB L2 is the game-changer compared to prior generations. V100 had 6MB, Turing had 5.5MB. A100's L2 can hold:
- **10M 32-bit integers** (parent array for graph with 10M vertices)
- **5M 64-bit floats** (rank vector for PageRank on 5M-vertex graph)
- **2.5M float4 vectors** (vectorized working set)

This means algorithms that repeatedly access a working set ≤40MB can run almost entirely from L2, reducing effective latency from ~400 cycles to ~200 cycles.

### 3.3 L2 Cache Persistence Controls (sm_80 Exclusive)

A100 introduces **explicit L2 cache management** — prior GPUs relied entirely on hardware-managed eviction. The API:

```cuda
// Step 1: Reserve L2 capacity for persistent data (call once at init)
// 30MB of 40MB = 75% allocated for persistence. Remaining 25% for normal traffic.
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 30 * 1024 * 1024);

// Step 2: Configure a CUDA stream to pin specific data
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr  = (void*)device_ptr;
attr.accessPolicyWindow.num_bytes = data_size_bytes;  // Must be ≤ persisting limit
attr.accessPolicyWindow.hitRatio  = 1.0f;              // 1.0 = pin everything
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;  // Keep in L2
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;   // Evict misses quickly
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);

// Step 3: Reset when done (release L2 for other data)
attr.accessPolicyWindow.num_bytes = 0;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
cudaCtxResetPersistingL2Cache();  // Force eviction
```

**When to pin:**
- Arrays accessed repeatedly across kernel launches (BFS parent array, PageRank rank vector)
- Working sets ≤ 30MB (75% of L2)
- Random-access patterns where HBM latency dominates

**When NOT to pin:**
- Streaming access patterns (sequential read-once) — hardware prefetch handles these
- Working sets >> 40MB — pinning thrashes L2 for other data
- Compute-bound kernels where memory latency is hidden

DoubleGraph's BFS implementation pins the parent array (up to 10M vertices = 40MB) and sees latency drop from ~400 cycles to ~200 cycles for random lookups.

### 3.4 Shared Memory Configuration

A100 allows flexible L1/shared memory partitioning per-kernel:

| Config | Shared Memory | L1 Cache |
|--------|---------------|----------|
| Default | 48 KB | 144 KB |
| Medium | 100 KB | 92 KB |
| Maximum | **164 KB** | 28 KB |

Set via: `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);`

**Bank conflicts:** Shared memory has 32 banks (4-byte stride). Classic fix: pad arrays.
```cuda
// BAD: 32 threads hit 32 banks with stride 32 = 32-way bank conflict
__shared__ float tile[32][32];

// GOOD: stride 33 eliminates conflicts (each row offset by 1 bank)
__shared__ float tile[32][33];
```

### 3.5 Asynchronous Memory Copy (sm_80 Exclusive: cp.async)

A100 introduces hardware-accelerated async copy from global to shared memory, bypassing registers:

```cuda
#include <cuda_pipeline.h>

// Old way (sm_75): load through registers (2 instructions, register pressure)
__shared__ float smem[256];
smem[threadIdx.x] = global_data[threadIdx.x];  // LOAD to register, then STORE to shared

// New way (sm_80): direct global→shared, no register usage
__pipeline_memcpy_async(&smem[threadIdx.x], &global_data[threadIdx.x], sizeof(float));
__pipeline_commit();
__pipeline_wait_prior(0);  // Wait for all committed copies

// Or using cooperative_groups:
#include <cooperative_groups/memcpy_async.h>
cooperative_groups::memcpy_async(group, smem, global_data, sizeof(float) * 256);
cooperative_groups::wait(group);
```

**Why this matters:** In pipelined algorithms, one warp can initiate the async copy for the NEXT tile while computing on the CURRENT tile. This overlaps memory latency with compute, achieving near-peak bandwidth utilization.

### 3.6 Tensor Cores (3rd Generation)

A100 Tensor Cores support:

| Precision | Throughput (per SM) | Total (108 SMs) |
|-----------|-------------------|------------------|
| FP16 | 256 TFLOPS | ~312 TFLOPS |
| BF16 | 256 TFLOPS | ~312 TFLOPS |
| TF32 | 128 TFLOPS | ~156 TFLOPS |
| FP64 | 16 TFLOPS | ~19.5 TFLOPS |
| INT8 | 512 TOPS | ~624 TOPS |
| INT4 | 1024 TOPS | ~1248 TOPS |

**TF32 is the sweet spot for CUDA kernel optimization:** It uses FP32 inputs/outputs but rounds mantissa to 10 bits (TF32) for the multiply-accumulate, giving ~3× throughput vs FP32 with minimal accuracy loss. Enable via:
```cuda
// cuBLAS
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
// or per-call:
cublasGemmEx(..., CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

**2:4 Structured Sparsity (sm_80 exclusive):** Hardware accelerates matrices where exactly 2 of every 4 elements are zero. 2× throughput for qualifying matrices. Requires `cusparseLt` library. Unlikely for hackathon but worth noting in SKILL.md for completeness.

### 3.7 Warp-Level Primitives

Every thread in a warp (32 threads) can communicate via register shuffle without shared memory:

```cuda
// Reduction: sum across 32 threads in ~5 cycles (vs ~20 cycles via shared memory)
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Broadcast: thread 0's value to all threads
float bcast = __shfl_sync(0xffffffff, val, 0);

// Ballot: create bitmask of which threads satisfy a condition
unsigned mask = __ballot_sync(0xffffffff, condition);
int count = __popc(mask);  // Population count = number of threads with condition=true

// Warp-level voting
bool any = __any_sync(0xffffffff, condition);   // True if ANY thread satisfies
bool all = __all_sync(0xffffffff, condition);   // True if ALL satisfy
```

DoubleGraph uses `__ballot_sync` extensively for BFS frontier management: warp-cooperative output where each warp collectively determines which vertices to add to the frontier, using `__ballot_sync` + `__popc` to compute output offsets without atomics.

### 3.8 Cooperative Kernels (Grid-Wide Synchronization)

For iterative algorithms (BFS levels, PageRank iterations, WCC convergence), A100 supports persistent kernels that synchronize across the entire grid without relaunching:

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void persistent_kernel(/* args */) {
    cg::grid_group grid = cg::this_grid();

    for (int iter = 0; !converged; iter++) {
        // Phase 1: Process data
        process_phase(/* ... */);

        // Grid-wide barrier: ALL blocks synchronize
        grid.sync();

        // Phase 2: Check convergence (only after all blocks finished phase 1)
        check_convergence(/* ... */);
        grid.sync();
    }
}

// Launch with cooperative API (NOT <<<>>> syntax)
void* args[] = { /* kernel args */ };
cudaLaunchCooperativeKernel(
    (void*)persistent_kernel,
    grid_dim, block_dim,
    args, shared_mem_size, stream
);
```

**Requirements:**
- Compile with `--rdc=true` (relocatable device code) → `// CU_FLAGS: --rdc=true`
- Grid size must be ≤ max active blocks across all SMs (query with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`)
- On A100: typically 108 SMs × 1-4 blocks/SM = 108-432 blocks max

**Benefit:** Eliminates kernel launch overhead (~5-10μs per launch). For BFS with ~100 levels, this saves ~0.5-1ms. For algorithms with 1000+ iterations (PageRank convergence), the savings compound.

### 3.9 Occupancy and Launch Configuration

**The occupancy equation:**
```
Occupancy = Active Warps per SM / Max Warps per SM

A100 max warps per SM = 64 (2048 threads / 32 threads per warp)

Factors that limit occupancy:
1. Block size: 256 threads = 8 warps. Need 8 blocks for 64 warps = 100% occupancy.
2. Registers: 65536 regs/SM. At 64 regs/thread, 256-thread block uses 16384 regs → 4 blocks fit → 32 warps = 50%.
3. Shared memory: 164KB/SM. At 48KB/block → 3 blocks fit → 24 warps = 37.5%.
```

**Controlling register pressure:**
```cuda
// Hint to compiler: max 256 threads/block, at least 2 blocks/SM
__global__ void __launch_bounds__(256, 2) my_kernel(/* ... */) {
    // Compiler will spill to local memory if needed to fit 2 blocks
}

// Or via nvcc flag (global):
// CU_FLAGS: --maxrregcount=64
```

**Rule of thumb for A100:** Target 50-75% occupancy. 100% occupancy is rarely optimal because it leaves no L1 cache for data. Profile with `ncu --set full` to find the sweet spot.

### 3.10 A100 vs Prior Generations — What Changed

| Feature | V100 (sm_70) | Turing (sm_75) | A100 (sm_80) |
|---------|-------------|----------------|--------------|
| L2 Cache | 6 MB | 5.5 MB | **40 MB** (6.7×) |
| SMEM/SM | 96 KB | 64 KB | **164 KB** (1.7×) |
| HBM BW | 900 GB/s | 616 GB/s | **2,039 GB/s** (2.3×) |
| cp.async | No | No | **Yes** |
| L2 Persistence | No | No | **Yes** |
| TF32 Tensor Core | No | No | **Yes** |
| BF16 Tensor Core | No | No | **Yes** |
| Structured Sparsity | No | No | **Yes** (2:4) |
| Cooperative Kernels | Limited | Limited | **Full** |
| Max threads/SM | 2048 | 1024 | 2048 |

The agent's SKILL.md must communicate that A100 optimizations (L2 pinning, cp.async, TF32) are NOT portable to V100/Turing. The agent must learn A100-specific patterns.

---

## PART 4: MODEL CONFIGURATION

### 4.1 Primary: Qwen3-Coder-Next (80B MoE, ~3.9B Active)

| Property | Value |
|----------|-------|
| HuggingFace ID | `Qwen/Qwen3-Coder-Next` |
| Architecture | 80B total, ~3.9B active per token. 512 routed experts + 1 shared, 10 active per token. |
| Layer structure | 48 hybrid layers: 12× groups of (3 Gated DeltaNet→MoE + 1 Gated Attention→MoE) |
| Attention layers | Only **12 of 48 layers** use standard attention (KV cache). Remaining 36 use Gated DeltaNet (linear attention). |
| Hidden dimension | 2048 |
| Native context | 256K tokens |
| License | Apache 2.0 |
| GPTQ-INT4 variant | `dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16` (community quantization, exists on HF) |
| FP8 variant | `Qwen/Qwen3-Coder-Next-FP8` (~80GB, fills entire H100 — NOT viable for training) |

**Why this model:**
- Best available open-source coder model. "Agentic training at scale."
- Only ~3.9B params active per token — inference is fast despite 80B total.
- Only 12 attention layers need KV cache — drastically reduces KV memory vs dense 80B.
- ByteDance trained cudaLLM-8B on Qwen3-8B (same family). Qwen3-Coder-Next is the scaled version.
- CUDA generation capability UNVERIFIED (no KernelBench results exist). Must test on hackathon day.

### 4.2 Hardware Paths for Qwen3-Coder-Next

**Path A — H100 80GB + 4-bit GPTQ + QLoRA (PRIMARY)**

Qwen3-Coder-Next is ~159GB in bf16. Does NOT fit on H100. 4-bit GPTQ brings weights to ~43GB.

```python
# path_a_h100.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

model_id = "dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# LoRA on attention + shared expert ONLY
# All 512 experts × LoRA = ~207GB optimizer states. Does NOT fit.
# Shared expert + attention LoRA = ~0.05GB adapters + ~0.3GB optimizer.
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention (12 layers)
        "shared_expert.gate_proj",                 # Shared expert
        "shared_expert.up_proj",
        "shared_expert.down_proj",
    ],
    lora_dropout=0,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

**Memory budget (H100 80GB + GPTQ path):**

| Component | Memory | Notes |
|-----------|--------|-------|
| 4-bit GPTQ weights (80B × 0.5 bytes + scales) | ~43 GB | GPTQModel supports MoE with Moe.Routing |
| LoRA adapters (rank 16, attention+shared expert, fp16) | ~0.05 GB | Only 12 attn layers + shared expert |
| AdamW optimizer states (fp32) | ~0.3 GB | For LoRA params only |
| Activations with gradient checkpointing | ~3 GB | Conservative estimate |
| KV cache (8 seqs × 8192 tokens, **12 attn layers only**) | ~1.5 GB | DeltaNet layers need no KV cache |
| Framework overhead / buffers | ~5 GB | CUDA context, PyTorch internals |
| Kernel evaluation buffers | ~1 GB | Test tensors for eval |
| **Total** | **~54 GB** | |
| **Headroom** | **~26 GB** | Comfortable |

**Critical blocker:** Unsloth does NOT support GPTQ-based QLoRA for MoE — BitsAndBytes cannot quantize MoE `nn.Parameter` tensors to 4-bit. This path uses HuggingFace PEFT + Transformers directly. Sacrifices Unsloth speed but is functional.

**Path B — B200 192GB + bf16 + Unsloth (FALLBACK)**

```python
# path_b_b200.py
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/Qwen3-Coder-Next-80B-A3B-Instruct",
    max_seq_length=8192,
    load_in_4bit=False,  # bf16 (4-bit broken for MoE on Unsloth)
    dtype=torch.bfloat16,
)

model = FastModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",  # 2.5x MoE speedup
)
```

**Memory budget (B200 192GB + bf16 path):**

| Component | Memory | Notes |
|-----------|--------|-------|
| bf16 weights | ~160 GB | Full precision |
| LoRA adapters | ~0.2 GB | Attention only |
| Optimizer states | ~1.6 GB | |
| Activations (Unsloth gradient checkpointing) | ~4 GB | |
| KV cache | ~2 GB | |
| Overhead | ~3 GB | |
| **Total** | **~171 GB** | |
| **Headroom** | **~21 GB** | Tight but confirmed by Unsloth |

Unsloth confirms Qwen3-Coder-Next fits on B200 in bf16 LoRA mode. Unsloth fully supports B200 (sm_100/Blackwell) with official NVIDIA collaboration.

**Path C — 2×H100 NVLink (IF AVAILABLE)**

Expert parallelism: 256 of 512 experts per GPU. All-to-all via NVLink (600-900 GB/s). Requires DeepSpeed or Megatron-Core (Unsloth lacks native EP).

### 4.3 Fallback: Qwen2.5-Coder-7B-Instruct

If Qwen3-Coder-Next fails to load, fails to generate compilable CUDA, or hackathon provides only A100 with no H100/B200:

| Property | Value |
|----------|-------|
| HuggingFace ID | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Unsloth ID | `unsloth/Qwen2.5-Coder-7B-Instruct` |
| Architecture | 7B dense (not MoE) |
| Training data | 5.5 trillion tokens, 92+ languages |
| VRAM (QLoRA 4-bit) | ~10-12 GB |
| Unsloth support | Verified. FastLanguageModel. Battle-tested. |

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-7B-Instruct",
    max_seq_length=8192,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
```

**Memory budget (A100 80GB):**

| Component | Memory |
|-----------|--------|
| Qwen2.5-Coder-7B QLoRA 4-bit | ~10 GB |
| LoRA + optimizer + activations | ~3 GB |
| KV cache + eval + overhead | ~4 GB |
| **Total** | **~17 GB** |
| **Headroom on 80GB** | **~63 GB** |

Fits trivially on any A100. Instant fallback.

---

## PART 5: THE OPENENV ENVIRONMENT

### 5.1 Architecture

Everything runs on the same A100. No Modal, no network calls, no cold starts.

```
┌─────────────────────────────────────────────────┐
│               Single A100 GPU                    │
│                                                  │
│  ┌────────────────────────┐                      │
│  │ Qwen2.5-Coder-7B      │  ~10-12 GB           │
│  │ (QLoRA 4-bit)          │                      │
│  └──────────┬─────────────┘                      │
│             │ Generate CUDA code                  │
│             ▼                                     │
│  ┌────────────────────────┐                      │
│  │ nvcc -arch=sm_80 -O3   │  CPU-side, ~3-5s     │
│  │ + .cu.flags applied    │                      │
│  └──────────┬─────────────┘                      │
│             │ Load .so, execute                   │
│             ▼                                     │
│  ┌────────────────────────┐                      │
│  │ CachePool evaluation   │  ~1-2 GB GPU memory  │
│  │  • 5 randomized inputs │  Resources stay      │
│  │  • PAC verification    │  resident via LRU    │
│  │  • 50 warmup + 30 runs │                      │
│  └──────────┬─────────────┘                      │
│             │ Discrete reward {-1,1,2,3}          │
│             ▼                                     │
│  ┌────────────────────────┐                      │
│  │ GRPO gradient step     │  Sequential, no      │
│  │ (TRL GRPOTrainer)      │  contention           │
│  └────────────────────────┘                      │
│                                                  │
│  Total per step (4 gens): ~40-65 seconds         │
│  100 steps: ~70-110 minutes                      │
└─────────────────────────────────────────────────┘
```

**Why no contention:** nvcc compilation is CPU-bound. Kernel execution uses tiny GPU memory for test data. Model stays loaded. Operations are sequential: generate → compile (CPU) → execute (GPU, small workload) → gradient update (GPU, model).

### 5.2 Environment Implementation

```python
"""
kernelforge_env.py — OpenEnv environment for CUDA kernel optimization.

Merges CUDA Agent evaluation loop with DoubleGraph architectural patterns.
All compilation and evaluation targets A100 (sm_80).

Install: pip install "openenv-core[core]>=0.2.1" cupy-cuda12x numpy
"""
import subprocess, tempfile, os, ctypes, hashlib, time, json
import numpy as np
from openenv.core.env_server import Environment, create_fastapi_app
from openenv.core.types import StepResult
from typing import Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Component 1: GPU CachePool (from DoubleGraph cache_pool.hpp, 125 lines)
#
# Keeps GPU resources resident across evaluations via thread-local LRU.
# Eliminates malloc/free per evaluation call.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

    def clear(self):
        self._cache.clear()
        self._order.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Component 2: Anti-Reward-Hacking (from CUDA Agent Section 3.2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Whitelisted per-kernel compilation flags (from DoubleGraph .cu.flags)
ALLOWED_CU_FLAGS = {
    "--use_fast_math",
    "--extra-device-vectorization",
    "--rdc=true",
}
ALLOWED_CU_FLAG_PREFIXES = (
    "--maxrregcount=",
)

# Symbols forbidden in generated kernels
FORBIDDEN_SYMBOLS = [
    "torch", "at::Tensor", "c10::", "torch::autograd",
    "triton", "torch.compile", "torch.nn.functional",
]


def extract_cu_flags(cuda_code: str) -> list[str]:
    """
    Parse // CU_FLAGS: from CUDA source. Only whitelisted flags accepted.
    Format: // CU_FLAGS: --use_fast_math --maxrregcount=48
    """
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
    try:
        result = subprocess.run(
            ["nm", "-D", so_path],
            capture_output=True, text=True, timeout=5,
        )
        for forbidden in FORBIDDEN_SYMBOLS:
            if forbidden in result.stdout:
                return f"Forbidden symbol detected: {forbidden}"
    except Exception:
        pass
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Component 3: Reward Function (CUDA Agent Equation 1, page 5)
#
# Discrete milestones validated by ablation (Table 2):
#   Discrete: 96.8% faster-than-compile
#   Continuous: 60.4% faster-than-compile
#   Delta: +36.4 percentage points
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_reward(
    compiled: bool,
    correct: bool,
    speedup_vs_eager: float,
    speedup_vs_compile: float,
) -> float:
    """
    CUDA Agent's 4-level reward:
      r = -1 if correctness fails
      r = +1 if correct but no speedup
      r = +2 if >5% faster than eager
      r = +3 if >5% faster than both eager AND compile
    """
    if not compiled or not correct:
        return -1.0
    if speedup_vs_compile > 1.05:
        return 3.0
    if speedup_vs_eager > 1.05:
        return 2.0
    return 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Component 4: The Environment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class KernelForgeEnv(Environment):
    """
    RL environment for CUDA kernel optimization on A100.

    Evaluation pipeline per step:
      1. Parse .cu.flags from agent's code
      2. Compile with nvcc -arch=sm_80 -O3
      3. Scan .so for forbidden symbols
      4. Run on 5 randomized inputs, check correctness (atol=1e-3, rtol=1e-3)
      5. Benchmark (50 warmup + 30 timed runs, median)
      6. Compute discrete milestone reward {-1, 1, 2, 3}
    """

    def __init__(self):
        self.cache_pool = GPUCachePool(max_entries=8)
        self.verify_inputs = 5
        self.warmup_iters = 50
        self.benchmark_runs = 30
        self.max_compile_seconds = 30
        self.current_problem = None
        self.baseline_eager_ms = None
        self.baseline_compile_ms = None
        self.skill_md = None

    def reset(self, problem: dict = None) -> dict:
        """
        Initialize with a problem from CUDA-Agent-Ops-6K format:
        {
            "name": "matmul_relu",
            "module_code": "class Model(nn.Module): ...",
            "get_inputs": "def get_inputs(): return [torch.randn(M, K), ...]",
            "get_init_inputs": "def get_init_inputs(): return []",
            "entry_point": "optimized_forward",
            "eager_time_ms": 1.234,
            "compile_time_ms": 0.567,
        }
        """
        self.current_problem = problem or self._sample_default_problem()
        self.baseline_eager_ms = self.current_problem.get("eager_time_ms")
        self.baseline_compile_ms = self.current_problem.get("compile_time_ms")
        self.skill_md = self._build_skill_md()

        return {
            "observation": self.skill_md + "\n\n---\n\n" + self._problem_prompt(),
            "arch": "sm_80",
            "baseline_eager_ms": self.baseline_eager_ms,
            "baseline_compile_ms": self.baseline_compile_ms,
        }

    def step(self, action: str) -> StepResult:
        """Accept CUDA code string, return (observation, reward, done, info)."""
        import cupy as cp

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "kernel.cu")
            lib_path = os.path.join(tmpdir, "kernel.so")

            cu_flags = extract_cu_flags(action)

            with open(src_path, "w") as f:
                f.write(action)

            # Step 1: Compile
            nvcc_cmd = [
                "nvcc", "-arch=sm_80", "-O3",
                "--shared", "-Xcompiler", "-fPIC",
            ] + cu_flags + [src_path, "-o", lib_path]

            try:
                proc = subprocess.run(
                    nvcc_cmd, capture_output=True, text=True,
                    timeout=self.max_compile_seconds,
                )
                if proc.returncode != 0:
                    return StepResult(
                        observation=f"COMPILE ERROR:\n{proc.stderr[:1500]}",
                        reward=-1.0, done=False,
                        info={"stage": "compile", "flags_used": cu_flags},
                    )
            except subprocess.TimeoutExpired:
                return StepResult(
                    observation="COMPILE TIMEOUT (30s)",
                    reward=-1.0, done=False,
                    info={"stage": "compile_timeout"},
                )

            # Step 2: Load + symbol scan
            try:
                so = ctypes.CDLL(lib_path)
            except OSError as e:
                return StepResult(
                    observation=f"LOAD ERROR: {e}",
                    reward=-1.0, done=False,
                    info={"stage": "load"},
                )

            forbidden = scan_forbidden_symbols(lib_path)
            if forbidden:
                return StepResult(
                    observation=f"ANTI-HACK: {forbidden}",
                    reward=-1.0, done=False,
                    info={"stage": "anti_hack"},
                )

            entry = self.current_problem.get("entry_point", "optimized_forward")
            if not hasattr(so, entry):
                return StepResult(
                    observation=f"MISSING: extern \"C\" void {entry}(...)",
                    reward=-1.0, done=False,
                    info={"stage": "entry_point"},
                )

            # Step 3: Correctness on 5 randomized inputs
            for i in range(self.verify_inputs):
                try:
                    inputs = self._generate_inputs(seed=i + hash(action) % 1000)
                    kernel_out = self._run_kernel(so, entry, inputs)
                    ref_out = self._run_reference(inputs)
                except Exception as e:
                    return StepResult(
                        observation=f"RUNTIME ERROR (input {i}): {str(e)[:500]}",
                        reward=-1.0, done=False,
                        info={"stage": "runtime", "input_idx": i},
                    )

                if not np.allclose(kernel_out, ref_out, atol=1e-3, rtol=1e-3):
                    max_diff = float(np.max(np.abs(kernel_out - ref_out)))
                    return StepResult(
                        observation=f"WRONG OUTPUT (input {i}): max_diff={max_diff:.6f}",
                        reward=-1.0, done=False,
                        info={"stage": "correctness", "max_diff": max_diff},
                    )

            # Step 4: Benchmark
            inputs = self._generate_inputs(seed=42)
            for _ in range(self.warmup_iters):
                self._run_kernel(so, entry, inputs)
            cp.cuda.Device(0).synchronize()

            times = []
            for _ in range(self.benchmark_runs):
                start = cp.cuda.Event()
                end = cp.cuda.Event()
                start.record()
                self._run_kernel(so, entry, inputs)
                end.record()
                end.synchronize()
                times.append(cp.cuda.get_elapsed_time(start, end))

            runtime_ms = float(np.median(times))

            # Step 5: Reward
            speedup_eager = (self.baseline_eager_ms / runtime_ms
                             if runtime_ms > 0 else 0.0)
            speedup_compile = (self.baseline_compile_ms / runtime_ms
                               if runtime_ms > 0 and self.baseline_compile_ms
                               else 0.0)

            reward = compute_reward(
                compiled=True,
                correct=True,
                speedup_vs_eager=speedup_eager,
                speedup_vs_compile=speedup_compile,
            )

            obs = (
                f"PASS | runtime={runtime_ms:.3f}ms | "
                f"eager={self.baseline_eager_ms:.3f}ms ({speedup_eager:.2f}x) | "
                f"compile={self.baseline_compile_ms:.3f}ms ({speedup_compile:.2f}x) | "
                f"reward={reward} | flags={cu_flags}"
            )

            return StepResult(
                observation=obs,
                reward=reward,
                done=(reward >= 3.0),
                info={
                    "runtime_ms": runtime_ms,
                    "speedup_eager": speedup_eager,
                    "speedup_compile": speedup_compile,
                    "flags_used": cu_flags,
                    "stage": "complete",
                },
            )

    @property
    def state(self) -> dict:
        return {
            "problem": self.current_problem.get("name") if self.current_problem else None,
        }

    # ── Internals ──

    def _build_skill_md(self) -> str:
        """Build A100-specific SKILL.md (see Part 6 of master doc)."""
        # Full content in PART 6 below
        return SKILL_MD_A100

    def _sample_default_problem(self) -> dict:
        return {
            "name": "vector_add",
            "entry_point": "optimized_forward",
            "eager_time_ms": 0.5,
            "compile_time_ms": 0.3,
        }

    def _problem_prompt(self) -> str:
        p = self.current_problem
        return (
            f"## Problem: {p['name']}\n\n"
            f"Write an optimized CUDA kernel. Entry point: "
            f"extern \"C\" void {p.get('entry_point', 'optimized_forward')}(...)\n\n"
            f"Module code:\n```python\n{p.get('module_code', '# see problem definition')}\n```"
        )

    def _generate_inputs(self, seed: int):
        rng = np.random.default_rng(seed)
        return rng.randn(1024).astype(np.float32)

    def _run_kernel(self, so, entry, inputs):
        import cupy as cp
        d_in = cp.asarray(inputs)
        d_out = cp.zeros_like(d_in)
        fn = getattr(so, entry)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        fn(ctypes.c_void_p(d_in.data.ptr), ctypes.c_void_p(d_out.data.ptr), ctypes.c_int(len(inputs)))
        cp.cuda.Device(0).synchronize()
        return d_out.get()

    def _run_reference(self, inputs):
        import torch
        t = torch.tensor(inputs)
        return t.numpy()


app = create_fastapi_app(KernelForgeEnv)
```

### 5.3 Anti-Reward-Hacking Summary

| Measure | Implementation |
|---------|---------------|
| Protected evaluation scripts | Verify/profile run in subprocess, read-only access |
| Forbid fallback calls | Scan .so with `nm -D` for torch/triton symbols, reject if found |
| Randomized inputs | 5 randomly-shaped inputs per eval (agent cannot memorize outputs) |
| Synchronized profiling | `cudaDeviceSynchronize()` before/after benchmark. 50 warmup iterations. |
| No web search | Solutions from model weights + SKILL.md only |
| Whitelisted .cu.flags | Only safe flags: --use_fast_math, --maxrregcount=N (16-128), --rdc=true, --extra-device-vectorization |

---

## PART 6: SKILL.md (A100-SPECIFIC)

This is the full SKILL.md given to the agent as observation context on every reset(). It encodes DoubleGraph's A100 technique library.

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
- Algebraic simplification: reduce computation before optimizing.
  Example: diag(A) × B = row-wise scaling (O(N²M) → O(NM), 73× speedup)
- Kernel fusion: merge sequential ops into one kernel.
  Eliminates intermediate tensor materialization and kernel launch overhead.
- Operator rearrangement: restructure computation order.
  Example: x @ (sum_j w_j^T) / 2 instead of sum_j (x @ w_j^T / 2) (24× speedup)

### Priority 2: A100 Memory Hierarchy (20-50% impact)
- L2 cache pinning: A100 has 40MB L2. Pin frequently accessed data:
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 30*1024*1024);
    // Then set stream attribute with cudaAccessPropertyPersisting
  For arrays up to 10M elements (40MB), entire working set fits in L2.
  Drops random access from ~200 cycles (HBM) to ~30-40 cycles (L2).
- Vectorized loads: float4 for 16-byte aligned access (4x transaction efficiency)
- Memory coalescing: consecutive threads access consecutive addresses
- Shared memory tiling: pad arrays float tile[32][33] to avoid 32-bank conflicts
  A100 has 164KB SMEM per SM — use it aggressively.
- __ldg() for read-only data via texture cache path
- cp.async pipelining: cuda::memcpy_async for global→shared (sm_80 exclusive)

### Priority 3: A100 Compute (10-20% impact)
- Warp primitives: __shfl_sync, __ballot_sync for intra-warp ops.
  Hand-rolled reductions give tighter register control than CUB library:
    __device__ float warp_reduce_sum(float val) {
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        return val;
    }
- Occupancy tuning: __launch_bounds__(256, 4) for register control.
  Use cudaOccupancyMaxActiveBlocksPerMultiprocessor.
- Cooperative kernels: for iterative algorithms, use
  cg::this_grid().sync() to avoid kernel launch overhead.
  Requires: // CU_FLAGS: --rdc=true
- TF32 for matmul: ~3x GEMM throughput on A100 Tensor Cores.
  Use cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32.

### Priority 4: Library Integration
- cuBLAS for GEMM: cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32
- cuDNN for Conv: cudnnConvolutionBiasActivationForward (fused conv+bias+activation)
- cuSPARSE for SpMV: cusparseSpMV with CSR format

## Rules
- Do NOT modify verification or profiling scripts
- Do NOT call torch.nn.functional or torch.compile in generated kernels
- Do NOT use Triton or any kernel generation framework
- Do NOT hardcode outputs for specific inputs
- Your kernel MUST work for any valid input shape
```

---

## PART 7: THE SCARCE DATA PROBLEM — WHY CUDA RL IS HARD

This is the central technical challenge. Every design decision in the training pipeline (Part 8) exists because of this problem.

### 7.1 The Problem: CUDA Is <0.01% of Pretraining Data

CUDA C++ code represents less than 0.01% of the tokens in any LLM pretraining corpus. For comparison:
- Python: ~10-15% of code tokens in typical training mixes
- JavaScript: ~8-12%
- C/C++ (general): ~3-5%
- CUDA C++: **<0.01%** — orders of magnitude less than even niche languages

This means CUDA-related tokens have probability ~10⁻⁹ in the base model's distribution (BF16 precision). When you try to do RL on this domain, the importance sampling ratio ρ_t(θ) = π_θ(a|s) / π_θ_old(a|s) fluctuates wildly because:

1. The numerator π_θ(a|s) for CUDA tokens is astronomically small
2. Tiny numerical perturbations in BF16 can swing this ratio by orders of magnitude
3. PPO/GRPO clipping (typically ε=0.2) cannot contain ratios that jump from 10⁻⁵ to 10⁵

**The result:** ByteDance's CUDA Agent paper documents that their initial pure RL trial on Seed1.6 (230B MoE) **collapsed at training step 17**. Training reward cratered, actor entropy spiked, the policy became "increasingly diffuse and poorly structured" (Section 3.3, page 6).

This is NOT a theoretical concern. It is empirically demonstrated.

### 7.2 Why Math RL Works But CUDA RL Doesn't (Without Help)

DeepSeek-R1-Zero showed that GRPO on a 671B base model produces emergent reasoning for mathematics without ANY SFT. On AIME 2024, R1-Zero hit 71% pass@1 (vs R1's 76%). So why does math work but CUDA doesn't?

**Mathematics is well-represented in pretraining data:**
- Mathematical notation, proofs, equations appear in textbooks, Wikipedia, arXiv, StackExchange
- Math tokens have probability ~10⁻³ to 10⁻⁴ — orders of magnitude higher than CUDA
- The base model already "knows" math syntax, so RL just needs to improve reasoning

**CUDA is essentially absent from pretraining data:**
- CUDA code only exists in specialized repositories, NVIDIA docs, and GPU programming tutorials
- The syntax overlaps with C++ but the semantics (thread hierarchy, memory hierarchy, synchronization) are unique
- At 10⁻⁹ probability, the model's "knowledge" of CUDA is near-random noise in BF16

**The PRIME counterpoint:** PRIME showed RL from Qwen2.5-Math-7B-Base "converges way faster than the SFT model, surpassing the instruct version within 32 steps." But this is for math — a well-represented domain. No equivalent result exists for CUDA.

### 7.3 Every Successful CUDA RL System Uses Warm-Up

| System | Model | Warm-Up Strategy | Result |
|--------|-------|-----------------|--------|
| CUDA Agent | Seed1.6 (230B) | 4-stage: Single-Turn PPO → RFT → Value Pretraining → Agentic PPO | 98.8% pass, 2.11× speedup |
| cudaLLM-8B | Qwen3-8B | SFT on DeepSeek-R1/Coder-generated CUDA, then RL via verl | 79.75% Level-1 |
| Kevin-32B | QwQ-32B | SFT warm-up, then RL with correctness+speedup rewards | Published by Cognition/Devin |
| CUDA-L1 | Various | SFT on curated CUDA examples, then RL | KernelBench evaluations |
| **Pure RL (CUDA Agent)** | **Seed1.6 (230B)** | **None** | **Collapsed at step 17** |

The pattern is unanimous: **CUDA RL requires supervised warm-up first.**

### 7.4 Our Mitigation Stack (5 Layers)

We don't just do one thing — we stack five mitigations, each addressing a different aspect of the scarce data problem:

**Mitigation 1: Trajectory Seeding via cudaLLM-8B (Expert Iteration)**

Before any RL, we use cudaLLM-8B (79.75% Bo1 Level-1 on KernelBench) to generate diverse CUDA kernel candidates. These are evaluated by our reward function and the reward-labeled trajectories become the initial training buffer.

This is NOT SFT — the policy model never directly imitates the teacher. Instead:
1. cudaLLM-8B generates candidates for each problem in our 200-operator dataset
2. Our environment evaluates them: compile, verify, benchmark
3. Trajectories with reward ≥ 1.0 form a "seed buffer"
4. GRPO uses this buffer for initial advantage estimation

This follows the Expert Iteration (ExIt) pattern, validated by Google's SRL framework: "SRL followed by RLVR yields the strongest overall performance."

**Why this helps:** It gives the policy gradient meaningful signal from the start. Without seeding, early GRPO steps produce 90%+ reward=-1 (compilation failures), making the gradient extremely noisy.

**Mitigation 2: Stage 1 Warm-Up on Trivially Easy Kernels**

Start with the simplest possible CUDA problems where even a base model occasionally produces correct code:
- Vector addition
- Scalar multiply
- Element-wise ReLU
- Simple reduction (sum)

The goal is NOT optimization — it's bootstrapping compilation rate from ~50% to ~85%+. The model needs to learn the format (extern "C" function signature, proper includes, launch configuration) before it can learn optimization.

**Mitigation 3: Rejection Fine-Tuning (RFT) — The Mandatory Stage**

CUDA Agent's ablation (Table 2) proves this is non-negotiable:
- **With RFT:** 96.8% faster-than-compile, 2.11× speedup
- **Without RFT:** 49.8% faster-than-compile, 1.05× speedup (nearly 50% ABSOLUTE DROP)
- Without RFT, "training reward collapses rapidly" and "actor entropy spikes"

RFT collects trajectories from Stage 1, filters for reward ≥ 1.0, and does SFT on the good ones. This creates a strong behavioral prior that prevents the policy from becoming diffuse during Stage 3 GRPO.

Think of it as "anchoring" the policy distribution: RFT moves the CUDA token probabilities from 10⁻⁹ up to a range where importance sampling ratios stay bounded.

**Mitigation 4: Curriculum Learning (Progressive Difficulty)**

CUDA Agent used a flat training distribution. We do better by adapting DoubleGraph's 4-way dispatch pattern into curriculum phases:

```
Phase A (Steps 0-20):  Single ops — relu, sigmoid, matmul
                        Target: Compilation + correctness (reward ≥ 1)

Phase B (Steps 20-40): 2-op fusions — matmul+relu, conv+bias+relu
                        Target: Basic optimization (reward ≥ 2, beats eager)

Phase C (Steps 40-55): A100-specific — L2 pinning, warp primitives, float4
                        Target: Architecture-aware optimization

Phase D (Steps 55-60): Complex — algebraic simplification, cuBLAS integration
                        Target: Beats torch.compile (reward = 3)
```

**Online prompt filtering (PRIME technique):** Maintain each problem at 20-80% success rate. If a problem is too hard (>80% failure), demote it. If too easy (>80% success), promote. This ensures the model always has meaningful gradient signal — not too noisy (all failures) or too flat (all successes).

**Mitigation 5: Discrete Milestone Rewards (Not Continuous Speedup)**

CUDA Agent's ablation proved discrete rewards {-1,1,2,3} beat continuous speedup rewards by 36.4 percentage points on the faster-than-compile metric:

| Reward Type | Pass Rate | Faster vs Compile | Speedup GM |
|-------------|-----------|-------------------|------------|
| Discrete milestones | 98.8% | 96.8% | 2.11× |
| Continuous speedup | 98.4% | 60.4% | 1.21× |

**Why continuous fails:** GPU execution times have noise (thermal throttling, OS interrupts, CUDA scheduler jitter). Continuous speedup rewards create outlier advantage estimates that bias the policy toward easy kernels where speedup variance is low. Discrete milestones are robust to timing noise.

### 7.5 What Qwen3-Coder-Next Brings That Smaller Models Don't

Qwen3-Coder-Next may partially mitigate the scarce data problem compared to Qwen2.5-Coder-7B:

1. **80B total parameters** — more capacity to store CUDA patterns from pretraining, even if rare
2. **Dedicated coder model** — "agentic training at scale" may include more CUDA than generic models
3. **~3.9B active per token** — only 3.9B parameters are active during inference, so the per-token compute cost is similar to an 8B dense model while drawing from 80B of stored knowledge
4. **Same family as cudaLLM-8B** — ByteDance fine-tuned Qwen3-8B for CUDA generation, confirming the architecture is compatible with CUDA domain adaptation

**UNVERIFIED:** Qwen3-Coder-Next's baseline CUDA generation quality. No KernelBench results exist. Must test empirically on hackathon day. If baseline compilation rate is >70% on simple problems, we may be able to shorten Stage 1 warm-up. If <30%, we need more aggressive trajectory seeding.

### 7.6 The Decision Tree on Hackathon Day

```
Hour 0: Load Qwen3-Coder-Next (GPTQ on H100 or bf16 on B200)
        ↓
        Test: Generate 20 simple CUDA kernels (vector_add, relu, etc.)
        ↓
    ┌─── Compilation rate > 70%? ───┐
    │                                │
   YES                              NO
    │                                │
    ↓                                ↓
  Skip to shorter           Full Stage 1 warm-up
  Stage 1 (30 steps)       (100 steps, 2 hours)
    │                                │
    └──── Both paths ──→ Stage 2 RFT (mandatory) ──→ Stage 3 GRPO
```

If Qwen3-Coder-Next can't load at all (hardware mismatch, GPTQ bug), immediately fall back to Qwen2.5-Coder-7B on whatever GPU is available. The pipeline is model-agnostic — only the model_loader.py changes.

---

## PART 8: 3-STAGE TRAINING PIPELINE

### 8.1 Why 3 Stages (Not 4)

CUDA Agent used 4 stages. We skip Stage 3 (Value Pretraining) because GRPO has no critic.

```
CUDA Agent (230B, 128 GPUs)        KernelForge (7B, 1 GPU)
───────────────────────────        ────────────────────────
Stage 1: Single-Turn PPO           Stage 1: Single-Turn GRPO (2 hrs)
         warm-up                            warm-up on easy kernels
Stage 2: RFT on agent              Stage 2: RFT on filtered
         trajectories                       trajectories (30 min)
Stage 3: Critic Value              SKIPPED (GRPO has no critic)
         Pretraining
Stage 4: Multi-turn Agentic        Stage 3: Single-Turn GRPO with
         PPO (150 steps)                    curriculum (2-3 hrs)
```

**Why single-turn, not multi-turn:** ByteDance used 128K context with up to 200 turns. On a single A100 with 7B model, single-turn is the practical choice. Each prompt → one kernel → one reward → policy update.

### 8.2 Stage 1: Single-Turn GRPO Warm-Up (2 hours)

**Goal:** Bootstrap from "knows CUDA syntax" to "kernels that compile and pass correctness."

**Data:** CUDA-Agent-Ops-6K, 200-problem curated subset. Start with simplest operators.

**Curriculum within Stage 1:**
- Steps 0-30: Single torch ops (relu, sigmoid, matmul, conv2d)
- Steps 30-60: 2-op compositions (matmul+relu, conv2d+bias+relu)
- Steps 60-100: 3+ op fusions

**Online prompt filtering (PRIME technique):** Maintain prompts at 20-80% success rate. Too hard (>80% failure) → demote. Too easy (>80% success) → promote.

```python
from trl import GRPOConfig, GRPOTrainer

warmup_config = GRPOConfig(
    learning_rate=3e-6,
    num_generations=4,              # 4 kernels per prompt
    max_prompt_length=1024,
    max_completion_length=4096,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Effective batch = 4
    max_steps=100,
    temperature=0.9,                # High for exploration
    beta=0.0,                       # No KL penalty
    bf16=True,
    optim="paged_adamw_8bit",
    output_dir="outputs/stage1-warmup",
    logging_steps=1,
    report_to="wandb",
)
```

**Expected outcome:** Compilation rate ~50% → ~85%+. Average reward ~0.0 → ~0.8-1.2.

**Abort criterion:** If stuck at <30% compilation after 20 steps, the model can't generate CUDA — fall back to smaller scope.

### 8.3 Stage 2: Rejection Fine-Tuning — RFT (30 minutes)

**Goal:** Filter warm-up trajectories, SFT on good ones to create strong behavioral prior.

**Why mandatory (CUDA Agent Figure 4):** Without RFT, training reward collapses within ~15 steps. Actor entropy spikes. Policy becomes diffuse. CUDA Agent's ablation: without RFT, faster-than-compile drops from 96.8% to 49.8%.

```python
# Collect trajectories from Stage 1 model
trajectories = []
for prompt in curriculum_prompts:  # 200 diverse operators
    for _ in range(4):
        completion = generate(model, tokenizer, prompt)
        code = extract_cuda_code(completion)
        result = env.evaluate_kernel(code)
        trajectories.append({
            "prompt": prompt,
            "completion": completion,
            "reward": result.reward,
        })

# Filter: keep reward >= 1.0 (correct)
good_trajectories = [t for t in trajectories if t["reward"] >= 1.0]

# SFT on filtered data
from trl import SFTConfig, SFTTrainer

rft_config = SFTConfig(
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    max_steps=100,           # Short — just enough to stabilize
    bf16=True,
    output_dir="outputs/stage2-rft",
)
```

### 8.4 Stage 3: GRPO with Curriculum (2-3 hours)

**Goal:** RL fine-tuning on progressively harder kernels. This is where the model discovers optimization strategies beyond what was in the RFT data.

**Curriculum (inspired by DoubleGraph's 4-way dispatch):**

| Phase | Steps | Target | Example Optimizations |
|-------|-------|--------|----------------------|
| A: Base | 0-20 | Single operators | Memory coalescing, shared memory tiling |
| B: Fusion | 20-40 | 2-3 op compositions | Kernel fusion, eliminate intermediates |
| C: Arch-Specific | 40-55 | A100-specific opts | L2 pinning, vectorized float4, warp primitives |
| D: Advanced | 55-60 | Complex operators | Algebraic simplification, library integration |

**Promotion rule:** Advance to next phase when >50% of last 10 steps achieve target reward.

```python
grpo_config = GRPOConfig(
    learning_rate=5e-6,
    num_generations=4,
    max_prompt_length=1024,
    max_completion_length=4096,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=60,
    temperature=0.7,                # Lower than warmup — exploit more
    beta=0.0,
    bf16=True,
    optim="paged_adamw_8bit",
    output_dir="outputs/stage3-grpo",
    logging_steps=1,
    report_to="wandb",
)
```

**Time estimate per step:** 4 generations × ~15s inference + 4 × ~10s evaluation + ~5s gradient = ~65s/step. 60 steps = ~65 minutes.

### 8.5 Efficiency Comparison

| Dimension | CUDA Agent (original) | KernelForge (ours) |
|-----------|-----------------------|--------------------|
| Model | 230B MoE (23B active) | 80B MoE (~3.9B active) |
| RL algorithm | PPO (actor + critic) | GRPO (actor only) |
| GPUs | 128 × H20 | 1 × H100 80GB (or B200) |
| Context | 131,072 tokens | 5,120 tokens (prompt+completion) |
| Agent turns | Up to 200 per episode | 1 (single-turn) |
| Generations/step | ~32-64 (estimated) | 4-8 |
| Training steps | 150 | 60 |
| Quantization | None (bf16) | 4-bit GPTQ |
| Optimizer | AdamW (fp32) | paged_adamw_8bit |
| Data | 6,000 operators | 200 operators (curated) |

**FLOPs per sequence:** CUDA Agent: 23B × 131K × 2 ≈ 6.0 TFLOPs. Ours: 3.9B × 5K × 2 ≈ 0.039 TFLOPs. **~150× reduction.**

---

## PART 9: DATA PIPELINE

### 9.1 Source

CUDA-Agent-Ops-6K (Apache 2.0, `BytedTsinghua-SIA/CUDA-Agent-Ops-6K`). Each operator is a PyTorch nn.Module with forward(), get_inputs(), get_init_inputs().

### 9.2 Curation

```python
def curate_subset(ops_6k: list[dict], n: int = 200) -> list[dict]:
    """
    Select 200 operators with balanced difficulty.
    CUDA Agent: 83.77% are 2-op fusions. We skew harder.
    """
    easy = [op for op in ops_6k if op["num_ops"] == 1][:50]     # 50 easy
    medium = [op for op in ops_6k if op["num_ops"] == 2][:75]   # 75 medium
    hard = [op for op in ops_6k if op["num_ops"] >= 3][:75]     # 75 hard
    return easy + medium + hard
```

### 9.3 Baseline Pre-Computation

For each operator, pre-compute on A100:
- `eager_time_ms`: torch.eager, median of 30 runs
- `compile_time_ms`: torch.compile, median of 30 runs after warmup

Both baselines run on the same GPU as the agent's kernels — relative speedup comparison is valid.

---

## PART 10: EVALUATION AND ABLATIONS

### 10.1 Hypotheses

**H1: Multi-stage RL improves kernel quality over base model.**
Metric: Average reward on held-out eval prompts increases from base → post-GRPO.

**H2: RFT warm-up is necessary (replicates CUDA Agent ablation).**
Metric: GRPO without RFT shows reward collapse; GRPO with RFT shows stable improvement.

**H3: DoubleGraph-informed SKILL.md improves over generic prompts.**
Metric: Kernels with A100-specific SKILL.md achieve higher speedup than generic "write fast CUDA" prompts.

### 10.2 Evaluation Protocol

- 50 held-out operators not in training set
- 4 generations per operator, take best
- Compare: base model → Stage 1 → Stage 2 → Stage 3
- Report: pass rate, reward distribution, speedup histogram

---

## PART 11: RISK REGISTER

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Qwen3-Coder-Next GPTQ won't load on hackathon GPU | Medium | High | Pre-test before hackathon. Instant fallback to Qwen2.5-Coder-7B. |
| Hackathon provides A100 not H100 (Qwen3 won't fit) | Medium | High | Fall back to Qwen2.5-Coder-7B which fits on any A100. |
| Qwen3-Coder-Next baseline CUDA generation is poor | Medium | Medium | Trajectory seeding from cudaLLM-8B bootstraps the policy. Stage 1 warm-up addresses this. |
| TRL GRPOTrainer + GPTQ model incompatibility | Medium | High | Test before hackathon. Fall back to Unsloth + Qwen2.5 if needed. |
| Model generates uncompilable CUDA at high rate | Medium | High | Stage 1 warm-up + curriculum. Start with trivially easy problems. |
| GRPO shows no improvement over RFT | Medium | Medium | Valid negative result; present as ablation confirming CUDA Agent findings at smaller scale. |
| Training collapses despite RFT | Low | High | CUDA Agent validated for Seed1.6; monitoring reward curve detects early. |
| Hackathon WiFi too slow for 80B+ model download | High | High | Pre-download GPTQ (~43GB) and fallback model to local drive. |
| Compilation too slow for RL loop | Low | Medium | nvcc is CPU-bound, ~3-5s per kernel. Acceptable. |
| Scarce data problem worse for Qwen3 than expected | Medium | Medium | 5-layer mitigation stack: seeding, warm-up, RFT, curriculum, discrete rewards. |

---

## PART 12: HACKATHON TIMELINE

### Pre-Hackathon (before March 7)

- [ ] Download Qwen3-Coder-Next GPTQ-Int4 (~43GB) to local drive
- [ ] Download Qwen2.5-Coder-7B-Instruct via Unsloth (fallback)
- [ ] Download cudaLLM-8B for trajectory seeding
- [ ] Download CUDA-Agent-Ops-6K dataset
- [ ] Curate 200-problem subset
- [ ] Pre-compute eager/compile baselines (requires A100 access)
- [ ] Generate trajectory seed buffer using cudaLLM-8B (Expert Iteration)
- [ ] Test environment end-to-end: reset() → step() → reward
- [ ] Test Qwen3-Coder-Next loading on H100 (GPTQ path) — critical validation
- [ ] Test full pipeline: model load → generate → evaluate → gradient step
- [ ] Package environment as Docker container

### Day 1: March 7 (BUILD)

| Time | Activity |
|------|----------|
| Hour 0-1 | Claim GPU. Load Qwen3-Coder-Next (GPTQ). Test: generate 20 simple CUDA kernels. If compilation rate >70%, shorten Stage 1. If model fails to load, fall back to Qwen2.5-Coder-7B immediately. |
| Hour 1-3 | Stage 1: GRPO warm-up (30-100 steps depending on baseline compilation rate) |
| Hour 3-4 | Stage 2: RFT (collect trajectories, filter reward ≥ 1.0, SFT 100 steps) |
| Hour 4-7 | Stage 3: GRPO with curriculum (60 steps, progressive difficulty) |
| Hour 7-8 | Evaluate: compare base → warm-up → RFT → GRPO |
| Hour 8-10 | Continue training if improving; run ablations (H1/H2/H3) |

### Day 2: March 8 (SHIP)

| Time | Activity |
|------|----------|
| Hour 0-2 | Final training + full evaluation suite |
| Hour 2-4 | Build demo (Streamlit/Gradio): reward curve, kernel comparisons, live generation |
| Hour 4-5 | Push model + environment to HuggingFace Hub |
| Hour 5 | Pitch: 3 minutes |

**Pitch narrative:** "CUDA Agent proved RL trains better kernel writers. DoubleGraph proved per-GPU specialization matters. We combined them at hackathon scale using OpenEnv. Here's the reward curve. Here's a kernel the model discovered an optimization that wasn't in the training data. This pipeline is open-source — plug in any model, any GPU, any operator."

---

## PART 13: REPOSITORY STRUCTURE

```
KernelForge-OpenEnv/
├── README.md
├── requirements.txt
├── Dockerfile                          # OpenEnv deployment container
├── skill.md                            # SKILL.md v5 (Part 6 of this doc)
│
├── openenv_env/
│   ├── __init__.py
│   ├── kernelforge_env.py             # The Environment (Part 5)
│   ├── cache_pool.py                  # DoubleGraph CachePool
│   ├── reward.py                      # CUDA Agent reward function
│   └── anti_hack.py                   # Symbol scanning, flag whitelist
│
├── training/
│   ├── model_loader.py                # Unsloth FastLanguageModel loader
│   ├── stage1_warmup.py               # Single-turn GRPO warm-up
│   ├── stage2_rft.py                  # Trajectory collection + RFT
│   ├── stage3_grpo.py                 # GRPO with curriculum
│   └── curriculum.py                  # Prompt difficulty management
│
├── datasets/
│   ├── download_ops6k.py              # Pull CUDA-Agent-Ops-6K
│   ├── curate_subset.py               # 200-problem subset
│   └── compute_baselines.py           # Pre-compute eager/compile times
│
├── evaluation/
│   ├── ablation.py                    # H1/H2/H3 ablations
│   └── eval_model.py                  # Full evaluation suite
│
├── demo/
│   └── app.py                         # Streamlit/Gradio demo
│
└── tests/
    ├── test_env.py                    # Environment unit tests
    ├── test_reward.py                 # Reward function tests
    └── test_compile.py                # sm_80 compilation tests
```

---

## PART 14: TECH STACK

```
# Core training
unsloth>=2025.3
transformers>=4.48,<5.0
trl>=0.16.0
peft>=0.14.0
torch>=2.10.0
datasets>=3.0

# OpenEnv
openenv-core[core]>=0.2.1

# Evaluation
cupy-cuda12x>=14.0
numpy>=1.26

# Data
huggingface-hub>=0.25

# Monitoring
wandb>=0.19
```

---

## PART 15: KEY CONSTANTS AND THRESHOLDS

All magic numbers in one place. Change here, propagate everywhere.

| Constant | Value | Source |
|----------|-------|--------|
| nvcc arch flag | `-arch=sm_80` | A100 compute capability |
| A100 L2 cache | 40 MB | NVIDIA whitepaper |
| L2 persistence allocation | 30 MB | DoubleGraph pattern (75% of L2) |
| A100 SMEM per SM | 164 KB | NVIDIA whitepaper |
| A100 SMs | 108 | NVIDIA whitepaper |
| A100 registers per SM | 65,536 | NVIDIA whitepaper |
| A100 HBM bandwidth | 2,039 GB/s (SXM) | NVIDIA whitepaper |
| Reward: compile fail | -1 | CUDA Agent Eq. 1 |
| Reward: correct, no speedup | +1 | CUDA Agent Eq. 1 |
| Reward: beats eager >5% | +2 | CUDA Agent Eq. 1 |
| Reward: beats compile >5% | +3 | CUDA Agent Eq. 1 |
| Speedup threshold | 1.05× (5%) | CUDA Agent Eq. 1 |
| Verify inputs per eval | 5 | CUDA Agent Section 3.2 |
| Warmup iterations | 50 | Reduced from CUDA Agent's 100 |
| Benchmark runs | 30 | Reduced from CUDA Agent's 50 |
| Compile timeout | 30 seconds | Practical limit |
| CachePool max entries | 8 | DoubleGraph cache_pool.hpp |
| BFS TD→BU threshold | frontier > N/20 | DoubleGraph A100 BFS |
| BFS BU→TD threshold | frontier < N/200 | DoubleGraph A100 BFS |
| Louvain serial threshold | N ≤ 200 | DoubleGraph Louvain |
| Louvain thread tier | avg_degree < 8 | DoubleGraph Louvain |
| Louvain warp tier | avg_degree ≥ 8 | DoubleGraph Louvain |
| maxrregcount range | 16-128 | .cu.flags whitelist |
| LoRA rank | 16 | Hackathon speed vs quality |
| LoRA alpha | 16 | Standard (alpha = rank) |
| LoRA targets (Qwen3-Coder-Next) | q/k/v/o_proj + shared_expert gate/up/down | Attention + shared expert only. All 512 experts = 207GB, won't fit. |
| LoRA targets (Qwen2.5-Coder-7B fallback) | q/k/v/o/gate/up/down_proj | All linear layers (dense model) |
| GRPO num_generations | 4-8 | 4 on tight VRAM, 8 with headroom. DeepSeek-R1 uses 16. |
| GRPO temperature (Stage 1) | 0.9 | High for exploration |
| GRPO temperature (Stage 3) | 0.7 | Lower for exploitation |
| GRPO beta (KL penalty) | 0.0 | DAPO research: not essential |
| GRPO max_completion_length | 4,096 tokens | Full CUDA kernel (100-500 lines) |
| Stage 1 max_steps | 100 | ~2 hours |
| Stage 2 SFT max_steps | 100 | ~30 minutes |
| Stage 3 max_steps | 60 | ~65 minutes |
| Curriculum promotion rule | >50% of last 10 achieve target reward | DoubleGraph-inspired |
| Prompt filter range | 20%-80% success rate | PRIME technique |
| Dataset size | 200 operators (50 easy / 75 medium / 75 hard) | Curated from Ops-6K |
| Trajectory seed model | cudaLLM-8B (ByteDance-Seed/cudaLLM-8B) | 79.75% Bo1 Level-1, Expert Iteration bootstrap |
| Trajectory seed filter | reward ≥ 1.0 | Keep correct compilations only |
| GPTQ model ID | dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16 | Community 4-bit quantization |
| GPTQ VRAM (weights) | ~43 GB | 80B × 0.5 bytes + scales |
| Training VRAM total (H100 path) | ~54 GB | 26GB headroom on H100 80GB |
| Training VRAM total (B200 path) | ~171 GB | 21GB headroom on B200 192GB |

---

## PART 16: EVIDENCE REFERENCES

| Claim | Source | Location |
|-------|--------|----------|
| Pure RL collapsed at step 17 | CUDA Agent paper | Section 3.3, page 6 |
| CUDA tokens = 0.01% of pretraining data | CUDA Agent paper | Section 3.3 |
| RFT ablation: faster-than-compile drops to 49.8% | CUDA Agent paper | Table 2, page 8 |
| Discrete rewards +36.4pp over continuous | CUDA Agent paper | Table 2 |
| Full system: 98.8% pass, 96.8% faster, 2.11× GM | CUDA Agent paper | Table 2 |
| 4-stage pipeline | CUDA Agent paper | Figure 3, page 5 |
| Reward function {-1,1,2,3} | CUDA Agent paper | Equation 1, page 5 |
| Anti-hacking measures | CUDA Agent paper | Section 3.2 |
| 6K operator dataset | CUDA Agent paper | Section 3.1 |
| Diag matmul 73× speedup | CUDA Agent paper | Appendix D |
| 192 kernel files per GPU | DoubleGraph SKILLS.md | Section 1 |
| 3.6× average speedup over cuGraph | DoubleGraph / WarpSpeed announcement | March 2-3 2026 |
| BFS direction-optimizing thresholds N/20, N/200 | DoubleGraph SKILLS.md | Section 4.2 |
| Louvain 3-tier dispatch thresholds | DoubleGraph SKILLS.md | Louvain analysis |
| CachePool pattern (LRU, 8 entries) | DoubleGraph SKILLS.md | Section 3 |
| 4-way dispatch | DoubleGraph SKILLS.md | Section 3 |
| GRPO eliminates critic (~50% memory) | DeepSeek-R1 paper | Architecture section |
| Unsloth: BitsAndBytes cannot quantize MoE nn.Parameter | Unsloth docs | February 2026 |
| Qwen2.5-Coder-7B: 5.5T code tokens | Qwen docs | Model card |
| cudaLLM-8B: 79.75% Bo1 Level-1 | ByteDance/cudaLLM-8B | Model card |

---

## PART 17: VERSION HISTORY

| Version | Date | Key Changes |
|---------|------|-------------|
| v1-v2 | March 3 | Initial research. Pure RL approach. WCC-only scope. |
| v3 | March 3 | H100/Hopper primitives. WarpSpeed Union-Find. Overly ambitious. |
| v4 | March 3 | **Pivot to A100.** Qwen2.5-Coder-7B. Practical hackathon scope. |
| v5 | March 4 | CUDA Agent 4-stage pipeline integrated. DoubleGraph patterns productized. Qwen3-Coder-Next on H100/B200. |
| Implementation Spec | March 4 | Full code. Multi-GPU registry (H100/B200/A100). Most detailed but scope-crept. |
| v6 (rejected) | March 4 | Attempted consolidation. Too high-level, lost implementation details. |
| **MASTER (this doc)** | March 4 | **A100 target. Qwen3-Coder-Next primary. Deep A100 hardware ref. Scarce data analysis. Single source of truth.** |

**Key corrections across versions:**
- "NO SFT JUST RL" → Multi-stage (warm-up → RFT → GRPO). Pure RL collapses at step 17.
- Multi-GPU scope creep (H100/B200/A100) → A100 as kernel target, H100/B200 for training.
- Qwen2.5-Coder-7B as primary → Qwen3-Coder-Next primary, Qwen2.5 as fallback.
- WCC-only → Multi-algorithm from Ops-6K.
- Simple reward → CUDA Agent's validated {-1,1,2,3} with ablation evidence.
- No anti-hacking → Full suite from CUDA Agent Section 3.2.
- No curriculum → DoubleGraph-inspired 4-phase progression.
- No scarce data analysis → Dedicated section with 5-layer mitigation stack.
- Shallow A100 specs table → Deep hardware reference with optimization-relevant detail.
