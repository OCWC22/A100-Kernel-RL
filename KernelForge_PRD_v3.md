# KernelForge-OpenEnv PRD v3.0

## Architectural Blueprint for Autonomous Hardware-Aware GPU Performance Engineering

**Date:** March 3, 2026  
**Target GPU:** H100 SXM5 (sm_90a) — `a` suffix mandatory for TMA/WGMMA  
**Algorithm:** WCC (Weakly Connected Components via Union-Find)  
**Agent Model:** Qwen3-Coder-Next (80B total / 3B active MoE)  
**RL Framework:** TRL GRPOTrainer + Unsloth FastModel  
**Environment:** Custom OpenEnv (openenv-core) CUDA sandbox on Modal H100  
**Hackathon:** OpenEnv Hackathon SF — March 7-8, 2026 (SHACK15, San Francisco)  
**Organizers:** Cerebral Valley + PyTorch/Meta + SHACK15  
**Prize Pool:** $100K+ cash | Teams up to 4

---

## 1. Executive Summary

### 1.1 The Problem

The trajectory of accelerated computing has reached an inflection point where the theoretical peak performance of silicon hardware significantly outpaces the capabilities of the software ecosystem designed to utilize it. As artificial intelligence models scale to trillion-parameter complexities and graph analytics workloads process graphs with billions of edges, the underlying computational hardware has evolved at a blistering pace. Modern graphics processing units, particularly those based on the NVIDIA Hopper microarchitecture, introduce a myriad of highly specialized, asynchronous hardware primitives. However, the algorithms and kernels required to drive this hardware to its theoretical limits consistently lag behind. This persistent performance gap is fundamentally bottlenecked by a global scarcity of elite GPU performance engineers capable of navigating the microarchitectural intricacies of modern accelerators.

### 1.2 The Solution

KernelForge-OpenEnv establishes a framework for Artificial Expert Intelligence that autonomously generates, aggressively optimizes, and mathematically verifies CUDA kernels that routinely surpass human-engineered baselines. It synthesizes two state-of-the-art methodologies:

**A. ByteDance CUDA Agent** (arXiv:2602.24286, February 27, 2026)

- Paper: "CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation"
- Authors: Weinan Dai, Hanlin Wu, et al. (ByteDance Seed + Tsinghua AIR SIA-Lab)
- Base model: Seed 1.6 (230B MoE, 23B active parameters)
- Results: 2.11x speedup over torch.compile on KernelBench (98.8% pass rate, 96.8% faster-than-compile rate)
- Critical finding: Metrics catastrophically drop to 77.1% pass and 14.1% faster rate if the multi-turn agentic loop is removed — proving the RL pipeline is load-bearing
- Training: 150 steps, 128 H20 GPUs, 131K context, up to 150 agent turns
- GitHub: github.com/BytedTsinghua-SIA/CUDA-Agent
- Related open-source: ByteDance-Seed/cudaLLM (Qwen3-8B base, Apache 2.0)

**B. DoubleAI WarpSpeed** (announced March 2, 2026)

- Company: AA-I Technologies Ltd, Tel Aviv. Founded by Prof. Amnon Shashua and Prof. Shai Shalev-Shwartz (Mobileye co-founders). $200M Series A (Lightspeed, Bessemer, NVIDIA).
- Product: Autonomously rewrote every kernel in NVIDIA cuGraph, producing doubleGraph — a drop-in replacement
- Results: 3.6x geometric mean speedup over a decade of NVIDIA expert code. 100% of algorithms faster. 55% achieve >2x. 18% exceed 10x. WCC alone: ~17x speedup.
- Architecture: 576 specialized kernels (192 per GPU architecture x 3: A100, L4, A10G)
- GitHub: github.com/double-ai/doubleGraph (Apache 2.0)
- Methodology: PAC-reasoning verification, time-travel with experience (agentic rollback), exhaustive per-config specialization, non-atomic Union-Find path compression, L2 cache pinning
- Note: All benchmarks are self-reported as of March 3, 2026. No independent validation yet.

### 1.3 Hardware Reality: Your 16GB M5 MacBook

**LOCAL (MacBook M5 16GB):**
- Code in Cursor (sponsor: AI pair programming credits)
- Modal CLI orchestration (modal run, modal deploy)
- OpenEnv HTTP server (pure Python, runs locally)
- Qwen3-Coder-Next inference via llama.cpp GGUF for local testing only
- Dataset generation scripts (networkx graph generation)
- Git, version control, documentation

**CLOUD (Modal H100 at ~$3.95/hr, or CoreWeave hackathon credits):**
- All CUDA compilation (nvcc -arch=sm_90a)
- All kernel benchmarking (cudaEvent timing with warm-up)
- All correctness verification (kernel output vs networkx reference)
- GRPO training loop (Unsloth FastModel + TRL GRPOTrainer)
- Baseline profiling (cuGraph WCC + doubleGraph reference)

**You CANNOT locally:** compile CUDA, fine-tune any model, run H100 benchmarks, or install openenv (wrong package — use openenv-core).

### 1.4 Scope Constraints (Strict — Do Not Expand)

- Algorithm: WCC only (cugraph.weakly_connected_components)
- Hardware: H100 only (-arch=sm_90a)
- Single-GPU only (no multi-GPU, no mg_ prefix APIs)
- Baselines: original cuGraph WCC + doubleGraph A100 reference
- Anti-hacking: agent cannot modify verification scripts, access network, call cuBLAS/cuSPARSE, or use torch fallbacks in C++ bindings

---

## 2. The Hardware Paradigm Shift: Ampere to Hopper

### 2.1 Qualitative Not Quantitative

The evolution from NVIDIA Ampere (A100, sm_80) to NVIDIA Hopper (H100, sm_90a) is not merely an iterative increase in clock speeds or core counts. It represents a fundamental, qualitative paradigm shift in memory hierarchy design, asynchronous execution capabilities, and parallel programming models. The KernelForge agent utilizes a comprehensive hardware specification matrix to dynamically route its optimization strategies based on the target microarchitecture. Legacy optimization strategies formulated for Ampere hardware frequently result in severe resource underutilization when executed on Hopper silicon.

### 2.2 Complete Architectural Transition Matrix

| Architectural Feature | NVIDIA Ampere A100 (sm_80) | NVIDIA Hopper H100 (sm_90a) | Implication for Agentic Synthesis |
|---|---|---|---|
| Manufacturing Node | TSMC 7nm (N7) | TSMC 4N (Custom 5nm class) | Increased logic gate density permits the inclusion of dedicated asynchronous copy engines |
| Streaming Multiprocessors | 108 SMs | 132 SMs (+22%) | Dictates dynamic adjustments in block sizing and grid configurations to ensure full chip occupancy |
| CUDA Core Count | 6,912 | 16,896 (+144%) | Enables aggressive unrolling of scalar operations without stalling warp schedulers |
| FP32 ops/cycle/SM | 1x | 2x | Doubles integer arithmetic throughput in Union-Find operations |
| Tensor Core Generation | Third Generation | Fourth Generation | Hopper Tensor Cores deliver double the raw MMA throughput per SM |
| Memory Subsystem | 80 GB HBM2e | 80 GB HBM3 | Same capacity, fundamentally different bandwidth |
| Memory Bandwidth | Up to 2.0 TB/s | Up to 3.35 TB/s (+68%) | Doubles the speed at which the agent can stream graph adjacency lists from global memory |
| L2 Cache Capacity | 40 MB | 50 MB (+25%) | Expands the viable boundary for cache pinning operations, reducing reliance on main memory. Parent array for ~12.5M vertices fits entirely in L2 (vs ~10M on A100) |
| Shared Memory per SM | 164 KB | 228 KB (+39%) | Larger frontier buffers per block; more room for local label caches |
| Execution Hierarchy | Threads -> Blocks -> Grids | Threads -> Blocks -> Clusters -> Grids | Fourth tier demands cooperative_groups API for SM-to-SM data locality |
| Thread Block Clusters | Not available | Up to 16 blocks co-scheduled | Cross-block label propagation via DSMEM without global memory round-trips |
| TMA | Not available | Dedicated hardware unit | Single thread initiates bulk copy; entire thread block freed for compute |
| DSMEM | Not available | Unified shared memory across cluster | Approximately 7x faster than global memory for inter-block communication |
| DPX Instructions | Not available | Hardware-fused DP primitives | min/max ops in label propagation fused to single instruction cycle |
| Interconnect Bandwidth | NVLink 3.0 (600 GB/s) | NVLink 4.0 (900 GB/s) | Reduces communication latency in multi-GPU configurations (future scope) |

### 2.3 The Law of Leaky Abstractions

The primary obstacle KernelForge seeks to dismantle is the persistent "law of leaky abstractions" in accelerated computing. High-level software abstractions, such as standard PyTorch libraries or out-of-the-box cuGraph implementations, are intrinsically generalized. They compile and execute across a wide spectrum of devices, which inherently prevents aggressive exploitation of generation-specific microarchitectural pathways.

A generic graph kernel compiled for sm_80 will execute correctly on an sm_90a device, but it fundamentally underutilizes the silicon because it continues to rely on legacy synchronous memory copies and global memory barriers. KernelForge bypasses these leaky abstractions entirely, specializing every algorithmic implementation to the exact hardware target via the injection of inline Parallel Thread Execution (PTX) code and hardware-aware C++ semantics.

---

## 3. Hopper-Specific Microarchitectural Primitives

The agent's action space is augmented with specific hardware primitives that redefine how computational work is scheduled and how memory is accessed.

### 3.1 The Tensor Memory Accelerator (TMA)

**What A100 does (legacy):** The cp.async instruction family facilitates asynchronous data movement, but data routes from global memory through L1 cache and into physical register files before being committed to shared memory. This staging consumes valuable register file allocations and utilizes general-purpose SM instructions for transfer management, limiting simultaneous compute.

**What H100 TMA does:** The TMA is a dedicated, fully asynchronous hardware copy engine. cp.async.bulk.tensor instructions asynchronously and bidirectionally transfer multi-dimensional tensors (1D to 5D) directly between global memory and shared memory, entirely bypassing the local register files.

A single thread within a warp issues a massive bulk data movement request to the physical TMA unit. Once issued, the remainder of the thread block is freed to continue processing overlapping mathematical computations while data remains in flight.

**TMA Multicast (H100-exclusive):** In traditional architectures, if multiple SMs need the same graph adjacency structures, each SM independently fetches from global memory. TMA multicast orchestrates a single global memory fetch that simultaneously broadcasts the data into the shared memory arrays of multiple SMs within a Thread Block Cluster. This drastically reduces overall HBM3 bandwidth pressure, eliminating memory saturation in highly parallel broadcasts.

```cpp
// TMA descriptor setup (host-side)
CUtensorMap tensor_map;
cuTensorMapEncodeTiled(&tensor_map,
    CU_TENSOR_MAP_DATA_TYPE_INT32,
    1,                              // 1D tensor (adjacency list segment)
    &adjacency_data,
    &size, &stride, &box_size, &elem_stride,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_NONE,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,   // Promote to L2 on load
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);

// In kernel: single thread initiates copy, all others freed for compute
if (threadIdx.x == 0) {
    // TMA async copy — hardware handles address generation
    cp_async_bulk_tensor_1d_global_to_shared(
        &shared_adj[0], &tensor_map, coords, barrier);
}
// Other 127 threads do path compression while adjacency data loads asynchronously
```

**Performance:** TMA achieves 1.45 TB/s GMEM throughput — 59% increase over A100 cp.async patterns.

### 3.2 Distributed Shared Memory (DSMEM) and Thread Block Clusters

**The A100 limitation:** When threads in one block need to communicate with threads in a separate block, the architecture forces a write-out to global DRAM followed by a read-back. This creates severe bottlenecks in iterative algorithms requiring constant state updates, such as label propagation in graph analytics.

**The H100 solution:** Thread Block Clusters allow software to programmatically group multiple thread blocks with the strict hardware guarantee that the entire cluster will be co-scheduled concurrently onto physically adjacent SMs within the same Graphics Processing Cluster (GPC).

DSMEM fundamentally alters the memory landscape by mapping the shared memory virtual address spaces of all thread blocks within a cluster into a unified, accessible memory pool. The agent constructs generic pointers via the cooperative_groups API that can address any shared memory segment within the active cluster.

**Performance:** Direct SM-to-SM network communications for memory loads, stores, and atomic operations via DSMEM achieve approximately 7x faster data exchange compared to standard global memory synchronization.

```cpp
// 4-block cluster: 4 x 228 KB = 912 KB distributed shared memory
__cluster_dims__(4, 1, 1)
__global__ void wcc_propagate_clustered(
    int* labels, const int* row_ptr, const int* col_idx, int N
) {
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();

    __shared__ int local_labels[BLOCK_SIZE];

    // Load local partition labels into shared memory
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) local_labels[threadIdx.x] = labels[tid];
    __syncthreads();

    // Access neighbor block's shared memory via cluster — NO global memory round-trip
    for (int neighbor_rank = 0; neighbor_rank < cluster.num_blocks(); neighbor_rank++) {
        int* neighbor_smem = cluster.map_shared_rank(local_labels, neighbor_rank);
        // Propagate labels across partition boundaries via DSMEM
        // This replaces what would be an atomicMin to global memory on A100
    }

    cluster.sync();
}

// Launch with cluster attribute
cudaLaunchConfig_t config = {};
cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim = {4, 1, 1};
config.attrs = attrs;
config.numAttrs = 1;
cudaLaunchKernelEx(&config, wcc_propagate_clustered, labels, row_ptr, col_idx, N);
```

### 3.3 Dynamic Programming Accelerators (DPX)

Dynamic programming algorithms are notoriously hostile to traditional GPU parallelization. These algorithms feature deep sequential dependencies and rely heavily on memoization, suffering from massive instruction overhead when executing repetitive minimum, maximum, and thresholding comparisons in scoring matrices.

DPX instructions are custom silicon pathways designed to perform advanced, fused mathematical operations within the inner loops of dynamic programming algorithms. The KernelForge agent's static analysis modules scan source code for recursive min/max operations and lower them directly into DPX PTX intrinsics.

**For WCC label propagation:**

```cpp
// A100: software min (multiple instructions per operation)
label[v] = min(label[v], label[neighbor]);

// H100: hardware-fused 3-way min via DPX — ONE instruction cycle
// __vimin3_s32(a, b, c) computes min(a, min(b, c)) in a single cycle
label[v] = __vimin3_s32(label[v], label[n1], label[n2]);

// For path halving with simultaneous neighbor check:
// __vimax3_s16x2_relu — fused 3-way max with clamp-to-zero (ReLU)
```

**Performance:** DPX delivers up to 7x speedup over optimal Ampere implementations for DP patterns. Smith-Waterman (genomics): 7.8x speedup vs A100. Combinatorial routing: up to 40x acceleration vs dual-socket CPU.

**SKILL.md directive:** When the agent detects recursive min/max operations during profiling, it must replace C++ constructs with DPX intrinsics. Guard all DPX code with `#if __CUDA_ARCH__ >= 900` compiler directives.

---

## 4. Algorithmic Mastery: WarpSpeed Optimization Heuristics

While exploitation of hardware primitives yields deterministic gains, achieving true Artificial Expert Intelligence requires algorithmic creativity that moves beyond standard human coding conventions. This section formalizes the extreme optimization paradigms from DoubleAI's WarpSpeed system.

### 4.1 The Deliberate Data Race: Non-Atomic Union-Find Path Compression

One of the most profound examples of WarpSpeed's divergence from human engineering intuition. The WCC kernel alone achieved ~17x speedup primarily through this technique.

**Diagram: Synergistic Graph Optimization (from architectural blueprint)**

The transformation is illustrated by comparing two workflows:

**Traditional Workflow:** SM threads all funnel through an atomic lock (atomicCAS). Writes are serialized — only one thread updates the parent array at a time. Atomic operations cost ~100+ cycles due to cache line ownership serialization. Data goes directly to global memory (DRAM). The bottleneck is structural: hardware-level lock contention.

**Optimized Workflow:** SM threads write concurrently with NO atomic lock. All writes target L2 cache (pinned), not global memory. Non-atomic stores coalesce freely into 128-byte transactions. The parent array never leaves L2 during the algorithm's lifetime. By replacing serialized atomic operations with concurrent non-atomic stores and pinning the parent array in the L2 cache, the algorithm collapses into a single, high-throughput kernel launch.

**Why the deliberate data race is mathematically safe:**

The AEI agent behind WarpSpeed recognized a deep mathematical property: the Union-Find parent hierarchy is monotonically converging. In the context of finding connected components, every value written to the parent array is strictly guaranteed to be an ancestor within the exact same component structure. Memory locks are therefore mathematically superfluous.

1. **Monotone convergence:** Parent pointers can only be updated to point closer to the root. A stale read produces a valid ancestor — just not maximally compressed. No update can ever move a pointer farther from the root.

2. **No value creation:** Every written value was previously read from another valid parent entry. The set of possible values is always valid node IDs within the same connected component. It is impossible for a race to inject an invalid node ID.

3. **Self-correcting:** If a race condition occurs during path compression, one thread may read a slightly stale ancestor pointer. This merely results in one additional traversal hop during the next iteration, but mathematical correctness is strictly and provably preserved.

4. **Path halving is inherently race-tolerant:** `parent[x] = parent[parent[x]]` at worst points x to a valid ancestor that is not the grandparent — still progress, never corruption. No race can induce infinite cycles or bridge isolated graph components together.

**Academic foundation:** ECL-CC algorithm (Jaiganesh and Burtscher, HPDC 2018). Additional analysis: Liu, VanAusdal, and Burtscher (IISWC 2024) confirmed that racy codes can produce unpredictable behavior across different hardware and compilers — our PAC verification ensures correctness regardless.

**Performance impact:** Atomic operations enforce strict serialization at the hardware level, creating massive latency bottlenecks (~100+ cycles per contended atomic). Non-atomic stores coalesce freely into 128-byte transactions. Combined with L2 pinning: random parent lookups drop from ~200+ cycles (HBM) to ~30-40 cycles (L2) — a 5-7x latency reduction on the hottest inner loop.

### 4.2 L2 Cache Pinning for Permanent Residency

While the elimination of atomic instructions removes serialization bottlenecks, executing racy non-atomic random writes against global memory (DRAM) would still incur devastating latency penalties due to the physical distance between SMs and memory controllers. The parent array must reside as close to the execution units as possible.

**The synergy:** Dropping atomics removes the logic bottleneck. L2 pinning removes the physical latency bottleneck. Together they render the deliberate data races operationally inexpensive.

```cpp
// H100: pin parent array to ~37.5 MB of L2 (75% of 50 MB)
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
size_t l2_set_aside = min(
    (size_t)(prop.l2CacheSize * 0.75),
    (size_t)prop.persistingL2CacheMaxSize
);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_set_aside);

// Configure access policy: parent array = persistent, everything else = streaming
cudaStreamAttrValue stream_attr = {};
stream_attr.accessPolicyWindow.base_ptr  = parent;
stream_attr.accessPolicyWindow.num_bytes = num_vertices * sizeof(int);
stream_attr.accessPolicyWindow.hitRatio  = 1.0f;
stream_attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
stream_attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
```

**Cache capacity math:**

| GPU | L2 Cache | 75% Set-Aside | Max Vertices (int32) |
|-----|----------|---------------|---------------------|
| H100 | 50 MB | 37.5 MB | ~9.4M vertices pinned |
| A100 | 40 MB | 30 MB | ~7.5M vertices pinned |

For graphs up to ~12.5M vertices, the entire parent array fits within H100 L2. Beyond this, the agent must partition the graph.

### 4.3 The Single Kernel Launch: Structural Advantage

Non-atomic path compression + L2 pinning unlock a profound structural advantage: the entire algorithm collapses into a single, continuous kernel launch.

**Why traditional implementations need multiple launches:**
- Thousands of thread blocks cannot natively synchronize grid-wide without risking deadlock
- Algorithm must periodically yield to host CPU to check convergence flags
- Each kernel boundary transition causes catastrophic L2 cache thrashing — data flushed to DRAM

**Why non-atomic WCC eliminates this:**
- Path compression is entirely race-tolerant — threads that fail a merge do not halt
- Threads re-find roots and retry instantaneously (no convergence array, no host round-trip)
- Iterative logic executes entirely within the lifecycle of a single thread
- No kernel boundary transition → parent array never leaves L2 → unbroken peak throughput from initialization to total graph convergence

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void wcc_single_launch(
    int* parent, const int* row_ptr, const int* col_idx, int N
) {
    auto grid = cg::this_grid();
    bool changed = true;

    while (changed) {
        changed = false;

        int v = blockIdx.x * blockDim.x + threadIdx.x;
        if (v < N) {
            // Non-atomic path compression (deliberate data race — provably safe)
            int root_v = find_root_nonatomic(parent, v);

            for (int e = row_ptr[v]; e < row_ptr[v + 1]; e++) {
                int u = col_idx[e];
                int root_u = find_root_nonatomic(parent, u);

                if (root_v != root_u) {
                    int lo = min(root_v, root_u);
                    int hi = max(root_v, root_u);
                    parent[hi] = lo;   // Non-atomic store — no atomicCAS
                    changed = true;
                }
            }
        }

        grid.sync();  // Grid-level barrier (cooperative launch required)
    }
}

// Helper: non-atomic path compression with path halving
__device__ int find_root_nonatomic(int* parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];   // Path halving — non-atomic, race-tolerant
        x = parent[x];
    }
    return x;
}

// Must use cooperative launch API
void* args[] = {&d_parent, &d_row_ptr, &d_col_idx, &N};
cudaLaunchCooperativeKernel(
    (void*)wcc_single_launch,
    grid_dim, block_dim, args, shared_mem, stream
);
```

### 4.4 Per-Vertex Adaptive Routing

Beyond Union-Find, WarpSpeed pioneered per-vertex adaptive routing for highly divergent workloads. In power-law graphs, a few vertices have millions of edges while the vast majority have very few, causing severe warp divergence.

**The technique:**
1. A lightweight pre-pass kernel calculates an upper-bound estimate of the 2-hop neighborhood size for each vertex (sum of direct neighbor degrees)
2. Based on this heuristic, the kernel dynamically routes execution:
   - Small, manageable neighborhoods: rapid on-chip hash tables
   - Extreme neighborhoods (>4M entries): sort-merge fallback algorithm

This intelligent workload distribution prevents massive hub nodes from stalling the execution pipeline. The KernelForge agent is trained to detect degree-skewed distributions during profiling and generate this routing logic.

### 4.5 Per-Architecture Exhaustive Specialization

Where cuGraph uses shared generic building blocks, WarpSpeed generates a distinct optimized kernel for every valid configuration combination: graph representation (CSR, CSC, COO), weight type (unweighted, int32, float32, float64), precomputation mode (renumbering, degree histograms, none). This combinatorial explosion yields 192 specialized kernel variants per GPU architecture — exhaustive specialization no human team would undertake.

We target H100 only (scope constraint) but apply the principle: generate multiple kernel variants for different graph characteristics (sparse vs dense, power-law vs uniform degree distribution).

---

## 5. Model Selection: Qwen3-Coder-Next

### 5.1 Primary Model

| Property | Value |
|----------|-------|
| HuggingFace ID | `Qwen/Qwen3-Coder-Next` |
| Unsloth GGUF | `unsloth/Qwen3-Coder-Next-GGUF` |
| Architecture | 80B total params, 3B active (MoE + hybrid attention) |
| Base | Built on Qwen3-Next-80B-A3B-Base |
| Context | 256K native (extensible to 1M via YaRN) |
| License | Open-weight |
| Thinking mode | Non-thinking only (no `<think>` blocks — clean for RL) |
| FIM | Fill-in-the-Middle supported (useful for kernel completion) |
| Tool calling | Native support with custom parser |

**Why this model:**
- Newest Qwen coder — purpose-built for coding agents with long-horizon reasoning, complex tool usage, and failure recovery
- Only 3B active despite 80B total — inference is extremely fast
- Specifically trained on "large-scale executable task synthesis, environment interaction, and reinforcement learning" — exactly our use case
- ByteDance's own cudaLLM used Qwen3-8B as base — strong Qwen family precedent for CUDA fine-tuning
- KernelForge environment requires up to 200 interaction turns with 128K context — Qwen3-Coder-Next's 256K context handles this natively

### 5.2 Fallback Model

`Qwen/Qwen3-Coder-30B-A3B-Instruct` — 30B total / 3B active. Smaller download, more battle-tested for Unsloth QLoRA (fits in 17.5GB VRAM). Use if Qwen3-Coder-Next has Unsloth compatibility issues at hackathon time.

### 5.3 Critical Unsloth Notes for MoE Models

```python
# USE FastModel, NOT FastLanguageModel — required for MoE architectures
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="Qwen/Qwen3-Coder-Next",
    max_seq_length=8192,
    load_in_4bit=True,       # QLoRA
    load_in_8bit=False,
    full_finetuning=False,
)

model = FastModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
```

- Router-layer fine-tuning disabled by default (correct for stability — do NOT enable)
- Full 16-bit model must download first, then quantized on-the-fly (BitsAndBytes MoE limitation)
- If QLoRA 4-bit causes instability, switch to bf16 LoRA (needs more VRAM)
- For GRPO with Unsloth on newer models: disable fast vLLM inference, use Unsloth inference instead

### 5.4 Local llama.cpp Setup (MacBook M5 Inference Testing)

```bash
# Download Q3 GGUF for 16GB Mac (Q4 may be tight)
huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF \
    --include "Qwen3-Coder-Next-Q3_K_M/*" \
    --local-dir ./models/

# Build llama.cpp with Metal
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON
cmake --build llama.cpp/build --config Release -j

# Run inference (non-thinking mode)
./llama.cpp/build/bin/llama-cli \
    -m ./models/Qwen3-Coder-Next-Q3_K_M/*.gguf \
    --ctx-size 16384 -ngl 99 \
    --temp 1.0 --top-k 40 --top-p 0.95
```

---

## 6. The SKILL.md Protocol

The SKILL.md document acts as the foundational prompt and rulebook for the agent's interaction loop, explicitly defining how it must approach optimization. It prevents blind code generation by mandating a rigorous, multi-phase execution loop.

### 6.1 Phase 1: Analytical Profiling

The protocol dictates that the agent must never draft CUDA code based on intuition alone. Upon receiving a source implementation, the agent must execute profile.py against the native implementation. The environment is deeply integrated with NVIDIA Nsight Compute (NCU), allowing the agent to parse hardware-level telemetry.

**Required metrics to parse:**
- Global memory load latency (cycles)
- Shared memory bank conflicts (count)
- Register pressure limiting warp occupancy (%)
- Active SM utilization (%)
- L2 cache hit rate (%)
- Kernel launch count (multi-launch = optimization opportunity)

### 6.2 Phase 2: Hardware-Aware Microarchitecture Mapping

The agent cross-references identified profiling bottlenecks against specific target architecture capabilities:

**Memory Bound Resolution (sm_90a):** If NCU profiling data indicates global memory throughput as primary limiter, the agent abandons synchronous loads and synthesizes inline PTX for TMA async bulk transfers.

**Communication Overhead Resolution:** If the algorithm demonstrates high volumes of atomic collisions across distinct thread blocks, the agent configures Thread Block Clusters and allocates DSMEM pointers for direct SM-to-SM communication.

**Instruction Bound Resolution:** If execution stalls on repetitive recursive logic indicative of dynamic programming, the agent replaces C++ logic with DPX intrinsics (__vimin3_s32, __vimax3_s16x2_relu).

**Memory Alignment Constraints:** Enforce strict 128-byte alignments for shared memory allocations. Ensure all Hopper-specific code is guarded by `#if __CUDA_ARCH__ >= 900` compiler directives.

### 6.3 Phase 3: Iterative Execution and Refinement

Compile, verify against 5 adversarial graph inputs, benchmark with synchronized warm-up. If compilation fails or outputs diverge, the environment captures full error traces and feeds them back for automated debugging. The environment supports up to 200 continuous interaction turns with 128K token context for solving exceptionally difficult optimization pathways.

### 6.4 Optimization Priority Hierarchy

**Priority 1 — Algorithmic (target >50% gains, do first):**
- Non-atomic Union-Find path compression (drop atomicCAS entirely)
- L2 persistent cache pinning for parent array
- Single cooperative kernel launch (eliminate host round-trips)
- Kernel fusion (combine hook + compress passes)

**Priority 2 — Hardware Utilization (target 20-50%, do after P1):**
- Thread block clusters (4-8 blocks) with DSMEM for cross-partition labels
- TMA for bulk adjacency list loading with multicast
- Vectorized loads (int4) for edge list traversal
- Warp-level primitives (__shfl_sync for intra-warp label exchange)

**Priority 3 — Instruction-Level (target <20%, polish):**
- DPX instructions for min-label propagation (__vimin3_s32)
- Shared memory bank conflict avoidance (128-byte alignment)
- Occupancy tuning via __launch_bounds__
- Memory access coalescing optimization

### 6.5 Complete SKILL.md File

```markdown
# KernelForge SKILL.md v3.0 — H100 WCC Optimization

## Target Hardware
- GPU: NVIDIA H100 SXM5 (Hopper architecture)
- Compute capability: sm_90a (the 'a' suffix is MANDATORY for TMA/WGMMA)
- nvcc flags: -arch=sm_90a -O3 -use_fast_math
- CUDA version: 12.1+
- L2 Cache: 50 MB (pin up to 37.5 MB for parent array)
- Shared memory/SM: 228 KB
- Thread block clusters: up to 16 co-scheduled blocks
- CUDA cores: 16,896 across 132 SMs
- HBM3 bandwidth: 3.35 TB/s

## Target Algorithm
Weakly Connected Components (Union-Find based)
Baseline: cugraph.weakly_connected_components(G)

## Mandatory 3-Phase Workflow
Deviation from any phase results in reward = -1.0.

### Phase 1: Analytical Profiling (NEVER SKIP)
Run `python verification/profile.py --baseline --ncu` to measure:
- cuGraph WCC runtime (ms) across 5 graph sizes
- Kernel launch count (multi-launch = optimization opportunity)
- Global memory load latency (cycles)
- L2 cache hit rate (%)
- SM utilization (%)
- Shared memory bank conflict count
- Register pressure / occupancy

### Phase 2: Hardware-Aware Implementation
Write CUDA code in kernels/ directory.

MANDATORY RULES:
- NO modification of any file in verification/ (chmod 444)
- NO torch.nn.functional fallback calls in C++ bindings
- NO cuBLAS/cuSPARSE library calls (must generate native CUDA)
- NO network access from sandbox
- Target sm_90a explicitly
- Guard Hopper-specific code: #if __CUDA_ARCH__ >= 900
- 128-byte alignment for all shared memory allocations

OPTIMIZATION PRIORITY (in order):
1. Drop ALL atomicCAS from Union-Find path compression (deliberate data race)
2. Pin parent array to L2 persistent cache via cudaStreamAttrValue
3. Implement single cooperative kernel launch (no host convergence check)
4. Exploit thread block clusters for cross-partition label propagation
5. Use TMA for adjacency list bulk loading with multicast
6. Apply DPX __vimin3_s32 for label propagation min operations
7. Per-vertex adaptive routing for power-law degree distributions

INTERFACE CONTRACT:
Provide Python binding accepting: CSR graph (row_ptr: int32[], col_idx: int32[], num_vertices: int)
Returning: labels array (int32[num_vertices])

### Phase 3: Compile, Verify, Benchmark

Compilation:
  nvcc -arch=sm_90a -O3 -use_fast_math kernel.cu -o kernel.so --shared -Xcompiler -fPIC

PAC Verification (ALL 5 adversarial graphs must pass):
  python verification/pac_verify.py --kernel kernel.so
  - 2x RMAT power-law (skewed degrees expose race conditions at hub nodes)
  - 2x SBM with planted communities (test inter-partition merging)
  - 1x Sparse Erdos-Renyi (isolated vertices, tiny components)
  3 Invariants per graph:
    1. Component COUNT matches networkx exactly
    2. All edges connect same-label vertices
    3. Labels from different reference components are distinct

Benchmarking:
  python verification/profile.py --kernel kernel.so --warmup 100 --runs 50
  - 100 warmup iterations (discarded, cache/hardware stabilization)
  - 50 timed runs with cudaEventRecord + cudaEventSynchronize
  - Report: mean, median, std, min, max (milliseconds)
  - Speedup = baseline_median / kernel_median

## Anti-Hacking Controls
1. verification/ directory: chmod 444 (immutable to agent)
2. Profiling/timing scripts immutable — agent cannot tamper
3. Mandatory warm-up prevents cold-start speedup inflation
4. No external binary calls (cuBLAS, cuSPARSE) — must generate native CUDA/PTX
5. Sandbox isolated from external network
6. AST check for disallowed patterns (torch fallbacks, hardcoded outputs)
7. 5 randomized graph inputs prevent lookup table hacking
8. Process isolation — agent cannot fork/exec outside sandbox
```

---

## 7. The Agentic RL Training Pipeline

### 7.1 Why Raw Speedup Rewards Destroy Training

A critical engineering insight: utilizing raw execution speedup ratios as a continuous, linear reward signal is inherently unstable and destructive to the training process. Raw execution times in a physical GPU cluster are highly susceptible to systemic noise, thermal throttling, interconnect congestion, and minor operating system interrupts. These micro-fluctuations create massive outliers that severely bias advantage estimation algorithms, leading the model to hallucinate false correlations between code changes and execution speed.

ByteDance's ablation studies: relying on raw speedup metrics reduces the overall success rate by up to 36 percentage points.

### 7.2 Robust Milestone-Based Discrete Reward Function

KernelForge discards raw speedups in favor of normalized, milestone-based discrete reward tiers:

```
r in {-1, +1, +2, +3}
```

| Reward | Condition | Meaning |
|--------|-----------|---------|
| **-1** | Compilation failure OR verification failure | Catastrophic penalty — corrupted logic |
| **+1** | Compiles + correct, but no meaningful speedup (<=5% over cuGraph) | Baseline success |
| **+2** | Correct + >5% faster than cuGraph baseline | Meaningful speedup — legitimate optimization |
| **+3** | Correct + >5% faster than BOTH cuGraph AND doubleGraph | State-of-the-art — agent exploited H100-specific primitives |

By discretizing the reward landscape, the system normalizes execution data, forcing policy networks to emphasize genuine, structural kernel quality improvements rather than chasing anomalous execution spikes.

```python
def cuda_kernel_reward(completions, **kwargs) -> list[float]:
    """
    ByteDance milestone-based discrete rewards.
    Dispatched to Modal H100 for evaluation.
    """
    import modal
    eval_fn = modal.Function.lookup("kernelforge-h100", "evaluate_kernel")
    rewards = []

    for completion in completions:
        code = extract_cuda_code(completion)
        if not code:
            rewards.append(-1.0)
            continue

        result = eval_fn.remote({
            "cuda_code": code,
            "verify_graphs": 5,
            "warmup_iters": 100,
            "benchmark_runs": 50,
        })

        if not result.get("compiles") or not result.get("correct"):
            rewards.append(-1.0)
        elif result.get("speedup_vs_dg", 0) > 1.05:
            rewards.append(3.0)    # Beats BOTH baselines
        elif result.get("speedup_vs_orig", 0) > 1.05:
            rewards.append(2.0)    # Beats cuGraph
        else:
            rewards.append(1.0)    # Correct but slow

    return rewards
```

### 7.3 The Four-Stage Training Methodology

Standard RL paradigms frequently fail when applied to complex code generation due to policy collapse and distribution mismatch during long-horizon reasoning. ByteDance found that naive RL collapses at step 17 because CUDA code is <0.01% of pretraining data, creating severe domain mismatch. The 4-stage pipeline safely ushers the model into handling 200-turn debug cycles:

**Stage 1 — Single-Turn PPO Warm-Up (ByteDance) / SFT Warm-Up (KernelForge)**

The foundational phase trains the model on a massive synthesized dataset of known, correct CUDA operators (CUDA-Agent-Ops-6K: 6,000 synthetic operators via LLM combinatorial synthesis). The explicit goal: establish strong baseline proficiency in C++ memory management and low-level PTX syntax. RL is restricted to single-turn optimization — no multi-turn debugging until baseline coding capability is stabilized. Context: 32K tokens.

Our adaptation: Supervised fine-tuning (SFT) on 50-100 WCC CUDA kernel examples using TRL SFTTrainer + Unsloth. Generated via Claude/GPT API, filtered by compilation + correctness on Modal H100.

**Stage 2 — Rejection Fine-Tuning (RFT) for Actor Initialization**

The agent explores the interactive environment and attempts multi-turn optimizations. The system samples resulting trajectories and aggressively filters:
- KEEP: Only trajectories with definitive positive outcome (reward >= 2)
- DISCARD: Trajectories with redundant debugging loops, repeated identical failures, invalid tool calls, or reward hacking attempts

The primary actor model is heavily fine-tuned on this pristine, filtered dataset.

*Why RFT is critical:* Bypassing it allows the actor's policy entropy to explode as it becomes lost in unproductive debugging cycles, leading to total training collapse.

**Stage 3 — Value Pretraining for Critic Network (ByteDance only — we skip)**

In ByteDance's PPO setup, the critic network must accurately estimate the underlying "value" of specific code states to guide the actor. An uninitialized critic demonstrates exceptionally low explained variance, resulting in highly inefficient exploration and pathological search behaviors.

*Our adaptation:* GRPO (Group Relative Policy Optimization) eliminates the critic entirely by normalizing rewards within generation groups. This saves VRAM and simplifies the pipeline — ideal for hackathon constraints.

**Stage 4 — Multi-Turn Agentic RL**

With the actor stabilized via RFT (and critic pretrained in PPO variants), models deploy into the full KernelForge environment. The agent engages maximum-capacity multi-turn interactions: iteratively utilizing Nsight Compute profiler, systematically rewriting code architectures based on deep execution feedback, and aggressively exploiting Hopper memory features.

ByteDance configuration: 150 steps, 128 H20 GPUs, 131K context, up to 150 agent turns.
KernelForge hackathon: 100 steps on Modal H100 (or CoreWeave credits).

### 7.4 Our Adapted Pipeline (3 Stages, GRPO)

```
ByteDance (PPO, 4 stages)              KernelForge (GRPO, 3 stages)
----------------------------------      ----------------------------------
Stage 1: Single-turn PPO warm-up   -->  Stage 1: SFT warm-up on CUDA examples
Stage 2: Rejection fine-tuning     -->  Stage 2: RFT (keep reward >= 2 only)
Stage 3: Critic value pretraining  -->  SKIPPED (GRPO has no critic)
Stage 4: Multi-turn agentic PPO   -->  Stage 3: GRPO with milestone rewards
```

---

## 8. System 3 Verification: PAC-Reasoning

### 8.1 The Data Scarcity Problem

A fundamental barrier to deploying Artificial Expert Intelligence for GPU performance engineering is extreme scarcity of high-quality ground truth data. Optimal GPU performance is not derived from simple syntax; it emerges from a massive, interacting chain of highly specific hardware decisions regarding memory layouts, warp behaviors, branching strategies, and instruction scheduling. Only a few thousand truly optimal, hardware-specific CUDA kernels exist globally. For novel algorithms, there is simply no existing human-engineered optimal baseline to train against.

### 8.2 PAC-Reasoning Architecture

KernelForge overcomes this data scarcity by integrating PAC (Probably Approximately Correct) Reasoning, derived from DoubleAI's methodology (arXiv:2412.02441, "Artificial Expert Intelligence through PAC-reasoning", December 2024).

**Core computational insight:** The mathematical verification of a complex solution is inherently simpler and less computationally expensive than the unstructured search for the optimal solution itself. This establishes a "System 3" reasoning process — analogous to the rigorous, empirical validation central to the scientific method — allowing the system to control error accumulation during long reasoning chains by investing inference-time compute into generating its own empirical correctness guarantees.

### 8.3 Two Adversarial Components

**1. Input Generator (IG)**

A secondary, adversarial model explores the problem space and synthesizes extremely challenging, edge-case test inputs specifically designed to break standard parallelization strategies. For Union-Find WCC:

- Massively scaled, heavily skewed power-law graphs (RMAT) designed to cause catastrophic warp divergence and memory saturation at hub nodes
- Stochastic Block Model graphs with planted communities testing inter-community merging across partition boundaries
- Sparse Erdos-Renyi graphs with many isolated vertices and tiny components (boundary conditions)
- Graphs with near-identical component sizes (stress-tests merge ordering)
- Graphs with single-edge bridges between large communities

**2. Algorithmic Verifier (AV)**

The system simultaneously synthesizes a brute-force, mathematically pure implementation of the target algorithm. Designed exclusively for guaranteed logical correctness, entirely ignoring execution latency or hardware efficiency (executed sequentially on host CPU via networkx).

**Three mathematical invariants checked:**
1. Component count matches reference exactly
2. Every edge connects vertices with the same label
3. Vertices in different reference components have different labels

### 8.4 The Self-Improving Flywheel

The agent's optimized, unproven, potentially race-tolerant kernel is executed against the adversarial IG datasets. Outputs are strictly compared against verified AV outputs.

As the primary agent generates faster and increasingly complex kernels, the verification engine automatically scales up, generating increasingly hostile inputs and more rigorous validation checks. This creates a massive, self-improving data flywheel that bootstraps reliable, mathematically guaranteed self-verification dynamically at inference time — completely eliminating reliance on scarce human-engineered ground truth data.

```python
def generate_test_graphs(num_vertices: int = 10000) -> list:
    """Input Generator: 5 adversarial graphs."""
    graphs = []

    # 1-2: RMAT power-law (skewed degrees -> race conditions at hubs)
    for seed in [42, 137]:
        edges = generate_rmat(num_vertices, num_vertices * 10, seed)
        graphs.append(("rmat", edges, num_vertices))

    # 3-4: SBM planted communities (cross-partition merging)
    for n_comm in [5, 50]:
        sizes = [num_vertices // n_comm] * n_comm
        p_matrix = [[0.1 if i==j else 0.001 for j in range(n_comm)]
                     for i in range(n_comm)]
        G = nx.stochastic_block_model(sizes, p_matrix, seed=n_comm)
        graphs.append(("sbm", list(G.edges()), sum(sizes)))

    # 5: Sparse Erdos-Renyi (many isolates, tiny components)
    G = nx.erdos_renyi_graph(num_vertices, 0.0005, seed=99)
    graphs.append(("er_sparse", list(G.edges()), num_vertices))

    return graphs


def verify_wcc(kernel_labels, edges, num_vertices) -> tuple:
    """Algorithmic Verifier: 3 mathematical invariants."""
    G = nx.Graph()
    G.add_nodes_from(range(num_vertices))
    G.add_edges_from(edges)
    ref = list(nx.connected_components(G))

    # Invariant 1: component count
    kernel_count = len(set(kernel_labels.get(v, v) for v in range(num_vertices)))
    if kernel_count != len(ref):
        return False, f"Count: kernel={kernel_count} vs ref={len(ref)}"

    # Invariant 2: edge consistency
    for u, v in edges:
        if kernel_labels.get(u, u) != kernel_labels.get(v, v):
            return False, f"Edge ({u},{v}) crosses components"

    # Invariant 3: cross-component distinctness
    label_to_comp = {}
    for comp_id, comp in enumerate(ref):
        for v in comp:
            l = kernel_labels.get(v, v)
            if l in label_to_comp and label_to_comp[l] != comp_id:
                return False, f"Label {l} spans multiple components"
            label_to_comp[l] = comp_id

    return True, f"Verified: {len(ref)} components, all invariants hold"


def generate_rmat(n, m, seed):
    """RMAT Kronecker graph generator (Graph500 standard params)."""
    rng = np.random.default_rng(seed)
    a, b, c = 0.57, 0.19, 0.19
    edges = set()
    for _ in range(m):
        u, v = 0, 0
        step = n // 2
        while step >= 1:
            r = rng.random()
            if r < a: pass
            elif r < a + b: v += step
            elif r < a + b + c: u += step
            else: u += step; v += step
            step //= 2
        if u != v and u < n and v < n:
            edges.add((min(u, v), max(u, v)))
    return list(edges)
```

---

## 9. Anti-Reward-Hacking Measures

A universally documented pathology within complex RL environments: the agent discovers shortcuts to artificially maximize reward without fulfilling engineering intent. For example, an agent might deduce the specific test inputs and hardcode a lookup table of correct answers, bypassing all computation while achieving zero latency and perfect correctness.

KernelForge enforces draconian operational controls:

| Control | What It Prevents |
|---------|-----------------|
| Profiling/validation scripts chmod 444 (immutable) | Agent tampering with timing mechanisms or correctness checks |
| Synchronized warm-up profiling (100 untimed iters) | Cold-start baseline exploitation (comparing cached vs cold) |
| No cuBLAS/cuSPARSE fallback calls | Shortcutting via pre-compiled, highly optimized external binaries |
| Must generate native CUDA/PTX | Forces actual algorithmic work, not library delegation |
| Sandbox isolated from network | Prevents downloading pre-optimized kernels or external code |
| 5 randomized graph inputs per evaluation | Prevents hardcoded output lookup tables |
| AST check for disallowed patterns | Catches torch fallbacks, system calls, hardcoded arrays |
| Process isolation | Agent cannot fork, exec, or escape sandbox boundaries |
| cudaDeviceSynchronize before all timing | No async cheating — true wall-clock measurement |

---

## 10. Core Implementation Files

### 10.1 Repository Structure

```
KernelForge-OpenEnv/
├── README.md
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── skill_h100.md                        # SKILL.md v3.0 (Section 6.5)
│
├── openenv_env/
│   ├── __init__.py
│   └── kernel_forge_env.py              # OpenEnv Environment subclass
│
├── training/
│   ├── sft_warmup.py                    # Stage 1: SFT warm-up on CUDA examples
│   ├── rft_filter.py                    # Stage 2: rejection fine-tuning filter
│   └── grpo_train.py                    # Stage 3: TRL GRPOTrainer + Unsloth
│
├── datasets/
│   ├── generate_wcc_dataset.py          # RMAT/SBM/ER graph generation
│   └── wcc_training.jsonl               # SFT training data (generated)
│
├── verification/
│   ├── pac_verify.py                    # PAC-reasoning (Input Generator + Algo Verifier)
│   └── profile.py                       # H100 benchmarking (NCU + cudaEvent)
│
├── modal_app.py                         # Modal H100 serverless functions
│
├── kernels/
│   ├── baseline_wcc.cu                  # Reference cuGraph-style WCC
│   ├── ecl_cc_h100.cu                   # Non-atomic Union-Find + L2 pinning
│   ├── clustered_wcc_h100.cu            # Thread block clusters + DSMEM
│   └── tma_wcc_h100.cu                  # TMA adjacency loading variant
│
└── demo/
    └── streamlit_demo.py                # Live demo for hackathon judging
```

### 10.2 openenv_env/kernel_forge_env.py

```python
"""
KernelForge OpenEnv Environment for H100 CUDA kernel RL training.

OpenEnv is an independent framework from Meta-PyTorch (NOT a Gymnasium
extension). HTTP client-server architecture with Docker container isolation.
Correct install: pip install "openenv-core[core]>=0.2.1" (NOT "openenv")
"""
from openenv.core.env_server import Environment, create_fastapi_app
from openenv.core.types import StepResult
import os


class KernelForgeEnv(Environment):
    """
    RL environment: agent submits CUDA source -> H100 compiles/verifies/benchmarks.

    Action space: CUDA source code string
    Observation: SKILL.md + compilation/verification/benchmark feedback + history
    Reward: discrete milestones {-1, +1, +2, +3}
    Max turns: 200 (ByteDance used 150; extended for hackathon exploration)
    Context: 128K tokens
    """

    def __init__(self, modal_function_name="kernelforge-h100"):
        self.modal_fn = modal_function_name
        self.history = []               # Time-travel snapshots (DoubleAI-inspired)
        self.turn = 0
        self.max_turns = 200
        self.best_reward = -1.0
        self.best_code = None
        self.original_baseline_ms = None
        self.doublegraph_baseline_ms = None

    def reset(self):
        """Reset environment. Profile baselines on first call."""
        self.history = []
        self.turn = 0
        self.best_reward = -1.0
        self.best_code = None

        if self.original_baseline_ms is None:
            baselines = self._modal("profile_baselines")
            self.original_baseline_ms = baselines["original_ms"]
            self.doublegraph_baseline_ms = baselines.get("doublegraph_ms")

        return {
            "observation": self._load_skill_md(),
            "baseline_original_ms": self.original_baseline_ms,
            "baseline_doublegraph_ms": self.doublegraph_baseline_ms,
            "hardware": {
                "gpu": "H100 SXM5",
                "arch": "sm_90a",
                "l2_cache_mb": 50,
                "smem_per_sm_kb": 228,
                "sms": 132,
                "cuda_cores": 16896,
                "hbm_bandwidth_tbs": 3.35,
                "cluster_max_blocks": 16,
                "has_tma": True,
                "has_dsmem": True,
                "has_dpx": True,
            },
        }

    def step(self, action: str) -> StepResult:
        """action: CUDA kernel source code string."""
        self.turn += 1

        result = self._modal("evaluate_kernel", {
            "cuda_code": action,
            "verify_graphs": 5,
            "warmup_iters": 100,
            "benchmark_runs": 50,
        })

        # Reward calculation (ByteDance milestone-based)
        if not result.get("compiles"):
            reward = -1.0
            obs = (f"COMPILATION FAILED (turn {self.turn}/{self.max_turns}):\n"
                   f"{result.get('error', 'Unknown error')[:1500]}")
        elif not result.get("correct"):
            reward = -1.0
            obs = (f"VERIFICATION FAILED (turn {self.turn}/{self.max_turns}):\n"
                   f"{result.get('verifier_msg', 'Unknown failure')}")
        else:
            rt = result["runtime_ms"]
            su_orig = self.original_baseline_ms / rt if rt > 0 else 0
            su_dg = None
            if self.doublegraph_baseline_ms and rt > 0:
                su_dg = self.doublegraph_baseline_ms / rt

            if su_dg and su_dg > 1.05:
                reward = 3.0     # Beats BOTH baselines
            elif su_orig > 1.05:
                reward = 2.0     # Beats cuGraph
            else:
                reward = 1.0     # Correct but no meaningful speedup

            obs = (f"BENCHMARK (turn {self.turn}/{self.max_turns}):\n"
                   f"  Runtime: {rt:.3f}ms\n"
                   f"  vs cuGraph: {su_orig:.2f}x")
            if su_dg is not None:
                obs += f"\n  vs doubleGraph: {su_dg:.2f}x"
            obs += f"\n  Stats: {result.get('runtime_stats', {})}"

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_code = action

        done = (self.turn >= self.max_turns) or (reward == 3.0)

        # Time-travel with experience (DoubleAI-inspired)
        # Each snapshot carries knowledge of what was tried and what failed
        self.history.append({
            "turn": self.turn,
            "reward": reward,
            "obs_summary": obs[:200],
        })

        # Include history in observation for time-travel context
        if len(self.history) > 1:
            history_ctx = "\n--- Previous attempts (time-travel context) ---\n"
            for h in self.history[:-1]:
                history_ctx += (f"Turn {h['turn']}: reward={h['reward']}, "
                               f"{h['obs_summary'][:80]}\n")
            obs = history_ctx + "\n--- Current result ---\n" + obs

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "turn": self.turn,
                "best_reward": self.best_reward,
                "speedup": su_orig if result.get("correct") else 0,
            },
        )

    @property
    def state(self):
        return {
            "turn": self.turn,
            "history": self.history,
            "best_reward": self.best_reward,
        }

    def _modal(self, fn_name, payload=None):
        """Dispatch to Modal H100."""
        import modal
        fn = modal.Function.lookup(self.modal_fn, fn_name)
        return fn.remote(payload or {})

    def _load_skill_md(self):
        skill_path = os.path.join(os.path.dirname(__file__), "..", "skill_h100.md")
        with open(skill_path) as f:
            return f.read()


# OpenEnv HTTP server entrypoint
# Run: uvicorn openenv_env.kernel_forge_env:app --host 0.0.0.0 --port 8080
app = create_fastapi_app(KernelForgeEnv)
```

### 10.3 training/grpo_train.py

```python
"""
GRPO Training for H100 CUDA Kernel Generation.

CRITICAL IMPORT CLARIFICATION:
- GRPOTrainer and GRPOConfig: from TRL (HuggingFace), NOT from Unsloth
- FastModel: from Unsloth (required for MoE model loading)
- Unsloth patches TRL behind the scenes for memory efficiency

ByteDance used PPO (4 stages, 230B model, 128 H20 GPUs).
We adapt to GRPO (3 stages) because:
1. No critic network needed -> saves VRAM
2. TRL has native GRPOTrainer
3. Hackathon-feasible compute budget
"""
from unsloth import FastModel          # NOT FastLanguageModel — MoE requires this
from trl import GRPOConfig, GRPOTrainer  # FROM TRL, not from Unsloth
from datasets import load_dataset


# === Stage 1: Load Model with Unsloth FastModel (MoE-compatible) ===

model, tokenizer = FastModel.from_pretrained(
    model_name="Qwen/Qwen3-Coder-Next",   # 80B total / 3B active MoE
    max_seq_length=8192,
    load_in_4bit=True,        # QLoRA for hackathon VRAM constraints
    load_in_8bit=False,
    full_finetuning=False,
)


# === Stage 2: Attach LoRA Adapters ===

model = FastModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",  # Reduces VRAM, extends context
)


# === Stage 3: Define Reward Function ===

def cuda_kernel_reward(completions, **kwargs) -> list[float]:
    """
    Composite reward dispatched to Modal H100.
    ByteDance milestone-based discrete rewards:
      -1.0  = compilation failure OR verification failure
      +1.0  = compiles and correct, but no speedup
      +2.0  = faster than cuGraph baseline by >5%
      +3.0  = faster than BOTH cuGraph AND doubleGraph by >5%

    Why discrete milestones > continuous speedup:
    ByteDance ablation: raw speedup metrics reduce success by 36 points.
    """
    import modal
    eval_fn = modal.Function.lookup("kernelforge-h100", "evaluate_kernel")
    rewards = []

    for completion in completions:
        code = _extract_cuda_code(completion)
        if not code:
            rewards.append(-1.0)
            continue

        result = eval_fn.remote({
            "cuda_code": code,
            "verify_graphs": 5,
            "warmup_iters": 100,
            "benchmark_runs": 50,
        })

        if not result.get("compiles") or not result.get("correct"):
            rewards.append(-1.0)
        elif result.get("speedup_vs_dg", 0) > 1.05:
            rewards.append(3.0)
        elif result.get("speedup_vs_orig", 0) > 1.05:
            rewards.append(2.0)
        else:
            rewards.append(1.0)

    return rewards


def _extract_cuda_code(completion: str) -> str:
    """Extract CUDA code block from model output."""
    for marker in ["```cuda", "```cpp", "```c"]:
        if marker in completion:
            start = completion.index(marker) + len(marker)
            end = completion.index("```", start)
            return completion[start:end].strip()
    return completion.strip()


# === Stage 4: Configure GRPO Training ===

training_args = GRPOConfig(
    learning_rate=5e-6,
    num_generations=4,               # Reduced from ByteDance's 1024
    max_prompt_length=512,
    max_completion_length=4096,      # CUDA kernels can be long
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=100,                   # Hackathon time budget
    optim="paged_adamw_8bit",
    bf16=True,
    report_to="none",                # Set to "wandb" if configured
    output_dir="outputs/kernelforge-grpo",
    logging_steps=1,
)


# === Stage 5: Load Dataset and Train ===

dataset = load_dataset("json", data_files="datasets/wcc_training.jsonl", split="train")

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[cuda_kernel_reward],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# Save LoRA adapters + tokenizer
model.save_pretrained("outputs/kernelforge-qwen3-coder-next")
tokenizer.save_pretrained("outputs/kernelforge-qwen3-coder-next")
```

### 10.4 modal_app.py

```python
"""
Modal Serverless Functions for H100 CUDA Kernel Evaluation.
All GPU work runs here — compilation, verification, benchmarking.
"""
import modal

cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        "cupy-cuda12x>=14.0",
        "networkx>=3.0",
        "numpy>=1.26",
        "scipy>=1.12",
    )
)

app = modal.App("kernelforge-h100")
kernel_cache = modal.Volume.from_name("kernelforge-cache", create_if_missing=True)


@app.function(
    gpu="H100",
    image=cuda_image,
    timeout=120,
    volumes={"/cache": kernel_cache},
)
def evaluate_kernel(payload: dict) -> dict:
    """
    Compile, verify (PAC-reasoning), and benchmark CUDA kernel on real H100.

    Input:
        cuda_code: str          - CUDA source
        verify_graphs: int      - number of test graphs (default 5)
        warmup_iters: int       - warmup iterations (default 100)
        benchmark_runs: int     - timed runs (default 50)

    Returns:
        compiles: bool
        correct: bool
        verifier_msg: str
        runtime_ms: float       - median of benchmark_runs
        runtime_stats: dict     - {mean, median, std, min, max}
        speedup_vs_orig: float
        speedup_vs_dg: float
        error: str
    """
    import subprocess
    import tempfile
    import os
    import ctypes
    import numpy as np

    cuda_code = payload.get("cuda_code", "")
    result = {
        "compiles": False, "correct": False, "verifier_msg": "",
        "runtime_ms": 0.0, "runtime_stats": {},
        "speedup_vs_orig": 0.0, "speedup_vs_dg": 0.0, "error": "",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.cu")
        lib_path = os.path.join(tmpdir, "kernel.so")

        with open(src_path, "w") as f:
            f.write(cuda_code)

        # Step 1: Compile with sm_90a (mandatory 'a' suffix for TMA/WGMMA)
        try:
            proc = subprocess.run(
                ["nvcc", "-arch=sm_90a", "-O3", "-use_fast_math",
                 src_path, "-o", lib_path, "--shared", "-Xcompiler", "-fPIC"],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode != 0:
                result["error"] = proc.stderr[:2000]
                return result
        except subprocess.TimeoutExpired:
            result["error"] = "Compilation timed out (30s limit)"
            return result

        result["compiles"] = True

        # Step 2: PAC Verification (5 adversarial graphs)
        try:
            from pac_verify import generate_test_graphs, verify_wcc

            graphs = generate_test_graphs(num_vertices=10000)
            lib = ctypes.CDLL(lib_path)

            all_passed = True
            for graph_type, edges, n_verts in graphs:
                kernel_labels = _run_kernel_ffi(lib, edges, n_verts)
                passed, msg = verify_wcc(kernel_labels, edges, n_verts)
                if not passed:
                    result["verifier_msg"] = f"FAILED on {graph_type}: {msg}"
                    all_passed = False
                    break

            result["correct"] = all_passed
            if all_passed:
                result["verifier_msg"] = f"All {len(graphs)} graphs verified"
        except Exception as e:
            result["correct"] = False
            result["verifier_msg"] = f"Verification exception: {str(e)[:500]}"
            return result

        # Step 3: Benchmark with cudaEvent timing
        if result["correct"]:
            try:
                import cupy as cp

                warmup = payload.get("warmup_iters", 100)
                runs = payload.get("benchmark_runs", 50)

                # Warmup (mandatory — prevents cold-start exploitation)
                for _ in range(warmup):
                    _run_kernel_ffi(lib, graphs[0][1], graphs[0][2])
                cp.cuda.Device(0).synchronize()

                # Timed runs with cudaEvent
                times = []
                for _ in range(runs):
                    start = cp.cuda.Event()
                    end = cp.cuda.Event()
                    start.record()
                    _run_kernel_ffi(lib, graphs[0][1], graphs[0][2])
                    end.record()
                    end.synchronize()
                    times.append(cp.cuda.get_elapsed_time(start, end))

                times = np.array(times)
                result["runtime_ms"] = float(np.median(times))
                result["runtime_stats"] = {
                    "mean": float(np.mean(times)),
                    "median": float(np.median(times)),
                    "std": float(np.std(times)),
                    "min": float(np.min(times)),
                    "max": float(np.max(times)),
                }
            except Exception as e:
                result["error"] = f"Benchmark exception: {str(e)[:500]}"

    return result


@app.function(gpu="H100", image=cuda_image, timeout=300)
def profile_baselines() -> dict:
    """Profile cuGraph WCC and doubleGraph baselines on standard test graphs."""
    # Implementation: install cuGraph + doubleGraph, measure on RMAT graphs
    return {
        "original_ms": 1.0,       # Stub — replace with actual measurement
        "doublegraph_ms": 0.1,    # Stub — replace with actual measurement
    }


def _run_kernel_ffi(lib, edges, n_verts) -> dict:
    """Run compiled kernel via ctypes. Depends on kernel's exported function."""
    # Implementation: convert edges to CSR, call lib.wcc_kernel(row_ptr, col_idx, N, labels)
    raise NotImplementedError("Wire up ctypes FFI to kernel's exported function")
```

---

## 11. Tech Stack and Infrastructure

### 11.1 Python Dependencies (requirements.txt)

```
# Core training
unsloth>=2025.3            # MoE fine-tuning with FastModel
trl>=0.16.0                # GRPOTrainer lives HERE, not in Unsloth
transformers>=4.48,<5.0    # Unsloth does NOT support transformers 5.0+ yet
torch>=2.10.0
datasets>=3.0

# OpenEnv
openenv-core[core]>=0.2.1  # NOT "openenv" (that's an unrelated 2020 package)
gymnasium>=1.0

# CUDA evaluation (runs on Modal H100, not locally)
cupy-cuda12x>=14.0         # Matches CUDA 12.1 Docker image
networkx>=3.0              # Reference WCC for PAC verification
numpy>=1.26
scipy>=1.12

# Infrastructure
modal>=0.70                # H100 cloud dispatch (~$3.95/hr)
wandb>=0.19                # Experiment tracking (optional)
```

### 11.2 Docker Image (Modal H100)

```python
# CUDA 12.1 — sm_90a features (TMA, WGMMA, DPX) available since CUDA 12.0
# CUDA 13.x exists but 12.1 has broadest PyTorch/Docker compatibility
cuda_image = modal.Image.from_registry(
    "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04",
    add_python="3.12",
)
```

### 11.3 Key Version Constraints

| Dependency | Constraint | Reason |
|------------|-----------|--------|
| transformers | >=4.48, <5.0 | Unsloth incompatible with transformers 5.0+ |
| unsloth | >=2025.3 | MoE FastModel support required |
| trl | >=0.16.0 | GRPOTrainer with rollout_func |
| CUDA toolkit | 12.1+ | sm_90a features (TMA, WGMMA, DPX) since 12.0 |
| openenv-core | >=0.2.1 | NOT "openenv" (wrong package) |
| cupy | cupy-cuda12x | Must match CUDA 12.1 Docker image |

---

## 12. Hackathon Execution Plan

### Day 0: Now Through March 6 (PREPARATION)

**Hour 1-2: Infrastructure Setup**
1. Create Modal account, verify H100 access: `modal setup`
2. Deploy stub modal_app.py: `modal deploy modal_app.py`
3. Test H100 compilation: `modal run modal_app.py::evaluate_kernel`
4. Install llama.cpp on MacBook for local Qwen3-Coder-Next inference testing
5. Download GGUF: `huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF --include "Qwen3-Coder-Next-Q3_K_M/*"`

**Hour 3-5: OpenEnv Environment**
6. Implement full KernelForgeEnv with Modal dispatch (Section 10.2)
7. Test reset() and step() with a trivial CUDA kernel
8. Implement PAC verification with all 5 adversarial graph types (Section 8.3)
9. Wire up cudaEvent benchmarking in Modal function

**Hour 6-8: Dataset and Baselines**
10. Profile cuGraph WCC on H100 via Modal (get actual baseline numbers)
11. Install doubleGraph A100 wheel and profile for reference
12. Generate 50-100 WCC CUDA kernel examples for SFT (use Claude API to generate, filter by compilation + correctness on Modal)
13. Format as datasets/wcc_training.jsonl

**Hour 9-10: Training Pipeline Dry Run**
14. Test Unsloth FastModel loading of Qwen3-Coder-Next on cloud GPU
15. Run 5-step SFT warm-up to verify full pipeline end-to-end
16. Verify GRPO reward dispatch cycle: generate -> Modal -> compile -> verify -> reward -> update

### Day 1: March 7 (Hackathon — BUILD)

**Morning (4 hours):**
1. Stage 1: SFT warm-up on WCC kernel examples (TRL SFTTrainer + Unsloth FastModel)
2. Collect trajectories from warm-up model
3. Stage 2: Rejection fine-tuning — filter trajectories, keep only reward >= 2

**Afternoon (4 hours):**
4. Stage 3: Start GRPO training with milestone rewards (target: 100 steps)
5. Monitor reward curve in real-time (wandb or terminal)
6. If reward plateaus at +1 (correct but slow): analyze NCU profiles, adjust SKILL.md prompts to emphasize non-atomic path compression + L2 pinning
7. If reward plateaus at +2 (beats cuGraph but not doubleGraph): introduce thread block cluster + TMA prompts

### Day 2: March 8 (Hackathon — SHIP)

**Morning (3 hours):**
1. Continue GRPO if still improving; freeze if plateaued
2. Run final evaluation: best kernel on large graphs (100K, 1M, 10M vertices)
3. Compute official speedup numbers vs cuGraph + doubleGraph

**Afternoon (3 hours):**
4. Build Streamlit demo showing agent optimizing a kernel in real-time
5. Push model + environment + results to HuggingFace Hub
6. Prepare 3-minute pitch: "We trained an RL agent to write H100 CUDA kernels that beat a decade of NVIDIA expert code using ByteDance's training pipeline and DoubleAI's optimization heuristics"

---

## 13. Sponsor Integration Strategy

| Sponsor | What They Provide | How We Leverage |
|---------|------------------|-----------------|
| **PyTorch/Meta** | OpenEnv framework, core organizer | Our Environment subclass runs on their spec |
| **Unsloth** | FastModel for MoE, 2x faster training, 70% less VRAM, Pro access + dedicated support | Qwen3-Coder-Next QLoRA training via FastModel |
| **Hugging Face** | TRL (GRPOTrainer), model hub, OpenEnv integration, $10K HF credits (OpenEnv Challenge) | Training loop + model/env publishing to Hub |
| **CoreWeave** | 250K+ GPUs, H100/H200/GB200 hardware, accelerator compute credits | Training infrastructure (supplement Modal) |
| **Cursor** | AI pair programming ($400+ credits per participant) | All local development |
| **Mercor** | APEX-Agents benchmark (33 worlds, 480 tasks), Archipelago execution infra | Evaluate agent's long-horizon planning capability |
| **Snorkel AI** | Safe RL practices | Milestone rewards prevent reward hacking |
| **Scale AI** | Data quality infrastructure | SFT dataset curation |
| **UC Berkeley / SkyRL** | RL research grounding | PAC verification theoretical foundation |

---

## 14. Corrections from Prior Versions

| PRD v1.x Error | v3.0 Correction | Reason |
|----------------|-----------------|--------|
| Model: `Qwen3.5-9B-Instruct` | `Qwen3-Coder-Next` (80B/3B MoE) | Qwen3.5 has no -Instruct suffix; Coder-Next purpose-built for coding agents |
| Import: `from unsloth import GRPOTrainer` | `from trl import GRPOConfig, GRPOTrainer` | GRPOTrainer is a TRL class, not Unsloth |
| Loader: `FastLanguageModel` | `FastModel` | MoE models require FastModel |
| Package: `pip install openenv` | `pip install openenv-core[core]` | "openenv" on PyPI is an unrelated 2020 package |
| Docker: `nvidia/cuda:13.0.2-devel` | `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` | CUDA 12.1 has broader PyTorch compatibility; sm_90a works since 12.0 |
| CuPy: `cupy-cuda13x` | `cupy-cuda12x` | Must match CUDA 12.1 Docker image |
| Arch flag: `-arch=sm_90` | `-arch=sm_90a` | The `a` suffix is mandatory for TMA and WGMMA instructions |
| OpenEnv: "Gymnasium extension" | Independent framework with Gymnasium-inspired API | HTTP client-server architecture, not a Gym wrapper |
| DoubleAI: "time-travel search" | "Time-travel with experience" | Correct terminology per their publications |
| ByteDance RL: claimed GRPO | They used PPO; we adapt to GRPO | GRPO needs no critic — simpler for hackathon |
| Reward: continuous speedup | Discrete milestones {-1, +1, +2, +3} | ByteDance ablation: raw speedup reduces success by 36 points |
| Missing hardware detail | Full A100 vs H100 transition matrix | Complete architectural comparison with agent synthesis implications |
| Missing TMA formalization | Complete TMA + multicast documentation | Including inline PTX examples and performance metrics |
| Missing DSMEM detail | Thread block cluster + DSMEM with code | Including cooperative_groups API and launch configuration |
| Missing DPX coverage | DPX instructions for label propagation | Including __vimin3_s32 and performance impact data |
| Missing deliberate data race theory | Full mathematical safety proof | 4 conditions: monotone convergence, no value creation, self-correction, race tolerance |
| Missing PAC-reasoning | System 3 verification architecture | Input Generator + Algorithmic Verifier with self-improving flywheel |
| Missing anti-hacking controls | 9 draconian enforcement mechanisms | From chmod 444 to AST checks to process isolation |
| Missing per-vertex adaptive routing | WarpSpeed divergent workload handling | Pre-pass kernel with dynamic routing based on 2-hop neighborhood estimate |

---

## 15. Works Cited

1. Weinan Dai, Hanlin Wu, et al. "CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation." arXiv:2602.24286, February 27, 2026.
2. DoubleAI. "WarpSpeed Beats a Decade of Expert-Engineered GPU Kernels — Every Single One of Them." Business Wire, March 2, 2026.
3. DoubleAI. "Surpassing Expert-Written Kernels At Scale." doubleai.com/research, March 2, 2026.
4. Amnon Shashua and Shai Shalev-Shwartz. "Artificial Expert Intelligence through PAC-reasoning." arXiv:2412.02441, December 2024.
5. Srinivas Jaiganesh and Martin Burtscher. "A GPU Implementation of the ECL-CC Connected Components Algorithm." HPDC 2018.
6. Yiqian Liu, Max VanAusdal, and Martin Burtscher. "Analysis of Data Races in GPU Implementations of Graph Algorithms." IISWC 2024.
7. NVIDIA. "Hopper Architecture In-Depth." NVIDIA Technical Blog, 2022.
8. NVIDIA. "H100 Tensor Core GPU Architecture Whitepaper." 2022.
9. NVIDIA. "Ampere Architecture In-Depth." NVIDIA Technical Blog, 2020.
10. NVIDIA. "A100 Tensor Core GPU Architecture Whitepaper." 2020.
11. PyTorch Blog. "Deep Dive on the Hopper TMA Unit for FP8 GEMMs." 2024.
12. NVIDIA. "Hopper Tuning Guide." CUDA Documentation.
13. NVIDIA. "Hopper GPU Architecture Accelerates Dynamic Programming Up to 40x Using DPX Instructions." NVIDIA Blog, 2022.
14. ByteDance-Seed/cudaLLM. GitHub repository. Apache 2.0 license.
15. BytedTsinghua-SIA/CUDA-Agent. GitHub repository.
16. double-ai/doubleGraph. GitHub repository. Apache 2.0 license.
17. NVIDIA. "CUDA C++ Programming Guide." CUDA 12.1 Documentation.
18. Colfax Research. "CUTLASS Tutorial: GEMM with Thread Block Clusters." 2024.
19. arXiv:2501.12084. "Dissecting the NVIDIA Hopper Architecture through Microbenchmarking." 2025.
