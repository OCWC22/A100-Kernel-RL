# KernelForge Addendum: A100 Hardware Bible + Multi-Model Strategy + Full Compiler Stack
**Version: March 4, 2026 — Paste AFTER the main KERNELFORGE_HACKATHON_GUIDE.md**

---

## A. NVIDIA A100 Hardware Specifications (What You're Optimizing FOR)

doubleGraph ships A100-optimized kernels. CUDA Agent benchmarked on A100. Your environment profiles kernels running ON this chip. Every optimization decision flows from these numbers.

### A.1 A100 SXM4 80GB — Complete Specs

```
┌─────────────────────────────────────────────────────────────────┐
│  NVIDIA A100 SXM4 80GB (GA100, sm_80)                          │
├─────────────────────────────────────────────────────────────────┤
│  COMPUTE                                                        │
│  • Streaming Multiprocessors (SMs): 108                         │
│  • CUDA Cores: 6,912 (64 per SM)                                │
│  • Tensor Cores: 432 (4 per SM, 3rd gen)                        │
│  • FP32 throughput: 19.5 TFLOPS                                 │
│  • TF32 Tensor Core: 156 TFLOPS (312 with sparsity)             │
│  • FP16/BF16 Tensor Core: 312 TFLOPS (624 with sparsity)        │
│  • INT8 Tensor Core: 624 TOPS (1248 with sparsity)              │
│  • Clock: 1410 MHz (boost)                                      │
│                                                                  │
│  MEMORY                                                          │
│  • HBM2e: 80 GB                                                 │
│  • Memory bandwidth: 2,039 GB/s (2.0 TB/s)                      │
│  • Memory bus: 5120-bit                                          │
│  • ECC: On (reduces usable bandwidth ~6%)                        │
│                                                                  │
│  CACHE HIERARCHY                                                 │
│  • L1 cache / shared memory per SM: 192 KB (configurable)        │
│    - Shared memory configs: 0/8/16/32/64/100/132/164 KB          │
│    - Remaining = L1 data cache                                   │
│  • L2 cache: 40 MB (massive — key optimization target)           │
│  • Register file per SM: 256 KB (65,536 x 32-bit registers)      │
│                                                                  │
│  THREADING                                                       │
│  • Max threads per SM: 2,048                                     │
│  • Max threads per block: 1,024                                  │
│  • Warp size: 32 threads                                         │
│  • Max warps per SM: 64                                          │
│  • Max blocks per SM: 32                                         │
│  • Max registers per thread: 255                                 │
│  • Max registers per block: 65,536                               │
│                                                                  │
│  INTERCONNECT                                                    │
│  • NVLink: 600 GB/s (12 links × 50 GB/s)                        │
│  • PCIe: Gen4 x16 (31.5 GB/s each direction)                    │
│                                                                  │
│  ARCHITECTURE FEATURES                                           │
│  • Compute capability: 8.0 (sm_80)                               │
│  • Async copy (cp.async): global → shared memory, bypasses L1    │
│  • Tensor Memory Accelerator (TMA): NO (H100 only, sm_90)        │
│  • Thread Block Clusters: NO (H100 only)                         │
│  • wgmma instructions: NO (H100 only)                            │
│  • mma.sync instructions: YES (this is what A100 uses)           │
│  • Hardware-accelerated barriers: YES (mbarrier)                 │
│  • Async pipeline: cp.async + mbarrier (software pipelining)     │
└─────────────────────────────────────────────────────────────────┘
```

### A.2 The Roofline Model for A100

The **roofline model** tells you whether a kernel is memory-bound or compute-bound.

```
Performance (TFLOPS)
    │
156 ├─────────────────────────────────── TF32 Tensor Core ceiling
    │                              ╱
    │                           ╱
    │                        ╱    ← Memory-bound region
    │                     ╱       (most kernels live here)
    │                  ╱
    │               ╱
    │            ╱    Ridge point: ~76 FLOPs/byte (TF32)
    │         ╱                    ~10 FLOPs/byte (FP32)
    │      ╱
    │   ╱
    │╱
    └──────────────────────────────────── Arithmetic Intensity
                                          (FLOPs / byte accessed)
```

**Key numbers for A100:**
- **Ridge point (FP32)**: 19.5 TFLOPS / 2.0 TB/s = **~10 FLOPs/byte**
- **Ridge point (TF32 Tensor)**: 156 TFLOPS / 2.0 TB/s = **~76 FLOPs/byte**
- **Ridge point (FP16 Tensor)**: 312 TFLOPS / 2.0 TB/s = **~153 FLOPs/byte**

**What this means for kernel optimization:**
- GEMM at large sizes (M,N,K > 1024): compute-bound → optimize Tensor Core usage
- Elementwise ops (ReLU, add, scale): memory-bound → optimize memory access patterns
- Reductions (softmax, layernorm): memory-bound → optimize coalescing + shared memory
- Fused ops: goal is to increase arithmetic intensity by doing more compute per byte loaded

### A.3 A100 vs H100 vs B200 — What Changes for Kernel Code

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
| Kernel portability | ✓ baseline | ✓ runs sm_80 | ✓ runs sm_80 |

**Critical implication:** Kernels optimized for A100 (sm_80) run on H100 and B200 but leave performance on the table because they don't use TMA, wgmma, or thread block clusters. Kernels written for H100 (sm_90) do NOT run on A100. **Since doubleGraph targets A100, and we want maximum compatibility, sm_80 is correct.**

If you have an H100 for the hackathon, you still compile with `-arch=sm_80` for A100-compatible kernels, but you can ALSO compile `-arch=sm_90` variants. Your environment should support both and benchmark both.

---

## B. The Full Compiler & Runtime Stack

### B.1 How PyTorch Code Becomes GPU Instructions

```
Python (torch.matmul)
    │
    ▼
TorchDynamo (captures computation graph)
    │
    ▼
FX Graph (intermediate representation)
    │
    ▼
TorchInductor (backend compiler)
    │
    ├──► Triton Kernels (default path)
    │       │
    │       ▼
    │    Triton IR → Triton GPU IR → LLVM IR → PTX → SASS
    │
    ├──► CUDA C++ Kernels (optional, via custom ops)
    │       │
    │       ▼
    │    nvcc → PTX → SASS
    │
    └──► cuBLAS/cuDNN calls (for known patterns like GEMM, conv)
             │
             ▼
          Pre-compiled SASS (vendor-optimized)
```

### B.2 Each Layer Explained

**Triton** (https://github.com/triton-lang/triton)
- OpenAI's GPU programming language, higher-level than CUDA
- Automatically handles tiling, memory coalescing, shared memory
- Default backend for torch.compile (TorchInductor generates Triton)
- Kernels written in Python-like syntax, compiled to PTX/SASS
- **For our environment:** Some teams write Triton instead of CUDA. Our environment could accept both. But CUDA gives more control for A100-specific optimizations.

**MLIR** (Multi-Level Intermediate Representation)
- Compiler infrastructure from LLVM project, used internally by Triton
- Triton IR → MLIR dialects → LLVM IR → PTX
- Not directly user-facing for kernel writing, but relevant because:
  - torch.compile's optimization passes happen at MLIR level
  - Understanding MLIR helps understand what torch.compile can/can't optimize

**PTX** (Parallel Thread Execution)
- NVIDIA's virtual ISA (instruction set architecture)
- Human-readable assembly-like language
- nvcc compiles CUDA → PTX, then ptxas compiles PTX → SASS
- **Optimization opportunity:** WarpSpeed injects inline PTX for fine-grained control. Our environment could allow/score this.

**SASS** (Shader ASSembly)
- The actual GPU machine code that runs on the silicon
- Architecture-specific (sm_80 SASS ≠ sm_90 SASS)
- WarpSpeed even does SASS-level optimization (not needed for hackathon, but impressive for demo)

**nvcc flags that matter for A100:**
```bash
nvcc -O3 \                     # Maximum optimization
     -arch=sm_80 \             # A100 compute capability
     -lineinfo \               # Source line mapping for profiling
     -std=c++17 \              # Modern C++
     --use_fast_math \         # Faster but less precise math
     --ptxas-options=-v \      # Show register/shared memory usage
     -Xptxas -O3 \             # PTX assembler optimization
     -lcublas -lcudnn \        # Link vendor libraries
     kernel.cu -o kernel.out
```

### B.3 Inference Runtimes (for serving the model that writes kernels)

**vLLM** (https://github.com/vllm-project/vllm)
- High-throughput LLM serving with PagedAttention
- Supports Qwen3.5, GLM-5, all models we care about
- **Use for:** Serving the model during GRPO training (model generates CUDA code → environment evaluates)
- Key command: `vllm serve Qwen/Qwen3.5-9B --gpu-memory-utilization 0.5` (leave room for CUDA sandbox)

**SGLang** (https://github.com/sgl-project/sglang)
- Faster than vLLM for many workloads, supports structured generation
- Better for Qwen3.5 specifically (recommended by Qwen team)
- **Use for:** If vLLM OOMs, SGLang's memory management may work better

**TensorRT-LLM** (https://github.com/NVIDIA/TensorRT-LLM)
- NVIDIA's own optimized inference, fastest raw throughput
- More complex setup, less flexible
- **Use for:** If you need maximum tokens/sec during GRPO rollouts

---

## C. Multi-Model Strategy — What Actually Fits on A100 80GB

### C.1 Reality Check: What CAN'T Run on a Single A100

**GLM-5 (744B total / 40B active):**
- FP8: ~744 GB → needs 8× H200. **IMPOSSIBLE on single A100.**
- Even with 4-bit quantization: ~370 GB. **IMPOSSIBLE.**
- **Use GLM-5 via API only** ($1.00/M input tokens via Z.ai). Use it as the SkyDiscover LLM backend — it writes kernel mutations, you don't train it.

**Qwen3.5-397B-A17B:**
- FP8: ~397 GB → needs 5-8 GPUs. **IMPOSSIBLE on single A100.**
- **API only** (via Alibaba Cloud Model Studio).

**Qwen3.5-27B (dense):**
- FP16: ~54 GB. Fits for inference, but no room for GRPO training + CUDA sandbox.
- 4-bit QLoRA: ~14 GB weights + ~10 GB optimizer = ~24 GB. Leaves ~56 GB for sandbox. **TIGHT but possible.**

### C.2 What DOES Fit — The Multi-Pronged Approach

Here's your three-model strategy, profiled for A100 80GB:

```
┌─────────────────────────────────────────────────────────────────────┐
│  PRONG 1: Qwen3.5-35B-A3B (MoE — BEST BANG FOR BUCK)              │
│                                                                     │
│  Total params: 35B, Active per token: 3B                            │
│  Architecture: Gated DeltaNet + Sparse MoE, 262K context           │
│  License: Apache 2.0                                                │
│  HuggingFace: Qwen/Qwen3.5-35B-A3B                                │
│                                                                     │
│  VRAM PROFILE (4-bit QLoRA + Unsloth):                             │
│  • Model weights (4-bit): ~17.5 GB                                 │
│  • LoRA adapters (r=16): ~0.5 GB                                   │
│  • Optimizer states: ~2 GB                                          │
│  • KV cache (8K context): ~4 GB                                    │
│  • Activations + gradients: ~8 GB                                   │
│  ─────────────────────────────────                                  │
│  • TOTAL: ~32 GB                                                    │
│  • REMAINING FOR CUDA SANDBOX: ~48 GB                               │
│                                                                     │
│  WHY THIS MODEL:                                                    │
│  • Only 3B active params → fast inference (more GRPO steps/hour)   │
│  • Full Qwen3.5 quality (beats Qwen3-235B-A22B on benchmarks!)     │
│  • Native thinking mode (<think>...</think>) for reasoning          │
│  • Hybrid DeltaNet attention = efficient long-context               │
│  • Unsloth + TRL explicitly support GRPO for this model             │
│                                                                     │
│  ESTIMATED GRPO THROUGHPUT (A100):                                  │
│  • ~4 kernel generations/min (with 8K context, 4 candidates)        │
│  • ~15-20 GRPO steps/hour                                           │
│  • 24h → ~360-480 steps (1,440-1,920 kernel samples)               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  PRONG 2: Qwen3.5-9B (Dense — HIGHEST QUALITY SMALL MODEL)         │
│                                                                     │
│  Total params: 9B (dense)                                           │
│  Architecture: Gated DeltaNet + MoE hybrid, 262K context           │
│  License: Apache 2.0                                                │
│  HuggingFace: Qwen/Qwen3.5-9B                                     │
│                                                                     │
│  VRAM PROFILE (4-bit QLoRA + Unsloth):                             │
│  • Model weights (4-bit): ~5 GB                                    │
│  • LoRA + optimizer: ~3 GB                                         │
│  • KV cache + activations: ~6 GB                                   │
│  ─────────────────────────────────                                  │
│  • TOTAL: ~14 GB                                                    │
│  • REMAINING FOR CUDA SANDBOX: ~66 GB                               │
│                                                                     │
│  WHY THIS MODEL:                                                    │
│  • Beats Qwen3-30B and GPT-OSS-120B on GPQA Diamond                │
│  • 81.7 GPQA Diamond — strongest reasoning per parameter            │
│  • Most headroom for CUDA compilation + profiling                   │
│  • Can run MORE GRPO steps (faster inference = more training)       │
│  • Perfect for "show scaling behavior" comparison vs 35B-A3B       │
│                                                                     │
│  ESTIMATED GRPO THROUGHPUT (A100):                                  │
│  • ~8-10 kernel generations/min                                     │
│  • ~30-40 GRPO steps/hour                                           │
│  • 24h → ~720-960 steps (2,880-3,840 kernel samples)               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  PRONG 3: GLM-5 or Qwen3.5-397B (API — SKYDISCOVER BACKEND)        │
│                                                                     │
│  NOT trained — used as the LLM that MUTATES kernels                │
│  in SkyDiscover's evolutionary search.                              │
│                                                                     │
│  GLM-5 API: $1.00/M input, $3.20/M output (Z.ai)                  │
│  Qwen3.5-Plus API: Alibaba Cloud Model Studio                     │
│  Claude Opus 4.6: $5/$25 (Anthropic)                               │
│  GPT-5.2: OpenAI                                                   │
│                                                                     │
│  WHY API:                                                           │
│  • 744B params can't fit on any single GPU                         │
│  • Frontier models produce better kernel mutations                  │
│  • SkyDiscover only calls the LLM ~80-120 times per run            │
│  • Total cost: ~$2-8 per evolution run                             │
│  • GLM-5 is MIT licensed, strongest open model for coding           │
│  • 77.8% SWE-bench → excellent code manipulation ability            │
│                                                                     │
│  SKYDISCOVER COMMAND:                                               │
│  uv run skydiscover-run kernel.cu evaluator.py \                   │
│    --search evox \                                                  │
│    --model glm-5 \         # or claude-opus-4-6, gpt-5-2           │
│    --iterations 100                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### C.3 Why NOT Qwen3-8B (old) or cudaLLM-8B?

- **Qwen3-8B** is last-gen. Qwen3.5-9B is strictly better on every benchmark (12+ points on MMLU, 15+ on HumanEval). Same memory footprint. No reason to use the old one.
- **cudaLLM-8B** (ByteDance) is a Qwen3-8B fine-tuned specifically for CUDA. It's useful as a **SFT warmup checkpoint** — start from cudaLLM-8B's CUDA knowledge, then GRPO train with Qwen3.5-9B architecture. But for the base model, Qwen3.5-9B is better.
- **Qwen3-Next-80B-A3B** ("QuarterNext") is an older ultra-sparse MoE from Sept 2025. Superseded by Qwen3.5-35B-A3B which has the same sparsity ratio but better architecture (Gated DeltaNet).

### C.4 The Actual Multi-Model Hackathon Plan

```
PARALLEL TRACK 1 (Background — overnight)
├─ SFT warmup: Qwen3.5-9B on cudaLLM training data (2-3 hours)
├─ GRPO training: Qwen3.5-9B through KernelForge env (20+ hours)
└─ Deliverable: "9B model that learns to write A100 kernels"

PARALLEL TRACK 2 (Background — overnight)  
├─ SFT warmup: Qwen3.5-35B-A3B on cudaLLM training data (3-4 hours)
├─ GRPO training: Qwen3.5-35B-A3B through KernelForge env (16+ hours)
└─ Deliverable: "35B MoE model, compare to 9B — same active params?"

PARALLEL TRACK 3 (Fast — 1-2 hours, run first)
├─ SkyDiscover EvoX with GLM-5 API as mutator
├─ Evolve 5-10 A100-specific kernels
└─ Deliverable: "Immediate 1.5-2x wins, live dashboard demo"

NOTE: Tracks 1 and 2 can't run simultaneously on 1 GPU.
Run Track 3 first (uses API, minimal local GPU).
Then run Track 1 or 2 overnight.
Run the other the next night if you have 48h+.
```

---

## D. A100-Specific Kernel Optimization Targets

These are the exact optimizations your environment should reward and your kernels should exploit.

### D.1 Memory System Optimizations

**Global memory coalescing (most impactful, easiest to detect):**
```
GOOD: Thread i reads address base + i × sizeof(float)   → 1 transaction
BAD:  Thread i reads address base + i × stride           → up to 32 transactions

Performance impact: up to 7x throughput difference
Detection: Nsight Compute "Global Memory Access Pattern" metric
```

**Shared memory usage (second most impactful):**
```
A100 shared memory per SM: up to 164 KB (configurable)
Optimal tile sizes for GEMM on A100:
  - BM=128, BN=128, BK=32 (for FP32)
  - BM=256, BN=128, BK=64 (for FP16/BF16 with Tensor Cores)

Bank conflicts: A100 has 32 banks, each 4 bytes wide
  - Stride-1 access: 0 bank conflicts (ideal)
  - Stride-32 access: 32-way bank conflict (32x slower)
  - Use padding: shared_mem[TILE_SIZE][TILE_SIZE + 1] to avoid
```

**L2 cache residency (A100-specific optimization — 40 MB L2!):**
```
A100's 40 MB L2 is HUGE. WarpSpeed exploited this for WCC:
  - Pin parent array in L2 → eliminates global memory round-trips
  - cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, bytes)
  - Use cudaAccessPolicyWindow to set cache residency policy

This is an A100/H100 feature that most kernels don't exploit.
An RL agent that discovers L2 pinning would be impressive.
```

**Async copy (cp.async — A100's killer feature vs Volta/Turing):**
```
// Old way (V100): load through registers
__shared__ float smem[TILE_SIZE];
smem[threadIdx.x] = global_data[idx];  // goes through L1 + registers

// A100 way: async copy, bypasses L1 and registers
__pipeline_memcpy_async(&smem[threadIdx.x], &global_data[idx], sizeof(float));
__pipeline_commit();
__pipeline_wait_prior(0);

Impact: 20-30% improvement on memory-bound kernels
This is THE optimization that separates A100-aware code from generic CUDA.
```

### D.2 Compute Optimizations

**Tensor Core usage (mma.sync on A100):**
```
A100 Tensor Cores do 16x8x16 (FP16) or 16x8x8 (TF32) matrix multiply-accumulate.
Using Tensor Cores via:
  - wmma API: __mma_sync (easiest, ~80% of peak)
  - PTX mma.sync: (harder, ~95% of peak)
  - cuBLAS: (automatic, ~99% of peak for standard shapes)

RL agent discovering wmma usage = huge speedup on compute-bound kernels.
```

**Vectorized loads:**
```
float4 data = *reinterpret_cast<float4*>(&global[idx]);
// Loads 16 bytes in one transaction instead of 4 × 4-byte loads
// 2-4x improvement on memory-bound kernels
```

**Warp-level primitives:**
```
__shfl_sync()     — exchange data between threads without shared memory
__ballot_sync()   — vote across warp (32 threads)
__reduce_add_sync() — warp-level reduction

These eliminate shared memory traffic for intra-warp operations.
```

### D.3 Launch Configuration for A100

```
// Optimal block sizes for A100 (108 SMs):
// Goal: maximize occupancy while leaving enough registers

// Memory-bound kernels: maximize threads for latency hiding
dim3 block(256);     // 256 threads = 8 warps
dim3 grid((N + 255) / 256);

// Compute-bound kernels: balance registers vs occupancy
dim3 block(128);     // 128 threads = 4 warps, more registers available
// Max occupancy at 128 threads: 2048/128 = 16 blocks/SM possible
// But register pressure typically limits to 4-8 blocks/SM

// 2D blocks for matrix operations:
dim3 block(32, 8);   // 256 threads, row-major for coalescing
dim3 block(16, 16);  // 256 threads, square for matrix tiles
```

---

## E. Updated Code Changes for A100 Targeting

### E.1 Updated Compiler (target sm_80)

In `server/compiler.py`, update the default:
```python
class CUDACompiler:
    def __init__(self, arch: str = "sm_80", timeout: int = 45):
        # sm_80 = A100 (our primary target, doubleGraph's target)
        # Also forward-compatible with H100 (sm_90) and B200 (sm_100)
```

### E.2 Updated Profiler (A100-aware metrics)

Add these Nsight Compute metrics to the profiler's observation:
```python
# A100-specific metrics to include in observation
NCU_METRICS = {
    # Memory system
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "memory_throughput_pct",
    "l2_cache_hit_rate": "l2_hit_rate",  # A100's 40MB L2 is key
    "lts__t_sector_hit_rate.pct": "l2_sector_hit_rate",
    
    # Compute
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed": "tensor_core_usage",
    
    # Occupancy
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed": "achieved_occupancy",
    
    # Efficiency
    "smsp__warp_issue_stalled_long_scoreboard_pct": "memory_stall_pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_pct": "compute_stall_pct",
}
```

### E.3 Updated GRPO Training Script (Qwen3.5-35B-A3B)

```python
from unsloth import FastLanguageModel

# PRONG 1: Qwen3.5-35B-A3B (MoE, 3B active)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-35B-A3B",
    max_seq_length=8192,
    dtype=None,           # Auto BF16
    load_in_4bit=True,    # ~17.5 GB
)

# OR PRONG 2: Qwen3.5-9B (dense)
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="Qwen/Qwen3.5-9B",
#     max_seq_length=8192,
#     dtype=None,
#     load_in_4bit=True,  # ~5 GB
# )

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

### E.4 SkyDiscover with GLM-5 API

```bash
# Set GLM-5 API key (Z.ai platform)
export OPENAI_API_KEY=your-zai-api-key
export OPENAI_API_BASE=https://api.z.ai/v1  # OpenAI-compatible endpoint

# Run SkyDiscover with GLM-5 as the mutator
uv run skydiscover-run \
    initial_a100_gemm.cu \
    skydiscover_evaluator.py \
    --search evox \
    --model glm-5 \           # GLM-5 via API
    --iterations 100 \
    --output a100_kernels_v1

# Alternative: use Claude for mutation, GLM-5 for meta-strategy
# (SkyDiscover supports multi-model configs)
```

---

## F. Qwen3.5 Model Family — Complete Reference

Released Feb 16 - Mar 2, 2026. All Apache 2.0.

| Model | Params | Active | Architecture | Context | GPQA Diamond | Fits A100 80GB (4-bit)? |
|-------|--------|--------|-------------|---------|-------------|----------------------|
| Qwen3.5-397B-A17B | 397B | 17B | MoE + DeltaNet | 262K | 71.2 | NO (API only) |
| Qwen3.5-35B-A3B | 35B | 3B | MoE + DeltaNet | 262K | ~68 | YES (~32 GB with QLoRA) |
| Qwen3.5-27B | 28B | 28B | Dense + DeltaNet | 262K | ~65 | TIGHT (~35 GB) |
| **Qwen3.5-9B** | 9B | ~9B | Hybrid DeltaNet+MoE | 262K | **81.7** | **YES (~14 GB)** |
| Qwen3.5-4B | 4B | ~4B | Hybrid DeltaNet+MoE | 262K | ~55 | YES (~8 GB) |
| Qwen3.5-2B | 2B | ~2B | Hybrid DeltaNet+MoE | 262K | ~45 | YES (~5 GB) |
| Qwen3.5-0.8B | 0.8B | ~0.8B | Hybrid DeltaNet+MoE | 262K | ~35 | YES (~3 GB) |

**Recommendation:** Qwen3.5-9B for GRPO training (best quality/size ratio, 81.7 GPQA Diamond beats GPT-OSS-120B). Qwen3.5-35B-A3B as stretch goal (same active params as 3B but full MoE routing intelligence).

---

## G. GLM-5 — Reference for API Usage

| Spec | Value |
|------|-------|
| Total params | 744B |
| Active params | 40B |
| Architecture | MoE + DeepSeek Sparse Attention |
| Context | 200K tokens |
| License | MIT |
| SWE-bench | 77.8% |
| API pricing | $1.00/M input, $3.20/M output |
| API endpoint | https://api.z.ai/v1 (OpenAI-compatible) |
| HuggingFace | zai-org/GLM-5 |
| Local deploy | Needs 8× H200 (FP8). NOT for single GPU. |

**For hackathon:** Use GLM-5 API as SkyDiscover's LLM backend. It's 5-8x cheaper than Claude/GPT-5 and strongest open model for code tasks. Cost for 100 SkyDiscover iterations: ~$1-3.

---

## H. What We're Actually Validating at the Hackathon

To be crystal clear — here's what the three pieces prove when combined:

```
THESIS: You don't need 128 GPUs and a trillion-parameter model
to generate expert-level A100 CUDA kernels. You need:

1. EFFICIENT SEARCH (SkyDiscover EvoX)
   - AdaEvolve's G-signal adapts explore/exploit automatically
   - EvoX meta-evolves the search strategy itself
   - Frontier LLM (GLM-5 API) as semantic mutator
   - Result: Comparable kernel quality in 80-120 LLM calls
     vs CUDA Agent's thousands of rollouts on 128 GPUs

2. EFFICIENT TRAINING (Unsloth GRPO on A100)
   - QLoRA reduces model memory 4x
   - GRPO eliminates critic model (vs PPO)
   - Qwen3.5-9B/35B-A3B: frontier quality at trainable sizes
   - Result: Internalize optimization patterns in 500 steps
     vs CUDA Agent's 150 steps × 1024 batch on massive compute

3. EXPERT BASELINES (doubleGraph A100 kernels)
   - Already-optimized A100 graph kernels as reward calibration
   - "How close can our agent get to the AI that beat NVIDIA?"
   - Eliminates the verification problem (known-good baselines)
   - Result: Calibrated, trustworthy reward signal

COMBINED: Single A100, 24-48 hours → kernels that beat torch.compile
on A100 by 1.5-2x+ for specific operators. Demonstrate that
the OpenEnv environment enables BOTH evolutionary AND RL approaches
to reach this performance level.
```

---

## I. Updated Links (New Models)

| Resource | URL |
|----------|-----|
| Qwen3.5-35B-A3B | https://huggingface.co/Qwen/Qwen3.5-35B-A3B |
| Qwen3.5-9B | https://huggingface.co/Qwen/Qwen3.5-9B |
| Qwen3.5-4B | https://huggingface.co/Qwen/Qwen3.5-4B |
| Qwen3.5 GitHub | https://github.com/QwenLM/Qwen3.5 |
| GLM-5 | https://huggingface.co/zai-org/GLM-5 |
| GLM-5 FP8 | https://huggingface.co/zai-org/GLM-5-FP8 |
| GLM-5 GGUF (Unsloth) | https://huggingface.co/unsloth/GLM-5-GGUF |
| GLM-5 GitHub | https://github.com/zai-org/GLM-5 |
| GLM-5 API | https://api.z.ai |
| cudaLLM-8B (SFT warmup) | https://huggingface.co/ByteDance-Seed/cudaLLM-8B |
| Qwen3.5 Blog | https://qwen.ai/blog?id=qwen3.5 |
| Unsloth Qwen3.5 support | https://github.com/unslothai/unsloth |
