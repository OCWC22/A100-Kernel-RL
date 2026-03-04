# KernelForge Implementation Specification

## Four Objectives, One System

**Hackathon:** OpenEnv Hackathon SF — March 7-8, 2026 (SHACK15, San Francisco)
**Date:** March 3, 2026

This document covers four objectives. Each section is self-contained: it states the problem, explains the design decisions (with evidence), provides implementation code, and lists exactly what another engineer needs to reproduce it.

---

## Objective 1: Build an RL Environment for the OpenEnv Hackathon

### 1.1 What the Environment Does

The environment accepts CUDA source code as an action, compiles it, verifies correctness, benchmarks it, and returns a reward. It merges two systems:

From **ByteDance CUDA Agent** (arXiv:2602.24286):
- The compile → verify → profile evaluation loop
- Discrete milestone rewards ({-1, 1, 2, 3}) validated by ablation to outperform continuous speedup by 36 percentage points
- Anti-reward-hacking measures (protected scripts, randomized inputs, forbidden fallbacks)
- SKILL.md protocol that encodes optimization knowledge as agent context

From **DoubleAI WarpSpeed / DoubleGraph**:
- GPU-specific kernel specialization (same algorithm, different implementation per architecture)
- CachePool for GPU resource reuse across evaluations (thread-local LRU, 8 entries)
- 4-way dispatch pattern (base / segmented / masked / seg+mask) repurposed as curriculum levels
- Per-kernel `.cu.flags` as a learnable compilation parameter dimension
- A100-specific technique library (direction-optimizing BFS, L2 pinning, warp primitives, 3-tier dispatch)

### 1.2 OpenEnv API Conformance

OpenEnv (github.com/meta-pytorch/OpenEnv, Meta-PyTorch, BSD-3-Clause) provides Gymnasium-style APIs. Environments are packaged as Docker containers and exposed via HTTP. The core contract:

```python
from openenv.core.env_server import Environment, create_fastapi_app
from openenv.core.types import StepResult

class KernelForgeEnv(Environment):
    def reset(self) -> dict:
        """Return initial observation (SKILL.md + problem description)."""

    def step(self, action: str) -> StepResult:
        """Accept CUDA code, return (observation, reward, done, info)."""

    @property
    def state(self) -> dict:
        """Return serializable environment state."""

app = create_fastapi_app(KernelForgeEnv)
```

TRL connects via `environment_factory` (OpenEnv's own examples directory includes `grpo_blackjack/` demonstrating TRL+OpenEnv integration).

### 1.3 Environment Implementation

```python
"""
kernelforge_env.py — OpenEnv environment for CUDA kernel optimization.

Merges CUDA Agent evaluation loop with DoubleGraph architectural patterns.
Cross-compiles for A100 (sm_80) on any training GPU.

Install:
    pip install "openenv-core[core]>=0.2.1" cupy-cuda12x numpy
"""
import subprocess, tempfile, os, ctypes, hashlib, time, json
import numpy as np
from openenv.core.env_server import Environment, create_fastapi_app
from openenv.core.types import StepResult
from typing import Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Component 1: GPU CachePool (from DoubleGraph cache_pool.hpp)
#
# DoubleGraph keeps GPU resources resident across evaluations via a
# thread-local LRU cache. We adapt this for Python: test data stays
# on GPU between evaluations instead of malloc/free per call.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GPUCachePool:
    """
    LRU cache for GPU-resident evaluation data.

    Why: CUDA Agent runs hundreds of evaluations per GRPO step.
    Without caching, each evaluation mallocs/frees test tensors.
    DoubleGraph's CachePool pattern (cache_pool.hpp, 125 lines)
    eliminates this overhead with thread-local LRU eviction.

    Usage:
        pool = GPUCachePool(max_entries=8)
        data = pool.get_or_create("problem_42", lambda: upload_to_gpu(...))
    """
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
            # Free GPU memory for evicted entry
            if hasattr(old, '_gpu_arrays'):
                del old._gpu_arrays
        val = factory()
        self._cache[key] = val
        self._order.append(key)
        return val

    def clear(self):
        self._cache.clear()
        self._order.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Component 2: GPU Architecture Registry (from DoubleGraph target_gpu.hpp)
#
# DoubleGraph uses compile-time GPU selection via target_gpu_map.json
# and #define AAI_GPU_A100 / AAI_IS_A100 macros. We make this runtime-
# configurable so the same environment supports multiple targets.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPU_REGISTRY = {
    "a100": {
        "arch": "sm_80",
        "l2_cache_mb": 40,
        "smem_per_sm_kb": 164,
        "sm_count": 108,
        "hbm_bandwidth_gbps": 2039,
        "registers_per_sm": 65536,
        "nvcc_flag": "-arch=sm_80",
        "features": [
            "cp.async", "l2_persistence", "bf16_tensor_core",
            "tf32_tensor_core", "structured_sparsity_2_4",
        ],
    },
    "h100": {
        "arch": "sm_90a",
        "l2_cache_mb": 50,
        "smem_per_sm_kb": 228,
        "sm_count": 132,
        "hbm_bandwidth_gbps": 3352,
        "registers_per_sm": 65536,
        "nvcc_flag": "-arch=sm_90a",
        "features": [
            "tma", "dsmem", "dpx", "thread_block_clusters",
            "warp_specialization", "fp8", "cp.async",
            "l2_persistence", "bf16_tensor_core", "tf32_tensor_core",
        ],
    },
    "b200": {
        "arch": "sm_100a",
        "l2_cache_mb": 96,        # estimated
        "smem_per_sm_kb": 228,    # estimated, may increase
        "sm_count": 192,          # estimated
        "hbm_bandwidth_gbps": 8000,  # HBM3e
        "registers_per_sm": 65536,
        "nvcc_flag": "-arch=sm_100a",
        "features": [
            "tma", "dsmem", "dpx", "thread_block_clusters",
            "nvlink_5", "fp4", "fp8", "cp.async",
            "l2_persistence",
        ],
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Component 3: Anti-Reward-Hacking (from CUDA Agent Section 3.2)
#
# CUDA Agent documented specific exploits and countermeasures.
# We implement all of them.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Whitelist of safe per-kernel compilation flags
# (from DoubleGraph .cu.flags sidecar pattern)
ALLOWED_CU_FLAGS = {
    "--use_fast_math",
    "--extra-device-vectorization",
    "--rdc=true",
}
ALLOWED_CU_FLAG_PREFIXES = (
    "--maxrregcount=",
)

# Symbols forbidden in generated kernels (anti-hacking)
FORBIDDEN_SYMBOLS = [
    "torch", "at::Tensor", "c10::", "torch::autograd",
    "triton", "torch.compile", "torch.nn.functional",
]


def extract_cu_flags(cuda_code: str) -> list[str]:
    """
    Parse per-kernel .cu.flags from a comment in the CUDA source.
    DoubleGraph uses .cu.flags sidecar files; we inline them.

    Format: // CU_FLAGS: --use_fast_math --maxrregcount=48

    Only whitelisted flags are accepted (anti-hacking).
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
                    # Validate numeric value
                    try:
                        val = int(tok.split('=')[1])
                        if 16 <= val <= 128:
                            flags.append(tok)
                    except (ValueError, IndexError):
                        pass
    return flags


def scan_forbidden_symbols(so_path: str) -> Optional[str]:
    """Check compiled .so for forbidden library calls."""
    try:
        result = subprocess.run(
            ["nm", "-D", so_path],
            capture_output=True, text=True, timeout=5,
        )
        symbols = result.stdout
        for forbidden in FORBIDDEN_SYMBOLS:
            if forbidden in symbols:
                return f"Forbidden symbol detected: {forbidden}"
    except Exception:
        pass
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Component 4: Reward Function (CUDA Agent Equation 1, page 5)
#
# Discrete milestone rewards validated by ablation (Table 2):
#   - Continuous speedup reward: 60.4% faster-than-compile
#   - Discrete milestone reward: 96.8% faster-than-compile
#   - Delta: +36.4 percentage points
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_reward(
    compiled: bool,
    correct: bool,
    speedup_vs_eager: float,
    speedup_vs_compile: float,
) -> float:
    """
    CUDA Agent's 4-level reward (Equation 1):
      r = -1 if correctness fails
      r = +1 if correct but no speedup
      r = +2 if >5% faster than eager
      r = +3 if >5% faster than both eager AND compile

    Why discrete: GPU execution times have noise (thermal throttling,
    OS interrupts). Continuous speedup rewards create outlier advantage
    estimates that bias policy toward easy kernels. CUDA Agent ablation
    proved this definitively.
    """
    if not compiled or not correct:
        return -1.0
    if speedup_vs_compile > 1.05:
        return 3.0   # Beats torch.compile (strong optimization)
    if speedup_vs_eager > 1.05:
        return 2.0   # Beats eager (basic optimization)
    return 1.0       # Correct but not faster


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Component 5: The Environment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class KernelForgeEnv(Environment):
    """
    RL environment for CUDA kernel optimization.

    Evaluation pipeline per step:
      1. Parse .cu.flags from agent's code
      2. Compile with nvcc (cross-compile for target arch)
      3. Scan .so for forbidden symbols
      4. Run on 5 randomized inputs, check correctness
      5. Benchmark (50 warmup + 30 timed runs, median)
      6. Compute discrete milestone reward

    The target GPU architecture is configurable.
    The training GPU can be different from the target GPU.
    """

    def __init__(self, target_gpu: str = "a100"):
        if target_gpu not in GPU_REGISTRY:
            raise ValueError(
                f"Unknown GPU: {target_gpu}. "
                f"Available: {list(GPU_REGISTRY.keys())}"
            )
        self.target_gpu = target_gpu
        self.gpu_spec = GPU_REGISTRY[target_gpu]
        self.cache_pool = GPUCachePool(max_entries=8)

        # Evaluation settings
        self.verify_inputs = 5          # Randomized correctness checks
        self.warmup_iters = 50
        self.benchmark_runs = 30
        self.max_compile_seconds = 30

        # Per-episode state
        self.current_problem = None
        self.baseline_eager_ms = None
        self.baseline_compile_ms = None
        self.skill_md = None

    def reset(self, problem: dict = None) -> dict:
        """
        Initialize with a problem definition.

        Problem format (matches CUDA-Agent-Ops-6K):
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

        # Baselines can be pre-computed or measured on first reset
        self.baseline_eager_ms = self.current_problem.get("eager_time_ms")
        self.baseline_compile_ms = self.current_problem.get("compile_time_ms")

        # Build SKILL.md with target GPU specs
        self.skill_md = self._build_skill_md()

        return {
            "observation": self.skill_md + "\n\n---\n\n" + self._problem_prompt(),
            "target_gpu": self.target_gpu,
            "arch": self.gpu_spec["arch"],
            "baseline_eager_ms": self.baseline_eager_ms,
            "baseline_compile_ms": self.baseline_compile_ms,
        }

    def step(self, action: str) -> StepResult:
        """
        Evaluate a CUDA kernel submission.

        Action: raw CUDA source code string.
        Returns: StepResult with reward, observation, done, info.
        """
        import cupy as cp

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "kernel.cu")
            lib_path = os.path.join(tmpdir, "kernel.so")

            # ── Parse per-kernel flags ──
            cu_flags = extract_cu_flags(action)

            with open(src_path, "w") as f:
                f.write(action)

            # ── Step 1: Compile ──
            nvcc_cmd = [
                "nvcc",
                self.gpu_spec["nvcc_flag"],
                "-O3", "--shared", "-Xcompiler", "-fPIC",
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

            # ── Step 2: Load + symbol scan ──
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

            # ── Step 3: Correctness on 5 randomized inputs ──
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

                if not self._outputs_match(kernel_out, ref_out, atol=1e-3, rtol=1e-3):
                    max_diff = float(np.max(np.abs(kernel_out - ref_out)))
                    return StepResult(
                        observation=f"WRONG OUTPUT (input {i}): max_diff={max_diff:.6f}",
                        reward=-1.0, done=False,
                        info={"stage": "correctness", "max_diff": max_diff},
                    )

            # ── Step 4: Benchmark ──
            inputs = self._generate_inputs(seed=42)

            # Warmup
            for _ in range(self.warmup_iters):
                self._run_kernel(so, entry, inputs)
            cp.cuda.Device(0).synchronize()

            # Timed runs
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

            # ── Step 5: Reward ──
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
                f"eager={self.baseline_eager_ms:.3f}ms "
                f"({speedup_eager:.2f}x) | "
                f"compile={self.baseline_compile_ms:.3f}ms "
                f"({speedup_compile:.2f}x) | "
                f"reward={reward} | flags={cu_flags}"
            )

            return StepResult(
                observation=obs,
                reward=reward,
                done=(reward >= 3.0),  # Episode ends on max reward
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
            "target_gpu": self.target_gpu,
            "problem": self.current_problem.get("name") if self.current_problem else None,
        }

    # ── SKILL.md Builder ──

    def _build_skill_md(self) -> str:
        """
        Build architecture-aware SKILL.md.

        This encodes DoubleGraph's per-GPU optimization techniques
        as agent context. The content changes per target GPU.
        """
        spec = self.gpu_spec
        gpu = self.target_gpu.upper()

        skill = f"""# SKILL.md — {gpu}-Optimized CUDA Kernel Generation

## Target Hardware
- GPU: NVIDIA {gpu} ({spec['arch']})
- SMs: {spec['sm_count']} | L2 Cache: {spec['l2_cache_mb']} MB
- Shared Memory/SM: {spec['smem_per_sm_kb']} KB
- HBM Bandwidth: {spec['hbm_bandwidth_gbps']} GB/s
- Registers/SM: {spec['registers_per_sm']}
- nvcc: {spec['nvcc_flag']} -O3

## Per-Kernel Compilation Flags
Specify in a comment: // CU_FLAGS: --use_fast_math --maxrregcount=48
Allowed: --use_fast_math, --maxrregcount=N (16-128),
         --rdc=true, --extra-device-vectorization

## Optimization Priorities (ranked by typical impact)

### Priority 1: Algorithmic Reduction (>50% impact)
- Algebraic simplification: reduce mathematical complexity before optimizing.
  Example: diag(A) @ B = row-wise scaling. O(N^2*M) -> O(N*M).
- Kernel fusion: merge sequential ops into one kernel.
  Eliminates intermediate tensor materialization and kernel launch overhead.
- Operator rearrangement: restructure computation order.
  Example: x @ (sum_j w_j^T) / 2 instead of sum_j (x @ w_j^T / 2).

### Priority 2: Memory Hierarchy (20-50% impact)"""

        # Architecture-specific memory techniques
        if self.target_gpu == "a100":
            skill += """
- L2 cache pinning: {gpu} has 40MB L2. Pin frequently accessed arrays:
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 30*1024*1024);
    // Then set stream attribute with cudaAccessPropertyPersisting
- Vectorized loads: float4 for 16-byte aligned access (4x transaction efficiency)
- Shared memory tiling: pad arrays float tile[32][33] to avoid bank conflicts
- Memory coalescing: consecutive threads access consecutive addresses
- __ldg() for read-only data via texture cache path"""

        elif self.target_gpu == "h100":
            skill += """
- TMA (Tensor Memory Accelerator): single thread initiates bulk async copy,
  frees other 127 threads for computation. 1.45 TB/s throughput.
- Distributed shared memory: SMs in a cluster access each other's SMEM
  without going through global memory.
- L2 cache pinning: {gpu} has 50MB L2 (larger than A100's 40MB).
- Thread Block Clusters: __cluster_dims__ for cross-block shared memory."""

        elif self.target_gpu == "b200":
            skill += """
- HBM3e: ~8 TB/s bandwidth. Memory-bound kernels benefit enormously.
- NVLink 5.0: if multi-GPU, 1.8 TB/s bidirectional.
- L2 cache: ~96MB estimated. Aggressive caching strategies viable.
- Profile before assuming A100/H100 patterns transfer."""

        skill += f"""

### Priority 3: Compute Optimization (10-20% impact)
- Warp primitives: __shfl_sync, __ballot_sync for intra-warp communication.
  Hand-rolled reductions give tighter register control than CUB library.
- Occupancy tuning: __launch_bounds__(threads, blocks) for register control.
- {'Cooperative kernels: cg::this_grid().sync() for iterative algorithms without kernel launch overhead. Requires: // CU_FLAGS: --rdc=true' if self.target_gpu in ('a100', 'h100') else 'Evaluate CUB vs hand-rolled for this architecture.'}
- TF32 for matmul: 3x GEMM throughput on Tensor Cores.

### Priority 4: Library Integration
- cuBLAS for GEMM: cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32
- cuDNN for Conv: cudnnConvolutionBiasActivationForward fuses conv+bias+activation
- cuSPARSE for SpMV: cusparseSpMV with CSR format

## Rules
- Do NOT modify verification or profiling scripts
- Do NOT call torch.nn.functional or torch.compile in generated kernels
- Do NOT use Triton or any kernel generation framework
- Do NOT hardcode outputs for specific inputs
- Your kernel MUST work for any valid input shape"""

        return skill

    # ── Internal Methods (problem-specific, simplified here) ──

    def _sample_default_problem(self) -> dict:
        """Return a simple default problem for testing."""
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
        """Generate randomized inputs for correctness checking."""
        # Implementation depends on problem type
        rng = np.random.default_rng(seed)
        return rng.randn(1024).astype(np.float32)

    def _run_kernel(self, so, entry, inputs):
        """Execute compiled kernel via ctypes FFI."""
        import cupy as cp
        d_in = cp.asarray(inputs)
        d_out = cp.zeros_like(d_in)
        fn = getattr(so, entry)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        fn(
            ctypes.c_void_p(d_in.data.ptr),
            ctypes.c_void_p(d_out.data.ptr),
            ctypes.c_int(len(inputs)),
        )
        cp.cuda.Device(0).synchronize()
        return d_out.get()

    def _run_reference(self, inputs):
        """Run PyTorch reference implementation."""
        import torch
        t = torch.tensor(inputs)
        # Problem-specific reference
        return t.numpy()

    def _outputs_match(self, a, b, atol, rtol):
        return np.allclose(a, b, atol=atol, rtol=rtol)


# HTTP server for OpenEnv deployment
app = create_fastapi_app(KernelForgeEnv)
```

### 1.4 Sponsor Integration Points

The hackathon is co-hosted by OpenEnv, Cerebral Valley, and SHACK15. Confirmed sponsors:

| Sponsor | What They Provide | How We Use It |
|---------|-------------------|---------------|
| **Meta-PyTorch** | OpenEnv framework | Our Environment subclass, HTTP deployment |
| **HuggingFace** | TRL GRPOTrainer + Hub | RL training loop + model publishing |
| **Unsloth** | FastModel + MoE optimization | Model loading + 2.5x MoE training speedup (B200 path) |
| **CoreWeave** | GPU compute (likely H100/B200) | Training + evaluation infrastructure |
| **Cursor** | Development environment | All code written during hackathon |

### 1.5 What to Build Before the Hackathon

1. The environment file (`kernelforge_env.py`) — fully tested with a toy problem
2. A Docker container that packages the environment with CUDA toolkit
3. 5-10 test problems with pre-computed baselines
4. A test script that runs `reset() → step() → reward` end-to-end

---

## Objective 2: Optimize the RL Training Pipeline for Qwen3-Coder-Next

### 2.1 The Efficiency Problem

ByteDance CUDA Agent used 128 H20 GPUs, a 230B model, and multi-week training. We need to compress this to 1-2 GPUs in a 24-hour hackathon. Every component must be optimized.

### 2.2 Model Loading: Two Paths

**Path A — H100 80GB + 4-bit GPTQ + QLoRA (primary)**

Qwen3-Coder-Next is 80B total parameters (512 routed experts + 1 shared, 10 active per token, 3B active). In bf16, that is ~160GB — does not fit on H100. In 4-bit GPTQ, it is ~43GB for weights, leaving 27GB headroom.

Blocker: Unsloth cannot do GPTQ-based QLoRA for MoE (BitsAndBytes does not support `nn.Parameter` for MoE experts). Use HuggingFace PEFT + Transformers directly.

```python
# path_a_h100.py — GPTQ + QLoRA on H100 80GB
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

model_id = "dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# LoRA on attention + shared expert only (not all 512 routed experts)
# 512 experts × LoRA = ~207GB optimizer states. Does not fit.
# Shared expert + attention LoRA = ~0.05GB adapters + ~0.3GB optimizer.
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "shared_expert.gate_proj",                 # Shared expert
        "shared_expert.up_proj",
        "shared_expert.down_proj",
    ],
    lora_dropout=0,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Memory budget:
# Weights (GPTQ 4-bit):   ~43 GB
# LoRA adapters (fp16):   ~0.05 GB
# Optimizer (fp32):       ~0.3 GB
# Activations (grad ckpt): ~3 GB
# KV cache (12 attn layers, 8 seqs × 8192 tokens): ~1.5 GB
# Framework overhead:     ~5 GB
# TOTAL:                  ~53 GB (27 GB headroom on H100 80GB)
```

**Path B — B200 192GB + bf16 + Unsloth (fallback)**

```python
# path_b_b200.py — bf16 LoRA with Unsloth on B200 192GB
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

# Memory budget:
# Weights (bf16):         ~160 GB
# LoRA adapters:          ~0.2 GB
# Optimizer:              ~1.6 GB
# Activations (unsloth):  ~4 GB
# KV cache:               ~2 GB
# Overhead:               ~3 GB
# TOTAL:                  ~171 GB (21 GB headroom on B200 192GB)
```

### 2.3 Training Loop: Why GRPO, Not PPO

CUDA Agent used PPO with a critic model. PPO on a single GPU requires loading actor + critic + reference = 3 model copies. For an 80B model, that is impossible.

GRPO (Group Relative Policy Optimization) eliminates the critic entirely. Instead of learning a value function, it normalizes rewards within a group of generations from the same prompt. This saves ~50% memory and works better on single-GPU setups.

Evidence that GRPO is sufficient: DeepSeek-R1 used GRPO (not PPO) on a 671B model and achieved near-parity with PPO-based systems. TRL's GRPOTrainer is the standard implementation.

```python
# grpo_training.py — The actual RL training loop
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    # ── RL hyperparameters ──
    num_generations=8,          # 8 kernels per prompt per step
                                # DeepSeek-R1 uses 16; 8 balances signal vs compute
    temperature=0.9,            # High for exploration (anneal to 0.7 later)
    beta=0.0,                   # No KL penalty (DAPO research shows not essential)

    # ── Sequence lengths ──
    max_prompt_length=1024,     # SKILL.md + problem description
    max_completion_length=4096, # Full CUDA kernel (100-500 lines)

    # ── Training efficiency ──
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Effective batch = 4
    max_steps=60,                   # 60 steps × ~90s/step = ~90 minutes
    learning_rate=5e-6,
    optim="paged_adamw_8bit",       # 8-bit optimizer saves ~50% optimizer memory
    bf16=True,

    # ── Logging ──
    output_dir="outputs/grpo",
    logging_steps=1,
    report_to="wandb",
)
```

### 2.4 The Multi-Stage Pipeline (Why We Cannot Skip It)

ByteDance CUDA Agent's initial pure RL trial collapsed at training step 17. Root cause: CUDA code is less than 0.01% of pretraining data, giving CUDA tokens probability ~10⁻⁹ in BF16. Importance sampling ratios explode.

Their ablation (Table 2, page 8) quantified the damage:
- Without RFT warm-up: training reward collapses, actor entropy spikes, policy becomes diffuse. Results could only be reported at "final validation step before training collapse."
- Without Value Pretraining: critic fails, trajectory lengths explode, training collapses.

**Our adapted 3-stage pipeline (no critic, so 3 stages not 4):**

```
Stage 1: Single-Turn GRPO Warm-Up     (2 hours)
  └─ Easy kernels (single ops). Goal: raise compilation rate from ~50% to ~85%.
  └─ Corresponds to CUDA Agent's "Single-Turn PPO warm-up" stage.

Stage 2: Rejection Fine-Tuning (RFT)  (30 minutes)
  └─ Collect trajectories from Stage 1 model, filter for reward >= 1.0.
  └─ SFT on filtered data to create strong behavioral prior.
  └─ CUDA Agent proved this is mandatory: without RFT, training collapses.

Stage 3: GRPO with Curriculum         (2-3 hours)
  └─ Progressive difficulty: single ops → 2-op fusion → arch-specific → advanced.
  └─ This is where the model discovers new optimization strategies.
  └─ Corresponds to CUDA Agent's "Full Agentic PPO" but single-turn GRPO.
```

### 2.5 Where the Efficiency Gains Come From

| Dimension | CUDA Agent (original) | KernelForge (ours) | Efficiency Gain |
|-----------|-----------------------|---------------------|-----------------|
| **Model** | 230B MoE (23B active) | 80B MoE (3B active) | 7.7x fewer active FLOPs per token |
| **RL algorithm** | PPO (actor + critic) | GRPO (actor only) | 2x memory savings, no critic training |
| **GPUs** | 128 × H20 | 1 × H100 or B200 | 128x fewer GPUs |
| **Context window** | 131,072 tokens | 5,120 tokens (prompt+completion) | 25x shorter sequences |
| **Agent turns** | Up to 200 per episode | 1 (single-turn) | 200x fewer inference calls per episode |
| **Generations per step** | Unknown (likely 32-64) | 8 | Conservative but practical |
| **Training steps** | 150 | 60 | 2.5x fewer steps |
| **Quantization** | None (bf16) | 4-bit GPTQ | 4x weight memory reduction |
| **Optimizer** | AdamW (fp32) | paged_adamw_8bit | 2x optimizer memory reduction |
| **Data** | 6,000 operators | 200 operators (curated subset) | Focused, not exhaustive |

**FLOPs estimation per training step:**

CUDA Agent: 23B active params × 131K tokens × 2 (fwd+bwd) ≈ 6.0 TFLOPs per sequence.

KernelForge: 3B active params × 5K tokens × 2 ≈ 0.03 TFLOPs per sequence.

**~200x FLOPs reduction per sequence.** With 8 generations per step (vs likely 32-64), total FLOPs per step are roughly **800-1600x smaller**.

### 2.6 Data Pipeline Optimization

Instead of generating 6,000 operators from scratch, use CUDA-Agent-Ops-6K (published by ByteDance, Apache 2.0, `BytedTsinghua-SIA/CUDA-Agent-Ops-6K`). Curate a 200-problem subset:

```python
# curate_dataset.py
# Download CUDA-Agent-Ops-6K, filter for hackathon scope

def curate_subset(ops_6k: list[dict], n: int = 200) -> list[dict]:
    """
    Select 200 operators with balanced difficulty:
    - 50 easy: single torch ops (relu, sigmoid, matmul, conv2d)
    - 75 medium: 2-op fusions (matmul+relu, conv2d+bias+relu)
    - 75 hard: 3+ op compositions

    From CUDA Agent data synthesis (Section 3.1):
    - 83.77% of their 6K dataset are 2-op fusions
    - We skew harder to challenge the model

    Pre-compute baselines for each:
    - torch.eager time (median of 30 runs)
    - torch.compile time (median of 30 runs after warmup)
    """
    easy = [op for op in ops_6k if op["num_ops"] == 1][:50]
    medium = [op for op in ops_6k if op["num_ops"] == 2][:75]
    hard = [op for op in ops_6k if op["num_ops"] >= 3][:75]
    return easy + medium + hard
```

### 2.7 Evaluation Signal Optimization

CUDA Agent runs each kernel for correctness on 5 inputs and profiles with 100 warmup + 50 timed runs. The full evaluation takes ~10-15 seconds per kernel.

We reduce this:
- **Correctness: 5 randomized inputs** (same as CUDA Agent — cannot reduce without risking false positives)
- **Profiling: 50 warmup + 30 timed runs** (reduced from 100+50; median is robust with 30 samples)
- **Compilation: ~3-5 seconds** (CPU-bound, cannot reduce)
- **Total per kernel: ~8-12 seconds**
- **Per GRPO step (8 generations): ~70-100 seconds**

CachePool saves ~1-2 seconds per evaluation by keeping test data on GPU.

---

## Objective 3: Automatically Generate Optimized A100 CUDA Kernels

### 3.1 What the Model Produces

For each problem, the model generates a complete .cu file containing:

1. An `extern "C"` entry point function
2. Any number of `__global__` and `__device__` helper kernels
3. Optional `// CU_FLAGS:` comment specifying compilation parameters
4. CUDA runtime calls for memory management and kernel launches

The environment compiles this with `nvcc -arch=sm_80 -O3`, runs it, and returns a reward.

### 3.2 A100-Specific Optimization Techniques (From DoubleGraph)

These are the concrete patterns DoubleGraph used for A100 that the RL agent should discover:

**Technique 1: L2 Cache Pinning (40MB on A100)**

A100 has 40MB of L2 cache with persistence controls. DoubleGraph pins frequently accessed data (parent arrays in BFS, rank vectors in PageRank) using:

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

For graphs up to 10M vertices (40MB / 4 bytes), the entire parent array fits in L2. This drops random access latency from ~200 cycles (HBM) to ~30-40 cycles (L2).

DoubleGraph used this on A10G (6MB L2) but NOT on A100 — because A100's large L2 naturally captures working sets. The agent should learn when explicit pinning helps vs when the hardware handles it.

**Technique 2: Direction-Optimizing Algorithms**

DoubleGraph's A100 BFS uses two phases with cost-model switching:

- Top-down (sparse frontier): queue-based, 1 warp per frontier vertex, `__ballot_sync` for collective insertion
- Bottom-up (dense frontier): bitmap-based, 1 thread per vertex, early termination

Switching thresholds (A100-specific, tuned for 40MB L2):
```
TD → BU when frontier_size > N/20
BU → TD when frontier_size < N/200
```

The agent should discover that A100's large L2 makes bitmap-based approaches cheaper, shifting the switching threshold compared to GPUs with smaller caches.

**Technique 3: Hand-Rolled Warp Primitives (No CUB)**

A100 kernels in DoubleGraph use custom warp reductions instead of the CUB library:

```cuda
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```

Why: CUB calls introduce function-call overhead and use more registers. On A100, hand-rolled gives tighter register control. On L4, DoubleGraph uses CUB instead — the optimal choice is architecture-dependent.

**Technique 4: Cooperative Kernels (Persistent Threads)**

For iterative algorithms, A100 supports cooperative kernel launch with grid-wide synchronization:

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void persistent_bfs(/* ... */) {
    cg::grid_group grid = cg::this_grid();
    for (int level = 0; !done; level++) {
        // Process frontier
        // ...
        grid.sync();  // Grid-wide barrier — no kernel relaunch
    }
}
```

Requires `// CU_FLAGS: --rdc=true` and `cudaLaunchCooperativeKernel`. DoubleGraph uses this for BFS mask and core_number on A100 but NOT on L4 or A10G.

**Technique 5: Kernel Fusion via Algebraic Simplification**

From CUDA Agent's optimization examples (Appendix D):

- Diagonal matmul: `torch.diag(A) @ B` → row-wise scaling. O(N²M) → O(NM). Result: 73× speedup.
- Fused reduction: `Σ_j (x·w_j^T)/2` → `x·(Σ_j w_j^T)/2`. Single column reduction + dot product. Result: 24× speedup.

These are algorithmic, not hardware-specific, but the agent needs to discover them through RL.

### 3.3 Cross-Compilation Strategy

We train on H100/B200 but generate kernels targeting A100:

```bash
# On H100, compile for A100
nvcc -arch=sm_80 -O3 -use_fast_math kernel.cu -o kernel.so --shared -Xcompiler -fPIC
```

The reward function uses **relative speedup** (optimized / naive baseline), not absolute timing. Both the agent's kernel and the baseline are compiled for sm_80, so the relative comparison is valid even when running on H100 hardware. Optimization patterns (coalescing, shared memory tiling, warp primitives, L2 management) transfer across architectures.

### 3.4 Curriculum Progression (DoubleGraph-Inspired)

DoubleGraph's 4-way dispatch (base → segmented → masked → seg+mask) represents increasing optimization difficulty. We repurpose this as training curriculum phases:

| Phase | Steps | Problem Type | Target Optimization | Expected Reward |
|-------|-------|--------------|---------------------|-----------------|
| A: Basics | 0-20 | Single operators | Correct compilation, basic coalescing | 1.0 (correct) |
| B: Fusion | 20-40 | 2-op compositions | Kernel fusion, eliminate intermediates | 2.0 (beats eager) |
| C: Arch-Specific | 40-55 | Complex operators | L2 pinning, warp primitives, launch bounds | 2.0-3.0 |
| D: Advanced | 55-60 | Multi-op + cuBLAS | Library integration, algebraic simplification | 3.0 (beats compile) |

Promotion rule: advance to next phase when >50% of last 10 steps achieve target reward.

### 3.5 WCC-Specific Deep Dive (Primary Demo Algorithm)

For the hackathon demo, WCC (Weakly Connected Components) is the focus because:
- DoubleGraph includes WCC with 4 variants per GPU (WCC base, seg, mask, seg_mask)
- The naive baseline is simple (atomic Union-Find) but has obvious optimization paths
- The optimized version demonstrates L2 pinning, non-atomic stores, and cooperative kernels

**Naive WCC baseline (what the agent must beat):**
```cuda
// Naive: atomicMin hooking + multi-kernel-launch convergence loop
// 3 kernel launches per iteration (init, hook, compress)
// ~100 iterations for power-law graphs
// Bottleneck: atomic operations + kernel launch overhead
```

**What the agent should discover through RL:**
1. Non-atomic path compression (parent pointers only move toward root — mathematically safe without atomics)
2. L2 cache pinning for parent array (10M vertices = 40MB, fits in A100 L2)
3. Single cooperative kernel launch (eliminate per-iteration kernel launch overhead)
4. Path halving: `parent[x] = parent[parent[x]]` instead of full path compression

**Reward progression for WCC:**
- Reward -1: Doesn't compile, or wrong component labels
- Reward +1: Correct but still uses atomics + multi-launch (same speed as baseline)
- Reward +2: Non-atomic path compression works, 2-5x faster
- Reward +3: L2 pinning + cooperative launch + non-atomic, 5-50x faster

---

## Objective 4: Design for Future GPU Architectures

### 4.1 The Modularity Principle

DoubleGraph's architecture proves that GPU-specific optimization requires fundamentally different kernel implementations per architecture — not just parameter tuning. A100 BFS uses queue↔bitmap dual representation. L4 uses bitmap-only. A10G uses 2-level batched top-down. Same algorithm, completely different code.

This means the system must be modular at three levels:

1. **GPU Registry** — Adding a new GPU is adding an entry to `GPU_REGISTRY` (already shown in Section 1.3)
2. **SKILL.md Generation** — Architecture-specific optimization knowledge is template-generated per GPU
3. **Evaluation** — Compilation and benchmarking use the target GPU's nvcc flag, not the training GPU

### 4.2 What Changes Per GPU

| Component | What Changes | What Stays the Same |
|-----------|-------------|---------------------|
| `nvcc_flag` | `-arch=sm_80` → `-arch=sm_90a` → `-arch=sm_100a` | Compilation pipeline |
| SKILL.md content | L2 size, SMEM size, available features | Reward function, environment API |
| `.cu.flags` whitelist | May add architecture-specific flags | Flag parsing logic |
| Baseline timing | Must re-profile per architecture | Relative speedup metric |
| Curriculum thresholds | Re-tune promotion criteria | Curriculum structure |

### 4.3 Adding a New GPU Target

Step-by-step for an engineer adding H100 support:

**Step 1: Add to GPU Registry (5 minutes)**

```python
GPU_REGISTRY["h100"] = {
    "arch": "sm_90a",
    "l2_cache_mb": 50,
    "smem_per_sm_kb": 228,
    "sm_count": 132,
    "hbm_bandwidth_gbps": 3352,
    "registers_per_sm": 65536,
    "nvcc_flag": "-arch=sm_90a",
    "features": [
        "tma", "dsmem", "dpx", "thread_block_clusters",
        "warp_specialization", "fp8", "cp.async",
        "l2_persistence", "bf16_tensor_core", "tf32_tensor_core",
    ],
}
```

**Step 2: Extend SKILL.md Template (30 minutes)**

Add H100-specific optimization guidance to `_build_skill_md()`:

```python
elif self.target_gpu == "h100":
    skill += """
- TMA (Tensor Memory Accelerator): bulk async copy from global to shared.
  One thread initiates copy, 127 freed for computation.
- Thread Block Clusters: __cluster_dims__(2,1,1) groups 2 blocks.
  Cluster-level shared memory enables cross-block communication.
- DPX instructions: __vimin3_s32(a,b,c) for 3-way min in one cycle.
- Distributed Shared Memory: SMs in a cluster access each other's SMEM.
- Warp Specialization: persistent producer/consumer warp roles."""
```

**Step 3: Re-Profile Baselines (1-2 hours)**

```python
# recompute_baselines.py
# Run on target hardware to get accurate eager/compile times
for problem in dataset:
    problem["eager_time_ms"] = profile_eager(problem, gpu="h100")
    problem["compile_time_ms"] = profile_compile(problem, gpu="h100")
```

**Step 4: Re-Tune Curriculum Thresholds (optional, 2-4 hours)**

Run 10 training steps on the new GPU, observe reward distribution, adjust promotion criteria.

**Step 5: Add .cu.flags for New Features (10 minutes)**

If H100 needs new compilation flags:
```python
# Add to whitelist
ALLOWED_CU_FLAGS.add("--threads-per-block=1024")  # if needed
```

**Total time to add a new GPU target: 2-4 hours.** No redesign required.

### 4.4 Future Architecture Considerations

From DoubleGraph's analysis (Section 7 of their SKILLS.md):

**H100 (SM90, Hopper):**
- TMA replaces explicit `cudaMemcpy` patterns for bulk data movement
- Thread Block Clusters add a new hierarchy level between blocks and grid
- DPX instructions accelerate dynamic programming (SSSP delta-stepping)
- 50MB L2 shifts bitmap thresholds relative to A100

**B200/B300 (SM100+, Blackwell):**
- Warp size, register file, SMEM capacity may all change — all hardcoded constants need re-evaluation
- NVLink 5.0 could enable multi-GPU kernels (currently single-GPU only)
- HBM3e bandwidth means memory-bound kernels (BFS bottom-up, PageRank SpMV) benefit most
- New ISA extensions require CUDA toolkit updates
- Compute capability ≥ 10.0 may require `--gpu-architecture=sm_100` minimum

**The meta-pattern (from DoubleGraph Section 8.4):** For any new GPU:
1. Profile before optimizing — run existing A100 kernels on new hardware first
2. Re-tune switching thresholds — BFS direction-switching, Louvain tier thresholds
3. Evaluate library tradeoffs — CUB vs hand-rolled may flip per architecture
4. L2 cache strategy — explicit pinning vs implicit depends on cache size
5. Register pressure — each SM generation has different register file size

### 4.5 Architecture-Aware Training

The system can train the model on multiple GPU targets simultaneously:

```python
# multi_arch_curriculum.py
# Interleave problems targeting different architectures

gpu_targets = ["a100", "h100"]
envs = {gpu: KernelForgeEnv(target_gpu=gpu) for gpu in gpu_targets}

def multi_arch_reward(completions, prompts, **kwargs):
    """
    Each prompt specifies its target GPU.
    Route to the appropriate environment.
    """
    rewards = []
    for completion, prompt in zip(completions, prompts):
        target = prompt.get("target_gpu", "a100")
        env = envs[target]
        result = env.step(completion)
        rewards.append(result.reward)
    return rewards
```

This teaches the model that optimization strategies differ per architecture — the same kernel that gets reward=3 on A100 might get reward=1 on H100 because it doesn't exploit TMA or clusters.

---

## Appendix A: Complete File Manifest

```
KernelForge-OpenEnv/
├── README.md
├── requirements.txt
├── Dockerfile                           # OpenEnv deployment container
│
├── openenv_env/
│   ├── __init__.py
│   ├── kernelforge_env.py              # Objective 1: The environment (Section 1.3)
│   ├── gpu_registry.py                 # Objective 4: GPU_REGISTRY (Section 1.3)
│   ├── cache_pool.py                   # From DoubleGraph (Section 1.3)
│   ├── reward.py                       # CUDA Agent rewards (Section 1.3)
│   ├── anti_hack.py                    # CUDA Agent anti-hacking (Section 1.3)
│   └── skill_builder.py               # Per-GPU SKILL.md generation (Section 1.3)
│
├── training/
│   ├── model_loader.py                 # Objective 2: Path A/B selection (Section 2.2)
│   ├── stage1_warmup.py                # Stage 1: Single-turn GRPO (Section 2.4)
│   ├── stage2_rft.py                   # Stage 2: Rejection fine-tuning (Section 2.4)
│   ├── stage3_grpo.py                  # Stage 3: GRPO with curriculum (Section 2.4)
│   └── curriculum.py                   # Objective 3: Phase progression (Section 3.4)
│
├── datasets/
│   ├── download_ops6k.py               # Pull CUDA-Agent-Ops-6K
│   ├── curate_subset.py                # 200-problem subset (Section 2.6)
│   └── compute_baselines.py            # Pre-compute eager/compile times
│
├── evaluation/
│   ├── ablation.py                     # H1: SFT-only vs GRPO
│   │                                   # H2: RFT necessary (replicate CUDA Agent)
│   │                                   # H3: SKILL.md impact
│   └── eval_model.py                   # Full evaluation suite
│
├── demo/
│   └── app.py                          # Streamlit/Gradio hackathon demo
│
└── tests/
    ├── test_env.py                     # Environment unit tests
    ├── test_reward.py                  # Reward function tests
    └── test_cross_compile.py           # sm_80 compilation on other GPUs
```

## Appendix B: Hackathon Timeline

| Time | Activity | Objective |
|------|----------|-----------|
| **Pre-hackathon** | Download GPTQ model, curate 200 problems, pre-compute baselines, test environment | 1, 2, 3 |
| **Hour 0-1** | Claim GPU, verify model loads, run environment end-to-end | 1, 2 |
| **Hour 1-3** | Stage 1: GRPO warm-up (60 steps, easy kernels) | 2, 3 |
| **Hour 3-4** | Stage 2: RFT (collect trajectories, filter, SFT) | 2 |
| **Hour 4-7** | Stage 3: GRPO with curriculum (60 steps, progressive difficulty) | 2, 3 |
| **Hour 7-8** | Evaluate: compare base → warm-up → RFT → GRPO | 3 |
| **Hour 8-10** | Continue training if improving; run ablations | 2, 3 |
| **Hour 10-12** | Build demo; push model+env to HuggingFace Hub | 1, 4 |
| **Hour 12-13** | Demonstrate multi-GPU-target SKILL.md (A100 + H100) | 4 |
| **Hour 13-14** | Pitch: 3-minute presentation | All |

## Appendix C: Key Evidence References

| Claim | Source | Location |
|-------|--------|----------|
| Pure RL collapsed at step 17 | CUDA Agent paper | Section 3.3, page 6 |
| CUDA tokens = 0.01% of pretraining data | CUDA Agent paper | Section 3.3 |
| RFT ablation: reward collapses without it | CUDA Agent paper | Table 2, Figure 4a |
| Discrete rewards +36pp over continuous | CUDA Agent paper | Table 2 comparison |
| 192 kernel files per GPU target | DoubleGraph SKILLS.md | Section 1, Scale table |
| A100 BFS uses queue↔bitmap dual | DoubleGraph SKILLS.md | Section 4.2 |
| CachePool pattern (LRU, 8 entries) | DoubleGraph SKILLS.md | Section 3 Step 2 |
| 4-way dispatch pattern | DoubleGraph SKILLS.md | Section 3 Step 3 |
| GRPO eliminates critic (~50% memory) | DeepSeek-R1 paper | Architecture section |
| Unsloth cannot do GPTQ QLoRA for MoE | Research transcript | BitsAndBytes nn.Parameter limitation |
| H100 GPTQ path: ~53GB total | Research transcript | Memory budget calculation |
| B200 bf16 path: ~171GB total | Research transcript | Memory budget calculation |
