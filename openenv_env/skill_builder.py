"""Dynamic SKILL.md generation per target GPU architecture.

Extracted from the Implementation Spec Section 1.3 / Section 4.2.
Each GPU gets architecture-specific optimization guidance.
"""
from __future__ import annotations

import os

from openenv_env.gpu_registry import get_gpu_spec


def build_skill_md(gpu_name: str = "a100") -> str:
    """Generate GPU-specific SKILL.md content.

    First tries to load a static skill file (e.g. skill_a100.md).
    Falls back to dynamic generation from GPU_REGISTRY specs.
    """
    # Try static file first (user may have customized it)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_file = os.getenv("KERNELFORGE_SKILL_FILE", "")
    if env_file:
        path = os.path.join(root, env_file)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return f.read()

    static_path = os.path.join(root, f"skill_{gpu_name.lower()}.md")
    if os.path.exists(static_path):
        with open(static_path, encoding="utf-8") as f:
            return f.read()

    # Dynamic generation from registry
    return _generate_skill_md(gpu_name)


def _generate_skill_md(gpu_name: str) -> str:
    """Generate SKILL.md dynamically from GPU_REGISTRY specs."""
    spec = get_gpu_spec(gpu_name)
    name = spec["name"]

    skill = f"""# SKILL.md — {name}-Optimized CUDA Kernel Generation

## Target Hardware
- GPU: NVIDIA {name} ({spec['arch']})
- SMs: {spec['sms']} | L2 Cache: {spec['l2_cache_mb']} MB
- Shared Memory/SM: {spec['smem_per_sm_kb']} KB
- HBM Bandwidth: {spec['hbm_bandwidth_tbs']} TB/s
- Registers/SM: {spec['registers_per_sm']}
- nvcc: {spec.get('nvcc_flag', '-arch=' + spec['arch'])} -O3

## Per-Kernel Compilation Flags
Specify in a comment: // CU_FLAGS: --use_fast_math --maxrregcount=48
Allowed: --use_fast_math, --maxrregcount=N (16-128),
         --rdc=true, --extra-device-vectorization

## Optimization Priorities

### Priority 1: Algorithmic Reduction (>50% impact)
- Algebraic simplification: reduce complexity before optimizing.
  Example: diag(A) @ B = row-wise scaling. O(N^2*M) -> O(N*M).
- Kernel fusion: merge sequential ops into one kernel.
  Eliminates intermediate tensor materialization.
- Operator rearrangement: restructure computation order.
"""

    # Architecture-specific memory guidance
    skill += "\n### Priority 2: Memory Hierarchy (20-50% impact)\n"

    if gpu_name.lower() == "a100":
        skill += f"""- L2 cache pinning: {name} has {spec['l2_cache_mb']}MB L2. Pin frequently accessed arrays:
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 30*1024*1024);
- Vectorized loads: float4 for 16-byte aligned access (4x transaction efficiency)
- Shared memory tiling: pad arrays float tile[32][33] to avoid bank conflicts
- Memory coalescing: consecutive threads access consecutive addresses
- __ldg() for read-only data via texture cache path
"""
    elif gpu_name.lower() == "h100":
        skill += f"""- TMA (Tensor Memory Accelerator): bulk async copy from global to shared.
  One thread initiates copy, 127 freed for computation. 1.45 TB/s.
- Distributed shared memory: SMs in a cluster access each other's SMEM.
- L2 cache pinning: {name} has {spec['l2_cache_mb']}MB L2 (larger than A100).
- Thread Block Clusters: __cluster_dims__ for cross-block shared memory.
- Vectorized loads: float4 for aligned accesses.
"""
    elif gpu_name.lower() == "b200":
        skill += f"""- HBM3e: ~{spec['hbm_bandwidth_tbs']} TB/s bandwidth.
- NVLink 5.0: 1.8 TB/s bidirectional for multi-GPU.
- L2 cache: ~{spec['l2_cache_mb']}MB. Aggressive caching strategies viable.
- TMA + Distributed SMEM from Hopper still apply.
- Profile before assuming A100/H100 patterns transfer.
"""
    else:
        skill += f"- L2 cache: {spec['l2_cache_mb']}MB. Profile to determine best strategy.\n"

    skill += """
### Priority 3: Compute Optimization (10-20% impact)
- Warp primitives: __shfl_sync, __ballot_sync for intra-warp communication.
- Occupancy tuning: __launch_bounds__(threads, blocks) for register control.
- TF32 for matmul: leverages Tensor Cores for ~3x GEMM throughput.
"""

    if spec.get("has_tma"):
        skill += "- Warp specialization: persistent producer/consumer warp roles.\n"

    skill += """
### Priority 4: Library Integration
- cuBLAS for GEMM: cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32
- cuDNN for Conv: cudnnConvolutionBiasActivationForward fuses conv+bias+activation
- cuSPARSE for SpMV: cusparseSpMV with CSR format

## Rules
- Do NOT modify verification or profiling scripts
- Do NOT call torch.nn.functional or torch.compile in generated kernels
- Do NOT use Triton or any kernel generation framework
- Do NOT hardcode outputs for specific inputs
- Your kernel MUST work for any valid input shape
"""
    return skill
