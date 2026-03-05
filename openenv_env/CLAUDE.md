# openenv_env/ — OpenEnv Environment

## GPU Split

> All performance reward (speedup, correctness) executes on **A100 via Modal**. B200 only for local compile checks.
> Judges call `step()` first — cache model on startup, warm CUDA context, pre-load tokenizer.

## OpenEnv Protocol

- **NOT Gymnasium**. HTTP client-server protocol.
- Install: `uv add "openenv-core[core]>=0.2.1"`
- Server: FastAPI app created via `create_fastapi_app(KernelForgeEnv, ...)`

## KernelForgeEnv (`kernel_forge_env.py`)

### Types

```python
class KernelForgeAction(Action):
    cuda_code: str

class KernelForgeObservation(Observation):
    text: str
    baseline_original_ms: float | None = None
    baseline_doublegraph_ms: float | None = None
    hardware: dict[str, Any]
    turn: int = 0
    best_reward: float = -1.0
    info: dict[str, Any]
```

### Constructor
```python
KernelForgeEnv(modal_function_name="kernelforge-a100")
```
- `self.target_gpu` from env `KERNELFORGE_TARGET_GPU` (default "a100")
- `self.max_turns = 200`

### Contract
```python
def reset(self, seed=None, episode_id=None, **kwargs) -> KernelForgeObservation
def step(self, action: KernelForgeAction, timeout_s=None, **kwargs) -> KernelForgeObservation
```

`step()` dispatches to Modal: `evaluate_kernel` with `verify_graphs=5, warmup_iters=50, benchmark_runs=30`

## Reward (`reward.py`)

```python
def compute_reward(compiled, correct, speedup_vs_eager, speedup_vs_compile,
                   occupancy=None, mem_coalescing=None, warp_efficiency=None) -> float
```

| Return | Condition |
|--------|-----------|
| -1.0 | not compiled OR not execution-correct (compile-only is NOT sufficient) |
| log(speedup) | fast path: CUDA events timing on A100 + execution correctness |
| log(speedup) + nsight_bonus | slow path (top-k): + 0.4*occ + 0.3*mem + 0.2*warp from ncu |

```python
def trloo_post_process(advantages: list[float], n: int) -> list[float]
```
Scales GRPO advantages by N/(N-1) to correct Dr. Kernel gradient shrinkage.

## Nsight Integration

Nsight metrics (occupancy, mem_coalescing, warp_efficiency) are passed as optional kwargs to `compute_reward()`. When available, they add a continuous bonus: `0.4*occ + 0.3*mem + 0.2*warp`.

## Anti-Hack (`anti_hack.py`)

### `extract_cu_flags(cuda_code) -> list[str]`
Extracts from `// CU_FLAGS: ...` comment lines.

**Allowed flags**: `--use_fast_math`, `--extra-device-vectorization`, `--rdc=true`, `--maxrregcount=*`

### `scan_forbidden_symbols(so_path) -> str | None`
Uses `nm -D` to scan compiled .so. Returns failure reason or None.

**Forbidden symbols**: `torch`, `at::Tensor`, `c10::`, `torch::autograd`, `triton`, `torch.compile`, `torch.nn.functional`

## GPU Registry (`gpu_registry.py`)

### `get_gpu_spec(gpu_name) -> dict[str, Any]`

| Key | A100 | H100 | B200 |
|-----|------|------|------|
| arch | sm_80 | sm_90a | sm_100a |
| SMs | 108 | 132 | 192 |
| L2 cache | 40 MB | 50 MB | 96 MB |
| SMEM/SM | 164 KB | 228 KB | 228 KB |
| HBM BW | 2.0 TB/s | 3.35 TB/s | 8.0 TB/s |
| Regs/SM | 65,536 | 65,536 | 65,536 |
| TMA | no | yes | yes |
| DSMEM | no | yes | yes |
| async_copy | yes | yes | yes |

**A100 features**: cp.async, l2_persistence, bf16_tensor_core, tf32_tensor_core, structured_sparsity_2_4

## Skill Builder (`skill_builder.py`)

### `build_skill_md(gpu_name="a100") -> str`
Priority chain:
1. Env var `KERNELFORGE_SKILL_FILE` (file path)
2. Static file `skill_{gpu_name}.md` (from project root)
3. `_generate_skill_md(gpu_name)` — dynamic from GPU_REGISTRY

### Generated SKILL.md priorities:
1. Algorithmic reduction (>50% impact)
2. Memory hierarchy (20-50%) — arch-specific patterns
3. Compute optimization (10-20%) — warp primitives, TF32
4. Library integration — cuBLAS, cuDNN, cuSPARSE

## CachePool (`cache_pool.py`)

```python
class GPUCachePool:
    def __init__(self, max_entries: int = 8)
    def get_or_create(self, key: str, factory: Callable[[], Any], metadata=None) -> Any
    def get(self, key: str, default=None) -> Any
    def clear(self) -> None
```

LRU eviction. Calls `close()` or `release()` on evicted entries.

Entry type: `GPUCacheEntry(key: str, value: Any, metadata: dict)`

## Transform Grammar — DEFERRED TO v2

Transformation grammar (12-40 rules) was planned but deferred. No production system has shipped this for CUDA kernels as of March 2026. For v1, use CUDA-Agent's SKILL.md verbatim + doubleGraph pattern paste.

See GRPO-13, `GRPO_DEEP_DIVE.md` line 1852 for reference design.

## Deep Dive Pointers

- Nsight rewards: `GRPO_DEEP_DIVE.md` line 1626 (GRPO-9)
- Transform grammar (deferred): `GRPO_DEEP_DIVE.md` line 1852 (GRPO-13)
- Reward function spec: `GRPO_DEEP_DIVE.md` line 557 (GRPO-3, Sec 3.2)


<claude-mem-context>

</claude-mem-context>