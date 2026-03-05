# openenv_env/ — OpenEnv Environment

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
def compute_reward(compiled: bool, correct: bool, speedup_vs_eager: float, speedup_vs_compile: float) -> float
```

| Return | Condition |
|--------|-----------|
| -1.0 | not compiled OR not correct |
| 3.0 | speedup_vs_compile > 1.05 |
| 2.0 | speedup_vs_eager > 1.05 |
| 1.0 | correct, no meaningful speedup |

## Nsight Bonus (NOT YET IMPLEMENTED)

Planned formula (see GRPO-9, `GRPO_DEEP_DIVE.md` line 1276):
- Extract ncu metrics: `sm__throughput.avg.pct_of_peak_sustained_elapsed`, `dram__throughput`, etc.
- Compute continuous bonus [0, 0.5] on top of discrete reward
- **File to create**: `reward_nsight.py`

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

## Transform Grammar (NOT YET IMPLEMENTED)

Planned: 12 core + 28 extended transformation rules for CUDA kernel optimization.
See GRPO-13, `GRPO_DEEP_DIVE.md` line 1490.

- Parser extracts applicable transforms from kernel source via regex
- Prompt template injects transform suggestions into LLM context
- **File to create**: `transform_grammar.py`

## Files to Create

- [ ] `transform_grammar.py` — 40-rule grammar parser + prompt template
- [ ] `reward_nsight.py` — Nsight Compute bonus reward [0, 0.5]

## Deep Dive Pointers

- Nsight rewards: `GRPO_DEEP_DIVE.md` line 1276 (GRPO-9)
- Transform grammar: `GRPO_DEEP_DIVE.md` line 1490 (GRPO-13)
- Reward function spec: `GRPO_DEEP_DIVE.md` line 237 (GRPO-3, Sec 3.2)
