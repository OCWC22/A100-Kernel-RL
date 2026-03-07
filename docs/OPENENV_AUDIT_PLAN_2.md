# KernelForge: OpenEnv + GRPO Implementation Spec

> **Purpose**: Self-contained engineering spec for making KernelForge's OpenEnv environment and GRPO training pipeline 100% functional. Intended for engineers and AI coding agents with no prior context.
>
> **Repo**: `A100-Kernel-RL/` — CUDA kernel optimization via RL on A100 GPUs.

---

## What This System Does

KernelForge trains an LLM (Qwen3-Coder-30B-A3B) to write optimized CUDA kernels using GRPO (Group Relative Policy Optimization). The system has two paths:

1. **Judge/Demo path**: An OpenEnv HTTP server where external users submit CUDA kernels and get compile/correctness/speedup feedback
2. **Training path**: A TRL GRPOTrainer that generates CUDA code, evaluates it on remote A100 GPUs via Modal, and uses discrete rewards {-1, 1, 2, 3} to improve the policy

Both paths share the same reward logic (`reward.py`) and Modal evaluation backend (`task_support.py`).

### System Topology
```
JUDGE / DEMO PATH                          TRAINING PATH
━━━━━━━━━━━━━━━━━                          ━━━━━━━━━━━━━━
Judges / Streamlit / curl                  GRPOTrainer (TRL)
         │                                        │
         ▼                                        ▼
KernelForgeClient (HTTP)               rollout_func (multi_turn_rollout.py)
         │                                        │
         ▼                                        │
OpenEnv HTTP Server                               │
  (server/app.py)                                 │
         │                                        │
         ▼                                        │
KernelForgeEnv.step()                             │
         │                                        │
         ├────────────────────────────────────────┘
         ▼
   SHARED CORE LOGIC
   ├── task_support.py  (Modal dispatch, task routing)
   ├── reward.py        (compute_reward: discrete {-1,1,2,3})
   ├── anti_hack.py     (forbidden symbol scan)
   └── skill_builder.py (SKILL.md generation)
         │
         ▼
   Modal A100 Backend (modal_app.py)
   ├── evaluate_kernel()       (WCC graph tasks)
   ├── evaluate_ops6k_kernel() (Ops-6K dense tasks)
   └── profile_baselines()     (baseline timing)
```

---

## What Needs to Be Fixed

There are **14 patches** across 3 workstreams. Execute in exact order.

### Current Bugs / Misalignments

| # | Problem | File | Severity |
|---|---------|------|----------|
| 1 | No `models.py` — Action/Observation defined inline, not in canonical OpenEnv location | `openenv_env/kernel_forge_env.py` | High |
| 2 | `class KernelForgeEnv(Environment[A,O,S])` uses generic subscript — echo_env reference does NOT | `openenv_env/kernel_forge_env.py` | Medium |
| 3 | `reset(seed, episode_id, **kw)` — OpenEnv contract: `reset()` takes NO args | `openenv_env/kernel_forge_env.py` | High |
| 4 | `step(action, timeout_s, **kw)` — OpenEnv contract: `step(action)` only | `openenv_env/kernel_forge_env.py` | High |
| 5 | `app = create_fastapi_app(...)` inline at module bottom — should be in `server/app.py` | `openenv_env/kernel_forge_env.py` | High |
| 6 | No `openenv.yaml` manifest | repo root | High |
| 7 | No `KernelForgeClient(EnvClient[...])` | missing file | Medium |
| 8 | No `server/` directory structure | missing directory | High |
| 9 | 7 of 11 reward tests FAIL — assert `math.log(speedup)` but code returns discrete {-1,1,2,3} | `tests/test_reward.py` | Critical |
| 10 | 2 of 9 env tests FAIL — same continuous vs discrete mismatch | `tests/test_env.py` | Critical |
| 11 | `conftest.py` mock `_MockAction` missing `metadata` field, `_MockState` not Pydantic | `tests/conftest.py` | High |
| 12 | `docker-compose.yml` uses port 8080, should be 8000; wrong app path | `docker-compose.yml` | Medium |
| 13 | `vllm_mode="colocate" if USE_VLLM else "server"` — **backwards**. When vLLM enabled, want `server` | `training/stage3_grpo.py` | Critical |
| 14 | Missing GRPO config: `beta`, `remove_unused_columns`, `scale_rewards`, `gradient_checkpointing`, `max_prompt_length` | `training/stage3_grpo.py` | High |
| 15 | Unbounded prompt growth in multi-turn: appends full completion+feedback each turn | `training/multi_turn_rollout.py` | High |
| 16 | `max_turns` default is 5 — should be 3 for hackathon GRPO efficiency | `training/multi_turn_rollout.py` | Medium |
| 17 | `modal_train.py` missing vLLM dep and env var forwarding | `modal_train.py` | Medium |

---

## OpenEnv Protocol Reference

OpenEnv is a **Pydantic-typed HTTP client-server protocol** from Meta-PyTorch. Built using Gymnasium-style APIs (step/reset/state) but is NOT the Gymnasium package.

### Base Types (`openenv.core.env_server.types`)
```python
class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    done: bool = Field(default=False)
    reward: bool | int | float | None = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class State(BaseModel):
    model_config = ConfigDict(extra="allow")
    episode_id: Optional[str] = None
    step_count: int = Field(default=0, ge=0)
```

### Environment Interface
```python
class Environment:
    def __init__(self): ...
    def reset(self) -> Observation: ...
    def step(self, action: Action) -> Observation: ...
    @property
    def state(self) -> State: ...
```

### Server Creation
```python
from openenv.core.env_server.http_server import create_app
app = create_app(EnvClass, ActionClass, ObsClass, env_name="myenv")
```

### Canonical Package Structure (from echo_env reference)
```
my_env/
├── __init__.py       # exports Client, Action, Observation
├── client.py         # MyEnvClient(EnvClient[...])
├── models.py         # MyAction(Action), MyObservation(Observation)
├── openenv.yaml      # manifest
├── server/
│   ├── __init__.py
│   └── app.py        # create_app(EnvClass, Act, Obs, env_name=...)
```

### TRL Integration Contract
```python
from trl.experimental.openenv import generate_rollout_completions

def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
    outputs = generate_rollout_completions(trainer, prompts)
    return {
        "prompt_ids": [...], "completion_ids": [...],
        "logprobs": [...], "env_reward": [...],  # extra field forwarded to reward_funcs
    }
```

---

## Reward Function Reference

**File**: `openenv_env/reward.py` — DO NOT MODIFY this file.

```python
def compute_reward(compiled, correct, speedup_vs_eager, speedup_vs_compile, ...) -> float:
    if not compiled or not correct: return -1.0
    if speedup_vs_compile > 1.05:   return  3.0   # Beats torch.compile
    if speedup_vs_eager > 1.05:     return  2.0   # Beats eager PyTorch
    return 1.0                                     # Correct but not faster
```

Nsight kwargs (occupancy, mem_coalescing, warp_efficiency) are accepted but **unused** in discrete mode.

---

## PATCH PLAN — Execute in This Exact Sequence

### Patch 1: CREATE `openenv_env/models.py`

Extract Action and Observation into their own module. Keep ALL existing fields (base has `extra="forbid"` so every field must be declared).

**Current location of these classes**: `openenv_env/kernel_forge_env.py` lines 26-50

**Create this file**:
```python
"""Pydantic models for KernelForge OpenEnv environment."""
from typing import Any

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class KernelForgeAction(Action):
    """Action payload: CUDA kernel source code."""
    cuda_code: str = Field(..., description="CUDA kernel source code string")


class KernelForgeObservation(Observation):
    """Observation payload returned by KernelForgeEnv."""
    text: str = Field(..., description="Environment feedback text")
    baseline_original_ms: float | None = None
    baseline_doublegraph_ms: float | None = None
    hardware: dict[str, Any] = Field(default_factory=dict)
    turn: int = 0
    best_reward: float = -1.0
    info: dict[str, Any] = Field(default_factory=dict)
    graph_properties: dict[str, Any] | None = Field(
        default=None,
        description="Graph topology properties (degree dist, density, diameter, etc.)",
    )
    topology_type: str | None = Field(
        default=None,
        description="Graph topology class: power-law, sparse-islands, dense-regular, etc.",
    )
```

**Test**: `python -c "from openenv_env.models import KernelForgeAction, KernelForgeObservation; print('OK')"`

---

### Patch 2: MODIFY `openenv_env/kernel_forge_env.py`

**4 changes required:**

**Change A** — Fix imports (line 14):
```python
# BEFORE:
from openenv.core.env_server import Environment, create_fastapi_app
from openenv.core.env_server.types import Action, Observation, State

# AFTER:
from openenv.core.env_server import Environment
from openenv.core.env_server.types import State
from openenv_env.models import KernelForgeAction, KernelForgeObservation
```

**Change B** — Delete inline class definitions (lines 26-50):
Delete the `class KernelForgeAction(Action)` and `class KernelForgeObservation(Observation)` blocks entirely. They now live in `models.py`. The import in Change A brings them into this module's namespace, so `from openenv_env.kernel_forge_env import KernelForgeAction` still works.

**Change C** — Fix class declaration and method signatures:
```python
# BEFORE:
class KernelForgeEnv(Environment[KernelForgeAction, KernelForgeObservation, State]):
    ...
    def reset(self, seed=None, episode_id=None, **kwargs) -> KernelForgeObservation:
        ...
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self.current_task = normalize_task_row(kwargs or self.current_task)
        ...

    def step(self, action, timeout_s=None, **kwargs) -> KernelForgeObservation:
        ...

# AFTER:
class KernelForgeEnv(Environment):
    ...
    def reset(self) -> KernelForgeObservation:
        ...
        self._state = State(episode_id=str(uuid4()), step_count=0)
        # Remove: self.current_task = normalize_task_row(kwargs or self.current_task)
        # Keep self.current_task as set in __init__
        ...

    def step(self, action: KernelForgeAction) -> KernelForgeObservation:
        ...
```

**Change D** — Delete the app creation at the bottom of the file (lines 249-251):
```python
# DELETE these lines:
# OpenEnv HTTP server entrypoint
# Run: uvicorn openenv_env.kernel_forge_env:app --host 0.0.0.0 --port 8080
app = create_fastapi_app(KernelForgeEnv, KernelForgeAction, KernelForgeObservation)
```

**Test**: `python -c "from openenv_env.kernel_forge_env import KernelForgeEnv; print('OK')"`

---

### Patch 3: CREATE `openenv_env/server/__init__.py` + `openenv_env/server/app.py`

First create the directory: `mkdir -p openenv_env/server`

**`openenv_env/server/__init__.py`** — empty file

**`openenv_env/server/app.py`**:
```python
"""OpenEnv HTTP server for KernelForge."""
from openenv.core.env_server.http_server import create_app

from openenv_env.kernel_forge_env import KernelForgeEnv
from openenv_env.models import KernelForgeAction, KernelForgeObservation

app = create_app(
    KernelForgeEnv, KernelForgeAction, KernelForgeObservation,
    env_name="kernelforge",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### Patch 4: CREATE `openenv.yaml` (repo root)

```yaml
spec_version: 1
name: kernelforge
type: space
runtime: fastapi
app: openenv_env.server.app:app
port: 8000
```

---

### Patch 5: CREATE `openenv_env/client.py`

```python
"""OpenEnv client for KernelForge — used by TRL and external consumers."""
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from openenv_env.models import KernelForgeAction, KernelForgeObservation


class KernelForgeClient(EnvClient[KernelForgeAction, KernelForgeObservation, State]):
    """HTTP client for the KernelForge environment."""
    pass
```

---

### Patch 6: MODIFY `openenv_env/__init__.py`

**Replace entire file contents with**:
```python
"""
KernelForge OpenEnv Environment for target-GPU CUDA kernel RL training.

OpenEnv is an independent framework from Meta-PyTorch (NOT a Gymnasium
extension). HTTP client-server architecture with Docker container isolation.
Correct install (uv): uv add "openenv-core[core]>=0.2.1" (NOT "openenv")
"""
from openenv_env.models import KernelForgeAction, KernelForgeObservation
from openenv_env.kernel_forge_env import KernelForgeEnv

__all__ = ["KernelForgeEnv", "KernelForgeAction", "KernelForgeObservation"]


def __getattr__(name: str):
    if name == "KernelForgeClient":
        from openenv_env.client import KernelForgeClient
        return KernelForgeClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**Test**: `python -c "from openenv_env import KernelForgeEnv, KernelForgeAction; print('OK')"`

---

### Patch 7: MODIFY `tests/conftest.py`

**Current file** at `tests/conftest.py`. Make these changes:

**Change A** — Add `metadata` field and `Field` import to mock classes:
```python
# BEFORE:
from pydantic import BaseModel, ConfigDict

class _MockAction(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

class _MockObservation(BaseModel):
    reward: float = 0.0
    done: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)

# AFTER:
from pydantic import BaseModel, ConfigDict, Field

class _MockAction(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    metadata: dict = Field(default_factory=dict)

class _MockObservation(BaseModel):
    reward: float = 0.0
    done: bool = False
    metadata: dict = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)
```

**Change B** — Make `_MockState` a Pydantic BaseModel:
```python
# BEFORE:
class _MockState:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# AFTER:
class _MockState(BaseModel):
    model_config = ConfigDict(extra="allow")
    episode_id: str | None = None
    step_count: int = 0
```

**Change C** — Remove `__class_getitem__` from `_MockEnvironment`:
```python
# BEFORE:
class _MockEnvironment:
    def __class_getitem__(cls, params):
        return cls

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __init__(self, *args, **kwargs):
        pass

# AFTER:
class _MockEnvironment:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __init__(self, *args, **kwargs):
        pass
```

**Change D** — Add `http_server` mock to sys.modules:
```python
# ADD after existing sys.modules.setdefault lines:
_http_server = MagicMock()
_http_server.create_app = MagicMock(return_value=MagicMock())
sys.modules.setdefault("openenv.core.env_server.http_server", _http_server)
```

---

### Patch 8: MODIFY `tests/test_reward.py`

The tests assert continuous `math.log(speedup)` rewards, but `compute_reward()` returns discrete `{-1, 1, 2, 3}`.

**Replace entire file with**:
```python
"""Tests for discrete milestone reward computation {-1, 1, 2, 3}."""
import pytest

from openenv_env.reward import compute_reward, trloo_post_process


def test_compile_fail():
    assert compute_reward(compiled=False, correct=False, speedup_vs_eager=0, speedup_vs_compile=0) == -1.0


def test_correct_fail():
    assert compute_reward(compiled=True, correct=False, speedup_vs_eager=0, speedup_vs_compile=0) == -1.0


def test_correct_no_speedup():
    """Correct but speedup=1.0 (not > 1.05) -> reward 1.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=1.0, speedup_vs_compile=0.9)
    assert r == 1.0


def test_modest_speedup():
    """speedup_vs_eager=1.5 > 1.05 -> reward 2.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=1.5, speedup_vs_compile=0.9)
    assert r == 2.0


def test_large_speedup():
    """speedup_vs_eager=3.0 > 1.05 but speedup_vs_compile=1.0 not > 1.05 -> reward 2.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=3.0, speedup_vs_compile=1.0)
    assert r == 2.0


def test_slower_than_baseline():
    """Correct but speedup=0.5 < 1.05 -> reward 1.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=0.5, speedup_vs_compile=0.3)
    assert r == 1.0


def test_very_slow_clamped():
    """Correct but speedup=0.01 < 1.05 -> reward 1.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=0.01, speedup_vs_compile=0)
    assert r == 1.0


def test_nsight_ignored():
    """Nsight metrics are accepted but unused in discrete mode — same reward as without."""
    base = compute_reward(compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.0)
    with_nsight = compute_reward(
        compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.0,
        occupancy=0.8, mem_coalescing=0.9, warp_efficiency=0.7,
    )
    assert with_nsight == base == 2.0


def test_nsight_extreme_values():
    """Nsight with out-of-range values still produces same discrete reward."""
    r = compute_reward(
        compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.0,
        occupancy=1.5, mem_coalescing=-0.1, warp_efficiency=0.5,
    )
    assert r == 2.0


def test_beats_torch_compile():
    """speedup_vs_compile=1.2 > 1.05 -> reward 3.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.2)
    assert r == 3.0


def test_trloo_post_process_g4():
    """TRLOO scales by N/(N-1) = 4/3 for G=4."""
    advantages = [0.5, -0.3, 1.2, -0.8]
    scaled = trloo_post_process(advantages, n=4)
    scale = 4 / 3
    for orig, result in zip(advantages, scaled):
        assert result == pytest.approx(orig * scale, abs=1e-6)


def test_trloo_post_process_g1():
    """TRLOO with N=1 returns unchanged."""
    advantages = [0.5]
    assert trloo_post_process(advantages, n=1) == advantages


def test_trloo_post_process_g2():
    """TRLOO scales by 2/1 = 2.0 for G=2."""
    advantages = [0.3, -0.3]
    scaled = trloo_post_process(advantages, n=2)
    assert scaled[0] == pytest.approx(0.6, abs=1e-6)
    assert scaled[1] == pytest.approx(-0.6, abs=1e-6)
```

**Test**: `uv run pytest tests/test_reward.py -q`

---

### Patch 9: MODIFY `tests/test_env.py`

**Change A** — Update import (line 6-9):
```python
# BEFORE:
from openenv_env.kernel_forge_env import (
    KernelForgeEnv,
    KernelForgeAction,
    KernelForgeObservation,
)

# AFTER:
from openenv_env import KernelForgeEnv, KernelForgeAction, KernelForgeObservation
```

**Change B** — Fix `test_step_correct_kernel` (lines 67-83):
```python
# BEFORE:
    # speedup = 10.0/12.0 ≈ 0.833 → log(0.833) ≈ -0.182
    expected = math.log(10.0 / 12.0)
    assert obs.reward == pytest.approx(expected, abs=1e-3)

# AFTER:
    # speedup = 10.0/12.0 ≈ 0.833, not > 1.05 → discrete reward 1.0
    assert obs.reward == 1.0
```

**Change C** — Fix `test_step_fast_kernel` (lines 86-102):
```python
# BEFORE:
    # speedup = 10.0/5.0 = 2.0 → log(2.0) ≈ 0.693
    assert obs.reward == pytest.approx(math.log(2.0), abs=1e-3)

# AFTER:
    # speedup = 10.0/5.0 = 2.0 > 1.05 → discrete reward 2.0
    assert obs.reward == 2.0
```

**Change D** — Remove unused `import math` (line 2).

**Test**: `uv run pytest tests/test_env.py -q`

---

### Patch 10: MODIFY `docker-compose.yml`

**Change the openenv-server service** (lines 25-41):
```yaml
# BEFORE:
  openenv-server:
    ...
    command: ["python", "-m", "uvicorn", "openenv_env.kernel_forge_env:app", "--host", "0.0.0.0", "--port", "8080"]
    ports:
      - "8080:8080"

# AFTER:
  openenv-server:
    ...
    command: ["python", "-m", "uvicorn", "openenv_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
    ports:
      - "8000:8000"
```

---

### Patch 11: MODIFY `Dockerfile`

**Change A** — Add port 8000 exposure (after line 55 `EXPOSE 8501`):
```dockerfile
EXPOSE 8000
```

**Change B** — Replace CMD with env-switchable entrypoint (line 62):
```dockerfile
# BEFORE:
CMD ["python", "demo/streamlit_demo.py"]

# AFTER:
CMD ["sh", "-c", "if [ \"$KERNELFORGE_MODE\" = 'server' ]; then python -m uvicorn openenv_env.server.app:app --host 0.0.0.0 --port 8000; else python demo/streamlit_demo.py; fi"]
```

---

### Patch 12: MODIFY `training/stage3_grpo.py`

**Change A** — Add imports and env var constants (after line 52):
```python
# ADD after existing constants:
VLLM_MODE = os.getenv("KERNELFORGE_VLLM_MODE", "server").strip().lower()
VLLM_SERVER_BASE_URL = os.getenv("KERNELFORGE_VLLM_SERVER_BASE_URL", "").strip()
USE_TRLOO = os.getenv("KERNELFORGE_USE_TRLOO", "1") == "1"
```

Add GRPOTrainer import (line 36):
```python
# BEFORE:
from trl import GRPOConfig

# AFTER:
from trl import GRPOConfig, GRPOTrainer
```

**Change B** — Fix max_completion_length default (line 60):
```python
# BEFORE:
MAX_COMPLETION_LENGTH = int(os.getenv("KERNELFORGE_STAGE3_MAX_COMPLETION_LENGTH", "1024"))

# AFTER:
MAX_COMPLETION_LENGTH = int(os.getenv("KERNELFORGE_STAGE3_MAX_COMPLETION_LENGTH", "768"))
```

**Change C** — Add config validation function (before `main()`):
```python
def _validate_config():
    effective_batch = 1 * 4  # per_device_train_batch_size * gradient_accumulation_steps
    num_gen = 2  # num_generations
    if effective_batch % num_gen != 0:
        raise ValueError(
            f"Effective batch size ({effective_batch}) must be divisible by "
            f"num_generations ({num_gen}) per TRL GRPO requirement."
        )
    if USE_VLLM and VLLM_MODE == "server" and not VLLM_SERVER_BASE_URL:
        raise ValueError(
            "KERNELFORGE_USE_VLLM=1 with KERNELFORGE_VLLM_MODE=server requires "
            "KERNELFORGE_VLLM_SERVER_BASE_URL to be set."
        )
```

**Change D** — Fix GRPOConfig (lines 163-182). Replace the entire `config = GRPOConfig(...)` block:
```python
    _validate_config()

    grpo_kwargs = dict(
        learning_rate=3e-6,
        temperature=0.7,
        num_generations=2,
        num_iterations=1,
        beta=0.0,                        # No ref model — saves memory
        epsilon=0.2,
        scale_rewards="batch",           # Better for sparse expensive env
        remove_unused_columns=False,     # Custom rollout needs extra columns
        max_prompt_length=3072,
        max_completion_length=MAX_COMPLETION_LENGTH,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=MAX_STEPS,
        optim=OPTIMIZER,
        bf16=USE_BF16,
        gradient_checkpointing=True,
        report_to="none",
        output_dir=OUTPUT_DIR,
        logging_steps=1,
        save_steps=25,
        save_total_limit=2,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.05,
        use_vllm=USE_VLLM,
        vllm_mode=VLLM_MODE if USE_VLLM else "colocate",
        vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
    )
    if USE_VLLM and VLLM_MODE == "server":
        grpo_kwargs["vllm_server_base_url"] = VLLM_SERVER_BASE_URL

    config = GRPOConfig(**grpo_kwargs)
```

**Change E** — Make trainer class selectable (line 184):
```python
# BEFORE:
    trainer = TRLOOGRPOTrainer(

# AFTER:
    trainer_cls = TRLOOGRPOTrainer if USE_TRLOO else GRPOTrainer
    print(f"  Trainer: {trainer_cls.__name__} (USE_TRLOO={USE_TRLOO})")
    trainer = trainer_cls(
```

---

### Patch 13: MODIFY `training/multi_turn_rollout.py`

**Change A** — Fix max_turns default (line 164):
```python
# BEFORE:
    max_turns: int = 5,

# AFTER:
    max_turns: int = 3,
```

**Change B** — Add truncation constants (after line 34):
```python
MAX_FEEDBACK_CHARS = int(os.getenv("KERNELFORGE_MAX_FEEDBACK_CHARS", "1200"))
MAX_ERROR_CHARS = int(os.getenv("KERNELFORGE_MAX_ERROR_CHARS", "800"))
```

**Change C** — Apply truncation in `_format_feedback` (line 110):
```python
# BEFORE:
        parts.append(f"COMPILATION FAILED:\n{error[:800]}")

# AFTER:
        parts.append(f"COMPILATION FAILED:\n{error[:MAX_ERROR_CHARS]}")
```

And at the end of the function (before `return`):
```python
    feedback = "\n".join(parts)
    return feedback[:MAX_FEEDBACK_CHARS]
```

**Change D** — Fix unbounded prompt growth (line 267):
```python
# BEFORE:
                current_prompt = f"{current_prompt}\n\n{completion_text}\n\n{feedback}"

# AFTER:
                current_prompt = build_generation_prompt(
                    task_row,
                    skill_context=skill_context,
                    topology_context=topology_ctx,
                ) + f"\n\n{feedback}"
```

---

### Patch 14: MODIFY `modal_train.py`

**Change A** — Add vLLM to train image deps (after line 43):
```python
# ADD to the .uv_pip_install() list:
        "vllm>=0.10.2",
```

**Change B** — Forward GRPO/vLLM env vars in train function (after line 103):
```python
    os.environ.setdefault("KERNELFORGE_VLLM_MODE", "server")
    os.environ.setdefault("KERNELFORGE_VLLM_SERVER_BASE_URL", "")
    os.environ.setdefault("KERNELFORGE_USE_TRLOO", "1")
    os.environ.setdefault("KERNELFORGE_STAGE3_SCALE_REWARDS", "batch")
    os.environ.setdefault("KERNELFORGE_STAGE3_BETA", "0.0")
    os.environ.setdefault("KERNELFORGE_STAGE3_MAX_PROMPT_LENGTH", "3072")
```

---

## Target File Tree After All Patches

```
A100-Kernel-RL/
├── openenv.yaml                              # NEW (Patch 4)
├── openenv_env/
│   ├── __init__.py                           # MODIFIED (Patch 6)
│   ├── models.py                             # NEW (Patch 1)
│   ├── client.py                             # NEW (Patch 5)
│   ├── kernel_forge_env.py                   # MODIFIED (Patch 2)
│   ├── server/                               # NEW (Patch 3)
│   │   ├── __init__.py
│   │   └── app.py
│   ├── reward.py                             # UNCHANGED
│   ├── anti_hack.py                          # UNCHANGED
│   ├── skill_builder.py                      # UNCHANGED
│   ├── gpu_registry.py                       # UNCHANGED
│   └── cache_pool.py                         # UNCHANGED
├── training/
│   ├── multi_turn_rollout.py                 # MODIFIED (Patch 13)
│   ├── stage3_grpo.py                        # MODIFIED (Patch 12)
│   ├── custom_grpo_trainer.py                # UNCHANGED
│   ├── task_support.py                       # UNCHANGED
│   └── ...
├── modal_app.py                              # UNCHANGED
├── modal_train.py                            # MODIFIED (Patch 14)
├── Dockerfile                                # MODIFIED (Patch 11)
├── docker-compose.yml                        # MODIFIED (Patch 10)
├── tests/
│   ├── conftest.py                           # MODIFIED (Patch 7)
│   ├── test_reward.py                        # MODIFIED (Patch 8)
│   ├── test_env.py                           # MODIFIED (Patch 9)
│   └── ...
└── ...
```

---

## Verification Sequence

Run these after all patches are applied:

```bash
# 1. Import checks (no OpenEnv SDK needed — conftest mocks handle it)
python -c "from openenv_env.models import KernelForgeAction; print('models OK')"
python -c "from openenv_env import KernelForgeEnv, KernelForgeAction; print('package OK')"
python -c "from openenv_env.kernel_forge_env import KernelForgeEnv; print('env OK')"

# 2. Full test suite — ALL tests must pass
uv run pytest tests/ -q

# 3. Stage 0 smoke test on Modal (requires Modal auth)
modal run modal_train.py --stage 0

# 4. OpenEnv server (requires openenv-core installed)
pip install "openenv-core[core]>=0.2.1"
uvicorn openenv_env.server.app:app --host 0.0.0.0 --port 8000

# 5. Test HTTP endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"cuda_code": "__global__ void k() {}"}'

# 6. GRPO smoke test (tiny, proves plumbing works)
KERNELFORGE_USE_VLLM=0 KERNELFORGE_STAGE3_MAX_STEPS=1 \
  modal run modal_train.py --stage 3
```

---

## Key Function Signatures (for cross-referencing)

### `training/task_support.py`
```python
def evaluate_code_on_modal(cuda_code: str, task_row: dict, modal_app_name: str,
                           baseline_orig_ms: float | None = None,
                           baseline_dg_ms: float | None = None) -> dict
def compute_task_reward(result: dict | None) -> float
def normalize_eval_result(result: dict | None) -> dict
def normalize_task_row(row: dict) -> dict
def build_generation_prompt(task_row: dict, skill_context: str = "",
                            topology_context: str = "") -> str
def build_prompt_lookup(rows: list[dict]) -> dict[str, dict]
```

### `training/curriculum.py`
```python
def format_topology_context(problem: dict) -> str
def format_problem_prompt(problem: dict) -> str
class CurriculumManager:
    def get_problem(self) -> dict
    def record_reward(self, reward: float) -> str | None
    def status(self) -> dict
```

### `openenv_env/reward.py` (DO NOT MODIFY)
```python
def compute_reward(compiled, correct, speedup_vs_eager, speedup_vs_compile,
                   occupancy=None, mem_coalescing=None, warp_efficiency=None) -> float
    # Returns: -1.0 | 1.0 | 2.0 | 3.0
def trloo_post_process(advantages: list[float], n: int) -> list[float]
```

---

## Architectural Decisions (Do Not Change)

1. **Training rollout talks directly to Modal** — NOT through the HTTP env server. This is intentional. The rollout needs multi-turn prompt management that step-level API doesn't handle.
2. **Discrete rewards {-1, 1, 2, 3}** — per CUDA Agent ablation (96.8% vs 60.4% faster rate over continuous). Do not switch to continuous.
3. **Local compile precheck** — `nvcc -arch=sm_80 -c` before Modal dispatch. Saves ~50% eval cost.
4. **GPU split** — H200 for training, A100 (Modal) for ALL performance evaluation. Never use training GPU timing for reward.
5. **vLLM server mode** (not colocate) when enabled — training + vLLM + remote reward on one GPU is too many moving parts.
