# KernelForge OpenEnv Audit + Hackathon Execution Context

## Context

The hackathon is March 7-9, 2026. We need a judge-runnable OpenEnv environment for CUDA kernel optimization on A100, integrated with TRL GRPO. This audit is based on real source inspection of: our repo, OpenEnv's actual SDK source on GitHub, the echo_env reference environment on HF Spaces, and TRL's official OpenEnv integration docs.

---

## 1. OpenEnv Reality Check

### What OpenEnv Actually Is (from verified source)

OpenEnv is a **Pydantic-typed HTTP client-server protocol** from Meta-PyTorch. NOT Gymnasium.

**Base types** (`openenv.core.env_server.types`):
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

**Environment** (`openenv.core.env_server.interfaces`):
```python
class Environment:  # No generic subscript in reference impl
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    def __init__(self): ...           # No-arg constructor
    def reset(self) -> Obs: ...       # No args
    def step(self, action: Act) -> Obs: ...  # Action only
    @property
    def state(self) -> State: ...
```

**Server** (`openenv.core.env_server.http_server`):
```python
from openenv.core.env_server.http_server import create_app
app = create_app(EnvClass, ActionClass, ObsClass, env_name="myenv")
# Runs: uvicorn server.app:app --host 0.0.0.0 --port 8000
```

**Client** (per-environment, for TRL):
```python
class MyEnvClient(EnvClient[MyAction, MyObs, State]):
    def _step_payload(self, action) -> dict: ...
    def _parse_result(self, payload) -> StepResult: ...
    def _parse_state(self, payload) -> State: ...
```

**Package structure** (echo_env reference):
```
my_env/
├── __init__.py       # exports Client, Action, Observation
├── client.py         # MyEnvClient(EnvClient[...])
├── models.py         # MyAction(Action), MyObservation(Observation)
├── openenv.yaml      # manifest: spec_version, name, type, runtime, app, port
├── pyproject.toml    # deps including openenv-core
├── Dockerfile        # FROM ghcr.io/meta-pytorch/openenv-base:latest
└── server/
    ├── __init__.py
    ├── app.py        # create_app(EnvClass, Act, Obs, env_name=...)
    └── environment.py
```

### What OpenEnv Does NOT Solve
- Remote GPU dispatch (that's Modal/K8s)
- Reward function design
- Multi-turn prompt management (that's your rollout_func)
- Training orchestration (that's TRL)

### TRL Integration Contract (from official docs)
```python
from trl.experimental.openenv import generate_rollout_completions

def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
    outputs = generate_rollout_completions(trainer, prompts)
    # Step through env, collect rewards
    return {
        "prompt_ids": [out["prompt_ids"] for out in outputs],
        "completion_ids": [out["completion_ids"] for out in outputs],
        "logprobs": [out["logprobs"] for out in outputs],
        "env_reward": env_rewards,  # Extra fields -> forwarded to reward_funcs via kwargs
    }
```

---

## 2. Repo Audit Against OpenEnv

### Component Classification

| Component | Impl Status | OpenEnv Alignment | Issue |
|---|---|---|---|
| `KernelForgeAction(Action)` | Implemented | **Aligned** | Missing `metadata` field from base; works if base provides it |
| `KernelForgeObservation(Observation)` | Implemented | **Aligned** | Custom fields declared; `done`/`reward` present; base has `extra="forbid"` so all fields must be declared — they are |
| `KernelForgeEnv(Environment[A,O,S])` | Implemented | **MISALIGNED** | Uses generic subscript `Environment[A,O,S]` — echo ref does NOT. `__class_getitem__` mocked in conftest; may fail with real SDK |
| `__init__(modal_function_name)` | Implemented | **Partially aligned** | Has default `None`, so no-arg call works. But constructor arg pattern differs from echo ref |
| `reset(seed, episode_id, **kw)` | Implemented | **MISALIGNED** | Echo ref: `reset()` takes NO args. Server likely calls `env.reset()` bare |
| `step(action, timeout_s, **kw)` | Implemented | **MISALIGNED** | Echo ref: `step(self, action)` only. Extra params may cause override mismatch |
| `state` property | Implemented | **Aligned** | State uses `extra="allow"`, extra fields (history, best_reward) valid |
| `create_fastapi_app()` call | Implemented | **Partially aligned** | Uses `create_fastapi_app` (exported alias for `create_app`). Missing `env_name` param |
| `openenv.yaml` | **MISSING** | **MISSING** | Required manifest for environment discovery |
| Client class | **MISSING** | **MISSING** | No `KernelForgeClient(EnvClient[...])`. Judges/TRL need this for HTTP |
| `models.py` | **MISSING** | **MISSING** | Action/Obs inline in env file, not separate module |
| `server/` directory | **MISSING** | **MISSING** | App creation inline, not in `server/app.py` |
| Dockerfile | Implemented | **MISALIGNED** | Uses `nvidia/cuda` base, not `openenv-base`. CMD is streamlit, not uvicorn |
| docker-compose openenv-server | Implemented | **Partially aligned** | Port 8080 (standard is 8000). Command is correct |
| `multi_turn_rollout.py` | Implemented | **PARALLEL PATH** | Correct TRL API (`generate_rollout_completions`, correct return dict). BUT bypasses OpenEnv env entirely — talks directly to Modal |
| `reward.py` | Implemented | N/A (internal) | Correct discrete {-1,1,2,3}. TRLOO post-process works |
| `modal_app.py` | Implemented | N/A (infra) | Real A100 eval. `evaluate_kernel`, `evaluate_ops6k_kernel` both work |
| `task_support.py` | Implemented | N/A (internal) | Task routing, Modal dispatch, reward computation. Well-factored |

### Test Suite: BROKEN

| Test File | Status | Issue |
|---|---|---|
| `test_reward.py` | **7 of 11 tests FAIL** | Tests assert `log(speedup)` + Nsight bonus. Actual code returns discrete {-1,1,2,3}. The docstring literally says "Tests for continuous reward computation" |
| `test_env.py` | **2 of 9 tests FAIL** | `test_step_correct_kernel` expects `math.log(10.0/12.0)`, actual returns `1.0`. `test_step_fast_kernel` expects `math.log(2.0)`, actual returns `2.0` |
| `conftest.py` | Works but fragile | `_MockAction` missing `metadata` field. `_MockState` not Pydantic. `__class_getitem__` hack for generic subscript |
| Other test files | Likely pass | `test_anti_hack.py`, `test_skill_builder.py`, `test_curriculum.py` etc test internal logic |

### Overclaims

1. **Tests**: Docs claim "113 tests passing" — at least 9 tests fail against the current discrete reward function
2. **Nsight**: CLAUDE.md marks Nsight as "DONE" — `compute_reward()` accepts but ignores Nsight kwargs entirely
3. **OpenEnv compliance**: Has `create_fastapi_app` line but misses: manifest, client, server directory, correct signatures, Dockerfile base

### Key Architectural Split

The training path (`multi_turn_rollout.py`) and the env server path (`kernel_forge_env.py`) are **two separate implementations** of the same logic. They share `reward.py` and `task_support.py` but:
- Training: rollout_func → generate → extract code → local compile → Modal eval → compute reward → append feedback
- Env server: HTTP step → build payload → Modal eval → compute reward → return observation

This is actually fine — training path needs multi-turn prompt management that step-level API doesn't handle. But it means:
- **Env server = judge/demo path** (what people interact with)
- **Training rollout = training path** (what TRL calls)
- Both must produce consistent rewards (they do, via shared `compute_reward`)

---

## 3. Best OpenEnv Formulation for CUDA Kernel Optimization

### Action
```python
class KernelForgeAction(Action):
    cuda_code: str = Field(..., description="CUDA kernel source code")
```
Correct as-is. One action = one kernel submission.

### Observation
```python
class KernelForgeObservation(Observation):
    # Inherited from base: done, reward, metadata
    text: str                                # Feedback for multi-turn
    turn: int = 0
    compiles: bool = False
    correct: bool = False
    runtime_ms: float | None = None
    speedup_vs_eager: float | None = None
    hardware: dict[str, Any] = Field(default_factory=dict)
    graph_properties: dict[str, Any] | None = None
    topology_type: str | None = None
```
Changes from current: remove `best_reward` (leaked state), remove `info` (use inherited `metadata`), remove `baseline_*_ms` (internal). Add `compiles`, `correct`, `runtime_ms`, `speedup_vs_eager` as structured fields (currently only in text).

### State
```python
State(episode_id="uuid", step_count=turn, best_reward=best_reward, history=[...])
```
State has `extra="allow"` so extra fields are valid. Current impl is correct.

### Reward
Discrete milestones `{-1.0, 1.0, 2.0, 3.0}`. Correct per CUDA Agent ablation. Implementation is correct.

### Done
```python
done = (turn >= max_turns) or (reward >= 3.0)
```
Early exit on beating torch.compile. Correct.

### Reset
Must return initial observation with SKILL.md context + problem prompt. **Must take no args** per OpenEnv contract.

### Multi-Turn
Handled in rollout_func, NOT in the environment. Each `env.step(action)` is one turn. Rollout appends observation.text to prompt for next generation. This matches TRL pattern.

---

## 4. TRL Integration Plan

### Current Architecture (KEEP for training)

```
GRPOTrainer.train()
  → rollout_func(prompts, trainer)     [multi_turn_rollout.py]
    → generate_rollout_completions()   [TRL helper]
    → extract_cuda_code()             [local]
    → _local_compile_check()          [nvcc on training GPU]
    → evaluate_code_on_modal()        [task_support.py → Modal A100]
    → compute_task_reward()           [reward.py discrete milestones]
    → _format_feedback()              [text for next turn]
    → return {prompt_ids, completion_ids, logprobs, env_reward}
  → reward_from_env(completions, **kwargs)
    → extract env_reward from kwargs
```

This is correct and should NOT be changed. The rollout func:
- Uses `generate_rollout_completions` correctly
- Returns the correct dict format with extra `env_reward` field
- Multi-turn appends feedback to prompt
- Local compile fast-fails save ~50% Modal cost
- Early exits at reward >= 3.0

### What Goes WHERE

| Concern | Location | Rationale |
|---|---|---|
| Model generation | rollout_func | TRL controls generation |
| CUDA code extraction | rollout_func | Before env interaction |
| Local compile check | rollout_func | Fast-fail on training GPU |
| Modal A100 dispatch | Both paths use `task_support.py` | Shared logic |
| Reward computation | `reward.py` (shared) | Single source of truth |
| Multi-turn prompt mgmt | rollout_func | Not env's job |
| Observation formatting | env `step()` | For HTTP/judge path |
| Curriculum management | rollout_func / stage3 | Training concern |

### Modal A100 Invocation Flow
```
rollout_func → evaluate_code_on_modal(code, task_row, app_name, baselines)
  → modal.Function.from_name(app_name, fn_name).remote(payload)
  → normalize_eval_result(result)
  → compute_task_reward(result)
```

---

## 5. Clean Target Architecture

### Component Diagram
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
   ├── reward.py        (compute_reward, discrete milestones)
   ├── anti_hack.py     (flag extraction, symbol scan)
   └── skill_builder.py (SKILL.md generation)
         │
         ▼
   Modal A100 Backend
   ├── evaluate_kernel()       (WCC tasks)
   ├── evaluate_ops6k_kernel() (Ops-6K tasks)
   └── profile_baselines()     (baseline timing)
```

### Target Repo Structure
```
A100-Kernel-RL/
├── openenv.yaml                              # NEW
├── openenv_env/
│   ├── __init__.py                           # MODIFY: export client, models
│   ├── models.py                             # NEW: Action + Observation
│   ├── client.py                             # NEW: KernelForgeClient(EnvClient)
│   ├── kernel_forge_env.py                   # MODIFY: fix signatures, import from models
│   ├── server/                               # NEW directory
│   │   ├── __init__.py                       # NEW
│   │   └── app.py                            # NEW: create_app(...)
│   ├── reward.py                             # UNCHANGED
│   ├── anti_hack.py                          # UNCHANGED
│   ├── skill_builder.py                      # UNCHANGED
│   ├── gpu_registry.py                       # UNCHANGED
│   └── cache_pool.py                         # UNCHANGED
├── training/
│   ├── multi_turn_rollout.py                 # UNCHANGED (training path stays direct)
│   ├── custom_grpo_trainer.py                # UNCHANGED
│   ├── stage3_grpo.py                        # UNCHANGED
│   ├── task_support.py                       # UNCHANGED
│   └── ...
├── modal_app.py                              # UNCHANGED
├── modal_train.py                            # UNCHANGED
├── Dockerfile                                # MODIFY: CMD for uvicorn
├── docker-compose.yml                        # MODIFY: port 8000
├── tests/
│   ├── conftest.py                           # MODIFY: fix mocks
│   ├── test_reward.py                        # MODIFY: fix for discrete rewards
│   ├── test_env.py                           # MODIFY: fix for discrete rewards
│   └── ...
└── ...
```

---

## 6. Shortest Credible Hackathon Path

### Recommendation: **Environment-fix + SFT-demonstrated + Search-demonstrated + Demo**

### Why NOT full GRPO
- 50 GRPO steps x 2 generations x multi-turn = thousands of Modal A100 calls
- Requires H200 for training ($4.54/hr) + A100 for eval ($2.50/hr) simultaneously
- Has never been run end-to-end. First run WILL have bugs
- Abort conditions in our own docs acknowledge this risk

### What IS credible this weekend

**P0 (3 hours): Fix OpenEnv contract compliance**
- Fix env signatures, add manifest/client/server, fix Dockerfile
- This is what judges will inspect first

**P1 (1 hour): Fix broken tests**
- 9+ tests fail against current discrete reward. Fix them
- Show a green test suite

**P2 (2-3 hours): Run and demonstrate SFT**
- 192 doubleGraph expert demos already in `datasets/doublegraph_sft.jsonl`
- Stage 2 RFT is implemented. Run it on Modal H200
- Produces a model that can generate compilable kernels

**P3 (2-3 hours): Run AdaEvolve search**
- `skydiscover_integration/` is implemented
- Uses real A100 eval via Modal
- Produces actual speedup numbers

**P4 (1-2 hours): Polish demo with real results**
- Feed real eval results into Streamlit demo
- Show: env contract + SFT learning curve + search speedups

### What judges can run
```bash
# 1. Deploy eval backend
modal deploy modal_app.py

# 2. Start OpenEnv server
uvicorn openenv_env.server.app:app --host 0.0.0.0 --port 8000

# 3. Submit a kernel
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"cuda_code": "__global__ void test_kernel() {}"}'

# 4. Run tests
uv run pytest tests/ -q

# 5. View demo
uv run streamlit run demo/streamlit_demo.py
```

---

## 7. Exact Implementation Plan

### Phase 1: Fix OpenEnv Contract (P0)

#### 1.1 Create `openenv_env/models.py`
Extract Action + Observation from `kernel_forge_env.py`:
```python
from pydantic import Field
from openenv.core.env_server.types import Action, Observation

class KernelForgeAction(Action):
    cuda_code: str = Field(..., description="CUDA kernel source code")

class KernelForgeObservation(Observation):
    # Inherits: done, reward, metadata
    text: str = Field(..., description="Environment feedback text")
    turn: int = 0
    best_reward: float = -1.0
    hardware: dict = Field(default_factory=dict)
    info: dict = Field(default_factory=dict)
    graph_properties: dict | None = None
    topology_type: str | None = None
    baseline_original_ms: float | None = None
    baseline_doublegraph_ms: float | None = None
```
Keep all existing fields to avoid breaking existing code. The base class has `extra="forbid"` so all must be declared — they are.

#### 1.2 Modify `openenv_env/kernel_forge_env.py`
- Remove generic subscript: `class KernelForgeEnv(Environment):`
- Import from models.py: `from openenv_env.models import KernelForgeAction, KernelForgeObservation`
- Remove inline Action/Observation class definitions
- Simplify `reset()`: remove `seed` and `episode_id` params (use kwargs for backward compat internally, but signature matches contract)
- Simplify `step()`: remove `timeout_s` param
- Remove `app = create_fastapi_app(...)` line at bottom (moved to server/app.py)

#### 1.3 Create `openenv_env/server/__init__.py`
Empty file.

#### 1.4 Create `openenv_env/server/app.py`
```python
from openenv.core.env_server.http_server import create_app
from openenv_env.models import KernelForgeAction, KernelForgeObservation
from openenv_env.kernel_forge_env import KernelForgeEnv

app = create_app(KernelForgeEnv, KernelForgeAction, KernelForgeObservation, env_name="kernelforge")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
```

#### 1.5 Create `openenv.yaml`
```yaml
spec_version: 1
name: kernelforge
type: space
runtime: fastapi
app: openenv_env.server.app:app
port: 8000
```

#### 1.6 Create `openenv_env/client.py`
```python
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State
from openenv_env.models import KernelForgeAction, KernelForgeObservation

class KernelForgeClient(EnvClient[KernelForgeAction, KernelForgeObservation, State]):
    # Subclass EnvClient with serialization hooks
    ...
```

#### 1.7 Update `openenv_env/__init__.py`
```python
from openenv_env.models import KernelForgeAction, KernelForgeObservation
from openenv_env.kernel_forge_env import KernelForgeEnv
# Lazy import client (requires openenv-core)
```

#### 1.8 Fix `docker-compose.yml`
- Change openenv-server port: `8080` -> `8000`
- Change command: `openenv_env.kernel_forge_env:app` -> `openenv_env.server.app:app`

#### 1.9 Fix `Dockerfile`
- Add `EXPOSE 8000` alongside 8501
- Change default CMD or add an ENV-based entrypoint that can serve either streamlit or uvicorn

### Phase 2: Fix Broken Tests (P1)

#### 2.1 Fix `tests/test_reward.py`
Replace all continuous reward assertions with discrete:
- `test_correct_no_speedup`: expect `1.0` (not `0.0`)
- `test_modest_speedup`: expect `2.0` (speedup 1.5 > 1.05 vs eager)
- `test_large_speedup`: expect `2.0` (speedup 3.0 vs eager, but speedup_vs_compile=1.0 not > 1.05)
- `test_slower_than_baseline`: expect `1.0` (correct but speedup 0.5 < 1.05)
- `test_very_slow_clamped`: expect `1.0` (correct, speedup 0.01 < 1.05)
- `test_nsight_bonus`: expect same as base (Nsight unused)
- `test_nsight_clamped`: expect same as base
- TRLOO tests pass as-is

#### 2.2 Fix `tests/test_env.py`
- `test_step_correct_kernel`: expect `1.0` (speedup 10/12=0.83 < 1.05)
- `test_step_fast_kernel`: expect `2.0` (speedup 10/5=2.0 > 1.05 vs eager)

#### 2.3 Fix `tests/conftest.py`
- Add `metadata: Dict[str, Any] = Field(default_factory=dict)` to `_MockAction` and `_MockObservation`
- Remove `__class_getitem__` hack from `_MockEnvironment` (no longer needed after removing generic subscript)
- Update imports in test files if Action/Observation now come from `models.py`

### Phase 3: Verify

#### Smoke tests (in order):
```bash
# 1. Import check (no OpenEnv SDK needed - conftest mocks)
uv run python -c "from openenv_env.kernel_forge_env import KernelForgeEnv; print('OK')"

# 2. Run fixed test suite
uv run pytest tests/ -q

# 3. Deploy Modal eval backend (requires Modal auth)
modal deploy modal_app.py

# 4. Preflight check
modal run modal_train.py --stage 0

# 5. Start OpenEnv server (requires openenv-core installed)
pip install "openenv-core[core]>=0.2.1"
uvicorn openenv_env.server.app:app --host 0.0.0.0 --port 8000

# 6. Test HTTP endpoint
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \
  -d '{"cuda_code": "__global__ void k() {}"}'
```

### Execution Order

| Step | File | Action | Test After |
|---|---|---|---|
| 1 | `openenv_env/models.py` | CREATE | `python -c "from openenv_env.models import KernelForgeAction"` |
| 2 | `openenv_env/kernel_forge_env.py` | MODIFY (remove inline types, fix signatures, remove app) | `python -c "from openenv_env.kernel_forge_env import KernelForgeEnv"` |
| 3 | `openenv_env/server/__init__.py` | CREATE | — |
| 4 | `openenv_env/server/app.py` | CREATE | `python -c "from openenv_env.server.app import app"` (with SDK) |
| 5 | `openenv.yaml` | CREATE | — |
| 6 | `openenv_env/client.py` | CREATE | — |
| 7 | `openenv_env/__init__.py` | MODIFY | `python -c "from openenv_env import KernelForgeAction"` |
| 8 | `tests/conftest.py` | MODIFY | — |
| 9 | `tests/test_reward.py` | MODIFY | `pytest tests/test_reward.py -q` |
| 10 | `tests/test_env.py` | MODIFY | `pytest tests/test_env.py -q` |
| 11 | `docker-compose.yml` | MODIFY | `docker-compose config` |
| 12 | `Dockerfile` | MODIFY | `docker build .` |
| 13 | Full test suite | — | `pytest tests/ -q` |

### What to DEFER until after hackathon
- Restructuring `server/` with separate `kernel_forge_environment.py` (cosmetic)
- Switching Dockerfile base to `openenv-base` (our image works, just non-standard)
- HF Spaces deployment via `openenv push`
- Full GRPO training run
- README overclaim cleanup (not judge-facing)

---

## PATCH PLAN

### Priority Order (do these first, in this exact sequence)

**Patch 1: `openenv_env/models.py` (CREATE)**
- Extract `KernelForgeAction` and `KernelForgeObservation` from `kernel_forge_env.py`
- Keep exact same fields. Import Action/Observation from `openenv.core.env_server.types`
- Test: import succeeds

**Patch 2: `openenv_env/kernel_forge_env.py` (MODIFY)**
- Change `class KernelForgeEnv(Environment[KernelForgeAction, KernelForgeObservation, State]):` → `class KernelForgeEnv(Environment):`
- Change `from openenv.core.env_server import Environment, create_fastapi_app` → `from openenv.core.env_server.interfaces import Environment`
- Import `from openenv_env.models import KernelForgeAction, KernelForgeObservation`
- Remove inline class definitions of KernelForgeAction and KernelForgeObservation
- Change `def reset(self, seed=None, episode_id=None, **kwargs)` → `def reset(self) -> KernelForgeObservation:`
  - Move seed/episode_id handling to internal defaults
- Change `def step(self, action, timeout_s=None, **kwargs)` → `def step(self, action: KernelForgeAction) -> KernelForgeObservation:`
- Remove last line: `app = create_fastapi_app(...)`
- Test: import succeeds

**Patch 3: `openenv_env/server/__init__.py` + `openenv_env/server/app.py` (CREATE)**
- `app.py`: `from openenv.core.env_server.http_server import create_app` + `app = create_app(KernelForgeEnv, KernelForgeAction, KernelForgeObservation, env_name="kernelforge")`
- Test: import succeeds (with SDK) or mock test

**Patch 4: `openenv.yaml` (CREATE)**
- 6-line manifest file
- Test: file exists

**Patch 5: `openenv_env/client.py` (CREATE)**
- Minimal `KernelForgeClient(EnvClient[...])` with serialization hooks
- Test: import succeeds (with SDK)

**Patch 6: `openenv_env/__init__.py` (MODIFY)**
- Export: KernelForgeEnv, KernelForgeAction, KernelForgeObservation, KernelForgeClient (lazy)
- Test: `from openenv_env import KernelForgeAction`

**Patch 7: `tests/conftest.py` (MODIFY)**
- Add `metadata` field to mock Action/Observation
- Remove `__class_getitem__` from `_MockEnvironment`
- Add mock for `openenv.core.env_server.interfaces` module
- Test: conftest loads

**Patch 8: `tests/test_reward.py` (MODIFY)**
- Replace all `math.log(...)` assertions with discrete values {1.0, 2.0, 3.0}
- Replace Nsight bonus assertions with same-as-base assertions
- Update docstring from "continuous" to "discrete"
- Test: `pytest tests/test_reward.py -q` all green

**Patch 9: `tests/test_env.py` (MODIFY)**
- `test_step_correct_kernel`: assert `obs.reward == 1.0`
- `test_step_fast_kernel`: assert `obs.reward == 2.0`
- Update `env()` fixture if reset/step signatures changed
- Test: `pytest tests/test_env.py -q` all green

**Patch 10: `docker-compose.yml` (MODIFY)**
- openenv-server port: 8080 → 8000
- command: `openenv_env.kernel_forge_env:app` → `openenv_env.server.app:app`
- Test: `docker-compose config`

**Patch 11: `Dockerfile` (MODIFY)**
- Add `EXPOSE 8000`
- Add alternative CMD for env server mode
- Test: `docker build .`

### After all patches:
```bash
# Full validation
uv run pytest tests/ -q             # All tests green
modal deploy modal_app.py           # Eval backend live
modal run modal_train.py --stage 0  # Smoke test passes
# With openenv-core installed:
uvicorn openenv_env.server.app:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/step -d '{"cuda_code":"__global__ void k(){}"}'
```
