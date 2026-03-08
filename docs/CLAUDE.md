# docs/ — Navigation Hub & Project Quick-Reference

## Active Runtime Posture (March 7, 2026)

- **Hackathon runtime target**: Northflank-managed workloads on CoreWeave-backed GPU infrastructure for remote A100 evaluation. Source: [Northflank GPU workloads](https://northflank.com/docs/v1/application/gpu-workloads/gpus-on-northflank), [CoreWeave on Northflank](https://northflank.com/docs/v1/application/bring-your-own-cloud/coreweave-on-northflank)
- **Training GPU posture**: H200-class for model generation and updates, with H100 fallback if that is what the provisioned node pool exposes. Source: [Northflank GPU workloads](https://northflank.com/docs/v1/application/gpu-workloads/gpus-on-northflank)
- **Eval GPU posture**: A100 80GB only for reward-bearing correctness and timing. Source: [Deploy GPUs in your own cloud](https://northflank.com/docs/v1/application/gpu-workloads/deploy-gpus-in-your-own-cloud)
- **Fallback path**: Modal is preserved as a fallback backend and legacy launcher surface, not the primary hackathon path.

## Active Documents

| File | Role |
|------|------|
| `KERNELFORGE_FINAL_PRD.md` | Main source of truth for architecture, rollout order, and launch criteria |
| `GRPO_DEEP_DIVE.md` | RL math, rollout strategy, TRLOO framing, and implementation guidance |
| `OPENENV_AUDIT_PLAN_3.md` | OpenEnv contract record plus architecture audit |
| `skills/CUDA_AGENT.md` | CUDA-Agent environment and task-format prior |
| `skills/DOUBLEGRAPH_A100.md` | DoubleGraph A100 kernel engineering prior |
| `skills/SKYDISCOVER_ADAEVOLVE_EVOX.md` | AdaEvolve/EvoX search-system prior |
| `skills/KERNELGYM_DR_KERNEL.md` | KernelGYM / Dr. Kernel execution-architecture prior |

## Architecture Summary

The current repo should be understood as three layers:

1. **Custom lightweight OpenEnv wrapper**
   - `openenv_env/`
   - public `step()` / `reset()` / `state()` contract for judges, demos, and TRL/OpenEnv compatibility
   - intentionally lightweight: it wraps the shared evaluator backend rather than trying to reproduce the whole execution plane inside the env package
   - Sources: [OpenEnv docs](https://meta-pytorch.github.io/OpenEnv/), [OpenEnv env authoring guide](https://github.com/meta-pytorch/OpenEnv/blob/main/envs/README.md)

2. **KernelForge task / reward / rollout logic**
   - `training/task_support.py`
   - `training/multi_turn_rollout.py`
   - `openenv_env/reward.py`
   - `training/custom_grpo_trainer.py`

3. **Remote execution backend**
   - `openenv_env/eval_backend.py` — backend switch
   - `eval_service/eval_core.py` — shared pure evaluator core
   - `eval_service/app.py` — Northflank/CoreWeave FastAPI service
   - `modal_app.py` — fallback wrapper around `eval_core`
   - Sources: [Dr. Kernel paper](https://arxiv.org/abs/2602.05885), [KernelGYM repo](https://github.com/hkust-nlp/KernelGYM)

## Current Backend Contract

`openenv_env/eval_backend.py` dispatches evaluation using:

- `KERNELFORGE_EVAL_BACKEND=coreweave` (default)
- `KERNELFORGE_EVAL_URL=https://...` for the Northflank service URL
- `KERNELFORGE_EVAL_BACKEND=modal` for fallback

Stable evaluator result shape used across training, env, and evaluation paths:

- `compiles`
- `correct`
- `runtime_ms`
- `runtime_stats`
- `speedup_vs_orig`
- `speedup_vs_dg`
- `verifier_msg`
- `error`

## GRPO Launch Posture

- **Primary model**: `Qwen3-Coder-30B-A3B-Instruct`
- **Training path**: `training/grpo_train.py`
- **Stage posture**:
  - Stage 1 = warmup
  - Stage 2 = RFT/SFT merge
  - Stage 3 = GRPO pilot
- **Multi-turn default**: 3 turns
- **vLLM**: disabled by default for the hackathon path
- **Reward**: discrete milestone `{-1, 1, 2, 3}` with TRLOO correction

TRL/OpenEnv supports custom rollout functions, which is why the direct rollout path remains acceptable while the custom lightweight OpenEnv wrapper stays the public environment surface. Source: [TRL OpenEnv integration](https://huggingface.co/docs/trl/en/openenv)

## Skills / Priors to Use

Use these together for the hackathon GRPO path:

- `docs/skills/CUDA_AGENT.md`
- `docs/skills/DOUBLEGRAPH_A100.md`
- `docs/skills/SKYDISCOVER_ADAEVOLVE_EVOX.md`
- `docs/skills/KERNELGYM_DR_KERNEL.md`

Operationally:

- **CUDA-Agent** gives the operator-task and environment prior. Source: [CUDA-Agent paper](https://arxiv.org/abs/2602.24286)
- **DoubleGraph** gives A100 graph-kernel engineering priors. Source: [doubleGraph repo](https://github.com/double-ai/doubleGraph)
- **SkyDiscover** gives the search hedge. Source: [SkyDiscover repo](https://github.com/skydiscover-ai/skydiscover)
- **KernelGYM / Dr. Kernel** gives the execution-plane reference. Sources: [Dr. Kernel paper](https://arxiv.org/abs/2602.05885), [KernelGYM repo](https://github.com/hkust-nlp/KernelGYM)

## What Is Functionally Ready in Code

- OpenEnv environment package and HTTP server
- Provider-neutral backend switch
- Shared pure evaluator core
- Northflank/CoreWeave eval service implementation
- Modal fallback wrapper
- GRPO preflight that checks backend health or Modal auth
- Shared reward contract across rollout, env, and evaluation paths
- 114 passing tests reported by the team

## What Still Needs Real Runtime Validation

- Live Northflank/CoreWeave A100 service deployment
- End-to-end Stage 1 run on the active backend
- End-to-end Stage 3 GRPO pilot on the active backend
- Throughput / stability measurements for the remote A100 service

## Environment Redesign (March 8, 2026)

Recent changes to the environment layer:

| Change | File | What |
|--------|------|------|
| Task pool sampling on `reset()` | `openenv_env/task_pool.py` | Replace hardcoded WCC with pool sampling (Ops-6K + doubleGraph) |
| `max_turns=3` | `openenv_env/kernel_forge_env.py` | Down from 200, matches Dr. Kernel MAX_TURN=3 |
| Anti-hack runtime checks | `openenv_env/anti_hack.py` | 5 Dr. Kernel-inspired checks wired into eval_core.py |
| Reference code in observations | `openenv_env/kernel_forge_env.py` | Task prompt + PyTorch reference code + interface contract |
| `_dispatch()` replaces `_modal()` | `openenv_env/kernel_forge_env.py` | Backend-neutral dispatch |
| Dependency inversion fix | `openenv_env/task_routing.py` | Re-exports from training/ so env doesn't import training |
| Benchmark tooling | `scripts/run_benchmark.py`, `scripts/compare_results.py` | KernelBench-compatible fast_p metrics |
| Task pool builder | `tasks/build_task_pool.py` | Curates Ops-6K from HuggingFace → stateless evaluable subset |

**Single-source-of-truth for environment design:** See `docs/SYSTEM_TRUTH.md`.

## Quick File Pointers

- **Single source of truth**: `docs/SYSTEM_TRUTH.md`
- OpenEnv server: `openenv_env/server/app.py`
- OpenEnv client: `openenv_env/client.py`
- Environment: `openenv_env/kernel_forge_env.py`
- Task pool: `openenv_env/task_pool.py`
- Anti-hack: `openenv_env/anti_hack.py`
- Backend switch: `openenv_env/eval_backend.py`
- Shared eval core: `eval_service/eval_core.py`
- CoreWeave/Northflank service: `eval_service/app.py`
- Training launcher: `training/grpo_train.py`
- Rollout loop: `training/multi_turn_rollout.py`
- Task routing + reward contract: `training/task_support.py`
- Task routing re-exports: `openenv_env/task_routing.py`
- Benchmark runner: `scripts/run_benchmark.py`
- Before/after comparison: `scripts/compare_results.py`
- Task pool builder: `tasks/build_task_pool.py`
- SkyDiscover evaluator: `skydiscover_integration/evaluator.py`
- SkyDiscover search: `skydiscover_integration/adaevolve.py`

<claude-mem-context>

</claude-mem-context>