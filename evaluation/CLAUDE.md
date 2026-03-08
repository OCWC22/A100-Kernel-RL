# evaluation/ — Eval, Verification & Profiling

## eval_model.py (219 lines)

### `evaluate_checkpoint(checkpoint_path, num_problems=50) -> dict`
Returns: `compiles`, `correct`, `rewards`, `speedups`, `compile_rate`, `correct_rate`, `avg_reward`, `median_speedup`

### `evaluate_multi_seed(checkpoint_path, num_problems=50, num_seeds=3, temperatures=(0.2, 0.7, 1.0)) -> dict`
Multi-seed evaluation with 95% CI (t-distribution).

### `compare_stages()`
Evaluates Base → Stage 1 → Stage 2 → Stage 3 progression.

Constants: `EVAL_BACKEND` (from `KERNELFORGE_EVAL_BACKEND`, default "coreweave"), `TARGET_GPU="A100"`, `EVAL_PROBLEMS=50`

## compare_stages.py (86 lines)

### `compare_all_stages(num_problems=50) -> list[dict]`

Stages evaluated:
| Label | Path |
|-------|------|
| Base | Qwen/Qwen3.5-35B-A3B |
| Stage 1 (GRPO warm-up) | outputs/kernelforge-stage1 |
| Stage 2 (RFT) | outputs/kernelforge-stage2 |
| Stage 3 (GRPO+curriculum) | outputs/kernelforge-stage3 |

## ablation.py (73 lines)

3 hypotheses:
- **H1**: Multi-stage RL improves over base model → `h1_multistage_improvement()`
- **H2**: RFT is necessary (without RFT → collapse) → `h2_rft_necessity()`
- **H3**: SKILL.md improves over generic prompts → `h3_skill_md_impact()`

## pass_at_k.py (58 lines)

### `pass_at_k(n, c, k) -> float`
Unbiased estimator (Chen et al. 2021): `1 - C(n-c, k) / C(n, k)`
- n = total samples, c = correct samples, k = pass@k

### `pass_at_k_problems(results, k_values=[1, 5, 10]) -> dict`
Mean pass@k across problems. Each result: `{"n": int, "c": int}`.

## reward_monitor.py — UPDATED for discrete rewards

### `check_reward_distribution(rewards) -> dict`
Returns: `distribution`, `total`, `mean`, `flags`, `entropy`, `tier_rates`

Discrete reward tiers {-1, 1, 2, 3}:

| Flag | Condition |
|------|-----------|
| Likely hacking | >90% at tier 3 (beats torch.compile) |
| Bimodal | only -1 and 3 (no intermediate tiers) |
| Model collapsed | uniform rewards |
| Cannot generate | no positive rewards |
| Stuck at tier 1 | >80% correct but not faster |

Tier rates: `fail_rate` (r<0), `correct_rate` (r>=1), `speedup_eager_rate` (r>=2), `speedup_compile_rate` (r>=3)

## verification/pac_verify.py (225 lines)

PAC (Probably Approximately Correct) verification for WCC kernels.

### `generate_test_graphs(num_vertices=10000) -> list[tuple[str, edges, n]]`
5 adversarial graphs: RMAT x2, SBM x2, Erdos-Renyi sparse

### `verify_wcc(kernel_labels, edges, num_vertices) -> tuple[bool, str]`
3 mathematical invariants:
1. Component count matches reference
2. Every edge connects same-label vertices
3. Different reference components get different labels

### `run_kernel_verification(kernel_lib_path, edges, num_vertices) -> dict[int, int]`
Runs compiled .so via ctypes FFI. Expected symbol: `wcc_kernel(const int* row_ptr, const int* col_idx, int num_vertices, int* labels)`

### `edges_to_csr(edges, num_vertices) -> tuple[np.ndarray, np.ndarray]`
Returns `(row_ptr, col_idx)` arrays.

## verification/profile.py (539 lines)

### `H100Profiler(kernel_path)`
- `profile_baseline(graph_sizes=None) -> dict` — cuGraph + NetworkX baselines
- `profile_kernel(warmup_iters=50, benchmark_runs=30, ncu_profile=False, graph_size=10000) -> dict`
- `generate_report(results) -> str`

NCU sections profiled: ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, LaunchStatistics, MemoryTopology, SMActivity

CLI args: `--kernel` (required), `--baseline`, `--ncu`, `--warmup` (100), `--runs` (50), `--graph-size` (10000)

## Eval Backend

### Primary service path

| Component | Role |
|-----------|------|
| `eval_service/eval_core.py` | Shared pure compile / verify / benchmark implementation |
| `eval_service/app.py` | Northflank/CoreWeave FastAPI service exposing the evaluator contract |
| `openenv_env/eval_backend.py` | Dispatch switch: CoreWeave HTTP default, Modal fallback |
| `modal_app.py` | Fallback wrapper around `eval_service.eval_core` |

Returns: `compiles`, `correct`, `speedup_vs_orig`, `speedup_vs_dg`, `runtime_ms`, `runtime_stats`, `verifier_msg`, `error`

## Training launch posture

- Preferred hackathon path: `training/grpo_train.py` with `KERNELFORGE_EVAL_BACKEND=coreweave` and `KERNELFORGE_EVAL_URL` set to the Northflank service URL.
- Fallback path: Modal launchers preserved for later / recovery scenarios.

## Eval Strategy

| Component | Status | Location |
|-----------|--------|----------|
| Local nvcc compile fast-path | **DONE** | `training/multi_turn_rollout.py:_local_compile_check()` |
| Remote eval (WCC) | **DONE in code** | `eval_service/eval_core.py:evaluate_kernel_impl()` via `eval_backend` |
| Remote eval (Ops-6K) | **DONE in code** | `eval_service/eval_core.py:evaluate_ops6k_kernel_impl()` via `eval_backend` |
| Nsight bonus in reward | **API COMPAT** | `openenv_env/reward.py:compute_reward()` — accepted but unused in discrete mode |
| Anti-hack runtime checks | **DONE** | `openenv_env/anti_hack.py` (5 checks wired into `eval_core.py` lines 732-773) |
| ~~CPPO cheap_cuda_score pre-filter~~ | DROPPED | At G=2, pruning leaves G=1 → zero advantage variance |
| Full hybrid turn-escalation | NOT YET | GRPO-10 line 1355 |

## ~~CPPO Heuristic~~ — DROPPED

CPPO completion pruning was planned but **dropped**: after SFT on expert kernels, the model generates syntactically valid CUDA by default. Heuristic checks (checking for `__shared__`, `float4`) provide no filtering value. At G=2, pruning one candidate leaves G=1 which produces zero advantage variance for TRLOO.

## Deep Dive Pointers

- Nsight rewards: `docs/GRPO_DEEP_DIVE.md` line 1276 (GRPO-9)
- Hybrid eval: `docs/GRPO_DEEP_DIVE.md` line 1355 (GRPO-10)
- CPPO pruning: `docs/GRPO_DEEP_DIVE.md` line 1403 (GRPO-11)
