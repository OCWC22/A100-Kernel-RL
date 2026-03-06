# evaluation/ — Eval, Verification & Profiling

## eval_model.py (219 lines)

### `evaluate_checkpoint(checkpoint_path, num_problems=50) -> dict`
Returns: `compiles`, `correct`, `rewards`, `speedups`, `compile_rate`, `correct_rate`, `avg_reward`, `median_speedup`

### `evaluate_multi_seed(checkpoint_path, num_problems=50, num_seeds=3, temperatures=(0.2, 0.7, 1.0)) -> dict`
Multi-seed evaluation with 95% CI (t-distribution).

### `compare_stages()`
Evaluates Base → Stage 1 → Stage 2 → Stage 3 progression.

Constants: `MODAL_APP_NAME="kernelforge-a100"`, `TARGET_GPU="A100"`, `EVAL_PROBLEMS=50`

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

## modal_app.py (project root)

### Modal Functions (GPU=A100, app="kernelforge-a100")

| Function | Timeout | Description | Status |
|----------|---------|-------------|--------|
| `evaluate_kernel(payload)` | 120s | Compile + PAC verify + benchmark (WCC) | Existing |
| `evaluate_ops6k_kernel(payload)` | 120s | Compile + verify + benchmark (Ops-6K) | **NEW** |
| `profile_baselines()` | 300s | Original + doublegraph timing | Existing |
| `test_h100_features()` | 60s | GPU feature detection | Existing |
| `evaluate_kernels_batch(payloads)` | 600s | Batch evaluation | Existing |

`evaluate_ops6k_kernel` payload: `cuda_code`, `task_code` (reference PyTorch model), `warmup_iters=10`, `benchmark_runs=10`

Returns: `compiles`, `correct`, `speedup_vs_orig`, `speedup_vs_dg`, `runtime_ms`, `error`

## modal_train.py (project root) — NEW

Modal training app. Runs 3-stage pipeline on H100 ($3.95/hr, 80GB).
- `modal run modal_train.py --stage 0` — smoke test
- `modal run modal_train.py --stage 1 --max-steps 10` — Stage 1 warmup
- `modal run modal_train.py --stage 3` — Stage 3 GRPO demo

## Eval Strategy

| Component | Status | Location |
|-----------|--------|----------|
| Local nvcc compile fast-path | **DONE** | `training/multi_turn_rollout.py:_local_compile_check()` |
| Modal eval (WCC) | **DONE** | `modal_app.py:evaluate_kernel()` |
| Modal eval (Ops-6K) | **DONE** | `modal_app.py:evaluate_ops6k_kernel()` |
| Nsight bonus in reward | **DONE** | `openenv_env/reward.py:compute_reward()` |
| CPPO cheap_cuda_score pre-filter | NOT YET | GRPO-11 line 1762 |
| Full hybrid turn-escalation | NOT YET | GRPO-10 line 1355 |

## ~~CPPO Heuristic~~ — DROPPED

CPPO completion pruning was planned but **dropped**: after SFT on expert kernels, the model generates syntactically valid CUDA by default. Heuristic checks (checking for `__shared__`, `float4`) provide no filtering value. At G=2, pruning one candidate leaves G=1 which produces zero advantage variance for TRLOO.

## Deep Dive Pointers

- Nsight rewards: `docs/GRPO_DEEP_DIVE.md` line 1276 (GRPO-9)
- Hybrid eval: `docs/GRPO_DEEP_DIVE.md` line 1355 (GRPO-10)
- CPPO pruning: `docs/GRPO_DEEP_DIVE.md` line 1403 (GRPO-11)
