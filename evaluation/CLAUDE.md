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
| Base | Qwen/Qwen2.5-Coder-7B-Instruct |
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

## reward_monitor.py (72 lines)

### `check_reward_distribution(rewards) -> dict`
Returns: `distribution`, `total`, `mean`, `flags`, `entropy`, `tier_rates`

**Suspicious flags**:
| Flag | Condition |
|------|-----------|
| Likely hacking | >90% at max reward (3.0) |
| Bimodal/overfitting | only -1 and +3 |
| Model collapsed | uniform rewards |
| Cannot generate | no positive rewards |
| No speedup | max reward <= 1.0 |

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

## modal_app.py (457 lines, project root)

### Modal Functions (GPU=A100, app="kernelforge-a100")

| Function | Timeout | Description |
|----------|---------|-------------|
| `evaluate_kernel(payload)` | 120s | Compile + PAC verify + benchmark |
| `profile_baselines()` | 300s | Baseline timing (original + doublegraph) |
| `test_h100_features()` | 60s | GPU feature detection |
| `evaluate_kernels_batch(payloads)` | 600s | Batch evaluation |

`evaluate_kernel` payload: `cuda_code`, `verify_graphs=5`, `warmup_iters=50`, `benchmark_runs=30`, `baseline_original_ms`, `baseline_doublegraph_ms`

Returns: `compiles`, `correct`, `verifier_msg`, `runtime_ms` (median), `speedup_vs_orig`, `speedup_vs_dg`, `error`

## CPPO Heuristic (NOT YET IMPLEMENTED)

Planned: `cheap_cuda_score()` for fast pre-filtering completions before expensive Modal eval.
See GRPO-11, `GRPO_DEEP_DIVE.md` line 1403.

Scoring rules (from deep dive):
- Has `__global__` keyword → +1
- Has `__shared__` memory → +1
- Has `threadIdx`/`blockIdx` → +1
- Compiles locally with nvcc → +2
- Score < threshold → skip Modal eval

## Nsight Reward Formula (NOT YET IMPLEMENTED)

See GRPO-9, `GRPO_DEEP_DIVE.md` line 1276.
- Extract: `sm__throughput.avg.pct_of_peak_sustained_elapsed`, `dram__throughput`, occupancy
- Bonus = weighted combination [0, 0.5] added to discrete reward
- **File to create**: `reward_nsight.py`

## Hybrid Eval Strategy (from GRPO-10, line 1355)

Turn-by-turn escalation:
1. **Turn 1-2**: Local nvcc compile check only (fast, free)
2. **Turn 3**: Local + PAC verification (5 test graphs)
3. **Turn 4+**: Modal + ncu profiling (expensive, accurate)

## Files to Create

- [ ] `reward_nsight.py` — Nsight Compute continuous bonus reward

## Deep Dive Pointers

- Nsight rewards: `docs/GRPO_DEEP_DIVE.md` line 1276 (GRPO-9)
- Hybrid eval: `docs/GRPO_DEEP_DIVE.md` line 1355 (GRPO-10)
- CPPO pruning: `docs/GRPO_DEEP_DIVE.md` line 1403 (GRPO-11)
