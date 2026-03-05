# docs/ — Navigation Hub & Project Quick-Reference

## Active Documents

| File | Lines | Size | Content |
|------|-------|------|---------|
| `KERNELFORGE_FINAL_PRD.md` | 1,457 | 80KB | Single source of truth — replaces all previous docs |
| `GRPO_DEEP_DIVE.md` | 2,075 | 81KB | Algorithm math, memory budgets, stacked mitigations |

## PRD Section Map (`KERNELFORGE_FINAL_PRD.md`)

| Section | Line | Title |
|---------|------|-------|
| 0 | 10 | Locked Decisions |
| 1 | 27 | What We're Building |
| 2 | 76 | Repository Structure |
| 3 | 148 | Complete Task List (Phases 0-3) |
| 4 | 1075 | Critical Path |
| 5 | 1103 | All Links |
| 6 | 1181 | Risk Matrix & Mitigations |
| 6.0 | 1200 | Fundamental Research Risks |
| 6.1 | 1349 | CUDA Agent Evaluation Pipeline Failures |
| 6.2 | 1361 | doubleGraph Expert Baselines Failures |
| 6.3 | 1373 | SkyDiscover Evolutionary Search Failures |
| 6.4 | 1385 | GRPO Training Failures |
| 6.5 | 1399 | Cross-Component Integration Risks |
| 6.6 | 1410 | Decision Gates (Go/No-Go) |
| 6.7 | 1425 | Realistic Timeline (with failure buffer) |
| 6.8 | 1447 | Minimum Viable Submission |

## GRPO Deep Dive Section Map (`GRPO_DEEP_DIVE.md`)

| Section | Line | Title |
|---------|------|-------|
| GRPO-1 | 335 | The Algorithm (Full Math) |
| GRPO-2 | 495 | B200 192GB Memory Budget (Exact) |
| GRPO-3 | 557 | TRL GRPOTrainer — Exact Configuration |
| GRPO-4 | 1171 | MARS+TRLOO Credit Assignment (Multi-Turn) |
| GRPO-5 | 1401 | Compute Budget on B200 |
| GRPO-6 | 1446 | What To Monitor During Training |
| GRPO-7 | 1504 | Critical Implementation Notes |
| GRPO-8 | 1573 | Quick Reference Card |
| GRPO-9 | 1623 | Nsight Compute Structured Rewards |
| GRPO-10 | 1714 | Hybrid Eval (Local + Modal) |
| GRPO-11 | 1762 | CPPO Completion Pruning |
| GRPO-12 | 1805 | MASPO Soft Trust Region |
| GRPO-13 | 1849 | Transformation Grammar (40 Rules) — DEFERRED TO v2 |
| GRPO-14 | 1978 | Full Stacked Architecture (Single-GPU Hackathon) |

---

## training/ — 3-Stage RL Pipeline

**Model**: Qwen/Qwen3-Coder-Next (80B MoE, 3B active, FP8 on B200 192GB), fallback Qwen/Qwen2.5-Coder-7B-Instruct

> **GPU Split:** B200 = model weights + generation + gradient updates + local compile checks. **A100 (Modal) = all performance reward** (speedup timing, execution correctness, Nsight profiling). B200 timing ≠ A100 timing.

### Stage Configs

| Param | Stage 1 (`stage1_warmup.py`) | Stage 2 (`stage2_rft.py`) | Stage 3 (`stage3_grpo.py`) |
|-------|------------------------------|---------------------------|----------------------------|
| **Type** | GRPO warm-up | RFT (SFT on filtered) | TRLOO-augmented GRPO + curriculum (P3 demo) |
| **LR** | 3e-6 | 5e-6 | 5e-6 |
| **Temperature** | 0.9 | 0.7 (generation) | 0.7 |
| **G (generations)** | 4 | — | 2 |
| **Max turns** | 3 | — | 3 |
| **Steps/Epochs** | 300 steps | 3 epochs | 10 steps (P3 demo) |
| **Batch** | 1 x 4 grad_accum | 1 x 4 grad_accum | 1 x 4 grad_accum |
| **Output** | `outputs/kernelforge-stage1` | `outputs/kernelforge-stage2` | `outputs/kernelforge-stage3` |

Shared: bf16=True, max_prompt=512, max_completion=4096, top_k=50, top_p=0.95, rep_penalty=1.05, vllm colocate

### LoRA (`model_loader.py`)

r=16, alpha=16, dropout=0, max_seq=8192, nf4 4-bit double_quant bf16. MoE targets: q/k/v/o_proj + shared_expert.gate/up/down_proj. Dense targets: q/k/v/o_proj + gate/up/down_proj.

### Curriculum (`curriculum.py`)

| Phase | Name | Target Reward | Problems |
|-------|------|---------------|----------|
| 0 | `single_ops` | 1.0 | vector_add, relu, softmax, matmul, gelu |
| 1 | `fusion_2op` | 2.0 | LayerNorm+ReLU, MatMul+BiasAdd, Softmax+Dropout, Conv2d+BatchNorm |
| 2 | `arch_specific` | 2.0 | WCC, reduce_sum, GEMM, attention |
| 3 | `advanced` | 3.0 | LayerNorm+GELU+Linear, Flash-Attention, Batched SpMV |

Window=10, promote >50%, demote <20%

### Multi-Turn Rollout (`multi_turn_rollout.py`)

`make_multi_turn_rollout(max_turns=3, skill_md_gpu=None) -> Callable` — TRL-compatible rollout_func. B200 generates, Modal A100 evaluates (5 graphs, 50 warmup, 30 runs). 2-tier reward: fast path (CUDA events + correctness) for all, slow path (Nsight ncu) for top-k. Early exit at reward>=1.6 (log(5.0), i.e. 5x+ speedup). Single-turn wrapper: `reward_from_env(prompts, completions, **kwargs) -> list[float]`.

### RFT Filter (`rft_filter.py`)

`TrajectoryCollector(modal_app_name, model_path)` — `collect_trajectories(n=100)`, `filter_trajectories(min_reward=1.0)`, `save_rft_dataset()`. Generation: max_new_tokens=2048, T=0.7, top_p=0.95.

### CUDA Agent (`cuda_agent_integration.py`)

Dataset: `BytedTsinghua-SIA/CUDA-Agent-Ops-6K`. `load_cuda_agent_prompt_dataset(max_samples=1024, seed=42)`, `load_cuda_agent_prompt_texts(max_samples=128)`.

### Stacked Mitigations (NOT YET IMPLEMENTED)

| Technique | Deep Dive Ref | File to Create |
|-----------|---------------|----------------|
| MARS+TRLOO credit assignment | GRPO-4 line 1171 | `training/custom_grpo_loop.py` |
| Nsight structured rewards | GRPO-9 line 1623 | `training/hybrid_rollout.py` |
| CPPO completion pruning | GRPO-11 line 1762 | (in custom_grpo_loop.py) |
| MASPO soft trust region | GRPO-12 line 1805 | `training/maspo_loss.py` |
| ~~Transformation grammar~~ | GRPO-13 line 1849 | DEFERRED to v2 — use CUDA-Agent SKILL.md + doubleGraph patterns instead |

### Abort Conditions

reward=-1 after 30 steps (dataset issue), std=0 (collapsed), NaN loss (LR too high), OOM (reduce G/batch/seq), Gate G-0.8 failure (non-zero gradients + >1.2x on 2/20 kernels) → abandon GRPO, use SFT + SkyDiscover only

---

## openenv_env/ — OpenEnv Environment

**Protocol**: HTTP client-server (NOT Gymnasium). `uv add "openenv-core[core]>=0.2.1"`

### KernelForgeEnv (`kernel_forge_env.py`)

```
KernelForgeAction(Action):   cuda_code: str
KernelForgeObservation(Observation): text, baseline_original_ms, baseline_doublegraph_ms, hardware, turn, best_reward, info
```

`reset(seed, episode_id) -> Observation` / `step(action, timeout_s) -> Observation` — dispatches to Modal evaluate_kernel (5 graphs, 50 warmup, 30 runs)

### Reward (`reward.py`)

`validate_eval_result(result) -> dict` — asserts required Modal keys (compiles, correct, speedup_vs_orig, speedup_vs_dg, error), clamps NaN/inf, returns safe defaults on missing keys.

`compute_reward(compiled, correct, speedup_vs_eager, speedup_vs_compile, occupancy=None, mem_coalescing=None, warp_efficiency=None) -> float`

| Return | Condition |
|--------|-----------|
| -1.0 | not compiled OR not execution-correct (compile-only is NOT sufficient per CUDABench) |
| log(speedup) | fast path: CUDA events timing on A100 + execution correctness |
| log(speedup) + nsight_bonus | slow path (top-k): + 0.4*occ + 0.3*mem + 0.2*warp from Nsight ncu |

Plus `trloo_post_process(advantages, n)` for N/(N-1) gradient correction (apply if Dr. Kernel derivation confirms; else vanilla GRPO with tight gates).

### Anti-Hack (`anti_hack.py`)

Allowed CU_FLAGS: `--use_fast_math`, `--extra-device-vectorization`, `--rdc=true`, `--maxrregcount=*`. Forbidden symbols: torch, at::Tensor, c10::, torch::autograd, triton, torch.compile, torch.nn.functional.

### GPU Registry (`gpu_registry.py`)

| Key | A100 | H100 | B200 |
|-----|------|------|------|
| arch | sm_80 | sm_90a | sm_100a |
| SMs | 108 | 132 | 192 |
| L2 | 40 MB | 50 MB | 96 MB |
| SMEM/SM | 164 KB | 228 KB | 228 KB |
| HBM BW | 2.0 TB/s | 3.35 TB/s | 8.0 TB/s |
| TMA | no | yes | yes |

### Skill Builder (`skill_builder.py`)

`build_skill_md(gpu="a100")` priority: env var `KERNELFORGE_SKILL_FILE` > static `skill_{gpu}.md` > dynamic from registry

### CachePool (`cache_pool.py`)

`GPUCachePool(max_entries=8)` — LRU. `get_or_create(key, factory, metadata)`, `get(key)`, `clear()`. Eviction calls close()/release().

### NOT YET IMPLEMENTED

- ~~`transform_grammar.py`~~ — DEFERRED to v2
- `reward_nsight.py` — Nsight metrics integration (occupancy, coalescing, warp efficiency) into continuous reward

---

## evaluation/ — Eval, Verification & Profiling

### eval_model.py (219 lines)

`evaluate_checkpoint(path, n=50) -> dict` — compile_rate, correct_rate, avg_reward, median_speedup. `evaluate_multi_seed(path, n=50, seeds=3, temps=(0.2,0.7,1.0))` — with 95% CI.

### compare_stages.py (86 lines)

`compare_all_stages(n=50)` — Base vs Stage 1-3 progression with reward deltas.

### ablation.py (73 lines)

H1: multi-stage > base. H2: RFT necessary (skip → collapse). H3: SKILL.md > generic prompts.

### pass_at_k.py (58 lines)

`pass_at_k(n, c, k) -> float` — unbiased estimator `1 - C(n-c,k)/C(n,k)`. `pass_at_k_problems(results, k=[1,5,10])`.

### reward_monitor.py (72 lines)

`check_reward_distribution(rewards) -> dict` — flags: hacking (>90% max), bimodal, collapsed, no positive, no speedup.

### verification/pac_verify.py (225 lines)

`generate_test_graphs(n=10000)` — 5 adversarial (RMAT x2, SBM x2, ER sparse). `verify_wcc(labels, edges, n)` — 3 invariants (count, edge consistency, label uniqueness). `run_kernel_verification(lib_path, edges, n)` — ctypes FFI, expects `wcc_kernel(row_ptr, col_idx, n, labels)`.

### verification/profile.py (539 lines)

`H100Profiler(kernel_path)` — `profile_baseline()`, `profile_kernel(warmup=50, runs=30, ncu=False, size=10000)`, `generate_report()`. NCU sections: Compute/Memory WorkloadAnalysis, LaunchStats, MemoryTopology, SMActivity.

### modal_app.py (457 lines, project root)

| Function | Timeout | Purpose |
|----------|---------|---------|
| `evaluate_kernel(payload)` | 120s | Compile + PAC verify + benchmark |
| `profile_baselines()` | 300s | Original + doublegraph timing |
| `test_h100_features()` | 60s | GPU feature detection |
| `evaluate_kernels_batch(payloads)` | 600s | Batch evaluation |

### Hybrid Eval Strategy (GRPO-10 line 1714)

Turn 1-2: local nvcc only. Turn 3: local + PAC verify. Turn 4+: Modal + ncu.

### NOT YET IMPLEMENTED

- Nsight metrics integration into Modal eval endpoint (occupancy, coalescing, warp efficiency passed to continuous reward)
- CPPO `cheap_cuda_score()` pre-filter (GRPO-11 line 1762): +1 for __global__, +1 __shared__, +1 threadIdx/blockIdx, +2 local nvcc compile; skip Modal if below threshold

---

## Skill Docs (`skills/`)

| File | Lines | Content |
|------|-------|---------|
| `doublegraph_a100.md` | 325 | DoubleGraph A100 engineering & replication guide |

Key sections: Sec 6 (line 148) kernel deep dive, Sec 10 (line 292) H100/B200 porting, Sec 11 (line 318) transferable strategies. Also see `/skill_a100.md` (root, 72 lines).

---

## Archive

All previous versions in `archive/`. Never reference for current decisions.


<claude-mem-context>

</claude-mem-context>