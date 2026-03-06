# docs/ — Navigation Hub & Project Quick-Reference

## Modal GPU Pricing (as of March 5, 2026)

Per-second billing, no minimum. $30/month free credits on Starter plan.

| GPU | VRAM | Arch | $/second | $/hour | Use Case |
|-----|------|------|----------|--------|----------|
| H200 | 141 GB HBM3e | sm_90 | $0.001261 | $4.54 | Training (alternative — fits but tight, ~41GB free) |
| B200 | 192 GB HBM3e | sm_100 | $0.001736 | $6.25 | Future scale-up (Qwen3-Coder-Next 80B FP8) |
| **H100** | 80 GB HBM3 | sm_90a | $0.001097 | **$3.95** | **Training** (primary — Qwen3-Coder-30B-A3B-Instruct fits with ~46-50GB free) |
| RTX PRO 6000 | 48 GB GDDR7 | ada | $0.000842 | $3.03 | Inference |
| **A100 80GB** | 80 GB HBM2e | sm_80 | $0.000694 | **$2.50** | **Eval/reward** (target GPU) |
| A100 40GB | 40 GB HBM2e | sm_80 | $0.000583 | $2.10 | Eval (budget) |
| L40S | 48 GB GDDR6 | ada | $0.000542 | $1.95 | Inference |
| A10 | 24 GB GDDR6 | sm_86 | $0.000306 | $1.10 | Light inference |
| L4 | 24 GB GDDR6 | ada | $0.000222 | $0.80 | Light inference |
| T4 | 16 GB GDDR6 | sm_75 | $0.000164 | $0.59 | Cheapest |

### KernelForge Cost Estimate (Hackathon Config)

| Component | GPU | Hours | $/hr | Cost |
|-----------|-----|-------|------|------|
| SFT warmup | H100 | 2.0 | $3.95 | $7.90 |
| Stage 1 warmup (100 steps) | H100 | 3.5 | $3.95 | $13.83 |
| Stage 2 RFT | H100 | 0.5 | $3.95 | $1.98 |
| Stage 3 GRPO pilot (50 steps) | H100 | 2.0 | $3.95 | $7.90 |
| Eval calls (~400 calls) | A100 80GB | 4.0 | $2.50 | $10.00 |
| Search / best-of-N + SkyDiscover | A100 80GB | 3.0 | $2.50 | $7.50 |
| Testing/debugging | H100 | 3.0 | $3.95 | $11.85 |
| **Total** | | **~18** | | **~$61-100** |
| **Budget ceiling** | | | | **$200** |

Source: [modal.com/pricing](https://modal.com/pricing)

---

## Active Documents

| File | Lines | Size | Content |
|------|-------|------|---------|
| `KERNELFORGE_FINAL_PRD.md` | ~1,700 | ~90KB | Hackathon PRD — single source of truth (hackathon-first framing, H100+A100, Qwen3-Coder-30B-A3B-Instruct) |
| `GRPO_DEEP_DIVE.md` | ~2,400 | ~95KB | Algorithm math, memory budgets, hackathon config (GRPO-15 split: hackathon + future scale-up) |

## PRD Section Map (`KERNELFORGE_FINAL_PRD.md`)

| Section | Title |
|---------|-------|
| 0 | Executive Summary + RL Feasibility + Locked Decisions |
| 1 | Strategic Framing (hackathon-first + long-term + what we're NOT claiming) |
| 2 | What We're Building (2.1 Hackathon Deliverable, 2.2 Long-Term Research Platform) |
| 3 | Scope Boundaries (hackathon scope + explicitly out of scope) |
| 4 | Training Strategy (Phase 0-3: env validation → SFT → GRPO pilot → search hedge) |
| 5 | Hackathon Configuration (model, hardware, RL posture, abort conditions) |
| 6 | Compute Budget (<$200) |
| 7 | Claim Discipline (Implemented / Tested / Benchmarked / Projected) |
| 8 | Repository Structure |
| 9 | Complete Task List (Phases 0-3) |
| 10 | Critical Path |
| 11 | All Links |
| 12 | Risk Matrix & Mitigations |
| 12.0-12.8 | Risk categories + Decision Gates + Timeline + Min Viable Submission |
| 12.9 | Deliverables for Judges |
| 12.10 | Bottom Line |
| 13 | Future Work: Scale-Up Feasibility Assessment (B200/Coder-Next — speculative) |

## GRPO Deep Dive Section Map (`GRPO_DEEP_DIVE.md`)

| Section | Title |
|---------|-------|
| Top | Core Recommendation (hackathon path + what we need to prove) |
| GRPO-1 | The Algorithm (Full Math) |
| GRPO-2 | Memory Budget (H100 primary, B200 future) |
| GRPO-3 | TRL GRPOTrainer — Exact Configuration |
| GRPO-4 | MARS+TRLOO Credit Assignment (Multi-Turn) — MARS is DROPPED |
| GRPO-5 | Compute Budget (Hackathon: H100 + A100) |
| GRPO-6 | What To Monitor During Training |
| GRPO-7 | Critical Implementation Notes |
| GRPO-8 | Quick Reference Card (hackathon config) |
| GRPO-9 | Nsight Compute Structured Rewards |
| GRPO-10 | Hybrid Eval (Local + Modal) |
| GRPO-11 | CPPO Completion Pruning — DROPPED |
| GRPO-12 | MASPO Soft Trust Region — DEFERRED (future research) |
| GRPO-13 | Transformation Grammar (40 Rules) — DEFERRED TO v2 |
| GRPO-14 | Full Stacked Architecture (future targets, not hackathon claims) |
| GRPO-15 | Hackathon Config (15.1) + Future Scale-Up Target (15.2) |
| GRPO-16 | Presentation Framing |
| GRPO-17 | Abort Gates (Consolidated) |

---

## training/ — 3-Stage RL Pipeline

**Model**: Qwen/Qwen3-Coder-30B-A3B-Instruct (30.5B MoE, 3.3B active, 4-bit on H100 80GB), fallback Qwen/Qwen3.5-35B-A3B

> **GPU Split:** **H100** ($3.95/hr) = model weights + generation + gradient updates + local compile checks. **A100 (Modal) = all performance reward** (speedup timing, execution correctness, Nsight profiling). H100 timing ≠ A100 timing.

### Stage Configs

| Param | Stage 1 (`stage1_warmup.py`) | Stage 2 (`stage2_rft.py`) | Stage 3 (`stage3_grpo.py`) |
|-------|------------------------------|---------------------------|----------------------------|
| **Type** | GRPO warm-up | RFT (SFT on filtered) | TRLOO-augmented GRPO pilot **(hackathon config per GRPO-15.1)** |
| **Trainer** | `TRLOOGRPOTrainer` | SFTTrainer | `TRLOOGRPOTrainer` |
| **LR** | 2e-6 | 5e-6 | 3e-6 |
| **Temperature** | 1.0 | 0.7 (generation) | 0.7 |
| **G (generations)** | 2 | — | 2 |
| **Max turns** | 3 | — | **3-5** (hackathon pilot; 20 is future target) |
| **Steps/Epochs** | 100 steps | 3 epochs | **50 steps** (hackathon pilot; 150 is future target) |
| **Context** | 8,192 | — | **8,192** (short context, budget-gated) |
| **Batch** | 1 x 4 grad_accum | 1 x 4 grad_accum | 1 x 4 grad_accum |
| **Output** | `outputs/kernelforge-stage1` | `outputs/kernelforge-stage2` | `outputs/kernelforge-stage3` |

Shared: bf16=True, max_prompt=512, top_k=50, top_p=0.95, rep_penalty=1.05, vllm colocate

### TRLOO Custom Trainer (`custom_grpo_trainer.py`) — IMPLEMENTED

`TRLOOGRPOTrainer(GRPOTrainer)` — drop-in replacement that applies N/(N-1) TRLOO advantage scaling after parent computes vanilla GRPO advantages. Fixes the 50% gradient shrinkage at G=2 from Dr. Kernel (arXiv 2602.05885). Used by both Stage 1 and Stage 3.

### LoRA (`model_loader.py`)

r=16, alpha=16, dropout=0, max_seq=8192, nf4 4-bit double_quant bf16. MoE targets: q/k/v/o_proj + shared_expert.gate/up/down_proj. Dense targets: q/k/v/o_proj + gate/up/down_proj.

### Curriculum (`curriculum.py`)

| Phase | Name | Target Reward | Problems |
|-------|------|---------------|----------|
| 0 | `single_ops` | 1.0 | vector_add, relu, softmax, matmul, gelu |
| 1 | `fusion_2op` | 2.0 | LayerNorm+ReLU, MatMul+BiasAdd, Softmax+Dropout, Conv2d+BatchNorm |
| 2 | `arch_specific` | 2.0 | WCC, reduce_sum, GEMM, attention, **+ BFS (power-law), PageRank (dense-regular), WCC (sparse-islands)** |
| 3 | `advanced` | 3.0 | LayerNorm+GELU+Linear, Flash-Attention, Batched SpMV, **+ Triangle Count (dense-community), Louvain (dense-community)** |

Window=10, promote >50%, demote <20%. Phase 2-3 include topology-aware graph problems from doubleGraph with specific graph structure descriptions.

### Multi-Turn Rollout (`multi_turn_rollout.py`)

`make_multi_turn_rollout(max_turns=3, skill_md_gpu=None) -> Callable` — TRL-compatible rollout_func. Stage 3 hackathon uses `max_turns=3-5` per GRPO-15.1.

Flow: generate → `extract_cuda_code()` → `_local_compile_check()` (nvcc syntax, **IMPLEMENTED**) → `_evaluate_on_modal()` (supports both WCC via `evaluate_kernel` and Ops-6K via `evaluate_ops6k_kernel`) → `_compute_reward_from_result()` → `_format_feedback()`. Early exit at reward >= 3.0 (beats torch.compile).

Local compile fast-path: `nvcc -arch=sm_80 -c` syntax check before Modal. Fails ~50% of early kernels for free, saves A100 eval cost.

Single-turn wrapper: `reward_from_env(completions, **kwargs) -> list[float]`.

### RFT Filter (`rft_filter.py`)

`TrajectoryCollector(modal_app_name, model_path)` — `collect_trajectories(n=100)`, `filter_trajectories(min_reward=0.0)` (revised per GRPO-15: any speedup is useful signal), `save_rft_dataset()`. Generation: max_new_tokens=2048, T=0.7, top_p=0.95.

### CUDA Agent (`cuda_agent_integration.py`)

Dataset: `BytedTsinghua-SIA/CUDA-Agent-Ops-6K`. `load_cuda_agent_prompt_dataset(max_samples=1024, seed=42)` — returns Dataset with columns: `prompt`, `task_code`, `ops`, `data_source`. The `task_code` column carries the reference PyTorch model code needed by `evaluate_ops6k_kernel`.

### Combined Dataset Pipeline (`datasets/build_combined_dataset.py`)

Merges:
- `docs/research/doublegraph/doublegraph_a100_manifest.jsonl` (192 doubleGraph A100 kernels, metadata-first)
- `BytedTsinghua-SIA/CUDA-Agent-Ops-6K` (~6,000 ops tasks)

Unified row schema:
- `prompt`, `ops`, `difficulty`, `data_source`
- optional: `task_code`, `topology`, `graph_properties`, `kernel_id`, `compile_flags`, `expert_code`

Output artifact:
- `datasets/combined_kernelforge.jsonl` (~6,192 rows)

Curriculum integration helper:
- `inject_into_curriculum(cm, dataset)` routes by difficulty:
  - 1 → `single_ops`
  - 2 → `fusion_2op`
  - 3 → `arch_specific`
  - 4 → `advanced`

### Stage-Aware Loader (`training/dataset_loader.py`)

`load_training_dataset(stage, ...)`:
- `stage1` → prompt `Dataset` (easy Ops-6K + doubleGraph base variants)
- `stage2` → `doublegraph_sft.jsonl` rows
- `stage3` → full combined rows (+ optional curriculum injection)

`training/stage1_warmup.py` and `training/stage3_grpo.py` now use this loader path.

### doubleGraph A100 Dataset (`datasets/`)

192 production A100-optimized CUDA kernels. Source of truth: `docs/research/doublegraph/doublegraph_a100_manifest.jsonl`.

| File | Format | Entries | Use |
|------|--------|---------|-----|
| `combined_kernelforge.jsonl` | Unified (prompt + ops + difficulty + topology) | ~6,192 | Stages 1 & 3 (via curriculum) |
| `doublegraph_sft.jsonl` | HF messages (system/user/assistant) | 192 | Stage 2 SFT |

Categories: traversal(20), components(6), link_analysis(32), community(42), centrality(32), link_prediction(48), cores(8), tree(4).
Topologies: power-law(20), sparse-islands(18), dense-regular(64), dense-community(42), bipartite(48).

Legacy files archived to `archive/datasets_legacy/`.

### Evolutionary Search — AdaEvolve + EvoX (`skydiscover_integration/`)

Implements SkyDiscover's two algorithms natively (no external dependency):

**Evaluator bridge** (`evaluator.py`): `KernelForgeEvaluator` — cascade eval on Modal A100.
- `evaluate_stage1(code)`: local nvcc compile check (fast, ~50% filter)
- `evaluate_stage2(code)`: full Modal A100 benchmark (accurate timing + correctness)
- `evaluate_program(code, id)`: async cascade entry point
- `evaluate(path)`: sync file-based interface

**AdaEvolve** (`adaevolve.py`): Multi-island evolutionary search with UCB scheduling.
- 4 islands: register_pressure, memory_coalescing, warp_divergence, occupancy_tuning
- UCB1 selects which island gets eval budget
- Paradigm breakthroughs (2x+ improvement) broadcast to all islands

**EvoX** (`evox_strategies.py`): Self-evolving mutation strategies.
- `LogWindowScorer`: tracks improvement velocity per strategy
- Stagnation detection → strategy evolution (hybrid strategies from top performers)
- Hot-swappable strategy database with 4 pre-defined hybrid templates

Seed kernels in `initial_kernels/`: 5 A100 graph kernels (wcc, bfs, pagerank, triangle_count, louvain).

Launch: `./skydiscover_integration/run_evolution.sh` (uses AdaEvolve, not external SkyDiscover).

### Modal Training (`modal_train.py`, project root) — IMPLEMENTED

Runs stages on Modal cloud GPU. Default: H100 ($3.95/hr, 80GB). Image: CUDA 12.4 + torch + trl[vllm]==0.29.0 + transformers + peft + flash-attn.

- `modal run modal_train.py --stage 0` — smoke test (model load, dataset, generation, eval endpoint)
- `modal run modal_train.py --stage 1 --max-steps 10` — Stage 1 warmup
- `modal run modal_train.py --stage 3` — Stage 3 GRPO demo

### Stacked Mitigations

| Technique | Status | Location |
|-----------|--------|----------|
| TRLOO advantage scaling (N/(N-1)) | **DONE** | `custom_grpo_trainer.py` |
| Local compile fast-path | **DONE** | `multi_turn_rollout.py:_local_compile_check()` |
| Ops-6K evaluation | **DONE** | `modal_app.py:evaluate_ops6k_kernel()` |
| Discrete reward {-1,1,2,3} | **DONE** | `reward.py:compute_reward()` |
| Nsight lightweight profiling | **DONE** | `modal_app.py:evaluate_kernel()` (ptxas occupancy + warp info) |
| doubleGraph A100 expert dataset | **DONE** | `datasets/doublegraph_*.jsonl` (192 entries) |
| Real A100 patterns in SKILL.md | **DONE** | `skill_builder.py:_append_a100_patterns()` (7 patterns) |
| Topology-aware curriculum | **DONE** | `curriculum.py` Phase 2-3 (5 graph problems with graph_properties) |
| Graph topology in RL observations | **DONE** | `kernel_forge_env.py`, `curriculum.py`, `multi_turn_rollout.py` |
| Evaluator bridge (cascade eval) | **DONE** | `skydiscover_integration/evaluator.py` |
| AdaEvolve multi-island search | **DONE** | `skydiscover_integration/adaevolve.py` (UCB + breakthrough) |
| EvoX self-evolving strategies | **DONE** | `skydiscover_integration/evox_strategies.py` (LogWindowScorer) |
| MARS return-to-go credit | **DROPPED** (degenerates with outcome rewards) | — (GRPO-4) |
| CPPO completion pruning | **DROPPED** (harmful for exploration at G=2) | — (GRPO-11) |
| MASPO soft trust region | DEFERRED (future research) | — (GRPO-12) |
| ~~Transformation grammar~~ | DEFERRED | v2 — use CUDA-Agent SKILL.md instead (GRPO-13) |

### Abort Conditions

reward=-1 after 30 steps (dataset issue), std=0 (collapsed), NaN loss (LR too high), OOM (reduce G/batch/seq), Gate G-0.8 failure (non-zero gradients + >1.2x on 2/20 kernels) → abandon GRPO, use SFT + AdaEvolve only

---

## openenv_env/ — OpenEnv Environment

**Protocol**: HTTP client-server (NOT Gymnasium). `uv add "openenv-core[core]>=0.2.1"`

### KernelForgeEnv (`kernel_forge_env.py`)

```
KernelForgeAction(Action):   cuda_code: str
KernelForgeObservation(Observation): text, baseline_original_ms, baseline_doublegraph_ms, hardware, turn, best_reward, info, graph_properties, topology_type
```

`reset(seed, episode_id) -> Observation` / `step(action, timeout_s) -> Observation` — dispatches to Modal evaluate_kernel (5 graphs, 50 warmup, 30 runs)

### Reward (`reward.py`)

`validate_eval_result(result) -> dict` — asserts required Modal keys (compiles, correct, speedup_vs_orig, speedup_vs_dg, error), clamps NaN/inf, returns safe defaults on missing keys.

`compute_reward(compiled, correct, speedup_vs_eager, speedup_vs_compile, occupancy=None, mem_coalescing=None, warp_efficiency=None) -> float`

| Return | Condition |
|--------|-----------|
| -1.0 | not compiled OR not execution-correct (compile-only is NOT sufficient per CUDABench) |
| 1.0 | correct but not faster than baselines |
| 2.0 | correct and faster than eager PyTorch (>5%) |
| 3.0 | correct and faster than torch.compile (>5%) |

`trloo_post_process(advantages, n)` — N/(N-1) gradient correction. **Applied** via `TRLOOGRPOTrainer._compute_advantages()` in Stages 1 & 3.

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

`build_skill_md(gpu="a100")` priority: env var `KERNELFORGE_SKILL_FILE` > static `skill_{gpu}.md` > dynamic from registry.

For A100, appends 7 real expert patterns via `_append_a100_patterns()`:
1. Degree-based kernel dispatch (warp-cooperative vs thread-per-vertex)
2. Warp-level shared-memory hash tables (atomicCAS per-slot)
3. Zero-copy convergence flag (cudaHostAlloc + cudaHostGetDevicePointer)
4. Bitmap frontier for BFS / direction-optimizing traversal
5. Path-halving Union-Find (lock-free, no atomics)
6. Compilation flag tuning by algorithm class (maxrregcount, rdc, dlcm)
7. __launch_bounds__ for register control

Output: 178 lines / 7.5KB for A100 (vs 55 lines for H100/B200).

### CachePool (`cache_pool.py`)

`GPUCachePool(max_entries=8)` — LRU. `get_or_create(key, factory, metadata)`, `get(key)`, `clear()`. Eviction calls close()/release().

### NOT YET IMPLEMENTED

- ~~`transform_grammar.py`~~ — DEFERRED to v2
- ~~`reward_nsight.py`~~ — Nsight bonus already in `compute_reward()` as optional kwargs; separate file not needed

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

### modal_app.py (project root)

| Function | Timeout | Purpose |
|----------|---------|---------|
| `evaluate_kernel(payload)` | 120s | Compile + PAC verify + benchmark (WCC) |
| `evaluate_ops6k_kernel(payload)` | 120s | **NEW** — Compile + verify + benchmark (Ops-6K tasks) |
| `profile_baselines()` | 300s | Original + doublegraph timing |
| `test_h100_features()` | 60s | GPU feature detection |
| `evaluate_kernels_batch(payloads)` | 600s | Batch evaluation |

`evaluate_ops6k_kernel` receives `cuda_code` + `task_code` (reference PyTorch model from Ops-6K), compiles via `torch.utils.cpp_extension.load()`, verifies correctness with `atol/rtol=1e-2`, profiles baseline vs torch.compile vs CUDA extension.

### modal_train.py (project root) — IMPLEMENTED

Modal training app. Runs 3-stage pipeline on H100 ($3.95/hr, 80GB). Smoke test, dry run, and full training modes.

### Eval Strategy

- **Local fast-path** (IMPLEMENTED): `nvcc -arch=sm_80 -c` syntax check in `multi_turn_rollout.py` before Modal eval. Fails non-compiling kernels for free.
- **Modal eval**: Full compile + correctness verify + benchmark on A100.
- **Hybrid escalation** (NOT YET): Turn-based ncu escalation (GRPO-10).

### DROPPED / DEFERRED

- ~~CPPO `cheap_cuda_score()` pre-filter~~ — **DROPPED**: harmful for exploration at G=2; after SFT, heuristic checks provide no value
- `reward_monitor.py` — **UPDATED** for discrete reward tiers {-1, 1, 2, 3}

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