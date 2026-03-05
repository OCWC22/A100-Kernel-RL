# training/ — 3-Stage RL Pipeline

## GPU Split

> **B200** = model weights + generation + gradient updates + local nvcc compile checks (fast-fail).
> **A100 (Modal)** = all performance reward (speedup timing, execution-based correctness, Nsight profiling).
> B200 timing ≠ A100 timing. Never use B200 execution for performance reward.

## Model

- **Primary**: Qwen/Qwen3-Coder-Next (80B MoE, 3B active, FP8 on B200 192GB)
- **Fallback**: Qwen/Qwen2.5-Coder-7B-Instruct
- **Env vars**: `PRIMARY_MODEL`, `FALLBACK_MODEL` in `model_loader.py`

## Stage Configs

| Param | Stage 1 (`stage1_warmup.py`) | Stage 2 (`stage2_rft.py`) | Stage 3 (`stage3_grpo.py`) |
|-------|------------------------------|---------------------------|----------------------------|
| **Type** | GRPO warm-up | RFT (SFT on filtered) | TRLOO-augmented GRPO + curriculum (P3 demo) |
| **LR** | 3e-6 | 5e-6 | 5e-6 |
| **Temperature** | 0.9 | 0.7 (generation) | 0.7 |
| **G (generations)** | 4 | — | 2 |
| **Max turns** | 3 | — | 3 |
| **Steps/Epochs** | 300 steps | 3 epochs | 10 steps (P3 demo) |
| **Batch** | 1 × 4 grad_accum | 1 × 4 grad_accum | 1 × 4 grad_accum |
| **Optimizer** | paged_adamw_8bit | — | paged_adamw_8bit |
| **Output** | `outputs/kernelforge-stage1` | `outputs/kernelforge-stage2` | `outputs/kernelforge-stage3` |

**Shared**: bf16=True, max_prompt_length=512, max_completion_length=4096, top_k=50, top_p=0.95, repetition_penalty=1.05, use_vllm=True (colocate)

## LoRA Config (`model_loader.py`)

| Param | Value |
|-------|-------|
| r | 16 |
| alpha | 16 |
| dropout | 0 |
| max_seq_length | 8192 |
| quant | nf4 4-bit, double_quant, bf16 compute |
| **MoE targets** | q/k/v/o_proj, shared_expert.gate/up/down_proj |
| **Dense targets** | q/k/v/o_proj, gate/up/down_proj |

## Curriculum (`curriculum.py`)

| Phase | Name | Target Reward | Problems |
|-------|------|---------------|----------|
| 0 | `single_ops` | 1.0 | vector_add, relu, softmax, matmul, gelu |
| 1 | `fusion_2op` | 2.0 | LayerNorm+ReLU, MatMul+BiasAdd, Softmax+Dropout, Conv2d+BatchNorm |
| 2 | `arch_specific` | 2.0 | WCC, reduce_sum, GEMM, attention |
| 3 | `advanced` | 3.0 | LayerNorm+GELU+Linear, Flash-Attention, Batched SpMV |

- **Window**: 10 rewards
- **Promote**: >50% hit target reward
- **Demote**: <20% positive

### Key class: `CurriculumManager`
- `get_problem() -> dict` — next problem from current phase
- `record_reward(reward) -> str | None` — returns event ("promoted"/"demoted") or None
- `status() -> dict` — phase index, name, history, rates

## Multi-Turn Rollout (`multi_turn_rollout.py`)

### `make_multi_turn_rollout(max_turns=3, skill_md_gpu=None) -> Callable`
Returns TRL-compatible `rollout_func(prompt_ids, ...) -> dict` with keys:
`prompt_ids`, `completion_ids`, `logprobs`, `env_reward`

### Flow per completion:
1. Generate response on B200 → `extract_cuda_code(text)` extracts ```cuda blocks
2. `_evaluate_on_modal(code)` → **A100 execution:** compile + correctness verify + benchmark (5 graphs, 50 warmup, 30 runs)
3. `_compute_reward_from_result(result)` → 2-tier: fast path (log(speedup) from CUDA events), slow path (+ Nsight bonus for top-k)
4. `_format_feedback(result, reward, turn)` → feedback text for next turn
5. **Early exit** at reward >= 1.6 (log(5.0), i.e. 5x+ speedup)

### `reward_from_env(completions, **kwargs) -> list[float]`
Single-turn wrapper for Stage 1.

## RFT Filter (`rft_filter.py`)

### `TrajectoryCollector(modal_app_name, model_path)`
- `collect_trajectories(num_trajectories=100) -> list[dict]`
- `filter_trajectories(min_reward=1.0) -> list[dict]` (was 2.0, P0-3 fix)
- `save_rft_dataset(filtered, output_path)` → HF messages format
- Generation: max_new_tokens=2048, temperature=0.7, top_p=0.95

## CUDA Agent Integration (`cuda_agent_integration.py`)

- Dataset: `BytedTsinghua-SIA/CUDA-Agent-Ops-6K`
- `load_cuda_agent_prompt_dataset(max_samples=1024, seed=42) -> Dataset`
- `load_cuda_agent_prompt_texts(max_samples=128, seed=42) -> list[str]`
- `_build_cuda_prompt(example, max_code_chars=6000) -> str | None`

## Stacked Mitigations (NOT YET IMPLEMENTED)

These techniques from GRPO Deep Dive are planned but have no code yet:

| Technique | Deep Dive Ref | File to Create |
|-----------|---------------|----------------|
| MARS+TRLOO credit assignment | GRPO-4 line 1171 | `custom_grpo_loop.py` |
| Nsight structured rewards | GRPO-9 line 1619 | `hybrid_rollout.py` |
| CPPO completion pruning | GRPO-11 line 1758 | (in custom_grpo_loop.py) |
| MASPO soft trust region | GRPO-12 line 1801 | `maspo_loss.py` |
| ~~Transformation grammar~~ | GRPO-13 line 1845 | DEFERRED to v2 — use CUDA-Agent SKILL.md + doubleGraph patterns instead |

## Files to Create

- [ ] `custom_grpo_loop.py` — MARS+TRLOO + CPPO pruning (replaces TRL default loop)
- [ ] `hybrid_rollout.py` — local fast-path + Modal ncu fallback
- [ ] `maspo_loss.py` — soft trust region loss term

## Abort Conditions

- reward = -1 consistently after 30 steps → dataset/prompt issue
- reward std = 0 → model collapsed
- NaN loss → LR too high or gradient explosion
- OOM → reduce G, batch, or seq_length
- Gate G-0.8 failure (non-zero gradients + >1.2x on 2/20 kernels) → abandon GRPO, use SFT + SkyDiscover only

## Deep Dive Pointers

- Quick reference card: `GRPO_DEEP_DIVE.md` line 1569 (GRPO-8)
- Full stacked architecture: `GRPO_DEEP_DIVE.md` line 1974 (GRPO-14)
- Monitor checklist: `GRPO_DEEP_DIVE.md` line 1446 (GRPO-6)


<claude-mem-context>

</claude-mem-context>