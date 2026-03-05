# KernelForge Training Plan — Multi-Turn Agentic GRPO

**Project:** KernelForge-OpenEnv — RL post-training for CUDA kernel generation
**Target:** A100 (sm_80), single-GPU training
**Hackathon:** March 7-8, 2026

---

## Timeline

### March 4, 2026 — AM: Infrastructure Complete

- Modal A100 deployment validated (5/5 eval checks pass, NVIDIA A100-SXM4-40GB detected)
- Baseline WCC kernel compiles, passes PAC verification, benchmarks at 39ms on A100
- CUDA 12.4.1 image deployed (`nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`)
- Datasets generated: `ops6k_full` (6K), `curated_200` (200), `wcc_training` (50)
- Per-problem baselines computed: `datasets/baselines.jsonl` (200 problems, A100 baseline 81.56ms)
- Dataset integrity verified: curated PASS, baselines PASS
- 96/96 unit tests pass, 7/7 smoke tests pass, 5/5 Modal eval checks pass
- Pipeline dry-run: all data stages CACHED, training stages ready

### March 4, 2026 — PM: Critical Audit

**Finding:** Training pipeline is **single-turn** — model generates CUDA code, gets scalar reward, moves on. No error feedback, no iteration. This is fundamentally different from CUDA Agent's multi-turn agentic approach.

**Decision:** Implement multi-turn via TRL's `rollout_func` + `generate_rollout_completions` pattern before training. Without this, the model can't learn from compilation errors or runtime feedback.

### March 4-5, 2026: Implementation

- Write `training/multi_turn_rollout.py` (shared rollout logic)
- Modify `training/stage1_warmup.py` (3 turns/episode, 300 steps)
- Modify `training/stage3_grpo.py` (5 turns/episode, 200 steps, curriculum)
- Add batch evaluation endpoint to `modal_app.py`
- Add `vllm` dependency for `rollout_func` support
- Test everything

### March 5-6, 2026: Training Execution

- Stage 1 GRPO Warm-up: ~4 hours (3,600 kernel evaluations)
- Stage 2 RFT: ~30 minutes (100 evaluations)
- Stage 3 GRPO + Curriculum: ~5 hours (4,000 evaluations)
- Total: ~9-10 hours wall-clock

### March 7-8, 2026: Hackathon

- Evaluate trained models via `evaluation/compare_stages.py`
- Report pass@k, compile_rate, correct_rate, avg_reward, median_speedup
- Iterate if time allows

---

## Research Foundation

### Paper 1: CUDA Agent — ByteDance (arXiv:2602.24286)

**Link:** https://arxiv.org/abs/2602.24286

**What they did:** Trained Seed1.6 (230B MoE, 23B active) on 128 H20 GPUs to generate optimized CUDA kernels via RL. Used their CUDA-Agent-Ops-6K dataset (6,000 PyTorch operator compositions, 83.77% are 2-op fusions).

**Results:**
- 98.8% pass rate (compile + correct)
- 96.8% faster-than-torch.compile rate
- 2.11x geometric mean speedup over torch.compile

**Key discoveries we use:**

1. **Discrete milestone rewards beat continuous by +36.4pp** (Table 2 ablation)
   - Continuous speedup reward: 60.4% faster-than-compile
   - Discrete milestones {-1, 1, 2, 3}: 96.8% faster-than-compile
   - Why: GPU timing noise makes continuous rewards unstable. Milestones give clean signal.

2. **Skipping RFT causes training collapse** (Table 2 ablation)
   - With RFT: 96.8% faster-than-compile
   - Without RFT: 49.8% faster-than-compile
   - Why: Without anchoring on correct trajectories, policy entropy explodes.

3. **Multi-turn agentic loop is the core capability** (Section 4)
   - Up to 200 turns per episode
   - Model sees compilation errors, verification failures, runtime feedback
   - Model iterates on its kernel across turns
   - This is where optimization discovery happens

4. **4-stage pipeline prevents training collapse** (Figure 3)
   - Stage 1: Single-turn PPO warm-up (bootstrap CUDA syntax)
   - Stage 2: RFT (filter for correct trajectories, SFT)
   - Stage 3: Value pretraining (train critic)
   - Stage 4: Full agentic PPO (multi-turn, 200 turns)

### Paper 2: DoubleGraph / WarpSpeed — DoubleAI

**What they did:** Per-GPU-architecture optimized graph kernels. 192 CUDA kernel files per target GPU. Same algorithm, completely different code for A100 vs L4 vs A10G.

**Results:** 3.6x average speedup over expert-tuned cuGraph, 18% of algorithms at 10-100x speedup.

**What we use:**
- **GPU Registry:** A100/H100/B200 specs (L2 cache, SM count, HBM bandwidth, feature flags). File: `openenv_env/gpu_registry.py`
- **CachePool (LRU, 8 entries):** GPU-resident test data between evaluations. File: `openenv_env/cache_pool.py`
- **4-way dispatch as curriculum:** base → fusion → architecture-specific → advanced. File: `training/curriculum.py`
- **Per-kernel .cu.flags:** Learnable compilation parameters (`--use_fast_math`, `--maxrregcount=N`). File: `openenv_env/anti_hack.py`
- **A100-specific techniques in SKILL.md:** L2 pinning (40MB), float4 vectorized loads, cooperative groups. File: `openenv_env/skill_builder.py`

### Paper 3: DeepSeek-R1 (arXiv:2501.12948)

**Link:** https://arxiv.org/abs/2501.12948

**What they proved:** GRPO (Group Relative Policy Optimization) works at 671B scale, achieving near-parity with PPO-based systems. GRPO eliminates the critic entirely — normalizes rewards within generation groups instead.

**Why we use GRPO instead of PPO:**
- PPO requires actor + critic + reference model = 3 copies of model weights
- For 80B model: 240B params in memory. Impossible on 1 GPU.
- GRPO: actor only. Saves ~50% memory.
- Trade-off: ~10-20% less sample-efficient, but fits on hardware.

### TRL OpenEnv Integration

**Link:** https://huggingface.co/docs/trl/main/en/openenv

**What it provides:** `rollout_func` parameter for `GRPOTrainer` that enables custom multi-turn rollout logic. Used with `generate_rollout_completions()` to generate completions within a rollout loop. Extra dict fields propagate to reward functions as kwargs.

**Pattern:** TRL's Wordle example shows multi-turn: loop over turns, generate per turn, step environment, accumulate `prompt_ids` + `completion_ids` + `logprobs` across turns, return best reward.

---

### Paper 4: MARS Credit Assignment — MARSHAL (arXiv:2510.15414)

**Link:** https://arxiv.org/abs/2510.15414
**Code:** https://github.com/thu-nics/MARS
**Status:** ICLR 2026 Accepted

**What it solves:** Long-horizon credit assignment in multi-turn RL. Standard GRPO assigns the same advantage to every token in a trajectory — the model can't distinguish "Turn 1 fixed the compile error" from "Turn 4 added an irrelevant comment".

**MARS Solution:** Turn-level advantage estimation via cumulative returns:

```
R_{i,k} = Σ_{k'=k}^{K} r_{i,k'}    # Cumulative return from turn k onward
A_{i,k} = (R_{i,k} - mean(R)) / std(R)    # Group-relative normalization
```

**Why it works:**
- Early turns get credit for everything they enabled later
- Low variance via group normalization
- No critic required (GRPO-compatible)
- ~50 lines to implement

**Results:**
- +28.7% on held-out games
- +10.0% on AIME, +6.6% on GPQA-Diamond
- Zero extra model overhead

**Implementation:** See `docs/research/MARS_CREDIT_ASSIGNMENT.md` for full algorithm.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training GPU (H100 80GB)                      │
│                                                                  │
│  ┌─────────────────────────────────┐                             │
│  │ Qwen3-Coder-Next 80B MoE       │ ~43 GB (4-bit NF4)          │
│  │ QLoRA: r=16, attention+shared   │                             │
│  │ Fallback: Qwen2.5-Coder-7B     │                             │
│  └──────────────┬──────────────────┘                             │
│                 │                                                 │
│  ┌──────────────▼──────────────────┐                             │
│  │ vLLM (colocate mode)            │ generate_rollout_completions│
│  │ Generates CUDA kernel code      │ per turn in multi-turn loop │
│  └──────────────┬──────────────────┘                             │
│                 │ CUDA source string                              │
│                 ▼                                                 │
│  ┌──────────────────────────────────┐     ┌────────────────────┐ │
│  │ Multi-Turn Rollout Loop          │     │ Modal A100 (remote) │ │
│  │                                  │     │                    │ │
│  │ Turn 1: generate → eval → -1    │────▶│ nvcc -arch=sm_80   │ │
│  │   feedback: "COMPILE FAILED:    │◀────│ PAC verify (5 grph)│ │
│  │   undefined __shfl_sync ln 47"  │     │ Benchmark 50w+30r  │ │
│  │                                  │     │ ~5-15 sec/eval     │ │
│  │ Turn 2: generate → eval → +1    │────▶│                    │ │
│  │   feedback: "CORRECT, 5.2ms,    │◀────│                    │ │
│  │   0.95x. Optimize for speed."   │     │                    │ │
│  │                                  │     │                    │ │
│  │ Turn 3: generate → eval → +2    │────▶│                    │ │
│  │   feedback: "CORRECT, 2.1ms,    │◀────│                    │ │
│  │   1.7x. Beat torch.compile."    │     │                    │ │
│  │                                  │     └────────────────────┘ │
│  │ Return: best_reward=+2          │                             │
│  │ + accumulated token IDs/logprobs│                             │
│  └──────────────┬──────────────────┘                             │
│                 │                                                 │
│  ┌──────────────▼──────────────────┐                             │
│  │ GRPO Gradient Update            │                             │
│  │ Group-relative advantage        │                             │
│  │ paged_adamw_8bit                │                             │
│  └─────────────────────────────────┘                             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Dependencies

### Compatible Version Stack

| Package | Version | Notes |
|---------|---------|-------|
| **unsloth** | 2026.3.3 | Install with `--no-deps` to bypass stale `trl<=0.24.0` cap |
| **trl** | >=0.29.0 | Has `rollout_func` + colocate + `generate_rollout_completions` |
| **transformers** | >=4.56.2, !=5.0.0, !=5.1.0 | Per trl 0.29.0 + unsloth exclusions |
| **torch** | (determined by vllm) | vllm pins it; don't specify ourselves |
| **vllm** | >=0.10.2, <0.13.0 | Per `trl[vllm]` 0.29.0 extra |
| **accelerate** | >=1.4.0 | Required by trl 0.29.0 |
| **datasets** | >=3.0 | No conflicts |

### Install Commands (Linux + CUDA)

```bash
# 1. Install unsloth WITHOUT its stale dependency caps
pip install --no-deps unsloth unsloth_zoo

# 2. Install TRL with vLLM support (pulls torch, vllm, transformers, accelerate)
pip install "trl[vllm]>=0.29.0"

# 3. Install remaining project dependencies
pip install -e ".[train,modal]"
```

### Why `--no-deps` for Unsloth

Unsloth 2026.3.3 caps `trl<=0.24.0` in its `pyproject.toml`, but `rollout_func` requires `trl>=0.25.0` (added in v0.25.0, stabilized in v0.29.0). The unsloth Dec 2025 release [added rollout_func support](https://github.com/unslothai/unsloth/releases/tag/December-2025) and works fine with trl 0.29.0 — the cap is just stale metadata.

After loading a model with `FastLanguageModel`, call `PatchFastRL("GRPO", FastLanguageModel)` so unsloth optimizations apply to vanilla TRL's `GRPOTrainer`.

**Import path:** `from trl.experimental.openenv import generate_rollout_completions` (NOT `trl.trainer.grpo_trainer`).

---

## Training Stages

### Stage 1: GRPO Warm-up (Multi-Turn)

**Goal:** Bootstrap CUDA syntax. Raise compilation rate from ~50% → ~85%.

| Parameter | Value | Rationale |
|---|---|---|
| Algorithm | GRPOTrainer + rollout_func | Multi-turn via TRL OpenEnv pattern |
| Max turns per episode | 3 | Short warm-up; most value in 1st error correction |
| Max training steps | 300 | Was 100. More budget for syntax learning |
| Dataset | 512 samples from Ops-6K (single-op) | Easy operators: vector_add, relu, softmax, etc. |
| Temperature | 0.9 | High exploration for syntax discovery |
| Learning rate | 3e-6 | Low to prevent catastrophic forgetting |
| KL penalty (beta) | 0.0 | No KL — let model explore freely |
| Num generations | 4 | 4 rollouts per prompt, group-relative normalization |
| Batch size | 1 (per device) × 4 (gradient accum) | Effective batch = 4 |
| Max completion length | 4096 tokens | Enough for full CUDA kernel |
| vLLM mode | colocate | Same GPU as training model |
| Total evaluations | ~3,600 | 300 steps × 4 gens × 3 turns |
| Estimated wall-clock | ~4 hours (batched) | |

**File:** `training/stage1_warmup.py`

### Stage 2: Rejection Fine-Tuning (Single-Turn)

**Goal:** Filter Stage 1 trajectories, SFT on correct kernels. Prevents entropy explosion.

| Parameter | Value | Rationale |
|---|---|---|
| Algorithm | SFTTrainer | Supervised fine-tuning on filtered data |
| Trajectories collected | 100 | Was 50. More data for RFT |
| Filter threshold | reward >= 1.0 | Keep correct kernels |
| Epochs | 3 | Standard SFT |
| Learning rate | 5e-6 | |
| Total evaluations | ~100 | Trajectory collection only |
| Estimated wall-clock | ~30 minutes | |

**File:** `training/stage2_rft.py`

### Stage 3: GRPO + Curriculum (Multi-Turn)

**Goal:** Learn optimization strategies through progressive difficulty.

| Parameter | Value | Rationale |
|---|---|---|
| Algorithm | GRPOTrainer + rollout_func | Multi-turn with curriculum |
| Max turns per episode | 5 | Harder problems need more iterations |
| Max training steps | 200 | Was 60. Main RL training, longest stage |
| Dataset | 200 curriculum-sampled prompts | 4 phases: single_ops → fusion → arch → advanced |
| Temperature | 0.7 | Lower for exploitation over exploration |
| Learning rate | 5e-6 | Faster convergence |
| Num generations | 4 | |
| Total evaluations | ~4,000 | 200 steps × 4 gens × 5 turns |
| Estimated wall-clock | ~5 hours (batched) | |

**Curriculum phases:**

| Phase | Difficulty | Target Reward | Example Problems |
|---|---|---|---|
| single_ops | 1 | 1.0 | vector_add, relu, softmax, matmul, gelu |
| fusion_2op | 2 | 2.0 | LayerNorm+ReLU, MatMul+BiasAdd, Softmax+Dropout |
| arch_specific | 3 | 2.0 | WCC+L2 pinning, Reduction+cooperative groups, GEMM+float4 |
| advanced | 4 | 3.0 | LayerNorm+GELU+Linear fusion, Flash-Attention, Batched SpMV |

**Promotion:** >50% of last 10 rewards hit target → advance phase
**Demotion:** <20% of last 10 rewards positive → regress phase

**File:** `training/stage3_grpo.py`

---

## Compute Budget

| Stage | Steps | Gens/step | Turns/gen | Total evals | Wall-clock (batched) |
|---|---|---|---|---|---|
| Stage 1 | 300 | 4 | 3 | 3,600 | ~4 hrs |
| Stage 2 | N/A | N/A | 1 | 100 | ~30 min |
| Stage 3 | 200 | 4 | 5 | 4,000 | ~5 hrs |
| **Total** | | | | **7,700** | **~9-10 hrs** |

**Modal A100 cost:** ~$3.50/hr → ~$30-35 for evaluations
**Training GPU (H100):** Separate cost, ~9-10 hours

---

## Reward Function

Identical to CUDA Agent (arXiv:2602.24286, Equation 1):

```
r = -1.0   if compilation fails OR correctness fails
r = +1.0   if correct, speedup_vs_eager <= 1.05
r = +2.0   if correct, speedup_vs_eager > 1.05
r = +3.0   if correct, speedup_vs_eager > 1.05 AND speedup_vs_compile > 1.05
```

**File:** `openenv_env/reward.py`

Validated by CUDA Agent Table 2 ablation: +36.4pp over continuous rewards.

---

## MARS Credit Assignment (Critical for Multi-Turn)

### Why Standard GRPO Fails in Multi-Turn

Standard GRPO assigns the **same advantage** to every token in a trajectory:

```
A = (R - mean(R)) / std(R)    # Same for all tokens
```

**Failure mode:** The model cannot distinguish:
- "Turn 1 fixed the compile error" from "Turn 4 added an irrelevant comment"
- Early good actions get diluted → slow learning or collapse

### MARS Solution: Turn-Level Advantage

**Algorithm:**

1. **Record per-turn rewards** during rollout:
   ```python
   turn_rewards = []  # [r_1, r_2, ..., r_K]
   ```

2. **Compute cumulative returns** (sum from turn k to end):
   ```python
   cumulative_returns = []
   current = 0.0
   for r in reversed(turn_rewards):
       current = r + gamma * current
       cumulative_returns.append(current)
   cumulative_returns = cumulative_returns[::-1]
   ```

3. **Group-relative normalization** (across all rollouts in generation group):
   ```python
   advantages = [(R_k - mean_R) / std_R for R_k in cumulative_returns]
   ```

4. **Apply to tokens:** Every token in turn k gets advantage `advantages[k]`

### Partial Rewards for Intermediate Turns

| Turn Outcome | Partial Reward | Rationale |
|--------------|----------------|-----------|
| Compilation fails | -1.0 | Clear failure signal |
| Compiles, verification fails | -0.5 | Partial progress |
| Correct, no speedup (≤1.05×) | +0.5 | Baseline achievement |
| Correct, >5% faster than eager | +1.0 | Promising optimization |

**Final turn:** Uses canonical {-1, 1, 2, 3} reward from CUDA Agent.

### Implementation in `training/multi_turn_rollout.py`

```python
def mars_rollout(prompt, env, model, max_turns=5, gamma=1.0):
    """Multi-turn rollout with MARS credit assignment."""
    history = [{"role": "user", "content": prompt}]
    turn_rewards = []
    turn_token_ids = []
    turn_logprobs = []
    best_reward = -1.0
    best_kernel = None

    for turn in range(max_turns):
        completion, ids, logprobs = model.generate_with_logprobs(history)
        kernel = extract_cuda_kernel(completion)
        result = env.step(kernel)

        # Partial reward (intermediate) or canonical (final)
        if turn < max_turns - 1:
            r_turn = compute_partial_reward(result)
        else:
            r_turn = result.reward  # Canonical {-1, 1, 2, 3}

        turn_rewards.append(r_turn)
        turn_token_ids.append(ids)
        turn_logprobs.append(logprobs)

        if result.reward > best_reward:
            best_reward = result.reward
            best_kernel = kernel

        if result.reward >= 2.0:
            break  # Early stop

        history.append({"role": "assistant", "content": completion})
        history.append({"role": "user", "content": format_feedback(result)})

    # MARS: Compute cumulative returns
    cumulative_returns = []
    current = 0.0
    for r in reversed(turn_rewards):
        current = r + gamma * current
        cumulative_returns.append(current)
    cumulative_returns = cumulative_returns[::-1]

    return {
        "best_kernel": best_kernel,
        "best_reward": best_reward,
        "turn_rewards": turn_rewards,
        "cumulative_returns": cumulative_returns,
        "turn_token_ids": turn_token_ids,
        "turn_logprobs": turn_logprobs,
    }


def compute_partial_reward(result: dict) -> float:
    """Partial rewards for intermediate turns."""
    if not result.get("compiles"):
        return -1.0
    if not result.get("correct"):
        return -0.5
    speedup = result.get("speedup_vs_orig", 1.0)
    if speedup > 1.05:
        return +1.0
    return +0.5


def mars_group_normalize(all_rollouts: list) -> list:
    """Group-relative normalization across rollouts."""
    all_returns = []
    for r in all_rollouts:
        all_returns.extend(r["cumulative_returns"])

    mean_R = sum(all_returns) / len(all_returns)
    std_R = max((sum((x - mean_R)**2 for x in all_returns) / len(all_returns))**0.5, 1e-8)

    for rollout in all_rollouts:
        rollout["turn_advantages"] = [(R_k - mean_R) / std_R for R_k in rollout["cumulative_returns"]]

    return all_rollouts
```

### Why MARS is Critical for KernelForge

| Without MARS | With MARS |
|--------------|-----------|
| Same advantage for all turns | Per-turn advantage |
| Early fixes get no credit | Early fixes get full credit |
| ~2× slower convergence | ~2-3× better sample efficiency |
| May collapse on long rollouts | Stable long-horizon learning |

**Bottom line:** MARS is the single highest-leverage change for making 3-5 turn episodes actually learn optimization instead of just syntax.

---

## Key Decisions and Rationale

| Decision | Choice | Alternative | Why |
|---|---|---|---|
| RL algorithm | GRPO | PPO | PPO needs 3 model copies, can't fit on 1 GPU |
| Multi-turn method | TRL rollout_func | Reward-func wrapper | rollout_func is native TRL, proper token tracking |
| **Credit assignment** | **MARS (turn-level)** | Trajectory-level | MARS gives 2-3× better sample efficiency; proven in ICLR 2026 |
| Max turns Stage 1 | 3 | 200 (CUDA Agent) | 200 turns × 300 steps = 27 days. 3 captures error correction value. |
| Max turns Stage 3 | 5 | 200 | Same reasoning. Harder problems get 2 more attempts. |
| Reward per episode | best_reward | last_reward | Don't penalize exploration; reward best achieved kernel |
| Generation backend | vLLM colocate | transformers.generate | Required by TRL rollout_func |
| Evaluation backend | Modal A100 (remote) | Local GPU | Decoupled: model trains on H100, kernels eval on A100 |
| Stage 1 steps | 300 | 100 (original) | 100 was undertraining risk; 300 matches ~20% of compute |
| Stage 3 steps | 200 | 60 (original) | 60 was undertraining risk; 200 is main RL stage (~60% compute) |
| Model | Qwen3-Coder-Next 80B MoE | Qwen2.5-Coder-7B | Prefer larger model; 7B is fallback if VRAM insufficient |

---

## Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Stage 1 doesn't converge (no reward >= 1.0) | HIGH | Monitor compile rate; abort if <20% after 100 steps |
| vLLM incompatible with QLoRA model | MEDIUM | Test locally first; fallback: use transformers.generate with manual loop |
| 7B fallback model can't learn CUDA | MEDIUM | 7B has less capacity; may only reach reward +1 consistently |
| Modal cold-start adds latency per turn | LOW | Batch evaluations; keep container warm between stages |
| TRL rollout_func API changes | LOW | Pin trl>=0.29.0; use `trl.experimental.openenv` import path |
| Context window overflow (multi-turn accumulation) | MEDIUM | Cap total context to 8192 tokens; truncate history if needed |

---

## Expected Outcomes

| Outcome | Criteria | Likelihood |
|---|---|---|
| **Minimum viable** | Pipeline runs end-to-end. GRPO shifts reward distribution upward. | High |
| **Good** | Model consistently generates reward >= 2.0 (beats eager). Clear reward curve. | Medium |
| **Great** | Model discovers optimization patterns through multi-turn iteration. Approaches torch.compile. | Low-Medium |

We are NOT expecting to match CUDA Agent's 2.11x result — they used 128 H20 GPUs and 230B model. We're proving the methodology works at single-GPU scale with multi-turn GRPO.

---

## File Index

| File | Role |
|---|---|
| `training/multi_turn_rollout.py` | **NEW** — Multi-turn rollout logic with MARS credit assignment |
| `training/stage1_warmup.py` | Stage 1: GRPO warm-up (modified for multi-turn) |
| `training/stage2_rft.py` | Stage 2: Rejection fine-tuning (unchanged) |
| `training/stage3_grpo.py` | Stage 3: GRPO + curriculum (modified for multi-turn) |
| `training/curriculum.py` | Curriculum manager (4 phases, promotion/demotion) |
| `training/model_loader.py` | Model loading (QLoRA, MoE LoRA targeting) |
| `openenv_env/kernelforge_env.py` | OpenEnv environment (reset/step/state) |
| `openenv_env/reward.py` | Discrete reward function {-1, 1, 2, 3} |
| `openenv_env/anti_hack.py` | Anti-reward-hacking (symbol scan, flag whitelist) |
| `openenv_env/gpu_registry.py` | GPU specs (A100/H100/B200) |
| `openenv_env/skill_builder.py` | SKILL.md generation (per-GPU optimization hints) |
| `openenv_env/cache_pool.py` | LRU cache for GPU-resident eval data |
| `verification/pac_verify.py` | PAC verification (5 graphs, 3 invariants) |
| `modal_app.py` | Modal serverless GPU backend |
| `evaluation/eval_model.py` | Per-checkpoint evaluation |
| `evaluation/compare_stages.py` | Stage-over-stage comparison |
| `evaluation/pass_at_k.py` | Unbiased pass@k estimator |
| `datasets/curated_200.jsonl` | 200 curated problems (difficulty 1-3) |
| `datasets/baselines.jsonl` | Per-problem A100 baselines |
| `scripts/run_pipeline.py` | Pipeline orchestrator |

### Research Documentation

| File | Content |
|---|---|
| `docs/research/MARS_CREDIT_ASSIGNMENT.md` | Full MARS algorithm, implementation, citations |
| `docs/research/MULTI_TURN_RL_TECHNIQUES.md` | Survey of credit assignment methods |
| `docs/research/CUDA_AGENT_ANALYSIS.md` | Deep dive into CUDA Agent paper |
| `docs/research/README.md` | Quick reference and implementation priority |
