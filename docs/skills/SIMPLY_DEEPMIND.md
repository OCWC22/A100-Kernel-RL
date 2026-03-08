# Google DeepMind Simply — Complete Technical Breakdown & KernelForge Comparison

## Purpose

Complete technical analysis of [google-deepmind/simply](https://github.com/google-deepmind/simply), DeepMind's minimal JAX-based LLM research framework, mapped against KernelForge's architecture. Identifies what to adopt, adapt, or skip.

> **Source:** [github.com/google-deepmind/simply](https://github.com/google-deepmind/simply)
> **What it is:** A hackable JAX framework for frontier LLM training — pretraining, RL (PPO/GRPO), tool use, MoE, distributed training. Designed for AI agent-driven research.
> **Reviewed:** March 8, 2026

---

## TL;DR — What Simply Gives Us

| What | Value to KernelForge | Priority |
|------|---------------------|----------|
| **Registry-pattern configs** | Replace ad-hoc stage configs with declarative, composable experiment definitions | HIGH |
| **PPO/GRPO loss with K3 KL estimator** | Reference implementation to validate our TRLOO correction against | HIGH |
| **Policy ratio cap + advantage clipping** | Safety guards that TRL may not have — need to audit | HIGH (audit) |
| **Agent-first codebase design** | AGENTS.md / CLAUDE.md pattern — validates our approach | VALIDATION |
| **Reward normalization registry** | Global vs ByGroup normalizer — directly applicable to discrete milestone rewards | MEDIUM |
| **RewardedSample / RLTrainingExampleBatch** | Structured RL data pipeline — cleaner than our current dict-passing | MEDIUM |
| **compute_return() with discounting** | Not needed (outcome-only rewards), useful if we add per-turn signal later | LOW |
| **JAX/XLA compilation** | Not applicable — we're PyTorch/TRL | N/A |
| **MoE qwen3_30b_a3b config** | Confirms our model's architecture (128 experts, 8 active, megablox) | VALIDATION |

---

## 1. Architecture Overview

Simply is a **3-layer JAX framework** with agent-first design philosophy.

### Simply's 3-Layer Stack

```
┌─────────────────────────────────────────────────────────┐
│  Config Layer                                           │
│  config_lib.py: ExperimentConfigRegistry,               │
│                 ShardingConfigRegistry                   │
│  Frozen dataclasses + registry pattern                  │
│  Composable via dataclasses.replace()                   │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Training Layer                                         │
│  main.py → dispatches to train loop by name             │
│  model_lib.py: Transformer, MoE, train_one_step        │
│  rl_lib.py: PPO/GRPO loop, reward normalization         │
│  data_lib.py: Grain pipeline (TFDS, HF, ArrayRecord)   │
│  tool_lib.py: Tool-use during RL sampling               │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Infrastructure Layer                                   │
│  utils/sharding.py: FSDP, TP, EP, multi-host mesh      │
│  utils/checkpoint_lib.py: Orbax checkpointing           │
│  utils/sampling_lib.py: Decoding / generation           │
│  utils/optimizers.py: Adam, AdamW, SGD + LR schedules   │
│  utils/module.py: SimplyModule + registry pattern       │
│  utils/common.py: AnnotatedArray, PyTree utilities      │
└─────────────────────────────────────────────────────────┘
```

### KernelForge's 3-Layer Stack (Current)

```
┌────────────────────────────────────┐
│  OpenEnv Wrapper (openenv_env/)    │  step()/reset()/state()
│  - kernel_forge_env.py             │  Task pool, skill builder
│  - reward.py, anti_hack.py         │  Discrete {-1,1,2,3}
└────────────────────────────────────┘
          ↓
┌────────────────────────────────────┐
│  Training Layer (training/)        │
│  - custom_grpo_trainer.py          │  TRLOO-augmented GRPOTrainer
│  - multi_turn_rollout.py           │  3-turn rollout
│  - stage1/2/3 configs              │  Ad-hoc Python files
│  - curriculum.py                   │  Phase-based progression
└────────────────────────────────────┘
          ↓
┌────────────────────────────────────┐
│  Eval Backend (eval_service/)      │
│  - eval_core.py                    │  Compile + verify + benchmark
│  - app.py                          │  FastAPI on CoreWeave A100
│  - eval_backend.py                 │  Provider switch
└────────────────────────────────────┘
```

**Key difference:** Simply owns the model and training loop end-to-end in JAX. KernelForge delegates model training to TRL/Unsloth and focuses on the environment + reward + eval backend. This is the correct split for us.

---

## 2. Module-by-Module Technical Breakdown

### 2.1 `main.py` — Entry Point

Loads experiment config (from registry name or JSON file), applies mesh/sharding overrides from CLI flags, dispatches to the registered train loop.

```python
# Core flow:
config = ExperimentConfigRegistry.get_config(name)  # or load from JSON
config = override_mesh_and_sharding(config)          # CLI mesh flags
run_experiment_fn = TrainLoopRegistry.get(config.train_loop_name)
run_experiment_fn(config=config, experiment_dir=dir)
```

Key features:
- `execute_code_patch(config)` — configs can carry arbitrary Python code patches (exec'd at load time)
- `jax.distributed.initialize()` — multi-host setup (falls back to single-host)
- Both `--experiment_config name` (registry) and `--experiment_config_path file.json` (serialized)

**KernelForge equivalent:** `training/grpo_train.py` — preflight, model load, dataset, trainer, `.train()`. Simpler because we delegate to TRL.

### 2.2 `config_lib.py` — Configuration System

#### `BaseExperimentConfig` (frozen dataclass, ~120 fields)

```python
@dataclasses.dataclass(frozen=True)
class BaseExperimentConfig:
    # Model architecture
    model_dim: int = 2048
    per_head_dim: int = 128
    n_heads: int = 16
    n_layers: int = 24
    n_kv_heads: int = 4
    ffn_expand_dim: int = 0
    expand_factor: int = 4
    ffn_activation: str = 'gelu'
    use_qk_norm: bool = False

    # MoE
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity_factor: float | None = 1.0
    gmm_impl: str = 'ragged_dot'

    # Training
    train_loop_name: str = 'lm'
    batch_size: int = 32
    num_train_steps: int = 10000
    seq_len: int = 2048
    optimizer: opt_lib.Optimizer = opt_lib.Adam()
    lr: opt_lib.LRSchedule = opt_lib.LinearWarmupCosineDecay(...)
    grad_accum_steps: int = -1
    clip_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # Sharding
    sharding_config: ShardingConfig = gspmd_sharding()
    mesh_shape: dict | None = None
    dcn_mesh_shape: dict | None = None

    # Checkpointing
    init_ckpt_dir: str = ''
    ckpt_interval: int = 100
    ckpt_max_to_keep: int = 5

    # Data
    dataset: data_lib.DatasetConfig = ...
    vocab_name: str = ''
    activation_dtype_name: str = 'bfloat16'
```

#### `RLExperimentConfig` (extends BaseExperimentConfig)

```python
@dataclasses.dataclass(frozen=True)
class RLExperimentConfig(BaseExperimentConfig):
    train_loop_name: str = 'rl'
    train_batch_size: int = 128
    num_samples_per_example: int = 4
    sampling_temperature: float = 1.0

    # PPO/GRPO
    gamma: float = 1.0
    kl_coeff: float = 0.001
    use_grpo: bool = False
    ppo_clip_eps: float = 0.2
    ppo_clip_eps_low: float | None = None   # Asymmetric clip
    ppo_clip_eps_high: float | None = None
    policy_ratio_cap: float | None = 10.0
    normalize_advantage: bool = True
    max_abs_advantage: float | None = 10.0
    normalize_reward_method: str = ''       # '', 'Global', 'ByGroup'

    # Ref model
    ref_params_dtype: str = 'bfloat16'

    # Sampling
    sampling_max_decode_steps: int = 32768
    train_max_seq_len: int = 9216
    sampling_prefill_size: int = 1024
    num_train_steps_per_batch: int = 4      # Replay buffer steps

    # Validation
    validation_dataset: data_lib.DatasetConfig | None = None
    validation_eval_interval: int = 50

    # Early stopping
    early_stop: opt_lib.EarlyStop | None = None

    # Tool use
    max_turns: int = 0

    # Filtering
    filter_truncated: bool = False
    use_policy_logp_as_sampler_logp: bool = False
```

#### Registry pattern

```python
@ExperimentConfigRegistry.register
def my_experiment():
    base = some_base_config()
    return dataclasses.replace(base, lr=3e-6, batch_size=64)
```

#### Production GRPO configs disable safety guards

```python
# deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v2:
normalize_reward_method='ByGroup'
policy_ratio_cap=None         # DISABLED
max_abs_advantage=None         # DISABLED
num_train_steps_per_batch=1   # No replay buffer
use_policy_logp_as_sampler_logp=True  # Pure on-policy
```

**KernelForge equivalent:** Separate Python files (`stage1_warmup.py`, `stage2_rft.py`, `stage3_grpo.py`) each with hardcoded `GRPOConfig(...)`. No registry, no composition. Changing a shared parameter requires editing 3 files.

---

### 2.3 `rl_lib.py` — The RL Training Loop (CRITICAL)

This is Simply's most important module for us. ~1,470 lines implementing the complete RL pipeline.

#### 2.3.1 Data Types

```python
@dataclasses.dataclass(frozen=True)
class RewardedSample:
    raw_example: Mapping[str, Any]
    step: int
    in_batch_example_index: int
    sampling_input: SamplingInput
    sampling_output: SamplingOutput | None = None
    is_valid_for_training: bool = True
    correct: bool | None = None
    reward: float | None = None
    reward_result: Any | None = None
    reward_types: list[str] | None = None

    def update_with_evaluation_result(self, eval_result):
        return dataclasses.replace(
            self,
            raw_evaluation_result=eval_result,
            correct=eval_result['correct'],
            reward=eval_result['reward'],
            reward_result=eval_result.get('reward_result'),
            reward_types=eval_result.get('reward_types'),
        )

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RLTrainingExampleBatch:
    input_tokens: Array      # int [..., seq_len]
    target_tokens: Array     # int [..., seq_len]
    logprobs: Array           # float [..., seq_len]
    target_mask: Array        # bool [..., seq_len]
    answer_mask: Array        # bool [..., seq_len]
    in_batch_example_id: Array
    reward: Array
    is_correct: Array
    is_valid_for_training: Array
    ref_logprobs: Array | None = None
    extra_inputs: PyTree | None = None
```

**KernelForge equivalent:** We pass dicts in `multi_turn_rollout.py` with ad-hoc keys. No frozen dataclasses, no type safety.

#### 2.3.2 Reward Normalization

```python
class RewardNormalizerRegistry(registry.RootRegistry):
    namespace: str = 'RewardNormalizer'

class RewardNormalizer:

    class Global(Base):
        def normalize(self, rewards, example_ids, masks):
            mean = np_safe_mean(rewards, where=masks)
            std = np_safe_std(rewards, where=masks)
            return (rewards - mean) / np.maximum(std, 1e-5)

    class ByGroup(Base):
        def normalize_by_group(self, rewards, example_ids, masks, std=None):
            # Per-example-group normalization (GRPO-native)
            # Groups identified by consecutive equal example_ids
            # Each group: (r_i - group_mean) / max(group_std, 1e-5)
            new_rewards = []
            i = 0
            while i < rewards.shape[0]:
                j = i + 1
                while j < rewards.shape[0] and example_ids[j] == example_ids[i]:
                    j += 1
                group_rewards = rewards[i:j]
                group_masks = masks[i:j]
                mean_reward = np_safe_mean(group_rewards, where=group_masks)
                if std is None:
                    std_reward = np_safe_std(group_rewards, where=group_masks)
                else:
                    std_reward = std
                for k in range(i, j):
                    new_rewards.append(
                        (rewards[k] - mean_reward) / np.maximum(std_reward, 1e-5)
                    )
                i = j
            return np.array(new_rewards)
```

**Key insight:** ByGroup normalizes within each prompt's G samples. With discrete rewards {-1,1,2,3} and G=2, if both samples get same reward, group_std=0 and 1e-5 floor prevents division by zero — advantages become ~0, which is correct (no signal when all samples agree).

**KernelForge equivalent:** TRL's GRPOTrainer does internal normalization. We have no explicit ByGroup normalizer.

#### 2.3.3 `compute_ppo_loss()` — The Core Loss Function

```python
def compute_ppo_loss(
    model, params, batch,
    gamma=1.0,
    kl_coeff=0.001,
    use_grpo=False,
    ppo_clip_eps_high=0.2,
    ppo_clip_eps_low=0.2,
    policy_ratio_cap=10.0,
    normalize_advantage=True,
    max_abs_advantage=10.0,
    use_policy_logp_as_sampler_logp=False,
):
```

**Algorithm step-by-step (GRPO mode, gamma=1.0):**

1. **Forward pass:** `logits = model.apply(params, inputs)` → compute `logpi`
2. **KL divergence (K3 estimator — Schulman's approximation):**
   ```python
   logr = logpi_ref - logpi        # log ratio ref/policy
   kl = expm1(logr) - logr          # K3: always non-negative
   ```
3. **Advantage computation (GRPO, gamma=1.0):**
   ```python
   adv = reward * answer_mask       # Broadcast reward to all answer tokens
   ```
4. **Advantage normalization:**
   ```python
   mean, std = masked_mean_std(adv, mask=answer_mask)
   adv = (adv - mean) / (std + 1e-5)
   ```
5. **Advantage clipping:**
   ```python
   adv = clip(adv, -max_abs_advantage, max_abs_advantage)  # ±10.0
   ```
6. **Policy ratio + dual-clip PPO:**
   ```python
   ratio = exp(logpi - logpi_old)
   ratio = min(ratio, policy_ratio_cap)      # Cap at 10.0 first
   clipped_ratio = clip(ratio, 1-eps_low, 1+eps_high)  # Then standard clip
   surr1 = ratio * adv
   surr2 = clipped_ratio * adv
   loss = -min(surr1, surr2)                  # PPO surrogate
   ```
7. **KL penalty (GRPO-specific, added after PPO loss):**
   ```python
   kl_loss = masked_mean(kl, mask=answer_mask)
   loss += kl_coeff * kl_loss
   ```

**Return value includes detailed metrics:**
```python
return loss, {
    'entropy': entropy,
    'kl_divergence': kl_divergence,
    'policy_ratio/mean': policy_ratio,
    'policy_ratio/max': policy_ratio_max,
    'policy_ratio/min': policy_ratio_min,
    'loss_weight': sum(answer_mask),
    'logp_diff_abs/mean': ...,
    'logp_diff_abs/max': ...,
}
```

#### Safety features comparison with TRL:

| Feature | Simply | Our Stack (TRL) | Risk if Missing |
|---------|--------|-----------------|-----------------|
| Policy ratio cap (10.0) | `policy_ratio_cap` | **UNKNOWN — AUDIT** | Catastrophic updates |
| Advantage clipping (±10.0) | `max_abs_advantage` | **UNKNOWN — AUDIT** | Gradient explosion |
| K3 KL estimator | `expm1(logr) - logr` | TRL likely uses standard KL | Stability |
| Asymmetric PPO clip | Separate `eps_low`/`eps_high` | Likely symmetric | Minor |
| Dual-clip PPO | Ratio cap + standard clip | **UNKNOWN** | Extreme ratios |
| On-policy log-prob override | `use_policy_logp_as_sampler_logp` | No equivalent | Sharding artifacts |

**KernelForge equivalent:** `custom_grpo_trainer.py` overrides `_compute_advantages()` for TRLOO `N/(N-1)` scaling. All loss computation delegated to TRL's `GRPOTrainer._compute_loss()`. Zero visibility into TRL's safety guards.

#### 2.3.4 `run_experiment()` — The Main RL Loop

Registered as `TrainLoopRegistry.register(name='rl')`:

```
for each outer step:
  1. SAMPLE: Convert training data → SamplingInputs
     - Generate completions using current policy (on decoding mesh)
     - Support tool_executor for multi-turn with tool calls
     - Async reward computation via ThreadPoolExecutor

  2. REWARD: Collect RewardedSamples
     - evaluation.evaluate(raw_example, output_text) → {correct, reward}
     - Filter truncated/throttled/NaN samples
     - Track valid sample counts across hosts (process_allgather)

  3. BATCH: create_train_batch()
     - Stack RewardedSamples → RLTrainingExampleBatch
     - Pad/truncate to max_seq_len
     - Apply reward normalization (Global or ByGroup)
     - Compute ref_logprobs if ref_params provided
     - ragged_stack_allgather across hosts

  4. REPLAY BUFFER: replay_buffer.extend(train_batch)
     - When buffer full: iterate with shuffling
     - num_train_steps_per_batch controls replay ratio

  5. TRAIN: For each mini-batch from replay buffer:
     - train_one_step(state, batch, lr, custom_loss_fn=compute_ppo_loss)
     - JIT-compiled with gradient accumulation
     - Checkpointing, TensorBoard logging, early stopping

  6. VALIDATE: Periodic validation eval
     - Generate on validation set
     - Compute eval_accuracy
     - Check early_stop conditions (SimpleEarlyStop with threshold tuples)
```

**Architectural differences from KernelForge:**

| Aspect | Simply | KernelForge |
|--------|--------|-------------|
| Model ownership | Builds Transformer from scratch | Loads pretrained via Unsloth/HF |
| Training loop | Custom JAX loop with JIT | TRL's GRPOTrainer.train() |
| Sampling | Own decoding on separate mesh | TRL's generation (or vLLM) |
| Reward computation | Async ThreadPoolExecutor + multihost | Sequential in rollout_func |
| Multi-turn | tool_lib.ToolExecutor with max_turns | Custom rollout_func, 3 turns |
| Replay buffer | Built-in, configurable replay ratio | No replay (single pass) |
| Ref model | Separate ref_params from init_ckpt | TRL's internal ref model (or beta=0) |
| Gradient accumulation | JAX lax.scan over microbatches | TRL's gradient_accumulation_steps |
| Multi-host | Full FSDP + TP + EP | Single GPU (H200) |

#### 2.3.5 `compute_return()` — Discounted Returns

```python
def compute_return(reward, mask, gamma=1.0):
    if gamma == 1.0:
        ret = flip(cumsum(flip(reward)))  # Simple cumulative sum
    else:
        _, ret = jax.lax.scan(          # Discounted return
            lambda g, r: (r + gamma*g, r + gamma*g), ...)
```

**KernelForge:** Not applicable. Outcome-only rewards, no per-turn discounting. MARS return-to-go was dropped (degenerates with outcome rewards).

#### 2.3.6 `compute_logprobs()` — Reference Log-Probability Computation

```python
def compute_logprobs(model, params, batch, microbatch_size=None):
    # Forward pass → logits → Categorical distribution → log_prob(targets)
    # Supports microbatching via einops rearrange + lax.scan
    logits, _ = model.apply(params, inputs, ...)
    logits = jnp.astype(logits, jnp.float32)  # Always float32 for stability
    m = distributions.Categorical(logits)
    logprobs = masked.masked(m.log_prob(targets), mask=mask)
```

Key detail: log-probs always computed in float32 regardless of model dtype.

#### 2.3.7 `compute_stats()` — Comprehensive Metrics

Tracks per-step: seq_len (mean/max/min), prompt_len, response_len, truncated rate, reward, accuracy, pass@k, eval_count, train_count, per-reward-type breakdowns. All aggregated across hosts via `process_allgather`.

---

### 2.4 `model_lib.py` — Model Architecture

Key components:
- `TransformerLM` — Full LLM: embedding → N blocks → output projection
- `TransformerBlock` — Self-attention + FFN with pre/post-LN
- `Attention` — Multi-head with GQA, RoPE, QK-norm, soft-capping
- `FeedForward` — Gated activation (SiLU/GeLU)
- `MoEFeedForward` — MoE FFN with top-k routing, load-balancing loss
- `LMInterface` — Generation wrapper
- `train_one_step()` — Training step with grad accumulation, clipping, weight decay

#### MoE details (our model — Qwen3-30B-A3B):

```python
class MoEFeedForward(FeedForward):
    num_experts: int = 128        # Qwen3-30B has 128 experts
    num_experts_per_token: int = 8  # 8 active per token
    expert_capacity_factor: float | None = None  # dropless routing
    gmm_impl: str = 'megablox'   # Optimized grouped matmul
    lbl_loss_weight: float = 0.01  # Load-balancing loss
    router_z_loss_weight: float = 0.0
```

#### `train_one_step()` — Generic Training Step

```python
def train_one_step(state, batch, model, opt, lr,
                   grad_accum_steps, clip_grad_norm,
                   clip_update_norm, clip_update_rms,
                   clip_local_update_rms, weight_decay,
                   custom_loss_fn, add_log_info):
    # 1. Compute gradient (with grad accumulation via lax.scan if needed)
    # 2. Clip gradient norm
    # 3. Apply optimizer → get update
    # 4. Clip update norm / RMS
    # 5. Apply weight decay
    # 6. Apply scaled update: params += lr * update
    # 7. Increment steps
```

Key: supports both global gradient norm clipping AND local per-parameter RMS clipping simultaneously.

**KernelForge:** Handled entirely by TRL/Transformers. We don't touch the training step.

---

### 2.5 `data_lib.py` — Data Pipeline

Data sources: `TFDSSource`, `HFSource`, `ArrayRecordSource`

Packing strategies:
- `concat_split` — Best throughput (pretraining)
- `first_fit` — Bin packing preserving boundaries (chat/SFT)
- `pad_or_truncate` — Simple padding (validation)
- `none` — No packing (RL raw data)

Chat formatting: `LMFormat` registry handles tokenization for different templates.

**KernelForge:** HuggingFace datasets + custom JSONL via `training/dataset_loader.py`. Much simpler.

### 2.6 `tool_lib.py` — Tool Use Framework

`ToolExecutor` enables function-calling during RL sampling:
- `sample_with_tool()` — multi-turn with tool calls interlaced
- Model generates tool tokens → tool executed → result appended → continue

**KernelForge:** Fundamentally different. Our model generates CUDA code evaluated on remote A100. Multi-turn = feedback-based retry, not function calling.

### 2.7 `utils/sharding.py` — Distributed Training

3 mesh axes: `replica`, `data`, `model`
- FSDP: `data` axis for weight sharding
- TP: `model` axis for tensor parallelism
- EP: Expert parallelism for MoE

**KernelForge:** Single GPU (H200 141GB). No sharding needed.

### 2.8 `utils/optimizers.py` — Optimization

Production GRPO uses:
```python
optimizer = opt_lib.Adam(beta1=0.9, beta2=0.95, epsilon=1e-8)
lr = opt_lib.LinearWarmupConstant(value=1e-6, warmup_steps=1)
weight_decay = 0.0
```

**KernelForge:** `paged_adamw_8bit` via TRL/Transformers, LR=3e-6 for Stage 3.

---

## 3. Head-to-Head Comparison

### 3.1 What Simply Does That We Don't

| Simply Feature | Why It Matters | Adopt? |
|---------------|---------------|--------|
| **Registry-pattern configs** | Composable, agent-friendly experiments | **YES** — HIGH |
| **Frozen dataclass RL types** | Type safety, immutability | **YES** — MEDIUM |
| **Explicit reward normalization** (Global/ByGroup) | Handles edge cases | **AUDIT TRL** |
| **Policy ratio cap (10.0)** | Prevents catastrophic updates | **AUDIT TRL** |
| **Advantage clipping (±10.0)** | Prevents gradient explosion | **AUDIT TRL** |
| **K3 KL estimator** | Better KL approximation for GRPO | **LOW** (we use beta=0) |
| **Replay buffer** | Multiple train steps per batch | **NO** — we do single-pass |
| **Multi-host training** | Distributed across TPU pods | **NO** — single H200 |
| **Own Transformer** | Full architecture control | **NO** — we use pretrained |

### 3.2 What We Do That Simply Doesn't

| KernelForge Feature | Simply Has It? |
|--------------------|----------------|
| **TRLOO N/(N-1) correction** | **NO** — Simply's GRPO has the same self-inclusion bias |
| **Discrete milestone rewards {-1,1,2,3}** | No — continuous rewards from evaluation |
| **Local compile pre-check** | No — text-based math evaluation |
| **Remote A100 execution** | Local evaluation or external APIs |
| **Curriculum with phase progression** | Static datasets |
| **Anti-hack runtime checks** | `is_valid_for_training` flag only |
| **Multi-turn feedback loop** | Tool-calling, not feedback-based retry |
| **Unsloth MoE loading** | Builds models from scratch |

### 3.3 Where Simply Validates Our Choices

1. **Agent-first design** — AGENTS.md / CLAUDE.md / GEMINI.md validates our docs/ approach
2. **GRPO for reasoning tasks** — confirms it works for multi-step problem solving
3. **Low kl_coeff in production** — validates our beta=0 (no ref model) approach
4. **ByGroup normalization** — confirms this is the right normalization for grouped samples
5. **MoE qwen3_30b_a3b config** — 128 experts, 8 active, megablox = our model exactly
6. **gamma=1.0 default** — confirms outcome-level reward (no per-token discounting) is standard

---

## 4. Actionable Items

### Phase 1: TRL Safety Audit (HIGH — before next training run)

Verify TRL's `GRPOTrainer` has equivalent safety guards to Simply's `compute_ppo_loss()`:

1. **Policy ratio cap** — Does TRL cap `exp(logpi - logpi_old)`? Simply caps at 10.0.
2. **Advantage clipping** — Does TRL clip advantages? Simply clips at ±10.0.
3. **ByGroup normalization** — Does TRL normalize per-group? Simply's ByGroup: `(r - group_mean) / max(group_std, 1e-5)`.
4. **NaN handling** — Simply filters NaN rewards via `is_valid_for_training`. Does TRL?

Files to audit: `trl/trainer/grpo_trainer.py` — `_compute_loss()`, `_compute_advantages()`

### Phase 2: Config Registry (MEDIUM — post-hackathon)

Replace 3 stage config files with Simply-inspired composable registry:

```python
# training/config_registry.py
@dataclasses.dataclass(frozen=True)
class KernelForgeExperimentConfig:
    stage: str
    model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    lr: float = 2e-6
    temperature: float = 1.0
    num_generations: int = 2
    max_turns: int = 3
    max_steps: int = 100
    batch_size: int = 1
    grad_accum: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 768
    use_trloo: bool = True
    use_vllm: bool = False
    eval_backend: str = "coreweave"
    reward_mode: str = "discrete"
    curriculum_enabled: bool = False

@KernelForgeConfigRegistry.register
def stage1_warmup():
    return KernelForgeExperimentConfig(stage="stage1")

@KernelForgeConfigRegistry.register
def stage3_grpo():
    return dataclasses.replace(stage1_warmup(), stage="stage3",
        lr=3e-6, temperature=0.7, curriculum_enabled=True)
```

### Phase 3: Structured RL Types (LOW — v2)

Replace dict-passing in `multi_turn_rollout.py` with frozen dataclasses modeled after Simply's `RewardedSample` / `RLTrainingExampleBatch`.

---

## 5. Simply's Production GRPO Hyperparameters

From `deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl_f32_v2` (their most evolved config):

```
batch_size:              64
num_samples_per_example: 8        (G=8)
sampling_temperature:    1.0
grad_accum_steps:        4
num_train_steps_per_batch: 1      (single pass, no replay)
normalize_reward_method: 'ByGroup'
policy_ratio_cap:        None     (DISABLED)
max_abs_advantage:       None     (DISABLED)
lr:                      1e-6     (constant after 1 warmup step)
optimizer:               Adam(beta1=0.9, beta2=0.95, eps=1e-8)
weight_decay:            0.0
activation_dtype:        float32  (or bfloat16 in bf16 variant)
use_flash_attention:     True
flash_attn_block_size:   512
train_max_seq_len:       9216
validation_eval_interval: 50
ckpt_interval:           20
use_policy_logp_as_sampler_logp: True (bf16 variant)
```

**Key takeaway:** DeepMind deliberately disables safety guards (ratio cap, advantage clipping) in production. They keep ByGroup normalization. This suggests the guards are training wheels that experienced practitioners remove once they have confidence in the training dynamics.

---

## 6. How Simply Maps to Our Research Papers

| Simply Component | KernelForge Equivalent | Paper Source |
|-----------------|----------------------|--------------|
| `compute_ppo_loss(use_grpo=True)` | `TRLOOGRPOTrainer._compute_advantages()` | Dr. Kernel (2602.05885) |
| `RewardNormalizer.ByGroup` | Implicit in TRL's GRPO | GRPO paper |
| `RewardedSample.is_valid_for_training` | `anti_hack.run_anti_hack_suite()` | Dr. Kernel |
| `ExperimentConfigRegistry` | Our stage*.py files (to be replaced) | Simply design |
| `compute_return(gamma)` | DROPPED (degenerates with outcome rewards) | GRPO-4 analysis |
| `early_stop` config | Abort conditions in training/CLAUDE.md | Operational |
| `qwen3_30b_a3b()` | `model_loader.py` Unsloth loading | Qwen3 model card |

---

## 7. Key Takeaway

Simply is a **validation of our architectural choices**, not a replacement. The most actionable items:

1. **TRL safety audit** — verify we have ratio caps and advantage clipping, or add them
2. **Registry-pattern configs** — the single change that most improves agent-friendliness
3. **TRLOO is novel** — Simply's GRPO has the same self-inclusion bias we correct. Our contribution stands.

The deeper insight: Simply proves that **agent-first codebase design** (registry patterns, frozen dataclasses, comprehensive AGENTS.md) is the future of ML research codebases. We're already on this path.

## Related Skill Docs

- `docs/skills/CUDA_AGENT.md` — CUDA-Agent environment prior
- `docs/skills/KERNELGYM_DR_KERNEL.md` — Dr. Kernel / TRLOO framing
- `docs/skills/SKYDISCOVER_ADAEVOLVE_EVOX.md` — Search hedge
- `docs/skills/DOUBLEGRAPH_A100.md` — A100 kernel engineering prior
