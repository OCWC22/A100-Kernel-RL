# Google DeepMind Simply — KernelForge Integration Skill Reference

## Purpose

This document maps **google-deepmind/simply** (a minimal JAX-based LLM research framework) against KernelForge's existing architecture to identify what we can steal, adapt, or learn from for our CUDA kernel RL environment.

> **Source:** [github.com/google-deepmind/simply](https://github.com/google-deepmind/simply)
> **What it is:** A hackable JAX framework for frontier LLM training — pretraining, RL (PPO/GRPO), tool use, MoE, distributed training. Designed explicitly for AI agent-driven research.
> **Reviewed:** March 8, 2026

---

## TL;DR — What Simply Gives Us

| What | Value to KernelForge | Priority |
|------|---------------------|----------|
| **Registry-pattern configs** | Replace our ad-hoc stage configs with declarative, composable experiment definitions | HIGH |
| **PPO/GRPO loss with K3 KL estimator** | Reference implementation to validate our TRLOO correction against | HIGH |
| **Agent-first codebase design** | AGENTS.md / CLAUDE.md pattern — we already do this, Simply validates the approach | VALIDATION |
| **Reward normalization registry** | Global vs ByGroup normalizer — directly applicable to our discrete milestone rewards | MEDIUM |
| **RewardedSample / RLTrainingExampleBatch** | Structured RL data pipeline — cleaner than our current dict-passing | MEDIUM |
| **compute_return() with discounting** | We don't use this (outcome-only rewards), but useful if we add per-turn signal later | LOW |
| **JAX/XLA compilation** | Not directly applicable — we're PyTorch/TRL, they're JAX | N/A |

---

## 1. Architecture Comparison

### Simply's 3-Layer Stack

```
┌────────────────────────────────────┐
│  Config Layer (config_lib.py)      │  Registry + frozen dataclasses
│  - ExperimentConfigRegistry        │  String-based lookup
│  - ShardingConfigRegistry          │  Composable via dataclasses.replace()
│  - override_from() for variants    │
└────────────────────────────────────┘
          ↓
┌────────────────────────────────────┐
│  Training Layer                    │
│  - model_lib.py (Transformer)     │  Attention, MoE, KV cache
│  - rl_lib.py (PPO/GRPO loop)     │  Sample → Reward → Train
│  - data_lib.py (Grain pipeline)   │  TFDS, HuggingFace, ArrayRecord
└────────────────────────────────────┘
          ↓
┌────────────────────────────────────┐
│  Infra Layer                       │
│  - utils/sharding.py              │  FSDP, TP, EP
│  - utils/checkpointing.py         │  Orbax
│  - utils/sampling.py              │  Decoding / generation
└────────────────────────────────────┘
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

**Key difference:** Simply owns the model and training loop end-to-end in JAX. KernelForge delegates model training to TRL/Unsloth and focuses on the environment + reward + eval backend. This is the correct split for us — we don't need to rewrite the training loop in JAX.

---

## 2. What to Steal: Registry-Pattern Configs

### Simply's Pattern

```python
@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class RLExperimentConfig(BaseExperimentConfig):
    train_loop_name: str = 'rl'
    num_samples_per_example: int = 4
    gamma: float = 1.0
    kl_coeff: float = 0.001
    normalize_reward_method: str = 'global'

# Variant via composition
@ExperimentConfigRegistry.register
def grpo_variant():
    return dataclasses.replace(
        RLExperimentConfig(),
        use_grpo=True,
        kl_coeff=0.0,
    )
```

### KernelForge Application

We currently define stage configs as separate Python files (`stage1_warmup.py`, `stage2_rft.py`, `stage3_grpo.py`). Each hardcodes its own `GRPOConfig`. This works but is fragile — changing a shared parameter requires editing 3 files.

**Proposed adaptation:**

```python
# training/config_registry.py

import dataclasses
from typing import ClassVar

class KernelForgeConfigRegistry:
    """Simply-inspired registry for experiment configs."""
    _registry: ClassVar[dict] = {}

    @classmethod
    def register(cls, func):
        cls._registry[func.__name__] = func
        return func

    @classmethod
    def get(cls, name: str):
        return cls._registry[name]()

@dataclasses.dataclass(frozen=True)
class KernelForgeExperimentConfig:
    stage: str = "stage1"
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
    reward_mode: str = "discrete"  # "discrete" or "continuous"
    curriculum_enabled: bool = False

@KernelForgeConfigRegistry.register
def stage1_warmup():
    return KernelForgeExperimentConfig(stage="stage1", max_steps=100, lr=2e-6)

@KernelForgeConfigRegistry.register
def stage3_grpo():
    return dataclasses.replace(
        stage1_warmup(),
        stage="stage3",
        max_steps=50,
        lr=3e-6,
        temperature=0.7,
        curriculum_enabled=True,
    )
```

**Why this matters:** Composable configs make it trivial for an AI agent to propose experiment variants. Simply's entire design philosophy is "agents should be able to read, modify, and launch experiments autonomously." Our current file-per-stage approach requires the agent to understand 3 separate files to change one shared parameter.

---

## 3. What to Steal: Reward Normalization Registry

### Simply's Pattern

```python
class RewardNormalizer:
    """Abstract base for reward normalization."""
    def normalize(self, rewards, example_ids) -> rewards: ...

class GlobalRewardNormalizer(RewardNormalizer):
    """Mean-std across all rewards in batch."""

class ByGroupRewardNormalizer(RewardNormalizer):
    """Per-example-group normalization (GRPO-native)."""
```

### KernelForge Application

Our `compute_reward()` returns raw discrete milestones `{-1, 1, 2, 3}`. The TRLOO correction in `custom_grpo_trainer.py` handles advantage scaling. But we have no explicit reward normalization strategy — TRL's `GRPOTrainer` does its own internal normalization.

**What to adopt:** Add a `ByGroup` normalizer as a pre-processing step before TRLOO correction. This matters because:
- With G=2 and discrete rewards, the variance within a group can be 0 (both samples get same milestone)
- Simply's `ByGroup` normalizer handles this edge case explicitly
- Our TRLOO `N/(N-1)` scaling amplifies any normalization artifacts

**Concrete next step:** Inspect TRL's internal advantage normalization and confirm it's doing group-level normalization. If not, we need to add it explicitly before the TRLOO scale factor.

---

## 4. What to Steal: Structured RL Data Types

### Simply's Pattern

```python
@dataclasses.dataclass(frozen=True)
class RewardedSample:
    raw_example: Mapping[str, Any]
    sampling_output: SamplingOutput | None
    is_valid_for_training: bool = True
    correct: bool | None = None
    reward: float | None = None
```

### KernelForge Application

Our `multi_turn_rollout.py` passes dicts around with ad-hoc keys. This works but is error-prone — a typo in a key name silently produces `None` values. Simply's frozen dataclass approach gives us:
1. Type safety at construction time
2. Immutability guarantees (no accidental mutation between turns)
3. Clear API contract for what a "rewarded sample" contains

**Low-priority but clean improvement** for post-hackathon.

---

## 5. What to Steal: PPO/GRPO Loss Reference

### Simply's `compute_ppo_loss()`

```python
def compute_ppo_loss(
    model, params, batch,
    gamma=1.0, kl_coeff=0.001,
    use_grpo=False,
    ppo_clip_eps_high=0.2, ppo_clip_eps_low=0.2,
    policy_ratio_cap=10.0,
    normalize_advantage=True,
    max_abs_advantage=10.0,
):
```

Key features:
- **K3 KL estimator** (Schulman's approximation) when GRPO is enabled
- **Policy ratio capping** at 10.0 (prevents extreme updates)
- **Advantage clipping** at `max_abs_advantage=10.0`
- **Asymmetric PPO clip** (separate high/low epsilon)

### KernelForge Comparison

We delegate loss computation entirely to TRL's `GRPOTrainer`. Our only modification is the TRLOO `N/(N-1)` advantage scaling. Simply's implementation reveals three safety features TRL may or may not have:

| Feature | Simply | KernelForge (via TRL) | Action |
|---------|--------|----------------------|--------|
| Policy ratio cap | 10.0 | Check TRL source | VERIFY |
| Advantage clipping | ±10.0 | Check TRL source | VERIFY |
| K3 KL estimator | Yes (GRPO mode) | Check TRL source | VERIFY |
| Asymmetric clip | Yes | Likely no | LOW PRIORITY |

**Concrete next step:** Audit TRL's `GRPOTrainer._compute_loss()` to confirm it has equivalent safety guards. If not, we may need to add them in `TRLOOGRPOTrainer`.

---

## 6. What to Steal: Agent-First Design Philosophy

### Simply's Approach

Three dedicated agent files:
- `AGENTS.md` — General: architecture overview, code standards, how to run experiments
- `CLAUDE.md` — Claude Code-specific guidance
- `GEMINI.md` — Gemini/Antigravity-specific guidance

The entire framework is designed so an AI agent can:
1. Read the codebase
2. Propose a new optimizer / training objective / RL algorithm
3. Implement it using registry patterns
4. Run experiments autonomously
5. Iterate on results

### KernelForge Comparison

We already do this well:
- `docs/CLAUDE.md` — navigation hub
- `docs/SYSTEM_TRUTH.md` — single source of truth
- `docs/skills/*.md` — domain priors
- Per-directory `CLAUDE.md` files in `openenv_env/`, `training/`

**What Simply validates:** Our approach is right. The registry-pattern configs (Section 2 above) would make our codebase even more agent-friendly by letting agents compose experiment variants declaratively rather than editing Python files.

---

## 7. What NOT to Take from Simply

| Simply Feature | Why We Skip It |
|----------------|----------------|
| JAX/XLA backend | We're PyTorch/TRL. No benefit from rewriting. |
| Grain data pipeline | We use HuggingFace datasets + custom JSONL. Grain is JAX-ecosystem. |
| Orbax checkpointing | We use HuggingFace/Unsloth checkpoint patterns. |
| Multi-host sharding | Our model fits on one H200. Distributed training is a future concern. |
| model_lib.py Transformer | We load pretrained Qwen3-Coder via Unsloth. We don't build the model. |
| tool_lib.py | Our "tool use" is CUDA code generation, not function calling. |
| compute_return() | We use outcome-only rewards. No per-turn discounting needed (MARS was dropped). |

---

## 8. Integration Roadmap for KernelForge

### Phase 1: Config Registry (Hackathon-adjacent)

1. Create `training/config_registry.py` with Simply-inspired registry pattern
2. Define `KernelForgeExperimentConfig` frozen dataclass
3. Register `stage1_warmup`, `stage2_rft`, `stage3_grpo` as composable variants
4. Modify `grpo_train.py` launcher to use registry lookup
5. **Benefit:** Agents can propose experiment variants by registering new configs

### Phase 2: Reward Normalization Audit (Post-hackathon)

1. Audit TRL's internal advantage normalization for group-level behavior
2. If needed, add explicit `ByGroup` normalizer before TRLOO correction
3. Add `normalize_reward_method` to experiment config
4. **Benefit:** Prevents silent failures when all G samples hit the same milestone

### Phase 3: Loss Safety Guards (Post-hackathon)

1. Audit TRL's `GRPOTrainer` for policy ratio caps and advantage clipping
2. If missing, add them to `TRLOOGRPOTrainer._compute_loss()` override
3. **Benefit:** Prevents training instability from extreme advantage values

### Phase 4: Structured RL Types (v2)

1. Replace dict-passing in `multi_turn_rollout.py` with frozen dataclasses
2. Model after Simply's `RewardedSample` / `RLTrainingExampleBatch`
3. **Benefit:** Type safety, immutability, clearer API contracts

---

## 9. How Simply Maps to Our Research Papers

| Simply Component | KernelForge Equivalent | Paper Source |
|-----------------|----------------------|--------------|
| `compute_ppo_loss(use_grpo=True)` | `TRLOOGRPOTrainer._compute_advantages()` | Dr. Kernel (2602.05885) |
| `RewardNormalizer.ByGroup` | Implicit in TRL's GRPO | GRPO paper |
| `RewardedSample.is_valid_for_training` | `anti_hack.run_anti_hack_suite()` | Dr. Kernel anti-hack |
| `ExperimentConfigRegistry` | Our stage*.py files (to be replaced) | Simply design pattern |
| `compute_return(gamma)` | DROPPED (MARS degenerates with outcome rewards) | GRPO-4 analysis |
| `early_stop` config | Abort conditions in training/CLAUDE.md | Operational |

---

## 10. Key Takeaway

Simply is a **validation of our architectural choices**, not a replacement for them. The most actionable thing to steal is the **registry-pattern config system** — it's the single change that would most improve our agent-friendliness and experiment composability. Everything else is either already handled by TRL (loss computation), not applicable (JAX), or a post-hackathon cleanup (structured types).

The deeper insight from reviewing Simply is about **agent-first codebase design**: the entire framework assumes AI agents will be the primary developers. Our CLAUDE.md / SYSTEM_TRUTH.md / skills/ approach is already aligned with this philosophy. Simply confirms we're on the right track.

## Related Skill Docs

- `docs/skills/CUDA_AGENT.md` — CUDA-Agent environment prior
- `docs/skills/KERNELGYM_DR_KERNEL.md` — Dr. Kernel / TRLOO framing
- `docs/skills/SKYDISCOVER_ADAEVOLVE_EVOX.md` — Search hedge
- `docs/skills/doublegraph_a100.md` — A100 kernel engineering prior
