# Multi-Turn RL Techniques for LLM Agents

**Survey Date:** March 4, 2026
**Focus:** Credit assignment methods for long-horizon agentic RL

---

## Overview

Multi-turn RL for LLM agents faces a fundamental challenge: **credit assignment**. When an agent takes multiple actions across many turns before receiving a reward, how do we correctly attribute credit (or blame) to each individual action?

This document surveys the state-of-the-art techniques for solving this problem, with a focus on methods applicable to KernelForge's CUDA kernel generation pipeline.

---

## 1. MARS (MARSHAL) — Turn-Level Cumulative Advantage

**Paper:** MARSHAL: Incentivizing Multi-Agent Reasoning via Self-Play with Strategic LLMs
**arXiv:** 2510.15414
**Published:** October 2025
**Status:** ICLR 2026 Accepted

### Core Technique

MARS uses **cumulative per-turn returns** with group-relative normalization:

```
R_{i,k} = Σ_{k'=k}^{K} r_{i,k'}
A_{i,k} = (R_{i,k} - mean(R)) / std(R)
```

### Key Properties

- **No critic required** — pure GRPO-compatible
- **Low variance** — group normalization stabilizes training
- **Long-horizon capable** — early turns get credit for later successes
- **Multi-agent extension** — role-specific normalization

### Results

- +28.7% on held-out games
- +10.0% on AIME, +6.6% on GPQA-Diamond
- Zero extra model overhead

**See full analysis:** `MARS_CREDIT_ASSIGNMENT.md`

---

## 2. GRPO-λ — Token-Level Credit with λ-Return

**Paper:** GRPO-λ: Credit Assignment improves LLM Reasoning
**arXiv:** 2510.00194
**Authors:** Prasanna Parthasarathi, Mathieu Reymond, et al. (Huawei Noah's Ark Lab, Mila)

### Core Technique

GRPO-λ extends GRPO with **eligibility traces** and **λ-return approximation**:

1. **Token-level log-probabilities** after each sequence generation
2. **Critic-free TD error approximation**
3. **Multiple weighting schemes** for λ-return

### Key Properties

- **Token-level granularity** — finer than MARS's turn-level
- **No critic** — uses novel approximation
- **30-40% improvement** over vanilla GRPO during training

### Comparison with MARS

| Aspect | GRPO-λ | MARS |
|--------|--------|------|
| Granularity | Token-level | Turn-level |
| Complexity | Higher (eligibility traces) | Lower (cumulative sum) |
| Best for | Fine-grained reasoning | Multi-turn interactions |
| Implementation | ~100 lines | ~50 lines |

### Applicability to KernelForge

GRPO-λ could be useful if we want **token-level** credit within each turn (e.g., distinguishing which lines of CUDA code contributed most to speedup). However, for hackathon scale, MARS's turn-level approach is simpler and proven effective.

---

## 3. Turn-PPO — Turn-Level MDP Formulation

**Paper:** Turn-PPO: Turn-Level Advantage Estimation with PPO for Improved Multi-Turn RL in Agentic LLMs
**arXiv:** 2512.17008
**Authors:** Junbo Li, Peng Zhou, et al. (UT Austin, Amazon)
**Published:** December 2025

### Core Technique

Turn-PPO reformulates the MDP at **turn granularity** instead of token granularity:

1. **State = conversation history** (all previous turns)
2. **Action = entire turn output** (full response)
3. **Reward = per-turn reward** (intermediate or final)

Then applies standard PPO with GAE at turn level.

### Key Properties

- **More robust than GRPO** in multi-turn settings
- **GAE(γ, λ) at turn level** — temporal difference learning
- **Requires critic** — unlike MARS, needs value function

### Results

On WebShop and Sokoban:
- Outperforms trajectory-level GRPO
- Stable training curves
- Better long-horizon performance

### Comparison with MARS

| Aspect | Turn-PPO | MARS |
|--------|----------|------|
| Algorithm | PPO | GRPO |
| Critic required | Yes | No |
| Advantage estimation | GAE(γ, λ) | Cumulative return |
| Memory overhead | Higher (critic) | Lower |
| Implementation complexity | Medium | Low |

### Applicability to KernelForge

Turn-PPO would require adding a critic model, which doubles memory requirements. For our single-GPU QLoRA setup, MARS is more practical. However, Turn-PPO's MDP formulation is conceptually useful for understanding multi-turn RL.

---

## 4. MARTI-MARS² — Multi-Agent Tree Search for Code

**Paper:** MARTI-MARS²: Scaling Multi-Agent Self-Search via Reinforcement Learning for Code Generation
**arXiv:** 2602.07848
**Authors:** Shijie Wang, Pengfei Li, et al. (Shanghai AI Lab, Tsinghua, multiple institutions)
**Published:** February 2026

### Core Technique

MARTI-MARS² combines:

1. **Multi-agent tree search** — agents explore solution branches collaboratively
2. **Self-search scaling** — iterative refinement within environment
3. **Heterogeneous multi-agent training** — different agents learn different roles
4. **MARS credit assignment** — turn-level advantage for each agent

### Key Innovation

Evolution from **parameter-sharing homogeneous multi-role training** to **heterogeneous multi-agent training** — different agents specialize in different aspects of code generation.

### Results

- Breaks performance ceilings in code generation
- Better error correction capabilities
- Strategic diversity across agents

### Applicability to KernelForge

MARTI-MARS² is the most advanced technique but requires:
- Multiple models (heterogeneous agents)
- Tree search infrastructure
- Significant compute

For hackathon scale, this is overkill. However, the **conceptual insight** of specialized agents (e.g., "syntax fixer" vs "optimizer") could inform future work.

---

## 5. Multi-Turn Reasoning via Turn-Level Credit Assignment

**Paper:** Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment
**arXiv:** 2505.11821
**Authors:** Siliang Zeng, Quan Wei, et al. (University of Minnesota, Morgan Stanley)
**Published:** May 2025

### Core Technique

Introduces **fine-grained turn-level advantage estimation** for GRPO:

1. Model multi-turn tool-use as MDP
2. Design **turn-level rewards** (verifiable or LLM-as-judge)
3. Extend GRPO to multi-turn variant with turn-level advantages

### Key Properties

- **MDP formulation** explicit at turn level
- **Two reward types**: verifiable (ground truth) or LLM-as-judge
- **100% tool execution success** vs 20-30% for baselines
- **50% exact answer accuracy** vs 20-30% for baselines

### Turn-Level Reward Design

| Reward Type | Use Case | Example |
|-------------|----------|---------|
| Verifiable | Ground truth available | Code compiles, test passes |
| LLM-as-judge | Subjective evaluation | Code quality, style |

### Applicability to KernelForge

This paper directly informs our **partial reward design** for intermediate turns:
- Compilation success = verifiable
- Correctness = verifiable
- Speedup = verifiable (via benchmark)

The MDP formulation is exactly what we need for CUDA kernel generation.

---

## 6. LLM-Guided Credit Assignment in MARL

**Paper:** Speaking the Language of Teamwork: LLM-Guided Credit Assignment in Multi-Agent Reinforcement Learning
**arXiv:** 2502.03723
**Authors:** Muhan Lin, Shuyang Shi, et al. (CMU, others)
**Published:** February 2025

### Core Technique

Uses LLMs to **generate dense reward functions** from sparse rewards:

1. Start with sparse team reward
2. LLM analyzes agent trajectories
3. LLM generates dense per-agent rewards
4. Train with dense rewards, evaluate on sparse

### Key Insight

LLMs can decompose sparse team rewards into meaningful per-agent credits by understanding agent roles and contributions.

### Applicability to KernelForge

Could be used to generate **per-turn reward explanations** that help the model understand why it got a particular reward. However, adds complexity without clear benefit for our setting.

---

## Comparison Matrix

| Method | Granularity | Critic | Memory | Complexity | Best For |
|--------|-------------|--------|--------|------------|----------|
| **MARS** | Turn | No | Low | Low | Multi-turn interactions |
| GRPO-λ | Token | No | Low | Medium | Fine-grained reasoning |
| Turn-PPO | Turn | Yes | Medium | Medium | Long-horizon PPO |
| MARTI-MARS² | Turn + Tree | No | High | High | Multi-agent code gen |
| Multi-Turn GRPO | Turn | No | Low | Low | Tool-use agents |

---

## Recommended Approach for KernelForge

### Primary: MARS Credit Assignment

**Why:**
1. **Proven effectiveness** — ICLR 2026 accepted, +28.7% improvements
2. **No critic** — fits single-GPU QLoRA setup
3. **Simple implementation** — ~50 lines of code
4. **Directly applicable** — designed for multi-turn agentic RL

### Implementation Priority

1. **P0:** Implement MARS in `training/multi_turn_rollout.py`
2. **P1:** Design partial rewards for intermediate turns
3. **P2:** Consider GRPO-λ for token-level credit within turns (future)

### Partial Reward Design

```python
def compute_partial_reward(result: dict) -> float:
    """
    Turn-level partial rewards for CUDA kernel generation.
    
    Design rationale:
    - Negative rewards for clear failures (compile/verify)
    - Small positive for progress (correct but slow)
    - Larger positive for optimization success
    """
    if not result.get("compiles"):
        return -1.0   # Clear failure signal
    
    if not result.get("correct"):
        return -0.5   # Partial progress (compiled but wrong)
    
    speedup = result.get("speedup_vs_orig", 1.0)
    
    if speedup > 1.05:
        return +1.0   # Promising optimization
    
    return +0.5       # Correct but no speedup
```

---

## Key References

1. **MARSHAL (MARS):** arXiv:2510.15414 — https://arxiv.org/abs/2510.15414
2. **GRPO-λ:** arXiv:2510.00194 — https://arxiv.org/abs/2510.00194
3. **Turn-PPO:** arXiv:2512.17008 — https://arxiv.org/abs/2512.17008
4. **MARTI-MARS²:** arXiv:2602.07848 — https://arxiv.org/abs/2602.07848
5. **Multi-Turn GRPO:** arXiv:2505.11821 — https://arxiv.org/abs/2505.11821
6. **LLM-Guided MARL:** arXiv:2502.03723 — https://arxiv.org/abs/2502.03723

---

## Timeline of Development

| Date | Paper | Key Innovation |
|------|-------|----------------|
| May 2025 | Multi-Turn GRPO | Turn-level advantage for tool-use |
| Oct 2025 | MARSHAL (MARS) | Cumulative return + group norm |
| Oct 2025 | GRPO-λ | Token-level eligibility traces |
| Dec 2025 | Turn-PPO | Turn-level MDP + GAE |
| Feb 2026 | MARTI-MARS² | Multi-agent tree search |

**Trend:** The field is rapidly converging on turn-level credit assignment as the key to multi-turn agentic RL. MARS is the simplest and most proven approach for single-GPU setups.
