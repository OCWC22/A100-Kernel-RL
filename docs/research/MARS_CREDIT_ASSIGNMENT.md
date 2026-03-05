# MARS Credit Assignment for Multi-Turn RL

**Paper:** MARSHAL: Incentivizing Multi-Agent Reasoning via Self-Play with Strategic LLMs
**arXiv:** 2510.15414
**Published:** October 17, 2025
**Accepted:** ICLR 2026
**Authors:** Huining Yuan, Zelai Xu, Zheyue Tan, et al. (Tsinghua University, Aalto University, Li Auto Inc.)
**Code:** https://github.com/thu-nics/MARS

---

## Executive Summary

MARS (Multi-Agent Reasoning through Self-play) is a turn-level advantage estimator that solves the **long-horizon credit assignment problem** in multi-turn, multi-agent reinforcement learning with LLMs. It is the core technique inside the MARSHAL framework and is now considered state-of-the-art for multi-turn agentic RL.

**Key Result:** MARSHAL agent trained from Qwen3-4B achieved:
- **+28.7%** performance improvement on held-out games
- **+10.0%** on AIME, **+6.6%** on GPQA-Diamond when integrated into multi-agent systems
- **+3.5%** average across all reasoning benchmarks

---

## The Problem: Why Standard GRPO Fails in Multi-Turn

### Trajectory-Level Advantage (Standard Approach)

In standard GRPO/PPO for multi-turn settings:

1. Run full episode (all turns)
2. Get one scalar reward R at the end
3. Assign the **same advantage** to every token in the entire trajectory:
   ```
   A = (R - mean(R)) / std(R)
   ```

**The Failure Mode:**
- Model cannot distinguish "Turn 1 fixed the compile error" from "Turn 4 added an irrelevant comment"
- Early good actions get diluted or miscredited
- Results in slow learning or policy collapse

This is exactly why CUDA Agent needed a full 4-stage pipeline and why naive multi-turn GRPO risks under-training on optimization discovery.

---

## The Solution: MARS Turn-Level Advantage

### Core Insight: Sum-Then-Normalize

MARS replaces trajectory-level advantage with **per-turn advantage estimation** using cumulative Monte-Carlo returns.

### Step-by-Step Algorithm

#### Step 1: Record Per-Turn Rewards During Rollout

```python
# During multi-turn rollout
turn_rewards = []  # r_{i,1}, r_{i,2}, ..., r_{i,K}

for turn in range(max_turns):
    completion = generate(prompt_with_history)
    result = evaluate(completion)
    
    # Assign partial reward per turn
    r_turn = compute_partial_reward(result)  # -1 / +0.5 / +1 / +2
    turn_rewards.append(r_turn)
```

#### Step 2: Compute Cumulative Return Per Turn

The key mathematical insight:

```
R_{i,k} = Σ_{k'=k}^{K} r_{i,k'}
```

This is the **total future reward from turn k onward**. Early turns get credit for everything they enabled later.

```python
# After full episode completes
cumulative_returns = []
current = 0.0
for r in reversed(turn_rewards):
    current += r  # or gamma * r for discounting
    cumulative_returns.append(current)
cumulative_returns = cumulative_returns[::-1]  # Now R_k for each turn
```

#### Step 3: Group-Relative Normalization (GRPO Style, Per-Turn)

```python
# Across all trajectories in the generation group
mean_R = sum(all_cumulative_returns) / len(all_returns)
advantages = [R_k - mean_R for R_k in cumulative_returns]
```

**Every token generated in turn k gets this same advantage A_{i,k}.**

#### Step 4: Agent-Specific / Role-Specific Normalization (Multi-Agent Extension)

If you have different "roles" (e.g., "syntax fixer" vs "optimizer"):

```python
# Normalize separately within each role's sub-group
advantages_by_role = {}
for role in roles:
    role_returns = [r for r in all_returns if r.role == role]
    mean_role = sum(role_returns) / len(role_returns)
    advantages_by_role[role] = [R_k - mean_role for R_k in role_returns]
```

This prevents a strong optimizer role from dominating the weaker syntax-fixer role.

#### Step 5: Plug Into GRPO Objective

The surrogate loss becomes:

```
J(θ) = E[ Σ_{k=1}^{K} Σ_{t ∈ tokens in turn k} clipped-PPO-surrogate(A_{i,k}) ]
```

---

## Why MARS Works So Well

### Theoretical Advantages

| Property | MARS | Standard GRPO |
|----------|------|---------------|
| Credit granularity | Per-turn | Per-trajectory |
| Long-horizon dependency | Captured via cumulative return | Lost (same advantage everywhere) |
| Variance | Low (group normalization) | High |
| Multi-turn stability | Excellent | Poor (entropy collapse) |

### Mathematical Equivalence

MARS is mathematically equivalent to **GAE(γ=1, λ=1)** with batch-mean baseline, but applied at **turn granularity** instead of token or full-trajectory.

### Empirical Validation (MARSHAL Paper)

| Benchmark | MARSHAL Improvement |
|-----------|---------------------|
| Held-out games | +28.7% |
| AIME (math reasoning) | +10.0% |
| GPQA-Diamond (science QA) | +6.6% |
| Average across benchmarks | +3.5% |

---

## MARS vs. Alternatives

| Method | Credit Granularity | Variance | Multi-turn Stability | Extra Models Required |
|--------|-------------------|----------|---------------------|----------------------|
| Standard GRPO | Trajectory | High | Poor | None |
| Naive per-turn norm | Turn (local) | Very High | Collapse | None |
| **MARS (sum-then-norm)** | Turn (cumulative) | Low | Excellent | None |
| GAE(0.95, 0.95) + critic | Token | Medium | Good | Critic network |
| PPO with value function | Token | Medium | Good | Critic network |

**Key Insight:** MARS achieves critic-level performance without a critic, making it perfect for single-GPU QLoRA setups.

---

## Partial Reward Design for KernelForge

For CUDA kernel generation, we can design per-turn rewards:

| Turn Outcome | Partial Reward | Rationale |
|--------------|----------------|-----------|
| Compilation fails | -1.0 | Clear negative signal |
| Compilation succeeds, verification fails | -0.5 | Partial progress |
| Correct but no speedup (≤1.05×) | +0.5 | Baseline achievement |
| Correct with speedup >1.05× vs eager | +1.0 | Good optimization |
| Correct with speedup >1.05× vs compile | +2.0 | Excellent optimization |

**Note:** The final turn still gets the canonical {-1, 1, 2, 3} reward, but intermediate turns get partial signals that MARS propagates backward.

---

## Implementation for KernelForge

### Code Template for `training/multi_turn_rollout.py`

```python
def mars_rollout(
    prompt: str,
    env,  # KernelForgeEnv
    model,  # vLLM model
    max_turns: int = 5,
    gamma: float = 1.0,  # No discounting (MARS default)
) -> dict:
    """
    Multi-turn rollout with MARS credit assignment.
    
    Returns:
        - best_kernel: str
        - best_reward: float
        - turn_advantages: List[float]  # Per-turn advantage for GRPO
        - token_ids: List[int]  # All tokens generated
        - logprobs: List[float]  # Log probabilities for each token
    """
    history = [{"role": "user", "content": prompt}]
    turn_rewards = []
    turn_token_ids = []  # List of lists
    turn_logprobs = []   # List of lists
    
    best_reward = -1.0
    best_kernel = None
    
    for turn in range(max_turns):
        # Generate completion
        completion, ids, logprobs = model.generate_with_logprobs(history)
        kernel = extract_cuda_kernel(completion)
        
        # Evaluate
        result = env.step(kernel)
        
        # Partial reward (intermediate turns)
        if turn < max_turns - 1:
            r_turn = compute_partial_reward(result)
        else:
            # Final turn: canonical reward
            r_turn = result.reward
        
        turn_rewards.append(r_turn)
        turn_token_ids.append(ids)
        turn_logprobs.append(logprobs)
        
        # Track best
        if result.reward > best_reward:
            best_reward = result.reward
            best_kernel = kernel
        
        # Early stop if excellent
        if result.reward >= 2.0:
            break
        
        # Add feedback to history
        history.append({"role": "assistant", "content": completion})
        history.append({"role": "user", "content": format_feedback(result)})
    
    # MARS: Compute cumulative returns (sum-then-normalize)
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
        "num_turns": len(turn_rewards),
    }


def compute_partial_reward(result: dict) -> float:
    """
    Compute partial reward for intermediate turns.
    
    This is the key design choice for MARS in CUDA kernel generation.
    """
    if not result.get("compiles"):
        return -1.0
    if not result.get("correct"):
        return -0.5
    
    speedup = result.get("speedup_vs_orig", 0)
    if speedup > 1.05:
        return +1.0  # Promising speedup
    return +0.5  # Correct but slow


def mars_group_normalize(
    all_rollouts: List[dict],
) -> List[dict]:
    """
    Apply MARS group-relative normalization across all rollouts.
    
    This is called after collecting multiple rollouts (e.g., 4 generations per prompt).
    """
    # Flatten all cumulative returns
    all_returns = []
    for rollout in all_rollouts:
        all_returns.extend(rollout["cumulative_returns"])
    
    mean_R = sum(all_returns) / len(all_returns)
    std_R = (sum((r - mean_R)**2 for r in all_returns) / len(all_returns))**0.5
    std_R = max(std_R, 1e-8)  # Prevent division by zero
    
    # Normalize each turn's advantage
    for rollout in all_rollouts:
        rollout["turn_advantages"] = [
            (R_k - mean_R) / std_R
            for R_k in rollout["cumulative_returns"]
        ]
    
    return all_rollouts
```

### Integration with GRPOTrainer

```python
# In training loop
rollouts = []
for _ in range(num_generations):
    rollout = mars_rollout(prompt, env, model, max_turns=5)
    rollouts.append(rollout)

# MARS normalization
rollouts = mars_group_normalize(rollouts)

# Build advantage-weighted loss
for rollout in rollouts:
    for turn_idx, (ids, logprobs, advantage) in enumerate(
        zip(rollout["turn_token_ids"], rollout["turn_logprobs"], rollout["turn_advantages"])
    ):
        # Each token in this turn gets the same advantage
        for token_id, logprob in zip(ids, logprobs):
            # GRPO surrogate loss with turn-level advantage
            loss += -advantage * logprob  # Simplified; use clipped PPO in practice
```

---

## Key Takeaways for KernelForge

1. **MARS is the missing piece** for multi-turn CUDA kernel optimization
2. **Zero extra models** — works with pure GRPO, no critic needed
3. **~2-3× better sample efficiency** on long rollouts (based on MARSHAL ablations)
4. **<50 lines of code** to implement
5. **Critical for Stage 3** — enables learning optimization patterns, not just syntax

---

## References

1. **MARSHAL Paper:** arXiv:2510.15414 — https://arxiv.org/abs/2510.15414
2. **Code:** https://github.com/thu-nics/MARS
3. **Project Page:** https://thu-nics.github.io/MARSHAL/
4. **License:** Apache 2.0

---

## Related Papers

- **MARTI-MARS²** (arXiv:2602.07848): Extension to code generation with multi-agent tree search
- **GRPO-λ** (arXiv:2510.00194): Token-level credit assignment with λ-return
- **Turn-PPO** (arXiv:2512.17008): Turn-level advantage with PPO for agentic LLMs
- **Multi-Turn Reasoning via Turn-Level Credit** (arXiv:2505.11821): Fine-grained turn-level advantage for GRPO
