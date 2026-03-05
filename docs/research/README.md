# KernelForge Research Documentation

This folder contains research synthesis documents for multi-turn RL techniques applicable to CUDA kernel generation.

---

## Documents

### 1. MARS Credit Assignment

**File:** `MARS_CREDIT_ASSIGNMENT.md`

The core technique for solving long-horizon credit assignment in multi-turn RL. MARS (Multi-Agent Reasoning through Self-play) uses cumulative per-turn returns with group-relative normalization to correctly attribute rewards to individual turns.

**Key findings:**
- +28.7% on held-out games, +10.0% on AIME
- No critic required (GRPO-compatible)
- ~50 lines to implement

**Source:** MARSHAL (arXiv:2510.15414), ICLR 2026

---

### 2. Multi-Turn RL Techniques Survey

**File:** `MULTI_TURN_RL_TECHNIQUES.md`

Comprehensive survey of credit assignment methods for long-horizon agentic RL:

| Method | Granularity | Critic | Best For |
|--------|-------------|--------|----------|
| MARS | Turn | No | Multi-turn interactions |
| GRPO-λ | Token | No | Fine-grained reasoning |
| Turn-PPO | Turn | Yes | Long-horizon PPO |
| MARTI-MARS² | Turn + Tree | No | Multi-agent code gen |

**Recommendation:** MARS is the optimal choice for KernelForge's single-GPU QLoRA setup.

---

### 3. CUDA Agent Deep Dive

**File:** `CUDA_AGENT_DEEP_DIVE.md`

Comprehensive analysis of ByteDance's CUDA Agent paper (arXiv:2602.24286):

- **4-stage pipeline** (we adapt to 3-stage for GRPO)
- **Discrete milestone rewards** {-1, 1, 2, 3} — +36.4pp over continuous
- **RFT is critical** — Without it, faster-than-compile drops from 96.8% to 49.8%
- **Anti-hacking measures** — Forbidden symbols, flag whitelist, randomized inputs
- **Ops-6K dataset** — 6,000 PyTorch operators (83.77% are 2-op fusion)
- **Training collapse analysis** — Why RFT is mandatory (step 17 collapse)

**Key result:** 98.8% pass rate, 96.8% faster-than-compile, 2.11× speedup

---

## Quick Reference

### MARS Algorithm (Pseudocode)

```python
# During rollout
turn_rewards = []
for turn in range(max_turns):
    result = evaluate(generate(prompt))
    turn_rewards.append(compute_partial_reward(result))

# After episode
cumulative_returns = []
current = 0.0
for r in reversed(turn_rewards):
    current = r + gamma * current
    cumulative_returns.append(current)
cumulative_returns = cumulative_returns[::-1]

# Group normalize
advantages = [(R_k - mean_R) / std_R for R_k in cumulative_returns]
```

### Partial Reward Design

| Outcome | Reward |
|---------|--------|
| Compile fails | -1.0 |
| Correctness fails | -0.5 |
| Correct, no speedup | +0.5 |
| Correct, >5% faster | +1.0 |

### CUDA Agent Reward (Final Turn)

| Outcome | Reward |
|---------|--------|
| Compile/correct fail | -1 |
| Correct, no speedup | +1 |
| >5% faster than eager | +2 |
| >5% faster than eager AND compile | +3 |

---

## Implementation Priority

1. **P0:** Implement MARS in `training/multi_turn_rollout.py`
2. **P1:** Design partial rewards for intermediate turns
3. **P1:** Ensure RFT filters at reward ≥ 1.0 (not 2.0)
4. **P2:** Consider GRPO-λ for token-level credit (future)

---

## References

| Paper | arXiv | Key Contribution |
|-------|-------|------------------|
| MARSHAL | 2510.15414 | MARS turn-level advantage |
| CUDA Agent | 2602.24286 | 4-stage pipeline, discrete rewards |
| GRPO-λ | 2510.00194 | Token-level eligibility traces |
| Turn-PPO | 2512.17008 | Turn-level MDP + GAE |
| MARTI-MARS² | 2602.07848 | Multi-agent tree search |
| Multi-Turn GRPO | 2505.11821 | Turn-level advantage for tool-use |

---

## Related Files

- `../TRAINING_PLAN.md` — Execution plan for hackathon
- `../KernelForge_Truth.md` — Single source of truth
- `../STATUS.md` — Implementation status and hackathon readiness
