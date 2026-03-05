# CUDA Agent Deep Dive

**Paper:** arXiv:2602.24286 - "CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation"

**Authors:** Weinan Dai, Hanlin Wu, Qiying Yu, et al. (ByteDance)

**Submitted:** February 27, 2026

---

## Executive Summary

CUDA Agent is a large-scale agentic reinforcement learning system that develops CUDA kernel expertise through three components:

1. **Scalable Data Synthesis Pipeline** — 6,000 training samples
2. **Skill-Augmented Environment** — Automated verification and profiling
3. **RL Algorithmic Techniques** — Multi-stage training for stability

**Key Results:**
- 100% faster-than-compile on KernelBench Level-1
- 100% faster-than-compile on KernelBench Level-2
- 92% faster-than-compile on KernelBench Level-3
- Outperforms Claude Opus 4.5 and Gemini 3 Pro by ~40% on Level-3

---

## Part 1: System Architecture

### 1.1 Three Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      CUDA Agent System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐  ┌───────────────────┐  ┌────────────────┐  │
│  │ Data Synthesis│  │ Skill-Augmented   │  │ RL Techniques  │  │
│  │ Pipeline      │  │ Environment       │  │                │  │
│  │               │  │                   │  │                │  │
│  │ • 6K operators│  │ • ReAct loop      │  │ • 4-stage PPO  │  │
│  │ • 5-op fusion │  │ • Compile-debug   │  │ • RFT warm-up  │  │
│  │ • Anti-hack   │  │ • Profiler-guided │  │ • Critic pretrain│
│  │ • Diversity   │  │ • Correctness     │  │ • Agentic PPO  │  │
│  └───────────────┘  └───────────────────┘  └────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Seed-1.6 (230B MoE, 23B active) |
| Context Length | 131,072 tokens |
| Agent Turns | Up to 200 per episode |
| Training GPUs | 128 × NVIDIA H20 |
| RL Algorithm | PPO (actor + critic) |

---

## Part 2: Data Synthesis Pipeline

### 2.1 CUDA-Agent-Ops-6K Dataset

**Source:** `BytedTsinghua-SIA/CUDA-Agent-Ops-6K` on HuggingFace

**Synthesis Process:**

1. **Operator Combination:** Combine 1-5 PyTorch operators
2. **Runtime Filtering:** Keep operators with 1ms < runtime < 100ms
3. **Anti-Hacking Checks:** Ensure diversity and prevent shortcuts
4. **Final Dataset:** 6,000 diverse operators

**Operator Distribution:**

| Category | Percentage | Example |
|----------|------------|---------|
| Single-op | ~16% | relu, sigmoid, matmul |
| 2-op fusion | 83.77% | matmul+relu, conv2d+bias+relu |
| 3+ op fusion | ~0.23% | layernorm+gelu+linear |

### 2.2 Data Quality Measures

**Anti-Hacking Filters:**

1. **Diversity Check:** Remove near-duplicate operators
2. **Runtime Bounds:** Filter trivially fast/slow operators
3. **Correctness Validation:** Verify PyTorch reference works
4. **Contamination Prevention:** Exclude operators from evaluation benchmarks

**Synthesis Algorithm:**

```python
def synthesize_operators():
    operators = []
    for num_ops in range(1, 6):  # 1 to 5 operators
        for combo in combinations(TORCH_OPS, num_ops):
            pytorch_code = build_operator(combo)
            runtime = benchmark(pytorch_code)
            if 1 < runtime < 100:  # milliseconds
                if passes_diversity_check(combo, operators):
                    operators.append({
                        "code": pytorch_code,
                        "ops": combo,
                        "runtime_ms": runtime,
                    })
    return operators[:6000]  # Cap at 6K
```

---

## Part 3: Skill-Augmented Environment

### 3.1 ReAct-Style Agent Loop

```
┌────────────────────────────────────────────────────────────┐
│                    Agent Environment                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ Observe │───>│ Think   │───>│ Act     │───>│ Reward  │──┐
│  │         │    │         │    │         │    │         │  │
│  │ Problem │    │ Reason  │    │ Code    │    │ Eval    │  │
│  │ Spec    │    │ Plan    │    │ Edit    │    │ Profile │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│       ↑                                              │     │
│       └──────────────────────────────────────────────┘     │
│                    (Multi-turn Loop)                       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 3.2 Environment Components

**Coding Tools:**
- Write CUDA kernel code
- Modify existing kernels
- Add compilation flags

**Compile-Debug Cycle:**
- nvcc compilation with error messages
- Automatic fix suggestions
- Iterative refinement

**Profiler-Guided Optimization:**
- Nsight Compute integration
- Memory throughput analysis
- Occupancy feedback

### 3.3 Verification Pipeline

**Multi-Input Verification:**

```python
def verify_kernel(kernel_code, reference_fn, num_inputs=5):
    """Test kernel against PyTorch reference on multiple inputs."""
    for seed in range(num_inputs):
        # Generate randomized input
        inputs = generate_inputs(seed)
        
        # Run kernel
        kernel_output = run_kernel(kernel_code, inputs)
        
        # Run reference
        ref_output = reference_fn(*inputs)
        
        # Compare
        if not allclose(kernel_output, ref_output, rtol=1e-3, atol=1e-3):
            return False, f"Mismatch on input {seed}"
    
    return True, "All inputs verified"
```

**Profiler Synchronization:**

```python
def profile_kernel(kernel_code, warmup=100, runs=50):
    """Synchronized profiling to prevent timing exploits."""
    # Warmup runs (prevent cold-start exploitation)
    for _ in range(warmup):
        run_kernel(kernel_code)
    
    cuda.synchronize()  # Ensure all work complete
    
    # Timed runs
    times = []
    for _ in range(runs):
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        run_kernel(kernel_code)
        end.record()
        end.synchronize()
        times.append(elapsed_time(start, end))
    
    return median(times)
```

---

## Part 4: Reward Function

### 4.1 Discrete Milestone Rewards

**Equation 1 from paper:**

```
r = -1  if compilation fails OR correctness fails
r = +1  if correct but no speedup
r = +2  if speedup > 1.05× vs eager
r = +3  if speedup > 1.05× vs eager AND vs compile
```

### 4.2 Why Discrete Beats Continuous

**Ablation Results (Table 2):**

| Reward Type | Pass Rate | Faster vs Compile | Speedup GM |
|-------------|-----------|-------------------|------------|
| Discrete milestones | 98.8% | 96.8% | 2.11× |
| Continuous speedup | 98.4% | 60.4% | 1.21× |

**Difference: +36.4 percentage points on faster-than-compile**

**Why Continuous Fails:**

1. **GPU Timing Noise:** Thermal throttling, OS interrupts, CUDA scheduler jitter
2. **Outlier Advantage Estimates:** Continuous rewards create noisy gradients
3. **Bias Toward Easy Kernels:** Where speedup variance is low
4. **Noisy Policy Updates:** Gradient estimates become unreliable

**Why Discrete Works:**

1. **Robust to Timing Noise:** Only cares about crossing 1.05× threshold
2. **Clear Signal:** {-1, 1, 2, 3} provides unambiguous feedback
3. **Stable Gradients:** Discrete rewards reduce variance
4. **Curriculum Effect:** Milestones naturally create difficulty progression

---

## Part 5: Four-Stage Training Pipeline

### 5.1 Pipeline Overview

```
Stage 1: Single-Turn PPO Warm-up (2 hours)
         ├── Goal: Bootstrap CUDA syntax
         ├── Data: Easy operators from Ops-6K
         └── Outcome: Compilation rate 50% → 85%+

         ↓

Stage 2: Rejection Fine-Tuning - RFT (30 minutes)
         ├── Goal: Create strong behavioral prior
         ├── Process: Filter trajectories, SFT on good ones
         └── Critical: Without RFT, training collapses at step 17

         ↓

Stage 3: Critic Value Pretraining
         ├── Goal: Stabilize PPO value estimates
         ├── Process: Pretrain critic on trajectory values
         └── Note: GRPO skips this (no critic)

         ↓

Stage 4: Multi-Turn Agentic PPO (150 steps, ~60% of compute)
         ├── Goal: Discover novel optimizations
         ├── Context: Up to 200 turns, 128K tokens
         └── Curriculum: Progressive difficulty
```

### 5.2 Stage 1: Single-Turn PPO Warm-Up

**Goal:** Bootstrap from "knows CUDA syntax" to "kernels that compile and pass correctness"

**Configuration:**

```python
ppo_config = PPOConfig(
    learning_rate=3e-6,
    num_steps=100,
    batch_size=32,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    entropy_coef=0.01,
    temperature=0.9,  # High exploration
)
```

**Curriculum within Stage 1:**
- Steps 0-30: Single torch ops (relu, sigmoid, matmul)
- Steps 30-60: 2-op compositions
- Steps 60-100: 3+ op fusions

**Expected Outcome:**
- Compilation rate: ~50% → ~85%+
- Average reward: ~0.0 → ~0.8-1.2

### 5.3 Stage 2: Rejection Fine-Tuning (RFT)

**Why Mandatory (Figure 4):**

Without RFT:
- Training reward collapses within ~15 steps
- Actor entropy spikes
- Policy becomes diffuse
- Faster-than-compile drops from 96.8% to 49.8%

**Process:**

```python
# 1. Collect trajectories from Stage 1 model
trajectories = []
for prompt in curriculum_prompts:
    for _ in range(4):  # Multiple samples per prompt
        completion = generate(model, prompt)
        result = env.evaluate(completion)
        trajectories.append({
            "prompt": prompt,
            "completion": completion,
            "reward": result.reward,
        })

# 2. Filter: keep reward >= 1.0 (correct)
good_trajectories = [t for t in trajectories if t["reward"] >= 1.0]

# 3. SFT on filtered data
sft_train(good_trajectories, epochs=3, lr=2e-5)
```

### 5.4 Stage 3: Critic Value Pretraining

**Purpose:** Stabilize PPO by pretraining the value function

**Process:**
1. Collect trajectories with rewards
2. Compute discounted returns
3. Pretrain critic network to predict returns
4. Freeze critic for initial PPO steps

**Note:** GRPO eliminates this stage entirely (no critic needed).

### 5.5 Stage 4: Multi-Turn Agentic PPO

**Configuration:**

```python
ppo_config = PPOConfig(
    learning_rate=5e-6,
    num_steps=150,
    max_turns=200,
    max_context_tokens=131072,
    temperature=0.7,  # Lower for exploitation
    clip_range=0.1,   # Tighter clipping
)
```

**Curriculum Progression:**

| Phase | Steps | Target | Example Optimizations |
|-------|-------|--------|----------------------|
| A | 0-40 | Single operators | Memory coalescing, shared memory tiling |
| B | 40-80 | 2-3 op fusions | Kernel fusion, eliminate intermediates |
| C | 80-120 | Complex operators | Algebraic simplification, library calls |
| D | 120-150 | Novel discoveries | Beyond training data |

---

## Part 6: Anti-Reward-Hacking Measures

### 6.1 Identified Hacking Strategies

| Strategy | Description | Prevention |
|----------|-------------|------------|
| Hardcoded Outputs | Return precomputed results | Multi-input verification |
| Input Detection | Branch on input values | Randomized inputs |
| Lazy Evaluation | Skip computation | Protected evaluation scripts |
| torch.compile Delegation | Call torch.compile | Forbidden symbol scanning |
| Triton Delegation | Use Triton primitives | Forbidden symbol scanning |
| Timing Exploitation | Detect profiling mode | Synchronized profiling |

### 6.2 Prevention Implementation

**Forbidden Symbols:**

```python
FORBIDDEN_SYMBOLS = [
    "torch",           # No PyTorch calls
    "at::Tensor",      # No ATen
    "c10::",           # No C10 dispatch
    "torch::autograd", # No autograd
    "triton",          # No Triton
    "torch.compile",   # No compile delegation
    "torch.nn.functional",  # No functional API
]
```

**Compilation Flag Whitelist:**

```python
ALLOWED_NVCC_FLAGS = {
    "--use_fast_math",
    "--extra-device-vectorization",
    "--rdc=true",
    "--maxrregcount=N",  # N in [16, 128]
}
```

**Protected Evaluation:**

```python
def protected_evaluate(kernel_code):
    """Run evaluation in isolated environment."""
    # 1. Compile in sandbox
    so_path = compile_sandboxed(kernel_code)
    
    # 2. Scan for forbidden symbols
    if scan_forbidden(so_path):
        return {"reward": -1, "error": "Forbidden symbol detected"}
    
    # 3. Run on randomized inputs
    inputs = generate_randomized_inputs(seed=random.randint())
    
    # 4. Profile with synchronization
    runtime = profile_synchronized(kernel_code, inputs)
    
    return {"runtime_ms": runtime}
```

---

## Part 7: Key Ablations

### 7.1 RFT Necessity (Table 2)

| Configuration | Pass Rate | Faster vs Compile | Speedup GM |
|---------------|-----------|-------------------|------------|
| Full System | 98.8% | 96.8% | 2.11× |
| w/o RFT | 98.4% | 49.8% | 1.21× |
| w/o Multi-turn | 97.2% | 78.4% | 1.67× |
| w/o Skill.md | 96.8% | 82.1% | 1.54× |

### 7.2 Reward Type (Table 2)

| Reward Type | Pass Rate | Faster vs Compile | Speedup GM |
|-------------|-----------|-------------------|------------|
| Discrete milestones | 98.8% | 96.8% | 2.11× |
| Continuous speedup | 98.4% | 60.4% | 1.21× |

### 7.3 Training Collapse Analysis

**Without RFT:**
- Step 0-10: Normal improvement
- Step 11-15: Plateau
- Step 16-17: Collapse begins
- Step 18+: Entropy explosion, policy diffusion

**Root Cause:**
- PPO requires good initialization
- Without RFT, actor explores too broadly
- Value function cannot keep up
- Policy gradient becomes noisy

---

## Part 8: KernelBench Results

### 8.1 Benchmark Overview

**KernelBench Levels:**
- Level-1: Single operators (100 problems)
- Level-2: 2-3 operator fusions (100 problems)
- Level-3: Complex multi-op fusions (100 problems)

### 8.2 CUDA Agent Performance

| Level | Pass Rate | Faster vs torch.compile | Speedup GM |
|-------|-----------|------------------------|------------|
| Level-1 | 100% | 100% | 2.34× |
| Level-2 | 100% | 100% | 2.18× |
| Level-3 | 98% | 92% | 1.87× |

### 8.3 Comparison with Other Models

| Model | Level-3 Faster Rate |
|-------|---------------------|
| CUDA Agent | 92% |
| Claude Opus 4.5 | 52% |
| Gemini 3 Pro | 48% |
| GPT-4o | 35% |
| DeepSeek-V3 | 30% |

**CUDA Agent outperforms by ~40 percentage points on hardest tasks.**

---

## Part 9: Notable Optimizations Discovered

### 9.1 Diagonal Matmul (73× speedup)

**Problem:** Matrix multiplication where one operand is diagonal

**Discovery:** CUDA Agent recognized the diagonal structure and eliminated unnecessary computation

```cuda
// Standard matmul: O(N^3)
// Diagonal-aware: O(N^2)

__global__ void diag_matmul(float* A, float* diag, float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float d = diag[row];
        for (int col = 0; col < N; col++) {
            C[row * N + col] = A[row * N + col] * d;
        }
    }
}
```

### 9.2 Memory Pattern Optimizations

**Discovered Patterns:**
- Shared memory double-buffering
- Coalesced memory access patterns
- Register blocking for small matrices
- Warp-level primitives for reductions

---

## Part 10: Implications for KernelForge

### 10.1 What to Adopt

| Component | CUDA Agent | KernelForge Adoption |
|-----------|------------|---------------------|
| Reward Function | Discrete {-1, 1, 2, 3} | ✅ Direct adoption |
| RFT Stage | Mandatory | ✅ Stage 2 RFT |
| Anti-Hacking | Full suite | ✅ Forbidden symbols, whitelist |
| Multi-turn | Up to 200 turns | ⚠️ Limited by single-GPU memory |
| Data Synthesis | 6K operators | ⚠️ Curated 200 subset |

### 10.2 What to Modify

| Component | CUDA Agent | KernelForge Modification |
|-----------|------------|-------------------------|
| RL Algorithm | PPO (actor + critic) | GRPO (actor only) — 50% memory savings |
| Model Size | 230B MoE | 80B MoE or 7B dense |
| GPUs | 128 × H20 | 1 × H100/B200 |
| Training Steps | 150 | 60 (reduced) |
| Context | 128K tokens | 5K tokens (single-turn) |

### 10.3 Efficiency Comparison

| Dimension | CUDA Agent | KernelForge |
|-----------|------------|-------------|
| Model | 230B MoE (23B active) | 80B MoE (~3.9B active) |
| RL algorithm | PPO (actor + critic) | GRPO (actor only) |
| GPUs | 128 × H20 | 1 × H100 80GB |
| Context | 131,072 tokens | 5,120 tokens |
| Agent turns | Up to 200 | 1 (single-turn) |
| Generations/step | ~32-64 | 4-8 |
| Training steps | 150 | 60 |
| FLOPs per sequence | ~6.0 TFLOPs | ~0.039 TFLOPs |
| **FLOPs reduction** | — | **~150×** |

---

## Part 11: References

| Claim | Source | Location |
|-------|--------|----------|
| Pure RL collapsed at step 17 | CUDA Agent paper | Section 3.3, page 6 |
| CUDA tokens = 0.01% of pretraining data | CUDA Agent paper | Section 3.3 |
| RFT ablation: faster-than-compile drops to 49.8% | CUDA Agent paper | Table 2, page 8 |
| Discrete rewards +36.4pp over continuous | CUDA Agent paper | Table 2 |
| Full system: 98.8% pass, 96.8% faster, 2.11× GM | CUDA Agent paper | Table 2 |
| 4-stage pipeline | CUDA Agent paper | Figure 3, page 5 |
| Reward function {-1,1,2,3} | CUDA Agent paper | Equation 1, page 5 |
| Anti-hacking measures | CUDA Agent paper | Section 3.2 |
| 6K operator dataset | CUDA Agent paper | Section 3.1 |
| Diagonal matmul 73× speedup | CUDA Agent paper | Appendix D |

---

## Appendix: Code Examples

### A.1 Reward Function Implementation

```python
def compute_reward(compiled: bool, correct: bool, 
                   speedup_vs_eager: float, 
                   speedup_vs_compile: float) -> float:
    """CUDA Agent discrete milestone reward function."""
    if not compiled or not correct:
        return -1.0
    if speedup_vs_compile > 1.05:
        return 3.0
    if speedup_vs_eager > 1.05:
        return 2.0
    return 1.0
```

### A.2 Trajectory Collection for RFT

```python
def collect_rft_trajectories(model, env, num_prompts=200, samples_per_prompt=4):
    """Collect and filter trajectories for RFT."""
    trajectories = []
    
    for prompt in sample_prompts(num_prompts):
        for _ in range(samples_per_prompt):
            # Generate completion
            completion = model.generate(prompt, temperature=0.9)
            
            # Evaluate
            result = env.evaluate(completion)
            
            trajectories.append({
                "prompt": prompt,
                "completion": completion,
                "reward": result.reward,
                "compiles": result.compiles,
                "correct": result.correct,
                "speedup": result.speedup,
            })
    
    # Filter: keep only correct trajectories
    good = [t for t in trajectories if t["reward"] >= 1.0]
    
    return good
```

### A.3 Anti-Hacking Scanner

```python
def scan_forbidden_symbols(so_path: str) -> str | None:
    """Scan compiled shared library for forbidden symbols."""
    import subprocess
    
    proc = subprocess.run(
        ["nm", "-D", so_path],
        capture_output=True,
        text=True,
        timeout=5,
    )
    
    symbols = proc.stdout
    for forbidden in FORBIDDEN_SYMBOLS:
        if forbidden in symbols:
            return f"Forbidden symbol detected: {forbidden}"
    
    return None
```
