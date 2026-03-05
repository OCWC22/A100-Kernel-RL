# KernelForge Status

**Last Updated:** 2026-03-04

---

## Current Target Profile

| Parameter | Value |
|-----------|-------|
| Target GPU | A100 |
| Target Architecture | sm_80 |
| Modal App | kernelforge-a100 |
| Skill File | skill_a100.md |

---

## Implementation Status

### ✅ Fully Implemented (22 components)

| Component | File | Notes |
|-----------|------|-------|
| OpenEnv Environment | `openenv_env/kernel_forge_env.py` | Typed Action/Observation/State |
| Discrete Reward Function | `openenv_env/reward.py` | `{-1, 1, 2, 3}` milestones |
| Anti-Reward-Hacking | `openenv_env/anti_hack.py` | Forbidden symbols, CU_FLAGS whitelist |
| GPU CachePool | `openenv_env/cache_pool.py` | LRU with 8 entries |
| GPU Registry | `openenv_env/gpu_registry.py` | A100/H100/B200 specs |
| Skill Builder | `openenv_env/skill_builder.py` | Per-GPU SKILL.md generation |
| Modal evaluate_kernel | `modal_app.py` | Compile, verify, benchmark |
| Modal profile_baselines | `modal_app.py` | Compiled kernel + NetworkX fallback |
| Modal test_h100_features | `modal_app.py` | TMA, DSMEM, DPX detection |
| Stage 1: GRPO Warm-up | `training/stage1_warmup.py` | Multi-turn with curriculum |
| Stage 2: RFT | `training/stage2_rft.py` | Trajectory collection + SFT |
| Stage 3: GRPO Curriculum | `training/stage3_grpo.py` | 4-phase curriculum |
| Model Loader | `training/model_loader.py` | Primary/fallback paths |
| Curriculum Manager | `training/curriculum.py` | Promotion/demotion logic |
| Multi-turn Rollout | `training/multi_turn_rollout.py` | Agentic loop |
| RFT Filter | `training/rft_filter.py` | Modal evaluation integration |
| CUDA-Agent Integration | `training/cuda_agent_integration.py` | Ops-6K prompt loading |
| PAC Verification | `verification/pac_verify.py` | 5 adversarial graphs |
| Graph Generators | `verification/pac_verify.py` | RMAT, SBM, Erdős-Rényi |
| WCC Verifier | `verification/pac_verify.py` | 3 invariants |
| Curated 200 Dataset | `datasets/curated_200.jsonl` | Balanced difficulty |
| Ops-6K Full Dataset | `datasets/ops6k_full.jsonl` | Full CUDA-Agent dataset |

### ⚠️ Partially Implemented (6 components)

| Component | Issue | Resolution |
|-----------|-------|------------|
| Verification Scope | WCC-only, not multi-algorithm | Extend `pac_verify.py` |
| Baseline Quality | Single WCC baseline | Add per-operator baselines |
| RFT Generation | Deterministic fallback when no checkpoint | Run training pipeline |
| Demo Dashboard | Simulated widgets | Wire to live artifacts |
| Multi-turn Context | 200 max turns, but single-turn default | Accept for hackathon |
| Dataset Curation | Single baseline, not per-operator | Pre-compute baselines |

### ❌ Missing (8 components)

| Component | Impact | Priority |
|-----------|--------|----------|
| Trained SFT Checkpoint | Cannot run Stage 2 RFT | P0 |
| RFT Dataset | Cannot run Stage 3 GRPO | P0 |
| GRPO Checkpoint | No trained model | P0 |
| Reward Trajectory Logs | No evaluation history | P1 |
| Secondary Baseline Kernel | Cannot achieve reward=3 | P0 |
| General Kernel Verification | Cannot verify non-WCC | P0 |
| Sandbox Isolation | Security risk | P1 |
| Network Isolation | Anti-hacking gap | P1 |

---

## Hackathon Readiness: 70%

### P0 Tasks (Blocking)

- [ ] Run Stage 1 GRPO warm-up → generate checkpoint
- [ ] Run RFT filter → generate `datasets/wcc_rft.jsonl`
- [ ] Run Stage 3 GRPO → generate trained model
- [ ] Configure secondary baseline kernel for reward=3
- [ ] Extend verification for non-WCC kernels

### P1 Tasks (Recommended)

- [ ] Pre-compute per-operator baselines
- [ ] Wire demo to live training artifacts
- [ ] Add sandbox isolation
- [ ] Add network isolation

### P2 Tasks (Nice-to-have)

- [ ] Add A100 kernel templates in `kernels/`
- [ ] Create one-command runbook script

---

## Command Sequence

```bash
# Environment setup
export KERNELFORGE_TARGET_GPU=A100
export KERNELFORGE_TARGET_ARCH=sm_80
export KERNELFORGE_MODAL_GPU=A100
export KERNELFORGE_CUDA_ARCH=sm_80
export KERNELFORGE_MODAL_APP=kernelforge-a100
export KERNELFORGE_SKILL_FILE=skill_a100.md
export KERNELFORGE_SECONDARY_BASELINE_KERNEL=kernels/clustered_wcc_h100.cu

# Dependency sync
uv sync

# Deploy GPU backend
modal deploy modal_app.py

# Training pipeline
uv run python training/stage1_warmup.py
uv run python training/rft_filter.py
uv run python training/stage2_rft.py
uv run python training/stage3_grpo.py

# Evaluation
uv run python evaluation/eval_model.py
```

---

## What Works Now

1. **uv-first project setup** — `uv lock`, `uv sync`, `uv run` paths wired
2. **OpenEnv runtime compatibility** — Typed environment with Modal dispatch
3. **Reward path uses real Modal evaluation** — GRPO/RFT pass baseline timings
4. **CUDA-Agent prompt integration** — Ops-6K prompts in GRPO/RFT pools
5. **A100 defaults configured** — Training scripts default to A100/sm_80
6. **Baseline profiling** — Compiled kernel first, measured CPU fallback
7. **RFT generation** — Real checkpoint generation with deterministic fallback

---

## What's Partially Working

1. **Baseline quality** — Primary baseline compiled, secondary optional
2. **RFT robustness** — Works with checkpoint, falls back to template
3. **Demo realism** — Backend real, frontend partially simulated

---

## What Does NOT Work End-to-End

1. **No trained SFT checkpoint in repo** — Need successful Stage 1 run
2. **No RFT dataset artifact** — Need successful RFT filter run
3. **No GRPO run result** — Need at least one completed training
4. **Demo not fully wired** — Partially illustrative

---

## Architecture Overview

```
KernelForge-OpenEnv/
├── skill_a100.md              # Agent prompt (default)
├── modal_app.py               # GPU backend (A100 default)
├── openenv_env/               # OpenEnv environment
│   ├── kernel_forge_env.py    # Core environment
│   ├── reward.py              # Discrete milestones
│   ├── anti_hack.py           # Forbidden symbols
│   ├── cache_pool.py          # LRU GPU cache
│   ├── gpu_registry.py        # GPU specs
│   └── skill_builder.py       # SKILL.md generator
├── training/                  # Training pipeline
│   ├── stage1_warmup.py       # GRPO warm-up
│   ├── stage2_rft.py          # RFT + SFT
│   ├── stage3_grpo.py         # GRPO curriculum
│   ├── model_loader.py        # Model loading
│   ├── curriculum.py          # Phase manager
│   └── rft_filter.py          # Trajectory filter
├── verification/              # PAC verification
│   └── pac_verify.py          # WCC verification
├── datasets/                  # Training data
│   ├── curated_200.jsonl      # Curated problems
│   └── ops6k_full.jsonl       # Full Ops-6K
├── docs/                      # Documentation
│   ├── KernelForge_Truth.md   # Single source of truth
│   └── KernelForge_Implementation_Spec.md
└── tests/                     # Test suite
```

---

## Key Evidence References

| Claim | Source | Location |
|-------|--------|----------|
| Pure RL collapsed at step 17 | CUDA Agent paper | Section 3.3, page 6 |
| CUDA tokens = 0.01% of pretraining | CUDA Agent paper | Section 3.3 |
| RFT ablation: +36.4pp faster-than-compile | CUDA Agent paper | Table 2 |
| Discrete rewards beat continuous | CUDA Agent paper | Table 2 |
| 4-way dispatch pattern | DoubleGraph SKILLS.md | Section 3 |
| CachePool 8 entries | DoubleGraph SKILLS.md | Section 3 |
| BFS TD→BU threshold N/20 | DoubleGraph SKILLS.md | Section 4.2 |
| GRPO eliminates critic | DeepSeek-R1 paper | Architecture section |
