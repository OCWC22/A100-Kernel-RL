# KernelForge Documentation

## Active Documents (Source of Truth)

| Document | Lines | Purpose |
|----------|-------|---------|
| [KERNELFORGE_FINAL_PRD.md](KERNELFORGE_FINAL_PRD.md) | 1,336 | **Single source of truth.** Locked decisions, 4 components, complete task list, repo structure, all links. |
| [GRPO_DEEP_DIVE.md](GRPO_DEEP_DIVE.md) | 1,713 | GRPO algorithm math, B200 memory budget, TRL GRPOTrainer config, 3-stage training code, stacked mitigations (MARS, Nsight, CPPO, MASPO, grammar). |
| [skills/doublegraph_a100.md](skills/doublegraph_a100.md) | 325 | DoubleGraph engineering reference: 4-layer architecture, A100 kernel deep dives, CachePool, build pipeline, porting guide. |

## Quick-Reference CLAUDE.md Files

Per-directory skill files that auto-load with Claude. Distilled from PRD + GRPO Deep Dive.

| File | Lines | Content |
|------|-------|---------|
| [CLAUDE.md](CLAUDE.md) | 230 | Navigation hub + full project quick-reference (all directories) |
| [../training/CLAUDE.md](../training/CLAUDE.md) | 115 | 3-stage pipeline configs, curriculum, LoRA, stacked mitigations |
| [../openenv_env/CLAUDE.md](../openenv_env/CLAUDE.md) | 138 | OpenEnv interface, reward thresholds, GPU registry, anti-hack |
| [../evaluation/CLAUDE.md](../evaluation/CLAUDE.md) | 137 | Eval functions, PAC verify, profiling, hybrid eval strategy |
| [skills/CLAUDE.md](skills/CLAUDE.md) | 40 | Index for doublegraph_a100.md |

## Archive

All previous versions (PRD v1-v5, Unified Spec, Truth.md, research papers) are in [archive/](archive/).
