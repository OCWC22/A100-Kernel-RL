# docs/skills/ — Skill Document Index

## Files

| File | Role |
|------|------|
| `CUDA_AGENT.md` | CUDA-Agent environment/task prior |
| `DOUBLEGRAPH_A100.md` | DoubleGraph A100 kernel engineering prior |
| `SKYDISCOVER_ADAEVOLVE_EVOX.md` | AdaEvolve/EvoX search-system prior |
| `KERNELGYM_DR_KERNEL.md` | KernelGYM backend architecture and Dr. Kernel/TRLOO framing |
| `SIMPLY_DEEPMIND.md` | Google DeepMind Simply — registry configs, PPO/GRPO reference, agent-first design |

## Recommended Reading Order

1. **`CUDA_AGENT.md`**
   - understand the operator-task and evaluator framing
2. **`DOUBLEGRAPH_A100.md`**
   - understand A100-specific kernel engineering priors
3. **`KERNELGYM_DR_KERNEL.md`**
   - understand the lightweight OpenEnv wrapper over a KernelGYM-style backend
4. **`SKYDISCOVER_ADAEVOLVE_EVOX.md`**
   - understand the search hedge beyond the GRPO loop
5. **`SIMPLY_DEEPMIND.md`**
   - understand registry-pattern configs, PPO/GRPO loss reference, agent-first design from DeepMind

## Current Repo Stance

These skill docs support the current KernelForge architecture:

- custom lightweight OpenEnv wrapper for the environment surface
- `TaskPool` sampling on `reset()` with task-specific reference code in observations
- discrete milestone reward `{-1, 1, 2, 3}` with anti-hack checks (Dr. Kernel-inspired)
- max_turns=3 (matching Dr. Kernel MAX_TURN)
- KernelGYM-style backend beneath that wrapper
- CoreWeave/Northflank as the primary eval path
- Modal as fallback only
- SkyDiscover integration as test-time search hedge (`skydiscover_integration/`)

**Single-source-of-truth for environment design:** See `docs/SYSTEM_TRUTH.md`.

External references:
- [OpenEnv overview](https://meta-pytorch.github.io/OpenEnv/)
- [TRL OpenEnv integration](https://huggingface.co/docs/trl/en/openenv)
- [KernelGYM repo](https://github.com/hkust-nlp/KernelGYM)
- [Dr. Kernel paper](https://arxiv.org/abs/2602.05885)
- [SkyDiscover repo](https://github.com/skydiscover-ai/skydiscover)
- [AdaEvolve paper](https://arxiv.org/abs/2602.20133)
- [EvoX paper](https://arxiv.org/abs/2602.23413)
- [Simply repo](https://github.com/google-deepmind/simply)
