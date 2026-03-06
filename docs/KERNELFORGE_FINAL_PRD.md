# KernelForge: Hackathon PRD
## Single Source of Truth for the OpenEnv / Cerebral Valley / PyTorch / Unsloth Hackathon
**Hackathon:** OpenEnv Hackathon SF — March 7–8, 2026 (SHACK15, San Francisco)
**Training GPU:** NVIDIA H100 via Modal ($3.95/hr) | **Eval GPU:** NVIDIA A100 80GB via Modal ($2.50/hr) | **Target Kernels:** NVIDIA A100 (`sm_80`)
**Primary Model:** Qwen3-Coder-30B-A3B-Instruct (30.5B total, 3.3B active, 128 experts / 8 active, 256K context)
**Fallback Model:** Qwen3.5-35B-A3B
**Last Updated:** March 5, 2026

## Executive Summary

KernelForge for the hackathon is **not** a claim that we will fully reproduce CUDA-Agent.

KernelForge for the hackathon **is**:
1. a low-cost, OpenEnv-compatible CUDA kernel optimization environment,
2. a real compile / correctness / timing feedback loop targeting **A100 kernels**,
3. a practical training + search stack using **Qwen3-Coder-30B-A3B-Instruct on H100** and **A100 for evaluation**, and
4. a foundation for later scaling into richer multi-turn RL, larger models, distillation, and future GPU targets.

The hackathon objective is to prove three things:
- the environment works end to end,
- the model can improve kernel candidates under real feedback,
- the system is reusable for larger research later.

The long-term objective remains the same: one-GPU, data-efficient CUDA kernel optimization infrastructure that can later scale to H100 / B200 / B300 and distill learned optimization behavior into smaller or cheaper models.

## RL Feasibility (Hackathon-Scoped)

Full CUDA-Agent replication is **not** the hackathon target. The hackathon target is a smaller, credible pilot:
- A100-targeted kernels with execution-based correctness and timing-based reward
- H100 for model generation and gradient updates via Modal
- A100 80GB for all performance measurement via Modal
- SFT warmup followed by limited-step GRPO pilot

The main bottleneck is evaluation throughput, not model VRAM. Even with a lightweight model, reward collection on remote A100 hardware accumulates latency. With G=2 and short rollout horizons, the compute budget is manageable under $200 total.

The repo should aim to prove:
1. the environment works (compile → correctness → timing → reward),
2. the model can improve candidates under real feedback,
3. the infra is reusable for future scaling (larger models, deeper RL, more tasks).

We are not claiming CUDA-Agent-level generalization or KernelBench reproduction. Those remain future research targets.

---

## 0. Locked Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Target kernels | **A100 `sm_80`** | The optimization target is A100 behavior, so performance reward must be measured on A100 rather than on a different architecture. |
| Training GPU | **H100** ($3.95/hr via Modal) | For the hackathon, H100 is the practical high-end training choice under budget. Qwen3-Coder-30B-A3B-Instruct fits comfortably on 80GB. Cheaper than B200 ($6.25/hr). |
| Eval GPU | **A100 80GB** via Modal ($2.50/hr) | **Hard requirement:** any reward involving speedup/runtime MUST execute on A100. H100-only eval is limited to: compile check, symbol scan, static analysis. |
| Primary model | **Qwen3-Coder-30B-A3B-Instruct** | Coding-agent model with 30.5B total / 3.3B active params, 48 layers, 128 experts / 8 active, 256K native context. Strong coding baseline without 80B-scale deployment burden. |
| Fallback model | **Qwen3.5-35B-A3B** | If primary model underperforms or has compatibility issues. |
| SkyDiscover LLM | GLM-5 via API ($1/M input) | Frontier 744B model for kernel evolution. ~$2-8/run. |
| Hackathon training path | **SFT warmup first, then small-budget GRPO pilot** | Safest way to get a working demo under budget and time pressure. |
| Parallel hedge | **SkyDiscover / evolutionary search** | Search gives a second route to demo results even if RL underperforms. |
| RL library | **TRL + OpenEnv** | TRL officially supports OpenEnv integration. TRLOO N/(N-1) correction applied via `TRLOOGRPOTrainer`. |
| Environment spec | OpenEnv (step/reset/state) | Hackathon framework requirement. |
| Kernel language | CUDA C++ | Aligns with CUDA Agent infra + doubleGraph patterns. |
| Long-term scale-up option | Qwen3-Coder-Next 80B / B200 | Keep as future work, not the primary hackathon path. |

### Training GPU Selection

Qwen3-Coder-30B-A3B-Instruct is an MoE model with 30.5B total parameters but only 3.3B active. In FP8, model weights are ~16-18GB. Training overhead (LoRA + optimizer + KV cache + activations) adds ~10-20GB depending on context length and G.

| GPU | VRAM | Model (FP8) | Free for Training | $/hr (Modal) | Verdict |
|-----|------|-------------|-------------------|-------------|---------|
| **H100** | 80 GB HBM3 | ~16-18 GB | **~60 GB** | **$3.95** | **Primary — ample headroom, best $/hr for hackathon** |
| B200 | 192 GB HBM3e | ~16-18 GB | ~174 GB | $6.25 | Overkill for 30B model; reserved for future 80B scale-up |
| A100 80GB | 80 GB HBM2e | ~16-18 GB | ~60 GB | $2.50 | **Eval only** — target architecture for kernel performance |

**H100 training overhead budget (~60GB free):**

| Component | Size | Notes |
|-----------|------|-------|
| LoRA adapters (r=16) | ~0.2 GB | Tiny — only trains adapter weights |
| Optimizer (paged_adamw_8bit) | ~1.5 GB | 8-bit Adam states for LoRA params only |
| KV cache (8K context, G=2) | ~4-6 GB | Conservative context for hackathon pilot |
| Activations + gradient checkpointing | ~8-12 GB | With gradient checkpointing enabled |
| **Total** | **~14-20 GB** | **Fits in 60GB with ~40-46GB margin** |

## 1. Strategic Framing

### Hackathon-first objective
Build the strongest possible submission for OpenEnv / Cerebral Valley / PyTorch / Unsloth under a real compute budget (**<$200 total**).

### Long-term research objective
Turn this into a reusable research platform for one-GPU CUDA kernel optimization, distillation, and later scaling across H100 / B200 / B300.

### What we are NOT claiming
We are not claiming:
- full CUDA-Agent reproduction,
- benchmark parity with CUDA-Agent,
- single-GPU SOTA,
- guaranteed KernelBench-level generalization.

Those are future targets, not current facts.

---

## 2. What We're Building

### 2.1 Hackathon Deliverable

A practical, budgeted system with five parts:

> **Training vs Eval GPU Split (First Principles):** We train on **H100** ($3.95/hr, 80GB) for model weights, generation, and gradient updates. **All performance reward must execute on A100** (via Modal). Cross-compiling `nvcc -arch=sm_80` on H100 is fine for syntax checking, but measuring runtime/occupancy/memory behavior on Hopper would optimize for the wrong hardware. Training GPU eval is limited to: compile check, symbol scan, `ptxas` static analysis. Any reward involving speedup, correctness verification, or Nsight profiling requires A100 execution.

**Component A — OpenEnv-compatible kernel environment**
- `step() / reset() / state()` contract
- prompt → candidate kernel → compile → correctness → timing → reward
- stable enough for judges to run

**Component B — Real evaluation harness (CUDA Agent pipeline)**
- CUDA-Agent-Ops-6K dataset (~6,000 PyTorch operator tasks as upstream data source; ~19 tasks are live-evaluable in the hackathon pilot)
- Compile → verify → profile scripts from their agent_workdir
- CUDA compile wrapper + anti-hack checks + execution-based correctness + A100 timing
- Discrete milestone rewards: `{-1, 1, 2, 3}` matching CUDA-Agent's ablation-proven scheme
- Floor baseline = torch.compile timing for Ops-6K tasks
- Source: https://github.com/BytedTsinghua-SIA/CUDA-Agent

**Component C — Small-budget model training loop**
- Qwen3-Coder-30B-A3B-Instruct on H100
- SFT warmup on curated examples (doubleGraph patterns + generated samples)
- Short GRPO pilot after SFT (G=2, discrete milestone rewards {-1, 1, 2, 3}, limited steps)
- TRLOO N/(N-1) correction via `TRLOOGRPOTrainer`
- CUDA-Agent SKILL.md verbatim + doubleGraph pattern paste as prompt context
- Hybrid eval: H100 local nvcc compile check for fast-fail, Modal A100 for all performance + correctness reward

**Component D — Search-based hedge (SkyDiscover / evolutionary search)**
- AdaEvolve + EvoX for kernel evolution via GLM-5 API
- Independent path to improved kernels even if RL is unstable
- 80-120 evolutions on seed .cu files
- Parallel track: runs on separate machine while GRPO trains (no GPU contention)
- Source: https://github.com/skydiscover-ai/skydiscover

**Component E — Demo / presentation layer**
- Clear architecture explanation
- Real metrics from actual A100 evaluation
- Reproducible workflow
- Grounded claims (see Claim Discipline section)

**doubleGraph Expert Baselines** (used as training data + prompt context, not runtime dependency)
- 192 A100-optimized CUDA kernels for graph algorithms (3.6x avg speedup over cuGraph)
- Core algorithms: BFS, Louvain, PageRank, WCC, Triangle Count
- Key A100 (SM80) patterns extracted for SKILL.md prompt context
- Source: https://github.com/double-ai/doubleGraph
- Full engineering reference: `docs/skills/doublegraph_a100.md` (11-section deep dive)

### 2.2 Long-Term Research Platform

After the hackathon, KernelForge should expand into:
- richer multi-turn RL with deeper rollout horizons,
- broader task coverage beyond the hackathon slice,
- stronger reward shaping and profiling rewards (Nsight ncu structured bonuses),
- distillation of learned kernel optimization behavior into smaller models,
- future hardware targets including H100 / B200 / B300,
- more advanced search and curriculum systems,
- larger models (Qwen3-Coder-Next 80B on B200).

---

## 3. Scope Boundaries

### 3.1 Hackathon scope
We will try to prove:
1. the OpenEnv environment is real and usable,
2. the reward is based on actual compile / correctness / timing behavior,
3. the model can improve kernel candidates under this loop,
4. the system can produce demo-worthy results under budget.

### 3.2 Explicitly out of scope for the hackathon
- full CUDA-Agent reproduction,
- large-scale 150-turn RL as a required path,
- broad benchmark claims across thousands of tasks,
- claiming parity with papers we did not reproduce end to end.

---

## 4. Training Strategy

### Phase 0 — Environment validation (highest priority)
Before any training:
- compile path must work,
- correctness path must work,
- A100 timing path must work,
- reward must not be hackable.

### Phase 1 — SFT warmup
Use curated generated examples, doubleGraph-inspired patterns, existing seed kernels, and any validated kernel-response pairs. Goal: improve compile rate, improve structural kernel quality, reduce dead-on-arrival GRPO steps.

### Phase 2 — Small-budget GRPO pilot
Run only after SFT. Primary purpose: validate real learning signal, not full-scale benchmark domination.
- `G=2`, short context, conservative completion length
- discrete milestone rewards {-1, 1, 2, 3}, execution-based correctness
- strict abort criteria

### Phase 3 — Search hedge / candidate selection
Run search in parallel or afterward:
- mutate / evolve candidate kernels (SkyDiscover / AdaEvolve),
- keep best-of-N,
- use final A100 eval for selection.

### Reward shape
Hackathon reward should be simple and defensible:

```python
reward = -1   # if not compiles or not correct
reward = 1    # compiles and correct, but no speedup
reward = 2    # faster than baseline
reward = 3    # significant speedup (>2x or within 10% of expert)
```

Optional structured bonus can be added only if actually implemented and tested:
- occupancy bonus, memory-efficiency bonus, warp-efficiency bonus.

Do not present these bonuses as core unless they are wired end to end.

---

## 5. Hackathon Configuration

### Model
- `Qwen3-Coder-30B-A3B-Instruct`

### Hardware
- H100 for generation + updates ($3.95/hr)
- A100 80GB for evaluation ($2.50/hr)

### RL posture
- SFT warmup first
- GRPO pilot second (only if SFT shows decent compile rate)
- shallow rollout horizon — avoid pretending deep multi-turn is mandatory on day one

### Candidate generation
- `G=2`
- short outputs
- best-of-N final selection

### Abort conditions
Abort or de-escalate GRPO if:
- compile rate stays too low after SFT,
- reward variance collapses,
- correctness never stabilizes,
- eval latency dominates the budget.

If that happens: ship SFT + search + environment. Do not force a fake RL story.

---

## 6. Realistic Compute Budget

The plan must stay under **$200** total.

| Budget bucket | Target |
|---------------|--------|
| H100 SFT + GRPO pilot | $60–100 |
| A100 evaluation | $40–70 |
| Buffer / retries / smoke tests | $20–40 |

This is feasible with current Modal pricing: H100 at $0.001097/sec ($3.95/hr), A100 80GB at $0.000694/sec ($2.50/hr).

Do not lock the PRD to fake exact step counts unless they are explicitly labeled as estimates.

---

## 7. Claim Discipline

This section is mandatory for the hackathon.

In demos, docs, and pitches, separate four categories:

| Category | Meaning |
|----------|---------|
| **Implemented** | Code exists |
| **Tested** | We ran it successfully |
| **Benchmarked** | We have measured results |
| **Projected** | Reasonable expectation, not yet proven |

Rules:
- Only benchmark claims should sound definitive.
- Projected outcomes must be labeled as targets or stretch goals.
- Theatrical framing is allowed.
- Fake certainty is not.

---

## 8. Repository Structure

```
kernelforge/
├── evaluation/                     # P0 — THE CORE
│   ├── __init__.py
│   ├── sandbox.py                  # Task 1: Subprocess isolation
│   ├── compiler.py                 # Task 2: nvcc -arch=sm_80 wrapper
│   ├── verifier.py                 # Task 3: Correctness checking
│   ├── profiler.py                 # Task 4: CUDA event timing
│   ├── reward.py                   # Task 5: Continuous Nsight + TRLOO reward
│   └── antihack.py                 # Task 6: Forbidden symbol scanning
│
├── data/
│   ├── __init__.py
│   ├── loader.py                   # Task 7: Load Ops-6K + doubleGraph tasks
│   ├── doublegraph_tasks.py        # Task 8: Extract graph algorithm tasks
│   ├── skill_a100.md               # Task 9: Agent SKILL.md context
│   ├── a100_patterns.md            # Task 10: doubleGraph optimization examples
│   └── sft_data.json               # Generated by Task 0.5
│
├── training/
│   ├── __init__.py
│   ├── run.py                      # Task 11: Main entry point (3-stage)
│   ├── stage1_warmup.py            # Task 12: GRPO warmup config
│   ├── stage2_rft.py               # Task 13: RFT filtering + SFT
│   └── stage3_grpo.py              # Task 14: GRPO pilot (H100 gen + A100 eval)
│
├── openenv_wrapper/                # P0 — hackathon judges test step() first
│   ├── __init__.py
│   ├── models.py                   # Task 15: Pydantic models
│   ├── environment.py              # Task 16: OpenEnv Environment class
│   ├── app.py                      # Task 17: FastAPI server
│   └── Dockerfile                  # Task 18: Container for HF Spaces
│
├── skydiscover_integration/        # P1 — parallel track ✅ IMPLEMENTED
│   ├── evaluator.py                # Task 19: KernelForgeEvaluator (cascade: stage1→stage2)
│   ├── run_evolution.sh            # Task 20: Launch script (--dry-run support)
│   └── initial_kernels/            # Task 21: Seed kernels from doubleGraph
│       └── wcc_doublegraph.cu      # Adapted WCC with extern "C" + zero-copy convergence
│
├── datasets/                       # Training data ✅ IMPLEMENTED
│   ├── build_combined_dataset.py   # Unified builder: manifest + Ops-6K → combined JSONL
│   ├── combined_kernelforge.jsonl  # Output: ~6,192 rows (192 doubleGraph + ~6K Ops-6K)
│   ├── extract_doublegraph_a100.py # Harvester + shared constants (ALGO_DISPLAY_NAMES, CATEGORY_META)
│   ├── doublegraph_sft.jsonl       # SFT messages format for Stage 2 (192 entries)
│   └── integrity.py                # Dataset validation checks
│
├── validation/                     # Task 0: All prep tests
│   ├── test_01_load_task.py
│   ├── test_02_profile_baseline.py
│   ├── test_03_compile_cuda.py
│   ├── test_04_subprocess_eval.py
│   ├── test_05_model_load.py
│   ├── test_06_grpo_init.py
│   ├── test_07_compilation_rate.py
│   └── test_08_full_grpo_step.py
│
├── tests/
│   ├── test_sandbox.py
│   ├── test_compiler.py
│   ├── test_verifier.py
│   ├── test_profiler.py
│   ├── test_reward.py
│   └── test_grpo_step.py
│
├── pyproject.toml
└── README.md
```

---

## 9. Complete Task List

### PHASE 0: PREP (Wednesday-Friday, before hackathon)

#### Task 0.1: Download All Assets
**Priority:** P0 BLOCKING — start first, downloads take hours
**Time:** 2-4 hours (parallel)
**Owner:** Anyone with internet

**Subtask 0.1.1: Download Qwen3-Coder-30B-A3B-Instruct (~18GB)**
```bash
mkdir -p ~/kernelforge && cd ~/kernelforge
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --local-dir ./models/qwen3-coder-30b
```
- Source: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
- Unsloth docs: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide

> **Future scale-up option:** Qwen3-Coder-Next 80B FP8 (~85GB) via `unsloth/Qwen3-Coder-Next-FP8-Dynamic` on B200. Not the hackathon default.

**Subtask 0.1.2: Download fallback model Qwen3.5-35B-A3B (~20GB)**
```bash
huggingface-cli download Qwen/Qwen3.5-35B-A3B --local-dir ./models/qwen35-35b
```
- Source: https://huggingface.co/Qwen/Qwen3.5-35B-A3B

**Subtask 0.1.3: Download CUDA-Agent-Ops-6K dataset**
```bash
python -c "
from datasets import load_dataset
ds = load_dataset('BytedTsinghua-SIA/CUDA-Agent-Ops-6K')
ds.save_to_disk('./data/ops_6k')
print(f'Downloaded {len(ds[\"train\"])} tasks')
"
```
- Source: https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K
- Format: Each row has `ops` (operation names), `data_source` (difficulty), `code` (complete Python module with Model class + get_inputs() + get_init_inputs())

Verify doubleGraph research manifest (metadata source of truth):
```bash
python -c "
import json
from pathlib import Path

path = Path('docs/research/doublegraph/doublegraph_a100_manifest.jsonl')
records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
print(f'doubleGraph manifest kernels: {len(records)}')
assert len(records) == 192
"
```

**Subtask 0.1.4: Clone all repos**
```bash
cd ~/kernelforge
git clone https://github.com/BytedTsinghua-SIA/CUDA-Agent
git clone https://github.com/double-ai/doubleGraph
git clone https://github.com/skydiscover-ai/skydiscover
git clone https://github.com/meta-pytorch/OpenEnv
```

**Subtask 0.1.5: Download doubleGraph A100 wheels** (STRETCH GOAL — skip if wheel incompatible)
```bash
# STRETCH GOAL: Wheel may not install on H100. If it fails, skip entirely.
# The 330-line docs/skills/doublegraph_a100.md provides all patterns needed for prompts.
mkdir -p ./wheels
wget -P ./wheels/ [A100_WHEEL_URL]  # Check https://github.com/double-ai/doubleGraph/releases/
```
- Source: https://github.com/double-ai/doubleGraph/releases/
- **Fallback:** Extract patterns into SKILL.md from docs/skills/DOUBLEGRAPH_A100.md (already complete)

**Done when:** All 5 subtasks complete. Verify with:
```bash
ls models/qwen3-coder-30b/  # Model files exist
ls models/qwen35-35b/       # Fallback model exists
python -c "from datasets import load_from_disk; print(len(load_from_disk('./data/ops_6k')['train']))"  # ~6000 (upstream data source; ~19 tasks live-evaluable in hackathon pilot)
ls CUDA-Agent/agent_workdir/utils/  # compile.sh, verification.py, profiling.py
ls doubleGraph/cpp/src/aai/impl/a100/  # CUDA kernel directories
ls skydiscover/  # SkyDiscover repo
```

---

#### Task 0.2: Set Up H100 Training Environment
**Priority:** P0 BLOCKING
**Time:** 1-2 hours

**Subtask 0.2.1: Install Python packages**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install --no-deps unsloth unsloth_zoo
pip install "trl[vllm]==0.29.0"  # Pin to version validated by Gate G-0.6 — do NOT upgrade without re-running gate
pip install transformers>=4.56.2 datasets accelerate peft
pip install openenv-core>=0.2.1
pip install cupy-cuda12x
```

**Subtask 0.2.2: Verify GPU and CUDA**
```bash
nvidia-smi                    # H100, 80GB
nvcc --version                # CUDA 12.x+
python -c "import torch; print(torch.cuda.get_device_name())"
python -c "import trl; print(trl.__version__)"  # Must match pinned version in pyproject.toml (validated by Gate G-0.6)
python -c "import unsloth; print('OK')"
```

**Subtask 0.2.2b: TRL wiring smoke test (P0, 15 min)**
```bash
# Verify reward function signature matches what TRL actually passes.
# TRL GRPO expects dataset column "prompt" (SINGULAR, not "prompts").
# Standardize all dataset builders to use "prompt" column.
# Reward funcs receive: prompts, completions, completion_ids, trainer_state, **kwargs
python -c "
from trl import GRPOConfig
# Verify remove_unused_columns=False keeps extra columns for reward **kwargs
config = GRPOConfig(output_dir='/tmp/test', remove_unused_columns=False, report_to='none')
print('TRL config OK, remove_unused_columns:', config.remove_unused_columns)
"
```

**Subtask 0.2.2c: Context length — start conservative**

Unsloth recommends smaller context for testing stability. Start with `max_seq_length=4096` (not 16384). Only increase after confirming no OOM and stable gradients at 4096.

**Subtask 0.2.3: Verify sm_80 cross-compilation**
```bash
echo '__global__ void test() {}' > /tmp/test.cu
nvcc -arch=sm_80 -c /tmp/test.cu -o /tmp/test.o && echo "sm_80 OK"
rm /tmp/test.cu /tmp/test.o
```

**Subtask 0.2.4: Install doubleGraph** (STRETCH GOAL)
```bash
pip install ./wheels/[DOUBLEGRAPH_A100_WHEEL].whl
python -c "import cugraph; print('doubleGraph loaded')"
```
STRETCH GOAL: If this fails, skip entirely. Dense ops + doubleGraph patterns from docs/skills/doublegraph_a100.md are sufficient. Runtime doubleGraph is NOT needed for Ops-6K reward calibration (those use torch.compile as floor).

**Done when:** All 4 subtasks pass.

---

#### Task 0.3: Inspect CUDA Agent's Evaluation Scripts
**Priority:** P0 — understanding these determines whether our wrapper works
**Time:** 2-3 hours
**Owner:** Lead engineer

**Subtask 0.3.1: Read and understand the evaluation scripts**
```bash
# Read each file. Take notes on:
# - Expected file paths and directory layout
# - How kernels are compiled (nvcc flags, output format)
# - How correctness is checked (tolerance, input generation)
# - How timing is measured (CUDA events? time.perf_counter?)
# - Output format (JSON? plain text? exit codes?)

cat CUDA-Agent/agent_workdir/SKILL.md
cat CUDA-Agent/agent_workdir/utils/compile.sh
cat CUDA-Agent/agent_workdir/utils/verification.py
cat CUDA-Agent/agent_workdir/utils/profiling.py

# CRITICAL: Copy these scripts VERBATIM into evaluation/
# Do NOT rewrite compile.sh, verification.py, or profiling.py.
# Wrap them with our subprocess isolation, but preserve their exact logic.
cp CUDA-Agent/agent_workdir/utils/compile.sh evaluation/cuda_agent_compile.sh
cp CUDA-Agent/agent_workdir/utils/verification.py evaluation/cuda_agent_verification.py
cp CUDA-Agent/agent_workdir/utils/profiling.py evaluation/cuda_agent_profiling.py
```

**Subtask 0.3.2: Test the scripts manually**
```bash
cd CUDA-Agent/agent_workdir

# Write a simple kernel
cat > kernels/kernel.cu << 'EOF'
#include <cuda_runtime.h>
__global__ void test(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 2.0f;
}
EOF

# Try compile script
bash utils/compile.sh

# Check: did it produce a .so? What flags does it use?
# Note any issues — this is what we're wrapping in our environment.
```

**Subtask 0.3.3: Document the interface**
Write a short note:
- What file does compile.sh expect? Where?
- What does verification.py output on success/failure?
- What does profiling.py output? What format?
- Can we call these scripts directly from our reward function?

**Done when:** You can manually compile a kernel, check correctness, and get timing using CUDA Agent's scripts.

---

#### Task 0.4: Extract doubleGraph A100 Patterns
**Priority:** P1 — enriches prompts and provides graph tasks
**Time:** 2 hours

**Subtask 0.4.1: Survey the A100 kernel directory**

Expected directory structure (from `docs/skills/doublegraph_a100.md`):
```
doubleGraph/cpp/src/aai/impl/a100/
├── traversal/           # BFS (bfs.cu + dispatch variants)
├── community/           # Louvain (louvain_f32.cu), Triangle Count (triangle_count.cu)
├── link_analysis/       # PageRank (pagerank.cu)
├── components/          # WCC (weakly_connected_components.cu)
├── link_prediction/     # Cosine similarity, etc.
└── *.cu.flags           # Per-kernel sidecar compiler flags
```

```bash
# Count kernels per algorithm category
for dir in doubleGraph/cpp/src/aai/impl/a100/*/; do
    count=$(find "$dir" -name "*.cu" | wc -l)
    echo "$(basename $dir): $count kernels"
done

# Also count .cu.flags sidecars
find doubleGraph/cpp/src/aai/impl/a100/ -name "*.cu.flags" | wc -l
```

**Subtask 0.4.2: Extract optimization patterns into a100_patterns.md**

Read the 5 primary A100 kernel files (see `docs/skills/doublegraph_a100.md` for full architecture):
```bash
# BFS: Direction-optimizing with dual frontiers, __ballot_sync warp aggregation
cat doubleGraph/cpp/src/aai/impl/a100/traversal/bfs.cu

# WCC: Parallel union-find with hook-sample/compress/hook-full, zero-copy convergence
cat doubleGraph/cpp/src/aai/impl/a100/components/weakly_connected_components.cu

# PageRank: Fused SpMV with warp-level __shfl_down_sync reduction
cat doubleGraph/cpp/src/aai/impl/a100/link_analysis/pagerank.cu

# Louvain: 3-tier dispatch (serial / thread-per-vertex / warp-per-vertex shared-mem hash)
cat doubleGraph/cpp/src/aai/impl/a100/community/louvain_f32.cu

# Triangle Count: DAG orientation + bilateral set intersection with __ldg() prefetch
cat doubleGraph/cpp/src/aai/impl/a100/community/triangle_count.cu
```

Also read the infrastructure headers:
```bash
# compact_graph_t: CSR/CSC struct with degree-based 4-bin segmentation
cat doubleGraph/cpp/include/cugraph/aai/compact_graph.hpp

# CachePool: Thread-local LRU GPU memory reuse (capacity=8)
cat doubleGraph/cpp/include/cugraph/aai/cache_pool.hpp

# Check .cu.flags sidecars for per-kernel compiler flags
find doubleGraph/cpp/src/aai/impl/a100/ -name "*.cu.flags" -exec echo "=== {} ===" \; -exec cat {} \;
```

Create `data/a100_patterns.md` with code snippets showing:
- Degree-based 4-bin segmentation (High ≥1024, Mid 32-1023, Low 1-31, Isolated 0)
- 4-way dispatch pattern (base/seg/mask/seg_mask) — eliminates warp divergence at host level
- `__ballot_sync` warp-level atomic aggregation (BFS frontier expansion)
- Shared-memory hash tables with `atomicCAS` + linear probing (Louvain Tier 3, 128 entries)
- Warp-level `__shfl_down_sync` reduction for convergence checking (PageRank L1 norm)
- Parallel union-find with path-halving: `parent[v] = parent[parent[v]]` (WCC)
- Zero-copy convergence via `cudaHostAllocMapped` (WCC)
- DAG orientation + bilateral set intersection with `__ldg()` prefetch (Triangle Count)
- `CachePool` thread-local LRU pattern (`static int tag;` + `Cacheable` base class)
- `.cu.flags` sidecar per-kernel compiler flags (`--maxrregcount`, `--use_fast_math`, `--rdc=true`)

These get pasted into the SKILL.md prompt that the model sees.

**Subtask 0.4.3: Create graph algorithm task definitions**

```python
# doublegraph_integration/graph_tasks.py
"""
Extract graph algorithm tasks from doubleGraph for the environment.

Each task has:
  - algorithm_name: e.g., "pagerank", "bfs", "wcc"
  - cuGraph_reference: the original (slow) cuGraph implementation
  - doubleGraph_expert: the optimized (fast) A100 kernel
  - reward calibration: cuGraph time = floor, doubleGraph time = ceiling
"""
from pathlib import Path

def extract_graph_tasks(dgraph_path: str = "./doubleGraph") -> list[dict]:
    a100_dir = Path(dgraph_path) / "cpp" / "src" / "aai" / "impl" / "a100"
    tasks = []
    
    if not a100_dir.exists():
        print(f"WARNING: {a100_dir} not found. Clone doubleGraph first.")
        return tasks
    
    for category_dir in sorted(a100_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        for cu_file in sorted(category_dir.glob("*.cu")):
            expert_code = cu_file.read_text()
            tasks.append({
                "type": "graph",
                "name": f"graph/{category_dir.name}/{cu_file.stem}",
                "category": category_dir.name,    # e.g., "components", "centrality"
                "algorithm": cu_file.stem,          # e.g., "weakly_connected_components_seg_mask"
                "expert_cuda": expert_code,          # The known-good A100-optimized kernel
                "expert_cuda_preview": expert_code[:3000],  # For prompt context
                "source": "doublegraph_a100",
                "has_expert_ceiling": True,           # We know how fast it CAN be
            })
    
    print(f"Extracted {len(tasks)} graph tasks from doubleGraph A100")
    return tasks
```

**Subtask 0.4.4: Profile cuGraph vs doubleGraph baselines (if doubleGraph installed)**

```python
# doublegraph_integration/baseline_profiler.py
"""
Profile cuGraph (baseline) vs doubleGraph (ceiling) for each graph algorithm.
Produces a baselines file used for reward calibration.
"""
import time
import json

try:
    import cugraph
    import cudf
    DOUBLEGRAPH_AVAILABLE = True
except ImportError:
    DOUBLEGRAPH_AVAILABLE = False

def profile_graph_baselines(algorithms=None):
    """
    Profile cuGraph/doubleGraph baselines for reward calibration.

    If doubleGraph wheel fails to install (H100 incompatible, A100/L4/A10G only),
    skip live profiling and use documented 3.6x average speedup estimates from
    docs/skills/doublegraph_a100.md patterns instead.

    WARNING: If doubleGraph monkey-patches or replaces cuGraph internals,
    the "cuGraph floor" timing may already be the optimized version.
    Run cuGraph timing in a clean env WITHOUT doubleGraph imported,
    then time doubleGraph separately. Otherwise floor ≈ ceiling.
    """
    if not DOUBLEGRAPH_AVAILABLE:
        print("WARNING: doubleGraph not installed — using estimated 3.6x baselines from docs")
        return _estimated_baselines()
    # Create a test graph (small — for quick profiling)
    import numpy as np
    n_vertices = 10000
    n_edges = 50000
    sources = np.random.randint(0, n_vertices, n_edges)
    destinations = np.random.randint(0, n_vertices, n_edges)
    
    gdf = cudf.DataFrame({'src': sources, 'dst': destinations})
    G = cugraph.Graph()
    G.from_cudf_edgelist(gdf, source='src', destination='dst')
    
    results = {}
    
    # PageRank
    for _ in range(5):  # warmup
        cugraph.pagerank(G)
    times = []
    for _ in range(20):
        start = time.perf_counter()
        cugraph.pagerank(G)
        times.append((time.perf_counter() - start) * 1000)
    results["pagerank"] = {
        "doublegraph_ms": sorted(times)[len(times)//4],  # Q1
        "estimated_cugraph_ms": sorted(times)[len(times)//4] * 3.6,
    }
    
    # BFS
    for _ in range(5):
        cugraph.bfs(G, start=0)
    times = []
    for _ in range(20):
        start = time.perf_counter()
        cugraph.bfs(G, start=0)
        times.append((time.perf_counter() - start) * 1000)
    results["bfs"] = {
        "doublegraph_ms": sorted(times)[len(times)//4],
        "estimated_cugraph_ms": sorted(times)[len(times)//4] * 3.6,
    }
    
    with open("data/graph_baselines.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Graph baselines saved to data/graph_baselines.json")
    return results

def _estimated_baselines():
    """Fallback when doubleGraph wheel is not installed."""
    # Use documented 3.6x average speedup from doubleGraph paper
    estimates = {
        "pagerank": {"doublegraph_ms": None, "estimated_cugraph_ms": None, "speedup_estimate": 3.6},
        "bfs": {"doublegraph_ms": None, "estimated_cugraph_ms": None, "speedup_estimate": 3.6},
        "wcc": {"doublegraph_ms": None, "estimated_cugraph_ms": None, "speedup_estimate": 3.6},
        "sssp": {"doublegraph_ms": None, "estimated_cugraph_ms": None, "speedup_estimate": 3.6},
    }
    with open("data/graph_baselines.json", "w") as f:
        json.dump(estimates, f, indent=2)
    print("Using estimated 3.6x baselines (doubleGraph not installed)")
    return estimates
```

**Done when:** `data/a100_patterns.md` exists with 5+ patterns. `graph_tasks.py` extracts tasks. Baseline profiling runs (or is skipped if doubleGraph not installed).

---

#### Task 0.5: Generate SFT Warmup Data
**Priority:** P0 — GRPO may fail without this
**Time:** 4-8 hours (run overnight Thursday)
**Cost:** ~$5-15 in GLM-5 API calls

```python
# data/generate_sft.py
"""
Generate SFT warmup data using GLM-5 API.
Produces working CUDA kernels for Ops-6K tasks.
Run overnight Thursday: nohup python data/generate_sft.py > sft.log 2>&1 &
"""
import os, json, subprocess, tempfile, re
from openai import OpenAI
from datasets import load_from_disk

client = OpenAI(
    base_url=os.environ.get("OPENAI_API_BASE", "https://api.z.ai/v1"),
    api_key=os.environ["OPENAI_API_KEY"],
)

# Load A100 patterns for few-shot context
patterns = ""
if os.path.exists("data/a100_patterns.md"):
    with open("data/a100_patterns.md") as f:
        patterns = f.read()[:2000]

ds = load_from_disk("./data/ops_6k")["train"]
tasks = [r for r in ds if "#1" in r["data_source"] or "#2" in r["data_source"]][:300]

sft_pairs = []
for i, task in enumerate(tasks):
    try:
        response = client.chat.completions.create(
            model="glm-5",
            messages=[{"role": "user", "content": (
                "Write a complete CUDA kernel (.cu file) for A100 (sm_80). Include:\n"
                "1. #include directives\n"
                "2. __global__ kernel function\n"
                "3. Host wrapper with extern \"C\"\n"
                f"A100 optimization patterns:\n{patterns}\n\n"
                f"Replace this PyTorch operation:\n```python\n{task['code']}\n```\n"
                "Write ONLY the .cu file."
            )}],
            max_tokens=4096, temperature=0.7,
        )
        cuda_code = response.choices[0].message.content
        
        # Strip markdown
        match = re.search(r'```(?:cuda|cpp)?\s*\n(.*?)```', cuda_code, re.DOTALL)
        if match:
            cuda_code = match.group(1)
        
        # Compile check
        with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
            f.write(cuda_code)
            cu = f.name
        result = subprocess.run(
            ["nvcc", "-arch=sm_80", "-O3", "-c", cu, "-o", "/dev/null"],
            capture_output=True, text=True, timeout=30
        )
        os.unlink(cu)
        compiles = result.returncode == 0
        
        sft_pairs.append({
            "ops": task["ops"], "task_code": task["code"],
            "generated_cuda": cuda_code, "compiles": compiles,
        })
        print(f"[{i+1}/{len(tasks)}] {'✓' if compiles else '✗'} {task['ops'][:50]}")
    except Exception as e:
        print(f"[{i+1}/{len(tasks)}] ERROR: {e}")

with open("data/sft_data.json", "w") as f:
    json.dump(sft_pairs, f, indent=2)

compilable = len([p for p in sft_pairs if p["compiles"]])
print(f"\nResults: {compilable}/{len(sft_pairs)} compile ({100*compilable/len(sft_pairs):.0f}%)")
```

**API Setup:**
- GLM-5: Sign up at https://api.z.ai → get API key
- Set: `export OPENAI_API_KEY=... OPENAI_API_BASE=https://api.z.ai/v1`
- Alternative: Any OpenAI-compatible API (Claude, GPT-5)

**Done when:** `data/sft_data.json` has 50+ compilable CUDA kernel examples.

---

#### Task 0.6: Test SkyDiscover
**Priority:** P1 — validates our parallel track
**Time:** 1-2 hours
**Cost:** ~$1-2

```bash
cd ~/kernelforge/skydiscover
uv sync --extra external
# NOTE: gpu_mode extra only covers ~4 benchmark tasks. Do NOT use for Ops-6K.

# Create test evaluator
cat > ../validation/sky_eval.py << 'PYEOF'
import subprocess, re, os, tempfile

def evaluate(program_path: str) -> dict:
    try:
        binary = program_path.replace(".cu", ".out")
        r = subprocess.run(["nvcc", "-arch=sm_80", "-O3", "-o", binary, program_path],
                           capture_output=True, text=True, timeout=45)
        if r.returncode != 0:
            return {"combined_score": -1e9, "artifacts": {"feedback": r.stderr[:500]}}
        r = subprocess.run([binary], capture_output=True, text=True, timeout=60)
        m = re.search(r"time:\s*([\d.]+)", r.stdout)
        score = 1.0 / float(m.group(1)) if m else 0.0
        os.unlink(binary)
        return {"combined_score": score, "artifacts": {"feedback": r.stdout[:500]}}
    except Exception as e:
        return {"combined_score": -1e9, "artifacts": {"feedback": str(e)}}
PYEOF

# Run 10 iterations
export OPENAI_API_KEY=your-key
export OPENAI_API_BASE=https://api.z.ai/v1

uv run skydiscover-run \
    ../skydiscover_integration/initial_kernels/gemm.cu \
    ../validation/sky_eval.py \
    --search evox --model glm-5 --iterations 10 --output ../validation/sky_test
```

**Done when:** Score improves over 10 iterations.

---

#### Task 0.7: Run All Validation Tests
**Priority:** P0 BLOCKING — must pass before hackathon
**Time:** 2-3 hours

Run tests 01-08 from the validation/ directory (see previous response for all test code). Each test validates one layer:

| Test | What It Validates | Must Pass? |
|------|-------------------|-----------|
| test_01_load_task.py | Can load Ops-6K, execute PyTorch model | YES |
| test_02_profile_baseline.py | Can profile torch.compile with CUDA events | YES |
| test_03_compile_cuda.py | Can compile .cu with nvcc -arch=sm_80 | YES |
| test_04_subprocess_eval.py | Subprocess isolation works (crash protection) | YES |
| test_05_model_load.py | Qwen3-Coder-30B-A3B-Instruct loads, generates CUDA | YES |
| test_06_grpo_init.py | TRL GRPOTrainer initializes without crash | YES |
| test_07_compilation_rate.py | Base model CUDA compilation rate (target: >50%) | INFORMATIONAL |
| test_08_full_grpo_step.py | Complete GRPO step with real evaluation | YES |

**In test_05, use Qwen3-Coder-30B-A3B-Instruct:**
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    max_seq_length=8192, load_in_4bit=False,
)
# Generate with temp=0.7, top_p=0.95, top_k=50
```

**Done when:** Tests 01-06 and 08 pass. Test 07 reports compilation rate.

---

### PHASE 1: BUILD (Friday, 8-10 hours of coding)

#### Task 1: `evaluation/sandbox.py` — Subprocess Isolation
**Priority:** P0 | **Time:** 30 min | **Depends on:** Nothing

Run any Python script in a subprocess. Return stdout/stderr/returncode. Handle timeouts and segfaults. ~60 LOC.

Key function: `run_in_sandbox(script: str, timeout: int) -> SandboxResult`

(Full implementation in Engineering PRD v2 — copy directly.)

**Test:** `pytest tests/test_sandbox.py` — basic execution, timeout, crash isolation, import error.

---

#### Task 2: `evaluation/compiler.py` — nvcc Wrapper
**Priority:** P0 | **Time:** 30 min | **Depends on:** Task 1

Compile CUDA source to .so with `nvcc -arch=sm_80 -O3 --use_fast_math -shared`. Parse `// CU_FLAGS:` comments (whitelisted flags only). ~80 LOC.

Key function: `compile_cuda(source: str, extra_flags: list) -> CompileResult`

(Full implementation in Engineering PRD v2.)

**Test:** Valid CUDA compiles. Invalid fails. CU_FLAGS parsing works. Dangerous flags rejected.

---

#### Task 3: `evaluation/verifier.py` — Correctness Checking
**Priority:** P0 | **Time:** 1 hour | **Depends on:** Tasks 1, 2

**Execution-based correctness is mandatory (P0).** CUDABench (arXiv 2603.02236) explicitly reports "mismatch between high compilation success and low functional correctness" — compile success alone is a bad proxy that enables reward hacking (model learns minimal kernels that compile but produce garbage). Verification must:
1. Check compiled .so for forbidden symbols via `nm -D`
2. **Run kernel on 5+ randomized inputs on A100** (via Modal)
3. **Compare outputs to PyTorch reference with tolerances** (rtol=1e-3, atol=1e-5 for fp32)
4. Fail hard on NaN, OOB, shape mismatch

Compile-only verification scores r=-1 (same as compile failure). Only execution-verified correct kernels get positive reward.

Key function: `verify_kernel(so_path: str, task_code: str) -> VerifyResult`

(Full implementation in Engineering PRD v2.)

---

#### Task 4: `evaluation/profiler.py` — Baseline Timing
**Priority:** P0 | **Time:** 1 hour | **Depends on:** Task 1

Profile torch.eager and torch.compile for a task on **A100 (Modal)**. CUDA events, 50 warmup, 30 runs, trimmed mean. **Must run on A100 — H100 timings are for the wrong architecture.**

Key function: `profile_task(task_code: str) -> ProfileResult` with `eager_ms` and `compile_ms`.

(Full implementation in Engineering PRD v2.)

---

#### Task 5: `openenv_env/reward.py` — GRPO Reward Function
**Priority:** P0 | **Time:** 1 hour | **Depends on:** Tasks 1-4

THE function passed to `GRPOTrainer(reward_funcs=...)`. Takes completions, extracts CUDA, dispatches to Modal A100 for execution-based correctness + speedup timing.

**2-tier reward architecture:**
- **Fast path (all rollouts):** CUDA events timing on A100 + execution correctness + cheap static metrics (ptxas register count, shared mem, occupancy estimate). Returns discrete milestone reward `{-1, 1, 2, 3}`.
- **Slow path (top-k only):** Full Nsight ncu on A100 for occupancy, mem_coalescing, warp_efficiency. Adds `0.4*occ + 0.3*mem + 0.2*warp` bonus.

TRLOO post-process: if Dr. Kernel derivation confirms N/(N-1) scaling, apply to advantages; otherwise run vanilla GRPO first with tight gates.

**Correctness gate:** `correct=False` → reward = -1.0 regardless of speedup. This prevents reward hacking where a fast-but-wrong kernel gets positive gradient signal. The reward function checks correctness BEFORE computing speedup.

**Modal `evaluate_kernel` return schema (locked):**
```python
# evaluate_kernel(payload) -> dict with these REQUIRED keys:
{
    "compiles": bool,           # nvcc compilation succeeded
    "correct": bool,            # execution-based correctness on 5+ random inputs vs PyTorch ref
    "speedup_vs_orig": float,   # speedup vs torch.eager baseline (0 if not measured)
    "speedup_vs_dg": float,     # speedup vs torch.compile/doubleGraph baseline (0 if not measured)
    "error": str,               # error message if compiles=False or correct=False, else ""
    # Optional (slow path / top-k only):
    "occupancy": float | None,  # SM occupancy from ncu (0.0-1.0)
    "mem_coalescing": float | None,  # memory coalescing efficiency (0.0-1.0)
    "warp_efficiency": float | None, # warp execution efficiency (0.0-1.0)
}
# Add a validation test: assert all required keys exist in eval result before computing reward.
```

Key function: `cuda_kernel_reward(prompts, completions, **kwargs) -> list[float]`

**Permissive signature:** TRL's internal plumbing for reward_funcs has changed across versions (completion_ids, trainer_state added in some). Use `**kwargs` to absorb any extra args TRL passes — don't enumerate them. Set `remove_unused_columns=False` in GRPOConfig to keep extra dataset columns (like task_code) reachable via `**kwargs`. For robustness, also embed task_code + metadata directly into the `prompt` column text (belt-and-suspenders: works even if column stripping changes). Dataset must have a `prompt` column (singular, not `prompts`).

(Full implementation in Engineering PRD v2.)

---

#### Task 6: `evaluation/antihack.py` — Anti-Reward-Hacking
**Priority:** P1 | **Time:** 30 min | **Depends on:** Nothing

Forbidden symbol list, CU_FLAGS whitelist, file size checks. Referenced by verifier.py.

From CUDA Agent Section 3.2:
- Forbidden: `torch`, `at::Tensor`, `c10::`, `triton`, `torch.compile`, `torch.nn.functional`
- Whitelist CU_FLAGS: `--use_fast_math`, `--maxrregcount=N` (16-128), `--rdc=true`

---

#### Task 7: `data/loader.py` — Dataset Loading
**Priority:** P0 | **Time:** 45 min | **Depends on:** Nothing

Load Ops-6K. Format prompts with SKILL.md. Filter by stage (warmup=single-op, curriculum=all). **Embed task_code + metadata directly into `prompt` column** for robustness (extra columns reach reward only if `remove_unused_columns=False`). Also set `remove_unused_columns=False` in GRPOConfig as belt-and-suspenders.

Key function: `load_training_dataset(stage: str) -> Dataset`

(Full implementation in Engineering PRD v2.)

#### Task 7b: `datasets/build_combined_dataset.py` — Combined Dataset Pipeline
**Priority:** P0 | **Time:** 1-2 hours | **Depends on:** Task 7, Task 22

Build a unified JSONL dataset that merges:
- doubleGraph A100 manifest (192 topology-aware graph kernels)
- CUDA-Agent Ops-6K (~6,000 PyTorch operator tasks as upstream data source; ~19 tasks are live-evaluable in the hackathon pilot)

Unified problem schema (curriculum-compatible):
- `prompt`, `ops`, `difficulty`, `data_source`
- `task_code` (Ops-6K), `topology` + `graph_properties` (doubleGraph)
- `kernel_id`, `compile_flags`, `expert_code` (optional)

Difficulty routing (for `CurriculumManager.add_problems()`):
- 1 → `single_ops`
- 2 → `fusion_2op`
- 3 → `arch_specific`
- 4 → `advanced`

Output artifact:
- `datasets/combined_kernelforge.jsonl` (~6,192 rows)

Implementation notes:
- Use manifest metadata at `docs/research/doublegraph/doublegraph_a100_manifest.jsonl` as source of truth for doubleGraph task generation.
- Reuse prompt builders from existing modules (`cuda_agent_integration.py`, `extract_doublegraph_a100.py`, `training/curriculum.py`) to avoid drift.

Integration companion:
- `training/dataset_loader.py` provides stage-aware loading (`stage1`, `stage2`, `stage3`) backed by the combined dataset.

---

#### Task 8: `data/doublegraph_tasks.py` — Graph Algorithm Tasks
**Priority:** P1 | **Time:** 1 hour | **Depends on:** Task 0.4

Load graph algorithm tasks from doubleGraph's A100 kernel directory. Each task includes the expert kernel (ceiling) and algorithm specification.

Key function: `load_graph_tasks() -> list[dict]`

(Implementation in Task 0.4 subtask 0.4.3.)

---

#### Task 9: `data/skill_a100.md` — Agent SKILL.md
**Priority:** P0 | **Time:** 30 min | **Depends on:** Task 0.4

The A100 optimization context given to the model in every prompt. Includes hardware specs, optimization priority order, doubleGraph patterns, and rules.

Content from Unified Spec Section 7. Add doubleGraph-extracted patterns from `a100_patterns.md`.

---

#### Task 10: `data/a100_patterns.md` — doubleGraph Examples
**Priority:** P1 | **Time:** 30 min | **Depends on:** Task 0.4

5-10 code snippets from doubleGraph's A100 kernels showing sm_80-specific optimizations. Used as few-shot examples in SKILL.md.

(Created during Task 0.4 subtask 0.4.2.)

---

#### Task 11: `training/run.py` — Main Entry Point
**Priority:** P0 | **Time:** 2 hours | **Depends on:** Tasks 1-7, 9

The main training script. Loads Qwen3-Coder-30B-A3B-Instruct with Unsloth, runs SFT warmup + GRPO pilot pipeline.

**Hackathon command:**

```python
# Default command (Saturday morning):
python -m training.run \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --output_dir ./checkpoints \
    --stage all

# If primary model has issues, fallback:
python -m training.run \
    --model Qwen/Qwen3.5-35B-A3B \
    --output_dir ./checkpoints \
    --stage all
```

Key settings for hackathon:
- Temperature: 0.9 (Stage 1), 0.7 (Stage 3)
- Learning rate: 3e-6 (Stage 1), 3e-6 (Stage 3)
- G=2 — conservative to save eval budget
- Max turns: 5 (Stage 1), 3-5 (Stage 3 hackathon pilot) — start simple, deepen only if budget allows
- Context: 8K — conservative for hackathon pilot
- Stage 1 may be skipped if compilation rate >70%

(Full implementation in Engineering PRD v2, with model loading updated per Model Update doc.)

---

#### Task 12-14: Stage Configs (Hackathon)
**Priority:** P0 | **Time:** Included in Task 11

Stage configs for SFT warmup and GRPO pilot on H100 + A100 eval.

| Stage | Steps | Temp | LR | Context | G | Max Turns |
|-------|-------|------|-----|---------|---|-----------|
| 1 (Warmup) | 300 | 0.9 | 3e-6 | 8,192 | 2 | 5 |
| 2 (SFT) | N/A (SFT 3 epochs) | 0.7 | 5e-6 | N/A | N/A | N/A |
| 3 (GRPO pilot) | **50 steps** (hackathon pilot) | 0.7 | **3e-6** | **8,192** | 2 | **3-5** |

---

#### Task 15-18: OpenEnv Wrapper
**Priority:** P0 — judges call `step()` first. Must be robust.
**Time:** 2-3 hours
**Depends on:** Tasks 1-5

Wrap `openenv_env/reward.py` in OpenEnv's `step()/reset()/state()` API. The core logic is identical — this is packaging.

**Critical: `step()` must be robust even if model loading or GPU setup is slow.** Cache the model on startup, warm the CUDA context (dummy kernel launch), and pre-load tokenizer. If Modal is slow on first call, return a timeout-safe response rather than crashing.

- Task 15: `models.py` — KernelAction, KernelObservation, StepOutcome Pydantic models
- Task 16: `environment.py` — KernelForgeEnvironment(Environment) class (warm cache in `__init__`)
- Task 17: `app.py` — FastAPI application (model + CUDA warmup in `@app.on_event("startup")`)
- Task 18: Dockerfile for HF Spaces deployment

---

#### Task 19-21: SkyDiscover Integration ✅ IMPLEMENTED
**Priority:** P1 — parallel track for demo numbers
**Time:** 2 hours
**Depends on:** Tasks 1-5
**Status:** All three tasks implemented. See `skydiscover_integration/`.

**Task 19:** `skydiscover_integration/evaluator.py` — wraps reward.py as SkyDiscover evaluator

**CRITICAL:** SkyDiscover evaluator MUST use A100 execution-based correctness + timing (fast path at minimum). Compile-only scoring will select garbage kernels that happen to compile — same reward hacking problem as GRPO (CUDABench arXiv 2603.02236).

```python
def evaluate(program_path: str) -> dict:
    with open(program_path) as f:
        cuda_code = f.read()
    # Use same A100 eval pipeline as GRPO reward (fast path: CUDA events + correctness)
    from openenv_env.reward import compute_reward
    import modal
    eval_fn = modal.Function.from_name("kernelforge-a100", "evaluate_kernel")
    result = eval_fn.remote({
        "cuda_code": cuda_code,
        "verify_graphs": 5, "warmup_iters": 50, "benchmark_runs": 30,
    })
    reward = compute_reward(
        compiled=result.get("compiles", False),
        correct=result.get("correct", False),
        speedup_vs_eager=result.get("speedup_vs_orig", 0),
        speedup_vs_compile=result.get("speedup_vs_dg", 0),
    )
    if reward <= -1.0:
        return {"combined_score": -1e9, "artifacts": {"feedback": result.get("error", "failed")[:500]}}
    return {"combined_score": float(reward), "artifacts": {"feedback": f"reward={reward:.3f}"}}
```

> **IMPLEMENTATION NOTE (March 5, 2026):** Actual implementation in `skydiscover_integration/evaluator.py` uses `KernelForgeEvaluator` class with:
> - `evaluate_stage1()`: local nvcc compile check (fast, ~50% filter)
> - `evaluate_stage2()`: full Modal A100 benchmark via `validate_eval_result()` + `compute_reward()`
> - `evaluate_program()`: async cascade (SkyDiscover-compatible)
> - `evaluate()`: sync file-based interface
> - `EvaluationResult` dataclass with `combined_score`, `metrics`, `artifacts`, `error`
> - Supports both WCC (`evaluate_kernel`) and Ops-6K (`evaluate_ops6k_kernel`) eval modes

**Task 20:** `run_evolution.sh` — launch script
```bash
#!/bin/bash
export OPENAI_API_KEY=${ZAI_API_KEY}
export OPENAI_API_BASE=https://api.z.ai/v1

cd ~/kernelforge/skydiscover
for kernel in ../skydiscover_integration/initial_kernels/*.cu; do
    name=$(basename $kernel .cu)
    echo "Evolving $name..."
    uv run skydiscover-run \
        "$kernel" \
        ../skydiscover_integration/evaluator.py \
        --search evox --model glm-5 \
        --iterations 80 \
        --output ../results/skydiscover/$name &
done
echo "All evolution jobs launched. Check results/skydiscover/"
```

**Task 21:** Write 3-5 initial CUDA kernels in `initial_kernels/`:
- `gemm.cu` — naive matrix multiplication
- `softmax.cu` — naive softmax
- `layernorm.cu` — naive layer normalization
- `fused_bias_relu.cu` — naive GEMM + bias + ReLU

> **IMPLEMENTATION NOTE (March 5, 2026):** Actual seed kernel is `wcc_doublegraph.cu` — adapted from doubleGraph's production WCC kernel (`components/weakly_connected_components.cu`). Uses `extern "C"` wrapper matching KernelForge eval interface (`wcc_kernel(row_ptr, col_idx, num_vertices, labels)`). Includes all A100 optimizations: zero-copy convergence flag (`cudaHostAlloc`), path-halving Union-Find (non-atomic), sampled hooking (k=2), `__launch_bounds__(256)`, `__ldg()` for read-only data. SkyDiscover evolves FROM optimized code, not FROM naive implementations.

---

#### Task 22-24: doubleGraph Integration ✅ IMPLEMENTED
**Priority:** P1 — expert baselines + graph tasks
**Time:** 3 hours
**Depends on:** Task 0.4, `docs/skills/doublegraph_a100.md` (SKILLS.md reference)
**Status:** Tasks 22-23 fully implemented, Task 24 partially done.

**Task 22:** `extract_patterns.py` — Extract A100 Patterns from DoubleGraph
- Parse the 4-layer architecture from the source tree:
  - Layer 4 (Integration): `cpp/src/aai/integration/` — template specialization with `#ifdef AAI_ROUTE_*`
  - Layer 3 (API Dispatch): `cpp/include/cugraph/aai/api/` — 4-way dispatch declarations
  - Layer 2 (Infrastructure): `cpp/include/cugraph/aai/` — `compact_graph_t`, `CachePool`
  - Layer 1 (Implementation): `cpp/src/aai/impl/a100/` — the `.cu` kernel files
- Extract code snippets for `data/a100_patterns.md`:
  - `compact_graph_t` struct with `segment_offsets` (degree-based 4-bin segmentation)
  - `CachePool` acquire/ensure pattern with `static int tag;`
  - 4-way dispatch host-side branching (base/seg/mask/seg_mask)
  - Per-algorithm A100 optimizations (see SKILLS.md Sections 6.1-6.5)
- Parse `.cu.flags` sidecar files to document per-kernel compiler flag patterns
- Identify RDC kernels (cooperative launches) separated into `cugraph_aai_rdc` static library

**Task 23:** `graph_tasks.py` — Format Graph Algorithm Tasks
- Create task definitions for 5 primary algorithms:
  - BFS (`traversal/bfs.cu`) — dual-frontier direction-optimizing
  - Louvain (`community/louvain_f32.cu`) — 3-tier adaptive dispatch
  - PageRank (`link_analysis/pagerank.cu`) — fused SpMV with warp reduction
  - WCC (`components/weakly_connected_components.cu`) — parallel union-find
  - Triangle Count (`community/triangle_count.cu`) — DAG orientation
- Each task includes: `expert_cuda` (the `.cu` source), algorithm category, dispatch variant
- Include `.cu.flags` content for each kernel (`--maxrregcount`, `--use_fast_math`, `--rdc=true`)

**Task 24:** `baseline_profiler.py` — Profile cuGraph vs doubleGraph Baselines
- Profile all 5 primary algorithms on test graphs of varying sizes
- Record per-algorithm speedup (not just the 3.6x average — speedups vary widely per algorithm)
- Test both masked and unmasked dispatch variants
- Store results for per-algorithm reward calibration:
  - cuGraph timing = floor, doubleGraph timing = ceiling
  - Handle doubleGraph-replaces-cuGraph issue: if doubleGraph is installed, `import cugraph` uses optimized version; profile doubleGraph directly and estimate cuGraph from known per-algorithm speedup ratios

These feed into:
- `data/loader.py` (Task 7) — adds graph tasks alongside dense ops
- `data/skill_a100.md` (Task 9) — includes doubleGraph patterns from `a100_patterns.md`
- `openenv_env/reward.py` (Task 5) — graph tasks use per-algorithm reward calibration:
  - r = -1: compilation fails OR correctness fails
  - r = +1: correct but slower than cuGraph baseline
  - r = +2: faster than cuGraph baseline
  - r = +3: within 10% of doubleGraph expert kernel timing
  - Per-algorithm thresholds computed from Task 24 profiling results

> **IMPORTANT (Fix 2):** The cuGraph floor / doubleGraph ceiling calibration above applies ONLY to the 5 graph algorithm tasks. For Ops-6K PyTorch operator tasks (GEMM, softmax, layernorm, etc.), doubleGraph has NO baselines. Those tasks use torch.compile timing as the floor, matching CUDA-Agent's approach. Ops-6K (~6,000 tasks) serves as the upstream data source; ~19 tasks are live-evaluable in the hackathon pilot. Do not conflate graph-task calibration with Ops-6K calibration.

> **IMPLEMENTATION NOTE (March 5, 2026):**
>
> **Task 22 — DONE** as `datasets/extract_doublegraph_a100.py` + `datasets/build_combined_dataset.py`: Source of truth is `docs/research/doublegraph/doublegraph_a100_manifest.jsonl` (192 kernels, all metadata). The combined builder reads the manifest, builds topology-aware prompts, and merges with Ops-6K into `datasets/combined_kernelforge.jsonl`. Active outputs:
> - `combined_kernelforge.jsonl` — unified dataset (~6,192 rows: 192 doubleGraph + ~6K Ops-6K)
> - `doublegraph_sft.jsonl` — 192 HF messages entries for Stage 2 SFT
> Legacy files (`doublegraph_a100_kernels.jsonl`, `doublegraph_grpo_prompts.jsonl`, `generate_wcc_dataset.py`) archived to `archive/datasets_legacy/`.
>
> **Task 23 — DONE** via `training/curriculum.py` Phase 2-3 additions: 5 topology-aware graph problems added:
> - Phase 2 (arch_specific): BFS (power-law), PageRank (dense-regular), WCC (sparse-islands)
> - Phase 3 (advanced): Triangle Count (dense-community), Louvain (dense-community)
> Each prompt describes physical graph topology + specific A100 optimization patterns + compilation flags.
>
> **Task 24 — PARTIALLY DONE**: Per-kernel compilation flags extracted and documented (all 192 .cu.flags files parsed). `_explain_flags()` provides hardware reasoning for each flag choice. Runtime profiling on Modal A100 deferred to hackathon day (requires deployed `modal_app.py`).
>
> **Additionally implemented:**
> - `openenv_env/skill_builder.py:_append_a100_patterns()` — 7 real expert patterns from doubleGraph production kernels injected into SKILL.md for A100 (degree-based dispatch, warp hash tables, zero-copy convergence, bitmap frontier, path-halving UF, compilation flag tuning, __launch_bounds__). Output: 178 lines / 7.5KB.

---

### PHASE 2: RUN (Saturday, hackathon day)

#### Task 25: Saturday Morning Decision Point
**Time:** 30 minutes

```bash
# Load model, test 20 kernels, check compilation rate
python validation/test_07_compilation_rate.py

# Result determines Stage 1:
#   ≥70% → Skip Stage 1, go to Stage 2 RFT
#   50-70% → Short Stage 1 (50 steps, ~1.5 hrs)
#   <50% → Full Stage 1 (100 steps, ~3.5 hrs)
```

> **SFT first:** Do NOT attempt GRPO until SFT warmup has run (Task 0.5 data + Stage 2 SFT). Verify >70% compile rate via local nvcc eval (no Modal) BEFORE attempting any GRPO steps.

#### Task 26: Launch SkyDiscover (Parallel Track)
**Time:** 2 hours setup, runs in background
```bash
bash skydiscover_integration/run_evolution.sh
```

#### Task 27: Launch GRPO Training
**Time:** Runs 7-12 hours in background
```bash
python -m training.run \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --stage all --output_dir ./checkpoints
```

#### Task 28: doubleGraph Comparison (if installed)
**Time:** 1-2 hours
```bash
python doublegraph_integration/baseline_profiler.py
```

### PHASE 3: SHIP (Saturday evening / Sunday)

#### Task 29: Collect Results
#### Task 30: Write Blog Post
#### Task 31: Build Demo
#### Task 32: Deploy to HuggingFace Spaces
#### Task 33: Record Demo Video

---

## 10. Critical Path

```
PREP (Wed-Fri):
  0.1 Downloads ──→ 0.2 Environment ──→ 0.3 CUDA Agent scripts ──→ 0.7 Validation
  0.4 doubleGraph ──→ a100_patterns.md + graph_tasks
  0.5 SFT data (overnight)
  0.6 SkyDiscover test

BUILD (Friday):
  Task 1 ──→ Task 2 ──→ Task 5 (reward.py) ──→ Task 11 (run.py)
              Task 3 ──↗
              Task 4 ──↗
  Task 7 (loader) ────────────────────────────↗
  Task 9 (SKILL.md) ─────────────────────────↗
  Task 8 + 10 (doubleGraph) ─────────────────↗

RUN (Saturday):
  Task 25 (decision) → Task 27 (GRPO, 7+ hrs)
  Task 26 (SkyDiscover, parallel)
  Task 28 (doubleGraph baselines, parallel)

SHIP (Saturday night):
  Tasks 29-33 (results, blog, demo, deploy)
```

---

## 11. All Links

### Repos
| Resource | URL |
|----------|-----|
| CUDA Agent | https://github.com/BytedTsinghua-SIA/CUDA-Agent |
| doubleGraph | https://github.com/double-ai/doubleGraph |
| SkyDiscover | https://github.com/skydiscover-ai/skydiscover |
| OpenEnv | https://github.com/meta-pytorch/OpenEnv |
| Unsloth | https://github.com/unslothai/unsloth |
| KernelBench | https://github.com/ScalingIntelligence/KernelBench |
| TRL | https://github.com/huggingface/trl |

### Models
| Model | URL |
|-------|-----|
| **Qwen3-Coder-30B-A3B-Instruct** (primary) | https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct |
| Qwen3.5-35B-A3B (fallback) | https://huggingface.co/Qwen/Qwen3.5-35B-A3B |
| Qwen3-Coder-Next (future scale-up) | https://huggingface.co/Qwen/Qwen3-Coder-Next |
| Qwen3-Coder-Next FP8 (future scale-up) | https://huggingface.co/unsloth/Qwen3-Coder-Next-FP8-Dynamic |
| GLM-5 (SkyDiscover API) | https://huggingface.co/zai-org/GLM-5 |

### Datasets
| Dataset | URL |
|---------|-----|
| CUDA-Agent-Ops-6K | https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K |
| doubleGraph A100 wheels | https://github.com/double-ai/doubleGraph/releases/ |

### Papers
| Paper | URL |
|-------|-----|
| **Qwen3-Coder-Next Technical Report** | https://arxiv.org/pdf/2603.00729 |
| CUDA Agent | https://arxiv.org/abs/2602.24286 |
| WarpSpeed/doubleGraph | https://doubleai.com/research/doubleais-warpspeed-surpassing-expert-written-kernels-at-scale |
| AdaEvolve (SkyDiscover) | https://arxiv.org/abs/2602.20133 |
| EvoX (SkyDiscover) | https://arxiv.org/abs/2602.23413 |
| MARS Credit Assignment | https://arxiv.org/abs/2510.15414 |
| GRPO (DeepSeekMath) | https://arxiv.org/abs/2402.03300 |

### Competitive Landscape (Feb-Mar 2026 — CUDA Kernel RL)

These concurrent papers directly challenge or supersede aspects of our approach. See Section 6.0 for analysis.

| Paper | arXiv | Key Finding | Implication for KernelForge |
|-------|-------|------------|----------------------------|
| **Dr. Kernel** (HKUST/TikTok) | [2602.05885](https://arxiv.org/abs/2602.05885) | GRPO has biased policy gradients; proposes TRLOO (Turn-level Reinforce Leave-One-Out) | **Fixed:** TRLOO N/(N-1) post-processing implemented in `custom_grpo_trainer.py`. With G=2, corrects 50% gradient shrinkage. |
| **CUDA-L1** (DeepReinforce) | [2507.14111](https://arxiv.org/abs/2507.14111) | Contrastive RL achieves 3.12x avg speedup on A100 KernelBench | Pairwise comparisons > group normalization. Alternative to GRPO. |
| **KernelBlaster** (Stanford/NVIDIA) | [2602.14293](https://arxiv.org/abs/2602.14293) | Memory-augmented in-context RL for kernel generation | In-context learning approach — no policy gradient training needed. |
| **OptiML** | [2602.12305](https://arxiv.org/abs/2602.12305) | MCTS over LLM edits, no RL training needed | Search-based alternative that's more sample-efficient than GRPO. |
| **ConCuR** | [2510.07356](https://arxiv.org/abs/2510.07356) | Data quality > algorithm; curated reasoning traces beat RL | SFT on quality data may beat our GRPO pipeline. |
| **CUDABench** | [2603.02236](https://arxiv.org/abs/2603.02236) | "Mismatch between high compilation success and low functional correctness" | Compilation-only reward (our current state) is a bad proxy for kernel quality. |

### Stacked RL Techniques (March 2026)

These papers provide the specific techniques that address each Fundamental Failure in Section 6.0, making GRPO viable on single-GPU. See GRPO_DEEP_DIVE sections GRPO-9 through GRPO-14.

| Paper | arXiv / Link | Role in Stack | Addresses |
|-------|-------------|---------------|-----------|
| ~~**MARSHAL / MARS**~~ | [2510.15414](https://arxiv.org/abs/2510.15414) | Turn-level cumulative advantage (+23% return, ICLR 2026) | ~~Failure #1 (GRPO bias)~~ **[DROPPED]** — degenerates with outcome rewards |
| **OAPL** | [2602.19362](https://arxiv.org/abs/2602.19362) | Off-policy advantage regression (handles 400-step async lag) | Failure #3 (async Modal eval) |
| **AReaL** | [2505.24298](https://arxiv.org/abs/2505.24298) | Async decoupled training (2.77x wall-time speedup) | Failure #3 (compute scale) |
| **MASPO** | [2602.17550](https://arxiv.org/abs/2602.17550) | Gaussian soft trust region replacing PPO clip (+2-3%) | Failure #1 (sparse reward collapse) **[DEFERRED]** — future research |
| ~~**CPPO**~~ | [2503.22342](https://arxiv.org/abs/2503.22342) | Completion pruning (7.98x speedup on GSM8K) | ~~Failure #3 (eval cost)~~ **[DROPPED]** — harmful for exploration |
| **SortedRL** | [openreview 5v3Gzuic8i](https://openreview.net/forum?id=5v3Gzuic8i) | Length-aware online scheduling (>50% less rollout bubble) | Failure #3 (throughput) |

### Docs
| Resource | URL |
|----------|-----|
| Unsloth Qwen3-Coder-Next | https://unsloth.ai/docs/models/qwen3-coder-next |
| Unsloth RL Guide | https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide |
| TRL GRPOTrainer | https://huggingface.co/docs/trl/main/en/grpo_trainer |
| TRL OpenEnv Integration | https://huggingface.co/docs/trl/main/en/openenv |
| OpenEnv Docs | https://meta-pytorch.org/OpenEnv/ |
| GLM-5 API | https://api.z.ai |
| SkyDiscover Blog | https://skydiscover-ai.github.io/ |

---

## 12. Risk Matrix & Mitigations

### Codebase Reality Check

| Layer | Status | LoC | Notes |
|-------|--------|-----|-------|
| OpenEnv environment | Complete | ~600 | Full step/reset/state contract, tested |
| Training pipeline | Skeletal | ~1,150 | Scaffolding exists, depends on Modal backend |
| Evaluation + Verification | Mostly complete | ~1,260 | PAC verify + profiling work, reward bridge missing |
| CUDA kernels | Minimal | ~670 | WCC only (3 variants). No BFS/PageRank/Louvain. |
| Datasets | Complete | ~1,300 | 6,200 tasks pre-generated and formatted |
| SkyDiscover integration | **Implemented** | ~250 | `evaluator.py` (cascade eval), seed kernel, launch script |
| doubleGraph integration | **Implemented** | ~550 | 192 kernels → 3 TRL datasets + 7 SKILL.md patterns + 5 curriculum tasks |
| `training/grpo_train.py` | Missing | 0 | Referenced in pyproject.toml + README but doesn't exist |

**Bottom line:** ~90% of infrastructure exists. Training data (192 expert A100 kernels + 6K Ops-6K) and integration layers (SkyDiscover evaluator + doubleGraph dataset harvester) are complete. The training pipeline assumes an external Modal deployment.

---

### 6.0 Fundamental Research Risks

> **Why this section exists:** Sections 6.1-6.8 cover *implementation* risks (stubs, missing code, memory budgets). This section asks the deeper question: **even if we implement everything perfectly, are the fundamental research approaches sound?** Six concurrent papers (Feb-Mar 2026) in the CUDA kernel RL space say no — at least not as currently designed.

#### 6.0.1 FUNDAMENTAL FAILURE #1: GRPO Is Mathematically Biased for Multi-Turn Kernel Generation

**The problem:** Dr. Kernel (arXiv [2602.05885](https://arxiv.org/abs/2602.05885)) proves that GRPO's advantage estimation is biased because of *self-inclusion* — computing the group mean/std using the current sample itself.

The expected gradient under GRPO is shrunk by a factor of `(1 - 1/N)`:
```
E[ĝ_GRPO] = (1 - 1/N) × ∇θJ(θ)
```

With our G=2: gradients are systematically 50% too small without correction. In multi-turn settings (which our PRD explicitly uses — "multi-turn with compiler feedback"), this compounds across turns because fewer samples survive to later turns, making N smaller and the bias worse. **Fix:** TRLOO N/(N-1) post-processing corrects this (implemented in `custom_grpo_trainer.py`).

**Empirical proof:** Dr. Kernel shows GRPO saturates at ~200 steps on KernelBench Level-2, while their TRLOO (Turn-level Reinforce Leave-One-Out) continues improving. TRLOO removes the current sample from the baseline:
```
Ā^(-i)_t = 1/(N_t - 1) × Σ_{j≠i} G_j,t
```

**The competitive landscape has already moved past GRPO for kernel generation:**

| System | Algorithm | Why not GRPO |
|--------|-----------|-------------|
| Dr. Kernel (HKUST) | TRLOO | Proved GRPO bias mathematically |
| CUDA-L1 (DeepReinforce) | Contrastive RL | Explicitly chose pairwise comparisons over group normalization |
| KernelBlaster (Stanford/NVIDIA) | Memory-augmented in-context RL | In-context learning, no policy gradient at all |
| OptiML | MCTS over LLM edits | Search-based, no RL training |
| CUDA Agent (ByteDance) | PPO (not GRPO) + custom mods | Asymmetric clipping ε_lower=0.2/ε_higher=0.28, value pretraining, 230B model on 128 GPUs |

**What we cite vs what they actually did:** Our PRD cites CUDA Agent as justification for GRPO. But CUDA Agent uses **PPO with a critic** (not GRPO), plus asymmetric clipping, value pretraining, and 128 H20 GPUs. We're copying their reward function but using a fundamentally different (and proven-weaker) algorithm at 1/128th the compute.

**Pivot options (RESOLVED — using option 1):**
1. ✅ **TRLOO post-processing** applied via `TRLOOGRPOTrainer` (N/(N-1) scaling, implemented in `custom_grpo_trainer.py`)
2. Switch to contrastive RL à la CUDA-L1 (compare good vs bad kernels directly) — not needed with TRLOO
3. Increase G from 2 to 4+ (requires more memory but reduces bias) — not needed with TRLOO
4. Drop multi-turn — rejected; multi-turn is #1 CUDA-Agent ablation factor (Section 7.2)

---

#### 6.0.2 FUNDAMENTAL FAILURE #2: Compilation-Only Reward = Proven Lazy Optimization

**The problem:** Our reward function has two unimplemented TODOs:
- GRPO_DEEP_DIVE line 339: "TODO: correctness check"
- GRPO_DEEP_DIVE line 368: `kernel_ms = compile_ms  # Placeholder`

This means rewards r=+2 and r=+3 are *mathematically unreachable*. The model can only ever get r=-1 (doesn't compile) or r=+1 (compiles). This is exactly what Dr. Kernel calls **"lazy optimization"** — the model learns to generate trivially correct, zero-effort kernels.

**Research evidence:**
- **Dr. Kernel:** Without profiling-based rewards, Fast@1.2x (kernels achieving ≥1.2x speedup) saturated at **5.6%**. With profiling rewards + rejection sampling: **20.0%**. That's a 3.6x difference from reward design alone.
- **CUDABench** (arXiv [2603.02236](https://arxiv.org/abs/2603.02236)): "notable mismatch between high compilation success rates and low functional correctness" — compilation is a bad proxy for kernel quality.
- **CUDA-L1:** Uses actual speedup measurement as contrastive signal. Their key insight: "the multiplicative nature of optimizations" — you MUST measure real performance to learn this.

**Current approach (revised):** Discrete milestone rewards `{-1, 1, 2, 3}` matching CUDA-Agent's ablation-proven scheme. CUDA-Agent demonstrated 2.11x geomean speedup using discrete rewards on Ops-6K, validating this design. The discrete scheme is simpler to implement and debug for the hackathon pilot.

- `reward = -1` for compile failure or correctness failure
- `reward = 1` for correct but no speedup
- `reward = 2` for faster than baseline
- `reward = 3` for significant speedup (>2x or within 10% of expert)
- Wrap with TRLOO N/(N-1) post-processing for unbiased advantage estimation
- Reuse CUDA-Agent's profiling.py and verification.py via subprocess (copy verbatim, do NOT rewrite)

**Future pivot options:**
1. Continuous reward (`log(speedup)`) if discrete bins prove too coarse for convergence
2. Add profiling-based rejection sampling (Dr. Kernel's PRS): only train on completions that BOTH compile AND show measurable speedup
3. Use contrastive pairs: generate 2 kernels, reward the faster one relative to the slower one

---

#### 6.0.3 FUNDAMENTAL FAILURE #3: Our Compute Scale Is 100x Below What Works

**The problem:** Every successful CUDA kernel RL system operates at a scale we cannot match:

| System | Model | GPUs | Batch | Context | Steps |
|--------|-------|------|-------|---------|-------|
| **CUDA Agent** | 230B (23B active) | 128x H20 | 1024 | 131K tokens | 150 |
| **Dr. Kernel** | 14B | Distributed KernelGYM (multiple workers) | Large | 32K+ | 200+ |
| **CUDA-L1** | Custom | A100 cluster | Full KernelBench (250 tasks) | — | Full RL loop |
| **Us (hackathon)** | 30.5B (3.3B active) | 1x H100 | 1 prompt × G=2 | 8K tokens | SFT + short GRPO pilot |

Our effective batch size is **2 completions** (G=2). CUDA Agent's is **1024**. That's 512× more gradient signal per step. CUDA Agent trains for 150 steps × 1024 batch = 153,600 effective samples. We train for ~450 steps × 2 × avg ~7 turns = ~6,300 effective samples. **That's ~25× fewer samples.** **[SPECULATIVE — depends on unimplemented techniques; see Section 7.8 for caveats.]**

**Why this matters from first principles:** RL for code generation has an astronomically sparse reward landscape. Most of the optimization space is "doesn't compile" (r=-1). The probability of randomly generating an optimized CUDA kernel is negligible. You need massive sampling to find the rare positive signals. With G=2, many steps will have all-negative rewards (std=0, zero gradient). CUDA Agent solved this with scale. We compensate with richer prompt context (7 SKILL.md patterns + 192 expert demos) and multi-turn iteration. MARS per-turn credit is [DROPPED] (degenerates with outcome rewards per ALPHXIV analysis). See Section 7.10 for remaining stacking techniques.

**Model capacity considerations:** Qwen3-Coder-30B-A3B-Instruct has 30.5B total / 3.3B active parameters — much smaller than CUDA Agent's 23B active. However:
- It is a coding-agent model, purpose-built for code generation and tool use
- MoE gives access to 128 expert blocks while keeping inference cost at 3.3B
- ConCuR (arXiv [2510.07356](https://arxiv.org/abs/2510.07356)) shows data quality matters more than model size — a well-curated dataset beats throwing a bigger model at the problem
- For the hackathon, the model just needs to be good enough that SFT + short GRPO produces visible improvement

**Pivot options:**
1. **Don't RL-train at all for the hackathon.** Use inference-time search instead (OptiML-style MCTS, or SkyDiscover evolutionary search). This is more sample-efficient and doesn't require our broken GRPO setup.
2. Reduce model to 14B (Dr. Kernel's size) to enable larger G and more steps
3. Use SFT on curated data (ConCuR approach) — quality data > RL algorithm
4. Use CUDA Agent's dataset (Ops-6K) as SFT data directly, skip RL, do evolutionary refinement

---

#### 6.0.4 Revised Assessment: Stacked Techniques Make GRPO Viable

The three Fundamental Failures above (6.0.1-6.0.3) are real. But they are addressable. Six papers from March 2026 provide specific, stackable mitigations:

| Fundamental Failure | Root Cause | Fix | Paper |
|-------------------|------------|-----|-------|
| **#1: GRPO bias** | Self-inclusion in group mean/std | TRLOO leave-one-out baseline (MARS [DROPPED] — degenerates with outcome rewards) | Dr. Kernel (arXiv 2602.05885) |
| **#1 (post-process)** | GRPO gradient shrinkage = (1 - 1/N) | TRLOO post-process: multiply advantages by N/(N-1) scaling factor after GRPO computation. Drop-in fix, no custom training loop needed. | Dr. Kernel (arXiv 2602.05885) |
| **#2: Lazy reward** | Reward was a stub; profiling not wired | Discrete milestone rewards {-1, 1, 2, 3} matching CUDA-Agent + optional Nsight bonus | CUDA-Agent ablation + CUDABench (arXiv 2603.02236) |
| **#2 (action space)** | Free-form 2000-token generation = huge search space | Rich SKILL.md prompt context (CUDA-Agent verbatim + doubleGraph patterns) constrains generation. Transformation grammar deferred to v2. | CUDA-Agent SKILL.md, doubleGraph docs |
| **#3: Compute scale** | ~110× fewer samples than CUDA Agent | Hybrid eval (local+Modal), smaller action space, SFT bootstrapping with 192 expert demos. CPPO [DROPPED] (harmful for exploration). | Local compile fast-path |
| **#3 (throughput)** | Sync eval blocks training | AReaL-style replay buffer (staleness η=4-8). MASPO [DEFERRED] (future research). | AReaL (arXiv 2505.24298) |

**Updated probability assessment:**

| Approach | Probability | Time | Why |
|----------|------------|------|-----|
| **SkyDiscover evolutionary search** | HIGH | 2-4 hrs | AdaEvolve proven on 200+ tasks. No training. Primary hedge. |
| **GRPO pilot with stacked techniques** | MEDIUM | ~4-8 hrs training | TRLOO fixes bias. Discrete milestone rewards match CUDA-Agent. SFT warmup anchors compile rate. Short pilot on Qwen3-Coder-30B-A3B-Instruct (H100 gen + A100 eval). Validates real learning signal under budget. |
| **SFT on curated data** (ConCuR-style) | MEDIUM-HIGH | 3-5 hrs | ConCuR shows curated traces > RL. Parallel track. |
| **GRPO as originally designed in PRD** | LOW | 7-12 hrs | Without stacking, still biased, stub reward, too few samples. |

**Implementation priority order (revised — 4-tier minimal viable approach):**

| Priority | When | Activity | Deliverable |
|----------|------|----------|-------------|
| **P0** | By Friday night | OpenEnv env that calls CUDA-Agent's EXACT scripts (copied verbatim into evaluation/) + discrete milestone rewards {-1, 1, 2, 3} + TRLOO N/(N-1) post-process in reward wrapper | Working eval pipeline with discrete reward |
| **P1** | Parallel with P0 | SkyDiscover evolution on 5 seed .cu files (gemm, softmax, layernorm, fused_bias_relu, reduce_sum). 80-120 evolutions per seed. | Demo-worthy evolved kernels |
| **P2** | After P0 | Rich SKILL.md with doubleGraph patterns (from docs/skills/doublegraph_a100.md) + CUDA-Agent SKILL.md verbatim. 50-example SFT warmup via Unsloth built-in trainer. | Anchored compilation ability |
| **P3** | After G-0.8 passes | **Short GRPO pilot:** H100 generates + updates, A100 (Modal) evaluates. G=2, short context, strict abort criteria. Gate G-0.8 must pass first. Best-of-N at inference + SkyDiscover for hardest tasks. | Validate real learning signal; demo credible improvement |

**Total new code:** ~200 LOC across 4 files (down from ~450 — transformation grammar dropped). Fits in 12-16 hours coding + 14 hours training.

**Target hackathon outcome (projected, not guaranteed):**
- SFT warmup improves compile rate and structural kernel quality
- GRPO pilot demonstrates measurable learning signal (candidate improvement under real feedback)
- SkyDiscover provides additional evolved kernels as independent hedge
- Total evals: budget-constrained, estimated hundreds of Modal calls

**Why this revision succeeds where original PRD fails:**
- Fixes GRPO bias (TRLOO post-process; MARS [DROPPED])
- Fixes lazy reward (discrete milestone rewards {-1, 1, 2, 3} + optional Nsight bonus)
- Fixes eval bottleneck (local nvcc for P3, no Modal dependency)
- Fixes action space (rich SKILL.md context, grammar deferred)
- Fixes scope (GRPO is P3 with 50 steps/3-5 turns (hackathon pilot) + best-of-8 after G-0.8; SkyDiscover/SFT are parallel hedges)
- Fixes doubleGraph scope (graph calibration for 5 tasks only, torch.compile for Ops-6K)
- Total timeline realistic (~36 hrs with 7 hr buffer)

---

### 6.1 Component 1: CUDA Agent Evaluation Pipeline — Top 3 Failures

| # | Failure | Severity | Evidence | Mitigation |
|---|---------|----------|----------|------------|
| 1.1 | **Reward function is a stub** — `compute_reward()` referenced in `training/multi_turn_rollout.py:54` but NOT DEFINED. No correctness checking (GRPO_DEEP_DIVE line 339: "TODO"). No kernel profiling (line 368: placeholder `kernel_ms = compile_ms`). | CRITICAL | Code inspection: function imported but missing. GRPO_DEEP_DIVE lines 332-369. | **Gate 0.3**: `openenv_env/reward.py` now implements discrete milestone rewards {-1, 1, 2, 3} with `trloo_post_process()` (~60 lines). Wire to compilation result from Modal. |
| 1.2 | **Modal is single point of failure** — every eval routes through `modal.Function.from_name()`. If Modal API is down/slow during hackathon, training halts. No local fallback. | CRITICAL | `training/multi_turn_rollout.py:35-51`. All GRPO steps call Modal. | Add local eval fallback: `nvcc -arch=sm_80` subprocess + `nm -D` symbol check. 45 min. Use Modal for profiling only; local for compile+verify. |
| 1.3 | **60s subprocess timeout too aggressive** — compilation (10-30s) + correctness (5s) + profiling (20-60s) = 35-95s total. Timeout kills valid kernels. | HIGH | GRPO_DEEP_DIVE line 398: `timeout=60`. Profiling budget line 956: "20-60s". | Increase to `timeout=120`. Or split: compile timeout=30s, profile timeout=90s separately. |

**Counter-argument (why it can still work):** CUDA Agent (arXiv 2602.24286) demonstrated 2.11x speedup using discrete rewards on Ops-6K. Our discrete milestone rewards {-1, 1, 2, 3} match their ablation-proven scheme (implemented in `openenv_env/reward.py`, ~60 lines). `trloo_post_process()` fixes the Dr. Kernel gradient shrinkage.

---

### 6.2 Component 2: doubleGraph Expert Baselines — Top 3 Failures

| # | Failure | Severity | Evidence | Mitigation |
|---|---------|----------|----------|------------|
| 2.1 | ~~**Zero integration code exists**~~ | ~~HIGH~~ **RESOLVED** | `datasets/extract_doublegraph_a100.py` harvests 192 A100 kernels → 3 TRL datasets. `skill_builder.py:_append_a100_patterns()` injects 7 real patterns. `curriculum.py` adds 5 topology-aware graph problems. | **Done.** 192 entries extracted (84K+ lines), topology-mapped, compilation-flag-aware. |
| 2.2 | **Wheel may not install on H100** — Task 0.1.5 is a placeholder (`wget [A100_WHEEL_URL]`). No actual URL. Cross-GPU wheel compatibility unverified. | LOW | Section 8, Task 0.1.5: placeholder URL. | **No longer needed.** We extract patterns directly from .cu source files (not via runtime wheel). The 192 kernels are read as text, not executed via doubleGraph library. |
| 2.3 | ~~**Only WCC kernels exist in repo**~~ | ~~MEDIUM~~ **MITIGATED** | Curriculum now includes BFS (power-law), PageRank (dense-regular), WCC (sparse-islands), Triangle Count (dense-community), Louvain (dense-community). Seed kernel `wcc_doublegraph.cu` in `skydiscover_integration/initial_kernels/`. | **Done.** 192 A100 kernels across 8 categories available as training data and prompts. |

**Counter-argument (updated):** doubleGraph's value is now fully realized as **training data + prompt context + curriculum**. The 192 production A100 kernels are extracted into TRL-compatible datasets for SFT (Stage 2) and GRPO (Stages 1 & 3). The `skill_builder.py:_append_a100_patterns()` function injects 7 real expert patterns (degree-based dispatch, warp hash tables, zero-copy convergence, bitmap frontier, path-halving UF, compilation flag tuning, __launch_bounds__) into every SKILL.md prompt. Topology-aware graph problems in curriculum Phases 2-3 ensure the model trains on realistic graph optimization tasks. No runtime doubleGraph wheel needed.

---

### 6.3 Component 3: SkyDiscover Evolutionary Search — Top 3 Failures

| # | Failure | Severity | Evidence | Mitigation |
|---|---------|----------|----------|------------|
| 3.1 | ~~**Zero implementation exists**~~ | ~~HIGH~~ **RESOLVED** | `skydiscover_integration/` created with: `evaluator.py` (KernelForgeEvaluator with cascade stage1→stage2), `run_evolution.sh` (executable, --dry-run support), `initial_kernels/wcc_doublegraph.cu` (adapted doubleGraph WCC seed). | **Done.** Evaluator bridges to Modal A100 eval via `validate_eval_result()` + `compute_reward()`. Supports both WCC and Ops-6K eval modes. |
| 3.2 | **GLM-5 API not configured** — no API key setup, no pyproject.toml dependency, no test of API endpoint. Cost estimate ($2-8/run) is unvalidated. | MEDIUM | No code references GLM-5 API client. Section 0, Locked Decisions just names the price. | Register for GLM-5 API key Wednesday. Test with 1 call. If API unavailable, substitute any OpenAI-compatible endpoint (Claude, GPT-4). AdaEvolve (arXiv 2602.20133) is model-agnostic. |
| 3.3 | **VRAM contention** — SkyDiscover and GRPO both need GPU access. | MEDIUM | No resource isolation in code. | Run SkyDiscover on a SEPARATE machine or Modal instance. It only needs nvcc + API calls, not full GPU VRAM. Or run sequentially: SkyDiscover first (2-3 hrs), then GRPO. |

**Counter-argument:** AdaEvolve (arXiv 2602.20133) demonstrated 10-40% gains over compiler baselines with 80-120 LLM calls. EvoX (arXiv 2602.23413) adds meta-evolution that automatically discovers optimization tactics. The framework is proven — the risk is purely implementation time, not algorithmic soundness. Even a minimal 10-iteration run produces demo-worthy results.

---

### 6.4 Component 4: GRPO Training — Top 3 Failures

| # | Failure | Severity | Evidence | Mitigation |
|---|---------|----------|----------|------------|
| 4.1 | **Memory budget unverified** — Qwen3-Coder-30B-A3B-Instruct FP8 (~16-18GB) + LoRA + optimizer + KV cache + activations = ~14-20GB estimated on H100 80GB. Not yet measured on actual hardware. | MEDIUM | Estimates only, no benchmarks. | **Gate G-0.2**: Run `nvidia-smi` during model load + single GRPO step. If OOM: (a) reduce context 8K->4K, (b) reduce G=2->1, (c) switch to 4-bit quantization, (d) use fallback Qwen3.5-35B-A3B. Decision ladder, not binary. |
| 4.2 | **TRL GRPOTrainer API compatibility** — `rollout_func` parameter is experimental. `reward_funcs` vs `reward_func` naming uncertain. `remove_unused_columns=False` behavior unverified. | HIGH | GRPO_DEEP_DIVE line 566 (`reward_funcs`), line 1080 (`remove_unused_columns`). | **Gate G-0.6**: `test_06_grpo_init.py` — initialize GRPOTrainer with our config, verify no crash. 15 min. TRL pinned to ==0.29.0 in pyproject.toml — do NOT upgrade without re-running gate. |
| 4.3 | **G=2 + low compilation rate = zero-gradient steps** — if base model compiles <30% of CUDA, P(all 2 fail) = 49%. Those steps produce std(r)=0 -> advantage=0 -> no learning. Discrete milestone rewards {-1, 1, 2, 3} provide variance when both compile at different quality levels. | HIGH | No empirical compilation rate for Qwen3-Coder-30B-A3B-Instruct. | **Gate G-25** (Saturday morning): test_07 measures compilation rate. If <50%: extend SFT warmup. If <30%: switch model. If persistently low: abandon GRPO, ship SFT + search + environment. |

**Counter-argument:** GRPO (arXiv 2402.03300, DeepSeekMath) has been validated in multiple domains. The "step-17 collapse" from CUDA Agent Section 3.3 is specifically for PURE RL without warmup — SFT-first addresses this. Qwen3-Coder-30B-A3B-Instruct is a coding-agent model with strong baseline code generation capability. The risk is real but bounded by our decision gates and abort criteria.

**Hackathon approach:** TRLOO N/(N-1) correction fixes the Dr. Kernel bias. SFT warmup anchors compile rate. G=2 keeps eval budget low. Discrete milestone rewards {-1, 1, 2, 3} provide reward variance when both candidates compile at different quality levels. Short GRPO pilot (50 steps, 3e-6 LR) validates real learning signal. If it fails: ship SFT + search + environment -- do not force a fake RL success story.

---

### 6.5 Cross-Component Integration Risks

| # | Integration Point | Risk | Mitigation |
|---|-------------------|------|------------|
| I.1 | GRPO reward -> Modal eval -> nvcc compile | If Modal latency >30s/call, each GRPO step takes 5+ min. 100 steps = 8+ hours. | Add local compile-only fast path. Use Modal only for profiling. Reduces per-step from 2-5 min to 30-60s for compile-only reward. |
| I.2 | Ops-6K dataset -> prompt formatting -> model generation | Dataset format assumes `ops`, `data_source`, `code` fields. If HF dataset updated, parsing breaks. | Pin dataset version. `training/cuda_agent_integration.py:66` already loads from HF — add `revision="main"` pin. |
| I.3 | SkyDiscover parallel + GRPO sequential | Both assume GPU access. No resource scheduler. | Run on separate machines. SkyDiscover needs only nvcc (CPU-heavy, API calls). GRPO needs GPU. |
| I.4 | `grpo_train.py` entry point missing | Referenced in README and pyproject.toml (`kernelforge-train`) but file doesn't exist. Training can't start. | Create `training/grpo_train.py` that orchestrates stage1->stage2->stage3. 1 hour. |

---

### 6.6 Decision Gates (Go/No-Go Checkpoints)

| Gate | When | Test | Pass Criteria | Fail Action |
|------|------|------|---------------|-------------|
| **G-0.1** | Wed night | Downloads complete | All 5 assets downloaded, checksums OK | Re-download overnight; if still failing, use backup model only |
| **G-0.2** | Thu morning | H100 environment | `nvidia-smi` shows H100 80GB, `nvcc --version` >=12.x, `nvcc -arch=sm_80 test.cu` works | Fix CUDA install; if H100 unavailable, try A100 or rent from Lambda/RunPod |
| **G-0.3** | Thu afternoon | Reward function works | `compute_reward(compiled=True, correct=True, speedup_vs_eager=1.5, speedup_vs_compile=1.0)` returns `log(1.5) ≈ 0.405` | Implement missing function (30 min, BLOCKING) |
| **G-0.5** | Thu night | SFT data generated | `data/sft_data.json` has >=50 compilable examples | Lower bar to 30; if API fails, use combined dataset as warmup data |
| **G-0.6** | Fri morning | GRPOTrainer initializes | test_06 passes, no OOM, no API errors | Adapt to TRL API changes (read changelog) |
| **G-0.7** | Fri afternoon | Full GRPO step works | test_08 completes: generate -> compile -> reward -> gradient update | Debug loop; if unfixable, fall back to SFT-only + SkyDiscover |
| **G-0.8** | Thu night | 5-step GRPO sanity | Run 5 GRPO steps with discrete milestone rewards {-1, 1, 2, 3} + TRLOO post-process. Must see: (a) non-zero gradients in 4/5 steps, (b) reward variance > 0 in 3/5 steps, (c) at least 2/20 test kernels achieve >1.2x speedup vs torch.compile. | Zero gradients → reward function broken. No speedup → model cannot optimize, fall back to SFT + SkyDiscover only. Do NOT continue to P3 GRPO. |
| **G-25** | Sat 9 AM | Compilation rate | test_07: base model >=50% compilation rate | <50%: extend warmup. <30%: switch model. <10%: abandon GRPO, go SkyDiscover-only. |

---

### 6.7 Realistic Timeline (with failure buffer)

**Updated per 10-fix critique + Section 7 feasibility analysis:** P0 (eval pipeline) must work before anything else. SkyDiscover runs in parallel as primary hedge. GRPO is P3 — 50-step serious run after G-0.8 passes (upgraded from 10-step demo per Section 7.5).

| Block | Time | Activity | Failure Buffer |
|-------|------|----------|---------------|
| **Wed night** | 4 hrs | Downloads + API key setup + GLM-5 API registration | Re-download Thu morning if needed |
| **Thu 9AM-12PM** | 3 hrs | H100 setup + CUDA verify + Gate G-0.2. Load Qwen3-Coder-30B-A3B-Instruct via Unsloth. | 1 hr debug buffer |
| **Thu 1PM-6PM** | 5 hrs | **P0 — Core pipeline:** Copy CUDA-Agent eval scripts verbatim into evaluation/. Wire discrete milestone reward function {-1, 1, 2, 3}. Add TRLOO N/(N-1) post-process wrapper. Gate G-0.3. | 2 hr buffer |
| **Thu 6PM-11PM** | 5 hrs | **P1 — SkyDiscover (parallel):** Set up evaluator on 5 seed .cu files. Launch 80-120 evolution runs. **P2 — SKILL.md:** Copy CUDA-Agent SKILL.md verbatim. Extract doubleGraph patterns into prompt context. | SkyDiscover runs overnight |
| **Fri 9AM-1PM** | 4 hrs | **P2 — SFT warmup:** 50-example SFT via Unsloth built-in trainer. Verify >70% compile rate with local nvcc. Gate G-0.8: 5-step GRPO sanity check. | If compile rate <70%, extend SFT |
| **Fri 1PM-Sat 3AM** | 14 hrs | **P3 — GRPO pilot:** H100 generates + updates weights; Modal A100 evaluates correctness + speedup for reward. Only if G-0.8 passes. Short run with strict abort gates. | If GRPO stalls, SFT + SkyDiscover results are the submission |
| **Sat 9AM** | 30 min | **Gate G-25**: compilation rate decision. Evaluate all tracks. | Decision tree in 6.6 |
| **Sat 9:30AM-5PM** | 7.5 hrs | Collect best kernels from all tracks. OpenEnv wrapper. Run final eval suite. | SkyDiscover is the hedge |
| **Sat 5PM-10PM** | 5 hrs | Results collection, demo, blog | 2 hr buffer |

**Total available: ~43 hrs. Total planned: ~36 hrs. Buffer: ~7 hrs (16%).**

**Key change from previous timeline:** P0 (eval pipeline + discrete milestone reward) must work before anything else. SkyDiscover runs in parallel as the primary hedge. GRPO is P3 — a 50-step run with 3-5 turns and 8K context (hackathon pilot config). It only runs if G-0.8 passes. Transformation grammar dropped from v1. The eval pipeline reuses CUDA-Agent's scripts verbatim rather than rewriting them.

---

### 6.8 Minimum Viable Submission (If Everything Fails)

Even if GRPO collapses AND SkyDiscover API is unavailable, we ship:

1. **OpenEnv environment** — fully functional, ~600 LoC, tested (ship as-is)
2. **WCC baseline kernels** — 3 variants with PAC verification (ship as-is)
3. **Dataset pipeline** — 6,200 tasks curated and formatted (ship as-is)
4. **Evaluation framework** — compilation, verification, profiling (ship as-is)
5. **doubleGraph SKILLS.md** — 330-line A100 optimization reference (ship as-is)

This is a credible hackathon submission. The infrastructure IS the contribution — a complete OpenEnv-compatible CUDA kernel optimization environment with real evaluation, real datasets, and real verification. The RL training is the stretch goal that makes it exceptional.

---

## 12.9 Deliverables for Judges

By submission time, the strongest credible package is:

1. **OpenEnv-compatible CUDA optimization environment** — real `step()/reset()/state()` contract
2. **Real compile / correctness / timing evaluation** — on A100 via Modal
3. **H100 + Qwen3-Coder-30B-A3B-Instruct training/search loop** — SFT + GRPO pilot + SkyDiscover
4. **A100-targeted kernel optimization demo** — with measured results
5. **Clear explanation of how this scales into future one-GPU kernel-optimization research**

That is a good hackathon story because it is:
- technically real,
- ambitious without being fake,
- aligned with OpenEnv / RL infra,
- extensible after the event.

---

## 12.10 Bottom Line

KernelForge is now defined as:
- a **budgeted, real, A100-targeted kernel optimization system** for the hackathon,
- using **Qwen3-Coder-30B-A3B-Instruct on H100** as the practical base,
- with **A100 timing as the performance truth**,
- and with **SFT + GRPO pilot + search hedge** as the actual execution plan.

That is the correct version of the project for this weekend.

---

## 13. Future Work: Scale-Up Feasibility Assessment

> **NOTE: This section describes future research targets, not hackathon deliverables. All projections below are speculative and have not been benchmarked.** The original analysis assumed B200 + Qwen3-Coder-Next 80B. For the hackathon, see Sections 0-7 for the actual plan.

### Original Assessment: Matching CUDA-Agent on Single B200 (FUTURE TARGET)

> **Added March 5, 2026. Revised March 5.** Quantitative analysis of how our stacked TRLOO-GRPO approach on a single B200 could approach CUDA-Agent's benchmark results. **[SPECULATIVE — projections below depend on MARS [DROPPED], CPPO [DROPPED], and other unimplemented techniques. Actual hackathon pilot uses 50 steps, discrete {-1,1,2,3} rewards, and TRLOO only.]**

### 7.1 CUDA-Agent Benchmark (The Target)

CUDA-Agent (Seed 1.6 230B MoE, 23B active, 128× H20 GPUs, PPO with critic, batch=1024, 150 steps):

| Metric | Level 1 (Simple) | Level 2 (Fused) | Level 3 (Full Models) | Overall |
|--------|-------------------|-----------------|----------------------|---------|
| Pass Rate | 100% | 100% | 94% | **98.8%** |
| Faster than torch.compile | 97% | 100% | 90% | **96.8%** |
| Geomean speedup vs compile | 1.87× | 2.80× | 1.52× | **2.11×** |

**Match targets:** 98.8% pass rate, 96.8% faster-than-compile, 2.11× geomean speedup. Achieved via training-time quality (93-96% pass@1) + inference-time best-of-8 selection + SkyDiscover for hardest tasks.

### 7.2 CUDA-Agent Ablation (What Actually Drives Performance)

| What's Removed | Pass Rate | Faster % | Speedup | Delta |
|---------------|-----------|----------|---------|-------|
| **Full system** | **98.8%** | **96.8%** | **2.11×** | — |
| Multi-turn agent loop (single-turn only) | 77.1% | 14.1% | 0.69× | **-82.7% faster, -1.42× speedup** |
| Robust reward (use raw speedup) | 96.8% | 60.4% | 1.25× | -36.4% faster, -0.86× speedup |
| RFT warmup | 95.6% | 49.8% | 1.05× | -47.0% faster, -1.06× speedup |
| Value pretraining | 98.6% | 50.9% | 1.00× | -45.9% faster, -1.11× speedup |

**Key insight:** Multi-turn iteration is the #1 factor by a wide margin. It accounts for more performance than reward design, RFT, and value pretraining combined.

### 7.3 KernelForge vs CUDA-Agent — Dimension-by-Dimension

| Dimension | CUDA-Agent | KernelForge | Who Wins? |
|-----------|------------|-------------|-----------|
| Model size | 230B MoE, **23B active** | 80B MoE, **3B active** | THEM (8× active params) |
| Base CUDA capability | Seed 1.6 (general LLM) | Qwen3-Coder-30B-A3B-Instruct (hackathon); Qwen3-Coder-Next 80B (future scale-up) | **US** (coding-agent baseline) |
| RL algorithm | PPO + critic network | TRLOO-GRPO (criticless, unbiased) | **US** (+72% Fast@1.2× per Dr. Kernel) |
| Compute scale | 128× H20 GPUs | 1× B200 | THEM (128×) |
| Effective RL samples | 153,600 (150 steps × 1024 batch) | **[SPECULATIVE]** ~6,000 (future target; hackathon pilot: ~19 live-evaluable tasks, 50 steps) | THEM (25×) |
| Multi-turn depth | 150 turns, 128K context | **20 turns, 32K context** (match config) | THEM (7.5× more turns) |
| Reward signal | Discrete {-1, 1, 2, 3} | Discrete {-1, 1, 2, 3} (matching CUDA-Agent) + optional Nsight bonus | EVEN (same scheme) |
| Advantage estimation | Standard PPO (biased for small groups) | TRLOO N/(N-1) (mathematically unbiased) | **US** (removes 50% gradient shrinkage at G=2) |
| Credit assignment | Flat (all turns same reward) | Flat (MARS [DROPPED] — degenerates with outcome rewards) | EVEN |
| Prompt context | Basic system prompt | CUDA-Agent SKILL.md + doubleGraph A100 patterns (1,600+ lines) | **US** (richer optimization priors) |
| Inference-time compute | 1 candidate per problem | **Best-of-8** (generate 8, pick fastest correct) | **US** (3-5× effective improvement) |
| Expert demonstrations | 0 | 192 A100 kernels from doubleGraph (84K+ lines) | **US** (∞ — they have zero expert signal) |
| Evolutionary hedge | None | SkyDiscover on same Modal A100 eval pipeline | **US** (independent search for hardest problems) |
| Eval cost | 128 GPUs → unlimited parallelism | Single A100 Modal → sequential | THEM |

### 7.4 Per-Metric Assessment

Each metric has two tiers: **training-time** (what the model produces per-candidate) and **inference-time** (after best-of-8 selection + SkyDiscover).

**Pass Rate: 98-99% → MATCHES 98.8% ✅**
- Training-time (pass@1): ~93-96% **[SPECULATIVE]**. RFT on 192 REAL expert demos + GRPO steps with TRLOO.
  CUDA-Agent ablation: RFT alone gets 95.6%; our RFT includes 192 expert kernels + filtered trajectories → stronger anchor.
- Inference-time (best-of-8): 1 - (1-0.95)^8 ≈ 99.99%. At 95% base pass rate, 8 candidates
  virtually guarantees a correct kernel for every problem.
- CUDA-Agent's 98.8% is pass@1 from a 23B-active model trained on 153K samples.
  Our best-of-8 on a 3B-active model trained on 6K high-quality samples exceeds this.

**Faster-than-compile: 93-97% → MATCHES 96.8% ✅**
- Training-time (single candidate): ~80-88% **[SPECULATIVE]**. SKILL.md gives 7 concrete A100 strategies. TRLOO fixes gradient bias at G=2.
- Inference-time (best-of-8): Select fastest correct kernel from 8 candidates.
  If each has 85% chance of being faster: best-of-8 = 1-(0.15)^8 ≈ 99.97%.
  But speedup isn't binary — we select the FASTEST, so effective rate even higher.
- SkyDiscover hedge: For hardest Level 3 tasks, evolutionary search produces
  optimized variants independent of GRPO. Fills the long tail.

**Geomean Speedup: 2.0-2.3× → MATCHES 2.11× ✅**
- Training-time (per-kernel average): ~1.8-2.1× **[SPECULATIVE]**. TRLOO fixes gradient bias.
  192 expert demos show the model what 3.6× looks like. Discrete milestone rewards provide clear optimization targets.
- Inference-time (best-of-8 max speedup): Select the fastest correct kernel from 8 candidates.
  With 8 samples, the max speedup is ~1.3-1.5× higher than the mean. Mean 1.9× → max-of-8 ≈ 2.2-2.5×.
- SkyDiscover: Starting from doubleGraph seeds (already 3.6× on graph algorithms),
  evolutionary search maintains or improves baseline speedups.

**Overall: [SPECULATIVE] projected to approach CUDA-Agent at lower cost. Not yet proven.**

### 7.5 Match Configuration: Full Stack on Single B200

The match strategy uses three layers: (1) expert bootstrapping via SFT, (2) efficient RL with TRLOO (MARS [DROPPED], CPPO [DROPPED]), (3) inference-time compute + SkyDiscover.

#### Layer 1: Expert Bootstrapping (Stages 1-2)

CUDA-Agent has ZERO expert demonstrations — it learns everything from RL. We SFT on 192 real A100 kernels FIRST, then do RL. This replaces ~10,000-20,000 RL exploration samples, putting our model at CUDA-Agent's ~step-50 level before Stage 3 even begins.

```
STAGE 1 — WARMUP (target: 90%+ compile rate)
  Model: Qwen3-Coder-Next 80B FP8 on B200 (192GB, $6.25/hr)
  Steps: 300, G: 2, Max turns: 5, Temp: 0.9, LR: 3e-6
  Context: 8,192 tokens
  Eval: Local nvcc fast-fail + Modal A100 for correctness + speedup
  Credit: TRLOO N/(N-1)
  Cost: ~3.5 hrs × $6.25 = $21.88

STAGE 2 — RFT (anchor with expert patterns)
  Filter: reward ≥ 0.0 (any kernel faster than baseline)
  SFT: 3 epochs on filtered trajectories + 192 doubleGraph expert demos
  Cost: ~0.5 hrs × $6.25 = $3.13
```

#### Layer 2: TRLOO GRPO (Stage 3 — the main event)

> **Hackathon pilot config:** 50 steps, 3e-6 LR, 3-5 turns, 8K context on H100. The config below is the future scale-up target on B200.

```
STAGE 3 — GRPO + CURRICULUM (future scale-up config — NOT hackathon)
  GPU: B200 (192GB, 92GB free after model load)
  Steps: 150 (matches CUDA-Agent's 150 steps — but on 1 GPU, not 128)
  G: 2
  Max turns: 20 (full compile→correct→optimize→tune lifecycle)
  Context: 32,768 tokens (KV cache ≈ 40GB → 52GB headroom on B200)
  Temperature: 0.7
  LR: 3e-6 (hackathon pilot; 5e-6 was original future target)
  Credit: TRLOO leave-one-out (MARS [DROPPED] — degenerates with outcome rewards)
  Reward: Discrete milestone {-1, 1, 2, 3} + optional Nsight bonus (top-k)
  Pruning: (CPPO [DROPPED] — harmful for exploration)
  Loss: Standard clip (MASPO [DEFERRED] — future research)

  B200 cost: 150 steps × ~4 min/step = ~10 hrs × $6.25 = $62.50
  A100 eval: ~1,200 Modal calls × 30s = ~10 hrs × $2.50 = $25.00
  Stage 3 total: ~$87.50
```

#### Layer 3: Inference-Time Compute + SkyDiscover

```
BEST-OF-N INFERENCE:
  Generate 8 candidates per problem (4 prompts × G=2)
  Evaluate all 8 on Modal A100
  Select fastest correct kernel
  Cost: 8 evals × 50 problems × $0.02/eval = ~$8

SKYDISCOVER EVOLUTIONARY SEARCH:
  For Level 3 (hardest) problems only (~20% of eval)
  Start from GRPO best outputs + doubleGraph seeds
  100 iterations of mutation + recombination on same Modal A100 pipeline
  Cost: ~$10

Inference total: ~$18
```

#### Total Cost

```
Stage 1 (300 warmup steps):     $22
Stage 2 (RFT):                  $3
Stage 3 (150 GRPO steps):       $87.50
Inference (best-of-8):          $8
SkyDiscover (evolutionary):     $10
Testing/debug buffer:           $25
────────────────────────────────────
TOTAL:                          ~$155.50
After $30 free credit:          ~$125.50

vs CUDA-Agent: 128× H20 GPUs = ~$5,000-10,000+
EFFICIENCY: [SPECULATIVE] 30-60× cheaper — cost is real, "equivalent results" is unproven
```

### 7.6 Expected Results (Match Configuration)

| Metric | Training-Time (pass@1) | Inference (best-of-8) | CUDA-Agent | Match? |
|--------|----------------------|----------------------|------------|--------|
| Pass rate | 93-96% | **98-99%** | 98.8% | **YES ✅** |
| Faster-than-compile | 80-88% | **93-97%** | 96.8% | **YES ✅** |
| Geomean speedup | 1.8-2.1× | **2.0-2.3×** | 2.11× | **YES ✅** |

**Why inference-time compute closes the gap:**
- Training-time results are per-candidate (pass@1). With 93-96% pass rate, each candidate has ~5% failure rate. Best-of-8 failure: (0.05)^8 < 0.00001%.
- For speedup, we select the FASTEST correct kernel from 8 candidates. This pushes the effective speedup from the mean (1.9×) toward the max (2.3×).
- CUDA-Agent's published results are ALSO inference-time (they evaluate their trained model). The difference: they achieve pass@1 with massive training compute. We achieve equivalent pass@8 with efficient training + cheap inference-time search.

### 7.7 Smaller Models Analysis

**Note: This section analyzes the future scale-up model (Qwen3-Coder-Next 80B), not the hackathon primary (Qwen3-Coder-30B-A3B-Instruct).** Qwen3-Coder-Next 80B MoE IS already a "small" model at inference time. It has 80B total parameters but only 3B active due to MoE routing. This gives 80B of stored knowledge at 3B inference cost — there is no free lunch from going smaller.

| Model | VRAM (FP8) | Active Params | Gen Speed | CUDA Capability |
|-------|-----------|---------------|-----------|-----------------|
| **Qwen3-Coder-Next 80B** (3B active) | ~100 GB | 3B | ~50 tok/s | **Very Strong** (800K executable tasks) |
| Qwen3.5-32B (dense) | ~32 GB | 32B | ~80 tok/s | Strong (general code) |
| Qwen3.5-9B (backup) | ~9 GB | 9B | ~150 tok/s | Moderate |

**When smaller wins:** Level 1 simple ops where base capability is sufficient; budget-constrained scenarios where 3× more training steps matters more than per-step quality.

**When smaller loses:** Level 2-3 fused/model tasks requiring deep CUDA knowledge; novel optimization pattern discovery; tasks where base compile rate < 50% (most steps wasted regardless of speed).

**Recommendation:** Stick with Qwen3-Coder-Next 80B FP8 on **B200** (192GB). Only fall back to 9B/32B if 80B doesn't fit (Gate G-0.2 fails) or base compilation rate < 30%.

### 7.8 Data Efficiency: Why 6,000 Samples Match 153,600

> **[SPECULATIVE] The "12-25x per-sample efficiency" claim below depends on MARS [DROPPED] and CPPO [DROPPED]. With only TRLOO implemented, actual per-sample efficiency improvement is the TRLOO +72% Fast@1.2x rate, not 12-25x. The hackathon pilot trains ~50 steps x G=2, not 150 steps x 20 turns.**

| Metric | CUDA-Agent | KernelForge | Ratio |
|--------|------------|-------------|-------|
| RL training samples | 153,600 | ~6,000 (150 steps × G=2 × avg 20 turns) **[future target]** | 25× fewer |
| Per-sample efficiency | Baseline (PPO, biased) | **[SPECULATIVE]** 12-25× (originally MARS+TRLOO+continuous; MARS [DROPPED], CPPO [DROPPED]) | Actual: TRLOO only |
| Effective RL signal | 153,600 | **[SPECULATIVE]** 72,000-150,000 | Unproven |
| Expert demonstrations | 0 | 192 (doubleGraph, 84K+ lines) | **∞** |
| Inference-time candidates | 1 (pass@1) | 8 (best-of-8) | **8×** |
| Evolutionary search | None | SkyDiscover (same Modal pipeline) | **∞** |
| Net effective signal | 153,600 | **~200,000-400,000** | **1.3-2.6× MORE** |

Combined supervised+RL dataset coverage used by KernelForge:
- Ops-6K tasks: ~6,000 (upstream data source)
- doubleGraph A100 tasks: 192
- Combined pool: ~6,192 tasks
- **Hackathon pilot live-evaluable:** ~19 tasks (subset with working eval pipelines)

Ops-6K serves as the upstream data source; ~19 tasks are live-evaluable in the hackathon pilot. This merged pool balances dense operator optimization (Ops-6K) with topology-aware graph optimization (doubleGraph), improving curriculum coverage across phases 1-4 while keeping a single prompt/problem contract.

**Why per-sample efficiency (revised — actual implemented techniques):**
1. **TRLOO** (unbiased G=2): Removes 50% gradient shrinkage at G=2 (Dr. Kernel: +72% Fast@1.2×) -- **IMPLEMENTED**
2. **192 expert demos**: Real A100 kernels as SFT anchor — CUDA-Agent has zero expert demonstrations -- **IMPLEMENTED**
3. **7 SKILL.md patterns**: Concrete optimization strategies from doubleGraph production code in every prompt -- **IMPLEMENTED**
4. ~~**MARS per-turn credit**~~: [DROPPED] — degenerates to flat returns with outcome-level (non-process) rewards per ALPHXIV analysis
5. ~~**CPPO filtering**~~: [DROPPED] — harmful for exploration; prunes candidates the model needs to learn from
6. ~~**Continuous reward**~~: Replaced with discrete milestone rewards {-1, 1, 2, 3} matching CUDA-Agent

**The compound math: [SPECULATIVE — original projection, not validated]**
```
[SPECULATIVE — depends on dropped MARS+CPPO techniques]
CUDA-Agent: 153,600 samples × 1.0× efficiency = 153,600 effective signal
KernelForge (original projection): 6,000 samples × 15× avg efficiency = 90,000 effective RL signal
             + 192 expert demos ≈ 15,000 effective signal (SFT bootstrapping)
             + best-of-8 at inference ≈ 3× effective improvement
             = (90,000 + 15,000) × 3 = ~315,000 effective signal

Original projection: KernelForge / CUDA-Agent = 315,000 / 153,600 = 2.05× MORE effective signal
NOTE: This 2.05x claim depends on MARS [DROPPED] and CPPO [DROPPED].
Actual hackathon pilot: TRLOO only, 50 steps, ~19 live-evaluable tasks.
```

### 7.9 Bottom Line

**Can we approach CUDA-Agent on a single B200? Projected YES — but not yet proven.**

The strategy has four layers:

1. **Expert bootstrapping** — 192 real A100 kernels via SFT puts us at CUDA-Agent's ~step-50 level before RL begins. They have ZERO expert demonstrations.
2. **Efficient RL** — TRLOO on 50 steps (hackathon pilot) with discrete milestone rewards. MARS [DROPPED], CPPO [DROPPED]. The original "12-25x more learning per sample" projection is **[SPECULATIVE]** and not achievable with current stack.
3. **Inference-time compute** — Best-of-8 candidate selection closes the remaining gap. Generate 8, evaluate all, pick the fastest correct kernel. Cheap on Modal A100.
4. **SkyDiscover hedge** — Evolutionary search on the hardest problems, starting from GRPO outputs + doubleGraph seeds. Independent of RL convergence.

**Cost:** ~$155 total ($125 after credits). vs CUDA-Agent's ~$5,000-10,000+.
**Efficiency:** **[SPECULATIVE]** Originally projected "30-60x cheaper for equivalent results" but this depends on MARS [DROPPED] and CPPO [DROPPED]. Actual cost advantage is real; performance equivalence is unproven.

The key insight: CUDA-Agent spent massive compute on EXPLORATION (discovering what good CUDA code looks like from scratch via RL). We skip exploration with 192 real expert demonstrations + 7 SKILL.md patterns from doubleGraph production code. Our RL budget goes entirely to EXPLOITATION (learning to optimize for specific problems), which is where TRLOO excels. (MARS [DROPPED] -- degenerates with outcome rewards.)

### 7.10 Stacked RL Techniques for CUDA Kernel Generation

> **Status update:** MARS [DROPPED] (degenerates with outcome rewards per ALPHXIV analysis). CPPO [DROPPED] (harmful for exploration). MASPO [DEFERRED] (future research). Only TRLOO is implemented and active. The analysis below is retained for reference but the MARS and CPPO sections describe dropped techniques.

#### The CUDA Kernel Iteration Lifecycle

Multi-turn CUDA kernel optimization follows a predictable lifecycle across 20 turns:

| Phase | Turns | What Happens | Typical Reward |
|-------|-------|-------------|----------------|
| **Compilation** | 1-5 | Write kernel → fix #include, syntax, launch config | r = -1.0 (compile errors) |
| **Correctness** | 5-8 | Fix numerical output, boundary conditions, race conditions | r = -1.0 → 0.0 (correct but slow) |
| **Optimization** | 8-15 | Profile → shared memory, float4 loads, coalescing, tiling | r = 0.3-0.8 (1.3×-2.2× speedup) |
| **Tuning** | 15-20 | launch_bounds, register pressure, L2 pinning, block size | r = 0.6-1.1 (1.8×-3.0× speedup) |

#### Why Standard GRPO Fails Here

Standard GRPO assigns the SAME advantage to every token in every turn based on the final reward. If the model achieves 2.0× speedup at turn 15, ALL tokens — including turn 1's compile error — get advantage proportional to log(2.0) = 0.69. The model cannot learn that compilation setup has modest value while optimization turns produce the actual speedup.

#### MARS Per-Turn Credit for CUDA [DROPPED]

> **[DROPPED]** MARS degenerates to flat returns when using outcome-level (non-process) rewards. With discrete milestone rewards {-1, 1, 2, 3} assigned at episode end, all turns receive the same cumulative return, eliminating the per-turn credit advantage. Retained below for reference only.

MARS computes per-turn cumulative returns R_k = r_k + γ·R_{k+1} (backward pass, γ=1.0):

```
Example: 10-turn CUDA kernel optimization

  Turn 1:  compile error        r₁ = -1.0  → R₁ = 0.47  (modest — downstream value)
  Turn 2:  fix #include         r₂ = -1.0  → R₂ = 1.47  (higher — enabled compilation)
  Turn 3:  compiles, wrong out  r₃ = -1.0  → R₃ = 2.47  (critical — gate to correctness)
  Turn 4:  correct, 0.9× slow  r₄ = -0.11 → R₄ = 2.36  (correct but slow — enables opt)
  Turn 5:  add shared mem, 1.4× r₅ = 0.34  → R₅ = 2.47  (first real optimization)
  Turn 6:  float4 loads, 1.8×   r₆ = 0.59  → R₆ = 2.13  (key A100 pattern from SKILL.md)
  Turn 7:  launch_bounds, 2.0×  r₇ = 0.69  → R₇ = 1.54  (tuning)
  Turn 8:  L2 pinning, 2.2×    r₈ = 0.79  → R₈ = 0.85  (diminishing returns)
  Turn 9:  register OK          r₉ = 0.06  → R₉ = 0.06  (early exit)

Without MARS: All turns get R = 0.79 (final reward only).
             Turn 1's compile error = Turn 6's float4 optimization. BAD.

With MARS:   Turn 3 (R₃=2.47) gets 3× more credit than Turn 8 (R₈=0.85).
             Correctness boundary = highest-value transition. GOOD.
```

**Key insight (theoretical — MARS is [DROPPED]):** MARS would discover that **getting to "correct" is the bottleneck** -- but only with per-turn process rewards. With outcome-level discrete rewards, MARS degenerates to flat credit assignment, providing no benefit over standard GRPO.

#### TRLOO Fixes G=2 Bias

With G=2, standard GRPO shrinks gradients by 50%: E[ĝ] = (1 - 1/2) × ∇J = 0.5 × ∇J. TRLOO applies N/(N-1) = 2× correction, restoring unbiased gradients. This is critical at G=2: without TRLOO, HALF the gradient signal is lost. Dr. Kernel ablation: TRLOO gives +72% Fast@1.2× rate. (MARS per-turn returns [DROPPED] — degenerates with outcome rewards.)

#### CPPO Filters Before Expensive Modal Eval [DROPPED]

> **[DROPPED]** CPPO completion pruning is harmful for exploration. With G=2, pruning one of two candidates eliminates the gradient signal entirely for that step. The model needs to see both good and bad candidates to learn. Retained below for reference only.

Before sending candidates to A100 eval ($2.50/hr), CPPO would score them cheaply via structural heuristics (+2 for `__global__`, +1 for `__shared__`, +0.5 for float4/shuffles/launch_bounds, -1 for suspiciously short code). With G=2: lower-scoring candidate would get r=-1.0 without Modal eval if score is below threshold.

#### Expert Demos + SKILL.md Amplification

192 doubleGraph A100 kernels + 7 SKILL.md patterns mean the model generates optimization-aware code from turn 1 (not turn 10). The model doesn't need 150 turns to discover `float4` loads -- it knows them from SKILL.md. It uses turns for problem-specific adaptation instead.

#### SkyDiscover as Floor + Supplement

Even if GRPO underperforms projections, SkyDiscover provides a floor:
- Starts from real doubleGraph WCC kernel (already 3.6× over cuGraph)
- Evolves via mutation + recombination using same Modal A100 eval infrastructure
- Produces demo-worthy results independent of GRPO convergence
- Projected floor with SkyDiscover only (no GRPO): ~1.5-2.0× on graph algorithms

#### Combined Impact Table

| Technique | What It Fixes | Quantified Impact | Status |
|-----------|--------------|-------------------|--------|
| **TRLOO** | 50% gradient shrinkage at G=2 | +72% Fast@1.2× rate (Dr. Kernel) | **DONE** |
| ~~**MARS**~~ | ~~All-turns-same-credit collapse~~ | ~~+23% return (MARSHAL ablations)~~ | **[DROPPED]** — degenerates with outcome rewards |
| ~~**CPPO**~~ | ~~Wasted Modal evals on bad code~~ | ~~~40% fewer Modal calls~~ | **[DROPPED]** — harmful for exploration at G=2 |
| **192 expert demos** | No real A100 patterns in training | SFT anchor = CUDA-Agent step-50 equivalent | **DONE** |
| **7 SKILL.md patterns** | Generic prompts | Concrete optimization strategies every prompt | **DONE** |
| **Topology curriculum** | One-size-fits-all training | 21 problems, 5 graph-structure-specific | **DONE** |
| **Best-of-8 inference** | Single candidate per problem | pass@8 ≈ 99.99%; max speedup selection | Eval-time |
| **SkyDiscover hedge** | GRPO might converge slowly | Evolutionary search floor on demo quality | **DONE** |

**Compound effect (revised):** TRLOO fixes the gradient bias at G=2 → expert demos anchor the starting point higher → SKILL.md guides generation toward known-good patterns → best-of-8 at inference picks the best outcome → SkyDiscover fills the hardest gaps. ~~MARS correctly distributes fixed gradients per turn~~ [DROPPED]. ~~CPPO ensures only viable candidates consume expensive eval~~ [DROPPED]. The original claim of **"2.05x more effective signal than CUDA-Agent at 30-60x lower cost"** is **[SPECULATIVE]** and depends on the dropped techniques.
