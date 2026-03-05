# KernelForge: Final Engineering PRD
## Single Source of Truth — Replaces ALL Previous Documents
**Hackathon:** OpenEnv Hackathon SF — March 7-8, 2026 (SHACK15, San Francisco)
**Training GPU:** NVIDIA B200 192GB | **Target Kernels:** NVIDIA A100 (sm_80)
**Primary Model:** Qwen3-Coder-Next (80B MoE, 3B active, agentic RL-trained)
**Last Updated:** March 4, 2026

---

## 0. Locked Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Target GPU | A100 sm_80 | doubleGraph + CUDA Agent both target A100. Cross-compile on B200. |
| Training GPU | B200 192GB | Qwen3-Coder-Next FP8 = ~100GB. 92GB free for sandbox. |
| Primary model | Qwen3-Coder-Next FP8 | Trained on 800K executable tasks with agentic RL. 70.6% SWE-Bench. Tool calling native. 3B active = fast generation. |
| Backup model | Qwen3.5-9B 4-bit | If Coder-Next fails to load. 81.7 GPQA Diamond. |
| SkyDiscover LLM | GLM-5 via API ($1/M input) | Frontier 744B model for kernel evolution. ~$2-8/run. |
| RL algorithm | TRLOO-augmented GRPO via TRL GRPOTrainer + stacked mitigations | MARS per-turn credit assignment + TRLOO post-process with N/(N-1) scaling (fixes Dr. Kernel 25% gradient shrinkage, arXiv 2510.15414 + 2602.05885). Nsight structured rewards + log(speedup) continuous signal replace discrete {-1,1,2,3} (fixes lazy optimization). CPPO pruning (arXiv 2503.22342) reduces eval cost 2-4x. MASPO soft trust region (arXiv 2602.17550). GRPO experimental 10 steps local-only; SkyDiscover/SFT remain primary hedges. See Section 6.0.4. |
| Training pipeline | 3-stage: Warmup → RFT → GRPO | Pure RL collapses at step 17 (CUDA Agent Section 3.3). |
| Environment spec | OpenEnv (step/reset/state) | Hackathon framework requirement. |
| Kernel language | CUDA C++ | Aligns with CUDA Agent infra + doubleGraph patterns. |

---

## 1. What We're Building

An RL training system + evolutionary search system that teaches Qwen3-Coder-Next to write optimized CUDA kernels for A100. Four components:

**Component 1: CUDA Agent Evaluation Pipeline** (ByteDance)
- 6,000 PyTorch operator tasks (CUDA-Agent-Ops-6K dataset)
- Compile → verify → profile scripts from their agent_workdir
- Continuous reward function: log(speedup) + occupancy + coalescing (replacing discrete {-1, 1, 2, 3}). Floor baseline = torch.compile timing for Ops-6K tasks.
- Anti-reward-hacking measures
- Source: https://github.com/BytedTsinghua-SIA/CUDA-Agent

**Component 2: doubleGraph Expert Baselines** (doubleAI)
- A100-optimized CUDA kernels for graph algorithms (3.6x avg speedup over cuGraph)
- 4-layer drop-in architecture: Integration (C++ template hijacking via `#ifdef AAI_ROUTE_*`) → API Dispatch (4-way: base/seg/mask/seg_mask) → Infrastructure (`compact_graph_t` + `CachePool`) → Implementation (per-GPU `.cu` files + `.cu.flags` sidecars)
- Core algorithms: BFS (dual-frontier direction-optimizing), Louvain (3-tier adaptive dispatch), PageRank (fused SpMV with warp-level reduction), WCC (parallel union-find), Triangle Count (DAG orientation + bilateral intersection)
- Key A100 (SM80) patterns: degree-based 4-bin segmentation (≥1024/32-1023/1-31/0), thread-local LRU `CachePool` (capacity=8, type-erased via `Cacheable`), `__ballot_sync` warp aggregation, shared-memory hash tables with `atomicCAS`, `cudaHostAllocMapped` zero-copy convergence
- Hard constraints: single GPU only, int32 only, no DCS/hypersparse, Stream 0 execution, no hot-path error checking
- Reward calibration (graph tasks ONLY — 5 algorithms): cuGraph baseline timing = floor, doubleGraph expert timing = ceiling, per-algorithm thresholds from profiling. **NOTE:** This calibration applies ONLY to the 5 graph algorithm tasks (BFS, Louvain, PageRank, WCC, TC). For the 6,000 Ops-6K PyTorch operator tasks, use torch.compile timing as floor instead (matching CUDA-Agent's approach).
- Source: https://github.com/double-ai/doubleGraph
- Full engineering reference: `docs/skills/doublegraph_a100.md` (11-section deep dive)

**Component 3: SkyDiscover Evolutionary Search** (UC Berkeley)
- AdaEvolve + EvoX for kernel evolution via GLM-5 API
- Pure parallel hedge: 80-120 evolutions on 5 seed .cu files (gemm, softmax, layernorm, fused_bias_relu, reduce_sum)
- Do NOT feed the full 6K Ops-6K dataset — SkyDiscover's gpu_mode supports only ~4 benchmark tasks natively
- Parallel track: runs on separate machine while GRPO trains (no GPU contention)
- Source: https://github.com/skydiscover-ai/skydiscover

**Component 4: GRPO Training** (our contribution)
- 3-stage pipeline training Qwen3-Coder-Next 80B MoE (3B active) via Unsloth FP8 Dynamic on B200
- Multi-turn with compiler feedback + MARS+TRLOO per-turn credit assignment
- Nsight Compute structured rewards (occupancy, mem coalescing, warp efficiency) replacing discrete {-1,1,2,3}
- CUDA-Agent SKILL.md verbatim + doubleGraph pattern paste as prompt context (transformation grammar deferred to v2)
- Hybrid eval: local nvcc+PAC for early turns, Modal A100+ncu for final turn
- CPPO pruning (top-2 of G candidates after cheap structural filter)
- MASPO soft trust region replacing hard PPO clip
- OpenEnv-compatible environment, Unsloth + TRL on B200
- **Strategy:** GRPO is experimental (10 steps, local nvcc eval only, no Modal). SkyDiscover + SFT are primary hedges. See GRPO_DEEP_DIVE sections GRPO-9 through GRPO-14 for stacked architecture (note: GRPO-13 transformation grammar deferred to v2).

---

## 2. Repository Structure

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
│   └── stage3_grpo.py              # Task 14: GRPO curriculum (max 10 steps local-only)
│
├── openenv_wrapper/                # P0 — hackathon judges test step() first
│   ├── __init__.py
│   ├── models.py                   # Task 15: Pydantic models
│   ├── environment.py              # Task 16: OpenEnv Environment class
│   ├── app.py                      # Task 17: FastAPI server
│   └── Dockerfile                  # Task 18: Container for HF Spaces
│
├── skydiscover_integration/        # P1 — parallel track
│   ├── evaluator.py                # Task 19: SkyDiscover wrapper
│   ├── run_evolution.sh            # Task 20: Launch script
│   └── initial_kernels/            # Task 21: Starting kernels to evolve
│       ├── gemm.cu
│       ├── softmax.cu
│       └── layernorm.cu
│
├── doublegraph_integration/        # P1 — expert baselines
│   ├── extract_patterns.py         # Task 22: Extract A100 patterns
│   ├── graph_tasks.py              # Task 23: Format graph algo tasks
│   └── baseline_profiler.py        # Task 24: Profile cuGraph vs doubleGraph
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

## 3. Complete Task List

### PHASE 0: PREP (Wednesday-Friday, before hackathon)

#### Task 0.1: Download All Assets
**Priority:** P0 BLOCKING — start first, downloads take hours
**Time:** 2-4 hours (parallel)
**Owner:** Anyone with internet

**Subtask 0.1.1: Download Qwen3-Coder-Next FP8 (~85GB)**
```bash
mkdir -p ~/kernelforge && cd ~/kernelforge
pip install huggingface_hub
huggingface-cli download unsloth/Qwen3-Coder-Next-FP8-Dynamic \
    --local-dir ./models/coder-next-fp8
```
- Source: https://huggingface.co/unsloth/Qwen3-Coder-Next-FP8-Dynamic
- Unsloth docs: https://unsloth.ai/docs/models/qwen3-coder-next
- Technical report: https://arxiv.org/pdf/2603.00729

**Subtask 0.1.2: Download backup model Qwen3.5-9B (~18GB)**
```bash
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ./models/qwen35-9b
```
- Source: https://huggingface.co/Qwen/Qwen3.5-9B

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

**Subtask 0.1.4: Clone all repos**
```bash
cd ~/kernelforge
git clone https://github.com/BytedTsinghua-SIA/CUDA-Agent
git clone https://github.com/double-ai/doubleGraph
git clone https://github.com/skydiscover-ai/skydiscover
git clone https://github.com/meta-pytorch/OpenEnv
```

**Subtask 0.1.5: Download doubleGraph A100 wheels** (STRETCH GOAL — skip if B200 wheel incompatible)
```bash
# STRETCH GOAL: Wheel may not install on B200. If it fails, skip entirely.
# The 330-line docs/skills/doublegraph_a100.md provides all patterns needed for prompts.
mkdir -p ./wheels
wget -P ./wheels/ [A100_WHEEL_URL]  # Check https://github.com/double-ai/doubleGraph/releases/
```
- Source: https://github.com/double-ai/doubleGraph/releases/
- **Fallback:** Extract patterns into SKILL.md from docs/skills/DOUBLEGRAPH_A100.md (already complete)

**Done when:** All 5 subtasks complete. Verify with:
```bash
ls models/coder-next-fp8/  # Model files exist
ls models/qwen35-9b/       # Backup model exists
python -c "from datasets import load_from_disk; print(len(load_from_disk('./data/ops_6k')['train']))"  # 6000
ls CUDA-Agent/agent_workdir/utils/  # compile.sh, verification.py, profiling.py
ls doubleGraph/cpp/src/aai/impl/a100/  # CUDA kernel directories
ls skydiscover/  # SkyDiscover repo
```

---

#### Task 0.2: Set Up B200 Environment
**Priority:** P0 BLOCKING
**Time:** 1-2 hours

**Subtask 0.2.1: Install Python packages**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install --no-deps unsloth unsloth_zoo
pip install "trl[vllm]>=0.29.0"
pip install transformers>=4.56.2 datasets accelerate peft
pip install openenv-core>=0.2.1
pip install cupy-cuda12x
```

**Subtask 0.2.2: Verify GPU and CUDA**
```bash
nvidia-smi                    # B200, 192GB
nvcc --version                # CUDA 12.x+
python -c "import torch; print(torch.cuda.get_device_name())"
python -c "import trl; print(trl.__version__)"  # ≥0.29.0
python -c "import unsloth; print('OK')"
```

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
STRETCH GOAL: If this fails (likely on B200), skip entirely. Dense ops + doubleGraph patterns from docs/skills/doublegraph_a100.md are sufficient. Runtime doubleGraph is NOT needed for Ops-6K reward calibration (those use torch.compile as floor).

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

    If doubleGraph wheel fails to install (B200 incompatible, A100/L4/A10G only),
    skip live profiling and use documented 3.6x average speedup estimates from
    docs/skills/doublegraph_a100.md patterns instead.
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
    """Fallback when doubleGraph wheel is not installed (e.g. B200)."""
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
| test_05_model_load.py | Qwen3-Coder-Next loads, generates CUDA | YES |
| test_06_grpo_init.py | TRL GRPOTrainer initializes without crash | YES |
| test_07_compilation_rate.py | Base model CUDA compilation rate (target: >50%) | INFORMATIONAL |
| test_08_full_grpo_step.py | Complete GRPO step with real evaluation | YES |

**In test_05, use Qwen3-Coder-Next with its recommended params:**
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-Coder-Next-FP8-Dynamic",
    max_seq_length=16384, load_in_4bit=False,
)
# Generate with temp=1.0, top_p=0.95, top_k=40, min_p=0.01
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

Check compiled .so for forbidden symbols via `nm -D`. Verify file exists and is non-trivial.

For hackathon MVP: compilation + symbol check = "verified." Full I/O correctness checking (run kernel on 5 inputs, compare to reference) is a P1 upgrade.

Key function: `verify_kernel(so_path: str, task_code: str) -> VerifyResult`

(Full implementation in Engineering PRD v2.)

---

#### Task 4: `evaluation/profiler.py` — Baseline Timing
**Priority:** P0 | **Time:** 1 hour | **Depends on:** Task 1

Profile torch.eager and torch.compile for a task. CUDA events, 50 warmup, 30 runs, trimmed mean. Runs in subprocess.

Key function: `profile_task(task_code: str) -> ProfileResult` with `eager_ms` and `compile_ms`.

(Full implementation in Engineering PRD v2.)

---

#### Task 5: `evaluation/reward.py` — GRPO Reward Function
**Priority:** P0 | **Time:** 1 hour | **Depends on:** Tasks 1-4

THE function passed to `GRPOTrainer(reward_funcs=...)`. Takes completions, extracts CUDA, compiles, verifies, returns continuous `log(speedup) + Nsight bonus` (see `openenv_env/reward.py`). TRLOO post-process with N/(N-1) scaling applied to advantages.

Key function: `cuda_kernel_reward(completions, prompts=None, task_code=None, **kwargs) -> list[float]`

**Signature MUST match TRL's expectation.** Extra dataset columns (task_code) forwarded via `**kwargs` when `remove_unused_columns=False`.

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

Load Ops-6K. Format prompts with SKILL.md. Filter by stage (warmup=single-op, curriculum=all). Include `task_code` column for reward function forwarding.

Key function: `load_training_dataset(stage: str) -> Dataset`

(Full implementation in Engineering PRD v2.)

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

The main training script. Loads Qwen3-Coder-Next with Unsloth, runs 3-stage pipeline.

**Updated for Qwen3-Coder-Next:**

```python
# Default command (Saturday morning):
python -m training.run \
    --model unsloth/Qwen3-Coder-Next-FP8-Dynamic \
    --use_fp8 \
    --output_dir ./checkpoints \
    --stage all

# If FP8 fails, fallback:
python -m training.run \
    --model Qwen/Qwen3-Coder-Next \
    --output_dir ./checkpoints \
    --stage all
    # (loads in 4-bit by default)
```

Key differences from generic model:
- Temperature: 1.0 (not 0.7-0.9) — Qwen3-Coder-Next trained at 1.0
- Learning rate: 2e-6 (Stage 1), 3e-6 (Stage 3) — lower to preserve existing capabilities
- Stage 1 may be skipped if compilation rate >70%

(Full implementation in Engineering PRD v2, with model loading updated per Model Update doc.)

---

#### Task 12-14: Stage Configs
**Priority:** P0 | **Time:** Included in Task 11

Three GRPOConfig objects for warmup / RFT / curriculum. All parameters specified in GRPO Deep Dive doc.

| Stage | Steps | Temp | LR | Max Tokens | Num Gens |
|-------|-------|------|-----|------------|----------|
| 1 (Warmup) | 50 | 1.0 | 2e-6 | 4096 | 4 |
| 2 (RFT) | N/A (SFT) | N/A | 5e-6 | N/A | N/A |
| 3 (GRPO) | 80 | 1.0 | 3e-6 | 4096 | 4 |

---

#### Task 15-18: OpenEnv Wrapper
**Priority:** P2 — build AFTER training works
**Time:** 2-3 hours
**Depends on:** Tasks 1-5

Wrap `evaluation/reward.py` in OpenEnv's `step()/reset()/state()` API. The core logic is identical — this is packaging.

- Task 15: `models.py` — KernelAction, KernelObservation, StepOutcome Pydantic models
- Task 16: `environment.py` — KernelForgeEnvironment(Environment) class
- Task 17: `app.py` — FastAPI application
- Task 18: Dockerfile for HF Spaces deployment

---

#### Task 19-21: SkyDiscover Integration
**Priority:** P1 — parallel track for demo numbers
**Time:** 2 hours
**Depends on:** Tasks 1-5

**Task 19:** `skydiscover_integration/evaluator.py` — wraps reward.py as SkyDiscover evaluator
```python
def evaluate(program_path: str) -> dict:
    with open(program_path) as f:
        cuda_code = f.read()
    # Reuse our evaluation pipeline
    from evaluation.compiler import compile_cuda
    from evaluation.verifier import verify_kernel
    result = compile_cuda(cuda_code)
    if not result.success:
        return {"combined_score": -1e9, "artifacts": {"feedback": result.stderr[:500]}}
    # Score based on compilation success + any profiling
    return {"combined_score": 1.0, "artifacts": {"feedback": "Compiled successfully"}}
```

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

---

#### Task 22-24: doubleGraph Integration
**Priority:** P1 — expert baselines + graph tasks
**Time:** 3 hours
**Depends on:** Task 0.4, `docs/skills/doublegraph_a100.md` (SKILLS.md reference)

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
- `evaluation/reward.py` (Task 5) — graph tasks use per-algorithm reward calibration:
  - r = -1: compilation fails OR correctness fails
  - r = +1: correct but slower than cuGraph baseline
  - r = +2: faster than cuGraph baseline
  - r = +3: within 10% of doubleGraph expert kernel timing
  - Per-algorithm thresholds computed from Task 24 profiling results

> **IMPORTANT (Fix 2):** The cuGraph floor / doubleGraph ceiling calibration above applies ONLY to the 5 graph algorithm tasks. For the 6,000 Ops-6K PyTorch operator tasks (GEMM, softmax, layernorm, etc.), doubleGraph has NO baselines. Those tasks use torch.compile timing as the floor, matching CUDA-Agent's approach. Do not conflate graph-task calibration with Ops-6K calibration.

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

> **Fix 6 (SFT first):** Do NOT attempt GRPO until Unsloth's built-in SFT trainer has run a warmup (Task 0.5 data + Stage 2 RFT). Verify >70% compile rate via local nvcc eval (no Modal) BEFORE attempting any GRPO steps. The TRL + Unsloth FP8 MoE + GRPO combination is untested at scale.

#### Task 26: Launch SkyDiscover (Parallel Track)
**Time:** 2 hours setup, runs in background
```bash
bash skydiscover_integration/run_evolution.sh
```

#### Task 27: Launch GRPO Training
**Time:** Runs 7-12 hours in background
```bash
python -m training.run \
    --model unsloth/Qwen3-Coder-Next-FP8-Dynamic \
    --use_fp8 --stage all --output_dir ./checkpoints
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

## 4. Critical Path

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

## 5. All Links

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
| **Qwen3-Coder-Next** | https://huggingface.co/Qwen/Qwen3-Coder-Next |
| Qwen3-Coder-Next FP8 | https://huggingface.co/unsloth/Qwen3-Coder-Next-FP8-Dynamic |
| Qwen3-Coder-Next GGUF | https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF |
| Qwen3.5-9B (backup) | https://huggingface.co/Qwen/Qwen3.5-9B |
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
| **Dr. Kernel** (HKUST/TikTok) | [2602.05885](https://arxiv.org/abs/2602.05885) | GRPO has biased policy gradients; proposes TRLOO (Turn-level Reinforce Leave-One-Out) | Our GRPO advantage estimation is systematically 25% too small (G=4). Multi-turn makes it worse. |
| **CUDA-L1** (DeepReinforce) | [2507.14111](https://arxiv.org/abs/2507.14111) | Contrastive RL achieves 3.12x avg speedup on A100 KernelBench | Pairwise comparisons > group normalization. Alternative to GRPO. |
| **KernelBlaster** (Stanford/NVIDIA) | [2602.14293](https://arxiv.org/abs/2602.14293) | Memory-augmented in-context RL for kernel generation | In-context learning approach — no policy gradient training needed. |
| **OptiML** | [2602.12305](https://arxiv.org/abs/2602.12305) | MCTS over LLM edits, no RL training needed | Search-based alternative that's more sample-efficient than GRPO. |
| **ConCuR** | [2510.07356](https://arxiv.org/abs/2510.07356) | Data quality > algorithm; curated reasoning traces beat RL | SFT on quality data may beat our GRPO pipeline. |
| **CUDABench** | [2603.02236](https://arxiv.org/abs/2603.02236) | "Mismatch between high compilation success and low functional correctness" | Compilation-only reward (our current state) is a bad proxy for kernel quality. |

### Stacked RL Techniques (March 2026)

These papers provide the specific techniques that address each Fundamental Failure in Section 6.0, making GRPO viable on single-GPU. See GRPO_DEEP_DIVE sections GRPO-9 through GRPO-14.

| Paper | arXiv / Link | Role in Stack | Addresses |
|-------|-------------|---------------|-----------|
| **MARSHAL / MARS** | [2510.15414](https://arxiv.org/abs/2510.15414) | Turn-level cumulative advantage (+23% return, ICLR 2026) | Failure #1 (GRPO bias) |
| **OAPL** | [2602.19362](https://arxiv.org/abs/2602.19362) | Off-policy advantage regression (handles 400-step async lag) | Failure #3 (async Modal eval) |
| **AReaL** | [2505.24298](https://arxiv.org/abs/2505.24298) | Async decoupled training (2.77x wall-time speedup) | Failure #3 (compute scale) |
| **MASPO** | [2602.17550](https://arxiv.org/abs/2602.17550) | Gaussian soft trust region replacing PPO clip (+2-3%) | Failure #1 (sparse reward collapse) |
| **CPPO** | [2503.22342](https://arxiv.org/abs/2503.22342) | Completion pruning (7.98x speedup on GSM8K) | Failure #3 (eval cost) |
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

## 6. Risk Matrix & Mitigations

### Codebase Reality Check

| Layer | Status | LoC | Notes |
|-------|--------|-----|-------|
| OpenEnv environment | Complete | ~600 | Full step/reset/state contract, tested |
| Training pipeline | Skeletal | ~1,150 | Scaffolding exists, depends on Modal backend |
| Evaluation + Verification | Mostly complete | ~1,260 | PAC verify + profiling work, reward bridge missing |
| CUDA kernels | Minimal | ~670 | WCC only (3 variants). No BFS/PageRank/Louvain. |
| Datasets | Complete | ~1,300 | 6,200 tasks pre-generated and formatted |
| SkyDiscover integration | Not implemented | 0 | Directory doesn't exist |
| doubleGraph integration | Not implemented | 0 | Directory doesn't exist; docs only |
| `training/grpo_train.py` | Missing | 0 | Referenced in pyproject.toml + README but doesn't exist |

**Bottom line:** ~70% of infrastructure exists. The training pipeline assumes an external Modal deployment. Two of four components have zero code.

---

### 6.0 Fundamental Research Risks

> **Why this section exists:** Sections 6.1-6.8 cover *implementation* risks (stubs, missing code, memory budgets). This section asks the deeper question: **even if we implement everything perfectly, are the fundamental research approaches sound?** Six concurrent papers (Feb-Mar 2026) in the CUDA kernel RL space say no — at least not as currently designed.

#### 6.0.1 FUNDAMENTAL FAILURE #1: GRPO Is Mathematically Biased for Multi-Turn Kernel Generation

**The problem:** Dr. Kernel (arXiv [2602.05885](https://arxiv.org/abs/2602.05885)) proves that GRPO's advantage estimation is biased because of *self-inclusion* — computing the group mean/std using the current sample itself.

The expected gradient under GRPO is shrunk by a factor of `(1 - 1/N)`:
```
E[ĝ_GRPO] = (1 - 1/N) × ∇θJ(θ)
```

With our G=4: gradients are systematically 25% too small. In multi-turn settings (which our PRD explicitly uses — "multi-turn with compiler feedback"), this compounds across turns because fewer samples survive to later turns, making N smaller and the bias worse.

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

**Pivot options:**
1. Switch from GRPO to TRLOO (TRL may not support this — would need custom training loop)
2. Switch to contrastive RL à la CUDA-L1 (compare good vs bad kernels directly)
3. Accept GRPO bias and increase G from 4 to at least 8 (requires memory optimization)
4. Drop multi-turn and use single-turn only (reduces bias accumulation)

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

**Why this is fundamental, not just a TODO:** Even if we implement profiling, discrete rewards {-1,1,2,3} lose critical signal. Dr. Kernel uses continuous profiling-based rewards. CUDA-L1 uses contrastive pairs of (slow kernel, fast kernel). Our 4-level discrete scheme cannot distinguish:
- A kernel that's 1.06x faster (barely r=+2) from one that's 5x faster (also r=+2)
- The model has zero incentive to optimize beyond the threshold

**Revised approach (Fix 5):** Replace discrete {-1,1,2,3} with continuous reward:
- `reward = log(speedup_vs_eager)` for correct kernels (continuous signal proportional to actual improvement)
- Add occupancy and coalescing bonuses from Nsight Compute (or subprocess profiling.py from CUDA-Agent)
- Wrap with TRLOO N/(N-1) post-processing for unbiased advantage estimation
- Reuse CUDA-Agent's profiling.py and verification.py via subprocess (copy verbatim, do NOT rewrite)

**Pivot options:**
1. Implement continuous reward: `reward = log(speedup)` instead of discrete bins
2. Add profiling-based rejection sampling (Dr. Kernel's PRS): only train on completions that BOTH compile AND show measurable speedup
3. Use contrastive pairs: generate 2 kernels, reward the faster one relative to the slower one
4. At minimum: implement the profiling TODO before any training run

---

#### 6.0.3 FUNDAMENTAL FAILURE #3: Our Compute Scale Is 100x Below What Works

**The problem:** Every successful CUDA kernel RL system operates at a scale we cannot match:

| System | Model | GPUs | Batch | Context | Steps |
|--------|-------|------|-------|---------|-------|
| **CUDA Agent** | 230B (23B active) | 128x H20 | 1024 | 131K tokens | 150 |
| **Dr. Kernel** | 14B | Distributed KernelGYM (multiple workers) | Large | 32K+ | 200+ |
| **CUDA-L1** | Custom | A100 cluster | Full KernelBench (250 tasks) | — | Full RL loop |
| **Us (PRD)** | 80B (3B active) | 1x B200 | 1 prompt x 4 completions | 4K tokens | 100 |

Our effective batch size is **4 completions**. CUDA Agent's is **1024**. That's 256x more gradient signal per step. CUDA Agent trains for 150 steps x 1024 batch = 153,600 effective samples. We train for 100 steps x 4 = 400 effective samples. **That's 384x fewer samples.**

**Why this matters from first principles:** RL for code generation has an astronomically sparse reward landscape. Most of the optimization space is "doesn't compile" (r=-1). The probability of randomly generating an optimized CUDA kernel is negligible. You need massive sampling to find the rare positive signals. With G=4, most steps will have all-negative rewards (std=0, zero gradient). CUDA Agent solved this with scale. We can't.

**The "3B active" misleadingness:** Our PRD highlights that Qwen3-Coder-Next has "only 3B active parameters" as if this makes it lightweight. But:
- The 80B total parameters still need 85GB for weights
- MoE routing overhead is real
- 3B active is much smaller than CUDA Agent's 23B active — less model capacity for learning kernel optimization
- ConCuR (arXiv [2510.07356](https://arxiv.org/abs/2510.07356)) shows data quality matters more than model size — a well-curated dataset beats throwing a bigger model at the problem

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
| **#1: GRPO bias** | Self-inclusion in group mean/std | MARS cumulative per-turn returns + TRLOO leave-one-out baseline | MARSHAL (arXiv 2510.15414) + Dr. Kernel (arXiv 2602.05885) |
| **#1 (post-process)** | GRPO gradient shrinkage = (1 - 1/N) | TRLOO post-process: multiply advantages by N/(N-1) scaling factor after GRPO computation. Drop-in fix, no custom training loop needed. | Dr. Kernel (arXiv 2602.05885) |
| **#2: Lazy reward** | Discrete {-1,1,2,3} loses signal; profiling is a stub | Continuous log(speedup) + Nsight Compute structured reward: occupancy + mem_coalescing + warp_efficiency | CUDABench (arXiv 2603.02236) |
| **#2 (action space)** | Free-form 2000-token generation = huge search space | Rich SKILL.md prompt context (CUDA-Agent verbatim + doubleGraph patterns) constrains generation. Transformation grammar deferred to v2. | CUDA-Agent SKILL.md, doubleGraph docs |
| **#3: Compute scale** | 384x fewer samples than CUDA Agent | CPPO pruning (only eval top-2 of G), hybrid eval (local+Modal), smaller action space | CPPO (arXiv 2503.22342) |
| **#3 (throughput)** | Sync eval blocks training | MASPO soft trust region + AReaL-style replay buffer (staleness η=4-8) | MASPO (arXiv 2602.17550), AReaL (arXiv 2505.24298) |

**Updated probability assessment:**

| Approach | Probability | Time | Why |
|----------|------------|------|-----|
| **SkyDiscover evolutionary search** | HIGH | 2-4 hrs | AdaEvolve proven on 200+ tasks. No training. Primary hedge. |
| **GRPO with stacked techniques** | MEDIUM | 4-6 hrs training | MARS+TRLOO fixes bias. Nsight fixes reward. CPPO cuts cost. Rich SKILL.md constrains generation. 10-step local-only experiment on Qwen3-Coder-Next 80B FP8. |
| **SFT on curated data** (ConCuR-style) | MEDIUM-HIGH | 3-5 hrs | ConCuR shows curated traces > RL. Parallel track. |
| **GRPO as originally designed in PRD** | LOW | 7-12 hrs | Without stacking, still biased, stub reward, too few samples. |

**Implementation priority order (revised — 4-tier minimal viable approach):**

| Priority | When | Activity | Deliverable |
|----------|------|----------|-------------|
| **P0** | By Friday night | OpenEnv env that calls CUDA-Agent's EXACT scripts (copied verbatim into evaluation/) + continuous speedup reward (log(speedup) + occupancy + coalescing) + TRLOO N/(N-1) post-process in reward wrapper | Working eval pipeline with continuous reward |
| **P1** | Parallel with P0 | SkyDiscover evolution on 5 seed .cu files (gemm, softmax, layernorm, fused_bias_relu, reduce_sum). 80-120 evolutions per seed. | Demo-worthy evolved kernels |
| **P2** | After P0 | Rich SKILL.md with doubleGraph patterns (from docs/skills/doublegraph_a100.md) + CUDA-Agent SKILL.md verbatim. 50-example SFT warmup via Unsloth built-in trainer. | Anchored compilation ability |
| **P3** | Optional demo | 10-step GRPO with local-only nvcc eval (no Modal). Gate G-0.8 must pass first. | Proof-of-concept RL training |

**Total new code:** ~200 LOC across 4 files (down from ~450 — transformation grammar dropped). Fits in 8-12 hours coding + 2-4 hours training.

**Realistic expected outcome with Qwen3-Coder-Next 80B FP8:**
- 85-95% compile+correct, 1.6-2.1x geo-mean on curated_200 + WCC
- Discovery of 8-12 A100-specific patterns (float4, L2 pinning, shuffles, tiling)
- Total evals: ~200-400 (P3 is 10-step demo; SkyDiscover provides bulk of evolved kernels)

**Why this revision succeeds where original PRD fails:**
- Fixes GRPO bias (MARS+TRLOO post-process)
- Fixes lazy reward (continuous log(speedup) + Nsight)
- Fixes eval bottleneck (local nvcc for P3, no Modal dependency)
- Fixes action space (rich SKILL.md context, grammar deferred)
- Fixes scope (GRPO demoted to P3 optional demo, SkyDiscover/SFT are primary)
- Fixes doubleGraph scope (graph calibration for 5 tasks only, torch.compile for Ops-6K)
- Total timeline realistic (~36 hrs with 7 hr buffer)

---

### 6.1 Component 1: CUDA Agent Evaluation Pipeline — Top 3 Failures

| # | Failure | Severity | Evidence | Mitigation |
|---|---------|----------|----------|------------|
| 1.1 | **Reward function is a stub** — `compute_reward()` referenced in `training/multi_turn_rollout.py:54` but NOT DEFINED. No correctness checking (GRPO_DEEP_DIVE line 339: "TODO"). No kernel profiling (line 368: placeholder `kernel_ms = compile_ms`). | CRITICAL | Code inspection: function imported but missing. GRPO_DEEP_DIVE lines 332-369. | **Gate 0.3**: `openenv_env/reward.py` now implements continuous `log(speedup) + Nsight bonus` with `trloo_post_process()` (~60 lines). Wire to compilation result from Modal. |
| 1.2 | **Modal is single point of failure** — every eval routes through `modal.Function.from_name()`. If Modal API is down/slow during hackathon, training halts. No local fallback. | CRITICAL | `training/multi_turn_rollout.py:35-51`. All GRPO steps call Modal. | Add local eval fallback: `nvcc -arch=sm_80` subprocess + `nm -D` symbol check. 45 min. Use Modal for profiling only; local for compile+verify. |
| 1.3 | **60s subprocess timeout too aggressive** — compilation (10-30s) + correctness (5s) + profiling (20-60s) = 35-95s total. Timeout kills valid kernels. | HIGH | GRPO_DEEP_DIVE line 398: `timeout=60`. Profiling budget line 956: "20-60s". | Increase to `timeout=120`. Or split: compile timeout=30s, profile timeout=90s separately. |

**Counter-argument (why it can still work):** CUDA Agent (arXiv 2602.24286) demonstrated 2.11x speedup using discrete rewards on Ops-6K. Our revised continuous `log(speedup) + Nsight bonus` (implemented in `openenv_env/reward.py`, ~60 lines) is strictly better — it provides gradient signal even between compilation milestones. `trloo_post_process()` fixes the Dr. Kernel gradient shrinkage.

---

### 6.2 Component 2: doubleGraph Expert Baselines — Top 3 Failures

| # | Failure | Severity | Evidence | Mitigation |
|---|---------|----------|----------|------------|
| 2.1 | **Zero integration code exists** — Tasks 22-24 are pseudocode in this PRD. No `doublegraph_integration/` directory. No `extract_patterns.py`, `graph_tasks.py`, `baseline_profiler.py`. | HIGH | Codebase scan: directory doesn't exist. | Demote to P2. Use doubleGraph docs (`docs/skills/doublegraph_a100.md`) as prompt context only — no runtime dependency. Extract patterns manually into `data/a100_patterns.md` as static file. 1 hour. |
| 2.2 | **Wheel may not install on B200** — Task 0.1.5 is a placeholder (`wget [A100_WHEEL_URL]`). No actual URL. Cross-GPU wheel compatibility unverified. | HIGH | Section 3, Task 0.1.5: placeholder URL. Task 0.2.4: "If this fails: stretch goal." | Accept this risk. If wheel fails, skip. The 330-line SKILLS.md provides enough pattern context for prompts WITHOUT runtime doubleGraph. |
| 2.3 | **Only WCC kernels exist in repo** — `kernels/` has 3 WCC variants. No BFS, Louvain, PageRank, Triangle Count. The "5 primary algorithms" coverage is aspirational. | MEDIUM | `kernels/`: `baseline_wcc.cu`, `ecl_cc_h100.cu`, `clustered_wcc_h100.cu`. Nothing else. | Focus hackathon on WCC + Ops-6K dense operators (6,000 tasks in dataset). Graph algo breadth is stretch goal. |

**Counter-argument:** doubleGraph's primary value is as **prompt context**, not runtime dependency. The 330-line `docs/skills/doublegraph_a100.md` contains all A100 optimization patterns (4-bin segmentation, CachePool, `__ballot_sync`, shared-mem hash tables). These get pasted into SKILL.md — the model sees the patterns regardless of whether doubleGraph wheels install. WarpSpeed research (doubleai.com) validates the 3.6x claim; we cite it, we don't need to reproduce it.

---

### 6.3 Component 3: SkyDiscover Evolutionary Search — Top 3 Failures

| # | Failure | Severity | Evidence | Mitigation |
|---|---------|----------|----------|------------|
| 3.1 | **Zero implementation exists** — no `skydiscover_integration/` directory, no evaluator, no launch script, no seed kernels. | HIGH | Codebase scan: directory doesn't exist. | Create minimal integration: 1 evaluator file (reuse `openenv_env/reward.py`), 1 launch script, 3 seed kernels. 2 hours total. OR skip entirely and use GRPO results for demo. |
| 3.2 | **GLM-5 API not configured** — no API key setup, no pyproject.toml dependency, no test of API endpoint. Cost estimate ($2-8/run) is unvalidated. | MEDIUM | No code references GLM-5 API client. Section 0, Locked Decisions just names the price. | Register for GLM-5 API key Wednesday. Test with 1 call. If API unavailable, substitute any OpenAI-compatible endpoint (Claude, GPT-4). AdaEvolve (arXiv 2602.20133) is model-agnostic. |
| 3.3 | **VRAM contention** — Section 1 says "parallel track: runs while GRPO trains" but both compete for B200 GPU. | MEDIUM | Section 1, Component 3. No resource isolation in code. | Run SkyDiscover on a SEPARATE machine or Modal instance. It only needs nvcc + API calls, not 192GB VRAM. Or run sequentially: SkyDiscover first (2-3 hrs), then GRPO. |

**Counter-argument:** AdaEvolve (arXiv 2602.20133) demonstrated 10-40% gains over compiler baselines with 80-120 LLM calls. EvoX (arXiv 2602.23413) adds meta-evolution that automatically discovers optimization tactics. The framework is proven — the risk is purely implementation time, not algorithmic soundness. Even a minimal 10-iteration run produces demo-worthy results.

---

### 6.4 Component 4: GRPO Training — Top 3 Failures

| # | Failure | Severity | Evidence | Mitigation |
|---|---------|----------|----------|------------|
| 4.1 | **Memory budget unverified** — FP8 (~85GB) + LoRA (0.2GB) + optimizer (3GB) + KV cache (12GB) + activations = ~100GB estimated. No actual B200 measurement. Gradient checkpointing + GRPO interaction untested. | CRITICAL | GRPO_DEEP_DIVE lines 182-193. Estimates only, no benchmarks. | **Gate G-0.2**: Run `nvidia-smi` during model load + single GRPO step. If OOM: (a) reduce max_seq_length 8192->4096, (b) reduce G=4->2, (c) switch to 4-bit quantization, (d) use backup Qwen3.5-9B (18GB). Decision ladder, not binary. |
| 4.2 | **TRL GRPOTrainer API compatibility** — `rollout_func` parameter is experimental. `reward_funcs` vs `reward_func` naming uncertain. `remove_unused_columns=False` behavior unverified with TRL >=0.29. | HIGH | GRPO_DEEP_DIVE line 566 (`reward_funcs`), line 1080 (`remove_unused_columns`). | **Gate G-0.6**: `test_06_grpo_init.py` — initialize GRPOTrainer with our config, verify no crash. 15 min. If API changed: read TRL 0.29 changelog, adapt. The core GRPO math is model-agnostic; only the TRL wrapper changes. |
| 4.3 | **G=4 + low compilation rate = zero-gradient steps** — if base model compiles <30% of CUDA, P(all 4 fail) > 24%. Those steps produce std(r)=0 -> advantage=0 -> no learning. | HIGH | GRPO_DEEP_DIVE line 73 (edge case), lines 1095-1110 (math). No empirical compilation rate for Qwen3-Coder-Next. | **Gate G-25** (Saturday morning): test_07 measures compilation rate. If <50%: extend Stage 1 warmup. If <30%: switch model. CUDA Agent (arXiv 2602.24286, Section 3.3) shows 3-stage pipeline prevents step-17 collapse — our warmup follows their protocol. |

**Counter-argument:** GRPO (arXiv 2402.03300, DeepSeekMath) has been validated in multiple domains. The "step-17 collapse" from CUDA Agent Section 3.3 is specifically for PURE RL without warmup — our 3-stage pipeline addresses this. Qwen3-Coder-Next (arXiv 2603.00729) was trained on 800K executable code tasks with agentic RL, giving it strong baseline CUDA familiarity. Temperature=1.0 matches its training distribution. The risk is real but bounded by our decision gates.

**Stacked mitigations (Section 6.0.4 revision):** MARS+TRLOO hybrid credit assignment directly fixes the Dr. Kernel bias. G=2 (reduced from G=4) cuts zero-gradient probability: P(all 2 fail) = 0.7² = 49% at 30% compilation rate vs 0.7⁴ = 24% — still high, but CPPO pruning means only top-2 candidates get full eval, and Nsight structured rewards provide continuous signal even when both candidates compile (no more std=0 dead zones for reward levels between {1,2,3}). 50 steps (not 100) limits exposure. Qwen3-Coder-Next 80B MoE (3B active, FP8) is the primary model — trained on 800K executable tasks with strong CUDA baseline.

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
| **G-0.2** | Thu morning | B200 environment | `nvidia-smi` shows B200 192GB, `nvcc --version` >=12.x, `nvcc -arch=sm_80 test.cu` works | Fix CUDA install; if B200 unavailable, rent A100 from Lambda/RunPod |
| **G-0.3** | Thu afternoon | Reward function works | `compute_reward(compiled=True, correct=True, speedup_vs_eager=1.5, speedup_vs_compile=1.0)` returns `log(1.5) ≈ 0.405` | Implement missing function (30 min, BLOCKING) |
| **G-0.5** | Thu night | SFT data generated | `data/sft_data.json` has >=50 compilable examples | Lower bar to 30; if API fails, use Ops-6K curated_200 as warmup data |
| **G-0.6** | Fri morning | GRPOTrainer initializes | test_06 passes, no OOM, no API errors | Adapt to TRL API changes (read changelog) |
| **G-0.7** | Fri afternoon | Full GRPO step works | test_08 completes: generate -> compile -> reward -> gradient update | Debug loop; if unfixable, fall back to SFT-only + SkyDiscover |
| **G-0.8** | Thu night | 5-step GRPO sanity | Run 5 GRPO steps with continuous reward + TRLOO post-process. Must see: (a) non-zero gradients in 4/5 steps, (b) reward variance > 0 in 3/5 steps, (c) at least 2/20 test kernels achieve >1.2x speedup vs torch.compile. | Zero gradients → reward function broken. No speedup → model cannot optimize, fall back to SFT + SkyDiscover only. Do NOT continue to P3 GRPO. |
| **G-25** | Sat 9 AM | Compilation rate | test_07: base model >=50% compilation rate | <50%: extend warmup. <30%: switch model. <10%: abandon GRPO, go SkyDiscover-only. |

---

### 6.7 Realistic Timeline (with failure buffer)

**Updated per 10-fix critique:** P0 (eval pipeline) must work before anything else. SkyDiscover runs in parallel as primary hedge. GRPO demoted to P3 (optional 10-step demo).

| Block | Time | Activity | Failure Buffer |
|-------|------|----------|---------------|
| **Wed night** | 4 hrs | Downloads + API key setup + GLM-5 API registration | Re-download Thu morning if needed |
| **Thu 9AM-12PM** | 3 hrs | B200 setup + CUDA verify + Gate G-0.2. Load Qwen3-Coder-Next 80B FP8 via Unsloth. | 1 hr debug buffer |
| **Thu 1PM-6PM** | 5 hrs | **P0 — Core pipeline:** Copy CUDA-Agent eval scripts verbatim into evaluation/. Wire continuous reward function (log(speedup) + occupancy + coalescing). Add TRLOO N/(N-1) post-process wrapper. Gate G-0.3. | 2 hr buffer |
| **Thu 6PM-11PM** | 5 hrs | **P1 — SkyDiscover (parallel):** Set up evaluator on 5 seed .cu files. Launch 80-120 evolution runs. **P2 — SKILL.md:** Copy CUDA-Agent SKILL.md verbatim. Extract doubleGraph patterns into prompt context. | SkyDiscover runs overnight |
| **Fri 9AM-1PM** | 4 hrs | **P2 — SFT warmup:** 50-example SFT via Unsloth built-in trainer. Verify >70% compile rate with local nvcc. Gate G-0.8: 5-step GRPO sanity check. | If compile rate <70%, extend SFT |
| **Fri 1PM-5PM** | 4 hrs | **P3 — Optional GRPO:** 10-step GRPO with local-only eval (no Modal). Only if G-0.8 passes. | If GRPO stalls, SFT + SkyDiscover results are the submission |
| **Sat 9AM** | 30 min | **Gate G-25**: compilation rate decision. Evaluate all tracks. | Decision tree in 6.6 |
| **Sat 9:30AM-5PM** | 7.5 hrs | Collect best kernels from all tracks. OpenEnv wrapper. Run final eval suite. | SkyDiscover is the hedge |
| **Sat 5PM-10PM** | 5 hrs | Results collection, demo, blog | 2 hr buffer |

**Total available: ~43 hrs. Total planned: ~36 hrs. Buffer: ~7 hrs (16%).**

**Key change from previous timeline:** P0 (eval pipeline + continuous reward) must work before anything else. SkyDiscover runs in parallel as the primary hedge. GRPO is demoted to P3 (optional 10-step demo) — it only runs if G-0.8 passes. Transformation grammar dropped from v1. The eval pipeline reuses CUDA-Agent's scripts verbatim rather than rewriting them.

---

### 6.8 Minimum Viable Submission (If Everything Fails)

Even if GRPO collapses AND SkyDiscover API is unavailable, we ship:

1. **OpenEnv environment** — fully functional, ~600 LoC, tested (ship as-is)
2. **WCC baseline kernels** — 3 variants with PAC verification (ship as-is)
3. **Dataset pipeline** — 6,200 tasks curated and formatted (ship as-is)
4. **Evaluation framework** — compilation, verification, profiling (ship as-is)
5. **doubleGraph SKILLS.md** — 330-line A100 optimization reference (ship as-is)

This is a credible hackathon submission. The infrastructure IS the contribution — a complete OpenEnv-compatible CUDA kernel optimization environment with real evaluation, real datasets, and real verification. The RL training is the stretch goal that makes it exceptional.
