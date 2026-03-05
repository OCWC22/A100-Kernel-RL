# KernelForge — Product Requirements Document
## PyTorch OpenEnv Hackathon | March 7-8, 2026 | SHACK15 SF
**Last Updated: March 4, 2026**

---

## 1. Product Overview

**Product:** KernelForge — An OpenEnv-compatible RL environment for CUDA kernel optimization targeting NVIDIA A100 (sm_80).

**One-line pitch:** The first open RL environment where AI agents learn to write CUDA kernels that beat torch.compile, combining ByteDance's CUDA Agent evaluation infrastructure, doubleAI's expert A100 graph kernels, and Berkeley's SkyDiscover evolutionary search.

**Training Hardware:** NVIDIA B200 192GB HBM3e
**Target Kernel Architecture:** NVIDIA A100 (sm_80)
**Primary Model:** Qwen3-Coder-Next (80B MoE, 3B active) at FP8
**Hackathon:** Cerebral Valley × PyTorch OpenEnv, $100K+ prizes, teams up to 4

---

## 2. Goals & Non-Goals

### Goals
1. Build a working OpenEnv environment that evaluates CUDA kernels (compile → verify → profile → reward)
2. Demonstrate kernels that beat torch.compile on A100-targeted workloads (via SkyDiscover)
3. Demonstrate RL training (GRPO) that shows measurable improvement over training steps
4. Support both dense ops (from CUDA-Agent-Ops-6K) and graph ops (from doubleGraph)
5. Deploy to HuggingFace Spaces

### Non-Goals
- Matching CUDA Agent's 2.11x speedup across all of KernelBench (they used 128 GPUs)
- Supporting Triton AND CUDA simultaneously in the GRPO loop (CUDA only — simpler, aligns with CUDA Agent's proven infrastructure)
- Training from scratch (we use SFT warmup + GRPO, not the 4-stage pipeline CUDA Agent used)
- Building our own verification/profiling scripts (we wrap CUDA Agent's existing ones)

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  CONSUMER A: SkyDiscover (GLM-5 API)                         │
│  Evolves full CUDA kernels iteratively                       │
│  Gives us "beat torch.compile" headline numbers              │
│  https://github.com/skydiscover-ai/skydiscover               │
└───────────────────────┬──────────────────────────────────────┘
                        │ calls evaluate()
┌───────────────────────┼──────────────────────────────────────┐
│  CONSUMER B: GRPO     │ (Qwen3-Coder-Next on B200)           │
│  Trains model to      │ write better CUDA kernels            │
│  Gives us "RL works"  │ training curve                       │
└───────────────────────┼──────────────────────────────────────┘
                        │ calls step()
┌═══════════════════════▼══════════════════════════════════════┐
║  KERNELFORGE OPENENV ENVIRONMENT                             ║
║                                                              ║
║  Built on: CUDA Agent's agent_workdir scripts                ║
║  https://github.com/BytedTsinghua-SIA/CUDA-Agent             ║
║                                                              ║
║  reset() → pick task, profile torch.compile baseline         ║
║  step(cuda_code) → compile → verify → profile → reward       ║
║                                                              ║
║  TASK SOURCES:                                               ║
║  ├─ CUDA-Agent-Ops-6K (6,000 dense ops)                     ║
║  │  https://huggingface.co/datasets/BytedTsinghua-SIA/       ║
║  │  CUDA-Agent-Ops-6K                                        ║
║  └─ doubleGraph A100 (graph algorithms)                      ║
║     https://github.com/double-ai/doubleGraph                 ║
║                                                              ║
║  EVALUATION: CUDA Agent's compile.sh + verification.py +     ║
║  profiling.py wrapped in subprocess isolation                ║
║                                                              ║
║  COMPILATION: nvcc -arch=sm_80 -O3                           ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 4. Components & Their Roles

### 4.1 OpenEnv (Meta/PyTorch)
- **What:** Standard RL environment API — step(), reset(), state()
- **Role:** The interface spec we implement. Our environment IS an OpenEnv environment.
- **Repo:** https://github.com/meta-pytorch/OpenEnv
- **Docs:** https://meta-pytorch.org/OpenEnv/
- **Install:** `pip install openenv-core`
- **Key files to read:**
  - `envs/coding_env/` — closest existing example to what we're building
  - `rfcs/001-abstractions.md` — the spec
  - `examples/grpo_blackjack/` — GRPO training example

### 4.2 CUDA Agent (ByteDance/Tsinghua)
- **What:** SOTA CUDA kernel generation via agentic RL
- **Role:** We use their evaluation infrastructure (compile/verify/profile scripts) and their 6K training dataset. We do NOT use their model (not released) or their massive compute setup.
- **Repo:** https://github.com/BytedTsinghua-SIA/CUDA-Agent
- **Paper:** https://arxiv.org/abs/2602.24286
- **Project page:** https://cuda-agent.github.io/
- **Dataset:** https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K
- **Key files to use:**
  - `agent_workdir/` — the evaluation environment we wrap
  - `agent_workdir/SKILL.md` — prompt template for kernel writing
  - `agent_workdir/utils/compile.sh` — CUDA compilation script
  - `agent_workdir/utils/verification.py` — correctness checking
  - `agent_workdir/utils/profiling.py` — performance benchmarking

### 4.3 doubleGraph (doubleAI)
- **What:** AI-generated drop-in replacement for NVIDIA cuGraph, optimized for A100
- **Role:** Three things:
  1. **Graph algorithm tasks** for the environment (PageRank, BFS, SSSP, WCC, Louvain, cosine similarity, betweenness centrality)
  2. **Expert A100 kernel baselines** — known ceiling for reward calibration on graph tasks
  3. **A100 CUDA pattern library** — source of sm_80-specific optimization examples for prompts/SFT
- **Repo:** https://github.com/double-ai/doubleGraph
- **Blog:** https://www.doubleai.com/research/doubleais-warpspeed-surpassing-expert-written-kernels-at-scale
- **Pre-built A100 wheels:** https://github.com/double-ai/doubleGraph/releases/
- **Key directories:**
  - `cpp/src/aai/impl/a100/` — all A100-optimized kernels organized by algorithm category
  - `cpp/src/aai/impl/a100/components/` — WCC, connected components
  - `cpp/src/aai/impl/a100/link_prediction/` — cosine similarity
  - `cpp/src/aai/impl/a100/community/` — Louvain, Leiden
  - `cpp/src/aai/impl/a100/centrality/` — betweenness, PageRank

**How doubleGraph graph tasks work in the environment:**
```
Dense task (from 6K):
  Input: PyTorch Model.forward()
  Baseline: torch.compile time
  Ceiling: unknown
  Reward: speedup over torch.compile

Graph task (from doubleGraph):
  Input: graph algorithm spec + cuGraph reference implementation
  Baseline: cuGraph original time (SLOW)
  Ceiling: doubleGraph A100 kernel time (FAST — known)
  Reward: (cuGraph_time - agent_time) / (cuGraph_time - doubleGraph_time)
    0.0 = same speed as cuGraph (no improvement)
    1.0 = matches doubleGraph (expert AI level)
    >1.0 = beats doubleGraph (superhuman — unlikely but possible)
```

### 4.4 SkyDiscover (UC Berkeley Sky Lab)
- **What:** LLM-driven code evolution with adaptive search
- **Role:** Evolves CUDA kernels through our environment evaluator. Gives us fast results (~1-2 hours) while GRPO trains overnight. Safety net if GRPO doesn't converge.
- **Repo:** https://github.com/skydiscover-ai/skydiscover
- **Blog:** https://skydiscover-ai.github.io/
- **AdaEvolve Paper:** https://arxiv.org/abs/2602.20133
- **EvoX Paper:** https://arxiv.org/abs/2602.23413
- **X Thread:** https://x.com/shulynnliu/status/2028892335875276919
- **Install:**
  ```bash
  git clone https://github.com/skydiscover-ai/skydiscover && cd skydiscover
  uv sync --extra gpu_mode --extra external
  ```
- **Key files:**
  - `benchmarks/gpu_mode/` — existing GPU optimization benchmarks (adapt as reference)
  - `skydiscover/search/` — AdaEvolve and EvoX implementations

### 4.5 Qwen3-Coder-Next (Alibaba)
- **What:** 80B MoE coding model, 3B active params, 256K context, non-reasoning
- **Role:** Primary model for GRPO training. Generates CUDA kernels.
- **HuggingFace:** https://huggingface.co/Qwen/Qwen3-Coder-Next
- **FP8 (Unsloth):** https://huggingface.co/unsloth/Qwen3-Coder-Next-FP8-Dynamic
- **GGUF (Unsloth):** https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF
- **Unsloth docs:** https://unsloth.ai/docs/models/qwen3-coder-next
- **B200 memory at FP8:** ~85 GB (leaves 107 GB for CUDA sandbox)
- **Generation speed:** ~50 tokens/sec with 3B active params

### 4.6 GLM-5 (Z.ai)
- **What:** 744B MoE, 40B active, MIT license, strongest open coding model
- **Role:** LLM backend for SkyDiscover. Called via API only.
- **HuggingFace:** https://huggingface.co/zai-org/GLM-5
- **API:** https://api.z.ai/v1 (OpenAI-compatible)
- **Pricing:** $1.00/M input, $3.20/M output
- **SWE-bench:** 77.8%

### 4.7 Unsloth
- **What:** Memory-efficient LLM fine-tuning and RL training
- **Role:** Makes GRPO training of Qwen3-Coder-Next feasible on single GPU
- **Repo:** https://github.com/unslothai/unsloth
- **RL Guide:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
- **Memory-efficient RL:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl
- **OpenEnv Colab:** https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb

### 4.8 Supporting Tools
| Tool | Role | Link |
|------|------|------|
| TRL | GRPO trainer implementation | https://huggingface.co/docs/trl/openenv |
| vLLM | Serve Qwen3-Coder-Next for inference | https://github.com/vllm-project/vllm |
| KernelBench | Standard eval benchmark (250 tasks) | https://github.com/ScalingIntelligence/KernelBench |
| cudaLLM-8B | Reference for SFT data format | https://huggingface.co/ByteDance-Seed/cudaLLM-8B |
| Qwen3.5-9B | Backup model (smaller, still good) | https://huggingface.co/Qwen/Qwen3.5-9B |

---

## 5. Task Breakdown

### PHASE 0: PREP (Wed March 4 - Fri March 6)

#### Task 0.1: Download Everything
**Priority:** CRITICAL — do this FIRST, downloads take hours
**Owner:** Anyone with internet
**Time:** 2-4 hours (parallel downloads)

```bash
# Create project directory
mkdir kernelforge && cd kernelforge

# Clone repos
git clone https://github.com/meta-pytorch/OpenEnv
git clone https://github.com/BytedTsinghua-SIA/CUDA-Agent
git clone https://github.com/double-ai/doubleGraph
git clone https://github.com/skydiscover-ai/skydiscover
git clone https://github.com/ScalingIntelligence/KernelBench

# Download models (these are BIG — start early)
pip install huggingface_hub
huggingface-cli download unsloth/Qwen3-Coder-Next-FP8-Dynamic --local-dir ./models/coder-next-fp8
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ./models/qwen35-9b
huggingface-cli download ByteDance-Seed/cudaLLM-8B --local-dir ./models/cudallm-8b

# Download dataset
python -c "from datasets import load_dataset; ds = load_dataset('BytedTsinghua-SIA/CUDA-Agent-Ops-6K'); ds.save_to_disk('./data/ops_6k')"

# Download doubleGraph A100 wheels
# Check https://github.com/double-ai/doubleGraph/releases/ for latest URL
wget -P ./wheels/ [DOUBLEGRAPH_A100_WHEEL_URL]
```

**Done when:** All repos cloned, model weights on disk, dataset cached locally.

---

#### Task 0.2: Set Up B200 Environment
**Priority:** CRITICAL
**Owner:** Person with GPU access
**Time:** 1-2 hours

```bash
# Verify GPU
nvidia-smi
# Should show: B200, 192 GB

# Verify CUDA
nvcc --version
# Need: CUDA 12.x or 13.x

# Install Python deps
pip install openenv-core torch triton unsloth trl datasets transformers vllm
pip install --break-system-packages huggingface_hub accelerate peft

# Verify Triton
python -c "import triton; print(triton.__version__)"

# Verify torch.compile works
python -c "
import torch
model = torch.nn.Linear(256, 256).cuda()
compiled = torch.compile(model)
x = torch.randn(32, 256).cuda()
out = compiled(x)
print(f'torch.compile works: output shape {out.shape}')
"

# Verify nvcc can target sm_80 (A100)
echo '__global__ void test() {}' > /tmp/test.cu
nvcc -arch=sm_80 -o /tmp/test.out /tmp/test.cu
echo "nvcc sm_80 compilation: OK"

# Install doubleGraph A100 wheel
pip install ./wheels/[DOUBLEGRAPH_WHEEL_NAME].whl
python -c "import cugraph; print('doubleGraph loaded')"
```

**Done when:** torch, triton, nvcc, doubleGraph all working on B200.

---

#### Task 0.3: Inspect CUDA Agent's Evaluation Scripts
**Priority:** HIGH — this is what we're wrapping
**Owner:** Lead engineer
**Time:** 2-3 hours

```bash
# Look at the structure
ls -la CUDA-Agent/agent_workdir/
ls -la CUDA-Agent/agent_workdir/utils/

# READ THESE FILES CAREFULLY:
cat CUDA-Agent/agent_workdir/SKILL.md
cat CUDA-Agent/agent_workdir/utils/compile.sh
cat CUDA-Agent/agent_workdir/utils/verification.py
cat CUDA-Agent/agent_workdir/utils/profiling.py
```

**What you're looking for:**
- How does `compile.sh` invoke nvcc? What flags? What output path?
- How does `verification.py` load the kernel and compare outputs? What's the tolerance? How does it handle the interface between PyTorch tensors and CUDA?
- How does `profiling.py` measure timing? CUDA events? How many runs?
- What's the expected file structure? Where does model.py go? Where does the kernel go?

**This is the most important prep task.** Understanding these scripts determines whether the OpenEnv wrapper works. If their scripts assume a specific directory layout, your wrapper must match it. If their profiling script outputs timing in a specific format, your wrapper must parse it.

**Done when:** You understand how to manually run: load a task → compile a kernel → check correctness → measure speedup using their scripts.

---

#### Task 0.4: Extract doubleGraph A100 Patterns
**Priority:** MEDIUM
**Owner:** Anyone
**Time:** 1-2 hours

```bash
# Explore doubleGraph's A100 kernel source
find doubleGraph/cpp/src/aai/impl/a100/ -name "*.cu" | head -20

# Count kernels per category
for dir in doubleGraph/cpp/src/aai/impl/a100/*/; do
    count=$(find "$dir" -name "*.cu" | wc -l)
    echo "$(basename $dir): $count kernels"
done

# Extract the most interesting patterns (WCC, cosine similarity)
# These are the ones the WarpSpeed blog highlighted
cat doubleGraph/cpp/src/aai/impl/a100/components/weakly_connected_components_seg_mask.cu
cat doubleGraph/cpp/src/aai/impl/a100/link_prediction/cosine_all_pairs_f64_seg.cu
```

**Create a file `a100_patterns.md` containing:**
- 5-10 code snippets showing A100-specific optimizations
- For each: what the optimization is, why it's fast on A100
- These will be pasted into GRPO prompts as few-shot examples

**Example entry:**
```markdown
## L2 Cache Residency Pinning (from doubleGraph WCC kernel)
A100 has 40MB L2 cache. Pinning hot data in L2 eliminates global memory round trips.
```cuda
cudaAccessPolicyWindow window;
window.base_ptr = parent_array;
window.num_bytes = num_vertices * sizeof(int);
window.hitRatio = 1.0f;
window.hitProp = cudaAccessPropertyPersisting;
cudaCtxSetAccessPolicyWindow(&window);
```
Why it's fast: parent array is accessed randomly during path compression.
Without L2 pinning: every access goes to HBM2e (2.0 TB/s).
With L2 pinning: hot data stays in L2 (~5-10x faster access).
Result: 17x speedup on WCC algorithm.
```

**Done when:** `a100_patterns.md` exists with 5-10 extractable patterns.

---

#### Task 0.5: Generate SFT Warmup Data
**Priority:** HIGH — GRPO will fail without this
**Owner:** Person with API key
**Time:** 4-8 hours (run overnight Thursday)
**Cost:** ~$5-15 in API calls

```python
"""
generate_sft_data.py

Uses GLM-5 (or Claude/GPT-5) API to generate working CUDA kernels
for tasks from the 6K dataset. These become SFT training data
so Qwen3-Coder-Next can generate valid CUDA before GRPO starts.

Run this Thursday night. Let it run overnight.
"""
from openai import OpenAI
from datasets import load_from_disk
import json
import subprocess
import tempfile
import os

# GLM-5 API (cheapest frontier model)
# Sign up at z.ai to get API key
client = OpenAI(
    base_url="https://api.z.ai/v1",  # OpenAI-compatible
    api_key=os.environ["ZAI_API_KEY"]
)

# Load 6K dataset
dataset = load_from_disk("./data/ops_6k")["train"]

# Load A100 patterns for few-shot examples
with open("a100_patterns.md") as f:
    a100_patterns = f.read()

sft_pairs = []
errors = []

for i, row in enumerate(dataset):
    if i >= 500:  # Start with 500 tasks
        break
    
    prompt = f"""Write a CUDA kernel (.cu file) that replaces this PyTorch operation.
The kernel must:
1. Include #include <torch/extension.h>
2. Have a __global__ kernel function
3. Have a host wrapper function
4. Be compilable with nvcc -arch=sm_80

Target: NVIDIA A100 (108 SMs, 2.0 TB/s bandwidth, 40MB L2 cache)

Here are expert A100 optimization patterns to use:
{a100_patterns[:2000]}

PyTorch code to replace:
```python
{row['code']}
```

Write ONLY the .cu file. No explanation."""

    try:
        response = client.chat.completions.create(
            model="glm-5",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.7,
        )
        
        cuda_code = response.choices[0].message.content
        
        # Quick compile check (doesn't verify correctness — just compilation)
        with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
            f.write(cuda_code)
            cu_path = f.name
        
        result = subprocess.run(
            ["nvcc", "-arch=sm_80", "-O3", "-c", cu_path, "-o", "/dev/null"],
            capture_output=True, text=True, timeout=30
        )
        os.unlink(cu_path)
        
        compiles = result.returncode == 0
        
        sft_pairs.append({
            "task_ops": row["ops"],
            "task_code": row["code"],
            "generated_cuda": cuda_code,
            "compiles": compiles,
            "prompt": prompt,
        })
        
        status = "✓ compiles" if compiles else "✗ compile error"
        print(f"[{i+1}/500] {row['ops'][:50]}: {status}")
        
    except Exception as e:
        errors.append({"task": i, "error": str(e)})
        print(f"[{i+1}/500] ERROR: {e}")

# Save results
with open("sft_data.json", "w") as f:
    json.dump(sft_pairs, f, indent=2)

# Summary
total = len(sft_pairs)
compilable = len([p for p in sft_pairs if p["compiles"]])
print(f"\nResults: {compilable}/{total} compile successfully ({100*compilable/total:.0f}%)")
print(f"Errors: {len(errors)}")
print(f"Saved to sft_data.json")
```

**Done when:** `sft_data.json` exists with 100+ compilable CUDA kernel examples.

---

#### Task 0.6: Test SkyDiscover on One Kernel
**Priority:** MEDIUM — validates our safety net
**Owner:** Anyone
**Time:** 1-2 hours
**Cost:** ~$1-2 in API calls

```bash
cd skydiscover
uv sync --extra gpu_mode --extra external

# Create a simple evaluator that wraps our environment
cat > test_evaluator.py << 'EOF'
import subprocess, tempfile, os, re

def evaluate(program_path: str) -> dict:
    """SkyDiscover calls this with path to a .cu file."""
    try:
        # Compile
        binary = program_path.replace(".cu", ".out")
        result = subprocess.run(
            ["nvcc", "-arch=sm_80", "-O3", "-o", binary, program_path],
            capture_output=True, text=True, timeout=45
        )
        if result.returncode != 0:
            return {"combined_score": -1e9, 
                    "artifacts": {"feedback": f"Compile error: {result.stderr[:500]}"}}
        
        # Run and time (simple benchmark)
        result = subprocess.run(
            [binary], capture_output=True, text=True, timeout=60
        )
        
        # Parse timing from output
        match = re.search(r"time:\s*([\d.]+)", result.stdout)
        if match:
            time_ms = float(match.group(1))
            score = 1.0 / (time_ms + 1e-9)
        else:
            score = 0.0
        
        return {"combined_score": score,
                "artifacts": {"feedback": result.stdout[:500]}}
    except Exception as e:
        return {"combined_score": -1e9,
                "artifacts": {"feedback": str(e)}}
EOF

# Create a simple starting kernel
cat > initial_kernel.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int N = 1 << 20;
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N*sizeof(float));
    cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_c, N*sizeof(float));
    
    // Warmup
    for(int i=0; i<10; i++)
        vector_add<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i=0; i<100; i++)
        vector_add<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);
    printf("time: %.4f\n", ms/100.0f);
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
EOF

# Set API key
export OPENAI_API_KEY=your-zai-key
export OPENAI_API_BASE=https://api.z.ai/v1

# Run SkyDiscover for 20 iterations (quick test)
uv run skydiscover-run initial_kernel.cu test_evaluator.py \
    --search evox --model glm-5 --iterations 20 --output test_run
```

**Done when:** SkyDiscover runs without crashing and the score improves over 20 iterations. If it doesn't improve, debug the evaluator.

---

### PHASE 1: BUILD THE ENVIRONMENT (Friday March 6)

#### Task 1.1: Copy CUDA Agent's agent_workdir
**Priority:** CRITICAL
**Time:** 30 minutes

```bash
# Copy the evaluation infrastructure
cp -r CUDA-Agent/agent_workdir/ ./kernelforge_workdir/

# Verify the key files exist
ls kernelforge_workdir/utils/compile.sh
ls kernelforge_workdir/utils/verification.py
ls kernelforge_workdir/utils/profiling.py
cat kernelforge_workdir/SKILL.md
```

**Modification needed:** You may need to adjust `compile.sh` to always use `-arch=sm_80`. Check what flags it currently uses.

---

#### Task 1.2: Write the Task Loader
**Priority:** CRITICAL
**Time:** 1 hour

```python
"""
kernelforge/task_loader.py

Loads tasks from both CUDA-Agent-Ops-6K and doubleGraph.
"""
from datasets import load_from_disk
from pathlib import Path
import random

class TaskBank:
    def __init__(self, ops_6k_path="./data/ops_6k", dgraph_path="./doubleGraph"):
        self.tasks = []
        self._load_dense_tasks(ops_6k_path)
        self._load_graph_tasks(dgraph_path)
        print(f"TaskBank: {len(self.tasks)} total tasks")
    
    def _load_dense_tasks(self, path):
        """Load from CUDA-Agent-Ops-6K."""
        try:
            ds = load_from_disk(path)["train"]
            for row in ds:
                self.tasks.append({
                    "type": "dense",
                    "name": str(row["ops"])[:80],
                    "ops": row["ops"],
                    "code": row["code"],
                    "source": "cuda_agent_6k",
                    # Dense tasks: baseline = torch.compile, no known ceiling
                    "has_expert_ceiling": False,
                })
            print(f"  Loaded {len(ds)} dense tasks from Ops-6K")
        except Exception as e:
            print(f"  WARNING: Could not load Ops-6K: {e}")
    
    def _load_graph_tasks(self, dgraph_path):
        """Load graph algorithm tasks from doubleGraph source."""
        a100_dir = Path(dgraph_path) / "cpp" / "src" / "aai" / "impl" / "a100"
        if not a100_dir.exists():
            print(f"  WARNING: doubleGraph A100 dir not found at {a100_dir}")
            return
        
        count = 0
        for category_dir in a100_dir.iterdir():
            if not category_dir.is_dir():
                continue
            for cu_file in category_dir.glob("*.cu"):
                expert_code = cu_file.read_text()
                self.tasks.append({
                    "type": "graph",
                    "name": f"graph/{category_dir.name}/{cu_file.stem}",
                    "ops": f"[{category_dir.name}.{cu_file.stem}]",
                    "code": expert_code[:4000],  # Truncate for prompt context
                    "expert_cuda": expert_code,   # Full expert kernel
                    "source": "doublegraph_a100",
                    "category": category_dir.name,
                    "algorithm": cu_file.stem,
                    # Graph tasks: baseline = cuGraph, ceiling = doubleGraph
                    "has_expert_ceiling": True,
                })
                count += 1
        
        print(f"  Loaded {count} graph tasks from doubleGraph A100")
    
    def sample(self, task_type=None):
        """Sample a random task. Optionally filter by type."""
        if task_type:
            filtered = [t for t in self.tasks if t["type"] == task_type]
            return random.choice(filtered)
        return random.choice(self.tasks)
    
    def sample_dense(self):
        return self.sample("dense")
    
    def sample_graph(self):
        return self.sample("graph")
```

**Done when:** `TaskBank()` loads 6000+ dense tasks and 100+ graph tasks.

---

#### Task 1.3: Write the OpenEnv Wrapper
**Priority:** CRITICAL
**Time:** 3-4 hours

This is the core of the project. See the environment code from the previous response — it wraps CUDA Agent's compile/verify/profile scripts in OpenEnv's step()/reset() interface.

**Key subtasks:**
- [ ] `models.py` — KernelAction, KernelObservation, StepOutcome
- [ ] `client.py` — KernelForgeEnv(EnvClient)
- [ ] `server/environment.py` — KernelForgeEnvironment(Environment)
- [ ] `server/app.py` — FastAPI application
- [ ] Test: `reset()` returns a task with baseline timing
- [ ] Test: `step()` with a valid kernel returns speedup
- [ ] Test: `step()` with invalid CUDA returns compile error
- [ ] Test: `step()` with wrong-output kernel returns correctness failure

**Done when:** All 4 tests pass.

---

#### Task 1.4: Wire SkyDiscover Evaluator
**Priority:** HIGH
**Time:** 1 hour

```python
"""
kernelforge/integrations/skydiscover_evaluator.py

Wraps the OpenEnv environment as a SkyDiscover evaluator.
SkyDiscover calls evaluate(file_path) → we call env.step().
"""
from kernelforge.client import KernelForgeEnv
from kernelforge.models import KernelAction

def evaluate(program_path: str) -> dict:
    with open(program_path) as f:
        cuda_code = f.read()
    
    with KernelForgeEnv(base_url="http://localhost:8000").sync() as env:
        env.reset()  # Load a task
        result = env.step(KernelAction(cuda_code=cuda_code))
        obs = result.observation
        
        return {
            "combined_score": obs.reward,
            "artifacts": {
                "feedback": obs.feedback,
                "speedup": obs.speedup,
                "outcome": obs.outcome.value,
            }
        }
```

**Done when:** SkyDiscover can call this evaluator and get scores back.

---

#### Task 1.5: Set Up GRPO Training Script
**Priority:** HIGH
**Time:** 2-3 hours

See the GRPO code from previous responses. Key steps:
- [ ] Load Qwen3-Coder-Next at FP8 with Unsloth
- [ ] Apply LoRA adapters
- [ ] SFT warmup on generated data (from Task 0.5)
- [ ] Write reward function that calls OpenEnv step()
- [ ] Configure GRPOConfig (num_generations=4, max_new_tokens=2048)
- [ ] Test: can you do 1 GRPO step without OOM or crash?

**Done when:** One GRPO step completes and the model's loss changes.

---

### PHASE 2: RUN EXPERIMENTS (Saturday March 7 — Hackathon Day)

#### Task 2.1: Launch SkyDiscover Evolution (Hour 1-2)
**Priority:** HIGH — gives us demo numbers fast
**Time:** Setup: 30 min. Runs for 2-4 hours in background.

```bash
# Start environment server
cd kernelforge && python -m server.app &

# Run SkyDiscover on 5-10 dense op tasks
for task_id in 0 10 50 100 200; do
    uv run skydiscover-run \
        task_${task_id}_initial.cu \
        integrations/skydiscover_evaluator.py \
        --search evox \
        --model glm-5 \
        --iterations 80 \
        --output results/skydiscover/task_${task_id} &
done
```

**Done when:** At least 3 out of 5 tasks show improved scores.

---

#### Task 2.2: Launch GRPO Training (Hour 2-3)
**Priority:** HIGH — runs overnight
**Time:** Setup: 30 min. Runs for 12-20+ hours.

```bash
# Start GRPO training
python kernelforge/integrations/grpo_training.py \
    --model unsloth/Qwen3-Coder-Next-FP8-Dynamic \
    --dataset ./data/ops_6k \
    --sft-data ./sft_data.json \
    --output ./results/grpo/ \
    --num-generations 4 \
    --max-new-tokens 2048 \
    --learning-rate 5e-6 \
    --logging-steps 1 \
    --save-steps 50
```

Monitor: check `results/grpo/` for training logs every few hours.

**Done when:** Training runs without crashing. Reward curve visible.

---

#### Task 2.3: Run doubleGraph Comparison (Hour 4-6)
**Priority:** MEDIUM — enriches the demo
**Time:** 2-3 hours

```bash
# For 3-5 graph algorithms, run:
# 1. cuGraph original (baseline)
# 2. doubleGraph A100 (expert ceiling)
# 3. SkyDiscover-evolved kernel (our agent)
#
# Record all three times for each algorithm.
# This shows where our agent lands between baseline and expert.
```

**Done when:** Table showing baseline / expert / our-agent for 3+ graph algorithms.

---

#### Task 2.4: Collect Results (Hour 12-20)
**Priority:** CRITICAL
**Time:** 2-3 hours

Gather:
- SkyDiscover speedup numbers (per operator)
- GRPO training curve (reward over steps)
- GRPO model's compilation success rate over training
- doubleGraph comparison table
- Examples of generated kernels (before/after)

---

### PHASE 3: SHIP (Saturday Night / Sunday)

#### Task 3.1: Write Blog Post
**Priority:** CRITICAL — hackathon evaluates based on blog
**Time:** 2-3 hours

Structure:
1. Problem: torch.compile leaves performance on the table for A100
2. Solution: KernelForge — OpenEnv for CUDA kernel optimization
3. Architecture: CUDA Agent eval scripts + doubleGraph baselines + SkyDiscover + GRPO
4. Results: speedup tables, training curve, generated kernel examples
5. How to use: pip install, quick start code
6. Links: all repos, all papers

---

#### Task 3.2: Deploy to HuggingFace Spaces
**Priority:** MEDIUM
**Time:** 1-2 hours

```bash
# From kernelforge directory
openenv push --repo-id your-username/kernel-forge
```

Note: This requires a GPU-enabled Space. May need to demo locally instead.

---

#### Task 3.3: Record Demo Video
**Priority:** HIGH
**Time:** 1 hour

Show:
1. Environment running: reset → step → reward
2. SkyDiscover dashboard evolving a kernel
3. GRPO training curve
4. Before/after kernel comparison
5. Speedup numbers

---

## 6. Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| CUDA Agent's scripts don't work out-of-box on B200 | HIGH | MEDIUM | Test Thursday. Read their code. May need path/flag adjustments. |
| SFT data generation produces <50 compilable kernels | HIGH | MEDIUM | Try different prompt formats. Fall back to Claude/GPT-5 API. |
| GRPO training crashes (OOM, NaN, collapse) | HIGH | HIGH | Parameter tuning fallback (JSON output, dense rewards). |
| SkyDiscover doesn't improve kernels | MEDIUM | LOW | It's iterative LLM prompting with profiling — it almost certainly finds SOMETHING. |
| Generated kernels segfault and crash process | HIGH | HIGH | Subprocess isolation for EVERY evaluation. Non-negotiable. |
| Profiling on B200 gives misleading A100 numbers | MEDIUM | HIGH | Be honest in blog. Rent A100 for 2 hours for final numbers ($2-4). |
| doubleGraph wheel doesn't install | LOW | MEDIUM | Graph tasks become stretch goal. Dense ops are sufficient for the demo. |
| HuggingFace Spaces GPU deployment fails | LOW | MEDIUM | Demo locally. Deploy is nice-to-have. |

---

## 7. Success Criteria

### Minimum Viable Demo (must hit ALL):
- [ ] OpenEnv environment runs: reset() and step() work
- [ ] At least 1 kernel beats torch.compile (via SkyDiscover)
- [ ] GRPO training runs for 100+ steps without crashing
- [ ] Blog post published with actual numbers

### Competitive Demo (hit 3+):
- [ ] 5+ operators where kernels beat torch.compile
- [ ] GRPO training curve shows upward trend
- [ ] Graph task comparison with doubleGraph baselines
- [ ] Deployed on HuggingFace Spaces
- [ ] Multi-model comparison (Coder-Next vs Qwen3.5-9B)

### Winning Demo (hit 2+):
- [ ] 1.5x+ average speedup over torch.compile across 10+ operators
- [ ] GRPO-trained model generates faster kernels than base model >50% of the time
- [ ] Agent approaches doubleGraph performance on at least 1 graph algorithm
- [ ] Clean, reproducible, pip-installable environment

---

## 8. Team Assignments (if team of 4)

| Person | Focus | Tasks |
|--------|-------|-------|
| **Person 1 (Lead)** | Environment core | 0.3, 1.1, 1.2, 1.3 |
| **Person 2** | Training pipeline | 0.5, 1.5, 2.2 |
| **Person 3** | SkyDiscover + doubleGraph | 0.4, 0.6, 1.4, 2.1, 2.3 |
| **Person 4** | Demo + blog + deploy | 3.1, 3.2, 3.3 |

Solo? Priority: 0.1 → 0.2 → 0.3 → 0.5 → 1.1 → 1.2 → 1.3 → 2.1 → 2.2 → 3.1

---

## 9. All Links (Master Reference)

### Repos
- OpenEnv: https://github.com/meta-pytorch/OpenEnv
- CUDA Agent: https://github.com/BytedTsinghua-SIA/CUDA-Agent
- doubleGraph: https://github.com/double-ai/doubleGraph
- SkyDiscover: https://github.com/skydiscover-ai/skydiscover
- KernelBench: https://github.com/ScalingIntelligence/KernelBench
- Unsloth: https://github.com/unslothai/unsloth
- cudaLLM: https://github.com/ByteDance-Seed/cudaLLM
- KernelGYM: https://github.com/hkust-nlp/KernelGYM
- torchforge: https://github.com/meta-pytorch/torchforge

### Models
- Qwen3-Coder-Next: https://huggingface.co/Qwen/Qwen3-Coder-Next
- Qwen3-Coder-Next FP8: https://huggingface.co/unsloth/Qwen3-Coder-Next-FP8-Dynamic
- Qwen3-Coder-Next GGUF: https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF
- Qwen3.5-9B: https://huggingface.co/Qwen/Qwen3.5-9B
- GLM-5: https://huggingface.co/zai-org/GLM-5
- cudaLLM-8B: https://huggingface.co/ByteDance-Seed/cudaLLM-8B

### Datasets
- CUDA-Agent-Ops-6K: https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K

### Papers
- CUDA Agent: https://arxiv.org/abs/2602.24286
- WarpSpeed/doubleGraph: https://doubleai.com/research/doubleais-warpspeed-surpassing-expert-written-kernels-at-scale
- AdaEvolve: https://arxiv.org/abs/2602.20133
- EvoX: https://arxiv.org/abs/2602.23413
- Dr. Kernel: https://arxiv.org/abs/2602.05885
- Kevin: https://arxiv.org/abs/2507.11948
- CudaForge: https://arxiv.org/abs/2511.01884
- KernelBench: https://arxiv.org/abs/2502.10517

### Docs
- OpenEnv: https://meta-pytorch.org/OpenEnv/
- Unsloth RL: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
- Unsloth Qwen3-Coder-Next: https://unsloth.ai/docs/models/qwen3-coder-next
- TRL OpenEnv: https://huggingface.co/docs/trl/openenv
- SkyRL + OpenEnv: https://skyrl.readthedocs.io/en/latest/examples/openenv.html
- GLM-5 API: https://api.z.ai

### Hackathon
- Event page: https://cerebralvalley.ai/e/openenv-hackathon-sf
- Announcement: https://x.com/cerebral_valley/status/2026391140312965168
- SkyDiscover thread: https://x.com/shulynnliu/status/2028892335875276919
- SkyDiscover project: https://skydiscover-ai.github.io/
