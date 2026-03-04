# KernelForge-OpenEnv PRD v4.0

## Replicate, Validate, and Improve ByteDance's RL-for-CUDA Pipeline Using OpenEnv

**Date:** March 3, 2026
**Hackathon:** OpenEnv Hackathon SF — March 7-8, 2026 (SHACK15, San Francisco)
**Organizers:** Cerebral Valley + PyTorch/Meta + SHACK15
**Prize Pool:** $100K+ cash | Teams up to 4

---

## 1. What We Are Actually Building

### 1.1 The Goal

Replicate ByteDance's CUDA Agent RL training methodology (arXiv:2602.24286) at hackathon scale using OpenEnv, validate that it works, and improve it with a more efficient, modular architecture that the community can build on.

This is NOT about writing the fastest possible CUDA kernel. This IS about proving that RL post-training can teach a small language model to write better CUDA kernels than it could before training, using a reproducible open-source pipeline.

### 1.2 What ByteDance Proved

ByteDance demonstrated (February 27, 2026) that a 230B MoE model trained with agentic PPO can write CUDA kernels achieving 2.11x speedup over torch.compile, with 98.8% compilation pass rate and 96.8% faster-than-compile rate. Their critical ablation: removing the multi-turn agentic RL loop drops performance to 77.1% pass and 14.1% faster — a 36-point collapse. The RL pipeline is load-bearing, not decorative.

But their setup is unreproducible outside ByteDance: 230B proprietary model, 128 H20 GPUs, custom PPO with critic pretraining, 6,000 synthetic operators, months of engineering.

### 1.3 What We Prove

That the same methodology works at 30x smaller scale:

| Dimension | ByteDance | KernelForge (ours) |
|-----------|-----------|-------------------|
| Model | Seed 1.6 (230B, 23B active) | Qwen2.5-Coder-7B (7B dense) |
| RL algorithm | PPO with critic pretraining | GRPO (no critic needed) |
| Training infra | 128 H20 GPUs | 1x A100 or A10G |
| Dataset | CUDA-Agent-Ops-6K (6,000 ops) | 50-100 WCC kernel examples |
| Context | 131K tokens | 8K tokens |
| Agent turns | Up to 150 | Up to 15 |
| Target GPU | Multiple architectures | A100 (sm_80) primary, A10G (sm_86) fallback |
| Environment | Custom sandbox | OpenEnv (openenv-core) — open standard |
| Algorithm scope | General CUDA operators | WCC (focused, deep) |

If a 7B model on 1 GPU can show measurable improvement in kernel quality through RL — even going from reward=1 (correct but slow) to reward=2 (faster than baseline) — that validates the entire paradigm at accessible scale.

### 1.4 Why This Wins the Hackathon

1. **It uses OpenEnv** — the hackathon's core framework. We build a real, reusable Environment.
2. **It demonstrates RL post-training** — the hackathon's thesis. Not just inference-time search, actual policy improvement.
3. **It's reproducible** — anyone with an A100 can run it. No proprietary models, no 128-GPU clusters.
4. **It targets CUDA kernel optimization** — the most technically impressive application of RL for code.
5. **It integrates every major sponsor** — Unsloth (training), HuggingFace (TRL + Hub), CoreWeave (compute), Cursor (dev), Meta (OpenEnv).

---

## 2. Target Hardware: A100 Primary, A10G Fallback

### 2.1 Why Not H100

H100 (sm_90a) has exclusive features — TMA, DSMEM, DPX, thread block clusters — that are fascinating but irrelevant for validating the RL methodology. They add kernel complexity without adding signal about whether RL training works. We keep H100 as a documented stretch target.

More practically: hackathon compute credits are likely A100s (CoreWeave) or A10Gs (common cloud). H100s are expensive and may not be available.

### 2.2 What We Run On

**Primary: A100 (sm_80, 40GB or 80GB HBM2e)**

| Property | Value | Impact |
|----------|-------|--------|
| VRAM | 40 or 80 GB | Qwen2.5-Coder-7B QLoRA: ~10-12GB. Leaves 28-68GB for CUDA compilation + kernel execution |
| L2 Cache | 40 MB | Parent array for ~10M vertices fits in L2 with 75% set-aside |
| HBM Bandwidth | 2.0 TB/s | Adequate for graph adjacency streaming |
| Shared Memory/SM | 164 KB | Standard frontier buffer sizing |
| SMs | 108 | Grid launch target |
| nvcc arch flag | `-arch=sm_80` | |

**Fallback: A10G (sm_86, 24GB GDDR6X)**

| Property | Value | Impact |
|----------|-------|--------|
| VRAM | 24 GB | Qwen2.5-Coder-7B QLoRA: ~10-12GB. Leaves ~12GB for kernel eval |
| L2 Cache | 6 MB | Much smaller — L2 pinning less impactful (only ~1.5M vertices) |
| HBM Bandwidth | 600 GB/s | 3x slower than A100 — bandwidth-bound kernels show bigger speedups |
| nvcc arch flag | `-arch=sm_86` | |

### 2.3 Critical Difference: Everything Runs on One Machine

This is the key architectural insight that makes the hackathon feasible. ByteDance used separate GPU clusters for training vs evaluation. We run both on the same GPU:

```
┌─────────────────────────────────────────────┐
│              Single A100 GPU                 │
│                                              │
│  ┌──────────────────────┐                    │
│  │  Qwen2.5-Coder-7B   │   Model loaded     │
│  │  (QLoRA, ~10GB)      │   in GPU memory    │
│  └──────────┬───────────┘                    │
│             │                                │
│             │  Generate CUDA code            │
│             ▼                                │
│  ┌──────────────────────┐                    │
│  │  nvcc compilation    │   Shell out to     │
│  │  (CPU-side, ~3-5s)   │   subprocess       │
│  └──────────┬───────────┘                    │
│             │                                │
│             │  Load compiled .so             │
│             ▼                                │
│  ┌──────────────────────┐                    │
│  │  Kernel execution    │   ctypes FFI,      │
│  │  + PAC verification  │   runs on same GPU │
│  │  + benchmarking      │   (~2-5s total)    │
│  └──────────┬───────────┘                    │
│             │                                │
│             │  Return reward                 │
│             ▼                                │
│  ┌──────────────────────┐                    │
│  │  GRPO gradient step  │   TRL updates      │
│  │  (same GPU memory)   │   model weights    │
│  └──────────────────────┘                    │
│                                              │
│  Total per evaluation: ~5-10 seconds         │
│  No network latency. No cold starts.         │
└─────────────────────────────────────────────┘
```

**Why this works:** nvcc compilation is CPU-bound (runs on the host processor, not the GPU). Kernel execution uses a small fraction of GPU memory for the test graph. The model stays loaded in GPU memory throughout. There's no contention because the operations are sequential: generate → compile (CPU) → execute (GPU, tiny workload) → gradient update (GPU, model).

**Total loop time per step:** With `num_generations=4`:
- 4x model inference: ~10-20s total (7B model, 4096 max tokens each)
- 4x compilation: ~12-20s total (nvcc, CPU-bound)
- 4x verification: ~4-8s total (5 small graphs each)
- 4x benchmarking: ~8-12s total (100 warmup + 50 timed runs each)
- 1x GRPO gradient update: ~2-5s
- **Total per training step: ~40-65 seconds**
- **100 steps: ~70-110 minutes**

That's feasible in a hackathon afternoon. Not comfortable, but feasible.

---

## 3. Model Selection: Qwen2.5-Coder-7B-Instruct (Primary)

### 3.1 Why Not Qwen3-Coder-Next

Qwen3-Coder-Next (80B/3B active) is the newest and flashiest, but:
- Nobody has fine-tuned it with Unsloth yet (released days ago)
- 80B download is massive — hours over hackathon WiFi
- MoE + hybrid attention + Unsloth FastModel = three untested integrations stacked
- If it breaks, there's no fallback plan and no community debugging resources

### 3.2 Why Qwen2.5-Coder-7B-Instruct

| Property | Value |
|----------|-------|
| HuggingFace ID | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Unsloth ID | `unsloth/Qwen2.5-Coder-7B-Instruct` |
| Architecture | 7B dense (not MoE) |
| Training data | 5.5 trillion tokens of code across 92+ programming languages |
| Context | 128K native |
| License | Apache 2.0 |
| Unsloth support | Verified, battle-tested, uses standard FastLanguageModel |
| VRAM (QLoRA 4-bit) | ~10-12 GB |

**Why it's right for this project:**
- **Proven Unsloth integration.** FastLanguageModel (not FastModel). No MoE edge cases. QLoRA works out of the box.
- **Fits on A10G.** 10-12GB leaves room for kernel evaluation on the same GPU.
- **Strong code baseline.** 5.5T code tokens means the model already knows C++ and CUDA syntax. RL just needs to improve optimization strategy, not teach the language.
- **ByteDance precedent.** Their open-source cudaLLM project fine-tuned Qwen3-8B — same architecture family, similar scale.
- **Fast inference.** 7B dense generates completions in seconds, not minutes.

### 3.3 Stretch Model: Qwen3-Coder-30B-A3B-Instruct

If A100 80GB is available and day 1 goes smoothly:
- 30B/3B active MoE — better reasoning, same inference speed
- Requires Unsloth FastModel (not FastLanguageModel)
- QLoRA fits in ~17.5GB — needs A100 40GB minimum
- Only attempt if the 7B pipeline is fully working first

### 3.4 Model Loading Code

```python
# PRIMARY: Qwen2.5-Coder-7B (dense, proven)
from unsloth import FastLanguageModel  # Standard loader for dense models

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-7B-Instruct",
    max_seq_length=8192,
    load_in_4bit=True,
    dtype=None,  # Auto-detect
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # Lower rank for hackathon speed
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
```

```python
# STRETCH: Qwen3-Coder-30B-A3B (MoE, if 7B works first)
from unsloth import FastModel  # MoE requires FastModel, NOT FastLanguageModel

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/Qwen3-Coder-30B-A3B-Instruct",
    max_seq_length=8192,
    load_in_4bit=True,
)
```

---

## 4. The OpenEnv Environment: What Actually Runs

### 4.1 Architecture: Local Evaluation, Not Modal

The fundamental design change from v3.0: **evaluation runs on the same machine as training.** No Modal. No network calls. No cold starts.

```python
"""
KernelForge OpenEnv Environment.

Runs nvcc compilation, PAC verification, and benchmarking
locally on the SAME GPU used for training. No external dispatch.

Install: pip install "openenv-core[core]>=0.2.1"
"""
import subprocess
import tempfile
import os
import ctypes
import numpy as np
import networkx as nx
from openenv.core.env_server import Environment, create_fastapi_app
from openenv.core.types import StepResult


class KernelForgeEnv(Environment):
    """
    RL environment for CUDA kernel optimization.

    Action: CUDA source code string (must define a specific C function signature)
    Observation: SKILL.md + compile/verify/benchmark feedback + history
    Reward: discrete milestones {-1, +1, +2, +3}

    The kernel interface contract is fixed and non-negotiable:
    The agent's CUDA code MUST define this extern C function:

        extern "C" void wcc_kernel(
            const int* row_ptr,    // CSR row pointers, length num_vertices+1
            const int* col_idx,    // CSR column indices, length num_edges
            int* labels,           // Output labels, length num_vertices
            int num_vertices,
            int num_edges
        );

    This is communicated to the agent via SKILL.md.
    """

    # --- Configuration (set per target GPU) ---
    ARCH_FLAG = "-arch=sm_80"     # A100. Change to sm_86 for A10G.
    NVCC_FLAGS = ["-O3", "-use_fast_math", "--shared", "-Xcompiler", "-fPIC"]
    MAX_TURNS = 15                # Hackathon scope (ByteDance: 150)
    VERIFY_GRAPHS = 5
    WARMUP_ITERS = 50             # Reduced for speed (ByteDance: would use more)
    BENCHMARK_RUNS = 30           # Reduced for speed
    MAX_COMPILE_SECONDS = 30
    TEST_GRAPH_VERTICES = 10000   # Small for fast iteration
    ENTRY_POINT = "wcc_kernel"    # The C function name the agent must export

    def __init__(self):
        self.history = []
        self.turn = 0
        self.best_reward = -1.0
        self.best_code = None
        self.best_runtime_ms = float("inf")
        self.baseline_ms = None
        self.test_graphs = None

    def reset(self):
        self.history = []
        self.turn = 0
        self.best_reward = -1.0
        self.best_code = None
        self.best_runtime_ms = float("inf")

        # Generate test graphs once per episode
        if self.test_graphs is None:
            self.test_graphs = self._generate_test_graphs()

        # Profile baseline (cuGraph or naive CUDA) once
        if self.baseline_ms is None:
            self.baseline_ms = self._profile_baseline()

        return {
            "observation": self._load_skill_md(),
            "baseline_ms": self.baseline_ms,
            "target_arch": self.ARCH_FLAG,
            "entry_point": self.ENTRY_POINT,
            "max_turns": self.MAX_TURNS,
        }

    def step(self, action: str) -> StepResult:
        self.turn += 1

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "kernel.cu")
            lib = os.path.join(tmpdir, "kernel.so")

            with open(src, "w") as f:
                f.write(action)

            # ── Step 1: Compile ──
            compile_cmd = (
                ["nvcc", self.ARCH_FLAG] + self.NVCC_FLAGS + [src, "-o", lib]
            )
            try:
                proc = subprocess.run(
                    compile_cmd,
                    capture_output=True, text=True,
                    timeout=self.MAX_COMPILE_SECONDS,
                )
                if proc.returncode != 0:
                    return self._fail(
                        -1.0,
                        f"COMPILE ERROR:\n{proc.stderr[:1500]}"
                    )
            except subprocess.TimeoutExpired:
                return self._fail(-1.0, "COMPILE TIMEOUT (30s)")

            # ── Step 2: Verify entry point exists ──
            try:
                so = ctypes.CDLL(lib)
                kernel_fn = getattr(so, self.ENTRY_POINT, None)
                if kernel_fn is None:
                    return self._fail(
                        -1.0,
                        f"MISSING ENTRY POINT: extern \"C\" void {self.ENTRY_POINT}(...) not found in compiled .so"
                    )
            except OSError as e:
                return self._fail(-1.0, f"LOAD ERROR: {e}")

            # ── Step 3: PAC Verification (5 adversarial graphs) ──
            for graph_type, row_ptr, col_idx, n_verts, n_edges, ref_labels in self.test_graphs:
                try:
                    kernel_labels = self._run_kernel(
                        so, row_ptr, col_idx, n_verts, n_edges
                    )
                except Exception as e:
                    return self._fail(
                        -1.0,
                        f"RUNTIME ERROR on {graph_type}: {str(e)[:500]}"
                    )

                passed, msg = self._verify(kernel_labels, ref_labels, row_ptr, col_idx, n_verts)
                if not passed:
                    return self._fail(
                        -1.0,
                        f"VERIFICATION FAILED on {graph_type}: {msg}"
                    )

            # ── Step 4: Benchmark ──
            import cupy as cp

            # Use first test graph for benchmarking
            _, row_ptr, col_idx, n_verts, n_edges, _ = self.test_graphs[0]

            # Warmup
            for _ in range(self.WARMUP_ITERS):
                self._run_kernel(so, row_ptr, col_idx, n_verts, n_edges)
            cp.cuda.Device(0).synchronize()

            # Timed runs
            times = []
            for _ in range(self.BENCHMARK_RUNS):
                start_event = cp.cuda.Event()
                end_event = cp.cuda.Event()
                start_event.record()
                self._run_kernel(so, row_ptr, col_idx, n_verts, n_edges)
                end_event.record()
                end_event.synchronize()
                times.append(cp.cuda.get_elapsed_time(start_event, end_event))

            times_arr = np.array(times)
            runtime_ms = float(np.median(times_arr))
            speedup = self.baseline_ms / runtime_ms if runtime_ms > 0 else 0

            # ── Reward (ByteDance milestone-based) ──
            if speedup > 1.50:
                reward = 3.0    # Major optimization
            elif speedup > 1.05:
                reward = 2.0    # Meaningful speedup (>5%)
            else:
                reward = 1.0    # Correct but not faster

            obs = (
                f"PASS (turn {self.turn}/{self.MAX_TURNS})\n"
                f"  Runtime: {runtime_ms:.3f}ms (baseline: {self.baseline_ms:.3f}ms)\n"
                f"  Speedup: {speedup:.2f}x\n"
                f"  Reward: {reward}\n"
                f"  Stats: mean={np.mean(times_arr):.3f} std={np.std(times_arr):.3f} "
                f"min={np.min(times_arr):.3f} max={np.max(times_arr):.3f}"
            )

            if reward > self.best_reward or runtime_ms < self.best_runtime_ms:
                self.best_reward = reward
                self.best_code = action
                self.best_runtime_ms = runtime_ms

        done = (self.turn >= self.MAX_TURNS) or (reward >= 3.0)

        self.history.append({
            "turn": self.turn,
            "reward": reward,
            "speedup": round(speedup, 2),
            "runtime_ms": round(runtime_ms, 3),
        })

        # Include history for agentic context
        if len(self.history) > 1:
            obs = self._format_history() + "\n---\n" + obs

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "turn": self.turn,
                "speedup": speedup,
                "runtime_ms": runtime_ms,
                "best_reward": self.best_reward,
            },
        )

    @property
    def state(self):
        return {
            "turn": self.turn,
            "history": self.history,
            "best_reward": self.best_reward,
            "best_runtime_ms": self.best_runtime_ms,
        }

    # ── Internal methods ──

    def _run_kernel(self, so, row_ptr, col_idx, n_verts, n_edges):
        """Execute the compiled kernel via ctypes FFI."""
        import cupy as cp

        # Allocate device arrays
        d_row = cp.array(row_ptr, dtype=cp.int32)
        d_col = cp.array(col_idx, dtype=cp.int32)
        d_labels = cp.zeros(n_verts, dtype=cp.int32)

        # Get raw device pointers
        fn = getattr(so, self.ENTRY_POINT)
        fn.argtypes = [
            ctypes.c_void_p,  # row_ptr
            ctypes.c_void_p,  # col_idx
            ctypes.c_void_p,  # labels (output)
            ctypes.c_int,     # num_vertices
            ctypes.c_int,     # num_edges
        ]

        fn(
            ctypes.c_void_p(d_row.data.ptr),
            ctypes.c_void_p(d_col.data.ptr),
            ctypes.c_void_p(d_labels.data.ptr),
            ctypes.c_int(n_verts),
            ctypes.c_int(n_edges),
        )

        cp.cuda.Device(0).synchronize()
        return d_labels.get()  # Copy back to host

    def _verify(self, kernel_labels, ref_labels, row_ptr, col_idx, n_verts):
        """
        PAC Verification: 3 mathematical invariants.
        Does NOT require exact label match — labels are arbitrary IDs.
        """
        # Invariant 1: same number of unique components
        kernel_components = len(set(int(kernel_labels[v]) for v in range(n_verts)))
        ref_components = len(set(int(ref_labels[v]) for v in range(n_verts)))
        if kernel_components != ref_components:
            return False, f"Component count: kernel={kernel_components} vs ref={ref_components}"

        # Invariant 2: all edges connect same-label vertices
        for v in range(n_verts):
            for e_idx in range(row_ptr[v], row_ptr[v + 1]):
                u = col_idx[e_idx]
                if kernel_labels[v] != kernel_labels[u]:
                    return False, f"Edge ({v},{u}): labels {kernel_labels[v]} vs {kernel_labels[u]}"

        # Invariant 3: vertices in different ref components have different kernel labels
        ref_comp_map = {}
        for v in range(n_verts):
            rl = int(ref_labels[v])
            kl = int(kernel_labels[v])
            if kl in ref_comp_map:
                if ref_comp_map[kl] != rl:
                    return False, f"Kernel label {kl} maps to ref components {ref_comp_map[kl]} and {rl}"
            else:
                ref_comp_map[kl] = rl

        return True, f"Verified: {ref_components} components"

    def _generate_test_graphs(self):
        """
        Input Generator: 5 adversarial graphs in CSR format.
        Returns list of (type, row_ptr, col_idx, n_verts, n_edges, ref_labels).
        """
        graphs = []
        N = self.TEST_GRAPH_VERTICES

        def nx_to_csr_and_labels(G, n):
            """Convert networkx graph to CSR + compute reference labels."""
            # Build CSR
            adj = [[] for _ in range(n)]
            for u, v in G.edges():
                adj[u].append(v)
                adj[v].append(u)
            row_ptr = [0]
            col_idx = []
            for v in range(n):
                neighbors = sorted(set(adj[v]))
                col_idx.extend(neighbors)
                row_ptr.append(len(col_idx))

            # Reference labels via networkx
            components = list(nx.connected_components(G))
            labels = np.zeros(n, dtype=np.int32)
            for comp_id, comp in enumerate(components):
                for v in comp:
                    labels[v] = comp_id

            return (
                np.array(row_ptr, dtype=np.int32),
                np.array(col_idx, dtype=np.int32),
                n,
                len(col_idx),
                labels,
            )

        # 1-2: RMAT power-law
        for seed in [42, 137]:
            G = nx.Graph()
            G.add_nodes_from(range(N))
            rng = np.random.default_rng(seed)
            for _ in range(N * 10):
                u, v = int(rng.zipf(2.0) % N), int(rng.zipf(2.0) % N)
                if u != v:
                    G.add_edge(u, v)
            rp, ci, nv, ne, rl = nx_to_csr_and_labels(G, N)
            graphs.append(("rmat", rp, ci, nv, ne, rl))

        # 3-4: SBM planted communities
        for n_comm in [5, 50]:
            sizes = [N // n_comm] * n_comm
            p = [[0.1 if i == j else 0.001 for j in range(n_comm)]
                 for i in range(n_comm)]
            G = nx.stochastic_block_model(sizes, p, seed=n_comm)
            rp, ci, nv, ne, rl = nx_to_csr_and_labels(G, sum(sizes))
            graphs.append(("sbm", rp, ci, nv, ne, rl))

        # 5: Sparse Erdos-Renyi
        G = nx.erdos_renyi_graph(N, 0.0005, seed=99)
        rp, ci, nv, ne, rl = nx_to_csr_and_labels(G, N)
        graphs.append(("er_sparse", rp, ci, nv, ne, rl))

        return graphs

    def _profile_baseline(self):
        """
        Profile a naive CUDA WCC baseline.
        This is a simple atomic-based implementation compiled at init.
        """
        # For hackathon: use a known-correct naive kernel as baseline
        # In production: profile cuGraph
        baseline_code = NAIVE_WCC_KERNEL  # Defined below
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "baseline.cu")
            lib_path = os.path.join(tmpdir, "baseline.so")
            with open(src, "w") as f:
                f.write(baseline_code)
            subprocess.run(
                ["nvcc", self.ARCH_FLAG, "-O3", "--shared",
                 "-Xcompiler", "-fPIC", src, "-o", lib_path],
                check=True, capture_output=True,
            )
            so = ctypes.CDLL(lib_path)
            import cupy as cp

            _, row_ptr, col_idx, n_verts, n_edges, _ = self.test_graphs[0]

            # Warmup
            for _ in range(20):
                self._run_kernel(so, row_ptr, col_idx, n_verts, n_edges)
            cp.cuda.Device(0).synchronize()

            # Time
            times = []
            for _ in range(20):
                start = cp.cuda.Event()
                end = cp.cuda.Event()
                start.record()
                self._run_kernel(so, row_ptr, col_idx, n_verts, n_edges)
                end.record()
                end.synchronize()
                times.append(cp.cuda.get_elapsed_time(start, end))

            return float(np.median(times))

    def _format_history(self):
        lines = ["Previous attempts:"]
        for h in self.history:
            lines.append(
                f"  Turn {h['turn']}: reward={h['reward']}, "
                f"speedup={h['speedup']}x, runtime={h['runtime_ms']}ms"
            )
        return "\n".join(lines)

    def _load_skill_md(self):
        skill_path = os.path.join(os.path.dirname(__file__), "..", "skill.md")
        if os.path.exists(skill_path):
            with open(skill_path) as f:
                return f.read()
        return "No SKILL.md found — write optimized WCC kernels."


# ── Naive WCC Baseline Kernel (atomic-based, unoptimized) ──
# This is what the agent must beat. It uses atomicMin for correctness
# but is intentionally unoptimized (multiple kernel launches, no L2 pinning,
# no path compression optimization).

NAIVE_WCC_KERNEL = r'''
#include <cuda_runtime.h>
#include <stdio.h>

__device__ int find_root_atomic(int* parent, int x) {
    while (parent[x] != x) {
        x = parent[x];
    }
    return x;
}

__global__ void init_labels(int* parent, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) parent[tid] = tid;
}

__global__ void hook_kernel(
    const int* row_ptr, const int* col_idx, int* parent,
    int N, int* changed
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < N) {
        int root_v = find_root_atomic(parent, v);
        for (int e = row_ptr[v]; e < row_ptr[v + 1]; e++) {
            int u = col_idx[e];
            int root_u = find_root_atomic(parent, u);
            if (root_v != root_u) {
                int hi = max(root_v, root_u);
                int lo = min(root_v, root_u);
                atomicMin(&parent[hi], lo);
                *changed = 1;
            }
        }
    }
}

__global__ void compress_kernel(int* parent, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        parent[tid] = find_root_atomic(parent, tid);
    }
}

extern "C" void wcc_kernel(
    const int* row_ptr, const int* col_idx, int* labels,
    int num_vertices, int num_edges
) {
    int threads = 256;
    int blocks = (num_vertices + threads - 1) / threads;

    int* d_changed;
    cudaMalloc(&d_changed, sizeof(int));

    // Initialize: each vertex is its own parent
    init_labels<<<blocks, threads>>>(labels, num_vertices);
    cudaDeviceSynchronize();

    // Iterative hooking + compression until convergence
    for (int iter = 0; iter < 100; iter++) {
        int h_changed = 0;
        cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);

        hook_kernel<<<blocks, threads>>>(
            row_ptr, col_idx, labels, num_vertices, d_changed
        );
        cudaDeviceSynchronize();

        compress_kernel<<<blocks, threads>>>(labels, num_vertices);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        if (!h_changed) break;
    }

    cudaFree(d_changed);
}
''';


# OpenEnv HTTP server
app = create_fastapi_app(KernelForgeEnv)
```

### 4.2 What's Different From v3.0

| v3.0 (broken) | v4.0 (works) |
|----------------|-------------|
| `_run_kernel_ffi` = `raise NotImplementedError` | Complete ctypes FFI with CuPy device pointers |
| Modal dispatch per evaluation (~25s latency) | Local subprocess nvcc + same-GPU execution (~5-10s) |
| No kernel interface contract | Fixed: `extern "C" void wcc_kernel(row_ptr, col_idx, labels, N, E)` |
| No baseline kernel | Complete naive atomic WCC kernel included |
| CSR conversion hand-waved | Full networkx-to-CSR conversion with reference labels |
| Verification checked labels dict | Verification checks numpy arrays from CuPy `.get()` |
| 200 max turns (impossible in 24h) | 15 max turns (realistic for hackathon) |
| cuGraph dependency (complex install) | Self-contained naive baseline (no cuGraph needed) |

---

## 5. The SKILL.md Protocol (Adapted for A100)

```markdown
# KernelForge SKILL.md v4.0 — A100 WCC Optimization

## Target Hardware
- GPU: NVIDIA A100 (Ampere architecture)
- Compute capability: sm_80
- nvcc flags: -arch=sm_80 -O3 -use_fast_math
- L2 Cache: 40 MB (pin up to 30 MB for parent array)
- Shared memory/SM: 164 KB
- SMs: 108
- HBM2e bandwidth: 2.0 TB/s

## Your Task
Write an optimized CUDA kernel for Weakly Connected Components (WCC)
using the Union-Find data structure. Your kernel must be faster than
the provided naive baseline which uses atomicMin.

## Kernel Interface Contract (MANDATORY)
Your code MUST define this exact function signature:

    extern "C" void wcc_kernel(
        const int* row_ptr,    // CSR row pointers, length num_vertices+1
        const int* col_idx,    // CSR column indices, length num_edges
        int* labels,           // Output: component label per vertex
        int num_vertices,
        int num_edges
    );

Input graph is in CSR (Compressed Sparse Row) format.
Output: labels[v] = component ID for vertex v.
Vertices in the same component must have the same label.
Vertices in different components must have different labels.
The specific label values don't matter, only grouping.

## Optimization Techniques (ranked by impact)

### Priority 1: Non-atomic path compression
The baseline uses atomicMin for thread safety. But Union-Find path
compression is mathematically safe without atomics because:
- Parent pointers only move toward root (monotone convergence)
- Stale reads produce valid ancestors (self-correcting)
- No race can bridge separate components

Drop atomicMin. Use plain stores: parent[hi] = lo;
Use path halving: parent[x] = parent[parent[x]];

### Priority 2: L2 cache pinning
Pin the parent array to L2 persistent cache:
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, ...);
  cudaStreamAttrValue with hitProp = cudaAccessPropertyPersisting

A100 L2 is 40 MB. For graphs up to 10M vertices (40 MB / 4 bytes),
the parent array fits entirely in L2. This drops random access latency
from ~200 cycles (HBM) to ~30-40 cycles (L2).

### Priority 3: Single kernel launch
The baseline launches 3 kernels per iteration (init, hook, compress)
with host-side convergence checking. Eliminate this:
- Use cooperative groups for grid-level sync
- Run the entire algorithm in one kernel
- No host round-trips, no kernel boundary cache thrashing

### Priority 4: Additional optimizations
- __ldg() for read-only edge data
- Vectorized loads (int4) for adjacency list
- __shfl_sync for intra-warp label exchange
- Launch bounds for occupancy tuning

## Rules
- Do NOT modify verification scripts
- Do NOT use cuBLAS, cuSPARSE, or other pre-built libraries
- Do NOT hardcode outputs for specific test inputs
- Your kernel MUST work for any valid CSR graph
```

---

## 6. Training Pipeline: What Actually Runs in 24 Hours

### 6.1 The Three Stages (Adapted from ByteDance's Four)

```
ByteDance (230B model, 128 GPUs, weeks)     Us (7B model, 1 GPU, 24 hours)
──────────────────────────────────────────   ──────────────────────────────
Stage 1: Single-turn PPO warm-up             Stage 1: SFT warm-up (2-3 hours)
Stage 2: Rejection fine-tuning               Stage 2: RFT filter (1 hour)
Stage 3: Critic value pretraining            SKIPPED (GRPO has no critic)
Stage 4: Multi-turn agentic PPO (150 steps)  Stage 3: GRPO training (2-3 hours)
```

### 6.2 Stage 1: SFT Warm-Up

**Goal:** Teach the model what good WCC CUDA kernels look like before asking it to optimize.

**Data generation (do BEFORE the hackathon):**

Generate 50-100 WCC kernel variants at different optimization levels:
- 10-15 naive kernels (atomicMin, multi-launch, no path compression)
- 15-20 intermediate kernels (path compression, still atomic)
- 10-15 optimized kernels (non-atomic path compression, L2 pinning)
- 10-15 broken kernels paired with fix explanations (teaches debugging)

Format each as a prompt-completion pair:

```jsonl
{"prompt": "Write an optimized CUDA kernel for WCC on A100. Use non-atomic path compression with L2 cache pinning. <SKILL.md content>", "completion": "```cuda\n#include <cuda_runtime.h>\n... kernel code ...\n```"}
```

**Generation method:** Use Claude API or GPT-4 to generate kernel variants. For each, compile with nvcc on Modal/CoreWeave, run PAC verification, and keep only correct ones. Tag with optimization level.

**Training:**

```python
from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(
    output_dir="outputs/sft-warmup",
    max_seq_length=4096,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=200,           # Small dataset, fast convergence
    learning_rate=2e-5,
    bf16=True,
    optim="paged_adamw_8bit",
    logging_steps=10,
)

sft_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=sft_dataset,
    args=sft_config,
)
sft_trainer.train()
```

**Time estimate:** ~200 steps at ~3s/step = ~10 minutes. The data generation before the hackathon takes longer.

### 6.3 Stage 2: Rejection Fine-Tuning

**Goal:** Run the SFT model in the OpenEnv environment, collect trajectories, filter for quality.

```python
# Collect trajectories
trajectories = []
env = KernelForgeEnv()

for prompt in evaluation_prompts:
    obs = env.reset()
    for turn in range(env.MAX_TURNS):
        # Generate kernel from model
        completion = generate(model, tokenizer, prompt + obs["observation"])
        result = env.step(completion)

        trajectories.append({
            "prompt": prompt,
            "completion": completion,
            "reward": result.reward,
            "turn": turn,
        })

        if result.done:
            break
        obs = {"observation": result.observation}

# Filter: keep only reward >= 1.0 (correct) trajectories
# ByteDance kept only reward >= 2, but we have less data
good_trajectories = [t for t in trajectories if t["reward"] >= 1.0]

# Re-run SFT on filtered data
rft_dataset = Dataset.from_list(good_trajectories)
```

**Time estimate:** 15 turns × 5 prompts × ~10s/turn = ~12 minutes for collection. SFT on filtered data: ~5 minutes. Total: ~20 minutes.

### 6.4 Stage 3: GRPO Training

**Goal:** RL fine-tuning with milestone-based rewards. This is where the model should actually improve its kernel optimization strategy.

```python
from trl import GRPOConfig, GRPOTrainer  # FROM TRL, not Unsloth

training_args = GRPOConfig(
    learning_rate=5e-6,
    num_generations=4,             # 4 kernels per prompt per step
    max_prompt_length=512,
    max_completion_length=4096,    # Must fit a full CUDA kernel
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=50,                  # Conservative — extend if time allows
    optim="paged_adamw_8bit",
    bf16=True,
    output_dir="outputs/grpo",
    logging_steps=1,
    report_to="none",
)

# The reward function runs LOCAL evaluation (no Modal)
def local_cuda_reward(completions, **kwargs) -> list[float]:
    """
    Evaluate each completion by compiling and running on the local GPU.
    This is the core RL signal.
    """
    env = KernelForgeEnv()
    if env.test_graphs is None:
        env.test_graphs = env._generate_test_graphs()
    if env.baseline_ms is None:
        env.baseline_ms = env._profile_baseline()

    rewards = []
    for completion in completions:
        code = extract_cuda_code(completion)
        if not code:
            rewards.append(-1.0)
            continue

        # Use the environment's step logic directly
        env.turn = 0  # Reset turn counter
        result = env.step(code)
        rewards.append(result.reward)

    return rewards

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[local_cuda_reward],
    args=training_args,
    train_dataset=prompt_dataset,
)

trainer.train()

# Save
model.save_pretrained("outputs/kernelforge-qwen25-coder-7b")
tokenizer.save_pretrained("outputs/kernelforge-qwen25-coder-7b")
```

**Time estimate per step:** 4 generations × ~10s inference + 4 × ~10s evaluation + ~5s gradient update = ~85s/step.
**50 steps:** ~70 minutes.
**100 steps (if time allows):** ~140 minutes.

### 6.5 Reward Function: Why Discrete Milestones

ByteDance's critical ablation finding: continuous speedup rewards are destructive. GPU execution times have inherent noise (thermal throttling, OS interrupts, interconnect congestion). These micro-fluctuations create outliers that bias advantage estimation, causing the model to hallucinate correlations between code changes and speed. Switching from continuous to discrete milestones improved success rate by 36 percentage points.

Our tiers:

| Reward | Condition | What it means |
|--------|-----------|--------------|
| -1.0 | Won't compile, or wrong output | Broken kernel |
| +1.0 | Correct output, <= 5% faster than baseline | Works but not optimized |
| +2.0 | Correct + > 5% faster than naive atomic baseline | Real optimization (likely non-atomic path compression) |
| +3.0 | Correct + > 50% faster than baseline | Major optimization (L2 pinning + single launch + non-atomic) |

The +3.0 threshold is set at 50% (not ByteDance's 2x) because our baseline is a naive CUDA kernel, not torch.compile. Beating a hand-written CUDA baseline by 50% on a graph algorithm is a genuine achievement.

---

## 7. What We Validate

### 7.1 The Core Hypothesis

**H1: RL post-training improves CUDA kernel quality measurably, even at small scale.**

Metric: Average reward across held-out evaluation prompts increases from SFT baseline to post-GRPO.

If the SFT model averages reward ~1.0 (correct but slow) and the GRPO model averages reward ~1.5-2.0 (sometimes faster), the hypothesis is validated. We don't need to match ByteDance's 2.11x. We need to show the gradient points in the right direction.

### 7.2 The Ablations We Run

**A1: SFT-only vs SFT+GRPO.** Does RL training add value over just showing the model good examples?

**A2: With SKILL.md vs without SKILL.md.** Does the structured protocol improve kernel quality? ByteDance found this critical.

**A3: Milestone rewards vs random rewards.** Sanity check that the reward signal is load-bearing (not just random noise driving improvement).

### 7.3 What "Success" Looks Like at the Hackathon

**Minimum viable demo (must have):**
- OpenEnv environment that compiles, verifies, and benchmarks CUDA kernels
- Naive baseline kernel with measured performance
- SFT model that generates compilable, correct WCC kernels
- Evidence that GRPO training changes the reward distribution (even slightly)

**Good demo:**
- GRPO-trained model consistently generates kernels with reward >= 2.0
- Clear reward curve showing improvement over training steps
- Ablation showing SFT-only < SFT+GRPO

**Great demo:**
- Model discovers non-atomic path compression through RL (not just from SFT data)
- Model discovers L2 cache pinning through RL exploration
- Kernels approach or exceed cuGraph performance
- Framework generalized: show it working on a second algorithm (e.g., PageRank)

---

## 8. H100 Stretch Goals (If A100 Pipeline Works)

If the core A100 pipeline is validated by Day 2 morning, extend to H100:

### 8.1 What Changes for H100

| Setting | A100 value | H100 value |
|---------|-----------|-----------|
| `ARCH_FLAG` | `-arch=sm_80` | `-arch=sm_90a` (mandatory `a` suffix) |
| L2 cache set-aside | 30 MB (75% of 40 MB) | 37.5 MB (75% of 50 MB) |
| Max pinnable vertices | ~7.5M | ~9.4M |
| Shared memory/SM | 164 KB | 228 KB |

### 8.2 H100-Exclusive Optimizations (add to SKILL.md)

These are the techniques from the v3.0 PRD that only apply to H100:

**Thread Block Clusters + DSMEM:** Group 4-8 blocks, share 912-1824 KB of distributed shared memory for cross-partition label propagation without global memory.

**TMA:** Single thread initiates bulk adjacency list copy; other 127 threads freed for computation. 1.45 TB/s throughput (59% over A100 patterns).

**DPX Instructions:** `__vimin3_s32(a, b, c)` fuses 3-way min into single cycle for label propagation.

**Guard with:** `#if __CUDA_ARCH__ >= 900`

### 8.3 DoubleAI WarpSpeed Reference Techniques

These were documented in v3.0 and remain valid reference material for the H100 stretch:

- Non-atomic Union-Find path compression (also works on A100 — this IS the core technique)
- L2 cache pinning for parent array (also works on A100)
- Single cooperative kernel launch (also works on A100)
- Per-vertex adaptive routing for power-law graphs
- Per-architecture exhaustive specialization (192 kernels per GPU)
- PAC-reasoning verification (we implement this)

The DoubleAI-specific H100 adaptation strategy: regenerate all 192 kernels targeting sm_90a, exploiting thread block clusters for cross-block label propagation, TMA for adjacency loading, DPX for min-label propagation, and the larger 50 MB L2.

---

## 9. Tech Stack

### 9.1 Python Dependencies

```
# Core training
unsloth>=2025.3
trl>=0.16.0                # GRPOTrainer is HERE, not in Unsloth
transformers>=4.48,<5.0    # Unsloth incompatible with transformers 5.0+
torch>=2.10.0
datasets>=3.0

# OpenEnv
openenv-core[core]>=0.2.1  # NOT "openenv" (wrong package)

# Evaluation (runs locally on same GPU)
cupy-cuda12x>=14.0
networkx>=3.0
numpy>=1.26
scipy>=1.12

# Infrastructure
wandb>=0.19                # Optional experiment tracking
```

### 9.2 System Requirements

| Requirement | A100 setup | A10G setup |
|-------------|-----------|-----------|
| GPU VRAM | 40+ GB | 24 GB |
| CUDA toolkit | 12.1+ | 12.1+ |
| nvcc | Required (in Docker image) | Required |
| Python | 3.10-3.12 | 3.10-3.12 |
| Model VRAM | ~10-12 GB (7B QLoRA) | ~10-12 GB (7B QLoRA) |
| Kernel eval VRAM | ~1-2 GB (small test graphs) | ~1-2 GB |
| Available for gradients | ~28-68 GB | ~10-12 GB (tight) |

### 9.3 Repository Structure

```
KernelForge-OpenEnv/
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── skill.md                              # SKILL.md v4.0 (Section 5)
│
├── openenv_env/
│   ├── __init__.py
│   └── kernel_forge_env.py               # OpenEnv Environment (Section 4.1)
│
├── training/
│   ├── sft_warmup.py                     # Stage 1
│   ├── rft_filter.py                     # Stage 2
│   └── grpo_train.py                     # Stage 3
│
├── datasets/
│   ├── generate_training_data.py         # Generate kernel variants via API
│   └── wcc_training.jsonl                # SFT data
│
├── evaluation/
│   ├── ablation.py                       # Run A1/A2/A3 ablations
│   └── eval_model.py                     # Evaluate trained model
│
├── kernels/
│   ├── naive_baseline.cu                 # Atomic WCC (the baseline to beat)
│   ├── nonatomic_wcc.cu                  # Reference: non-atomic + L2 pinning
│   └── cooperative_wcc.cu                # Reference: single cooperative launch
│
└── demo/
    └── streamlit_demo.py                 # Live training visualization
```

---

## 10. Hackathon Execution Plan

### 10.1 Before the Hackathon (Now Through March 6)

**Day -4 to -3: Data generation (CRITICAL PATH)**

1. Generate 50-100 WCC kernel variants using Claude API / GPT-4
2. For each: compile with nvcc on a cloud GPU, verify correctness, tag optimization level
3. Format as wcc_training.jsonl
4. Test that the naive baseline kernel compiles and runs correctly

**Day -2 to -1: Infrastructure validation**

5. Set up cloud GPU instance (CoreWeave credits or Modal A100)
6. Install all dependencies, verify Unsloth + Qwen2.5-Coder-7B loads
7. Test the complete OpenEnv environment: reset() → step() → reward
8. Run a 5-step SFT training to verify the pipeline end-to-end
9. Run a 3-step GRPO training to verify reward dispatch works
10. Time everything. Know exactly how long each stage takes.

### 10.2 Day 1: March 7 (BUILD)

**Hour 0-1: Setup (get running immediately)**
- Clone repo, install deps on hackathon compute
- Verify GPU access and CUDA toolkit
- Load model, verify SKILL.md

**Hour 1-3: Stage 1 — SFT Warm-Up**
- Load pre-generated wcc_training.jsonl
- Run SFT: 200 steps (~10 min)
- Evaluate: generate 10 kernels, check how many compile and pass verification

**Hour 3-5: Stage 2 — Rejection Fine-Tuning**
- Run SFT model in OpenEnv loop: 5 prompts × 15 turns
- Collect trajectories, filter for reward >= 1.0
- Re-train SFT on filtered data (~5 min)

**Hour 5-8: Stage 3 — GRPO Training**
- Start GRPO: target 50 steps
- Monitor reward distribution per step (wandb or print)
- If reward stuck at -1.0 (can't compile): problem with prompt format, debug
- If reward stuck at +1.0 (correct but slow): model needs more optimization examples in SFT data
- If reward reaches +2.0: it's working. Extend to 100 steps.

### 10.3 Day 2: March 8 (SHIP)

**Hour 0-2: Finalize Training**
- Continue GRPO if improving
- Run final evaluation: 20 prompts, compare SFT-only vs GRPO
- Run ablations A1, A2, A3

**Hour 2-4: Build Demo**
- Streamlit app showing:
  - Reward curve over GRPO training steps
  - Side-by-side: SFT kernel vs GRPO kernel (actual code diff)
  - Speedup numbers: naive baseline → SFT → GRPO
  - Live: type a prompt, model generates kernel, environment evaluates it
- Push model + environment to HuggingFace Hub

**Hour 4-5: Pitch**
- 3-minute presentation:
  - "ByteDance proved RL makes better CUDA kernels. We reproduced it at 30x smaller scale using OpenEnv."
  - Show the reward curve going up
  - Show an actual kernel the model wrote that beats the baseline
  - "This environment is open source. Plug in any model, any GPU, any algorithm."

---

## 11. Sponsor Integration

| Sponsor | How We Use Them | Evidence in Demo |
|---------|----------------|-----------------|
| **Meta/PyTorch** | OpenEnv framework | Our Environment subclass, HTTP server |
| **Unsloth** | FastLanguageModel QLoRA | 2x faster training, 70% less VRAM |
| **HuggingFace** | TRL GRPOTrainer + model Hub | Training loop + published model |
| **CoreWeave** | A100 GPU compute | Training infrastructure |
| **Cursor** | Development environment | All code written in Cursor |
| **Mercor** | Agent evaluation framing | Multi-turn optimization as agent task |

---

## 12. Honest Risk Assessment

| Risk | Probability | Mitigation |
|------|------------|------------|
| Unsloth + Qwen2.5-Coder-7B incompatibility | Low (well-tested combo) | Pre-verify before hackathon |
| Model generates uncompilable code | Medium | SFT warm-up teaches syntax; reward=-1 penalizes |
| GRPO shows no improvement over SFT | Medium | This is still a valid result ("RL doesn't help at this scale") — present as ablation |
| CUDA compilation too slow for RL loop | Low | nvcc on CPU is ~3-5s; acceptable |
| Not enough time for GRPO | Medium | SFT + environment demo is still valuable |
| Test graphs too small to show speedup differences | Medium | Increase to 50K-100K vertices if eval is fast enough |
| OpenEnv API changes at hackathon | Low | Pin openenv-core version |

---

## 13. Corrections from All Prior Versions

| Prior PRD Error | v4.0 Fix |
|----------------|----------|
| Target: H100 only | A100 primary, A10G fallback, H100 stretch |
| Model: Qwen3-Coder-Next (80B, untested) | Qwen2.5-Coder-7B-Instruct (proven Unsloth) |
| Eval: Modal dispatch (~25s latency) | Local same-GPU evaluation (~5-10s) |
| Kernel interface: undefined | Fixed: `extern "C" void wcc_kernel(row_ptr, col_idx, labels, N, E)` |
| `_run_kernel_ffi`: NotImplementedError | Complete ctypes FFI with CuPy device pointers |
| Baseline: cuGraph (complex install) | Self-contained naive CUDA baseline kernel |
| Max turns: 200 (impossible in 24h) | 15 (realistic for hackathon) |
| GRPO steps: 100+ | 50 primary, 100 stretch |
| No ablation plan | Three ablations defined (A1, A2, A3) |
| No success metrics | Clear minimum/good/great thresholds |
| Loader: FastModel (MoE) | FastLanguageModel (dense, for 7B) |
| Package: `openenv` | `openenv-core[core]>=0.2.1` |
| Import: `from unsloth import GRPOTrainer` | `from trl import GRPOConfig, GRPOTrainer` |
| transformers version unspecified | `>=4.48, <5.0` (Unsloth constraint) |
| Claimed DoubleAI integration | DoubleAI techniques documented as reference/stretch |
| H100-specific code in core path | All H100 code behind `#if __CUDA_ARCH__ >= 900` guard, in stretch section |

---

## 14. Works Cited

1. Weinan Dai, Hanlin Wu, et al. "CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation." arXiv:2602.24286, February 2026.
2. DoubleAI. "WarpSpeed: Surpassing Expert-Written Kernels At Scale." doubleai.com, March 2, 2026.
3. Amnon Shashua, Shai Shalev-Shwartz. "Artificial Expert Intelligence through PAC-reasoning." arXiv:2412.02441, December 2024.
4. Srinivas Jaiganesh, Martin Burtscher. "A GPU Implementation of the ECL-CC Algorithm." HPDC 2018.
5. NVIDIA. "Hopper Architecture In-Depth." NVIDIA Technical Blog, 2022.
6. NVIDIA. "A100 Tensor Core GPU Architecture Whitepaper." 2020.
7. NVIDIA. "CUDA C++ Programming Guide." CUDA 12.1 Documentation.
8. ByteDance-Seed/cudaLLM. GitHub. Apache 2.0.
9. double-ai/doubleGraph. GitHub. Apache 2.0.
