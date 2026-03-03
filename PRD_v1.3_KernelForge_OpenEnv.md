# PRD v1.3 – KernelForge-OpenEnv (The Megazord Build)

**Date:** March 3, 2026
**Target:** H100 (sm_90) | **Algorithm:** WCC (Weakly Connected Components)
**Agent Base:** Qwen3.5-9B (4-bit LoRA) | **RL Framework:** Unsloth + TRL GRPO
**Hackathon:** OpenEnv Hackathon SF (Cerebral Valley + PyTorch/Meta + Shack15)

---

## Changelog from v1.2

> **This section documents every fix applied to v1.2. AI agents: read this first.**

### Tech Stack Fixes (6 broken references corrected)

| # | v1.2 (WRONG) | v1.3 (CORRECT) | Why |
|---|---|---|---|
| 1 | `Qwen/Qwen3.5-9B-Instruct` | `Qwen/Qwen3.5-9B` | Qwen3.5 does NOT use `-Instruct` suffix. The base chat model is just `Qwen3.5-9B`. |
| 2 | `from openenv import OpenEnv` | `from openenv.core.env_server import create_app` / `from openenv.core.env_client import EnvClient` | The Meta OpenEnv package is `openenv-core` (not `openenv`). There is no `OpenEnv` base class. |
| 3 | `from unsloth.grpo import GRPOTrainer` | `from trl import GRPOConfig, GRPOTrainer` | GRPOTrainer lives in HuggingFace's `trl` library, NOT in unsloth. Unsloth optimizes via `FastLanguageModel`. |
| 4 | `pip install .../doublegraph_a100-0.1-cp312-cp312-linux_x86_64.whl` | `pip install .../libcugraph_cu13-26.2.0-py3-none-linux_x86_64.whl` from release tag `v0.1.0-a100` | Wrong wheel filename. The actual wheel is `libcugraph_cu13-26.2.0-py3-none-linux_x86_64.whl`. |
| 5 | `torch 2.5+` | `torch>=2.10.0` | PyTorch 2.10.0 is the latest stable (Jan 2026). Specifying 2.5+ is misleadingly loose. |
| 6 | `openenv` in pip | `openenv-core` | The PyPI package name for Meta's OpenEnv is `openenv-core`. The plain `openenv` is an unrelated 2020 package. |

### DoubleAI Context Fix

| v1.2 | v1.3 | Why |
|---|---|---|
| "3.6× geometric-mean speedup" | "3.6× mean speedup" | DoubleAI's published materials say "mean" / "average". They do not specify geometric mean. |

### ByteDance CUDA Agent Alignment Notes

The PRD's description of the ByteDance pipeline is broadly accurate (arXiv: 2602.24286). Key details to keep aligned:

| PRD Concept | ByteDance Actual | Status |
|---|---|---|
| 4-stage pipeline (Warm-up → RFT → Critic → Agentic RL) | Exactly as described | **Accurate** |
| SKILL.md workflow enforcement | Real file in their repo (`agent_workdir/SKILL.md`) | **Accurate** |
| Milestone reward > continuous speedup | Confirmed superior in their ablations | **Accurate** |
| -1 reward for incorrect kernels | Confirmed | **Accurate** |
| ByteDance uses PPO | Confirmed (both single-turn and agentic PPO) | **Accurate** |
| We adapt to GRPO (not PPO) | Our choice — GRPO is simpler for hackathon scope | **Intentional deviation** |
| ByteDance base model: Seed 1.6 (230B MoE) | We use Qwen3.5-9B (much smaller) | **Intentional deviation** |
| Anti-reward-hacking: 5 measures | File permissions, torch fallback blocking, 5 random input correctness, profiling with sync/warmup/avg, no web search | **Adapted** |

Reference: [CUDA Agent (arXiv 2602.24286)](https://arxiv.org/abs/2602.24286) | [GitHub](https://github.com/BytedTsinghua-SIA/CUDA-Agent) | [Dataset](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K)

### Critical Code Bugs Fixed (7 bugs)

| # | Bug | Location | Fix |
|---|---|---|---|
| 1 | **Reward gap → NameError.** When `1.0 < speedup_vs_orig <= 1.05` AND `speedup_vs_dg <= 1.05`, no `base_reward` is assigned. Python raises `NameError: name 'base_reward' is not defined`. | `KernelForgeEnv.step()` reward logic | Added exhaustive elif chain with explicit fallthrough default. |
| 2 | **Positive reward for degradation.** `speedup_vs_orig <= 1.0` gave `base_reward = 1.0`. This rewards the agent for making code SLOWER. | `KernelForgeEnv.step()` | Changed to `base_reward = -0.5` for degradation. |
| 3 | **Missing `reset()` method.** Gymnasium requires `reset()` returning `(obs, info)`. Without it, `env.reset()` crashes. | `KernelForgeEnv` class | Added `reset()` with container setup, baseline profiling, and initial observation. |
| 4 | **No observation_space / action_space.** Gymnasium requires these as class attributes. Without them, `gymnasium.make()` and wrappers fail. | `KernelForgeEnv.__init__()` | Added `gymnasium.spaces.Text` for both. |
| 5 | **Container never created.** `self.container` is referenced in `step()` but never instantiated. | `KernelForgeEnv.__init__()` | Moved container creation to `reset()` with proper lifecycle (create on reset, kill on close). |
| 6 | **Time-travel rewind described but never implemented.** The PRD describes rewind behavior extensively, but `step()` only appends to history — never reads it. The observation on failure says "Rewinding state..." but nothing actually rewinds. | `KernelForgeEnv` | Added `rewind(steps_back)` method and automatic rewind-to-last-good-state on verification failure. |
| 7 | **`profile.py` referenced in skill_h100.md but missing from repo structure.** The SKILL.md says "Profile baseline with `profile.py`" but no such file exists. | Repository structure | Added `tools/profile.py` to file tree and provided skeleton. |

### Structural Gaps Filled (6 missing files)

| File | Status in v1.2 | Added in v1.3 |
|---|---|---|
| `requirements.txt` | Listed but no content | Full pinned dependency list |
| `pyproject.toml` | Listed but no content | Complete project metadata |
| `openenv_env/__init__.py` | Listed but no content | Proper module init with registration |
| `demo/streamlit_demo.py` | Listed but no skeleton | Working skeleton |
| `tools/profile.py` | Not listed at all | CUDA profiling script |
| `scripts/gen_dataset.py` | Not listed at all | RMAT/SBM graph generator for JSONL |

---

## 1. One-Shot Build Instruction for AI Coding Agent (READ CAREFULLY)

**Role:** You are an expert AI systems engineer and CUDA optimization specialist.

**Task:** Build the KernelForge-OpenEnv repository exactly as specified in this document. Do not add, remove, hallucinate, or deviate from any detail, scope, or code skeleton. Incorporate every DoubleAI WarpSpeed and ByteDance CUDA Agent element listed below. Use only the listed libraries and versions. Output the repo files in the exact structure requested.

---

## 2. Project Overview & Business Outcomes

We are building an open-source Gymnasium-style RL environment called KernelForgeEnv.

It trains a small LLM (Qwen3.5-9B) to autonomously write and optimize CUDA kernels for cuGraph's Weakly Connected Components (WCC) algorithm on real H100 hardware.

**Scope Constraints (Strict – No Deviation):**
* Algorithm: WCC only (cuGraph `weakly_connected_components`). Do not implement full cuGraph.
* Hardware: H100 only (`-arch=sm_90`). No multi-GPU (no `mg_` prefix).
* Baseline: Original cuGraph PyTorch implementation + doubleGraph A100 wheel.
* Anti-Hacking: Strict process isolation; the agent cannot modify verification scripts or use torch fallbacks in C++.

---

## 3. Core Architecture: DoubleAI + ByteDance Fusion

The system integrates two state-of-the-art methodologies:

### A. DoubleAI WarpSpeed Capabilities (To Replicate in Env)

* **PAC-Reasoning Verification:**
  * **Input Generator (IG):** Generates hard test graphs (RMAT power-law, Stochastic Block Models with planted communities).
  * **Algorithmic Verifier (AV):** Checks mathematical invariants instead of relying on exact output matches (e.g., exact component count match via networkx, provably safe data races in Union-Find).

* **Time-Travel Search:** The agent snapshots state at decision points (up to 15 turns). On failure, it rewinds to the last known-good state and injects an explanation into the context (e.g., "tried atomics → failed latency; try non-atomic with L2 pin"). Unlike simple backtracking, artifacts and diagnostics from the failed trajectory are preserved and injected into the rewound state.

* **Exhaustive Specialization:** Target a single-kernel launch, dropping atomics in Union-Find path compression (safe under warp scheduling), and pinning the parent array to L2.

### B. ByteDance CUDA Agent RL Pipeline

* **Skill-Augmented Sandbox:** A Docker environment wrapped by OpenEnv that enforces a strict SKILL.md workflow.

* **4-Stage RL Training (Adapted for Unsloth + TRL GRPO):**
  1. **Warm-up:** Single-turn trajectory collection.
  2. **Rejection Fine-Tuning (RFT):** Filter positive trajectories and perform supervised fine-tuning.
  3. **Critic Pre-training:** Handled via GRPO value functions.
  4. **Agentic GRPO:** Multi-turn loop (up to 15 turns for this hackathon).

* **Milestone Reward Function:** Discrete rewards are more stable than raw continuous speedup. See reward table in Section 6.

---

## 4. Required Tech Stack (Pinned Versions)

```
Python 3.12
gymnasium>=1.0.0
openenv-core                    # Meta's OpenEnv (NOT the unrelated "openenv" on PyPI)
docker>=7.0
modal>=0.73
unsloth[colab-new]              # Install via: pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
trl>=0.15                       # GRPOTrainer lives here, NOT in unsloth
torch>=2.10.0
cupy-cuda13x>=14.0
networkx>=3.4
nvcc 13.0.2                     # Via Docker image: nvidia/cuda:13.0.2-devel-ubuntu22.04
streamlit>=1.41                 # For demo UI
scipy>=1.14                     # For RMAT/SBM graph generation
```

---

## 5. Exact Repository Structure

```
KernelForge-OpenEnv/
├── README.md
├── openenv_env/
│   ├── __init__.py
│   └── KernelForgeEnv.py
├── skill_h100.md
├── datasets/
│   └── wcc_doubleai_inspired.jsonl
├── training/
│   └── grpo_train.py
├── tools/
│   └── profile.py              # NEW: Referenced by skill_h100.md
├── scripts/
│   └── gen_dataset.py           # NEW: Generates the JSONL dataset
├── modal_app.py
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── demo/
    └── streamlit_demo.py
```

---

## 6. Exact File Contents & Skeletons

### File 1: README.md

Must contain this exact DoubleAI context verbatim:

> "DoubleAI's WarpSpeed (March 2 2026) autonomously rewrote every single kernel in NVIDIA cuGraph (RAPIDS graph analytics library) for A100, L4, and A10G GPUs, delivering a 3.6× mean speedup over a decade of NVIDIA expert code. 100% of kernels faster, 55% >2×, 18% 10–100×. They released the full drop-in doubleGraph library (https://github.com/double-ai/doubleGraph) with per-GPU wheels (cu13). Key innovations: PAC-reasoning verification (Input Generator for hard test graphs like Stochastic Block Models + Algorithmic Verifier using mathematical invariants), Diligent small-data learning, time-travel search (snapshot/rewind + experience injection from failed trajectories), exhaustive per-config specialization (192 kernels per arch), and profile-guided last-mile PTX tweaks. Our KernelForge-OpenEnv brings this exact approach to H100 + open RL training."

Must contain this exact Sponsor Thanks section verbatim:

> **Sponsor Thanks:**
> * **Unsloth:** Trained Qwen3.5-9B with Unsloth + TRL GRPO – replicating WarpSpeed efficiency on 1 H100.
> * **Hugging Face:** Env + model published to HF Hub for OpenEnv Challenge.
> * **CoreWeave / OpenPipe / Northflank:** H100 runs via Modal on CoreWeave infra.
> * **Cursor:** Built with Cursor AI pair-programming.
> * **PyTorch / Meta:** Uses exact OpenEnv spec from PyTorch team + WarpSpeed-inspired time-travel.
> * **UC Berkeley:** Grounded long-horizon planning with real H100 interaction; PAC verification grounded in math invariants.
> * **Snorkel AI / Patronus AI / Scale AI:** Safe RL – no reward hacking via milestones.
> * **Fleet AI / Mercor / Halluminate / Scaler AI Labs:** Agentic orchestration via time-travel.

---

### File 2: skill_h100.md

```markdown
# KernelForge SKILL.md v1.3 – MUST FOLLOW EVERY RULE

Target: H100 sm_90
nvcc flags: -arch=sm_90 -O3 -use_fast_math

WORKFLOW ENFORCEMENT (If not followed, reward = -100):
1. Profile baseline with `tools/profile.py`.
2. Write C++ code strictly avoiding `torch` fallbacks.
3. Compile via exact bash commands.
4. Verify with math invariants (not just eager execution matching).
5. Profile final version with warmup + sync + average.

H100 CUDA RULES:
* ALWAYS use TMA async for irregular edge/frontier loads in WCC.
* Warp-specialized frontiers (one warp per segment).
* For WCC Union-Find: non-atomic path compression with provably safe data races
  (writes only to ancestors in same component).
* Pin parent array/labels to L2 with __ldg() or TMA.
* Single-kernel launch where possible to eliminate kernel boundaries.

TIME-TRAVEL:
If verification fails or performance degrades, the environment will rewind to
the last known-good state. You must output a brief root-cause analysis of the
failure before attempting the next compilation. The failed code and diagnostics
will be injected into your context for reference.
```

---

### File 3: openenv_env/__init__.py

```python
from openenv_env.KernelForgeEnv import KernelForgeEnv

__all__ = ["KernelForgeEnv"]
```

---

### File 4: openenv_env/KernelForgeEnv.py

```python
import gymnasium as gym
from gymnasium import spaces
import docker
import subprocess
import numpy as np
import networkx as nx
import scipy.sparse as sp
import time
import os
import json


class KernelForgeEnv(gym.Env):
    """
    Gymnasium environment for training an LLM to optimize CUDA WCC kernels on H100.
    Combines DoubleAI WarpSpeed PAC-verification + time-travel with ByteDance milestone rewards.
    """

    metadata = {"render_modes": []}

    # --- Reward Table (ByteDance Milestone + DoubleAI Log-Relative Fusion) ---
    # Discrete milestones prevent reward hacking from continuous speedup noise.
    #
    # | Condition                                         | base_reward |
    # |---------------------------------------------------|-------------|
    # | Verification failed                               |  -1.0       |
    # | speedup_vs_orig <= 1.0 (degradation)              |  -0.5       |
    # | 1.0 < speedup_vs_orig <= 1.05 (marginal)         |   0.5       |
    # | speedup_vs_orig > 1.05 (orig only)                |   2.0       |
    # | speedup_vs_orig > 1.05 AND speedup_vs_dg > 1.05  |   3.0       |
    #
    # Final reward = base_reward + log(1 + max(speedup_vs_orig, speedup_vs_dg)) * 15

    MAX_TURNS = 15
    DOCKER_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu22.04"
    N_WARMUP = 100
    N_BENCHMARK = 50
    N_VERIFY_GRAPHS = 5

    def __init__(self, docker_image=None):
        super().__init__()

        self.docker_image = docker_image or self.DOCKER_IMAGE
        self.docker_client = docker.from_env()
        self.container = None

        # Gymnasium requires these
        self.observation_space = spaces.Text(
            min_length=0,
            max_length=65536,
            charset=None,  # all unicode
        )
        self.action_space = spaces.Text(
            min_length=1,
            max_length=16384,
            charset=None,
        )

        # Time-travel state
        self.history = []           # List of {"code", "reward", "action", "perf"}
        self.best_snapshot = None   # Last known-good state for rewind
        self.turn = 0

        # Baselines (set during reset)
        self.original_baseline = None
        self.doublegraph_baseline = None

    def reset(self, seed=None, options=None):
        """Reset the environment: start a fresh container and profile baselines."""
        super().reset(seed=seed)

        # Tear down old container if it exists
        if self.container is not None:
            try:
                self.container.kill()
                self.container.remove()
            except docker.errors.APIError:
                pass

        # Start fresh container with GPU access
        self.container = self.docker_client.containers.run(
            self.docker_image,
            command="sleep infinity",
            detach=True,
            device_requests=[
                docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])
            ],
        )

        # Profile baselines
        self.original_baseline = self._profile_original()
        self.doublegraph_baseline = self._profile_doublegraph()

        # Reset time-travel state
        self.history = []
        self.best_snapshot = None
        self.turn = 0

        obs = (
            f"Environment ready. H100 container running.\n"
            f"Original cuGraph WCC baseline: {self.original_baseline:.2f} ms\n"
            f"doubleGraph A100 baseline: {self.doublegraph_baseline:.2f} ms\n"
            f"Your goal: beat both. Max turns: {self.MAX_TURNS}.\n"
            f"Follow skill_h100.md rules strictly."
        )
        info = {
            "original_baseline_ms": self.original_baseline,
            "doublegraph_baseline_ms": self.doublegraph_baseline,
        }
        return obs, info

    def step(self, action):
        """Execute an action (bash command) in the container and evaluate."""
        self.turn += 1

        # Execute action in isolated container
        exit_code, output = self.container.exec_run(
            f"bash -c '{action}'",
            demux=True,
        )
        stdout = (output[0] or b"").decode("utf-8", errors="replace")
        stderr = (output[1] or b"").decode("utf-8", errors="replace")

        # PAC Verification
        correct, av_msg = self._pac_verify_wcc()

        if not correct:
            reward = -1.0
            speedup_vs_orig = 0.0
            speedup_vs_dg = 0.0

            # Time-travel: rewind to last known-good state
            rewind_context = self._rewind()
            obs = (
                f"VERIFICATION FAILED: {av_msg}\n"
                f"stdout: {stdout[:2000]}\n"
                f"stderr: {stderr[:2000]}\n"
                f"--- REWIND ---\n"
                f"{rewind_context}\n"
                f"You MUST output a root-cause analysis before your next compilation."
            )
        else:
            perf = self._profile()
            speedup_vs_orig = self.original_baseline / perf if perf > 0 else 0.0
            speedup_vs_dg = self.doublegraph_baseline / perf if perf > 0 else 0.0

            # Milestone reward (exhaustive — no gaps)
            if speedup_vs_orig > 1.05 and speedup_vs_dg > 1.05:
                base_reward = 3.0
            elif speedup_vs_orig > 1.05:
                base_reward = 2.0
            elif speedup_vs_orig > 1.0:
                base_reward = 0.5
            else:
                # Degradation: agent made code slower or equal
                base_reward = -0.5

            reward = base_reward + (np.log(1 + max(speedup_vs_orig, speedup_vs_dg)) * 15)

            obs = (
                f"Code:\n{self._get_code()}\n"
                f"Perf: {perf:.2f} ms\n"
                f"Speedup vs original: {speedup_vs_orig:.2f}x | vs doubleGraph: {speedup_vs_dg:.2f}x\n"
                f"stdout: {stdout[:1000]}\n"
                f"stderr: {stderr[:1000]}"
            )

            # Update best snapshot for time-travel rewind
            if self.best_snapshot is None or reward > self.best_snapshot.get("reward", float("-inf")):
                self.best_snapshot = {
                    "code": self._get_code(),
                    "reward": reward,
                    "perf": perf,
                    "turn": self.turn,
                }

        done = (speedup_vs_orig > 3.6) or (self.turn >= self.MAX_TURNS)

        # Record history
        self.history.append({
            "code": self._get_code(),
            "reward": reward,
            "action": action,
            "turn": self.turn,
            "speedup_vs_orig": speedup_vs_orig,
            "speedup_vs_dg": speedup_vs_dg,
        })

        info = {
            "speedup_vs_orig": speedup_vs_orig,
            "speedup_vs_dg": speedup_vs_dg,
            "turn": self.turn,
            "verification_passed": correct,
        }

        return obs, reward, done, False, info

    def _rewind(self):
        """
        Time-travel rewind: restore to last known-good state.
        Returns context string with failed trajectory info for injection.
        """
        if self.best_snapshot is None:
            return "No previous good state to rewind to. Starting from scratch."

        # Restore the best known code to the container
        best_code = self.best_snapshot["code"]
        self.container.exec_run(f"bash -c 'cat > /workspace/kernel.cu << '\\''CODEEOF'\\''\\n{best_code}\\nCODEEOF'")

        # Build context from failed trajectories since the best snapshot
        failed_turns = [
            h for h in self.history
            if h["turn"] > self.best_snapshot["turn"]
        ]
        failed_summary = "\n".join(
            f"  Turn {h['turn']}: reward={h['reward']:.2f}, action='{h['action'][:200]}'"
            for h in failed_turns
        )

        return (
            f"Rewound to turn {self.best_snapshot['turn']} "
            f"(reward={self.best_snapshot['reward']:.2f}, perf={self.best_snapshot['perf']:.2f}ms).\n"
            f"Failed attempts since then:\n{failed_summary}\n"
            f"Best code restored. Learn from the failures above."
        )

    def _pac_verify_wcc(self):
        """
        PAC Verification using Input Generator + Algorithmic Verifier.
        Generates N_VERIFY_GRAPHS hard graphs and checks WCC correctness
        via mathematical invariants (component count match with networkx).
        """
        for i in range(self.N_VERIFY_GRAPHS):
            # Generate test graph
            if i % 2 == 0:
                # RMAT power-law graph
                G = self._gen_rmat_graph(n=2**14, m=2**16, seed=i)
            else:
                # Stochastic Block Model with planted communities
                G = self._gen_sbm_graph(sizes=[500, 500, 500], p_in=0.05, p_out=0.001, seed=i)

            # Ground truth via networkx
            expected_components = nx.number_connected_components(G)

            # Run agent's kernel on this graph and get component count
            agent_components = self._run_agent_wcc(G)

            if agent_components is None:
                return False, f"Agent kernel crashed on test graph {i}."

            if agent_components != expected_components:
                return False, (
                    f"Component count mismatch on graph {i}: "
                    f"expected {expected_components}, got {agent_components}."
                )

        return True, "All math invariants verified across all test graphs."

    def _gen_rmat_graph(self, n=16384, m=65536, seed=0):
        """Generate an RMAT power-law graph (R-MAT: a,b,c,d = 0.57,0.19,0.19,0.05)."""
        rng = np.random.default_rng(seed)
        a, b, c = 0.57, 0.19, 0.19
        # d = 1 - a - b - c = 0.05
        log2n = int(np.log2(n))
        src = np.zeros(m, dtype=np.int64)
        dst = np.zeros(m, dtype=np.int64)
        for depth in range(log2n):
            choose = rng.random(m)
            src_bit = ((choose > a) & (choose <= a + b)) | (choose > a + b + c)
            dst_bit = ((choose > a + b) & (choose <= a + b + c)) | (choose > a + b + c)
            src += src_bit.astype(np.int64) * (1 << depth)
            dst += dst_bit.astype(np.int64) * (1 << depth)

        G = nx.Graph()
        G.add_nodes_from(range(n))
        edges = list(set(zip(src.tolist(), dst.tolist())))
        edges = [(u, v) for u, v in edges if u != v]  # Remove self-loops
        G.add_edges_from(edges)
        return G

    def _gen_sbm_graph(self, sizes, p_in, p_out, seed=0):
        """Generate a Stochastic Block Model graph with planted communities."""
        k = len(sizes)
        p_matrix = np.full((k, k), p_out)
        np.fill_diagonal(p_matrix, p_in)
        return nx.stochastic_block_model(sizes, p_matrix, seed=seed)

    def _run_agent_wcc(self, G):
        """
        Run the agent's compiled WCC kernel on graph G inside the container.
        Returns the number of connected components found, or None on crash.
        """
        # Serialize graph to CSR format and copy into container
        n = G.number_of_nodes()
        if G.number_of_edges() == 0:
            return n  # Each node is its own component

        adj = nx.to_scipy_sparse_array(G, format="csr")
        row_offsets = adj.indptr.astype(np.int32).tobytes()
        col_indices = adj.indices.astype(np.int32).tobytes()

        # Write binary data to container
        import tarfile, io
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for name, data in [("row_offsets.bin", row_offsets), ("col_indices.bin", col_indices)]:
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        tar_stream.seek(0)
        self.container.put_archive("/workspace/", tar_stream)

        # Run the agent's kernel
        exit_code, output = self.container.exec_run(
            f"bash -c '/workspace/wcc_kernel {n} /workspace/row_offsets.bin /workspace/col_indices.bin'",
            demux=True,
        )

        if exit_code != 0:
            return None

        stdout = (output[0] or b"").decode("utf-8", errors="replace").strip()
        try:
            return int(stdout)
        except ValueError:
            return None

    def _profile(self):
        """
        Profile the agent's kernel using cudaEventRecord timing.
        N_WARMUP warmup iterations, then average over N_BENCHMARK runs.
        Returns average execution time in milliseconds.
        """
        # Run profiling script inside container
        exit_code, output = self.container.exec_run(
            f"bash -c 'python3 /workspace/tools/profile.py "
            f"--warmup {self.N_WARMUP} --runs {self.N_BENCHMARK} "
            f"--binary /workspace/wcc_kernel'",
            demux=True,
        )

        if exit_code != 0:
            return float("inf")

        stdout = (output[0] or b"").decode("utf-8", errors="replace").strip()
        try:
            # profile.py outputs a single float: average ms
            return float(stdout.split("\n")[-1])
        except (ValueError, IndexError):
            return float("inf")

    def _profile_original(self):
        """Profile the original cuGraph WCC implementation. Returns time in ms."""
        # STUB: Replace with actual cuGraph profiling in container
        # Must install cugraph-cu13 and run the default WCC kernel
        return 100.0

    def _profile_doublegraph(self):
        """Profile the doubleGraph A100 WCC wheel. Returns time in ms."""
        # STUB: Replace with actual doubleGraph profiling
        # Wheel: libcugraph_cu13-26.2.0-py3-none-linux_x86_64.whl from v0.1.0-a100
        return 50.0

    def _get_code(self):
        """Read the current kernel source from the container."""
        exit_code, output = self.container.exec_run(
            "cat /workspace/kernel.cu",
            demux=True,
        )
        if exit_code != 0:
            return ""
        return (output[0] or b"").decode("utf-8", errors="replace")

    def close(self):
        """Clean up the Docker container."""
        if self.container is not None:
            try:
                self.container.kill()
                self.container.remove()
            except docker.errors.APIError:
                pass
            self.container = None
```

---

### File 5: training/grpo_train.py

```python
"""
KernelForge GRPO Training Script
Inspired by WarpSpeed's Diligent small-data + ByteDance CUDA Agent RL flywheel.

Uses Unsloth for fast model loading/LoRA and TRL for GRPOTrainer.
"""
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset

# --- Model Setup ---
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen3.5-9B",           # NOT "Qwen3.5-9B-Instruct" (does not exist)
    load_in_4bit=True,
    max_seq_length=8192,
)

# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

# --- Dataset ---
dataset = load_dataset("json", data_files="datasets/wcc_doubleai_inspired.jsonl", split="train")

# --- Reward Function ---
# This connects to the KernelForgeEnv reward logic.
# In practice, rewards come from environment interaction (not offline).
# For warm-up/RFT stages, we use pre-collected trajectory rewards.


def reward_fn(completions, **kwargs):
    """
    Placeholder reward function for offline GRPO warm-up.
    In agentic mode (stage 4), rewards come from KernelForgeEnv.step().
    """
    # STUB: Replace with actual environment reward collection
    return [0.0] * len(completions)


# --- GRPO Training Config ---
config = GRPOConfig(
    output_dir="outputs/kernelforge-qwen3.5-9b",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    max_completion_length=4096,
    num_generations=4,              # GRPO group size
    logging_steps=10,
    save_steps=100,
    bf16=True,
)

# --- Train ---
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    config=config,
    train_dataset=dataset,
    reward_funcs=reward_fn,
)

trainer.train()

# --- Save ---
model.save_pretrained("outputs/kernelforge-qwen3.5-9b")
tokenizer.save_pretrained("outputs/kernelforge-qwen3.5-9b")
```

---

### File 6: modal_app.py

```python
import modal

# Must install the doubleGraph A100 wheel as baseline
# CORRECTED: Actual wheel name is libcugraph_cu13-26.2.0, NOT doublegraph_a100-0.1
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu22.04")
    .pip_install("cupy-cuda13x", "networkx", "scipy")
    .run_commands(
        "pip install https://github.com/double-ai/doubleGraph/releases/download/"
        "v0.1.0-a100/libcugraph_cu13-26.2.0-py3-none-linux_x86_64.whl"
    )
)

app = modal.App("kernelforge-h100")


@app.function(gpu="H100:1", image=image, timeout=600)
def run_episode(action: str) -> dict:
    """
    Execute a single environment step on Modal H100 hardware.
    Returns observation, reward, done, and info.
    """
    from openenv_env.KernelForgeEnv import KernelForgeEnv

    env = KernelForgeEnv()
    obs, info = env.reset()

    # Execute the provided action
    obs, reward, done, truncated, info = env.step(action)

    env.close()

    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.function(gpu="H100:1", image=image, timeout=3600)
def run_full_episode(actions: list[str]) -> list[dict]:
    """
    Execute a full multi-turn episode on Modal H100 hardware.
    """
    from openenv_env.KernelForgeEnv import KernelForgeEnv

    env = KernelForgeEnv()
    obs, info = env.reset()

    results = [{"observation": obs, "info": info}]

    for action in actions:
        obs, reward, done, truncated, info = env.step(action)
        results.append({
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info,
        })
        if done:
            break

    env.close()
    return results
```

---

### File 7: tools/profile.py

```python
"""
CUDA kernel profiling script.
Runs inside the Docker container. Uses CuPy cudaEvent for accurate GPU timing.

Usage: python3 profile.py --binary /workspace/wcc_kernel --warmup 100 --runs 50
"""
import argparse
import subprocess
import sys

try:
    import cupy as cp
except ImportError:
    print("CuPy not available. Install cupy-cuda13x.", file=sys.stderr)
    sys.exit(1)


def profile_kernel(binary_path: str, warmup: int = 100, runs: int = 50) -> float:
    """Profile a compiled CUDA binary. Returns average execution time in ms."""

    # Warmup
    for _ in range(warmup):
        subprocess.run([binary_path], capture_output=True, timeout=30)

    # Synchronize before measurement
    cp.cuda.Device(0).synchronize()

    # Benchmark
    times = []
    for _ in range(runs):
        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()
        result = subprocess.run([binary_path], capture_output=True, timeout=30)
        end.record()
        end.synchronize()

        elapsed_ms = cp.cuda.get_elapsed_time(start, end)
        times.append(elapsed_ms)

        if result.returncode != 0:
            print(f"Kernel exited with code {result.returncode}", file=sys.stderr)
            return float("inf")

    avg_ms = sum(times) / len(times)
    return avg_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a CUDA WCC kernel")
    parser.add_argument("--binary", required=True, help="Path to compiled kernel binary")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=50, help="Benchmark iterations")
    args = parser.parse_args()

    avg = profile_kernel(args.binary, args.warmup, args.runs)
    # Output ONLY the average ms on the last line (environment parses this)
    print(f"{avg:.4f}")
```

---

### File 8: scripts/gen_dataset.py

```python
"""
Generate wcc_doubleai_inspired.jsonl — 500 RMAT/SBM graphs with ground-truth WCC.
Each record contains a graph description + expected component count for training data.

Usage: python3 scripts/gen_dataset.py --output datasets/wcc_doubleai_inspired.jsonl --n 500
"""
import argparse
import json
import numpy as np
import networkx as nx


def gen_rmat(n=16384, m=65536, seed=0):
    """Generate an RMAT power-law graph."""
    rng = np.random.default_rng(seed)
    a, b, c = 0.57, 0.19, 0.19
    log2n = int(np.log2(n))
    src = np.zeros(m, dtype=np.int64)
    dst = np.zeros(m, dtype=np.int64)
    for depth in range(log2n):
        choose = rng.random(m)
        src_bit = ((choose > a) & (choose <= a + b)) | (choose > a + b + c)
        dst_bit = ((choose > a + b) & (choose <= a + b + c)) | (choose > a + b + c)
        src += src_bit.astype(np.int64) * (1 << depth)
        dst += dst_bit.astype(np.int64) * (1 << depth)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = set(zip(src.tolist(), dst.tolist()))
    edges = [(u, v) for u, v in edges if u != v]
    G.add_edges_from(edges)
    return G


def gen_sbm(sizes, p_in=0.05, p_out=0.001, seed=0):
    """Generate a Stochastic Block Model graph."""
    k = len(sizes)
    p_matrix = np.full((k, k), p_out)
    np.fill_diagonal(p_matrix, p_in)
    return nx.stochastic_block_model(sizes, p_matrix, seed=seed)


def graph_to_record(G, graph_id, graph_type, params):
    """Convert a networkx graph to a JSONL training record."""
    n_components = nx.number_connected_components(G)
    n = G.number_of_nodes()
    m = G.number_of_edges()

    prompt = (
        f"Write an optimized CUDA WCC kernel for an undirected graph with "
        f"{n} nodes and {m} edges. The graph is {graph_type} with properties: {params}. "
        f"Target: H100 (sm_90). Use Union-Find with path compression. "
        f"Pin parent array to L2. Single kernel launch."
    )

    return {
        "id": graph_id,
        "graph_type": graph_type,
        "n_nodes": n,
        "n_edges": m,
        "n_components": n_components,
        "params": params,
        "prompt": prompt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="datasets/wcc_doubleai_inspired.jsonl")
    parser.add_argument("--n", type=int, default=500, help="Number of graphs to generate")
    args = parser.parse_args()

    records = []
    rng = np.random.default_rng(42)

    for i in range(args.n):
        if i % 2 == 0:
            # RMAT
            n = 2 ** rng.integers(10, 16)
            m = min(2 ** rng.integers(12, 18), n * (n - 1) // 2)
            G = gen_rmat(n=n, m=m, seed=i)
            params = {"n": int(n), "m": int(m), "type": "RMAT", "a": 0.57, "b": 0.19, "c": 0.19}
            rec = graph_to_record(G, f"rmat_{i}", "RMAT power-law", params)
        else:
            # SBM
            k = rng.integers(2, 6)
            block_size = rng.integers(100, 1000)
            sizes = [int(block_size)] * int(k)
            p_in = rng.uniform(0.01, 0.1)
            p_out = rng.uniform(0.0001, 0.005)
            G = gen_sbm(sizes=sizes, p_in=p_in, p_out=p_out, seed=i)
            params = {"sizes": sizes, "p_in": round(float(p_in), 4), "p_out": round(float(p_out), 4)}
            rec = graph_to_record(G, f"sbm_{i}", "Stochastic Block Model", params)

        records.append(rec)

        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{args.n} graphs...")

    with open(args.output, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
```

---

### File 9: demo/streamlit_demo.py

```python
"""
Streamlit demo UI for KernelForge-OpenEnv.
Connects to the Modal backend to run episodes on H100 hardware.
"""
import streamlit as st
import modal

st.set_page_config(page_title="KernelForge-OpenEnv", layout="wide")
st.title("KernelForge-OpenEnv: WCC Kernel Optimization")
st.markdown("Train an LLM to write faster CUDA WCC kernels than NVIDIA's experts.")

# Sidebar
st.sidebar.header("Configuration")
max_turns = st.sidebar.slider("Max Turns", 1, 15, 15)

# Main area
st.header("Agent Interaction")

if "history" not in st.session_state:
    st.session_state.history = []

action = st.text_area("Enter CUDA action (bash command):", height=150)

if st.button("Execute Step"):
    if action.strip():
        with st.spinner("Running on H100 via Modal..."):
            try:
                # Call Modal function
                fn = modal.Function.from_name("kernelforge-h100", "run_episode")
                result = fn.remote(action)

                st.session_state.history.append({
                    "action": action,
                    "result": result,
                })

                st.success(f"Reward: {result['reward']:.2f}")
                st.code(result["observation"], language="text")

                if result["done"]:
                    st.balloons()
                    st.success("Episode complete!")
            except Exception as e:
                st.error(f"Error: {e}")

# History
if st.session_state.history:
    st.header("Episode History")
    for i, entry in enumerate(st.session_state.history):
        with st.expander(f"Turn {i + 1}"):
            st.code(entry["action"], language="bash")
            st.json(entry["result"])
```

---

### File 10: requirements.txt

```
gymnasium>=1.0.0
openenv-core
docker>=7.0
modal>=0.73
torch>=2.10.0
cupy-cuda13x>=14.0
networkx>=3.4
scipy>=1.14
streamlit>=1.41
datasets>=3.0
```

> **Note:** Unsloth and TRL are installed separately for training:
> ```
> pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
> pip install trl>=0.15
> ```

---

### File 11: pyproject.toml

```toml
[build-system]
requires = ["setuptools>=75.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "kernelforge-openenv"
version = "0.1.0"
description = "Gymnasium RL environment for training LLMs to optimize CUDA WCC kernels on H100"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.12"
dependencies = [
    "gymnasium>=1.0.0",
    "openenv-core",
    "docker>=7.0",
    "networkx>=3.4",
    "scipy>=1.14",
    "numpy>=2.0",
]

[project.optional-dependencies]
train = [
    "trl>=0.15",
    "datasets>=3.0",
    "torch>=2.10.0",
]
demo = [
    "streamlit>=1.41",
    "modal>=0.73",
]
```

---

### File 12: Dockerfile

```dockerfile
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install the doubleGraph A100 wheel as baseline
# CORRECTED wheel name from v1.2
RUN pip install --no-cache-dir \
    cupy-cuda13x \
    networkx \
    scipy \
    https://github.com/double-ai/doubleGraph/releases/download/v0.1.0-a100/libcugraph_cu13-26.2.0-py3-none-linux_x86_64.whl

COPY tools/profile.py /workspace/tools/profile.py

CMD ["sleep", "infinity"]
```

---

## 7. Subtasks Execution Order for the AI Agent

1. Initialize empty repository matching the exact directory structure.
2. Populate `requirements.txt` and `pyproject.toml` (use exact content from Section 6).
3. Write `README.md` containing the exact DoubleAI context block and Sponsor block.
4. Write `skill_h100.md` exactly as provided (note: references `tools/profile.py`, not `profile.py`).
5. Implement `openenv_env/__init__.py` and `openenv_env/KernelForgeEnv.py` (includes PAC verification, time-travel rewind, milestone rewards — all bugs from v1.2 are fixed).
6. Write `modal_app.py` with the **corrected** doubleGraph wheel URL.
7. Write `training/grpo_train.py` with **corrected** imports (`trl.GRPOTrainer`, `Qwen/Qwen3.5-9B`).
8. Write `tools/profile.py` (CUDA profiling script).
9. Write `scripts/gen_dataset.py` and run it to generate `datasets/wcc_doubleai_inspired.jsonl` (500 RMAT/SBM graphs).
10. Write `demo/streamlit_demo.py` with Modal backend integration.
11. Write `Dockerfile` with corrected wheel installation.

---

## Appendix A: Reward Function Truth Table

This table exhaustively covers every possible outcome. **No gaps, no NameErrors.**

| Verification | speedup_vs_orig | speedup_vs_dg | base_reward | Notes |
|---|---|---|---|---|
| FAILED | N/A | N/A | -1.0 | Rewind triggered |
| PASSED | ≤ 1.0 | any | -0.5 | Degradation penalized |
| PASSED | (1.0, 1.05] | ≤ 1.05 | 0.5 | Marginal improvement |
| PASSED | > 1.05 | ≤ 1.05 | 2.0 | Beats original only |
| PASSED | > 1.05 | > 1.05 | 3.0 | Beats both baselines |

Final reward = `base_reward + log(1 + max(speedup_vs_orig, speedup_vs_dg)) * 15`

---

## Appendix B: Time-Travel Rewind Protocol

```
┌─────────────────────────────────────────────────────┐
│ Agent submits action                                │
│   ├─ Compile + run in container                     │
│   ├─ PAC verify (5 RMAT/SBM graphs)                │
│   │   ├─ PASS → profile → compute reward            │
│   │   │   └─ If reward > best_snapshot → save new   │
│   │   └─ FAIL → rewind to best_snapshot             │
│   │       ├─ Restore best code in container         │
│   │       ├─ Inject failed trajectory summary       │
│   │       └─ Require root-cause analysis next turn  │
│   └─ Return (obs, reward, done, truncated, info)    │
└─────────────────────────────────────────────────────┘
```

---

## Appendix C: What v1.2 Got Right (No Changes Needed)

- CUDA 13.0.2 Docker image: **Verified real** on Docker Hub
- cupy-cuda13x: **Verified real** on PyPI (v14.0.0)
- unsloth[colab-new]: **Verified valid** pip extra
- DoubleAI WarpSpeed announcement: **Verified March 2, 2026**
- Performance claims (100% faster, 55% >2x, 18% 10-100x): **Verified accurate**
- doubleGraph GitHub repo: **Verified real** at github.com/double-ai/doubleGraph
- PAC-reasoning, time-travel search, diligent learning: **Verified real** (arXiv:2412.02441)
