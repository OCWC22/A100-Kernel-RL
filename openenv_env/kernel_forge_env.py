"""
KernelForge OpenEnv Environment for target-GPU CUDA kernel RL training.

OpenEnv is an independent framework from Meta-PyTorch (NOT a Gymnasium
extension). HTTP client-server architecture with Docker container isolation.
Correct install (uv): uv add "openenv-core[core]>=0.2.1" (NOT "openenv")
"""
import os
from typing import Any
from uuid import uuid4

from openenv.core.env_server import Environment
from openenv.core.env_server.types import State
from openenv_env.models import KernelForgeAction, KernelForgeObservation
from openenv_env.gpu_registry import get_gpu_spec
from openenv_env.reward import compute_reward
from openenv_env.skill_builder import build_skill_md
from training.task_support import (
    build_modal_payload,
    normalize_eval_result,
    normalize_task_row,
)


class KernelForgeEnv(Environment):
    """
    RL environment: agent submits CUDA source -> target GPU compiles/verifies/benchmarks.

    Action space: CUDA source code string
    Observation: SKILL.md + compilation/verification/benchmark feedback + history
    Reward: discrete milestones {-1, +1, +2, +3}
    Max turns: 200 (ByteDance used 150; extended for hackathon exploration)
    Context: 128K tokens
    """

    def __init__(self, modal_function_name: str | None = None):
        super().__init__()
        self.modal_fn = modal_function_name or os.getenv("KERNELFORGE_MODAL_APP", "kernelforge-a100")
        self.target_gpu = os.getenv("KERNELFORGE_TARGET_GPU", "a100").lower()
        self.gpu_spec = get_gpu_spec(self.target_gpu)
        self.history = []               # Time-travel snapshots (DoubleAI-inspired)
        self.turn = 0
        self.max_turns = 200
        self.best_reward = -1.0
        self.best_code = None
        self.original_baseline_ms = None
        self.doublegraph_baseline_ms = None
        self.current_task = normalize_task_row(
            {
                "prompt": "Optimize a Weakly Connected Components CUDA kernel.",
                "ops": ["weakly_connected_components"],
                "data_source": "openenv_default",
                "difficulty": 1,
            }
        )
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self) -> KernelForgeObservation:
        """Reset environment. Profile baselines on first call."""
        self.history = []
        self.turn = 0
        self.best_reward = -1.0
        self.best_code = None
        self._state = State(episode_id=str(uuid4()), step_count=0)

        if (
            self.original_baseline_ms is None
            and self.current_task.get("evaluation_backend") == "wcc"
        ):
            baselines = self._modal("profile_baselines")
            self.original_baseline_ms = baselines["original_ms"]
            self.doublegraph_baseline_ms = baselines.get("doublegraph_ms")

        return KernelForgeObservation(
            text=(
                f"{build_skill_md(self.target_gpu)}\n\n---\n\n"
                f"{self.current_task.get('prompt', '')}"
            ),
            baseline_original_ms=self.original_baseline_ms,
            baseline_doublegraph_ms=self.doublegraph_baseline_ms,
            hardware=self.gpu_spec,
            reward=0.0,
            done=False,
            turn=self.turn,
            best_reward=self.best_reward,
            info={"phase": "reset"},
            graph_properties=self.current_task.get("graph_properties"),
            topology_type=self.current_task.get("topology"),
        )

    def step(self, action: KernelForgeAction) -> KernelForgeObservation:
        """Execute one environment step from CUDA source code action."""
        self.turn += 1
        self._state.step_count = self.turn

        try:
            fn_name, payload = build_modal_payload(
                action.cuda_code,
                self.current_task,
                baseline_orig_ms=self.original_baseline_ms,
                baseline_dg_ms=self.doublegraph_baseline_ms,
            )
            result = normalize_eval_result(self._modal(fn_name, payload))
        except Exception as exc:
            result = {
                "compiles": False,
                "correct": False,
                "runtime_ms": 0.0,
                "runtime_stats": {},
                "speedup_vs_orig": 0.0,
                "speedup_vs_dg": 0.0,
                "error": str(exc),
            }

        # Compute speedups and reward via canonical function (P1-8)
        if not result.get("compiles"):
            reward = -1.0
            su_orig = 0
            obs = (f"COMPILATION FAILED (turn {self.turn}/{self.max_turns}):\n"
                   f"{result.get('error', 'Unknown error')[:1500]}")
        elif not result.get("correct"):
            reward = -1.0
            su_orig = 0
            obs = (f"VERIFICATION FAILED (turn {self.turn}/{self.max_turns}):\n"
                   f"{result.get('verifier_msg', 'Unknown failure')}")
        else:
            rt = float(result["runtime_ms"])
            su_orig = (
                self.original_baseline_ms / rt
                if self.original_baseline_ms and rt > 0
                else float(result.get("speedup_vs_orig", 0.0) or 0.0)
            )
            su_dg = (
                self.doublegraph_baseline_ms / rt
                if self.doublegraph_baseline_ms and rt > 0
                else float(result.get("speedup_vs_dg", 0.0) or 0.0)
            )

            reward = compute_reward(
                compiled=True,
                correct=True,
                speedup_vs_eager=su_orig,
                speedup_vs_compile=su_dg,
            )

            obs = (f"BENCHMARK (turn {self.turn}/{self.max_turns}):\n"
                   f"  Runtime: {rt:.3f}ms\n"
                   f"  vs cuGraph: {su_orig:.2f}x")
            if su_dg:
                obs += f"\n  vs doubleGraph: {su_dg:.2f}x"
            obs += f"\n  Stats: {result.get('runtime_stats', {})}"

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_code = action.cuda_code

        done = (self.turn >= self.max_turns) or (reward >= 3.0)

        # Time-travel with experience (DoubleAI-inspired)
        # Each snapshot carries knowledge of what was tried and what failed
        self.history.append({
            "turn": self.turn,
            "reward": reward,
            "obs_summary": obs[:200],
        })

        # Include history in observation for time-travel context
        if len(self.history) > 1:
            history_ctx = "\n--- Previous attempts (time-travel context) ---\n"
            for h in self.history[:-1]:
                history_ctx += (f"Turn {h['turn']}: reward={h['reward']}, "
                               f"{h['obs_summary'][:80]}\n")
            obs = history_ctx + "\n--- Current result ---\n" + obs

        return KernelForgeObservation(
            text=obs,
            reward=reward,
            done=done,
            turn=self.turn,
            best_reward=self.best_reward,
            info={
                "turn": self.turn,
                "best_reward": self.best_reward,
                "speedup": su_orig if result.get("correct") else 0,
                "evaluation_backend": self.current_task.get("evaluation_backend"),
            },
            graph_properties=self.current_task.get("graph_properties"),
            topology_type=self.current_task.get("topology"),
        )

    @property
    def state(self) -> State:
        return State(
            episode_id=self._state.episode_id,
            step_count=self.turn,
            history=self.history,
            best_reward=self.best_reward,
        )

    def close(self):
        """Clean up episode resources."""
        self.history = []
        self.current_task = None

    def _modal(self, fn_name, payload=None):
        """Dispatch to configured Modal app."""
        import modal
        fn = modal.Function.from_name(self.modal_fn, fn_name)
        if payload is None:
            return fn.remote()
        return fn.remote(payload)
