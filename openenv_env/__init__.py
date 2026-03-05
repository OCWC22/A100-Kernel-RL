"""
KernelForge OpenEnv Environment for target-GPU CUDA kernel RL training.

OpenEnv is an independent framework from Meta-PyTorch (NOT a Gymnasium
extension). HTTP client-server architecture with Docker container isolation.
Correct install (uv): uv add "openenv-core[core]>=0.2.1" (NOT "openenv")
"""


def __getattr__(name: str):
    if name == "KernelForgeEnv":
        from .kernel_forge_env import KernelForgeEnv

        return KernelForgeEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["KernelForgeEnv"]
