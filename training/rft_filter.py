"""
Rejection Fine-Tuning (RFT) Filter for KernelForge.

Stage 2 of the 3-stage training pipeline:
- Collect trajectories from SFT warm-up model
- Filter aggressively: keep only trajectories with reward >= 2
- Creates high-quality dataset for GRPO training

Critical: Skipping RFT causes policy entropy explosion and training collapse.
"""
import modal
import json
import random
from typing import List, Dict, Any
from datasets import Dataset
import os

from training.cuda_agent_integration import load_cuda_agent_prompt_texts
from openenv_env.reward import compute_reward


TARGET_GPU = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
DEFAULT_MODAL_APP = os.getenv("KERNELFORGE_MODAL_APP", "kernelforge-a100")
DEFAULT_MODEL_PATH = os.getenv("KERNELFORGE_RFT_MODEL_PATH", "outputs/kernelforge-sft")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("KERNELFORGE_RFT_MAX_NEW_TOKENS", "2048"))


class TrajectoryCollector:
    """Collect and filter trajectories for RFT."""
    
    def __init__(self, modal_app_name: str | None = None, model_path: str | None = None):
        self.modal_app_name = modal_app_name or DEFAULT_MODAL_APP
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.trajectories = []
        self._baselines = None
        self._generator = None

    def _get_baselines(self) -> tuple[float | None, float | None]:
        """Fetch and cache baseline runtimes from Modal."""
        if self._baselines is None:
            try:
                baseline_fn = modal.Function.from_name(self.modal_app_name, "profile_baselines")
                self._baselines = baseline_fn.remote() or {}
            except Exception as e:
                print(f"Baseline profiling failed: {e}")
                self._baselines = {}

        return self._baselines.get("original_ms"), self._baselines.get("doublegraph_ms")
        
    def collect_trajectories(self, num_trajectories: int = 100) -> List[Dict[str, Any]]:
        """
        Collect trajectories by running the model on various WCC prompts.
        
        Args:
            num_trajectories: Number of trajectories to collect
            
        Returns:
            List of trajectory dictionaries
        """
        print(f"Collecting {num_trajectories} trajectories...")
        
        prompts = self._generate_wcc_prompts()
        
        for i in range(num_trajectories):
            prompt = random.choice(prompts)
            trajectory = self._run_single_trajectory(prompt, i)
            
            if trajectory:
                self.trajectories.append(trajectory)
                print(f"Trajectory {i+1}/{num_trajectories}: reward={trajectory['reward']}")
            
            if (i + 1) % 10 == 0:
                print(f"Collected {len(self.trajectories)} trajectories so far...")
        
        return self.trajectories
    
    def _generate_wcc_prompts(self) -> List[str]:
        """Generate diverse CUDA kernel prompts (P1-11: not WCC-only)."""
        prompts = [
            f"Write a CUDA vector addition kernel for {TARGET_GPU} ({TARGET_ARCH}). Take float* A, float* B, float* C, int N.",
            f"Write a CUDA matrix multiplication kernel using shared memory tiling for {TARGET_GPU} ({TARGET_ARCH}).",
            f"Write a CUDA softmax kernel computing row-wise softmax for {TARGET_GPU} ({TARGET_ARCH}).",
            f"Write a CUDA ReLU activation kernel for {TARGET_GPU} ({TARGET_ARCH}).",
            f"Write a fused CUDA LayerNorm + GELU kernel for {TARGET_GPU} ({TARGET_ARCH}).",
            f"Write a CUDA WCC kernel using non-atomic Union-Find for {TARGET_GPU} ({TARGET_ARCH}).",
            f"Write a CUDA reduction kernel using cooperative groups for {TARGET_GPU} ({TARGET_ARCH}).",
            f"Write a CUDA GEMM kernel with float4 vectorized loads for {TARGET_GPU} ({TARGET_ARCH}).",
            f"Write a fused CUDA MatMul + BiasAdd kernel for {TARGET_GPU} ({TARGET_ARCH}).",
            f"Write a CUDA batch normalization kernel for {TARGET_GPU} ({TARGET_ARCH}).",
        ]

        # Optional augmentation from CUDA-Agent-Ops-6K operator tasks.
        # These samples increase prompt diversity without requiring labeled completions.
        try:
            extra_count = int(os.getenv("CUDA_AGENT_RFT_PROMPTS", "64"))
            if extra_count > 0:
                cuda_agent_prompts = load_cuda_agent_prompt_texts(max_samples=extra_count)
                if cuda_agent_prompts:
                    prompts.extend(cuda_agent_prompts)
                    print(f"Augmented RFT prompt pool with {len(cuda_agent_prompts)} CUDA-Agent prompts")
        except Exception as e:
            print(f"Could not augment RFT prompts from CUDA-Agent-Ops-6K: {e}")
        
        return prompts
    
    def _run_single_trajectory(self, prompt: str, trajectory_id: int) -> Dict[str, Any]:
        """
        Run a single trajectory: prompt -> model -> evaluation -> reward.
        
        Args:
            prompt: The WCC optimization prompt
            trajectory_id: Unique identifier for this trajectory
            
        Returns:
            Trajectory dictionary or None if failed
        """
        try:
            model_output = self._get_model_response(prompt)
            
            # Evaluate on Modal GPU backend
            reward_result = self._evaluate_kernel(model_output)
            
            trajectory = {
                "id": trajectory_id,
                "prompt": prompt,
                "model_output": model_output,
                "reward": reward_result["reward"],
                "compiles": reward_result["compiles"],
                "correct": reward_result["correct"],
                "speedup_vs_orig": reward_result.get("speedup_vs_orig", 0),
                "speedup_vs_dg": reward_result.get("speedup_vs_dg", 0),
                "error": reward_result.get("error", ""),
                "timestamp": trajectory_id,  # Simplified timestamp
            }
            
            return trajectory
            
        except Exception as e:
            print(f"Error in trajectory {trajectory_id}: {e}")
            return None
    
    def _get_generator(self):
        """Lazily initialize text-generation pipeline from SFT checkpoint."""
        if self._generator is not None:
            return self._generator

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        self._generator = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        return self._generator

    def _fallback_kernel_template(self) -> str:
        """Deterministic fallback when model checkpoint is unavailable."""
        return """```cuda
#include <cuda_runtime.h>

__device__ int find_root(int* parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

extern "C" __global__ void wcc_kernel(int* parent, const int* row_ptr, const int* col_idx, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int root_v = find_root(parent, tid);
    for (int e = row_ptr[tid]; e < row_ptr[tid + 1]; ++e) {
        int u = col_idx[e];
        int root_u = find_root(parent, u);
        if (root_u != root_v) {
            int lo = min(root_u, root_v);
            int hi = max(root_u, root_v);
            parent[hi] = lo;
            root_v = lo;
        }
    }
}
```"""

    def _get_model_response(self, prompt: str) -> str:
        """Generate a model response from the SFT checkpoint with safe fallback."""
        try:
            generator = self._get_generator()
            outputs = generator(
                prompt,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=1,
            )
            text = outputs[0]["generated_text"]
            if text.startswith(prompt):
                text = text[len(prompt):]
            return text.strip()
        except Exception as e:
            print(f"Model generation failed ({self.model_path}): {e}. Using deterministic fallback kernel template.")
            return self._fallback_kernel_template()
    
    def _evaluate_kernel(self, kernel_code: str) -> Dict[str, Any]:
        """
        Evaluate kernel code on Modal GPU backend.
        
        Args:
            kernel_code: CUDA kernel source code
            
        Returns:
            Evaluation result dictionary
        """
        try:
            import modal
            eval_fn = modal.Function.from_name(self.modal_app_name, "evaluate_kernel")
            baseline_orig_ms, baseline_dg_ms = self._get_baselines()
            
            result = eval_fn.remote({
                "cuda_code": kernel_code,
                "verify_graphs": 5,
                "warmup_iters": 50,
                "benchmark_runs": 30,
                "baseline_original_ms": baseline_orig_ms,
                "baseline_doublegraph_ms": baseline_dg_ms,
            })
            
            # Compute reward via canonical function (P1-8: single source of truth)
            reward = compute_reward(
                compiled=result.get("compiles", False),
                correct=result.get("correct", False),
                speedup_vs_eager=result.get("speedup_vs_orig", 0),
                speedup_vs_compile=result.get("speedup_vs_dg", 0),
            )
            
            return {
                "reward": reward,
                "compiles": result.get("compiles", False),
                "correct": result.get("correct", False),
                "speedup_vs_orig": result.get("speedup_vs_orig", 0),
                "speedup_vs_dg": result.get("speedup_vs_dg", 0),
                "error": result.get("error", ""),
            }
            
        except Exception as e:
            # Fail closed: do not generate synthetic high rewards when evaluation fails.
            print(f"Modal evaluation failed: {e}")
            return {
                "reward": -1.0,
                "compiles": False,
                "correct": False,
                "speedup_vs_orig": 0.0,
                "speedup_vs_dg": 0.0,
                "error": str(e),
            }
    
    def filter_trajectories(self, min_reward: float = 1.0) -> List[Dict[str, Any]]:
        """
        Filter trajectories based on reward threshold.

        P0-3 fix: threshold changed from 2.0 to 1.0 to keep correct-but-not-fast trajectories.

        Args:
            min_reward: Minimum reward to keep trajectory (default 1.0 = correct)

        Returns:
            Filtered list of high-quality trajectories
        """
        filtered = [t for t in self.trajectories if t["reward"] >= min_reward]

        # P0-4 fix: guard against ZeroDivisionError when no trajectories collected
        total = len(self.trajectories)
        if total > 0:
            print(f"Filtered trajectories: {len(filtered)}/{total} "
                  f"({len(filtered)/total*100:.1f}%)")
        else:
            print("No trajectories collected.")

        return filtered
    
    def save_rft_dataset(self, filtered_trajectories: List[Dict[str, Any]], output_path: str):
        """
        Save filtered trajectories as RFT dataset.
        
        Args:
            filtered_trajectories: High-quality trajectories
            output_path: Output file path
        """
        rft_examples = []
        
        for traj in filtered_trajectories:
            # Format for SFT training
            example = {
                "messages": [
                    {"role": "user", "content": traj["prompt"]},
                    {"role": "assistant", "content": traj["model_output"]}
                ],
                "reward": traj["reward"],
                "compiles": traj["compiles"],
                "correct": traj["correct"],
                "speedup_vs_orig": traj["speedup_vs_orig"],
                "speedup_vs_dg": traj["speedup_vs_dg"],
            }
            rft_examples.append(example)
        
        # Save as JSONL
        with open(output_path, 'w') as f:
            for example in rft_examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"Saved {len(rft_examples)} RFT examples to {output_path}")
        
        # Also save as HuggingFace dataset
        dataset = Dataset.from_list(rft_examples)
        dataset.save_to_disk(output_path.replace('.jsonl', '_hf'))
        
        return dataset


def main():
    """Main RFT filtering process."""
    print("Starting Rejection Fine-Tuning (RFT) filtering...")
    
    collector = TrajectoryCollector()
    
    # Collect trajectories
    trajectories = collector.collect_trajectories(num_trajectories=50)
    
    # Filter high-quality trajectories
    filtered = collector.filter_trajectories(min_reward=1.0)
    
    if not filtered:
        print("No trajectories met the quality threshold!")
        return
    
    # Save RFT dataset
    os.makedirs("datasets", exist_ok=True)
    rft_dataset = collector.save_rft_dataset(filtered, "datasets/wcc_rft.jsonl")
    
    print(f"RFT filtering completed! Created dataset with {len(filtered)} high-quality examples.")
    
    # Print statistics
    rewards = [t["reward"] for t in filtered]
    print(f"Reward statistics: min={min(rewards):.1f}, max={max(rewards):.1f}, mean={sum(rewards)/len(rewards):.1f}")


if __name__ == "__main__":
    main()
