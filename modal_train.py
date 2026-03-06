"""
Modal training app — runs the 3-stage GRPO pipeline on cloud GPU.

Default: H200 (141GB, $4.54/hr on Modal). bf16 for Qwen3-Coder-30B-A3B-Instruct via Unsloth.
A100 (Modal) handles all eval/reward. You cannot optimize A100 perf by measuring on H200.

Usage:
    # Stage 1 warmup (default)
    modal run modal_train.py --stage 1

    # Stage 3 GRPO demo (10 steps)
    modal run modal_train.py --stage 3

    # Smoke test (1 step, no real training)
    modal run modal_train.py --stage 0
"""
import os
import modal

TRAIN_GPU = os.getenv("KERNELFORGE_TRAIN_GPU", "H200")
APP_NAME = os.getenv("KERNELFORGE_TRAIN_APP", "kernelforge-train")

train_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .uv_pip_install(
        "torch>=2.4",
        "trl[vllm]==0.29.0",
        "transformers>=4.56.2",
        "datasets>=3.0",
        "accelerate>=1.4.0",
        "peft>=0.14",
        "bitsandbytes>=0.45",
        "numpy>=1.26",
        "modal>=0.70",
        "flash-attn>=2.7",
    )
    # Unsloth installed separately to bypass stale trl<=0.24.0 cap
    .run_commands("pip install --no-deps unsloth unsloth_zoo 2>/dev/null || true")
    .add_local_python_source(
        "training", "openenv_env", "evaluation", "verification",
    )
    .add_local_file("skill_a100.md", "/root/skill_a100.md")
    .add_local_file("modal_app.py", "/root/modal_app.py")
)

app = modal.App(APP_NAME)
checkpoints_vol = modal.Volume.from_name("kernelforge-checkpoints", create_if_missing=True)
datasets_vol = modal.Volume.from_name("kernelforge-datasets", create_if_missing=True)


@app.function(
    gpu=TRAIN_GPU,
    image=train_image,
    timeout=3600 * 12,  # 12 hour max
    volumes={
        "/checkpoints": checkpoints_vol,
        "/datasets": datasets_vol,
    },
    include_source=True,
)
def train(stage: int = 1, max_steps: int | None = None, dry_run: bool = False):
    """Run a training stage on Modal GPU.

    Args:
        stage: 0=smoke test, 1=warmup, 2=RFT, 3=GRPO
        max_steps: Override max training steps (for cost control)
        dry_run: Just load model + dataset, don't train
    """
    import torch
    print(f"=== KernelForge Training Stage {stage} ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"CUDA: {torch.version.cuda}")

    if stage == 0:
        return _smoke_test()

    # Set output dirs to Modal volumes
    os.environ["KERNELFORGE_STAGE1_OUTPUT"] = "/checkpoints/stage1"
    os.environ["KERNELFORGE_STAGE2_OUTPUT"] = "/checkpoints/stage2"
    os.environ["KERNELFORGE_STAGE3_OUTPUT"] = "/checkpoints/stage3"

    if max_steps is not None:
        os.environ[f"KERNELFORGE_STAGE{stage}_MAX_STEPS"] = str(max_steps)

    if stage == 1:
        from training.stage1_warmup import main as stage1_main
        if dry_run:
            return _dry_run_stage1()
        stage1_main()
    elif stage == 2:
        from training.stage2_rft import main as stage2_main
        stage2_main()
    elif stage == 3:
        from training.stage3_grpo import main as stage3_main
        stage3_main()
    else:
        print(f"Unknown stage: {stage}")
        return {"error": f"Unknown stage {stage}"}

    # Commit volumes so checkpoints persist
    checkpoints_vol.commit()
    return {"status": "complete", "stage": stage}


def _smoke_test() -> dict:
    """Quick validation: model loads, dataset loads, one forward pass."""
    import torch

    results = {}

    # Test 1: Model loading
    print("\n[1/4] Loading model...")
    try:
        from training.model_loader import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer()
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results["model"] = {
            "loaded": True,
            "total_params": params,
            "trainable_params": trainable,
            "vram_gb": torch.cuda.memory_allocated() / 1e9,
        }
        print(f"  Model loaded: {trainable:,} trainable / {params:,} total")
        print(f"  VRAM used: {results['model']['vram_gb']:.1f} GB")
    except Exception as e:
        results["model"] = {"loaded": False, "error": str(e)[:500]}
        print(f"  FAILED: {e}")

    # Test 2: Dataset loading
    print("\n[2/4] Loading dataset...")
    try:
        from training.stage1_warmup import load_stage1_dataset
        dataset = load_stage1_dataset()
        results["dataset"] = {
            "loaded": True,
            "size": len(dataset),
            "columns": dataset.column_names,
        }
        print(f"  Dataset loaded: {len(dataset)} examples, columns: {dataset.column_names}")
    except Exception as e:
        results["dataset"] = {"loaded": False, "error": str(e)[:500]}
        print(f"  FAILED: {e}")

    # Test 3: Generation
    print("\n[3/4] Testing generation...")
    try:
        test_prompt = "Write a CUDA vector addition kernel for A100."
        inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=64, temperature=0.7)
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        results["generation"] = {
            "works": True,
            "output_tokens": len(output[0]),
            "preview": generated[:200],
        }
        print(f"  Generated {len(output[0])} tokens")
    except Exception as e:
        results["generation"] = {"works": False, "error": str(e)[:500]}
        print(f"  FAILED: {e}")

    # Test 4: Eval connectivity
    print("\n[4/4] Testing Modal eval endpoint...")
    try:
        eval_fn = modal.Function.from_name("kernelforge-a100", "evaluate_ops6k_kernel")
        test_result = eval_fn.remote({
            "cuda_code": '__global__ void test(float* x, int n) { int i = threadIdx.x; if (i < n) x[i] *= 2.0f; }',
            "task_code": 'import torch\nclass Model(torch.nn.Module):\n  def __init__(self): super().__init__()\n  def forward(self, x): return x * 2\ndef get_inputs(): return [torch.randn(256).cuda()]\ndef get_init_inputs(): return []',
        })
        results["eval"] = {"connected": True, "compiles": test_result.get("compiles", False)}
        print(f"  Eval endpoint: compiles={test_result.get('compiles')}")
    except Exception as e:
        results["eval"] = {"connected": False, "error": str(e)[:500]}
        print(f"  FAILED: {e}")

    print("\n=== Smoke Test Complete ===")
    for key, val in results.items():
        status = "PASS" if val.get("loaded") or val.get("works") or val.get("connected") else "FAIL"
        print(f"  {key}: {status}")

    return results


def _dry_run_stage1() -> dict:
    """Load model + dataset but don't train. For cost verification."""
    import torch
    from training.model_loader import load_model_and_tokenizer
    from training.stage1_warmup import load_stage1_dataset

    model, tokenizer = load_model_and_tokenizer()
    dataset = load_stage1_dataset()

    vram = torch.cuda.memory_allocated() / 1e9
    total_vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"Model VRAM: {vram:.1f} / {total_vram:.1f} GB")
    print(f"Dataset: {len(dataset)} examples")
    print(f"Free VRAM: {total_vram - vram:.1f} GB (for KV cache + optimizer)")

    return {
        "status": "dry_run",
        "vram_used_gb": vram,
        "vram_total_gb": total_vram,
        "dataset_size": len(dataset),
        "fits": vram < total_vram * 0.85,
    }


@app.local_entrypoint()
def main(stage: int = 1, max_steps: int = 0, dry_run: bool = False):
    """CLI entrypoint: modal run modal_train.py --stage 1 --max-steps 10"""
    steps = max_steps if max_steps > 0 else None
    result = train.remote(stage=stage, max_steps=steps, dry_run=dry_run)
    print(f"\nResult: {result}")
