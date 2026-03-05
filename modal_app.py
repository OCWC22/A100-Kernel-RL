"""
Modal serverless functions for CUDA kernel evaluation.
All GPU work runs here: compilation, verification, benchmarking, and baselines.
"""
import modal
import subprocess
import tempfile
import os
import ctypes
import numpy as np
from typing import Any

# CUDA 12.4 Docker image with Ampere/Hopper support
cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .uv_pip_install(
        "cupy-cuda12x>=14.0",
        "networkx>=3.0",
        "numpy>=1.26",
        "scipy>=1.12",
    )
    # Python packages → add_local_python_source (auto-added to PYTHONPATH at /root)
    .add_local_python_source("verification", "openenv_env")
    # Non-Python CUDA sources → add_local_dir
    .add_local_dir("kernels", remote_path="/root/kernels")
)

TARGET_GPU = os.getenv("KERNELFORGE_MODAL_GPU", os.getenv("KERNELFORGE_TARGET_GPU", "A100"))
TARGET_CUDA_ARCH = os.getenv("KERNELFORGE_CUDA_ARCH", os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80"))
APP_NAME = os.getenv("KERNELFORGE_MODAL_APP", "kernelforge-a100")
BASELINE_KERNEL = os.getenv("KERNELFORGE_BASELINE_KERNEL", "baseline_wcc.cu")
SECONDARY_BASELINE_KERNEL = os.getenv("KERNELFORGE_SECONDARY_BASELINE_KERNEL", "")

app = modal.App(APP_NAME)
kernel_cache = modal.Volume.from_name("kernelforge-cache", create_if_missing=True)



def _resolve_kernel_path(kernel_name: str) -> str | None:
    """Resolve kernel source path from common locations."""
    if not kernel_name:
        return None

    if os.path.isabs(kernel_name) and os.path.exists(kernel_name):
        return kernel_name

    root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(root, "kernels", kernel_name),
        os.path.join(root, kernel_name),
        os.path.join("/root/kernels", kernel_name),
        os.path.join(os.getcwd(), "kernels", kernel_name),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def _load_kernel_source(kernel_name: str) -> str:
    """Load CUDA kernel source by filename."""
    resolved = _resolve_kernel_path(kernel_name)
    if not resolved:
        raise FileNotFoundError(f"Kernel source not found: {kernel_name}")

    with open(resolved, "r", encoding="utf-8") as f:
        return f.read()


def _benchmark_kernel_source(
    cuda_code: str,
    warmup_iters: int = 40,
    benchmark_runs: int = 30,
    num_vertices: int = 10000,
) -> dict:
    """Compile, verify, and benchmark a kernel source string once."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "baseline.cu")
        lib_path = os.path.join(tmpdir, "baseline.so")

        with open(src_path, "w", encoding="utf-8") as f:
            f.write(cuda_code)

        proc = subprocess.run(
            [
                "nvcc",
                f"-arch={TARGET_CUDA_ARCH}",
                "-O3",
                "-use_fast_math",
                src_path,
                "-o",
                lib_path,
                "--shared",
                "-Xcompiler",
                "-fPIC",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Baseline compile failed: {proc.stderr[:1200]}")

        from verification.pac_verify import (
            generate_test_graphs,
            run_kernel_verification,
            verify_wcc,
        )
        import cupy as cp

        graphs = generate_test_graphs(num_vertices=num_vertices)
        edges = graphs[0][1]
        n_verts = graphs[0][2]

        labels = run_kernel_verification(lib_path, edges, n_verts)
        passed, msg = verify_wcc(labels, edges, n_verts)
        if not passed:
            raise RuntimeError(f"Baseline verification failed: {msg}")

        for _ in range(warmup_iters):
            run_kernel_verification(lib_path, edges, n_verts)
        cp.cuda.Device(0).synchronize()

        times = []
        for _ in range(benchmark_runs):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            run_kernel_verification(lib_path, edges, n_verts)
            end.record()
            end.synchronize()
            times.append(cp.cuda.get_elapsed_time(start, end))

        times_np = np.array(times)
        return {
            "runtime_ms": float(np.median(times_np)),
            "runtime_stats": {
                "mean": float(np.mean(times_np)),
                "median": float(np.median(times_np)),
                "std": float(np.std(times_np)),
                "min": float(np.min(times_np)),
                "max": float(np.max(times_np)),
            },
        }


@app.function(
    gpu=TARGET_GPU,
    image=cuda_image,
    timeout=120,
    volumes={"/cache": kernel_cache},
    include_source=True,
)
def evaluate_kernel(payload: dict) -> dict:
    """
    Compile, verify (PAC-reasoning), and benchmark CUDA kernel on target GPU.

    Input:
        cuda_code: str          - CUDA source
        verify_graphs: int      - number of test graphs (default 5)
        warmup_iters: int       - warmup iterations (default 50)
        benchmark_runs: int     - timed runs (default 30)

    Returns:
        compiles: bool
        correct: bool
        verifier_msg: str
        runtime_ms: float       - median of benchmark_runs
        runtime_stats: dict     - {mean, median, std, min, max}
        speedup_vs_orig: float
        speedup_vs_dg: float
        error: str
    """
    cuda_code = payload.get("cuda_code", "")
    baseline_orig_ms = payload.get("baseline_original_ms")
    baseline_dg_ms = payload.get("baseline_doublegraph_ms")
    result = {
        "compiles": False, "correct": False, "verifier_msg": "",
        "runtime_ms": 0.0, "runtime_stats": {},
        "speedup_vs_orig": 0.0, "speedup_vs_dg": 0.0, "error": "",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.cu")
        lib_path = os.path.join(tmpdir, "kernel.so")

        with open(src_path, "w") as f:
            f.write(cuda_code)

        # Step 1: Compile for target architecture
        try:
            proc = subprocess.run(
                ["nvcc", f"-arch={TARGET_CUDA_ARCH}", "-O3", "-use_fast_math",
                 src_path, "-o", lib_path, "--shared", "-Xcompiler", "-fPIC"],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode != 0:
                result["error"] = proc.stderr[:2000]
                return result
        except subprocess.TimeoutExpired:
            result["error"] = "Compilation timed out (30s limit)"
            return result

        result["compiles"] = True

        # Step 2: PAC Verification (5 adversarial graphs)
        try:
            from verification.pac_verify import generate_test_graphs, verify_wcc, run_kernel_verification

            graphs = generate_test_graphs(num_vertices=10000)
            ctypes.CDLL(lib_path)

            all_passed = True
            for graph_type, edges, n_verts in graphs:
                kernel_labels = run_kernel_verification(lib_path, edges, n_verts)
                passed, msg = verify_wcc(kernel_labels, edges, n_verts)
                if not passed:
                    result["verifier_msg"] = f"FAILED on {graph_type}: {msg}"
                    all_passed = False
                    break

            result["correct"] = all_passed
            if all_passed:
                result["verifier_msg"] = f"All {len(graphs)} graphs verified"
        except Exception as e:
            result["correct"] = False
            result["verifier_msg"] = f"Verification exception: {str(e)[:500]}"
            return result

        # Step 3: Benchmark with cudaEvent timing
        if result["correct"]:
            try:
                import cupy as cp

                warmup = payload.get("warmup_iters", 50)
                runs = payload.get("benchmark_runs", 30)

                # Use first graph for benchmarking
                edges = graphs[0][1]
                n_verts = graphs[0][2]

                # Warmup (mandatory — prevents cold-start exploitation)
                for _ in range(warmup):
                    run_kernel_verification(lib_path, edges, n_verts)
                cp.cuda.Device(0).synchronize()

                # Timed runs with cudaEvent
                times = []
                for _ in range(runs):
                    start = cp.cuda.Event()
                    end = cp.cuda.Event()
                    start.record()
                    run_kernel_verification(lib_path, edges, n_verts)
                    end.record()
                    end.synchronize()
                    times.append(cp.cuda.get_elapsed_time(start, end))

                times = np.array(times)
                result["runtime_ms"] = float(np.median(times))
                result["runtime_stats"] = {
                    "mean": float(np.mean(times)),
                    "median": float(np.median(times)),
                    "std": float(np.std(times)),
                    "min": float(np.min(times)),
                    "max": float(np.max(times)),
                }

                if baseline_orig_ms and result["runtime_ms"] > 0:
                    result["speedup_vs_orig"] = float(baseline_orig_ms) / result["runtime_ms"]
                if baseline_dg_ms and result["runtime_ms"] > 0:
                    result["speedup_vs_dg"] = float(baseline_dg_ms) / result["runtime_ms"]
            except Exception as e:
                result["error"] = f"Benchmark exception: {str(e)[:500]}"

    return result


@app.function(gpu=TARGET_GPU, image=cuda_image, timeout=300, include_source=True)
def profile_baselines() -> dict:
    """
    Profile baseline kernels on the target GPU using the same verifier/timing path.
    
    Returns:
        original_ms: baseline runtime in milliseconds
        doublegraph_ms: optional secondary baseline runtime in milliseconds
    """
    details: dict[str, Any] = {
        "target_gpu": TARGET_GPU,
        "target_arch": TARGET_CUDA_ARCH,
        "baseline_kernel": BASELINE_KERNEL,
    }

    original_ms = None
    doublegraph_ms = None

    try:
        baseline_code = _load_kernel_source(BASELINE_KERNEL)
        baseline_result = _benchmark_kernel_source(baseline_code)
        original_ms = baseline_result["runtime_ms"]
        details["baseline_runtime_stats"] = baseline_result["runtime_stats"]
    except Exception as e:
        details["baseline_error"] = str(e)

    if SECONDARY_BASELINE_KERNEL:
        details["secondary_kernel"] = SECONDARY_BASELINE_KERNEL
        try:
            secondary_code = _load_kernel_source(SECONDARY_BASELINE_KERNEL)
            secondary_result = _benchmark_kernel_source(secondary_code)
            doublegraph_ms = secondary_result["runtime_ms"]
            details["secondary_runtime_stats"] = secondary_result["runtime_stats"]
        except Exception as e:
            details["secondary_error"] = str(e)

    if original_ms is None:
        # Last-resort measured fallback (CPU reference) instead of synthetic constants.
        try:
            import networkx as nx
            import time
            from verification.pac_verify import generate_test_graphs

            graphs = generate_test_graphs(num_vertices=50000)
            edges = graphs[0][1]
            n_verts = graphs[0][2]

            graph = nx.Graph()
            graph.add_nodes_from(range(n_verts))
            graph.add_edges_from(edges)

            start_t = time.time()
            _ = list(nx.connected_components(graph))
            original_ms = (time.time() - start_t) * 1000.0
            details["fallback_source"] = "networkx_cpu_measured"
        except Exception as e:
            details["fallback_error"] = str(e)

    return {
        "original_ms": original_ms,
        "doublegraph_ms": doublegraph_ms,
        "baseline_source": details.get("fallback_source", "compiled_kernel"),
        "details": details,
    }


def edges_to_csr(edges, num_vertices):
    """Convert edge list to CSR format."""
    adj = [[] for _ in range(num_vertices)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    row_ptr = np.zeros(num_vertices + 1, dtype=np.int32)
    col_idx = []
    
    for i, neighbors in enumerate(adj):
        row_ptr[i + 1] = row_ptr[i] + len(neighbors)
        col_idx.extend(sorted(neighbors))
    
    return row_ptr, np.array(col_idx, dtype=np.int32)


@app.function(gpu=TARGET_GPU, image=cuda_image, timeout=60, include_source=True)
def test_h100_features() -> dict:
    """
    Test target GPU feature availability.
    
    Returns:
        tma_available: bool
        dsmem_available: bool  
        dpx_available: bool
        compute_capability: str
    """
    try:
        import cupy as cp

        device = cp.cuda.Device(0)
        cc = device.compute_capability  # str like "80" or "90"
        major = int(str(cc)[:1])
        compute_cap = f"{major}.{str(cc)[1:]}" if len(str(cc)) > 1 else str(cc)

        tma_available = major >= 9
        dsmem_available = major >= 9
        dpx_available = major >= 9
        async_copy_available = major >= 8

        props = cp.cuda.runtime.getDeviceProperties(0)
        raw_name = props.get("name", b"unknown")
        device_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)

        return {
            "target_gpu": TARGET_GPU,
            "target_arch": TARGET_CUDA_ARCH,
            "tma_available": tma_available,
            "dsmem_available": dsmem_available,
            "dpx_available": dpx_available,
            "async_copy_available": async_copy_available,
            "compute_capability": compute_cap,
            "device_name": device_name,
        }

    except Exception as e:
        return {
            "error": str(e),
            "target_gpu": TARGET_GPU,
            "target_arch": TARGET_CUDA_ARCH,
            "tma_available": False,
            "dsmem_available": False,
            "dpx_available": False,
            "async_copy_available": False,
            "compute_capability": "unknown",
        }


@app.function(
    gpu=TARGET_GPU,
    image=cuda_image,
    timeout=600,
    volumes={"/cache": kernel_cache},
    include_source=True,
)
def evaluate_kernels_batch(payloads: list[dict]) -> list[dict]:
    """
    Evaluate multiple kernels in a single Modal call.

    Avoids per-kernel cold-start overhead during multi-turn training.
    Each payload follows the same format as evaluate_kernel().
    """
    results = []
    for payload in payloads:
        try:
            result = evaluate_kernel.local(payload)
            results.append(result)
        except Exception as e:
            results.append({
                "compiles": False, "correct": False, "verifier_msg": "",
                "runtime_ms": 0.0, "runtime_stats": {},
                "speedup_vs_orig": 0.0, "speedup_vs_dg": 0.0,
                "error": f"Batch eval exception: {str(e)[:500]}",
            })
    return results


if __name__ == "__main__":
    # Test Modal functions locally
    with app.run():
        print("Testing GPU features...")
        result = test_h100_features()
        print(f"GPU features: {result}")

        print("\nTesting baseline profiling...")
        baselines = profile_baselines()
        print(f"Baselines: {baselines}")
