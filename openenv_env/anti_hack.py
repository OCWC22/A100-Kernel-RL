"""Anti-reward-hacking utilities for kernel evaluation."""

from __future__ import annotations

import subprocess
from pathlib import Path

ALLOWED_CU_FLAGS = {
    "--use_fast_math",
    "--extra-device-vectorization",
    "--rdc=true",
}
ALLOWED_CU_FLAG_PREFIXES = ("--maxrregcount=",)

FORBIDDEN_SYMBOLS = [
    "torch",
    "at::Tensor",
    "c10::",
    "torch::autograd",
    "triton",
    "torch.compile",
    "torch.nn.functional",
]


def extract_cu_flags(cuda_code: str) -> list[str]:
    """Extract whitelisted ``// CU_FLAGS: ...`` tokens from CUDA source."""
    extracted: list[str] = []
    for line in cuda_code.splitlines():
        stripped = line.strip()
        if not stripped.startswith("// CU_FLAGS:"):
            continue

        for token in stripped.replace("// CU_FLAGS:", "").strip().split():
            if token in ALLOWED_CU_FLAGS:
                extracted.append(token)
                continue

            if any(token.startswith(prefix) for prefix in ALLOWED_CU_FLAG_PREFIXES):
                try:
                    value = int(token.split("=", 1)[1])
                except (IndexError, ValueError):
                    continue
                if 16 <= value <= 128:
                    extracted.append(token)

    # preserve order, drop duplicates
    return list(dict.fromkeys(extracted))


def scan_forbidden_symbols(so_path: str | Path) -> str | None:
    """Return a failure reason if forbidden dynamic symbols are detected."""
    so_path = str(so_path)
    try:
        proc = subprocess.run(
            ["nm", "-D", so_path],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        # Do not hard-fail if nm is unavailable; caller can decide fallback behavior.
        return None

    symbols = proc.stdout
    for forbidden in FORBIDDEN_SYMBOLS:
        if forbidden in symbols:
            return f"Forbidden symbol detected: {forbidden}"
    return None
