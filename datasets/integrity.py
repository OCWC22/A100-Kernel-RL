"""Dataset integrity verification — SHA-256 hashing and structural validation.

Run before training to catch data corruption or unexpected schema changes.
"""
from __future__ import annotations

import hashlib
import json
import os


def hash_file(path: str) -> str | None:
    """Compute SHA-256 hash of a file. Returns None if file doesn't exist."""
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_jsonl(path: str, required_keys: list[str] | None = None) -> dict:
    """Verify each line is valid JSON with optional required keys.

    Returns:
        Dict with line_count, errors list, and valid bool.
    """
    if not os.path.exists(path):
        return {"path": path, "exists": False, "line_count": 0, "errors": ["File not found"], "valid": False}

    errors = []
    line_count = 0
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if required_keys:
                    missing = [k for k in required_keys if k not in obj]
                    if missing:
                        errors.append(f"Line {i}: missing keys {missing}")
                line_count += 1
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON — {e}")

    return {
        "path": path,
        "exists": True,
        "line_count": line_count,
        "errors": errors[:20],  # cap error output
        "total_errors": len(errors),
        "valid": len(errors) == 0,
    }


def verify_curated_dataset(path: str = "datasets/curated_200.jsonl") -> dict:
    """Validate the curated 200-problem training dataset."""
    result = verify_jsonl(path, required_keys=["ops", "difficulty"])
    if result["valid"]:
        # Check difficulty distribution
        difficulties = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    difficulties.append(json.loads(line).get("difficulty", 0))
        result["difficulty_distribution"] = {
            d: difficulties.count(d) for d in sorted(set(difficulties))
        }
    result["sha256"] = hash_file(path)
    return result


def verify_baselines(path: str = "datasets/baselines.jsonl") -> dict:
    """Validate the pre-computed baselines dataset."""
    result = verify_jsonl(path, required_keys=["name", "baseline_eager_ms"])
    result["sha256"] = hash_file(path)
    return result


def main():
    """Run all integrity checks and print report."""
    print("Dataset Integrity Report")
    print("=" * 50)

    for name, check_fn in [
        ("Curated 200", verify_curated_dataset),
        ("Baselines", verify_baselines),
    ]:
        result = check_fn()
        status = "PASS" if result.get("valid") else "FAIL"
        print(f"\n{name}: {status}")
        print(f"  Path: {result.get('path', 'N/A')}")
        print(f"  Exists: {result.get('exists', False)}")
        print(f"  Lines: {result.get('line_count', 0)}")
        print(f"  SHA-256: {result.get('sha256', 'N/A')}")
        if result.get("errors"):
            for e in result["errors"][:5]:
                print(f"  ERROR: {e}")
        if result.get("difficulty_distribution"):
            print(f"  Difficulties: {result['difficulty_distribution']}")


if __name__ == "__main__":
    main()
