#!/usr/bin/env python3
"""Smoke test: proves core logic works without GPU/network.

Exit code 0 = all core logic verified.
Run: uv run python scripts/smoke_test.py
"""
import os
import sys

# Ensure project root is on sys.path when invoked as `python scripts/smoke_test.py`
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import traceback


def main():
    errors = []
    passed = 0

    # 1. Reward computation (4 discrete tiers)
    try:
        from openenv_env.reward import compute_reward
        assert compute_reward(False, False, 0, 0) == -1.0, "compile fail"
        assert compute_reward(True, False, 0, 0) == -1.0, "verify fail"
        assert compute_reward(True, True, 1.0, 0.9) == 1.0, "correct, no speedup"
        assert compute_reward(True, True, 1.10, 0.9) == 2.0, "beats eager"
        assert compute_reward(True, True, 1.10, 1.10) == 3.0, "beats both"
        print("PASS: reward.compute_reward (5 assertions)")
        passed += 1
    except Exception as e:
        errors.append(f"FAIL: reward - {e}")

    # 2. GPU registry (3 GPUs)
    try:
        from openenv_env.gpu_registry import get_gpu_spec
        a100 = get_gpu_spec("a100")
        assert a100["arch"] == "sm_80"
        assert a100["sms"] == 108
        assert a100["has_tma"] is False
        h100 = get_gpu_spec("h100")
        assert h100["arch"] == "sm_90a"
        assert h100["has_tma"] is True
        b200 = get_gpu_spec("b200")
        assert b200["arch"] == "sm_100a"
        print("PASS: gpu_registry (3 GPUs verified)")
        passed += 1
    except Exception as e:
        errors.append(f"FAIL: gpu_registry - {e}")

    # 3. Anti-hack (cu_flags + forbidden symbols)
    try:
        from openenv_env.anti_hack import extract_cu_flags, FORBIDDEN_SYMBOLS
        assert extract_cu_flags("") == []
        assert extract_cu_flags("// CU_FLAGS: --use_fast_math") == ["--use_fast_math"]
        assert extract_cu_flags("// CU_FLAGS: --maxrregcount=48") == ["--maxrregcount=48"]
        assert extract_cu_flags("// CU_FLAGS: --maxrregcount=256") == []  # out of range
        assert extract_cu_flags("// CU_FLAGS: --evil-flag") == []  # disallowed
        assert "torch" in FORBIDDEN_SYMBOLS
        assert "triton" in FORBIDDEN_SYMBOLS
        print("PASS: anti_hack (6 assertions)")
        passed += 1
    except Exception as e:
        errors.append(f"FAIL: anti_hack - {e}")

    # 4. Cache pool (LRU eviction)
    try:
        from openenv_env.cache_pool import GPUCachePool
        pool = GPUCachePool(max_entries=2)
        pool.get_or_create("a", lambda: 1)
        pool.get_or_create("b", lambda: 2)
        pool.get_or_create("c", lambda: 3)  # evicts "a"
        assert pool.get("a") is None, "evicted entry should be gone"
        assert pool.get("c") == 3
        assert len(pool) == 2
        print("PASS: cache_pool (LRU eviction)")
        passed += 1
    except Exception as e:
        errors.append(f"FAIL: cache_pool - {e}")

    # 5. Skill builder
    try:
        from openenv_env.skill_builder import build_skill_md
        md_a = build_skill_md("a100")
        assert "A100" in md_a or "sm_80" in md_a
        assert len(md_a) > 100
        md_h = build_skill_md("h100")
        assert "H100" in md_h or "TMA" in md_h
        print("PASS: skill_builder (a100 + h100)")
        passed += 1
    except Exception as e:
        errors.append(f"FAIL: skill_builder - {e}")

    # 6. Curriculum manager
    try:
        from training.curriculum import CurriculumManager
        cm = CurriculumManager()
        assert cm.phase_name == "single_ops"
        p = cm.get_problem()
        assert "prompt" in p
        s = cm.status()
        assert s["phase"] == "single_ops"
        assert s["total_phases"] == 4
        print("PASS: curriculum (4 phases)")
        passed += 1
    except Exception as e:
        errors.append(f"FAIL: curriculum - {e}")

    # 7. PAC verification (graph invariants)
    try:
        import networkx as nx
        from verification.pac_verify import verify_wcc, edges_to_csr

        edges = [(0, 1), (1, 2), (3, 4)]
        n = 5
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        labels = {}
        for comp in nx.connected_components(G):
            root = min(comp)
            for v in comp:
                labels[v] = root

        ok, msg = verify_wcc(labels, edges, n)
        assert ok, f"PAC verification should pass: {msg}"

        # Negative test: wrong labels should fail
        bad_labels = {v: 0 for v in range(n)}  # all in one component (wrong)
        ok2, _ = verify_wcc(bad_labels, edges, n)
        assert not ok2, "Wrong labels should fail"

        print("PASS: pac_verify (correct + incorrect labels)")
        passed += 1
    except Exception as e:
        errors.append(f"FAIL: pac_verify - {e}")

    # Summary
    total = passed + len(errors)
    print(f"\n{'='*50}")
    if errors:
        print(f"{len(errors)}/{total} FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print(f"All {passed} smoke tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
