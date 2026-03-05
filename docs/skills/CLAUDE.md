# docs/skills/ — Skill Document Index

## Files

| File | Lines | Content |
|------|-------|---------|
| `doublegraph_a100.md` | 325 | DoubleGraph engineering & replication guide for A100 |

## doublegraph_a100.md Section Index

| Line | Section |
|------|---------|
| 1 | Title & Overview |
| 11 | 1. Problem Statement: Cost of Genericism |
| 30 | 2. System Architecture: 4-Layer Drop-in |
| 47 | 3. Layer 1: Graph Abstraction (`compact_graph_t`) |
| 75 | 4. Layer 2: Resource Management (`CachePool`) |
| 106 | 5. Layer 3: 4-Way Dispatch Execution Matrix |
| 148 | 6. Layer 4: A100-Specific Kernel Deep Dive |
| 161 | 6.1 BFS: Dual-Frontier & Direction Optimization |
| 165 | 6.2 Louvain: 3-Tier Adaptive Dispatch |
| 177 | 6.3 PageRank: Fused SpMV & Warp-Level Reduction |
| 187 | 6.4 WCC: Parallel Union-Find & Host-Mapped Flags |
| 205 | 6.5 Triangle Count: DAG Orientation |
| 214 | 7. Cross-Architecture Divergence (A100 vs L4 vs A10G) |
| 224 | 8. Integration Layer & Build Pipeline |
| 249 | 9. Hard Constraints & Known Limitations |
| 292 | 10. Porting to Future Hardware (H100, B200) |
| 318 | 11. Transferable Engineering Strategies |

## When to Read

- **Graph kernel implementation** → Sec 6 (line 148)
- **Baseline timing data** → Sec 6 subsections (timing per algorithm)
- **Porting to H100/B200** → Sec 10 (line 292)
- **Build pipeline** → Sec 8 (line 224)

## Also See

- `/skill_a100.md` (root, 72 lines) — A100 hardware quick-reference
