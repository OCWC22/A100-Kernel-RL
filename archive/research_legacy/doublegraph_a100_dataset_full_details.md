# DoubleGraph A100 Dataset: Full Extraction and Agent Handoff

This file is an exhaustive, codebase-derived dataset handoff for other coding agents.

- Key index: `docs/RL_DATASET_INDEX.md`
- Agent playbook: `docs/SKILLS.md`

- Generated from: `cpp/src/aai/impl/a100/`
- Generated at: `2026-03-05 12:12:22 UTC`
- Total kernels: `192`
- Total sidecar flag files: `172`
- Missing sidecar flags: `20`

Companion machine-readable artifacts:
- `docs/doublegraph_a100_manifest.jsonl`
- `docs/doublegraph_a100_manifest.csv`

## Web-Verified Reference Links
These links were looked up via web search to anchor standards/flag semantics used by this dataset contract.

- JSON Schema Draft 2020-12: https://json-schema.org/draft/2020-12
- JSON Schema specification landing page: https://json-schema.org/specification
- JSON Lines format docs: https://jsonlines.org/
- NVCC compiler driver docs (current): https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
- NVCC `--rdc=true` option reference: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#cmdoption-nvcc-rdc
- CUDA best practices compiler switch overview: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

## Codebase Wiring (Authoritative in this repo)
- GPU target selection is required by CMake (`-DTARGET_GPU=<A100|L4|A10G>`).
- A100 impl root is selected via `AAI_TARGET_GPU_SUFFIX=a100` in `cpp/CMakeLists.txt`.
- Impl sources are selected by algorithm globs (`_AAI_IMPL_GLOBS_*`) and only `.cu` files are compiled.
- Sidecar flags come from `<kernel>.cu.flags` and are applied as per-source CUDA compile options.
- `--rdc=true` sidecar entries are stripped from per-source options and routed into `cugraph_aai_rdc` static lib with separable compilation enabled.

### Routed Algorithm IDs (`AAI_ROUTED_ALGORITHMS`)

`BFS, SSSP, K_HOP_NBRS, PAGERANK, HITS, BETWEENNESS_CENTRALITY, EIGENVECTOR_CENTRALITY, KATZ_CENTRALITY, LOUVAIN, LEIDEN, ECG, TRIANGLE_COUNT, K_TRUSS, EGONET, SPECTRAL_MODULARITY_MAXIMIZATION, ANALYZE_CLUSTERING_MODULARITY, ANALYZE_CLUSTERING_EDGE_CUT, ANALYZE_CLUSTERING_RATIO_CUT, WCC, SCC, CORE_NUMBER, K_CORE, COSINE, JACCARD, OVERLAP, SORENSEN, MST`

### Algorithm-to-Impl Glob Map (from `cpp/CMakeLists.txt`)

| algorithm_id | globs |
|---|---|
| `ANALYZE_CLUSTERING_EDGE_CUT` | `${_AAI_IMPL_DIR}/community/analyze_clustering_edge_cut*` |
| `ANALYZE_CLUSTERING_MODULARITY` | `${_AAI_IMPL_DIR}/community/analyze_clustering_modularity*` |
| `ANALYZE_CLUSTERING_RATIO_CUT` | `${_AAI_IMPL_DIR}/community/analyze_clustering_ratio_cut*` |
| `BETWEENNESS_CENTRALITY` | `${_AAI_IMPL_DIR}/centrality/betweenness_centrality* ; ${_AAI_IMPL_DIR}/centrality/edge_betweenness_centrality*` |
| `BFS` | `${_AAI_IMPL_DIR}/traversal/bfs*` |
| `CORE_NUMBER` | `${_AAI_IMPL_DIR}/cores/core_number*` |
| `COSINE` | `${_AAI_IMPL_DIR}/link_prediction/cosine*` |
| `ECG` | `${_AAI_IMPL_DIR}/community/ecg*` |
| `EGONET` | `${_AAI_IMPL_DIR}/community/extract_ego*` |
| `EIGENVECTOR_CENTRALITY` | `${_AAI_IMPL_DIR}/centrality/eigenvector_centrality*` |
| `HITS` | `${_AAI_IMPL_DIR}/link_analysis/hits*` |
| `JACCARD` | `${_AAI_IMPL_DIR}/link_prediction/jaccard*` |
| `KATZ_CENTRALITY` | `${_AAI_IMPL_DIR}/centrality/katz_centrality*` |
| `K_CORE` | `${_AAI_IMPL_DIR}/cores/k_core*` |
| `K_HOP_NBRS` | `${_AAI_IMPL_DIR}/traversal/k_hop_nbrs*` |
| `K_TRUSS` | `${_AAI_IMPL_DIR}/community/k_truss*` |
| `LEIDEN` | `${_AAI_IMPL_DIR}/community/leiden*` |
| `LOUVAIN` | `${_AAI_IMPL_DIR}/community/louvain*` |
| `MST` | `${_AAI_IMPL_DIR}/tree/mst*` |
| `OVERLAP` | `${_AAI_IMPL_DIR}/link_prediction/overlap*` |
| `PAGERANK` | `${_AAI_IMPL_DIR}/link_analysis/pagerank*` |
| `SCC` | `${_AAI_IMPL_DIR}/components/scc* ; ${_AAI_IMPL_DIR}/components/strongly_connected_components*` |
| `SORENSEN` | `${_AAI_IMPL_DIR}/link_prediction/sorensen*` |
| `SPECTRAL_MODULARITY_MAXIMIZATION` | `${_AAI_IMPL_DIR}/community/spectral_modularity_maximization*` |
| `SSSP` | `${_AAI_IMPL_DIR}/traversal/sssp*` |
| `TRIANGLE_COUNT` | `${_AAI_IMPL_DIR}/community/triangle_count*` |
| `WCC` | `${_AAI_IMPL_DIR}/components/wcc* ; ${_AAI_IMPL_DIR}/components/weakly_connected_components*` |

## Inventory Summary

### Category counts

| category | kernels | with_flags | missing_flags |
|---|---:|---:|---:|
| `centrality` | 32 | 29 | 3 |
| `community` | 42 | 40 | 2 |
| `components` | 6 | 6 | 0 |
| `cores` | 8 | 8 | 0 |
| `link_analysis` | 32 | 24 | 8 |
| `link_prediction` | 48 | 44 | 4 |
| `traversal` | 20 | 17 | 3 |
| `tree` | 4 | 4 | 0 |

### Variant counts

| variant | count |
|---|---:|
| `base` | 70 |
| `seg` | 70 |
| `mask` | 26 |
| `seg_mask` | 26 |

### Feature counts

| feature | count |
|---|---:|
| `requires_rdc` | 5 |
| `uses_cooperative_groups` | 5 |
| `uses_cache_pool` | 186 |
| `uses_atomic_ops` | 157 |
| `uses_cusparse` | 29 |

### Sidecar flag token frequencies

| flag token | count |
|---|---:|
| `--use_fast_math` | 172 |
| `--rdc=true` | 5 |
| `--extra-device-vectorization` | 4 |
| `--maxrregcount=40` | 2 |
| `--maxrregcount=64` | 2 |
| `-Xptxas` | 1 |
| `-dlcm=ca` | 1 |
| `--maxrregcount=48` | 1 |
| `--expt-relaxed-constexpr` | 1 |
| `--maxrregcount=56` | 1 |

## Missing Sidecar Flags (20)

- `cpp/src/aai/impl/a100/centrality/eigenvector_centrality_f64.cu`
- `cpp/src/aai/impl/a100/centrality/katz_centrality_f64_mask.cu`
- `cpp/src/aai/impl/a100/centrality/katz_centrality_f64_seg_mask.cu`
- `cpp/src/aai/impl/a100/community/louvain_f64.cu`
- `cpp/src/aai/impl/a100/community/louvain_f64_seg.cu`
- `cpp/src/aai/impl/a100/link_analysis/hits_f64_mask.cu`
- `cpp/src/aai/impl/a100/link_analysis/pagerank_f64_mask.cu`
- `cpp/src/aai/impl/a100/link_analysis/pagerank_f64_p64.cu`
- `cpp/src/aai/impl/a100/link_analysis/pagerank_f64_p64_mask.cu`
- `cpp/src/aai/impl/a100/link_analysis/pagerank_f64_p64_seg.cu`
- `cpp/src/aai/impl/a100/link_analysis/pagerank_f64_seg.cu`
- `cpp/src/aai/impl/a100/link_analysis/pagerank_mask.cu`
- `cpp/src/aai/impl/a100/link_analysis/pagerank_seg_mask.cu`
- `cpp/src/aai/impl/a100/link_prediction/cosine_f64.cu`
- `cpp/src/aai/impl/a100/link_prediction/jaccard_all_pairs_f64.cu`
- `cpp/src/aai/impl/a100/link_prediction/overlap_all_pairs_f64.cu`
- `cpp/src/aai/impl/a100/link_prediction/overlap_f64.cu`
- `cpp/src/aai/impl/a100/traversal/sssp_f64.cu`
- `cpp/src/aai/impl/a100/traversal/sssp_f64_mask.cu`
- `cpp/src/aai/impl/a100/traversal/sssp_f64_seg_mask.cu`

## Full Kernel Inventory (All 192)

Columns: `kernel_id`, `algorithm_name`, `variant`, `line_count`, `has_flags`, `requires_rdc`, `flags`

### `centrality`

| kernel_id | algorithm_name | variant | line_count | has_flags | requires_rdc | flags |
|---|---|---|---:|---:|---:|---|
| `a100/centrality/betweenness_centrality` | `betweenness_centrality` | `base` | 469 | true | false | `--use_fast_math` |
| `a100/centrality/betweenness_centrality_mask` | `betweenness_centrality` | `mask` | 668 | true | false | `--use_fast_math` |
| `a100/centrality/betweenness_centrality_seg` | `betweenness_centrality` | `seg` | 380 | true | false | `--use_fast_math` |
| `a100/centrality/betweenness_centrality_seg_mask` | `betweenness_centrality` | `seg_mask` | 382 | true | false | `--use_fast_math` |
| `a100/centrality/edge_betweenness_centrality` | `edge_betweenness_centrality` | `base` | 351 | true | false | `--use_fast_math` |
| `a100/centrality/edge_betweenness_centrality_mask` | `edge_betweenness_centrality` | `mask` | 469 | true | false | `--use_fast_math` |
| `a100/centrality/edge_betweenness_centrality_seg` | `edge_betweenness_centrality` | `seg` | 363 | true | false | `--use_fast_math` |
| `a100/centrality/edge_betweenness_centrality_seg_mask` | `edge_betweenness_centrality` | `seg_mask` | 440 | true | false | `--use_fast_math` |
| `a100/centrality/eigenvector_centrality` | `eigenvector_centrality` | `base` | 287 | true | false | `--use_fast_math` |
| `a100/centrality/eigenvector_centrality_f32` | `eigenvector_centrality_f32` | `base` | 317 | true | false | `--use_fast_math` |
| `a100/centrality/eigenvector_centrality_f32_mask` | `eigenvector_centrality_f32` | `mask` | 405 | true | false | `--use_fast_math` |
| `a100/centrality/eigenvector_centrality_f32_seg` | `eigenvector_centrality_f32` | `seg` | 392 | true | true | `--use_fast_math --maxrregcount=40 --rdc=true` |
| `a100/centrality/eigenvector_centrality_f32_seg_mask` | `eigenvector_centrality_f32` | `seg_mask` | 366 | true | false | `--use_fast_math` |
| `a100/centrality/eigenvector_centrality_f64` | `eigenvector_centrality_f64` | `base` | 237 | false | false | `(none)` |
| `a100/centrality/eigenvector_centrality_f64_mask` | `eigenvector_centrality_f64` | `mask` | 400 | true | false | `--use_fast_math` |
| `a100/centrality/eigenvector_centrality_f64_seg` | `eigenvector_centrality_f64` | `seg` | 312 | true | false | `--use_fast_math` |
| `a100/centrality/eigenvector_centrality_f64_seg_mask` | `eigenvector_centrality_f64` | `seg_mask` | 391 | true | true | `--use_fast_math --rdc=true` |
| `a100/centrality/eigenvector_centrality_mask` | `eigenvector_centrality` | `mask` | 414 | true | false | `--use_fast_math` |
| `a100/centrality/eigenvector_centrality_seg` | `eigenvector_centrality` | `seg` | 335 | true | false | `--use_fast_math` |
| `a100/centrality/eigenvector_centrality_seg_mask` | `eigenvector_centrality` | `seg_mask` | 386 | true | false | `--use_fast_math` |
| `a100/centrality/katz_centrality` | `katz_centrality` | `base` | 380 | true | false | `--use_fast_math` |
| `a100/centrality/katz_centrality_f32` | `katz_centrality_f32` | `base` | 296 | true | false | `--use_fast_math` |
| `a100/centrality/katz_centrality_f32_mask` | `katz_centrality_f32` | `mask` | 348 | true | false | `--use_fast_math` |
| `a100/centrality/katz_centrality_f32_seg` | `katz_centrality_f32` | `seg` | 331 | true | false | `--use_fast_math` |
| `a100/centrality/katz_centrality_f32_seg_mask` | `katz_centrality_f32` | `seg_mask` | 395 | true | false | `--use_fast_math` |
| `a100/centrality/katz_centrality_f64` | `katz_centrality_f64` | `base` | 281 | true | false | `--use_fast_math` |
| `a100/centrality/katz_centrality_f64_mask` | `katz_centrality_f64` | `mask` | 289 | false | false | `(none)` |
| `a100/centrality/katz_centrality_f64_seg` | `katz_centrality_f64` | `seg` | 248 | true | false | `--use_fast_math` |
| `a100/centrality/katz_centrality_f64_seg_mask` | `katz_centrality_f64` | `seg_mask` | 465 | false | false | `(none)` |
| `a100/centrality/katz_centrality_mask` | `katz_centrality` | `mask` | 446 | true | false | `--use_fast_math` |
| `a100/centrality/katz_centrality_seg` | `katz_centrality` | `seg` | 398 | true | false | `--use_fast_math` |
| `a100/centrality/katz_centrality_seg_mask` | `katz_centrality` | `seg_mask` | 471 | true | false | `--use_fast_math` |

### `community`

| kernel_id | algorithm_name | variant | line_count | has_flags | requires_rdc | flags |
|---|---|---|---:|---:|---:|---|
| `a100/community/analyze_clustering_edge_cut_f32` | `analyze_clustering_edge_cut_f32` | `base` | 295 | true | false | `--use_fast_math` |
| `a100/community/analyze_clustering_edge_cut_f32_seg` | `analyze_clustering_edge_cut_f32` | `seg` | 204 | true | false | `--use_fast_math` |
| `a100/community/analyze_clustering_edge_cut_f64` | `analyze_clustering_edge_cut_f64` | `base` | 118 | true | false | `--use_fast_math` |
| `a100/community/analyze_clustering_edge_cut_f64_seg` | `analyze_clustering_edge_cut_f64` | `seg` | 234 | true | false | `--use_fast_math` |
| `a100/community/analyze_clustering_modularity_f32` | `analyze_clustering_modularity_f32` | `base` | 945 | true | false | `--use_fast_math` |
| `a100/community/analyze_clustering_modularity_f32_seg` | `analyze_clustering_modularity_f32` | `seg` | 445 | true | false | `--use_fast_math` |
| `a100/community/analyze_clustering_modularity_f64` | `analyze_clustering_modularity_f64` | `base` | 242 | true | false | `--use_fast_math` |
| `a100/community/analyze_clustering_modularity_f64_seg` | `analyze_clustering_modularity_f64` | `seg` | 470 | true | false | `--use_fast_math` |
| `a100/community/analyze_clustering_ratio_cut_f32` | `analyze_clustering_ratio_cut_f32` | `base` | 504 | true | false | `--use_fast_math --extra-device-vectorization` |
| `a100/community/analyze_clustering_ratio_cut_f32_seg` | `analyze_clustering_ratio_cut_f32` | `seg` | 753 | true | false | `--use_fast_math` |
| `a100/community/analyze_clustering_ratio_cut_f64` | `analyze_clustering_ratio_cut_f64` | `base` | 422 | true | false | `--use_fast_math` |
| `a100/community/analyze_clustering_ratio_cut_f64_seg` | `analyze_clustering_ratio_cut_f64` | `seg` | 488 | true | false | `--use_fast_math` |
| `a100/community/ecg_f32` | `ecg_f32` | `base` | 340 | true | false | `--use_fast_math` |
| `a100/community/ecg_f32_seg` | `ecg_f32` | `seg` | 662 | true | false | `--use_fast_math --maxrregcount=40` |
| `a100/community/ecg_f64` | `ecg_f64` | `base` | 528 | true | false | `--use_fast_math` |
| `a100/community/ecg_f64_seg` | `ecg_f64` | `seg` | 509 | true | false | `--use_fast_math` |
| `a100/community/extract_ego` | `extract_ego` | `base` | 382 | true | false | `--use_fast_math` |
| `a100/community/extract_ego_seg` | `extract_ego` | `seg` | 945 | true | false | `--use_fast_math` |
| `a100/community/extract_ego_weighted_f32` | `extract_ego_weighted_f32` | `base` | 306 | true | false | `--use_fast_math` |
| `a100/community/extract_ego_weighted_f32_seg` | `extract_ego_weighted_f32` | `seg` | 496 | true | false | `--use_fast_math -Xptxas -dlcm=ca` |
| `a100/community/extract_ego_weighted_f64` | `extract_ego_weighted_f64` | `base` | 458 | true | false | `--use_fast_math` |
| `a100/community/extract_ego_weighted_f64_seg` | `extract_ego_weighted_f64` | `seg` | 510 | true | false | `--use_fast_math` |
| `a100/community/k_truss` | `k_truss` | `base` | 482 | true | false | `--use_fast_math` |
| `a100/community/k_truss_mask` | `k_truss` | `mask` | 448 | true | false | `--use_fast_math` |
| `a100/community/k_truss_seg` | `k_truss` | `seg` | 481 | true | false | `--use_fast_math` |
| `a100/community/k_truss_seg_mask` | `k_truss` | `seg_mask` | 376 | true | false | `--use_fast_math` |
| `a100/community/leiden_f32` | `leiden_f32` | `base` | 668 | true | false | `--use_fast_math --maxrregcount=48` |
| `a100/community/leiden_f32_seg` | `leiden_f32` | `seg` | 690 | true | false | `--use_fast_math` |
| `a100/community/leiden_f64` | `leiden_f64` | `base` | 745 | true | false | `--use_fast_math` |
| `a100/community/leiden_f64_seg` | `leiden_f64` | `seg` | 766 | true | false | `--use_fast_math` |
| `a100/community/louvain_f32` | `louvain_f32` | `base` | 626 | true | false | `--use_fast_math` |
| `a100/community/louvain_f32_seg` | `louvain_f32` | `seg` | 995 | true | false | `--use_fast_math` |
| `a100/community/louvain_f64` | `louvain_f64` | `base` | 1192 | false | false | `(none)` |
| `a100/community/louvain_f64_seg` | `louvain_f64` | `seg` | 1045 | false | false | `(none)` |
| `a100/community/spectral_modularity_maximization_f32` | `spectral_modularity_maximization_f32` | `base` | 998 | true | false | `--use_fast_math` |
| `a100/community/spectral_modularity_maximization_f32_seg` | `spectral_modularity_maximization_f32` | `seg` | 601 | true | false | `--use_fast_math` |
| `a100/community/spectral_modularity_maximization_f64` | `spectral_modularity_maximization_f64` | `base` | 496 | true | false | `--use_fast_math` |
| `a100/community/spectral_modularity_maximization_f64_seg` | `spectral_modularity_maximization_f64` | `seg` | 692 | true | false | `--use_fast_math` |
| `a100/community/triangle_count` | `triangle_count` | `base` | 379 | true | false | `--use_fast_math` |
| `a100/community/triangle_count_mask` | `triangle_count` | `mask` | 520 | true | false | `--use_fast_math` |
| `a100/community/triangle_count_seg` | `triangle_count` | `seg` | 649 | true | false | `--use_fast_math` |
| `a100/community/triangle_count_seg_mask` | `triangle_count` | `seg_mask` | 336 | true | false | `--use_fast_math --maxrregcount=64` |

### `components`

| kernel_id | algorithm_name | variant | line_count | has_flags | requires_rdc | flags |
|---|---|---|---:|---:|---:|---|
| `a100/components/strongly_connected_components` | `strongly_connected_components` | `base` | 870 | true | false | `--use_fast_math` |
| `a100/components/strongly_connected_components_seg` | `strongly_connected_components` | `seg` | 964 | true | false | `--use_fast_math` |
| `a100/components/weakly_connected_components` | `weakly_connected_components` | `base` | 218 | true | false | `--use_fast_math --extra-device-vectorization` |
| `a100/components/weakly_connected_components_mask` | `weakly_connected_components` | `mask` | 142 | true | false | `--use_fast_math` |
| `a100/components/weakly_connected_components_seg` | `weakly_connected_components` | `seg` | 292 | true | false | `--use_fast_math` |
| `a100/components/weakly_connected_components_seg_mask` | `weakly_connected_components` | `seg_mask` | 277 | true | false | `--use_fast_math` |

### `cores`

| kernel_id | algorithm_name | variant | line_count | has_flags | requires_rdc | flags |
|---|---|---|---:|---:|---:|---|
| `a100/cores/core_number` | `core_number` | `base` | 348 | true | true | `--use_fast_math --rdc=true` |
| `a100/cores/core_number_mask` | `core_number` | `mask` | 460 | true | false | `--use_fast_math` |
| `a100/cores/core_number_seg` | `core_number` | `seg` | 302 | true | true | `--use_fast_math --expt-relaxed-constexpr --rdc=true` |
| `a100/cores/core_number_seg_mask` | `core_number` | `seg_mask` | 456 | true | false | `--use_fast_math` |
| `a100/cores/k_core` | `k_core` | `base` | 427 | true | false | `--use_fast_math` |
| `a100/cores/k_core_mask` | `k_core` | `mask` | 401 | true | false | `--use_fast_math` |
| `a100/cores/k_core_seg` | `k_core` | `seg` | 514 | true | false | `--use_fast_math` |
| `a100/cores/k_core_seg_mask` | `k_core` | `seg_mask` | 256 | true | false | `--use_fast_math --extra-device-vectorization` |

### `link_analysis`

| kernel_id | algorithm_name | variant | line_count | has_flags | requires_rdc | flags |
|---|---|---|---:|---:|---:|---|
| `a100/link_analysis/hits` | `hits` | `base` | 426 | true | false | `--use_fast_math` |
| `a100/link_analysis/hits_f64` | `hits_f64` | `base` | 467 | true | false | `--use_fast_math` |
| `a100/link_analysis/hits_f64_mask` | `hits_f64` | `mask` | 326 | false | false | `(none)` |
| `a100/link_analysis/hits_f64_seg` | `hits_f64` | `seg` | 530 | true | false | `--use_fast_math` |
| `a100/link_analysis/hits_f64_seg_mask` | `hits_f64` | `seg_mask` | 452 | true | false | `--use_fast_math` |
| `a100/link_analysis/hits_mask` | `hits` | `mask` | 405 | true | false | `--use_fast_math` |
| `a100/link_analysis/hits_seg` | `hits` | `seg` | 520 | true | false | `--use_fast_math` |
| `a100/link_analysis/hits_seg_mask` | `hits` | `seg_mask` | 352 | true | false | `--use_fast_math` |
| `a100/link_analysis/pagerank` | `pagerank` | `base` | 314 | true | false | `--use_fast_math` |
| `a100/link_analysis/pagerank_f32` | `pagerank_f32` | `base` | 435 | true | false | `--use_fast_math` |
| `a100/link_analysis/pagerank_f32_mask` | `pagerank_f32` | `mask` | 412 | true | false | `--use_fast_math` |
| `a100/link_analysis/pagerank_f32_p32` | `pagerank_f32_p32` | `base` | 496 | true | false | `--use_fast_math` |
| `a100/link_analysis/pagerank_f32_p32_mask` | `pagerank_f32_p32` | `mask` | 457 | true | false | `--use_fast_math` |
| `a100/link_analysis/pagerank_f32_p32_seg` | `pagerank_f32_p32` | `seg` | 387 | true | false | `--use_fast_math` |
| `a100/link_analysis/pagerank_f32_p32_seg_mask` | `pagerank_f32_p32` | `seg_mask` | 560 | true | false | `--use_fast_math` |
| `a100/link_analysis/pagerank_f32_seg` | `pagerank_f32` | `seg` | 465 | true | false | `--use_fast_math` |
| `a100/link_analysis/pagerank_f32_seg_mask` | `pagerank_f32` | `seg_mask` | 432 | true | false | `--use_fast_math` |
| `a100/link_analysis/pagerank_f64` | `pagerank_f64` | `base` | 636 | true | false | `--use_fast_math` |
| `a100/link_analysis/pagerank_f64_mask` | `pagerank_f64` | `mask` | 440 | false | false | `(none)` |
| `a100/link_analysis/pagerank_f64_p64` | `pagerank_f64_p64` | `base` | 430 | false | false | `(none)` |
| `a100/link_analysis/pagerank_f64_p64_mask` | `pagerank_f64_p64` | `mask` | 590 | false | false | `(none)` |
| `a100/link_analysis/pagerank_f64_p64_seg` | `pagerank_f64_p64` | `seg` | 510 | false | false | `(none)` |
| `a100/link_analysis/pagerank_f64_p64_seg_mask` | `pagerank_f64_p64` | `seg_mask` | 635 | true | false | `--use_fast_math` |
| `a100/link_analysis/pagerank_f64_seg` | `pagerank_f64` | `seg` | 345 | false | false | `(none)` |
| `a100/link_analysis/pagerank_f64_seg_mask` | `pagerank_f64` | `seg_mask` | 340 | true | false | `--use_fast_math` |
| `a100/link_analysis/pagerank_mask` | `pagerank` | `mask` | 485 | false | false | `(none)` |
| `a100/link_analysis/pagerank_p32` | `pagerank_p32` | `base` | 529 | true | false | `--use_fast_math` |
| `a100/link_analysis/pagerank_p32_mask` | `pagerank_p32` | `mask` | 464 | true | false | `--use_fast_math` |
| `a100/link_analysis/pagerank_p32_seg` | `pagerank_p32` | `seg` | 543 | true | false | `--use_fast_math` |
| `a100/link_analysis/pagerank_p32_seg_mask` | `pagerank_p32` | `seg_mask` | 462 | true | false | `--use_fast_math` |
| `a100/link_analysis/pagerank_seg` | `pagerank` | `seg` | 376 | true | false | `--use_fast_math` |
| `a100/link_analysis/pagerank_seg_mask` | `pagerank` | `seg_mask` | 554 | false | false | `(none)` |

### `link_prediction`

| kernel_id | algorithm_name | variant | line_count | has_flags | requires_rdc | flags |
|---|---|---|---:|---:|---:|---|
| `a100/link_prediction/cosine` | `cosine` | `base` | 229 | true | false | `--use_fast_math` |
| `a100/link_prediction/cosine_all_pairs` | `cosine_all_pairs` | `base` | 721 | true | false | `--use_fast_math` |
| `a100/link_prediction/cosine_all_pairs_f32` | `cosine_all_pairs_f32` | `base` | 528 | true | false | `--use_fast_math` |
| `a100/link_prediction/cosine_all_pairs_f32_seg` | `cosine_all_pairs_f32` | `seg` | 713 | true | false | `--use_fast_math` |
| `a100/link_prediction/cosine_all_pairs_f64` | `cosine_all_pairs_f64` | `base` | 645 | true | false | `--use_fast_math` |
| `a100/link_prediction/cosine_all_pairs_f64_seg` | `cosine_all_pairs_f64` | `seg` | 765 | true | false | `--use_fast_math --extra-device-vectorization` |
| `a100/link_prediction/cosine_all_pairs_seg` | `cosine_all_pairs` | `seg` | 539 | true | false | `--use_fast_math` |
| `a100/link_prediction/cosine_f32` | `cosine_f32` | `base` | 167 | true | false | `--use_fast_math` |
| `a100/link_prediction/cosine_f32_seg` | `cosine_f32` | `seg` | 145 | true | false | `--use_fast_math` |
| `a100/link_prediction/cosine_f64` | `cosine_f64` | `base` | 146 | false | false | `(none)` |
| `a100/link_prediction/cosine_f64_seg` | `cosine_f64` | `seg` | 206 | true | false | `--use_fast_math` |
| `a100/link_prediction/cosine_seg` | `cosine` | `seg` | 125 | true | false | `--use_fast_math` |
| `a100/link_prediction/jaccard` | `jaccard` | `base` | 350 | true | false | `--use_fast_math` |
| `a100/link_prediction/jaccard_all_pairs` | `jaccard_all_pairs` | `base` | 428 | true | false | `--use_fast_math` |
| `a100/link_prediction/jaccard_all_pairs_f32` | `jaccard_all_pairs_f32` | `base` | 611 | true | false | `--use_fast_math` |
| `a100/link_prediction/jaccard_all_pairs_f32_seg` | `jaccard_all_pairs_f32` | `seg` | 465 | true | false | `--use_fast_math` |
| `a100/link_prediction/jaccard_all_pairs_f64` | `jaccard_all_pairs_f64` | `base` | 633 | false | false | `(none)` |
| `a100/link_prediction/jaccard_all_pairs_f64_seg` | `jaccard_all_pairs_f64` | `seg` | 612 | true | false | `--use_fast_math --maxrregcount=56` |
| `a100/link_prediction/jaccard_all_pairs_seg` | `jaccard_all_pairs` | `seg` | 638 | true | false | `--use_fast_math` |
| `a100/link_prediction/jaccard_f32` | `jaccard_f32` | `base` | 277 | true | false | `--use_fast_math` |
| `a100/link_prediction/jaccard_f32_seg` | `jaccard_f32` | `seg` | 278 | true | false | `--use_fast_math` |
| `a100/link_prediction/jaccard_f64` | `jaccard_f64` | `base` | 170 | true | false | `--use_fast_math` |
| `a100/link_prediction/jaccard_f64_seg` | `jaccard_f64` | `seg` | 237 | true | false | `--use_fast_math` |
| `a100/link_prediction/jaccard_seg` | `jaccard` | `seg` | 301 | true | false | `--use_fast_math` |
| `a100/link_prediction/overlap` | `overlap` | `base` | 303 | true | false | `--use_fast_math` |
| `a100/link_prediction/overlap_all_pairs` | `overlap_all_pairs` | `base` | 420 | true | false | `--use_fast_math` |
| `a100/link_prediction/overlap_all_pairs_f32` | `overlap_all_pairs_f32` | `base` | 617 | true | false | `--use_fast_math` |
| `a100/link_prediction/overlap_all_pairs_f32_seg` | `overlap_all_pairs_f32` | `seg` | 449 | true | false | `--use_fast_math` |
| `a100/link_prediction/overlap_all_pairs_f64` | `overlap_all_pairs_f64` | `base` | 507 | false | false | `(none)` |
| `a100/link_prediction/overlap_all_pairs_f64_seg` | `overlap_all_pairs_f64` | `seg` | 470 | true | false | `--use_fast_math` |
| `a100/link_prediction/overlap_all_pairs_seg` | `overlap_all_pairs` | `seg` | 655 | true | false | `--use_fast_math` |
| `a100/link_prediction/overlap_f32` | `overlap_f32` | `base` | 192 | true | false | `--use_fast_math` |
| `a100/link_prediction/overlap_f32_seg` | `overlap_f32` | `seg` | 210 | true | false | `--use_fast_math` |
| `a100/link_prediction/overlap_f64` | `overlap_f64` | `base` | 237 | false | false | `(none)` |
| `a100/link_prediction/overlap_f64_seg` | `overlap_f64` | `seg` | 204 | true | false | `--use_fast_math` |
| `a100/link_prediction/overlap_seg` | `overlap` | `seg` | 339 | true | false | `--use_fast_math` |
| `a100/link_prediction/sorensen` | `sorensen` | `base` | 228 | true | false | `--use_fast_math` |
| `a100/link_prediction/sorensen_all_pairs` | `sorensen_all_pairs` | `base` | 645 | true | false | `--use_fast_math` |
| `a100/link_prediction/sorensen_all_pairs_f32` | `sorensen_all_pairs_f32` | `base` | 453 | true | false | `--use_fast_math` |
| `a100/link_prediction/sorensen_all_pairs_f32_seg` | `sorensen_all_pairs_f32` | `seg` | 589 | true | false | `--use_fast_math --maxrregcount=64` |
| `a100/link_prediction/sorensen_all_pairs_f64` | `sorensen_all_pairs_f64` | `base` | 442 | true | false | `--use_fast_math` |
| `a100/link_prediction/sorensen_all_pairs_f64_seg` | `sorensen_all_pairs_f64` | `seg` | 648 | true | false | `--use_fast_math` |
| `a100/link_prediction/sorensen_all_pairs_seg` | `sorensen_all_pairs` | `seg` | 476 | true | false | `--use_fast_math` |
| `a100/link_prediction/sorensen_f32` | `sorensen_f32` | `base` | 185 | true | false | `--use_fast_math` |
| `a100/link_prediction/sorensen_f32_seg` | `sorensen_f32` | `seg` | 221 | true | false | `--use_fast_math` |
| `a100/link_prediction/sorensen_f64` | `sorensen_f64` | `base` | 164 | true | false | `--use_fast_math` |
| `a100/link_prediction/sorensen_f64_seg` | `sorensen_f64` | `seg` | 170 | true | false | `--use_fast_math` |
| `a100/link_prediction/sorensen_seg` | `sorensen` | `seg` | 266 | true | false | `--use_fast_math` |

### `traversal`

| kernel_id | algorithm_name | variant | line_count | has_flags | requires_rdc | flags |
|---|---|---|---:|---:|---:|---|
| `a100/traversal/bfs` | `bfs` | `base` | 459 | true | false | `--use_fast_math` |
| `a100/traversal/bfs_direction_optimizing` | `bfs_direction_optimizing` | `base` | 483 | true | false | `--use_fast_math` |
| `a100/traversal/bfs_direction_optimizing_mask` | `bfs_direction_optimizing` | `mask` | 417 | true | false | `--use_fast_math` |
| `a100/traversal/bfs_direction_optimizing_seg` | `bfs_direction_optimizing` | `seg` | 373 | true | false | `--use_fast_math` |
| `a100/traversal/bfs_direction_optimizing_seg_mask` | `bfs_direction_optimizing` | `seg_mask` | 413 | true | false | `--use_fast_math` |
| `a100/traversal/bfs_mask` | `bfs` | `mask` | 460 | true | true | `--use_fast_math --rdc=true` |
| `a100/traversal/bfs_seg` | `bfs` | `seg` | 199 | true | false | `--use_fast_math` |
| `a100/traversal/bfs_seg_mask` | `bfs` | `seg_mask` | 469 | true | false | `--use_fast_math` |
| `a100/traversal/k_hop_nbrs` | `k_hop_nbrs` | `base` | 359 | true | false | `--use_fast_math` |
| `a100/traversal/k_hop_nbrs_mask` | `k_hop_nbrs` | `mask` | 354 | true | false | `--use_fast_math` |
| `a100/traversal/k_hop_nbrs_seg` | `k_hop_nbrs` | `seg` | 851 | true | false | `--use_fast_math` |
| `a100/traversal/k_hop_nbrs_seg_mask` | `k_hop_nbrs` | `seg_mask` | 519 | true | false | `--use_fast_math` |
| `a100/traversal/sssp_f32` | `sssp_f32` | `base` | 306 | true | false | `--use_fast_math` |
| `a100/traversal/sssp_f32_mask` | `sssp_f32` | `mask` | 368 | true | false | `--use_fast_math` |
| `a100/traversal/sssp_f32_seg` | `sssp_f32` | `seg` | 289 | true | false | `--use_fast_math` |
| `a100/traversal/sssp_f32_seg_mask` | `sssp_f32` | `seg_mask` | 318 | true | false | `--use_fast_math` |
| `a100/traversal/sssp_f64` | `sssp_f64` | `base` | 326 | false | false | `(none)` |
| `a100/traversal/sssp_f64_mask` | `sssp_f64` | `mask` | 331 | false | false | `(none)` |
| `a100/traversal/sssp_f64_seg` | `sssp_f64` | `seg` | 381 | true | false | `--use_fast_math` |
| `a100/traversal/sssp_f64_seg_mask` | `sssp_f64` | `seg_mask` | 330 | false | false | `(none)` |

### `tree`

| kernel_id | algorithm_name | variant | line_count | has_flags | requires_rdc | flags |
|---|---|---|---:|---:|---:|---|
| `a100/tree/mst_f32` | `mst_f32` | `base` | 302 | true | false | `--use_fast_math` |
| `a100/tree/mst_f32_seg` | `mst_f32` | `seg` | 399 | true | false | `--use_fast_math` |
| `a100/tree/mst_f64` | `mst_f64` | `base` | 385 | true | false | `--use_fast_math` |
| `a100/tree/mst_f64_seg` | `mst_f64` | `seg` | 376 | true | false | `--use_fast_math` |

## Reproducible Extraction Command

Use the generated JSONL as source-of-truth for downstream agents. If you need to regenerate:

```bash
python3 scripts/extract_a100_manifest.py  # if ported into your other repo
```

In this repo, generation was done from direct filesystem traversal + CMake parsing at snapshot time.
