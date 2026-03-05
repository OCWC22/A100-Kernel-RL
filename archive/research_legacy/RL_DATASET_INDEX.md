# RL Dataset Index: DoubleGraph A100 Kernels

This is the key navigation index for engineers and coding agents using the A100 kernel dataset in SDLC workflows.

## Purpose

Use this index to quickly find:
- Source-of-truth dataset artifacts
- Dataset schema contracts
- Full per-kernel inventory and metadata
- Safe workflows for ingestion, validation, and extension

## Canonical Artifacts

| file | purpose | use when |
|---|---|---|
| `docs/doublegraph_a100_dataset_formats.md` | Contract-level schema + examples + validation rules | Building parsers, RL interfaces, eval adapters |
| `docs/doublegraph_a100_dataset_full_details.md` | Exhaustive codebase-derived handoff (all 192 kernels) | Reviewing all kernels, flags, routing, category coverage |
| `docs/doublegraph_a100_manifest.jsonl` | Machine-readable per-kernel records | Programmatic ingestion by RL/evolution agents |
| `docs/doublegraph_a100_manifest.csv` | Spreadsheet-friendly inventory | Human review, PM/eng reporting, quick filtering |
| `docs/SKILLS.md` | Agent playbook for navigating and using this dataset | Assigning coding agents to tasks |

## Read Order

1. `docs/RL_DATASET_INDEX.md` (this file)
2. `docs/doublegraph_a100_dataset_formats.md`
3. `docs/doublegraph_a100_dataset_full_details.md`
4. `docs/doublegraph_a100_manifest.jsonl`

## Snapshot Facts (Current Repo)

- Kernel root: `cpp/src/aai/impl/a100/`
- Total kernels: `192`
- Kernels with sidecar flags: `172`
- Kernels missing sidecar flags: `20`
- Categories: `centrality`, `community`, `components`, `cores`, `link_analysis`, `link_prediction`, `traversal`, `tree`

## Fast Navigation Tasks

### Find one kernel in all artifacts

Use `a100/<category>/<stem>` as `kernel_id`.

Example kernel id:
- `a100/traversal/bfs`

Where to look:
- Source code path in JSONL field `source_path`
- Compile flags in JSONL fields `flags` and `requires_rdc`
- High-level details in full details table

### Build RL task payloads

Required source fields (from JSONL):
- `kernel_id`
- `category`
- `algorithm_name`
- `variant`
- `source_path`
- `source_sha256`
- `flags`
- `requires_rdc`

Then map into `KernelTaskRecord` and `RLEpisodeObservationRecord` per `docs/doublegraph_a100_dataset_formats.md`.

### Validate ingestion correctness

Minimum checks:
- Record count equals discovered `.cu` count (`192` for this snapshot)
- `kernel_id` uniqueness
- Category-path consistency
- Variant parser correctness (`base/seg/mask/seg_mask`)
- Missing `.cu.flags` handled explicitly

## SDLC Usage Pattern

1. Discovery
- Read `docs/doublegraph_a100_dataset_formats.md` and `docs/SKILLS.md`.

2. Ingestion
- Load `docs/doublegraph_a100_manifest.jsonl` into pipeline tables.

3. Validation
- Run schema + integrity checks from the format doc.

4. Training/Evolution
- Generate RL task/observation payloads.
- Seed AdaEvolve/EvoX from category-aligned kernel groups.

5. Change Management
- If A100 kernel files change, regenerate manifest and re-run validation.

## Ownership and Source of Truth

- Source code truth: `cpp/src/aai/impl/a100/*.cu`
- Build/flag truth: `cpp/CMakeLists.txt` + `*.cu.flags`
- Dataset contract truth: `docs/doublegraph_a100_dataset_formats.md`
- Agent handoff truth: `docs/doublegraph_a100_dataset_full_details.md` + JSONL manifest

