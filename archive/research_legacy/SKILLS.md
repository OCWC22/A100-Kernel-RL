# SKILLS.md — Agent Playbook for DoubleGraph A100 RL Dataset

This file tells coding agents and engineers how to navigate and use the A100 RL dataset artifacts in this repo.

## Skill Name

`doublegraph-a100-dataset-navigation`

## When to Use This Skill

Use this skill when a task requires any of the following:
- Loading A100 kernel metadata into RL/evolution systems
- Building parser/validator code from dataset contracts
- Auditing kernel coverage, flags, variants, or category splits
- Preparing handoff packages for other coding agents

## Inputs

- Dataset contract: `docs/doublegraph_a100_dataset_formats.md`
- Full details: `docs/doublegraph_a100_dataset_full_details.md`
- Machine manifest: `docs/doublegraph_a100_manifest.jsonl`
- Spreadsheet manifest: `docs/doublegraph_a100_manifest.csv`
- Source tree: `cpp/src/aai/impl/a100/`
- Build routing/flags: `cpp/CMakeLists.txt`

## Outputs

Depending on the request, produce one or more:
- RL task records aligned to `KernelTaskRecord`
- RL observation records aligned to `RLEpisodeObservationRecord`
- Evo seed records aligned to `EvolutionSeedRecord`
- Validation reports (counts, uniqueness, missing sidecars, rdc requirements)
- Updated docs/manifests after kernel-level changes

## Core Workflow

1. Grounding
- Confirm artifact files exist and manifest row count is expected.

2. Contract Alignment
- Use schema/field definitions from `docs/doublegraph_a100_dataset_formats.md`.
- Do not invent alternate field names unless explicitly requested.

3. Metadata Join
- Join code and compile metadata by `kernel_id`.
- Preserve `flags` ordering from sidecar files.
- Preserve `requires_rdc` logic.

4. Validation
- Enforce count and integrity checks before handing off.
- Surface discrepancies as blocking issues.

5. Handoff
- Provide both machine-readable and human-readable outputs.
- Include file paths and exact assumptions.

## Hard Rules

- Use `kernel_id` as the primary key: `a100/<category>/<stem>`.
- Variant parsing precedence is fixed: `_seg_mask`, `_seg`, `_mask`, else `base`.
- Missing `.cu.flags` is valid and must be represented explicitly.
- `requires_rdc` must be true if flags contain `--rdc=true` or kernel uses cooperative groups.
- Keep full source fidelity; do not truncate kernel content in extraction pipelines.

## Common Agent Tasks

### Task A: Build RL training input table
- Load JSONL
- Select task-level fields
- Attach topology/context metadata
- Emit `KernelTaskRecord` rows

### Task B: Build Evo seed population
- Group by `category`
- Select baseline kernels per island
- Emit `EvolutionSeedRecord` rows

### Task C: Validate repository snapshot drift
- Re-scan `cpp/src/aai/impl/a100/*.cu`
- Compare count, ids, and sidecar presence against manifest
- Regenerate artifacts if drift detected

## Handoff Checklist

- Counts match source tree
- All `kernel_id` values unique
- Category and variant parsing validated
- Missing sidecars explicitly listed
- Manifest and docs cross-linked
- Consumer team has index link: `docs/RL_DATASET_INDEX.md`

## Quick Entry Point for Any Agent

Start at:
- `docs/RL_DATASET_INDEX.md`

Then follow:
- `docs/doublegraph_a100_dataset_formats.md`
- `docs/doublegraph_a100_dataset_full_details.md`
- `docs/doublegraph_a100_manifest.jsonl`

