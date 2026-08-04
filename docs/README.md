# v2e Documentation Map

Every document below has **one** job. If you are about to add a section, find
the file whose job it is; do not restate it somewhere more convenient.

## Ownership contract

| Question | Owner | Contains | Must not contain |
|---|---|---|---|
| How does the model work? | [`core_model_mapping.md`](core_model_mapping.md) | Paper math mapped to searchable `KEY[...]` anchors | Status, defects, tasks |
| What do the opt-in extensions do? | [`README_error_models_extensions.md`](README_error_models_extensions.md) | Flags, formulas, semantics, reference classification | Task lists, validation results |
| What works right now? | [`repo_capabilities_status.md`](repo_capabilities_status.md) | Capability table, validation snapshot, integration map | Defect explanations, task lists |
| What is wrong, and why? | [`findings/`](findings/) | One durable entry per defect: issue, why, evidence, fix | Checkboxes, ordering, scheduling |
| What are we going to do, in what order? | [`developments/`](developments/) | Action items with `- [ ]` / `- [x]`, gates, sequencing | Defect explanations (cite finding IDs) |
| How do I work on this repo? | [`development_core.md`](development_core.md) | Repo map, invariants, benchmark and test workflow | Status, plans, defects |
| What did we conclude on date X? | [`reports/`](reports/) | Frozen dated snapshots, never edited after archiving | Anything expected to stay current |

The rule that makes this work: **an action item lives in exactly one plan, and a
defect is explained in exactly one finding.** Everything else cites an
identifier instead of copying the text.

## Reference

- [`core_model_mapping.md`](core_model_mapping.md) — core v2e event model,
  paper equations to code anchors.
- [`README_error_models_extensions.md`](README_error_models_extensions.md) —
  IEBCS- and V2CE-inspired opt-in mechanisms, their CLI surface, and how far
  each is from its reference.
- [`../README_comparative_benchmark.md`](../README_comparative_benchmark.md) —
  comparative benchmark suite, metrics, and reproducibility checklist.

## Current state

- [`repo_capabilities_status.md`](repo_capabilities_status.md) — what the
  repository can do today and how far it has been validated.

## Findings

- [`findings/`](findings/) — the defect register. Entries are identified as
  `CORE`, `LIFE`, `IEBCS`, `V2CE`, `STREAM`, `TOOL` plus a number, and those
  identifiers are stable. Start at [`findings/README.md`](findings/README.md)
  for the full summary table.

## Plans

Ordering between plans: correctness first, then V2CE modelling, then
acceleration. Each downstream plan states its own prerequisite gate.

1. [`developments/consolidation_staged_plan.md`](developments/consolidation_staged_plan.md)
   — correctness and merge readiness.
2. [`developments/v2ce_timing_staged_plan.md`](developments/v2ce_timing_staged_plan.md)
   — per-pixel analytic timestamp inference.
3. [`developments/performance_optimization_opportunities.md`](developments/performance_optimization_opportunities.md)
   — shared C++/CUDA backend and parity gates.

## Contributor guide

- [`development_core.md`](development_core.md) — repository map, correctness
  invariants, hotspots, benchmarking and testing workflow.
- [`../v2ecore/README.md`](../v2ecore/README.md) — module-level orientation for
  `v2ecore/` itself.

## Archived reports

Dated snapshots. They are deliberately **not** updated: their test counts, file
paths, and conclusions describe the tree as it was on their status date. Read
them for provenance, not for current truth.

- [`reports/implementation_review_2026-07-28.md`](reports/implementation_review_2026-07-28.md)
- [`reports/full_review_report_2026-07-06.md`](reports/full_review_report_2026-07-06.md)
- [`reports/implementation_review_report_2026-07-04.md`](reports/implementation_review_report_2026-07-04.md)
- [`reports/optimization_pass_2026-02-15.md`](reports/optimization_pass_2026-02-15.md)

## Conventions

- **File names**: lowercase `snake_case.md`, except `README.md` for directory
  landing pages. Archived reports carry a `_YYYY-MM-DD` suffix.
- **Directory**: `docs/` only. The historical `doc/` tree no longer exists;
  references to it in archived reports are frozen historical text.
- **Anchors**: link to implementation with `KEY[...]` anchor IDs, never copied
  line numbers.

  ```bash
  rg -n "KEY\[[^]]+\]" v2ecore/emulator.py v2ecore/emulator_utils.py
  ```

- **"Stage"**: reserved for the numbered phases inside a single plan. Model
  blocks use the letters in `core_model_mapping.md`; IEBCS feature batches are
  described by mechanism name, not by stage number.
- **Equivalence vocabulary**: use exactly one of the three levels defined in
  [`README_error_models_extensions.md`](README_error_models_extensions.md) —
  *implemented*, *component-aligned*, *output-equivalent*. No extension
  currently qualifies as output-equivalent.
