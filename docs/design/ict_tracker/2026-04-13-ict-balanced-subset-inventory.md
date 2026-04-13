# ICT Balanced Subset Inventory

Date: `2026-04-13`

Report order note:

- this is a later same-day follow-up report
- it was created after the earlier compact continuity report to inventory all targets/runs and define the balanced evaluation subset

## Scope

This note inventories the current ICT runs and records the balanced multi-target subset used for the next continuity evaluation pass.

Artifacts for this note live under:

- [`docs/design/artifacts/ict_tracker_2026-04-13_balanced_eval/inventory`](/home/kara4/demo/acies-os/docs/design/artifacts/ict_tracker_2026-04-13_balanced_eval/inventory)

Primary CSVs:

- [`run_inventory.csv`](/home/kara4/demo/acies-os/docs/design/artifacts/ict_tracker_2026-04-13_balanced_eval/inventory/run_inventory.csv)
- [`balanced_subset.csv`](/home/kara4/demo/acies-os/docs/design/artifacts/ict_tracker_2026-04-13_balanced_eval/inventory/balanced_subset.csv)

## Dataset Inventory

Discovered ICT targets:

- `mustang`
- `miata`
- `gle350`
- `cx30`

Discovered runs:

- `run0` to `run7`

Important constraint:

- the current ICT corpus has exactly `2` runs per target, not `3`
- that means a `3 runs/target` balanced split is not achievable yet
- the maximum balanced multi-target subset is therefore `8` runs total: all usable runs

Target/run count summary:

| Label | Runs | Run ids |
| --- | ---: | --- |
| `mustang` | `2` | `0, 1` |
| `miata` | `2` | `2, 3` |
| `gle350` | `2` | `4, 5` |
| `cx30` | `2` | `6, 7` |

## Per-Run Summary

From [`run_inventory.csv`](/home/kara4/demo/acies-os/docs/design/artifacts/ict_tracker_2026-04-13_balanced_eval/inventory/run_inventory.csv):

| Run | Label | Split | Duration (s) | Usable loops | Median loop length (s) | Labeling usable | Loop extraction usable | Quality issue |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `0` | `mustang` | `train` | `899.0` | `16` | `49.0` | `yes` | `yes` | none |
| `1` | `mustang` | `test` | `2807.0` | `56` | `47.0` | `yes` | `yes` | none |
| `2` | `miata` | `train` | `899.0` | `13` | `59.0` | `yes` | `yes` | high ambiguous fraction |
| `3` | `miata` | `test` | `3010.0` | `49` | `58.0` | `yes` | `yes` | high ambiguous fraction; irregular loop lengths |
| `4` | `gle350` | `train` | `899.0` | `17` | `47.0` | `yes` | `yes` | none |
| `5` | `gle350` | `test` | `2678.0` | `60` | `43.0` | `yes` | `yes` | none |
| `6` | `cx30` | `train` | `899.0` | `16` | `50.0` | `yes` | `yes` | weaker runtime continuity |
| `7` | `cx30` | `test` | `2789.0` | `52` | `50.0` | `yes` | `yes` | weaker runtime continuity |

Interpretation:

- all `8` runs have usable labels and at least one extractable full loop segment
- long test runs (`1, 3, 5, 7`) naturally contribute many more candidate loop segments than the short train runs
- `miata` is the noisiest label in the inventory because both runs show elevated ambiguous-direction fraction, and `run3` also has more irregular loop lengths
- `cx30` is the weakest label under the current runtime continuity candidate on pooled templates, even though its loops are still extractable and usable

## Balanced Subset Proposal

The subset was ranked within each label by:

1. usable loop count, descending
2. loop-length coefficient of variation, ascending
3. runtime mean step, ascending
4. run id, ascending

From [`balanced_subset.csv`](/home/kara4/demo/acies-os/docs/design/artifacts/ict_tracker_2026-04-13_balanced_eval/inventory/balanced_subset.csv):

| Label | Selected runs | Notes |
| --- | --- | --- |
| `mustang` | `run1`, `run0` | only `2` usable runs available |
| `miata` | `run3`, `run2` | only `2` usable runs available |
| `gle350` | `run5`, `run4` | only `2` usable runs available |
| `cx30` | `run7`, `run6` | only `2` usable runs available |

Why each run stays in:

- every label falls short of the desired `3`, so excluding a usable run would make the split less balanced, not more
- all runs have stable enough loop extraction to support representative-loop continuity plots
- the longer runs provide many candidate loops, while the shorter train runs still matter because they are the only second sample for their label

Why the ranking still matters:

- it makes the selection rule explicit and machine-readable
- if more runs are added later, the same CSV generation logic can immediately cap the subset at the best `3` per target

## Recommendation

Use the balanced subset defined in [`balanced_subset.csv`](/home/kara4/demo/acies-os/docs/design/artifacts/ict_tracker_2026-04-13_balanced_eval/inventory/balanced_subset.csv) for the next continuity-first ICT evaluation.

Practical recommendation:

- for the current corpus, use all `8` usable runs
- do not keep evaluating only the earlier `mustang`/`gle350` compact view
- for the next real split improvement, collect at least one additional run per target or move to leave-one-run-out reporting, because the current dataset cannot support a `3-per-target` or stronger held-out target analysis yet
