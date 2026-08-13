# Numerai research ledger

This file is a navigation index. The linked experiment records remain the
canonical sources for protocols, metrics, artifact hashes, and terminal
decisions.

## Current program state

No Ender20-family model is approved for deployment. No upload, assignment,
staking, submission, or Numerai account mutation is authorized. Ender20,
Ender21, Ender22, and Ender23 are closed research families; their frozen
artifacts and decision records must not be overwritten or reused as new runs.

## Active source-only design

[Ender24 EMA seed stability](ender24_ema_seed_stability_v53/experiment.md) is
the only active research design. It tests one fixed EMA weight-stabilization
procedure against matched model seeds. Round 1 is source-reviewed and sealed:
the [mechanical-activity receipt](ender24_ema_seed_stability_v53/protocol/mechanical_activity_receipt.json)
records 12/12 passing proofs, and the
[source manifest](ender24_ema_seed_stability_v53/source_manifest_round1.json)
binds the exact 31-file source/evidence set, runtime, and two discovery-input
receipts. The immutable Git chain is source merge `aebc577`, receipt-only
`a2bfe0f`, manifest-only `5a1a75d`, and seal merge `789a91f`.

Current state: `ROUND1_SEALED_AWAITING_EXPLICIT_LAUNCH_AUTHORITY`. No Ender24
training or scoring has occurred. The next admissible action is a separate
pre-launch audit and explicit launch decision for exactly four one-shot
Round-1 runs. The seal itself grants no data access, output reservation,
training, scoring, confirmation, deployment, or account authority.

## Ender20-23 outcome chain

| Family | Terminal state | Key conclusion | Canonical record | Archival point |
| --- | --- | --- | --- | --- |
| Ender20 | `STOP_NO_ELIGIBLE_CANDIDATE` / not promotion eligible | K64/500k was the strongest architecture scout, but later stability, hybrid, seed, and auxiliary-target gates produced no eligible deployment candidate. | [Architecture](ender20_nn_architecture_v53/experiment.md), [seed stability](ender20_seed_ensemble_stability_v53/experiment.md), [hybrid stability](ender20_hybrid_stability_v53/experiment.md), [terminal checkpoint](ender20_aux_target_rank_ensemble_v53/CONTINUATION_CHECKPOINT.md) | `4669d42`, tag `numerai-ender20-terminal` |
| Ender21 | `NEGATIVE` confirmation terminal | Block-DRO remained positive and passed the absolute risk gates, but retained only 40.745% of discovery BMC against the required 60%. | [Confirmation postmortem](ender21_residual_stability_v53/confirmation_postmortem.md) | `bd0fe11`, tag `numerai-ender21-terminal` |
| Ender22 | Operationally invalid; no experiment decision | The half-life-52 run failed before its first fit on a 3.17 GiB allocation. The cohort was not scored and Round 2 was never authorized. | [Round-1 execution postmortem](ender22_temporal_retention_v53/round1_execution_postmortem.md) | `f9f45f6`, tag `numerai-ender22-terminal` |
| Ender23 | `NEGATIVE_SEED_INSTABILITY` | The memory repair worked and window-78 won Round 1. Two of three Round-2 realizations qualified, but the fixed ensemble failed its Sharpe and drawdown gates. | [Terminal postmortem](ender23_temporal_retention_v53/round2_terminal_postmortem.md) | `aff188b`, tag `numerai-ender23-terminal` |

## What the sequence established

- Residual Block-DRO TabM produced real positive benchmark contribution, but
  Ender21 showed that its discovery-period magnitude did not retain strongly
  enough on the one authorized historical confirmation cohort.
- Ender22's outcome was an infrastructure failure, not model evidence.
- Ender23 repaired that infrastructure without changing the frozen research
  question. Recency-window training improved recent discovery BMC, but the
  improvement was not stable enough across the predeclared replication gate.
- Ender23's sample-seed replication was mechanically inactive because no
  window-78 training fold reached the 500,000-row sampling cap. It is not
  independent support for the selected procedure.

## Next admissible work

1. Do not rescue Ender23 with another seed, neighboring window, blend weight,
   threshold change, or reuse of its output paths.
2. If more research is authorized, start a newly named family from a genuinely
   new hypothesis aimed at reducing model-seed variance, not another temporal
   retention sweep.
3. Freeze the protocol, data authority, candidate set, evaluation boundary,
   replication design, and stopping rule before any score is read.
4. Prove every replication axis is mechanically active for every candidate
   before launch. Do not use sample seed as a replicate when sampling is not
   actually triggered.
5. Do not reuse consumed eras `0865`-`1021` for model selection. Historical
   eras `1022`-`1230` are not a substitute confirmation cohort.
6. Keep deployment separate: it requires a successful, separately frozen
   prospective validation and explicit user authorization for any account
   action.

## Repository hygiene

- One directory represents one line of inquiry. Its canonical terminal
  decision and postmortem stay inside that directory.
- This ledger links outcomes; it does not duplicate full metrics, receipts, or
  source manifests.
- Large datasets, ignored predictions, logs, model bundles, and protected GPU
  build residue remain outside normal Git history unless a frozen protocol
  explicitly requires a small evidence artifact.
- Preserve the archival commits and tags above. Do not squash, rebase, or
  rewrite the research chain because manifests and receipts bind historical
  commit identities.
- A future family must use new output and receipt paths; prior terminal
  artifacts are immutable evidence, not reusable workspace.
