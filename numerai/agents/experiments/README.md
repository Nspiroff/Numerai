# Numerai research ledger

This file is a navigation index. The linked experiment records remain the
canonical sources for protocols, metrics, artifact hashes, and terminal
decisions.

## Current program state

No Ender20-family model is approved for deployment. No upload, assignment,
staking, submission, or Numerai account mutation is authorized. Ender20
through Ender24 are closed research families; their frozen artifacts and
decision records must not be overwritten or reused as new runs.

## Latest terminal experiment

[Ender24 EMA seed stability](ender24_ema_seed_stability_v53/round1_execution_postmortem.md)
is closed at
`STOPPED_AT_ENDER24_ROUND1_EVALUATOR_PRECONDITION_FAILURE`. All four frozen
training components completed exactly once, but the sole evaluator invocation
stopped before truth or result parsing because it compared CRLF physical JSON
bytes against LF Git-blob fingerprints. No scientific decision was produced;
Round 2 is unauthorized. The immutable training artifacts are preserved for a
separately reviewed recovery gate, not for an Ender24 retry.

## Active source gate

[Ender25 Ender24 evaluation recovery](ender25_ender24_evaluation_recovery_v53/experiment.md)
is the source-only recovery gate for those four preserved runs. It introduces
a new evaluator stage and create-only decision namespace, while keeping the
old Ender24 evaluator and decision path closed. Its protocol binds physical
bytes independently from strict LF/CRLF-canonical JSON authority. No Ender25
manifest, artifact read, evaluation, score, Round-2 execution, deployment, or
account action is authorized by the source scaffold.

## Ender20-24 outcome chain

| Family | Terminal state | Key conclusion | Canonical record | Archival point |
| --- | --- | --- | --- | --- |
| Ender20 | `STOP_NO_ELIGIBLE_CANDIDATE` / not promotion eligible | K64/500k was the strongest architecture scout, but later stability, hybrid, seed, and auxiliary-target gates produced no eligible deployment candidate. | [Architecture](ender20_nn_architecture_v53/experiment.md), [seed stability](ender20_seed_ensemble_stability_v53/experiment.md), [hybrid stability](ender20_hybrid_stability_v53/experiment.md), [terminal checkpoint](ender20_aux_target_rank_ensemble_v53/CONTINUATION_CHECKPOINT.md) | `4669d42`, tag `numerai-ender20-terminal` |
| Ender21 | `NEGATIVE` confirmation terminal | Block-DRO remained positive and passed the absolute risk gates, but retained only 40.745% of discovery BMC against the required 60%. | [Confirmation postmortem](ender21_residual_stability_v53/confirmation_postmortem.md) | `bd0fe11`, tag `numerai-ender21-terminal` |
| Ender22 | Operationally invalid; no experiment decision | The half-life-52 run failed before its first fit on a 3.17 GiB allocation. The cohort was not scored and Round 2 was never authorized. | [Round-1 execution postmortem](ender22_temporal_retention_v53/round1_execution_postmortem.md) | `f9f45f6`, tag `numerai-ender22-terminal` |
| Ender23 | `NEGATIVE_SEED_INSTABILITY` | The memory repair worked and window-78 won Round 1. Two of three Round-2 realizations qualified, but the fixed ensemble failed its Sharpe and drawdown gates. | [Terminal postmortem](ender23_temporal_retention_v53/round2_terminal_postmortem.md) | `aff188b`, tag `numerai-ender23-terminal` |
| Ender24 | No decision; evaluator precondition failure | Four matched control/EMA runs finalized, but a CRLF/LF authority-fingerprint defect stopped the evaluator before metrics. Round 2 was not authorized. | [Round-1 execution postmortem](ender24_ema_seed_stability_v53/round1_execution_postmortem.md) | `d12f755`, tag `numerai-ender24-terminal` |

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
- Ender24 produced complete matched-seed artifacts but no scientific evidence:
  the evaluator failed before metric parsing because text-byte authority was
  not portable across LF and CRLF checkouts.

## Next admissible work

1. Do not retry Ender24 training or evaluation, populate its old decision path,
   infer an EMA winner from its stored results, or launch its Round 2.
2. Review the Ender25 evaluation-recovery source and its synthetic portability
   and custody regressions independently.
3. Only after that review, create a separate manifest-only seal for the exact
   Ender25 source set. Evaluation still requires later explicit launch
   authorization.
4. Treat the manifest as raw-byte custody while validating JSON authority by
   canonical semantics, with regressions for LF/CRLF equivalence and semantic
   mutation rejection.
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
