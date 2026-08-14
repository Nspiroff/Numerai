# Numerai research ledger

This file is a navigation index. The linked experiment records remain the
canonical sources for protocols, metrics, artifact hashes, and terminal
decisions.

## Current program state

No Ender20-family model is approved for deployment. No upload, assignment,
staking, submission, or Numerai account mutation is authorized. Ender20
through Ender26 are closed research families; their frozen artifacts and
decision records must not be overwritten or reused as new runs.

## Latest terminal experiment

[Ender26 Gaussian-rank benchmark residual](ender26_gaussian_rank_residual_v53/terminal_postmortem.md)
is closed at `ENDER26_NEGATIVE_NO_BMC_ALIGNMENT_GAIN`. Mean full and recent-40
BMC improved and benchmark correlation fell, but seed 1337 lost recent-40 BMC
and exceeded its matched drawdown allowance. The predeclared cross-seed gate
therefore rejected the procedure. Round 2 is unauthorized.

## Active research gate

[Ender27 tempered Gaussian-rank benchmark residual](ender27_tempered_gaussian_rank_residual_v53/experiment.md)
is an active **source-only** Round-1 gate. It tests one preregistered midpoint
target-residual blend (`lambda=0.5`) while holding the Ender26 control,
architecture, training procedure, matched seeds, cohort, and decision law
fixed. No manifest, data access, output reservation, training, scoring,
evaluation, Round 2, deployment, or account action is authorized.

## Ender20-26 outcome chain

| Family | Terminal state | Key conclusion | Canonical record | Archival point |
| --- | --- | --- | --- | --- |
| Ender20 | `STOP_NO_ELIGIBLE_CANDIDATE` / not promotion eligible | K64/500k was the strongest architecture scout, but later stability, hybrid, seed, and auxiliary-target gates produced no eligible deployment candidate. | [Architecture](ender20_nn_architecture_v53/experiment.md), [seed stability](ender20_seed_ensemble_stability_v53/experiment.md), [hybrid stability](ender20_hybrid_stability_v53/experiment.md), [terminal checkpoint](ender20_aux_target_rank_ensemble_v53/CONTINUATION_CHECKPOINT.md) | `4669d42`, tag `numerai-ender20-terminal` |
| Ender21 | `NEGATIVE` confirmation terminal | Block-DRO remained positive and passed the absolute risk gates, but retained only 40.745% of discovery BMC against the required 60%. | [Confirmation postmortem](ender21_residual_stability_v53/confirmation_postmortem.md) | `bd0fe11`, tag `numerai-ender21-terminal` |
| Ender22 | Operationally invalid; no experiment decision | The half-life-52 run failed before its first fit on a 3.17 GiB allocation. The cohort was not scored and Round 2 was never authorized. | [Round-1 execution postmortem](ender22_temporal_retention_v53/round1_execution_postmortem.md) | `f9f45f6`, tag `numerai-ender22-terminal` |
| Ender23 | `NEGATIVE_SEED_INSTABILITY` | The memory repair worked and window-78 won Round 1. Two of three Round-2 realizations qualified, but the fixed ensemble failed its Sharpe and drawdown gates. | [Terminal postmortem](ender23_temporal_retention_v53/round2_terminal_postmortem.md) | `aff188b`, tag `numerai-ender23-terminal` |
| Ender24 | No decision; evaluator precondition failure | Four matched control/EMA runs finalized, but a CRLF/LF authority-fingerprint defect stopped the evaluator before metrics. Round 2 was not authorized. | [Round-1 execution postmortem](ender24_ema_seed_stability_v53/round1_execution_postmortem.md) | `d12f755`, tag `numerai-ender24-terminal` |
| Ender25 | `ENDER25_NEGATIVE_NO_EMA_STABILITY_GAIN` | The repaired one-shot evaluator produced the missing scientific decision. EMA greatly compressed seed variance, but lost average full and recent BMC and worsened seed-1337 Sharpe/drawdown. | [Terminal postmortem](ender25_ender24_evaluation_recovery_v53/terminal_postmortem.md) | `9f6a08c`, tags `numerai-ender25-terminal` and `numerai-ender25-terminal-custody` |
| Ender26 | `ENDER26_NEGATIVE_NO_BMC_ALIGNMENT_GAIN` | Gaussian-rank target residualization improved aggregate full/recent BMC and decorrelated from the benchmark, but seed 1337 failed the matched recent-40 and drawdown guards. | [Terminal postmortem](ender26_gaussian_rank_residual_v53/terminal_postmortem.md) | tag `numerai-ender26-terminal` |

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
- Ender25 repaired only that authority boundary and recovered the frozen
  decision without retraining. EMA reduced seed dispersion, but the lower mean
  BMC and seed-1337 risk regression proved that variance compression alone was
  not a stronger modeling procedure.
- Ender26 directly improved mean benchmark-unique contribution and reduced
  benchmark correlation. The gain was nevertheless non-robust: seed 1337 lost
  recent-40 BMC and breached the matched drawdown guard, so aggregate gains did
  not justify promotion.

## Next admissible work

1. Do not retry Ender24, Ender25, or Ender26; replace cohort members; add or
   select seeds; loosen thresholds; or launch Round 2.
2. Review and merge the Ender27 source-only gate. Its sole new scientific
   variable is the fixed `lambda=0.5` target-residual midpoint; do not add a
   strength sweep, architecture, temporal window, seed, or threshold.
3. Only after source review may Ender27 receive a separate manifest-only seal.
   Training and evaluation still require later explicit user authorization.
4. Do not reuse consumed eras `0301`-`1021` for post-hoc model selection.
   Historical eras `1022`-`1230` are not a substitute confirmation cohort.
5. Keep deployment separate: it requires a successful, separately frozen
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
