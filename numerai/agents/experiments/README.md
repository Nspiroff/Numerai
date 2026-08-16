# Numerai research ledger

This file is the navigation index for the governed Numerai research program.
The linked experiment records remain the canonical sources for protocols,
metrics, artifact hashes, custody rules, and terminal decisions.

## Current program state

No Ender-family model is approved for deployment. No upload, assignment,
staking, submission, model creation, or Numerai account mutation is authorized.
Ender20 through Ender26 are closed research families; their frozen artifacts
and decision records must not be overwritten, renamed, deleted, or reused as
new runs.

The live repository checkpoint remains:

`c63e0465426c580d6144bcc092e199bc2f1dbbe4`

That checkpoint is the merge of Ender27's reviewed source packet and its
manifest-only seal. The exact Ender27 four-run Round-1 training cohort has now
completed locally without retry, but the one-shot evaluator has not been
invoked. The canonical decision path remains absent, so Ender27 has no
scientific verdict yet.

## Latest terminal experiment

[Ender26 Gaussian-rank benchmark residual](ender26_gaussian_rank_residual_v53/terminal_postmortem.md)
is closed at `ENDER26_NEGATIVE_NO_BMC_ALIGNMENT_GAIN`. Mean full and recent-40
BMC improved and benchmark correlation fell, but seed 1337 lost recent-40 BMC
and exceeded its matched drawdown allowance. The predeclared cross-seed gate
therefore rejected the procedure. Round 2 is unauthorized.

## Active research gate

[Ender27 tempered Gaussian-rank benchmark residual](ender27_tempered_gaussian_rank_residual_v53/experiment.md)
is at:

`ENDER27_ROUND1_FOUR_RUN_COHORT_COMPLETE_AWAITING_SEPARATE_EVALUATOR_AUTHORITY`

Ender27 tests one preregistered midpoint target-residual blend
(`lambda=0.5`) while holding the Ender26 control, architecture, training
procedure, matched seeds, cohort, and decision law fixed.

The reviewed lifecycle chain is:

- source PR #23, reviewed source commit
  `a23ed023f3d592b05cb5368ca68b8a84a83658d8`;
- merged source checkpoint A
  `f578fef884b1c38a0c7141cd65f7fe3f221b6c59`;
- manifest-only commit B
  `2d95e361dbc7f724a14835c3b4c491112e80f2bb`;
- merged seal checkpoint M / current `main`
  `c63e0465426c580d6144bcc092e199bc2f1dbbe4`;
- exact manifest Git blob
  `4942af7b5576d465678e07ad004e329555c7cf0e`.

The four original training invocations completed once, strictly serially, in
the frozen order:

| Run | Exit | Prediction bytes | Result bytes | Completion bytes |
| --- | ---: | ---: | ---: | ---: |
| `r1_control_rawresid_seed1337` | 0 | 15,779,944 | 6,170 | 1,455 |
| `r1_tempered_grank_resid_seed1337` | 0 | 15,772,440 | 6,374 | 1,471 |
| `r1_control_rawresid_seed2027` | 0 | 15,786,216 | 6,166 | 1,455 |
| `r1_tempered_grank_resid_seed2027` | 0 | 15,788,359 | 6,379 | 1,471 |

All twelve canonical training artifacts are local, nonempty, and intentionally
Git-ignored. Their identities are bound by the completion envelopes. No run was
retried, no output was repaired or redirected, and no prediction, result, or
metric was manually read.

The evaluator decision path remains absent:

`ender27_tempered_gaussian_rank_residual_v53/receipts/round1_tempered_alignment.json`

Issue #22 is the live lifecycle authority record. The evaluator must not start
while another job may launch Unreal Engine, Blender, another trainer, or another
substantial workload. A fresh preflight and a separate explicit authorization
are required immediately before the single evaluator invocation.

## Ender20-26 outcome chain

| Family | Terminal state | Key conclusion | Canonical record | Archival point |
| --- | --- | --- | --- | --- |
| Ender20 | `STOP_NO_ELIGIBLE_CANDIDATE` / not promotion eligible | K64/500k was the strongest architecture scout, but later stability, hybrid, seed, and auxiliary-target gates produced no eligible deployment candidate. | [Architecture](ender20_nn_architecture_v53/experiment.md), [seed stability](ender20_seed_ensemble_stability_v53/experiment.md), [hybrid stability](ender20_hybrid_stability_v53/experiment.md), [terminal checkpoint](ender20_aux_target_rank_ensemble_v53/CONTINUATION_CHECKPOINT.md) | `4669d42`, tag `numerai-ender20-terminal` |
| Ender21 | `NEGATIVE` confirmation terminal | Block-DRO remained positive and passed the absolute risk gates, but retained only 40.745% of discovery BMC against the required 60%. | [Confirmation postmortem](ender21_residual_stability_v53/confirmation_postmortem.md) | `bd0fe11`, tag `numerai-ender21-terminal` |
| Ender22 | Operationally invalid; no experiment decision | The half-life-52 run failed before its first fit on a 3.17 GiB allocation. The cohort was not scored and Round 2 was never authorized. | [Round-1 execution postmortem](ender22_temporal_retention_v53/round1_execution_postmortem.md) | `f9f45f6`, tag `numerai-ender22-terminal` |
| Ender23 | `NEGATIVE_SEED_INSTABILITY` | The memory repair worked and window-78 won Round 1. Two of three Round-2 realizations qualified, but the fixed ensemble failed its Sharpe and drawdown gates. | [Terminal postmortem](ender23_temporal_retention_v53/round2_terminal_postmortem.md) | `aff188b`, tag `numerai-ender23-terminal` |
| Ender24 | No decision; evaluator precondition failure | Four matched control/EMA runs finalized, but a CRLF/LF authority-fingerprint defect stopped the evaluator before metrics. Round 2 was not authorized. | [Round-1 execution postmortem](ender24_ema_seed_stability_v53/round1_execution_postmortem.md) | `d12f755`, tag `numerai-ender24-terminal` |
| Ender25 | `ENDER25_NEGATIVE_NO_EMA_STABILITY_GAIN` | The repaired one-shot evaluator produced the missing scientific decision. EMA greatly compressed seed variance, but lost average full and recent BMC and worsened seed-1337 Sharpe/drawdown. | [Terminal postmortem](ender25_ender24_evaluation_recovery_v53/terminal_postmortem.md) | `9f6a08c`, tags `numerai-ender25-terminal` and `numerai-ender25-terminal-custody` |
| Ender26 | `ENDER26_NEGATIVE_NO_BMC_ALIGNMENT_GAIN` | Gaussian-rank target residualization improved aggregate full/recent BMC and decorrelated from the benchmark, but seed 1337 failed the matched recent-40 and drawdown guards. | [Terminal postmortem](ender26_gaussian_rank_residual_v53/terminal_postmortem.md) | `f5ef885`, tag `numerai-ender26-terminal` |

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
- Ender27 preserves the exact Ender26 procedure and tests only the fixed
  half-strength target-residual midpoint. Its four training artifacts are
  complete, but their scientific meaning remains intentionally sealed until
  the one-shot evaluator is separately authorized and completes.

## Next admissible work

1. Keep `main` pinned at
   `c63e0465426c580d6144bcc092e199bc2f1dbbe4` until the Ender27 evaluator gate
   is complete.
2. Do not start the evaluator while another job may launch Unreal Engine,
   Blender, another trainer, or another substantial workload.
3. After the competing job finishes, perform a fresh read-only evaluator
   preflight: verify the exact Git/runtime/source envelope, the twelve training
   artifact identities, the two governed Parquet identities, resource posture,
   and decision-path absence.
4. Only after that preflight may the user explicitly authorize exactly one
   canonical Ender27 evaluator invocation.
5. Preserve the first truthful evaluator result or failure. Never retry the
   decision identity or delete a zero-byte/partial reservation.
6. Only after the scientific result is reviewed may a separate evidence packet
   update the ledger, publish a terminal postmortem and decision, and create an
   archival tag.
7. Do not reuse consumed eras `0301`-`1021` for post-hoc model selection.
   Historical eras `1022`-`1230` are not a substitute confirmation cohort.
8. Keep deployment separate. It requires a successful, separately frozen
   prospective validation and explicit authorization for every account action.

## Repository hygiene

- One directory represents one line of inquiry. Its canonical terminal
  decision and postmortem stay inside that directory.
- This ledger links outcomes; it does not replace experiment protocols,
  decisions, receipts, or full metric records.
- Large datasets, ignored predictions, logs, model bundles, and protected GPU
  build residue remain outside normal Git history unless a frozen protocol
  explicitly requires a small evidence artifact.
- Preserve the archival commits and tags above. Do not squash, rebase, or
  rewrite the research chain because manifests and receipts bind historical
  commit identities.
- Historical feature/source branches may be removed only after their merge and
  archival identities have been independently verified. Deleting a branch is
  repository housekeeping, never evidence deletion.
- A future family must use new output and receipt paths. Prior terminal or
  consumed artifacts are immutable evidence, not reusable workspace.
