# Numerai research ledger

This file is the navigation index for the governed Numerai research program.
The linked experiment records remain the canonical sources for protocols,
metrics, artifact hashes, custody rules, and terminal decisions.

## Current program state

No Ender-family model is approved for deployment. No upload, assignment,
staking, submission, model creation, or Numerai account mutation is authorized.
**Ender20 through Ender27 are closed research families**; their frozen
artifacts and decision records must not be overwritten, renamed, deleted, or
reused as new runs.

The live remote `main` checkpoint is:

`99df04ff07ca7cdc9b02bc0cd275d3ecb105a520`

That checkpoint is the Ender27 terminal-evidence merge (evidence PR #27,
archival tag `numerai-ender27-terminal`). **There is currently no active
scientific gate.**

## Latest terminal experiment

[Ender27 tempered Gaussian-rank benchmark residual](ender27_tempered_gaussian_rank_residual_v53/terminal_postmortem.md)
is closed at:

`ENDER27_NEGATIVE_NO_TEMPERED_ALIGNMENT_GAIN`

The separately authorized one-shot evaluator ran exactly once, exited 0, and
committed the canonical decision
([`round1_tempered_alignment.json`](ender27_tempered_gaussian_rank_residual_v53/receipts/round1_tempered_alignment.json),
21,670 bytes, SHA-256
`261ed2661dbc282388498ecd2fd8fec668a14ef78bf256b7549bbbdda73fc007`).
20 of 22 frozen checks passed. Exactly two failed, both in the recent-40
window: the aggregate recent-40 uplift gate and seed 1337's matched recent-40
gate. Mean full BMC improved about +5.86% and seed-1337 drawdown passed, but
aggregate recent-40 BMC fell about -4.21% and seed 1337 stayed below its
matched control, so the frozen all-conditions law rejected the tempered
procedure. Round 2 is unauthorized.

The reviewed lifecycle chain is:

- source PR #23, reviewed source commit
  `a23ed023f3d592b05cb5368ca68b8a84a83658d8`;
- merged source checkpoint A
  `f578fef884b1c38a0c7141cd65f7fe3f221b6c59`;
- manifest-only commit B
  `2d95e361dbc7f724a14835c3b4c491112e80f2bb`;
- merged seal checkpoint M
  `c63e0465426c580d6144bcc092e199bc2f1dbbe4`;
- exact manifest Git blob
  `4942af7b5576d465678e07ad004e329555c7cf0e`;
- evidence PR #27, evidence commit
  `8916a9a57d0ba6c0634b989d592f222e88f11edc`, evidence merge
  `99df04ff07ca7cdc9b02bc0cd275d3ecb105a520`, tag
  `numerai-ender27-terminal`.

The twelve generated training artifacts remain local, nonempty, and
intentionally Git-ignored; their identities are bound by the completion
envelopes and the canonical decision's input receipts. No run or evaluator
invocation was retried, and no artifact was repaired or redirected.

## Active research gate

There is currently no active scientific gate. Issue #22 (the Ender27
lifecycle record) is closed as completed. Any successor modeling requires a
newly named, separately frozen, preregistered hypothesis and fresh explicit
user authorization. Repository maintenance is tracked separately in Issue #26.

## Ender20-27 outcome chain

| Family | Terminal state | Key conclusion | Canonical record | Archival point |
| --- | --- | --- | --- | --- |
| Ender20 | `STOP_NO_ELIGIBLE_CANDIDATE` / not promotion eligible | K64/500k was the strongest architecture scout, but later stability, hybrid, seed, and auxiliary-target gates produced no eligible deployment candidate. | [Architecture](ender20_nn_architecture_v53/experiment.md), [seed stability](ender20_seed_ensemble_stability_v53/experiment.md), [hybrid stability](ender20_hybrid_stability_v53/experiment.md), [terminal checkpoint](ender20_aux_target_rank_ensemble_v53/CONTINUATION_CHECKPOINT.md) | `4669d42`, tag `numerai-ender20-terminal` |
| Ender21 | `NEGATIVE` confirmation terminal | Block-DRO remained positive and passed the absolute risk gates, but retained only 40.745% of discovery BMC against the required 60%. | [Confirmation postmortem](ender21_residual_stability_v53/confirmation_postmortem.md) | `bd0fe11`, tag `numerai-ender21-terminal` |
| Ender22 | Operationally invalid; no experiment decision | The half-life-52 run failed before its first fit on a 3.17 GiB allocation. The cohort was not scored and Round 2 was never authorized. | [Round-1 execution postmortem](ender22_temporal_retention_v53/round1_execution_postmortem.md) | `f9f45f6`, tag `numerai-ender22-terminal` |
| Ender23 | `NEGATIVE_SEED_INSTABILITY` | The memory repair worked and window-78 won Round 1. Two of three Round-2 realizations qualified, but the fixed ensemble failed its Sharpe and drawdown gates. | [Terminal postmortem](ender23_temporal_retention_v53/round2_terminal_postmortem.md) | `aff188b`, tag `numerai-ender23-terminal` |
| Ender24 | No decision; evaluator precondition failure | Four matched control/EMA runs finalized, but a CRLF/LF authority-fingerprint defect stopped the evaluator before metrics. Round 2 was not authorized. | [Round-1 execution postmortem](ender24_ema_seed_stability_v53/round1_execution_postmortem.md) | `d12f755`, tag `numerai-ender24-terminal` |
| Ender25 | `ENDER25_NEGATIVE_NO_EMA_STABILITY_GAIN` | The repaired one-shot evaluator produced the missing scientific decision. EMA greatly compressed seed variance, but lost average full and recent BMC and worsened seed-1337 Sharpe/drawdown. | [Terminal postmortem](ender25_ender24_evaluation_recovery_v53/terminal_postmortem.md) | `9f6a08c`, tags `numerai-ender25-terminal` and `numerai-ender25-terminal-custody` |
| Ender26 | `ENDER26_NEGATIVE_NO_BMC_ALIGNMENT_GAIN` | Gaussian-rank target residualization improved aggregate full/recent BMC and decorrelated from the benchmark, but seed 1337 failed the matched recent-40 and drawdown guards. | [Terminal postmortem](ender26_gaussian_rank_residual_v53/terminal_postmortem.md) | `f5ef885`, tag `numerai-ender26-terminal` |
| Ender27 | `ENDER27_NEGATIVE_NO_TEMPERED_ALIGNMENT_GAIN` | Half-strength tempering kept the full-period BMC gain, repaired seed-1337 drawdown, and compressed recent-window seed dispersion, but seed-1337 recent-40 BMC and the aggregate recent-40 uplift still failed the frozen law. | [Terminal postmortem](ender27_tempered_gaussian_rank_residual_v53/terminal_postmortem.md), [canonical decision](ender27_tempered_gaussian_rank_residual_v53/receipts/round1_tempered_alignment.json) | `99df04f`, tag `numerai-ender27-terminal` |

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
- Ender27 preserved the exact Ender26 procedure and tested only the fixed
  half-strength target-residual midpoint. The full-period BMC improvement
  survived tempering, recent-window two-seed dispersion compressed sharply
  (0.0026315400136204814 to 0.00018397972470021468), and seed-1337 drawdown
  was repaired — but seed-1337 recent-40 BMC and the aggregate recent-40
  uplift both failed, so the repair was insufficient in the window that
  decides the gate. The result is promising but non-robust negative evidence,
  not a no-signal result.

## Next admissible work

1. Independent review and merge of the control-ledger documentation (draft
   PR #25). Documentation is not research authority.
2. Repository maintenance under Issue #26 (CI trigger modernization, branch
   review after custody verification, labels, protection). Maintenance is
   housekeeping, never evidence deletion, and grants no research authority.
3. Any successor modeling requires a genuinely new, newly named,
   separately frozen, preregistered hypothesis and fresh explicit user
   authorization. This ledger deliberately does not design one.
4. Do not reuse consumed eras `0301`-`1021` for post-hoc model selection.
   Historical eras `1022`-`1230` are not a substitute confirmation cohort.
5. Keep deployment separate. It requires a successful, separately frozen
   prospective validation and explicit authorization for every account action.
   No such authority exists.

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
