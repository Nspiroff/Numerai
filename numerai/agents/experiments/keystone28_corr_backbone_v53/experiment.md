# Keystone28 — CORR-backbone research family on v5.3 (Round 0)

Family directory: `numerai/agents/experiments/keystone28_corr_backbone_v53/`
Round-0 assignment: KA28 (current-data authority and true Numerai scoring harness).

## Status

**Round 0 only. No model has been trained, tuned, fitted, or selected in this
family. No holdout era has been scored. No upload, model creation, submission,
staking, or Numerai account action has occurred or is authorized.**

Round 0 establishes three things and nothing else:

1. the exact current data/target/benchmark/meta-model/scoring authority,
   frozen in [`round0_data_authority.json`](round0_data_authority.json) and
   [`round0_score_authority.json`](round0_score_authority.json);
2. a corrected chronological DEV/GAP/HOLDOUT partition of the actual current
   v5.3 validation data, frozen in [`round0_partition.json`](round0_partition.json);
3. a reusable official-parity scoring harness
   (`agents/code/metrics/keystone_round0.py`) proven end-to-end on real
   historical data with a plumbing-only smoke
   ([`round0_smoke.py`](round0_smoke.py) →
   [`round0_smoke_result.json`](round0_smoke_result.json)).

The full narrative audit is in [`round0_audit.md`](round0_audit.md).

## Corrected hypothesis

The archived Ender program (Ender20–27, all terminal) optimized
benchmark-unique contribution (BMC against `v53_lgbm_ender20`) and repeatedly
produced non-robust gains in the recent-window gates. Keystone reorients the
program to the **actual current payout objective**:

> A CORR-backbone model family — selected and gated directly on the official
> weighted payout score (CORR20V2 × 0.75 + MMC20 × 2.25 against
> `target_ender_20` and the published Meta Model), rather than on
> single-benchmark BMC — can produce a candidate whose weighted score is
> positive and era-stable on a frozen pre-registered development protocol, and
> that survives an untouched, accumulating post-archive holdout.

The **target-authority correction** embedded in this hypothesis: the MR28
brief's payout target assertion (`target_ender_20`) was distrusted and
re-derived from live authority; a competing expectation (`target_cyrus_20`)
was found to be stale (Cyrus was retired as the payout target on 2025-12-31).
The independently verified current payout target is **`target_ender_20`**,
with the announced Ender-60 cutover recorded as pending. The bare `target`
column in v5.3 aliases `target_ender_60` and is never used as a scoring
authority.

## Current payout objective (frozen 2026-08-17, must be re-audited before use)

- Payout target: `target_ender_20` (effective 2026-01-01; reaffirmed by the
  v5.3 "Quantum" release, 2026-07-15).
- Scores: CORR20V2 (`numerai_tools.scoring.numerai_corr`) and MMC20
  (`numerai_tools.scoring.correlation_contribution` against
  `numerai_meta_model`).
- Weighted score: `corr * 0.75 + mmc * 2.25` (multipliers effective
  2026-01-01).
- BMC: not reproducible from published files
  (`BMC_AGGREGATE_NOT_REPRODUCIBLE_FROM_PUBLISHED_FILES`); never substituted.

## Round-0 packet contents

| File | Purpose |
| --- | --- |
| `round0_data_authority.json` | Dataset inventory, storage/download identities (SHA-256), schemas, era/null audit, findings |
| `round0_score_authority.json` | Dated machine-readable scoring authority; loadable by `ScoreAuthority.from_json` |
| `round0_partition.json` | Frozen corrected DEV/GAP/HOLDOUT partition with embargo rationale and accumulating-holdout policy |
| `round0_smoke.py` | Deterministic, offline, plumbing-only real-data smoke runner |
| `round0_smoke_result.json` | Canonical smoke output (byte-stable across runs) |
| `round0_audit.md` | Human-readable Round-0 audit narrative |

## What Round 0 does NOT authorize

- training or tuning of any model (Round 1 requires its own explicit
  authorization and pre-registration);
- scoring, summarizing, or selecting on holdout eras (≥ 1231);
- any Numerai account mutation, upload, submission, or staking;
- treating the smoke metrics as evidence about any model — the smoke
  prediction vector is a published benchmark column used only to prove
  plumbing.
