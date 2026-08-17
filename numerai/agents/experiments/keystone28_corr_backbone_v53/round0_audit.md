# Keystone28 Round-0 audit (KA28)

Audit date: 2026-08-17 (UTC). Base commit: `7eae9e4850782bbc5362834f65b28fb1cd6c2818`
(current `origin/main` at branch creation). Branch:
`research/keystone28-round0-data-metrics-r1`.

This audit records how the current Numerai data/scoring authority was
independently resolved, how the corrected partition was frozen, and how the
Round-0 harness was proven on real historical data. **No model was trained. No
holdout era was scored. No API-authenticated or account action occurred** (the
only network operations were unauthenticated public dataset listing/downloads
and public documentation reads).

## 1. Target-authority resolution (the MR28 correction)

The KA28 brief required distrusting MR28's payout-target assertion
(`target_ender_20`) and independently re-deriving the live authority, with
`target_cyrus_20` named as the expected answer. The live evidence resolved the
conflict the other way:

- Cyrus was the payout target from 2023 until the 2026 season. Effective
  **2026-01-01**, "Ender20 is the new scoring, payout, and leaderboard
  target" — the first payout-target change since 2023 — and the multipliers
  moved to **CORR 0.75** (from 0.5) and **MMC 2.25** (from 2.0)
  (blog.numer.ai, "Numerai Monthly: … 2026 Payout Updates").
- The **v5.3 "Quantum"** release (forum, 2026-07-15) added 807 new features,
  kept the target list, stated "**the payout target is remaining the same**",
  and shipped benchmarks `v53_lgbm_ender20` and `v53_lgbm_ender60`.
- Current scoring docs list CORR20V2/CORR60/CORJ60/CORT20 and describe a
  "**separately coordinated Ender-60 scoring cutover round**" after which CORR
  and MMC will use the 60-day Ender target. As of retrieval no cutover round
  was effective: payouts remain on the 20-day Ender target.
- The current v5.3 files corroborate: there is **no plain cyrus target
  column** (only auxiliary `target_cyrusd_20/60`); `target_ender_20` is
  present and resolves on the 20D2L cadence (frontier era 1227 vs 1219 for the
  60D targets); and the bare `target` column is **byte-identical to
  `target_ender_60`** on pre-holdout eras (NaN pattern included), confirming
  the documented alias.

**Resolution: the current official payout target is `target_ender_20`.**
MR28's assertion was correct; the `target_cyrus_20` expectation was stale.
Documentation and files agree, so the
`STOPPED_AT_KA28_TARGET_AUTHORITY_MISMATCH` condition was not triggered. The
announced Ender-60 cutover is recorded in `round0_score_authority.json` as a
pending transition that stales the record when effected.

## 2. Storage and data acquisition

- External data root: `D:\numerai-data\keystone28\v5.3` on the non-system
  `D:` volume (ExtraStorage, ≈265 GB free before download; free space before
  and after recorded in `round0_data_authority.json`). No dataset lives under
  the Git worktree; `*.parquet` and `v5.3/` remain gitignored.
- Downloads (unauthenticated `NumerAPI().download_dataset`; sizes and SHA-256
  in `round0_data_authority.json`): `features.json` (387,149 B),
  `meta_model.parquet` (13,590,154 B), `validation_example_preds.parquet`
  (97,057,487 B), `validation_benchmark_models.parquet` (126,278,375 B),
  `validation.parquet` (5,611,121,884 B, SHA-256
  `8ed7859b707aee8e4d6c4476fb9ecb1123caa429311dbf9f45c574722c24fae6`).
  No pre-existing file was overwritten (the data root was created fresh).
- Dataset inventory: 48 datasets across v5.0–v5.2 and v5.3 (12 files each);
  v5.3 is the current recommended Tournament version; v4.x is no longer
  listed.

## 3. Schema and era audit (authorized operations only)

Operations used: schema inspection, era enumeration, row counts, null counts,
and column identity checks (identity checks restricted to pre-holdout eras
≤ 1222). Holdout-era target values were touched only as null-count metadata.

- `validation.parquet`: 4,114,072 rows; 3,599 columns = 3,555 `feature_*`
  + 41 targets + `era`/`data_type`/`id`; `data_type` = validation (4,021,919)
  and test (92,153); eras **0575–1232**, 658 contiguous weekly eras, no gaps;
  5,450–7,309 rows per era.
- Targets: bare `target` plus 20 named pairs × {20, 60}: agnes, alpha, bravo,
  caroline, charlie, claudia, cyrusd, delta, echo, **ender**, jasper, jeremy,
  ralph, rowan, sam, teager2b, tyler, victor, waldo, xerxes.
- Resolution frontiers: `target_ender_20` fully resolved 0575–**1227**
  (unresolved 1228–1232; no partially resolved era anywhere);
  `target_ender_60` and bare `target` resolved through **1219** (unresolved
  1220–1232) — cadences consistent with 20D2L vs 60D2L.
- `meta_model.parquet`: columns `era`/`data_type`/`numerai_meta_model`/`id`;
  eras **1133–1219** (87 contiguous), 575,597 rows, no duplicate ids.
- `validation_benchmark_models.parquet`: keys `era`/`id`; exactly **two**
  benchmark columns — `v53_lgbm_ender20`, `v53_lgbm_ender60` — eras
  0575–1232. **No official Benchmark Meta Model aggregate column and no
  historical stake weights are published**, so official BMC is
  `BMC_AGGREGATE_NOT_REPRODUCIBLE_FROM_PUBLISHED_FILES`; no substitute is
  fabricated or reported as BMC.
- `validation_example_preds.parquet`: `prediction`/`id`, 4,114,072 rows.
- `features.json`: `feature_sets` (19 sets: all = 3,555, medium = 780,
  small = 42, quantum = 807, faith = 372, midnight = 244, rain = 666,
  sunshine = 325, serenity = 95, agility = 145, wisdom = 140,
  constitution = 335, dexterity = 51, strength = 135, charisma = 290,
  intelligence = 35, v2_equivalent = 304, v3_equivalent = 1,000,
  fncv3 = 400) and 41 `targets`.

## 4. Corrected partition (why it differs from a naive copy of MR28)

The archived ledger (`numerai/agents/experiments/README.md`) documents the
consumption history: selection consumed eras 0301–1021; eras 1022–1230 were
observed by the archived confirmation/evaluations and are "not a substitute
confirmation cohort". The corrected partition is derived from that history
plus the **actual** current era range and resolution frontiers:

- **DEV 0575–1222** (648 validation-file eras) with documented subzones:
  archive-selection-consumed (0575–1021), archive-observed (1022–1230
  portion), and the **MMC-capable zone 1133–1219** taken from the actual
  published meta-model coverage — a constraint MR28's fixed boundary never
  encoded.
- **GAP 1223–1230** (8 full weekly eras, the required minimum). The 9-week
  era-date spacing between DEV end (1222) and HOLDOUT start (1231) is 63
  days — beyond the 22-day forward window of the current 20D2L payout target
  *and* beyond the 62-day window of a 60D2L target — so the embargo stays
  valid across the announced Ender-60 cutover.
- **HOLDOUT ≥ 1231, ACCUMULATING.** Post-archive eras are reserved
  preferentially: the holdout starts immediately after era 1230, the last
  archive-observed era. At freeze the file contains holdout eras 1231–1232
  and **zero of them are fully resolved** for `target_ender_20` (frontier
  1227), so fewer than 13 suitable holdout eras exist and the holdout is
  marked accumulating **without weakening the 13-era requirement**. No
  post-hoc selection on the holdout; Round 0 grants no authority to score it.

## 5. Harness (`agents/code/metrics/keystone_round0.py`)

Official implementations are retained, not rederived: per-era CORR is
`numerai_tools.scoring.numerai_corr` (CORR20V2 math) and per-era MMC/BMC is
`numerai_tools.scoring.correlation_contribution`. The harness adds the
authority boundary and the failure discipline around them:

- `ScoreAuthority` (frozen dataclass, loadable from
  `round0_score_authority.json`): payout target, CORR/MMC score names,
  multipliers, meta-model column, BMC aggregate authority, retrieval
  provenance. No authority value has a default; the bare `target` alias is
  rejected as a payout-target configuration.
- `score_round0(...)`: strict one-to-one id joins with era-equality checks;
  loud `ValueError` on duplicate/missing/misaligned ids, non-finite
  predictions, missing target values in scored eras, malformed or mixed-width
  era strings, and unexpected era sets (`expected_eras`); deterministic
  `(era, id)` ordering before scoring; weighted score
  `corr * corr_multiplier + mmc * mmc_multiplier`.
- Meta-model coverage: wholly uncovered eras either fail (default) or are
  cleanly excluded under the explicitly declared policy; partially covered
  eras always fail.
- Summary (reproducible from the per-era frame via `summarize_round0`): mean,
  population std (ddof = 0), era Sharpe (= mean/std, no annualization, NaN at
  std 0 — convention recorded in the output), cumulative-sum max drawdown
  (positive magnitude, convention recorded), recent-window mean, per-block
  means, correlation with the Meta Model, and optional benchmark-column
  correlations (both via the official Numerai Corr transform with the
  reference vector in the target slot — diagnostic, not payout scores).
- BMC gating: computed only when the authority declares the exact official
  historical BMM aggregate column; otherwise reported as
  `{"status": "unavailable", "reason":
  "BMC_AGGREGATE_NOT_REPRODUCIBLE_FROM_PUBLISHED_FILES"}` — a single
  benchmark or an equal-weight average is never reported as BMC.

## 6. Focused tests (`agents/tests/test_keystone28_round0_metrics.py`)

29 synthetic, deterministic, CPU-only unittest cases (no dataset, no network,
no pyarrow/numerapi import): exact per-era parity with `numerai_corr` and
`correlation_contribution`; exact weighted-score arithmetic; row-order
invariance across all inputs and repeat-run identity; explicit (non-default)
target and multiplier authority incl. bare-alias rejection and
score-authority-record round-trip; loud failures for duplicate/missing ids,
era disagreement, non-finite predictions, missing targets, malformed eras,
unexpected era sets; meta-model coverage policy behavior (fail / clean whole-
era exclusion / partial-coverage always fails); BMC unavailability without
official BMM authority and BMC parity when one is declared; summary
reproducibility against hand computation. Local run: **29/29 OK** (Python
3.13.14, CI-pinned numpy 2.5.1 / pandas 3.0.5 / numerai-tools 0.6.0).

## 7. Real-data smoke (plumbing evidence only)

`round0_smoke.py` scored the published `v53_lgbm_ender20` benchmark column as
the prediction vector on the frozen pre-holdout smoke slice (eras 1216–1219:
inside DEV, inside meta-model coverage, fully resolved for the payout
target). 27,897 rows per source joined one-to-one (identical id sets across
validation/meta-model/benchmarks; zero nulls; zero duplicates). All per-era
CORR/MMC/weighted scores are finite; era-set enforcement (`expected_eras`)
and the embargo/holdout guard held; **no holdout era was read for scoring**.
Two independent executions produced **byte-identical** output
(`round0_smoke_result.json`, payload SHA-256
`1dbad8b681ea1ec52e78d7c6e103d3e45b150d004795b1c69a446c1a42756789`). The
numbers exist to prove plumbing, not to rank models: the benchmark's
correlation with the Meta Model (≈ 0.63–0.69 per era) and its small negative
MMC on those eras are exactly the expected shape for a benchmark model close
to the Meta Model, and no benchmark is selected or ranked from this smoke.

## 8. Hygiene and revalidation

- The protected checkout `Desktop\example-scripts` was not modified (fetch
  and worktree registration only). Archived Ender experiments, their tests,
  PR #31, Issue #26 maintenance, branch protection, and CI workflows were not
  touched.
- Generated multi-gigabyte data stays outside Git on `D:`.
- Round-0 records are dated snapshots. **The scoring authority changes over
  time** (multipliers and payout target changed 2026-01-01; an Ender-60
  cutover is announced): every value in this packet must be re-audited
  against live documentation, round configuration, and current files before
  deployment-relevant use.
