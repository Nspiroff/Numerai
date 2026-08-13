# Ender21 Gate Contract

This file is the concise fail-closed contract for
`ender21_residual_stability_v53`. The detailed rationale and formulas live in
`experiment.md` and are part of this contract.

## Authority and prohibited actions

- Ender20 experiment directories and their results are read-only evidence.
- The protected repository residue `.gpu-lgbm-source-build/` is out of scope.
- No Numerai upload, model assignment, staking, social connection, key display,
  or other account mutation is authorized.
- Historical v5.3 validation is not called unseen. The strongest possible state
  from this gate is `SHADOW_READY`.
- Candidate definitions, seeds, split rules, losses, ranking, blend formula, and
  thresholds freeze before the first Ender21 score is read.
- Round-1 inputs must be physically isolated Parquets whose selected source row
  groups have maximum era <=0861. Runtime filtering of a later target-bearing
  file is insufficient. The mixed 1025/1029 row group remains forbidden from
  any later Ender21 confirmation extract.
- Round-1 `data.era_allowlist_path` must bind the exact committed discovery list
  through 0861 and filter the CV universe before any fit. The separate 0865-1021
  confirmation list remains closed until both discovery rounds pass.

## Required run order

1. Commit the protocol.
2. Implement loss support and tests.
3. Create the five exact Round-1 configs and a content-hashed source manifest.
4. Commit that pre-scoring checkpoint.
5. For each named config, create-new reserve its two canonical outputs, then
   verify the exact committed source/runtime/data manifest before config or
   modeling-data access. Run each candidate once; never overwrite, delete,
   rename, or retry an existing result or prediction.
6. Evaluate and write the Round-1 decision.
7. If and only if Round 1 passes, commit the four exact seed-replication configs,
   evaluator, and a second source manifest; then run the two matched control /
   selected pairs exactly once.
8. Round 2 passed 2/3. Commit the exact one-model confirmation config, runner,
   evaluator, rules, protocol lists, and third source manifest before prediction.
9. Reserve every canonical confirmation output once. Fit only the selected
   seed-1337 family on exact eras 0161-0809; preserve eras 0813-0861 as the
   outer embargo. Predict the exact 263,551 rows in eras 0865-1021 without
   reading their target, then commit the completion and unscored result.
10. Only after that evidence is committed and clean, evaluate the exact
    Ender21-only 0865-1021 confirmation once and write
    `receipts/confirmation_research.json` with create-new semantics.
11. Do not open the later full-consecutive gate while Ender20's locked period is
   protected.
12. Stop at `NEGATIVE`, `SCOUT_WINNER`, or
   `SEED_REPLICATION_PASS`, or
   `HISTORICAL_RESEARCH_PASS_FORWARD_VALIDATION_REQUIRED`; do not upload.
   Only `SEED_REPLICATION_PASS` from the exact three-realization Round-2 receipt
   authorizes the Ender21 family-locked confirmation.

## Round-1 eligibility

Against the fresh matched control, a challenger must retain >=90% of full and
most-recent-fold BMC, reduce full BMC max drawdown by >=15%, keep BMC Sharpe within 0.05,
have positive BMC in every outer fold, and keep Corr in [0.005, 0.04). Ranking is
full BMC, most-recent-fold BMC, then lower drawdown.

## Round-2 eligibility

Round 2 contains exactly three matched realizations: the reused base-seed-1337
pair, the model-seed-2027 pair, and the row-sample-seed-2027 pair. The selected
family advances only if at least two of the three pass every exact Round-1
eligibility check against their matched control. No relaxed or approximate
threshold comparison is allowed.

## Ender21 confirmation eligibility

The exact 0865-1021 confirmation opens once after Round 2. Only
`c1_selected_tabm_k64_block_dro` is authorized. Its config may differ from the
selected Round-1 config only in `output.results_name`; no confirmation control,
new seed, retry, or blend is allowed. The fit era list is exactly 0161-0809 and
the outer embargo list is exactly 0813-0861. The confirmation cohort is exactly
263,551 unique IDs over 40 eras 0865-1021. Training/prediction must not read the
confirmation target. A committed-clean completion and unscored result, with an
exact prediction hash binding, are prerequisites to the target-bearing evaluator.

Required: BMC
>=0.0020, Sharpe >0.25, drawdown <0.10, Corr >=0.008, benchmark correlation
<0.25, at least 3/4 chronological blocks positive, worst block >-0.001, and at
least 60% of the frozen discovery BMC `0.006876950492356912`. A pass stops at
`HISTORICAL_RESEARCH_PASS_FORWARD_VALIDATION_REQUIRED` and still requires an
unchanged predictor to collect 52 future resolved eras. It does not authorize
upload, assignment, staking, or the deferred full-consecutive gate.

## Deferred full-consecutive eligibility

The frozen two-seed 50/50 within-era-rank ensemble must meet:

`BMC >= 0.0055`, `last200 BMC >= 0.0035`, `Corr in [0.0075, 0.04)`,
`BMC Sharpe >= 0.50`, `max drawdown <= 0.225`, benchmark correlation <=0.15,
positive BMC in all four chronological quartiles, and positive full/recent BMC
for both component seeds.

These rules are inert until a separate explicit decision releases the protected
period. They are recorded now so that a later decision cannot tune them after
seeing those eras.

## No post-hoc rescue

An ineligible candidate cannot be repaired on the same scored eras by changing
weights, blocks, temperatures, seeds, target mixtures, neutralization, or
thresholds. Such a change is a new named experiment with a new protocol.
