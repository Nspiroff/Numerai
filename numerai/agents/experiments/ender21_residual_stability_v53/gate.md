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
8. Open the exact Ender21-only 0865-1021 confirmation once only after Round 2.
9. Do not open the later full-consecutive gate while Ender20's locked period is
   protected.
10. Stop at `NEGATIVE`, `SCOUT_WINNER`, or
   `HISTORICAL_RESEARCH_PASS_FORWARD_VALIDATION_REQUIRED`; do not upload.

## Round-1 eligibility

Against the fresh matched control, a challenger must retain >=90% of full and
most-recent-fold BMC, reduce full BMC max drawdown by >=15%, keep BMC Sharpe within 0.05,
have positive BMC in every outer fold, and keep Corr in [0.005, 0.04). Ranking is
full BMC, most-recent-fold BMC, then lower drawdown.

## Ender21 confirmation eligibility

The exact 0865-1021 confirmation opens once after Round 2. Required: BMC
>=0.0020, Sharpe >0.25, drawdown <0.10, Corr >=0.008, benchmark correlation
<0.25, at least 3/4 chronological blocks positive, worst block >-0.001, and at
least 60% discovery-BMC retention. A pass still requires 52 future resolved eras.

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
