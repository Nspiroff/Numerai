# Ender23 terminal postmortem

## Outcome

Ender23 terminated at `NEGATIVE_SEED_INSTABILITY`. Round 1 selected
`r1_recent_window78`, but the fixed three-member Round-2 ensemble failed two
mandatory stability gates. No forward-validation, upload, staking, model
creation, or account action is authorized.

The infrastructure objective was achieved: all three fresh Round-1 candidates
loaded the exact 957,366-row, 176-era, 3,555-feature discovery cohort and
completed. In particular, `r1_recent_half_life52` crossed the former Ender22
pre-fit 3.17 GiB allocation failure and produced a finalized completion
receipt. No Ender22 prediction, result, or completion was reused.

## Frozen provenance

- Memory repair and Ender23 protocol commit: `967cd78546c2d431759bea85a034750760d35775`.
- Round-1 manifest-only seal: `37fa3c7` (manifest `git_head` is the repair
  commit above).
- Committed Round-1 evidence and decision: `8bc9b4f`.
- Round-2 manifest-only seal: `0c1d37a` (manifest `git_head` is the Round-1
  evidence commit above).
- Round-1 decision SHA-256:
  `8548c0cde2f518276a5253d397c1aa846b780943ca1dc27601b98e7c7efbb925`.
- Round-2 terminal receipt SHA-256:
  `fdab6ca9a91d6f1acfc09d0f40406d6f1fa9f55a9c7b89190c38685dacfa4c54`.

## Round 1

| Procedure | Full BMC | Recent-40 BMC | Sharpe | Drawdown | Result |
|---|---:|---:|---:|---:|---|
| control Block-DRO | 0.00687695 | 0.00737398 | 0.644354 | 0.043614 | control |
| half-life 52 | 0.00658219 | 0.00667034 | 0.552205 | 0.109461 | ineligible |
| recent window 78 | 0.00666491 | 0.00849022 | 0.629060 | 0.043614 | selected |

The half-life candidate failed the recent-gain, worst-block, Sharpe, and
drawdown gates. The 78-era window passed every predeclared challenger gate and
advanced without any threshold or weight adjustment.

## Round 2

| Realization | Full BMC | Recent-40 BMC | Sharpe | Drawdown | Qualified |
|---|---:|---:|---:|---:|---|
| selected base, seeds 1337/1337 | 0.00666491 | 0.00849022 | 0.629060 | 0.043614 | yes |
| model seed 2027 | 0.00495794 | 0.00585868 | 0.440663 | 0.058255 | no |
| sample seed 2027 | 0.00666491 | 0.00849022 | 0.629060 | 0.043614 | yes |
| fixed equal-rank ensemble | 0.00637761 | 0.00794986 | 0.591972 | 0.044284 | no |

The model-seed realization failed full-BMC retention, recent-BMC retention,
Sharpe, and drawdown. Two of three individual entries therefore qualified, but
the fixed ensemble still failed the mandatory Sharpe and drawdown comparisons
against the original control.

The sample-seed realization was byte-identical to the selected base prediction
(SHA-256
`3e5285e3d5b9e3dea7df8da3d121a0157fa83e91bbec371c30526a8539aced73`).
This is expected from the realized window-78 geometry: no outer training fold
reached the 500,000-row sampling cap, so changing only `sample_seed` could not
change the fit. It must not be interpreted as independent stochastic support.
The already-negative ensemble outcome means this limitation cannot promote the
experiment to a pass.

## Decision

Stop this family. Do not rescue it with another seed, neighboring window,
different blend weight, threshold relaxation, or reuse of the consumed
0865-1021 confirmation cohort. A future experiment, if desired, must start
from a new hypothesis and freeze a replication axis that is mechanically
active for every candidate before any score is read.
