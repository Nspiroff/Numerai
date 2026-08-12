# Ender21 Confirmation Postmortem

## Outcome

The frozen Ender21 experiment ended at `NEGATIVE` on 2026-08-11. The sole
authorized confirmation candidate was `c1_selected_tabm_k64_block_dro`. Its
unchanged seed-1337 predictor trained once, stopped at epoch 14 with the best
epoch restored from epoch 10, produced raw target-free predictions, and was
then scored once on exactly 263,551 unique IDs across 40 eras `0865`-`1021`.

## Frozen-check result

| Check | Requirement | Observed | Result |
| --- | ---: | ---: | --- |
| Mean BMC | >= 0.0020 | 0.00280203 | pass |
| BMC Sharpe | > 0.25 | 0.33489 | pass |
| BMC max drawdown | < 0.10 | 0.02877 | pass |
| Mean Corr | >= 0.008 | 0.00827321 | pass |
| Benchmark correlation | < 0.25 | 0.14474 | pass |
| Positive 10-era blocks | >= 3 of 4 | 4 of 4 | pass |
| Worst 10-era block BMC | > -0.001 | 0.00226679 | pass |
| Discovery BMC retention | >= 60% | 40.745% | **fail** |

The required retention BMC was `0.0041261703`, calculated from the frozen
Round-1 BMC `0.0068769505`. Confirmation BMC was `0.0028020256`, a shortfall of
`0.0013241447`. The chronological block means were all positive:
`0.00272269`, `0.00249867`, `0.00226679`, and `0.00371996`.

## Interpretation

The Block-DRO TabM signal remained positive, diversified, stable across all
four confirmation blocks, and above the absolute BMC/Corr risk gates. It did
not preserve enough of its discovery-period unique contribution. The result is
therefore useful negative evidence: the architecture improved robustness and
drawdown, but discovery BMC overstated the magnitude that generalized into the
held-out historical period.

## Terminal action

No retry, seed change, blend, threshold relaxation, neutralization change, or
post-hoc rescue is permitted on these scored eras. The deferred
full-consecutive gate remains closed, and this result does not authorize a
Numerai upload, model assignment, staking, or account mutation. Any future
work must be a newly named experiment with a newly frozen protocol and must not
reuse this confirmation cohort for selection.
