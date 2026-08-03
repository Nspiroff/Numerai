# Ender-20 Two-Seed Stability Experiment (v5.3)

**Started:** 2026-08-03
**State:** gate frozen; seed-2027 training pending

## Research question

Can an equal-rank ensemble of the frozen K64 TabM architecture at model seeds
1337 and 2027 reduce the benchmark-hybrid's BMC variance enough to clear the
unchanged stability gate?

## Prior evidence

The single-seed hybrid round had no calibration-eligible candidate. The closest,
`hybrid_w35`, passed calibration drawdown (`0.143763`), BMC retention, Corr
retention, and benchmark-similarity checks, but missed BMC Sharpe (`0.442194`
versus strict `> 0.45`). It would also miss holdout Sharpe (`0.334145` versus
strict `> 0.35`).

The architecture and second-seed configuration were frozen by the earlier
six-round search. This experiment changes only model initialization through the
existing `scale_disk_tabm_k64_train500k_seed2027.py` config, then applies an
equal-rank ensemble and the same five benchmark weights.

## Protocol

The exact inputs, transforms, split, selection rule, thresholds, stopping rule,
and deployment boundary are frozen in [`gate.md`](gate.md). The seed-2027 model
will train on the existing disk feature store with CUDA and write generated OOF
predictions/results to the original architecture experiment's ignored output
folders.

## Results

Pending.

## Decision

Pending. No packaging or upload is permitted unless every frozen check passes.
