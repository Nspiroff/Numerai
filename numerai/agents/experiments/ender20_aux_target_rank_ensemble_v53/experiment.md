# Ender20 Auxiliary-Target Rank Ensemble (v5.3)

**Started:** 2026-08-03

**State:** protocol definition pending pre-scoring checkpoint; no new component
training or ensemble scoring performed

## Abstract

This experiment tests whether a fixed-architecture ensemble of five diverse
20-day Numerai targets can improve the BMC stability of the prior direct
Xerxes20 LightGBM. Jasper, Teager2b, Victor, Xerxes, and Tyler span the quality
and diversity range recorded in Numerai's official target-ensemble tutorial.

The experiment changes only the component training label and Tyler's
predeclared ensemble weight. Every other learner, feature, CV, sampling, and
ranking choice is fixed before scoring.

## Prior evidence and hypothesis

The sealed depth-8 Xerxes scout achieved calibration BMC mean `0.001659` and
Ender Corr `0.021912`, but failed the strict stability screen with BMC Sharpe
`0.175127`. Numerai's official tutorial shows that equal-rank models trained on
different auxiliary targets can smooth Corr, while also warning that target
ensembles may remain benchmark-like.

The hypothesis is that target diversity can lift BMC Sharpe and reduce drawdown
without sacrificing the Xerxes signal's Ender Corr or becoming too similar to
the official Ender20/Ender60 models or the prior two-seed TabM residual.

## Frozen method

The exact components, five blend candidates, chronological split, similarity
guards, thresholds, deterministic selection, confirmation boundary, and stop
rule are defined in [`gate.md`](gate.md). Exact data, tutorial, historical
artifact, runtime, and configuration receipts are pinned in
`source_manifest.json`.

No new performance result belongs in this report until the pre-scoring protocol
checkpoint and later evaluator/training checkpoint have both been committed.

## Results

Pending frozen scout execution.

## Decision

Pending. No upload, assignment, submission, or staking action is authorized by
this experiment definition.
