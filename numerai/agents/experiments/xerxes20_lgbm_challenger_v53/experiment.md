# Xerxes-20 LightGBM Challenger (v5.3)

**Started:** 2026-08-03
**State:** gate frozen; implementation and scout pending

## Abstract

This experiment tests a direct `target_xerxes_20` LightGBM as a genuinely
different challenger to the benchmark-residual Ender20 TabM research line. The
primary decision metric remains BMC against `v53_lgbm_ender20`, scored on
`target_ender_20` rather than the auxiliary training target.

## Motivation

The two-seed TabM ensemble improved residual stability but no frozen benchmark
blend satisfied both unique-signal retention and target-Corr retention. The
checked-in official target-ensemble tutorial provides a useful prior: its
downsampled small-feature Xerxes model had the strongest Corr and Sharpe among
four auxiliary-target examples. That result is not independent evidence for
this experiment, but it motivates testing a tree model, different target, and
more capable medium feature set under a leakage-safe gate.

## Method

The exact sources, four scout profiles, deterministic winner rule, consecutive
confirmation, thresholds, packaging boundary, and stop rule are frozen in
[`gate.md`](gate.md). Exact input and configuration receipts are pinned in
[`source_manifest.json`](source_manifest.json). No Numerai upload or staking is
authorized.

## Results

Pending.

## Decision

Pending.
