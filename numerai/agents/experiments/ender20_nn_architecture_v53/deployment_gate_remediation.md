# Predeclared cross-architecture drawdown remediation gate

Frozen on 2026-08-02 after the first consecutive-era TabM run failed the
original gate's per-run drawdown ceiling, and before any consecutive-era MLP
result existed.

## Why this gate exists

`scale_disk_tabm_k64_train500k` produced positive full and recent BMC with
plausible target Corr and acceptable Sharpe, but its full-period BMC maximum
drawdown was 0.3009819625 versus the original strict limit of 0.15. Because the
original gate requires every TabM seed to pass, the remaining two runs cannot
repair that gate and will not be launched.

The architecture-search report had already frozen a 50/50 per-era-rank blend of
the residual MLP and TabM as its lowest-drawdown cross-architecture candidate
before any full consecutive-era output was observed. This remediation evaluates
that exact formula once; it does not search blend weights or seeds.

## Runs and candidate

1. Existing TabM component: `scale_disk_tabm_k64_train500k`, model seed 1337,
   sample seed 1337, 500,000-row cap.
2. New MLP component: `scale_disk_mlp_residual_train250k`, model seed 1337,
   sample seed 1337, 250,000-row cap.
3. Candidate: rank each component independently within era using pandas
   `groupby(era, sort=False).rank(method="average", pct=True)`, then compute the
   equal-weight arithmetic mean. Do not rank the blend again.

Both components use the exact full consecutive benchmark-covered feature store,
five expanding outer splits, and 52-era inner and outer embargoes. Fold 0 has no
training history and is skipped; folds 1 through 4 must align exactly by ID,
era, target, benchmark, and fold.

## Fail-closed checks

- Source hashes, store generation, prediction semantics, IDs, coverage,
  alignment, and finite-value checks from the original gate remain mandatory.
- The new MLP component must have positive full-period and last-200-era BMC,
  target Corr in the inclusive range 0.005 to 0.04, full BMC Sharpe above 0.25,
  and absolute average correlation with the official benchmark below 0.25.
- The 50/50 candidate must have positive full-period and last-200-era BMC,
  target Corr in the inclusive range 0.005 to 0.04, full BMC Sharpe above 0.45,
  maximum BMC drawdown below 0.15, and absolute average correlation with the
  official benchmark below 0.25.
- The candidate's full-period and last-200-era BMC means must each retain at
  least 80% of the corresponding TabM component mean. This prevents accepting a
  low-volatility blend that removes too much of the validated signal.
- Torch-to-NumPy raw inference parity must pass with `rtol=1e-5` and
  `atol=1e-6`. The deployed predictor must preserve the exact input index and
  emit one finite `prediction` column in `[0, 1]`.
- The complete two-model predictor must run on the current 7,058-row v5.3 live
  file with one CPU, no more than 4 GiB RAM, and under 10 minutes in the current
  production-compatible Python 3.12 package set.

## Frozen decision rule

Promote only the exact 50/50 MLP/TabM per-era-rank blend if every check passes.
There is no weight tuning, best-component fallback, seed substitution, or
second-rank transform. If any quality, correctness, portability, memory, or
runtime check fails, stop without producing an upload-ready model.

