# Predeclared consecutive-era deployment gate

Frozen before any `scale_disk_*` result existed on 2026-08-02.

## Runs

All runs use the full consecutive benchmark-covered disk feature store, the
scout winner's K64/3x512/ReLU/dropout-0.1 architecture, a 500,000-row training
cap, and 52-era inner and outer embargoes.

1. `scale_disk_tabm_k64_train500k`: model seed 1337, sample seed 1337.
2. `scale_disk_tabm_k64_train500k_seed2027`: model seed 2027, sample seed 1337.
3. `scale_disk_tabm_k64_train500k_sample_seed2027`: model seed 1337, sample seed 2027.

## Fail-closed checks

- Every artifact must have unique IDs, exact era/index alignment, complete
  benchmark coverage, finite targets/benchmarks/predictions, and the declared
  residual-signal prediction semantics.
- Every run must have positive full-period and last-200-era BMC. The median
  full-period and last-200-era BMC must also be positive, and target Corr must
  remain in the repository's plausible non-leakage range of 0.005 to 0.04.
- Every run must have full-period BMC Sharpe above 0.25, maximum BMC drawdown
  below 0.15, and absolute average correlation with the official benchmark
  below 0.25. The median BMC Sharpe must exceed 0.40. These are pass/fail
  guardrails, not post-result tuning targets.
- Recompute metrics after per-era percentile ranking, because that is the live
  submission transform. The frozen transform is pandas
  `groupby(era, sort=False).rank(method="average", pct=True)` on finite raw
  predictions, with missing eras or non-finite values rejected. A candidate
  that fails after ranking is rejected.
- NumPy inference must match the fitted Torch model within `rtol=1e-5` and
  `atol=1e-6`, preserve input index, emit one finite `prediction` column in
  `[0, 1]`, and pass the current Numerai runtime contract.
- Hosted inference must fit one CPU, 4 GiB RAM, and 10 minutes. No candidate is
  promoted without a measured live-size runtime check.

## Frozen selection rule

The preferred deployment candidate is the equal-weight mean of the three
models' per-era percentile ranks, provided all fail-closed checks pass and the
three-model bundle fits the hosted runtime budget. If the ensemble misses the
runtime budget, the fallback is the original model/sample seed 1337 candidate;
the best-looking seed will not be selected post hoc. If either choice fails a
quality or correctness check, deployment stops.

The matrix deliberately represents three predeclared stochastic realizations
from two independently varied randomness sources; it does not claim three
distinct numeric seed values. The exact source/config and disk-store metadata
hashes are recorded separately in `gate_source_manifest.json` before reading a
gate result.
