# Ender-20 Two-Seed Stability Experiment (v5.3)

**Started:** 2026-08-03
**State:** completed; no calibration-eligible candidate

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

The unchanged seed-2027 config completed all four scored OOF folds on CUDA. Its
early-stopping receipts were: fold 1 best epoch 2 of 6, fold 2 best epoch 1 of
5, fold 3 best epoch 14 of 18, and fold 4 best epoch 1 of 5. The generated
prediction artifact contains the exact expected 5,112,039 rows.

Before scoring, the evaluator verified the committed pre-training checkpoint,
the historical feature-store metadata anchor, exact manifest hash, fully
resolved model/config/result receipts, canonical prediction semantics, and both
prediction files against the independently derived ID/era/target/fold cohort.

Calibration results:

| signal | BMC mean | BMC Sharpe | BMC max drawdown | Corr | benchmark Spearman | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| benchmark only (reference) | 0.000000 | 0.0689 | 0.0000 | 0.041137 | 1.0000 | n/a |
| seed 1337 (reference) | 0.006431 | 0.5402 | 0.3010 | 0.011422 | 0.0986 | n/a |
| seed 2027 (reference) | 0.006090 | 0.5267 | 0.1684 | 0.010762 | 0.1022 | n/a |
| two-seed residual (reference) | 0.007243 | 0.6355 | 0.2432 | 0.012592 | 0.1051 | n/a |
| `two_seed_hybrid_w35` | 0.002787 | 0.4837 | 0.1216 | 0.037260 | 0.8972 | no |
| `two_seed_hybrid_w45` | 0.003797 | 0.5298 | 0.1468 | 0.035018 | 0.8012 | no |
| `two_seed_hybrid_w55` | 0.004763 | 0.5591 | 0.1734 | 0.032211 | 0.6665 | no |
| `two_seed_hybrid_w65` | 0.005555 | 0.5792 | 0.1949 | 0.028849 | 0.5209 | no |
| `two_seed_hybrid_w75` | 0.006199 | 0.5986 | 0.2129 | 0.025102 | 0.3848 | no |

The two-seed ensemble materially improved stability, but the frozen candidates
still do not overlap all calibration constraints:

- `two_seed_hybrid_w35` passed Sharpe, drawdown, target-Corr retention, and
  benchmark-similarity checks, but retained `38.4726%` of the two-seed
  residual's BMC versus the required inclusive `40%`.
- `two_seed_hybrid_w45` retained `52.4187%` of residual BMC and passed Sharpe
  and drawdown, but retained only `85.1239%` of benchmark target Corr versus the
  required inclusive `90%`.
- weights 0.55-0.75 failed both the target-Corr retention and strict `< 0.15`
  drawdown requirements.

The holdout was not allowed to choose a replacement. For context only, the w35
formula had holdout BMC `0.002588`, Sharpe `0.5541`, drawdown `0.0399`, and Corr
`0.022645`; those favorable values do not override its calibration failure.

Generated machine-readable outputs use content-addressed CSV filenames under
`results/`, with one atomic result pointer. Generation ID:
`489b334e7e75b86a124e`.

The compact, hash-verifying reader artifact is
[`ender20_seed_ensemble_stability_analysis.ipynb`](ender20_seed_ensemble_stability_analysis.ipynb).
It was executed successfully with all five code cells complete and no error
outputs.

## Independent validation

A separate bounded audit recomputed all 27 segment/candidate summaries from the
7,695-row per-era output and reproduced the recorded mean, standard deviation,
Sharpe, maximum drawdown, and symmetric benchmark-similarity values. It also
verified the exact 855-era chronology, 655/200 split, finite nine-signal
coverage, every recorded provenance and artifact hash, generation ID, immutable
output names, and atomic pointer. The audit independently reproduced zero
eligible candidates and the null selection, with no correctness blocker found.

## Reproduction

From the repository root with `PYTHONPATH=numerai`:

```powershell
.\.venv\Scripts\python.exe -m agents.code.modeling --config numerai\agents\experiments\ender20_nn_architecture_v53\configs\scale_disk_tabm_k64_train500k_seed2027.py
.\.venv\Scripts\python.exe numerai\agents\code\analysis\evaluate_ender20_seed_ensemble_stability.py
.\.venv\Scripts\python.exe numerai\agents\experiments\ender20_seed_ensemble_stability_v53\build_analysis_notebook.py --execute
```

Key SHA-256 receipts:

- seed-2027 OOF predictions: `58027368888ba806383003acb8cdbcc6252223b0b7539537c66d7cedd94601e4`
- seed-2027 result JSON: `0fd6c896b6536ee5038ffec0884ebe3a7586bf9322028d7c5b21ff4d9958b1db`
- frozen gate: `3097219be90c4fb49d07a461c129e8121942e49a74d9af75397cfe7eec841cc4`
- result pointer: `960b299f85bc68dfdb6c84a88d38008058b6f6c38aee8cfc0887850ed03bf95c`

## Decision

`NOT_PROMOTION_ELIGIBLE`. This was the predeclared final stability iteration,
so the current benchmark-blend approach stops without another seed, weight, or
threshold search. No hybrid pickle was packaged and nothing was uploaded to
Numerai. The original single-seed TabM pickle remains experimental and is not a
production-approved upload candidate.
