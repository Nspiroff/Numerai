# Xerxes-20 LightGBM Challenger (v5.3)

**Started:** 2026-08-03

**Completed:** 2026-08-03

**State:** `STOP_NO_SCOUT_CALIBRATION_WINNER`

## Abstract

This experiment tested four fixed LightGBM capacity profiles trained directly
on `target_xerxes_20` with Numerai's 780 medium features. The models were scored
on `target_ender_20` against `v53_lgbm_ender20` under a gate frozen before any
Xerxes scout result was computed. None cleared calibration eligibility: every
profile missed the strict BMC Sharpe `> 0.20` threshold, and the 2,000-tree
profile also missed BMC mean `> 0.0010`.

The family therefore stopped without selecting a scout, opening the locked
50-era metrics, running a consecutive confirmation, packaging a model, or
uploading anything to Numerai.

## Motivation

The preceding TabM line retained unique Ender20 signal but did not satisfy its
stability gate. This challenger deliberately changed both the training target
and inductive bias: a direct auxiliary-target tree model instead of another
benchmark-residual neural network. The checked-in target-ensemble tutorial
provided motivation for `target_xerxes_20`, but not independent evidence for
this exact medium-feature, leakage-safe protocol.

## Method

The complete source definitions, four scout profiles, chronological split,
thresholds, deterministic winner rule, confirmation boundary, and stop rule are
frozen in [`gate.md`](gate.md). Exact source and runtime receipts are recorded in
[`source_manifest.json`](source_manifest.json) and
[`gpu_runtime.json`](gpu_runtime.json).

All scouts used GPU LightGBM, `colsample_bytree=0.1`,
`min_data_in_leaf=10000`, a deterministic 500,000-row training cap, seed 1337,
and fixed-tree training. Four expanding folds produced 1,279,658 OOF rows over
214 every-fourth eras from `0373` through `1225`. Selection used only the first
164 eras (`0373`-`1025`); the final 50 eras remained locked because no scout was
calibration-eligible.

Calibration required all of the following: Ender BMC mean `> 0.0010`, BMC
Sharpe `> 0.20`, BMC maximum drawdown `< 0.15`, Ender Corr `> 0.010`, and
average symmetric benchmark Spearman `< 0.85`.

## Calibration results

| scout | trees | depth | BMC mean | BMC Sharpe | BMC max drawdown | Ender Corr | benchmark Spearman | failed checks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `r1_base_d6_t6000` | 6,000 | 6 | 0.001613 | 0.168329 | 0.042231 | 0.021567 | 0.451152 | BMC Sharpe |
| `r1_trees2k` | 2,000 | 6 | 0.000402 | 0.042282 | 0.069234 | 0.017667 | 0.392992 | BMC mean, BMC Sharpe |
| `r1_depth5` | 6,000 | 5 | 0.001225 | 0.128079 | 0.048439 | 0.020754 | 0.441641 | BMC Sharpe |
| `r1_depth8` | 6,000 | 8 | 0.001659 | 0.175127 | 0.042528 | 0.021912 | 0.458295 | BMC Sharpe |

`r1_depth8` had the highest BMC mean and Ender Corr, but it was not eligible:
its BMC Sharpe was `0.175127`, below the strict `> 0.20` screen. No post-result
replacement or alternate capacity search is permitted by the frozen stop rule.

## Independent validation

The executed reader notebook is
[`xerxes20_lgbm_challenger_analysis.ipynb`](xerxes20_lgbm_challenger_analysis.ipynb).
It performs no training and does not open prediction parquets or per-run result
JSONs. It verifies the canonical/content-addressed evaluator result, summary,
calibration per-era output, runtime receipt, source manifest, evaluator, and all
four configs by SHA-256.

The notebook independently checked 656 saved calibration rows (164 exact eras
for each of four scouts), rejected duplicate or non-finite rows, confirmed that
no locked phase appears, recomputed every reported calibration metric within
`1e-12`, and reproduced each eligibility decision. All four code cells executed
without errors under Python 3.12.13 in `numerai-lgbm-gpu312`. Two consecutive
executed builds produced the same notebook hash.

A separate read-only audit then recomputed Corr, BMC mean, BMC Sharpe, BMC
maximum drawdown, and symmetric benchmark Spearman directly from each sealed
prediction artifact and the raw Ender/benchmark sources. It was restricted to
the same 164 calibration eras through `1025`, matched every evaluator value to
floating-point roundoff, reproduced the empty eligible set, and did not read
the locked 50 eras.

## Decision and stopping rationale

`NOT_PROMOTION_ELIGIBLE`. The frozen selection set is empty, so the evaluator
correctly emitted `STOP_NO_SCOUT_CALIBRATION_WINNER`. The locked scout slice and
confirmation gate were not opened. No confirmation config, final fit, pickle,
Docker validation, upload, assignment, API submission, or staking action was
performed.

Further work, if authorized, should begin as a new predeclared model family or
research hypothesis rather than post hoc tuning of this stopped Xerxes20 sweep.

## Reproduction

From the repository root in PowerShell, the historical scout runs are reproduced
sequentially with the pinned GPU environment:

```powershell
$env:PYTHONPATH = "numerai"
$conda = "$env:USERPROFILE\miniforge3\Scripts\conda.exe"
$configs = @(
    "numerai\agents\experiments\xerxes20_lgbm_challenger_v53\configs\r1_base_d6_t6000.py",
    "numerai\agents\experiments\xerxes20_lgbm_challenger_v53\configs\r1_trees2k.py",
    "numerai\agents\experiments\xerxes20_lgbm_challenger_v53\configs\r1_depth5.py",
    "numerai\agents\experiments\xerxes20_lgbm_challenger_v53\configs\r1_depth8.py"
)
foreach ($config in $configs) {
    & $conda run -n numerai-lgbm-gpu312 --no-capture-output `
        python -m agents.code.modeling --config $config
}
```

Reproduce the frozen calibration evaluation and then rebuild the executed reader:

```powershell
$env:PYTHONPATH = "numerai"
$conda = "$env:USERPROFILE\miniforge3\Scripts\conda.exe"
& $conda run -n numerai-lgbm-gpu312 --no-capture-output `
    python -m agents.code.analysis.evaluate_xerxes20_lgbm_challenger `
    --pretraining-commit 2639d98140c99ec501b56e8c2f8b0419b90f9852
# A non-zero exit is the expected, valid frozen-gate stop for this result.
& $conda run -n numerai-lgbm-gpu312 --no-capture-output `
    python numerai\agents\experiments\xerxes20_lgbm_challenger_v53\build_analysis_notebook.py --execute
```

## Frozen hashes and checkpoints

- Pre-scoring protocol commit: `0f892c712c870bfd97eb7735eec944f3f2c60d2f`
- Pretraining implementation commit: `2639d98140c99ec501b56e8c2f8b0419b90f9852`
- Evaluator: `d6b3e0290a28e4fbd12376227f2dcbda21defe6263bb595e51042c139097568c`
- Source manifest: `4b3dd7e30dbcb8e532ffbdd484031c98efc30cf4d82804290f62846e19675a8d`
- GPU runtime receipt: `d6656bc39a0d603860c9b327569bd453b1556b8a3aae99f8567edefbc214f135`
- Canonical and content-addressed evaluator result: `c5939fc19c57688788fc2fdd2e28e8a49e99394ecab5aac019ddf1069cd62c6d`
- Calibration summary CSV: `9eacdcdcf5a94f9b9fc0194173efd6588fe5cf37588bda77ff2690273c3fa862`
- Calibration per-era CSV: `41259804c93fca8d57ed6eeb72bf597e4ce558ac14dfea6223a1e4baa89b4cf2`
- Notebook builder: `6ae0341ceea16aa5e6c0fa99fdce77258c3f1644ef5bc0307c9c10d19b6a2f62`
- Executed notebook: `cbcfeef6cb9c76045caee142d0976d06a3aac4b044e3555cedb88bab848c628d`

The four config hashes and sealed scout prediction/result receipts remain in
the canonical evaluator result. The notebook verifies the config files but
intentionally does not reopen sealed prediction or per-run result artifacts.
