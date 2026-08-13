"""Evaluate the frozen Ender20 benchmark/TabM hybrid stability experiment.

This module is deliberately training-free.  It joins the already validated
TabM out-of-fold predictions to the exact v5.3 feature-store manifest, builds
only the five blends frozen in ``ender20_hybrid_stability_v53/gate.md``, selects
on the first 655 eras, and evaluates the untouched final 200 eras.  Integrity
or coverage failures stop the run instead of producing a partial result.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from agents.code.metrics import numerai_metrics


EXPERIMENT_NAME = "ender20_hybrid_stability_v53"
AS_OF_DATE = "2026-08-03"

ID_COLUMN = "id"
ERA_COLUMN = "era"
TARGET_COLUMN = "target_ender_20"
BENCHMARK_COLUMN = "v53_lgbm_ender20"
RESIDUAL_COLUMN = "prediction"
FOLD_COLUMN = "cv_fold"

EXPECTED_ROWS = 5_112_039
EXPECTED_ERAS = 855
EXPECTED_FIRST_ERA = "0371"
EXPECTED_LAST_ERA = "1225"
CALIBRATION_ERAS = 655
HOLDOUT_ERAS = 200

WEIGHTS: dict[str, float] = {
    "hybrid_w35": 0.35,
    "hybrid_w45": 0.45,
    "hybrid_w55": 0.55,
    "hybrid_w65": 0.65,
    "hybrid_w75": 0.75,
}

REFERENCE_COLUMNS = ("benchmark_only", "residual_only")
CANDIDATE_COLUMNS = tuple(WEIGHTS)
ALL_SIGNAL_COLUMNS = (*REFERENCE_COLUMNS, *CANDIDATE_COLUMNS)

CALIBRATION_THRESHOLDS = {
    "bmc_mean_min_exclusive": 0.0,
    "residual_bmc_retention_min_inclusive": 0.40,
    "bmc_sharpe_min_exclusive": 0.45,
    "bmc_max_drawdown_max_exclusive": 0.15,
    "benchmark_corr_retention_min_inclusive": 0.90,
    "avg_corr_with_benchmark_max_exclusive": 0.95,
}

PROMOTION_THRESHOLDS = {
    "holdout_bmc_mean_min_exclusive": 0.0,
    "holdout_residual_bmc_retention_min_inclusive": 0.35,
    "holdout_bmc_sharpe_min_exclusive": 0.35,
    "holdout_bmc_max_drawdown_max_exclusive": 0.15,
    "holdout_benchmark_corr_retention_min_inclusive": 0.90,
    "full_bmc_mean_min_exclusive": 0.0,
    "last_200_bmc_mean_min_exclusive": 0.0,
    "full_bmc_sharpe_min_exclusive": 0.45,
    "full_bmc_max_drawdown_max_exclusive": 0.15,
    "full_corr_mean_min_inclusive": 0.005,
    "full_corr_mean_max_inclusive": 0.04,
    "full_avg_corr_with_benchmark_max_exclusive": 0.95,
}

EXPECTED_PREDICTION_SEMANTICS = {
    "artifact_kind": "out_of_fold_validation",
    "column": RESIDUAL_COLUMN,
    "era_column": ERA_COLUMN,
    "fold_column": FOLD_COLUMN,
    "fold_index_base": 0,
    "inverse_target_transform_applied": False,
    "pipeline_postprocess": {"type": "identity"},
    "producer": "model.predict",
    "schema_version": 1,
    "stored_target": {
        "column": TARGET_COLUMN,
        "transform": {"type": "identity"},
    },
    "training_target": {
        "column": TARGET_COLUMN,
        "transform": {
            "benchmark_col": BENCHMARK_COLUMN,
            "era_col": ERA_COLUMN,
            "fit_intercept": True,
            "per_era": True,
            "type": "residual_to_benchmark",
        },
    },
}


class HybridEvaluationError(ValueError):
    """Raised when frozen-input integrity or evaluation contracts fail."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HybridEvaluationError(message)


def _at_least(value: float, threshold: float) -> bool:
    """Inclusive comparison without rejecting an exact decimal at float noise."""

    return bool(value >= threshold or np.isclose(value, threshold, rtol=1e-12, atol=1e-15))


def _at_most(value: float, threshold: float) -> bool:
    """Inclusive comparison without rejecting an exact decimal at float noise."""

    return bool(value <= threshold or np.isclose(value, threshold, rtol=1e-12, atol=1e-15))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_prediction_semantics(path: Path) -> dict[str, Any]:
    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(b"numerai.agents.prediction_semantics")
    _require(raw is not None, "OOF parquet is missing prediction-semantics metadata.")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HybridEvaluationError(
            "OOF prediction-semantics metadata is not valid UTF-8 JSON."
        ) from error
    _require(
        value == EXPECTED_PREDICTION_SEMANTICS,
        "OOF prediction semantics do not match the frozen residual contract.",
    )
    return value


def resolve_manifest(path_or_glob: str | Path) -> Path:
    value = Path(path_or_glob)
    if any(char in str(value) for char in "*?["):
        matches = sorted(value.parent.glob(value.name))
        _require(
            len(matches) == 1,
            f"Expected exactly one feature-store manifest, found {len(matches)}.",
        )
        return matches[0].resolve()
    _require(value.is_file(), f"Feature-store manifest not found: {value}")
    return value.resolve()


def load_and_validate_cohort(
    predictions_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Join the frozen OOF rows to the exact manifest and validate alignment."""

    _require(predictions_path.is_file(), f"OOF predictions not found: {predictions_path}")
    _require(manifest_path.is_file(), f"Manifest not found: {manifest_path}")
    semantics = _read_prediction_semantics(predictions_path)

    prediction_columns = [
        ID_COLUMN,
        ERA_COLUMN,
        TARGET_COLUMN,
        RESIDUAL_COLUMN,
        FOLD_COLUMN,
    ]
    manifest_columns = [ID_COLUMN, ERA_COLUMN, TARGET_COLUMN, BENCHMARK_COLUMN]
    oof = pd.read_parquet(predictions_path, columns=prediction_columns)
    manifest = pd.read_parquet(manifest_path, columns=manifest_columns)

    _require(len(oof) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS:,} OOF rows; got {len(oof):,}.")
    _require(oof[ID_COLUMN].notna().all(), "OOF IDs contain missing values.")
    _require(oof[ID_COLUMN].is_unique, "OOF IDs are not unique.")
    _require(oof[ERA_COLUMN].notna().all(), "OOF eras contain missing values.")
    _require(
        np.isfinite(oof[[TARGET_COLUMN, RESIDUAL_COLUMN]].to_numpy(dtype=np.float64)).all(),
        "OOF targets or residual predictions contain non-finite values.",
    )
    _require(oof[FOLD_COLUMN].notna().all(), "OOF fold assignments contain missing values.")

    oof = oof.reset_index(drop=True)
    oof["_oof_order"] = np.arange(len(oof), dtype=np.int64)
    manifest = manifest.rename(
        columns={
            ERA_COLUMN: "_manifest_era",
            TARGET_COLUMN: "_manifest_target",
        }
    )
    try:
        cohort = oof.merge(
            manifest,
            how="left",
            on=ID_COLUMN,
            sort=False,
            validate="one_to_one",
            indicator=True,
        )
    except pd.errors.MergeError as error:
        raise HybridEvaluationError(
            "OOF and manifest IDs do not form an exact one-to-one mapping."
        ) from error
    finally:
        del manifest
        gc.collect()

    _require((cohort["_merge"] == "both").all(), "At least one OOF ID is absent from the manifest.")
    cohort = cohort.sort_values("_oof_order", kind="stable").reset_index(drop=True)
    _require(
        np.array_equal(
            cohort[ERA_COLUMN].astype(str).to_numpy(),
            cohort["_manifest_era"].astype(str).to_numpy(),
        ),
        "OOF and manifest eras differ for at least one ID.",
    )
    _require(
        np.array_equal(
            cohort[TARGET_COLUMN].to_numpy(), cohort["_manifest_target"].to_numpy()
        ),
        "OOF and manifest targets differ for at least one ID.",
    )
    _require(
        np.isfinite(cohort[BENCHMARK_COLUMN].to_numpy(dtype=np.float64)).all(),
        "Manifest benchmark predictions contain non-finite values.",
    )

    cohort = cohort.drop(
        columns=["_oof_order", "_manifest_era", "_manifest_target", "_merge"]
    )
    era_fold_counts = cohort.groupby(ERA_COLUMN, sort=False)[FOLD_COLUMN].nunique()
    _require((era_fold_counts == 1).all(), "At least one era spans multiple OOF folds.")

    eras = sorted(cohort[ERA_COLUMN].astype(str).unique().tolist(), key=int)
    _require(len(eras) == EXPECTED_ERAS, f"Expected {EXPECTED_ERAS} eras; got {len(eras)}.")
    _require(eras[0] == EXPECTED_FIRST_ERA, f"Unexpected first OOF era: {eras[0]}.")
    _require(eras[-1] == EXPECTED_LAST_ERA, f"Unexpected last OOF era: {eras[-1]}.")
    _require(
        len(eras[:CALIBRATION_ERAS]) == CALIBRATION_ERAS
        and len(eras[CALIBRATION_ERAS:]) == HOLDOUT_ERAS,
        "Frozen calibration/holdout split is not 655/200 eras.",
    )
    return cohort, eras, semantics


def rank_hybrid_signals(
    frame: pd.DataFrame,
    weights: Mapping[str, float] = WEIGHTS,
) -> pd.DataFrame:
    """Create the two references and frozen per-era rank blends."""

    required = {ERA_COLUMN, BENCHMARK_COLUMN, RESIDUAL_COLUMN}
    _require(required.issubset(frame.columns), f"Missing blend columns: {sorted(required - set(frame.columns))}")
    result = frame.copy()
    grouped = result.groupby(ERA_COLUMN, sort=False, observed=True)
    result["benchmark_only"] = grouped[BENCHMARK_COLUMN].rank(
        method="average", pct=True
    )
    result["residual_only"] = grouped[RESIDUAL_COLUMN].rank(
        method="average", pct=True
    )
    for name, residual_weight in weights.items():
        _require(0.0 < float(residual_weight) < 1.0, f"Invalid residual weight: {residual_weight}")
        raw_score = (
            (1.0 - float(residual_weight)) * result["benchmark_only"]
            + float(residual_weight) * result["residual_only"]
        )
        result[name] = raw_score.groupby(result[ERA_COLUMN], sort=False).rank(
            method="average", pct=True
        )
    generated_columns = (*REFERENCE_COLUMNS, *tuple(weights))
    values = result[list(generated_columns)].to_numpy(dtype=np.float64)
    _require(np.isfinite(values).all(), "Generated signals contain non-finite values.")
    _require(((values >= 0.0) & (values <= 1.0)).all(), "Generated signals leave [0, 1].")
    return result


def _validate_per_era_scores(scores: pd.DataFrame, eras: Sequence[str], label: str) -> pd.DataFrame:
    result = scores.copy()
    result.index = result.index.astype(str)
    result = result.loc[list(eras), list(ALL_SIGNAL_COLUMNS)]
    _require(result.index.tolist() == list(eras), f"{label} eras are not in frozen order.")
    _require(np.isfinite(result.to_numpy(dtype=np.float64)).all(), f"{label} contains non-finite values.")
    return result


def compute_per_era_metrics(
    cohort: pd.DataFrame,
    eras: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Compute Numerai Corr, BMC, and benchmark similarity once for all signals."""

    corr = numerai_metrics.per_era_corr(
        cohort, ALL_SIGNAL_COLUMNS, TARGET_COLUMN, ERA_COLUMN
    )
    bmc = numerai_metrics.per_era_bmc(
        cohort, ALL_SIGNAL_COLUMNS, BENCHMARK_COLUMN, TARGET_COLUMN, ERA_COLUMN
    )
    benchmark_corr = per_era_rank_similarity(
        cohort,
        ALL_SIGNAL_COLUMNS,
        BENCHMARK_COLUMN,
        ERA_COLUMN,
    )
    return {
        "corr": _validate_per_era_scores(corr, eras, "Per-era Corr"),
        "bmc": _validate_per_era_scores(bmc, eras, "Per-era BMC"),
        "benchmark_corr": _validate_per_era_scores(
            benchmark_corr, eras, "Per-era benchmark correlation"
        ),
    }


def per_era_rank_similarity(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    reference_column: str,
    era_column: str = ERA_COLUMN,
) -> pd.DataFrame:
    """Symmetric per-era Spearman correlation between prediction signals.

    Numerai Corr is intentionally asymmetric because its second argument is a
    target.  Prediction-vs-prediction similarity instead requires the same
    transform on both sides; Pearson correlation of average tie ranks is the
    standard Spearman definition and makes a reference compared with itself
    exactly one.
    """

    signals = list(signal_columns)
    columns_to_rank = list(dict.fromkeys([*signals, reference_column]))
    required = {era_column, *columns_to_rank}
    _require(
        required.issubset(frame.columns),
        f"Missing similarity columns: {sorted(required - set(frame.columns))}",
    )
    ranked = frame.groupby(era_column, sort=False, observed=True)[columns_to_rank].rank(
        method="average", pct=True
    )
    ranked[era_column] = frame[era_column].to_numpy()
    eras: list[str] = []
    rows: list[np.ndarray] = []
    for era, group in ranked.groupby(era_column, sort=False, observed=True):
        predictions = group[signals].to_numpy(dtype=np.float64)
        reference = group[reference_column].to_numpy(dtype=np.float64)
        predictions = predictions - predictions.mean(axis=0, keepdims=True)
        reference = reference - reference.mean()
        numerator = predictions.T @ reference
        denominator = np.sqrt(
            np.square(predictions).sum(axis=0) * np.square(reference).sum()
        )
        _require((denominator > 0.0).all(), f"Constant prediction signal in era {era}.")
        rows.append(numerator / denominator)
        eras.append(str(era))
    result = pd.DataFrame(rows, index=eras, columns=signals)
    _require(
        np.isfinite(result.to_numpy(dtype=np.float64)).all(),
        "Per-era prediction similarity contains non-finite values.",
    )
    return result


def _score_summary(scores: pd.Series) -> dict[str, float]:
    values = scores.to_numpy(dtype=np.float64)
    _require(values.size > 0 and np.isfinite(values).all(), "Cannot summarize empty/non-finite scores.")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    return {
        "mean": mean,
        "std": std,
        "sharpe": float(mean / std) if std != 0.0 else 0.0,
        "max_drawdown": float(numerai_metrics.max_drawdown(pd.Series(values))),
    }


def summarize_segments(
    per_era: Mapping[str, pd.DataFrame],
    eras: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    segments = {
        "calibration": list(eras[:CALIBRATION_ERAS]),
        "holdout": list(eras[CALIBRATION_ERAS:]),
        "full": list(eras),
    }
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for segment_name, segment_eras in segments.items():
        candidates: dict[str, dict[str, Any]] = {}
        for candidate in ALL_SIGNAL_COLUMNS:
            candidates[candidate] = {
                "era_count": len(segment_eras),
                "corr": _score_summary(per_era["corr"].loc[segment_eras, candidate]),
                "bmc": _score_summary(per_era["bmc"].loc[segment_eras, candidate]),
                "avg_corr_with_benchmark": float(
                    per_era["benchmark_corr"].loc[segment_eras, candidate].mean()
                ),
            }
        output[segment_name] = candidates
    return output


def calibration_checks(
    candidate_metrics: Mapping[str, Any],
    residual_bmc_mean: float,
    benchmark_corr_mean: float,
) -> dict[str, bool]:
    bmc = candidate_metrics["bmc"]
    return {
        "bmc_mean_positive": float(bmc["mean"]) > 0.0,
        "residual_bmc_retention": _at_least(
            float(bmc["mean"]),
            CALIBRATION_THRESHOLDS["residual_bmc_retention_min_inclusive"]
            * residual_bmc_mean,
        ),
        "bmc_sharpe": float(bmc["sharpe"])
        > CALIBRATION_THRESHOLDS["bmc_sharpe_min_exclusive"],
        "bmc_max_drawdown": float(bmc["max_drawdown"])
        < CALIBRATION_THRESHOLDS["bmc_max_drawdown_max_exclusive"],
        "benchmark_corr_retention": _at_least(
            float(candidate_metrics["corr"]["mean"]),
            CALIBRATION_THRESHOLDS["benchmark_corr_retention_min_inclusive"]
            * benchmark_corr_mean,
        ),
        "benchmark_similarity": float(candidate_metrics["avg_corr_with_benchmark"])
        < CALIBRATION_THRESHOLDS["avg_corr_with_benchmark_max_exclusive"],
    }


def select_calibration_candidate(
    summaries: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    calibration = summaries["calibration"]
    residual_bmc_mean = float(calibration["residual_only"]["bmc"]["mean"])
    benchmark_corr_mean = float(calibration["benchmark_only"]["corr"]["mean"])
    evaluations: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    for candidate in CANDIDATE_COLUMNS:
        checks = calibration_checks(
            calibration[candidate], residual_bmc_mean, benchmark_corr_mean
        )
        is_eligible = all(checks.values())
        evaluations[candidate] = {"eligible": is_eligible, "checks": checks}
        if is_eligible:
            eligible.append(candidate)
    if not eligible:
        return None, evaluations
    selected = sorted(
        eligible,
        key=lambda name: (
            -float(calibration[name]["bmc"]["mean"]),
            float(calibration[name]["bmc"]["max_drawdown"]),
            float(WEIGHTS[name]),
        ),
    )[0]
    return selected, evaluations


def promotion_checks(
    selected: str,
    summaries: Mapping[str, Mapping[str, Mapping[str, Any]]],
    coverage_ok: bool,
) -> dict[str, bool]:
    holdout = summaries["holdout"]
    full = summaries["full"]
    selected_holdout = holdout[selected]
    selected_full = full[selected]
    residual_holdout_bmc = float(holdout["residual_only"]["bmc"]["mean"])
    benchmark_holdout_corr = float(holdout["benchmark_only"]["corr"]["mean"])
    return {
        "exact_finite_coverage": bool(coverage_ok),
        "holdout_bmc_mean_positive": float(selected_holdout["bmc"]["mean"]) > 0.0,
        "holdout_residual_bmc_retention": _at_least(
            float(selected_holdout["bmc"]["mean"]),
            PROMOTION_THRESHOLDS["holdout_residual_bmc_retention_min_inclusive"]
            * residual_holdout_bmc,
        ),
        "holdout_bmc_sharpe": float(selected_holdout["bmc"]["sharpe"])
        > PROMOTION_THRESHOLDS["holdout_bmc_sharpe_min_exclusive"],
        "holdout_bmc_max_drawdown": float(selected_holdout["bmc"]["max_drawdown"])
        < PROMOTION_THRESHOLDS["holdout_bmc_max_drawdown_max_exclusive"],
        "holdout_benchmark_corr_retention": _at_least(
            float(selected_holdout["corr"]["mean"]),
            PROMOTION_THRESHOLDS["holdout_benchmark_corr_retention_min_inclusive"]
            * benchmark_holdout_corr,
        ),
        "full_bmc_mean_positive": float(selected_full["bmc"]["mean"]) > 0.0,
        "last_200_bmc_mean_positive": float(selected_holdout["bmc"]["mean"]) > 0.0,
        "full_bmc_sharpe": float(selected_full["bmc"]["sharpe"])
        > PROMOTION_THRESHOLDS["full_bmc_sharpe_min_exclusive"],
        "full_bmc_max_drawdown": float(selected_full["bmc"]["max_drawdown"])
        < PROMOTION_THRESHOLDS["full_bmc_max_drawdown_max_exclusive"],
        "full_corr_lower_bound": _at_least(
            float(selected_full["corr"]["mean"]),
            PROMOTION_THRESHOLDS["full_corr_mean_min_inclusive"],
        ),
        "full_corr_upper_bound": _at_most(
            float(selected_full["corr"]["mean"]),
            PROMOTION_THRESHOLDS["full_corr_mean_max_inclusive"],
        ),
        "full_benchmark_similarity": float(selected_full["avg_corr_with_benchmark"])
        < PROMOTION_THRESHOLDS["full_avg_corr_with_benchmark_max_exclusive"],
    }


def _write_summary_csv(
    path: Path,
    summaries: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    rows: list[dict[str, Any]] = []
    for segment, candidates in summaries.items():
        for candidate, metrics in candidates.items():
            rows.append(
                {
                    "segment": segment,
                    "candidate": candidate,
                    "selectable": candidate in CANDIDATE_COLUMNS,
                    "residual_weight": WEIGHTS.get(candidate),
                    "era_count": metrics["era_count"],
                    "corr_mean": metrics["corr"]["mean"],
                    "corr_std": metrics["corr"]["std"],
                    "corr_sharpe": metrics["corr"]["sharpe"],
                    "bmc_mean": metrics["bmc"]["mean"],
                    "bmc_std": metrics["bmc"]["std"],
                    "bmc_sharpe": metrics["bmc"]["sharpe"],
                    "bmc_max_drawdown": metrics["bmc"]["max_drawdown"],
                    "avg_corr_with_benchmark": metrics["avg_corr_with_benchmark"],
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_per_era_csv(path: Path, per_era: Mapping[str, pd.DataFrame]) -> None:
    rows: list[pd.DataFrame] = []
    for candidate in ALL_SIGNAL_COLUMNS:
        rows.append(
            pd.DataFrame(
                {
                    "era": per_era["bmc"].index.astype(str),
                    "candidate": candidate,
                    "corr": per_era["corr"][candidate].to_numpy(),
                    "bmc": per_era["bmc"][candidate].to_numpy(),
                    "benchmark_corr": per_era["benchmark_corr"][candidate].to_numpy(),
                }
            )
        )
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)


def evaluate(
    predictions_path: Path,
    manifest_path: Path,
    gate_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    repo_root = _repo_root()
    cohort, eras, semantics = load_and_validate_cohort(predictions_path, manifest_path)
    cohort = rank_hybrid_signals(cohort)
    coverage_ok = (
        len(cohort) == EXPECTED_ROWS
        and cohort[ID_COLUMN].is_unique
        and np.isfinite(cohort[list(ALL_SIGNAL_COLUMNS)].to_numpy(dtype=np.float64)).all()
    )
    per_era = compute_per_era_metrics(cohort, eras)
    summaries = summarize_segments(per_era, eras)
    selected, candidate_evaluations = select_calibration_candidate(summaries)
    checks = promotion_checks(selected, summaries, coverage_ok) if selected else {}
    promotion_eligible = selected is not None and all(checks.values())

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "hybrid_stability_summary.csv"
    per_era_path = output_dir / "hybrid_stability_per_era.csv"
    result_path = output_dir / "hybrid_stability_result.json"
    _write_summary_csv(summary_path, summaries)
    _write_per_era_csv(per_era_path, per_era)

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "as_of_date": AS_OF_DATE,
        "state": (
            "PROMOTION_ELIGIBLE_LOCAL_PACKAGING_ONLY"
            if promotion_eligible
            else "NOT_PROMOTION_ELIGIBLE"
        ),
        "promotion_eligible": promotion_eligible,
        "selected_candidate": selected,
        "selected_residual_weight": WEIGHTS.get(selected) if selected else None,
        "deployment_boundary": "No upload or staking is authorized by this result.",
        "inputs": {
            "predictions": {
                "path": _relative_path(predictions_path, repo_root),
                "sha256": _sha256_file(predictions_path),
            },
            "manifest": {
                "path": _relative_path(manifest_path, repo_root),
                "sha256": _sha256_file(manifest_path),
            },
            "gate": {
                "path": _relative_path(gate_path, repo_root),
                "sha256": _sha256_file(gate_path),
            },
            "prediction_semantics": semantics,
        },
        "cohort": {
            "rows": len(cohort),
            "eras": len(eras),
            "first_era": eras[0],
            "last_era": eras[-1],
            "calibration_eras": CALIBRATION_ERAS,
            "holdout_eras": HOLDOUT_ERAS,
            "first_holdout_era": eras[CALIBRATION_ERAS],
        },
        "weights": WEIGHTS,
        "calibration_thresholds": CALIBRATION_THRESHOLDS,
        "promotion_thresholds": PROMOTION_THRESHOLDS,
        "calibration_candidates": candidate_evaluations,
        "promotion_checks": checks,
        "summaries": summaries,
        "outputs": {
            "summary_csv": _relative_path(summary_path, repo_root),
            "per_era_csv": _relative_path(per_era_path, repo_root),
        },
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": str(result_path),
                "state": result["state"],
                "selected_candidate": selected,
                "promotion_eligible": promotion_eligible,
            },
            sort_keys=True,
        )
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = _repo_root()
    experiment_dir = repo_root / "numerai/agents/experiments" / EXPERIMENT_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=repo_root
        / "numerai/agents/experiments/ender20_nn_architecture_v53/predictions/"
        "scale_disk_tabm_k64_train500k.parquet",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root
        / "numerai/v5.3/target_ender_20_feature_store/manifest-*.parquet",
    )
    parser.add_argument("--gate", type=Path, default=experiment_dir / "gate.md")
    parser.add_argument("--output-dir", type=Path, default=experiment_dir / "results")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = resolve_manifest(args.manifest)
    evaluate(
        args.predictions.resolve(),
        manifest,
        args.gate.resolve(),
        args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
