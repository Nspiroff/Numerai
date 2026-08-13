"""Fail-closed evaluator for the frozen Ender20 TabM deployment gate.

The evaluator deliberately does not train, fit, export, or upload a model.  It
validates the three predeclared OOF result/prediction pairs against the frozen
source manifest and feature-store cohort, recomputes raw and live-ranked
metrics, and evaluates the predeclared quality thresholds.  NumPy parity and
hosted-runtime checks are reported as required external checks; an artifact
quality pass is never represented as deployment approval.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from agents.code.metrics import numerai_metrics


GATE_NAME = "ender20_nn_architecture_v53"
SOURCE_MANIFEST_SHA256 = (
    "2598382b910fcfe8c3f9adc0fce909cdbc8cadb69a4d36ebf85f24b8b808f7a1"
)
DEFAULT_SOURCE_MANIFEST = Path(
    "agents/experiments/ender20_nn_architecture_v53/gate_source_manifest.json"
)
STORE_METADATA_RELATIVE = Path(
    "v5.3/target_ender_20_feature_store/metadata.json"
)
PREDICTION_SEMANTICS_METADATA_KEY = b"numerai.agents.prediction_semantics"

ID_COLUMN = "id"
ERA_COLUMN = "era"
TARGET_COLUMN = "target_ender_20"
BENCHMARK_COLUMN = "v53_lgbm_ender20"
PREDICTION_COLUMN = "prediction"
FOLD_COLUMN = "cv_fold"

RUN_SPECS: dict[str, dict[str, Any]] = {
    "scale_disk_tabm_k64_train500k": {
        "config": "agents/experiments/ender20_nn_architecture_v53/configs/"
        "scale_disk_tabm_k64_train500k.py",
        "model_seed": 1337,
        "sample_seed": 1337,
    },
    "scale_disk_tabm_k64_train500k_seed2027": {
        "config": "agents/experiments/ender20_nn_architecture_v53/configs/"
        "scale_disk_tabm_k64_train500k_seed2027.py",
        "model_seed": 2027,
        "sample_seed": 1337,
    },
    "scale_disk_tabm_k64_train500k_sample_seed2027": {
        "config": "agents/experiments/ender20_nn_architecture_v53/configs/"
        "scale_disk_tabm_k64_train500k_sample_seed2027.py",
        "model_seed": 1337,
        "sample_seed": 2027,
    },
}

STRICT_THRESHOLDS = {
    "bmc_mean_min_exclusive": 0.0,
    "bmc_last_200_mean_min_exclusive": 0.0,
    "corr_mean_min_inclusive": 0.005,
    "corr_mean_max_inclusive": 0.04,
    "bmc_sharpe_min_exclusive": 0.25,
    "bmc_max_drawdown_max_exclusive": 0.15,
    "abs_avg_corr_with_benchmark_max_exclusive": 0.25,
    "median_bmc_sharpe_min_exclusive": 0.40,
}


class GateEvaluationError(ValueError):
    """An integrity or contract failure that prevents quality evaluation."""


@dataclass(frozen=True)
class RunArtifact:
    name: str
    result_path: Path
    predictions_path: Path


@dataclass(frozen=True)
class FrozenSource:
    repo_root: Path
    source_manifest_path: Path
    source_manifest: dict[str, Any]
    source_manifest_sha256: str
    store_metadata_path: Path
    store_metadata: dict[str, Any]
    store_manifest_path: Path
    configs: dict[str, dict[str, Any]]
    recorded_commit_present: bool


@dataclass(frozen=True)
class ExpectedCohort:
    frame: pd.DataFrame
    full_rows: int
    full_eras: int
    oof_rows: int
    oof_eras: int
    first_full_era: str
    last_full_era: str
    first_oof_era: str
    last_oof_era: str
    folds: tuple[dict[str, int], ...]
    internal_folds_by_run: dict[str, tuple[dict[str, int], ...]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateEvaluationError(message)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_normalize(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise GateEvaluationError(f"Cannot read {label}: {path.name}: {error}") from error
    _require(isinstance(value, dict), f"{label} must be a JSON object.")
    return value


def _safe_repo_path(repo_root: Path, relative: str | Path) -> Path:
    relative_path = Path(relative)
    _require(not relative_path.is_absolute(), "Frozen source paths must be relative.")
    candidate = (repo_root / relative_path).resolve()
    try:
        candidate.relative_to(repo_root)
    except ValueError as error:
        raise GateEvaluationError(
            f"Frozen source path escapes repository root: {relative_path}."
        ) from error
    return candidate


def _reject_nonfinite_json_numbers(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        _require(np.isfinite(float(value)), f"{label} contains a non-finite number.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite_json_numbers(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite_json_numbers(item, f"{label}.{key}")


def _exact_equal(actual: Any, expected: Any, label: str) -> None:
    try:
        actual_normalized = _json_normalize(actual)
        expected_normalized = _json_normalize(expected)
    except (TypeError, ValueError) as error:
        raise GateEvaluationError(f"{label} is not canonical finite JSON.") from error
    _require(actual_normalized == expected_normalized, f"{label} does not match frozen config.")


def verify_frozen_source(
    repo_root: Path, source_manifest_path: Path | None = None
) -> FrozenSource:
    """Verify every frozen source hash and load the three exact configs."""

    repo_root = repo_root.expanduser().resolve()
    manifest_path = (
        source_manifest_path.expanduser().resolve()
        if source_manifest_path is not None
        else (repo_root / DEFAULT_SOURCE_MANIFEST).resolve()
    )
    _require(manifest_path.is_file(), f"Frozen source manifest is missing: {manifest_path.name}.")
    manifest_sha = _sha256_file(manifest_path)
    _require(
        manifest_sha == SOURCE_MANIFEST_SHA256,
        "Frozen source manifest SHA-256 does not match the predeclared anchor.",
    )
    manifest = _load_json_object(manifest_path, "frozen source manifest")
    _require(manifest.get("hash_algorithm") == "sha256", "Source hash algorithm must be sha256.")
    files = manifest.get("files")
    _require(isinstance(files, dict) and files, "Frozen source manifest has no files.")

    required_files = {
        str(DEFAULT_SOURCE_MANIFEST.parent / "deployment_gate.md").replace("\\", "/"),
        str(STORE_METADATA_RELATIVE).replace("\\", "/"),
        *(spec["config"] for spec in RUN_SPECS.values()),
    }
    _require(required_files.issubset(files), "Frozen source manifest omits required gate sources.")
    for relative, expected_sha in sorted(files.items()):
        _require(
            isinstance(relative, str)
            and isinstance(expected_sha, str)
            and len(expected_sha) == 64,
            "Frozen source manifest contains a malformed file entry.",
        )
        path = _safe_repo_path(repo_root, relative)
        _require(path.is_file(), f"Frozen source file is missing: {relative}.")
        actual_sha = _sha256_file(path)
        _require(actual_sha == expected_sha, f"Frozen source hash mismatch: {relative}.")

    recorded_head = manifest.get("git_head")
    _require(
        isinstance(recorded_head, str) and len(recorded_head) == 40,
        "Frozen source manifest git_head is malformed.",
    )
    try:
        commit_check = subprocess.run(
            ["git", "cat-file", "-e", f"{recorded_head}^{{commit}}"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        recorded_commit_present = commit_check.returncode == 0
    except OSError:
        recorded_commit_present = False
    _require(recorded_commit_present, "Frozen source git commit is not present locally.")

    metadata_path = _safe_repo_path(repo_root, STORE_METADATA_RELATIVE)
    metadata = _load_json_object(metadata_path, "feature-store metadata")
    _require(metadata.get("complete") is True, "Feature-store generation is not complete.")
    _require(metadata.get("target_column") == TARGET_COLUMN, "Feature-store target is not frozen target.")
    _require(
        metadata.get("benchmark_column") == BENCHMARK_COLUMN,
        "Feature-store benchmark is not the frozen benchmark.",
    )
    manifest_meta = metadata.get("manifest")
    feature_meta = metadata.get("features")
    _require(isinstance(manifest_meta, dict), "Feature-store manifest metadata is malformed.")
    _require(isinstance(feature_meta, dict), "Feature-store feature metadata is malformed.")
    store_manifest_path = metadata_path.parent / str(manifest_meta.get("filename", ""))
    feature_path = metadata_path.parent / str(feature_meta.get("filename", ""))
    _require(store_manifest_path.parent == metadata_path.parent, "Unsafe store manifest filename.")
    _require(feature_path.parent == metadata_path.parent, "Unsafe feature payload filename.")
    _require(store_manifest_path.is_file(), "Feature-store manifest artifact is missing.")
    _require(feature_path.is_file(), "Feature-store feature artifact is missing.")
    _require(
        store_manifest_path.stat().st_size == int(manifest_meta.get("size_bytes", -1)),
        "Feature-store manifest size does not match frozen metadata.",
    )
    _require(
        feature_path.stat().st_size == int(feature_meta.get("size_bytes", -1)),
        "Feature-store feature payload size does not match frozen metadata.",
    )
    _require(
        _sha256_file(store_manifest_path) == manifest_meta.get("sha256"),
        "Feature-store manifest payload hash does not match frozen metadata.",
    )

    configs: dict[str, dict[str, Any]] = {}
    for run_name, spec in RUN_SPECS.items():
        config_path = _safe_repo_path(repo_root, spec["config"])
        namespace = runpy.run_path(str(config_path))
        config = namespace.get("CONFIG", namespace.get("config"))
        _require(isinstance(config, dict), f"Frozen config {run_name} has no CONFIG dict.")
        configs[run_name] = config

    return FrozenSource(
        repo_root=repo_root,
        source_manifest_path=manifest_path,
        source_manifest=manifest,
        source_manifest_sha256=manifest_sha,
        store_metadata_path=metadata_path,
        store_metadata=metadata,
        store_manifest_path=store_manifest_path,
        configs=configs,
        recorded_commit_present=recorded_commit_present,
    )


def _finite_numeric_array(series: pd.Series, label: str) -> np.ndarray:
    _require(
        not pd.api.types.is_bool_dtype(series.dtype)
        and pd.api.types.is_numeric_dtype(series.dtype)
        and not pd.api.types.is_complex_dtype(series.dtype),
        f"{label} must be finite real numeric data.",
    )
    values = series.to_numpy(dtype=np.float64, copy=False)
    _require(np.isfinite(values).all(), f"{label} contains non-finite values.")
    return values


def derive_internal_split_counts(
    outer_train_eras: Sequence[Any] | np.ndarray,
    *,
    sample_seed: int,
    max_train_samples: int,
    val_fraction: float,
    internal_val_embargo: int,
) -> dict[str, int]:
    """Mirror outer row sampling and Torch recent-era validation exactly."""

    eras = np.asarray(outer_train_eras)
    _require(eras.ndim == 1 and eras.size > 0, "Outer training eras must be non-empty and one-dimensional.")
    _require(max_train_samples > 0, "max_train_samples must be positive.")
    _require(0.0 < val_fraction < 1.0, "val_fraction must be strictly between zero and one.")
    _require(internal_val_embargo >= 0, "internal_val_embargo must be non-negative.")
    if len(eras) > max_train_samples:
        # numerai_cv._subset_data creates a fresh generator for every outer
        # fold, then applies these positions without sorting them.
        rng = np.random.default_rng(sample_seed)
        sampled_positions = rng.choice(
            len(eras), size=max_train_samples, replace=False
        )
        sampled_eras = eras[sampled_positions]
    else:
        sampled_eras = eras

    unique_eras = sorted(set(sampled_eras), key=lambda value: int(value))
    _require(len(unique_eras) >= 2, "Internal recent-era validation needs at least two sampled eras.")
    val_era_count = max(1, int(len(unique_eras) * val_fraction))
    split = len(unique_eras) - val_era_count
    train_end = max(0, split - internal_val_embargo)
    train_eras = set(unique_eras[:train_end])
    validation_eras = set(unique_eras[split:])
    train_mask = np.fromiter(
        (era in train_eras for era in sampled_eras),
        dtype=bool,
        count=len(sampled_eras),
    )
    validation_mask = np.fromiter(
        (era in validation_eras for era in sampled_eras),
        dtype=bool,
        count=len(sampled_eras),
    )
    train_rows = int(train_mask.sum())
    validation_rows = int(validation_mask.sum())
    _require(train_rows > 0 and validation_rows > 0, "Internal recent-era split is empty.")
    sampled_rows = int(len(sampled_eras))
    embargo_rows = sampled_rows - train_rows - validation_rows
    _require(embargo_rows >= 0, "Internal split row accounting is invalid.")
    return {
        "outer_sample_rows": sampled_rows,
        "sampled_eras": len(unique_eras),
        "internal_train_eras": len(train_eras),
        "internal_validation_eras": len(validation_eras),
        "internal_embargo_eras": split - train_end,
        "disk_train_rows": train_rows,
        "disk_validation_rows": validation_rows,
        "internal_embargo_rows": embargo_rows,
    }


def build_expected_oof_cohort(source: FrozenSource) -> ExpectedCohort:
    """Build the exact fold-1..4 cohort from the frozen consecutive-era store."""

    metadata = source.store_metadata
    manifest_meta = metadata["manifest"]
    expected_columns = [
        "row_offset",
        ID_COLUMN,
        ERA_COLUMN,
        TARGET_COLUMN,
        BENCHMARK_COLUMN,
    ]
    _require(manifest_meta.get("columns") == expected_columns, "Store manifest columns are not frozen schema.")
    parquet = pq.ParquetFile(source.store_manifest_path)
    try:
        schema = parquet.schema_arrow
        rows = parquet.metadata.num_rows
    finally:
        parquet.close()
    _require(schema.names == expected_columns, "Store manifest parquet schema names differ.")
    _require(rows == int(metadata.get("row_count", -1)), "Store manifest row count differs.")
    _require(schema.field("row_offset").type == pa.int64(), "Store row_offset is not int64.")
    _require(
        pa.types.is_string(schema.field(ID_COLUMN).type)
        or pa.types.is_large_string(schema.field(ID_COLUMN).type),
        "Store ids are not strings.",
    )
    _require(
        pa.types.is_string(schema.field(ERA_COLUMN).type)
        or pa.types.is_large_string(schema.field(ERA_COLUMN).type),
        "Store eras are not strings.",
    )

    manifest = pd.read_parquet(source.store_manifest_path, columns=expected_columns)
    _require(len(manifest) == rows, "Store manifest materialized row count differs.")
    offsets = manifest["row_offset"].to_numpy(dtype=np.int64, copy=False)
    _require(np.array_equal(offsets, np.arange(rows, dtype=np.int64)), "Store row offsets are not exact and contiguous.")
    _require(not manifest[ID_COLUMN].isna().any(), "Store manifest contains null ids.")
    _require(not manifest[ID_COLUMN].duplicated().any(), "Store manifest contains duplicate ids.")
    _require(not manifest[ERA_COLUMN].isna().any(), "Store manifest contains null eras.")
    _finite_numeric_array(manifest[TARGET_COLUMN], "Store target")
    _finite_numeric_array(manifest[BENCHMARK_COLUMN], "Store benchmark")

    era_values = manifest[ERA_COLUMN].astype(str)
    _require(
        np.array_equal(era_values.to_numpy(), manifest[ERA_COLUMN].to_numpy()),
        "Store era representation is not exact string data.",
    )
    unique_eras = sorted(era_values.unique().tolist(), key=int)
    _require(unique_eras, "Store contains no eras.")
    _require(all(len(era) == 4 and era.isdigit() for era in unique_eras), "Store eras are not four-digit numeric strings.")
    era_numbers = np.asarray([int(era) for era in unique_eras], dtype=np.int64)
    _require(np.all(np.diff(era_numbers) == 1), "Store eras are not consecutive.")

    n_splits = 5
    embargo = 52
    fold_size = len(unique_eras) // n_splits
    remainder = len(unique_eras) % n_splits
    era_to_fold: dict[str, int] = {}
    fold_specs: list[dict[str, int]] = []
    fold_validation_eras: list[tuple[int, list[str]]] = []
    fold_outer_train_eras: list[tuple[int, list[str]]] = []
    max_train_samples = 500_000
    for fold in range(n_splits):
        start = fold * fold_size
        end = (fold + 1) * fold_size
        if fold == n_splits - 1:
            end += remainder
        validation_eras = unique_eras[start:end]
        train_end = max(0, start - embargo)
        train_eras = unique_eras[:train_end]
        if not train_eras or not validation_eras:
            continue
        for era in validation_eras:
            _require(era not in era_to_fold, "Frozen CV validation eras overlap.")
            era_to_fold[era] = fold
        fold_validation_eras.append((fold, validation_eras))
        fold_outer_train_eras.append((fold, train_eras))
        train_rows_uncapped = int(era_values.isin(train_eras).sum())
        val_rows = int(era_values.isin(validation_eras).sum())
        fold_specs.append(
            {
                "fold": fold,
                "train_eras": len(train_eras),
                "val_eras": len(validation_eras),
                "train_rows": min(train_rows_uncapped, max_train_samples),
                "val_rows": val_rows,
            }
        )

    # Mirror build_oof_predictions exactly: each validation fold is loaded by
    # sorted manifest position, then fold frames are concatenated in fold order.
    cohort_parts: list[pd.DataFrame] = []
    for fold, validation_eras in fold_validation_eras:
        selected = era_values.isin(validation_eras)
        part = manifest.loc[
            selected, [ID_COLUMN, ERA_COLUMN, TARGET_COLUMN, BENCHMARK_COLUMN]
        ].copy()
        part[FOLD_COLUMN] = fold
        cohort_parts.append(part)
    cohort = pd.concat(cohort_parts, ignore_index=True)
    selected_eras = sorted(cohort[ERA_COLUMN].unique().tolist(), key=int)
    _require(len(cohort) == sum(fold["val_rows"] for fold in fold_specs), "Derived OOF row coverage is inconsistent.")
    _require(len(selected_eras) == sum(fold["val_eras"] for fold in fold_specs), "Derived OOF era coverage is inconsistent.")

    split_cache: dict[tuple[int, int, float, int, int], dict[str, int]] = {}
    internal_folds_by_run: dict[str, tuple[dict[str, int], ...]] = {}
    era_array = era_values.to_numpy(copy=False)
    outer_eras_by_fold = {
        fold: era_array[era_values.isin(train_eras).to_numpy()]
        for fold, train_eras in fold_outer_train_eras
    }
    for run_name, config in source.configs.items():
        training = config["training"]
        params = config["model"]["params"]
        sample_seed = int(training["sample_seed"])
        max_train_samples = int(training["max_train_samples"])
        val_fraction = float(params["val_fraction"])
        internal_embargo = int(params["internal_val_embargo"])
        internal_folds: list[dict[str, int]] = []
        for fold, _train_eras in fold_outer_train_eras:
            outer_eras = outer_eras_by_fold[fold]
            cache_key = (
                fold,
                sample_seed,
                val_fraction,
                internal_embargo,
                max_train_samples,
            )
            counts = split_cache.get(cache_key)
            if counts is None:
                counts = derive_internal_split_counts(
                    outer_eras,
                    sample_seed=sample_seed,
                    max_train_samples=max_train_samples,
                    val_fraction=val_fraction,
                    internal_val_embargo=internal_embargo,
                )
                split_cache[cache_key] = counts
            internal_folds.append({"fold": fold, **counts})
        internal_folds_by_run[run_name] = tuple(internal_folds)
    return ExpectedCohort(
        frame=cohort,
        full_rows=len(manifest),
        full_eras=len(unique_eras),
        oof_rows=len(cohort),
        oof_eras=len(selected_eras),
        first_full_era=unique_eras[0],
        last_full_era=unique_eras[-1],
        first_oof_era=selected_eras[0],
        last_oof_era=selected_eras[-1],
        folds=tuple(fold_specs),
        internal_folds_by_run=internal_folds_by_run,
    )


def expected_prediction_semantics(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config["model"]
    transform = model.get("target_transform") or {"type": "identity"}
    return _json_normalize(
        {
            "schema_version": 1,
            "column": PREDICTION_COLUMN,
            "artifact_kind": "out_of_fold_validation",
            "producer": "model.predict",
            "training_target": {
                "column": TARGET_COLUMN,
                "transform": transform,
            },
            "stored_target": {
                "column": TARGET_COLUMN,
                "transform": {"type": "identity"},
            },
            "inverse_target_transform_applied": False,
            "pipeline_postprocess": {"type": "identity"},
            "era_column": ERA_COLUMN,
            "fold_column": FOLD_COLUMN,
            "fold_index_base": 0,
        }
    )


def _path_leaf(value: Any) -> str:
    _require(isinstance(value, str) and value, "Artifact path must be a non-empty string.")
    return value.replace("\\", "/").rstrip("/").split("/")[-1]


def validate_result_json(
    run: RunArtifact,
    result: dict[str, Any],
    config: dict[str, Any],
    source: FrozenSource,
    cohort: ExpectedCohort,
) -> dict[str, Any]:
    """Validate result provenance/config/coverage before reading predictions."""

    _reject_nonfinite_json_numbers(result, f"Result {run.name}")
    spec = RUN_SPECS[run.name]
    _require(run.result_path.name == f"{run.name}.json", f"{run.name} result filename is not exact.")
    _require(
        run.predictions_path.name == f"{run.name}.parquet",
        f"{run.name} prediction filename is not exact.",
    )

    model_config = config["model"]
    expected_model: dict[str, Any] = {
        "type": model_config["type"],
        "params": model_config["params"],
    }
    for key in (
        "x_groups",
        "data_needed",
        "target_transform",
        "prediction_transform",
        "era_weighting",
        "prediction_batch_size",
        "benchmark",
        "baseline",
    ):
        if key in model_config:
            expected_model[key] = model_config[key]
    _exact_equal(result.get("model"), expected_model, f"{run.name} model")
    _require(
        result["model"]["params"].get("seed") == spec["model_seed"],
        f"{run.name} model seed differs from frozen matrix.",
    )

    preprocessing = config.get("preprocessing", {})
    expected_preprocessing = {
        "nan_missing_all_twos": preprocessing.get("nan_missing_all_twos", False),
        "missing_value": preprocessing.get("missing_value", 2.0),
    }
    _exact_equal(
        result.get("preprocessing"),
        expected_preprocessing,
        f"{run.name} preprocessing",
    )

    data = result.get("data")
    _require(isinstance(data, dict), f"{run.name} result has no data object.")
    expected_data = {
        "data_version": "v5.3",
        "feature_set": "all",
        "target": TARGET_COLUMN,
        "full_data_path": None,
        "full_rows": cohort.full_rows,
        "full_eras": cohort.full_eras,
        "oof_rows": cohort.oof_rows,
        "oof_eras": cohort.oof_eras,
        "embargo_eras": 52,
        "require_benchmark_coverage": True,
        "data_mode": "disk_feature_store",
    }
    for key, expected in expected_data.items():
        _exact_equal(data.get(key), expected, f"{run.name} data.{key}")

    diagnostics = data.get("disk_feature_store")
    _require(isinstance(diagnostics, dict), f"{run.name} lacks disk-store diagnostics.")
    metadata = source.store_metadata
    exact_diagnostics = {
        "generation_id": metadata["generation_id"],
        "row_count": metadata["row_count"],
        "feature_count": metadata["feature_count"],
        "feature_bytes": metadata["features"]["size_bytes"],
        "manifest_bytes": metadata["manifest"]["size_bytes"],
        "feature_order_sha256": metadata["feature_order_sha256"],
        "feature_sha256": metadata["features"]["sha256"],
        "manifest_sha256": metadata["manifest"]["sha256"],
    }
    for key, expected in exact_diagnostics.items():
        _exact_equal(diagnostics.get(key), expected, f"{run.name} store.{key}")
    _require(
        _path_leaf(diagnostics.get("manifest_path")) == metadata["manifest"]["filename"],
        f"{run.name} store manifest path differs.",
    )
    _require(
        _path_leaf(diagnostics.get("feature_path")) == metadata["features"]["filename"],
        f"{run.name} store feature path differs.",
    )

    benchmark = result.get("benchmark")
    _require(isinstance(benchmark, dict), f"{run.name} has no benchmark object.")
    _require(benchmark.get("model") == BENCHMARK_COLUMN, f"{run.name} benchmark model differs.")
    _require(
        _path_leaf(benchmark.get("file")) == metadata["manifest"]["filename"],
        f"{run.name} benchmark source is not frozen manifest.",
    )

    training = result.get("training")
    _require(isinstance(training, dict), f"{run.name} has no training object.")
    _exact_equal(
        training.get("data_sampling"),
        {"max_train_samples": 500_000, "sample_seed": spec["sample_seed"]},
        f"{run.name} sampling",
    )
    _require(training.get("data_mode") == "disk_feature_store", f"{run.name} training mode differs.")
    _exact_equal(
        training.get("cv"),
        {
            "enabled": True,
            "n_splits": 5,
            "embargo": 52,
            "mode": "expanding",
            "min_train_size": 0,
        },
        f"{run.name} training CV",
    )

    cv = result.get("cv")
    _require(isinstance(cv, dict), f"{run.name} has no CV object.")
    for key, expected in {
        "n_splits": 5,
        "embargo": 52,
        "mode": "expanding",
        "min_train_size": 0,
        "folds_used": len(cohort.folds),
    }.items():
        _exact_equal(cv.get(key), expected, f"{run.name} cv.{key}")
    result_folds = cv.get("folds")
    _require(
        isinstance(result_folds, list) and len(result_folds) == len(cohort.folds),
        f"{run.name} CV fold list is incomplete.",
    )
    internal_by_fold = {
        fold["fold"]: fold for fold in cohort.internal_folds_by_run[run.name]
    }
    for actual, expected in zip(result_folds, cohort.folds, strict=True):
        _require(isinstance(actual, dict), f"{run.name} CV fold is malformed.")
        for key, value in expected.items():
            _exact_equal(actual.get(key), value, f"{run.name} fold {expected['fold']} {key}")
        diagnostics = actual.get("model_diagnostics")
        _require(isinstance(diagnostics, dict), f"{run.name} fold diagnostics are missing.")
        for key in ("best_epoch", "epochs_ran", "n_parameters"):
            value = diagnostics.get(key)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value > 0,
                f"{run.name} fold diagnostic {key} is invalid.",
            )
        _require(
            diagnostics["best_epoch"] <= diagnostics["epochs_ran"] <= model_config["params"]["max_epochs"],
            f"{run.name} fold epoch diagnostics are inconsistent.",
        )
        for key in ("best_val_loss", "final_train_loss"):
            value = diagnostics.get(key)
            _require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(float(value)),
                f"{run.name} fold diagnostic {key} is invalid.",
            )
        _require(
            diagnostics.get("data_mode") == "disk_feature_store",
            f"{run.name} fold was not trained from disk feature store.",
        )
        internal = internal_by_fold[expected["fold"]]
        _exact_equal(
            diagnostics.get("disk_train_rows"),
            internal["disk_train_rows"],
            f"{run.name} fold disk_train_rows",
        )
        _exact_equal(
            diagnostics.get("disk_validation_rows"),
            internal["disk_validation_rows"],
            f"{run.name} fold disk_validation_rows",
        )
        prediction_batches = diagnostics.get("disk_prediction_batches")
        _require(
            isinstance(prediction_batches, int) and prediction_batches > 0,
            f"{run.name} fold prediction batch count is invalid.",
        )
        rows_per_epoch = diagnostics.get("disk_rows_per_epoch")
        batches_per_epoch = diagnostics.get("disk_batches_per_epoch")
        _require(
            isinstance(rows_per_epoch, list)
            and len(rows_per_epoch) == diagnostics["epochs_ran"]
            and all(row == internal["disk_train_rows"] for row in rows_per_epoch),
            f"{run.name} fold disk rows-per-epoch are inconsistent.",
        )
        _require(
            isinstance(batches_per_epoch, list)
            and len(batches_per_epoch) == diagnostics["epochs_ran"]
            and all(isinstance(count, int) and count > 0 for count in batches_per_epoch),
            f"{run.name} fold disk batches-per-epoch are inconsistent.",
        )

    output = result.get("output")
    _require(isinstance(output, dict), f"{run.name} has no output object.")
    _require(
        _path_leaf(output.get("predictions_file")) == f"{run.name}.parquet",
        f"{run.name} result points to a different prediction artifact.",
    )
    semantics = expected_prediction_semantics(config)
    _exact_equal(output.get("prediction_semantics"), semantics, f"{run.name} prediction semantics")
    metrics = result.get("metrics")
    _require(isinstance(metrics, dict), f"{run.name} has no metrics object.")
    _require(
        set(metrics) == {"corr", "bmc", "bmc_last_200_eras"},
        f"{run.name} result metrics schema differs.",
    )
    return semantics


def rank_predictions_exact(predictions: pd.Series, eras: pd.Series) -> np.ndarray:
    """Apply the frozen live transform exactly, including average tie ranks."""

    _finite_numeric_array(predictions, "Raw prediction")
    _require(not eras.isna().any(), "Prediction eras contain null values.")
    ranked = predictions.groupby(eras, sort=False).rank(method="average", pct=True)
    values = ranked.to_numpy(dtype=np.float64, copy=False)
    _require(np.isfinite(values).all(), "Ranked predictions contain non-finite values.")
    _require(
        np.logical_and(values >= 0.0, values <= 1.0).all(),
        "Ranked predictions fall outside [0, 1].",
    )
    return np.array(values, dtype=np.float64, copy=True)


def validate_prediction_artifact(
    path: Path,
    expected: pd.DataFrame,
    expected_semantics: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Validate one OOF parquet exactly and return raw and ranked signals."""

    path = path.expanduser().resolve()
    _require(path.is_file(), f"Prediction artifact is missing: {path.name}.")
    required_columns = [
        ID_COLUMN,
        ERA_COLUMN,
        TARGET_COLUMN,
        PREDICTION_COLUMN,
        FOLD_COLUMN,
    ]
    parquet = pq.ParquetFile(path)
    try:
        schema = parquet.schema_arrow
        num_rows = parquet.metadata.num_rows
        metadata = dict(schema.metadata or {})
    finally:
        parquet.close()
    _require(schema.names == required_columns, f"{path.name} parquet columns are not exact OOF schema.")
    _require(num_rows == len(expected), f"{path.name} parquet row count is incomplete.")
    encoded_semantics = metadata.get(PREDICTION_SEMANTICS_METADATA_KEY)
    _require(encoded_semantics is not None, f"{path.name} lacks prediction-semantics metadata.")
    try:
        semantics_text = encoded_semantics.decode("utf-8")
        parquet_semantics = json.loads(semantics_text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateEvaluationError(f"{path.name} semantics metadata is malformed.") from error
    expected_semantics_text = _canonical_json(expected_semantics)
    _require(
        semantics_text == expected_semantics_text,
        f"{path.name} semantics metadata is not exact canonical frozen semantics.",
    )
    _exact_equal(parquet_semantics, expected_semantics, f"{path.name} parquet semantics")

    frame = pd.read_parquet(path, columns=required_columns)
    _require(len(frame) == len(expected), f"{path.name} materialized row count differs.")
    _require(not frame[ID_COLUMN].isna().any(), f"{path.name} contains null ids.")
    _require(not frame[ID_COLUMN].duplicated().any(), f"{path.name} contains duplicate ids.")
    _require(not frame[ERA_COLUMN].isna().any(), f"{path.name} contains null eras.")
    _require(
        np.array_equal(frame[ID_COLUMN].to_numpy(), expected[ID_COLUMN].to_numpy()),
        f"{path.name} ids or row order do not exactly match frozen OOF cohort.",
    )
    _require(
        np.array_equal(frame[ERA_COLUMN].to_numpy(), expected[ERA_COLUMN].to_numpy()),
        f"{path.name} eras do not exactly align by id.",
    )
    target = _finite_numeric_array(frame[TARGET_COLUMN], f"{path.name} target")
    expected_target = _finite_numeric_array(expected[TARGET_COLUMN], "Expected target")
    _require(
        np.array_equal(target, expected_target),
        f"{path.name} targets do not exactly match frozen OOF cohort.",
    )
    fold_values = _finite_numeric_array(frame[FOLD_COLUMN], f"{path.name} cv_fold")
    _require(np.equal(fold_values, np.floor(fold_values)).all(), f"{path.name} cv_fold is not integral.")
    expected_folds = expected[FOLD_COLUMN].to_numpy(dtype=np.int64, copy=False)
    _require(
        np.array_equal(fold_values.astype(np.int64), expected_folds),
        f"{path.name} CV folds do not exactly match frozen OOF coverage.",
    )
    _finite_numeric_array(frame[PREDICTION_COLUMN], f"{path.name} prediction")
    # Preserve the stored dtype for the raw-metric receipt comparison.  This
    # mirrors the training pipeline's scorer before the separate live rank.
    raw = frame[PREDICTION_COLUMN].to_numpy(copy=True)
    ranked = rank_predictions_exact(frame[PREDICTION_COLUMN], frame[ERA_COLUMN])
    diagnostics = {
        "file": path.name,
        "sha256": _sha256_file(path),
        "rows": len(frame),
        "eras": int(frame[ERA_COLUMN].nunique()),
        "first_era": str(frame[ERA_COLUMN].iloc[0]),
        "last_era": str(frame[ERA_COLUMN].iloc[-1]),
        "unique_ids": True,
        "finite_raw_predictions": True,
        "ranked_min": float(ranked.min()),
        "ranked_max": float(ranked.max()),
        "semantics_sha256": hashlib.sha256(encoded_semantics).hexdigest(),
    }
    return raw, ranked, diagnostics


def _score_columns(
    cohort: pd.DataFrame, columns: Mapping[str, np.ndarray]
) -> dict[str, dict[str, dict[str, float]]]:
    _require(columns, "No prediction columns supplied for scoring.")
    scoring = cohort[[ERA_COLUMN, TARGET_COLUMN, BENCHMARK_COLUMN]].copy()
    for name, values in columns.items():
        array = np.asarray(values)
        _require(array.shape == (len(cohort),), f"Scoring column {name} has wrong shape.")
        _require(
            np.issubdtype(array.dtype, np.number)
            and not np.issubdtype(array.dtype, np.bool_)
            and not np.issubdtype(array.dtype, np.complexfloating),
            f"Scoring column {name} is not finite real numeric data.",
        )
        _require(
            np.isfinite(array.astype(np.float64, copy=False)).all(),
            f"Scoring column {name} is non-finite.",
        )
        scoring[name] = array
    prediction_columns = list(columns)
    corr = numerai_metrics.per_era_corr(
        scoring, prediction_columns, TARGET_COLUMN, ERA_COLUMN
    )
    bmc = numerai_metrics.per_era_bmc(
        scoring,
        prediction_columns,
        BENCHMARK_COLUMN,
        TARGET_COLUMN,
        ERA_COLUMN,
    )
    benchmark_corr = numerai_metrics.per_era_pred_corr(
        scoring, prediction_columns, BENCHMARK_COLUMN, ERA_COLUMN
    )
    expected_eras = sorted(cohort[ERA_COLUMN].unique().tolist(), key=int)
    for label, frame in (
        ("Corr", corr),
        ("BMC", bmc),
        ("benchmark correlation", benchmark_corr),
    ):
        _require(
            list(frame.columns) == prediction_columns,
            f"{label} output columns differ from requested predictions.",
        )
        _require(
            [str(value) for value in frame.index] == expected_eras,
            f"{label} output does not cover every expected era exactly.",
        )
        _require(
            np.isfinite(frame.to_numpy(dtype=np.float64)).all(),
            f"{label} contains non-finite per-era scores.",
        )

    corr_summary = numerai_metrics.summarize_scores(corr)
    bmc_summary = numerai_metrics.summarize_scores(bmc)
    recent_bmc = numerai_metrics._last_n_eras(bmc, 200)
    recent_benchmark_corr = numerai_metrics._last_n_eras(benchmark_corr, 200)
    recent_summary = numerai_metrics.summarize_scores(recent_bmc)
    benchmark_mean = benchmark_corr.mean()
    recent_benchmark_mean = recent_benchmark_corr.mean()

    output: dict[str, dict[str, dict[str, float]]] = {}
    for name in prediction_columns:
        corr_values = {key: float(value) for key, value in corr_summary.loc[name].items()}
        bmc_values = {key: float(value) for key, value in bmc_summary.loc[name].items()}
        bmc_values["avg_corr_with_benchmark"] = float(benchmark_mean.loc[name])
        recent_values = {
            key: float(value) for key, value in recent_summary.loc[name].items()
        }
        recent_values["avg_corr_with_benchmark"] = float(
            recent_benchmark_mean.loc[name]
        )
        metrics = {
            "corr": corr_values,
            "bmc": bmc_values,
            "bmc_last_200_eras": recent_values,
        }
        _reject_nonfinite_json_numbers(metrics, f"Computed metrics {name}")
        output[name] = metrics
    return output


def _require_reported_metrics_match(
    reported: Mapping[str, Any], computed: Mapping[str, Any], run_name: str
) -> None:
    _require(set(reported) == set(computed), f"{run_name} reported metric groups differ.")
    for group, expected_values in computed.items():
        actual_values = reported.get(group)
        _require(isinstance(actual_values, dict), f"{run_name} metric {group} is malformed.")
        _require(
            set(actual_values) == set(expected_values),
            f"{run_name} metric keys for {group} differ.",
        )
        for key, expected in expected_values.items():
            actual = actual_values[key]
            _require(
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and np.isfinite(float(actual)),
                f"{run_name} reported {group}.{key} is non-finite.",
            )
            _require(
                np.isclose(float(actual), float(expected), rtol=1e-10, atol=1e-12),
                f"{run_name} reported {group}.{key} does not match raw artifact.",
            )


def evaluate_quality_thresholds(
    run_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    ensemble_metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Evaluate all frozen numerical rules, including ensemble candidate quality."""

    _require(set(run_metrics) == set(RUN_SPECS), "Quality matrix must contain exactly three frozen runs.")
    checks: list[dict[str, Any]] = []

    def add(
        name: str,
        scope: str,
        value: float,
        comparison: str,
        threshold: float | tuple[float, float],
        passed: bool,
    ) -> None:
        _require(np.isfinite(float(value)), f"Quality check {name} has non-finite value.")
        checks.append(
            {
                "name": name,
                "scope": scope,
                "value": float(value),
                "comparison": comparison,
                "threshold": list(threshold) if isinstance(threshold, tuple) else threshold,
                "passed": bool(passed),
            }
        )

    def candidate_checks(scope: str, metrics: Mapping[str, Mapping[str, float]]) -> None:
        full_bmc = float(metrics["bmc"]["mean"])
        recent_bmc = float(metrics["bmc_last_200_eras"]["mean"])
        corr = float(metrics["corr"]["mean"])
        sharpe = float(metrics["bmc"]["sharpe"])
        drawdown = float(metrics["bmc"]["max_drawdown"])
        benchmark_corr = float(metrics["bmc"]["avg_corr_with_benchmark"])
        add("positive_full_bmc", scope, full_bmc, ">", 0.0, full_bmc > 0.0)
        add("positive_last_200_bmc", scope, recent_bmc, ">", 0.0, recent_bmc > 0.0)
        add(
            "plausible_target_corr",
            scope,
            corr,
            "inclusive_range",
            (0.005, 0.04),
            0.005 <= corr <= 0.04,
        )
        add("bmc_sharpe", scope, sharpe, ">", 0.25, sharpe > 0.25)
        add("bmc_max_drawdown", scope, drawdown, "<", 0.15, drawdown < 0.15)
        add(
            "abs_avg_corr_with_benchmark",
            scope,
            abs(benchmark_corr),
            "<",
            0.25,
            abs(benchmark_corr) < 0.25,
        )

    for run_name in RUN_SPECS:
        candidate_checks(run_name, run_metrics[run_name])

    median_full_bmc = statistics.median(
        float(run_metrics[name]["bmc"]["mean"]) for name in RUN_SPECS
    )
    median_recent_bmc = statistics.median(
        float(run_metrics[name]["bmc_last_200_eras"]["mean"])
        for name in RUN_SPECS
    )
    median_sharpe = statistics.median(
        float(run_metrics[name]["bmc"]["sharpe"]) for name in RUN_SPECS
    )
    add("median_full_bmc", "three_run_matrix", median_full_bmc, ">", 0.0, median_full_bmc > 0.0)
    add(
        "median_last_200_bmc",
        "three_run_matrix",
        median_recent_bmc,
        ">",
        0.0,
        median_recent_bmc > 0.0,
    )
    add("median_bmc_sharpe", "three_run_matrix", median_sharpe, ">", 0.40, median_sharpe > 0.40)
    candidate_checks("rank_mean_ensemble", ensemble_metrics)
    return {
        "status": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "checks": checks,
        "aggregate": {
            "median_bmc_mean": float(median_full_bmc),
            "median_bmc_last_200_mean": float(median_recent_bmc),
            "median_bmc_sharpe": float(median_sharpe),
        },
    }


def _validate_run_matrix(runs: Sequence[RunArtifact]) -> dict[str, RunArtifact]:
    _require(len(runs) == 3, "Exactly three result/prediction pairs are required.")
    by_name: dict[str, RunArtifact] = {}
    for run in runs:
        _require(run.name in RUN_SPECS, f"Unknown or post-hoc run name: {run.name}.")
        _require(run.name not in by_name, f"Duplicate run name: {run.name}.")
        by_name[run.name] = RunArtifact(
            run.name,
            run.result_path.expanduser().resolve(),
            run.predictions_path.expanduser().resolve(),
        )
    _require(set(by_name) == set(RUN_SPECS), "Run matrix does not exactly match frozen runs.")
    result_paths = [run.result_path for run in by_name.values()]
    prediction_paths = [run.predictions_path for run in by_name.values()]
    _require(len(set(result_paths)) == 3, "Result JSON paths must be distinct.")
    _require(len(set(prediction_paths)) == 3, "Prediction parquet paths must be distinct.")
    return by_name


def evaluate_gate(
    runs: Sequence[RunArtifact],
    *,
    repo_root: Path,
    source_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen artifact-quality gate or raise on integrity failure."""

    run_matrix = _validate_run_matrix(runs)
    source = verify_frozen_source(repo_root, source_manifest_path)
    cohort = build_expected_oof_cohort(source)
    raw_predictions: dict[str, np.ndarray] = {}
    ranked_predictions: dict[str, np.ndarray] = {}
    run_results: dict[str, dict[str, Any]] = {}
    run_reports: dict[str, dict[str, Any]] = {}

    for run_name in RUN_SPECS:
        run = run_matrix[run_name]
        _require(run.result_path.is_file(), f"Result JSON is missing: {run.result_path.name}.")
        result = _load_json_object(run.result_path, f"{run_name} result JSON")
        semantics = validate_result_json(
            run,
            result,
            source.configs[run_name],
            source,
            cohort,
        )
        raw, ranked, artifact_report = validate_prediction_artifact(
            run.predictions_path, cohort.frame, semantics
        )
        raw_predictions[run_name] = raw
        ranked_predictions[run_name] = ranked
        run_results[run_name] = result
        run_reports[run_name] = {
            "result_file": run.result_path.name,
            "result_sha256": _sha256_file(run.result_path),
            "predictions": artifact_report,
            "internal_split_receipts": list(
                cohort.internal_folds_by_run[run_name]
            ),
        }

    raw_metrics = _score_columns(cohort.frame, raw_predictions)
    for run_name in RUN_SPECS:
        _require_reported_metrics_match(
            run_results[run_name]["metrics"], raw_metrics[run_name], run_name
        )
        run_reports[run_name]["raw_metrics"] = raw_metrics[run_name]
    del raw_predictions

    ensemble = np.zeros(cohort.oof_rows, dtype=np.float64)
    for run_name in RUN_SPECS:
        ensemble += ranked_predictions[run_name]
    ensemble /= len(RUN_SPECS)
    _require(np.isfinite(ensemble).all(), "Rank-mean ensemble is non-finite.")
    _require(
        np.logical_and(ensemble >= 0.0, ensemble <= 1.0).all(),
        "Rank-mean ensemble falls outside [0, 1].",
    )
    ranked_for_scoring = dict(ranked_predictions)
    ranked_for_scoring["rank_mean_ensemble"] = ensemble
    ranked_metrics_all = _score_columns(cohort.frame, ranked_for_scoring)
    ranked_metrics = {
        name: ranked_metrics_all[name] for name in RUN_SPECS
    }
    ensemble_metrics = ranked_metrics_all["rank_mean_ensemble"]
    for run_name in RUN_SPECS:
        run_reports[run_name]["ranked_metrics"] = ranked_metrics[run_name]
    quality = evaluate_quality_thresholds(ranked_metrics, ensemble_metrics)
    quality_pass = quality["status"] == "PASS"

    external_checks = {
        "numpy_torch_parity": {
            "required": True,
            "status": "NOT_EVALUATED",
            "rtol": 1e-5,
            "atol": 1e-6,
        },
        "numpy_predictor_contract": {
            "required": True,
            "status": "NOT_EVALUATED",
            "requirements": [
                "preserve_input_index",
                "one_finite_prediction_column",
                "prediction_in_closed_unit_interval",
                "current_numerai_runtime_contract",
            ],
        },
        "hosted_live_size_runtime": {
            "required": True,
            "status": "NOT_EVALUATED",
            "cpu_count": 1,
            "memory_gib": 4,
            "deadline_seconds": 600,
        },
    }
    report = {
        "schema_version": 1,
        "gate": GATE_NAME,
        "status": (
            "PASS_PENDING_EXTERNAL_CHECKS" if quality_pass else "FAIL_CLOSED"
        ),
        "deployment_approved": False,
        "source": {
            "source_manifest_file": source.source_manifest_path.name,
            "source_manifest_sha256": source.source_manifest_sha256,
            "recorded_git_head": source.source_manifest["git_head"],
            "recorded_git_commit_present": source.recorded_commit_present,
            "verified_file_count": len(source.source_manifest["files"]),
            "store_generation_id": source.store_metadata["generation_id"],
            "store_manifest_sha256": source.store_metadata["manifest"]["sha256"],
        },
        "cohort": {
            "full_rows": cohort.full_rows,
            "full_eras": cohort.full_eras,
            "full_era_range": [cohort.first_full_era, cohort.last_full_era],
            "oof_rows": cohort.oof_rows,
            "oof_eras": cohort.oof_eras,
            "oof_era_range": [cohort.first_oof_era, cohort.last_oof_era],
            "folds": list(cohort.folds),
            "exact_cross_run_alignment": True,
            "complete_benchmark_coverage": True,
        },
        "runs": run_reports,
        "quality_gate": quality,
        "rank_mean_ensemble": {
            "construction": "equal_weight_mean_of_three_per_era_average_pct_ranks",
            "metrics": ensemble_metrics,
            "quality_status": (
                "PASS" if all(
                    check["passed"]
                    for check in quality["checks"]
                    if check["scope"] == "rank_mean_ensemble"
                ) else "FAIL"
            ),
        },
        "external_checks": external_checks,
        "selection": {
            "preferred_if_external_checks_pass": "rank_mean_ensemble",
            "runtime_fallback": "scale_disk_tabm_k64_train500k",
            "post_hoc_best_seed_selection_allowed": False,
            "ready_for_upload": False,
        },
    }
    _canonical_json(report)
    return report


def evaluate_gate_fail_closed(
    runs: Sequence[RunArtifact],
    *,
    repo_root: Path,
    source_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Return a machine-readable failure report instead of leaking exceptions."""

    try:
        return evaluate_gate(
            runs,
            repo_root=repo_root,
            source_manifest_path=source_manifest_path,
        )
    except Exception as error:
        report = {
            "schema_version": 1,
            "gate": GATE_NAME,
            "status": "FAIL_CLOSED",
            "deployment_approved": False,
            "quality_gate": {"status": "NOT_EVALUATED"},
            "errors": [
                {
                    "type": type(error).__name__,
                    "message": str(error),
                }
            ],
            "selection": {"ready_for_upload": False},
        }
        _canonical_json(report)
        return report


def _parse_run(values: Sequence[str]) -> RunArtifact:
    name, result_path, predictions_path = values
    return RunArtifact(name, Path(result_path), Path(predictions_path))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen three-run Ender20 deployment quality gate."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        required=True,
        metavar=("NAME", "RESULT_JSON", "PREDICTIONS_PARQUET"),
        help="Repeat exactly three times, once for each frozen run.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Numerai repository root (defaults to the root containing agents/).",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="Frozen gate source manifest (default is resolved under repo root).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report output path; parent directory must already exist.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    runs = [_parse_run(values) for values in args.run]
    report = evaluate_gate_fail_closed(
        runs,
        repo_root=args.repo_root,
        source_manifest_path=args.source_manifest,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        if not report_path.parent.is_dir():
            failure = {
                "schema_version": 1,
                "gate": GATE_NAME,
                "status": "FAIL_CLOSED",
                "deployment_approved": False,
                "errors": [{"type": "FileNotFoundError", "message": "Report parent directory does not exist."}],
            }
            print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
            return 2
        report_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS_PENDING_EXTERNAL_CHECKS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
