from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numerapi import NumerAPI

from agents.code.metrics import numerai_metrics
from .config import (
    load_config,
    resolve_predictions_path,
    resolve_results_path,
)
from .constants import (
    BASE_DIR,
    DEFAULT_BASELINES_DIR,
    DEFAULT_BENCHMARK_MODEL,
    DEFAULT_OUTPUT_DIR,
    NUMERAI_DIR,
    REPO_DIR,
)
from .data import (
    apply_missing_all_twos_as_nan,
    attach_baseline_column,
    attach_benchmark_models,
    load_features,
    load_full_data,
)
from .model_data import build_model_data_loader, build_x_cols, normalize_x_groups
from .numerai_cv import build_oof_predictions


PREDICTION_SEMANTICS_METADATA_KEY = b"numerai.agents.prediction_semantics"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _normalize_target_transform(transform: object) -> dict:
    if transform is None or transform == {} or transform == "":
        return {"type": "identity"}
    if isinstance(transform, str):
        return {"type": transform}
    if not isinstance(transform, dict):
        raise TypeError(
            "model.target_transform must be a dict, a string identifier, or None."
        )
    if not transform.get("type"):
        raise ValueError("model.target_transform.type is required.")
    return json.loads(_canonical_json(transform))


def _has_nonempty_prediction_transform(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes, dict, list, tuple, set)):
        return bool(value)
    return True


def build_prediction_semantics(
    model_config: dict, target_col: str, era_col: str
) -> dict:
    """Describe the exact meaning of the persisted OOF prediction artifact."""
    if _has_nonempty_prediction_transform(model_config.get("prediction_transform")):
        raise ValueError(
            "model.prediction_transform is not implemented; persisted predictions "
            "must remain the direct model.predict output."
        )
    semantics = {
        "schema_version": 1,
        "column": "prediction",
        "artifact_kind": "out_of_fold_validation",
        "producer": "model.predict",
        "training_target": {
            "column": target_col,
            "transform": _normalize_target_transform(
                model_config.get("target_transform")
            ),
        },
        "stored_target": {
            "column": target_col,
            "transform": {"type": "identity"},
        },
        "inverse_target_transform_applied": False,
        "pipeline_postprocess": {"type": "identity"},
        "era_column": era_col,
        "fold_column": "cv_fold",
        "fold_index_base": 0,
    }
    return json.loads(_canonical_json(semantics))


def resolve_output_locations(
    config: dict, output_dir_override: Path | None
) -> tuple[Path, Path, Path, Path]:
    output_config = config.get("output", {})
    data_config = config.get("data", {})

    output_dir = _resolve_repo_dir(
        output_dir_override or output_config.get("output_dir"),
        DEFAULT_OUTPUT_DIR,
    )
    baselines_dir = _resolve_repo_dir(
        output_config.get("baselines_dir") or data_config.get("baselines_dir"),
        DEFAULT_BASELINES_DIR,
    )

    results_dir = output_dir / "results"
    predictions_dir = output_dir / "predictions"
    return output_dir, baselines_dir, results_dir, predictions_dir


def _resolve_repo_dir(path: str | Path | None, default: Path) -> Path:
    if not path:
        return default
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] == BASE_DIR.name:
        return (BASE_DIR.parent / candidate).resolve()
    return (BASE_DIR / candidate).resolve()


def _resolve_feature_store_dir(
    path: str | Path | None, *, data_version: str, target_col: str
) -> Path:
    if path is None:
        return (NUMERAI_DIR / data_version / f"{target_col}_feature_store").resolve()
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] == NUMERAI_DIR.name:
        return (REPO_DIR / candidate).resolve()
    return (NUMERAI_DIR / candidate).resolve()


def resolve_model_config(model_config: dict) -> tuple[str, dict]:
    model_type = model_config.get("type", "LGBMRegressor")
    model_params = model_config.get("params")
    if model_params is None:
        raise ValueError("model.params must be specified; pipeline does not set defaults.")
    return model_type, model_params


def load_and_prepare_data(
    napi: NumerAPI,
    data_version: str,
    feature_set: str,
    target_col: str,
    era_col: str,
    id_col: str,
    full_data_path: str | Path | None,
    nan_missing_all_twos: bool,
    missing_value: float,
) -> tuple[pd.DataFrame, list[str]]:
    features = load_features(napi, data_version, feature_set)
    full = load_full_data(
        napi,
        data_version,
        features,
        era_col,
        target_col,
        id_col,
        full_data_path=full_data_path,
    )

    if nan_missing_all_twos:
        full = apply_missing_all_twos_as_nan(full, features, era_col, missing_value)

    return full, features


def select_prediction_columns(
    predictions: pd.DataFrame,
    id_col: str | None,
    era_col: str,
    target_col: str,
) -> pd.DataFrame:
    named_columns = {
        "id_col": id_col,
        "era_col": era_col,
        "target_col": target_col,
    }
    invalid = [
        label
        for label, column in named_columns.items()
        if not isinstance(column, str) or not column
    ]
    if invalid:
        raise ValueError(f"OOF prediction column names are required: {invalid}.")
    required_cols = [id_col, era_col, target_col, "prediction", "cv_fold"]
    missing = [col for col in required_cols if col not in predictions.columns]
    if missing:
        raise ValueError(f"OOF predictions are missing required columns: {missing}.")
    return predictions[required_cols].copy()


def save_predictions(
    predictions: pd.DataFrame,
    config: dict,
    config_path: Path,
    predictions_dir: Path,
    output_dir: Path,
    prediction_semantics: dict,
) -> tuple[Path, Path]:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = resolve_predictions_path(config, config_path, predictions_dir)
    table = pa.Table.from_pandas(predictions, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[PREDICTION_SEMANTICS_METADATA_KEY] = _canonical_json(
        prediction_semantics
    ).encode("utf-8")
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, predictions_path)
    print(f"Saved predictions to {predictions_path}")
    predictions_relative = predictions_path.relative_to(output_dir)
    return predictions_path, predictions_relative


def summarize_predictions(
    predictions_path: Path,
    target_col: str,
    data_version: str,
    benchmark_model: str,
    benchmark_data_path: str | None,
    era_col: str,
    id_col: str,
    *,
    benchmark_data: pd.DataFrame | None = None,
) -> dict:
    return numerai_metrics.summarize_prediction_file_with_bmc(
        predictions_path,
        ["prediction"],
        target_col,
        data_version,
        benchmark_model=benchmark_model,
        benchmark_data_path=benchmark_data_path,
        era_col=era_col,
        id_col=id_col,
        benchmark_data=benchmark_data,
    )


def build_results_payload(
    *,
    model_type: str,
    model_params: dict,
    model_config: dict,
    nan_missing_all_twos: bool,
    missing_value: float,
    data_version: str,
    feature_set: str,
    target_col: str,
    full_data_path: str | Path | None,
    full: pd.DataFrame,
    predictions: pd.DataFrame,
    era_col: str,
    embargo_eras: int,
    benchmark_model: str,
    benchmark_data_path: str | Path | None,
    output_dir: Path,
    predictions_relative: Path,
    summaries: dict,
    cv_meta: dict,
    cv_enabled: bool,
    max_train_samples: int | None,
    sample_seed: int,
    require_benchmark_coverage: bool,
    data_mode: str,
    disk_store_diagnostics: dict[str, object] | None,
    prediction_semantics: dict,
) -> dict:
    model_meta = {
        "type": model_type,
        "params": model_params,
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
            model_meta[key] = model_config[key]

    results = {
        "model": model_meta,
        "preprocessing": {
            "nan_missing_all_twos": nan_missing_all_twos,
            "missing_value": missing_value,
        },
        "data": {
            "data_version": data_version,
            "feature_set": feature_set,
            "target": target_col,
            "full_data_path": full_data_path,
            "full_rows": int(full.shape[0]),
            "full_eras": int(full[era_col].nunique()),
            "oof_rows": int(predictions.shape[0]),
            "oof_eras": int(predictions[era_col].nunique()),
            "embargo_eras": embargo_eras,
            "require_benchmark_coverage": require_benchmark_coverage,
            "data_mode": data_mode,
        },
        "benchmark": {
            "model": benchmark_model,
            "file": benchmark_data_path
            or f"{data_version}/full_benchmark_models.parquet",
        },
        "output": {
            "output_dir": str(output_dir),
            "predictions_file": str(predictions_relative),
            "prediction_semantics": prediction_semantics,
        },
        "metrics": {
            "corr": summaries["corr"].loc["prediction"].to_dict(),
            "bmc": summaries["bmc"].loc["prediction"].to_dict(),
            "bmc_last_200_eras": summaries["bmc_last_200_eras"]
            .loc["prediction"]
            .to_dict(),
        },
        "cv": cv_meta,
        "training": {
            "data_sampling": {
                "max_train_samples": max_train_samples,
                "sample_seed": sample_seed if max_train_samples else None,
            },
            "data_mode": data_mode,
            "cv": {
                "enabled": cv_enabled,
                "n_splits": cv_meta["n_splits"],
                "embargo": cv_meta["embargo"],
                "mode": cv_meta["mode"],
                "min_train_size": cv_meta["min_train_size"],
            },
        },
    }
    if disk_store_diagnostics is not None:
        results["data"]["disk_feature_store"] = disk_store_diagnostics
    return results


def save_results(results: dict, results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"Saved results to {results_path}")


def run_training(
    config_path: Path, output_dir_override: Path | None = None
) -> tuple[Path, Path]:
    config = load_config(config_path)

    data_config = config.get("data", {})
    preprocessing_config = config.get("preprocessing", {})
    training_config = config.get("training", {})
    model_config = config.get("model", {})

    data_version = data_config.get("data_version", "v5.3")
    feature_set = data_config.get("feature_set", "small")
    target_col = data_config.get("target_col", "target")
    era_col = data_config.get("era_col", "era")
    id_col = data_config.get("id_col", "id")
    full_data_path = data_config.get("full_data_path")
    benchmark_data_path = data_config.get("benchmark_data_path")
    embargo_eras = data_config.get("embargo_eras", 13)
    benchmark_model = data_config.get("benchmark_model", DEFAULT_BENCHMARK_MODEL)
    require_benchmark_coverage = bool(
        data_config.get("require_benchmark_coverage", False)
    )
    if not isinstance(id_col, str) or not id_col:
        raise ValueError("data.id_col must name the explicit OOF id column.")
    prediction_semantics = build_prediction_semantics(
        model_config, target_col, era_col
    )

    nan_missing_all_twos = preprocessing_config.get("nan_missing_all_twos", False)
    missing_value = preprocessing_config.get("missing_value", 2.0)

    max_train_samples = training_config.get("max_train_samples")
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)
    sample_seed = int(training_config.get("sample_seed", 1337))
    data_mode = str(training_config.get("data_mode", "eager")).lower()
    if data_mode not in {"eager", "disk_feature_store"}:
        raise ValueError(
            "training.data_mode must be 'eager' or 'disk_feature_store'."
        )

    output_dir, baselines_dir, results_dir, predictions_dir = resolve_output_locations(
        config, output_dir_override
    )

    napi = NumerAPI()
    model_type, model_params = resolve_model_config(model_config)
    raw_x_groups = model_config.get("x_groups") or model_config.get("data_needed")
    x_groups = normalize_x_groups(raw_x_groups)
    benchmark_cols: list[str] = []
    baseline_col: str | None = None
    disk_store_diagnostics: dict[str, object] | None = None
    scoring_benchmark_data_path: str | Path | None = benchmark_data_path
    scoring_benchmark_data: pd.DataFrame | None = None

    if data_mode == "eager":
        full, features = load_and_prepare_data(
            napi,
            data_version,
            feature_set,
            target_col,
            era_col,
            id_col,
            full_data_path=full_data_path,
            nan_missing_all_twos=nan_missing_all_twos,
            missing_value=missing_value,
        )

        if "benchmark_models" in x_groups:
            if not id_col:
                raise ValueError("id_col is required to attach benchmark models.")
            full, benchmark_cols = attach_benchmark_models(
                full,
                napi,
                data_version,
                benchmark_data_path,
                era_col,
                id_col,
            )
            if require_benchmark_coverage:
                if benchmark_model not in benchmark_cols:
                    raise ValueError(
                        f"Required benchmark '{benchmark_model}' is not present in "
                        f"the benchmark file. Available: {benchmark_cols}"
                    )
                rows_before = len(full)
                full = full[full[benchmark_model].notna()].copy()
                print(
                    "Restricted modeling data to benchmark-covered rows: "
                    f"{len(full):,}/{rows_before:,}."
                )

        if "baseline" in x_groups:
            baseline_spec = model_config.get("baseline", {})
            baseline_name = baseline_spec.get("name")
            baseline_path = baseline_spec.get("predictions_path")
            pred_col = baseline_spec.get("pred_col", "prediction")
            if not baseline_name or not baseline_path:
                raise ValueError(
                    "model.baseline.name and model.baseline.predictions_path "
                    "are required when baseline data is requested."
                )
            if not id_col:
                raise ValueError("id_col is required to attach baseline predictions.")
            full, baseline_col = attach_baseline_column(
                full,
                baseline_name,
                baseline_path,
                era_col,
                id_col,
                pred_col=pred_col,
            )
    else:
        if model_type != "TorchTabularRegressor":
            raise ValueError(
                "training.data_mode='disk_feature_store' currently requires "
                "model.type='TorchTabularRegressor'."
            )
        if nan_missing_all_twos:
            raise ValueError(
                "nan_missing_all_twos is unavailable for immutable int8 disk features."
            )
        if full_data_path is not None or benchmark_data_path is not None:
            raise ValueError(
                "Disk feature-store mode uses its own manifest; remove "
                "data.full_data_path and data.benchmark_data_path."
            )
        if id_col != "id" or era_col != "era":
            raise ValueError(
                "Disk feature-store mode requires id_col='id' and era_col='era'."
            )
        if "baseline" in x_groups:
            raise ValueError(
                "model.x_groups='baseline' is unavailable in disk feature-store mode."
            )
        configured_path = data_config.get("disk_feature_store_path")
        alias_path = data_config.get("feature_store_path")
        if configured_path is not None and alias_path is not None:
            if Path(configured_path) != Path(alias_path):
                raise ValueError(
                    "data.disk_feature_store_path and data.feature_store_path disagree."
                )
        store_path = _resolve_feature_store_dir(
            configured_path if configured_path is not None else alias_path,
            data_version=data_version,
            target_col=target_col,
        )
        from .disk_feature_store import DiskFeatureStoreLoader

        configured_features = load_features(napi, data_version, feature_set)
        data_loader = DiskFeatureStoreLoader(
            store_path,
            era_col=era_col,
            target_col=target_col,
            id_col=id_col,
            benchmark_col=benchmark_model,
        )
        try:
            features = list(data_loader.feature_columns)
            if configured_features != features:
                raise ValueError(
                    "Configured data.feature_set does not exactly match the disk "
                    "feature-store feature order."
                )
            benchmark_cols = [benchmark_model]
            full = data_loader.manifest
            scoring_benchmark_data_path = str(data_loader.manifest_path)
            scoring_benchmark_data = data_loader.manifest
            disk_store_diagnostics = data_loader.diagnostics
            require_benchmark_coverage = True
        except BaseException:
            data_loader.close()
            raise

    try:
        x_cols = build_x_cols(
            x_groups=x_groups,
            features=features,
            benchmark_cols=benchmark_cols,
            era_col=era_col,
            id_col=id_col,
            baseline_col=baseline_col,
        )
        if data_mode == "eager":
            data_loader = build_model_data_loader(
                full=full,
                x_cols=x_cols,
                era_col=era_col,
                target_col=target_col,
                id_col=id_col,
            )
        else:
            data_loader.configure_x_cols(x_cols)

        cv_config = dict(training_config.get("cv", {}))
        cv_config.setdefault("embargo", embargo_eras)
        cv_enabled = cv_config.get("enabled", True)
        if not cv_enabled:
            raise ValueError("CV/OOF pipeline is required for all experiments.")

        predictions, cv_meta = build_oof_predictions(
            full[era_col],
            data_loader,
            model_type,
            model_params,
            model_config,
            cv_config,
            max_train_samples,
            sample_seed,
            id_col,
            era_col,
            target_col,
            feature_cols=features,
        )
    finally:
        if data_mode == "disk_feature_store":
            data_loader.close()

    predictions = select_prediction_columns(predictions, id_col, era_col, target_col)
    predictions_path, predictions_relative = save_predictions(
        predictions,
        config,
        config_path,
        predictions_dir,
        output_dir,
        prediction_semantics,
    )

    summaries = summarize_predictions(
        predictions_path,
        target_col,
        data_version,
        benchmark_model,
        scoring_benchmark_data_path,
        era_col,
        id_col,
        benchmark_data=scoring_benchmark_data,
    )

    results_path = resolve_results_path(config, config_path, results_dir)
    results = build_results_payload(
        model_type=model_type,
        model_params=model_params,
        model_config=model_config,
        nan_missing_all_twos=nan_missing_all_twos,
        missing_value=missing_value,
        data_version=data_version,
        feature_set=feature_set,
        target_col=target_col,
        full_data_path=full_data_path,
        full=full,
        predictions=predictions,
        era_col=era_col,
        embargo_eras=embargo_eras,
        benchmark_model=benchmark_model,
        benchmark_data_path=scoring_benchmark_data_path,
        output_dir=output_dir,
        predictions_relative=predictions_relative,
        summaries=summaries,
        cv_meta=cv_meta,
        cv_enabled=cv_enabled,
        max_train_samples=max_train_samples,
        sample_seed=sample_seed,
        require_benchmark_coverage=require_benchmark_coverage,
        data_mode=data_mode,
        disk_store_diagnostics=disk_store_diagnostics,
        prediction_semantics=prediction_semantics,
    )
    save_results(results, results_path)

    return predictions_path, results_path
