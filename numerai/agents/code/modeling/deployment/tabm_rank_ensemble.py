"""Portable three-model rank ensemble for NumPy TabM predictors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from agents.code.modeling.deployment.tabm_numpy import build_tabm_numpy_forward


_REQUIRED_SPEC_KEYS = frozenset(
    {"feature_names", "blocks", "output_weight", "output_bias"}
)
_OPTIONAL_SPEC_KEYS = frozenset(
    {
        "feature_center",
        "feature_scale",
        "batch_size",
        "activation",
        "era_column",
        "prediction_column",
    }
)
_FORWARD_KEYS = frozenset(
    {
        "blocks",
        "output_weight",
        "output_bias",
        "feature_center",
        "feature_scale",
        "batch_size",
        "activation",
    }
)
_SHARED_STORE_KEYS = (
    "generation_id",
    "metadata_sha256",
    "manifest_sha256",
    "feature_sha256",
    "feature_order_sha256",
    "row_count",
    "feature_count",
    "era_start",
    "era_end",
    "era_count",
)
_EXPECTED_TARGET_TRANSFORM = {
    "type": "residual_to_benchmark",
    "benchmark_col": "v53_lgbm_ender20",
    "era_col": "era",
    "per_era": True,
    "fit_intercept": True,
}


def _model_specs_tuple(model_specs: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(model_specs, (str, bytes, Mapping)):
        raise TypeError("model_specs must be a sequence of three mappings.")
    try:
        specs = tuple(model_specs)
    except TypeError as exc:
        raise TypeError("model_specs must be a sequence of three mappings.") from exc
    if len(specs) != 3:
        raise ValueError(
            f"The frozen ensemble requires exactly three model specs; got {len(specs)}."
        )
    if any(not isinstance(spec, Mapping) for spec in specs):
        raise TypeError("Every model spec must be a mapping.")
    return specs


def _validated_spec_keys(spec: Mapping[str, Any], *, index: int) -> None:
    if any(not isinstance(key, str) for key in spec):
        raise TypeError(f"model_specs[{index}] keys must all be strings.")
    keys = frozenset(spec)
    missing = sorted(_REQUIRED_SPEC_KEYS - keys)
    unexpected = sorted(keys - _REQUIRED_SPEC_KEYS - _OPTIONAL_SPEC_KEYS)
    if missing or unexpected:
        raise ValueError(
            f"model_specs[{index}] has invalid keys; "
            f"missing={missing}, unexpected={unexpected}."
        )


def _feature_names(spec: Mapping[str, Any], *, index: int) -> tuple[str, ...]:
    value = spec["feature_names"]
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"model_specs[{index}]['feature_names'] must be a sequence."
        )
    try:
        names = tuple(value)
    except TypeError as exc:
        raise TypeError(
            f"model_specs[{index}]['feature_names'] must be a sequence."
        ) from exc
    if not names:
        raise ValueError(f"model_specs[{index}]['feature_names'] must not be empty.")
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError(
            f"model_specs[{index}] feature names must be non-empty strings."
        )
    if len(set(names)) != len(names):
        raise ValueError(f"model_specs[{index}] feature names must be unique.")
    return names


def _resolve_era_column(
    specs: tuple[Mapping[str, Any], ...], era_column: str | None
) -> str:
    declared = [spec["era_column"] for spec in specs if "era_column" in spec]
    for value in declared:
        if not isinstance(value, str) or not value:
            raise TypeError("Every declared era_column must be a non-empty string.")
    if era_column is None:
        resolved = declared[0] if declared else "era"
    else:
        if not isinstance(era_column, str) or not era_column:
            raise TypeError("era_column must be a non-empty string or None.")
        resolved = era_column
    if any(value != resolved for value in declared):
        raise ValueError(
            "All model specs must declare the same era_column as the ensemble."
        )
    return resolved


def build_three_tabm_rank_ensemble_predictor(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    era_column: str | None = None,
    prediction_column: str = "prediction",
):
    """Build a portable mean-of-three-per-era-ranks Numerai predictor.

    Each input must be a predictor spec accepted by
    ``build_tabm_numpy_predictor`` (including ``feature_names``).  The three raw
    model outputs are ranked independently within era with average-tie
    percentile ranks.  Those three ranks are averaged directly; the ensemble
    mean is deliberately not ranked again.
    """

    specs = _model_specs_tuple(model_specs)
    for index, spec in enumerate(specs):
        _validated_spec_keys(spec, index=index)
    if not isinstance(prediction_column, str) or not prediction_column:
        raise TypeError("prediction_column must be a non-empty string.")
    resolved_era_column = _resolve_era_column(specs, era_column)

    names_by_model = tuple(
        _feature_names(spec, index=index) for index, spec in enumerate(specs)
    )
    feature_names = names_by_model[0]
    if any(names != feature_names for names in names_by_model[1:]):
        raise ValueError(
            "All three TabM models must use the same frozen feature order."
        )
    if resolved_era_column in feature_names:
        raise ValueError("era_column must not also be a model feature.")

    raw_forwards = []
    model_metadata = []
    for index, spec in enumerate(specs):
        if "prediction_column" in spec:
            value = spec["prediction_column"]
            if not isinstance(value, str) or not value:
                raise TypeError(
                    f"model_specs[{index}]['prediction_column'] must be a "
                    "non-empty string."
                )
        forward_kwargs = {key: spec[key] for key in _FORWARD_KEYS if key in spec}
        forward = build_tabm_numpy_forward(**forward_kwargs)
        if forward.feature_count != len(feature_names):
            raise ValueError(
                f"model_specs[{index}] has {len(feature_names)} feature names, "
                f"but its TabM forward expects {forward.feature_count}."
            )
        raw_forwards.append(forward)
        model_metadata.append(
            {
                "feature_count": forward.feature_count,
                "ensemble_size": forward.ensemble_size,
                "block_count": forward.block_count,
                "batch_size": forward.batch_size,
            }
        )
    frozen_forwards = tuple(raw_forwards)
    frozen_metadata = tuple(model_metadata)

    def predict(live_features, live_benchmark_models):
        import numpy as _np
        import pandas as _pd

        # Numerai supplies this argument, but this residual-signal ensemble has
        # no live benchmark dependency.
        del live_benchmark_models

        if not isinstance(live_features, _pd.DataFrame):
            raise TypeError("live_features must be a pandas DataFrame.")
        if live_features.empty:
            raise ValueError("live_features must contain at least one row.")
        if not live_features.columns.is_unique:
            raise ValueError("live_features column names must be unique.")
        if live_features.index.nlevels != 1:
            raise ValueError("live_features must use a one-dimensional index.")
        if not live_features.index.is_unique:
            raise ValueError("live_features index must be unique.")
        if live_features.index.hasnans:
            raise ValueError("live_features index must not contain missing values.")
        if resolved_era_column not in live_features.columns:
            raise ValueError(
                "live_features is missing required era column "
                f"{resolved_era_column!r}."
            )
        missing = [name for name in feature_names if name not in live_features.columns]
        if missing:
            preview = missing[:5]
            suffix = "..." if len(missing) > len(preview) else ""
            raise ValueError(
                f"live_features is missing required model features: {preview}{suffix}"
            )

        eras = live_features[resolved_era_column]
        if eras.isna().any():
            raise ValueError(
                f"{resolved_era_column!r} must not contain missing values."
            )
        try:
            feature_values = live_features.loc[:, list(feature_names)].to_numpy(
                dtype=_np.float32, copy=True
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("All required model features must be numeric.") from exc

        rank_sum = _np.zeros(len(live_features), dtype=_np.float64)
        for raw_forward in frozen_forwards:
            raw_predictions = raw_forward(feature_values)
            if raw_predictions.shape != (len(live_features),):
                raise RuntimeError(
                    "TabM raw prediction count does not match live feature rows."
                )
            if not _np.isfinite(raw_predictions).all():
                raise FloatingPointError(
                    "TabM raw predictions contain non-finite values."
                )
            raw = _pd.Series(raw_predictions, index=live_features.index)
            ranked = raw.groupby(
                eras, sort=False, dropna=False
            ).rank(method="average", pct=True)
            ranked_values = ranked.to_numpy(dtype=_np.float64, copy=False)
            if not _np.isfinite(ranked_values).all():
                raise FloatingPointError("Per-model ranks contain non-finite values.")
            if ((ranked_values < 0.0) | (ranked_values > 1.0)).any():
                raise RuntimeError("Per-model ranks are outside the [0, 1] interval.")
            rank_sum += ranked_values

        ensemble_values = rank_sum / _np.float64(3.0)
        if not _np.isfinite(ensemble_values).all():
            raise FloatingPointError(
                "Ensemble predictions contain non-finite values."
            )
        if ((ensemble_values < 0.0) | (ensemble_values > 1.0)).any():
            raise RuntimeError(
                "Ensemble predictions are outside the [0, 1] interval."
            )
        output = _pd.DataFrame(
            {prediction_column: ensemble_values}, index=live_features.index
        )
        if not output.index.equals(live_features.index):
            raise RuntimeError("Prediction output index does not match live_features.")
        return output

    predict.__name__ = "predict"
    predict.__qualname__ = "predict"
    predict.feature_names = feature_names
    predict.era_column = resolved_era_column
    predict.prediction_column = prediction_column
    predict.uses_live_benchmark_models = False
    predict.model_count = 3
    predict.ensemble_method = "mean_of_three_per_era_average_percentile_ranks"
    predict.reranks_ensemble_mean = False
    predict.tabm_models = frozen_metadata
    return predict


def _required_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} mapping is required.")
    return value


def _sha256_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a SHA-256 digest.") from exc
    return value.lower()


def _semantic_weight_sha256(spec: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    arrays = []
    for index, block in enumerate(spec["blocks"]):
        for parameter in ("weight", "r", "s", "bias"):
            arrays.append((f"block_{index}_{parameter}", block[parameter]))
    arrays.extend(
        (
            ("output_weight", spec["output_weight"]),
            ("output_bias", spec["output_bias"]),
        )
    )
    for name, value in arrays:
        array = np.ascontiguousarray(value, dtype=np.float32)
        digest.update(name.encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_frozen_ender20_rank_ensemble_from_bundles(
    *,
    bundle_dirs: Sequence[str | Path],
    gate_source_manifest_path: str | Path | None = None,
    era_column: str | None = None,
    prediction_column: str = "prediction",
):
    """Build the frozen ensemble only from the three approved bundle lineages."""

    if isinstance(bundle_dirs, (str, bytes, Mapping)):
        raise TypeError("bundle_dirs must be a sequence of three directories.")
    try:
        directories = tuple(Path(path).expanduser().resolve() for path in bundle_dirs)
    except TypeError as exc:
        raise TypeError("bundle_dirs must be a sequence of three directories.") from exc
    if len(directories) != 3:
        raise ValueError(
            f"The frozen ensemble requires exactly three bundle directories; got {len(directories)}."
        )
    if len(set(directories)) != 3:
        raise ValueError("Frozen ensemble bundle directories must be distinct.")

    from agents.code.modeling.deployment.final_fit_export import (
        GATE_CONFIG_RELATIVE_PREFIX,
        SUPPORTED_RUNS,
        load_gate_source_manifest_pin,
        load_intermediate_predictor_spec,
    )

    gate_source = load_gate_source_manifest_pin(gate_source_manifest_path)
    records: dict[str, dict[str, Any]] = {}
    for directory in directories:
        spec = load_intermediate_predictor_spec(directory)
        provenance_path = directory / "provenance.json"
        try:
            with provenance_path.open("r", encoding="utf-8") as stream:
                provenance = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load bundle provenance: {provenance_path}.") from exc
        provenance = _required_mapping(provenance, "bundle provenance")
        if provenance.get("format") != "numerai-ender20-final-fit-provenance":
            raise ValueError("Bundle provenance format is not the frozen Ender20 format.")
        if provenance.get("format_version") != 1:
            raise ValueError("Bundle provenance format_version must be 1.")
        if provenance.get("artifact_state") != "intermediates_only_no_pickle_no_upload":
            raise ValueError("Bundle provenance artifact_state is not an approved intermediate.")

        config = _required_mapping(provenance.get("config"), "bundle config")
        config_name = config.get("name")
        if config_name not in SUPPORTED_RUNS:
            raise ValueError(f"Unexpected frozen bundle config: {config_name!r}.")
        if config_name in records:
            raise ValueError(f"Duplicate frozen bundle config: {config_name!r}.")
        config_relative_path = f"{GATE_CONFIG_RELATIVE_PREFIX}/{config_name}"
        expected_config_sha256 = gate_source["files"].get(config_relative_path)
        if expected_config_sha256 is None:
            raise ValueError(
                f"Gate source manifest does not pin config {config_relative_path!r}."
            )
        if _sha256_digest(config.get("file_sha256"), "bundle config file_sha256") != expected_config_sha256:
            raise ValueError(
                f"Bundle {config_name!r} config hash does not match the gate source."
            )

        model_seed, sample_seed = SUPPORTED_RUNS[config_name]
        training = _required_mapping(provenance.get("training"), "bundle training")
        sample = _required_mapping(provenance.get("sample"), "bundle sample")
        if training.get("model_seed") != model_seed:
            raise ValueError(f"Bundle {config_name!r} has the wrong model seed.")
        if sample.get("seed") != sample_seed:
            raise ValueError(f"Bundle {config_name!r} has the wrong sample seed.")
        if training.get("target_transform") != _EXPECTED_TARGET_TRANSFORM:
            raise ValueError(f"Bundle {config_name!r} has the wrong target transform.")
        sample_positions_sha256 = _sha256_digest(
            sample.get("manifest_positions_sha256"),
            f"bundle {config_name!r} sample-position hash",
        )

        store = _required_mapping(provenance.get("store"), "bundle store")
        store_values = {}
        for key in _SHARED_STORE_KEYS:
            if key not in store:
                raise ValueError(f"Bundle {config_name!r} store is missing {key!r}.")
            store_values[key] = store[key]
        store_values["metadata_sha256"] = _sha256_digest(
            store_values["metadata_sha256"], "bundle store metadata_sha256"
        )
        if store_values["metadata_sha256"] != gate_source["store_metadata_sha256"]:
            raise ValueError(
                f"Bundle {config_name!r} store metadata does not match the gate source."
            )
        for key in ("manifest_sha256", "feature_sha256", "feature_order_sha256"):
            store_values[key] = _sha256_digest(
                store_values[key], f"bundle store {key}"
            )

        recorded_gate = provenance.get("gate_source")
        if recorded_gate is not None:
            recorded_gate = _required_mapping(recorded_gate, "bundle gate_source")
            if _sha256_digest(
                recorded_gate.get("manifest_sha256"),
                "bundle gate-source manifest_sha256",
            ) != gate_source["manifest_sha256"]:
                raise ValueError(
                    f"Bundle {config_name!r} names a different gate source manifest."
                )
            if recorded_gate.get("store_metadata_relative_path") != gate_source[
                "store_metadata_relative_path"
            ]:
                raise ValueError(
                    f"Bundle {config_name!r} names a different gated store path."
                )
            if _sha256_digest(
                recorded_gate.get("expected_store_metadata_sha256"),
                "bundle expected store metadata SHA-256",
            ) != gate_source["store_metadata_sha256"]:
                raise ValueError(
                    f"Bundle {config_name!r} records a different gated store hash."
                )
            if recorded_gate.get("config_relative_path") != config_relative_path:
                raise ValueError(
                    f"Bundle {config_name!r} records a different gated config path."
                )
            if _sha256_digest(
                recorded_gate.get("expected_config_sha256"),
                "bundle expected config SHA-256",
            ) != expected_config_sha256:
                raise ValueError(
                    f"Bundle {config_name!r} records a different gated config hash."
                )

        records[config_name] = {
            "spec": spec,
            "store": store_values,
            "sample_positions_sha256": sample_positions_sha256,
            "weight_sha256": _semantic_weight_sha256(spec),
        }

    expected_names = tuple(SUPPORTED_RUNS)
    if set(records) != set(expected_names):
        missing = sorted(set(expected_names) - set(records))
        unexpected = sorted(set(records) - set(expected_names))
        raise ValueError(
            f"Frozen bundle config set is incomplete; missing={missing}, unexpected={unexpected}."
        )
    ordered = tuple(records[name] for name in expected_names)
    reference_store = ordered[0]["store"]
    for name, record in zip(expected_names[1:], ordered[1:]):
        if record["store"] != reference_store:
            raise ValueError(f"Bundle {name!r} does not share the frozen store lineage.")
    weight_hashes = tuple(record["weight_sha256"] for record in ordered)
    if len(set(weight_hashes)) != 3:
        raise ValueError("Frozen ensemble bundles must contain three distinct models.")
    base_sample_hash = ordered[0]["sample_positions_sha256"]
    if ordered[1]["sample_positions_sha256"] != base_sample_hash:
        raise ValueError(
            "The two sample-seed-1337 bundles must share manifest positions."
        )
    if ordered[2]["sample_positions_sha256"] == base_sample_hash:
        raise ValueError(
            "The sample-seed-2027 bundle must use different manifest positions."
        )

    refreshed_gate_source = load_gate_source_manifest_pin(gate_source["path"])
    if refreshed_gate_source["manifest_sha256"] != gate_source["manifest_sha256"]:
        raise RuntimeError("Frozen gate source manifest changed during assembly.")
    predictor = build_three_tabm_rank_ensemble_predictor(
        model_specs=[record["spec"] for record in ordered],
        era_column=era_column,
        prediction_column=prediction_column,
    )
    predictor.bundle_config_names = expected_names
    predictor.bundle_weight_sha256 = weight_hashes
    predictor.gate_source_manifest_sha256 = gate_source["manifest_sha256"]
    predictor.store_generation_id = reference_store["generation_id"]
    return predictor


__all__ = [
    "build_frozen_ender20_rank_ensemble_from_bundles",
    "build_three_tabm_rank_ensemble_predictor",
]
