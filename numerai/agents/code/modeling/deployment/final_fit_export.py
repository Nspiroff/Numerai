"""Conservative two-pass final fitting for the frozen Ender20 TabM runs.

This module stops at portable, non-pickle intermediates.  It deliberately does
not create a Numerai upload artifact: the generated NumPy arrays and JSON
documents are intended to be rebuilt and cloudpickled later in the validated
Python 3.12 runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
from typing import Any
import uuid

import numpy as np

from agents.code.modeling.utils.config import load_config
from agents.code.modeling.utils.constants import NUMERAI_DIR, REPO_DIR
from agents.code.modeling.utils.model_data import ModelDataBatch, build_x_cols


SUPPORTED_RUNS: Mapping[str, tuple[int, int]] = {
    "scale_disk_tabm_k64_train500k.py": (1337, 1337),
    "scale_disk_tabm_k64_train500k_seed2027.py": (2027, 1337),
    "scale_disk_tabm_k64_train500k_sample_seed2027.py": (1337, 2027),
}

DEFAULT_GATE_SOURCE_MANIFEST_PATH = (
    NUMERAI_DIR
    / "agents"
    / "experiments"
    / "ender20_nn_architecture_v53"
    / "gate_source_manifest.json"
)
GATE_STORE_METADATA_RELATIVE_PATH = (
    "v5.3/target_ender_20_feature_store/metadata.json"
)
GATE_CONFIG_RELATIVE_PREFIX = (
    "agents/experiments/ender20_nn_architecture_v53/configs"
)

_SPEC_FORMAT = "numerai-tabm-numpy-predictor-spec"
_SPEC_VERSION = 1
_PROVENANCE_FORMAT = "numerai-ender20-final-fit-provenance"
_PROVENANCE_VERSION = 1
_POSITION_HASH_ALGORITHM = "sha256-int64-le-c-order-v1"
_WEIGHT_NAMES = ("weight", "r", "s", "bias")


@dataclass(frozen=True)
class FinalFitProtocol:
    """Frozen final-fit controls; smaller values may be injected in unit tests."""

    sample_size: int = 500_000
    selector_max_epochs: int = 30
    selector_patience: int = 4
    selector_val_fraction: float = 0.1
    internal_val_embargo: int = 52
    predictor_batch_size: int = 32

    def validate(self) -> None:
        for name in (
            "sample_size",
            "selector_max_epochs",
            "selector_patience",
            "internal_val_embargo",
            "predictor_batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"FinalFitProtocol.{name} must be a positive integer.")
        if not 0.0 < float(self.selector_val_fraction) < 1.0:
            raise ValueError(
                "FinalFitProtocol.selector_val_fraction must be strictly between 0 and 1."
            )


@dataclass(frozen=True)
class FinalFitIntermediates:
    output_dir: Path
    weights_path: Path
    predictor_spec_path: Path
    sample_positions_path: Path
    provenance_path: Path
    best_epoch: int
    sample_positions_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a SHA-256 digest.") from exc
    return value.lower()


def load_gate_source_manifest_pin(
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the frozen gate source manifest used to bind final artifacts."""

    path = Path(
        manifest_path
        if manifest_path is not None
        else DEFAULT_GATE_SOURCE_MANIFEST_PATH
    ).expanduser().resolve()
    try:
        with path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load frozen gate source manifest: {path}.") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Frozen gate source manifest must be a JSON object.")
    if manifest.get("hash_algorithm") != "sha256":
        raise ValueError("Frozen gate source manifest must use SHA-256.")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Frozen gate source manifest files mapping is required.")
    normalized_files: dict[str, str] = {}
    for name, digest in files.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Frozen gate source manifest file names must be strings.")
        normalized_files[name.replace("\\", "/")] = _validated_sha256(
            digest, f"gate source digest for {name!r}"
        )
    if GATE_STORE_METADATA_RELATIVE_PATH not in normalized_files:
        raise ValueError(
            "Frozen gate source manifest does not pin the Ender20 store metadata."
        )
    return {
        "path": path,
        "manifest_sha256": _sha256_file(path),
        "files": normalized_files,
        "store_metadata_relative_path": GATE_STORE_METADATA_RELATIVE_PATH,
        "store_metadata_sha256": normalized_files[
            GATE_STORE_METADATA_RELATIVE_PATH
        ],
    }


def _feature_order_sha256(feature_names: Sequence[str]) -> str:
    encoded = json.dumps(
        list(feature_names), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _positions_sha256(positions: np.ndarray) -> str:
    canonical = np.ascontiguousarray(positions, dtype="<i8")
    return _sha256_bytes(canonical.tobytes(order="C"))


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} must be {expected!r}; got {actual!r}.")


def _validate_frozen_config(
    config: dict[str, Any], config_path: Path, protocol: FinalFitProtocol
) -> tuple[int, int]:
    try:
        expected_model_seed, expected_sample_seed = SUPPORTED_RUNS[config_path.name]
    except KeyError as exc:
        raise ValueError(
            "Final fitting supports only the three frozen scale-disk TabM configs; "
            f"got {config_path.name!r}."
        ) from exc

    data = config.get("data")
    model = config.get("model")
    training = config.get("training")
    preprocessing = config.get("preprocessing")
    if not all(isinstance(value, dict) for value in (data, model, training, preprocessing)):
        raise ValueError("Frozen config data/model/training/preprocessing sections are required.")
    params = model.get("params")
    target_transform = model.get("target_transform")
    cv = training.get("cv")
    if not isinstance(params, dict) or not isinstance(target_transform, dict):
        raise ValueError("Frozen model params and target_transform mappings are required.")
    if not isinstance(cv, dict):
        raise ValueError("Frozen training.cv mapping is required.")

    expected_values = (
        (data.get("data_version"), "v5.3", "data.data_version"),
        (data.get("feature_set"), "all", "data.feature_set"),
        (data.get("target_col"), "target_ender_20", "data.target_col"),
        (data.get("era_col"), "era", "data.era_col"),
        (data.get("id_col"), "id", "data.id_col"),
        (
            data.get("benchmark_model"),
            "v53_lgbm_ender20",
            "data.benchmark_model",
        ),
        (
            data.get("embargo_eras"),
            protocol.internal_val_embargo,
            "data.embargo_eras",
        ),
        (data.get("require_benchmark_coverage"), True, "data.require_benchmark_coverage"),
        (model.get("type"), "TorchTabularRegressor", "model.type"),
        (
            model.get("x_groups"),
            ["features", "era", "benchmark_models"],
            "model.x_groups",
        ),
        (target_transform.get("type"), "residual_to_benchmark", "target_transform.type"),
        (
            target_transform.get("benchmark_col"),
            "v53_lgbm_ender20",
            "target_transform.benchmark_col",
        ),
        (target_transform.get("era_col"), "era", "target_transform.era_col"),
        (target_transform.get("per_era"), True, "target_transform.per_era"),
        (
            target_transform.get("fit_intercept"),
            True,
            "target_transform.fit_intercept",
        ),
        (params.get("architecture"), "tabm", "model.params.architecture"),
        (params.get("activation"), "relu", "model.params.activation"),
        (params.get("tabm_arch_type"), "tabm", "model.params.tabm_arch_type"),
        (params.get("tabm_k"), 64, "model.params.tabm_k"),
        (params.get("tabm_width"), 512, "model.params.tabm_width"),
        (params.get("tabm_blocks"), 3, "model.params.tabm_blocks"),
        (params.get("dropout"), 0.1, "model.params.dropout"),
        (params.get("batch_size"), 1024, "model.params.batch_size"),
        (
            params.get("prediction_batch_size"),
            2048,
            "model.params.prediction_batch_size",
        ),
        (params.get("learning_rate"), 0.002, "model.params.learning_rate"),
        (params.get("weight_decay"), 0.0003, "model.params.weight_decay"),
        (
            params.get("max_epochs"),
            protocol.selector_max_epochs,
            "model.params.max_epochs",
        ),
        (
            params.get("patience"),
            protocol.selector_patience,
            "model.params.patience",
        ),
        (
            params.get("val_fraction"),
            protocol.selector_val_fraction,
            "model.params.val_fraction",
        ),
        (params.get("val_split"), "recent_eras", "model.params.val_split"),
        (
            params.get("internal_val_embargo"),
            protocol.internal_val_embargo,
            "model.params.internal_val_embargo",
        ),
        (params.get("feature_center"), 2.0, "model.params.feature_center"),
        (params.get("feature_scale"), 2.0, "model.params.feature_scale"),
        (params.get("device"), "cuda", "model.params.device"),
        (params.get("amp"), True, "model.params.amp"),
        (params.get("seed"), expected_model_seed, "model.params.seed"),
        (
            training.get("max_train_samples"),
            protocol.sample_size,
            "training.max_train_samples",
        ),
        (training.get("sample_seed"), expected_sample_seed, "training.sample_seed"),
        (training.get("data_mode"), "disk_feature_store", "training.data_mode"),
        (
            cv.get("embargo"),
            protocol.internal_val_embargo,
            "training.cv.embargo",
        ),
        (cv.get("enabled"), True, "training.cv.enabled"),
        (cv.get("mode"), "expanding", "training.cv.mode"),
        (cv.get("n_splits"), 5, "training.cv.n_splits"),
        (preprocessing.get("nan_missing_all_twos"), False, "preprocessing.nan_missing_all_twos"),
        (preprocessing.get("missing_value"), 2.0, "preprocessing.missing_value"),
    )
    for actual, expected, label in expected_values:
        _require_equal(actual, expected, label)
    if "full_data_path" in data or "benchmark_data_path" in data:
        raise ValueError("Frozen disk-store configs must not name eager data paths.")
    if float(target_transform.get("proportion", 1.0)) != 1.0:
        raise ValueError("Frozen residual target transform proportion must be 1.0.")
    return expected_model_seed, expected_sample_seed


def _resolve_store_path(config: dict[str, Any]) -> Path:
    data = config["data"]
    primary = data.get("disk_feature_store_path")
    alias = data.get("feature_store_path")
    if primary is not None and alias is not None and Path(primary) != Path(alias):
        raise ValueError("data.disk_feature_store_path and data.feature_store_path disagree.")
    configured = primary if primary is not None else alias
    if configured is None:
        configured = f"{data['data_version']}/{data['target_col']}_feature_store"
    candidate = Path(configured).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] == NUMERAI_DIR.name:
        return (REPO_DIR / candidate).resolve()
    return (NUMERAI_DIR / candidate).resolve()


def _load_configured_feature_order(
    config: dict[str, Any], _config_path: Path
) -> tuple[str, ...]:
    data = config["data"]
    features_path = NUMERAI_DIR / data["data_version"] / "features.json"
    try:
        with features_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        feature_names = metadata["feature_sets"][data["feature_set"]]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Cannot load configured feature order from {features_path}."
        ) from exc
    return _validated_feature_names(feature_names, "configured feature order")


def _validated_feature_names(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of names.")
    try:
        names = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{label} must be a sequence of names.") from exc
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{label} must contain non-empty string names.")
    if len(names) != len(set(names)):
        raise ValueError(f"{label} must contain unique names.")
    return names


def _consecutive_eras(values: Sequence[Any]) -> tuple[str, ...]:
    eras = tuple(dict.fromkeys(str(value) for value in values))
    if not eras:
        raise ValueError("Disk feature store contains no eras.")
    try:
        numeric = [(int(era), era) for era in eras]
    except ValueError as exc:
        raise ValueError("Every frozen Ender20 era must be an integer string.") from exc
    if len({number for number, _ in numeric}) != len(numeric):
        raise ValueError("Disk feature-store eras have ambiguous numeric spellings.")
    numeric.sort(key=lambda item: item[0])
    numbers = [item[0] for item in numeric]
    expected = list(range(numbers[0], numbers[-1] + 1))
    if numbers != expected:
        missing = sorted(set(expected) - set(numbers))
        raise ValueError(
            "Disk feature-store eras are not consecutive; "
            f"first missing numeric eras: {missing[:5]}."
        )
    return tuple(era for _, era in numeric)


def _subset_value(value: Any, indices: np.ndarray) -> Any:
    if value is None:
        return None
    if getattr(value, "is_disk_feature_view", False):
        return value.take(indices)
    if hasattr(value, "iloc"):
        return value.iloc[indices]
    return value[indices]


def _draw_sample_once(
    data: ModelDataBatch, sample_size: int, sample_seed: int
) -> tuple[ModelDataBatch, np.ndarray]:
    row_count = len(data.X)
    if row_count <= sample_size:
        raise ValueError(
            "Frozen final fit requires a real without-replacement sample: "
            f"store rows={row_count:,}, requested={sample_size:,}."
        )
    rng = np.random.default_rng(sample_seed)
    relative_positions = rng.choice(row_count, size=sample_size, replace=False)
    sampled = ModelDataBatch(
        X=_subset_value(data.X, relative_positions),
        y=_subset_value(data.y, relative_positions),
        era=_subset_value(data.era, relative_positions),
        id=_subset_value(data.id, relative_positions) if data.id is not None else None,
    )
    manifest_positions = np.asarray(
        getattr(sampled.X, "manifest_positions", relative_positions), dtype=np.int64
    )
    if manifest_positions.shape != (sample_size,):
        raise ValueError("Sampled manifest positions have the wrong shape.")
    if len(np.unique(manifest_positions)) != sample_size:
        raise ValueError("Sampled manifest positions are not unique.")
    if manifest_positions.min() < 0 or manifest_positions.max() >= row_count:
        raise ValueError("Sampled manifest positions are outside the store manifest.")
    return sampled, np.array(manifest_positions, dtype=np.int64, order="C", copy=True)


def _validate_store(
    loader: Any, configured_features: tuple[str, ...]
) -> tuple[dict[str, Any], tuple[str, ...], str]:
    stored_features = _validated_feature_names(
        loader.feature_columns, "disk feature-store feature order"
    )
    if stored_features != configured_features:
        raise ValueError(
            "Configured data.feature_set does not exactly match the disk "
            "feature-store feature order."
        )
    diagnostics = dict(loader.diagnostics)
    required = {
        "directory",
        "feature_path",
        "manifest_path",
        "generation_id",
        "row_count",
        "feature_count",
        "feature_bytes",
        "manifest_bytes",
        "feature_order_sha256",
        "feature_sha256",
        "manifest_sha256",
    }
    missing = sorted(required - set(diagnostics))
    if missing:
        raise ValueError(f"Disk feature-store diagnostics are missing: {missing}.")
    _require_equal(
        int(diagnostics["feature_count"]), len(stored_features), "store feature_count"
    )
    _require_equal(
        int(diagnostics["row_count"]), len(loader.manifest), "store row_count"
    )
    expected_feature_hash = _feature_order_sha256(stored_features)
    _require_equal(
        diagnostics["feature_order_sha256"],
        expected_feature_hash,
        "store feature_order_sha256",
    )
    for key in ("feature_sha256", "manifest_sha256"):
        value = diagnostics[key]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Store {key} is not a SHA-256 digest.")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(f"Store {key} is not a SHA-256 digest.") from exc
    metadata_path = Path(loader.directory) / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Disk feature-store metadata is missing: {metadata_path}.")
    return diagnostics, stored_features, _sha256_file(metadata_path)


def _best_epoch(model: Any, maximum: int) -> int:
    value = getattr(model, "best_epoch_", None)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Selector best_epoch_ must be an integer in the frozen range.")
    try:
        epoch = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Selector did not expose a valid best_epoch_."
        ) from exc
    if value != epoch or not 1 <= epoch <= maximum:
        raise ValueError(
            f"Selector best_epoch_ must be in [1, {maximum}]; got {value!r}."
        )
    return epoch


def _validate_fixed_fit(model: Any, expected_epochs: int) -> None:
    _require_equal(getattr(model, "val_split", None), "none", "fixed model val_split")
    _require_equal(getattr(model, "val_fraction", None), 0.0, "fixed model val_fraction")
    _require_equal(
        getattr(model, "max_epochs", None), expected_epochs, "fixed model max_epochs"
    )
    history = getattr(model, "training_history_", None)
    if not isinstance(history, list) or len(history) != expected_epochs:
        raise RuntimeError(
            "Fixed pass did not retain evidence for every requested terminal epoch: "
            f"expected {expected_epochs}, got "
            f"{len(history) if isinstance(history, list) else None}."
        )


def _validated_array(name: str, value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Exported weight {name!r} must be a CPU NumPy array.")
    if value.dtype != np.float32:
        raise TypeError(f"Exported weight {name!r} must have dtype float32.")
    if value.ndim == 0 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"Exported weight {name!r} must be non-empty and finite.")
    return np.array(value, dtype=np.float32, order="C", copy=True)


def _split_predictor_spec(
    spec: Mapping[str, Any], expected_features: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    feature_names = _validated_feature_names(spec.get("feature_names"), "export feature order")
    if feature_names != expected_features:
        raise ValueError("Exported predictor feature order does not match the store.")
    blocks = spec.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("Exported TabM predictor must contain at least one block.")

    arrays: dict[str, np.ndarray] = {}
    block_refs: list[dict[str, str]] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, Mapping) or set(block) != set(_WEIGHT_NAMES):
            raise ValueError(f"Exported TabM block {index} has an unsupported layout.")
        refs: dict[str, str] = {}
        for parameter in _WEIGHT_NAMES:
            key = f"block_{index}_{parameter}"
            arrays[key] = _validated_array(key, block[parameter])
            refs[parameter] = key
        block_refs.append(refs)
    arrays["output_weight"] = _validated_array(
        "output_weight", spec.get("output_weight")
    )
    arrays["output_bias"] = _validated_array("output_bias", spec.get("output_bias"))

    scalar_fields = {
        "feature_center": float(spec.get("feature_center")),
        "feature_scale": float(spec.get("feature_scale")),
        "batch_size": int(spec.get("batch_size")),
        "activation": spec.get("activation"),
        "era_column": spec.get("era_column"),
        "prediction_column": spec.get("prediction_column"),
    }
    if not np.isfinite(scalar_fields["feature_center"]):
        raise ValueError("Export feature_center must be finite.")
    if (
        not np.isfinite(scalar_fields["feature_scale"])
        or scalar_fields["feature_scale"] == 0.0
    ):
        raise ValueError("Export feature_scale must be finite and non-zero.")
    if scalar_fields["batch_size"] <= 0:
        raise ValueError("Export batch_size must be positive.")
    _require_equal(scalar_fields["activation"], "relu", "export activation")
    for key in ("era_column", "prediction_column"):
        if not isinstance(scalar_fields[key], str) or not scalar_fields[key]:
            raise ValueError(f"Export {key} must be a non-empty string.")

    array_descriptors = {
        name: {
            "dtype": "float32",
            "shape": list(array.shape),
            "sha256": _sha256_bytes(array.tobytes(order="C")),
        }
        for name, array in arrays.items()
    }
    portable = {
        "format": _SPEC_FORMAT,
        "format_version": _SPEC_VERSION,
        "builder": "agents.code.modeling.deployment.tabm_numpy.build_tabm_numpy_predictor",
        "weights_file": "weights.npz",
        "feature_names": list(feature_names),
        "blocks": block_refs,
        "output_weight": "output_weight",
        "output_bias": "output_bias",
        "arrays": array_descriptors,
        **scalar_fields,
    }
    return portable, arrays


def _write_bytes_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_npy_fsynced(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _write_npz_fsynced(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def write_intermediate_bundle(
    output_dir: str | Path,
    *,
    predictor_spec: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    sample_manifest_positions: np.ndarray,
    provenance: Mapping[str, Any],
    replace: Callable[[str | bytes | os.PathLike, str | bytes | os.PathLike], None] = os.replace,
) -> FinalFitIntermediates:
    """Publish the four-file intermediate bundle with one atomic directory rename."""

    canonical_positions = np.ascontiguousarray(sample_manifest_positions, dtype="<i8")
    if canonical_positions.ndim != 1 or canonical_positions.size == 0:
        raise ValueError("sample_manifest_positions must be a non-empty vector.")
    expected_positions_hash = provenance.get("sample", {}).get(
        "manifest_positions_sha256"
    )
    _require_equal(
        _positions_sha256(canonical_positions),
        expected_positions_hash,
        "sample manifest-position hash",
    )

    destination = Path(output_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to replace existing final-fit output directory: {destination}"
        )
    staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        weights_path = staging / "weights.npz"
        spec_path = staging / "predictor_spec.json"
        positions_path = staging / "sample_manifest_positions.npy"
        provenance_path = staging / "provenance.json"

        _write_npz_fsynced(weights_path, arrays)
        _write_bytes_fsynced(spec_path, _canonical_json_bytes(dict(predictor_spec)))
        _write_npy_fsynced(positions_path, canonical_positions)

        final_provenance = deepcopy(dict(provenance))
        final_provenance["intermediates"] = {
            "weights": {
                "file": weights_path.name,
                "sha256": _sha256_file(weights_path),
            },
            "predictor_spec": {
                "file": spec_path.name,
                "sha256": _sha256_file(spec_path),
            },
            "sample_manifest_positions": {
                "file": positions_path.name,
                "sha256": _sha256_file(positions_path),
            },
        }
        _write_bytes_fsynced(
            provenance_path, _canonical_json_bytes(final_provenance)
        )
        replace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    best_epoch = int(provenance["training"]["selected_best_epoch"])
    positions_hash = str(provenance["sample"]["manifest_positions_sha256"])
    return FinalFitIntermediates(
        output_dir=destination,
        weights_path=destination / "weights.npz",
        predictor_spec_path=destination / "predictor_spec.json",
        sample_positions_path=destination / "sample_manifest_positions.npy",
        provenance_path=destination / "provenance.json",
        best_epoch=best_epoch,
        sample_positions_sha256=positions_hash,
    )


def load_intermediate_predictor_spec(bundle_dir: str | Path) -> dict[str, Any]:
    """Rehydrate tabm_numpy builder arguments without importing torch or pickle."""

    directory = Path(bundle_dir).expanduser().resolve()
    spec_path = directory / "predictor_spec.json"
    weights_path = directory / "weights.npz"
    positions_path = directory / "sample_manifest_positions.npy"
    provenance_path = directory / "provenance.json"
    with provenance_path.open("r", encoding="utf-8") as stream:
        provenance = json.load(stream)
    _require_equal(provenance.get("format"), _PROVENANCE_FORMAT, "provenance format")
    _require_equal(
        provenance.get("format_version"),
        _PROVENANCE_VERSION,
        "provenance format version",
    )
    intermediate_paths = {
        "weights": weights_path,
        "predictor_spec": spec_path,
        "sample_manifest_positions": positions_path,
    }
    recorded_intermediates = provenance.get("intermediates")
    if not isinstance(recorded_intermediates, dict):
        raise ValueError("Provenance intermediates mapping is required.")
    for name, artifact_path in intermediate_paths.items():
        record = recorded_intermediates.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"Provenance is missing the {name!r} intermediate.")
        _require_equal(record.get("file"), artifact_path.name, f"{name} filename")
        _require_equal(
            _sha256_file(artifact_path), record.get("sha256"), f"{name} file sha256"
        )
    positions = np.load(positions_path, allow_pickle=False)
    if positions.ndim != 1 or positions.dtype != np.dtype("int64"):
        raise ValueError("Sample manifest-position intermediate must be int64 vector.")
    _require_equal(
        _positions_sha256(positions),
        provenance.get("sample", {}).get("manifest_positions_sha256"),
        "sample manifest-position hash",
    )
    with spec_path.open("r", encoding="utf-8") as stream:
        portable = json.load(stream)
    _require_equal(portable.get("format"), _SPEC_FORMAT, "predictor spec format")
    _require_equal(portable.get("format_version"), _SPEC_VERSION, "predictor spec version")
    _require_equal(portable.get("weights_file"), weights_path.name, "predictor weights file")
    descriptors = portable.get("arrays")
    if not isinstance(descriptors, dict):
        raise ValueError("Predictor spec arrays mapping is required.")
    with np.load(weights_path, allow_pickle=False) as archive:
        if set(archive.files) != set(descriptors):
            raise ValueError("Predictor weights do not match the spec array names.")
        arrays = {}
        for name, descriptor in descriptors.items():
            if not isinstance(descriptor, dict):
                raise ValueError(f"Predictor array descriptor {name!r} is malformed.")
            array = np.array(archive[name], dtype=np.float32, order="C", copy=True)
            _require_equal(str(array.dtype), descriptor.get("dtype"), f"{name} dtype")
            _require_equal(list(array.shape), descriptor.get("shape"), f"{name} shape")
            _require_equal(
                _sha256_bytes(array.tobytes(order="C")),
                descriptor.get("sha256"),
                f"{name} sha256",
            )
            arrays[name] = array
    blocks = [
        {parameter: arrays[references[parameter]] for parameter in _WEIGHT_NAMES}
        for references in portable["blocks"]
    ]
    return {
        "feature_names": tuple(portable["feature_names"]),
        "blocks": blocks,
        "output_weight": arrays[portable["output_weight"]],
        "output_bias": arrays[portable["output_bias"]],
        "feature_center": float(portable["feature_center"]),
        "feature_scale": float(portable["feature_scale"]),
        "batch_size": int(portable["batch_size"]),
        "activation": portable["activation"],
        "era_column": portable["era_column"],
        "prediction_column": portable["prediction_column"],
    }


def run_final_fit_export(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    gate_source_manifest_path: str | Path | None = None,
    protocol: FinalFitProtocol = FinalFitProtocol(),
    loader_factory: Callable[..., Any] | None = None,
    feature_order_loader: Callable[[dict[str, Any], Path], Sequence[str]] | None = None,
    model_builder: Callable[..., Any] | None = None,
    spec_exporter: Callable[..., Mapping[str, Any]] | None = None,
    utcnow: Callable[[], datetime] | None = None,
) -> FinalFitIntermediates:
    """Run selector + fixed-epoch final fit and publish non-pickle intermediates."""

    protocol.validate()
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Final-fit config does not exist: {path}")
    initial_config_sha256 = _sha256_file(path)
    config = load_config(path)
    if not isinstance(config, dict):
        raise TypeError("Final-fit config must evaluate to a dictionary.")
    model_seed, sample_seed = _validate_frozen_config(config, path, protocol)
    canonical_config_sha256 = _sha256_bytes(_canonical_json_bytes(config))
    gate_source = load_gate_source_manifest_pin(gate_source_manifest_path)
    config_relative_path = f"{GATE_CONFIG_RELATIVE_PREFIX}/{path.name}"
    expected_config_sha256 = gate_source["files"].get(config_relative_path)
    if expected_config_sha256 is None:
        raise ValueError(
            "Frozen gate source manifest does not pin final-fit config "
            f"{config_relative_path!r}."
        )
    _require_equal(
        initial_config_sha256,
        expected_config_sha256,
        "final-fit config SHA-256 pinned by the gate",
    )

    if loader_factory is None:
        from agents.code.modeling.utils.disk_feature_store import (
            DiskFeatureStoreLoader,
        )

        loader_factory = DiskFeatureStoreLoader
    if feature_order_loader is None:
        feature_order_loader = _load_configured_feature_order
    if model_builder is None:
        from agents.code.modeling.utils.model_factory import build_model

        model_builder = build_model
    if spec_exporter is None:
        from agents.code.modeling.deployment.tabm_export import (
            extract_tabm_numpy_predictor_spec,
        )

        spec_exporter = extract_tabm_numpy_predictor_spec
    if utcnow is None:
        utcnow = lambda: datetime.now(timezone.utc)

    data_config = config["data"]
    model_config = config["model"]
    model_params = model_config["params"]
    configured_features = _validated_feature_names(
        feature_order_loader(config, path), "configured feature order"
    )
    store_path = _resolve_store_path(config)
    loader = loader_factory(
        store_path,
        era_col=data_config["era_col"],
        target_col=data_config["target_col"],
        id_col=data_config["id_col"],
        benchmark_col=data_config["benchmark_model"],
    )
    try:
        store_diagnostics, feature_names, initial_metadata_sha256 = _validate_store(
            loader, configured_features
        )
        _require_equal(
            initial_metadata_sha256,
            gate_source["store_metadata_sha256"],
            "feature-store metadata SHA-256 pinned by the gate",
        )
        eras = _consecutive_eras(loader.eras)
        x_cols = build_x_cols(
            x_groups=model_config["x_groups"],
            features=feature_names,
            benchmark_cols=[data_config["benchmark_model"]],
            era_col=data_config["era_col"],
            id_col=data_config["id_col"],
        )
        loader.configure_x_cols(x_cols)
        all_data = loader.load(eras)
        _require_equal(len(all_data.X), int(store_diagnostics["row_count"]), "loaded row count")
        sampled, manifest_positions = _draw_sample_once(
            all_data, protocol.sample_size, sample_seed
        )
        positions_hash = _positions_sha256(manifest_positions)

        selector_params = deepcopy(model_params)
        selector = model_builder(
            model_config["type"],
            selector_params,
            deepcopy(model_config),
            feature_cols=list(feature_names),
        )
        selector.fit(sampled.X, sampled.y)
        selected_epoch = _best_epoch(selector, protocol.selector_max_epochs)

        fixed_params = deepcopy(model_params)
        fixed_params.update(
            {
                "val_split": "none",
                "val_fraction": 0.0,
                "max_epochs": selected_epoch,
            }
        )
        fixed = model_builder(
            model_config["type"],
            fixed_params,
            deepcopy(model_config),
            feature_cols=list(feature_names),
        )
        fixed.fit(sampled.X, sampled.y)
        _validate_fixed_fit(fixed, selected_epoch)
        raw_spec = spec_exporter(
            fixed,
            batch_size=protocol.predictor_batch_size,
            era_column=data_config["era_col"],
            prediction_column="prediction",
        )
        predictor_spec, arrays = _split_predictor_spec(raw_spec, feature_names)

        metadata_path = Path(loader.directory) / "metadata.json"
        if _sha256_file(path) != initial_config_sha256:
            raise RuntimeError("Frozen final-fit config changed while training.")
        if _sha256_file(metadata_path) != initial_metadata_sha256:
            raise RuntimeError("Disk feature-store metadata changed while training.")
        if _sha256_file(gate_source["path"]) != gate_source["manifest_sha256"]:
            raise RuntimeError("Frozen gate source manifest changed while training.")

        created = utcnow()
        if created.tzinfo is None:
            raise ValueError("utcnow() must return a timezone-aware datetime.")
        provenance = {
            "format": _PROVENANCE_FORMAT,
            "format_version": _PROVENANCE_VERSION,
            "created_at_utc": created.astimezone(timezone.utc).isoformat(),
            "artifact_state": "intermediates_only_no_pickle_no_upload",
            "gate_source": {
                "manifest_path": str(gate_source["path"]),
                "manifest_sha256": gate_source["manifest_sha256"],
                "store_metadata_relative_path": gate_source[
                    "store_metadata_relative_path"
                ],
                "expected_store_metadata_sha256": gate_source[
                    "store_metadata_sha256"
                ],
                "config_relative_path": config_relative_path,
                "expected_config_sha256": expected_config_sha256,
            },
            "config": {
                "path": str(path),
                "name": path.name,
                "file_sha256": initial_config_sha256,
                "canonical_sha256": canonical_config_sha256,
            },
            "store": {
                **store_diagnostics,
                "metadata_sha256": initial_metadata_sha256,
                "era_start": eras[0],
                "era_end": eras[-1],
                "era_count": len(eras),
                "consecutive_eras": True,
            },
            "sample": {
                "method": "numpy.default_rng.choice_without_replacement",
                "seed": sample_seed,
                "row_count": len(manifest_positions),
                "manifest_positions_hash_algorithm": _POSITION_HASH_ALGORITHM,
                "manifest_positions_sha256": positions_hash,
            },
            "training": {
                "model_seed": model_seed,
                "target_transform": deepcopy(model_config["target_transform"]),
                "selector": {
                    "val_split": "recent_eras",
                    "val_fraction": protocol.selector_val_fraction,
                    "internal_val_embargo": protocol.internal_val_embargo,
                    "max_epochs": protocol.selector_max_epochs,
                    "patience": protocol.selector_patience,
                },
                "selected_best_epoch": selected_epoch,
                "fixed_pass": {
                    "fresh_model": True,
                    "same_sample_manifest_positions_sha256": positions_hash,
                    "val_split": "none",
                    "val_fraction": 0.0,
                    "max_epochs": selected_epoch,
                    "terminal_state_required": True,
                },
            },
            "export": {
                "bridge": "extract_tabm_numpy_predictor_spec",
                "predictor_spec_format": _SPEC_FORMAT,
                "feature_order_sha256": _feature_order_sha256(feature_names),
                "pickle_created": False,
                "upload_performed": False,
                "intended_rebuild_runtime": {
                    "python_major_minor": "3.12",
                    "requires": ["numpy", "pandas", "cloudpickle"],
                },
            },
            "build_environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
        }
        return write_intermediate_bundle(
            output_dir,
            predictor_spec=predictor_spec,
            arrays=arrays,
            sample_manifest_positions=manifest_positions,
            provenance=provenance,
        )
    finally:
        loader.close()


__all__ = [
    "FinalFitIntermediates",
    "FinalFitProtocol",
    "DEFAULT_GATE_SOURCE_MANIFEST_PATH",
    "GATE_CONFIG_RELATIVE_PREFIX",
    "GATE_STORE_METADATA_RELATIVE_PATH",
    "SUPPORTED_RUNS",
    "load_gate_source_manifest_pin",
    "load_intermediate_predictor_spec",
    "run_final_fit_export",
    "write_intermediate_bundle",
]
