"""Evaluate the frozen Ender20 auxiliary-target rank-ensemble protocol.

This module is deliberately training-free.  It validates sealed pipeline
artifacts and exposes write-once stages for component sealing, calibration,
locked scout evaluation, and the two confirmation gates.  It never packages,
uploads, assigns, submits, or stakes a model.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from agents.code.analysis import evaluate_ender20_hybrid_stability as hybrid
from agents.code.analysis import evaluate_xerxes20_lgbm_challenger as xerxes
from agents.code.data.build_full_datasets import (
    feature_order_sha256,
    parquet_source_fingerprint,
)
from agents.code.metrics import numerai_metrics
from agents.code.modeling.utils.numerai_cv import era_cv_splits
from agents.code.modeling.utils.pipeline import (
    PREDICTION_SEMANTICS_METADATA_KEY,
    build_prediction_semantics,
)


EXPERIMENT_NAME = "ender20_aux_target_rank_ensemble_v53"
AS_OF_DATE = "2026-08-03"
PRE_SCORING_COMMIT = "ef4ee304d6088f10d27e4d49a80d67ec925dbbf3"
SOURCE_MANIFEST_SHA256 = (
    "3cc96dce9938306cc1f2e7d4ef6b6628f24494f5c30a1ca87d791b64ace662a8"
)
GATE_SHA256 = "c851e3e0637e26bff5b2c26eda5752a46a9d72fce2621678bd39ffa320983ffe"

ID_COLUMN = "id"
ERA_COLUMN = "era"
ENDER_TARGET = "target_ender_20"
BENCHMARK_ENDER20 = "v53_lgbm_ender20"
BENCHMARK_ENDER60 = "v53_lgbm_ender60"
PREDICTION_COLUMN = "prediction"
FOLD_COLUMN = "cv_fold"

COMPONENT_TARGETS = {
    "jasper": "target_jasper_20",
    "teager2b": "target_teager2b_20",
    "victor": "target_victor_20",
    "xerxes": "target_xerxes_20",
    "tyler": "target_tyler_20",
}
SCOUT_NEW_COMPONENTS = ("jasper", "teager2b", "victor", "tyler")
ALL_COMPONENTS = ("jasper", "teager2b", "victor", "xerxes", "tyler")
BLEND_WEIGHTS = {
    "tyler_w00": {"tyler": 0.0, "core": 0.25},
    "tyler_w10": {"tyler": 0.1, "core": 0.225},
    "tyler_w20_equal5": {"tyler": 0.2, "core": 0.2},
    "tyler_w30": {"tyler": 0.3, "core": 0.175},
    "tyler_w40": {"tyler": 0.4, "core": 0.15},
}
CANDIDATE_NAMES = tuple(BLEND_WEIGHTS)

SCOUT_ROWS = 1_279_658
SCOUT_ERAS = 214
SCOUT_FIRST_ERA = "0373"
SCOUT_LAST_ERA = "1225"
SCOUT_CALIBRATION_ERAS = 164
SCOUT_LOCKED_ERAS = 50
SCOUT_LAST_CALIBRATION_ERA = "1025"
SCOUT_FIRST_LOCKED_ERA = "1029"

CONFIRMATION_ROWS = 5_112_039
CONFIRMATION_ERAS = 855
CONFIRMATION_FIRST_ERA = "0371"
CONFIRMATION_LAST_ERA = "1225"
CONFIRMATION_CALIBRATION_ERAS = 655
CONFIRMATION_LOCKED_ERAS = 200
CONFIRMATION_LAST_CALIBRATION_ERA = "1025"
CONFIRMATION_FIRST_LOCKED_ERA = "1026"

CALIBRATION_THRESHOLDS = {
    "bmc_mean_min_inclusive": 0.0020,
    "bmc_sharpe_min_exclusive": 0.25,
    "bmc_max_drawdown_max_exclusive": 0.10,
    "corr_mean_min_inclusive": 0.012,
    "similarity_max_exclusive": 0.75,
}
LOCKED_THRESHOLDS = {
    "bmc_mean_min_exclusive": 0.0,
    "bmc_sharpe_min_exclusive": 0.20,
    "bmc_max_drawdown_max_exclusive": 0.10,
    "corr_mean_min_exclusive": 0.008,
}
CONFIRMATION_THRESHOLDS = {
    "bmc_mean_min_inclusive": 0.0020,
    "bmc_sharpe_min_exclusive": 0.35,
    "bmc_max_drawdown_max_exclusive": 0.15,
    "corr_mean_min_inclusive": 0.012,
    "similarity_max_exclusive": 0.75,
    "locked_bmc_mean_min_exclusive": 0.0,
    "locked_bmc_sharpe_min_exclusive": 0.20,
    "locked_bmc_max_drawdown_max_exclusive": 0.15,
    "locked_corr_mean_min_exclusive": 0.008,
}

PROTOCOL_CHECKPOINT_PATHS = (
    "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53/configs/base_d8.py",
    "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53/configs/r1_jasper_d8_t6000.py",
    "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53/configs/r1_teager2b_d8_t6000.py",
    "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53/configs/r1_victor_d8_t6000.py",
    "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53/configs/r1_tyler_d8_t6000.py",
    "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53/gate.md",
    "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53/source_manifest.json",
)
TRAINING_CHECKPOINT_PATHS = (
    "numerai/agents/code/analysis/evaluate_ender20_aux_target_rank_ensemble.py",
    "numerai/agents/code/analysis/evaluate_ender20_hybrid_stability.py",
    "numerai/agents/code/analysis/evaluate_xerxes20_lgbm_challenger.py",
    "numerai/agents/code/modeling",
    "numerai/agents/code/metrics/numerai_metrics.py",
    "numerai/agents/code/data/build_full_datasets.py",
    "numerai/agents/experiments/xerxes20_lgbm_challenger_v53/gpu_runtime.json",
)


class EnderEnsembleEvaluationError(ValueError):
    """Raised when a frozen source, artifact, or stage contract is violated."""


@dataclass(frozen=True)
class FrozenProtocol:
    repo_root: Path
    experiment_dir: Path
    source_manifest_path: Path
    source_manifest: dict[str, Any]
    scout_configs: dict[str, dict[str, Any]]
    scout_config_paths: dict[str, Path]
    medium_features: tuple[str, ...]
    pretraining_commit: str
    gpu_runtime_path: Path
    gpu_runtime_receipt: dict[str, Any]


@dataclass(frozen=True)
class ExpectedCohort:
    frame: pd.DataFrame
    full_rows: int
    full_eras: int
    eras: tuple[str, ...]
    folds: tuple[dict[str, int], ...]


@dataclass(frozen=True)
class ComponentPaths:
    name: str
    config: Path
    result: Path
    predictions: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EnderEnsembleEvaluationError(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _receipt_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EnderEnsembleEvaluationError(f"{label} is not valid UTF-8 JSON.") from error
    _require(isinstance(value, dict), f"{label} must be a JSON object.")
    _reject_nonfinite(value, label)
    return value


def _reject_nonfinite(value: Any, label: str) -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"{label} contains a non-finite number.")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{label}[{index}]")


def _exact_equal(actual: Any, expected: Any, label: str) -> None:
    _require(actual == expected, f"{label} differs: expected {expected!r}, got {actual!r}.")


def _safe_repo_path(repo_root: Path, relative: str | Path) -> Path:
    value = Path(relative)
    _require(not value.is_absolute(), f"Frozen path must be repo-relative: {relative}")
    candidate = (repo_root / value).resolve()
    root = repo_root.resolve()
    _require(
        candidate == root or root in candidate.parents,
        f"Frozen path escapes the repository: {relative}",
    )
    return candidate


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise EnderEnsembleEvaluationError(
            f"Artifact escapes repository root: {path}"
        ) from error


def _run_git(repo_root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def verify_checkpoint_boundaries(repo_root: Path, pretraining_commit: str) -> None:
    """Bind immutable protocol paths and later implementation paths to Git."""

    _require(
        isinstance(pretraining_commit, str)
        and len(pretraining_commit) == 40
        and all(character in "0123456789abcdef" for character in pretraining_commit),
        "Pretraining commit must be a full lowercase 40-character Git object ID.",
    )
    for label, commit in (
        ("protocol", PRE_SCORING_COMMIT),
        ("pretraining", pretraining_commit),
    ):
        exists = _run_git(repo_root, ["cat-file", "-e", f"{commit}^{{commit}}"])
        _require(exists.returncode == 0, f"Frozen {label} Git checkpoint is unavailable.")
        ancestor = _run_git(repo_root, ["merge-base", "--is-ancestor", commit, "HEAD"])
        _require(
            ancestor.returncode == 0,
            f"Frozen {label} checkpoint is not an ancestor of HEAD.",
        )
    ordered = _run_git(
        repo_root,
        ["merge-base", "--is-ancestor", PRE_SCORING_COMMIT, pretraining_commit],
    )
    _require(
        ordered.returncode == 0,
        "Pretraining checkpoint predates the frozen protocol checkpoint.",
    )
    for commit, paths, label in (
        (PRE_SCORING_COMMIT, PROTOCOL_CHECKPOINT_PATHS, "protocol"),
        (pretraining_commit, TRAINING_CHECKPOINT_PATHS, "training"),
    ):
        unchanged = _run_git(repo_root, ["diff", "--quiet", commit, "--", *paths])
        _require(
            unchanged.returncode == 0,
            f"Frozen {label} paths changed after their checkpoint.",
        )
        clean = _run_git(
            repo_root,
            ["status", "--porcelain", "--untracked-files=all", "--", *paths],
        )
        _require(
            clean.returncode == 0 and not clean.stdout.strip(),
            f"Frozen {label} paths have uncommitted or untracked changes.",
        )


def _write_content_addressed_receipt(
    output_dir: Path,
    prefix: str,
    receipt: Mapping[str, Any],
) -> Path:
    """Create one hash-named receipt and always refuse an existing destination."""

    payload = _receipt_bytes(receipt)
    digest = hashlib.sha256(payload).hexdigest()
    _require(
        bool(prefix)
        and all(character.isalnum() or character in "-_" for character in prefix),
        "Receipt prefix is unsafe.",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _require(
        not any(output_dir.glob(f"{prefix}-*.json")),
        f"A receipt already exists for immutable prefix: {prefix}",
    )
    claim = output_dir / f".{prefix}.claimed"
    try:
        with claim.open("xb") as stream:
            stream.write(f"{digest}\n".encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise EnderEnsembleEvaluationError(
            f"A receipt already exists for immutable prefix: {prefix}"
        ) from error
    path = output_dir / f"{prefix}-{digest}.json"
    _require(not path.exists(), f"Receipt output already exists: {path}")
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise EnderEnsembleEvaluationError(
            f"Receipt output already exists: {path}"
        ) from error
    _exact_equal(_sha256_file(path), digest, f"{prefix} receipt content address")
    return path


def _load_bound_receipt(
    path: Path,
    expected_sha256: str,
    *,
    expected_stage: str,
) -> dict[str, Any]:
    _require(
        len(expected_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_sha256),
        "Receipt hash must be a full lowercase SHA-256 digest.",
    )
    actual = _sha256_file(path)
    _exact_equal(actual, expected_sha256, f"{expected_stage} receipt hash")
    _require(path.name.endswith(f"-{actual}.json"), "Receipt filename is not content-addressed.")
    receipt = _load_json(path, f"{expected_stage} receipt")
    _exact_equal(receipt.get("experiment"), EXPERIMENT_NAME, "receipt experiment")
    _exact_equal(receipt.get("stage"), expected_stage, "receipt stage")
    return receipt


def _parse_seal_bindings(values: Sequence[Sequence[str]] | None) -> dict[str, tuple[Path, str]]:
    bindings: dict[str, tuple[Path, str]] = {}
    for value in values or ():
        _require(len(value) == 3, "Each --seal-receipt requires COMPONENT PATH SHA256.")
        component, path, digest = value
        _require(component in SCOUT_NEW_COMPONENTS, f"Unknown seal component: {component}")
        _require(component not in bindings, f"Duplicate seal component: {component}")
        bindings[component] = (Path(path), digest)
    return bindings


def _validate_path_receipt(
    repo_root: Path,
    receipt: Mapping[str, Any],
    label: str,
) -> Path:
    _require(
        isinstance(receipt.get("path"), str)
        and isinstance(receipt.get("sha256"), str)
        and isinstance(receipt.get("size_bytes"), int),
        f"{label} file receipt is malformed.",
    )
    path = _safe_repo_path(repo_root, str(receipt["path"]))
    _require(path.is_file(), f"{label} is missing: {path}")
    _exact_equal(path.stat().st_size, receipt["size_bytes"], f"{label} size")
    _exact_equal(_sha256_file(path), receipt["sha256"], f"{label} hash")
    return path


def _walk_explicit_path_receipts(
    repo_root: Path,
    value: Any,
    label: str = "source_manifest",
) -> None:
    if isinstance(value, Mapping):
        if {"path", "sha256", "size_bytes"}.issubset(value):
            _validate_path_receipt(repo_root, value, label)
        for key, child in value.items():
            _walk_explicit_path_receipts(repo_root, child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_explicit_path_receipts(repo_root, child, f"{label}[{index}]")


def _load_config(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        namespace = runpy.run_path(str(path))
    except Exception as error:
        raise EnderEnsembleEvaluationError(f"{label} could not be evaluated.") from error
    config = namespace.get("CONFIG", namespace.get("config"))
    _require(isinstance(config, dict), f"{label} has no CONFIG mapping.")
    _reject_nonfinite(config, label)
    return config


def validate_component_config(
    name: str,
    config: Mapping[str, Any],
    *,
    confirmation: bool = False,
) -> None:
    """Validate the only target-qualified scout or confirmation config shape."""

    _require(name in ALL_COMPONENTS, f"Unknown component config: {name}")
    target = COMPONENT_TARGETS[name]
    data = config.get("data")
    model = config.get("model")
    training = config.get("training")
    preprocessing = config.get("preprocessing")
    output = config.get("output")
    _exact_equal(
        set(config),
        {"data", "model", "training", "preprocessing", "output"},
        f"{name} top-level config keys",
    )
    _require(
        all(isinstance(value, Mapping) for value in (
            data,
            model,
            training,
            preprocessing,
            output,
        )),
        f"{name} config sections are malformed.",
    )
    assert isinstance(data, Mapping)
    assert isinstance(model, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(preprocessing, Mapping)
    assert isinstance(output, Mapping)

    _exact_equal(set(model), {"type", "x_groups", "params"}, f"{name} model keys")
    _exact_equal(
        set(preprocessing),
        {"nan_missing_all_twos", "missing_value"},
        f"{name} preprocessing keys",
    )
    _exact_equal(
        set(output),
        {"output_dir", "results_name"},
        f"{name} output keys",
    )

    _exact_equal(model.get("type"), "LGBMRegressor", f"{name} model type")
    _exact_equal(
        model.get("x_groups"),
        ["features", "era", "benchmark_models"],
        f"{name} model inputs",
    )
    _require(not model.get("target_transform"), f"{name} must train directly on its target.")
    _require(
        not model.get("prediction_transform"),
        f"{name} may not transform component predictions.",
    )
    _exact_equal(
        model.get("params"),
        {
            "n_estimators": 6_000,
            "learning_rate": 0.003,
            "max_depth": 8,
            "num_leaves": 255,
            "colsample_bytree": 0.1,
            "min_data_in_leaf": 10_000,
            "device_type": "gpu",
            "n_jobs": 12,
            "random_state": 1337,
            "verbosity": -1,
        },
        f"{name} LightGBM parameters",
    )
    _exact_equal(
        preprocessing,
        {"nan_missing_all_twos": False, "missing_value": 2.0},
        f"{name} preprocessing",
    )
    for key, expected in {
        "max_train_samples": 500_000,
        "sample_seed": 1337,
    }.items():
        _exact_equal(training.get(key), expected, f"{name} training.{key}")
    _exact_equal(
        training.get("cv"),
        {
            "enabled": True,
            "n_splits": 5,
            "embargo": 52 if confirmation else 13,
            "mode": "expanding",
            "min_train_size": 0,
        },
        f"{name} CV config",
    )
    expected_result_name = (
        f"confirmation_{name}_d8_t6000"
        if confirmation
        else ("r1_depth8" if name == "xerxes" else f"r1_{name}_d8_t6000")
    )
    _exact_equal(output.get("results_name"), expected_result_name, f"{name} result name")
    if name != "xerxes" or confirmation:
        _exact_equal(
            output.get("output_dir"),
            "experiments/ender20_aux_target_rank_ensemble_v53",
            f"{name} output directory",
        )

    common_data = {
        "data_version": "v5.3",
        "feature_set": "medium",
        "target_col": target,
        "era_col": ERA_COLUMN,
        "id_col": ID_COLUMN,
        "benchmark_model": BENCHMARK_ENDER20,
        "require_benchmark_coverage": True,
        "embargo_eras": 52 if confirmation else 13,
    }
    for key, expected in common_data.items():
        _exact_equal(data.get(key), expected, f"{name} data.{key}")
    if not confirmation:
        _exact_equal(data.get("full_data_path"), "v5.3/downsampled_full.parquet", f"{name} data source")
        _exact_equal(
            data.get("benchmark_data_path"),
            "v5.3/downsampled_full_benchmark_models.parquet",
            f"{name} benchmark source",
        )
        _exact_equal(set(data), {*common_data, "full_data_path", "benchmark_data_path"}, f"{name} data keys")
        _require("data_mode" not in training, f"{name} scout may not use a disk mode.")
        _exact_equal(
            set(training),
            {"max_train_samples", "sample_seed", "cv"},
            f"{name} scout training keys",
        )
    else:
        _require(
            not data.get("full_data_path") and not data.get("benchmark_data_path"),
            f"{name} confirmation may not use eager full sources.",
        )
        store_value = data.get("disk_feature_store_path", data.get("feature_store_path"))
        _require(
            isinstance(store_value, str) and bool(store_value),
            f"{name} confirmation has no disk feature store.",
        )
        if "disk_feature_store_path" in data and "feature_store_path" in data:
            _exact_equal(
                data["disk_feature_store_path"],
                data["feature_store_path"],
                f"{name} disk-store aliases",
            )
        allowed_data = {
            *common_data,
            "disk_feature_store_path",
            "feature_store_path",
            "label_sidecar_path",
        }
        _require(set(data).issubset(allowed_data), f"{name} confirmation data keys differ.")
        _exact_equal(training.get("data_mode"), "disk_feature_store", f"{name} data mode")
        _require(
            set(training) == {"max_train_samples", "sample_seed", "cv", "data_mode"},
            f"{name} confirmation training keys differ.",
        )


def default_scout_component_paths(
    protocol: FrozenProtocol,
    name: str,
) -> ComponentPaths:
    _require(name in SCOUT_NEW_COMPONENTS, f"Unknown new scout component: {name}")
    stem = f"r1_{name}_d8_t6000"
    return ComponentPaths(
        name=name,
        config=protocol.experiment_dir / "configs" / f"{stem}.py",
        result=protocol.experiment_dir / "results" / f"{stem}.json",
        predictions=protocol.experiment_dir / "predictions" / f"{stem}.parquet",
    )


def default_confirmation_component_paths(
    protocol: FrozenProtocol,
    name: str,
) -> ComponentPaths:
    _require(name in ALL_COMPONENTS, f"Unknown confirmation component: {name}")
    contract = protocol.source_manifest["confirmation_output_contract"][name]
    return ComponentPaths(
        name=name,
        config=_safe_repo_path(protocol.repo_root, contract["config_path"]),
        result=_safe_repo_path(protocol.repo_root, contract["results_path"]),
        predictions=_safe_repo_path(protocol.repo_root, contract["predictions_path"]),
    )


def verify_frozen_protocol(
    source_manifest_path: Path | None = None,
    *,
    pretraining_commit: str,
) -> FrozenProtocol:
    """Verify the protocol commit, manifest, configs, sources, and GPU receipt."""

    repo_root = _repo_root()
    experiment_dir = (
        repo_root / "numerai/agents/experiments" / EXPERIMENT_NAME
    ).resolve()
    manifest_path = (
        source_manifest_path.resolve()
        if source_manifest_path is not None
        else experiment_dir / "source_manifest.json"
    )
    _exact_equal(_sha256_file(manifest_path), SOURCE_MANIFEST_SHA256, "source manifest hash")
    manifest = _load_json(manifest_path, "source manifest")
    verify_checkpoint_boundaries(repo_root, pretraining_commit)

    _exact_equal(manifest.get("schema_version"), 1, "source manifest schema")
    _exact_equal(manifest.get("as_of_date"), AS_OF_DATE, "source manifest date")
    _exact_equal(manifest.get("data_version"), "v5.3", "source manifest data version")
    _exact_equal(manifest.get("component_targets"), COMPONENT_TARGETS, "component targets")
    expected_blends = {
        name: {
            "tyler_weight": weights["tyler"],
            "core_weight_each": weights["core"],
        }
        for name, weights in BLEND_WEIGHTS.items()
    }
    _exact_equal(manifest.get("blend_candidates"), expected_blends, "blend candidates")

    experiment_files = manifest.get("experiment_files")
    _require(isinstance(experiment_files, Mapping), "Manifest experiment_files is malformed.")
    for relative, receipt in experiment_files.items():
        _require(isinstance(receipt, Mapping), f"Malformed experiment receipt: {relative}")
        path = (experiment_dir / str(relative)).resolve()
        _require(experiment_dir in path.parents, f"Experiment path escapes directory: {relative}")
        _exact_equal(path.stat().st_size, receipt.get("size_bytes"), f"{relative} size")
        _exact_equal(_sha256_file(path), receipt.get("sha256"), f"{relative} hash")
    _exact_equal(
        experiment_files.get("gate.md", {}).get("sha256"),
        GATE_SHA256,
        "gate hash receipt",
    )

    scout_sources = manifest.get("scout_sources")
    _require(isinstance(scout_sources, Mapping), "Manifest scout_sources is malformed.")
    for relative, receipt in scout_sources.items():
        _require(isinstance(receipt, Mapping), f"Malformed scout source: {relative}")
        path = _safe_repo_path(repo_root, str(relative))
        _exact_equal(path.stat().st_size, receipt.get("size_bytes"), f"{relative} size")
        _exact_equal(_sha256_file(path), receipt.get("sha256"), f"{relative} hash")

    # All objects carrying path/size/full-hash fields are independently checked.
    _walk_explicit_path_receipts(repo_root, manifest)
    confirmation_sources = manifest.get("confirmation_sources")
    _require(
        isinstance(confirmation_sources, list) and len(confirmation_sources) == 4,
        "Manifest confirmation_sources is malformed.",
    )
    for receipt in confirmation_sources:
        _require(isinstance(receipt, Mapping), "Confirmation source receipt is malformed.")
        path = _safe_repo_path(repo_root, str(receipt["path"]))
        actual = parquet_source_fingerprint(path)
        for key in (
            "size_bytes",
            "mtime_ns",
            "num_rows",
            "num_row_groups",
            "schema_sha256",
            "footer_sha256",
        ):
            _exact_equal(actual[key], receipt[key], f"{receipt['path']} {key}")

    feature_receipt = manifest.get("feature_metadata")
    _require(isinstance(feature_receipt, Mapping), "Feature metadata receipt is malformed.")
    feature_path = _safe_repo_path(repo_root, str(feature_receipt["path"]))
    features = _load_json(feature_path, "features.json")
    medium = features.get("feature_sets", {}).get("medium")
    _require(
        isinstance(medium, list) and all(isinstance(value, str) for value in medium),
        "Medium feature set is malformed.",
    )
    _exact_equal(len(medium), feature_receipt["medium_feature_count"], "medium feature count")
    _exact_equal(
        feature_order_sha256(medium),
        feature_receipt["medium_feature_order_sha256"],
        "medium feature order hash",
    )

    config_paths: dict[str, Path] = {}
    configs: dict[str, dict[str, Any]] = {}
    for name in SCOUT_NEW_COMPONENTS:
        stem = f"r1_{name}_d8_t6000"
        relative = f"configs/{stem}.py"
        path = experiment_dir / relative
        config = _load_config(path, f"{name} scout config")
        validate_component_config(name, config)
        config_paths[name] = path
        configs[name] = config

    gpu_receipt = manifest.get("gpu_runtime")
    _require(isinstance(gpu_receipt, Mapping), "GPU runtime receipt is malformed.")
    gpu_path = _safe_repo_path(repo_root, str(gpu_receipt["path"]))
    try:
        gpu_runtime = xerxes.verify_live_gpu_runtime(gpu_path)
    except Exception as error:
        raise EnderEnsembleEvaluationError("GPU runtime validation failed.") from error
    return FrozenProtocol(
        repo_root=repo_root,
        experiment_dir=experiment_dir,
        source_manifest_path=manifest_path,
        source_manifest=manifest,
        scout_configs=configs,
        scout_config_paths=config_paths,
        medium_features=tuple(medium),
        pretraining_commit=pretraining_commit,
        gpu_runtime_path=gpu_path,
        gpu_runtime_receipt=gpu_runtime,
    )


def _derive_expected_oof(
    full: pd.DataFrame,
    *,
    embargo: int,
    expected_rows: int,
    expected_eras: int,
    first_era: str,
    last_era: str,
    calibration_eras: int,
    last_calibration_era: str,
    first_locked_era: str,
) -> ExpectedCohort:
    full = full.copy()
    full[ERA_COLUMN] = full[ERA_COLUMN].astype(str)
    _require(full[ID_COLUMN].notna().all(), "Expected cohort contains null IDs.")
    _require(full[ID_COLUMN].is_unique, "Expected cohort IDs are not unique.")
    all_eras = sorted(full[ERA_COLUMN].unique().tolist(), key=int)
    splits = era_cv_splits(
        all_eras,
        n_splits=5,
        embargo=embargo,
        mode="expanding",
        min_train_size=0,
    )
    counts = full.groupby(ERA_COLUMN, sort=False, observed=True).size()
    fold_by_era: dict[str, int] = {}
    fold_receipts: list[dict[str, int]] = []
    for fold, (train_eras, validation_eras) in enumerate(splits):
        if not train_eras or not validation_eras:
            continue
        train_labels = [str(era) for era in train_eras]
        validation_labels = [str(era) for era in validation_eras]
        overlap = set(fold_by_era).intersection(validation_labels)
        _require(not overlap, "CV validation eras overlap between folds.")
        fold_by_era.update({era: fold for era in validation_labels})
        train_rows = int(counts.reindex(train_labels).sum())
        validation_rows = int(counts.reindex(validation_labels).sum())
        fold_receipts.append(
            {
                "fold": fold,
                "train_eras": len(train_labels),
                "val_eras": len(validation_labels),
                "train_rows": min(train_rows, 500_000),
                "val_rows": validation_rows,
            }
        )
    oof = full[full[ERA_COLUMN].isin(fold_by_era)].copy()
    oof[FOLD_COLUMN] = oof[ERA_COLUMN].map(fold_by_era).astype(np.int16)
    eras = tuple(sorted(oof[ERA_COLUMN].unique().tolist(), key=int))
    _exact_equal(len(oof), expected_rows, "OOF row count")
    _exact_equal(len(eras), expected_eras, "OOF era count")
    _exact_equal(eras[0], first_era, "first OOF era")
    _exact_equal(eras[-1], last_era, "last OOF era")
    _exact_equal(eras[calibration_eras - 1], last_calibration_era, "last calibration era")
    _exact_equal(eras[calibration_eras], first_locked_era, "first locked era")
    return ExpectedCohort(
        frame=oof.reset_index(drop=True),
        full_rows=len(full),
        full_eras=len(all_eras),
        eras=eras,
        folds=tuple(fold_receipts),
    )


def _validate_finite_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    _require(not frame[list(columns)].isna().any().any(), f"{label} contains null values.")
    values = frame[list(columns)].to_numpy(dtype=np.float64)
    _require(np.isfinite(values).all(), f"{label} contains non-finite values.")


def _merge_sources_one_to_one(
    data: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    _require(data[ID_COLUMN].notna().all() and data[ID_COLUMN].is_unique, f"{label} data IDs are invalid.")
    _require(
        benchmark[ID_COLUMN].notna().all() and benchmark[ID_COLUMN].is_unique,
        f"{label} benchmark IDs are invalid.",
    )
    renamed = benchmark.rename(columns={ERA_COLUMN: "_benchmark_era"})
    try:
        merged = data.merge(
            renamed,
            how="inner",
            on=ID_COLUMN,
            validate="one_to_one",
            sort=False,
        )
    except pd.errors.MergeError as error:
        raise EnderEnsembleEvaluationError(f"{label} sources are not one-to-one.") from error
    _require(
        np.array_equal(
            merged[ERA_COLUMN].astype(str).to_numpy(),
            merged["_benchmark_era"].astype(str).to_numpy(),
        ),
        f"{label} source eras differ by ID.",
    )
    return merged.drop(columns=["_benchmark_era"])


def build_scout_expected_cohort(protocol: FrozenProtocol) -> ExpectedCohort:
    data_path = protocol.repo_root / "numerai/v5.3/downsampled_full.parquet"
    benchmark_path = (
        protocol.repo_root / "numerai/v5.3/downsampled_full_benchmark_models.parquet"
    )
    data_columns = [ID_COLUMN, ERA_COLUMN, *COMPONENT_TARGETS.values(), ENDER_TARGET]
    benchmark_columns = [ID_COLUMN, ERA_COLUMN, BENCHMARK_ENDER20, BENCHMARK_ENDER60]
    data = pd.read_parquet(data_path, columns=list(dict.fromkeys(data_columns)))
    benchmark = pd.read_parquet(benchmark_path, columns=benchmark_columns)
    full = _merge_sources_one_to_one(data, benchmark, label="Scout")
    _validate_finite_columns(
        full,
        [*COMPONENT_TARGETS.values(), ENDER_TARGET, BENCHMARK_ENDER20, BENCHMARK_ENDER60],
        "Scout targets and benchmarks",
    )
    coverage = protocol.source_manifest["target_coverage"]
    _exact_equal(len(full), coverage["benchmark_covered_rows"], "scout covered rows")
    _exact_equal(full[ERA_COLUMN].astype(str).nunique(), coverage["benchmark_covered_eras"], "scout covered eras")
    return _derive_expected_oof(
        full,
        embargo=13,
        expected_rows=SCOUT_ROWS,
        expected_eras=SCOUT_ERAS,
        first_era=SCOUT_FIRST_ERA,
        last_era=SCOUT_LAST_ERA,
        calibration_eras=SCOUT_CALIBRATION_ERAS,
        last_calibration_era=SCOUT_LAST_CALIBRATION_ERA,
        first_locked_era=SCOUT_FIRST_LOCKED_ERA,
    )


def _read_full_confirmation_sources(protocol: FrozenProtocol) -> pd.DataFrame:
    source_by_name = {
        Path(str(receipt["path"])).name: _safe_repo_path(
            protocol.repo_root, str(receipt["path"])
        )
        for receipt in protocol.source_manifest["confirmation_sources"]
    }
    data_columns = [ID_COLUMN, ERA_COLUMN, *COMPONENT_TARGETS.values(), ENDER_TARGET]
    benchmark_columns = [ID_COLUMN, ERA_COLUMN, BENCHMARK_ENDER20, BENCHMARK_ENDER60]
    data_frames: list[pd.DataFrame] = []
    for filename in ("train.parquet", "validation.parquet"):
        path = source_by_name[filename]
        schema_names = pq.read_schema(path).names
        columns = list(dict.fromkeys(data_columns))
        if "data_type" in schema_names:
            columns.append("data_type")
        frame = pd.read_parquet(path, columns=columns)
        if "data_type" in frame:
            frame = frame[frame["data_type"].astype(str).isin(("train", "validation"))]
            frame = frame.drop(columns=["data_type"])
        data_frames.append(frame)
    benchmark_frames = [
        pd.read_parquet(source_by_name[filename], columns=benchmark_columns)
        for filename in (
            "train_benchmark_models.parquet",
            "validation_benchmark_models.parquet",
        )
    ]
    data = pd.concat(data_frames, ignore_index=True)
    benchmark = pd.concat(benchmark_frames, ignore_index=True)
    full = _merge_sources_one_to_one(data, benchmark, label="Confirmation")
    numeric_columns = [
        *COMPONENT_TARGETS.values(),
        ENDER_TARGET,
        BENCHMARK_ENDER20,
        BENCHMARK_ENDER60,
    ]
    finite = np.isfinite(full[numeric_columns].to_numpy(dtype=np.float64)).all(axis=1)
    full = full.loc[finite].reset_index(drop=True)
    _validate_finite_columns(full, numeric_columns, "Confirmation targets and benchmarks")
    coverage = protocol.source_manifest["final_fit_target_coverage"]
    _exact_equal(len(full), coverage["finite_all_targets_and_benchmarks_rows"], "confirmation finite rows")
    return full


def _validate_confirmation_input_receipt(
    protocol: FrozenProtocol,
    path: Path | None,
    expected_sha256: str | None,
) -> dict[str, Any]:
    _require(
        path is not None and expected_sha256 is not None,
        "Confirmation input receipt path/hash are both required.",
    )
    assert path is not None and expected_sha256 is not None
    path = path.resolve()
    receipt = _load_bound_receipt(
        path,
        expected_sha256,
        expected_stage="confirmation-pretraining",
    )
    _exact_equal(receipt.get("passed"), True, "confirmation pretraining passage")
    _exact_equal(receipt.get("state"), "PASS", "confirmation pretraining state")
    _validate_protocol_binding(receipt, protocol)
    _exact_equal(
        receipt.get("checkpoint"),
        protocol.pretraining_commit,
        "confirmation pretraining checkpoint",
    )

    configs = receipt.get("configs")
    _require(isinstance(configs, Mapping), "Confirmation config receipts are malformed.")
    _exact_equal(set(configs), set(ALL_COMPONENTS), "confirmation config components")
    for name in ALL_COMPONENTS:
        config_receipt = configs[name]
        _require(isinstance(config_receipt, Mapping), f"{name} config receipt is malformed.")
        expected_relative = protocol.source_manifest["confirmation_output_contract"][name]["config_path"]
        _exact_equal(config_receipt.get("path"), expected_relative, f"{name} config path")
        config_path = _validate_path_receipt(
            protocol.repo_root, config_receipt, f"{name} confirmation config"
        )
        unchanged = _run_git(
            protocol.repo_root,
            ["diff", "--quiet", protocol.pretraining_commit, "--", expected_relative],
        )
        _require(
            unchanged.returncode == 0,
            f"{name} confirmation config differs from its checkpoint.",
        )
        config = _load_config(config_path, f"{name} confirmation config")
        validate_component_config(name, config, confirmation=True)

    loader = receipt.get("loader")
    _require(isinstance(loader, Mapping), "Confirmation loader receipt is malformed.")
    _exact_equal(
        loader.get("checkpoint"),
        protocol.pretraining_commit,
        "confirmation loader checkpoint",
    )
    loader_files = loader.get("files")
    _require(
        isinstance(loader_files, list) and bool(loader_files),
        "Confirmation loader files are missing.",
    )
    required_loader_paths = {
        "numerai/agents/code/modeling/utils/disk_feature_store.py",
        "numerai/agents/code/modeling/utils/pipeline.py",
    }
    actual_loader_paths: set[str] = set()
    for index, item in enumerate(loader_files):
        _require(isinstance(item, Mapping), f"Loader file receipt {index} is malformed.")
        loader_path = _validate_path_receipt(
            protocol.repo_root, item, f"confirmation loader file {index}"
        )
        relative = _relative_path(loader_path, protocol.repo_root)
        actual_loader_paths.add(relative)
        unchanged = _run_git(
            protocol.repo_root,
            ["diff", "--quiet", protocol.pretraining_commit, "--", relative],
        )
        _require(
            unchanged.returncode == 0,
            f"Confirmation loader file differs from its checkpoint: {relative}",
        )
    _require(
        required_loader_paths.issubset(actual_loader_paths),
        "Confirmation receipt omits a required loader implementation file.",
    )

    _exact_equal(
        receipt.get("canonical_store"),
        protocol.source_manifest["confirmation_xerxes_medium_store_anchor"],
        "canonical confirmation store binding",
    )
    layout = receipt.get("input_layout")
    _require(isinstance(layout, Mapping), "Confirmation input layout is malformed.")
    _exact_equal(
        layout.get("type"),
        "dedicated_target_stores",
        "confirmation input layout type",
    )
    stores = layout.get("stores")
    _require(isinstance(stores, Mapping), "Confirmation store receipts are malformed.")
    _exact_equal(set(stores), set(ALL_COMPONENTS), "confirmation target stores")
    canonical = protocol.source_manifest["confirmation_xerxes_medium_store_anchor"]
    for name in ALL_COMPONENTS:
        store = stores[name]
        _require(isinstance(store, Mapping), f"{name} store receipt is malformed.")
        for key in ("metadata", "manifest", "features"):
            item = store.get(key)
            _require(isinstance(item, Mapping), f"{name} store {key} receipt is missing.")
            _validate_path_receipt(protocol.repo_root, item, f"{name} store {key}")
        metadata_path = _safe_repo_path(protocol.repo_root, store["metadata"]["path"])
        metadata = _load_json(metadata_path, f"{name} store metadata")
        for key, value in {
            "complete": True,
            "row_count": canonical["row_count"],
            "feature_count": canonical["feature_count"],
            "feature_order_sha256": canonical["feature_order_sha256"],
            "target_column": COMPONENT_TARGETS[name],
        }.items():
            _exact_equal(metadata.get(key), value, f"{name} store metadata.{key}")
        _exact_equal(
            metadata.get("features", {}).get("sha256"),
            canonical["features"]["sha256"],
            f"{name} canonical feature bytes",
        )
        _exact_equal(
            store["features"].get("sha256"),
            canonical["features"]["sha256"],
            f"{name} feature receipt hash",
        )
        for key in ("manifest", "features"):
            embedded = metadata.get(key)
            _require(isinstance(embedded, Mapping), f"{name} metadata.{key} is malformed.")
            _exact_equal(
                embedded.get("sha256"),
                store[key].get("sha256"),
                f"{name} metadata.{key} hash",
            )
            _exact_equal(
                embedded.get("size_bytes"),
                store[key].get("size_bytes"),
                f"{name} metadata.{key} size",
            )
        if name == "xerxes":
            _exact_equal(store, canonical, "canonical Xerxes store receipt")

        config = _load_config(
            _safe_repo_path(protocol.repo_root, configs[name]["path"]),
            f"{name} confirmation config",
        )
        configured = config["data"].get(
            "disk_feature_store_path", config["data"].get("feature_store_path")
        )
        _require(isinstance(configured, str), f"{name} store path is malformed.")
        configured_path = Path(configured)
        if configured_path.parts and configured_path.parts[0] == "v5.3":
            configured_path = protocol.repo_root / "numerai" / configured_path
        else:
            configured_path = _safe_repo_path(protocol.repo_root, configured_path)
        _exact_equal(
            configured_path.resolve(),
            metadata_path.parent.resolve(),
            f"{name} configured target store",
        )

    destinations = receipt.get("output_destinations")
    _require(
        isinstance(destinations, Mapping),
        "Confirmation output-destination receipt is malformed.",
    )
    _exact_equal(set(destinations), set(ALL_COMPONENTS), "confirmation destinations")
    for name in ALL_COMPONENTS:
        contract = protocol.source_manifest["confirmation_output_contract"][name]
        item = destinations[name]
        _require(isinstance(item, Mapping), f"{name} destination receipt is malformed.")
        expected_destination = {
            "results_path": contract["results_path"],
            "predictions_path": contract["predictions_path"],
            "results_absent_at_checkpoint": True,
            "predictions_absent_at_checkpoint": True,
        }
        _exact_equal(item, expected_destination, f"{name} confirmation destinations")
        for key in ("results_path", "predictions_path"):
            relative = contract[key]
            exists = _run_git(
                protocol.repo_root,
                ["cat-file", "-e", f"{protocol.pretraining_commit}:{relative}"],
            )
            _require(
                exists.returncode != 0,
                f"{name} {key} existed at the confirmation checkpoint.",
            )
    return receipt


def build_confirmation_expected_cohort(
    protocol: FrozenProtocol,
    *,
    confirmation_input_receipt: Path | None = None,
    confirmation_input_receipt_sha256: str | None = None,
) -> ExpectedCohort:
    """Derive the consecutive OOF cohort and bind it to the canonical store."""

    _validate_confirmation_input_receipt(
        protocol,
        confirmation_input_receipt,
        confirmation_input_receipt_sha256,
    )
    full = _read_full_confirmation_sources(protocol)
    anchor = protocol.source_manifest["confirmation_xerxes_medium_store_anchor"]
    manifest_path = _safe_repo_path(protocol.repo_root, anchor["manifest"]["path"])
    store = pd.read_parquet(
        manifest_path,
        columns=["row_offset", ID_COLUMN, ERA_COLUMN, COMPONENT_TARGETS["xerxes"], BENCHMARK_ENDER20],
    )
    _exact_equal(len(store), len(full), "canonical store row count")
    _require(
        np.array_equal(store["row_offset"].to_numpy(dtype=np.int64), np.arange(len(store), dtype=np.int64)),
        "Canonical store row offsets are not consecutive.",
    )
    for column in (ID_COLUMN, ERA_COLUMN):
        _require(
            np.array_equal(store[column].astype(str).to_numpy(), full[column].astype(str).to_numpy()),
            f"Canonical store {column} order differs from raw sources.",
        )
    for column in (COMPONENT_TARGETS["xerxes"], BENCHMARK_ENDER20):
        _require(
            np.array_equal(store[column].to_numpy(), full[column].to_numpy()),
            f"Canonical store {column} differs from raw sources.",
        )
    coverage = protocol.source_manifest["confirmation_target_coverage"]
    expected = _derive_expected_oof(
        full,
        embargo=52,
        expected_rows=CONFIRMATION_ROWS,
        expected_eras=CONFIRMATION_ERAS,
        first_era=CONFIRMATION_FIRST_ERA,
        last_era=CONFIRMATION_LAST_ERA,
        calibration_eras=CONFIRMATION_CALIBRATION_ERAS,
        last_calibration_era=CONFIRMATION_LAST_CALIBRATION_ERA,
        first_locked_era=CONFIRMATION_FIRST_LOCKED_ERA,
    )
    _exact_equal(len(expected.frame), coverage["rows"], "confirmation OOF rows")
    _exact_equal(len(expected.eras), coverage["eras"], "confirmation OOF eras")
    return expected


def rank_within_era(raw: Sequence[float], eras: Sequence[Any]) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    era_series = pd.Series(eras, copy=False).reset_index(drop=True).astype(str)
    _require(len(values) == len(era_series), "Prediction and era lengths differ.")
    _require(np.isfinite(values).all(), "Cannot rank non-finite predictions.")
    ranked = pd.Series(values).groupby(era_series, sort=False, observed=True).rank(
        method="average", pct=True
    )
    result = ranked.to_numpy(dtype=np.float64)
    _require(np.isfinite(result).all(), "Ranked predictions contain non-finite values.")
    return result


def build_rank_blends(frame: pd.DataFrame) -> pd.DataFrame:
    """Create only the five predeclared rank blends from raw component columns."""

    required = {ERA_COLUMN, *ALL_COMPONENTS}
    _require(required.issubset(frame), f"Missing blend columns: {sorted(required - set(frame))}")
    result = frame.copy()
    ranked: dict[str, np.ndarray] = {
        name: rank_within_era(result[name], result[ERA_COLUMN])
        for name in ALL_COMPONENTS
    }
    core_names = ("jasper", "teager2b", "victor", "xerxes")
    for candidate, weights in BLEND_WEIGHTS.items():
        raw_blend = weights["core"] * sum(ranked[name] for name in core_names)
        raw_blend = raw_blend + weights["tyler"] * ranked["tyler"]
        result[candidate] = rank_within_era(raw_blend, result[ERA_COLUMN])
    return result


def build_selected_rank_blend(frame: pd.DataFrame, candidate: str) -> pd.DataFrame:
    """Build one already-selected formula without materializing alternatives."""

    _require(candidate in CANDIDATE_NAMES, f"Unknown selected blend: {candidate}")
    required = {ERA_COLUMN, *ALL_COMPONENTS}
    _require(required.issubset(frame), f"Missing blend columns: {sorted(required - set(frame))}")
    result = frame.copy()
    ranked = {
        name: rank_within_era(result[name], result[ERA_COLUMN])
        for name in ALL_COMPONENTS
    }
    weights = BLEND_WEIGHTS[candidate]
    core_names = ("jasper", "teager2b", "victor", "xerxes")
    raw_blend = weights["core"] * sum(ranked[name] for name in core_names)
    raw_blend = raw_blend + weights["tyler"] * ranked["tyler"]
    result[candidate] = rank_within_era(raw_blend, result[ERA_COLUMN])
    return result


def symmetric_per_era_similarity(
    left: Sequence[float],
    right: Sequence[float],
    eras: Sequence[Any],
) -> pd.Series:
    """Equal-era symmetric Spearman similarity with average tie ranks."""

    frame = pd.DataFrame(
        {
            ERA_COLUMN: pd.Series(eras, copy=False).astype(str),
            "_left": np.asarray(left, dtype=np.float64),
            "_right": np.asarray(right, dtype=np.float64),
        }
    )
    _validate_finite_columns(frame, ["_left", "_right"], "Similarity signals")
    rows: dict[str, float] = {}
    for era, group in frame.groupby(ERA_COLUMN, sort=False, observed=True):
        ranked = group[["_left", "_right"]].rank(method="average", pct=True)
        values = ranked.to_numpy(dtype=np.float64)
        values = values - values.mean(axis=0, keepdims=True)
        denominator = math.sqrt(float(np.square(values[:, 0]).sum() * np.square(values[:, 1]).sum()))
        _require(denominator > 0.0 and math.isfinite(denominator), f"Constant similarity signal in era {era}.")
        score = float(values[:, 0] @ values[:, 1] / denominator)
        _require(math.isfinite(score), f"Non-finite similarity in era {era}.")
        rows[str(era)] = score
    return pd.Series(rows, dtype=np.float64)


def _score_summary(scores: pd.Series) -> dict[str, Any]:
    values = scores.to_numpy(dtype=np.float64)
    _require(values.size > 0 and np.isfinite(values).all(), "Cannot summarize empty/non-finite scores.")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    std_valid = bool(math.isfinite(std) and std > 0.0)
    cumulative = pd.Series(values).cumsum()
    running_max = cumulative.expanding(min_periods=1).max()
    max_drawdown = float((running_max - cumulative).max())
    return {
        "mean": mean,
        "std": std,
        "std_valid": std_valid,
        "sharpe": float(mean / std) if std_valid else None,
        "max_drawdown": max_drawdown,
    }


def compute_per_era_metrics(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    eras: Sequence[str],
    *,
    tabm_column: str,
) -> dict[str, pd.DataFrame]:
    signals = list(signal_columns)
    ordered_eras = tuple(sorted((str(era) for era in eras), key=int))
    corr = numerai_metrics.per_era_corr(frame, signals, ENDER_TARGET, ERA_COLUMN)
    bmc = numerai_metrics.per_era_bmc(
        frame, signals, BENCHMARK_ENDER20, ENDER_TARGET, ERA_COLUMN
    )
    result: dict[str, pd.DataFrame] = {}
    for label, scores in (("corr", corr), ("bmc", bmc)):
        scores = scores.copy()
        scores.index = scores.index.astype(str)
        _require(set(scores.index) == set(ordered_eras), f"{label} era coverage differs.")
        scores = scores.loc[list(ordered_eras), signals]
        _require(np.isfinite(scores.to_numpy(dtype=np.float64)).all(), f"{label} contains non-finite scores.")
        result[label] = scores
    references = {
        "ender20_similarity": BENCHMARK_ENDER20,
        "ender60_similarity": BENCHMARK_ENDER60,
        "tabm_similarity": tabm_column,
    }
    for label, reference in references.items():
        columns: dict[str, pd.Series] = {}
        for signal in signals:
            scores = symmetric_per_era_similarity(
                frame[signal], frame[reference], frame[ERA_COLUMN]
            )
            _require(set(scores.index) == set(ordered_eras), f"{label} era coverage differs.")
            columns[signal] = scores.loc[list(ordered_eras)]
        result[label] = pd.DataFrame(columns, index=list(ordered_eras))
    return result


def summarize_signal(
    per_era: Mapping[str, pd.DataFrame],
    signal: str,
) -> dict[str, Any]:
    summary = {
        "era_count": len(per_era["bmc"]),
        "corr": _score_summary(per_era["corr"][signal]),
        "bmc": _score_summary(per_era["bmc"][signal]),
        "avg_ender20_similarity": float(per_era["ender20_similarity"][signal].mean()),
        "avg_ender60_similarity": float(per_era["ender60_similarity"][signal].mean()),
        "avg_tabm_similarity": float(per_era["tabm_similarity"][signal].mean()),
    }
    _reject_nonfinite(summary, f"{signal} summary")
    return summary


def calibration_checks(summary: Mapping[str, Any]) -> dict[str, bool]:
    bmc = summary["bmc"]
    corr = summary["corr"]
    sharpe = bmc.get("sharpe")
    return {
        "bmc_mean": float(bmc["mean"]) >= CALIBRATION_THRESHOLDS["bmc_mean_min_inclusive"],
        "bmc_sharpe": bool(bmc.get("std_valid", True))
        and sharpe is not None
        and float(sharpe) > CALIBRATION_THRESHOLDS["bmc_sharpe_min_exclusive"],
        "bmc_max_drawdown": float(bmc["max_drawdown"])
        < CALIBRATION_THRESHOLDS["bmc_max_drawdown_max_exclusive"],
        "corr_mean": float(corr["mean"]) >= CALIBRATION_THRESHOLDS["corr_mean_min_inclusive"],
        "ender20_similarity": float(summary["avg_ender20_similarity"])
        < CALIBRATION_THRESHOLDS["similarity_max_exclusive"],
        "ender60_similarity": float(summary["avg_ender60_similarity"])
        < CALIBRATION_THRESHOLDS["similarity_max_exclusive"],
        "tabm_similarity": float(summary["avg_tabm_similarity"])
        < CALIBRATION_THRESHOLDS["similarity_max_exclusive"],
    }


def select_scout_candidate(
    summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    evaluations: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    for candidate in CANDIDATE_NAMES:
        checks = calibration_checks(summaries[candidate])
        passed = all(checks.values())
        evaluations[candidate] = {"eligible": passed, "checks": checks, "in_tie_set": False}
        if passed:
            eligible.append(candidate)
    if not eligible:
        return None, evaluations
    max_mean = max(float(summaries[name]["bmc"]["mean"]) for name in eligible)
    tie_set = [
        name
        for name in eligible
        if max_mean - float(summaries[name]["bmc"]["mean"]) <= 0.0001
    ]
    for name in tie_set:
        evaluations[name]["in_tie_set"] = True
    selected = sorted(
        tie_set,
        key=lambda name: (
            -float(summaries[name]["bmc"]["sharpe"]),
            float(summaries[name]["bmc"]["max_drawdown"]),
            float(BLEND_WEIGHTS[name]["tyler"]),
            name,
        ),
    )[0]
    return selected, evaluations


def locked_checks(summary: Mapping[str, Any]) -> dict[str, bool]:
    bmc = summary["bmc"]
    corr = summary["corr"]
    sharpe = bmc.get("sharpe")
    return {
        "bmc_mean": float(bmc["mean"]) > LOCKED_THRESHOLDS["bmc_mean_min_exclusive"],
        "bmc_sharpe": bool(bmc.get("std_valid", True))
        and sharpe is not None
        and float(sharpe) > LOCKED_THRESHOLDS["bmc_sharpe_min_exclusive"],
        "bmc_max_drawdown": float(bmc["max_drawdown"])
        < LOCKED_THRESHOLDS["bmc_max_drawdown_max_exclusive"],
        "corr_mean": float(corr["mean"]) > LOCKED_THRESHOLDS["corr_mean_min_exclusive"],
    }


def confirmation_checks(
    calibration: Mapping[str, Any],
    locked: Mapping[str, Any],
    full: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for segment, summary in (("calibration", calibration), ("full", full)):
        if summary is None:
            continue
        bmc = summary["bmc"]
        corr = summary["corr"]
        checks.update(
            {
                f"{segment}_bmc_mean": float(bmc["mean"])
                >= CONFIRMATION_THRESHOLDS["bmc_mean_min_inclusive"],
                f"{segment}_bmc_sharpe": bool(bmc.get("std_valid", True))
                and bmc.get("sharpe") is not None
                and float(bmc["sharpe"]) > CONFIRMATION_THRESHOLDS["bmc_sharpe_min_exclusive"],
                f"{segment}_bmc_max_drawdown": float(bmc["max_drawdown"])
                < CONFIRMATION_THRESHOLDS["bmc_max_drawdown_max_exclusive"],
                f"{segment}_corr_mean": float(corr["mean"])
                >= CONFIRMATION_THRESHOLDS["corr_mean_min_inclusive"],
                f"{segment}_ender20_similarity": float(summary["avg_ender20_similarity"])
                < CONFIRMATION_THRESHOLDS["similarity_max_exclusive"],
                f"{segment}_ender60_similarity": float(summary["avg_ender60_similarity"])
                < CONFIRMATION_THRESHOLDS["similarity_max_exclusive"],
                f"{segment}_tabm_similarity": float(summary["avg_tabm_similarity"])
                < CONFIRMATION_THRESHOLDS["similarity_max_exclusive"],
            }
        )
    locked_bmc = locked["bmc"]
    locked_corr = locked["corr"]
    checks.update(
        {
            "locked_bmc_mean": float(locked_bmc["mean"])
            > CONFIRMATION_THRESHOLDS["locked_bmc_mean_min_exclusive"],
            "locked_bmc_sharpe": bool(locked_bmc.get("std_valid", True))
            and locked_bmc.get("sharpe") is not None
            and float(locked_bmc["sharpe"])
            > CONFIRMATION_THRESHOLDS["locked_bmc_sharpe_min_exclusive"],
            "locked_bmc_max_drawdown": float(locked_bmc["max_drawdown"])
            < CONFIRMATION_THRESHOLDS["locked_bmc_max_drawdown_max_exclusive"],
            "locked_corr_mean": float(locked_corr["mean"])
            > CONFIRMATION_THRESHOLDS["locked_corr_mean_min_exclusive"],
        }
    )
    return checks


def confirmation_calibration_checks(summary: Mapping[str, Any]) -> dict[str, bool]:
    """Evaluate only the authorized 655-era confirmation calibration slice."""

    bmc = summary["bmc"]
    corr = summary["corr"]
    return {
        "calibration_bmc_mean": float(bmc["mean"])
        >= CONFIRMATION_THRESHOLDS["bmc_mean_min_inclusive"],
        "calibration_bmc_sharpe": bool(bmc.get("std_valid", True))
        and bmc.get("sharpe") is not None
        and float(bmc["sharpe"])
        > CONFIRMATION_THRESHOLDS["bmc_sharpe_min_exclusive"],
        "calibration_bmc_max_drawdown": float(bmc["max_drawdown"])
        < CONFIRMATION_THRESHOLDS["bmc_max_drawdown_max_exclusive"],
        "calibration_corr_mean": float(corr["mean"])
        >= CONFIRMATION_THRESHOLDS["corr_mean_min_inclusive"],
        "calibration_ender20_similarity": float(summary["avg_ender20_similarity"])
        < CONFIRMATION_THRESHOLDS["similarity_max_exclusive"],
        "calibration_ender60_similarity": float(summary["avg_ender60_similarity"])
        < CONFIRMATION_THRESHOLDS["similarity_max_exclusive"],
        "calibration_tabm_similarity": float(summary["avg_tabm_similarity"])
        < CONFIRMATION_THRESHOLDS["similarity_max_exclusive"],
    }


def _path_leaf(value: Any) -> str:
    _require(isinstance(value, str) and bool(value), "Artifact path is malformed.")
    return Path(value.replace("\\", "/")).name


def _expected_model_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config["model"]
    payload: dict[str, Any] = {"type": model["type"], "params": model["params"]}
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
        if key in model:
            payload[key] = model[key]
    return payload


def _expected_semantics(config: Mapping[str, Any]) -> dict[str, Any]:
    return build_prediction_semantics(
        dict(config["model"]),
        str(config["data"]["target_col"]),
        str(config["data"]["era_col"]),
    )


def validate_result_json(
    component: ComponentPaths,
    result: Mapping[str, Any],
    config: Mapping[str, Any],
    expected: ExpectedCohort,
    *,
    confirmation: bool = False,
    store_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a pipeline result to its target, config, folds, GPU, and semantics."""

    result_name = str(config["output"]["results_name"])
    _exact_equal(component.result.name, f"{result_name}.json", f"{component.name} result filename")
    _exact_equal(component.predictions.name, f"{result_name}.parquet", f"{component.name} prediction filename")
    _exact_equal(result.get("model"), _expected_model_payload(config), f"{component.name} model receipt")
    preprocessing = config.get("preprocessing", {})
    _exact_equal(
        result.get("preprocessing"),
        {
            "nan_missing_all_twos": preprocessing.get("nan_missing_all_twos", False),
            "missing_value": preprocessing.get("missing_value", 2.0),
        },
        f"{component.name} preprocessing receipt",
    )
    target = COMPONENT_TARGETS[component.name]
    data = result.get("data")
    _require(isinstance(data, Mapping), f"{component.name} result lacks data receipt.")
    expected_data = {
        "data_version": "v5.3",
        "feature_set": "medium",
        "target": target,
        "full_rows": expected.full_rows,
        "full_eras": expected.full_eras,
        "oof_rows": len(expected.frame),
        "oof_eras": len(expected.eras),
        "embargo_eras": 52 if confirmation else 13,
        "require_benchmark_coverage": True,
        "data_mode": "disk_feature_store" if confirmation else "eager",
    }
    for key, value in expected_data.items():
        _exact_equal(data.get(key), value, f"{component.name} data.{key}")
    if not confirmation:
        _exact_equal(
            data.get("full_data_path"),
            "v5.3/downsampled_full.parquet",
            f"{component.name} full data path",
        )
    else:
        diagnostics = data.get("disk_feature_store")
        _require(isinstance(diagnostics, Mapping), f"{component.name} lacks disk-store diagnostics.")
        if store_metadata is not None:
            for key, expected_value in {
                "generation_id": store_metadata.get("generation_id"),
                "row_count": store_metadata.get("row_count"),
                "feature_count": store_metadata.get("feature_count"),
                "feature_order_sha256": store_metadata.get("feature_order_sha256"),
            }.items():
                _exact_equal(diagnostics.get(key), expected_value, f"{component.name} store.{key}")

    benchmark = result.get("benchmark")
    _require(isinstance(benchmark, Mapping), f"{component.name} lacks benchmark receipt.")
    _exact_equal(benchmark.get("model"), BENCHMARK_ENDER20, f"{component.name} benchmark model")
    if not confirmation:
        _exact_equal(
            benchmark.get("file"),
            "v5.3/downsampled_full_benchmark_models.parquet",
            f"{component.name} benchmark path",
        )

    training = result.get("training")
    _require(isinstance(training, Mapping), f"{component.name} lacks training receipt.")
    _exact_equal(
        training.get("data_sampling"),
        {"max_train_samples": 500_000, "sample_seed": 1337},
        f"{component.name} sampling receipt",
    )
    _exact_equal(training.get("data_mode"), expected_data["data_mode"], f"{component.name} training mode")
    _exact_equal(
        training.get("cv"),
        {
            "enabled": True,
            "n_splits": 5,
            "embargo": 52 if confirmation else 13,
            "mode": "expanding",
            "min_train_size": 0,
        },
        f"{component.name} training CV",
    )
    cv = result.get("cv")
    _require(isinstance(cv, Mapping), f"{component.name} lacks CV receipt.")
    for key, value in {
        "n_splits": 5,
        "embargo": 52 if confirmation else 13,
        "mode": "expanding",
        "min_train_size": 0,
        "folds_used": len(expected.folds),
    }.items():
        _exact_equal(cv.get(key), value, f"{component.name} cv.{key}")
    folds = cv.get("folds")
    _require(
        isinstance(folds, list) and len(folds) == len(expected.folds),
        f"{component.name} CV fold receipt is incomplete.",
    )
    for actual, frozen in zip(folds, expected.folds, strict=True):
        _require(isinstance(actual, Mapping), f"{component.name} CV fold is malformed.")
        for key, value in frozen.items():
            _exact_equal(actual.get(key), value, f"{component.name} fold {frozen['fold']} {key}")
        diagnostics = actual.get("model_diagnostics")
        _require(isinstance(diagnostics, Mapping), f"{component.name} fold lacks diagnostics.")
        _exact_equal(diagnostics.get("effective_device_type"), "gpu", f"{component.name} fold GPU device")
        _exact_equal(diagnostics.get("gpu_fallback_used"), False, f"{component.name} fold GPU fallback")

    output = result.get("output")
    _require(isinstance(output, Mapping), f"{component.name} lacks output receipt.")
    _exact_equal(_path_leaf(output.get("predictions_file")), component.predictions.name, f"{component.name} output file")
    semantics = _expected_semantics(config)
    _exact_equal(output.get("prediction_semantics"), semantics, f"{component.name} result semantics")
    metrics = result.get("metrics")
    _require(
        isinstance(metrics, Mapping)
        and set(metrics) == {"corr", "bmc", "bmc_last_200_eras"},
        f"{component.name} automatic metric schema differs.",
    )
    return semantics


def _read_prediction_semantics(path: Path) -> dict[str, Any] | None:
    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(PREDICTION_SEMANTICS_METADATA_KEY)
    if raw is None:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EnderEnsembleEvaluationError(f"{path.name} semantics are invalid JSON.") from error
    _require(isinstance(value, dict), f"{path.name} semantics are malformed.")
    return value


def _independent_fold_map(expected: ExpectedCohort) -> dict[str, int]:
    """Derive a producer fold map from the frozen chronology, not another artifact."""

    mapping: dict[str, int] = {}
    offset = 0
    for fold in expected.folds:
        count = int(fold["val_eras"])
        labels = expected.eras[offset : offset + count]
        _exact_equal(len(labels), count, "independent validation-era count")
        mapping.update({era: int(fold["fold"]) for era in labels})
        offset += count
    _exact_equal(offset, len(expected.eras), "independent fold era coverage")
    return mapping


def validate_prediction_artifact(
    path: Path,
    expected: pd.DataFrame,
    expected_semantics: Mapping[str, Any] | None,
    *,
    target_column: str,
    expected_fold_by_era: Mapping[str, int] | None = None,
    require_semantics: bool = True,
) -> np.ndarray:
    """Validate exact one-to-one artifact alignment and return raw predictions."""

    _require(path.is_file(), f"Prediction artifact is missing: {path}")
    semantics = _read_prediction_semantics(path)
    if require_semantics:
        _exact_equal(semantics, expected_semantics, f"{path.name} Parquet semantics")
    else:
        _exact_equal(semantics, None, f"{path.name} legacy semantics absence")
    columns = [ID_COLUMN, ERA_COLUMN, target_column, PREDICTION_COLUMN, FOLD_COLUMN]
    _exact_equal(pq.read_schema(path).names, columns, f"{path.name} column schema")
    artifact = pd.read_parquet(path, columns=columns)
    _exact_equal(len(artifact), len(expected), f"{path.name} row count")
    _require(artifact[ID_COLUMN].notna().all() and artifact[ID_COLUMN].is_unique, f"{path.name} IDs are invalid.")
    _validate_finite_columns(artifact, [target_column, PREDICTION_COLUMN, FOLD_COLUMN], path.name)
    if expected_fold_by_era is None:
        expected_fold_by_era = {
            str(era): int(fold)
            for era, fold in expected[[ERA_COLUMN, FOLD_COLUMN]].drop_duplicates().itertuples(index=False)
        }
    artifact_eras = artifact[ERA_COLUMN].astype(str)
    derived_fold = artifact_eras.map(expected_fold_by_era)
    _require(derived_fold.notna().all(), f"{path.name} contains an unexpected validation era.")
    artifact_fold_values = artifact[FOLD_COLUMN].to_numpy(dtype=np.float64)
    _require(
        np.equal(artifact_fold_values, np.floor(artifact_fold_values)).all(),
        f"{path.name} fold provenance contains a fractional fold label.",
    )
    _require(
        np.array_equal(
            derived_fold.to_numpy(dtype=np.int64),
            artifact_fold_values.astype(np.int64),
        ),
        f"{path.name} fold provenance differs from its frozen producer.",
    )
    renamed = artifact.rename(
        columns={
            ERA_COLUMN: "_artifact_era",
            target_column: "_artifact_target",
            FOLD_COLUMN: "_artifact_fold",
        }
    )
    ordered = expected.reset_index(drop=True).copy()
    ordered["_expected_order"] = np.arange(len(ordered), dtype=np.int64)
    try:
        aligned = ordered.merge(
            renamed,
            how="left",
            on=ID_COLUMN,
            sort=False,
            validate="one_to_one",
            indicator=True,
        )
    except pd.errors.MergeError as error:
        raise EnderEnsembleEvaluationError(f"{path.name} IDs do not align one-to-one.") from error
    _require((aligned["_merge"] == "both").all(), f"{path.name} is missing expected IDs.")
    aligned = aligned.sort_values("_expected_order", kind="stable")
    _require(
        np.array_equal(aligned[ERA_COLUMN].astype(str).to_numpy(), aligned["_artifact_era"].astype(str).to_numpy()),
        f"{path.name} eras differ by ID.",
    )
    _require(
        np.array_equal(aligned[target_column].to_numpy(), aligned["_artifact_target"].to_numpy()),
        f"{path.name} stored target differs by ID.",
    )
    return aligned[PREDICTION_COLUMN].to_numpy(dtype=np.float64, copy=True)


def _validate_tabm_result(result: Mapping[str, Any], label: str, *, metadata_required: bool) -> dict[str, Any] | None:
    model = result.get("model")
    _require(isinstance(model, Mapping), f"{label} result lacks model receipt.")
    _exact_equal(
        model.get("target_transform"),
        {
            "type": "residual_to_benchmark",
            "benchmark_col": BENCHMARK_ENDER20,
            "era_col": ERA_COLUMN,
            "per_era": True,
            "fit_intercept": True,
        },
        f"{label} target transform",
    )
    output = result.get("output")
    _require(isinstance(output, Mapping), f"{label} result lacks output receipt.")
    semantics = output.get("prediction_semantics")
    if metadata_required:
        _require(isinstance(semantics, Mapping), f"{label} result lacks prediction semantics.")
        _exact_equal(
            semantics.get("training_target", {}).get("transform", {}).get("type"),
            "residual_to_benchmark",
            f"{label} semantic target transform",
        )
        return dict(semantics)
    return None


def load_frozen_two_seed_residual(
    protocol: FrozenProtocol,
    expected: ExpectedCohort,
    *,
    confirmation: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate and construct only the gate-authorized two-seed TabM reference."""

    section_name = "confirmation" if confirmation else "scout"
    section = protocol.source_manifest["tabm_similarity_reference"][section_name]
    expected_fold_map = _independent_fold_map(expected)
    raw: list[np.ndarray] = []
    receipts: dict[str, Any] = {}
    for seed in ("seed1337", "seed2027"):
        entry = section[seed]
        config_path = _safe_repo_path(protocol.repo_root, entry["config"]["path"])
        result_path = _safe_repo_path(protocol.repo_root, entry["result"]["path"])
        predictions_path = _safe_repo_path(protocol.repo_root, entry["predictions"]["path"])
        config = _load_config(config_path, f"TabM {section_name} {seed} config")
        _exact_equal(
            config.get("model", {}).get("target_transform", {}).get("type"),
            "residual_to_benchmark",
            f"TabM {section_name} {seed} config transform",
        )
        result = _load_json(result_path, f"TabM {section_name} {seed} result")
        semantics = _validate_tabm_result(
            result,
            f"TabM {section_name} {seed}",
            metadata_required=confirmation,
        )
        raw.append(
            validate_prediction_artifact(
                predictions_path,
                expected.frame,
                semantics,
                target_column=ENDER_TARGET,
                expected_fold_by_era=expected_fold_map,
                require_semantics=confirmation,
            )
        )
        receipts[seed] = {
            "config": _sha256_file(config_path),
            "result": _sha256_file(result_path),
            "predictions": _sha256_file(predictions_path),
        }
    eras = expected.frame[ERA_COLUMN]
    first = rank_within_era(raw[0], eras)
    second = rank_within_era(raw[1], eras)
    return rank_within_era(0.5 * (first + second), eras), receipts


def _file_receipt(path: Path, repo_root: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Receipt input is missing: {path}")
    return {
        "path": _relative_path(path, repo_root),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _protocol_binding(protocol: FrozenProtocol) -> dict[str, Any]:
    gate_path = protocol.experiment_dir / "gate.md"
    evaluator_path = Path(__file__).resolve()
    return {
        "pre_scoring_commit": PRE_SCORING_COMMIT,
        "pretraining_commit": protocol.pretraining_commit,
        "source_manifest": _file_receipt(
            protocol.source_manifest_path, protocol.repo_root
        ),
        "gate": _file_receipt(gate_path, protocol.repo_root),
        "evaluator": _file_receipt(evaluator_path, protocol.repo_root),
        "imported_evaluators": {
            "hybrid": _file_receipt(Path(hybrid.__file__).resolve(), protocol.repo_root),
            "xerxes": _file_receipt(Path(xerxes.__file__).resolve(), protocol.repo_root),
        },
        "gpu_runtime": _file_receipt(
            protocol.gpu_runtime_path, protocol.repo_root
        ),
    }


def _validate_protocol_binding(
    receipt: Mapping[str, Any],
    protocol: FrozenProtocol,
    *,
    allow_prior_pretraining_commit: bool = False,
) -> None:
    binding = receipt.get("protocol")
    _require(isinstance(binding, Mapping), "Stage receipt protocol binding is malformed.")
    expected = _protocol_binding(protocol)
    if not allow_prior_pretraining_commit:
        _exact_equal(binding, expected, "stage protocol binding")
        return
    actual_commit = binding.get("pretraining_commit")
    _require(
        isinstance(actual_commit, str) and len(actual_commit) == 40,
        "Prior stage pretraining commit is malformed.",
    )
    actual_without_commit = dict(binding)
    expected_without_commit = dict(expected)
    actual_without_commit.pop("pretraining_commit", None)
    expected_without_commit.pop("pretraining_commit", None)
    _exact_equal(
        actual_without_commit,
        expected_without_commit,
        "prior stage protocol binding",
    )
    ancestor = _run_git(
        protocol.repo_root,
        ["merge-base", "--is-ancestor", actual_commit, protocol.pretraining_commit],
    )
    _require(
        ancestor.returncode == 0,
        "Prior stage pretraining checkpoint is not an ancestor of the current checkpoint.",
    )


def _cohort_receipt(expected: ExpectedCohort) -> dict[str, Any]:
    return {
        "rows": len(expected.frame),
        "eras": len(expected.eras),
        "first_era": expected.eras[0],
        "last_era": expected.eras[-1],
        "full_rows": expected.full_rows,
        "full_eras": expected.full_eras,
        "folds": [dict(fold) for fold in expected.folds],
    }


def _artifact_receipt(
    protocol: FrozenProtocol,
    component: ComponentPaths,
) -> dict[str, Any]:
    return {
        "component": component.name,
        "target": COMPONENT_TARGETS[component.name],
        "config": _file_receipt(component.config, protocol.repo_root),
        "result": _file_receipt(component.result, protocol.repo_root),
        "predictions": _file_receipt(component.predictions, protocol.repo_root),
    }


def _validate_scout_component(
    protocol: FrozenProtocol,
    component: ComponentPaths,
    expected: ExpectedCohort,
) -> tuple[np.ndarray, dict[str, Any]]:
    _require(component.name in ALL_COMPONENTS, f"Unknown Scout component: {component.name}")
    config = _load_config(component.config, f"{component.name} Scout config")
    validate_component_config(component.name, config)
    result = _load_json(component.result, f"{component.name} Scout result")
    semantics = validate_result_json(component, result, config, expected)
    raw = validate_prediction_artifact(
        component.predictions,
        expected.frame,
        semantics,
        target_column=COMPONENT_TARGETS[component.name],
        expected_fold_by_era=_independent_fold_map(expected),
    )
    return raw, _artifact_receipt(protocol, component)


def _reused_xerxes_paths(protocol: FrozenProtocol) -> ComponentPaths:
    frozen = protocol.source_manifest["reused_xerxes_component"]
    return ComponentPaths(
        name="xerxes",
        config=_safe_repo_path(protocol.repo_root, frozen["config"]["path"]),
        result=_safe_repo_path(protocol.repo_root, frozen["result"]["path"]),
        predictions=_safe_repo_path(
            protocol.repo_root, frozen["predictions"]["path"]
        ),
    )


def seal_scout_component(
    protocol: FrozenProtocol,
    component: ComponentPaths,
    receipt_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Seal one successful new Scout run without persisting any Ender metric."""

    _require(
        component.name in SCOUT_NEW_COMPONENTS,
        "Only the four new Scout components may be sealed.",
    )
    expected_paths = default_scout_component_paths(protocol, component.name)
    _exact_equal(component, expected_paths, f"{component.name} Scout destinations")
    expected = build_scout_expected_cohort(protocol)
    _, artifact = _validate_scout_component(protocol, component, expected)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "stage": "seal-scout-component",
        "state": "SEALED",
        "passed": True,
        "protocol": _protocol_binding(protocol),
        "component": component.name,
        "cohort": _cohort_receipt(expected),
        "artifact": artifact,
        "gpu_folds_verified": len(expected.folds),
    }
    path = _write_content_addressed_receipt(
        receipt_dir,
        f"scout-seal-{component.name}",
        receipt,
    )
    return path, receipt


def validate_seal_receipts(
    protocol: FrozenProtocol,
    bindings: Mapping[str, tuple[Path, str]],
    expected: ExpectedCohort | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Revalidate four exact seals and their underlying artifacts independently."""

    _exact_equal(set(bindings), set(SCOUT_NEW_COMPONENTS), "Scout seal components")
    cohort = expected if expected is not None else build_scout_expected_cohort(protocol)
    signals: dict[str, np.ndarray] = {}
    normalized: dict[str, dict[str, Any]] = {}
    for name in SCOUT_NEW_COMPONENTS:
        path, digest = bindings[name]
        path = path.resolve()
        seal = _load_bound_receipt(
            path,
            digest,
            expected_stage="seal-scout-component",
        )
        _exact_equal(seal.get("passed"), True, f"{name} seal passage")
        _exact_equal(seal.get("state"), "SEALED", f"{name} seal state")
        _exact_equal(seal.get("component"), name, f"{name} seal component")
        _validate_protocol_binding(seal, protocol)
        _exact_equal(seal.get("cohort"), _cohort_receipt(cohort), f"{name} seal cohort")
        component = default_scout_component_paths(protocol, name)
        raw, artifact = _validate_scout_component(protocol, component, cohort)
        _exact_equal(seal.get("artifact"), artifact, f"{name} sealed artifact")
        _exact_equal(
            seal.get("gpu_folds_verified"),
            len(cohort.folds),
            f"{name} sealed GPU folds",
        )
        signals[name] = raw
        normalized[name] = {
            "path": _relative_path(path, protocol.repo_root),
            "sha256": digest,
            "artifact": artifact,
        }
    return signals, normalized


def _build_scout_scoring_frame(
    protocol: FrozenProtocol,
    expected: ExpectedCohort,
    bindings: Mapping[str, tuple[Path, str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    signals, seals = validate_seal_receipts(protocol, bindings, expected)
    xerxes_paths = _reused_xerxes_paths(protocol)
    xerxes_raw, xerxes_artifact = _validate_scout_component(
        protocol, xerxes_paths, expected
    )
    signals["xerxes"] = xerxes_raw
    tabm, tabm_receipts = load_frozen_two_seed_residual(
        protocol, expected, confirmation=False
    )
    columns = [
        ID_COLUMN,
        ERA_COLUMN,
        ENDER_TARGET,
        BENCHMARK_ENDER20,
        BENCHMARK_ENDER60,
        FOLD_COLUMN,
    ]
    frame = expected.frame[columns].copy()
    for name in ALL_COMPONENTS:
        frame[name] = signals[name]
    frame["tabm_two_seed_residual"] = tabm
    return frame, {
        "seal_receipts": seals,
        "reused_xerxes": xerxes_artifact,
        "tabm_two_seed_residual": tabm_receipts,
        "cohort": _cohort_receipt(expected),
    }


def _slice_eras(
    frame: pd.DataFrame,
    eras: Sequence[str],
) -> pd.DataFrame:
    era_set = set(eras)
    sliced = frame[frame[ERA_COLUMN].astype(str).isin(era_set)].copy()
    _exact_equal(
        tuple(sorted(sliced[ERA_COLUMN].astype(str).unique(), key=int)),
        tuple(eras),
        "scoring slice eras",
    )
    return sliced.reset_index(drop=True)


def _serialize_per_era(
    per_era: Mapping[str, pd.DataFrame],
    signals: Sequence[str],
) -> dict[str, dict[str, list[list[Any]]]]:
    payload: dict[str, dict[str, list[list[Any]]]] = {}
    for metric, scores in per_era.items():
        payload[metric] = {
            signal: [
                [str(era), float(value)]
                for era, value in scores[signal].items()
            ]
            for signal in signals
        }
    _reject_nonfinite(payload, "per-era metrics")
    return payload


def _seal_bindings_from_receipt(
    protocol: FrozenProtocol,
    value: Any,
) -> dict[str, tuple[Path, str]]:
    _require(isinstance(value, Mapping), "Stage seal receipt bindings are malformed.")
    _exact_equal(set(value), set(SCOUT_NEW_COMPONENTS), "Stage seal components")
    result: dict[str, tuple[Path, str]] = {}
    for name in SCOUT_NEW_COMPONENTS:
        item = value[name]
        _require(isinstance(item, Mapping), f"{name} stage seal binding is malformed.")
        path = _safe_repo_path(protocol.repo_root, str(item.get("path")))
        digest = item.get("sha256")
        _require(isinstance(digest, str), f"{name} stage seal hash is malformed.")
        result[name] = (path, digest)
    return result


def run_calibrate(
    protocol: FrozenProtocol,
    seal_bindings: Mapping[str, tuple[Path, str]],
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Score all and only the five frozen formulas on the first 164 eras."""

    expected = build_scout_expected_cohort(protocol)
    full_frame, inputs = _build_scout_scoring_frame(
        protocol, expected, seal_bindings
    )
    calibration_eras = expected.eras[:SCOUT_CALIBRATION_ERAS]
    calibration = _slice_eras(full_frame, calibration_eras)
    _exact_equal(
        calibration_eras[-1],
        SCOUT_LAST_CALIBRATION_ERA,
        "Scout calibration boundary",
    )
    scored = build_rank_blends(calibration)
    metrics = compute_per_era_metrics(
        scored,
        CANDIDATE_NAMES,
        calibration_eras,
        tabm_column="tabm_two_seed_residual",
    )
    summaries = {
        candidate: summarize_signal(metrics, candidate)
        for candidate in CANDIDATE_NAMES
    }
    selected, evaluations = select_scout_candidate(summaries)
    candidates = {
        candidate: {
            "summary": summaries[candidate],
            **evaluations[candidate],
        }
        for candidate in CANDIDATE_NAMES
    }
    passed = selected is not None
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "stage": "calibrate",
        "state": "PASS" if passed else "STOP_NO_ELIGIBLE_CANDIDATE",
        "passed": passed,
        "protocol": _protocol_binding(protocol),
        "inputs": inputs,
        "selected_formula": (
            {
                "name": selected,
                "weights": dict(BLEND_WEIGHTS[selected]),
            }
            if selected is not None
            else None
        ),
        "calibration": {
            "rows": len(calibration),
            "eras": len(calibration_eras),
            "first_era": calibration_eras[0],
            "last_era": calibration_eras[-1],
            "candidates": candidates,
            "per_era": _serialize_per_era(metrics, CANDIDATE_NAMES),
        },
    }
    path = _write_content_addressed_receipt(output_dir, "calibrate", receipt)
    return path, receipt


def _load_passing_stage_receipt(
    protocol: FrozenProtocol,
    path: Path,
    digest: str,
    *,
    stage: str,
    allow_prior_pretraining_commit: bool = False,
) -> dict[str, Any]:
    receipt = _load_bound_receipt(path, digest, expected_stage=stage)
    _exact_equal(receipt.get("passed"), True, f"{stage} passage")
    _exact_equal(receipt.get("state"), "PASS", f"{stage} state")
    _validate_protocol_binding(
        receipt,
        protocol,
        allow_prior_pretraining_commit=allow_prior_pretraining_commit,
    )
    return receipt


def _selected_formula(receipt: Mapping[str, Any]) -> str:
    formula = receipt.get("selected_formula")
    _require(isinstance(formula, Mapping), "Passing stage has no selected formula.")
    name = formula.get("name")
    _require(name in CANDIDATE_NAMES, "Passing stage selected an unknown formula.")
    assert isinstance(name, str)
    _exact_equal(formula.get("weights"), BLEND_WEIGHTS[name], "selected weights")
    return name


def run_locked(
    protocol: FrozenProtocol,
    calibration_receipt_path: Path,
    calibration_receipt_sha256: str,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Open the Scout holdout for only the immutable selected formula."""

    calibration_receipt_path = calibration_receipt_path.resolve()
    calibration_receipt = _load_passing_stage_receipt(
        protocol,
        calibration_receipt_path,
        calibration_receipt_sha256,
        stage="calibrate",
    )
    selected = _selected_formula(calibration_receipt)
    inputs = calibration_receipt.get("inputs")
    _require(isinstance(inputs, Mapping), "Calibration inputs are malformed.")
    bindings = _seal_bindings_from_receipt(protocol, inputs.get("seal_receipts"))
    expected = build_scout_expected_cohort(protocol)
    full_frame, current_inputs = _build_scout_scoring_frame(
        protocol, expected, bindings
    )
    _exact_equal(current_inputs, inputs, "locked Scout input revalidation")
    locked_eras = expected.eras[-SCOUT_LOCKED_ERAS:]
    _exact_equal(locked_eras[0], SCOUT_FIRST_LOCKED_ERA, "Scout locked boundary")
    locked = _slice_eras(full_frame, locked_eras)
    scored = build_selected_rank_blend(locked, selected)
    metrics = compute_per_era_metrics(
        scored,
        [selected],
        locked_eras,
        tabm_column="tabm_two_seed_residual",
    )
    summary = summarize_signal(metrics, selected)
    checks = locked_checks(summary)
    passed = all(checks.values())
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "stage": "locked",
        "state": "PASS" if passed else "STOP_SCOUT_LOCKED_FAILED",
        "passed": passed,
        "protocol": _protocol_binding(protocol),
        "input_receipt": {
            "path": _relative_path(calibration_receipt_path, protocol.repo_root),
            "sha256": calibration_receipt_sha256,
        },
        "selected_formula": {
            "name": selected,
            "weights": dict(BLEND_WEIGHTS[selected]),
        },
        "locked": {
            "rows": len(locked),
            "eras": len(locked_eras),
            "first_era": locked_eras[0],
            "last_era": locked_eras[-1],
            "summary": summary,
            "checks": checks,
            "per_era": _serialize_per_era(metrics, [selected]),
        },
    }
    path = _write_content_addressed_receipt(output_dir, "locked", receipt)
    return path, receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _repo_root()
    experiment_dir = root / "numerai/agents/experiments" / EXPERIMENT_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "seal-scout-component",
            "calibrate",
            "locked",
            "confirmation-calibrate",
            "confirmation-locked",
        ),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=experiment_dir / "source_manifest.json",
    )
    parser.add_argument(
        "--pretraining-commit",
        required=True,
        help="Full implementation/config checkpoint committed before the relevant run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment_dir / "receipts",
    )
    parser.add_argument("--component", choices=SCOUT_NEW_COMPONENTS)
    parser.add_argument("--receipt-dir", type=Path)
    parser.add_argument(
        "--seal-receipt",
        action="append",
        nargs=3,
        metavar=("COMPONENT", "PATH", "SHA256"),
        help="Exact content-addressed scout seal; repeat once per new component.",
    )
    parser.add_argument("--input-receipt", type=Path)
    parser.add_argument("--input-receipt-sha256")
    parser.add_argument("--confirmation-input-receipt", type=Path)
    parser.add_argument("--confirmation-input-receipt-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = verify_frozen_protocol(
        args.source_manifest.resolve(),
        pretraining_commit=args.pretraining_commit,
    )
    if args.stage == "seal-scout-component":
        _require(args.component is not None, "--component is required for sealing.")
        receipt_dir = (args.receipt_dir or args.output_dir).resolve()
        path, receipt = seal_scout_component(
            protocol,
            default_scout_component_paths(protocol, args.component),
            receipt_dir,
        )
    elif args.stage == "calibrate":
        path, receipt = run_calibrate(
            protocol,
            _parse_seal_bindings(args.seal_receipt),
            args.output_dir.resolve(),
        )
    else:
        _require(args.input_receipt is not None, "--input-receipt is required.")
        _require(
            args.input_receipt_sha256 is not None,
            "--input-receipt-sha256 is required.",
        )
        if args.stage == "locked":
            path, receipt = run_locked(
                protocol,
                args.input_receipt.resolve(),
                args.input_receipt_sha256,
                args.output_dir.resolve(),
            )
        elif args.stage == "confirmation-calibrate":
            path, receipt = run_confirmation_calibrate(
                protocol,
                args.input_receipt.resolve(),
                args.input_receipt_sha256,
                args.output_dir.resolve(),
                confirmation_input_receipt=(
                    args.confirmation_input_receipt.resolve()
                    if args.confirmation_input_receipt is not None
                    else None
                ),
                confirmation_input_receipt_sha256=(
                    args.confirmation_input_receipt_sha256
                ),
            )
        else:
            path, receipt = run_confirmation_locked(
                protocol,
                args.input_receipt.resolve(),
                args.input_receipt_sha256,
                args.output_dir.resolve(),
            )
    print(
        json.dumps(
            {
                "receipt": str(path),
                "receipt_sha256": _sha256_file(path),
                "stage": receipt["stage"],
                "state": receipt["state"],
                "passed": receipt["passed"],
            },
            sort_keys=True,
        )
    )
    return 0 if bool(receipt["passed"]) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EnderEnsembleEvaluationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
