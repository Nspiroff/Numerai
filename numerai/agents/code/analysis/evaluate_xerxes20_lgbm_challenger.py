"""Evaluate the frozen Xerxes20 LightGBM challenger protocol.

The evaluator is deliberately training-free.  It verifies the pre-scoring Git
checkpoint and source manifest, validates every config/result/OOF artifact,
scores the four scout profiles on Ender20, and opens locked metrics only for
the calibration winner.  An optional confirmation stage validates the sole
winner on the consecutive cohort and compares it with the frozen two-seed
TabM residual.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from agents.code.analysis import evaluate_ender20_hybrid_stability as hybrid
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


EXPERIMENT_NAME = "xerxes20_lgbm_challenger_v53"
AS_OF_DATE = "2026-08-03"
PRE_SCORING_COMMIT = "0f892c712c870bfd97eb7735eec944f3f2c60d2f"
SOURCE_MANIFEST_SHA256 = (
    "4b3dd7e30dbcb8e532ffbdd484031c98efc30cf4d82804290f62846e19675a8d"
)
GPU_RUNTIME_SHA256 = (
    "d6656bc39a0d603860c9b327569bd453b1556b8a3aae99f8567edefbc214f135"
)
EXPECTED_LIGHTGBM_VERSION = "4.7.0"
EXPECTED_PYTHON_MAJOR_MINOR = (3, 12)
EXPECTED_LIGHTGBM_DLL_SIZE = 4_169_216
EXPECTED_LIGHTGBM_DLL_SHA256 = (
    "2ab79db409bba74a97b40485126528358058ebec914cac176e38d9fb1bfdb356"
)
TWO_SEED_RECEIPT_SHA256 = (
    "960b299f85bc68dfdb6c84a88d38008058b6f6c38aee8cfc0887850ed03bf95c"
)
TWO_SEED_PREDICTION_SHA256 = {
    "seed1337": "196f56053eb23a04d32c80118dd90e99c41290c99cefcf1de2afd05bb1c4597e",
    "seed2027": "58027368888ba806383003acb8cdbcc6252223b0b7539537c66d7cedd94601e4",
}

ID_COLUMN = "id"
ERA_COLUMN = "era"
XERXES_TARGET = "target_xerxes_20"
ENDER_TARGET = "target_ender_20"
BENCHMARK_COLUMN = "v53_lgbm_ender20"
PREDICTION_COLUMN = "prediction"
FOLD_COLUMN = "cv_fold"

SCOUT_ROWS = 1_279_658
SCOUT_ERAS = 214
SCOUT_FIRST_ERA = "0373"
SCOUT_LAST_ERA = "1225"
SCOUT_CALIBRATION_ERAS = 164
SCOUT_HOLDOUT_ERAS = 50
SCOUT_LAST_CALIBRATION_ERA = "1025"
SCOUT_FIRST_HOLDOUT_ERA = "1029"

CONFIRMATION_ROWS = 5_112_039
CONFIRMATION_ERAS = 855
CONFIRMATION_FIRST_ERA = "0371"
CONFIRMATION_LAST_ERA = "1225"
CONFIRMATION_CALIBRATION_ERAS = 655
CONFIRMATION_HOLDOUT_ERAS = 200
CONFIRMATION_LAST_CALIBRATION_ERA = "1025"
CONFIRMATION_FIRST_HOLDOUT_ERA = "1026"
DISK_PREDICTION_BATCH_SIZE_MAX = 65_536

SCOUT_CONFIG_HASHES = {
    "r1_base_d6_t6000": (
        "5ec02a6647eb6f14dea6fc3a7c8c358f9fd69e3e37dcd4220d41c25c3fbf143b"
    ),
    "r1_trees2k": (
        "b1211a94b72f5f8236d4d9d3a4e82e6b5e7cdc2df9b8f4b8fcd746e753056c2a"
    ),
    "r1_depth5": (
        "05205e3852984db5b1ed09a65ecc19bf4e376e845bc2c3d59c395a8b1da3deb5"
    ),
    "r1_depth8": (
        "ed186040dcd6899090302575c1a1503c892dba953365049ee5ceefd3467f6d69"
    ),
}
SCOUT_DEPTHS = {
    "r1_base_d6_t6000": 6,
    "r1_trees2k": 6,
    "r1_depth5": 5,
    "r1_depth8": 8,
}
SCOUT_NAMES = tuple(SCOUT_CONFIG_HASHES)

SCOUT_CALIBRATION_THRESHOLDS = {
    "bmc_mean_min_exclusive": 0.0010,
    "bmc_sharpe_min_exclusive": 0.20,
    "bmc_max_drawdown_max_exclusive": 0.15,
    "corr_mean_min_exclusive": 0.010,
    "benchmark_similarity_max_exclusive": 0.85,
}
SCOUT_HOLDOUT_THRESHOLDS = {
    "bmc_mean_min_exclusive": 0.0,
    "bmc_sharpe_min_exclusive": 0.20,
    "bmc_max_drawdown_max_exclusive": 0.10,
    "corr_mean_min_exclusive": 0.008,
}
CONFIRMATION_THRESHOLDS = {
    "calibration_bmc_mean_min_inclusive": 0.0015,
    "calibration_bmc_sharpe_min_exclusive": 0.35,
    "calibration_bmc_max_drawdown_max_exclusive": 0.15,
    "calibration_corr_mean_min_inclusive": 0.012,
    "calibration_benchmark_similarity_max_exclusive": 0.75,
    "calibration_tabm_similarity_max_exclusive": 0.75,
    "holdout_bmc_mean_min_exclusive": 0.0,
    "holdout_bmc_max_drawdown_max_exclusive": 0.15,
    "holdout_corr_mean_min_exclusive": 0.0,
    "full_bmc_mean_min_inclusive": 0.0015,
    "full_bmc_sharpe_min_exclusive": 0.35,
    "full_bmc_max_drawdown_max_exclusive": 0.15,
    "full_corr_mean_min_inclusive": 0.012,
}

PROTOCOL_CHECKPOINT_PATHS = (
    "numerai/agents/experiments/xerxes20_lgbm_challenger_v53/configs",
    "numerai/agents/experiments/xerxes20_lgbm_challenger_v53/gate.md",
    "numerai/agents/experiments/xerxes20_lgbm_challenger_v53/source_manifest.json",
)
TRAINING_CHECKPOINT_PATHS = (
    "numerai/agents/code/analysis/evaluate_ender20_hybrid_stability.py",
    "numerai/agents/code/analysis/evaluate_xerxes20_lgbm_challenger.py",
    "numerai/agents/code/modeling",
    "numerai/agents/code/metrics/numerai_metrics.py",
    "numerai/agents/code/data/build_full_datasets.py",
    "numerai/agents/experiments/xerxes20_lgbm_challenger_v53/gpu_runtime.json",
)


class XerxesEvaluationError(ValueError):
    """Raised when a frozen input, artifact, or decision contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XerxesEvaluationError(message)


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
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise XerxesEvaluationError(f"Invalid {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} must be a JSON object.")
    _reject_nonfinite(value, label)
    return value


def _reject_nonfinite(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        _require(np.isfinite(float(value)), f"{label} contains a non-finite number.")
        return
    if isinstance(value, list):
        for item in value:
            _reject_nonfinite(item, label)
        return
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite(item, label)
        return
    raise XerxesEvaluationError(f"{label} contains an unsupported JSON value.")


def _exact_equal(actual: Any, expected: Any, label: str) -> None:
    _require(actual == expected, f"{label} differs from the frozen contract.")


def _exact_keys(value: Any, keys: Sequence[str], label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} must be a JSON object.")
    _exact_equal(set(value), set(keys), f"{label} schema")
    return value


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _safe_relative_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    _require(
        candidate == root.resolve() or root.resolve() in candidate.parents,
        f"Source-manifest path escapes repository: {relative}",
    )
    return candidate


def _path_leaf(value: Any) -> str:
    _require(isinstance(value, str) and value, "Artifact path is missing.")
    return value.replace("\\", "/").rstrip("/").split("/")[-1]


def _run_git(repo_root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )


@dataclass(frozen=True)
class FrozenProtocol:
    repo_root: Path
    experiment_dir: Path
    source_manifest_path: Path
    source_manifest: dict[str, Any]
    configs: dict[str, dict[str, Any]]
    config_paths: dict[str, Path]
    medium_features: tuple[str, ...]
    ender_metadata_path: Path
    ender_manifest_path: Path
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
class RunPaths:
    name: str
    config: Path
    result: Path
    predictions: Path


def _validate_parquet_fingerprint(
    path: Path, expected: Mapping[str, Any], label: str
) -> None:
    actual = parquet_source_fingerprint(path)
    for key in (
        "size_bytes",
        "mtime_ns",
        "num_rows",
        "num_row_groups",
        "schema_sha256",
        "footer_sha256",
    ):
        _exact_equal(actual[key], expected[key], f"{label} {key}")


def verify_checkpoint_boundaries(repo_root: Path, pretraining_commit: str) -> None:
    """Keep the frozen protocol and later training implementation separately bound."""

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
        ancestor = _run_git(
            repo_root, ["merge-base", "--is-ancestor", commit, "HEAD"]
        )
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
    boundaries = (
        (PRE_SCORING_COMMIT, PROTOCOL_CHECKPOINT_PATHS, "protocol"),
        (pretraining_commit, TRAINING_CHECKPOINT_PATHS, "training"),
    )
    for commit, paths, label in boundaries:
        unchanged = _run_git(
            repo_root, ["diff", "--quiet", commit, "--", *paths]
        )
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


def validate_gpu_runtime_receipt(path: Path) -> dict[str, Any]:
    """Validate the exact committed GPU-runtime receipt and nested schema."""

    _require(path.is_file(), f"GPU runtime receipt is missing: {path}")
    _exact_equal(_sha256_file(path), GPU_RUNTIME_SHA256, "GPU runtime receipt hash")
    receipt = _load_json(path, "GPU runtime receipt")
    _exact_keys(
        receipt,
        ("as_of_date", "environment", "gpu", "lightgbm", "proof", "schema_version"),
        "GPU runtime receipt",
    )
    _exact_equal(receipt["schema_version"], 1, "GPU runtime schema version")
    _exact_equal(receipt["as_of_date"], AS_OF_DATE, "GPU runtime as-of date")
    environment = _exact_keys(
        receipt["environment"], ("manager", "name", "packages"), "GPU environment"
    )
    _exact_equal(environment["name"], "numerai-lgbm-gpu312", "GPU environment name")
    packages = _exact_keys(
        environment["packages"],
        (
            "cloudpickle",
            "cmake",
            "khronos-opencl-icd-loader",
            "libboost",
            "libboost-devel",
            "libboost-headers",
            "ninja",
            "numerai-tools",
            "numerapi",
            "numpy",
            "opencl-headers",
            "pandas",
            "pyarrow",
            "python",
            "requests",
            "scikit-learn",
            "scipy",
            "tqdm",
        ),
        "GPU environment packages",
    )
    _require(
        str(packages["python"]).startswith("3.12."),
        "GPU runtime package receipt is not Python 3.12.",
    )
    gpu = _exact_keys(
        receipt["gpu"],
        (
            "compute_capability",
            "driver_version",
            "memory_mib",
            "name",
            "opencl_device_receipt",
        ),
        "GPU device",
    )
    _exact_equal(gpu["name"], "NVIDIA RTX A4500", "GPU device name")
    lightgbm = _exact_keys(
        receipt["lightgbm"],
        (
            "build_defines",
            "build_patch",
            "dll",
            "dynamic_dependencies",
            "source",
            "version",
        ),
        "LightGBM runtime",
    )
    _exact_equal(
        lightgbm["version"], EXPECTED_LIGHTGBM_VERSION, "LightGBM receipt version"
    )
    _exact_keys(
        lightgbm["build_defines"],
        ("BOOST_ROOT", "OpenCL_INCLUDE_DIR", "OpenCL_LIBRARY", "USE_GPU"),
        "LightGBM build defines",
    )
    _exact_equal(
        lightgbm["build_defines"]["USE_GPU"], "ON", "LightGBM GPU build flag"
    )
    _exact_keys(
        lightgbm["build_patch"],
        ("description", "patched_cmakelists_sha256"),
        "LightGBM build patch",
    )
    dll = _exact_keys(
        lightgbm["dll"], ("filename", "sha256", "size_bytes"), "LightGBM DLL"
    )
    _exact_equal(dll["filename"], "lib_lightgbm.dll", "LightGBM DLL filename")
    _exact_equal(
        dll["size_bytes"], EXPECTED_LIGHTGBM_DLL_SIZE, "LightGBM DLL receipt size"
    )
    _exact_equal(
        dll["sha256"], EXPECTED_LIGHTGBM_DLL_SHA256, "LightGBM DLL receipt hash"
    )
    dependencies = _exact_keys(
        lightgbm["dynamic_dependencies"],
        ("boost_filesystem.dll", "boost_system.dll", "OpenCL.lib"),
        "LightGBM dynamic dependencies",
    )
    for name, dependency in dependencies.items():
        _exact_keys(
            dependency, ("sha256", "size_bytes"), f"LightGBM dependency {name}"
        )
    _exact_keys(
        lightgbm["source"], ("filename", "pypi_sha256"), "LightGBM source"
    )
    proof = _exact_keys(
        receipt["proof"],
        (
            "direct_log_receipts",
            "direct_prediction_finite",
            "passed",
            "synthetic_features",
            "synthetic_rows",
            "wrapper_effective_device_type",
            "wrapper_gpu_fallback_used",
            "wrapper_prediction_finite",
        ),
        "GPU proof",
    )
    _exact_equal(proof["passed"], True, "GPU proof passed")
    _exact_equal(
        proof["wrapper_effective_device_type"], "gpu", "GPU wrapper device"
    )
    _exact_equal(proof["wrapper_gpu_fallback_used"], False, "GPU wrapper fallback")
    _exact_equal(
        proof["direct_prediction_finite"], True, "GPU direct prediction proof"
    )
    _exact_equal(
        proof["wrapper_prediction_finite"], True, "GPU wrapper prediction proof"
    )
    return receipt


def _probe_live_gpu_runtime() -> dict[str, Any]:
    """Inspect the LightGBM library loaded by this Python interpreter."""

    try:
        import lightgbm
        from lightgbm import basic as lightgbm_basic
    except ImportError as error:
        raise XerxesEvaluationError(
            "The evaluator interpreter does not provide the pinned LightGBM runtime."
        ) from error
    dll_name = getattr(getattr(lightgbm_basic, "_LIB", None), "_name", None)
    _require(
        isinstance(dll_name, str) and dll_name,
        "Loaded LightGBM DLL path is unavailable.",
    )
    dll_path = Path(dll_name).resolve()
    _require(dll_path.is_file(), f"Loaded LightGBM DLL is missing: {dll_path}")
    return {
        "python_major_minor": [sys.version_info.major, sys.version_info.minor],
        "python_version": sys.version.split()[0],
        "python_executable": str(Path(sys.executable).resolve()),
        "lightgbm_version": str(lightgbm.__version__),
        "dll_path": str(dll_path),
        "dll_size_bytes": dll_path.stat().st_size,
        "dll_sha256": _sha256_file(dll_path),
    }


def verify_live_gpu_runtime(path: Path) -> dict[str, Any]:
    """Bind the committed runtime receipt to the active evaluator interpreter."""

    validate_gpu_runtime_receipt(path)
    live = _probe_live_gpu_runtime()
    _exact_keys(
        live,
        (
            "python_major_minor",
            "python_version",
            "python_executable",
            "lightgbm_version",
            "dll_path",
            "dll_size_bytes",
            "dll_sha256",
        ),
        "Live GPU runtime probe",
    )
    _exact_equal(
        live["python_major_minor"],
        list(EXPECTED_PYTHON_MAJOR_MINOR),
        "Evaluator Python major/minor",
    )
    _exact_equal(
        live["lightgbm_version"],
        EXPECTED_LIGHTGBM_VERSION,
        "Evaluator LightGBM version",
    )
    _exact_equal(
        Path(live["dll_path"]).name,
        "lib_lightgbm.dll",
        "Loaded LightGBM DLL filename",
    )
    _exact_equal(
        live["dll_size_bytes"], EXPECTED_LIGHTGBM_DLL_SIZE, "Loaded LightGBM DLL size"
    )
    _exact_equal(
        live["dll_sha256"], EXPECTED_LIGHTGBM_DLL_SHA256, "Loaded LightGBM DLL hash"
    )
    return {
        "path": str(path.resolve()),
        "runtime_receipt_sha256": GPU_RUNTIME_SHA256,
        **live,
    }


def verify_frozen_protocol(
    source_manifest_path: Path | None = None,
    *,
    pretraining_commit: str,
) -> FrozenProtocol:
    """Verify the pre-scoring checkpoint and every source-manifest anchor."""

    repo_root = _repo_root()
    experiment_dir = (
        repo_root / "numerai/agents/experiments" / EXPERIMENT_NAME
    ).resolve()
    manifest_path = (
        source_manifest_path.resolve()
        if source_manifest_path is not None
        else experiment_dir / "source_manifest.json"
    )
    _require(
        _sha256_file(manifest_path) == SOURCE_MANIFEST_SHA256,
        "Xerxes source manifest differs from the pre-scoring checkpoint.",
    )
    manifest = _load_json(manifest_path, "Xerxes source manifest")

    verify_checkpoint_boundaries(repo_root, pretraining_commit)
    gpu_runtime_path = experiment_dir / "gpu_runtime.json"
    gpu_runtime_receipt = verify_live_gpu_runtime(gpu_runtime_path)

    experiment_files = manifest.get("experiment_files")
    _require(isinstance(experiment_files, dict), "Manifest experiment_files is malformed.")
    config_paths: dict[str, Path] = {}
    configs: dict[str, dict[str, Any]] = {}
    for relative, receipt in experiment_files.items():
        _require(isinstance(receipt, dict), f"Malformed receipt for {relative}.")
        path = _safe_relative_path(experiment_dir, relative)
        _require(path.is_file(), f"Frozen experiment file is missing: {relative}")
        _exact_equal(path.stat().st_size, receipt.get("size_bytes"), f"{relative} size")
        _exact_equal(_sha256_file(path), receipt.get("sha256"), f"{relative} hash")
    for name, expected_hash in SCOUT_CONFIG_HASHES.items():
        path = experiment_dir / "configs" / f"{name}.py"
        _exact_equal(_sha256_file(path), expected_hash, f"{name} config hash")
        namespace = runpy.run_path(str(path))
        config = namespace.get("CONFIG", namespace.get("config"))
        _require(isinstance(config, dict), f"{name} config has no CONFIG mapping.")
        validate_scout_config(name, config)
        config_paths[name] = path
        configs[name] = config

    feature_meta = manifest.get("feature_metadata")
    _require(isinstance(feature_meta, dict), "Manifest feature metadata is malformed.")
    features_path = _safe_relative_path(repo_root, feature_meta["path"])
    _exact_equal(features_path.stat().st_size, feature_meta["size_bytes"], "features.json size")
    _exact_equal(_sha256_file(features_path), feature_meta["sha256"], "features.json hash")
    features_json = _load_json(features_path, "features.json")
    feature_sets = features_json.get("feature_sets")
    _require(isinstance(feature_sets, dict), "features.json has no feature_sets mapping.")
    medium = feature_sets.get("medium")
    _require(
        isinstance(medium, list) and all(isinstance(value, str) for value in medium),
        "features.json medium feature set is malformed.",
    )
    _exact_equal(len(medium), feature_meta["medium_feature_count"], "medium feature count")
    _exact_equal(
        feature_order_sha256(medium),
        feature_meta["medium_feature_order_sha256"],
        "medium feature order hash",
    )

    scout_sources = manifest.get("scout_sources")
    _require(isinstance(scout_sources, dict), "Manifest scout_sources is malformed.")
    for relative, receipt in scout_sources.items():
        path = _safe_relative_path(repo_root, relative)
        _exact_equal(path.stat().st_size, receipt["size_bytes"], f"{relative} size")
        _exact_equal(_sha256_file(path), receipt["sha256"], f"{relative} hash")

    full_sources = manifest.get("full_source_fingerprints")
    _require(
        isinstance(full_sources, list) and len(full_sources) == 4,
        "Manifest full_source_fingerprints is malformed.",
    )
    for receipt in full_sources:
        path = _safe_relative_path(repo_root, receipt["path"])
        _validate_parquet_fingerprint(path, receipt, receipt["path"])

    anchor = manifest.get("existing_ender_store_anchor")
    _require(isinstance(anchor, dict), "Ender store anchor is malformed.")
    ender_dir = repo_root / "numerai/v5.3/target_ender_20_feature_store"
    ender_metadata_path = ender_dir / "metadata.json"
    ender_metadata = _load_json(ender_metadata_path, "Ender store metadata")
    for key in ("generation_id", "row_count", "feature_count", "target_column"):
        _exact_equal(ender_metadata.get(key), anchor.get(key), f"Ender store {key}")
    _exact_equal(
        ender_metadata["features"]["sha256"],
        anchor["feature_sha256"],
        "Ender feature hash receipt",
    )
    _exact_equal(
        ender_metadata["manifest"]["sha256"],
        anchor["manifest_sha256"],
        "Ender manifest hash receipt",
    )
    ender_manifest_path = ender_dir / ender_metadata["manifest"]["filename"]
    _exact_equal(
        _sha256_file(ender_manifest_path),
        anchor["manifest_sha256"],
        "Ender manifest payload hash",
    )
    return FrozenProtocol(
        repo_root=repo_root,
        experiment_dir=experiment_dir,
        source_manifest_path=manifest_path,
        source_manifest=manifest,
        configs=configs,
        config_paths=config_paths,
        medium_features=tuple(medium),
        ender_metadata_path=ender_metadata_path,
        ender_manifest_path=ender_manifest_path,
        pretraining_commit=pretraining_commit,
        gpu_runtime_path=gpu_runtime_path,
        gpu_runtime_receipt=gpu_runtime_receipt,
    )


def validate_scout_config(name: str, config: Mapping[str, Any]) -> None:
    """Validate semantic invariants in addition to the frozen file hash."""

    _require(name in SCOUT_NAMES, f"Unknown scout config: {name}")
    data = config.get("data")
    model = config.get("model")
    training = config.get("training")
    output = config.get("output")
    _require(all(isinstance(value, dict) for value in (data, model, training, output)),
             f"{name} config sections are malformed.")
    expected_data = {
        "data_version": "v5.3",
        "feature_set": "medium",
        "target_col": XERXES_TARGET,
        "era_col": ERA_COLUMN,
        "id_col": ID_COLUMN,
        "benchmark_model": BENCHMARK_COLUMN,
        "full_data_path": "v5.3/downsampled_full.parquet",
        "benchmark_data_path": "v5.3/downsampled_full_benchmark_models.parquet",
        "require_benchmark_coverage": True,
        "embargo_eras": 13,
    }
    _exact_equal(data, expected_data, f"{name} data config")
    _exact_equal(model.get("type"), "LGBMRegressor", f"{name} model type")
    _exact_equal(model.get("x_groups"), ["features", "era", "benchmark_models"],
                 f"{name} x_groups")
    _require(not model.get("target_transform"), f"{name} must train directly on Xerxes20.")
    _require(not model.get("prediction_transform"), f"{name} may not transform OOF predictions.")
    params = model.get("params")
    _require(isinstance(params, dict), f"{name} model params are malformed.")
    for key, expected in {
        "learning_rate": 0.003,
        "colsample_bytree": 0.1,
        "min_data_in_leaf": 10_000,
        "device_type": "gpu",
        "random_state": 1337,
    }.items():
        _exact_equal(params.get(key), expected, f"{name} model.{key}")
    _exact_equal(params.get("max_depth"), SCOUT_DEPTHS[name], f"{name} max_depth")
    _exact_equal(params.get("num_leaves"), (2 ** SCOUT_DEPTHS[name]) - 1,
                 f"{name} num_leaves")
    _exact_equal(params.get("n_estimators"), 2_000 if name == "r1_trees2k" else 6_000,
                 f"{name} n_estimators")
    _exact_equal(training.get("max_train_samples"), 500_000, f"{name} row cap")
    _exact_equal(training.get("sample_seed"), 1337, f"{name} sample seed")
    _exact_equal(
        training.get("cv"),
        {
            "enabled": True,
            "n_splits": 5,
            "embargo": 13,
            "mode": "expanding",
            "min_train_size": 0,
        },
        f"{name} CV config",
    )
    _exact_equal(output.get("results_name"), name, f"{name} result name")


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
    first_holdout_era: str,
) -> ExpectedCohort:
    full = full.copy()
    full[ERA_COLUMN] = full[ERA_COLUMN].astype(str)
    all_eras = sorted(full[ERA_COLUMN].unique().tolist(), key=int)
    splits = era_cv_splits(
        all_eras,
        n_splits=5,
        embargo=embargo,
        mode="expanding",
        min_train_size=0,
    )
    fold_by_era: dict[str, int] = {}
    fold_specs: list[dict[str, int]] = []
    counts = full.groupby(ERA_COLUMN, sort=False, observed=True).size()
    for fold, (train_eras, val_eras) in enumerate(splits):
        if not train_eras or not val_eras:
            continue
        fold_by_era.update({str(era): fold for era in val_eras})
        train_rows = int(counts.reindex([str(era) for era in train_eras]).sum())
        val_rows = int(counts.reindex([str(era) for era in val_eras]).sum())
        fold_specs.append(
            {
                "fold": fold,
                "train_eras": len(train_eras),
                "val_eras": len(val_eras),
                "train_rows": min(train_rows, 500_000),
                "val_rows": val_rows,
            }
        )
    oof = full[full[ERA_COLUMN].isin(fold_by_era)].copy()
    oof[FOLD_COLUMN] = oof[ERA_COLUMN].map(fold_by_era).astype(np.int16)
    eras = tuple(sorted(oof[ERA_COLUMN].unique().tolist(), key=int))
    _require(len(oof) == expected_rows, f"Expected {expected_rows:,} OOF rows; got {len(oof):,}.")
    _require(len(eras) == expected_eras, f"Expected {expected_eras} OOF eras; got {len(eras)}.")
    _exact_equal(eras[0], first_era, "first OOF era")
    _exact_equal(eras[-1], last_era, "last OOF era")
    _exact_equal(eras[calibration_eras - 1], last_calibration_era,
                 "last calibration era")
    _exact_equal(eras[calibration_eras], first_holdout_era, "first holdout era")
    _require(oof[ID_COLUMN].is_unique, "Expected OOF IDs are not unique.")
    return ExpectedCohort(
        frame=oof.reset_index(drop=True),
        full_rows=len(full),
        full_eras=len(all_eras),
        eras=eras,
        folds=tuple(fold_specs),
    )


def build_scout_expected_cohort(protocol: FrozenProtocol) -> ExpectedCohort:
    data_path = protocol.repo_root / "numerai/v5.3/downsampled_full.parquet"
    benchmark_path = (
        protocol.repo_root / "numerai/v5.3/downsampled_full_benchmark_models.parquet"
    )
    data = pd.read_parquet(
        data_path, columns=[ID_COLUMN, ERA_COLUMN, XERXES_TARGET, ENDER_TARGET]
    )
    benchmark = pd.read_parquet(
        benchmark_path, columns=[ID_COLUMN, ERA_COLUMN, BENCHMARK_COLUMN]
    )
    _require(data[ID_COLUMN].notna().all() and data[ID_COLUMN].is_unique,
             "Scout data IDs are missing or duplicated.")
    _require(benchmark[ID_COLUMN].notna().all() and benchmark[ID_COLUMN].is_unique,
             "Scout benchmark IDs are missing or duplicated.")
    benchmark = benchmark.rename(columns={ERA_COLUMN: "_benchmark_era"})
    try:
        full = benchmark.merge(
            data,
            how="left",
            on=ID_COLUMN,
            validate="one_to_one",
            indicator=True,
            sort=False,
        )
    except pd.errors.MergeError as error:
        raise XerxesEvaluationError("Scout sources are not one-to-one by ID.") from error
    _require((full["_merge"] == "both").all(), "Scout benchmark ID is absent from data.")
    _require(
        np.array_equal(
            full["_benchmark_era"].astype(str).to_numpy(),
            full[ERA_COLUMN].astype(str).to_numpy(),
        ),
        "Scout source eras differ by ID.",
    )
    full = full.drop(columns=["_benchmark_era", "_merge"])
    values = full[[XERXES_TARGET, ENDER_TARGET, BENCHMARK_COLUMN]].to_numpy(
        dtype=np.float64
    )
    _require(np.isfinite(values).all(), "Scout targets or benchmark contain non-finite values.")
    return _derive_expected_oof(
        full,
        embargo=13,
        expected_rows=SCOUT_ROWS,
        expected_eras=SCOUT_ERAS,
        first_era=SCOUT_FIRST_ERA,
        last_era=SCOUT_LAST_ERA,
        calibration_eras=SCOUT_CALIBRATION_ERAS,
        last_calibration_era=SCOUT_LAST_CALIBRATION_ERA,
        first_holdout_era=SCOUT_FIRST_HOLDOUT_ERA,
    )


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
    run: RunPaths,
    result: Mapping[str, Any],
    config: Mapping[str, Any],
    expected: ExpectedCohort,
    *,
    store_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a pipeline result receipt to its config and expected OOF cohort."""

    result_name = str(config["output"]["results_name"])
    _exact_equal(run.result.name, f"{result_name}.json", f"{run.name} result filename")
    _exact_equal(run.predictions.name, f"{result_name}.parquet",
                 f"{run.name} prediction filename")
    _exact_equal(result.get("model"), _expected_model_payload(config), f"{run.name} model")
    preprocessing = config.get("preprocessing", {})
    _exact_equal(
        result.get("preprocessing"),
        {
            "nan_missing_all_twos": preprocessing.get("nan_missing_all_twos", False),
            "missing_value": preprocessing.get("missing_value", 2.0),
        },
        f"{run.name} preprocessing",
    )
    data_config = config["data"]
    data = result.get("data")
    _require(isinstance(data, dict), f"{run.name} result has no data receipt.")
    expected_data = {
        "data_version": "v5.3",
        "feature_set": "medium",
        "target": XERXES_TARGET,
        "full_rows": expected.full_rows,
        "full_eras": expected.full_eras,
        "oof_rows": len(expected.frame),
        "oof_eras": len(expected.eras),
        "embargo_eras": int(data_config["embargo_eras"]),
        "require_benchmark_coverage": True,
        "data_mode": str(config.get("training", {}).get("data_mode", "eager")),
    }
    for key, value in expected_data.items():
        _exact_equal(data.get(key), value, f"{run.name} data.{key}")
    if expected_data["data_mode"] == "eager":
        _exact_equal(data.get("full_data_path"), data_config["full_data_path"],
                     f"{run.name} full data path")
    else:
        _require(store_metadata is not None, f"{run.name} lacks frozen store metadata.")
        diagnostics = data.get("disk_feature_store")
        _require(isinstance(diagnostics, dict),
                 f"{run.name} lacks disk-feature-store diagnostics.")
        exact_store = {
            "generation_id": store_metadata["generation_id"],
            "row_count": store_metadata["row_count"],
            "feature_count": store_metadata["feature_count"],
            "feature_bytes": store_metadata["features"]["size_bytes"],
            "manifest_bytes": store_metadata["manifest"]["size_bytes"],
            "feature_order_sha256": store_metadata["feature_order_sha256"],
            "feature_sha256": store_metadata["features"]["sha256"],
            "manifest_sha256": store_metadata["manifest"]["sha256"],
        }
        for key, value in exact_store.items():
            _exact_equal(diagnostics.get(key), value, f"{run.name} store.{key}")
        _exact_equal(
            _path_leaf(diagnostics.get("manifest_path")),
            store_metadata["manifest"]["filename"],
            f"{run.name} store manifest path",
        )
        _exact_equal(
            _path_leaf(diagnostics.get("feature_path")),
            store_metadata["features"]["filename"],
            f"{run.name} store feature path",
        )

    benchmark = result.get("benchmark")
    _require(isinstance(benchmark, dict), f"{run.name} has no benchmark receipt.")
    _exact_equal(benchmark.get("model"), BENCHMARK_COLUMN, f"{run.name} benchmark model")
    if data_config.get("benchmark_data_path"):
        _exact_equal(benchmark.get("file"), data_config["benchmark_data_path"],
                     f"{run.name} benchmark path")
    elif store_metadata is not None:
        _exact_equal(
            _path_leaf(benchmark.get("file")),
            store_metadata["manifest"]["filename"],
            f"{run.name} benchmark store manifest",
        )

    training = result.get("training")
    _require(isinstance(training, dict), f"{run.name} has no training receipt.")
    _exact_equal(
        training.get("data_sampling"),
        {"max_train_samples": 500_000, "sample_seed": 1337},
        f"{run.name} sampling",
    )
    _exact_equal(training.get("data_mode"), expected_data["data_mode"],
                 f"{run.name} training data mode")
    cv_config = config["training"]["cv"]
    _exact_equal(
        training.get("cv"),
        {
            "enabled": True,
            "n_splits": 5,
            "embargo": int(cv_config["embargo"]),
            "mode": "expanding",
            "min_train_size": 0,
        },
        f"{run.name} training CV",
    )
    cv = result.get("cv")
    _require(isinstance(cv, dict), f"{run.name} has no CV receipt.")
    for key, value in {
        "n_splits": 5,
        "embargo": int(cv_config["embargo"]),
        "mode": "expanding",
        "min_train_size": 0,
        "folds_used": len(expected.folds),
    }.items():
        _exact_equal(cv.get(key), value, f"{run.name} cv.{key}")
    actual_folds = cv.get("folds")
    _require(
        isinstance(actual_folds, list) and len(actual_folds) == len(expected.folds),
        f"{run.name} CV fold receipt is incomplete.",
    )
    for actual, frozen_fold in zip(actual_folds, expected.folds, strict=True):
        _require(isinstance(actual, dict), f"{run.name} CV fold is malformed.")
        for key, value in frozen_fold.items():
            _exact_equal(actual.get(key), value, f"{run.name} fold {frozen_fold['fold']} {key}")
        diagnostics = actual.get("model_diagnostics")
        _require(isinstance(diagnostics, dict),
                 f"{run.name} fold {frozen_fold['fold']} lacks model diagnostics.")
        _exact_equal(
            diagnostics.get("effective_device_type"),
            "gpu",
            f"{run.name} fold effective LightGBM device",
        )
        _exact_equal(
            diagnostics.get("gpu_fallback_used"),
            False,
            f"{run.name} fold GPU fallback receipt",
        )
        if expected_data["data_mode"] == "disk_feature_store":
            _exact_equal(diagnostics.get("data_mode"), "disk_feature_store",
                         f"{run.name} fold data mode")
            _exact_equal(diagnostics.get("disk_train_rows"), frozen_fold["train_rows"],
                         f"{run.name} fold disk train rows")
            _exact_equal(diagnostics.get("disk_validation_rows"), frozen_fold["val_rows"],
                         f"{run.name} fold disk validation rows")
            prediction_batches = diagnostics.get("disk_prediction_batches")
            configured_batch_size = config["model"].get("prediction_batch_size")
            _require(
                isinstance(configured_batch_size, int)
                and not isinstance(configured_batch_size, bool)
                and 0 < configured_batch_size <= DISK_PREDICTION_BATCH_SIZE_MAX,
                f"{run.name} has no bounded prediction_batch_size config.",
            )
            _exact_equal(
                diagnostics.get("disk_prediction_batch_size"),
                configured_batch_size,
                f"{run.name} fold prediction batch-size receipt",
            )
            _exact_equal(
                prediction_batches,
                int(np.ceil(frozen_fold["val_rows"] / configured_batch_size)),
                f"{run.name} fold disk prediction batch count",
            )
            rows_per_epoch = diagnostics.get("disk_rows_per_epoch")
            batches_per_epoch = diagnostics.get("disk_batches_per_epoch")
            _exact_equal(rows_per_epoch, [frozen_fold["train_rows"]],
                         f"{run.name} fold disk rows per fit")
            _require(
                isinstance(batches_per_epoch, list)
                and len(batches_per_epoch) == 1
                and isinstance(batches_per_epoch[0], int)
                and batches_per_epoch[0]
                == int(np.ceil(frozen_fold["train_rows"] / configured_batch_size)),
                f"{run.name} fold training materialization was not bounded.",
            )

    output = result.get("output")
    _require(isinstance(output, dict), f"{run.name} has no output receipt.")
    _exact_equal(_path_leaf(output.get("predictions_file")), run.predictions.name,
                 f"{run.name} output prediction")
    semantics = _expected_semantics(config)
    _exact_equal(output.get("prediction_semantics"), semantics,
                 f"{run.name} prediction semantics")
    metrics = result.get("metrics")
    _require(
        isinstance(metrics, dict)
        and set(metrics) == {"corr", "bmc", "bmc_last_200_eras"},
        f"{run.name} result metrics schema differs.",
    )
    return semantics


def _read_parquet_semantics(path: Path) -> dict[str, Any]:
    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(PREDICTION_SEMANTICS_METADATA_KEY)
    _require(raw is not None, f"{path.name} lacks prediction-semantics metadata.")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise XerxesEvaluationError(
            f"{path.name} prediction semantics are not valid UTF-8 JSON."
        ) from error
    _require(isinstance(value, dict), f"{path.name} prediction semantics are malformed.")
    return value


def validate_prediction_artifact(
    path: Path,
    expected: pd.DataFrame,
    expected_semantics: Mapping[str, Any],
    *,
    stored_target: str = XERXES_TARGET,
) -> np.ndarray:
    """Validate exact ID/era/target/fold alignment and return aligned raw predictions."""

    _require(path.is_file(), f"Prediction artifact is missing: {path}")
    _exact_equal(_read_parquet_semantics(path), expected_semantics,
                 f"{path.name} Parquet semantics")
    columns = [ID_COLUMN, ERA_COLUMN, stored_target, PREDICTION_COLUMN, FOLD_COLUMN]
    _exact_equal(pq.read_schema(path).names, columns, f"{path.name} column schema")
    predictions = pd.read_parquet(path, columns=columns)
    _exact_equal(len(predictions), len(expected), f"{path.name} row count")
    _require(predictions[ID_COLUMN].notna().all() and predictions[ID_COLUMN].is_unique,
             f"{path.name} IDs are missing or duplicated.")
    numeric = predictions[[stored_target, PREDICTION_COLUMN, FOLD_COLUMN]].to_numpy(
        dtype=np.float64
    )
    _require(np.isfinite(numeric).all(), f"{path.name} contains non-finite values.")
    predictions = predictions.rename(
        columns={
            ERA_COLUMN: "_artifact_era",
            stored_target: "_artifact_target",
            FOLD_COLUMN: "_artifact_fold",
        }
    )
    expected_ordered = expected.reset_index(drop=True).copy()
    expected_ordered["_expected_order"] = np.arange(len(expected_ordered), dtype=np.int64)
    try:
        aligned = expected_ordered.merge(
            predictions,
            how="left",
            on=ID_COLUMN,
            sort=False,
            validate="one_to_one",
            indicator=True,
        )
    except pd.errors.MergeError as error:
        raise XerxesEvaluationError(f"{path.name} IDs do not align one-to-one.") from error
    _require((aligned["_merge"] == "both").all(), f"{path.name} is missing expected IDs.")
    aligned = aligned.sort_values("_expected_order", kind="stable")
    _require(
        np.array_equal(
            aligned[ERA_COLUMN].astype(str).to_numpy(),
            aligned["_artifact_era"].astype(str).to_numpy(),
        ),
        f"{path.name} eras differ by ID.",
    )
    _require(
        np.array_equal(
            aligned[stored_target].to_numpy(), aligned["_artifact_target"].to_numpy()
        ),
        f"{path.name} stored {stored_target} differs by ID.",
    )
    _require(
        np.array_equal(
            aligned[FOLD_COLUMN].to_numpy(), aligned["_artifact_fold"].to_numpy()
        ),
        f"{path.name} fold assignments differ by ID.",
    )
    return aligned[PREDICTION_COLUMN].to_numpy(dtype=np.float64, copy=True)


def rank_within_era(raw: Sequence[float], eras: pd.Series) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    _require(len(values) == len(eras) and np.isfinite(values).all(),
             "Cannot rank missing or non-finite predictions.")
    ranked = pd.Series(values).groupby(eras.reset_index(drop=True), sort=False).rank(
        method="average", pct=True
    )
    result = ranked.to_numpy(dtype=np.float64)
    _require(np.isfinite(result).all(), "Ranked predictions contain non-finite values.")
    return result


def symmetric_per_era_similarity(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    reference_column: str,
) -> pd.DataFrame:
    """Symmetric Spearman similarity using average tie ranks on both signals."""

    return hybrid.per_era_rank_similarity(
        frame, signal_columns, reference_column, ERA_COLUMN
    )


def _validated_score_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
    eras: Sequence[str],
    label: str,
) -> pd.DataFrame:
    result = frame.copy()
    result.index = result.index.astype(str)
    _require(set(result.index) == set(eras), f"{label} era coverage differs.")
    result = result.loc[list(eras), list(columns)]
    _require(np.isfinite(result.to_numpy(dtype=np.float64)).all(),
             f"{label} contains non-finite values.")
    return result


def compute_per_era_metrics(
    frame: pd.DataFrame,
    signals: Sequence[str],
    eras: Sequence[str],
    *,
    tabm_reference: str | None = None,
) -> dict[str, pd.DataFrame]:
    signals = list(signals)
    corr = numerai_metrics.per_era_corr(frame, signals, ENDER_TARGET, ERA_COLUMN)
    bmc = numerai_metrics.per_era_bmc(
        frame, signals, BENCHMARK_COLUMN, ENDER_TARGET, ERA_COLUMN
    )
    benchmark_similarity = symmetric_per_era_similarity(
        frame, signals, BENCHMARK_COLUMN
    )
    result = {
        "corr": _validated_score_frame(corr, signals, eras, "Corr"),
        "bmc": _validated_score_frame(bmc, signals, eras, "BMC"),
        "benchmark_similarity": _validated_score_frame(
            benchmark_similarity, signals, eras, "benchmark similarity"
        ),
    }
    if tabm_reference is not None:
        similarity = symmetric_per_era_similarity(frame, signals, tabm_reference)
        result["tabm_similarity"] = _validated_score_frame(
            similarity, signals, eras, "TabM similarity"
        )
    return result


def summarize_signal(
    per_era: Mapping[str, pd.DataFrame], signal: str
) -> dict[str, Any]:
    summary = {
        "era_count": len(per_era["bmc"]),
        "corr": hybrid._score_summary(per_era["corr"][signal]),
        "bmc": hybrid._score_summary(per_era["bmc"][signal]),
        "avg_benchmark_similarity": float(
            per_era["benchmark_similarity"][signal].mean()
        ),
    }
    if "tabm_similarity" in per_era:
        summary["avg_tabm_similarity"] = float(
            per_era["tabm_similarity"][signal].mean()
        )
    _reject_nonfinite(summary, f"{signal} metric summary")
    return summary


def scout_calibration_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "bmc_mean": float(metrics["bmc"]["mean"])
        > SCOUT_CALIBRATION_THRESHOLDS["bmc_mean_min_exclusive"],
        "bmc_sharpe": float(metrics["bmc"]["sharpe"])
        > SCOUT_CALIBRATION_THRESHOLDS["bmc_sharpe_min_exclusive"],
        "bmc_max_drawdown": float(metrics["bmc"]["max_drawdown"])
        < SCOUT_CALIBRATION_THRESHOLDS["bmc_max_drawdown_max_exclusive"],
        "corr_mean": float(metrics["corr"]["mean"])
        > SCOUT_CALIBRATION_THRESHOLDS["corr_mean_min_exclusive"],
        "benchmark_similarity": float(metrics["avg_benchmark_similarity"])
        < SCOUT_CALIBRATION_THRESHOLDS["benchmark_similarity_max_exclusive"],
    }


def select_scout_candidate(
    summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    evaluations: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    for name in SCOUT_NAMES:
        checks = scout_calibration_checks(summaries[name])
        is_eligible = all(checks.values())
        evaluations[name] = {"eligible": is_eligible, "checks": checks}
        if is_eligible:
            eligible.append(name)
    if not eligible:
        return None, evaluations
    selected = sorted(
        eligible,
        key=lambda name: (
            -float(summaries[name]["bmc"]["mean"]),
            float(summaries[name]["bmc"]["max_drawdown"]),
            SCOUT_DEPTHS[name],
            name,
        ),
    )[0]
    return selected, evaluations


def scout_holdout_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "bmc_mean": float(metrics["bmc"]["mean"])
        > SCOUT_HOLDOUT_THRESHOLDS["bmc_mean_min_exclusive"],
        "bmc_sharpe": float(metrics["bmc"]["sharpe"])
        > SCOUT_HOLDOUT_THRESHOLDS["bmc_sharpe_min_exclusive"],
        "bmc_max_drawdown": float(metrics["bmc"]["max_drawdown"])
        < SCOUT_HOLDOUT_THRESHOLDS["bmc_max_drawdown_max_exclusive"],
        "corr_mean": float(metrics["corr"]["mean"])
        > SCOUT_HOLDOUT_THRESHOLDS["corr_mean_min_exclusive"],
    }


def confirmation_promotion_checks(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    provenance_ok: bool,
) -> dict[str, bool]:
    calibration = summaries["confirmation_calibration"]
    holdout = summaries["confirmation_holdout"]
    full = summaries["confirmation_full"]
    return {
        "exact_finite_provenance": bool(provenance_ok),
        "calibration_bmc_mean": hybrid._at_least(
            float(calibration["bmc"]["mean"]),
            CONFIRMATION_THRESHOLDS["calibration_bmc_mean_min_inclusive"],
        ),
        "calibration_bmc_sharpe": float(calibration["bmc"]["sharpe"])
        > CONFIRMATION_THRESHOLDS["calibration_bmc_sharpe_min_exclusive"],
        "calibration_bmc_max_drawdown": float(calibration["bmc"]["max_drawdown"])
        < CONFIRMATION_THRESHOLDS["calibration_bmc_max_drawdown_max_exclusive"],
        "calibration_corr_mean": hybrid._at_least(
            float(calibration["corr"]["mean"]),
            CONFIRMATION_THRESHOLDS["calibration_corr_mean_min_inclusive"],
        ),
        "calibration_benchmark_similarity": float(
            calibration["avg_benchmark_similarity"]
        ) < CONFIRMATION_THRESHOLDS[
            "calibration_benchmark_similarity_max_exclusive"
        ],
        "calibration_tabm_similarity": float(calibration["avg_tabm_similarity"])
        < CONFIRMATION_THRESHOLDS["calibration_tabm_similarity_max_exclusive"],
        "holdout_bmc_mean": float(holdout["bmc"]["mean"])
        > CONFIRMATION_THRESHOLDS["holdout_bmc_mean_min_exclusive"],
        "holdout_bmc_max_drawdown": float(holdout["bmc"]["max_drawdown"])
        < CONFIRMATION_THRESHOLDS["holdout_bmc_max_drawdown_max_exclusive"],
        "holdout_corr_mean": float(holdout["corr"]["mean"])
        > CONFIRMATION_THRESHOLDS["holdout_corr_mean_min_exclusive"],
        "full_bmc_mean": hybrid._at_least(
            float(full["bmc"]["mean"]),
            CONFIRMATION_THRESHOLDS["full_bmc_mean_min_inclusive"],
        ),
        "full_bmc_sharpe": float(full["bmc"]["sharpe"])
        > CONFIRMATION_THRESHOLDS["full_bmc_sharpe_min_exclusive"],
        "full_bmc_max_drawdown": float(full["bmc"]["max_drawdown"])
        < CONFIRMATION_THRESHOLDS["full_bmc_max_drawdown_max_exclusive"],
        "full_corr_mean": hybrid._at_least(
            float(full["corr"]["mean"]),
            CONFIRMATION_THRESHOLDS["full_corr_mean_min_inclusive"],
        ),
    }


def validate_confirmation_config(
    selected: str,
    scout_config: Mapping[str, Any],
    confirmation_config: Mapping[str, Any],
) -> None:
    """Allow only the four confirmation changes predeclared in the gate."""

    base = copy.deepcopy(dict(scout_config))
    actual = copy.deepcopy(dict(confirmation_config))
    _exact_equal(actual.get("model", {}).get("params"), base["model"]["params"],
                 "confirmation LightGBM parameters")
    _exact_equal(actual.get("model", {}).get("x_groups"), base["model"]["x_groups"],
                 "confirmation model inputs")
    batch_size = actual.get("model", {}).get("prediction_batch_size")
    _require(
        isinstance(batch_size, int)
        and not isinstance(batch_size, bool)
        and 0 < batch_size <= DISK_PREDICTION_BATCH_SIZE_MAX,
        "Confirmation must declare a bounded positive model.prediction_batch_size.",
    )
    data = actual.get("data", {})
    for key in (
        "data_version", "feature_set", "target_col", "era_col", "id_col",
        "benchmark_model", "require_benchmark_coverage",
    ):
        _exact_equal(data.get(key), base["data"].get(key), f"confirmation data.{key}")
    _exact_equal(data.get("embargo_eras"), 52, "confirmation data embargo")
    store_value = data.get("disk_feature_store_path", data.get("feature_store_path"))
    _require(isinstance(store_value, str) and "target_xerxes_20_feature_store" in store_value,
             "Confirmation must use the Xerxes20 disk feature store.")
    if data.get("disk_feature_store_path") is not None and data.get("feature_store_path") is not None:
        _exact_equal(
            data["disk_feature_store_path"],
            data["feature_store_path"],
            "confirmation disk-store aliases",
        )
    _require(not data.get("full_data_path") and not data.get("benchmark_data_path"),
             "Confirmation may not use eager full or benchmark paths.")
    training = actual.get("training", {})
    _exact_equal(training.get("max_train_samples"), 500_000, "confirmation row cap")
    _exact_equal(training.get("sample_seed"), 1337, "confirmation sample seed")
    _exact_equal(training.get("data_mode"), "disk_feature_store",
                 "confirmation data mode")
    _exact_equal(
        training.get("cv"),
        {
            "enabled": True,
            "n_splits": 5,
            "embargo": 52,
            "mode": "expanding",
            "min_train_size": 0,
        },
        "confirmation CV",
    )
    result_name = actual.get("output", {}).get("results_name")
    _require(
        isinstance(result_name, str)
        and selected in result_name
        and "confirmation" in result_name,
        "Confirmation result name must identify the frozen scout and confirmation stage.",
    )

    # Normalize only the four changes authorized by the gate, then demand exact
    # equality so an unrelated preprocessing/model/data knob cannot hitchhike.
    normalized_base = copy.deepcopy(base)
    normalized_actual = copy.deepcopy(actual)
    for key in ("full_data_path", "benchmark_data_path", "embargo_eras"):
        normalized_base["data"].pop(key, None)
    for key in (
        "disk_feature_store_path",
        "feature_store_path",
        "full_data_path",
        "benchmark_data_path",
        "embargo_eras",
    ):
        normalized_actual["data"].pop(key, None)
    normalized_base["training"]["cv"].pop("embargo", None)
    normalized_actual["training"]["cv"].pop("embargo", None)
    normalized_actual["training"].pop("data_mode", None)
    normalized_actual["model"].pop("prediction_batch_size", None)
    normalized_base["output"].pop("results_name", None)
    normalized_actual["output"].pop("results_name", None)
    _exact_equal(
        normalized_actual,
        normalized_base,
        "confirmation config outside authorized changes",
    )


def _validate_store(
    metadata_path: Path,
    protocol: FrozenProtocol,
    *,
    verify_feature_payload: bool,
) -> tuple[dict[str, Any], Path, dict[str, str]]:
    metadata = _load_json(metadata_path, "Xerxes store metadata")
    _exact_equal(metadata.get("complete"), True, "Xerxes store completion")
    _exact_equal(metadata.get("target_column"), XERXES_TARGET, "Xerxes store target")
    _exact_equal(metadata.get("benchmark_column"), BENCHMARK_COLUMN,
                 "Xerxes store benchmark")
    _exact_equal(metadata.get("feature_count"), 780, "Xerxes store feature count")
    _exact_equal(metadata.get("feature_columns"), list(protocol.medium_features),
                 "Xerxes store feature order")
    feature_meta = protocol.source_manifest["feature_metadata"]
    _exact_equal(metadata.get("feature_order_sha256"),
                 feature_meta["medium_feature_order_sha256"],
                 "Xerxes store feature-order hash")
    expected_sources = protocol.source_manifest["full_source_fingerprints"]
    actual_sources = metadata.get("source_fingerprints")
    _require(isinstance(actual_sources, list) and len(actual_sources) == 4,
             "Xerxes store source fingerprints are incomplete.")
    for actual, expected in zip(actual_sources, expected_sources, strict=True):
        for key in (
            "size_bytes", "mtime_ns", "num_rows", "num_row_groups",
            "schema_sha256", "footer_sha256",
        ):
            _exact_equal(actual.get(key), expected.get(key), f"Xerxes store source {key}")
        _exact_equal(Path(str(actual.get("path"))).name, Path(expected["path"]).name,
                     "Xerxes store source filename")
    manifest_receipt = metadata.get("manifest")
    feature_receipt = metadata.get("features")
    _require(isinstance(manifest_receipt, dict) and isinstance(feature_receipt, dict),
             "Xerxes store payload receipts are malformed.")
    manifest_path = (metadata_path.parent / str(manifest_receipt.get("filename"))).resolve()
    feature_path = (metadata_path.parent / str(feature_receipt.get("filename"))).resolve()
    _require(manifest_path.parent == metadata_path.parent.resolve()
             and feature_path.parent == metadata_path.parent.resolve(),
             "Xerxes store payload escapes its directory.")
    _exact_equal(manifest_path.stat().st_size, manifest_receipt.get("size_bytes"),
                 "Xerxes store manifest size")
    _exact_equal(_sha256_file(manifest_path), manifest_receipt.get("sha256"),
                 "Xerxes store manifest hash")
    _exact_equal(feature_path.stat().st_size, feature_receipt.get("size_bytes"),
                 "Xerxes store feature size")
    if verify_feature_payload:
        _exact_equal(_sha256_file(feature_path), feature_receipt.get("sha256"),
                     "Xerxes store feature hash")
    hashes = {
        "metadata": _sha256_file(metadata_path),
        "manifest": str(manifest_receipt["sha256"]),
        "features": str(feature_receipt["sha256"]),
    }
    return metadata, manifest_path, hashes


def _resolve_configured_store(repo_root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] == "numerai":
        return (repo_root / candidate).resolve()
    return (repo_root / "numerai" / candidate).resolve()


def build_confirmation_expected_cohort(
    protocol: FrozenProtocol,
    xerxes_metadata_path: Path,
) -> tuple[ExpectedCohort, dict[str, str], dict[str, Any]]:
    metadata, xerxes_manifest_path, hashes = _validate_store(
        xerxes_metadata_path, protocol, verify_feature_payload=True
    )
    ender = pd.read_parquet(
        protocol.ender_manifest_path,
        columns=["row_offset", ID_COLUMN, ERA_COLUMN, ENDER_TARGET, BENCHMARK_COLUMN],
    )
    xerxes = pd.read_parquet(
        xerxes_manifest_path,
        columns=["row_offset", ID_COLUMN, ERA_COLUMN, XERXES_TARGET, BENCHMARK_COLUMN],
    )
    _require(ender[ID_COLUMN].is_unique and xerxes[ID_COLUMN].is_unique,
             "Confirmation store IDs are not unique.")
    xerxes = xerxes.rename(
        columns={
            "row_offset": "_xerxes_row_offset",
            ERA_COLUMN: "_xerxes_era",
            BENCHMARK_COLUMN: "_xerxes_benchmark",
        }
    )
    try:
        full = ender.merge(
            xerxes,
            on=ID_COLUMN,
            how="outer",
            sort=False,
            validate="one_to_one",
            indicator=True,
        )
    except pd.errors.MergeError as error:
        raise XerxesEvaluationError("Ender and Xerxes stores are not one-to-one.") from error
    _require((full["_merge"] == "both").all(),
             "Ender and Xerxes stores do not contain the same finite cohort.")
    _require(
        np.array_equal(full["row_offset"].to_numpy(), full["_xerxes_row_offset"].to_numpy()),
        "Ender and Xerxes row offsets differ.",
    )
    _require(
        np.array_equal(full[ERA_COLUMN].astype(str).to_numpy(),
                       full["_xerxes_era"].astype(str).to_numpy()),
        "Ender and Xerxes eras differ by ID.",
    )
    _require(
        np.array_equal(full[BENCHMARK_COLUMN].to_numpy(),
                       full["_xerxes_benchmark"].to_numpy()),
        "Ender and Xerxes benchmarks differ by ID.",
    )
    full = full.drop(
        columns=["_xerxes_row_offset", "_xerxes_era", "_xerxes_benchmark", "_merge"]
    )
    values = full[[ENDER_TARGET, XERXES_TARGET, BENCHMARK_COLUMN]].to_numpy(
        dtype=np.float64
    )
    _require(np.isfinite(values).all(), "Confirmation sources contain non-finite values.")
    _exact_equal(len(full), int(metadata["row_count"]), "Xerxes store row count")
    expected = _derive_expected_oof(
        full,
        embargo=52,
        expected_rows=CONFIRMATION_ROWS,
        expected_eras=CONFIRMATION_ERAS,
        first_era=CONFIRMATION_FIRST_ERA,
        last_era=CONFIRMATION_LAST_ERA,
        calibration_eras=CONFIRMATION_CALIBRATION_ERAS,
        last_calibration_era=CONFIRMATION_LAST_CALIBRATION_ERA,
        first_holdout_era=CONFIRMATION_FIRST_HOLDOUT_ERA,
    )
    hashes["ender_manifest"] = _sha256_file(protocol.ender_manifest_path)
    return expected, hashes, metadata


def load_frozen_two_seed_residual(
    expected: ExpectedCohort,
    receipt_path: Path,
    seed1337_path: Path,
    seed2027_path: Path,
    ender_manifest_path: Path,
) -> tuple[np.ndarray, dict[str, str]]:
    receipt_hash = _sha256_file(receipt_path)
    _exact_equal(
        receipt_hash,
        TWO_SEED_RECEIPT_SHA256,
        "two-seed stability receipt hash",
    )
    receipt = _load_json(receipt_path, "two-seed stability receipt")
    _exact_equal(receipt.get("experiment"), "ender20_seed_ensemble_stability_v53",
                 "two-seed receipt experiment")
    inputs = receipt.get("inputs")
    _require(isinstance(inputs, dict), "Two-seed receipt inputs are malformed.")
    manifest_receipt = inputs.get("manifest")
    _require(isinstance(manifest_receipt, dict), "Two-seed manifest receipt is missing.")
    _exact_equal(_sha256_file(ender_manifest_path), manifest_receipt.get("sha256"),
                 "two-seed Ender manifest hash")
    semantics_receipt = inputs.get("prediction_semantics")
    _require(isinstance(semantics_receipt, dict), "Two-seed semantics receipt is missing.")
    raw_signals: list[np.ndarray] = []
    hashes: dict[str, str] = {"receipt": receipt_hash}
    for label, path, receipt_key, semantics_key in (
        ("seed1337", seed1337_path, "seed1337_predictions",
         "scale_disk_tabm_k64_train500k"),
        ("seed2027", seed2027_path, "seed2027_predictions",
         "scale_disk_tabm_k64_train500k_seed2027"),
    ):
        artifact_receipt = inputs.get(receipt_key)
        _require(isinstance(artifact_receipt, dict), f"Two-seed {label} receipt is missing.")
        actual_hash = _sha256_file(path)
        _exact_equal(
            actual_hash,
            TWO_SEED_PREDICTION_SHA256[label],
            f"frozen {label} prediction hash",
        )
        _exact_equal(actual_hash, artifact_receipt.get("sha256"), f"{label} prediction hash")
        expected_semantics = semantics_receipt.get(semantics_key)
        _require(isinstance(expected_semantics, dict), f"{label} semantics receipt is missing.")
        raw_signals.append(
            validate_prediction_artifact(
                path,
                expected.frame,
                expected_semantics,
                stored_target=ENDER_TARGET,
            )
        )
        hashes[label] = actual_hash
    eras = expected.frame[ERA_COLUMN]
    ranked_a = rank_within_era(raw_signals[0], eras)
    ranked_b = rank_within_era(raw_signals[1], eras)
    residual = rank_within_era(0.5 * (ranked_a + ranked_b), eras)
    return residual, hashes


def _segment_metrics(
    frame: pd.DataFrame,
    signal: str,
    eras: Sequence[str],
    *,
    tabm_reference: str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    subset = frame[frame[ERA_COLUMN].astype(str).isin(eras)].copy()
    per_era = compute_per_era_metrics(
        subset, [signal], eras, tabm_reference=tabm_reference
    )
    return per_era, summarize_signal(per_era, signal)


def _append_per_era_rows(
    rows: list[pd.DataFrame],
    phase: str,
    candidate: str,
    per_era: Mapping[str, pd.DataFrame],
) -> None:
    payload = {
        "phase": phase,
        "era": per_era["bmc"].index.astype(str),
        "candidate": candidate,
        "corr": per_era["corr"][candidate].to_numpy(),
        "bmc": per_era["bmc"][candidate].to_numpy(),
        "benchmark_similarity": per_era["benchmark_similarity"][candidate].to_numpy(),
    }
    if "tabm_similarity" in per_era:
        payload["tabm_similarity"] = per_era["tabm_similarity"][candidate].to_numpy()
    rows.append(pd.DataFrame(payload))


def _summary_rows(summaries: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for segment, candidates in summaries.items():
        for candidate, metrics in candidates.items():
            rows.append(
                {
                    "segment": segment,
                    "candidate": candidate,
                    "era_count": metrics["era_count"],
                    "corr_mean": metrics["corr"]["mean"],
                    "corr_std": metrics["corr"]["std"],
                    "corr_sharpe": metrics["corr"]["sharpe"],
                    "bmc_mean": metrics["bmc"]["mean"],
                    "bmc_std": metrics["bmc"]["std"],
                    "bmc_sharpe": metrics["bmc"]["sharpe"],
                    "bmc_max_drawdown": metrics["bmc"]["max_drawdown"],
                    "avg_benchmark_similarity": metrics["avg_benchmark_similarity"],
                    "avg_tabm_similarity": metrics.get("avg_tabm_similarity"),
                }
            )
    return pd.DataFrame(rows)


def _pending_path(output_dir: Path, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=".xerxes20_", suffix=suffix, dir=output_dir, delete=False
    ) as stream:
        return Path(stream.name)


def _install_content_addressed(pending: Path, final: Path) -> None:
    if final.exists():
        _require(_sha256_file(final) == _sha256_file(pending),
                 f"Existing content-addressed output differs: {final.name}")
        pending.unlink()
        return
    pending.replace(final)


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    pending = _pending_path(path.parent, ".json")
    try:
        with pending.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(path)
    finally:
        if pending.exists():
            pending.unlink()


def _write_outputs(
    output_dir: Path,
    generation_id: str,
    result: dict[str, Any],
    summaries: Mapping[str, Mapping[str, Any]],
    per_era_rows: Sequence[pd.DataFrame],
    repo_root: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"xerxes20_summary-{generation_id}.csv"
    per_era_path = output_dir / f"xerxes20_per_era-{generation_id}.csv"
    result_path = output_dir / f"xerxes20_result-{generation_id}.json"
    summary_pending = _pending_path(output_dir, ".summary.csv")
    per_era_pending = _pending_path(output_dir, ".per_era.csv")
    _summary_rows(summaries).to_csv(summary_pending, index=False)
    pd.concat(list(per_era_rows), ignore_index=True).to_csv(per_era_pending, index=False)
    summary_hash = _sha256_file(summary_pending)
    per_era_hash = _sha256_file(per_era_pending)
    _install_content_addressed(summary_pending, summary_path)
    _install_content_addressed(per_era_pending, per_era_path)
    result["outputs"] = {
        "summary_csv": _relative_path(summary_path, repo_root),
        "summary_csv_sha256": summary_hash,
        "per_era_csv": _relative_path(per_era_path, repo_root),
        "per_era_csv_sha256": per_era_hash,
    }
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    pending_result = _pending_path(output_dir, ".result.json")
    pending_result.write_bytes(payload)
    _install_content_addressed(pending_result, result_path)
    _atomic_replace_bytes(output_dir / "xerxes20_result.json", payload)
    return result_path, summary_path, per_era_path


def evaluate(
    protocol: FrozenProtocol,
    scout_runs: Mapping[str, RunPaths],
    output_dir: Path,
    *,
    confirmation_run: RunPaths | None = None,
    xerxes_store_metadata: Path | None = None,
    two_seed_receipt: Path | None = None,
    seed1337_predictions: Path | None = None,
    seed2027_predictions: Path | None = None,
) -> dict[str, Any]:
    expected = build_scout_expected_cohort(protocol)
    scout_frame = expected.frame.copy()
    input_receipts: dict[str, Any] = {
        "source_manifest": {
            "path": _relative_path(protocol.source_manifest_path, protocol.repo_root),
            "sha256": SOURCE_MANIFEST_SHA256,
        },
        "pre_scoring_commit": PRE_SCORING_COMMIT,
        "pretraining_commit": protocol.pretraining_commit,
        "gpu_runtime": {
            **protocol.gpu_runtime_receipt,
            "path": _relative_path(protocol.gpu_runtime_path, protocol.repo_root),
        },
        "scout_runs": {},
    }
    for name in SCOUT_NAMES:
        run = scout_runs[name]
        _exact_equal(run.name, name, "scout run mapping")
        _exact_equal(run.config.resolve(), protocol.config_paths[name].resolve(),
                     f"{name} config path")
        result = _load_json(run.result, f"{name} result")
        semantics = validate_result_json(run, result, protocol.configs[name], expected)
        raw = validate_prediction_artifact(run.predictions, expected.frame, semantics)
        scout_frame[name] = rank_within_era(raw, expected.frame[ERA_COLUMN])
        input_receipts["scout_runs"][name] = {
            "config_sha256": _sha256_file(run.config),
            "result_sha256": _sha256_file(run.result),
            "predictions_sha256": _sha256_file(run.predictions),
        }

    calibration_eras = expected.eras[:SCOUT_CALIBRATION_ERAS]
    calibration = scout_frame[
        scout_frame[ERA_COLUMN].astype(str).isin(calibration_eras)
    ]
    calibration_per_era = compute_per_era_metrics(
        calibration, SCOUT_NAMES, calibration_eras
    )
    calibration_summaries = {
        name: summarize_signal(calibration_per_era, name) for name in SCOUT_NAMES
    }
    selected, calibration_evaluations = select_scout_candidate(
        calibration_summaries
    )
    summaries: dict[str, dict[str, Any]] = {
        "scout_calibration": calibration_summaries
    }
    per_era_rows: list[pd.DataFrame] = []
    for name in SCOUT_NAMES:
        one = {key: value[[name]] for key, value in calibration_per_era.items()}
        _append_per_era_rows(per_era_rows, "scout_calibration", name, one)

    state = "STOP_NO_SCOUT_CALIBRATION_WINNER"
    holdout_checks: dict[str, bool] = {}
    confirmation_checks: dict[str, bool] = {}
    offline_gate_passed = False
    confirmation_cohort_receipt: dict[str, Any] | None = None
    if selected is not None:
        holdout_eras = expected.eras[SCOUT_CALIBRATION_ERAS:]
        holdout_per_era, holdout_summary = _segment_metrics(
            scout_frame, selected, holdout_eras
        )
        summaries["scout_holdout"] = {selected: holdout_summary}
        _append_per_era_rows(
            per_era_rows, "scout_holdout", selected, holdout_per_era
        )
        holdout_checks = scout_holdout_checks(holdout_summary)
        if all(holdout_checks.values()):
            state = "SCOUT_ACCEPTED_CONFIRMATION_PENDING"
        else:
            state = "STOP_SCOUT_HOLDOUT_FAILED"

    confirmation_inputs = (
        confirmation_run,
        xerxes_store_metadata,
        two_seed_receipt,
        seed1337_predictions,
        seed2027_predictions,
    )
    supplied_confirmation = [value is not None for value in confirmation_inputs]
    _require(
        all(supplied_confirmation) or not any(supplied_confirmation),
        "Confirmation inputs must be supplied together.",
    )
    if all(supplied_confirmation):
        _require(
            selected is not None and all(holdout_checks.values()),
            "Confirmation may run only after the sole scout winner passes locked holdout.",
        )
        assert confirmation_run is not None
        assert xerxes_store_metadata is not None
        assert two_seed_receipt is not None
        assert seed1337_predictions is not None
        assert seed2027_predictions is not None
        namespace = runpy.run_path(str(confirmation_run.config))
        confirmation_config = namespace.get("CONFIG", namespace.get("config"))
        _require(isinstance(confirmation_config, dict),
                 "Confirmation config has no CONFIG mapping.")
        validate_confirmation_config(
            selected, protocol.configs[selected], confirmation_config
        )
        configured_store = confirmation_config["data"].get(
            "disk_feature_store_path",
            confirmation_config["data"].get("feature_store_path"),
        )
        _require(
            _resolve_configured_store(protocol.repo_root, configured_store)
            == xerxes_store_metadata.parent.resolve(),
            "Confirmation config does not point to the validated Xerxes store.",
        )
        confirmation_expected, store_hashes, store_metadata = build_confirmation_expected_cohort(
            protocol, xerxes_store_metadata
        )
        confirmation_result = _load_json(
            confirmation_run.result, "confirmation result"
        )
        semantics = validate_result_json(
            confirmation_run,
            confirmation_result,
            confirmation_config,
            confirmation_expected,
            store_metadata=store_metadata,
        )
        raw = validate_prediction_artifact(
            confirmation_run.predictions,
            confirmation_expected.frame,
            semantics,
        )
        confirmation_frame = confirmation_expected.frame.copy()
        confirmation_cohort_receipt = {
            "rows": len(confirmation_expected.frame),
            "eras": len(confirmation_expected.eras),
            "first_era": confirmation_expected.eras[0],
            "last_era": confirmation_expected.eras[-1],
            "calibration_eras": CONFIRMATION_CALIBRATION_ERAS,
            "holdout_eras": CONFIRMATION_HOLDOUT_ERAS,
            "first_holdout_era": CONFIRMATION_FIRST_HOLDOUT_ERA,
        }
        confirmation_frame[selected] = rank_within_era(
            raw, confirmation_frame[ERA_COLUMN]
        )
        tabm, tabm_hashes = load_frozen_two_seed_residual(
            confirmation_expected,
            two_seed_receipt,
            seed1337_predictions,
            seed2027_predictions,
            protocol.ender_manifest_path,
        )
        confirmation_frame["two_seed_tabm_residual"] = tabm
        segment_eras = {
            "confirmation_calibration": confirmation_expected.eras[
                :CONFIRMATION_CALIBRATION_ERAS
            ],
            "confirmation_holdout": confirmation_expected.eras[
                CONFIRMATION_CALIBRATION_ERAS:
            ],
            "confirmation_full": confirmation_expected.eras,
        }
        for segment, eras in segment_eras.items():
            metrics, summary = _segment_metrics(
                confirmation_frame,
                selected,
                eras,
                tabm_reference="two_seed_tabm_residual",
            )
            summaries[segment] = {selected: summary}
            _append_per_era_rows(per_era_rows, segment, selected, metrics)
        confirmation_checks = confirmation_promotion_checks(
            {
                segment: candidates[selected]
                for segment, candidates in summaries.items()
                if segment.startswith("confirmation_")
            },
            provenance_ok=True,
        )
        offline_gate_passed = all(confirmation_checks.values())
        state = (
            "OFFLINE_GATE_PASSED_LOCAL_PACKAGING_PERMITTED"
            if offline_gate_passed
            else "NOT_PROMOTION_ELIGIBLE_STOP_FAMILY"
        )
        input_receipts["confirmation"] = {
            "name": confirmation_run.name,
            "config_sha256": _sha256_file(confirmation_run.config),
            "result_sha256": _sha256_file(confirmation_run.result),
            "predictions_sha256": _sha256_file(confirmation_run.predictions),
            "store": store_hashes,
            "two_seed_tabm": tabm_hashes,
        }

    evaluator_hash = _sha256_file(Path(__file__).resolve())
    generation_payload = {
        "schema_version": 1,
        "inputs": input_receipts,
        "evaluator_sha256": evaluator_hash,
    }
    generation_id = hashlib.sha256(_canonical_json(generation_payload)).hexdigest()[:20]
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "as_of_date": AS_OF_DATE,
        "generation_id": generation_id,
        "state": state,
        "selected_scout": selected,
        "offline_gate_passed": offline_gate_passed,
        "promotion_eligible": False,
        "deployment_boundary": (
            "Offline passage permits local packaging only; upload, model assignment, "
            "submission, and staking remain unauthorized."
        ),
        "stop_rule": (
            "A failed calibration, locked scout holdout, or confirmation stops this "
            "family without replacement or post-result tuning."
        ),
        "inputs": input_receipts,
        "evaluator": {
            "path": _relative_path(Path(__file__).resolve(), protocol.repo_root),
            "sha256": evaluator_hash,
        },
        "cohorts": {
            **({"confirmation": confirmation_cohort_receipt}
               if confirmation_cohort_receipt is not None else {}),
            "scout": {
                "rows": len(expected.frame),
                "eras": len(expected.eras),
                "first_era": expected.eras[0],
                "last_era": expected.eras[-1],
                "calibration_eras": SCOUT_CALIBRATION_ERAS,
                "holdout_eras": SCOUT_HOLDOUT_ERAS,
            }
        },
        "thresholds": {
            "scout_calibration": SCOUT_CALIBRATION_THRESHOLDS,
            "scout_holdout": SCOUT_HOLDOUT_THRESHOLDS,
            "confirmation": CONFIRMATION_THRESHOLDS,
        },
        "scout_calibration_candidates": calibration_evaluations,
        "scout_holdout_checks": holdout_checks,
        "confirmation_checks": confirmation_checks,
        "summaries": summaries,
    }
    result_path, _, _ = _write_outputs(
        output_dir.resolve(),
        generation_id,
        result,
        summaries,
        per_era_rows,
        protocol.repo_root,
    )
    print(
        json.dumps(
            {
                "result": str(result_path),
                "state": state,
                "selected_scout": selected,
                "offline_gate_passed": offline_gate_passed,
            },
            sort_keys=True,
        )
    )
    return result


def _default_scout_runs(experiment_dir: Path) -> dict[str, RunPaths]:
    return {
        name: RunPaths(
            name=name,
            config=experiment_dir / "configs" / f"{name}.py",
            result=experiment_dir / "results" / f"{name}.json",
            predictions=experiment_dir / "predictions" / f"{name}.parquet",
        )
        for name in SCOUT_NAMES
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _repo_root()
    experiment_dir = root / "numerai/agents/experiments" / EXPERIMENT_NAME
    architecture_dir = root / "numerai/agents/experiments/ender20_nn_architecture_v53"
    seed_dir = root / "numerai/agents/experiments/ender20_seed_ensemble_stability_v53"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path,
                        default=experiment_dir / "source_manifest.json")
    parser.add_argument(
        "--pretraining-commit",
        required=True,
        help=(
            "Full commit ID recorded after training/modeling implementation was "
            "frozen and before any Xerxes training began."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=experiment_dir / "results")
    parser.add_argument("--confirmation-config", type=Path)
    parser.add_argument("--confirmation-result", type=Path)
    parser.add_argument("--confirmation-predictions", type=Path)
    parser.add_argument("--xerxes-store-metadata", type=Path,
                        default=root / "numerai/v5.3/target_xerxes_20_feature_store/metadata.json")
    parser.add_argument("--two-seed-receipt", type=Path,
                        default=seed_dir / "results/two_seed_stability_result.json")
    parser.add_argument("--seed1337-predictions", type=Path,
                        default=architecture_dir / "predictions/scale_disk_tabm_k64_train500k.parquet")
    parser.add_argument("--seed2027-predictions", type=Path,
                        default=architecture_dir / "predictions/scale_disk_tabm_k64_train500k_seed2027.parquet")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = verify_frozen_protocol(
        args.source_manifest.resolve(),
        pretraining_commit=args.pretraining_commit,
    )
    confirmation_values = (
        args.confirmation_config,
        args.confirmation_result,
        args.confirmation_predictions,
    )
    _require(all(value is None for value in confirmation_values)
             or all(value is not None for value in confirmation_values),
             "All three confirmation artifact paths are required together.")
    confirmation_run = None
    if args.confirmation_config is not None:
        namespace = runpy.run_path(str(args.confirmation_config))
        config = namespace.get("CONFIG", namespace.get("config"))
        _require(isinstance(config, dict), "Confirmation config has no CONFIG mapping.")
        name = str(config.get("output", {}).get("results_name", ""))
        confirmation_run = RunPaths(
            name=name,
            config=args.confirmation_config.resolve(),
            result=args.confirmation_result.resolve(),
            predictions=args.confirmation_predictions.resolve(),
        )
    result = evaluate(
        protocol,
        _default_scout_runs(protocol.experiment_dir),
        args.output_dir,
        confirmation_run=confirmation_run,
        xerxes_store_metadata=(args.xerxes_store_metadata.resolve()
                               if confirmation_run else None),
        two_seed_receipt=(args.two_seed_receipt.resolve() if confirmation_run else None),
        seed1337_predictions=(args.seed1337_predictions.resolve()
                              if confirmation_run else None),
        seed2027_predictions=(args.seed2027_predictions.resolve()
                              if confirmation_run else None),
    )
    return 0 if result["state"] in {
        "SCOUT_ACCEPTED_CONFIRMATION_PENDING",
        "OFFLINE_GATE_PASSED_LOCAL_PACKAGING_PERMITTED",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
