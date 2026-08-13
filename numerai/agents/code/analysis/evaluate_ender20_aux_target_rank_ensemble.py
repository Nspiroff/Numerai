"""Evaluate the frozen Ender20 auxiliary-target rank-ensemble protocol.

This module is deliberately training-free.  It validates sealed pipeline
artifacts and exposes write-once stages for component sealing, calibration,
locked scout evaluation, and the two confirmation gates.  It never packages,
uploads, assigns, submits, or stakes a model.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import stat
import subprocess
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from agents.code.analysis import evaluate_ender20_hybrid_stability as hybrid
from agents.code.analysis import evaluate_xerxes20_lgbm_challenger as xerxes
from agents.code.data.build_full_datasets import (
    feature_order_sha256,
    parquet_source_fingerprint,
)
from agents.code.metrics import numerai_metrics
from agents.code.modeling.utils.numerai_cv import era_cv_splits
from agents.code.modeling.utils.disk_feature_store import _ReadOnlyFileLease
from agents.code.modeling.utils.pipeline import (
    PREDICTION_SEMANTICS_METADATA_KEY,
    _require_frozen_python_runtime,
    build_prediction_semantics,
)


EXPERIMENT_NAME = "ender20_aux_target_rank_ensemble_v53"
AS_OF_DATE = "2026-08-03"
PRE_SCORING_COMMIT = "ef4ee304d6088f10d27e4d49a80d67ec925dbbf3"
CONFIRMATION_STORE_INVENTORY_PATH = (
    "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53/"
    "confirmation_store_inventory.json"
)
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
SCOUT_RUN_ORDER = SCOUT_NEW_COMPONENTS
CONFIRMATION_RUN_ORDER = ALL_COMPONENTS
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
SCOUT_CALIBRATION_ERA_SEQUENCE = tuple(
    f"{value:04d}" for value in range(373, 1026, 4)
)
SCOUT_LOCKED_ERA_SEQUENCE = tuple(
    f"{value:04d}" for value in range(1029, 1226, 4)
)

CONFIRMATION_ROWS = 5_112_039
CONFIRMATION_ERAS = 855
CONFIRMATION_FIRST_ERA = "0371"
CONFIRMATION_LAST_ERA = "1225"
CONFIRMATION_CALIBRATION_ERAS = 655
CONFIRMATION_LOCKED_ERAS = 200
CONFIRMATION_LAST_CALIBRATION_ERA = "1025"
CONFIRMATION_FIRST_LOCKED_ERA = "1026"
CONFIRMATION_CALIBRATION_ERA_SEQUENCE = tuple(
    f"{value:04d}" for value in range(371, 1026)
)
CONFIRMATION_LOCKED_ERA_SEQUENCE = tuple(
    f"{value:04d}" for value in range(1026, 1226)
)
CONFIRMATION_FULL_ERA_SEQUENCE = tuple(
    f"{value:04d}" for value in range(371, 1226)
)

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
CONFIRMATION_CONFIG_HELPER_PATHS = (
    "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53/configs/base_d8.py",
)
CONFIRMATION_LOADER_PATHS = (
    "numerai/agents/code/analysis/evaluate_ender20_aux_target_rank_ensemble.py",
    "numerai/agents/code/modeling/utils/cli.py",
    "numerai/agents/code/modeling/utils/disk_feature_store.py",
    "numerai/agents/code/modeling/utils/pipeline.py",
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
    authority_file_receipts: dict[str, Any] | None = None


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


def _require_exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is malformed.")
    _exact_equal(set(value), expected, f"{label} keys")
    return value


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, label: str) -> str:
    _require(_is_lower_hex(value, 64), f"{label} must be a lowercase 64-character SHA-256 digest.")
    assert isinstance(value, str)
    return value


def _lexical_absolute(path: Path) -> Path:
    """Make a path absolute without following its final or ancestor links."""

    _require(
        ".." not in path.parts,
        f"Lexical path may not contain parent traversal: {path}",
    )
    return Path(os.path.abspath(path))


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


def _lexical_repo_path(repo_root: Path, relative: str | Path) -> Path:
    """Resolve `.` components without following a destination link or junction."""

    value = Path(relative)
    _require(not value.is_absolute(), f"Frozen path must be repo-relative: {relative}")
    _require(
        ".." not in value.parts,
        f"Frozen path may not contain parent traversal: {relative}",
    )
    candidate = Path(os.path.abspath(repo_root / value))
    root = Path(os.path.abspath(repo_root))
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


def _lexical_relative_path(path: Path, repo_root: Path) -> str:
    try:
        return Path(os.path.abspath(path)).relative_to(
            Path(os.path.abspath(repo_root))
        ).as_posix()
    except ValueError as error:
        raise EnderEnsembleEvaluationError(
            f"Artifact escapes repository root: {path}"
        ) from error


def _require_absent_destination(path: Path, label: str) -> None:
    _require(not os.path.lexists(path), f"{label} already exists: {path}")


def _require_regular_unlinked_file(path: Path, label: str) -> None:
    """Reject symlinks, reparse points, non-files, and hard-linked files."""

    _require(not path.is_symlink(), f"{label} may not be a symbolic link.")
    try:
        file_stat = path.lstat()
    except OSError as error:
        raise EnderEnsembleEvaluationError(f"{label} cannot be inspected: {path}") from error
    attributes = getattr(file_stat, "st_file_attributes", 0)
    _require(
        not bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)),
        f"{label} may not be a reparse point.",
    )
    _require(stat.S_ISREG(file_stat.st_mode), f"{label} is not a regular file.")
    _exact_equal(file_stat.st_nlink, 1, f"{label} hard-link count")


def _require_regular_directory(path: Path, label: str) -> None:
    """Reject canonical-directory aliases such as symlinks and junctions."""

    _require(not path.is_symlink(), f"{label} may not be a symbolic link.")
    try:
        directory_stat = path.lstat()
    except OSError as error:
        raise EnderEnsembleEvaluationError(f"{label} cannot be inspected: {path}") from error
    attributes = getattr(directory_stat, "st_file_attributes", 0)
    _require(
        not bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)),
        f"{label} may not be a reparse point.",
    )
    _require(stat.S_ISDIR(directory_stat.st_mode), f"{label} is not a directory.")


def _require_lexical_directory_chain(
    repo_root: Path,
    directory: Path,
    label: str,
    *,
    allow_missing_leaf: bool = False,
    create_leaf: bool = False,
) -> None:
    """Validate every lexical directory from the repository root to one leaf.

    Only the final directory may be absent.  A caller may either leave that
    direct leaf absent for a preflight check or create exactly that one leaf
    for a just-in-time output claim.  Parent creation is intentionally never
    recursive so a missing or aliased ancestor fails closed.
    """

    root = Path(os.path.abspath(repo_root))
    candidate = Path(os.path.abspath(directory))
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise EnderEnsembleEvaluationError(
            f"{label} escapes the repository: {directory}"
        ) from error
    _require_regular_directory(root, f"{label} repository root")
    current = root
    for part in relative.parts:
        current = current / part
        if not os.path.lexists(current):
            _require(
                current == candidate and (allow_missing_leaf or create_leaf),
                f"{label} has a missing intermediate directory: {current}",
            )
            if create_leaf:
                try:
                    current.mkdir()
                except FileExistsError:
                    pass
                except OSError as error:
                    raise EnderEnsembleEvaluationError(
                        f"{label} direct output directory could not be created: {current}"
                    ) from error
                _require_regular_directory(current, label)
            return
        _require_regular_directory(current, label)


def _prepare_output_destination_parent(
    protocol: FrozenProtocol,
    path: Path,
    label: str,
    *,
    create_direct_parent: bool,
) -> None:
    """Validate an output's full lexical parent chain without following links."""

    _lexical_relative_path(path, protocol.repo_root)
    _require_lexical_directory_chain(
        protocol.repo_root,
        path.parent,
        f"{label} parent",
        allow_missing_leaf=not create_direct_parent,
        create_leaf=create_direct_parent,
    )


def _require_regular_output_file(
    protocol: FrozenProtocol,
    path: Path,
    label: str,
) -> None:
    """Require a post-run output to be a unique regular file on a safe chain."""

    _prepare_output_destination_parent(
        protocol,
        path,
        label,
        create_direct_parent=False,
    )
    _require_regular_unlinked_file(path, label)


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
    """Convenience wrapper for claim-then-finalize receipt primitives."""

    claim_path = _claim_receipt_prefix(output_dir, prefix)
    return _write_claimed_content_addressed_receipt(
        output_dir,
        prefix,
        claim_path,
        receipt,
    )


def _validate_receipt_prefix(prefix: str) -> None:
    _require(
        bool(prefix)
        and all(character.isalnum() or character in "-_" for character in prefix),
        "Receipt prefix is unsafe.",
    )


def _claim_payload(prefix: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "prefix": prefix,
        "state": "CLAIMED",
    }


def _claim_receipt_prefix(receipt_dir: Path, prefix: str) -> Path:
    """Exclusively claim a receipt prefix before any protected access.

    The claim is deliberately never removed or overwritten.  If the process
    stops before finalization, the incomplete claim therefore remains a
    durable, fail-closed record and the stage cannot be retried.
    """

    _validate_receipt_prefix(prefix)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    _require_regular_directory(receipt_dir, "receipt directory")
    receipt_dir = Path(os.path.abspath(receipt_dir))
    _require(
        not any(receipt_dir.glob(f"{prefix}-*.json")),
        f"A receipt already exists for immutable prefix: {prefix}",
    )
    _require(
        not os.path.lexists(receipt_dir / f".{prefix}.finalized.json"),
        f"A receipt already exists for immutable prefix: {prefix}",
    )
    claim_path = receipt_dir / f".{prefix}.claimed.json"
    payload = _receipt_bytes(_claim_payload(prefix))
    try:
        with claim_path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise EnderEnsembleEvaluationError(
            f"A receipt already exists for immutable prefix: {prefix}"
        ) from error
    _require_regular_unlinked_file(claim_path, f"{prefix} receipt claim")
    _exact_equal(
        _load_json(claim_path, f"{prefix} receipt claim"),
        _claim_payload(prefix),
        f"{prefix} receipt claim",
    )
    return claim_path


def _finalization_payload(
    prefix: str,
    claim_path: Path,
    claim_sha256: str,
    receipt_path: Path,
    receipt_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "prefix": prefix,
        "state": "FINALIZED",
        "claim": {
            "path": claim_path.name,
            "sha256": claim_sha256,
        },
        "receipt": {
            "path": receipt_path.name,
            "sha256": receipt_sha256,
        },
    }


def _write_claimed_content_addressed_receipt(
    receipt_dir: Path,
    prefix: str,
    claim_path: Path,
    receipt: Mapping[str, Any],
) -> Path:
    """Finalize an existing exclusive claim with one immutable receipt."""

    _validate_receipt_prefix(prefix)
    _require_regular_directory(receipt_dir, "receipt directory")
    receipt_dir = Path(os.path.abspath(receipt_dir))
    claim_path = Path(os.path.abspath(claim_path))
    expected_claim_path = receipt_dir / f".{prefix}.claimed.json"
    _exact_equal(claim_path, expected_claim_path, f"{prefix} claim path")
    _require_regular_unlinked_file(claim_path, f"{prefix} receipt claim")
    _exact_equal(
        _load_json(claim_path, f"{prefix} receipt claim"),
        _claim_payload(prefix),
        f"{prefix} receipt claim",
    )
    finalization_path = receipt_dir / f".{prefix}.finalized.json"
    _require(
        not os.path.lexists(finalization_path),
        f"A receipt already exists for immutable prefix: {prefix}",
    )
    _require(
        not any(receipt_dir.glob(f"{prefix}-*.json")),
        f"A receipt already exists for immutable prefix: {prefix}",
    )

    payload = _receipt_bytes(receipt)
    digest = hashlib.sha256(payload).hexdigest()
    receipt_path = receipt_dir / f"{prefix}-{digest}.json"
    try:
        with receipt_path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise EnderEnsembleEvaluationError(
            f"Receipt output already exists: {receipt_path}"
        ) from error
    _require_regular_unlinked_file(receipt_path, f"{prefix} receipt")
    _exact_equal(
        _sha256_file(receipt_path),
        digest,
        f"{prefix} receipt content address",
    )

    claim_digest = _sha256_file(claim_path)
    finalization = _finalization_payload(
        prefix,
        claim_path,
        claim_digest,
        receipt_path,
        digest,
    )
    try:
        with finalization_path.open("xb") as stream:
            stream.write(_receipt_bytes(finalization))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise EnderEnsembleEvaluationError(
            f"A receipt already exists for immutable prefix: {prefix}"
        ) from error
    _require_regular_unlinked_file(
        finalization_path,
        f"{prefix} receipt finalization",
    )
    _exact_equal(
        _load_json(finalization_path, f"{prefix} receipt finalization"),
        finalization,
        f"{prefix} receipt finalization",
    )
    return receipt_path


def _validate_finalized_receipt_claim(
    path: Path,
    expected_sha256: str,
    *,
    receipt_dir: Path,
    expected_prefix: str,
) -> None:
    """Bind a receipt to its canonical parent, exact prefix, and final claim."""

    _validate_receipt_prefix(expected_prefix)
    _require_regular_directory(receipt_dir, "receipt directory")
    receipt_dir = Path(os.path.abspath(receipt_dir))
    path = Path(os.path.abspath(path))
    _require_regular_unlinked_file(path, f"{expected_prefix} receipt")
    _exact_equal(path.parent, receipt_dir, f"{expected_prefix} receipt parent")
    _exact_equal(
        path.name,
        f"{expected_prefix}-{expected_sha256}.json",
        f"{expected_prefix} receipt filename",
    )
    matching = sorted(receipt_dir.glob(f"{expected_prefix}-*.json"))
    _exact_equal(matching, [path], f"{expected_prefix} canonical receipt set")

    claim_path = receipt_dir / f".{expected_prefix}.claimed.json"
    _require_regular_unlinked_file(claim_path, f"{expected_prefix} receipt claim")
    _exact_equal(
        _load_json(claim_path, f"{expected_prefix} receipt claim"),
        _claim_payload(expected_prefix),
        f"{expected_prefix} receipt claim",
    )
    finalization_path = receipt_dir / f".{expected_prefix}.finalized.json"
    _require_regular_unlinked_file(
        finalization_path,
        f"{expected_prefix} receipt finalization",
    )
    expected_finalization = _finalization_payload(
        expected_prefix,
        claim_path,
        _sha256_file(claim_path),
        path,
        expected_sha256,
    )
    _exact_equal(
        _load_json(finalization_path, f"{expected_prefix} receipt finalization"),
        expected_finalization,
        f"{expected_prefix} receipt finalization",
    )


def _load_bound_receipt(
    path: Path,
    expected_sha256: str,
    *,
    expected_stage: str,
    receipt_dir: Path | None = None,
    expected_prefix: str | None = None,
) -> dict[str, Any]:
    _require(
        len(expected_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_sha256),
        "Receipt hash must be a full lowercase SHA-256 digest.",
    )
    _require(
        (receipt_dir is None) == (expected_prefix is None),
        "Receipt parent and prefix bindings must be supplied together.",
    )
    path = Path(os.path.abspath(path))
    _require_regular_unlinked_file(path, f"{expected_stage} receipt")
    if receipt_dir is not None and expected_prefix is not None:
        _validate_finalized_receipt_claim(
            path,
            expected_sha256,
            receipt_dir=receipt_dir,
            expected_prefix=expected_prefix,
        )
    actual = _sha256_file(path)
    _exact_equal(actual, expected_sha256, f"{expected_stage} receipt hash")
    if receipt_dir is None:
        _require(path.name.endswith(f"-{actual}.json"), "Receipt filename is not content-addressed.")
    receipt = _load_json(path, f"{expected_stage} receipt")
    _exact_equal(receipt.get("experiment"), EXPERIMENT_NAME, "receipt experiment")
    _exact_equal(receipt.get("stage"), expected_stage, "receipt stage")
    return receipt


_RECEIPT_ENVELOPE_KEYS = {
    "schema_version",
    "experiment",
    "stage",
    "state",
    "passed",
    "protocol",
}
_FORBIDDEN_PRE_SCORING_KEYS = {
    "metrics",
    "corr",
    "bmc",
    "summary",
    "per_era",
    "calibration",
    "locked",
    "full",
    "candidates",
    "selected_formula",
}


def _validate_binding_schema(value: Any, label: str) -> Mapping[str, Any]:
    binding = _require_exact_keys(value, {"path", "sha256"}, label)
    _require(isinstance(binding.get("path"), str) and bool(binding["path"]), f"{label} path is malformed.")
    _require_sha256(binding.get("sha256"), f"{label} hash")
    return binding


def _validate_canonical_receipt_binding_schema(
    value: Any,
    prefix: str,
    label: str,
) -> Mapping[str, Any]:
    binding = _validate_binding_schema(value, label)
    expected_path = (
        f"numerai/agents/experiments/{EXPERIMENT_NAME}/receipts/"
        f"{prefix}-{binding['sha256']}.json"
    )
    _exact_equal(binding.get("path"), expected_path, f"{label} canonical path")
    return binding


def _validate_prior_seal_binding_schema(
    value: Any,
    label: str,
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    binding = _require_exact_keys(value, {"component", "path", "sha256"}, label)
    _require(binding.get("component") in ALL_COMPONENTS, f"{label} component is malformed.")
    _require(isinstance(binding.get("path"), str) and bool(binding["path"]), f"{label} path is malformed.")
    _require_sha256(binding.get("sha256"), f"{label} hash")
    return binding


def _validate_file_receipt_schema(value: Any, label: str) -> Mapping[str, Any]:
    receipt = _require_exact_keys(value, {"path", "sha256", "size_bytes"}, label)
    _require(isinstance(receipt.get("path"), str) and bool(receipt["path"]), f"{label} path is malformed.")
    _require_sha256(receipt.get("sha256"), f"{label} hash")
    _require(type(receipt.get("size_bytes")) is int and receipt["size_bytes"] >= 0, f"{label} size is malformed.")
    return receipt


def _validate_checkpointed_file_receipt_schema(
    value: Any,
    label: str,
) -> Mapping[str, Any]:
    receipt = _require_exact_keys(
        value,
        {"path", "sha256", "size_bytes", "checkpoint_commit", "git_blob_id"},
        label,
    )
    _require(isinstance(receipt.get("path"), str) and bool(receipt["path"]), f"{label} path is malformed.")
    _require_sha256(receipt.get("sha256"), f"{label} hash")
    _require(type(receipt.get("size_bytes")) is int and receipt["size_bytes"] >= 0, f"{label} size is malformed.")
    _require(_is_lower_hex(receipt.get("checkpoint_commit"), 40), f"{label} checkpoint is malformed.")
    _require(_is_lower_hex(receipt.get("git_blob_id"), 40), f"{label} Git blob is malformed.")
    return receipt


def _validate_selected_formula_schema(value: Any, label: str) -> str:
    formula = _require_exact_keys(value, {"name", "weights"}, label)
    name = formula.get("name")
    _require(name in CANDIDATE_NAMES, f"{label} selected an unknown formula.")
    assert isinstance(name, str)
    weights = _require_exact_keys(formula.get("weights"), {"tyler", "core"}, f"{label} weights")
    _exact_equal(dict(weights), BLEND_WEIGHTS[name], f"{label} weights")
    return name


def _reject_forbidden_pre_scoring_keys(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        forbidden = set(value).intersection(_FORBIDDEN_PRE_SCORING_KEYS)
        _require(not forbidden, f"{label} contains forbidden pre-scoring keys: {sorted(forbidden)}")
        for key, child in value.items():
            _reject_forbidden_pre_scoring_keys(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_pre_scoring_keys(child, f"{label}[{index}]")


def _validate_destination_schema(value: Any, label: str) -> None:
    destinations = _require_exact_keys(value, {"result", "predictions"}, label)
    for name in ("result", "predictions"):
        destination = _require_exact_keys(destinations.get(name), {"path", "absent"}, f"{label}.{name}")
        _require(isinstance(destination.get("path"), str) and bool(destination["path"]), f"{label}.{name} path is malformed.")
        _exact_equal(destination.get("absent"), True, f"{label}.{name} absence")


def _validate_artifact_schema(value: Any, label: str) -> None:
    artifact = _require_exact_keys(
        value,
        {"component", "target", "config", "result", "predictions"},
        label,
    )
    _require(artifact.get("component") in ALL_COMPONENTS, f"{label} component is malformed.")
    _require(isinstance(artifact.get("target"), str), f"{label} target is malformed.")
    for name in ("config", "result", "predictions"):
        _validate_file_receipt_schema(artifact.get(name), f"{label}.{name}")


def _validate_confirmation_store_schema(value: Any, label: str) -> None:
    store = _require_exact_keys(
        value,
        {
            "generation_id",
            "row_count",
            "feature_count",
            "feature_order_sha256",
            "target_column",
            "metadata",
            "manifest",
            "features",
        },
        label,
    )
    _require(
        _is_lower_hex(store.get("generation_id"), 32),
        f"{label} generation ID is malformed.",
    )
    for name in ("row_count", "feature_count"):
        _require(
            type(store.get(name)) is int and store[name] > 0,
            f"{label}.{name} is malformed.",
        )
    _require_sha256(store.get("feature_order_sha256"), f"{label} feature-order hash")
    _require(
        isinstance(store.get("target_column"), str) and bool(store["target_column"]),
        f"{label} target column is malformed.",
    )
    for name in ("metadata", "manifest", "features"):
        _validate_file_receipt_schema(store.get(name), f"{label}.{name}")


def _validate_confirmation_output_destinations_schema(value: Any, label: str) -> None:
    destinations = _require_exact_keys(value, set(ALL_COMPONENTS), label)
    for component, component_value in destinations.items():
        item = _require_exact_keys(
            component_value,
            {
                "results_path",
                "predictions_path",
                "results_absent_at_checkpoint",
                "predictions_absent_at_checkpoint",
            },
            f"{label}.{component}",
        )
        for name in ("results_path", "predictions_path"):
            _require(
                isinstance(item.get(name), str) and bool(item[name]),
                f"{label}.{component}.{name} is malformed.",
            )
        for name in (
            "results_absent_at_checkpoint",
            "predictions_absent_at_checkpoint",
        ):
            _exact_equal(
                item.get(name),
                True,
                f"{label}.{component}.{name}",
            )


def _validate_cohort_schema(value: Any, label: str) -> None:
    cohort = _require_exact_keys(
        value,
        {"rows", "eras", "first_era", "last_era", "full_rows", "full_eras", "folds"},
        label,
    )
    for name in ("rows", "eras", "full_rows", "full_eras"):
        _require(type(cohort.get(name)) is int and cohort[name] > 0, f"{label}.{name} is malformed.")
    for name in ("first_era", "last_era"):
        _require(isinstance(cohort.get(name), str) and bool(cohort[name]), f"{label}.{name} is malformed.")
    folds = cohort.get("folds")
    _require(isinstance(folds, list) and bool(folds), f"{label}.folds is malformed.")
    for index, fold_value in enumerate(folds):
        fold = _require_exact_keys(
            fold_value,
            {"fold", "train_eras", "val_eras", "train_rows", "val_rows"},
            f"{label}.folds[{index}]",
        )
        for name in ("fold", "train_eras", "val_eras", "train_rows", "val_rows"):
            _require(type(fold.get(name)) is int and fold[name] >= 0, f"{label}.folds[{index}].{name} is malformed.")


def _validate_tabm_receipts_schema(value: Any, label: str) -> None:
    seeds = _require_exact_keys(value, {"seed1337", "seed2027"}, label)
    for seed, seed_value in seeds.items():
        files = _require_exact_keys(
            seed_value, {"config", "result", "predictions"}, f"{label}.{seed}"
        )
        for name, digest in files.items():
            _require_sha256(digest, f"{label}.{seed}.{name}")


def _validate_normalized_seal_inputs_schema(
    value: Any,
    *,
    confirmation: bool,
    label: str,
) -> Mapping[str, Any]:
    order = CONFIRMATION_RUN_ORDER if confirmation else SCOUT_RUN_ORDER
    seals = _require_exact_keys(value, set(order), label)
    seen_paths: set[str] = set()
    seen_digests: set[str] = set()
    expected_pretraining: Mapping[str, Any] | None = None
    for index, component in enumerate(order):
        expected_keys = {
            "path", "sha256", "prior_finalized_seal",
            "pre_run_absence_receipt", "run_consumption_claim",
            "run_completion_receipt", "artifact",
        }
        if confirmation:
            expected_keys.add("confirmation_pretraining_receipt")
        seal = _require_exact_keys(seals.get(component), expected_keys, f"{label}.{component}")
        digest = _require_sha256(seal.get("sha256"), f"{label}.{component} hash")
        prefix = (
            f"confirmation-seal-{component}"
            if confirmation
            else f"scout-seal-{component}"
        )
        expected_path = (
            f"numerai/agents/experiments/{EXPERIMENT_NAME}/receipts/"
            f"{prefix}-{digest}.json"
        )
        _exact_equal(
            seal.get("path"),
            expected_path,
            f"{label}.{component} canonical seal path",
        )
        _require(expected_path not in seen_paths, f"{label} reuses a seal path.")
        _require(digest not in seen_digests, f"{label} reuses a seal digest.")
        seen_paths.add(expected_path)
        seen_digests.add(digest)
        prior = _validate_prior_seal_binding_schema(
            seal.get("prior_finalized_seal"), f"{label}.{component} predecessor"
        )
        if index == 0:
            _exact_equal(prior, None, f"{label}.{component} first predecessor")
        else:
            _require(prior is not None, f"{label}.{component} predecessor is missing.")
            assert prior is not None
            previous = order[index - 1]
            previous_seal = seals[previous]
            _exact_equal(
                dict(prior),
                {
                    "component": previous,
                    "path": previous_seal["path"],
                    "sha256": previous_seal["sha256"],
                },
                f"{label}.{component} contiguous predecessor",
            )
        pre_run_prefix = (
            f"confirmation-pre-run-{component}"
            if confirmation
            else f"scout-pre-run-{component}"
        )
        _validate_canonical_receipt_binding_schema(
            seal.get("pre_run_absence_receipt"),
            pre_run_prefix,
            f"{label}.{component} pre-run",
        )
        _validate_file_receipt_schema(
            seal.get("run_consumption_claim"),
            f"{label}.{component} run consumption claim",
        )
        family = "confirmation" if confirmation else "scout"
        _exact_equal(
            seal["run_consumption_claim"]["path"],
            (
                f"numerai/agents/experiments/{EXPERIMENT_NAME}/receipts/"
                f".{family}-train-{component}.consumed.json"
            ),
            f"{label}.{component} canonical run consumption claim path",
        )
        _validate_file_receipt_schema(
            seal.get("run_completion_receipt"),
            f"{label}.{component} run completion receipt",
        )
        _exact_equal(
            seal["run_completion_receipt"]["path"],
            (
                f"numerai/agents/experiments/{EXPERIMENT_NAME}/receipts/"
                f"{family}-train-{component}-completion-"
                f"{seal['run_completion_receipt']['sha256']}.json"
            ),
            f"{label}.{component} canonical run completion receipt path",
        )
        if confirmation:
            pretraining = _validate_canonical_receipt_binding_schema(
                seal.get("confirmation_pretraining_receipt"),
                "confirmation-pretraining",
                f"{label}.{component} pretraining",
            )
            if expected_pretraining is None:
                expected_pretraining = pretraining
            else:
                _exact_equal(
                    dict(pretraining),
                    dict(expected_pretraining),
                    f"{label}.{component} shared pretraining",
                )
        artifact = seal.get("artifact")
        _validate_artifact_schema(artifact, f"{label}.{component} artifact")
        _exact_equal(
            artifact.get("component"), component, f"{label}.{component} artifact component"
        )
        _exact_equal(
            artifact.get("target"),
            COMPONENT_TARGETS[component],
            f"{label}.{component} artifact target",
        )
    _reject_forbidden_pre_scoring_keys(seals, label)
    return seals


def _validate_calibration_inputs_schema(
    value: Any,
    *,
    confirmation: bool,
    label: str,
) -> None:
    if confirmation:
        inputs = _require_exact_keys(
            value,
            {
                "scout_locked_receipt", "confirmation_pretraining_receipt",
                "confirmation_seal_receipts", "tabm_two_seed_residual", "cohort",
            },
            label,
        )
        _validate_canonical_receipt_binding_schema(
            inputs.get("scout_locked_receipt"), "locked", f"{label}.Scout locked"
        )
        pretraining = _validate_canonical_receipt_binding_schema(
            inputs.get("confirmation_pretraining_receipt"),
            "confirmation-pretraining",
            f"{label}.pretraining",
        )
        seals = _validate_normalized_seal_inputs_schema(
            inputs.get("confirmation_seal_receipts"),
            confirmation=True,
            label=f"{label}.seals",
        )
        for component, seal in seals.items():
            _exact_equal(
                seal.get("confirmation_pretraining_receipt"),
                pretraining,
                f"{label}.{component} pretraining binding",
            )
    else:
        inputs = _require_exact_keys(
            value,
            {"seal_receipts", "reused_xerxes", "tabm_two_seed_residual", "cohort"},
            label,
        )
        _validate_normalized_seal_inputs_schema(
            inputs.get("seal_receipts"), confirmation=False, label=f"{label}.seals"
        )
        _validate_artifact_schema(inputs.get("reused_xerxes"), f"{label}.reused Xerxes")
    _validate_tabm_receipts_schema(
        inputs.get("tabm_two_seed_residual"), f"{label}.TabM"
    )
    _validate_cohort_schema(inputs.get("cohort"), f"{label}.cohort")


def _validate_signal_summary_schema(value: Any, label: str) -> None:
    summary = _require_exact_keys(
        value,
        {
            "era_count",
            "corr",
            "bmc",
            "avg_ender20_similarity",
            "avg_ender60_similarity",
            "avg_tabm_similarity",
        },
        label,
    )
    _require(type(summary.get("era_count")) is int and summary["era_count"] > 0, f"{label}.era_count is malformed.")
    for score_name in ("corr", "bmc"):
        score = _require_exact_keys(
            summary.get(score_name),
            {"mean", "std", "std_valid", "sharpe", "max_drawdown"},
            f"{label}.{score_name}",
        )
        _require(type(score.get("std_valid")) is bool, f"{label}.{score_name}.std_valid is malformed.")
        for key in ("mean", "std", "max_drawdown"):
            number = score.get(key)
            _require(
                isinstance(number, (int, float))
                and not isinstance(number, bool)
                and math.isfinite(float(number)),
                f"{label}.{score_name}.{key} is malformed.",
            )
        sharpe = score.get("sharpe")
        _require(
            sharpe is None
            or (
                isinstance(sharpe, (int, float))
                and not isinstance(sharpe, bool)
                and math.isfinite(float(sharpe))
            ),
            f"{label}.{score_name}.sharpe is malformed.",
        )
        _exact_equal(
            sharpe is not None,
            bool(score["std_valid"]),
            f"{label}.{score_name} Sharpe/std validity",
        )
    for key in (
        "avg_ender20_similarity",
        "avg_ender60_similarity",
        "avg_tabm_similarity",
    ):
        number = summary.get(key)
        _require(
            isinstance(number, (int, float))
            and not isinstance(number, bool)
            and math.isfinite(float(number)),
            f"{label}.{key} is malformed.",
        )
    _reject_nonfinite(summary, label)


def _validate_era_rows_schema(
    value: Any,
    *,
    expected_sequence: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    expected = tuple(expected_sequence)
    _require(
        isinstance(value, list) and len(value) == len(expected),
        f"{label} row count is malformed.",
    )
    eras: list[str] = []
    for index, row in enumerate(value):
        _require(
            isinstance(row, list) and len(row) == 2,
            f"{label}[{index}] is malformed.",
        )
        era, score = row
        _require(
            isinstance(era, str) and len(era) == 4 and era.isdigit(),
            f"{label}[{index}] era is malformed.",
        )
        _require(
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(float(score)),
            f"{label}[{index}] score is malformed.",
        )
        if eras:
            _require(eras[-1] < era, f"{label} eras are not strictly increasing.")
        eras.append(era)
    actual = tuple(eras)
    _exact_equal(actual, expected, f"{label} authorized era sequence")
    return actual


def _validate_per_era_schema(
    value: Any,
    selected: str,
    label: str,
    *,
    expected_sequence: Sequence[str],
) -> None:
    metrics = _require_exact_keys(
        value,
        {"corr", "bmc", "ender20_similarity", "ender60_similarity", "tabm_similarity"},
        label,
    )
    authorized_sequence = tuple(expected_sequence)
    observed_sequence: tuple[str, ...] | None = None
    for metric_name, metric_value in metrics.items():
        signals = _require_exact_keys(metric_value, {selected}, f"{label}.{metric_name}")
        sequence = _validate_era_rows_schema(
            signals[selected],
            expected_sequence=authorized_sequence,
            label=f"{label}.{metric_name}.{selected}",
        )
        if observed_sequence is None:
            observed_sequence = sequence
        else:
            _exact_equal(
                sequence,
                observed_sequence,
                f"{label}.{metric_name} era sequence",
            )


def _per_era_frames_from_receipt(
    value: Mapping[str, Any],
    signals: Sequence[str],
    expected_sequence: Sequence[str],
) -> dict[str, pd.DataFrame]:
    index = list(expected_sequence)
    frames: dict[str, pd.DataFrame] = {}
    for metric_name in (
        "corr",
        "bmc",
        "ender20_similarity",
        "ender60_similarity",
        "tabm_similarity",
    ):
        metric = value[metric_name]
        frames[metric_name] = pd.DataFrame(
            {
                signal: [float(row[1]) for row in metric[signal]]
                for signal in signals
            },
            index=index,
        )
    return frames


def _validate_scout_calibration_schema(
    value: Any,
    label: str,
    expected_sequence: Sequence[str],
) -> str | None:
    calibration = _require_exact_keys(
        value,
        {"rows", "eras", "first_era", "last_era", "candidates", "per_era"},
        label,
    )
    for name in ("rows", "eras"):
        _require(type(calibration.get(name)) is int and calibration[name] > 0, f"{label}.{name} is malformed.")
    for name in ("first_era", "last_era"):
        _require(
            isinstance(calibration.get(name), str) and bool(calibration[name]),
            f"{label}.{name} is malformed.",
        )
    expected_eras = tuple(expected_sequence)
    _exact_equal(calibration["eras"], len(expected_eras), f"{label} era count")
    _exact_equal(calibration["first_era"], expected_eras[0], f"{label} first era")
    _exact_equal(calibration["last_era"], expected_eras[-1], f"{label} last era")
    candidates = _require_exact_keys(
        calibration.get("candidates"), set(CANDIDATE_NAMES), f"{label}.candidates"
    )
    check_keys = {
        "bmc_mean", "bmc_sharpe", "bmc_max_drawdown", "corr_mean",
        "ender20_similarity", "ender60_similarity", "tabm_similarity",
    }
    for candidate, candidate_value in candidates.items():
        item = _require_exact_keys(
            candidate_value,
            {"summary", "eligible", "checks", "in_tie_set"},
            f"{label}.candidates.{candidate}",
        )
        _validate_signal_summary_schema(
            item.get("summary"), f"{label}.candidates.{candidate}.summary"
        )
        _exact_equal(
            item["summary"]["era_count"],
            calibration["eras"],
            f"{label}.{candidate} summary era count",
        )
        checks = _require_exact_keys(
            item.get("checks"), check_keys,
            f"{label}.candidates.{candidate}.checks",
        )
        _require(all(type(result) is bool for result in checks.values()), f"{label}.{candidate} checks are malformed.")
        _exact_equal(item.get("eligible"), all(checks.values()), f"{label}.{candidate} eligibility")
        _require(type(item.get("in_tie_set")) is bool, f"{label}.{candidate} tie flag is malformed.")
    metrics = _require_exact_keys(
        calibration.get("per_era"),
        {"corr", "bmc", "ender20_similarity", "ender60_similarity", "tabm_similarity"},
        f"{label}.per_era",
    )
    expected_sequence: tuple[str, ...] | None = None
    for metric_name, metric_value in metrics.items():
        signals = _require_exact_keys(
            metric_value, set(CANDIDATE_NAMES), f"{label}.per_era.{metric_name}"
        )
        for candidate, rows in signals.items():
            sequence = _validate_era_rows_schema(
                rows,
                expected_sequence=expected_eras,
                label=f"{label}.per_era.{metric_name}.{candidate}",
            )
            if expected_sequence is None:
                expected_sequence = sequence
            else:
                _exact_equal(
                    sequence,
                    expected_sequence,
                    f"{label}.{metric_name}.{candidate} era sequence",
                )
    frames = _per_era_frames_from_receipt(
        metrics,
        CANDIDATE_NAMES,
        expected_eras,
    )
    summaries = {
        candidate: summarize_signal(frames, candidate)
        for candidate in CANDIDATE_NAMES
    }
    selected, evaluations = select_scout_candidate(summaries)
    for candidate in CANDIDATE_NAMES:
        item = candidates[candidate]
        _exact_equal(
            item["summary"],
            summaries[candidate],
            f"{label}.{candidate} summary derivation",
        )
        for key in ("eligible", "checks", "in_tie_set"):
            _exact_equal(
                item[key],
                evaluations[candidate][key],
                f"{label}.{candidate} {key} derivation",
            )
    return selected


def _validate_confirmation_calibration_schema(
    value: Any,
    selected: str,
    label: str,
    expected_sequence: Sequence[str],
) -> bool:
    calibration = _require_exact_keys(
        value,
        {"rows", "eras", "first_era", "last_era", "summary", "checks", "per_era"},
        label,
    )
    for name in ("rows", "eras"):
        _require(type(calibration.get(name)) is int and calibration[name] > 0, f"{label}.{name} is malformed.")
    for name in ("first_era", "last_era"):
        _require(
            isinstance(calibration.get(name), str) and bool(calibration[name]),
            f"{label}.{name} is malformed.",
        )
    expected_eras = tuple(expected_sequence)
    _exact_equal(calibration["eras"], len(expected_eras), f"{label} era count")
    _exact_equal(calibration["first_era"], expected_eras[0], f"{label} first era")
    _exact_equal(calibration["last_era"], expected_eras[-1], f"{label} last era")
    _validate_signal_summary_schema(calibration.get("summary"), f"{label}.summary")
    _exact_equal(
        calibration["summary"]["era_count"],
        calibration["eras"],
        f"{label} summary era count",
    )
    checks = _require_exact_keys(
        calibration.get("checks"),
        {
            "calibration_bmc_mean", "calibration_bmc_sharpe",
            "calibration_bmc_max_drawdown", "calibration_corr_mean",
            "calibration_ender20_similarity", "calibration_ender60_similarity",
            "calibration_tabm_similarity",
        },
        f"{label}.checks",
    )
    _require(all(type(result) is bool for result in checks.values()), f"{label}.checks are malformed.")
    _validate_per_era_schema(
        calibration.get("per_era"),
        selected,
        f"{label}.per_era",
        expected_sequence=expected_eras,
    )
    frames = _per_era_frames_from_receipt(
        calibration["per_era"],
        [selected],
        expected_eras,
    )
    expected_summary = summarize_signal(frames, selected)
    _exact_equal(
        calibration["summary"],
        expected_summary,
        f"{label} summary derivation",
    )
    expected_checks = confirmation_calibration_checks(expected_summary)
    _exact_equal(
        calibration["checks"],
        expected_checks,
        f"{label} checks derivation",
    )
    return all(expected_checks.values())


def _validate_scoring_block_schema(
    value: Any,
    selected: str,
    label: str,
    expected_check_keys: set[str],
    expected_sequence: Sequence[str],
    check_function: Any,
) -> bool:
    block = _require_exact_keys(
        value,
        {"rows", "eras", "first_era", "last_era", "summary", "checks", "per_era"},
        label,
    )
    for name in ("rows", "eras"):
        _require(type(block.get(name)) is int and block[name] > 0, f"{label}.{name} is malformed.")
    for name in ("first_era", "last_era"):
        _require(
            isinstance(block.get(name), str) and bool(block[name]),
            f"{label}.{name} is malformed.",
        )
    expected_eras = tuple(expected_sequence)
    _exact_equal(block["eras"], len(expected_eras), f"{label} era count")
    _exact_equal(block["first_era"], expected_eras[0], f"{label} first era")
    _exact_equal(block["last_era"], expected_eras[-1], f"{label} last era")
    _validate_signal_summary_schema(block.get("summary"), f"{label}.summary")
    _exact_equal(
        block["summary"]["era_count"],
        block["eras"],
        f"{label} summary era count",
    )
    checks = _require_exact_keys(block.get("checks"), expected_check_keys, f"{label}.checks")
    _require(all(type(result) is bool for result in checks.values()), f"{label}.checks values are malformed.")
    _validate_per_era_schema(
        block.get("per_era"),
        selected,
        f"{label}.per_era",
        expected_sequence=expected_eras,
    )
    frames = _per_era_frames_from_receipt(
        block["per_era"],
        [selected],
        expected_eras,
    )
    expected_summary = summarize_signal(frames, selected)
    _exact_equal(block["summary"], expected_summary, f"{label} summary derivation")
    expected_checks = check_function(expected_summary)
    _exact_equal(set(expected_checks), expected_check_keys, f"{label} derived check keys")
    _exact_equal(dict(checks), expected_checks, f"{label} checks derivation")
    return all(expected_checks.values())


def _validate_stage_receipt_schema(receipt: Mapping[str, Any], stage: str) -> None:
    """Close every production receipt schema before any downstream access."""

    stage_keys: dict[str, set[str]] = {
        "create-confirmation-store-inventory": {
            "scout_locked_receipt", "selected_formula", "inventory"
        },
        "claim-scout-component-run": {"component", "prior_finalized_seal", "destinations"},
        "seal-scout-component": {
            "component", "prior_finalized_seal", "pre_run_absence_receipt",
            "run_consumption_claim", "run_completion_receipt", "cohort",
            "artifact", "gpu_folds_verified",
        },
        "calibrate": {"inputs", "selected_formula", "calibration"},
        "locked": {"input_receipt", "selected_formula", "locked"},
        "confirmation-pretraining": {
            "checkpoint", "scout_locked_receipt", "configs", "config_helpers",
            "loader", "store_inventory", "canonical_store", "input_layout",
            "output_destinations",
        },
        "claim-confirmation-component-run": {
            "component", "confirmation_pretraining_receipt",
            "prior_finalized_seal", "destinations",
        },
        "seal-confirmation-component": {
            "component", "confirmation_pretraining_receipt",
            "prior_finalized_seal", "pre_run_absence_receipt",
            "run_consumption_claim", "run_completion_receipt", "cohort",
            "artifact", "gpu_folds_verified",
        },
        "confirmation-calibrate": {"inputs", "selected_formula", "calibration"},
        "confirmation-locked": {
            "input_receipt", "confirmation_pretraining_receipt",
            "confirmation_seal_receipts", "selected_formula", "locked",
        },
    }
    _require(stage in stage_keys, f"No closed receipt schema exists for stage: {stage}")
    expected = _RECEIPT_ENVELOPE_KEYS | stage_keys[stage]
    if stage == "confirmation-locked" and "full" in receipt:
        expected = expected | {"full"}
    _require_exact_keys(receipt, expected, f"{stage} receipt")
    _exact_equal(receipt.get("schema_version"), 1, f"{stage} receipt schema")
    _exact_equal(receipt.get("experiment"), EXPERIMENT_NAME, f"{stage} receipt experiment")
    _exact_equal(receipt.get("stage"), stage, f"{stage} receipt stage")
    _require(type(receipt.get("passed")) is bool, f"{stage} receipt passed flag is malformed.")

    if stage in {
        "claim-scout-component-run", "seal-scout-component",
        "confirmation-pretraining", "claim-confirmation-component-run",
        "seal-confirmation-component",
    }:
        payload = {key: value for key, value in receipt.items() if key not in _RECEIPT_ENVELOPE_KEYS}
        _reject_forbidden_pre_scoring_keys(payload, f"{stage} receipt")

    if stage == "create-confirmation-store-inventory":
        _validate_canonical_receipt_binding_schema(
            receipt.get("scout_locked_receipt"), "locked", "inventory Scout receipt"
        )
        _validate_selected_formula_schema(receipt.get("selected_formula"), "inventory formula")
        _validate_file_receipt_schema(receipt.get("inventory"), "inventory file")
    elif stage == "confirmation-pretraining":
        _require(
            _is_lower_hex(receipt.get("checkpoint"), 40),
            "confirmation pretraining checkpoint is malformed.",
        )
        _validate_canonical_receipt_binding_schema(
            receipt.get("scout_locked_receipt"),
            "locked",
            "confirmation pretraining Scout receipt",
        )
        configs = _require_exact_keys(
            receipt.get("configs"), set(ALL_COMPONENTS),
            "confirmation pretraining configs",
        )
        for component, item in configs.items():
            _validate_checkpointed_file_receipt_schema(
                item, f"confirmation pretraining {component} config"
            )
        helpers = receipt.get("config_helpers")
        _require(
            isinstance(helpers, list)
            and len(helpers) == len(CONFIRMATION_CONFIG_HELPER_PATHS),
            "confirmation pretraining config helpers are malformed.",
        )
        for index, item in enumerate(helpers):
            _validate_checkpointed_file_receipt_schema(
                item, f"confirmation pretraining config helper {index}"
            )
        loader = _require_exact_keys(
            receipt.get("loader"), {"checkpoint", "files"},
            "confirmation pretraining loader",
        )
        _require(
            _is_lower_hex(loader.get("checkpoint"), 40),
            "confirmation loader checkpoint is malformed.",
        )
        files = loader.get("files")
        _require(
            isinstance(files, list) and len(files) == len(CONFIRMATION_LOADER_PATHS),
            "confirmation loader files are malformed.",
        )
        for index, item in enumerate(files):
            _validate_checkpointed_file_receipt_schema(
                item, f"confirmation loader file {index}"
            )
        _validate_checkpointed_file_receipt_schema(
            receipt.get("store_inventory"), "confirmation store inventory"
        )
        _validate_confirmation_store_schema(
            receipt.get("canonical_store"), "confirmation canonical store"
        )
        layout = _require_exact_keys(
            receipt.get("input_layout"), {"type", "stores"},
            "confirmation pretraining input layout",
        )
        _exact_equal(
            layout.get("type"), "dedicated_target_stores",
            "confirmation pretraining input layout type",
        )
        stores = _require_exact_keys(
            layout.get("stores"), set(ALL_COMPONENTS),
            "confirmation pretraining stores",
        )
        for component, store in stores.items():
            _validate_confirmation_store_schema(
                store, f"confirmation pretraining store {component}"
            )
        _validate_confirmation_output_destinations_schema(
            receipt.get("output_destinations"),
            "confirmation pretraining output destinations",
        )
    elif stage in {"claim-scout-component-run", "claim-confirmation-component-run"}:
        _validate_prior_seal_binding_schema(
            receipt.get("prior_finalized_seal"), f"{stage} predecessor"
        )
        _validate_destination_schema(receipt.get("destinations"), f"{stage} destinations")
        if stage == "claim-confirmation-component-run":
            _validate_canonical_receipt_binding_schema(
                receipt.get("confirmation_pretraining_receipt"),
                "confirmation-pretraining",
                "confirmation pre-run pretraining receipt",
            )
    elif stage in {"seal-scout-component", "seal-confirmation-component"}:
        _validate_prior_seal_binding_schema(
            receipt.get("prior_finalized_seal"), f"{stage} predecessor"
        )
        _validate_binding_schema(receipt.get("pre_run_absence_receipt"), f"{stage} pre-run receipt")
        _validate_file_receipt_schema(
            receipt.get("run_consumption_claim"),
            f"{stage} run consumption claim",
        )
        family = "confirmation" if stage == "seal-confirmation-component" else "scout"
        component = receipt.get("component")
        _require(
            isinstance(component, str) and bool(component),
            f"{stage} component is malformed.",
        )
        _exact_equal(
            receipt["run_consumption_claim"]["path"],
            (
                f"numerai/agents/experiments/{EXPERIMENT_NAME}/receipts/"
                f".{family}-train-{component}.consumed.json"
            ),
            f"{stage} canonical run consumption claim path",
        )
        _validate_file_receipt_schema(
            receipt.get("run_completion_receipt"),
            f"{stage} run completion receipt",
        )
        _exact_equal(
            receipt["run_completion_receipt"]["path"],
            (
                f"numerai/agents/experiments/{EXPERIMENT_NAME}/receipts/"
                f"{family}-train-{component}-completion-"
                f"{receipt['run_completion_receipt']['sha256']}.json"
            ),
            f"{stage} canonical run completion receipt path",
        )
        _validate_cohort_schema(receipt.get("cohort"), f"{stage} cohort")
        _validate_artifact_schema(receipt.get("artifact"), f"{stage} artifact")
        _require(type(receipt.get("gpu_folds_verified")) is int and receipt["gpu_folds_verified"] > 0, f"{stage} GPU fold count is malformed.")
        if stage == "seal-confirmation-component":
            _validate_canonical_receipt_binding_schema(
                receipt.get("confirmation_pretraining_receipt"),
                "confirmation-pretraining",
                "confirmation seal pretraining receipt",
            )
    elif stage == "calibrate":
        _validate_calibration_inputs_schema(
            receipt.get("inputs"), confirmation=False, label="Scout calibration inputs"
        )
        derived_selected = _validate_scout_calibration_schema(
            receipt.get("calibration"),
            "Scout calibration",
            SCOUT_CALIBRATION_ERA_SEQUENCE,
        )
        derived_passed = derived_selected is not None
        _exact_equal(receipt.get("passed"), derived_passed, "Scout calibration passage")
        _exact_equal(
            receipt.get("state"),
            "PASS" if derived_passed else "STOP_NO_ELIGIBLE_CANDIDATE",
            "Scout calibration state",
        )
        if derived_passed:
            selected = _validate_selected_formula_schema(
                receipt.get("selected_formula"), "Scout calibration formula"
            )
            _exact_equal(selected, derived_selected, "Scout calibration selection")
        else:
            _exact_equal(
                receipt.get("selected_formula"), None,
                "failed Scout calibration formula",
            )
    elif stage == "confirmation-calibrate":
        _validate_calibration_inputs_schema(
            receipt.get("inputs"),
            confirmation=True,
            label="confirmation calibration inputs",
        )
        selected = _validate_selected_formula_schema(
            receipt.get("selected_formula"), "confirmation calibration formula"
        )
        calibration_passed = _validate_confirmation_calibration_schema(
            receipt.get("calibration"),
            selected,
            "confirmation calibration",
            CONFIRMATION_CALIBRATION_ERA_SEQUENCE,
        )
        _exact_equal(
            receipt.get("passed"),
            calibration_passed,
            "confirmation calibration passage",
        )
        _exact_equal(
            receipt.get("state"),
            "PASS" if calibration_passed else "STOP_CONFIRMATION_CALIBRATION_FAILED",
            "confirmation calibration state",
        )
    elif stage == "locked":
        _validate_canonical_receipt_binding_schema(
            receipt.get("input_receipt"), "calibrate", "locked input receipt"
        )
        selected = _validate_selected_formula_schema(receipt.get("selected_formula"), "locked formula")
        block_passed = _validate_scoring_block_schema(
            receipt.get("locked"), selected, "locked block",
            {"bmc_mean", "bmc_sharpe", "bmc_max_drawdown", "corr_mean"},
            SCOUT_LOCKED_ERA_SEQUENCE,
            locked_checks,
        )
        _exact_equal(receipt.get("passed"), block_passed, "locked receipt passage")
        _exact_equal(
            receipt.get("state"),
            "PASS" if block_passed else "STOP_SCOUT_LOCKED_FAILED",
            "locked receipt state",
        )
    elif stage == "confirmation-locked":
        _validate_canonical_receipt_binding_schema(
            receipt.get("input_receipt"),
            "confirmation-calibrate",
            "confirmation locked input",
        )
        pretraining = _validate_canonical_receipt_binding_schema(
            receipt.get("confirmation_pretraining_receipt"),
            "confirmation-pretraining",
            "confirmation locked pretraining",
        )
        seals = _validate_normalized_seal_inputs_schema(
            receipt.get("confirmation_seal_receipts"),
            confirmation=True,
            label="confirmation locked seals",
        )
        for component, seal in seals.items():
            _exact_equal(
                seal.get("confirmation_pretraining_receipt"),
                pretraining,
                f"confirmation locked {component} pretraining binding",
            )
        selected = _validate_selected_formula_schema(receipt.get("selected_formula"), "confirmation locked formula")
        locked_passed = _validate_scoring_block_schema(
            receipt.get("locked"), selected, "confirmation locked block",
            {"locked_bmc_mean", "locked_bmc_sharpe", "locked_bmc_max_drawdown", "locked_corr_mean"},
            CONFIRMATION_LOCKED_ERA_SEQUENCE,
            confirmation_locked_checks,
        )
        state = receipt.get("state")
        if not locked_passed:
            _exact_equal(state, "STOP_CONFIRMATION_LOCKED_FAILED", "failed confirmation locked state")
            _require("full" not in receipt and receipt.get("passed") is False, "Failed confirmation lock may not contain full metrics.")
        else:
            _require(state in {"PASS", "STOP_CONFIRMATION_FULL_FAILED"} and "full" in receipt, "Confirmation full receipt state is inconsistent.")
            full_passed = _validate_scoring_block_schema(
                receipt.get("full"), selected, "confirmation full block",
                {
                    "full_bmc_mean", "full_bmc_sharpe", "full_bmc_max_drawdown",
                    "full_corr_mean", "full_ender20_similarity",
                    "full_ender60_similarity", "full_tabm_similarity",
                },
                CONFIRMATION_FULL_ERA_SEQUENCE,
                confirmation_full_checks,
            )
            _exact_equal(receipt.get("passed"), full_passed, "confirmation full passage")
            _exact_equal(
                state,
                "PASS" if full_passed else "STOP_CONFIRMATION_FULL_FAILED",
                "confirmation full state",
            )


def _parse_seal_bindings(values: Sequence[Sequence[str]] | None) -> dict[str, tuple[Path, str]]:
    bindings: dict[str, tuple[Path, str]] = {}
    for value in values or ():
        _require(len(value) == 3, "Each --seal-receipt requires COMPONENT PATH SHA256.")
        component, path, digest = value
        _require(component in SCOUT_NEW_COMPONENTS, f"Unknown seal component: {component}")
        _require(component not in bindings, f"Duplicate seal component: {component}")
        bindings[component] = (Path(path), digest)
    return bindings


def _parse_confirmation_seal_bindings(
    values: Sequence[Sequence[str]] | None,
) -> dict[str, tuple[Path, str]]:
    bindings: dict[str, tuple[Path, str]] = {}
    for value in values or ():
        _require(
            len(value) == 3,
            "Each --confirmation-seal-receipt requires COMPONENT PATH SHA256.",
        )
        component, path, digest = value
        _require(component in ALL_COMPONENTS, f"Unknown confirmation seal component: {component}")
        _require(component not in bindings, f"Duplicate confirmation seal component: {component}")
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


def _confirmation_config_relative(name: str) -> str:
    _require(name in ALL_COMPONENTS, f"Unknown confirmation config: {name}")
    return (
        f"numerai/agents/experiments/{EXPERIMENT_NAME}/configs/"
        f"confirmation_{name}_d8_t6000.py"
    )


def _confirmation_store_relative(name: str) -> str:
    _require(name in ALL_COMPONENTS, f"Unknown confirmation store: {name}")
    return f"v5.3/{COMPONENT_TARGETS[name]}_feature_store"


def _expected_confirmation_config_source(name: str) -> str:
    """Return the only executable source authorized for one confirmation config."""

    target = COMPONENT_TARGETS[name]
    results_name = f"confirmation_{name}_d8_t6000"
    store = _confirmation_store_relative(name)
    return (
        "from pathlib import Path\n"
        "import runpy\n"
        "\n"
        "\n"
        "MAKE_CONFIG = runpy.run_path(str(Path(__file__).with_name(\"base_d8.py\")))[\n"
        "    \"make_config\"\n"
        "]\n"
        f"CONFIG = MAKE_CONFIG(\"{target}\", \"{results_name}\")\n"
        "CONFIG[\"data\"].pop(\"full_data_path\")\n"
        "CONFIG[\"data\"].pop(\"benchmark_data_path\")\n"
        f"CONFIG[\"data\"][\"disk_feature_store_path\"] = \"{store}\"\n"
        "CONFIG[\"data\"][\"disk_feature_store_inventory_path\"] = "
        f"\"{CONFIRMATION_STORE_INVENTORY_PATH.removeprefix('numerai/')}\"\n"
        "CONFIG[\"data\"][\"embargo_eras\"] = 52\n"
        "CONFIG[\"training\"][\"cv\"][\"embargo\"] = 52\n"
        "CONFIG[\"training\"][\"data_mode\"] = \"disk_feature_store\"\n"
    )


@contextmanager
def _lease_confirmation_config(
    protocol: FrozenProtocol,
    name: str,
    path: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Lease the exact wrapper/helper bytes through config evaluation and use."""

    expected_relative = _confirmation_config_relative(name)
    contract = protocol.source_manifest["confirmation_output_contract"][name]
    expected_contract = {
        "config_path": expected_relative,
        "must_be_absent_before_run": True,
        "predictions_path": (
            f"numerai/agents/experiments/{EXPERIMENT_NAME}/predictions/"
            f"confirmation_{name}_d8_t6000.parquet"
        ),
        "results_name": f"confirmation_{name}_d8_t6000",
        "results_path": (
            f"numerai/agents/experiments/{EXPERIMENT_NAME}/results/"
            f"confirmation_{name}_d8_t6000.json"
        ),
    }
    _exact_equal(dict(contract), expected_contract, f"{name} confirmation output contract")
    expected_path = _lexical_repo_path(protocol.repo_root, expected_relative)
    actual_path = expected_path if path is None else Path(os.path.abspath(path))
    _exact_equal(actual_path, expected_path, f"{name} confirmation config lexical path")
    _require_lexical_directory_chain(
        protocol.repo_root,
        actual_path.parent,
        f"{name} confirmation config",
    )
    _require_regular_unlinked_file(actual_path, f"{name} confirmation config")

    base_relative = CONFIRMATION_CONFIG_HELPER_PATHS[0]
    base_path = _lexical_repo_path(protocol.repo_root, base_relative)
    _exact_equal(
        base_path,
        actual_path.with_name("base_d8.py"),
        f"{name} confirmation base helper path",
    )
    _require_lexical_directory_chain(
        protocol.repo_root,
        base_path.parent,
        f"{name} confirmation base helper",
    )
    _require_regular_unlinked_file(base_path, f"{name} confirmation base helper")

    base_receipt = protocol.source_manifest["experiment_files"][
        "configs/base_d8.py"
    ]
    base_entry = {
        "path": _lexical_relative_path(base_path, protocol.repo_root),
        "sha256": base_receipt["sha256"],
        "size_bytes": base_receipt["size_bytes"],
    }
    with _lease_frozen_manifest_artifacts(
        protocol,
        {"base_d8": base_entry},
        f"{name} confirmation config helper",
    ):
        wrapper_lease = _ReadOnlyFileLease(
            actual_path, f"{name} confirmation config"
        )
        try:
            inspected = os.fstat(wrapper_lease.fileno())
            path_inspected = actual_path.lstat()
            _exact_equal(
                (int(inspected.st_dev), int(inspected.st_ino)),
                (int(path_inspected.st_dev), int(path_inspected.st_ino)),
                f"{name} confirmation config leased identity",
            )
            _exact_equal(
                int(inspected.st_nlink),
                1,
                f"{name} confirmation config link count",
            )
            expected_source = _expected_confirmation_config_source(name).encode(
                "utf-8"
            )
            _exact_equal(
                wrapper_lease.read_bytes(),
                expected_source,
                f"{name} confirmation config source",
            )
            config = _load_config(actual_path, f"{name} confirmation config")
            validate_component_config(name, config, confirmation=True)
            yield config
        finally:
            wrapper_lease.close()


def _load_confirmation_config(
    protocol: FrozenProtocol,
    name: str,
    path: Path | None = None,
) -> dict[str, Any]:
    with _lease_confirmation_config(protocol, name, path) as config:
        return copy.deepcopy(config)


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
        _exact_equal(
            set(data),
            {*common_data, "disk_feature_store_path", "disk_feature_store_inventory_path"},
            f"{name} confirmation data keys",
        )
        _exact_equal(
            data.get("disk_feature_store_inventory_path"),
            CONFIRMATION_STORE_INVENTORY_PATH.removeprefix("numerai/"),
            f"{name} confirmation store inventory path",
        )
        _require(
            "label_sidecar_path" not in data,
            f"{name} confirmation may not use a label sidecar.",
        )
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
        config=_lexical_repo_path(protocol.repo_root, contract["config_path"]),
        result=_lexical_repo_path(protocol.repo_root, contract["results_path"]),
        predictions=_lexical_repo_path(
            protocol.repo_root, contract["predictions_path"]
        ),
    )


def _canonical_receipt_dir(protocol: FrozenProtocol) -> Path:
    canonical = Path(os.path.abspath(protocol.experiment_dir / "receipts"))
    _relative_path(canonical, protocol.repo_root)
    return canonical


def _require_canonical_receipt_dir(
    protocol: FrozenProtocol,
    receipt_dir: Path,
) -> Path:
    canonical = _canonical_receipt_dir(protocol)
    provided = Path(os.path.abspath(receipt_dir))
    _exact_equal(provided, canonical, "canonical receipt directory")
    if os.path.lexists(canonical):
        _require_regular_directory(canonical, "canonical receipt directory")
    return canonical


def _claim_canonical_receipt_prefix(
    protocol: FrozenProtocol,
    receipt_dir: Path,
    prefix: str,
) -> tuple[Path, Path]:
    canonical = _require_canonical_receipt_dir(protocol, receipt_dir)
    return canonical, _claim_receipt_prefix(canonical, prefix)


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
    verify_checkpoint_boundaries(repo_root, pretraining_commit)
    manifest_path = _lexical_absolute(
        source_manifest_path
        if source_manifest_path is not None
        else experiment_dir / "source_manifest.json"
    )
    _exact_equal(
        manifest_path,
        _lexical_absolute(experiment_dir / "source_manifest.json"),
        "canonical source manifest",
    )
    _require_lexical_directory_chain(
        repo_root, manifest_path.parent, "source manifest parent"
    )
    _require_regular_unlinked_file(manifest_path, "source manifest")
    manifest_lease = _ReadOnlyFileLease(manifest_path, "source manifest")
    try:
        manifest_bytes = manifest_lease.read_bytes()
        manifest_stat = os.fstat(manifest_lease.fileno())
        manifest_path_stat = manifest_path.lstat()
        _exact_equal(
            (int(manifest_stat.st_dev), int(manifest_stat.st_ino)),
            (int(manifest_path_stat.st_dev), int(manifest_path_stat.st_ino)),
            "source manifest leased identity",
        )
        _exact_equal(
            hashlib.sha256(manifest_bytes).hexdigest(),
            SOURCE_MANIFEST_SHA256,
            "source manifest hash",
        )
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EnderEnsembleEvaluationError(
                "source manifest is invalid JSON."
            ) from error
        _require(isinstance(manifest, dict), "source manifest must be a JSON object.")
        source_manifest_receipt = {
            "path": _lexical_relative_path(manifest_path, repo_root),
            "sha256": SOURCE_MANIFEST_SHA256,
            "size_bytes": int(manifest_stat.st_size),
        }
    finally:
        manifest_lease.close()

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
    config_entries: dict[str, dict[str, Any]] = {}
    base_relative = "configs/base_d8.py"
    base_receipt = experiment_files[base_relative]
    config_entries["base_d8"] = {
        "path": _lexical_relative_path(
            experiment_dir / base_relative, repo_root
        ),
        "sha256": base_receipt["sha256"],
        "size_bytes": base_receipt["size_bytes"],
    }
    for name in SCOUT_NEW_COMPONENTS:
        stem = f"r1_{name}_d8_t6000"
        relative = f"configs/{stem}.py"
        path = experiment_dir / relative
        receipt = experiment_files[relative]
        config_paths[name] = path
        config_entries[name] = {
            "path": _lexical_relative_path(path, repo_root),
            "sha256": receipt["sha256"],
            "size_bytes": receipt["size_bytes"],
        }

    config_protocol = FrozenProtocol(
        repo_root=repo_root,
        experiment_dir=experiment_dir,
        source_manifest_path=manifest_path,
        source_manifest=manifest,
        scout_configs={},
        scout_config_paths=config_paths,
        medium_features=tuple(medium),
        pretraining_commit=pretraining_commit,
        gpu_runtime_path=repo_root / "__pending_gpu_runtime__",
        gpu_runtime_receipt={},
    )
    configs: dict[str, dict[str, Any]] = {}
    with _lease_frozen_manifest_artifacts(
        config_protocol,
        config_entries,
        "frozen Scout config",
    ):
        for name in SCOUT_NEW_COMPONENTS:
            config = _load_config(config_paths[name], f"{name} scout config")
            validate_component_config(name, config)
            configs[name] = config

    gpu_receipt = manifest.get("gpu_runtime")
    _require(isinstance(gpu_receipt, Mapping), "GPU runtime receipt is malformed.")
    gpu_path = _safe_repo_path(repo_root, str(gpu_receipt["path"]))
    with _lease_frozen_manifest_artifacts(
        protocol=FrozenProtocol(
            repo_root=repo_root,
            experiment_dir=experiment_dir,
            source_manifest_path=manifest_path,
            source_manifest=manifest,
            scout_configs=configs,
            scout_config_paths=config_paths,
            medium_features=tuple(medium),
            pretraining_commit=pretraining_commit,
            gpu_runtime_path=gpu_path,
            gpu_runtime_receipt={},
        ),
        entries={
            "gpu_runtime": {
                key: gpu_receipt[key]
                for key in ("path", "sha256", "size_bytes")
            }
        },
        label="GPU runtime",
    ):
        try:
            gpu_runtime = xerxes.verify_live_gpu_runtime(gpu_path)
        except Exception as error:
            raise EnderEnsembleEvaluationError("GPU runtime validation failed.") from error
    gate_authority = _leased_checkpoint_file_receipt(
        experiment_dir / "gate.md",
        repo_root,
        PRE_SCORING_COMMIT,
        "frozen gate",
    )
    _exact_equal(
        gate_authority,
        {
            "path": _lexical_relative_path(
                experiment_dir / "gate.md", repo_root
            ),
            "sha256": experiment_files["gate.md"]["sha256"],
            "size_bytes": experiment_files["gate.md"]["size_bytes"],
        },
        "frozen gate manifest receipt",
    )
    evaluator_authority = _leased_checkpoint_file_receipt(
        Path(__file__).resolve(),
        repo_root,
        pretraining_commit,
        "frozen evaluator",
    )
    hybrid_authority = _leased_checkpoint_file_receipt(
        Path(hybrid.__file__).resolve(),
        repo_root,
        pretraining_commit,
        "hybrid evaluator",
    )
    xerxes_authority = _leased_checkpoint_file_receipt(
        Path(xerxes.__file__).resolve(),
        repo_root,
        pretraining_commit,
        "Xerxes evaluator",
    )
    _exact_equal(
        xerxes_authority,
        dict(manifest["reused_xerxes_component"]["evaluator"]),
        "Xerxes evaluator manifest receipt",
    )
    authority_file_receipts = {
        "source_manifest": source_manifest_receipt,
        "gate": gate_authority,
        "evaluator": evaluator_authority,
        "imported_evaluators": {
            "hybrid": hybrid_authority,
            "xerxes": xerxes_authority,
        },
        "gpu_runtime": {
            key: gpu_receipt[key] for key in ("path", "sha256", "size_bytes")
        },
    }
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
        authority_file_receipts=authority_file_receipts,
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
    source_entries = {
        relative: {
            "path": relative,
            "sha256": receipt["sha256"],
            "size_bytes": receipt["size_bytes"],
        }
        for relative, receipt in protocol.source_manifest["scout_sources"].items()
    }
    with _lease_frozen_manifest_artifacts(
        protocol,
        source_entries,
        "Scout source",
    ) as source_paths:
        data_path = source_paths["numerai/v5.3/downsampled_full.parquet"]
        benchmark_path = source_paths[
            "numerai/v5.3/downsampled_full_benchmark_models.parquet"
        ]
        data_columns = [
            ID_COLUMN,
            ERA_COLUMN,
            *COMPONENT_TARGETS.values(),
            ENDER_TARGET,
        ]
        benchmark_columns = [
            ID_COLUMN,
            ERA_COLUMN,
            BENCHMARK_ENDER20,
            BENCHMARK_ENDER60,
        ]
        data = pd.read_parquet(
            data_path, columns=list(dict.fromkeys(data_columns))
        )
        benchmark = pd.read_parquet(benchmark_path, columns=benchmark_columns)
        full = _merge_sources_one_to_one(data, benchmark, label="Scout")
        _validate_finite_columns(
            full,
            [
                *COMPONENT_TARGETS.values(),
                ENDER_TARGET,
                BENCHMARK_ENDER20,
                BENCHMARK_ENDER60,
            ],
            "Scout targets and benchmarks",
        )
        coverage = protocol.source_manifest["target_coverage"]
        _exact_equal(
            len(full), coverage["benchmark_covered_rows"], "scout covered rows"
        )
        _exact_equal(
            full[ERA_COLUMN].astype(str).nunique(),
            coverage["benchmark_covered_eras"],
            "scout covered eras",
        )
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
    source_entries = {
        Path(str(receipt["path"])).name: {
            "path": receipt["path"],
            "sha256": receipt["sha256"],
            "size_bytes": receipt["size_bytes"],
        }
        for receipt in protocol.source_manifest["confirmation_sources"]
    }
    with _lease_frozen_manifest_artifacts(
        protocol,
        source_entries,
        "confirmation source",
    ) as source_by_name:
        data_columns = [
            ID_COLUMN,
            ERA_COLUMN,
            *COMPONENT_TARGETS.values(),
            ENDER_TARGET,
        ]
        benchmark_columns = [
            ID_COLUMN,
            ERA_COLUMN,
            BENCHMARK_ENDER20,
            BENCHMARK_ENDER60,
        ]
        data_frames: list[pd.DataFrame] = []
        for filename in ("train.parquet", "validation.parquet"):
            path = source_by_name[filename]
            schema_names = pq.read_schema(path).names
            columns = list(dict.fromkeys(data_columns))
            if "data_type" in schema_names:
                columns.append("data_type")
            frame = pd.read_parquet(path, columns=columns)
            if "data_type" in frame:
                frame = frame[
                    frame["data_type"].astype(str).isin(("train", "validation"))
                ]
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
        finite = np.isfinite(
            full[numeric_columns].to_numpy(dtype=np.float64)
        ).all(axis=1)
        full = full.loc[finite].reset_index(drop=True)
        _validate_finite_columns(
            full, numeric_columns, "Confirmation targets and benchmarks"
        )
        coverage = protocol.source_manifest["final_fit_target_coverage"]
        _exact_equal(
            len(full),
            coverage["finite_all_targets_and_benchmarks_rows"],
            "confirmation finite rows",
        )
        return full


def _checkpointed_file_receipt(
    protocol: FrozenProtocol,
    path: Path,
    label: str,
) -> dict[str, Any]:
    """Prove a live file is byte-identical to its blob in the current checkpoint."""

    path = Path(os.path.abspath(path))
    relative = _lexical_relative_path(path, protocol.repo_root)
    _require_lexical_directory_chain(
        protocol.repo_root,
        path.parent,
        f"{label} checkpoint path",
    )
    _require_regular_unlinked_file(path, label)
    checkpoint_object = f"{protocol.pretraining_commit}:{relative}"
    exists = _run_git(protocol.repo_root, ["cat-file", "-e", checkpoint_object])
    _require(exists.returncode == 0, f"{label} is absent from the confirmation checkpoint.")
    committed = _run_git(protocol.repo_root, ["rev-parse", checkpoint_object])
    _require(committed.returncode == 0, f"{label} checkpoint blob cannot be resolved.")
    committed_blob = committed.stdout.strip()
    live = _run_git(
        protocol.repo_root,
        ["hash-object", f"--path={relative}", "--", str(path)],
    )
    _require(live.returncode == 0, f"{label} live blob cannot be hashed.")
    _exact_equal(live.stdout.strip(), committed_blob, f"{label} checkpoint blob")
    clean = _run_git(
        protocol.repo_root,
        ["status", "--porcelain=v1", "--untracked-files=all", "--", relative],
    )
    _require(
        clean.returncode == 0 and not clean.stdout.strip(),
        f"{label} has uncommitted or untracked changes.",
    )
    receipt = _file_receipt(path, protocol.repo_root)
    receipt.update(
        {
            "checkpoint_commit": protocol.pretraining_commit,
            "git_blob_id": committed_blob,
        }
    )
    return receipt


def _checkpoint_path_absent(
    protocol: FrozenProtocol,
    relative: str,
    label: str,
) -> None:
    """Require a successful tree query with no entry at an exact checkpoint path."""

    _lexical_repo_path(protocol.repo_root, relative)
    query = _run_git(
        protocol.repo_root,
        [
            "ls-tree",
            "--full-tree",
            "-r",
            "--name-only",
            protocol.pretraining_commit,
            "--",
            relative,
        ],
    )
    _require(
        query.returncode == 0,
        f"{label} checkpoint tree query failed.",
    )
    _exact_equal(query.stdout.strip(), "", f"{label} checkpoint absence")


def _validate_checkpointed_file_receipt(
    protocol: FrozenProtocol,
    receipt: Mapping[str, Any],
    expected_relative: str,
    label: str,
) -> Path:
    _exact_equal(receipt.get("path"), expected_relative, f"{label} path")
    path = _safe_repo_path(protocol.repo_root, expected_relative)
    _exact_equal(
        dict(receipt),
        _checkpointed_file_receipt(protocol, path, label),
        f"{label} committed-live receipt",
    )
    return path


def _configured_confirmation_store_directory(
    protocol: FrozenProtocol,
    name: str,
    config: Mapping[str, Any],
) -> Path:
    data = config["data"]
    _require("label_sidecar_path" not in data, f"{name} confirmation sidecar is forbidden.")
    configured = data.get("disk_feature_store_path", data.get("feature_store_path"))
    _require(isinstance(configured, str) and bool(configured), f"{name} store path is malformed.")
    _exact_equal(
        configured,
        _confirmation_store_relative(name),
        f"{name} exact confirmation store path",
    )
    configured_path = Path(configured)
    _require(
        configured_path.parts
        and configured_path.parts[0] == "v5.3"
        and ".." not in configured_path.parts,
        f"{name} confirmation store must be a traversal-free v5.3 path.",
    )
    configured_path = _lexical_repo_path(
        protocol.repo_root,
        Path("numerai") / configured_path,
    )
    _require_lexical_directory_chain(
        protocol.repo_root,
        configured_path,
        f"{name} confirmation store directory",
    )
    return configured_path


def _require_regular_unlinked_receipt_file(
    protocol: FrozenProtocol,
    receipt: Mapping[str, Any],
    label: str,
) -> Path:
    relative = receipt.get("path")
    _require(isinstance(relative, str), f"{label} path is malformed.")
    unresolved = _lexical_repo_path(protocol.repo_root, relative)
    _require_lexical_directory_chain(
        protocol.repo_root,
        unresolved.parent,
        f"{label} parent directory",
    )
    _require_regular_unlinked_file(unresolved, label)
    return _validate_path_receipt(protocol.repo_root, receipt, label)


def _confirmation_source_fingerprints(protocol: FrozenProtocol) -> list[dict[str, Any]]:
    fingerprints: list[dict[str, Any]] = []
    for index, pinned in enumerate(protocol.source_manifest["confirmation_sources"]):
        path = _safe_repo_path(protocol.repo_root, pinned["path"])
        current = parquet_source_fingerprint(path)
        expected_current = {
            "path": str(path.resolve()),
            "size_bytes": pinned["size_bytes"],
            "mtime_ns": pinned["mtime_ns"],
            "num_rows": pinned["num_rows"],
            "num_row_groups": pinned["num_row_groups"],
            "schema_sha256": pinned["schema_sha256"],
            "footer_sha256": pinned["footer_sha256"],
        }
        _exact_equal(current, expected_current, f"confirmation source fingerprint {index}")
        fingerprints.append(
            {
                "role": "data" if index < 2 else "benchmark",
                "position": index if index < 2 else index - 2,
                **current,
            }
        )
    return fingerprints


def _confirmation_store_receipt(
    protocol: FrozenProtocol,
    name: str,
) -> dict[str, Any]:
    component = default_confirmation_component_paths(protocol, name)
    config = _load_confirmation_config(protocol, name, component.config)
    directory = _configured_confirmation_store_directory(protocol, name, config)
    metadata_path = directory / "metadata.json"
    _require_regular_unlinked_file(metadata_path, f"{name} store metadata")
    metadata = _load_json(metadata_path, f"{name} store metadata")
    features = metadata.get("features")
    manifest = metadata.get("manifest")
    _require(isinstance(features, Mapping), f"{name} metadata.features is malformed.")
    _require(isinstance(manifest, Mapping), f"{name} metadata.manifest is malformed.")
    feature_filename = features.get("filename")
    manifest_filename = manifest.get("filename")
    _require(
        isinstance(feature_filename, str)
        and Path(feature_filename).parts == (feature_filename,),
        f"{name} feature filename is not a lexical leaf.",
    )
    _require(
        isinstance(manifest_filename, str)
        and Path(manifest_filename).parts == (manifest_filename,),
        f"{name} manifest filename is not a lexical leaf.",
    )
    feature_path = directory / feature_filename
    manifest_path = directory / manifest_filename
    _require_regular_unlinked_file(feature_path, f"{name} store features")
    _require_regular_unlinked_file(manifest_path, f"{name} store manifest")
    return {
        "generation_id": metadata.get("generation_id"),
        "row_count": metadata.get("row_count"),
        "feature_count": metadata.get("feature_count"),
        "feature_order_sha256": metadata.get("feature_order_sha256"),
        "target_column": metadata.get("target_column"),
        "metadata": _file_receipt(metadata_path, protocol.repo_root),
        "manifest": _file_receipt(manifest_path, protocol.repo_root),
        "features": _file_receipt(feature_path, protocol.repo_root),
    }


def _validate_distinct_confirmation_store_files(
    protocol: FrozenProtocol,
    stores: Mapping[str, Any],
) -> None:
    """Require fifteen distinct regular files across the five target stores."""

    _exact_equal(set(stores), set(ALL_COMPONENTS), "confirmation target stores")
    paths: list[tuple[str, Path]] = []
    for name in ALL_COMPONENTS:
        store = stores.get(name)
        _require(isinstance(store, Mapping), f"{name} store receipt is malformed.")
        for key in ("metadata", "manifest", "features"):
            item = store.get(key)
            _require(isinstance(item, Mapping), f"{name} store {key} receipt is malformed.")
            path = _require_regular_unlinked_receipt_file(
                protocol,
                item,
                f"{name} store {key}",
            )
            paths.append((f"{name}.{key}", Path(os.path.abspath(path))))
    _exact_equal(
        len({str(path).casefold() for _, path in paths}),
        len(paths),
        "confirmation store lexical file paths",
    )
    for index, (left_label, left) in enumerate(paths):
        for right_label, right in paths[index + 1 :]:
            try:
                same = os.path.samefile(left, right)
            except OSError as error:
                raise EnderEnsembleEvaluationError(
                    f"Confirmation store files cannot be compared: {left_label}, {right_label}."
                ) from error
            _require(
                not same,
                f"Confirmation store files are not physically distinct: "
                f"{left_label}, {right_label}.",
            )


def _validate_confirmation_store_receipt(
    protocol: FrozenProtocol,
    name: str,
    store: Mapping[str, Any],
    raw: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deeply bind one physical target store to frozen features and raw labels."""

    _require(name in ALL_COMPONENTS, f"Unknown confirmation store: {name}")
    canonical = protocol.source_manifest["confirmation_xerxes_medium_store_anchor"]
    _exact_equal(
        set(store),
        set(canonical),
        f"{name} confirmation store receipt keys",
    )
    for key in ("metadata", "manifest", "features"):
        item = store.get(key)
        _require(isinstance(item, Mapping), f"{name} store {key} receipt is malformed.")
        _require_regular_unlinked_receipt_file(protocol, item, f"{name} store {key}")
    metadata_path = _safe_repo_path(protocol.repo_root, store["metadata"]["path"])
    metadata = _load_json(metadata_path, f"{name} store metadata")
    generation_id = metadata.get("generation_id")
    _require(
        isinstance(generation_id, str)
        and len(generation_id) == 32
        and all(character in "0123456789abcdef" for character in generation_id),
        f"{name} store generation ID is malformed.",
    )
    expected_metadata = {
        "format": "numerai-v5.3-int8-feature-store",
        "format_version": 1,
        "complete": True,
        "generation_id": generation_id,
        "row_count": canonical["row_count"],
        "feature_count": canonical["feature_count"],
        "feature_columns": list(protocol.medium_features),
        "feature_order_sha256": canonical["feature_order_sha256"],
        "target_column": COMPONENT_TARGETS[name],
        "benchmark_column": BENCHMARK_ENDER20,
        "source_fingerprints": _confirmation_source_fingerprints(protocol),
    }
    for key, expected_value in expected_metadata.items():
        _exact_equal(metadata.get(key), expected_value, f"{name} store metadata.{key}")
    _exact_equal(
        feature_order_sha256(metadata["feature_columns"]),
        canonical["feature_order_sha256"],
        f"{name} recomputed feature order",
    )
    expected_feature_size = canonical["row_count"] * canonical["feature_count"]
    embedded_features = metadata.get("features")
    embedded_manifest = metadata.get("manifest")
    _require(isinstance(embedded_features, Mapping), f"{name} metadata.features is malformed.")
    _require(isinstance(embedded_manifest, Mapping), f"{name} metadata.manifest is malformed.")
    _exact_equal(
        embedded_features,
        {
            "filename": f"features-{generation_id}.int8.bin",
            "dtype": "int8",
            "layout": "C",
            "size_bytes": expected_feature_size,
            "sha256": canonical["features"]["sha256"],
        },
        f"{name} embedded feature receipt",
    )
    manifest_columns = [
        "row_offset",
        ID_COLUMN,
        ERA_COLUMN,
        COMPONENT_TARGETS[name],
        BENCHMARK_ENDER20,
    ]
    _exact_equal(
        embedded_manifest,
        {
            "filename": f"manifest-{generation_id}.parquet",
            "columns": manifest_columns,
            "size_bytes": store["manifest"]["size_bytes"],
            "sha256": store["manifest"]["sha256"],
        },
        f"{name} embedded manifest receipt",
    )
    expected_store = {
        "generation_id": generation_id,
        "row_count": canonical["row_count"],
        "feature_count": canonical["feature_count"],
        "feature_order_sha256": canonical["feature_order_sha256"],
        "target_column": COMPONENT_TARGETS[name],
        "metadata": _file_receipt(metadata_path, protocol.repo_root),
        "manifest": _file_receipt(
            metadata_path.parent / embedded_manifest["filename"], protocol.repo_root
        ),
        "features": _file_receipt(
            metadata_path.parent / embedded_features["filename"], protocol.repo_root
        ),
    }
    _exact_equal(dict(store), expected_store, f"{name} store file receipts")
    _exact_equal(
        store["features"]["sha256"],
        canonical["features"]["sha256"],
        f"{name} canonical feature bytes",
    )
    _exact_equal(
        store["features"]["size_bytes"],
        canonical["features"]["size_bytes"],
        f"{name} canonical feature size",
    )
    if name == "xerxes":
        _exact_equal(dict(store), canonical, "canonical Xerxes store receipt")

    if config is None:
        config = _load_confirmation_config(
            protocol,
            name,
            default_confirmation_component_paths(protocol, name).config,
        )
    validate_component_config(name, config, confirmation=True)
    _exact_equal(
        _configured_confirmation_store_directory(protocol, name, config),
        Path(os.path.abspath(metadata_path.parent)),
        f"{name} configured target store",
    )

    manifest_path = _safe_repo_path(protocol.repo_root, store["manifest"]["path"])
    parquet = pq.ParquetFile(manifest_path)
    try:
        _exact_equal(parquet.schema_arrow.names, manifest_columns, f"{name} manifest columns")
        _exact_equal(parquet.metadata.num_rows, canonical["row_count"], f"{name} manifest rows")
        _exact_equal(
            [field.type for field in parquet.schema_arrow],
            [pa.int64(), pa.string(), pa.string(), pa.float32(), pa.float64()],
            f"{name} manifest schema",
        )
    finally:
        parquet.close()
    manifest_frame = pd.read_parquet(manifest_path, columns=manifest_columns)
    _require(
        np.array_equal(
            manifest_frame["row_offset"].to_numpy(dtype=np.int64),
            np.arange(len(raw), dtype=np.int64),
        ),
        f"{name} manifest row offsets are not consecutive.",
    )
    _require(
        manifest_frame[ID_COLUMN].notna().all() and manifest_frame[ID_COLUMN].is_unique,
        f"{name} manifest IDs are invalid.",
    )
    _require(manifest_frame[ERA_COLUMN].notna().all(), f"{name} manifest eras are invalid.")
    _validate_finite_columns(
        manifest_frame,
        [COMPONENT_TARGETS[name], BENCHMARK_ENDER20],
        f"{name} manifest labels",
    )
    for column in (ID_COLUMN, ERA_COLUMN):
        _require(
            np.array_equal(
                manifest_frame[column].astype(str).to_numpy(),
                raw[column].astype(str).to_numpy(),
            ),
            f"{name} manifest {column} order differs from raw sources.",
        )
    for column in (COMPONENT_TARGETS[name], BENCHMARK_ENDER20):
        _require(
            np.array_equal(manifest_frame[column].to_numpy(), raw[column].to_numpy()),
            f"{name} manifest {column} differs from raw sources.",
        )
    return metadata


def _confirmation_store_inventory_file(protocol: FrozenProtocol) -> Path:
    path = _lexical_repo_path(protocol.repo_root, CONFIRMATION_STORE_INVENTORY_PATH)
    _exact_equal(
        path.parent,
        Path(os.path.abspath(protocol.experiment_dir)),
        "confirmation store inventory parent",
    )
    _require_lexical_directory_chain(
        protocol.repo_root,
        path.parent,
        "confirmation store inventory",
    )
    return path


def _confirmation_store_inventory_payload(
    protocol: FrozenProtocol,
    scout_locked_receipt_path: Path,
    scout_locked_receipt_sha256: str,
    selected: str,
    stores: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "artifact": "confirmation-store-inventory-v1",
        "scout_locked_receipt": {
            "path": _lexical_relative_path(scout_locked_receipt_path, protocol.repo_root),
            "sha256": scout_locked_receipt_sha256,
        },
        "selected_formula": {
            "name": selected,
            "weights": dict(BLEND_WEIGHTS[selected]),
        },
        "canonical_store": copy.deepcopy(
            protocol.source_manifest["confirmation_xerxes_medium_store_anchor"]
        ),
        "input_layout": {
            "type": "dedicated_target_stores",
            "stores": copy.deepcopy(dict(stores)),
        },
    }


def _validate_confirmation_store_inventory(
    protocol: FrozenProtocol,
    inventory: Mapping[str, Any],
    scout_locked_receipt_path: Path,
    scout_locked_receipt_sha256: str,
    selected: str,
) -> Mapping[str, Any]:
    """Validate the closed, commit-free static inventory schema."""

    _exact_equal(
        set(inventory),
        {
            "schema_version",
            "experiment",
            "artifact",
            "scout_locked_receipt",
            "selected_formula",
            "canonical_store",
            "input_layout",
        },
        "confirmation store inventory keys",
    )
    _exact_equal(inventory.get("schema_version"), 1, "confirmation store inventory schema")
    _exact_equal(inventory.get("experiment"), EXPERIMENT_NAME, "confirmation store inventory experiment")
    _exact_equal(
        inventory.get("artifact"),
        "confirmation-store-inventory-v1",
        "confirmation store inventory artifact",
    )
    expected_locked = {
        "path": _lexical_relative_path(scout_locked_receipt_path, protocol.repo_root),
        "sha256": scout_locked_receipt_sha256,
    }
    _exact_equal(
        inventory.get("scout_locked_receipt"),
        expected_locked,
        "confirmation store inventory Scout authorization",
    )
    _exact_equal(
        inventory.get("selected_formula"),
        {"name": selected, "weights": dict(BLEND_WEIGHTS[selected])},
        "confirmation store inventory selected formula",
    )
    canonical = protocol.source_manifest["confirmation_xerxes_medium_store_anchor"]
    _exact_equal(
        inventory.get("canonical_store"),
        canonical,
        "confirmation store inventory canonical store",
    )
    layout = inventory.get("input_layout")
    _require(isinstance(layout, Mapping), "Confirmation store inventory layout is malformed.")
    _exact_equal(set(layout), {"type", "stores"}, "confirmation store inventory layout keys")
    _exact_equal(layout.get("type"), "dedicated_target_stores", "confirmation store inventory layout")
    stores = layout.get("stores")
    _require(isinstance(stores, Mapping), "Confirmation store inventory stores are malformed.")
    _exact_equal(set(stores), set(ALL_COMPONENTS), "confirmation store inventory components")
    for name in ALL_COMPONENTS:
        store = stores.get(name)
        _require(isinstance(store, Mapping), f"{name} inventory store is malformed.")
        _exact_equal(
            set(store),
            set(canonical),
            f"{name} inventory store keys",
        )
    _validate_distinct_confirmation_store_files(protocol, stores)
    return stores


def create_confirmation_store_inventory(
    protocol: FrozenProtocol,
    scout_locked_receipt_path: Path,
    scout_locked_receipt_sha256: str,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Create the one fixed static inventory after a passing Scout lock."""

    canonical_dir, claim_path = _claim_canonical_receipt_prefix(
        protocol,
        output_dir,
        "confirmation-store-inventory",
    )
    inventory_path = _confirmation_store_inventory_file(protocol)
    _require_absent_destination(inventory_path, "confirmation store inventory")

    scout_locked_receipt_path = Path(os.path.abspath(scout_locked_receipt_path))
    scout_locked = _load_passing_scout_locked_receipt(
        protocol,
        scout_locked_receipt_path,
        scout_locked_receipt_sha256,
    )
    selected = _selected_formula(scout_locked)
    configs = {
        name: _load_confirmation_config(
            protocol,
            name,
            default_confirmation_component_paths(protocol, name).config,
        )
        for name in ALL_COMPONENTS
    }
    raw = _read_full_confirmation_sources(protocol)
    stores: dict[str, Any] = {}
    for name in ALL_COMPONENTS:
        store = _confirmation_store_receipt(protocol, name)
        _validate_confirmation_store_receipt(
            protocol,
            name,
            store,
            raw,
            config=configs[name],
        )
        stores[name] = store
    _validate_distinct_confirmation_store_files(protocol, stores)
    canonical = protocol.source_manifest["confirmation_xerxes_medium_store_anchor"]
    _exact_equal(stores["xerxes"], canonical, "canonical Xerxes store receipt")
    inventory = _confirmation_store_inventory_payload(
        protocol,
        scout_locked_receipt_path,
        scout_locked_receipt_sha256,
        selected,
        stores,
    )
    _validate_confirmation_store_inventory(
        protocol,
        inventory,
        scout_locked_receipt_path,
        scout_locked_receipt_sha256,
        selected,
    )
    try:
        with inventory_path.open("xb") as stream:
            stream.write(_receipt_bytes(inventory))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise EnderEnsembleEvaluationError(
            f"Confirmation store inventory already exists: {inventory_path}"
        ) from error
    _require_regular_unlinked_file(inventory_path, "confirmation store inventory")
    _exact_equal(
        _load_json(inventory_path, "confirmation store inventory"),
        inventory,
        "written confirmation store inventory",
    )
    stage_receipt: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "stage": "create-confirmation-store-inventory",
        "state": "PASS",
        "passed": True,
        "protocol": _protocol_binding(protocol),
        "scout_locked_receipt": {
            "path": _lexical_relative_path(scout_locked_receipt_path, protocol.repo_root),
            "sha256": scout_locked_receipt_sha256,
        },
        "selected_formula": {
            "name": selected,
            "weights": dict(BLEND_WEIGHTS[selected]),
        },
        "inventory": _file_receipt(inventory_path, protocol.repo_root),
    }
    path = _write_claimed_content_addressed_receipt(
        canonical_dir,
        "confirmation-store-inventory",
        claim_path,
        stage_receipt,
    )
    return path, stage_receipt


def _confirmation_destination_receipts(
    protocol: FrozenProtocol,
    *,
    require_live_absence: bool,
) -> dict[str, Any]:
    destinations: dict[str, Any] = {}
    for name in ALL_COMPONENTS:
        contract = protocol.source_manifest["confirmation_output_contract"][name]
        _exact_equal(contract.get("must_be_absent_before_run"), True, f"{name} absence contract")
        item = {
            "results_path": contract["results_path"],
            "predictions_path": contract["predictions_path"],
            "results_absent_at_checkpoint": True,
            "predictions_absent_at_checkpoint": True,
        }
        for key in ("results_path", "predictions_path"):
            relative = contract[key]
            destination = _lexical_repo_path(protocol.repo_root, relative)
            _prepare_output_destination_parent(
                protocol,
                destination,
                f"{name} {key}",
                create_direct_parent=False,
            )
            _checkpoint_path_absent(
                protocol,
                relative,
                f"{name} {key}",
            )
            if require_live_absence:
                _require_absent_destination(
                    destination,
                    f"{name} {key} before confirmation pretraining",
                )
        destinations[name] = item
    return destinations


def _confirmation_pretraining_binding(
    protocol: FrozenProtocol,
    path: Path,
    digest: str,
) -> dict[str, str]:
    return {
        "path": _lexical_relative_path(_lexical_absolute(path), protocol.repo_root),
        "sha256": digest,
    }


def _validate_confirmation_pretraining_receipt(
    protocol: FrozenProtocol,
    path: Path | None,
    expected_sha256: str | None,
) -> dict[str, Any]:
    _require(
        path is not None and expected_sha256 is not None,
        "Confirmation pretraining receipt path/hash are both required.",
    )
    assert path is not None and expected_sha256 is not None
    receipt = _load_bound_receipt(
        _lexical_absolute(path),
        expected_sha256,
        expected_stage="confirmation-pretraining",
        receipt_dir=_canonical_receipt_dir(protocol),
        expected_prefix="confirmation-pretraining",
    )
    _validate_stage_receipt_schema(receipt, "confirmation-pretraining")
    _exact_equal(receipt.get("passed"), True, "confirmation pretraining passage")
    _exact_equal(receipt.get("state"), "PASS", "confirmation pretraining state")
    _validate_protocol_binding(receipt, protocol)
    _exact_equal(receipt.get("checkpoint"), protocol.pretraining_commit, "confirmation checkpoint")

    scout_locked = receipt.get("scout_locked_receipt")
    _require(isinstance(scout_locked, Mapping), "Scout locked authorization is malformed.")
    scout_path = _lexical_repo_path(protocol.repo_root, str(scout_locked.get("path")))
    scout_digest = scout_locked.get("sha256")
    _require(isinstance(scout_digest, str), "Scout locked authorization hash is malformed.")
    scout_receipt = _load_passing_scout_locked_receipt(
        protocol,
        scout_path,
        scout_digest,
    )
    selected = _selected_formula(scout_receipt)

    configs = receipt.get("configs")
    _require(isinstance(configs, Mapping), "Confirmation config receipts are malformed.")
    _exact_equal(set(configs), set(ALL_COMPONENTS), "confirmation config components")
    loaded_configs: dict[str, dict[str, Any]] = {}
    for name in ALL_COMPONENTS:
        config_receipt = configs[name]
        _require(isinstance(config_receipt, Mapping), f"{name} config receipt is malformed.")
        relative = protocol.source_manifest["confirmation_output_contract"][name]["config_path"]
        config_path = _validate_checkpointed_file_receipt(
            protocol, config_receipt, relative, f"{name} confirmation config"
        )
        config = _load_confirmation_config(protocol, name, config_path)
        loaded_configs[name] = config

    helpers = receipt.get("config_helpers")
    _require(isinstance(helpers, list), "Confirmation config-helper receipts are malformed.")
    _exact_equal(len(helpers), len(CONFIRMATION_CONFIG_HELPER_PATHS), "config-helper count")
    for item, relative in zip(helpers, CONFIRMATION_CONFIG_HELPER_PATHS, strict=True):
        _require(isinstance(item, Mapping), "Confirmation config-helper receipt is malformed.")
        _validate_checkpointed_file_receipt(protocol, item, relative, "confirmation config helper")

    loader = receipt.get("loader")
    _require(isinstance(loader, Mapping), "Confirmation loader receipt is malformed.")
    _exact_equal(loader.get("checkpoint"), protocol.pretraining_commit, "loader checkpoint")
    loader_files = loader.get("files")
    _require(isinstance(loader_files, list), "Confirmation loader files are malformed.")
    _exact_equal(len(loader_files), len(CONFIRMATION_LOADER_PATHS), "loader file count")
    for item, relative in zip(loader_files, CONFIRMATION_LOADER_PATHS, strict=True):
        _require(isinstance(item, Mapping), "Confirmation loader file receipt is malformed.")
        _validate_checkpointed_file_receipt(protocol, item, relative, "confirmation loader file")

    inventory_receipt = receipt.get("store_inventory")
    _require(
        isinstance(inventory_receipt, Mapping),
        "Confirmation store inventory receipt is malformed.",
    )
    inventory_path = _validate_checkpointed_file_receipt(
        protocol,
        inventory_receipt,
        CONFIRMATION_STORE_INVENTORY_PATH,
        "confirmation store inventory",
    )
    inventory = _load_json(inventory_path, "confirmation store inventory")
    inventory_stores = _validate_confirmation_store_inventory(
        protocol,
        inventory,
        scout_path,
        scout_digest,
        selected,
    )

    canonical = protocol.source_manifest["confirmation_xerxes_medium_store_anchor"]
    _exact_equal(receipt.get("canonical_store"), canonical, "canonical confirmation store")
    layout = receipt.get("input_layout")
    _require(isinstance(layout, Mapping), "Confirmation input layout is malformed.")
    _exact_equal(set(layout), {"type", "stores"}, "confirmation input-layout keys")
    _exact_equal(layout.get("type"), "dedicated_target_stores", "confirmation input layout")
    stores = layout.get("stores")
    _require(isinstance(stores, Mapping), "Confirmation store receipts are malformed.")
    _exact_equal(set(stores), set(ALL_COMPONENTS), "confirmation target stores")
    _exact_equal(stores, inventory_stores, "committed confirmation store inventory layout")
    raw = _read_full_confirmation_sources(protocol)
    for name in ALL_COMPONENTS:
        store = stores[name]
        _require(isinstance(store, Mapping), f"{name} store receipt is malformed.")
        live_store = _confirmation_store_receipt(protocol, name)
        _exact_equal(live_store, store, f"{name} live store inventory")
        _validate_confirmation_store_receipt(
            protocol,
            name,
            store,
            raw,
            config=loaded_configs[name],
        )
    _validate_distinct_confirmation_store_files(protocol, stores)
    _exact_equal(
        receipt.get("output_destinations"),
        _confirmation_destination_receipts(protocol, require_live_absence=False),
        "confirmation output destinations",
    )
    return receipt


def _validate_confirmation_input_receipt(
    protocol: FrozenProtocol,
    path: Path | None,
    expected_sha256: str | None,
) -> dict[str, Any]:
    """Backward-compatible internal alias for the frozen pretraining receipt."""

    return _validate_confirmation_pretraining_receipt(protocol, path, expected_sha256)


def create_confirmation_pretraining_receipt(
    protocol: FrozenProtocol,
    scout_locked_receipt_path: Path,
    scout_locked_receipt_sha256: str,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Freeze committed confirmation configs, loader, stores, and absent outputs."""

    canonical_dir, claim_path = _claim_canonical_receipt_prefix(
        protocol, output_dir, "confirmation-pretraining"
    )
    scout_locked_receipt_path = _lexical_absolute(scout_locked_receipt_path)
    scout_locked = _load_passing_scout_locked_receipt(
        protocol,
        scout_locked_receipt_path,
        scout_locked_receipt_sha256,
    )
    selected = _selected_formula(scout_locked)
    configs: dict[str, Any] = {}
    loaded_configs: dict[str, dict[str, Any]] = {}
    for name in ALL_COMPONENTS:
        relative = protocol.source_manifest["confirmation_output_contract"][name]["config_path"]
        path = _lexical_repo_path(protocol.repo_root, relative)
        configs[name] = _checkpointed_file_receipt(
            protocol, path, f"{name} confirmation config"
        )
        config = _load_confirmation_config(protocol, name, path)
        loaded_configs[name] = config
    helpers = [
        _checkpointed_file_receipt(
            protocol,
            _safe_repo_path(protocol.repo_root, relative),
            "confirmation config helper",
        )
        for relative in CONFIRMATION_CONFIG_HELPER_PATHS
    ]
    loader_files = [
        _checkpointed_file_receipt(
            protocol,
            _safe_repo_path(protocol.repo_root, relative),
            "confirmation loader file",
        )
        for relative in CONFIRMATION_LOADER_PATHS
    ]
    inventory_path = _confirmation_store_inventory_file(protocol)
    inventory_receipt = _checkpointed_file_receipt(
        protocol,
        inventory_path,
        "confirmation store inventory",
    )
    inventory = _load_json(inventory_path, "confirmation store inventory")
    inventory_stores = _validate_confirmation_store_inventory(
        protocol,
        inventory,
        scout_locked_receipt_path,
        scout_locked_receipt_sha256,
        selected,
    )
    raw = _read_full_confirmation_sources(protocol)
    stores: dict[str, Any] = {}
    for name in ALL_COMPONENTS:
        store = inventory_stores[name]
        _require(isinstance(store, Mapping), f"{name} inventory store is malformed.")
        live_store = _confirmation_store_receipt(protocol, name)
        _exact_equal(live_store, store, f"{name} live store inventory")
        _validate_confirmation_store_receipt(
            protocol, name, store, raw, config=loaded_configs[name]
        )
        stores[name] = copy.deepcopy(dict(store))
    _validate_distinct_confirmation_store_files(protocol, stores)
    canonical = protocol.source_manifest["confirmation_xerxes_medium_store_anchor"]
    _exact_equal(stores["xerxes"], canonical, "canonical Xerxes store receipt")
    destinations = _confirmation_destination_receipts(
        protocol, require_live_absence=True
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "stage": "confirmation-pretraining",
        "state": "PASS",
        "passed": True,
        "protocol": _protocol_binding(protocol),
        "checkpoint": protocol.pretraining_commit,
        "scout_locked_receipt": {
            "path": _lexical_relative_path(scout_locked_receipt_path, protocol.repo_root),
            "sha256": scout_locked_receipt_sha256,
        },
        "configs": configs,
        "config_helpers": helpers,
        "loader": {
            "checkpoint": protocol.pretraining_commit,
            "files": loader_files,
        },
        "store_inventory": inventory_receipt,
        "canonical_store": canonical,
        "input_layout": {
            "type": "dedicated_target_stores",
            "stores": stores,
        },
        "output_destinations": destinations,
    }
    path = _write_claimed_content_addressed_receipt(
        canonical_dir,
        "confirmation-pretraining",
        claim_path,
        receipt,
    )
    return path, receipt


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
    with _lease_frozen_manifest_artifacts(
        protocol,
        {"manifest": anchor["manifest"]},
        "confirmation canonical store",
    ) as store_paths:
        store = pd.read_parquet(
            store_paths["manifest"],
            columns=[
                "row_offset",
                ID_COLUMN,
                ERA_COLUMN,
                COMPONENT_TARGETS["xerxes"],
                BENCHMARK_ENDER20,
            ],
        )
        _exact_equal(len(store), len(full), "canonical store row count")
        _require(
            np.array_equal(
                store["row_offset"].to_numpy(dtype=np.int64),
                np.arange(len(store), dtype=np.int64),
            ),
            "Canonical store row offsets are not consecutive.",
        )
        for column in (ID_COLUMN, ERA_COLUMN):
            _require(
                np.array_equal(
                    store[column].astype(str).to_numpy(),
                    full[column].astype(str).to_numpy(),
                ),
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


def confirmation_locked_checks(summary: Mapping[str, Any]) -> dict[str, bool]:
    """Evaluate only the authorized final 200-era confirmation holdout."""

    bmc = summary["bmc"]
    corr = summary["corr"]
    return {
        "locked_bmc_mean": float(bmc["mean"])
        > CONFIRMATION_THRESHOLDS["locked_bmc_mean_min_exclusive"],
        "locked_bmc_sharpe": bool(bmc.get("std_valid", True))
        and bmc.get("sharpe") is not None
        and float(bmc["sharpe"])
        > CONFIRMATION_THRESHOLDS["locked_bmc_sharpe_min_exclusive"],
        "locked_bmc_max_drawdown": float(bmc["max_drawdown"])
        < CONFIRMATION_THRESHOLDS["locked_bmc_max_drawdown_max_exclusive"],
        "locked_corr_mean": float(corr["mean"])
        > CONFIRMATION_THRESHOLDS["locked_corr_mean_min_exclusive"],
    }


def confirmation_full_checks(summary: Mapping[str, Any]) -> dict[str, bool]:
    """Evaluate the full 855 eras only after the locked 200 eras pass."""

    bmc = summary["bmc"]
    corr = summary["corr"]
    return {
        "full_bmc_mean": float(bmc["mean"])
        >= CONFIRMATION_THRESHOLDS["bmc_mean_min_inclusive"],
        "full_bmc_sharpe": bool(bmc.get("std_valid", True))
        and bmc.get("sharpe") is not None
        and float(bmc["sharpe"])
        > CONFIRMATION_THRESHOLDS["bmc_sharpe_min_exclusive"],
        "full_bmc_max_drawdown": float(bmc["max_drawdown"])
        < CONFIRMATION_THRESHOLDS["bmc_max_drawdown_max_exclusive"],
        "full_corr_mean": float(corr["mean"])
        >= CONFIRMATION_THRESHOLDS["corr_mean_min_inclusive"],
        "full_ender20_similarity": float(summary["avg_ender20_similarity"])
        < CONFIRMATION_THRESHOLDS["similarity_max_exclusive"],
        "full_ender60_similarity": float(summary["avg_ender60_similarity"])
        < CONFIRMATION_THRESHOLDS["similarity_max_exclusive"],
        "full_tabm_similarity": float(summary["avg_tabm_similarity"])
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
    store_receipt: Mapping[str, Any] | None = None,
    store_inventory_receipt: Mapping[str, Any] | None = None,
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
        _require(
            isinstance(store_metadata, Mapping)
            and isinstance(store_receipt, Mapping)
            and isinstance(store_inventory_receipt, Mapping),
            f"{component.name} lacks immutable store provenance.",
        )
        assert store_metadata is not None
        assert store_receipt is not None
        assert store_inventory_receipt is not None
        _exact_equal(
            set(diagnostics),
            {
                "directory",
                "feature_path",
                "manifest_path",
                "generation_id",
                "row_count",
                "feature_count",
                "feature_bytes",
                "manifest_bytes",
                "feature_order_sha256",
                "metadata_sha256",
                "feature_sha256",
                "manifest_sha256",
                "committed_inventory",
            },
            f"{component.name} store diagnostics keys",
        )
        for key, expected_value in {
            "generation_id": store_metadata.get("generation_id"),
            "row_count": store_metadata.get("row_count"),
            "feature_count": store_metadata.get("feature_count"),
            "feature_order_sha256": store_metadata.get("feature_order_sha256"),
            "feature_bytes": store_receipt["features"]["size_bytes"],
            "manifest_bytes": store_receipt["manifest"]["size_bytes"],
            "metadata_sha256": store_receipt["metadata"]["sha256"],
            "feature_sha256": store_receipt["features"]["sha256"],
            "manifest_sha256": store_receipt["manifest"]["sha256"],
        }.items():
            _exact_equal(diagnostics.get(key), expected_value, f"{component.name} store.{key}")
        _exact_equal(
            diagnostics.get("committed_inventory"),
                {
                    "path": store_inventory_receipt.get("path"),
                    "git_blob_id": store_inventory_receipt.get("git_blob_id"),
                    "checkpoint_commit": store_inventory_receipt.get(
                        "checkpoint_commit"
                    ),
                },
            f"{component.name} committed inventory diagnostics",
        )

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


@contextmanager
def _lease_frozen_manifest_artifacts(
    protocol: FrozenProtocol,
    entries: Mapping[str, Mapping[str, Any]],
    label: str,
) -> Iterator[dict[str, Path]]:
    """Lease one exact manifest snapshot through every historical artifact read."""

    leases: list[_ReadOnlyFileLease] = []
    paths: dict[str, Path] = {}
    physical_ids: set[tuple[int, int]] = set()
    try:
        for name, value in entries.items():
            receipt = _require_exact_keys(
                value,
                {"path", "sha256", "size_bytes"},
                f"{label} {name} receipt",
            )
            relative = receipt.get("path")
            _require(
                isinstance(relative, str) and bool(relative),
                f"{label} {name} path is malformed.",
            )
            path = _lexical_repo_path(protocol.repo_root, relative)
            _require_lexical_directory_chain(
                protocol.repo_root,
                path.parent,
                f"{label} {name} parent",
            )
            _require_regular_unlinked_file(path, f"{label} {name}")
            lease = _ReadOnlyFileLease(path, f"{label} {name}")
            leases.append(lease)
            inspected = os.fstat(lease.fileno())
            path_inspected = path.lstat()
            identity = (int(inspected.st_dev), int(inspected.st_ino))
            _exact_equal(
                identity,
                (int(path_inspected.st_dev), int(path_inspected.st_ino)),
                f"{label} {name} leased identity",
            )
            _require(
                identity not in physical_ids,
                f"{label} historical artifacts reuse a physical file.",
            )
            physical_ids.add(identity)
            _exact_equal(
                int(inspected.st_nlink),
                1,
                f"{label} {name} link count",
            )
            _exact_equal(
                int(inspected.st_size),
                receipt.get("size_bytes"),
                f"{label} {name} size",
            )
            _exact_equal(
                lease.sha256(),
                receipt.get("sha256"),
                f"{label} {name} hash",
            )
            paths[name] = path
        yield paths
    finally:
        for lease in reversed(leases):
            lease.close()


def load_frozen_two_seed_residual(
    protocol: FrozenProtocol,
    expected: ExpectedCohort,
    *,
    confirmation: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate and construct only the gate-authorized two-seed TabM reference."""

    section_name = "confirmation" if confirmation else "scout"
    reference = protocol.source_manifest["tabm_similarity_reference"]
    section = reference[section_name]
    manifest_entries: dict[str, Mapping[str, Any]] = {
        **{
            f"dependency.{name}": receipt
            for name, receipt in reference["config_dependency_chain"].items()
        },
        "historical_source_manifest": {
            key: reference["historical_source_manifest"][key]
            for key in ("path", "sha256", "size_bytes")
        },
    }
    for seed in ("seed1337", "seed2027"):
        for name in ("config", "result", "predictions"):
            manifest_entries[f"{seed}.{name}"] = section[seed][name]
    with _lease_frozen_manifest_artifacts(
        protocol,
        manifest_entries,
        f"TabM {section_name}",
    ) as paths:
        expected_fold_map = _independent_fold_map(expected)
        raw: list[np.ndarray] = []
        receipts: dict[str, Any] = {}
        for seed in ("seed1337", "seed2027"):
            entry = section[seed]
            config_path = paths[f"{seed}.config"]
            result_path = paths[f"{seed}.result"]
            predictions_path = paths[f"{seed}.predictions"]
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
                name: entry[name]["sha256"]
                for name in ("config", "result", "predictions")
            }
        eras = expected.frame[ERA_COLUMN]
        first = rank_within_era(raw[0], eras)
        second = rank_within_era(raw[1], eras)
        blended = rank_within_era(0.5 * (first + second), eras)
        return blended, receipts


def _file_receipt(path: Path, repo_root: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Receipt input is missing: {path}")
    return {
        "path": _relative_path(path, repo_root),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _leased_file_receipt(path: Path, repo_root: Path, label: str) -> dict[str, Any]:
    """Hash one exact regular file handle and bind it to the current path."""

    path = _lexical_absolute(path)
    _require_lexical_directory_chain(repo_root, path.parent, f"{label} parent")
    _require_regular_unlinked_file(path, label)
    lease = _ReadOnlyFileLease(path, label)
    try:
        inspected = os.fstat(lease.fileno())
        path_inspected = path.lstat()
        _exact_equal(
            (int(inspected.st_dev), int(inspected.st_ino)),
            (int(path_inspected.st_dev), int(path_inspected.st_ino)),
            f"{label} leased identity",
        )
        _exact_equal(int(inspected.st_nlink), 1, f"{label} link count")
        return {
            "path": _lexical_relative_path(path, repo_root),
            "sha256": lease.sha256(),
            "size_bytes": int(inspected.st_size),
        }
    finally:
        lease.close()


def _leased_checkpoint_file_receipt(
    path: Path,
    repo_root: Path,
    checkpoint: str,
    label: str,
) -> dict[str, Any]:
    """Bind one leased live file to the exact bytes stored in a Git checkpoint."""

    path = _lexical_absolute(path)
    relative = _lexical_relative_path(path, repo_root)
    committed = subprocess.run(
        ["git", "show", f"{checkpoint}:{relative}"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    _require(
        committed.returncode == 0,
        f"{label} checkpoint blob cannot be read.",
    )
    _require_lexical_directory_chain(repo_root, path.parent, f"{label} parent")
    _require_regular_unlinked_file(path, label)
    lease = _ReadOnlyFileLease(path, label)
    try:
        inspected = os.fstat(lease.fileno())
        path_inspected = path.lstat()
        _exact_equal(
            (int(inspected.st_dev), int(inspected.st_ino)),
            (int(path_inspected.st_dev), int(path_inspected.st_ino)),
            f"{label} leased identity",
        )
        _exact_equal(int(inspected.st_nlink), 1, f"{label} link count")
        live_bytes = lease.read_bytes()
        _exact_equal(live_bytes, committed.stdout, f"{label} checkpoint bytes")
        return {
            "path": relative,
            "sha256": hashlib.sha256(live_bytes).hexdigest(),
            "size_bytes": int(inspected.st_size),
        }
    finally:
        lease.close()


def _protocol_binding(protocol: FrozenProtocol) -> dict[str, Any]:
    if protocol.authority_file_receipts is not None:
        files = copy.deepcopy(protocol.authority_file_receipts)
        _require_exact_keys(
            files,
            {
                "source_manifest",
                "gate",
                "evaluator",
                "imported_evaluators",
                "gpu_runtime",
            },
            "verified protocol authority receipts",
        )
        return {
            "pre_scoring_commit": PRE_SCORING_COMMIT,
            "pretraining_commit": protocol.pretraining_commit,
            **files,
        }
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
    expected_prior_pretraining_commit: str | None = None,
) -> None:
    binding = receipt.get("protocol")
    _require(isinstance(binding, Mapping), "Stage receipt protocol binding is malformed.")
    expected = _protocol_binding(protocol)
    if not allow_prior_pretraining_commit:
        _require(
            expected_prior_pretraining_commit is None,
            "An exact prior checkpoint requires prior-commit validation mode.",
        )
        _exact_equal(binding, expected, "stage protocol binding")
        return
    actual_commit = binding.get("pretraining_commit")
    _require(
        _is_lower_hex(actual_commit, 40),
        "Prior stage pretraining commit is malformed.",
    )
    assert isinstance(actual_commit, str)
    if expected_prior_pretraining_commit is not None:
        _require(
            _is_lower_hex(expected_prior_pretraining_commit, 40),
            "Expected prior stage checkpoint is malformed.",
        )
        _exact_equal(
            actual_commit,
            expected_prior_pretraining_commit,
            "shared prior stage pretraining checkpoint",
        )
    if actual_commit == protocol.pretraining_commit:
        _exact_equal(binding, expected, "stage protocol binding")
        return
    actual_without_commit = dict(binding)
    expected_without_commit = dict(expected)
    actual_without_commit.pop("pretraining_commit", None)
    expected_without_commit.pop("pretraining_commit", None)
    _exact_equal(
        actual_without_commit,
        expected_without_commit,
        "prior stage protocol binding",
    )
    exists = _run_git(
        protocol.repo_root,
        ["cat-file", "-e", f"{actual_commit}^{{commit}}"],
    )
    _require(exists.returncode == 0, "Prior stage pretraining checkpoint is unavailable.")
    lower_bound = _run_git(
        protocol.repo_root,
        ["merge-base", "--is-ancestor", PRE_SCORING_COMMIT, actual_commit],
    )
    _require(
        lower_bound.returncode in {0, 1},
        "Prior stage lower-bound ancestry check failed.",
    )
    _require(
        lower_bound.returncode == 0,
        "Prior stage pretraining checkpoint predates the frozen protocol checkpoint.",
    )
    ancestor = _run_git(
        protocol.repo_root,
        ["merge-base", "--is-ancestor", actual_commit, protocol.pretraining_commit],
    )
    _require(
        ancestor.returncode in {0, 1},
        "Prior stage upper-bound ancestry check failed.",
    )
    _require(
        ancestor.returncode == 0,
        "Prior stage pretraining checkpoint is not an ancestor of the current checkpoint.",
    )
    unchanged = _run_git(
        protocol.repo_root,
        [
            "diff",
            "--quiet",
            actual_commit,
            protocol.pretraining_commit,
            "--",
            *TRAINING_CHECKPOINT_PATHS,
        ],
    )
    _require(
        unchanged.returncode == 0,
        "Training implementation changed after the prior stage checkpoint.",
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


@contextmanager
def _lease_scout_component_config(
    protocol: FrozenProtocol,
    component: ComponentPaths,
) -> Iterator[dict[str, Any]]:
    """Bind every Scout config use to its frozen wrapper and helper bytes."""

    if component.name in SCOUT_NEW_COMPONENTS:
        expected_path = protocol.scout_config_paths[component.name]
        _exact_equal(
            _lexical_absolute(component.config),
            _lexical_absolute(expected_path),
            f"{component.name} Scout config path",
        )
        experiment_files = protocol.source_manifest["experiment_files"]
        wrapper_relative = f"configs/r1_{component.name}_d8_t6000.py"
        wrapper_receipt = experiment_files[wrapper_relative]
        base_receipt = experiment_files["configs/base_d8.py"]
        entries = {
            "base_d8": {
                "path": _lexical_relative_path(
                    protocol.experiment_dir / "configs/base_d8.py",
                    protocol.repo_root,
                ),
                "sha256": base_receipt["sha256"],
                "size_bytes": base_receipt["size_bytes"],
            },
            "config": {
                "path": _lexical_relative_path(
                    expected_path, protocol.repo_root
                ),
                "sha256": wrapper_receipt["sha256"],
                "size_bytes": wrapper_receipt["size_bytes"],
            },
        }
        with _lease_frozen_manifest_artifacts(
            protocol,
            entries,
            f"{component.name} Scout config",
        ):
            config = copy.deepcopy(protocol.scout_configs[component.name])
            validate_component_config(component.name, config)
            yield config
        return

    _exact_equal(component.name, "xerxes", "reused Scout component")
    frozen = protocol.source_manifest["reused_xerxes_component"]
    entries = {
        "base_config": frozen["base_config"],
        "config": frozen["config"],
    }
    with _lease_frozen_manifest_artifacts(
        protocol,
        entries,
        "reused Xerxes config",
    ) as paths:
        _exact_equal(
            _lexical_absolute(component.config),
            _lexical_absolute(paths["config"]),
            "reused Xerxes config path",
        )
        config = _load_config(paths["config"], "xerxes Scout config")
        validate_component_config(component.name, config)
        yield config


def _validate_scout_component(
    protocol: FrozenProtocol,
    component: ComponentPaths,
    expected: ExpectedCohort,
) -> tuple[np.ndarray, dict[str, Any]]:
    _require(component.name in ALL_COMPONENTS, f"Unknown Scout component: {component.name}")
    _require_regular_output_file(
        protocol,
        component.result,
        f"{component.name} Scout result",
    )
    _require_regular_output_file(
        protocol,
        component.predictions,
        f"{component.name} Scout predictions",
    )
    with _lease_scout_component_config(protocol, component) as config:
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


def _scout_destination_receipt(
    protocol: FrozenProtocol,
    component: ComponentPaths,
) -> dict[str, Any]:
    contract = protocol.source_manifest["new_scout_outputs"][component.name]
    _exact_equal(
        contract.get("must_be_absent_before_run"),
        True,
        f"{component.name} Scout absence contract",
    )
    expected = {
        "result": {
            "path": _lexical_relative_path(component.result, protocol.repo_root),
            "absent": True,
        },
        "predictions": {
            "path": _lexical_relative_path(component.predictions, protocol.repo_root),
            "absent": True,
        },
    }
    _exact_equal(
        contract.get("results_path"),
        expected["result"]["path"],
        f"{component.name} Scout result destination",
    )
    _exact_equal(
        contract.get("predictions_path"),
        expected["predictions"]["path"],
        f"{component.name} Scout prediction destination",
    )
    for label, path in (
        ("result", component.result),
        ("predictions", component.predictions),
    ):
        _prepare_output_destination_parent(
            protocol,
            path,
            f"{component.name} Scout {label}",
            create_direct_parent=False,
        )
    return expected


def _validate_prior_finalized_seal(
    protocol: FrozenProtocol,
    run_order: Sequence[str],
    component_name: str,
    prior_seal_receipt_path: Path | None,
    prior_seal_receipt_sha256: str | None,
    *,
    confirmation_pretraining_receipt_path: Path | None = None,
    confirmation_pretraining_receipt_sha256: str | None = None,
    allow_prior_pretraining_commit: bool = False,
    expected_prior_pretraining_commit: str | None = None,
) -> dict[str, str] | None:
    """Validate the complete fixed predecessor chain for one component run."""

    _require(component_name in run_order, f"Unknown sequential component: {component_name}")
    index = tuple(run_order).index(component_name)
    if index == 0:
        _require(
            prior_seal_receipt_path is None and prior_seal_receipt_sha256 is None,
            f"{component_name} is first and may not bind a predecessor seal.",
        )
        return None
    _require(
        prior_seal_receipt_path is not None and prior_seal_receipt_sha256 is not None,
        f"{component_name} requires the immediately preceding finalized seal.",
    )
    assert prior_seal_receipt_path is not None
    assert prior_seal_receipt_sha256 is not None
    previous = tuple(run_order)[index - 1]
    confirmation = tuple(run_order) == tuple(CONFIRMATION_RUN_ORDER)
    prefix = (
        f"confirmation-seal-{previous}"
        if confirmation
        else f"scout-seal-{previous}"
    )
    stage = "seal-confirmation-component" if confirmation else "seal-scout-component"
    path = _lexical_absolute(prior_seal_receipt_path)
    seal = _load_bound_receipt(
        path,
        prior_seal_receipt_sha256,
        expected_stage=stage,
        receipt_dir=_canonical_receipt_dir(protocol),
        expected_prefix=prefix,
    )
    _validate_stage_receipt_schema(seal, stage)
    _exact_equal(seal.get("passed"), True, f"{previous} predecessor seal passage")
    _exact_equal(seal.get("state"), "SEALED", f"{previous} predecessor seal state")
    _exact_equal(seal.get("component"), previous, f"{previous} predecessor component")
    _validate_protocol_binding(
        seal,
        protocol,
        allow_prior_pretraining_commit=allow_prior_pretraining_commit,
        expected_prior_pretraining_commit=expected_prior_pretraining_commit,
    )
    if confirmation:
        _require(
            confirmation_pretraining_receipt_path is not None
            and confirmation_pretraining_receipt_sha256 is not None,
            "Confirmation predecessor validation requires pretraining provenance.",
        )
        assert confirmation_pretraining_receipt_path is not None
        assert confirmation_pretraining_receipt_sha256 is not None
        _exact_equal(
            seal.get("confirmation_pretraining_receipt"),
            _confirmation_pretraining_binding(
                protocol,
                confirmation_pretraining_receipt_path,
                confirmation_pretraining_receipt_sha256,
            ),
            f"{previous} predecessor confirmation pretraining binding",
        )
    pre_run_binding = _validate_binding_schema(
        seal.get("pre_run_absence_receipt"),
        f"{previous} predecessor pre-run receipt",
    )
    pre_run_path = _lexical_repo_path(
        protocol.repo_root, str(pre_run_binding["path"])
    )
    completion_binding = _validate_file_receipt_schema(
        seal.get("run_completion_receipt"),
        f"{previous} predecessor completion receipt",
    )
    completion_path = _lexical_repo_path(
        protocol.repo_root, str(completion_binding["path"])
    )
    if confirmation:
        pre_run = _validate_confirmation_pre_run_absence_receipt(
            protocol,
            default_confirmation_component_paths(protocol, previous),
            pre_run_path,
            str(pre_run_binding["sha256"]),
            confirmation_pretraining_receipt_path,
            confirmation_pretraining_receipt_sha256,
        )
    else:
        pre_run = _validate_scout_pre_run_absence_receipt(
            protocol,
            default_scout_component_paths(protocol, previous),
            pre_run_path,
            str(pre_run_binding["sha256"]),
            allow_prior_pretraining_commit=allow_prior_pretraining_commit,
            expected_prior_pretraining_commit=expected_prior_pretraining_commit,
        )
    component = (
        default_confirmation_component_paths(protocol, previous)
        if confirmation
        else default_scout_component_paths(protocol, previous)
    )
    with _lease_validated_component_outputs(
        protocol,
        component,
        pre_run_path,
        str(pre_run_binding["sha256"]),
        completion_path,
        str(completion_binding["sha256"]),
        confirmation=confirmation,
        confirmation_pretraining_receipt_path=(
            confirmation_pretraining_receipt_path
        ),
        confirmation_pretraining_receipt_sha256=(
            confirmation_pretraining_receipt_sha256
        ),
    ) as (consumption_claim, completion_receipt):
        _exact_equal(
            seal.get("run_consumption_claim"),
            consumption_claim,
            f"{previous} predecessor run consumption claim",
        )
        _exact_equal(
            seal.get("run_completion_receipt"),
            completion_receipt,
            f"{previous} predecessor run completion receipt",
        )
        _exact_equal(
            seal.get("prior_finalized_seal"),
            pre_run.get("prior_finalized_seal"),
            f"{previous} seal/pre-run predecessor chain",
        )
        if confirmation:
            assert confirmation_pretraining_receipt_path is not None
            assert confirmation_pretraining_receipt_sha256 is not None
            pretraining = _validate_confirmation_pretraining_receipt(
                protocol,
                confirmation_pretraining_receipt_path,
                confirmation_pretraining_receipt_sha256,
            )
            expected = build_confirmation_expected_cohort(
                protocol,
                confirmation_input_receipt=confirmation_pretraining_receipt_path,
                confirmation_input_receipt_sha256=(
                    confirmation_pretraining_receipt_sha256
                ),
            )
            _, artifact = _validate_confirmation_component(
                protocol,
                component,
                expected,
                pretraining,
            )
        else:
            expected = build_scout_expected_cohort(protocol)
            _, artifact = _validate_scout_component(protocol, component, expected)
        _exact_equal(
            seal.get("cohort"),
            _cohort_receipt(expected),
            f"{previous} predecessor cohort",
        )
        _exact_equal(
            seal.get("artifact"),
            artifact,
            f"{previous} predecessor artifact",
        )
        _exact_equal(
            seal.get("gpu_folds_verified"),
            len(expected.folds),
            f"{previous} predecessor GPU folds",
        )
    return {
        "component": previous,
        "path": _lexical_relative_path(path, protocol.repo_root),
        "sha256": prior_seal_receipt_sha256,
    }


def _component_training_consumption_claim_path(
    protocol: FrozenProtocol,
    component_name: str,
    *,
    confirmation: bool,
) -> Path:
    family = "confirmation" if confirmation else "scout"
    return _canonical_receipt_dir(protocol) / (
        f".{family}-train-{component_name}.consumed.json"
    )


def _component_training_completion_prefix(
    component_name: str,
    *,
    confirmation: bool,
) -> str:
    family = "confirmation" if confirmation else "scout"
    return f"{family}-train-{component_name}-completion"


def _normalize_component_output_reservations(
    protocol: FrozenProtocol,
    component: ComponentPaths,
    value: Mapping[str, Any],
    *,
    require_empty: bool,
) -> dict[str, dict[str, Any]]:
    reservations = _require_exact_keys(
        value,
        {"predictions", "result"},
        f"{component.name} output reservations",
    )
    normalized: dict[str, dict[str, Any]] = {}
    for label, expected_path in (
        ("predictions", component.predictions),
        ("result", component.result),
    ):
        item = _require_exact_keys(
            reservations.get(label),
            {"path", "device", "inode"},
            f"{component.name} {label} reservation",
        )
        path_value = item.get("path")
        _require(
            isinstance(path_value, str) and bool(path_value),
            f"{component.name} {label} reservation path is malformed.",
        )
        candidate = Path(path_value)
        path = (
            _lexical_absolute(candidate)
            if candidate.is_absolute()
            else _lexical_repo_path(protocol.repo_root, candidate)
        )
        _exact_equal(path, expected_path, f"{component.name} {label} reservation path")
        for name in ("device", "inode"):
            _require(
                type(item.get(name)) is int and item[name] >= 0,
                f"{component.name} {label} reservation {name} is malformed.",
            )
        _require_regular_output_file(
            protocol,
            path,
            f"{component.name} reserved {label} output",
        )
        inspected = path.lstat()
        _exact_equal(
            int(inspected.st_dev), item["device"], f"{component.name} {label} device"
        )
        _exact_equal(
            int(inspected.st_ino), item["inode"], f"{component.name} {label} inode"
        )
        if require_empty:
            _exact_equal(
                int(inspected.st_size), 0, f"{component.name} {label} reserved size"
            )
        normalized[label] = {
            "path": _lexical_relative_path(path, protocol.repo_root),
            "device": item["device"],
            "inode": item["inode"],
        }
    return normalized


def _component_training_consumption_payload(
    protocol: FrozenProtocol,
    component: ComponentPaths,
    pre_run_receipt_path: Path,
    pre_run_receipt_sha256: str,
    output_reservations: Mapping[str, Any],
    *,
    confirmation: bool,
    confirmation_pretraining_receipt_path: Path | None = None,
    confirmation_pretraining_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    stage = (
        "consume-confirmation-component-run"
        if confirmation
        else "consume-scout-component-run"
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "stage": stage,
        "state": "CONSUMED",
        "component": component.name,
        "protocol": _protocol_binding(protocol),
        "python_import_policy": {
            "dont_write_bytecode": True,
            "isolated_empty_pycache_prefix": True,
        },
        "pre_run_receipt": {
            "path": _lexical_relative_path(
                _lexical_absolute(pre_run_receipt_path), protocol.repo_root
            ),
            "sha256": _require_sha256(
                pre_run_receipt_sha256, "training pre-run receipt hash"
            ),
        },
        "output_reservations": _normalize_component_output_reservations(
            protocol,
            component,
            output_reservations,
            require_empty=False,
        ),
    }
    if confirmation:
        _require(
            confirmation_pretraining_receipt_path is not None
            and confirmation_pretraining_receipt_sha256 is not None,
            "Confirmation training consumption requires pretraining provenance.",
        )
        assert confirmation_pretraining_receipt_path is not None
        assert confirmation_pretraining_receipt_sha256 is not None
        payload["confirmation_pretraining_receipt"] = (
            _confirmation_pretraining_binding(
                protocol,
                _lexical_absolute(confirmation_pretraining_receipt_path),
                confirmation_pretraining_receipt_sha256,
            )
        )
    else:
        _require(
            confirmation_pretraining_receipt_path is None
            and confirmation_pretraining_receipt_sha256 is None,
            "Scout training consumption may not bind confirmation provenance.",
        )
    return payload


def claim_component_training_consumption(
    protocol: FrozenProtocol,
    component: ComponentPaths,
    pre_run_receipt_path: Path,
    pre_run_receipt_sha256: str,
    output_reservations: Mapping[str, Any],
    *,
    confirmation: bool,
    confirmation_pretraining_receipt_path: Path | None = None,
    confirmation_pretraining_receipt_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Consume one finalized pre-run authorization exactly once before fitting."""

    expected_component = (
        default_confirmation_component_paths(protocol, component.name)
        if confirmation
        else default_scout_component_paths(protocol, component.name)
    )
    _exact_equal(component, expected_component, f"{component.name} training paths")
    if confirmation:
        _require(component.name in ALL_COMPONENTS, "Unknown confirmation component run.")
        expected_stage = "claim-confirmation-component-run"
        prefix = f"confirmation-pre-run-{component.name}"
    else:
        _require(
            component.name in SCOUT_NEW_COMPONENTS,
            "Only the four new Scout components may consume a run.",
        )
        expected_stage = "claim-scout-component-run"
        prefix = f"scout-pre-run-{component.name}"

    pre_run_receipt_path = _lexical_absolute(pre_run_receipt_path)
    pre_run = _load_bound_receipt(
        pre_run_receipt_path,
        pre_run_receipt_sha256,
        expected_stage=expected_stage,
        receipt_dir=_canonical_receipt_dir(protocol),
        expected_prefix=prefix,
    )
    _validate_stage_receipt_schema(pre_run, expected_stage)
    _exact_equal(pre_run.get("passed"), True, f"{component.name} pre-run passage")
    _exact_equal(
        pre_run.get("state"),
        "ABSENCE_PROVEN",
        f"{component.name} pre-run state",
    )
    _exact_equal(
        pre_run.get("component"), component.name, f"{component.name} pre-run component"
    )
    _validate_protocol_binding(pre_run, protocol)
    expected_destinations = (
        _confirmation_component_destination_receipt(protocol, component)
        if confirmation
        else _scout_destination_receipt(protocol, component)
    )
    _exact_equal(
        pre_run.get("destinations"),
        expected_destinations,
        f"{component.name} pre-run destinations",
    )
    if confirmation:
        assert confirmation_pretraining_receipt_path is not None
        assert confirmation_pretraining_receipt_sha256 is not None
        _exact_equal(
            pre_run.get("confirmation_pretraining_receipt"),
            _confirmation_pretraining_binding(
                protocol,
                _lexical_absolute(confirmation_pretraining_receipt_path),
                confirmation_pretraining_receipt_sha256,
            ),
            f"{component.name} pre-run confirmation pretraining binding",
        )
    normalized_reservations = _normalize_component_output_reservations(
        protocol,
        component,
        output_reservations,
        require_empty=True,
    )

    claim_path = _component_training_consumption_claim_path(
        protocol, component.name, confirmation=confirmation
    )
    _require_frozen_python_runtime()
    _require_lexical_directory_chain(
        protocol.repo_root,
        claim_path.parent,
        f"{component.name} training consumption claim parent",
    )
    payload = _component_training_consumption_payload(
        protocol,
        component,
        pre_run_receipt_path,
        pre_run_receipt_sha256,
        normalized_reservations,
        confirmation=confirmation,
        confirmation_pretraining_receipt_path=(
            confirmation_pretraining_receipt_path
        ),
        confirmation_pretraining_receipt_sha256=(
            confirmation_pretraining_receipt_sha256
        ),
    )
    try:
        with claim_path.open("xb") as stream:
            stream.write(_receipt_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise EnderEnsembleEvaluationError(
            f"{component.name} training authorization was already consumed."
        ) from error
    _require_regular_unlinked_file(
        claim_path, f"{component.name} training consumption claim"
    )
    _exact_equal(
        _load_json(claim_path, f"{component.name} training consumption claim"),
        payload,
        f"{component.name} training consumption claim",
    )
    return claim_path, payload


def _validate_component_training_consumption_claim(
    protocol: FrozenProtocol,
    component: ComponentPaths,
    pre_run_receipt_path: Path,
    pre_run_receipt_sha256: str,
    *,
    confirmation: bool,
    confirmation_pretraining_receipt_path: Path | None = None,
    confirmation_pretraining_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    claim_path = _component_training_consumption_claim_path(
        protocol, component.name, confirmation=confirmation
    )
    _require_lexical_directory_chain(
        protocol.repo_root,
        claim_path.parent,
        f"{component.name} training consumption claim parent",
    )
    _require_regular_unlinked_file(
        claim_path, f"{component.name} training consumption claim"
    )
    actual = _load_json(
        claim_path, f"{component.name} training consumption claim"
    )
    _require(
        isinstance(actual, Mapping),
        f"{component.name} training consumption claim is malformed.",
    )
    reservations = actual.get("output_reservations")
    _require(
        isinstance(reservations, Mapping),
        f"{component.name} output reservations are malformed.",
    )
    expected = _component_training_consumption_payload(
        protocol,
        component,
        pre_run_receipt_path,
        pre_run_receipt_sha256,
        reservations,
        confirmation=confirmation,
        confirmation_pretraining_receipt_path=(
            confirmation_pretraining_receipt_path
        ),
        confirmation_pretraining_receipt_sha256=(
            confirmation_pretraining_receipt_sha256
        ),
    )
    _exact_equal(
        actual,
        expected,
        f"{component.name} training consumption claim",
    )
    return _file_receipt(claim_path, protocol.repo_root)


def claim_component_training_completion(
    protocol: FrozenProtocol,
    component: ComponentPaths,
    *,
    confirmation: bool,
) -> Path:
    """Claim the one completion receipt before any authorized model code runs."""

    expected_component = (
        default_confirmation_component_paths(protocol, component.name)
        if confirmation
        else default_scout_component_paths(protocol, component.name)
    )
    _exact_equal(component, expected_component, f"{component.name} completion paths")
    allowed = ALL_COMPONENTS if confirmation else SCOUT_NEW_COMPONENTS
    _require(component.name in allowed, "Unknown component completion claim.")
    _, claim_path = _claim_canonical_receipt_prefix(
        protocol,
        _canonical_receipt_dir(protocol),
        _component_training_completion_prefix(
            component.name, confirmation=confirmation
        ),
    )
    return claim_path


def _normalize_component_output_completions(
    protocol: FrozenProtocol,
    component: ComponentPaths,
    value: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    outputs = _require_exact_keys(
        value,
        {"predictions", "result"},
        f"{component.name} completed outputs",
    )
    normalized: dict[str, dict[str, Any]] = {}
    for label, expected_path in (
        ("predictions", component.predictions),
        ("result", component.result),
    ):
        item = _require_exact_keys(
            outputs.get(label),
            {"path", "device", "inode", "size_bytes", "sha256"},
            f"{component.name} completed {label}",
        )
        path_value = item.get("path")
        _require(
            isinstance(path_value, str) and bool(path_value),
            f"{component.name} completed {label} path is malformed.",
        )
        candidate = Path(path_value)
        path = (
            _lexical_absolute(candidate)
            if candidate.is_absolute()
            else _lexical_repo_path(protocol.repo_root, candidate)
        )
        _exact_equal(path, expected_path, f"{component.name} completed {label} path")
        for name in ("device", "inode"):
            _require(
                type(item.get(name)) is int and item[name] >= 0,
                f"{component.name} completed {label} {name} is malformed.",
            )
        _require(
            type(item.get("size_bytes")) is int and item["size_bytes"] > 0,
            f"{component.name} completed {label} size is malformed.",
        )
        digest = _require_sha256(
            item.get("sha256"), f"{component.name} completed {label} hash"
        )
        _require_regular_output_file(
            protocol,
            path,
            f"{component.name} completed {label} output",
        )
        inspected = path.lstat()
        _exact_equal(
            int(inspected.st_dev),
            item["device"],
            f"{component.name} completed {label} device",
        )
        _exact_equal(
            int(inspected.st_ino),
            item["inode"],
            f"{component.name} completed {label} inode",
        )
        _exact_equal(
            int(inspected.st_size),
            item["size_bytes"],
            f"{component.name} completed {label} size",
        )
        normalized[label] = {
            "path": _lexical_relative_path(path, protocol.repo_root),
            "device": item["device"],
            "inode": item["inode"],
            "size_bytes": item["size_bytes"],
            "sha256": digest,
        }
    return normalized


def _component_training_completion_payload(
    protocol: FrozenProtocol,
    component: ComponentPaths,
    pre_run_receipt_path: Path,
    pre_run_receipt_sha256: str,
    consumption_claim: Mapping[str, Any],
    output_reservations: Mapping[str, Any],
    outputs: Mapping[str, Any],
    *,
    confirmation: bool,
    confirmation_pretraining_receipt_path: Path | None = None,
    confirmation_pretraining_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    family = "confirmation" if confirmation else "scout"
    claim_receipt = _validate_file_receipt_schema(
        consumption_claim,
        f"{component.name} completion consumption claim",
    )
    _exact_equal(
        claim_receipt.get("path"),
        _lexical_relative_path(
            _component_training_consumption_claim_path(
                protocol, component.name, confirmation=confirmation
            ),
            protocol.repo_root,
        ),
        f"{component.name} completion consumption claim path",
    )
    normalized_reservations = _normalize_component_output_reservations(
        protocol,
        component,
        output_reservations,
        require_empty=False,
    )
    normalized_outputs = _normalize_component_output_completions(
        protocol,
        component,
        outputs,
    )
    for label in ("predictions", "result"):
        _exact_equal(
            {
                key: normalized_outputs[label][key]
                for key in ("path", "device", "inode")
            },
            normalized_reservations[label],
            f"{component.name} completed {label} reservation",
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "stage": f"complete-{family}-component-run",
        "state": "OUTPUTS_FINALIZED",
        "passed": True,
        "protocol": _protocol_binding(protocol),
        "component": component.name,
        "pre_run_receipt": {
            "path": _lexical_relative_path(
                _lexical_absolute(pre_run_receipt_path), protocol.repo_root
            ),
            "sha256": _require_sha256(
                pre_run_receipt_sha256, "completion pre-run receipt hash"
            ),
        },
        "run_consumption_claim": dict(claim_receipt),
        "output_reservations": normalized_reservations,
        "outputs": normalized_outputs,
    }
    if confirmation:
        _require(
            confirmation_pretraining_receipt_path is not None
            and confirmation_pretraining_receipt_sha256 is not None,
            "Confirmation completion requires pretraining provenance.",
        )
        assert confirmation_pretraining_receipt_path is not None
        assert confirmation_pretraining_receipt_sha256 is not None
        payload["confirmation_pretraining_receipt"] = (
            _confirmation_pretraining_binding(
                protocol,
                _lexical_absolute(confirmation_pretraining_receipt_path),
                confirmation_pretraining_receipt_sha256,
            )
        )
    else:
        _require(
            confirmation_pretraining_receipt_path is None
            and confirmation_pretraining_receipt_sha256 is None,
            "Scout completion may not bind confirmation provenance.",
        )
    return payload


def complete_component_training_consumption(
    protocol: FrozenProtocol,
    component: ComponentPaths,
    pre_run_receipt_path: Path,
    pre_run_receipt_sha256: str,
    output_artifacts: Mapping[str, Any],
    completion_claim_path: Path,
    *,
    confirmation: bool,
    confirmation_pretraining_receipt_path: Path | None = None,
    confirmation_pretraining_receipt_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Finalize exact output bytes while their original handles remain held."""

    claim_receipt = _validate_component_training_consumption_claim(
        protocol,
        component,
        pre_run_receipt_path,
        pre_run_receipt_sha256,
        confirmation=confirmation,
        confirmation_pretraining_receipt_path=(
            confirmation_pretraining_receipt_path
        ),
        confirmation_pretraining_receipt_sha256=(
            confirmation_pretraining_receipt_sha256
        ),
    )
    consumption_payload = _load_json(
        _component_training_consumption_claim_path(
            protocol, component.name, confirmation=confirmation
        ),
        f"{component.name} training consumption claim",
    )
    receipt = _component_training_completion_payload(
        protocol,
        component,
        pre_run_receipt_path,
        pre_run_receipt_sha256,
        claim_receipt,
        consumption_payload.get("output_reservations"),
        output_artifacts,
        confirmation=confirmation,
        confirmation_pretraining_receipt_path=(
            confirmation_pretraining_receipt_path
        ),
        confirmation_pretraining_receipt_sha256=(
            confirmation_pretraining_receipt_sha256
        ),
    )
    prefix = _component_training_completion_prefix(
        component.name, confirmation=confirmation
    )
    path = _write_claimed_content_addressed_receipt(
        _canonical_receipt_dir(protocol),
        prefix,
        _lexical_absolute(completion_claim_path),
        receipt,
    )
    return path, receipt


def _validate_component_training_completion_receipt(
    protocol: FrozenProtocol,
    component: ComponentPaths,
    pre_run_receipt_path: Path,
    pre_run_receipt_sha256: str,
    completion_receipt_path: Path,
    completion_receipt_sha256: str,
    *,
    confirmation: bool,
    confirmation_pretraining_receipt_path: Path | None = None,
    confirmation_pretraining_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    prefix = _component_training_completion_prefix(
        component.name, confirmation=confirmation
    )
    stage = (
        "complete-confirmation-component-run"
        if confirmation
        else "complete-scout-component-run"
    )
    path = _lexical_absolute(completion_receipt_path)
    actual = _load_bound_receipt(
        path,
        completion_receipt_sha256,
        expected_stage=stage,
        receipt_dir=_canonical_receipt_dir(protocol),
        expected_prefix=prefix,
    )
    claim_receipt = _validate_component_training_consumption_claim(
        protocol,
        component,
        pre_run_receipt_path,
        pre_run_receipt_sha256,
        confirmation=confirmation,
        confirmation_pretraining_receipt_path=(
            confirmation_pretraining_receipt_path
        ),
        confirmation_pretraining_receipt_sha256=(
            confirmation_pretraining_receipt_sha256
        ),
    )
    consumption_payload = _load_json(
        _component_training_consumption_claim_path(
            protocol, component.name, confirmation=confirmation
        ),
        f"{component.name} training consumption claim",
    )
    expected = _component_training_completion_payload(
        protocol,
        component,
        pre_run_receipt_path,
        pre_run_receipt_sha256,
        claim_receipt,
        consumption_payload.get("output_reservations"),
        actual.get("outputs"),
        confirmation=confirmation,
        confirmation_pretraining_receipt_path=(
            confirmation_pretraining_receipt_path
        ),
        confirmation_pretraining_receipt_sha256=(
            confirmation_pretraining_receipt_sha256
        ),
    )
    _exact_equal(
        actual,
        expected,
        f"{component.name} training completion receipt",
    )
    return _file_receipt(path, protocol.repo_root)


@contextmanager
def _lease_validated_component_outputs(
    protocol: FrozenProtocol,
    component: ComponentPaths,
    pre_run_receipt_path: Path,
    pre_run_receipt_sha256: str,
    completion_receipt_path: Path,
    completion_receipt_sha256: str,
    *,
    confirmation: bool,
    confirmation_pretraining_receipt_path: Path | None = None,
    confirmation_pretraining_receipt_sha256: str | None = None,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Hold completion, marker, and exact output bytes through artifact reads."""

    completion_receipt = _validate_component_training_completion_receipt(
        protocol,
        component,
        pre_run_receipt_path,
        pre_run_receipt_sha256,
        completion_receipt_path,
        completion_receipt_sha256,
        confirmation=confirmation,
        confirmation_pretraining_receipt_path=(
            confirmation_pretraining_receipt_path
        ),
        confirmation_pretraining_receipt_sha256=(
            confirmation_pretraining_receipt_sha256
        ),
    )
    claim_receipt = _validate_component_training_consumption_claim(
        protocol,
        component,
        pre_run_receipt_path,
        pre_run_receipt_sha256,
        confirmation=confirmation,
        confirmation_pretraining_receipt_path=(
            confirmation_pretraining_receipt_path
        ),
        confirmation_pretraining_receipt_sha256=(
            confirmation_pretraining_receipt_sha256
        ),
    )
    claim_path = _component_training_consumption_claim_path(
        protocol, component.name, confirmation=confirmation
    )
    completion_path = _lexical_absolute(completion_receipt_path)
    completion_prefix = _component_training_completion_prefix(
        component.name, confirmation=confirmation
    )
    receipt_dir = _canonical_receipt_dir(protocol)
    completion_claim_path = receipt_dir / f".{completion_prefix}.claimed.json"
    completion_finalization_path = (
        receipt_dir / f".{completion_prefix}.finalized.json"
    )
    leases: list[_ReadOnlyFileLease] = []
    try:
        completion_claim_lease = _ReadOnlyFileLease(
            completion_claim_path,
            f"{component.name} training completion claim",
        )
        leases.append(completion_claim_lease)
        _exact_equal(
            completion_claim_lease.read_bytes(),
            _receipt_bytes(_claim_payload(completion_prefix)),
            f"{component.name} leased completion claim",
        )
        completion_lease = _ReadOnlyFileLease(
            completion_path,
            f"{component.name} training completion receipt",
        )
        leases.append(completion_lease)
        completion_bytes = completion_lease.read_bytes()
        _exact_equal(
            hashlib.sha256(completion_bytes).hexdigest(),
            completion_receipt_sha256,
            f"{component.name} leased completion receipt hash",
        )
        _exact_equal(
            len(completion_bytes),
            completion_receipt.get("size_bytes"),
            f"{component.name} leased completion receipt size",
        )
        completion_finalization_lease = _ReadOnlyFileLease(
            completion_finalization_path,
            f"{component.name} training completion finalization",
        )
        leases.append(completion_finalization_lease)
        expected_finalization = _finalization_payload(
            completion_prefix,
            completion_claim_path,
            completion_claim_lease.sha256(),
            completion_path,
            completion_receipt_sha256,
        )
        _exact_equal(
            completion_finalization_lease.read_bytes(),
            _receipt_bytes(expected_finalization),
            f"{component.name} leased completion finalization",
        )
        claim_lease = _ReadOnlyFileLease(
            claim_path, f"{component.name} training consumption claim"
        )
        leases.append(claim_lease)
        claim_bytes = claim_lease.read_bytes()
        _exact_equal(
            hashlib.sha256(claim_bytes).hexdigest(),
            claim_receipt.get("sha256"),
            f"{component.name} leased training consumption claim hash",
        )
        _exact_equal(
            len(claim_bytes),
            claim_receipt.get("size_bytes"),
            f"{component.name} leased training consumption claim size",
        )
        claim_stat = os.fstat(claim_lease.fileno())
        claim_path_stat = claim_path.lstat()
        _exact_equal(
            (int(claim_stat.st_dev), int(claim_stat.st_ino)),
            (int(claim_path_stat.st_dev), int(claim_path_stat.st_ino)),
            f"{component.name} training consumption claim identity",
        )
        try:
            claim_payload = json.loads(claim_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EnderEnsembleEvaluationError(
                f"{component.name} leased training consumption claim is invalid JSON."
            ) from error
        _require(
            isinstance(claim_payload, Mapping),
            f"{component.name} leased training consumption claim is malformed.",
        )
        expected_payload = _component_training_consumption_payload(
            protocol,
            component,
            pre_run_receipt_path,
            pre_run_receipt_sha256,
            claim_payload.get("output_reservations"),
            confirmation=confirmation,
            confirmation_pretraining_receipt_path=(
                confirmation_pretraining_receipt_path
            ),
            confirmation_pretraining_receipt_sha256=(
                confirmation_pretraining_receipt_sha256
            ),
        )
        _exact_equal(
            claim_payload,
            expected_payload,
            f"{component.name} leased training consumption claim",
        )
        try:
            completion_payload = json.loads(completion_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EnderEnsembleEvaluationError(
                f"{component.name} leased completion receipt is invalid JSON."
            ) from error
        _require(
            isinstance(completion_payload, Mapping),
            f"{component.name} leased completion receipt is malformed.",
        )
        expected_completion = _component_training_completion_payload(
            protocol,
            component,
            pre_run_receipt_path,
            pre_run_receipt_sha256,
            claim_receipt,
            expected_payload["output_reservations"],
            completion_payload.get("outputs"),
            confirmation=confirmation,
            confirmation_pretraining_receipt_path=(
                confirmation_pretraining_receipt_path
            ),
            confirmation_pretraining_receipt_sha256=(
                confirmation_pretraining_receipt_sha256
            ),
        )
        _exact_equal(
            completion_payload,
            expected_completion,
            f"{component.name} leased training completion receipt",
        )
        reservations = expected_payload["output_reservations"]
        completed_outputs = expected_completion["outputs"]
        for label, path in (
            ("predictions", component.predictions),
            ("result", component.result),
        ):
            _require_regular_output_file(
                protocol,
                path,
                f"{component.name} sealed {label} output",
            )
            output_lease = _ReadOnlyFileLease(
                path, f"{component.name} sealed {label} output"
            )
            leases.append(output_lease)
            leased_stat = os.fstat(output_lease.fileno())
            path_stat = path.lstat()
            _exact_equal(
                (int(leased_stat.st_dev), int(leased_stat.st_ino)),
                (
                    int(reservations[label]["device"]),
                    int(reservations[label]["inode"]),
                ),
                f"{component.name} leased {label} reservation identity",
            )
            _exact_equal(
                (int(path_stat.st_dev), int(path_stat.st_ino)),
                (int(leased_stat.st_dev), int(leased_stat.st_ino)),
                f"{component.name} sealed {label} path identity",
            )
            _exact_equal(
                int(leased_stat.st_nlink),
                1,
                f"{component.name} sealed {label} link count",
            )
            _exact_equal(
                int(leased_stat.st_size),
                completed_outputs[label]["size_bytes"],
                f"{component.name} sealed {label} completed size",
            )
            _exact_equal(
                output_lease.sha256(),
                completed_outputs[label]["sha256"],
                f"{component.name} sealed {label} completed hash",
            )
        yield claim_receipt, completion_receipt
    finally:
        for lease in reversed(leases):
            lease.close()


def claim_scout_component_run(
    protocol: FrozenProtocol,
    component_name: str,
    receipt_dir: Path,
    prior_seal_receipt_path: Path | None = None,
    prior_seal_receipt_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Durably prove both Scout outputs absent immediately before one run."""

    _require(
        component_name in SCOUT_NEW_COMPONENTS,
        "Only the four new Scout components may claim a run.",
    )
    prefix = f"scout-pre-run-{component_name}"
    canonical_dir, claim_path = _claim_canonical_receipt_prefix(
        protocol, receipt_dir, prefix
    )
    prior_seal = _validate_prior_finalized_seal(
        protocol,
        SCOUT_RUN_ORDER,
        component_name,
        prior_seal_receipt_path,
        prior_seal_receipt_sha256,
    )
    component = default_scout_component_paths(protocol, component_name)
    for label, path in (
        ("result", component.result),
        ("predictions", component.predictions),
    ):
        _prepare_output_destination_parent(
            protocol,
            path,
            f"{component_name} Scout {label}",
            create_direct_parent=True,
        )
        _require_absent_destination(
            path,
            f"{component_name} Scout {label} destination",
        )
    destinations = _scout_destination_receipt(protocol, component)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "stage": "claim-scout-component-run",
        "state": "ABSENCE_PROVEN",
        "passed": True,
        "protocol": _protocol_binding(protocol),
        "component": component_name,
        "prior_finalized_seal": prior_seal,
        "destinations": destinations,
    }
    path = _write_claimed_content_addressed_receipt(
        canonical_dir,
        prefix,
        claim_path,
        receipt,
    )
    return path, receipt


def _validate_scout_pre_run_absence_receipt(
    protocol: FrozenProtocol,
    component: ComponentPaths,
    path: Path,
    digest: str,
    *,
    allow_prior_pretraining_commit: bool = False,
    expected_prior_pretraining_commit: str | None = None,
) -> dict[str, Any]:
    prefix = f"scout-pre-run-{component.name}"
    path = _lexical_absolute(path)
    receipt = _load_bound_receipt(
        path,
        digest,
        expected_stage="claim-scout-component-run",
        receipt_dir=_canonical_receipt_dir(protocol),
        expected_prefix=prefix,
    )
    _validate_stage_receipt_schema(receipt, "claim-scout-component-run")
    _exact_equal(receipt.get("passed"), True, f"{component.name} pre-run passage")
    _exact_equal(
        receipt.get("state"),
        "ABSENCE_PROVEN",
        f"{component.name} pre-run state",
    )
    _exact_equal(
        receipt.get("component"),
        component.name,
        f"{component.name} pre-run component",
    )
    _validate_protocol_binding(
        receipt,
        protocol,
        allow_prior_pretraining_commit=allow_prior_pretraining_commit,
        expected_prior_pretraining_commit=expected_prior_pretraining_commit,
    )
    prior_value = receipt.get("prior_finalized_seal")
    prior_binding = _validate_prior_seal_binding_schema(
        prior_value, f"{component.name} Scout predecessor"
    )
    prior_path: Path | None = None
    prior_sha: str | None = None
    if prior_binding is not None:
        prior_path = _lexical_repo_path(protocol.repo_root, str(prior_binding["path"]))
        prior_sha = str(prior_binding["sha256"])
    _exact_equal(
        prior_value,
        _validate_prior_finalized_seal(
            protocol,
            SCOUT_RUN_ORDER,
            component.name,
            prior_path,
            prior_sha,
            allow_prior_pretraining_commit=allow_prior_pretraining_commit,
            expected_prior_pretraining_commit=expected_prior_pretraining_commit,
        ),
        f"{component.name} Scout predecessor binding",
    )
    _exact_equal(
        receipt.get("destinations"),
        _scout_destination_receipt(protocol, component),
        f"{component.name} pre-run destinations",
    )
    return receipt


def seal_scout_component(
    protocol: FrozenProtocol,
    component: ComponentPaths,
    receipt_dir: Path,
    pre_run_receipt_path: Path,
    pre_run_receipt_sha256: str,
    completion_receipt_path: Path,
    completion_receipt_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    """Seal one successful new Scout run without persisting any Ender metric."""

    _require(
        component.name in SCOUT_NEW_COMPONENTS,
        "Only the four new Scout components may be sealed.",
    )
    expected_paths = default_scout_component_paths(protocol, component.name)
    _exact_equal(component, expected_paths, f"{component.name} Scout destinations")
    prefix = f"scout-seal-{component.name}"
    canonical_dir, claim_path = _claim_canonical_receipt_prefix(
        protocol, receipt_dir, prefix
    )
    pre_run_receipt_path = _lexical_absolute(pre_run_receipt_path)
    pre_run_receipt = _validate_scout_pre_run_absence_receipt(
        protocol,
        component,
        pre_run_receipt_path,
        pre_run_receipt_sha256,
    )
    expected = build_scout_expected_cohort(protocol)
    with _lease_validated_component_outputs(
        protocol,
        component,
        pre_run_receipt_path,
        pre_run_receipt_sha256,
        _lexical_absolute(completion_receipt_path),
        completion_receipt_sha256,
        confirmation=False,
    ) as (consumption_claim, completion_receipt):
        _, artifact = _validate_scout_component(protocol, component, expected)
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "experiment": EXPERIMENT_NAME,
            "stage": "seal-scout-component",
            "state": "SEALED",
            "passed": True,
            "protocol": _protocol_binding(protocol),
            "component": component.name,
            "prior_finalized_seal": copy.deepcopy(
                pre_run_receipt.get("prior_finalized_seal")
            ),
            "pre_run_absence_receipt": {
                "path": _lexical_relative_path(
                    pre_run_receipt_path, protocol.repo_root
                ),
                "sha256": pre_run_receipt_sha256,
            },
            "run_consumption_claim": consumption_claim,
            "run_completion_receipt": completion_receipt,
            "cohort": _cohort_receipt(expected),
            "artifact": artifact,
            "gpu_folds_verified": len(expected.folds),
        }
        path = _write_claimed_content_addressed_receipt(
            canonical_dir,
            prefix,
            claim_path,
            receipt,
        )
        return path, receipt


def validate_seal_receipts(
    protocol: FrozenProtocol,
    bindings: Mapping[str, tuple[Path, str]],
    expected: ExpectedCohort | None = None,
    *,
    allow_prior_pretraining_commit: bool = False,
    expected_prior_pretraining_commit: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Revalidate four exact seals and their underlying artifacts independently."""

    _exact_equal(set(bindings), set(SCOUT_NEW_COMPONENTS), "Scout seal components")
    lexical_paths = [_lexical_absolute(path) for path, _ in bindings.values()]
    digests = [digest for _, digest in bindings.values()]
    _exact_equal(len(set(lexical_paths)), len(SCOUT_RUN_ORDER), "Scout seal path uniqueness")
    _exact_equal(len(set(digests)), len(SCOUT_RUN_ORDER), "Scout seal hash uniqueness")
    cohort = expected if expected is not None else build_scout_expected_cohort(protocol)
    signals: dict[str, np.ndarray] = {}
    normalized: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(SCOUT_RUN_ORDER):
        path, digest = bindings[name]
        path = _lexical_absolute(path)
        seal = _load_bound_receipt(
            path,
            digest,
            expected_stage="seal-scout-component",
            receipt_dir=_canonical_receipt_dir(protocol),
            expected_prefix=f"scout-seal-{name}",
        )
        _validate_stage_receipt_schema(seal, "seal-scout-component")
        _exact_equal(seal.get("passed"), True, f"{name} seal passage")
        _exact_equal(seal.get("state"), "SEALED", f"{name} seal state")
        _exact_equal(seal.get("component"), name, f"{name} seal component")
        _validate_protocol_binding(
            seal,
            protocol,
            allow_prior_pretraining_commit=allow_prior_pretraining_commit,
            expected_prior_pretraining_commit=expected_prior_pretraining_commit,
        )
        component = default_scout_component_paths(protocol, name)
        pre_run = seal.get("pre_run_absence_receipt")
        _require(
            isinstance(pre_run, Mapping),
            f"{name} seal pre-run absence binding is malformed.",
        )
        pre_run_path_value = pre_run.get("path")
        pre_run_digest = pre_run.get("sha256")
        _require(
            isinstance(pre_run_path_value, str)
            and isinstance(pre_run_digest, str),
            f"{name} seal pre-run absence binding is malformed.",
        )
        pre_run_path = _lexical_repo_path(protocol.repo_root, pre_run_path_value)
        pre_run_receipt = _validate_scout_pre_run_absence_receipt(
            protocol,
            component,
            pre_run_path,
            pre_run_digest,
            allow_prior_pretraining_commit=allow_prior_pretraining_commit,
            expected_prior_pretraining_commit=expected_prior_pretraining_commit,
        )
        completion = _validate_file_receipt_schema(
            seal.get("run_completion_receipt"),
            f"{name} sealed run completion receipt",
        )
        completion_path = _lexical_repo_path(
            protocol.repo_root, str(completion["path"])
        )
        expected_prior: dict[str, str] | None = None
        if index:
            previous = SCOUT_RUN_ORDER[index - 1]
            previous_path, previous_digest = bindings[previous]
            expected_prior = {
                "component": previous,
                "path": _lexical_relative_path(
                    _lexical_absolute(previous_path), protocol.repo_root
                ),
                "sha256": previous_digest,
            }
        _exact_equal(
            pre_run_receipt.get("prior_finalized_seal"),
            expected_prior,
            f"{name} pre-run contiguous predecessor",
        )
        _exact_equal(
            seal.get("prior_finalized_seal"),
            expected_prior,
            f"{name} seal contiguous predecessor",
        )
        _exact_equal(seal.get("cohort"), _cohort_receipt(cohort), f"{name} seal cohort")
        with _lease_validated_component_outputs(
            protocol,
            component,
            pre_run_path,
            pre_run_digest,
            completion_path,
            str(completion["sha256"]),
            confirmation=False,
        ) as (consumption_claim, completion_receipt):
            _exact_equal(
                seal.get("run_consumption_claim"),
                consumption_claim,
                f"{name} sealed run consumption claim",
            )
            _exact_equal(
                seal.get("run_completion_receipt"),
                completion_receipt,
                f"{name} sealed run completion receipt",
            )
            raw, artifact = _validate_scout_component(protocol, component, cohort)
            _exact_equal(seal.get("artifact"), artifact, f"{name} sealed artifact")
        _exact_equal(
            seal.get("gpu_folds_verified"),
            len(cohort.folds),
            f"{name} sealed GPU folds",
        )
        signals[name] = raw
        normalized[name] = {
            "path": _lexical_relative_path(path, protocol.repo_root),
            "sha256": digest,
            "prior_finalized_seal": copy.deepcopy(expected_prior),
            "pre_run_absence_receipt": {
                "path": pre_run_path_value,
                "sha256": pre_run_digest,
            },
            "run_consumption_claim": consumption_claim,
            "run_completion_receipt": completion_receipt,
            "artifact": artifact,
        }
    return signals, normalized


def _confirmation_component_destination_receipt(
    protocol: FrozenProtocol,
    component: ComponentPaths,
) -> dict[str, Any]:
    contract = protocol.source_manifest["confirmation_output_contract"][component.name]
    _exact_equal(
        contract.get("must_be_absent_before_run"),
        True,
        f"{component.name} confirmation absence contract",
    )
    expected = {
        "result": {
            "path": _lexical_relative_path(component.result, protocol.repo_root),
            "absent": True,
        },
        "predictions": {
            "path": _lexical_relative_path(component.predictions, protocol.repo_root),
            "absent": True,
        },
    }
    _exact_equal(
        contract.get("results_path"),
        expected["result"]["path"],
        f"{component.name} confirmation result destination",
    )
    _exact_equal(
        contract.get("predictions_path"),
        expected["predictions"]["path"],
        f"{component.name} confirmation prediction destination",
    )
    for label, path in (
        ("result", component.result),
        ("predictions", component.predictions),
    ):
        _prepare_output_destination_parent(
            protocol,
            path,
            f"{component.name} confirmation {label}",
            create_direct_parent=False,
        )
    return expected


def claim_confirmation_component_run(
    protocol: FrozenProtocol,
    component_name: str,
    confirmation_pretraining_receipt_path: Path,
    confirmation_pretraining_receipt_sha256: str,
    receipt_dir: Path,
    prior_seal_receipt_path: Path | None = None,
    prior_seal_receipt_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Claim one confirmation run and prove its two outputs absent just in time."""

    _require(component_name in ALL_COMPONENTS, "Unknown confirmation component run.")
    prefix = f"confirmation-pre-run-{component_name}"
    canonical_dir, claim_path = _claim_canonical_receipt_prefix(
        protocol, receipt_dir, prefix
    )
    confirmation_pretraining_receipt_path = _lexical_absolute(
        confirmation_pretraining_receipt_path
    )
    _validate_confirmation_pretraining_receipt(
        protocol,
        confirmation_pretraining_receipt_path,
        confirmation_pretraining_receipt_sha256,
    )
    prior_seal = _validate_prior_finalized_seal(
        protocol,
        CONFIRMATION_RUN_ORDER,
        component_name,
        prior_seal_receipt_path,
        prior_seal_receipt_sha256,
        confirmation_pretraining_receipt_path=confirmation_pretraining_receipt_path,
        confirmation_pretraining_receipt_sha256=confirmation_pretraining_receipt_sha256,
    )
    component = default_confirmation_component_paths(protocol, component_name)
    for label, path in (
        ("result", component.result),
        ("predictions", component.predictions),
    ):
        _prepare_output_destination_parent(
            protocol,
            path,
            f"{component_name} confirmation {label}",
            create_direct_parent=True,
        )
        _require_absent_destination(
            path,
            f"{component_name} confirmation {label} destination",
        )
    destinations = _confirmation_component_destination_receipt(protocol, component)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "stage": "claim-confirmation-component-run",
        "state": "ABSENCE_PROVEN",
        "passed": True,
        "protocol": _protocol_binding(protocol),
        "component": component_name,
        "confirmation_pretraining_receipt": _confirmation_pretraining_binding(
            protocol,
            confirmation_pretraining_receipt_path,
            confirmation_pretraining_receipt_sha256,
        ),
        "prior_finalized_seal": prior_seal,
        "destinations": destinations,
    }
    path = _write_claimed_content_addressed_receipt(
        canonical_dir,
        prefix,
        claim_path,
        receipt,
    )
    return path, receipt


def _validate_confirmation_pre_run_absence_receipt(
    protocol: FrozenProtocol,
    component: ComponentPaths,
    path: Path,
    digest: str,
    confirmation_pretraining_receipt_path: Path,
    confirmation_pretraining_receipt_sha256: str,
) -> dict[str, Any]:
    prefix = f"confirmation-pre-run-{component.name}"
    receipt = _load_bound_receipt(
        _lexical_absolute(path),
        digest,
        expected_stage="claim-confirmation-component-run",
        receipt_dir=_canonical_receipt_dir(protocol),
        expected_prefix=prefix,
    )
    _validate_stage_receipt_schema(receipt, "claim-confirmation-component-run")
    _exact_equal(receipt.get("passed"), True, f"{component.name} confirmation pre-run passage")
    _exact_equal(
        receipt.get("state"),
        "ABSENCE_PROVEN",
        f"{component.name} confirmation pre-run state",
    )
    _exact_equal(receipt.get("component"), component.name, f"{component.name} pre-run component")
    _validate_protocol_binding(receipt, protocol)
    _validate_confirmation_pretraining_receipt(
        protocol,
        confirmation_pretraining_receipt_path,
        confirmation_pretraining_receipt_sha256,
    )
    _exact_equal(
        receipt.get("confirmation_pretraining_receipt"),
        _confirmation_pretraining_binding(
            protocol,
            confirmation_pretraining_receipt_path,
            confirmation_pretraining_receipt_sha256,
        ),
        f"{component.name} pre-run confirmation pretraining binding",
    )
    prior_value = receipt.get("prior_finalized_seal")
    prior_binding = _validate_prior_seal_binding_schema(
        prior_value, f"{component.name} confirmation predecessor"
    )
    prior_path: Path | None = None
    prior_sha: str | None = None
    if prior_binding is not None:
        prior_path = _lexical_repo_path(protocol.repo_root, str(prior_binding["path"]))
        prior_sha = str(prior_binding["sha256"])
    _exact_equal(
        prior_value,
        _validate_prior_finalized_seal(
            protocol,
            CONFIRMATION_RUN_ORDER,
            component.name,
            prior_path,
            prior_sha,
            confirmation_pretraining_receipt_path=confirmation_pretraining_receipt_path,
            confirmation_pretraining_receipt_sha256=confirmation_pretraining_receipt_sha256,
        ),
        f"{component.name} confirmation predecessor binding",
    )
    _exact_equal(
        receipt.get("destinations"),
        _confirmation_component_destination_receipt(protocol, component),
        f"{component.name} confirmation pre-run destinations",
    )
    return receipt


def _validate_confirmation_component(
    protocol: FrozenProtocol,
    component: ComponentPaths,
    expected: ExpectedCohort,
    confirmation_pretraining_receipt: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    _require_regular_output_file(
        protocol,
        component.result,
        f"{component.name} confirmation result",
    )
    _require_regular_output_file(
        protocol,
        component.predictions,
        f"{component.name} confirmation predictions",
    )
    with _lease_confirmation_config(
        protocol, component.name, component.config
    ) as config:
        layout = confirmation_pretraining_receipt.get("input_layout")
        _require(isinstance(layout, Mapping), "Confirmation input layout is malformed.")
        stores = layout.get("stores")
        _require(isinstance(stores, Mapping), "Confirmation store bindings are malformed.")
        store = stores.get(component.name)
        _require(isinstance(store, Mapping), f"{component.name} store binding is malformed.")
        store_inventory_receipt = confirmation_pretraining_receipt.get("store_inventory")
        _require(
            isinstance(store_inventory_receipt, Mapping),
            "Confirmation store inventory receipt is malformed.",
        )
        metadata_receipt = store.get("metadata")
        _require(
            isinstance(metadata_receipt, Mapping),
            f"{component.name} store metadata receipt is malformed.",
        )
        metadata_path = _require_regular_unlinked_receipt_file(
            protocol,
            metadata_receipt,
            f"{component.name} store metadata",
        )
        store_metadata = _load_json(metadata_path, f"{component.name} store metadata")
        result = _load_json(component.result, f"{component.name} confirmation result")
        data_receipt = result.get("data")
        _require(isinstance(data_receipt, Mapping), f"{component.name} result lacks data receipt.")
        diagnostics = data_receipt.get("disk_feature_store")
        _require(isinstance(diagnostics, Mapping), f"{component.name} lacks disk-store diagnostics.")
        expected_diagnostic_paths = {
            "directory": metadata_path.parent,
            "feature_path": _lexical_repo_path(
                protocol.repo_root,
                str(store["features"]["path"]),
            ),
            "manifest_path": _lexical_repo_path(
                protocol.repo_root,
                str(store["manifest"]["path"]),
            ),
        }
        for key, expected_path in expected_diagnostic_paths.items():
            actual_path = diagnostics.get(key)
            _require(
                isinstance(actual_path, str) and actual_path,
                f"{component.name} store.{key} path is malformed.",
            )
            _exact_equal(
                _lexical_absolute(Path(actual_path)),
                expected_path,
                f"{component.name} store.{key}",
            )
        semantics = validate_result_json(
            component,
            result,
            config,
            expected,
            confirmation=True,
            store_metadata=store_metadata,
            store_receipt=store,
            store_inventory_receipt=store_inventory_receipt,
        )
        raw = validate_prediction_artifact(
            component.predictions,
            expected.frame,
            semantics,
            target_column=COMPONENT_TARGETS[component.name],
            expected_fold_by_era=_independent_fold_map(expected),
        )
        return raw, _artifact_receipt(protocol, component)


def seal_confirmation_component(
    protocol: FrozenProtocol,
    component: ComponentPaths,
    pre_run_receipt_path: Path,
    pre_run_receipt_sha256: str,
    completion_receipt_path: Path,
    completion_receipt_sha256: str,
    confirmation_pretraining_receipt_path: Path,
    confirmation_pretraining_receipt_sha256: str,
    receipt_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Seal one confirmation component without persisting any Ender metric."""

    _require(component.name in ALL_COMPONENTS, "Unknown confirmation component seal.")
    prefix = f"confirmation-seal-{component.name}"
    canonical_dir, claim_path = _claim_canonical_receipt_prefix(
        protocol, receipt_dir, prefix
    )
    _exact_equal(
        component,
        default_confirmation_component_paths(protocol, component.name),
        f"{component.name} confirmation paths",
    )
    pre_run_receipt_path = _lexical_absolute(pre_run_receipt_path)
    confirmation_pretraining_receipt_path = _lexical_absolute(
        confirmation_pretraining_receipt_path
    )
    pretraining = _validate_confirmation_pretraining_receipt(
        protocol,
        confirmation_pretraining_receipt_path,
        confirmation_pretraining_receipt_sha256,
    )
    pre_run_receipt = _validate_confirmation_pre_run_absence_receipt(
        protocol,
        component,
        pre_run_receipt_path,
        pre_run_receipt_sha256,
        confirmation_pretraining_receipt_path,
        confirmation_pretraining_receipt_sha256,
    )
    expected = build_confirmation_expected_cohort(
        protocol,
        confirmation_input_receipt=confirmation_pretraining_receipt_path,
        confirmation_input_receipt_sha256=confirmation_pretraining_receipt_sha256,
    )
    with _lease_validated_component_outputs(
        protocol,
        component,
        pre_run_receipt_path,
        pre_run_receipt_sha256,
        _lexical_absolute(completion_receipt_path),
        completion_receipt_sha256,
        confirmation=True,
        confirmation_pretraining_receipt_path=(
            confirmation_pretraining_receipt_path
        ),
        confirmation_pretraining_receipt_sha256=(
            confirmation_pretraining_receipt_sha256
        ),
    ) as (consumption_claim, completion_receipt):
        _, artifact = _validate_confirmation_component(
            protocol,
            component,
            expected,
            pretraining,
        )
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "experiment": EXPERIMENT_NAME,
            "stage": "seal-confirmation-component",
            "state": "SEALED",
            "passed": True,
            "protocol": _protocol_binding(protocol),
            "component": component.name,
            "confirmation_pretraining_receipt": (
                _confirmation_pretraining_binding(
                    protocol,
                    confirmation_pretraining_receipt_path,
                    confirmation_pretraining_receipt_sha256,
                )
            ),
            "prior_finalized_seal": copy.deepcopy(
                pre_run_receipt.get("prior_finalized_seal")
            ),
            "pre_run_absence_receipt": {
                "path": _lexical_relative_path(
                    pre_run_receipt_path, protocol.repo_root
                ),
                "sha256": pre_run_receipt_sha256,
            },
            "run_consumption_claim": consumption_claim,
            "run_completion_receipt": completion_receipt,
            "cohort": _cohort_receipt(expected),
            "artifact": artifact,
            "gpu_folds_verified": len(expected.folds),
        }
        path = _write_claimed_content_addressed_receipt(
            canonical_dir,
            prefix,
            claim_path,
            receipt,
        )
        return path, receipt


def validate_confirmation_seal_receipts(
    protocol: FrozenProtocol,
    bindings: Mapping[str, tuple[Path, str]],
    confirmation_pretraining_receipt_path: Path,
    confirmation_pretraining_receipt_sha256: str,
    expected: ExpectedCohort | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Revalidate exactly five unique confirmation seals against one pretraining receipt."""

    _exact_equal(set(bindings), set(ALL_COMPONENTS), "confirmation seal components")
    resolved_paths = [_lexical_absolute(path) for path, _ in bindings.values()]
    digests = [digest for _, digest in bindings.values()]
    _exact_equal(len(set(resolved_paths)), len(ALL_COMPONENTS), "confirmation seal path uniqueness")
    _exact_equal(len(set(digests)), len(ALL_COMPONENTS), "confirmation seal hash uniqueness")
    confirmation_pretraining_receipt_path = _lexical_absolute(
        confirmation_pretraining_receipt_path
    )
    pretraining = _validate_confirmation_pretraining_receipt(
        protocol,
        confirmation_pretraining_receipt_path,
        confirmation_pretraining_receipt_sha256,
    )
    cohort = expected
    if cohort is None:
        cohort = build_confirmation_expected_cohort(
            protocol,
            confirmation_input_receipt=confirmation_pretraining_receipt_path,
            confirmation_input_receipt_sha256=confirmation_pretraining_receipt_sha256,
        )
    expected_pretraining_binding = _confirmation_pretraining_binding(
        protocol,
        confirmation_pretraining_receipt_path,
        confirmation_pretraining_receipt_sha256,
    )
    signals: dict[str, np.ndarray] = {}
    normalized: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(CONFIRMATION_RUN_ORDER):
        path, digest = bindings[name]
        path = _lexical_absolute(path)
        seal = _load_bound_receipt(
            path,
            digest,
            expected_stage="seal-confirmation-component",
            receipt_dir=_canonical_receipt_dir(protocol),
            expected_prefix=f"confirmation-seal-{name}",
        )
        _validate_stage_receipt_schema(seal, "seal-confirmation-component")
        _exact_equal(seal.get("passed"), True, f"{name} confirmation seal passage")
        _exact_equal(seal.get("state"), "SEALED", f"{name} confirmation seal state")
        _exact_equal(seal.get("component"), name, f"{name} confirmation seal component")
        _validate_protocol_binding(seal, protocol)
        _exact_equal(
            seal.get("confirmation_pretraining_receipt"),
            expected_pretraining_binding,
            f"{name} confirmation seal pretraining binding",
        )
        component = default_confirmation_component_paths(protocol, name)
        pre_run = seal.get("pre_run_absence_receipt")
        _require(isinstance(pre_run, Mapping), f"{name} pre-run seal binding is malformed.")
        pre_run_path = _lexical_repo_path(protocol.repo_root, str(pre_run.get("path")))
        pre_run_digest = pre_run.get("sha256")
        _require(isinstance(pre_run_digest, str), f"{name} pre-run seal hash is malformed.")
        pre_run_receipt = _validate_confirmation_pre_run_absence_receipt(
            protocol,
            component,
            pre_run_path,
            pre_run_digest,
            confirmation_pretraining_receipt_path,
            confirmation_pretraining_receipt_sha256,
        )
        completion = _validate_file_receipt_schema(
            seal.get("run_completion_receipt"),
            f"{name} sealed run completion receipt",
        )
        completion_path = _lexical_repo_path(
            protocol.repo_root, str(completion["path"])
        )
        expected_prior: dict[str, str] | None = None
        if index:
            previous = CONFIRMATION_RUN_ORDER[index - 1]
            previous_path, previous_digest = bindings[previous]
            expected_prior = {
                "component": previous,
                "path": _lexical_relative_path(
                    _lexical_absolute(previous_path), protocol.repo_root
                ),
                "sha256": previous_digest,
            }
        _exact_equal(
            pre_run_receipt.get("prior_finalized_seal"),
            expected_prior,
            f"{name} confirmation pre-run contiguous predecessor",
        )
        _exact_equal(
            seal.get("prior_finalized_seal"),
            expected_prior,
            f"{name} confirmation seal contiguous predecessor",
        )
        _exact_equal(seal.get("cohort"), _cohort_receipt(cohort), f"{name} seal cohort")
        with _lease_validated_component_outputs(
            protocol,
            component,
            pre_run_path,
            pre_run_digest,
            completion_path,
            str(completion["sha256"]),
            confirmation=True,
            confirmation_pretraining_receipt_path=(
                confirmation_pretraining_receipt_path
            ),
            confirmation_pretraining_receipt_sha256=(
                confirmation_pretraining_receipt_sha256
            ),
        ) as (consumption_claim, completion_receipt):
            _exact_equal(
                seal.get("run_consumption_claim"),
                consumption_claim,
                f"{name} sealed run consumption claim",
            )
            _exact_equal(
                seal.get("run_completion_receipt"),
                completion_receipt,
                f"{name} sealed run completion receipt",
            )
            raw, artifact = _validate_confirmation_component(
                protocol,
                component,
                cohort,
                pretraining,
            )
            _exact_equal(seal.get("artifact"), artifact, f"{name} sealed artifact")
        _exact_equal(
            seal.get("gpu_folds_verified"),
            len(cohort.folds),
            f"{name} sealed GPU folds",
        )
        signals[name] = raw
        normalized[name] = {
            "path": _lexical_relative_path(path, protocol.repo_root),
            "sha256": digest,
            "confirmation_pretraining_receipt": expected_pretraining_binding,
            "prior_finalized_seal": copy.deepcopy(expected_prior),
            "pre_run_absence_receipt": {
                "path": _lexical_relative_path(pre_run_path, protocol.repo_root),
                "sha256": pre_run_digest,
            },
            "run_consumption_claim": consumption_claim,
            "run_completion_receipt": completion_receipt,
            "artifact": artifact,
        }
    return signals, normalized


def _build_confirmation_scoring_frame(
    protocol: FrozenProtocol,
    expected: ExpectedCohort,
    bindings: Mapping[str, tuple[Path, str]],
    confirmation_pretraining_receipt_path: Path,
    confirmation_pretraining_receipt_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    signals, seals = validate_confirmation_seal_receipts(
        protocol,
        bindings,
        confirmation_pretraining_receipt_path,
        confirmation_pretraining_receipt_sha256,
        expected,
    )
    tabm, tabm_receipts = load_frozen_two_seed_residual(
        protocol,
        expected,
        confirmation=True,
    )
    frame = expected.frame[
        [
            ID_COLUMN,
            ERA_COLUMN,
            ENDER_TARGET,
            BENCHMARK_ENDER20,
            BENCHMARK_ENDER60,
            FOLD_COLUMN,
        ]
    ].copy()
    for name in ALL_COMPONENTS:
        frame[name] = signals[name]
    frame["tabm_two_seed_residual"] = tabm
    return frame, {
        "confirmation_pretraining_receipt": _confirmation_pretraining_binding(
            protocol,
            confirmation_pretraining_receipt_path,
            confirmation_pretraining_receipt_sha256,
        ),
        "confirmation_seal_receipts": seals,
        "tabm_two_seed_residual": tabm_receipts,
        "cohort": _cohort_receipt(expected),
    }


def _build_scout_scoring_frame(
    protocol: FrozenProtocol,
    expected: ExpectedCohort,
    bindings: Mapping[str, tuple[Path, str]],
    *,
    allow_prior_pretraining_commit: bool = False,
    expected_prior_pretraining_commit: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    signals, seals = validate_seal_receipts(
        protocol,
        bindings,
        expected,
        allow_prior_pretraining_commit=allow_prior_pretraining_commit,
        expected_prior_pretraining_commit=expected_prior_pretraining_commit,
    )
    frozen_xerxes = protocol.source_manifest["reused_xerxes_component"]
    xerxes_entries = {
        name: frozen_xerxes[name]
        for name in (
            "base_config",
            "config",
            "evaluator",
            "predictions",
            "result",
            "source_manifest",
        )
    }
    with _lease_frozen_manifest_artifacts(
        protocol,
        xerxes_entries,
        "reused Xerxes",
    ):
        xerxes_paths = _reused_xerxes_paths(protocol)
        xerxes_raw, xerxes_artifact = _validate_scout_component(
            protocol, xerxes_paths, expected
        )
        for name in ("config", "result", "predictions"):
            _exact_equal(
                xerxes_artifact[name],
                frozen_xerxes[name],
                f"reused Xerxes {name} manifest binding",
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
        path = _lexical_repo_path(protocol.repo_root, str(item.get("path")))
        digest = item.get("sha256")
        _require(isinstance(digest, str), f"{name} stage seal hash is malformed.")
        result[name] = (path, digest)
    return result


def _derive_scout_calibration(
    full_frame: pd.DataFrame,
    expected: ExpectedCohort,
) -> dict[str, Any]:
    """Deterministically derive every calibration candidate and selection."""

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
    return {
        "state": "PASS" if passed else "STOP_NO_ELIGIBLE_CANDIDATE",
        "passed": passed,
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


def _validate_scout_calibration_derivation(
    calibration_receipt: Mapping[str, Any],
    full_frame: pd.DataFrame,
    expected: ExpectedCohort,
) -> str:
    """Recompute calibration before any Scout locked-era slice is opened."""

    derived = _derive_scout_calibration(full_frame, expected)
    for key in ("state", "passed", "selected_formula", "calibration"):
        _exact_equal(
            calibration_receipt.get(key),
            derived[key],
            f"recomputed calibration {key}",
        )
    _exact_equal(derived["passed"], True, "recomputed calibration passage")
    return _selected_formula(derived)


def run_calibrate(
    protocol: FrozenProtocol,
    seal_bindings: Mapping[str, tuple[Path, str]],
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Score all and only the five frozen formulas on the first 164 eras."""

    canonical_dir, claim_path = _claim_canonical_receipt_prefix(
        protocol, output_dir, "calibrate"
    )
    expected = build_scout_expected_cohort(protocol)
    full_frame, inputs = _build_scout_scoring_frame(
        protocol, expected, seal_bindings
    )
    derived = _derive_scout_calibration(full_frame, expected)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "stage": "calibrate",
        "state": derived["state"],
        "passed": derived["passed"],
        "protocol": _protocol_binding(protocol),
        "inputs": inputs,
        "selected_formula": derived["selected_formula"],
        "calibration": derived["calibration"],
    }
    path = _write_claimed_content_addressed_receipt(
        canonical_dir,
        "calibrate",
        claim_path,
        receipt,
    )
    return path, receipt


def _load_passing_stage_receipt(
    protocol: FrozenProtocol,
    path: Path,
    digest: str,
    *,
    stage: str,
    allow_prior_pretraining_commit: bool = False,
    expected_prior_pretraining_commit: str | None = None,
) -> dict[str, Any]:
    receipt = _load_bound_receipt(
        path,
        digest,
        expected_stage=stage,
        receipt_dir=_canonical_receipt_dir(protocol),
        expected_prefix=stage,
    )
    _validate_stage_receipt_schema(receipt, stage)
    _exact_equal(receipt.get("passed"), True, f"{stage} passage")
    _exact_equal(receipt.get("state"), "PASS", f"{stage} state")
    _validate_protocol_binding(
        receipt,
        protocol,
        allow_prior_pretraining_commit=allow_prior_pretraining_commit,
        expected_prior_pretraining_commit=expected_prior_pretraining_commit,
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


def _load_passing_scout_locked_receipt(
    protocol: FrozenProtocol,
    path: Path,
    digest: str,
) -> dict[str, Any]:
    """Rederive a passing Scout lock from its exact sealed model artifacts."""

    locked_receipt = _load_passing_stage_receipt(
        protocol,
        _lexical_absolute(path),
        digest,
        stage="locked",
        allow_prior_pretraining_commit=True,
    )
    locked_protocol = locked_receipt.get("protocol")
    _require(isinstance(locked_protocol, Mapping), "Scout locked protocol is malformed.")
    scout_pretraining_commit = locked_protocol.get("pretraining_commit")
    _require(
        _is_lower_hex(scout_pretraining_commit, 40),
        "Scout locked pretraining checkpoint is malformed.",
    )
    assert isinstance(scout_pretraining_commit, str)
    calibration_binding = _validate_canonical_receipt_binding_schema(
        locked_receipt.get("input_receipt"),
        "calibrate",
        "Scout locked calibration input",
    )
    calibration_path = _lexical_repo_path(
        protocol.repo_root,
        str(calibration_binding["path"]),
    )
    calibration_receipt = _load_passing_stage_receipt(
        protocol,
        calibration_path,
        str(calibration_binding["sha256"]),
        stage="calibrate",
        allow_prior_pretraining_commit=True,
        expected_prior_pretraining_commit=scout_pretraining_commit,
    )
    inputs = calibration_receipt.get("inputs")
    _require(isinstance(inputs, Mapping), "Calibration inputs are malformed.")
    bindings = _seal_bindings_from_receipt(protocol, inputs.get("seal_receipts"))
    expected = build_scout_expected_cohort(protocol)
    full_frame, current_inputs = _build_scout_scoring_frame(
        protocol,
        expected,
        bindings,
        allow_prior_pretraining_commit=True,
        expected_prior_pretraining_commit=scout_pretraining_commit,
    )
    _exact_equal(current_inputs, inputs, "Scout locked authorization inputs")
    selected = _validate_scout_calibration_derivation(
        calibration_receipt,
        full_frame,
        expected,
    )
    _exact_equal(
        _selected_formula(locked_receipt),
        selected,
        "Scout locked authorization formula",
    )
    locked_eras = expected.eras[-SCOUT_LOCKED_ERAS:]
    _exact_equal(
        locked_eras,
        SCOUT_LOCKED_ERA_SEQUENCE,
        "Scout locked authorization eras",
    )
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
    derived = {
        "rows": len(locked),
        "eras": len(locked_eras),
        "first_era": locked_eras[0],
        "last_era": locked_eras[-1],
        "summary": summary,
        "checks": checks,
        "per_era": _serialize_per_era(metrics, [selected]),
    }
    _exact_equal(
        locked_receipt.get("locked"),
        derived,
        "Scout locked authorization derivation",
    )
    _exact_equal(all(checks.values()), True, "Scout locked authorization passage")
    return locked_receipt


def run_locked(
    protocol: FrozenProtocol,
    calibration_receipt_path: Path,
    calibration_receipt_sha256: str,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Open the Scout holdout for only the immutable selected formula."""

    canonical_dir, claim_path = _claim_canonical_receipt_prefix(
        protocol, output_dir, "locked"
    )
    calibration_receipt_path = _lexical_absolute(calibration_receipt_path)
    calibration_receipt = _load_passing_stage_receipt(
        protocol,
        calibration_receipt_path,
        calibration_receipt_sha256,
        stage="calibrate",
    )
    inputs = calibration_receipt.get("inputs")
    _require(isinstance(inputs, Mapping), "Calibration inputs are malformed.")
    bindings = _seal_bindings_from_receipt(protocol, inputs.get("seal_receipts"))
    expected = build_scout_expected_cohort(protocol)
    full_frame, current_inputs = _build_scout_scoring_frame(
        protocol, expected, bindings
    )
    _exact_equal(current_inputs, inputs, "locked Scout input revalidation")
    selected = _validate_scout_calibration_derivation(
        calibration_receipt,
        full_frame,
        expected,
    )
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
            "path": _lexical_relative_path(calibration_receipt_path, protocol.repo_root),
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
    path = _write_claimed_content_addressed_receipt(
        canonical_dir,
        "locked",
        claim_path,
        receipt,
    )
    return path, receipt


def _confirmation_seal_bindings_from_receipt(
    protocol: FrozenProtocol,
    value: Any,
) -> dict[str, tuple[Path, str]]:
    _require(isinstance(value, Mapping), "Confirmation stage seal bindings are malformed.")
    _exact_equal(set(value), set(ALL_COMPONENTS), "confirmation stage seal components")
    bindings: dict[str, tuple[Path, str]] = {}
    for name in ALL_COMPONENTS:
        item = value[name]
        _require(isinstance(item, Mapping), f"{name} confirmation seal binding is malformed.")
        path_value = item.get("path")
        digest = item.get("sha256")
        _require(
            isinstance(path_value, str) and isinstance(digest, str),
            f"{name} confirmation seal path/hash is malformed.",
        )
        bindings[name] = (_lexical_repo_path(protocol.repo_root, path_value), digest)
    return bindings


def _derive_confirmation_calibration(
    full_frame: pd.DataFrame,
    expected: ExpectedCohort,
    selected: str,
) -> dict[str, Any]:
    calibration_eras = expected.eras[:CONFIRMATION_CALIBRATION_ERAS]
    _exact_equal(calibration_eras[0], CONFIRMATION_FIRST_ERA, "confirmation calibration first era")
    _exact_equal(
        calibration_eras[-1],
        CONFIRMATION_LAST_CALIBRATION_ERA,
        "confirmation calibration last era",
    )
    calibration = _slice_eras(full_frame, calibration_eras)
    scored = build_selected_rank_blend(calibration, selected)
    metrics = compute_per_era_metrics(
        scored,
        [selected],
        calibration_eras,
        tabm_column="tabm_two_seed_residual",
    )
    summary = summarize_signal(metrics, selected)
    checks = confirmation_calibration_checks(summary)
    passed = all(checks.values())
    return {
        "state": "PASS" if passed else "STOP_CONFIRMATION_CALIBRATION_FAILED",
        "passed": passed,
        "calibration": {
            "rows": len(calibration),
            "eras": len(calibration_eras),
            "first_era": calibration_eras[0],
            "last_era": calibration_eras[-1],
            "summary": summary,
            "checks": checks,
            "per_era": _serialize_per_era(metrics, [selected]),
        },
    }


def _validate_confirmation_calibration_derivation(
    calibration_receipt: Mapping[str, Any],
    full_frame: pd.DataFrame,
    expected: ExpectedCohort,
    selected: str,
) -> None:
    derived = _derive_confirmation_calibration(full_frame, expected, selected)
    for key in ("state", "passed", "calibration"):
        _exact_equal(
            calibration_receipt.get(key),
            derived[key],
            f"recomputed confirmation calibration {key}",
        )
    _exact_equal(derived["passed"], True, "recomputed confirmation calibration passage")


def run_confirmation_calibrate(
    protocol: FrozenProtocol,
    scout_locked_receipt_path: Path,
    scout_locked_receipt_sha256: str,
    confirmation_pretraining_receipt_path: Path,
    confirmation_pretraining_receipt_sha256: str,
    seal_bindings: Mapping[str, tuple[Path, str]],
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Score the fixed Scout-selected formula on only the first 655 eras."""

    canonical_dir, claim_path = _claim_canonical_receipt_prefix(
        protocol, output_dir, "confirmation-calibrate"
    )
    scout_locked_receipt_path = _lexical_absolute(scout_locked_receipt_path)
    scout_locked = _load_passing_scout_locked_receipt(
        protocol,
        scout_locked_receipt_path,
        scout_locked_receipt_sha256,
    )
    selected = _selected_formula(scout_locked)
    confirmation_pretraining_receipt_path = _lexical_absolute(
        confirmation_pretraining_receipt_path
    )
    pretraining = _validate_confirmation_pretraining_receipt(
        protocol,
        confirmation_pretraining_receipt_path,
        confirmation_pretraining_receipt_sha256,
    )
    scout_binding = {
        "path": _lexical_relative_path(scout_locked_receipt_path, protocol.repo_root),
        "sha256": scout_locked_receipt_sha256,
    }
    _exact_equal(
        pretraining.get("scout_locked_receipt"),
        scout_binding,
        "confirmation pretraining Scout authorization",
    )
    expected = build_confirmation_expected_cohort(
        protocol,
        confirmation_input_receipt=confirmation_pretraining_receipt_path,
        confirmation_input_receipt_sha256=confirmation_pretraining_receipt_sha256,
    )
    full_frame, current_inputs = _build_confirmation_scoring_frame(
        protocol,
        expected,
        seal_bindings,
        confirmation_pretraining_receipt_path,
        confirmation_pretraining_receipt_sha256,
    )
    derived = _derive_confirmation_calibration(full_frame, expected, selected)
    inputs = {"scout_locked_receipt": scout_binding, **current_inputs}
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "stage": "confirmation-calibrate",
        "state": derived["state"],
        "passed": derived["passed"],
        "protocol": _protocol_binding(protocol),
        "inputs": inputs,
        "selected_formula": {
            "name": selected,
            "weights": dict(BLEND_WEIGHTS[selected]),
        },
        "calibration": derived["calibration"],
    }
    path = _write_claimed_content_addressed_receipt(
        canonical_dir,
        "confirmation-calibrate",
        claim_path,
        receipt,
    )
    return path, receipt


def run_confirmation_locked(
    protocol: FrozenProtocol,
    confirmation_calibration_receipt_path: Path,
    confirmation_calibration_receipt_sha256: str,
    confirmation_pretraining_receipt_path: Path,
    confirmation_pretraining_receipt_sha256: str,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Score locked 200 first; score all 855 only after that holdout passes."""

    canonical_dir, claim_path = _claim_canonical_receipt_prefix(
        protocol, output_dir, "confirmation-locked"
    )
    confirmation_calibration_receipt_path = _lexical_absolute(
        confirmation_calibration_receipt_path
    )
    calibration_receipt = _load_passing_stage_receipt(
        protocol,
        confirmation_calibration_receipt_path,
        confirmation_calibration_receipt_sha256,
        stage="confirmation-calibrate",
    )
    selected = _selected_formula(calibration_receipt)
    confirmation_pretraining_receipt_path = _lexical_absolute(
        confirmation_pretraining_receipt_path
    )
    _validate_confirmation_pretraining_receipt(
        protocol,
        confirmation_pretraining_receipt_path,
        confirmation_pretraining_receipt_sha256,
    )
    inputs = calibration_receipt.get("inputs")
    _require(isinstance(inputs, Mapping), "Confirmation calibration inputs are malformed.")
    pretraining_binding = _confirmation_pretraining_binding(
        protocol,
        confirmation_pretraining_receipt_path,
        confirmation_pretraining_receipt_sha256,
    )
    _exact_equal(
        inputs.get("confirmation_pretraining_receipt"),
        pretraining_binding,
        "confirmation locked pretraining binding",
    )
    scout_binding = inputs.get("scout_locked_receipt")
    _require(isinstance(scout_binding, Mapping), "Scout locked input binding is malformed.")
    scout_path = _lexical_repo_path(protocol.repo_root, str(scout_binding.get("path")))
    scout_digest = scout_binding.get("sha256")
    _require(isinstance(scout_digest, str), "Scout locked input hash is malformed.")
    scout_locked = _load_passing_scout_locked_receipt(
        protocol,
        scout_path,
        scout_digest,
    )
    _exact_equal(_selected_formula(scout_locked), selected, "confirmation selected Scout formula")
    bindings = _confirmation_seal_bindings_from_receipt(
        protocol,
        inputs.get("confirmation_seal_receipts"),
    )
    expected = build_confirmation_expected_cohort(
        protocol,
        confirmation_input_receipt=confirmation_pretraining_receipt_path,
        confirmation_input_receipt_sha256=confirmation_pretraining_receipt_sha256,
    )
    full_frame, current_inputs = _build_confirmation_scoring_frame(
        protocol,
        expected,
        bindings,
        confirmation_pretraining_receipt_path,
        confirmation_pretraining_receipt_sha256,
    )
    _exact_equal(
        {"scout_locked_receipt": dict(scout_binding), **current_inputs},
        dict(inputs),
        "confirmation locked input revalidation",
    )
    _validate_confirmation_calibration_derivation(
        calibration_receipt,
        full_frame,
        expected,
        selected,
    )

    locked_eras = expected.eras[-CONFIRMATION_LOCKED_ERAS:]
    _exact_equal(locked_eras[0], CONFIRMATION_FIRST_LOCKED_ERA, "confirmation locked first era")
    _exact_equal(locked_eras[-1], CONFIRMATION_LAST_ERA, "confirmation locked last era")
    locked = _slice_eras(full_frame, locked_eras)
    locked_scored = build_selected_rank_blend(locked, selected)
    locked_metrics = compute_per_era_metrics(
        locked_scored,
        [selected],
        locked_eras,
        tabm_column="tabm_two_seed_residual",
    )
    locked_summary = summarize_signal(locked_metrics, selected)
    locked_checks_result = confirmation_locked_checks(locked_summary)
    locked_passed = all(locked_checks_result.values())
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "stage": "confirmation-locked",
        "state": "PASS" if locked_passed else "STOP_CONFIRMATION_LOCKED_FAILED",
        "passed": locked_passed,
        "protocol": _protocol_binding(protocol),
        "input_receipt": {
            "path": _lexical_relative_path(confirmation_calibration_receipt_path, protocol.repo_root),
            "sha256": confirmation_calibration_receipt_sha256,
        },
        "confirmation_pretraining_receipt": pretraining_binding,
        "confirmation_seal_receipts": current_inputs["confirmation_seal_receipts"],
        "selected_formula": {
            "name": selected,
            "weights": dict(BLEND_WEIGHTS[selected]),
        },
        "locked": {
            "rows": len(locked),
            "eras": len(locked_eras),
            "first_era": locked_eras[0],
            "last_era": locked_eras[-1],
            "summary": locked_summary,
            "checks": locked_checks_result,
            "per_era": _serialize_per_era(locked_metrics, [selected]),
        },
    }
    if locked_passed:
        full_scored = build_selected_rank_blend(full_frame, selected)
        full_metrics = compute_per_era_metrics(
            full_scored,
            [selected],
            expected.eras,
            tabm_column="tabm_two_seed_residual",
        )
        full_summary = summarize_signal(full_metrics, selected)
        full_checks_result = confirmation_full_checks(full_summary)
        full_passed = all(full_checks_result.values())
        receipt["full"] = {
            "rows": len(full_frame),
            "eras": len(expected.eras),
            "first_era": expected.eras[0],
            "last_era": expected.eras[-1],
            "summary": full_summary,
            "checks": full_checks_result,
            "per_era": _serialize_per_era(full_metrics, [selected]),
        }
        receipt["passed"] = full_passed
        receipt["state"] = "PASS" if full_passed else "STOP_CONFIRMATION_FULL_FAILED"
    path = _write_claimed_content_addressed_receipt(
        canonical_dir,
        "confirmation-locked",
        claim_path,
        receipt,
    )
    return path, receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _repo_root()
    experiment_dir = root / "numerai/agents/experiments" / EXPERIMENT_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "claim-scout-component-run",
            "seal-scout-component",
            "calibrate",
            "locked",
            "create-confirmation-store-inventory",
            "create-confirmation-pretraining",
            "claim-confirmation-component-run",
            "seal-confirmation-component",
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
    parser.add_argument("--component", choices=ALL_COMPONENTS)
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
    parser.add_argument("--pre-run-receipt", type=Path)
    parser.add_argument("--pre-run-receipt-sha256")
    parser.add_argument("--run-completion-receipt", type=Path)
    parser.add_argument("--run-completion-receipt-sha256")
    parser.add_argument("--prior-seal-receipt", type=Path)
    parser.add_argument("--prior-seal-receipt-sha256")
    parser.add_argument("--confirmation-pretraining-receipt", type=Path)
    parser.add_argument("--confirmation-pretraining-receipt-sha256")
    parser.add_argument(
        "--confirmation-seal-receipt",
        action="append",
        nargs=3,
        metavar=("COMPONENT", "PATH", "SHA256"),
        help="Exact confirmation seal; repeat once for each of the five components.",
    )
    return parser.parse_args(argv)


def _validate_cli_receipt_binding(
    receipt_dir: Path,
    path: Path,
    digest: str,
    prefix: str,
    label: str,
) -> tuple[Path, str]:
    _require_sha256(digest, f"{label} receipt SHA-256")
    lexical = _lexical_absolute(path)
    _exact_equal(lexical.parent, receipt_dir, f"{label} canonical receipt parent")
    _exact_equal(
        lexical.name,
        f"{prefix}-{digest}.json",
        f"{label} content-addressed receipt filename",
    )
    return lexical, digest


def _validate_unique_cli_bindings(
    bindings: Sequence[tuple[Path, str]],
    label: str,
) -> None:
    paths = [_lexical_absolute(path) for path, _ in bindings]
    digests = [digest for _, digest in bindings]
    _exact_equal(len(set(paths)), len(paths), f"{label} unique receipt paths")
    _exact_equal(len(set(digests)), len(digests), f"{label} unique receipt digests")


def _validate_cli_stage_arguments(args: argparse.Namespace) -> None:
    """Reject every syntactic/path error before protocol or source access."""

    _require(
        _is_lower_hex(args.pretraining_commit, 40),
        "Pretraining commit must be a lowercase 40-character Git SHA-1.",
    )
    root = _repo_root()
    experiment_dir = root / "numerai/agents/experiments" / EXPERIMENT_NAME
    canonical_manifest = _lexical_absolute(experiment_dir / "source_manifest.json")
    canonical_receipts = _lexical_absolute(experiment_dir / "receipts")
    _exact_equal(
        _lexical_absolute(args.source_manifest),
        canonical_manifest,
        "canonical source manifest",
    )
    _exact_equal(
        _lexical_absolute(args.output_dir),
        canonical_receipts,
        "canonical output receipt directory",
    )
    _exact_equal(
        _lexical_absolute(args.receipt_dir or args.output_dir),
        canonical_receipts,
        "canonical receipt directory",
    )

    optional_names = {
        "component",
        "seal_receipt",
        "input_receipt",
        "input_receipt_sha256",
        "pre_run_receipt",
        "pre_run_receipt_sha256",
        "run_completion_receipt",
        "run_completion_receipt_sha256",
        "prior_seal_receipt",
        "prior_seal_receipt_sha256",
        "confirmation_pretraining_receipt",
        "confirmation_pretraining_receipt_sha256",
        "confirmation_seal_receipt",
    }
    allowed_by_stage = {
        "claim-scout-component-run": {
            "component", "prior_seal_receipt", "prior_seal_receipt_sha256"
        },
        "seal-scout-component": {
            "component", "input_receipt", "input_receipt_sha256",
            "run_completion_receipt", "run_completion_receipt_sha256",
        },
        "calibrate": {"seal_receipt"},
        "locked": {"input_receipt", "input_receipt_sha256"},
        "create-confirmation-store-inventory": {
            "input_receipt", "input_receipt_sha256"
        },
        "create-confirmation-pretraining": {
            "input_receipt", "input_receipt_sha256"
        },
        "claim-confirmation-component-run": {
            "component", "confirmation_pretraining_receipt",
            "confirmation_pretraining_receipt_sha256", "prior_seal_receipt",
            "prior_seal_receipt_sha256",
        },
        "seal-confirmation-component": {
            "component", "pre_run_receipt", "pre_run_receipt_sha256",
            "run_completion_receipt", "run_completion_receipt_sha256",
            "confirmation_pretraining_receipt",
            "confirmation_pretraining_receipt_sha256",
        },
        "confirmation-calibrate": {
            "input_receipt", "input_receipt_sha256",
            "confirmation_pretraining_receipt",
            "confirmation_pretraining_receipt_sha256",
            "confirmation_seal_receipt",
        },
        "confirmation-locked": {
            "input_receipt", "input_receipt_sha256",
            "confirmation_pretraining_receipt",
            "confirmation_pretraining_receipt_sha256",
        },
    }
    _require(args.stage in allowed_by_stage, f"Unknown evaluator stage: {args.stage}")
    for name in optional_names - allowed_by_stage[args.stage]:
        _require(getattr(args, name) is None, f"--{name.replace('_', '-')} is not valid for {args.stage}.")

    collected: list[tuple[Path, str, str, str]] = []

    def require_pair(path_name: str, digest_name: str, prefix: str, label: str) -> None:
        path = getattr(args, path_name)
        digest = getattr(args, digest_name)
        _require(path is not None, f"--{path_name.replace('_', '-')} is required.")
        _require(digest is not None, f"--{digest_name.replace('_', '-')} is required.")
        assert path is not None and digest is not None
        _require_sha256(digest, f"{label} receipt SHA-256")
        collected.append((path, digest, prefix, label))

    if args.stage == "claim-scout-component-run":
        _require(args.component in SCOUT_RUN_ORDER, "A new Scout --component is required.")
        assert args.component is not None
        index = SCOUT_RUN_ORDER.index(args.component)
        if index == 0:
            _require(
                args.prior_seal_receipt is None
                and args.prior_seal_receipt_sha256 is None,
                "The first Scout component may not bind a prior seal.",
            )
        else:
            require_pair(
                "prior_seal_receipt",
                "prior_seal_receipt_sha256",
                f"scout-seal-{SCOUT_RUN_ORDER[index - 1]}",
                "Scout predecessor",
            )
    elif args.stage == "seal-scout-component":
        _require(args.component in SCOUT_RUN_ORDER, "A new Scout --component is required.")
        assert args.component is not None
        require_pair(
            "input_receipt", "input_receipt_sha256",
            f"scout-pre-run-{args.component}", "Scout pre-run",
        )
        require_pair(
            "run_completion_receipt",
            "run_completion_receipt_sha256",
            f"scout-train-{args.component}-completion",
            "Scout run completion",
        )
    elif args.stage == "calibrate":
        bindings = _parse_seal_bindings(args.seal_receipt)
        _exact_equal(set(bindings), set(SCOUT_RUN_ORDER), "Scout CLI seal components")
        for component in SCOUT_RUN_ORDER:
            path, digest = bindings[component]
            _require_sha256(digest, f"{component} Scout seal SHA-256")
            collected.append((path, digest, f"scout-seal-{component}", f"{component} Scout seal"))
    elif args.stage in {"locked", "create-confirmation-store-inventory", "create-confirmation-pretraining"}:
        prefix = "calibrate" if args.stage == "locked" else "locked"
        require_pair("input_receipt", "input_receipt_sha256", prefix, f"{args.stage} input")
    elif args.stage == "claim-confirmation-component-run":
        _require(args.component in CONFIRMATION_RUN_ORDER, "A confirmation --component is required.")
        assert args.component is not None
        require_pair(
            "confirmation_pretraining_receipt",
            "confirmation_pretraining_receipt_sha256",
            "confirmation-pretraining",
            "confirmation pretraining",
        )
        index = CONFIRMATION_RUN_ORDER.index(args.component)
        if index == 0:
            _require(
                args.prior_seal_receipt is None
                and args.prior_seal_receipt_sha256 is None,
                "The first confirmation component may not bind a prior seal.",
            )
        else:
            require_pair(
                "prior_seal_receipt",
                "prior_seal_receipt_sha256",
                f"confirmation-seal-{CONFIRMATION_RUN_ORDER[index - 1]}",
                "confirmation predecessor",
            )
    elif args.stage == "seal-confirmation-component":
        _require(args.component in CONFIRMATION_RUN_ORDER, "A confirmation --component is required.")
        assert args.component is not None
        require_pair(
            "pre_run_receipt", "pre_run_receipt_sha256",
            f"confirmation-pre-run-{args.component}", "confirmation pre-run",
        )
        require_pair(
            "confirmation_pretraining_receipt",
            "confirmation_pretraining_receipt_sha256",
            "confirmation-pretraining", "confirmation pretraining",
        )
        require_pair(
            "run_completion_receipt",
            "run_completion_receipt_sha256",
            f"confirmation-train-{args.component}-completion",
            "confirmation run completion",
        )
    elif args.stage == "confirmation-calibrate":
        require_pair("input_receipt", "input_receipt_sha256", "locked", "Scout locked")
        require_pair(
            "confirmation_pretraining_receipt",
            "confirmation_pretraining_receipt_sha256",
            "confirmation-pretraining", "confirmation pretraining",
        )
        bindings = _parse_confirmation_seal_bindings(args.confirmation_seal_receipt)
        _exact_equal(set(bindings), set(CONFIRMATION_RUN_ORDER), "confirmation CLI seal components")
        for component in CONFIRMATION_RUN_ORDER:
            path, digest = bindings[component]
            _require_sha256(digest, f"{component} confirmation seal SHA-256")
            collected.append((path, digest, f"confirmation-seal-{component}", f"{component} confirmation seal"))
    elif args.stage == "confirmation-locked":
        require_pair(
            "input_receipt", "input_receipt_sha256",
            "confirmation-calibrate", "confirmation calibration",
        )
        require_pair(
            "confirmation_pretraining_receipt",
            "confirmation_pretraining_receipt_sha256",
            "confirmation-pretraining", "confirmation pretraining",
        )

    _validate_unique_cli_bindings(
        [(path, digest) for path, digest, _, _ in collected],
        f"{args.stage} CLI",
    )
    for path, digest, prefix, label in collected:
        _validate_cli_receipt_binding(canonical_receipts, path, digest, prefix, label)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_cli_stage_arguments(args)
    protocol = verify_frozen_protocol(
        _lexical_absolute(args.source_manifest),
        pretraining_commit=args.pretraining_commit,
    )
    output_dir = Path(os.path.abspath(args.output_dir))
    receipt_dir = Path(os.path.abspath(args.receipt_dir or args.output_dir))
    if args.stage == "claim-scout-component-run":
        assert args.component is not None
        path, receipt = claim_scout_component_run(
            protocol,
            args.component,
            receipt_dir,
            (
                _lexical_absolute(args.prior_seal_receipt)
                if args.prior_seal_receipt is not None
                else None
            ),
            args.prior_seal_receipt_sha256,
        )
    elif args.stage == "seal-scout-component":
        assert args.component is not None
        assert args.input_receipt is not None
        assert args.input_receipt_sha256 is not None
        assert args.run_completion_receipt is not None
        assert args.run_completion_receipt_sha256 is not None
        path, receipt = seal_scout_component(
            protocol,
            default_scout_component_paths(protocol, args.component),
            receipt_dir,
            _lexical_absolute(args.input_receipt),
            args.input_receipt_sha256,
            _lexical_absolute(args.run_completion_receipt),
            args.run_completion_receipt_sha256,
        )
    elif args.stage == "calibrate":
        path, receipt = run_calibrate(
            protocol,
            _parse_seal_bindings(args.seal_receipt),
            output_dir,
        )
    elif args.stage == "locked":
        assert args.input_receipt is not None
        assert args.input_receipt_sha256 is not None
        path, receipt = run_locked(
            protocol,
            _lexical_absolute(args.input_receipt),
            args.input_receipt_sha256,
            output_dir,
        )
    elif args.stage == "create-confirmation-store-inventory":
        assert args.input_receipt is not None
        assert args.input_receipt_sha256 is not None
        path, receipt = create_confirmation_store_inventory(
            protocol,
            _lexical_absolute(args.input_receipt),
            args.input_receipt_sha256,
            output_dir,
        )
    elif args.stage == "create-confirmation-pretraining":
        assert args.input_receipt is not None
        assert args.input_receipt_sha256 is not None
        path, receipt = create_confirmation_pretraining_receipt(
            protocol,
            _lexical_absolute(args.input_receipt),
            args.input_receipt_sha256,
            output_dir,
        )
    elif args.stage == "claim-confirmation-component-run":
        assert args.component is not None
        assert args.confirmation_pretraining_receipt is not None
        assert args.confirmation_pretraining_receipt_sha256 is not None
        path, receipt = claim_confirmation_component_run(
            protocol,
            args.component,
            _lexical_absolute(args.confirmation_pretraining_receipt),
            args.confirmation_pretraining_receipt_sha256,
            receipt_dir,
            (
                _lexical_absolute(args.prior_seal_receipt)
                if args.prior_seal_receipt is not None
                else None
            ),
            args.prior_seal_receipt_sha256,
        )
    elif args.stage == "seal-confirmation-component":
        assert args.component is not None
        assert args.pre_run_receipt is not None
        assert args.pre_run_receipt_sha256 is not None
        assert args.run_completion_receipt is not None
        assert args.run_completion_receipt_sha256 is not None
        assert args.confirmation_pretraining_receipt is not None
        assert args.confirmation_pretraining_receipt_sha256 is not None
        path, receipt = seal_confirmation_component(
            protocol,
            default_confirmation_component_paths(protocol, args.component),
            _lexical_absolute(args.pre_run_receipt),
            args.pre_run_receipt_sha256,
            _lexical_absolute(args.run_completion_receipt),
            args.run_completion_receipt_sha256,
            _lexical_absolute(args.confirmation_pretraining_receipt),
            args.confirmation_pretraining_receipt_sha256,
            receipt_dir,
        )
    elif args.stage == "confirmation-calibrate":
        assert args.input_receipt is not None
        assert args.input_receipt_sha256 is not None
        assert args.confirmation_pretraining_receipt is not None
        assert args.confirmation_pretraining_receipt_sha256 is not None
        path, receipt = run_confirmation_calibrate(
            protocol,
            _lexical_absolute(args.input_receipt),
            args.input_receipt_sha256,
            _lexical_absolute(args.confirmation_pretraining_receipt),
            args.confirmation_pretraining_receipt_sha256,
            _parse_confirmation_seal_bindings(args.confirmation_seal_receipt),
            output_dir,
        )
    else:
        assert args.input_receipt is not None
        assert args.input_receipt_sha256 is not None
        assert args.confirmation_pretraining_receipt is not None
        assert args.confirmation_pretraining_receipt_sha256 is not None
        path, receipt = run_confirmation_locked(
            protocol,
            _lexical_absolute(args.input_receipt),
            args.input_receipt_sha256,
            _lexical_absolute(args.confirmation_pretraining_receipt),
            args.confirmation_pretraining_receipt_sha256,
            output_dir,
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
