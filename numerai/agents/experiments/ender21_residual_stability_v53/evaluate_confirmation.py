"""Evaluate the frozen Ender21 family-locked confirmation exactly once."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import runpy
import stat
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_NAME = "ender21_residual_stability_v53"
CONFIRMATION_NAME = "c1_selected_tabm_k64_block_dro"
CONFIRMATION_ROWS = 263_551
DISCOVERY_BMC_BASELINE = 0.006876950492356912
ID = "id"
ERA = "era"
TARGET = "target_ender_20"
PREDICTION = "prediction"
BENCHMARK = "v53_lgbm_ender20"
CONFIRMATION_COLUMNS = [ID, ERA, TARGET, PREDICTION, BENCHMARK]
EXPECTED_CONFIRMATION_ERAS = tuple(f"{era:04d}" for era in range(865, 1022, 4))
EXPECTED_TRAIN_ERAS = tuple(f"{era:04d}" for era in range(161, 810, 4))
EXPECTED_EMBARGO_ERAS = tuple(f"{era:04d}" for era in range(813, 862, 4))
EXPECTED_SAMPLE_POSITIONS_SHA256 = (
    "a3a985613515c80e9a4d8d4c1c201ea497c7307786e0a0181730e5e0b69e0732"
)
PREDICTION_SEMANTICS_KEY = b"numerai.agents.prediction_semantics"
EXPECTED_PREDICTION_SEMANTICS = {
    "artifact_kind": "locked_holdout_prediction",
    "column": PREDICTION,
    "era_column": ERA,
    "inverse_target_transform_applied": False,
    "pipeline_postprocess": {"type": "identity"},
    "producer": "model.predict",
    "schema_version": 1,
    "training_target": {
        "type": "residual_to_benchmark",
        "benchmark_col": BENCHMARK,
        "era_col": ERA,
        "per_era": True,
        "fit_intercept": True,
        "proportion": 1.0,
    },
}
EXPECTED_INPUTS = {
    "discovery_full": (
        "numerai/v5.3/ender21_discovery_full_through_0861.parquet"
    ),
    "discovery_benchmark": (
        "numerai/v5.3/ender21_discovery_benchmark_models_through_0861.parquet"
    ),
    "confirmation_full": "numerai/v5.3/ender21_dev_full_through_1021.parquet",
    "confirmation_benchmark": (
        "numerai/v5.3/ender21_dev_benchmark_models_through_1021.parquet"
    ),
    "features_json": "numerai/v5.3/features.json",
}
EVALUATION_MANIFEST_NAME = "source_manifest_confirmation_evaluation.json"
TRAINING_MANIFEST_RELATIVE = (
    "numerai/agents/experiments/ender21_residual_stability_v53/"
    "source_manifest_confirmation.json"
)
TRAINING_CODE_COMMIT = "dcd3dd10eff7b121d6fff6c285680f527e588207"
TRAINING_MANIFEST_COMMIT = "2cc2bde920e980806d075266f06c561536c4c202"
TRAINING_EVIDENCE_COMMIT = "366c1219fcff342efb124068b1690b4c2fb70f99"
TRAINING_MANIFEST_SHA256 = (
    "186c5b92eda6400f218c31a301955126fe65a92c550167cd12fa8c31a45325ca"
)
TRAINING_MANIFEST_GIT_BLOB = "b6889a09cea85a1b4fb3e3ec90548cd452971679"
EXPECTED_EVIDENCE_BLOBS = {
    "numerai/agents/experiments/ender21_residual_stability_v53/"
    "logs/c1_selected_tabm_k64_block_dro.log": (
        "ff5836fd54996cd70912cfd325b1ca4f15e52dde"
    ),
    "numerai/agents/experiments/ender21_residual_stability_v53/"
    "models/c1_selected_tabm_k64_block_dro/predictor_spec.json": (
        "2abcb1a91d003918dddf28d600cb88b3a00166c6"
    ),
    "numerai/agents/experiments/ender21_residual_stability_v53/"
    "models/c1_selected_tabm_k64_block_dro/provenance.json": (
        "159ee9f4b8113dfa35bbf2a8f63ea738dbc1aa57"
    ),
    "numerai/agents/experiments/ender21_residual_stability_v53/"
    "models/c1_selected_tabm_k64_block_dro/sample_manifest_positions.npy": (
        "b62a7a1c85789ee4cdb3482ce04ad74904c37473"
    ),
    "numerai/agents/experiments/ender21_residual_stability_v53/"
    "models/c1_selected_tabm_k64_block_dro/weights.npz": (
        "e452dac30cc214bedf6ce65fe419a59656a65f81"
    ),
    "numerai/agents/experiments/ender21_residual_stability_v53/"
    "predictions/c1_selected_tabm_k64_block_dro.parquet": (
        "a86fe3d3dce44c53135f1a4a0fdfbebe8138d1f6"
    ),
    "numerai/agents/experiments/ender21_residual_stability_v53/"
    "receipts/c1_selected_tabm_k64_block_dro.completion.json": (
        "2448d8d776a11cb2afbd1f3700ec55dbe2ce63b0"
    ),
    "numerai/agents/experiments/ender21_residual_stability_v53/"
    "results/c1_selected_tabm_k64_block_dro.json": (
        "d1ce20d111530358685ecbca036053ee050b902e"
    ),
}
EVIDENCE_LEASE_KEYS = {
    relative: (
        "training_log"
        if "/logs/" in relative
        else "prediction"
        if "/predictions/" in relative
        else "completion"
        if "/receipts/" in relative
        else "result"
        if "/results/" in relative
        else f"bundle:{Path(relative).name}"
    )
    for relative in EXPECTED_EVIDENCE_BLOBS
}
EXPECTED_TRAINING_AUTHORITY = {
    "source_manifest": {
        "path": TRAINING_MANIFEST_RELATIVE,
        "sha256": TRAINING_MANIFEST_SHA256,
        "git_blob": TRAINING_MANIFEST_GIT_BLOB,
        "git_head": TRAINING_CODE_COMMIT,
        "manifest_commit": TRAINING_MANIFEST_COMMIT,
    },
    "evidence_commit": TRAINING_EVIDENCE_COMMIT,
    "evidence_blobs": EXPECTED_EVIDENCE_BLOBS,
}
EXPECTED_EVALUATION_MANIFEST_FILES = frozenset(
    {
        "numerai/agents/code/analysis/ender21_confirmation_rules.py",
        "numerai/agents/code/metrics/numerai_metrics.py",
        "numerai/agents/code/modeling/utils/constants.py",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "evaluate_confirmation.py",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "protocol/confirmation_train_eras_through_0809.json",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "protocol/confirmation_embargo_eras_0813_through_0861.json",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "protocol/confirmation_eras_0865_through_1021.json",
    }
)
EXPECTED_EVALUATION_EXTERNAL_ARTIFACTS = {
    EXPECTED_INPUTS["confirmation_full"]: {
        "last_era": "1021",
        "sha256": "952d043bf1673ff525124defd9f24a7e110a92909de1bc2feec35a119c45ae71",
        "size_bytes": 1_637_635_919,
    },
    EXPECTED_INPUTS["confirmation_benchmark"]: {
        "last_era": "1021",
        "sha256": "f435b4aed17c759011709ff2d8a8a4f235a7444154a740733fb80b55d7350605",
        "size_bytes": 29_796_569,
    },
}
EXPECTED_EVALUATION_RUNTIME = {
    "python": "3.13.14",
    "packages": {
        "numpy": "2.5.1",
        "pandas": "3.0.5",
        "pyarrow": "25.0.0",
        "numerai-tools": "0.6.0",
        "numerapi": "2.23.3",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_plain_file(path: Path, label: str) -> os.stat_result:
    try:
        inspected = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing {label}: {path}") from error
    attributes = getattr(inspected, "st_file_attributes", 0)
    if path.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ):
        raise ValueError(f"{label} may not be a symlink or reparse point.")
    if not stat.S_ISREG(inspected.st_mode) or inspected.st_nlink != 1:
        raise ValueError(f"{label} must be a unique regular file.")
    return inspected


def _require_plain_directory(path: Path, label: str) -> os.stat_result:
    inspected = path.lstat()
    attributes = getattr(inspected, "st_file_attributes", 0)
    if path.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ) or not stat.S_ISDIR(inspected.st_mode):
        raise ValueError(f"{label} must be a plain directory.")
    return inspected


def _require_plain_directory_chain(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    for directory in reversed([absolute, *absolute.parents]):
        if directory == directory.parent:
            continue
        inspected = directory.lstat()
        attributes = getattr(inspected, "st_file_attributes", 0)
        if directory.is_symlink() or bool(
            attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ) or not stat.S_ISDIR(inspected.st_mode):
            raise ValueError(f"{label} parent chain is not plain: {directory}")


def _require_frozen_launch_policy() -> None:
    """Require an immutable no-bytecode launch and an external empty cache root."""

    if sys.flags.dont_write_bytecode != 1 or sys.dont_write_bytecode is not True:
        raise ValueError("Confirmation evaluator must launch with immutable -B.")
    options = getattr(sys, "_xoptions", None)
    option_prefix = (
        options.get("pycache_prefix") if isinstance(options, dict) else None
    )
    live_prefix = getattr(sys, "pycache_prefix", None)
    if (
        type(option_prefix) is not str
        or not option_prefix
        or type(live_prefix) is not str
        or live_prefix != option_prefix
    ):
        raise ValueError(
            "Confirmation evaluator requires an exact -X pycache_prefix."
        )
    prefix = Path(option_prefix)
    if not prefix.is_absolute() or Path(os.path.abspath(prefix)) != prefix:
        raise ValueError("Confirmation pycache_prefix must be absolute and canonical.")
    try:
        prefix.relative_to(REPO_DIR)
    except ValueError:
        pass
    else:
        raise ValueError("Confirmation pycache_prefix must be outside the repository.")
    _require_plain_directory_chain(prefix.parent, "confirmation pycache_prefix")
    _require_plain_directory(prefix, "confirmation pycache_prefix")
    if next(prefix.iterdir(), None) is not None:
        raise ValueError("Confirmation pycache_prefix must be empty at launch.")


class _BootstrapReadOnlyLease:
    """Pin one regular file without sharing write or delete access on Windows."""

    def __init__(self, path: Path, label: str) -> None:
        self.path = Path(os.path.abspath(path))
        self.label = label
        self.stream = None
        _require_plain_directory_chain(self.path.parent, label)
        expected = _require_plain_file(self.path, label)
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            import msvcrt

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            create_file = kernel32.CreateFileW
            create_file.argtypes = (
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            )
            create_file.restype = wintypes.HANDLE
            handle = create_file(
                str(self.path),
                0x80000000,  # GENERIC_READ
                0x00000001,  # FILE_SHARE_READ only
                None,
                3,  # OPEN_EXISTING
                0x00000080,  # FILE_ATTRIBUTE_NORMAL
                None,
            )
            invalid = ctypes.c_void_p(-1).value
            if handle == invalid:
                error = ctypes.WinError(ctypes.get_last_error())
                raise ValueError(f"Cannot lease {label}: {self.path}") from error
            try:
                descriptor = msvcrt.open_osfhandle(
                    int(handle), os.O_RDONLY | getattr(os, "O_BINARY", 0)
                )
            except BaseException:
                kernel32.CloseHandle(handle)
                raise
            self.stream = os.fdopen(descriptor, "rb", buffering=0)
        else:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self.path, flags)
            self.stream = os.fdopen(descriptor, "rb", buffering=0)
        observed = os.fstat(self.stream.fileno())
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or int(observed.st_dev) != int(expected.st_dev)
            or int(observed.st_ino) != int(expected.st_ino)
            or int(observed.st_size) != int(expected.st_size)
        ):
            self.close()
            raise ValueError(f"Leased {label} identity changed during open.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def fileno(self) -> int:
        if self.stream is None:
            raise RuntimeError(f"Lease is closed: {self.label}")
        return self.stream.fileno()

    def read_bytes(self) -> bytes:
        if self.stream is None:
            raise RuntimeError(f"Lease is closed: {self.label}")
        self.stream.seek(0)
        value = self.stream.read()
        self.stream.seek(0)
        return value

    def sha256(self, *, chunk_size: int = 8 * 1024 * 1024) -> str:
        if self.stream is None:
            raise RuntimeError(f"Lease is closed: {self.label}")
        digest = hashlib.sha256()
        self.stream.seek(0)
        while chunk := self.stream.read(chunk_size):
            digest.update(chunk)
        self.stream.seek(0)
        return digest.hexdigest()

    def git_blob_oid(self, *, chunk_size: int = 8 * 1024 * 1024) -> str:
        """Hash the held bytes using this repository's SHA-1 blob format."""

        if self.stream is None:
            raise RuntimeError(f"Lease is closed: {self.label}")
        size = os.fstat(self.stream.fileno()).st_size
        digest = hashlib.sha1(f"blob {size}\0".encode("ascii"))
        self.stream.seek(0)
        while chunk := self.stream.read(chunk_size):
            digest.update(chunk)
        self.stream.seek(0)
        return digest.hexdigest()

    def stat(self) -> os.stat_result:
        return os.fstat(self.fileno())

    def close(self) -> None:
        stream = self.stream
        self.stream = None
        if stream is not None:
            stream.close()


def _run_git(*arguments: str, allow_one: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in ({0, 1} if allow_one else {0}):
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"Confirmation Git check failed for {arguments}: {detail}")
    return result


def _canonical_repo_file(relative_text: str, label: str) -> Path:
    if not isinstance(relative_text, str) or not relative_text:
        raise ValueError(f"{label} path must be a non-empty repository path.")
    relative = Path(relative_text)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"{label} path is not canonical: {relative_text}")
    absolute = Path(os.path.abspath(REPO_DIR / relative))
    try:
        absolute.relative_to(REPO_DIR)
    except ValueError as error:
        raise ValueError(f"{label} path escapes the repository.") from error
    return absolute


def _exact_commit_parent(commit: str) -> str:
    line = _run_git("rev-list", "--parents", "-n", "1", commit).stdout.strip()
    parts = line.split()
    if len(parts) != 2 or parts[0] != commit:
        raise ValueError(f"Confirmation checkpoint is not a single-parent commit: {commit}")
    return parts[1]


def _commit_changed_paths(commit: str) -> tuple[str, ...]:
    return tuple(
        line
        for line in _run_git(
            "diff-tree", "--no-commit-id", "--name-only", "-r", commit
        ).stdout.splitlines()
        if line
    )


def _require_exact_evaluation_topology(evaluation_code_commit: str) -> None:
    """Bind the evaluator seal to the immutable training/evidence lineage."""

    for commit in (
        TRAINING_CODE_COMMIT,
        TRAINING_MANIFEST_COMMIT,
        TRAINING_EVIDENCE_COMMIT,
        evaluation_code_commit,
    ):
        resolved = _run_git(
            "rev-parse", "--verify", f"{commit}^{{commit}}"
        ).stdout.strip()
        if resolved != commit:
            raise ValueError(f"Confirmation checkpoint does not resolve exactly: {commit}")
    if _exact_commit_parent(TRAINING_MANIFEST_COMMIT) != TRAINING_CODE_COMMIT:
        raise ValueError("Training-manifest checkpoint topology differs.")
    if _exact_commit_parent(TRAINING_EVIDENCE_COMMIT) != TRAINING_MANIFEST_COMMIT:
        raise ValueError("Training-evidence checkpoint topology differs.")
    if _exact_commit_parent(evaluation_code_commit) != TRAINING_EVIDENCE_COMMIT:
        raise ValueError("Evaluation-code checkpoint topology differs.")
    if set(_commit_changed_paths(TRAINING_MANIFEST_COMMIT)) != {
        TRAINING_MANIFEST_RELATIVE
    }:
        raise ValueError("Training-manifest checkpoint path set differs.")
    if set(_commit_changed_paths(TRAINING_EVIDENCE_COMMIT)) != set(
        EXPECTED_EVIDENCE_BLOBS
    ):
        raise ValueError("Training-evidence checkpoint path set differs.")
    for relative, expected_blob in EXPECTED_EVIDENCE_BLOBS.items():
        actual_blob = _run_git(
            "rev-parse", f"{TRAINING_EVIDENCE_COMMIT}:{relative}"
        ).stdout.strip()
        if actual_blob != expected_blob:
            raise ValueError(f"Training-evidence Git blob differs: {relative}")

    head = _run_git("rev-parse", "HEAD").stdout.strip()
    if not _is_hex(head, 40) or _exact_commit_parent(head) != evaluation_code_commit:
        raise ValueError("Evaluation-manifest seal topology differs.")
    expected_manifest = (
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        f"{EVALUATION_MANIFEST_NAME}"
    )
    if set(_commit_changed_paths(head)) != {expected_manifest}:
        raise ValueError("Evaluation-manifest seal path set differs.")


def _validate_evaluation_manifest_schema(manifest: object) -> dict:
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "frozen_at",
        "git_head",
        "hash_algorithm",
        "files",
        "external_artifacts",
        "runtime",
        "training_authority",
    }:
        raise ValueError("Confirmation evaluation manifest schema differs.")
    if manifest["schema_version"] != 1 or manifest["hash_algorithm"] != "sha256":
        raise ValueError("Confirmation evaluation manifest version differs.")
    if not _is_hex(manifest["git_head"], 40):
        raise ValueError("Confirmation evaluation Git checkpoint is malformed.")
    if (
        not isinstance(manifest["files"], dict)
        or set(manifest["files"]) != EXPECTED_EVALUATION_MANIFEST_FILES
    ):
        raise ValueError("Confirmation evaluation source set differs.")
    if manifest["external_artifacts"] != EXPECTED_EVALUATION_EXTERNAL_ARTIFACTS:
        raise ValueError("Confirmation evaluation external input set differs.")
    if manifest["runtime"] != EXPECTED_EVALUATION_RUNTIME:
        raise ValueError("Confirmation evaluation runtime contract differs.")
    if manifest["training_authority"] != EXPECTED_TRAINING_AUTHORITY:
        raise ValueError("Confirmation training authority differs.")
    return manifest


def _bootstrap_verify_and_lease_sources(
    stack: ExitStack,
    experiment: Path,
) -> tuple[dict, dict, dict[str, _BootstrapReadOnlyLease]]:
    """Verify the new evaluator seal and the immutable training authority."""

    expected_experiment = Path(
        os.path.abspath(
            REPO_DIR / "numerai/agents/experiments" / EXPERIMENT_NAME
        )
    )
    if Path(os.path.abspath(experiment)) != expected_experiment:
        raise ValueError("Confirmation bootstrap experiment path is not canonical.")
    manifest_path = experiment / EVALUATION_MANIFEST_NAME
    manifest_lease = stack.enter_context(
        _BootstrapReadOnlyLease(manifest_path, "confirmation evaluation manifest")
    )
    try:
        manifest = json.loads(manifest_lease.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Confirmation evaluation manifest is invalid JSON.") from error
    manifest = _validate_evaluation_manifest_schema(manifest)
    commit = manifest["git_head"]
    files = manifest["files"]
    if platform.python_version() != EXPECTED_EVALUATION_RUNTIME["python"]:
        raise ValueError("Confirmation bootstrap Python runtime differs.")
    for package, expected_version in EXPECTED_EVALUATION_RUNTIME["packages"].items():
        try:
            actual_version = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError as error:
            raise ValueError(
                f"Confirmation bootstrap runtime package is absent: {package}"
            ) from error
        if actual_version != expected_version:
            raise ValueError(
                f"Confirmation bootstrap runtime package drifted: {package}"
            )

    _require_exact_evaluation_topology(commit)
    manifest_relative = manifest_path.relative_to(REPO_DIR).as_posix()
    _require_committed_clean(manifest_path, "confirmation evaluation manifest")
    _run_git("cat-file", "-e", f"HEAD:{manifest_relative}")

    leases = {"source_manifest": manifest_lease}
    for relative_text in sorted(files):
        expected_hash = files[relative_text]
        if not _is_hex(expected_hash, 64):
            raise ValueError(f"Invalid confirmation source digest: {relative_text}")
        path = _canonical_repo_file(relative_text, "confirmation source")
        lease = stack.enter_context(
            _BootstrapReadOnlyLease(path, f"confirmation source {relative_text}")
        )
        if lease.sha256() != expected_hash:
            raise ValueError(f"Confirmation source hash drifted: {relative_text}")
        _run_git("cat-file", "-e", f"{commit}:{relative_text}")
        if _run_git(
            "diff", "--quiet", commit, "--", relative_text, allow_one=True
        ).returncode:
            raise ValueError(
                f"Confirmation source differs from checkpoint: {relative_text}"
            )
        leases[relative_text] = lease

    training_manifest_path = _canonical_repo_file(
        TRAINING_MANIFEST_RELATIVE, "training source manifest"
    )
    training_manifest_lease = stack.enter_context(
        _BootstrapReadOnlyLease(
            training_manifest_path, "immutable training source manifest"
        )
    )
    if training_manifest_lease.sha256() != TRAINING_MANIFEST_SHA256:
        raise ValueError("Immutable training source-manifest hash differs.")
    if training_manifest_lease.git_blob_oid() != TRAINING_MANIFEST_GIT_BLOB:
        raise ValueError("Immutable training source-manifest Git blob differs.")
    commit_blob = _run_git(
        "rev-parse", f"{TRAINING_MANIFEST_COMMIT}:{TRAINING_MANIFEST_RELATIVE}"
    ).stdout.strip()
    if commit_blob != TRAINING_MANIFEST_GIT_BLOB:
        raise ValueError("Training source-manifest commit binding differs.")
    _require_committed_clean(
        training_manifest_path, "immutable training source manifest"
    )
    try:
        training_manifest = json.loads(training_manifest_lease.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Immutable training source manifest is invalid JSON.") from error
    if (
        not isinstance(training_manifest, dict)
        or training_manifest.get("git_head") != TRAINING_CODE_COMMIT
        or training_manifest.get("hash_algorithm") != "sha256"
        or training_manifest.get("schema_version") != 1
    ):
        raise ValueError("Immutable training source-manifest envelope differs.")
    leases[TRAINING_MANIFEST_RELATIVE] = training_manifest_lease
    return manifest, training_manifest, leases


def _require_committed_clean(path: Path, label: str) -> None:
    try:
        relative = path.relative_to(REPO_DIR).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} must be inside the repository.") from error
    status = _run_git(
        "status", "--porcelain=v1", "--untracked-files=all", "--", relative
    ).stdout
    if status:
        raise ValueError(f"{label} is not committed and clean.")
    _run_git("cat-file", "-e", f"HEAD:{relative}")
    live_blob = _run_git("hash-object", f"--path={relative}", relative).stdout.strip()
    committed_blob = _run_git("rev-parse", f"HEAD:{relative}").stdout.strip()
    if live_blob != committed_blob:
        raise ValueError(f"{label} differs from its HEAD blob.")


def _acquire_evaluation_leases(
    stack: ExitStack,
    experiment: Path,
    numerai_dir: Path,
) -> dict[str, _BootstrapReadOnlyLease]:
    """Pin every scored input and evidence artifact through receipt durability."""

    bundle = experiment / f"models/{CONFIRMATION_NAME}"
    _require_plain_directory(bundle, "confirmation portable-bundle")
    expected_bundle_files = {
        "weights.npz",
        "predictor_spec.json",
        "sample_manifest_positions.npy",
        "provenance.json",
    }
    observed_bundle_files = {
        path.name for path in bundle.iterdir() if path.is_file()
    }
    if observed_bundle_files != expected_bundle_files or any(
        not path.is_file() for path in bundle.iterdir()
    ):
        raise ValueError("Confirmation portable-bundle directory differs.")
    paths = {
        "training_log": experiment / f"logs/{CONFIRMATION_NAME}.log",
        "completion": experiment
        / f"receipts/{CONFIRMATION_NAME}.completion.json",
        "result": experiment / f"results/{CONFIRMATION_NAME}.json",
        "prediction": experiment / f"predictions/{CONFIRMATION_NAME}.parquet",
        "confirmation_benchmark": numerai_dir
        / "v5.3/ender21_dev_benchmark_models_through_1021.parquet",
        "confirmation_full": numerai_dir
        / "v5.3/ender21_dev_full_through_1021.parquet",
        **{
            f"bundle:{filename}": bundle / filename
            for filename in sorted(expected_bundle_files)
        },
    }
    leases = {}
    for label, path in paths.items():
        leases[label] = stack.enter_context(
            _BootstrapReadOnlyLease(path, f"confirmation evaluation {label}")
        )
    return leases


def _validate_evidence_commit_leases(
    leases: dict[str, _BootstrapReadOnlyLease],
) -> None:
    """Require every held evidence file to equal its frozen Git blob and HEAD."""

    for relative, expected_blob in EXPECTED_EVIDENCE_BLOBS.items():
        lease = leases[EVIDENCE_LEASE_KEYS[relative]]
        path = _canonical_repo_file(relative, "training evidence")
        inspected = _require_plain_file(path, f"training evidence {relative}")
        held = lease.stat()
        if (
            int(held.st_dev) != int(inspected.st_dev)
            or int(held.st_ino) != int(inspected.st_ino)
            or int(held.st_size) != int(inspected.st_size)
        ):
            raise ValueError(f"Training-evidence lease identity differs: {relative}")
        _require_committed_clean(path, f"training evidence {relative}")
        head_blob = _run_git("rev-parse", f"HEAD:{relative}").stdout.strip()
        if head_blob != expected_blob or lease.git_blob_oid() != expected_blob:
            raise ValueError(f"Live training-evidence Git blob differs: {relative}")


def _validate_evaluation_external_leases(
    leases: dict[str, _BootstrapReadOnlyLease],
) -> None:
    for label in ("confirmation_full", "confirmation_benchmark"):
        relative = EXPECTED_INPUTS[label]
        expected = EXPECTED_EVALUATION_EXTERNAL_ARTIFACTS[relative]
        lease = leases[label]
        if (
            lease.stat().st_size != expected["size_bytes"]
            or lease.sha256() != expected["sha256"]
        ):
            raise ValueError(f"Confirmation evaluation external drifted: {relative}")


def _validate_file_receipt(
    receipt: object,
    expected_path: Path,
    label: str,
    *,
    require_identity: bool,
    lease: _BootstrapReadOnlyLease | None = None,
) -> None:
    expected_keys = {"path", "size_bytes", "sha256"}
    if require_identity:
        expected_keys |= {"device", "inode"}
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise ValueError(f"{label} receipt schema differs.")
    inspected = _require_plain_file(expected_path, label)
    if lease is not None:
        if lease.path != Path(os.path.abspath(expected_path)):
            raise ValueError(f"{label} lease path differs.")
        leased_stat = lease.stat()
        if (
            int(leased_stat.st_dev) != int(inspected.st_dev)
            or int(leased_stat.st_ino) != int(inspected.st_ino)
            or int(leased_stat.st_size) != int(inspected.st_size)
        ):
            raise ValueError(f"{label} lease identity differs.")
        actual_sha256 = lease.sha256()
    else:
        actual_sha256 = _sha256(expected_path)
    if (
        receipt["path"] != str(expected_path)
        or type(receipt["size_bytes"]) is not int
        or receipt["size_bytes"] != inspected.st_size
        or not _is_hex(receipt["sha256"], 64)
        or receipt["sha256"] != actual_sha256
    ):
        raise ValueError(f"{label} receipt differs from its artifact.")
    if require_identity and (
        type(receipt["device"]) is not int
        or type(receipt["inode"]) is not int
        or receipt["device"] != int(inspected.st_dev)
        or receipt["inode"] != int(inspected.st_ino)
    ):
        raise ValueError(f"{label} file identity differs from completion evidence.")


def _validate_prediction_artifact(path: Path) -> None:
    """Validate exact raw-prediction schema and frozen semantic metadata."""

    _require_plain_directory(path.parent, "confirmation prediction parent")
    _require_plain_file(path, "confirmation prediction")
    parquet = pq.ParquetFile(path)
    try:
        if parquet.schema.names != [ID, ERA, PREDICTION]:
            raise ValueError("Confirmation prediction Parquet columns differ.")
        metadata = parquet.schema_arrow.metadata or {}
        raw = metadata.get(PREDICTION_SEMANTICS_KEY)
        if raw is None:
            raise ValueError("Confirmation prediction semantics are absent.")
        try:
            semantics = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Confirmation prediction semantics are invalid.") from error
        if semantics != EXPECTED_PREDICTION_SEMANTICS:
            raise ValueError("Confirmation prediction semantics differ.")
        if parquet.metadata.num_rows != CONFIRMATION_ROWS:
            raise ValueError("Confirmation prediction Parquet row count differs.")
    finally:
        parquet.close()


def _protocol_binding(experiment: Path, filename: str, eras: tuple[str, ...]) -> dict:
    path = experiment / "protocol" / filename
    payload = path.read_bytes()
    parsed = json.loads(payload)
    if tuple(parsed) != eras:
        raise ValueError(f"Protocol era list differs: {filename}")
    return {
        "path": path.relative_to(REPO_DIR).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "era_count": len(eras),
        "first_era": eras[0],
        "last_era": eras[-1],
    }


def _validate_training_evidence(
    experiment: Path,
    manifest: dict,
    leases: dict[str, _BootstrapReadOnlyLease] | None = None,
) -> dict:
    """Validate all unscored training evidence before target data may be read."""

    completion_path = experiment / f"receipts/{CONFIRMATION_NAME}.completion.json"
    result_path = experiment / f"results/{CONFIRMATION_NAME}.json"
    prediction_path = experiment / f"predictions/{CONFIRMATION_NAME}.parquet"
    bundle_path = experiment / f"models/{CONFIRMATION_NAME}"
    _require_plain_file(completion_path, "confirmation completion receipt")
    _require_plain_file(result_path, "confirmation training result")
    _require_committed_clean(completion_path, "confirmation completion receipt")
    _require_committed_clean(result_path, "confirmation training result")
    completion_bytes = (
        leases["completion"].read_bytes()
        if leases is not None
        else completion_path.read_bytes()
    )
    try:
        completion = json.loads(completion_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Confirmation completion receipt is invalid JSON.") from error
    expected_keys = {
        "schema_version",
        "stage",
        "state",
        "component",
        "source_manifest",
        "config",
        "era_contract",
        "inputs",
        "sample",
        "training",
        "outputs",
    }
    if not isinstance(completion, dict) or set(completion) != expected_keys:
        raise ValueError("Confirmation completion schema differs.")
    if (
        completion["schema_version"] != 1
        or completion["stage"] != "ender21-confirmation-training-completion"
        or completion["state"] != "OUTPUTS_FINALIZED"
        or completion["component"] != CONFIRMATION_NAME
    ):
        raise ValueError("Confirmation completion envelope differs.")

    manifest_path = experiment / "source_manifest_confirmation.json"
    expected_manifest = {
        "path": manifest_path.relative_to(REPO_DIR).as_posix(),
        "sha256": _sha256(manifest_path),
        "git_head": manifest["git_head"],
    }
    config_path = experiment / f"configs/{CONFIRMATION_NAME}.py"
    config_relative = config_path.relative_to(REPO_DIR).as_posix()
    if completion["source_manifest"] != expected_manifest or completion[
        "config"
    ] != {
        "path": config_relative,
        "sha256": manifest["files"][config_relative],
    }:
        raise ValueError("Confirmation completion source binding differs.")

    expected_era_contract = {
        "fit": _protocol_binding(
            experiment,
            "confirmation_train_eras_through_0809.json",
            EXPECTED_TRAIN_ERAS,
        ),
        "embargo": _protocol_binding(
            experiment,
            "confirmation_embargo_eras_0813_through_0861.json",
            EXPECTED_EMBARGO_ERAS,
        ),
        "confirmation": _protocol_binding(
            experiment,
            "confirmation_eras_0865_through_1021.json",
            EXPECTED_CONFIRMATION_ERAS,
        ),
    }
    if completion["era_contract"] != expected_era_contract:
        raise ValueError("Confirmation era contract differs.")

    inputs = completion["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != set(EXPECTED_INPUTS):
        raise ValueError("Confirmation input receipt set differs.")
    external = manifest["external_artifacts"]
    for name, relative in EXPECTED_INPUTS.items():
        expected = {"path": relative, **external[relative]}
        if inputs[name] != expected:
            raise ValueError(f"Confirmation input binding differs: {name}")

    if completion["sample"] != {
        "method": "numpy.default_rng.choice_without_replacement",
        "seed": 1337,
        "source_rows": 880_075,
        "row_count": 500_000,
        "positions_sha256": EXPECTED_SAMPLE_POSITIONS_SHA256,
    }:
        raise ValueError("Confirmation sample contract differs.")
    training = completion["training"]
    if (
        not isinstance(training, dict)
        or set(training) != {
            "best_epoch",
            "model_seed",
            "sample_seed",
            "loss_mode",
            "inner_validation",
        }
        or isinstance(training["best_epoch"], bool)
        or not isinstance(training["best_epoch"], int)
        or training["best_epoch"] <= 0
        or training["best_epoch"] > 30
        or training["model_seed"] != 1337
        or training["sample_seed"] != 1337
        or training["loss_mode"] != "chronological_block_dro"
        or training["inner_validation"] != {
            "type": "recent_eras",
            "fraction": 0.1,
            "embargo": 13,
        }
    ):
        raise ValueError("Confirmation training contract differs.")

    outputs = completion["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != {
        "predictions",
        "result",
        "portable_bundle",
    }:
        raise ValueError("Confirmation completion output set differs.")
    _validate_file_receipt(
        outputs["predictions"], prediction_path, "confirmation predictions",
        require_identity=True,
        lease=leases["prediction"] if leases is not None else None,
    )
    _validate_prediction_artifact(prediction_path)
    _validate_file_receipt(
        outputs["result"], result_path, "confirmation result", require_identity=True,
        lease=leases["result"] if leases is not None else None,
    )
    bundle = outputs["portable_bundle"]
    if not isinstance(bundle, dict) or set(bundle) != {"path", "files"}:
        raise ValueError("Confirmation portable-bundle receipt schema differs.")
    if bundle["path"] != str(bundle_path):
        raise ValueError("Confirmation portable-bundle path differs.")
    _require_plain_directory(bundle_path, "confirmation portable-bundle")
    expected_bundle_files = {
        "weights.npz",
        "predictor_spec.json",
        "sample_manifest_positions.npy",
        "provenance.json",
    }
    if not isinstance(bundle["files"], dict) or set(bundle["files"]) != (
        expected_bundle_files
    ):
        raise ValueError("Confirmation portable-bundle file set differs.")
    for filename in sorted(expected_bundle_files):
        _validate_file_receipt(
            bundle["files"][filename],
            bundle_path / filename,
            f"confirmation bundle {filename}",
            require_identity=False,
            lease=leases[f"bundle:{filename}"] if leases is not None else None,
        )

    result_bytes = (
        leases["result"].read_bytes()
        if leases is not None
        else result_path.read_bytes()
    )
    try:
        result = json.loads(result_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Confirmation training result is invalid JSON.") from error
    if not isinstance(result, dict) or set(result) != {
        "schema_version",
        "stage",
        "state",
        "component",
        "source_manifest",
        "config",
        "era_contract",
        "inputs",
        "sample",
        "training",
        "output",
    }:
        raise ValueError("Confirmation training result schema differs.")
    if (
        result["schema_version"] != 1
        or result["stage"] != "ender21-family-locked-confirmation-prediction"
        or result["state"] != "PREDICTIONS_FINALIZED_UNSCORED"
        or result["component"] != CONFIRMATION_NAME
        or any(
            result[key] != completion[key]
            for key in (
                "source_manifest",
                "config",
                "era_contract",
                "inputs",
                "sample",
                "training",
            )
        )
        or result["output"] != {
            "prediction_file": str(prediction_path),
            "prediction_rows": CONFIRMATION_ROWS,
            "prediction_eras": 40,
            "portable_bundle": bundle,
        }
    ):
        raise ValueError("Confirmation training result envelope differs.")
    return completion


def _validate_confirmation_frame(
    frame: pd.DataFrame,
    expected_eras: tuple[str, ...],
) -> pd.DataFrame:
    """Validate and canonicalize the exact joined confirmation scoring cohort."""

    if (
        not isinstance(frame, pd.DataFrame)
        or len(frame.columns) != len(CONFIRMATION_COLUMNS)
        or set(frame.columns) != set(CONFIRMATION_COLUMNS)
    ):
        raise ValueError("Confirmation scoring frame columns differ from the freeze.")
    if len(frame) != CONFIRMATION_ROWS:
        raise ValueError("Confirmation scoring row count differs from the freeze.")
    if frame.empty or frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError("Confirmation scoring ids are empty, null, or duplicated.")
    if frame[ERA].isna().any():
        raise ValueError("Confirmation scoring eras contain nulls.")
    validated = frame.loc[:, CONFIRMATION_COLUMNS].copy()
    validated[ERA] = validated[ERA].astype(str)
    if tuple(sorted(validated[ERA].unique(), key=int)) != tuple(expected_eras):
        raise ValueError("Confirmation scoring era universe differs from the freeze.")
    counts = validated.groupby(ERA, sort=False).size()
    if len(counts) != 40 or (counts <= 0).any():
        raise ValueError("Confirmation scoring era coverage differs from the freeze.")
    for column in (TARGET, PREDICTION, BENCHMARK):
        if pd.api.types.is_bool_dtype(validated[column].dtype):
            raise ValueError(f"Confirmation {column} must be finite numeric data.")
        try:
            values = validated[column].to_numpy(dtype="float64", copy=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Confirmation {column} must be finite numeric data."
            ) from error
        if not np.isfinite(values).all():
            raise ValueError(f"Confirmation {column} contains non-finite values.")
    validated["__era_int"] = validated[ERA].map(int)
    validated = validated.sort_values(["__era_int", ID], kind="stable")
    validated = validated.drop(columns="__era_int").reset_index(drop=True)
    if tuple(validated[ERA].drop_duplicates()) != tuple(expected_eras):
        raise ValueError("Confirmation canonical era order differs from the freeze.")
    return validated


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite confirmation metric: {label}")
    return number


def _decision(checks: dict[str, bool]) -> dict:
    expected = {
        "bmc_floor",
        "sharpe_floor",
        "drawdown_ceiling",
        "corr_floor",
        "benchmark_corr_ceiling",
        "positive_block_count",
        "worst_block_floor",
        "discovery_bmc_retention",
    }
    if not isinstance(checks, dict) or set(checks) != expected:
        raise ValueError("Confirmation decision requires every exact frozen check.")
    if any(type(value) is not bool for value in checks.values()):
        raise ValueError("Confirmation check states must be exact booleans.")
    passed = all(checks.values())
    return {
        "passed": passed,
        "state": (
            "HISTORICAL_RESEARCH_PASS_FORWARD_VALIDATION_REQUIRED"
            if passed
            else "NEGATIVE"
        ),
    }


def _load_governed_scoring() -> tuple[object, object]:
    """Load scoring code only after its source bytes are verified and leased."""

    governed_modules = {
        "agents.code.metrics.numerai_metrics",
        "agents.code.modeling.utils.constants",
    }
    preloaded = sorted(governed_modules & set(sys.modules))
    if preloaded:
        raise RuntimeError(
            f"Governed scoring modules were imported before leasing: {preloaded}"
        )
    rules_path = REPO_DIR / (
        "numerai/agents/code/analysis/ender21_confirmation_rules.py"
    )
    rules = runpy.run_path(str(rules_path))
    if rules.get("DISCOVERY_BMC_BASELINE") != DISCOVERY_BMC_BASELINE:
        raise ValueError("Confirmation rules discovery baseline differs.")
    checks = rules.get("confirmation_checks")
    if not callable(checks):
        raise ValueError("Confirmation rules callable is unavailable.")
    metrics = importlib.import_module("agents.code.metrics.numerai_metrics")
    return metrics, checks


def _score_confirmation(
    frame: pd.DataFrame,
    numerai_metrics_module,
    confirmation_checks_fn,
) -> tuple[dict, dict]:
    bmc = numerai_metrics_module.per_era_bmc(
        frame, [PREDICTION], BENCHMARK, TARGET
    )
    corr = numerai_metrics_module.per_era_corr(frame, [PREDICTION], TARGET)
    similarity = numerai_metrics_module.per_era_pred_corr(
        frame, [PREDICTION], BENCHMARK
    )
    bmc_summary = numerai_metrics_module.score_summary(bmc[PREDICTION])
    corr_summary = numerai_metrics_module.score_summary(corr[PREDICTION])
    blocks = {}
    for index in range(4):
        block_eras = EXPECTED_CONFIRMATION_ERAS[index * 10 : (index + 1) * 10]
        value = bmc.loc[bmc.index.astype(str).isin(block_eras), PREDICTION].mean()
        blocks[str(index)] = _finite(value, f"chronological block {index} BMC")
    metrics = {
        "bmc": {
            key: _finite(value, f"BMC {key}")
            for key, value in bmc_summary.items()
        },
        "corr": {
            key: _finite(value, f"Corr {key}")
            for key, value in corr_summary.items()
        },
        "avg_corr_with_benchmark": _finite(
            similarity[PREDICTION].mean(), "benchmark correlation"
        ),
        "chronological_block_bmc": blocks,
    }
    checks = confirmation_checks_fn(
        {
            "bmc": {
                key: metrics["bmc"][key]
                for key in ("mean", "sharpe", "max_drawdown")
            },
            "corr": {"mean": metrics["corr"]["mean"]},
            "avg_corr_with_benchmark": metrics["avg_corr_with_benchmark"],
            "chronological_block_bmc": blocks,
        },
        DISCOVERY_BMC_BASELINE,
    )
    per_era = {
        str(era): {
            "bmc": _finite(bmc.loc[era, PREDICTION], f"era {era} BMC"),
            "corr": _finite(corr.loc[era, PREDICTION], f"era {era} Corr"),
            "corr_with_benchmark": _finite(
                similarity.loc[era, PREDICTION], f"era {era} benchmark correlation"
            ),
        }
        for era in bmc.index
    }
    return {"metrics": metrics, "checks": checks}, per_era


def _load_confirmation_truth(
    numerai_dir: Path,
    expected_eras: tuple[str, ...],
) -> pd.DataFrame:
    """Perform the evaluator's sole target-bearing confirmation read."""

    return pd.read_parquet(
        numerai_dir / "v5.3/ender21_dev_full_through_1021.parquet",
        columns=[ID, ERA, TARGET],
        filters=[(ERA, "in", list(expected_eras))],
    )


def _evaluate_confirmation(
    experiment: Path,
    numerai_dir: Path,
    custody_stack: ExitStack | None = None,
) -> dict:
    """Validate unscored evidence, open the target once, and derive the receipt."""

    if custody_stack is None:
        with ExitStack() as local_stack:
            return _evaluate_confirmation(experiment, numerai_dir, local_stack)

    # Independently verify and pin every governed source before running or importing
    # any of it. Pin all evidence and scored inputs before their validation/read.
    evaluation_manifest, training_manifest, source_leases = (
        _bootstrap_verify_and_lease_sources(
        custody_stack, experiment
        )
    )
    evidence_leases = _acquire_evaluation_leases(
        custody_stack, experiment, numerai_dir
    )
    _validate_evidence_commit_leases(evidence_leases)
    _validate_evaluation_external_leases(evidence_leases)
    numerai_metrics_module, confirmation_checks_fn = _load_governed_scoring()
    completion = _validate_training_evidence(
        experiment, training_manifest, evidence_leases
    )
    expected_eras = tuple(
        json.loads(
            source_leases[
                "numerai/agents/experiments/ender21_residual_stability_v53/"
                "protocol/confirmation_eras_0865_through_1021.json"
            ].read_bytes()
        )
    )
    if expected_eras != EXPECTED_CONFIRMATION_ERAS:
        raise ValueError("Confirmation era protocol differs from the freeze.")

    prediction = pd.read_parquet(
        experiment / f"predictions/{CONFIRMATION_NAME}.parquet",
        columns=[ID, ERA, PREDICTION],
    )
    benchmark = pd.read_parquet(
        numerai_dir / "v5.3/ender21_dev_benchmark_models_through_1021.parquet",
        columns=[ID, ERA, BENCHMARK],
        filters=[(ERA, "in", list(expected_eras))],
    )
    prediction[ERA] = prediction[ERA].astype(str)
    benchmark[ERA] = benchmark[ERA].astype(str)
    if (
        list(prediction.columns) != [ID, ERA, PREDICTION]
        or len(prediction) != CONFIRMATION_ROWS
        or prediction[ID].isna().any()
        or prediction[ID].duplicated().any()
        or prediction[ERA].isna().any()
        or tuple(sorted(prediction[ERA].unique(), key=int)) != expected_eras
        or not np.isfinite(
            prediction[PREDICTION].to_numpy(dtype="float64", copy=False)
        ).all()
    ):
        raise ValueError("Confirmation prediction cohort differs from the freeze.")
    if (
        list(benchmark.columns) != [ID, ERA, BENCHMARK]
        or len(benchmark) != CONFIRMATION_ROWS
        or benchmark[ID].isna().any()
        or benchmark[ID].duplicated().any()
        or benchmark[ERA].isna().any()
        or tuple(sorted(benchmark[ERA].unique(), key=int)) != expected_eras
        or not np.isfinite(
            benchmark[BENCHMARK].to_numpy(dtype="float64", copy=False)
        ).all()
    ):
        raise ValueError("Confirmation benchmark cohort differs from the freeze.")
    pretarget = prediction.merge(
        benchmark,
        on=[ID, ERA],
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(pretarget) != CONFIRMATION_ROWS or len(prediction) != CONFIRMATION_ROWS:
        raise ValueError("Prediction/benchmark confirmation cohort differs.")

    # This is the first target-bearing confirmation read in the evaluator.
    truth = _load_confirmation_truth(numerai_dir, expected_eras)
    if (
        list(truth.columns) != [ID, ERA, TARGET]
        or len(truth) != CONFIRMATION_ROWS
        or truth[ID].isna().any()
        or truth[ID].duplicated().any()
        or truth[ERA].isna().any()
    ):
        raise ValueError("Confirmation truth cohort differs from the freeze.")
    truth[ERA] = truth[ERA].astype(str)
    if tuple(sorted(truth[ERA].unique(), key=int)) != expected_eras:
        raise ValueError("Confirmation truth eras differ from the freeze.")
    frame = truth.merge(
        pretarget,
        on=[ID, ERA],
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    frame = _validate_confirmation_frame(frame[CONFIRMATION_COLUMNS], expected_eras)
    scoring, per_era = _score_confirmation(
        frame, numerai_metrics_module, confirmation_checks_fn
    )
    decision = _decision(scoring["checks"])
    return {
        "schema_version": 1,
        "stage": "ender21-family-locked-confirmation",
        **decision,
        "component": CONFIRMATION_NAME,
        "discovery_bmc_baseline": DISCOVERY_BMC_BASELINE,
        "rows": len(frame),
        "eras": len(expected_eras),
        "first_era": expected_eras[0],
        "last_era": expected_eras[-1],
        "evaluation_manifest": {
            "path": (
                experiment / EVALUATION_MANIFEST_NAME
            ).relative_to(REPO_DIR).as_posix(),
            "sha256": _sha256(experiment / EVALUATION_MANIFEST_NAME),
            "git_head": evaluation_manifest["git_head"],
        },
        "training_completion": completion,
        **scoring,
        "per_era": per_era,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--numerai-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    expected_experiment = Path(
        os.path.abspath(REPO_DIR / "numerai/agents/experiments" / EXPERIMENT_NAME)
    )
    experiment = Path(os.path.abspath(args.experiment))
    numerai_dir = Path(os.path.abspath(args.numerai_dir))
    output = Path(os.path.abspath(args.output))
    if experiment != expected_experiment:
        raise ValueError("Confirmation evaluator requires the canonical experiment.")
    if numerai_dir != Path(os.path.abspath(REPO_DIR / "numerai")):
        raise ValueError("Confirmation evaluator requires the canonical data root.")
    if output != experiment / "receipts/confirmation_research.json":
        raise ValueError("Confirmation evaluator output path differs from the freeze.")
    parent = output.parent
    _require_plain_directory(parent, "confirmation receipt parent")
    try:
        reserved = output.open("xb", buffering=0)
    except FileExistsError as error:
        raise FileExistsError(
            f"Refusing to overwrite confirmation receipt: {output}"
        ) from error
    with reserved, ExitStack() as custody_stack:
        _require_frozen_launch_policy()
        payload = _evaluate_confirmation(
            experiment, numerai_dir, custody_stack
        )
        encoded = (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        reserved.write(encoded)
        reserved.flush()
        os.fsync(reserved.fileno())
    print(
        json.dumps(
            {"state": payload["state"], "passed": payload["passed"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
