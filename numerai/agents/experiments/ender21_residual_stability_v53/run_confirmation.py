"""Fit and seal the single frozen Ender21 family-locked confirmation model.

This module deliberately does not score the confirmation target.  It reads only
IDs, eras, features, and the benchmark from the locked confirmation slice.  The
separate evaluator may open the target only after these outputs are finalized.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import importlib.machinery
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
from types import ModuleType


REPO_DIR = Path(__file__).resolve().parents[4]
PREDICTION_SEMANTICS_METADATA_KEY = b"numerai.agents.prediction_semantics"


CONFIRMATION_NAME = "c1_selected_tabm_k64_block_dro"
CONFIRMATION_ROWS = 263_551
TARGET = "target_ender_20"
BENCHMARK = "v53_lgbm_ender20"
ERA = "era"
ID = "id"
_MANIFEST_NAME = "source_manifest_confirmation.json"
_CONFIG_RELATIVE = (
    "numerai/agents/experiments/ender21_residual_stability_v53/"
    "configs/c1_selected_tabm_k64_block_dro.py"
)
_PARENT_CONFIG_RELATIVE = (
    "numerai/agents/experiments/ender21_residual_stability_v53/"
    "configs/r1_tabm_k64_block_dro.py"
)
_PROTOCOL_FILES = {
    "fit": "protocol/confirmation_train_eras_through_0809.json",
    "embargo": "protocol/confirmation_embargo_eras_0813_through_0861.json",
    "confirmation": "protocol/confirmation_eras_0865_through_1021.json",
}
_EXPECTED_ERA_BOUNDS = {
    "fit": (163, "0161", "0809"),
    "embargo": (13, "0813", "0861"),
    "confirmation": (40, "0865", "1021"),
}
_EXTERNAL_PATHS = {
    "discovery_full": "numerai/v5.3/ender21_discovery_full_through_0861.parquet",
    "discovery_benchmark": (
        "numerai/v5.3/ender21_discovery_benchmark_models_through_0861.parquet"
    ),
    "confirmation_full": "numerai/v5.3/ender21_dev_full_through_1021.parquet",
    "confirmation_benchmark": (
        "numerai/v5.3/ender21_dev_benchmark_models_through_1021.parquet"
    ),
    "features_json": "numerai/v5.3/features.json",
}
_EXPECTED_MANIFEST_FILES = frozenset(
    {
        "numerai/agents/code/analysis/ender21_confirmation_rules.py",
        "numerai/agents/code/metrics/numerai_metrics.py",
        "numerai/agents/code/modeling/deployment/tabm_export.py",
        "numerai/agents/code/modeling/deployment/tabm_numpy.py",
        "numerai/agents/code/modeling/deployment/final_fit_export.py",
        "numerai/agents/code/modeling/models/torch_tabular_regressor.py",
        "numerai/agents/code/modeling/utils/config.py",
        "numerai/agents/code/modeling/utils/constants.py",
        "numerai/agents/code/modeling/utils/data.py",
        "numerai/agents/code/modeling/utils/model_data.py",
        "numerai/agents/code/modeling/utils/model_factory.py",
        "numerai/agents/code/modeling/utils/numerai_cv.py",
        "numerai/agents/code/modeling/utils/pipeline.py",
        "numerai/agents/code/modeling/utils/target_transforms.py",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "configs/base_r1.py",
        _PARENT_CONFIG_RELATIVE,
        _CONFIG_RELATIVE,
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "development_extract_receipt.json",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "evaluate_confirmation.py",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "experiment.md",
        "numerai/agents/experiments/ender21_residual_stability_v53/gate.md",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "protocol/confirmation_train_eras_through_0809.json",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "protocol/confirmation_embargo_eras_0813_through_0861.json",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "protocol/confirmation_eras_0865_through_1021.json",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "protocol/feature_columns_all_v53.json",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "receipts/round1_discovery.json",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "receipts/round2_seed_replication.json",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "run_confirmation.py",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "source_manifest.json",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "source_manifest_round2.json",
    }
)
_EXPECTED_RUNTIME = {
    "python": "3.13.14",
    "packages": {
        "numpy": "2.5.1",
        "pandas": "3.0.5",
        "pyarrow": "25.0.0",
        "numerai-tools": "0.6.0",
        "numerapi": "2.23.3",
        "torch": "2.13.0+cu130",
        "tabm": "0.0.3",
        "rtdl-num-embeddings": "0.0.12",
    },
}
_GOVERNED_IMPORTS_AUTHORIZED = False


def _revoke_governed_imports() -> None:
    global _GOVERNED_IMPORTS_AUTHORIZED
    _GOVERNED_IMPORTS_AUTHORIZED = False


def _require_frozen_launch_policy() -> None:
    """Require immutable launch evidence and a fresh external cache root."""

    if sys.flags.dont_write_bytecode != 1 or sys.dont_write_bytecode is not True:
        raise ValueError("Confirmation Python must launch with immutable -B.")
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
            "Confirmation Python requires an exact -X pycache_prefix."
        )
    resolved = Path(option_prefix)
    if not resolved.is_absolute() or Path(os.path.abspath(resolved)) != resolved:
        raise ValueError("Confirmation pycache_prefix must be absolute and canonical.")
    try:
        resolved.relative_to(REPO_DIR)
    except ValueError:
        pass
    else:
        raise ValueError("Confirmation pycache_prefix must be outside the repository.")
    _bootstrap_require_plain_directory_chain(resolved.parent)
    inspected = resolved.lstat()
    attributes = getattr(inspected, "st_file_attributes", 0)
    if (
        resolved.is_symlink()
        or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
        or not stat.S_ISDIR(inspected.st_mode)
    ):
        raise ValueError("Confirmation pycache_prefix must be a plain directory.")
    if next(resolved.iterdir(), None) is not None:
        raise ValueError("Confirmation pycache_prefix must be empty at launch.")


def _load_governed_runtime() -> dict:
    """Import manifest-governed and third-party code only after source leasing."""

    if not _GOVERNED_IMPORTS_AUTHORIZED:
        raise RuntimeError(
            "Governed confirmation imports require verified held source leases."
        )

    governed_modules = {
        "agents.code.modeling.deployment.final_fit_export",
        "agents.code.modeling.deployment.tabm_export",
        "agents.code.modeling.deployment.tabm_numpy",
        "agents.code.modeling.models.torch_tabular_regressor",
        "agents.code.modeling.utils.config",
        "agents.code.modeling.utils.constants",
        "agents.code.modeling.utils.model_data",
        "agents.code.modeling.utils.model_factory",
        "agents.code.modeling.utils.target_transforms",
    }
    preloaded = sorted(governed_modules & set(sys.modules))
    if preloaded:
        raise RuntimeError(
            f"Governed confirmation modules were imported before leasing: {preloaded}"
        )

    # Bypass deployment/__init__.py, which is intentionally outside this frozen
    # source set.  The two exact leased submodules import only their named peers.
    deployment_name = "agents.code.modeling.deployment"
    if deployment_name not in sys.modules:
        deployment = ModuleType(deployment_name)
        deployment.__package__ = deployment_name
        deployment.__path__ = [
            str(REPO_DIR / "numerai/agents/code/modeling/deployment")
        ]
        deployment.__spec__ = importlib.machinery.ModuleSpec(
            deployment_name, loader=None, is_package=True
        )
        sys.modules[deployment_name] = deployment

    np = importlib.import_module("numpy")
    pd = importlib.import_module("pandas")
    pa = importlib.import_module("pyarrow")
    pq = importlib.import_module("pyarrow.parquet")
    final_fit_export = importlib.import_module(
        "agents.code.modeling.deployment.final_fit_export"
    )
    tabm_export = importlib.import_module(
        "agents.code.modeling.deployment.tabm_export"
    )
    config = importlib.import_module("agents.code.modeling.utils.config")
    model_data = importlib.import_module("agents.code.modeling.utils.model_data")
    model_factory = importlib.import_module("agents.code.modeling.utils.model_factory")

    return {
        "np": np,
        "pd": pd,
        "pa": pa,
        "pq": pq,
        "positions_sha256": final_fit_export._positions_sha256,
        "split_predictor_spec": final_fit_export._split_predictor_spec,
        "write_npy_fsynced": final_fit_export._write_npy_fsynced,
        "write_npz_fsynced": final_fit_export._write_npz_fsynced,
        "extract_tabm_numpy_predictor_spec": (
            tabm_export.extract_tabm_numpy_predictor_spec
        ),
        "load_config": config.load_config,
        "ModelDataBatch": model_data.ModelDataBatch,
        "build_model_data_loader": model_data.build_model_data_loader,
        "build_x_cols": model_data.build_x_cols,
        "build_model": model_factory.build_model,
    }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_lower_hex(value: object, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_plain_file(path: Path, label: str) -> os.stat_result:
    inspected = path.lstat()
    attributes = getattr(inspected, "st_file_attributes", 0)
    if (
        path.is_symlink()
        or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
        or not stat.S_ISREG(inspected.st_mode)
        or inspected.st_nlink != 1
    ):
        raise ValueError(f"{label} must be a regular unlinked file: {path}")
    return inspected


def _bootstrap_require_plain_directory_chain(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    for directory in reversed([absolute, *absolute.parents]):
        if directory == directory.parent or not os.path.lexists(directory):
            continue
        inspected = directory.lstat()
        attributes = getattr(inspected, "st_file_attributes", 0)
        if directory.is_symlink() or bool(
            attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise ValueError(f"Confirmation directory may not be a reparse point: {directory}")
        if not stat.S_ISDIR(inspected.st_mode):
            raise ValueError(f"Confirmation directory is not plain: {directory}")


class _BootstrapReadOnlyFileLease:
    def __init__(self, path: Path, label: str) -> None:
        self.path = Path(path)
        self.label = label
        self.stream = None
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            import msvcrt

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            create_file = kernel32.CreateFileW
            create_file.argtypes = (
                wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
            )
            create_file.restype = wintypes.HANDLE
            handle = create_file(
                str(self.path), 0x80000000, 0x00000001, None, 3, 0x00000080, None
            )
            if handle == ctypes.c_void_p(-1).value:
                raise ValueError(f"Cannot lease confirmation {label}: {path}") from ctypes.WinError(ctypes.get_last_error())
            try:
                descriptor = msvcrt.open_osfhandle(
                    int(handle), os.O_RDONLY | getattr(os, "O_BINARY", 0)
                )
            except BaseException:
                kernel32.CloseHandle(handle)
                raise
            self.stream = os.fdopen(descriptor, "rb", buffering=0)
        else:
            self.stream = self.path.open("rb", buffering=0)

    def read_bytes(self) -> bytes:
        if self.stream is None:
            raise RuntimeError(f"Confirmation {self.label} lease is closed.")
        self.stream.seek(0)
        value = self.stream.read()
        self.stream.seek(0)
        return value

    def sha256(self, *, chunk_size: int = 8 * 1024 * 1024) -> str:
        digest = hashlib.sha256()
        if self.stream is None:
            raise RuntimeError(f"Confirmation {self.label} lease is closed.")
        self.stream.seek(0)
        while chunk := self.stream.read(chunk_size):
            digest.update(chunk)
        self.stream.seek(0)
        return digest.hexdigest()

    def close(self) -> None:
        stream = self.stream
        self.stream = None
        if stream is not None:
            stream.close()


class _ExclusiveOutputReservations:
    def __init__(self, predictions_path: Path, results_path: Path, completion_path: Path):
        self.predictions_path = Path(os.path.abspath(predictions_path))
        self.results_path = Path(os.path.abspath(results_path))
        self.completion_path = Path(os.path.abspath(completion_path))
        self.predictions_stream = None
        self.results_stream = None
        self.completion_stream = None

    @staticmethod
    def _open(path: Path, label: str):
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            import msvcrt

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            create_file = kernel32.CreateFileW
            create_file.argtypes = (
                wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
            )
            create_file.restype = wintypes.HANDLE
            handle = create_file(
                str(path), 0xC0000000, 0x00000001, None, 1, 0x00000080, None
            )
            if handle == ctypes.c_void_p(-1).value:
                raise ValueError(f"Cannot reserve exclusive {label} output: {path}") from ctypes.WinError(ctypes.get_last_error())
            try:
                descriptor = msvcrt.open_osfhandle(
                    int(handle), os.O_RDWR | getattr(os, "O_BINARY", 0)
                )
            except BaseException:
                kernel32.CloseHandle(handle)
                raise
            return os.fdopen(descriptor, "w+b", buffering=0)
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        return os.fdopen(descriptor, "w+b", buffering=0)

    def __enter__(self):
        self.predictions_stream = self._open(self.predictions_path, "prediction")
        try:
            self.results_stream = self._open(self.results_path, "result")
            self.completion_stream = self._open(self.completion_path, "completion receipt")
        except BaseException:
            for stream in (self.results_stream, self.predictions_stream):
                if stream is not None:
                    stream.close()
            self.results_stream = self.predictions_stream = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for name in ("completion_stream", "results_stream", "predictions_stream"):
            stream = getattr(self, name)
            setattr(self, name, None)
            if stream is not None:
                stream.close()

    def _final_receipts(self) -> dict[str, dict[str, object]]:
        values = {}
        for label, path, stream in (
            ("predictions", self.predictions_path, self.predictions_stream),
            ("result", self.results_path, self.results_stream),
        ):
            if stream is None:
                raise RuntimeError("Confirmation output reservation is closed.")
            stream.flush(); os.fsync(stream.fileno())
            inspected = os.fstat(stream.fileno())
            path_stat = path.lstat()
            if inspected.st_nlink != 1 or inspected.st_size <= 0 or (
                inspected.st_dev, inspected.st_ino
            ) != (path_stat.st_dev, path_stat.st_ino):
                raise ValueError(f"Completed confirmation {label} identity differs.")
            digest = hashlib.sha256()
            position = stream.tell(); stream.seek(0)
            while chunk := stream.read(8 * 1024 * 1024): digest.update(chunk)
            stream.seek(position)
            values[label] = {
                "path": str(path), "device": int(inspected.st_dev),
                "inode": int(inspected.st_ino), "size_bytes": int(inspected.st_size),
                "sha256": digest.hexdigest(),
            }
        return values

    def completion_identities(self) -> dict[str, dict[str, object]]:
        return self._final_receipts()

    def write_completion(self, payload: dict) -> bytes:
        if self.completion_stream is None:
            raise RuntimeError("Confirmation completion reservation is closed.")
        if os.fstat(self.completion_stream.fileno()).st_size != 0:
            raise ValueError("Reserved confirmation completion is no longer empty.")
        value = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        self.completion_stream.write(value)
        self.completion_stream.flush(); os.fsync(self.completion_stream.fileno())
        return value


def _run_git(*arguments: str, allow_one: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *arguments], cwd=REPO_DIR, capture_output=True, text=True, check=False
    )
    if result.returncode not in ({0, 1} if allow_one else {0}):
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"Ender21 confirmation Git verification failed: {detail}")
    return result


def _acquire_confirmation_input_leases(
    stack: ExitStack,
    experiment: Path,
    manifest: dict,
) -> dict[str, _BootstrapReadOnlyFileLease]:
    """Pin all mutable bytes used by confirmation until completion is durable."""

    required_relatives = {
        "source_manifest": (
            experiment / _MANIFEST_NAME
        ).relative_to(REPO_DIR).as_posix(),
        "config": _CONFIG_RELATIVE,
        "parent_config": _PARENT_CONFIG_RELATIVE,
        "fit_eras": (
            experiment / _PROTOCOL_FILES["fit"]
        ).relative_to(REPO_DIR).as_posix(),
        "embargo_eras": (
            experiment / _PROTOCOL_FILES["embargo"]
        ).relative_to(REPO_DIR).as_posix(),
        "confirmation_eras": (
            experiment / _PROTOCOL_FILES["confirmation"]
        ).relative_to(REPO_DIR).as_posix(),
        "feature_columns": (
            experiment / "protocol/feature_columns_all_v53.json"
        ).relative_to(REPO_DIR).as_posix(),
        "round2_receipt": (
            experiment / "receipts/round2_seed_replication.json"
        ).relative_to(REPO_DIR).as_posix(),
        **_EXTERNAL_PATHS,
    }
    required_relatives.update(
        {
            f"source:{relative}": relative
            for relative in sorted(manifest["files"])
        }
    )
    leases: dict[str, _BootstrapReadOnlyFileLease] = {}
    leased_relatives: set[str] = set()
    for label, relative in required_relatives.items():
        if relative in leased_relatives:
            continue
        path = Path(os.path.abspath(REPO_DIR / relative))
        lease = _BootstrapReadOnlyFileLease(path, f"confirmation {label}")
        stack.callback(lease.close)
        expected = None
        if label == "source_manifest":
            try:
                if json.loads(lease.read_bytes()) != manifest:
                    raise ValueError("Leased confirmation manifest content differs.")
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Leased confirmation manifest is invalid JSON.") from exc
            expected = lease.sha256()
        elif relative in manifest["files"]:
            expected = manifest["files"][relative]
        elif relative in manifest["external_artifacts"]:
            expected = manifest["external_artifacts"][relative]["sha256"]
        if expected is None or lease.sha256() != expected:
            raise ValueError(f"Leased confirmation input hash differs: {relative}")
        lease.expected_sha256 = expected
        leases[label] = lease
        leased_relatives.add(relative)
    return leases


def verify_confirmation_manifest(experiment: Path, numerai_dir: Path) -> dict:
    """Verify the exact frozen source/runtime/input manifest before any data read."""

    experiment = Path(os.path.abspath(experiment))
    numerai_dir = Path(os.path.abspath(numerai_dir))
    expected_experiment = Path(
        os.path.abspath(
            REPO_DIR / "numerai/agents/experiments/ender21_residual_stability_v53"
        )
    )
    expected_numerai = Path(os.path.abspath(REPO_DIR / "numerai"))
    if experiment != expected_experiment or numerai_dir != expected_numerai:
        raise ValueError("Ender21 confirmation paths differ from the canonical repository.")
    manifest_path = experiment / _MANIFEST_NAME
    _require_plain_file(manifest_path, "Confirmation source manifest")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Confirmation source manifest is not valid UTF-8 JSON.") from exc
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "frozen_at",
        "git_head",
        "hash_algorithm",
        "files",
        "external_artifacts",
        "runtime",
    }:
        raise ValueError("Confirmation source manifest schema differs from the freeze.")
    if manifest["schema_version"] != 1 or manifest["hash_algorithm"] != "sha256":
        raise ValueError("Confirmation source manifest version or hash differs.")
    commit = manifest["git_head"]
    if not _is_lower_hex(commit, 40):
        raise ValueError("Confirmation source manifest git_head is invalid.")
    files = manifest["files"]
    external = manifest["external_artifacts"]
    if not isinstance(files, dict) or set(files) != _EXPECTED_MANIFEST_FILES:
        raise ValueError("Confirmation source manifest file set differs from the freeze.")
    if not isinstance(external, dict) or set(external) != set(_EXTERNAL_PATHS.values()):
        raise ValueError("Confirmation source manifest artifact set differs from the freeze.")
    if manifest["runtime"] != _EXPECTED_RUNTIME or platform.python_version() != _EXPECTED_RUNTIME["python"]:
        raise ValueError("Confirmation Python runtime differs from the freeze.")
    for package, version in _EXPECTED_RUNTIME["packages"].items():
        try:
            actual = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError as exc:
            raise ValueError(f"Confirmation runtime package is absent: {package}") from exc
        if actual != version:
            raise ValueError(f"Confirmation runtime package drifted: {package}")

    resolved = _run_git("rev-parse", "--verify", f"{commit}^{{commit}}").stdout.strip()
    if resolved != commit:
        raise ValueError("Confirmation frozen git_head does not resolve exactly.")
    if _run_git("merge-base", "--is-ancestor", commit, "HEAD", allow_one=True).returncode:
        raise ValueError("Confirmation frozen git_head is not an ancestor of HEAD.")
    manifest_relative = manifest_path.relative_to(REPO_DIR).as_posix()
    if _run_git(
        "status", "--porcelain=v1", "--untracked-files=all", "--", manifest_relative
    ).stdout:
        raise ValueError("Confirmation source manifest is not committed and clean.")
    _run_git("cat-file", "-e", f"HEAD:{manifest_relative}")

    for relative_text, expected_hash in sorted(files.items()):
        if not _is_lower_hex(expected_hash, 64):
            raise ValueError(f"Invalid confirmation source digest: {relative_text}")
        path = Path(os.path.abspath(REPO_DIR / relative_text))
        _require_plain_file(path, f"Confirmation source {relative_text}")
        if _sha256_file(path) != expected_hash:
            raise ValueError(f"Confirmation source hash drifted: {relative_text}")
        _run_git("cat-file", "-e", f"{commit}:{relative_text}")
        if _run_git(
            "diff", "--quiet", commit, "--", relative_text, allow_one=True
        ).returncode:
            raise ValueError(f"Confirmation source differs from git_head: {relative_text}")

    for relative_text, receipt in sorted(external.items()):
        path = Path(os.path.abspath(REPO_DIR / relative_text))
        _require_plain_file(path, f"Confirmation input {relative_text}")
        expected_keys = {"size_bytes", "sha256"}
        if path.suffix == ".parquet":
            expected_keys.add("last_era")
        if not isinstance(receipt, dict) or set(receipt) != expected_keys:
            raise ValueError(f"Invalid confirmation artifact receipt: {relative_text}")
        if (
            isinstance(receipt["size_bytes"], bool)
            or not isinstance(receipt["size_bytes"], int)
            or receipt["size_bytes"] <= 0
            or not _is_lower_hex(receipt["sha256"], 64)
            or path.stat().st_size != receipt["size_bytes"]
            or _sha256_file(path) != receipt["sha256"]
        ):
            raise ValueError(f"Confirmation artifact drifted: {relative_text}")
        if path.suffix == ".parquet":
            expected_last = "0861" if "discovery" in relative_text else "1021"
            if receipt["last_era"] != expected_last:
                raise ValueError(f"Confirmation artifact era bound differs: {relative_text}")
    return manifest


def _era_receipt(path: Path, eras: tuple[str, ...]) -> dict:
    data = path.read_bytes()
    return {
        "path": path.relative_to(REPO_DIR).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "era_count": len(eras),
        "first_era": eras[0],
        "last_era": eras[-1],
    }


def _load_era_contract(experiment: Path) -> dict:
    result: dict[str, object] = {"receipts": {}}
    sets: dict[str, set[str]] = {}
    for label, relative in _PROTOCOL_FILES.items():
        path = Path(os.path.abspath(experiment / relative))
        _require_plain_file(path, f"Confirmation {label} era list")
        try:
            value = json.loads(path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Confirmation {label} era list is invalid JSON.") from exc
        if (
            not isinstance(value, list)
            or any(not isinstance(era, str) or len(era) != 4 for era in value)
            or len(value) != len(set(value))
            or value != sorted(value, key=int)
        ):
            raise ValueError(f"Confirmation {label} era list is malformed.")
        eras = tuple(value)
        expected_count, expected_first, expected_last = _EXPECTED_ERA_BOUNDS[label]
        if (len(eras), eras[0], eras[-1]) != (
            expected_count,
            expected_first,
            expected_last,
        ):
            raise ValueError(f"Confirmation {label} era bounds differ.")
        result[label] = eras
        result["receipts"][label] = _era_receipt(path, eras)  # type: ignore[index]
        sets[label] = set(eras)
    if any(sets[left] & sets[right] for left, right in (("fit", "embargo"), ("fit", "confirmation"), ("embargo", "confirmation"))):
        raise ValueError("Confirmation fit, embargo, and holdout eras must be disjoint.")
    if [int(x) for x in result["fit"] + result["embargo"] + result["confirmation"]] != list(  # type: ignore[operator]
        range(161, 1022, 4)
    ):
        raise ValueError("Confirmation era contract is not the exact retained-era sequence.")
    return result


def _validate_config(config: dict, experiment: Path, *, load_config_fn=None) -> None:
    if load_config_fn is None:
        parent_source = (
            experiment / "configs/r1_tabm_k64_block_dro.py"
        ).read_text(encoding="utf-8")
        child_source = (
            experiment / f"configs/{CONFIRMATION_NAME}.py"
        ).read_text(encoding="utf-8")
        expected_parent = (
            'from pathlib import Path\nimport runpy\n\n\n'
            'variant = runpy.run_path(str(Path(__file__).with_name("base_r1.py")))["variant"]\n'
            'CONFIG = variant(\n'
            '    "r1_tabm_k64_block_dro",\n'
            '    loss_mode="chronological_block_dro",\n'
            '    tabm_arch_type="tabm",\n'
            ')\n'
        )
        if parent_source.replace("\r\n", "\n") != expected_parent:
            raise ValueError("Confirmation parent config source differs.")
        expected_child = (
            'from copy import deepcopy\nfrom pathlib import Path\nimport runpy\n\n\n'
            'parent = runpy.run_path(\n'
            '    str(Path(__file__).with_name("r1_tabm_k64_block_dro.py"))\n'
            ')["CONFIG"]\n'
            'CONFIG = deepcopy(parent)\n'
            f'CONFIG["output"]["results_name"] = "{CONFIRMATION_NAME}"\n'
        )
        if child_source.replace("\r\n", "\n") != expected_child:
            raise ValueError("Confirmation config source differs.")
        base = json.loads(json.dumps(config))
        expected_model = {
            "type": "TorchTabularRegressor",
            "x_groups": ["features", "era", "benchmark_models"],
            "target_transform": {
                "type": "residual_to_benchmark",
                "benchmark_col": BENCHMARK,
                "era_col": ERA,
                "per_era": True,
                "fit_intercept": True,
                "proportion": 1.0,
            },
        }
        if any(base.get("model", {}).get(k) != v for k, v in expected_model.items()):
            raise ValueError("Confirmation model family or target transform drifted.")
        parent = deepcopy(base)
        parent["output"]["results_name"] = "r1_tabm_k64_block_dro"
    else:
        parent = load_config_fn(experiment / "configs/r1_tabm_k64_block_dro.py")
    expected = deepcopy(parent)
    expected["output"]["results_name"] = CONFIRMATION_NAME
    if config != expected:
        raise ValueError("Confirmation config differs outside its frozen output name.")
    params = config["model"]["params"]
    training = config["training"]
    if (
        params.get("seed") != 1337
        or params.get("loss_mode") != "chronological_block_dro"
        or params.get("val_split") != "recent_eras"
        or params.get("val_fraction") != 0.1
        or params.get("internal_val_embargo") != 13
        or training.get("max_train_samples") != 500_000
        or training.get("sample_seed") != 1337
    ):
        raise ValueError("Confirmation seed, loss, sampling, or inner split drifted.")


def _output_paths(experiment: Path) -> dict[str, Path]:
    root = Path(os.path.abspath(experiment))
    return {
        "predictions": root / "predictions" / f"{CONFIRMATION_NAME}.parquet",
        "result": root / "results" / f"{CONFIRMATION_NAME}.json",
        "completion": root / "receipts" / f"{CONFIRMATION_NAME}.completion.json",
        "bundle": root / "models" / CONFIRMATION_NAME,
    }


class _ConfirmationOutputReservations:
    """Create-new reservation for every confirmation destination."""

    def __init__(self, experiment: Path, name: str) -> None:
        if name != CONFIRMATION_NAME:
            raise ValueError("Unknown Ender21 confirmation component.")
        self.paths = _output_paths(experiment)
        self._files = _ExclusiveOutputReservations(
            self.paths["predictions"], self.paths["result"], self.paths["completion"]
        )
        self._bundle_open = False

    def __enter__(self):
        for key in ("predictions", "result", "completion"):
            self.paths[key].parent.mkdir(parents=True, exist_ok=True)
        self.paths["bundle"].parent.mkdir(parents=True, exist_ok=True)
        for key in ("predictions", "result", "completion", "bundle"):
            _bootstrap_require_plain_directory_chain(self.paths[key].parent)
        if self.paths["bundle"].exists() or self.paths["bundle"].is_symlink():
            raise ValueError(
                f"Cannot reserve exclusive confirmation bundle: {self.paths['bundle']}"
            )
        try:
            self._files.__enter__()
        except BaseException as exc:
            raise ValueError(
                f"Cannot reserve exclusive confirmation outputs: {exc}"
            ) from exc
        try:
            self.paths["bundle"].mkdir()
            self._bundle_open = True
        except BaseException as exc:
            self._files.__exit__(type(exc), exc, exc.__traceback__)
            raise ValueError(
                f"Cannot reserve exclusive confirmation bundle: {self.paths['bundle']}"
            ) from exc
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._files.__exit__(exc_type, exc_value, traceback)
        self._bundle_open = False

    @property
    def predictions_stream(self):
        return self._files.predictions_stream

    @property
    def result_stream(self):
        return self._files.results_stream

    def publish_bundle(self) -> None:
        if not self._bundle_open or not self.paths["bundle"].is_dir():
            raise RuntimeError("Confirmation bundle reservation is not open.")
        if set(path.name for path in self.paths["bundle"].iterdir()) != {
            "weights.npz",
            "predictor_spec.json",
            "sample_manifest_positions.npy",
            "provenance.json",
        }:
            raise ValueError("Confirmation portable bundle file set differs.")

    def write_completion(self, payload: dict) -> bytes:
        return self._files.write_completion(payload)

    def completed_file_identities(self) -> dict[str, dict[str, object]]:
        values = self._files.completion_identities()
        return {"predictions": values["predictions"], "result": values["result"]}


def _read_features(path: Path) -> tuple[str, ...]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, list) or len(value) != 3555 or len(value) != len(set(value)):
        raise ValueError("Frozen confirmation feature list is malformed.")
    if any(not isinstance(item, str) or not item.startswith("feature_") for item in value):
        raise ValueError("Frozen confirmation feature names are malformed.")
    return tuple(value)


def _read_confirmation_predictors(
    full_path: Path,
    benchmark_path: Path,
    confirmation_eras: tuple[str, ...],
    feature_columns: tuple[str, ...],
    *,
    pd_module=None,
):
    """Read the locked slice without ever projecting the confirmation target."""

    full_columns = [ID, ERA, *feature_columns]
    if TARGET in full_columns:
        raise ValueError("Confirmation predictor projection may not contain target.")
    pd_module = pd_module or _load_governed_runtime()["pd"]
    frame = pd_module.read_parquet(
        full_path,
        columns=full_columns,
        filters=[(ERA, "in", list(confirmation_eras))],
    )
    benchmark = pd_module.read_parquet(
        benchmark_path,
        columns=[ID, ERA, BENCHMARK],
        filters=[(ERA, "in", list(confirmation_eras))],
    )
    frame[ERA] = frame[ERA].astype(str)
    benchmark[ERA] = benchmark[ERA].astype(str)
    joined = frame.merge(benchmark, on=[ID, ERA], how="left", validate="one_to_one")
    if (
        joined.empty
        or len(joined) != CONFIRMATION_ROWS
        or joined[ID].isna().any()
        or joined[ID].duplicated().any()
        or joined[BENCHMARK].isna().any()
        or tuple(sorted(joined[ERA].unique(), key=int)) != confirmation_eras
    ):
        raise ValueError("Confirmation predictor coverage differs from the frozen slice.")
    return joined


def _predict_target_free(
    model,
    confirmation,
    feature_columns: tuple[str, ...],
    benchmark_col: str = BENCHMARK,
    *,
    np_module=None,
    pd_module=None,
):
    if np_module is None or pd_module is None:
        runtime = _load_governed_runtime()
        np_module = np_module or runtime["np"]
        pd_module = pd_module or runtime["pd"]
    if TARGET in confirmation.columns:
        raise ValueError("Confirmation target must not be materialized before sealing.")
    x_columns = [*feature_columns, ERA, benchmark_col]
    if list(confirmation.columns) != [ID, ERA, *feature_columns, benchmark_col]:
        raise ValueError("Confirmation predictor columns differ from the frozen order.")
    raw = np_module.asarray(model.predict(confirmation[x_columns])).reshape(-1)
    if raw.shape != (len(confirmation),) or not np_module.isfinite(raw).all():
        raise ValueError("Confirmation model produced invalid raw predictions.")
    return pd_module.DataFrame(
        {
            ID: confirmation[ID].to_numpy(copy=True),
            ERA: confirmation[ERA].to_numpy(copy=True),
            "prediction": raw,
        }
    )


def _sample_fit_data(data, *, runtime=None):
    runtime = runtime or _load_governed_runtime()
    np_module = runtime["np"]
    row_count = len(data.X)
    if row_count != 880_075:
        raise ValueError(f"Confirmation fit coverage differs: {row_count}")
    positions = np_module.random.default_rng(1337).choice(
        row_count, size=500_000, replace=False
    )
    sampled = runtime["ModelDataBatch"](
        X=data.X.iloc[positions],
        y=data.y.iloc[positions],
        era=data.era.iloc[positions],
        id=data.id.iloc[positions] if data.id is not None else None,
    )
    if sampled.era.astype(str).nunique() != 163:
        raise ValueError("Confirmation sample does not represent every fit era.")
    return sampled, np_module.asarray(positions, dtype=np_module.int64)


def _write_bundle(staging: Path, model, features: tuple[str, ...], provenance: dict, positions, *, runtime=None) -> dict:
    runtime = runtime or _load_governed_runtime()
    raw_spec = runtime["extract_tabm_numpy_predictor_spec"](
        model, batch_size=32, era_column=ERA, prediction_column="prediction"
    )
    portable, arrays = runtime["split_predictor_spec"](raw_spec, features)
    weights = staging / "weights.npz"
    spec = staging / "predictor_spec.json"
    sample_positions = staging / "sample_manifest_positions.npy"
    provenance_path = staging / "provenance.json"
    runtime["write_npz_fsynced"](weights, arrays)
    with spec.open("xb") as stream:
        stream.write(_canonical_json_bytes(portable))
        stream.flush()
        os.fsync(stream.fileno())
    runtime["write_npy_fsynced"](sample_positions, positions)
    provisional = {
        "weights.npz": _sha256_file(weights),
        "predictor_spec.json": _sha256_file(spec),
        "sample_manifest_positions.npy": _sha256_file(sample_positions),
    }
    final_provenance = {**provenance, "files": provisional}
    with provenance_path.open("xb") as stream:
        stream.write(_canonical_json_bytes(final_provenance))
        stream.flush()
        os.fsync(stream.fileno())
    return {
        name: {"sha256": _sha256_file(staging / name), "size_bytes": (staging / name).stat().st_size}
        for name in ("weights.npz", "predictor_spec.json", "sample_manifest_positions.npy", "provenance.json")
    }


def run_confirmation(experiment: Path, numerai_dir: Path) -> tuple[Path, dict]:
    """Run the sole authorized confirmation fit and seal its unscored outputs."""

    _require_frozen_launch_policy()
    experiment = Path(os.path.abspath(experiment))
    numerai_dir = Path(os.path.abspath(numerai_dir))
    with ExitStack() as input_stack, _ConfirmationOutputReservations(
        experiment, CONFIRMATION_NAME
    ) as reserved:
        # Reserve every canonical destination before source/config/data inspection.
        # A failed preflight therefore leaves terminal create-new evidence and may
        # never be silently retried over the same confirmation paths.
        manifest = verify_confirmation_manifest(experiment, numerai_dir)
        leases = _acquire_confirmation_input_leases(
            input_stack, experiment, manifest
        )
        global _GOVERNED_IMPORTS_AUTHORIZED
        _GOVERNED_IMPORTS_AUTHORIZED = True
        input_stack.callback(_revoke_governed_imports)
        runtime = _load_governed_runtime()
        np_module = runtime["np"]
        pd_module = runtime["pd"]
        pa_module = runtime["pa"]
        pq_module = runtime["pq"]
        era_contract = _load_era_contract(experiment)
        config_path = Path(os.path.abspath(REPO_DIR / _CONFIG_RELATIVE))
        config = runtime["load_config"](config_path)
        _validate_config(
            config, experiment, load_config_fn=runtime["load_config"]
        )
        round2 = json.loads(leases["round2_receipt"].read_bytes())
        if (
            round2.get("state") != "SEED_REPLICATION_PASS"
            or round2.get("passed") is not True
        ):
            raise ValueError(
                "Confirmation requires the exact Round-2 passage receipt."
            )

        features_path = experiment / "protocol/feature_columns_all_v53.json"
        features = _read_features(features_path)
        paths = {
            key: Path(os.path.abspath(REPO_DIR / relative))
            for key, relative in _EXTERNAL_PATHS.items()
        }
        feature_metadata = json.loads(leases["features_json"].read_bytes())
        if feature_metadata.get("feature_sets", {}).get("all") != list(features):
            raise ValueError(
                "Frozen confirmation feature order differs from features.json."
            )

        fit_full = pd_module.read_parquet(
            paths["discovery_full"],
            columns=[ID, ERA, TARGET, *features],
            filters=[(ERA, "in", list(era_contract["fit"]))],
        )
        fit_benchmark = pd_module.read_parquet(
            paths["discovery_benchmark"],
            columns=[ID, ERA, BENCHMARK],
            filters=[(ERA, "in", list(era_contract["fit"]))],
        )
        fit_full[ERA] = fit_full[ERA].astype(str)
        fit_benchmark[ERA] = fit_benchmark[ERA].astype(str)
        fit = fit_full.merge(
            fit_benchmark, on=[ID, ERA], how="inner", validate="one_to_one"
        )
        if (
            len(fit) != 880_075
            or fit[ID].isna().any()
            or fit[ID].duplicated().any()
            or tuple(sorted(fit[ERA].unique(), key=int)) != era_contract["fit"]
        ):
            raise ValueError(
                "Confirmation fit rows/eras differ from the frozen contract."
            )
        x_cols = runtime["build_x_cols"](
            x_groups=config["model"]["x_groups"],
            features=features,
            benchmark_cols=[BENCHMARK],
            era_col=ERA,
            id_col=ID,
        )
        loader = runtime["build_model_data_loader"](
            full=fit, x_cols=x_cols, era_col=ERA, target_col=TARGET, id_col=ID
        )
        sampled, positions = _sample_fit_data(
            loader.load(era_contract["fit"]), runtime=runtime
        )
        model = runtime["build_model"](
            config["model"]["type"],
            deepcopy(config["model"]["params"]),
            deepcopy(config["model"]),
            feature_cols=list(features),
        )
        model.fit(sampled.X, sampled.y)
        best_epoch = getattr(model, "best_epoch_", None)
        if (
            isinstance(best_epoch, bool)
            or not isinstance(best_epoch, int)
            or not 1 <= best_epoch <= 30
        ):
            raise ValueError("Confirmation fit did not expose a valid best epoch.")
        confirmation = _read_confirmation_predictors(
            paths["confirmation_full"],
            paths["confirmation_benchmark"],
            era_contract["confirmation"],
            features,
            pd_module=pd_module,
        )
        predictions = _predict_target_free(
            model,
            confirmation,
            features,
            np_module=np_module,
            pd_module=pd_module,
        )
        semantics = {
            "artifact_kind": "locked_holdout_prediction",
            "column": "prediction",
            "era_column": ERA,
            "inverse_target_transform_applied": False,
            "pipeline_postprocess": {"type": "identity"},
            "producer": "model.predict",
            "schema_version": 1,
            "training_target": deepcopy(config["model"]["target_transform"]),
        }
        table = pa_module.Table.from_pandas(predictions, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata[PREDICTION_SEMANTICS_METADATA_KEY] = _canonical_json_bytes(semantics)
        table = table.replace_schema_metadata(metadata)
        pq_module.write_table(table, reserved.predictions_stream)
        reserved.predictions_stream.flush()
        os.fsync(reserved.predictions_stream.fileno())

        manifest_path = experiment / _MANIFEST_NAME
        manifest_identity = {
            "path": manifest_path.relative_to(REPO_DIR).as_posix(),
            "sha256": _sha256_file(manifest_path),
            "git_head": manifest["git_head"],
        }
        config_identity = {
            "path": config_path.relative_to(REPO_DIR).as_posix(),
            "sha256": manifest["files"][_CONFIG_RELATIVE],
        }
        input_receipts = {
            label: {"path": relative, **manifest["external_artifacts"][relative]}
            for label, relative in _EXTERNAL_PATHS.items()
        }
        sample = {
            "method": "numpy.default_rng.choice_without_replacement",
            "seed": 1337,
            "source_rows": 880_075,
            "row_count": 500_000,
            "positions_sha256": runtime["positions_sha256"](positions),
        }
        training = {
            "best_epoch": best_epoch,
            "model_seed": 1337,
            "sample_seed": 1337,
            "loss_mode": "chronological_block_dro",
            "inner_validation": {
                "type": "recent_eras",
                "fraction": 0.1,
                "embargo": 13,
            },
        }
        bundle_files = _write_bundle(
            reserved.paths["bundle"],
            model,
            features,
            {
                "schema_version": 1,
                "stage": "ender21-confirmation-portable-predictor",
                "component": CONFIRMATION_NAME,
                "config": config_identity,
                "source_manifest": manifest_identity,
                "sample": sample,
                "training": training,
            },
            positions,
            runtime=runtime,
        )
        reserved.publish_bundle()
        bundle_receipt = {
            "path": str(reserved.paths["bundle"]),
            "files": {
                filename: {
                    "path": str(reserved.paths["bundle"] / filename),
                    **receipt,
                }
                for filename, receipt in bundle_files.items()
            },
        }
        result = {
            "schema_version": 1,
            "stage": "ender21-family-locked-confirmation-prediction",
            "state": "PREDICTIONS_FINALIZED_UNSCORED",
            "component": CONFIRMATION_NAME,
            "source_manifest": manifest_identity,
            "config": config_identity,
            "era_contract": era_contract["receipts"],
            "inputs": input_receipts,
            "sample": sample,
            "training": training,
            "output": {
                "prediction_file": str(reserved.paths["predictions"]),
                "prediction_rows": len(predictions),
                "prediction_eras": int(predictions[ERA].nunique()),
                "portable_bundle": bundle_receipt,
            },
        }
        result_bytes = json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        reserved.result_stream.write(result_bytes)
        reserved.result_stream.flush()
        os.fsync(reserved.result_stream.fileno())
        output_receipts = reserved.completed_file_identities()
        completion = {
            "schema_version": 1,
            "stage": "ender21-confirmation-training-completion",
            "state": "OUTPUTS_FINALIZED",
            "component": CONFIRMATION_NAME,
            "source_manifest": manifest_identity,
            "config": config_identity,
            "era_contract": era_contract["receipts"],
            "inputs": input_receipts,
            "sample": sample,
            "training": training,
            "outputs": {**output_receipts, "portable_bundle": bundle_receipt},
        }
        for label, lease in leases.items():
            if lease.sha256() != lease.expected_sha256:
                raise ValueError(f"Leased confirmation input changed during run: {label}")
        reserved.write_completion(completion)
        return reserved.paths["completion"], completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--numerai-dir", type=Path, required=True)
    args = parser.parse_args()
    _require_frozen_launch_policy()
    path, payload = run_confirmation(args.experiment, args.numerai_dir)
    print(json.dumps({"path": str(path), "state": payload["state"]}, sort_keys=True))


if __name__ == "__main__":
    main()
