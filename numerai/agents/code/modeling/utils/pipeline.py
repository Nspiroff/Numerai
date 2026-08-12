from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
from types import ModuleType

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


_FROZEN_SOURCE_MODULES = {
    "agents.code.modeling.models.lgbm_regressor": (
        "numerai/agents/code/modeling/models/lgbm_regressor.py"
    ),
    "agents.code.modeling.utils.target_transforms": (
        "numerai/agents/code/modeling/utils/target_transforms.py"
    ),
}
_MISSING_MODULE = object()
_ENDER_EXPERIMENT_NAME = "ender20_aux_target_rank_ensemble_v53"
_SCOUT_CONFIG_BY_COMPONENT = {
    "jasper": "r1_jasper_d8_t6000.py",
    "teager2b": "r1_teager2b_d8_t6000.py",
    "victor": "r1_victor_d8_t6000.py",
    "tyler": "r1_tyler_d8_t6000.py",
}
_CONFIRMATION_COMPONENTS = (
    "jasper",
    "teager2b",
    "victor",
    "xerxes",
    "tyler",
)
_CONFIRMATION_RESULTS_NAMES = (
    *(f"confirmation_{component}_d8_t6000" for component in _CONFIRMATION_COMPONENTS),
)
_ENDER21_EXPERIMENT_NAME = "ender21_residual_stability_v53"
_ENDER21_ROUND1_NAMES = (
    "r1_control_tabm_k64",
    "r1_tabm_mini_k64",
    "r1_tabm_k64_era_balanced",
    "r1_tabm_k64_block_dro",
    "r1_tabm_mini_k64_block_dro",
)
_ENDER21_MANIFEST_FILES = frozenset(
    {
        "numerai/agents/code/modeling/models/torch_tabular_regressor.py",
        "numerai/agents/code/modeling/utils/config.py",
        "numerai/agents/code/modeling/utils/constants.py",
        "numerai/agents/code/modeling/utils/data.py",
        "numerai/agents/code/modeling/utils/model_data.py",
        "numerai/agents/code/modeling/utils/model_factory.py",
        "numerai/agents/code/modeling/utils/pipeline.py",
        "numerai/agents/code/modeling/utils/target_transforms.py",
        "numerai/agents/code/modeling/utils/numerai_cv.py",
        "numerai/agents/code/metrics/numerai_metrics.py",
        "numerai/agents/experiments/ender21_residual_stability_v53/experiment.md",
        "numerai/agents/experiments/ender21_residual_stability_v53/gate.md",
        (
            "numerai/agents/experiments/ender21_residual_stability_v53/"
            "development_extract_receipt.json"
        ),
        (
            "numerai/agents/experiments/ender21_residual_stability_v53/"
            "protocol/discovery_eras_through_0861.json"
        ),
        (
            "numerai/agents/experiments/ender21_residual_stability_v53/"
            "protocol/confirmation_eras_0865_through_1021.json"
        ),
        (
            "numerai/agents/experiments/ender21_residual_stability_v53/"
            "configs/base_r1.py"
        ),
        *(
            (
                "numerai/agents/experiments/ender21_residual_stability_v53/"
                f"configs/{name}.py"
            )
            for name in _ENDER21_ROUND1_NAMES
        ),
        (
            "numerai/agents/experiments/ender21_residual_stability_v53/"
            "evaluate_round1.py"
        ),
        (
            "numerai/agents/experiments/ender21_residual_stability_v53/"
            "tools/build_development_extract.py"
        ),
    }
)
_ENDER21_EXTERNAL_ARTIFACTS = frozenset(
    {
        "numerai/v5.3/ender21_discovery_full_through_0861.parquet",
        (
            "numerai/v5.3/"
            "ender21_discovery_benchmark_models_through_0861.parquet"
        ),
    }
)


def _governed_output_paths() -> set[Path]:
    experiment = Path(
        os.path.abspath(
            REPO_DIR
            / "numerai/agents/experiments"
            / _ENDER_EXPERIMENT_NAME
        )
    )
    names = (
        *(Path(name).stem for name in _SCOUT_CONFIG_BY_COMPONENT.values()),
        *_CONFIRMATION_RESULTS_NAMES,
    )
    governed = {
        *(experiment / "predictions" / f"{name}.parquet" for name in names),
        *(experiment / "results" / f"{name}.json" for name in names),
    }
    xerxes_experiment = Path(
        os.path.abspath(
            REPO_DIR
            / "numerai/agents/experiments/xerxes20_lgbm_challenger_v53"
        )
    )
    governed.update(
        {
            xerxes_experiment / "predictions/r1_depth8.parquet",
            xerxes_experiment / "results/r1_depth8.json",
        }
    )
    tabm_experiment = Path(
        os.path.abspath(
            REPO_DIR
            / "numerai/agents/experiments/ender20_nn_architecture_v53"
        )
    )
    tabm_names = (
        "r5_tabm_k64_train500k",
        "r6_tabm_k64_train500k_seed2027",
        "scale_disk_tabm_k64_train500k",
        "scale_disk_tabm_k64_train500k_seed2027",
    )
    governed.update(
        {
            *(tabm_experiment / "predictions" / f"{name}.parquet" for name in tabm_names),
            *(tabm_experiment / "results" / f"{name}.json" for name in tabm_names),
        }
    )
    ender21_experiment = Path(
        os.path.abspath(
            REPO_DIR
            / "numerai/agents/experiments"
            / _ENDER21_EXPERIMENT_NAME
        )
    )
    governed.update(
        {
            *(
                ender21_experiment / "predictions" / f"{name}.parquet"
                for name in _ENDER21_ROUND1_NAMES
            ),
            *(
                ender21_experiment / "results" / f"{name}.json"
                for name in _ENDER21_ROUND1_NAMES
            ),
        }
    )
    return governed


def _require_safe_unreserved_output(path: Path, label: str) -> None:
    """Reject aliases that could overwrite a governed artifact without authority."""

    candidate = Path(os.path.abspath(path))
    for directory in reversed([candidate.parent, *candidate.parent.parents]):
        if directory == directory.parent or not os.path.lexists(directory):
            continue
        inspected = directory.lstat()
        attributes = getattr(inspected, "st_file_attributes", 0)
        if directory.is_symlink() or bool(
            attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise ValueError(
                f"Unreserved {label} output parent may not be a reparse point: "
                f"{directory}"
            )
        if not stat.S_ISDIR(inspected.st_mode):
            raise ValueError(
                f"Unreserved {label} output parent is not a directory: {directory}"
            )

    if not os.path.lexists(candidate):
        return
    inspected = candidate.lstat()
    attributes = getattr(inspected, "st_file_attributes", 0)
    if candidate.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ):
        raise ValueError(
            f"Unreserved {label} output may not be a reparse point: {candidate}"
        )
    if not stat.S_ISREG(inspected.st_mode):
        raise ValueError(
            f"Unreserved {label} output is not a regular file: {candidate}"
        )
    if inspected.st_nlink != 1:
        raise ValueError(
            f"Unreserved {label} output may not be a hardlink: {candidate}"
        )
    candidate_identity = (int(inspected.st_dev), int(inspected.st_ino))
    for governed_path in _governed_output_paths():
        if not os.path.lexists(governed_path):
            continue
        governed_stat = governed_path.lstat()
        if candidate_identity == (
            int(governed_stat.st_dev),
            int(governed_stat.st_ino),
        ):
            raise ValueError(
                f"Unreserved {label} output aliases a governed artifact: {candidate}"
            )


@dataclass(frozen=True)
class _TrainingAuthority:
    mode: str
    component_name: str
    checkpoint: str
    protocol: object
    component: object
    pre_run_receipt_path: Path
    pre_run_receipt_sha256: str
    inventory_blob: str | None = None
    confirmation_pretraining_receipt_path: Path | None = None
    confirmation_pretraining_receipt_sha256: str | None = None
    data_leases: tuple = ()


class _ExclusiveOutputReservations:
    """Create and hold both authorized output paths without overwrite sharing."""

    def __init__(self, predictions_path: Path, results_path: Path) -> None:
        self.predictions_path = Path(os.path.abspath(predictions_path))
        self.results_path = Path(os.path.abspath(results_path))
        self.predictions_stream = None
        self.results_stream = None

    @staticmethod
    def _open(path: Path, label: str):
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
                str(path),
                0xC0000000,  # GENERIC_READ | GENERIC_WRITE
                0x00000001,  # FILE_SHARE_READ only
                None,
                1,  # CREATE_NEW
                0x00000080,  # FILE_ATTRIBUTE_NORMAL
                None,
            )
            invalid_handle = ctypes.c_void_p(-1).value
            if handle == invalid_handle:
                error = ctypes.WinError(ctypes.get_last_error())
                raise ValueError(
                    f"Cannot reserve exclusive {label} output: {path}"
                ) from error
            try:
                descriptor = msvcrt.open_osfhandle(
                    int(handle), os.O_RDWR | getattr(os, "O_BINARY", 0)
                )
            except BaseException:
                kernel32.CloseHandle(handle)
                raise
            return os.fdopen(descriptor, "w+b", buffering=0)
        try:
            descriptor = os.open(
                path,
                os.O_RDWR | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except OSError as error:
            raise ValueError(
                f"Cannot reserve exclusive {label} output: {path}"
            ) from error
        return os.fdopen(descriptor, "w+b", buffering=0)

    def __enter__(self):
        self.predictions_stream = self._open(
            self.predictions_path, "prediction"
        )
        try:
            self.results_stream = self._open(self.results_path, "result")
        except BaseException:
            self.predictions_stream.close()
            self.predictions_stream = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for name in ("results_stream", "predictions_stream"):
            stream = getattr(self, name)
            setattr(self, name, None)
            if stream is not None:
                stream.close()

    def identities(self) -> dict[str, dict[str, object]]:
        if self.predictions_stream is None or self.results_stream is None:
            raise RuntimeError("Exclusive output reservations are not open.")
        values = {}
        for label, path, stream in (
            ("predictions", self.predictions_path, self.predictions_stream),
            ("result", self.results_path, self.results_stream),
        ):
            inspected = os.fstat(stream.fileno())
            if inspected.st_nlink != 1 or inspected.st_size != 0:
                raise ValueError(f"Reserved {label} output identity is malformed.")
            values[label] = {
                "path": str(path),
                "device": int(inspected.st_dev),
                "inode": int(inspected.st_ino),
            }
        return values

    def completion_identities(self) -> dict[str, dict[str, object]]:
        """Hash final bytes through the same CREATE_NEW handles used to write them."""

        if self.predictions_stream is None or self.results_stream is None:
            raise RuntimeError("Exclusive output reservations are not open.")
        values = {}
        for label, path, stream in (
            ("predictions", self.predictions_path, self.predictions_stream),
            ("result", self.results_path, self.results_stream),
        ):
            stream.flush()
            os.fsync(stream.fileno())
            inspected = os.fstat(stream.fileno())
            path_inspected = path.lstat()
            if inspected.st_nlink != 1 or inspected.st_size <= 0:
                raise ValueError(f"Completed {label} output identity is malformed.")
            if (
                int(path_inspected.st_dev),
                int(path_inspected.st_ino),
            ) != (int(inspected.st_dev), int(inspected.st_ino)):
                raise ValueError(f"Completed {label} output path identity changed.")
            digest = hashlib.sha256()
            position = stream.tell()
            stream.seek(0)
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
            stream.seek(position)
            values[label] = {
                "path": str(path),
                "device": int(inspected.st_dev),
                "inode": int(inspected.st_ino),
                "size_bytes": int(inspected.st_size),
                "sha256": digest.hexdigest(),
            }
        return values


def _ender21_output_reservations(
    config_path: Path,
    output_dir_override: Path | None,
) -> _ExclusiveOutputReservations | None:
    """Reserve each frozen Ender21 output once, before config or data access."""

    experiment = Path(
        os.path.abspath(
            REPO_DIR / "numerai/agents/experiments" / _ENDER21_EXPERIMENT_NAME
        )
    )
    configs = experiment / "configs"
    supplied = Path(config_path)
    if any(part in {".", ".."} for part in supplied.parts):
        raise ValueError("Ender21 Round-1 config paths must be canonical.")
    lexical = Path(os.path.abspath(supplied))
    if lexical.parent != configs:
        return None
    if lexical.suffix != ".py" or lexical.stem not in _ENDER21_ROUND1_NAMES:
        raise ValueError("Only frozen named Ender21 Round-1 configs may execute.")
    if lexical != configs / f"{lexical.stem}.py":
        raise ValueError("Ender21 Round-1 config path is not canonical.")
    if output_dir_override is not None:
        raise ValueError("Ender21 Round-1 outputs may not be redirected.")
    _bootstrap_require_plain_directory_chain(experiment)
    _bootstrap_require_plain_file(lexical, "Ender21 Round-1 config")
    predictions_dir = experiment / "predictions"
    results_dir = experiment / "results"
    predictions_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    _bootstrap_require_plain_directory_chain(predictions_dir)
    _bootstrap_require_plain_directory_chain(results_dir)
    return _ExclusiveOutputReservations(
        predictions_dir / f"{lexical.stem}.parquet",
        results_dir / f"{lexical.stem}.json",
    )


def _bootstrap_require_plain_directory_chain(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    for directory in reversed([absolute, *absolute.parents]):
        if directory == directory.parent:
            continue
        inspected = directory.lstat()
        attributes = getattr(inspected, "st_file_attributes", 0)
        if directory.is_symlink() or bool(
            attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise ValueError(f"Bootstrap directory may not be a reparse point: {directory}")
        if not stat.S_ISDIR(inspected.st_mode):
            raise ValueError(f"Bootstrap path is not a directory: {directory}")


def _bootstrap_require_plain_file(path: Path, label: str) -> None:
    inspected = path.lstat()
    attributes = getattr(inspected, "st_file_attributes", 0)
    if path.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ):
        raise ValueError(f"Bootstrap {label} may not be a reparse point: {path}")
    if not stat.S_ISREG(inspected.st_mode) or inspected.st_nlink != 1:
        raise ValueError(f"Bootstrap {label} is not a unique regular file: {path}")


def _load_era_allowlist(value: object) -> tuple[tuple[str, ...] | None, dict | None]:
    """Load an exact, repo-bound era universe before any modeling data access."""

    if value is None:
        return None, None
    if not isinstance(value, str) or not value:
        raise ValueError("data.era_allowlist_path must be a non-empty repo-relative path.")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("data.era_allowlist_path must be canonical and repo-relative.")
    path = Path(os.path.abspath(REPO_DIR / relative))
    try:
        path.relative_to(REPO_DIR)
    except ValueError as error:
        raise ValueError("data.era_allowlist_path escapes the repository.") from error
    _bootstrap_require_plain_directory_chain(path.parent)
    _bootstrap_require_plain_file(path, "era allowlist")
    payload = path.read_bytes()
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("data.era_allowlist_path must contain valid UTF-8 JSON.") from error
    if (
        not isinstance(parsed, list)
        or not parsed
        or not all(isinstance(era, str) and era for era in parsed)
        or len(set(parsed)) != len(parsed)
    ):
        raise ValueError("Era allowlist must be a non-empty list of unique strings.")
    try:
        sorted_eras = sorted(parsed, key=lambda era: int(era))
    except ValueError as error:
        raise ValueError("Era allowlist entries must be integer-like strings.") from error
    if any(str(int(era)).zfill(4) != era for era in parsed):
        raise ValueError("Era allowlist entries must use canonical four-digit strings.")
    if parsed != sorted_eras:
        raise ValueError("Era allowlist must be in ascending chronological order.")
    return tuple(parsed), {
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "era_count": len(parsed),
        "first_era": parsed[0],
        "last_era": parsed[-1],
    }


def _filter_to_era_allowlist(
    full: pd.DataFrame,
    era_col: str,
    allowed_eras: tuple[str, ...] | None,
) -> pd.DataFrame:
    if allowed_eras is None:
        return full
    if era_col not in full.columns:
        raise ValueError(f"Era allowlist requires data column '{era_col}'.")
    eras = full[era_col].astype(str)
    present = set(eras.unique())
    missing = [era for era in allowed_eras if era not in present]
    if missing:
        raise ValueError(f"Era allowlist entries are absent from modeling data: {missing[:5]}")
    filtered = full.loc[eras.isin(set(allowed_eras))].copy()
    observed = sorted(filtered[era_col].astype(str).unique(), key=lambda era: int(era))
    if observed != list(allowed_eras):
        raise ValueError("Filtered modeling eras do not exactly equal the era allowlist.")
    print(
        "Restricted modeling data to the frozen era allowlist: "
        f"{len(filtered):,}/{len(full):,} rows, {len(observed)} eras."
    )
    return filtered


class _BootstrapReadOnlyFileLease:
    """Minimal stdlib-only immutable lease used before local source verification."""

    def __init__(self, path: Path, label: str) -> None:
        self.path = path
        self.label = label
        self.stream = None
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
                str(path),
                0x80000000,
                0x00000001,
                None,
                3,
                0x00000080,
                None,
            )
            invalid_handle = ctypes.c_void_p(-1).value
            if handle == invalid_handle:
                error = ctypes.WinError(ctypes.get_last_error())
                raise ValueError(f"Cannot lease bootstrap {label}: {path}") from error
            try:
                descriptor = msvcrt.open_osfhandle(
                    int(handle), os.O_RDONLY | getattr(os, "O_BINARY", 0)
                )
            except BaseException:
                kernel32.CloseHandle(handle)
                raise
            self.stream = os.fdopen(descriptor, "rb", buffering=0)
        else:
            self.stream = path.open("rb", buffering=0)

    def read_bytes(self) -> bytes:
        if self.stream is None:
            raise RuntimeError(f"Bootstrap {self.label} lease is closed.")
        self.stream.seek(0)
        value = self.stream.read()
        self.stream.seek(0)
        return value

    def fileno(self) -> int:
        if self.stream is None:
            raise RuntimeError(f"Bootstrap {self.label} lease is closed.")
        return self.stream.fileno()

    def size_bytes(self) -> int:
        return int(os.fstat(self.fileno()).st_size)

    def sha256(self, *, chunk_size: int = 8 * 1024 * 1024) -> str:
        if self.stream is None:
            raise RuntimeError(f"Bootstrap {self.label} lease is closed.")
        digest = hashlib.sha256()
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


def _verify_ender21_round1_manifest() -> dict:
    """Fail closed unless Round-1 source and physical inputs match the freeze."""

    experiment_relative = Path(
        "numerai/agents/experiments/ender21_residual_stability_v53"
    )
    manifest_relative = experiment_relative / "source_manifest.json"
    manifest_path = Path(os.path.abspath(REPO_DIR / manifest_relative))
    _bootstrap_require_plain_directory_chain(manifest_path.parent)
    _bootstrap_require_plain_file(manifest_path, "Ender21 source manifest")
    manifest_lease = _BootstrapReadOnlyFileLease(
        manifest_path, "Ender21 source manifest"
    )
    try:
        manifest_bytes = manifest_lease.read_bytes()
    finally:
        manifest_lease.close()
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Ender21 source manifest is not valid UTF-8 JSON.") from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "frozen_at",
        "git_head",
        "hash_algorithm",
        "files",
        "external_artifacts",
        "runtime",
    }:
        raise ValueError("Ender21 source manifest schema differs from the freeze.")
    if manifest["schema_version"] != 1 or manifest["hash_algorithm"] != "sha256":
        raise ValueError("Ender21 source manifest version or hash differs.")
    commit = manifest["git_head"]
    if not _is_lower_hex(commit, 40):
        raise ValueError("Ender21 source manifest git_head is not a commit SHA.")
    files = manifest["files"]
    external = manifest["external_artifacts"]
    runtime = manifest["runtime"]
    if not isinstance(files, dict) or set(files) != _ENDER21_MANIFEST_FILES:
        raise ValueError("Ender21 source manifest file set differs from the freeze.")
    if not isinstance(external, dict) or set(external) != _ENDER21_EXTERNAL_ARTIFACTS:
        raise ValueError("Ender21 source manifest artifact set differs from the freeze.")
    expected_runtime = {
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
    if runtime != expected_runtime or platform.python_version() != runtime["python"]:
        raise ValueError("Ender21 Python runtime differs from the freeze.")
    for package, expected_version in runtime["packages"].items():
        try:
            actual_version = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError as error:
            raise ValueError(f"Ender21 runtime package is absent: {package}") from error
        if actual_version != expected_version:
            raise ValueError(f"Ender21 runtime package drifted: {package}")

    def git(*arguments: str, allow_one: bool = False) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        allowed = {0, 1} if allow_one else {0}
        if result.returncode not in allowed:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ValueError(
                f"Ender21 Git verification failed for {arguments}: {detail}"
            )
        return result

    resolved = git("rev-parse", "--verify", f"{commit}^{{commit}}").stdout.strip()
    if resolved != commit:
        raise ValueError("Ender21 frozen git_head does not resolve exactly.")
    if git("merge-base", "--is-ancestor", commit, "HEAD", allow_one=True).returncode:
        raise ValueError("Ender21 frozen git_head is not an ancestor of HEAD.")
    status = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        manifest_relative.as_posix(),
    ).stdout
    if status:
        raise ValueError("Ender21 source manifest is not committed and clean.")
    git("cat-file", "-e", f"HEAD:{manifest_relative.as_posix()}")

    for relative_text in sorted(files):
        expected = files[relative_text]
        if not _is_lower_hex(expected, 64):
            raise ValueError(f"Invalid Ender21 source digest: {relative_text}")
        relative = Path(relative_text)
        path = Path(os.path.abspath(REPO_DIR / relative))
        _bootstrap_require_plain_directory_chain(path.parent)
        _bootstrap_require_plain_file(path, f"Ender21 source {relative_text}")
        lease = _BootstrapReadOnlyFileLease(path, f"Ender21 source {relative_text}")
        try:
            actual = lease.sha256()
        finally:
            lease.close()
        if actual != expected:
            raise ValueError(f"Ender21 source hash drifted: {relative_text}")
        git("cat-file", "-e", f"{commit}:{relative.as_posix()}")
        if git(
            "diff",
            "--quiet",
            commit,
            "--",
            relative.as_posix(),
            allow_one=True,
        ).returncode:
            raise ValueError(f"Ender21 source differs from git_head: {relative_text}")

    for relative_text in sorted(external):
        receipt = external[relative_text]
        if not isinstance(receipt, dict) or set(receipt) != {
            "size_bytes",
            "sha256",
            "last_era",
        }:
            raise ValueError(f"Invalid Ender21 artifact receipt: {relative_text}")
        if (
            isinstance(receipt["size_bytes"], bool)
            or not isinstance(receipt["size_bytes"], int)
            or receipt["size_bytes"] <= 0
            or not _is_lower_hex(receipt["sha256"], 64)
            or receipt["last_era"] != "0861"
        ):
            raise ValueError(f"Malformed Ender21 artifact receipt: {relative_text}")
        path = Path(os.path.abspath(REPO_DIR / relative_text))
        _bootstrap_require_plain_directory_chain(path.parent)
        _bootstrap_require_plain_file(path, f"Ender21 artifact {relative_text}")
        lease = _BootstrapReadOnlyFileLease(path, f"Ender21 artifact {relative_text}")
        try:
            if lease.size_bytes() != receipt["size_bytes"]:
                raise ValueError(f"Ender21 artifact size drifted: {relative_text}")
            actual = lease.sha256()
        finally:
            lease.close()
        if actual != receipt["sha256"]:
            raise ValueError(f"Ender21 artifact hash drifted: {relative_text}")
    return manifest


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _run_git(*arguments: str, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_DIR,
        text=text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _require_frozen_python_runtime() -> Path:
    """Require a fresh alternate cache root so adjacent ignored pyc is unreachable."""

    launch_prefix = getattr(sys, "_xoptions", {}).get("pycache_prefix")
    if (
        sys.flags.dont_write_bytecode != 1
        or not sys.dont_write_bytecode
        or sys.pycache_prefix is None
        or not isinstance(launch_prefix, str)
        or not launch_prefix
    ):
        raise ValueError(
            "Receipt-authorized training requires Python -B with an isolated "
            "-X pycache_prefix directory."
        )
    configured = Path(sys.pycache_prefix)
    launch_configured = Path(launch_prefix)
    if not configured.is_absolute() or not launch_configured.is_absolute():
        raise ValueError("Frozen Python pycache_prefix must be absolute.")
    prefix = Path(os.path.abspath(configured))
    if prefix != Path(os.path.abspath(launch_configured)):
        raise ValueError("Frozen Python pycache_prefix differs from launch state.")
    try:
        prefix.relative_to(Path(os.path.abspath(REPO_DIR)))
    except ValueError:
        pass
    else:
        raise ValueError("Frozen Python pycache_prefix must be outside the repository.")

    _bootstrap_require_plain_directory_chain(prefix)
    try:
        if next(prefix.iterdir(), None) is not None:
            raise ValueError("Frozen Python pycache_prefix must be freshly empty.")
    except OSError as error:
        raise ValueError("Frozen Python pycache_prefix cannot be inspected.") from error
    return prefix


def _lease_finalized_receipt_envelope(
    path: Path,
    expected_sha256: str,
    *,
    receipt_dir: Path,
    prefix: str,
    stage: str,
):
    """Parse one finalized receipt from the same leased bytes later revalidated."""

    if not _is_lower_hex(expected_sha256, 64):
        raise ValueError("Receipt hash must be a lowercase SHA-256 digest.")
    if ".." in Path(path).parts or ".." in Path(receipt_dir).parts:
        raise ValueError(f"{prefix} receipt path contains parent traversal.")
    path = Path(os.path.abspath(path))
    receipt_dir = Path(os.path.abspath(receipt_dir))
    _bootstrap_require_plain_directory_chain(receipt_dir)
    if path.parent != receipt_dir:
        raise ValueError(f"{prefix} receipt has a noncanonical parent.")
    if path.name != f"{prefix}-{expected_sha256}.json":
        raise ValueError(f"{prefix} receipt has a noncanonical filename.")
    if sorted(receipt_dir.glob(f"{prefix}-*.json")) != [path]:
        raise ValueError(f"{prefix} canonical receipt set is malformed.")
    claim_path = receipt_dir / f".{prefix}.claimed.json"
    finalization_path = receipt_dir / f".{prefix}.finalized.json"
    paths = (
        (path, f"{prefix} receipt"),
        (claim_path, f"{prefix} receipt claim"),
        (finalization_path, f"{prefix} receipt finalization"),
    )
    leases = []
    try:
        for candidate, label in paths:
            _bootstrap_require_plain_directory_chain(candidate.parent)
            _bootstrap_require_plain_file(candidate, label)
            lease = _BootstrapReadOnlyFileLease(candidate, label)
            leases.append(lease)
            opened = os.fstat(lease.fileno())
            lexical = candidate.lstat()
            if (opened.st_dev, opened.st_ino) != (
                lexical.st_dev,
                lexical.st_ino,
            ):
                raise ValueError(f"{label} changed during lease acquisition.")

        receipt_bytes = leases[0].read_bytes()
        claim_bytes = leases[1].read_bytes()
        finalization_bytes = leases[2].read_bytes()
        if hashlib.sha256(receipt_bytes).hexdigest() != expected_sha256:
            raise ValueError(f"{prefix} receipt hash differs from its binding.")
        try:
            receipt = json.loads(receipt_bytes.decode("utf-8"))
            claim = json.loads(claim_bytes.decode("utf-8"))
            finalization = json.loads(finalization_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{prefix} receipt envelope is not valid JSON.") from error
        expected_claim = {
            "schema_version": 1,
            "experiment": _ENDER_EXPERIMENT_NAME,
            "prefix": prefix,
            "state": "CLAIMED",
        }
        if claim != expected_claim:
            raise ValueError(f"{prefix} receipt claim is malformed.")
        expected_finalization = {
            "schema_version": 1,
            "experiment": _ENDER_EXPERIMENT_NAME,
            "prefix": prefix,
            "state": "FINALIZED",
            "claim": {
                "path": claim_path.name,
                "sha256": hashlib.sha256(claim_bytes).hexdigest(),
            },
            "receipt": {
                "path": path.name,
                "sha256": expected_sha256,
            },
        }
        if finalization != expected_finalization:
            raise ValueError(f"{prefix} receipt finalization is malformed.")
        if not isinstance(receipt, dict):
            raise ValueError(f"{prefix} receipt payload is malformed.")
        if receipt.get("experiment") != _ENDER_EXPERIMENT_NAME:
            raise ValueError(f"{prefix} receipt experiment is malformed.")
        if receipt.get("stage") != stage:
            raise ValueError(f"{prefix} receipt stage is malformed.")
        envelope_keys = {
            "schema_version",
            "experiment",
            "stage",
            "state",
            "passed",
            "protocol",
        }
        stage_keys = {
            "claim-scout-component-run": {
                "component",
                "prior_finalized_seal",
                "destinations",
            },
            "claim-confirmation-component-run": {
                "component",
                "confirmation_pretraining_receipt",
                "prior_finalized_seal",
                "destinations",
            },
            "confirmation-pretraining": {
                "checkpoint",
                "scout_locked_receipt",
                "configs",
                "config_helpers",
                "loader",
                "store_inventory",
                "canonical_store",
                "input_layout",
                "output_destinations",
            },
        }
        expected_keys = envelope_keys | stage_keys.get(stage, set())
        if not stage_keys.get(stage) or set(receipt) != expected_keys:
            raise ValueError(f"{prefix} receipt schema is malformed.")
        if receipt.get("schema_version") != 1 or type(receipt.get("passed")) is not bool:
            raise ValueError(f"{prefix} receipt envelope is malformed.")
    except BaseException:
        for lease in reversed(leases):
            lease.close()
        raise
    return receipt, tuple(leases)


def _receipt_authority_directory() -> Path:
    return Path(
        os.path.abspath(
            REPO_DIR
            / "numerai/agents/experiments"
            / _ENDER_EXPERIMENT_NAME
            / "receipts"
        )
    )


def _preflight_scout_training_authority(
    component_name: str,
    pre_run_receipt_path: Path,
    pre_run_receipt_sha256: str,
) -> tuple[str, tuple]:
    if component_name not in _SCOUT_CONFIG_BY_COMPONENT:
        raise ValueError("Unknown Scout training component.")
    receipt, leases = _lease_finalized_receipt_envelope(
        pre_run_receipt_path,
        pre_run_receipt_sha256,
        receipt_dir=_receipt_authority_directory(),
        prefix=f"scout-pre-run-{component_name}",
        stage="claim-scout-component-run",
    )
    try:
        if (
            receipt.get("passed") is not True
            or receipt.get("state") != "ABSENCE_PROVEN"
            or receipt.get("component") != component_name
        ):
            raise ValueError("Scout pre-run authority is not passing.")
        protocol = receipt.get("protocol")
        checkpoint = (
            protocol.get("pretraining_commit")
            if isinstance(protocol, dict)
            else None
        )
        if not _is_lower_hex(checkpoint, 40):
            raise ValueError("Scout pre-run checkpoint is malformed.")
    except BaseException:
        for lease in reversed(leases):
            lease.close()
        raise
    return checkpoint, leases


def _preflight_confirmation_training_authority(
    component_name: str,
    pre_run_receipt_path: Path,
    pre_run_receipt_sha256: str,
    pretraining_receipt_path: Path,
    pretraining_receipt_sha256: str,
) -> tuple[str, tuple]:
    if component_name not in _CONFIRMATION_COMPONENTS:
        raise ValueError("Unknown confirmation training component.")
    receipt_dir = _receipt_authority_directory()
    pretraining, pretraining_leases = _lease_finalized_receipt_envelope(
        pretraining_receipt_path,
        pretraining_receipt_sha256,
        receipt_dir=receipt_dir,
        prefix="confirmation-pretraining",
        stage="confirmation-pretraining",
    )
    try:
        pre_run, pre_run_leases = _lease_finalized_receipt_envelope(
            pre_run_receipt_path,
            pre_run_receipt_sha256,
            receipt_dir=receipt_dir,
            prefix=f"confirmation-pre-run-{component_name}",
            stage="claim-confirmation-component-run",
        )
    except BaseException:
        for lease in reversed(pretraining_leases):
            lease.close()
        raise
    leases = (*pretraining_leases, *pre_run_leases)
    try:
        checkpoint = pretraining.get("checkpoint")
        if not _is_lower_hex(checkpoint, 40):
            raise ValueError("Confirmation pretraining checkpoint is malformed.")
        if (
            pre_run.get("passed") is not True
            or pre_run.get("state") != "ABSENCE_PROVEN"
            or pre_run.get("component") != component_name
        ):
            raise ValueError("Confirmation pre-run authority is not passing.")
        protocol = pre_run.get("protocol")
        if not isinstance(protocol, dict) or protocol.get(
            "pretraining_commit"
        ) != checkpoint:
            raise ValueError("Confirmation receipt checkpoints differ.")
        expected_pretraining = {
            "path": Path(os.path.abspath(pretraining_receipt_path))
            .relative_to(Path(os.path.abspath(REPO_DIR)))
            .as_posix(),
            "sha256": pretraining_receipt_sha256,
        }
        if pre_run.get("confirmation_pretraining_receipt") != expected_pretraining:
            raise ValueError(
                "Confirmation pre-run binds another pretraining receipt."
            )
    except BaseException:
        for lease in reversed(leases):
            lease.close()
        raise
    return checkpoint, tuple(leases)


def _verify_frozen_training_source(config_path: Path, expected_commit: str) -> tuple:
    """Reject drift before evaluating a confirmation config or opening its store."""

    if not _is_lower_hex(expected_commit, 40):
        raise ValueError("Frozen training commit must be a lowercase Git object ID.")
    candidate = Path(config_path).expanduser()
    if ".." in candidate.parts:
        raise ValueError("Frozen training config may not contain parent traversal.")
    absolute = Path(os.path.abspath(candidate))
    try:
        relative_config = absolute.relative_to(REPO_DIR).as_posix()
    except ValueError as error:
        raise ValueError("Frozen training config escapes the repository.") from error

    _bootstrap_require_plain_directory_chain(absolute.parent)
    _bootstrap_require_plain_file(absolute, "training config")
    exists = _run_git("cat-file", "-e", f"{expected_commit}^{{commit}}")
    if exists.returncode != 0:
        raise ValueError("Frozen training checkpoint is unavailable.")
    head = _run_git("rev-parse", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != expected_commit:
        raise ValueError("HEAD is not the frozen training checkpoint.")

    frozen_support_paths = [
        (
            "numerai/agents/code/analysis/"
            "evaluate_ender20_aux_target_rank_ensemble.py"
        ),
        (
            "numerai/agents/code/analysis/"
            "evaluate_ender20_hybrid_stability.py"
        ),
        (
            "numerai/agents/code/analysis/"
            "evaluate_xerxes20_lgbm_challenger.py"
        ),
        "numerai/agents/code/modeling",
        "numerai/agents/code/metrics/numerai_metrics.py",
        "numerai/agents/code/data/build_full_datasets.py",
        (
            "numerai/agents/experiments/"
            "xerxes20_lgbm_challenger_v53/gpu_runtime.json"
        ),
        (
            "numerai/agents/experiments/"
            "ender20_aux_target_rank_ensemble_v53/gate.md"
        ),
        (
            "numerai/agents/experiments/"
            "ender20_aux_target_rank_ensemble_v53/source_manifest.json"
        ),
        (
            "numerai/agents/experiments/"
            "ender20_aux_target_rank_ensemble_v53/configs"
        ),
    ]
    if absolute.name.startswith("confirmation_"):
        frozen_support_paths.append(
            "numerai/agents/experiments/"
            "ender20_aux_target_rank_ensemble_v53/"
            "confirmation_store_inventory.json"
        )
    guarded_paths = [
        relative_config,
        *frozen_support_paths,
    ]
    base_helper = absolute.with_name("base_d8.py")
    uses_frozen_base_helper = (
        absolute.name.startswith("confirmation_")
        or absolute.name in set(_SCOUT_CONFIG_BY_COMPONENT.values())
    )
    if uses_frozen_base_helper:
        _bootstrap_require_plain_file(base_helper, "training config helper")
        guarded_paths.append(base_helper.relative_to(REPO_DIR).as_posix())
    status = _run_git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *guarded_paths,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise ValueError("Frozen training sources are not clean at the checkpoint.")
    diff = _run_git("diff", "--quiet", expected_commit, "--", *guarded_paths)
    if diff.returncode != 0:
        raise ValueError("Frozen training sources differ from the checkpoint.")
    source_tree = _run_git(
        "ls-tree",
        "-r",
        "--name-only",
        expected_commit,
        "--",
        *frozen_support_paths,
    )
    if source_tree.returncode != 0:
        raise ValueError("Frozen executable source inventory cannot be read.")
    frozen_nonpython_paths = {
        path
        for path in frozen_support_paths
        if Path(path).suffix in {".json", ".md"}
    }
    executable_relatives = [
        value
        for value in source_tree.stdout.splitlines()
        if value.endswith(".py") or value in frozen_nonpython_paths
    ]
    if not executable_relatives:
        raise ValueError("Frozen executable source inventory is empty.")
    leased_paths = [(absolute, "training config")]
    if uses_frozen_base_helper:
        leased_paths.append((base_helper, "training config helper"))
    leased_paths.extend(
        (Path(os.path.abspath(REPO_DIR / relative)), "training source")
        for relative in executable_relatives
    )
    unique_leased_paths = []
    seen_paths = set()
    for path, label in leased_paths:
        key = str(path).casefold()
        if key not in seen_paths:
            seen_paths.add(key)
            unique_leased_paths.append((path, label))
    leases = []
    try:
        for path, label in unique_leased_paths:
            _bootstrap_require_plain_directory_chain(path.parent)
            _bootstrap_require_plain_file(path, label)
            lease = _BootstrapReadOnlyFileLease(path, label)
            leases.append(lease)
            relative = path.relative_to(REPO_DIR).as_posix()
            expected_blob = _run_git("rev-parse", f"{expected_commit}:{relative}")
            live_blob = _run_git("hash-object", f"--path={relative}", str(path))
            if (
                expected_blob.returncode != 0
                or live_blob.returncode != 0
                or expected_blob.stdout.strip() != live_blob.stdout.strip()
            ):
                raise ValueError(f"Frozen {label} differs from the checkpoint blob.")
    except BaseException:
        for lease in reversed(leases):
            lease.close()
        raise
    return tuple(leases)


@contextmanager
def _frozen_source_module_scope(source_leases: tuple):
    """Load lazy confirmation modules only from already verified leased source."""

    if not source_leases:
        yield
        return

    leases_by_path = {
        os.path.normcase(os.path.abspath(lease.path)): lease
        for lease in source_leases
    }
    module_sources = []
    for module_name, relative_path in _FROZEN_SOURCE_MODULES.items():
        source_path = Path(os.path.abspath(REPO_DIR / relative_path))
        lease = leases_by_path.get(os.path.normcase(str(source_path)))
        if lease is None:
            raise ValueError(
                f"Frozen source lease is missing for module {module_name}."
            )
        module_sources.append((module_name, source_path, lease))

    previous_modules = {}
    previous_parent_attributes = {}
    try:
        for module_name, source_path, lease in module_sources:
            parent_name, _, child_name = module_name.rpartition(".")
            parent = importlib.import_module(parent_name)
            previous_modules[module_name] = sys.modules.get(
                module_name, _MISSING_MODULE
            )
            previous_parent_attributes[module_name] = (
                parent,
                child_name,
                getattr(parent, child_name, _MISSING_MODULE),
            )

            source = lease.read_bytes()
            code = compile(
                source,
                str(source_path),
                "exec",
                dont_inherit=True,
                optimize=0,
            )
            module = ModuleType(module_name)
            module.__file__ = str(source_path)
            module.__package__ = parent_name
            module.__cached__ = None
            module.__loader__ = None
            module.__spec__ = importlib.util.spec_from_loader(
                module_name,
                loader=None,
                origin=str(source_path),
            )
            sys.modules[module_name] = module
            setattr(parent, child_name, module)
            exec(code, module.__dict__)
        yield
    finally:
        for module_name, _, _ in reversed(module_sources):
            parent, child_name, previous_attribute = previous_parent_attributes.get(
                module_name,
                (None, None, _MISSING_MODULE),
            )
            if parent is not None:
                if previous_attribute is _MISSING_MODULE:
                    try:
                        delattr(parent, child_name)
                    except AttributeError:
                        pass
                else:
                    setattr(parent, child_name, previous_attribute)
            previous_module = previous_modules.get(module_name, _MISSING_MODULE)
            if previous_module is _MISSING_MODULE:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module


def _lease_scout_training_inputs(protocol) -> tuple:
    """Lease and rehash the three manifest-bound Scout inputs through fitting."""

    from agents.code.analysis import (
        evaluate_ender20_aux_target_rank_ensemble as evaluator,
    )
    from .disk_feature_store import (
        _ReadOnlyFileLease,
        _require_plain_directory_chain,
        _require_plain_file,
    )

    feature_metadata = protocol.source_manifest.get("feature_metadata")
    scout_sources = protocol.source_manifest.get("scout_sources")
    if not isinstance(feature_metadata, dict) or not isinstance(scout_sources, dict):
        raise ValueError("Frozen Scout input receipts are malformed.")
    entries = [("feature metadata", feature_metadata)]
    entries.extend(
        (f"Scout source {relative}", {"path": relative, **receipt})
        for relative, receipt in scout_sources.items()
        if isinstance(receipt, dict)
    )
    if len(entries) != 3:
        raise ValueError("Frozen Scout input receipt count is malformed.")

    leases = []
    seen_paths = set()
    try:
        for label, receipt in entries:
            relative = receipt.get("path")
            expected_sha256 = receipt.get("sha256")
            expected_size = receipt.get("size_bytes")
            if (
                not isinstance(relative, str)
                or not _is_lower_hex(expected_sha256, 64)
                or type(expected_size) is not int
                or expected_size <= 0
            ):
                raise ValueError(f"Frozen {label} receipt is malformed.")
            path = evaluator._lexical_repo_path(protocol.repo_root, relative)
            path_key = os.path.normcase(str(path))
            if path_key in seen_paths:
                raise ValueError("Frozen Scout input paths are not distinct.")
            seen_paths.add(path_key)
            _require_plain_directory_chain(path.parent)
            _require_plain_file(path, label)
            lease = _ReadOnlyFileLease(path, label)
            leases.append(lease)
            if lease.size_bytes() != expected_size:
                raise ValueError(f"Frozen {label} size differs from its receipt.")
            if lease.sha256() != expected_sha256:
                raise ValueError(f"Frozen {label} SHA-256 differs from its receipt.")
    except BaseException:
        for lease in reversed(leases):
            lease.close()
        raise
    return tuple(leases)


def _derive_scout_training_authority(
    config_path: Path,
    *,
    component_name: str,
    pre_run_receipt_path: Path,
    pre_run_receipt_sha256: str,
) -> _TrainingAuthority:
    """Derive one Scout fit from its finalized just-in-time authorization."""

    from agents.code.analysis import (
        evaluate_ender20_aux_target_rank_ensemble as evaluator,
    )

    if component_name not in evaluator.SCOUT_NEW_COMPONENTS:
        raise ValueError("Unknown Scout training component.")
    receipt_dir = Path(
        os.path.abspath(
            REPO_DIR
            / "numerai/agents/experiments"
            / evaluator.EXPERIMENT_NAME
            / "receipts"
        )
    )
    evaluator._require_lexical_directory_chain(
        REPO_DIR,
        receipt_dir,
        "Scout training receipt directory",
    )
    pre_run_receipt_path = evaluator._lexical_absolute(
        Path(pre_run_receipt_path)
    )
    preliminary = evaluator._load_bound_receipt(
        pre_run_receipt_path,
        pre_run_receipt_sha256,
        expected_stage="claim-scout-component-run",
        receipt_dir=receipt_dir,
        expected_prefix=f"scout-pre-run-{component_name}",
    )
    evaluator._validate_stage_receipt_schema(
        preliminary, "claim-scout-component-run"
    )
    if (
        preliminary.get("passed") is not True
        or preliminary.get("state") != "ABSENCE_PROVEN"
        or preliminary.get("component") != component_name
    ):
        raise ValueError("Scout pre-run authority is not passing.")
    protocol_binding = preliminary.get("protocol")
    checkpoint = (
        protocol_binding.get("pretraining_commit")
        if isinstance(protocol_binding, dict)
        else None
    )
    if not _is_lower_hex(checkpoint, 40):
        raise ValueError("Scout pre-run checkpoint is malformed.")
    protocol = evaluator.verify_frozen_protocol(pretraining_commit=checkpoint)
    component = evaluator.default_scout_component_paths(protocol, component_name)
    if Path(os.path.abspath(config_path)) != component.config:
        raise ValueError("Training config does not match the authorized component.")

    data_leases = _lease_scout_training_inputs(protocol)
    try:
        evaluator._validate_scout_pre_run_absence_receipt(
            protocol,
            component,
            pre_run_receipt_path,
            pre_run_receipt_sha256,
        )
        for label, path in (
            ("result", component.result),
            ("predictions", component.predictions),
        ):
            evaluator._prepare_output_destination_parent(
                protocol,
                path,
                f"{component_name} Scout {label}",
                create_direct_parent=False,
            )
            evaluator._require_absent_destination(
                path,
                f"{component_name} Scout {label} before training",
            )
    except BaseException:
        for lease in reversed(data_leases):
            lease.close()
        raise
    return _TrainingAuthority(
        mode="scout",
        component_name=component_name,
        checkpoint=checkpoint,
        protocol=protocol,
        component=component,
        pre_run_receipt_path=pre_run_receipt_path,
        pre_run_receipt_sha256=pre_run_receipt_sha256,
        data_leases=data_leases,
    )


def _derive_confirmation_training_authority(
    config_path: Path,
    *,
    component_name: str,
    pre_run_receipt_path: Path,
    pre_run_receipt_sha256: str,
    pretraining_receipt_path: Path,
    pretraining_receipt_sha256: str,
) -> _TrainingAuthority:
    """Derive trainer authority from the finalized confirmation receipts."""

    from agents.code.analysis import (
        evaluate_ender20_aux_target_rank_ensemble as evaluator,
    )

    if component_name not in evaluator.ALL_COMPONENTS:
        raise ValueError("Unknown confirmation training component.")
    receipt_dir = Path(
        os.path.abspath(
            REPO_DIR
            / "numerai/agents/experiments"
            / evaluator.EXPERIMENT_NAME
            / "receipts"
        )
    )
    evaluator._require_lexical_directory_chain(
        REPO_DIR,
        receipt_dir,
        "confirmation training receipt directory",
    )
    preliminary = evaluator._load_bound_receipt(
        evaluator._lexical_absolute(Path(pretraining_receipt_path)),
        pretraining_receipt_sha256,
        expected_stage="confirmation-pretraining",
        receipt_dir=receipt_dir,
        expected_prefix="confirmation-pretraining",
    )
    evaluator._validate_stage_receipt_schema(
        preliminary, "confirmation-pretraining"
    )
    preliminary_pre_run = evaluator._load_bound_receipt(
        evaluator._lexical_absolute(Path(pre_run_receipt_path)),
        pre_run_receipt_sha256,
        expected_stage="claim-confirmation-component-run",
        receipt_dir=receipt_dir,
        expected_prefix=f"confirmation-pre-run-{component_name}",
    )
    evaluator._validate_stage_receipt_schema(
        preliminary_pre_run, "claim-confirmation-component-run"
    )
    if (
        preliminary_pre_run.get("passed") is not True
        or preliminary_pre_run.get("state") != "ABSENCE_PROVEN"
        or preliminary_pre_run.get("component") != component_name
    ):
        raise ValueError("Confirmation pre-run authority is not passing.")
    expected_pretraining_binding = {
        "path": evaluator._lexical_relative_path(
            evaluator._lexical_absolute(Path(pretraining_receipt_path)),
            REPO_DIR,
        ),
        "sha256": pretraining_receipt_sha256,
    }
    if (
        preliminary_pre_run.get("confirmation_pretraining_receipt")
        != expected_pretraining_binding
    ):
        raise ValueError("Confirmation pre-run binds another pretraining receipt.")
    checkpoint = preliminary.get("checkpoint")
    if not _is_lower_hex(checkpoint, 40):
        raise ValueError("Confirmation pretraining checkpoint is malformed.")
    protocol = evaluator.verify_frozen_protocol(
        pretraining_commit=checkpoint,
    )
    component = evaluator.default_confirmation_component_paths(
        protocol, component_name
    )
    if Path(os.path.abspath(config_path)) != component.config:
        raise ValueError("Training config does not match the authorized component.")
    pretraining = evaluator._validate_confirmation_pretraining_receipt(
        protocol,
        evaluator._lexical_absolute(Path(pretraining_receipt_path)),
        pretraining_receipt_sha256,
    )
    evaluator._validate_confirmation_pre_run_absence_receipt(
        protocol,
        component,
        evaluator._lexical_absolute(Path(pre_run_receipt_path)),
        pre_run_receipt_sha256,
        evaluator._lexical_absolute(Path(pretraining_receipt_path)),
        pretraining_receipt_sha256,
    )
    for label, path in (
        ("result", component.result),
        ("predictions", component.predictions),
    ):
        evaluator._prepare_output_destination_parent(
            protocol,
            path,
            f"{component_name} confirmation {label}",
            create_direct_parent=False,
        )
        evaluator._require_absent_destination(
            path,
            f"{component_name} confirmation {label} before training",
        )
    inventory = pretraining.get("store_inventory")
    if not isinstance(inventory, dict) or not _is_lower_hex(
        inventory.get("git_blob_id"), 40
    ):
        raise ValueError("Confirmation inventory authority is malformed.")
    return _TrainingAuthority(
        mode="confirmation",
        component_name=component_name,
        checkpoint=checkpoint,
        protocol=protocol,
        component=component,
        pre_run_receipt_path=evaluator._lexical_absolute(
            Path(pre_run_receipt_path)
        ),
        pre_run_receipt_sha256=pre_run_receipt_sha256,
        inventory_blob=inventory["git_blob_id"],
        confirmation_pretraining_receipt_path=evaluator._lexical_absolute(
            Path(pretraining_receipt_path)
        ),
        confirmation_pretraining_receipt_sha256=(
            pretraining_receipt_sha256
        ),
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
        return Path(
            os.path.abspath(NUMERAI_DIR / data_version / f"{target_col}_feature_store")
        )
    candidate = Path(path).expanduser()
    if ".." in candidate.parts:
        raise ValueError("Feature-store path may not contain parent traversal.")
    if candidate.is_absolute():
        return Path(os.path.abspath(candidate))
    if candidate.parts and candidate.parts[0] == NUMERAI_DIR.name:
        return Path(os.path.abspath(REPO_DIR / candidate))
    return Path(os.path.abspath(NUMERAI_DIR / candidate))


def _load_committed_feature_store_identity(
    inventory_value: str | Path,
    *,
    target_col: str,
    store_path: Path,
    expected_commit: str,
    expected_blob: str,
) -> tuple[dict, dict[str, str]]:
    """Load one externally committed store identity without trusting store metadata."""

    candidate = Path(inventory_value).expanduser()
    if ".." in candidate.parts:
        raise ValueError("Feature-store inventory path may not contain parent traversal.")
    if candidate.is_absolute():
        inventory_path = Path(os.path.abspath(candidate))
    elif candidate.parts and candidate.parts[0] == NUMERAI_DIR.name:
        inventory_path = Path(os.path.abspath(REPO_DIR / candidate))
    elif candidate.parts and candidate.parts[0] == BASE_DIR.name:
        inventory_path = Path(os.path.abspath(NUMERAI_DIR / candidate))
    else:
        inventory_path = Path(os.path.abspath(NUMERAI_DIR / candidate))
    try:
        relative = inventory_path.relative_to(REPO_DIR).as_posix()
    except ValueError as error:
        raise ValueError("Feature-store inventory escapes the repository.") from error
    from .disk_feature_store import _require_plain_directory_chain, _require_plain_file

    _require_plain_directory_chain(inventory_path.parent)
    _require_plain_file(inventory_path, "inventory")

    if not _is_lower_hex(expected_commit, 40):
        raise ValueError("Frozen inventory commit is malformed.")
    if not _is_lower_hex(expected_blob, 40):
        raise ValueError("Frozen inventory blob is malformed.")
    head = _run_git("rev-parse", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != expected_commit:
        raise ValueError("HEAD is not the frozen inventory checkpoint.")
    status = _run_git(
        "status", "--porcelain", "--untracked-files=all", "--", relative
    )
    if status.returncode != 0 or status.stdout.strip():
        raise ValueError("Feature-store inventory is not a clean committed file.")
    blob = _run_git("rev-parse", f"{expected_commit}:{relative}")
    blob_id = blob.stdout.strip()
    if (
        blob.returncode != 0
        or len(blob_id) != 40
        or not all(character in "0123456789abcdef" for character in blob_id)
    ):
        raise ValueError("Feature-store inventory is absent from the frozen checkpoint.")
    if blob_id != expected_blob:
        raise ValueError("Feature-store inventory blob differs from frozen authority.")
    live_blob = _run_git("hash-object", f"--path={relative}", str(inventory_path))
    if live_blob.returncode != 0 or live_blob.stdout.strip() != expected_blob:
        raise ValueError("Feature-store inventory differs from its committed Git blob.")
    blob_type = _run_git("cat-file", "-t", expected_blob)
    if blob_type.returncode != 0 or blob_type.stdout.strip() != "blob":
        raise ValueError("Frozen feature-store inventory object is not a Git blob.")
    frozen_blob = _run_git("cat-file", "blob", expected_blob, text=False)
    if frozen_blob.returncode != 0:
        raise ValueError("Frozen feature-store inventory blob cannot be read.")
    try:
        inventory = json.loads(frozen_blob.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Frozen feature-store inventory is not valid UTF-8 JSON.") from error
    if not isinstance(inventory, dict):
        raise ValueError("Feature-store inventory must be a JSON object.")
    layout = inventory.get("input_layout")
    stores = layout.get("stores") if isinstance(layout, dict) else None
    if not isinstance(stores, dict):
        raise ValueError("Feature-store inventory has no store layout.")
    matches = [
        receipt
        for receipt in stores.values()
        if isinstance(receipt, dict) and receipt.get("target_column") == target_col
    ]
    if len(matches) != 1:
        raise ValueError("Feature-store inventory target binding is ambiguous or missing.")
    receipt = matches[0]
    metadata = receipt.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("path"), str):
        raise ValueError("Feature-store inventory metadata receipt is malformed.")
    metadata_relative = Path(metadata["path"])
    if metadata_relative.is_absolute() or ".." in metadata_relative.parts:
        raise ValueError("Feature-store inventory metadata path is not repo-relative.")
    expected_directory = Path(os.path.abspath(REPO_DIR / metadata_relative)).parent
    if expected_directory != Path(os.path.abspath(store_path)):
        raise ValueError("Feature-store path differs from the committed inventory.")
    return receipt, {
        "path": relative,
        "git_blob_id": blob_id,
        "checkpoint_commit": expected_commit,
    }


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
    *,
    reserved_stream=None,
) -> tuple[Path, Path]:
    if reserved_stream is None:
        predictions_dir.mkdir(parents=True, exist_ok=True)
    elif not predictions_dir.is_dir():
        raise ValueError("Reserved prediction output directory is unavailable.")
    predictions_path = resolve_predictions_path(config, config_path, predictions_dir)
    table = pa.Table.from_pandas(predictions, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[PREDICTION_SEMANTICS_METADATA_KEY] = _canonical_json(
        prediction_semantics
    ).encode("utf-8")
    table = table.replace_schema_metadata(metadata)
    if reserved_stream is None:
        pq.write_table(table, predictions_path)
    else:
        if os.fstat(reserved_stream.fileno()).st_size != 0:
            raise ValueError("Reserved prediction output is no longer empty.")
        reserved_stream.seek(0)
        pq.write_table(table, reserved_stream)
        reserved_stream.flush()
        os.fsync(reserved_stream.fileno())
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
    era_allowlist_receipt: dict | None = None,
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
    if era_allowlist_receipt is not None:
        results["data"]["era_allowlist"] = era_allowlist_receipt
    return results


def save_results(results: dict, results_path: Path, *, reserved_stream=None) -> None:
    if reserved_stream is None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)
    else:
        if not results_path.parent.is_dir():
            raise ValueError("Reserved result output directory is unavailable.")
        if os.fstat(reserved_stream.fileno()).st_size != 0:
            raise ValueError("Reserved result output is no longer empty.")
        payload = json.dumps(results, indent=2, sort_keys=True).encode("utf-8")
        reserved_stream.seek(0)
        reserved_stream.write(payload)
        reserved_stream.flush()
        os.fsync(reserved_stream.fileno())
    print(f"Saved results to {results_path}")


def run_training(
    config_path: Path,
    output_dir_override: Path | None = None,
    *,
    scout_component: str | None = None,
    scout_pre_run_receipt: Path | None = None,
    scout_pre_run_receipt_sha256: str | None = None,
    confirmation_component: str | None = None,
    confirmation_pre_run_receipt: Path | None = None,
    confirmation_pre_run_receipt_sha256: str | None = None,
    confirmation_pretraining_receipt: Path | None = None,
    confirmation_pretraining_receipt_sha256: str | None = None,
) -> tuple[Path, Path]:
    scout_authority_values = (
        scout_component,
        scout_pre_run_receipt,
        scout_pre_run_receipt_sha256,
    )
    confirmation_authority_values = (
        confirmation_component,
        confirmation_pre_run_receipt,
        confirmation_pre_run_receipt_sha256,
        confirmation_pretraining_receipt,
        confirmation_pretraining_receipt_sha256,
    )
    if any(value is not None for value in scout_authority_values) and not all(
        value is not None for value in scout_authority_values
    ):
        raise ValueError(
            "All Scout training authority arguments must be supplied together."
        )
    if any(
        value is not None for value in confirmation_authority_values
    ) and not all(
        value is not None for value in confirmation_authority_values
    ):
        raise ValueError(
            "All confirmation training authority arguments must be supplied together."
        )
    has_scout_authority = all(
        value is not None for value in scout_authority_values
    )
    has_confirmation_authority = all(
        value is not None for value in confirmation_authority_values
    )
    if has_scout_authority and has_confirmation_authority:
        raise ValueError("Scout and confirmation authority are mutually exclusive.")
    has_receipt_authority = has_scout_authority or has_confirmation_authority
    if has_receipt_authority and output_dir_override is not None:
        raise ValueError(
            "Receipt-authorized training may not override its bound output directory."
        )
    if not has_confirmation_authority and Path(config_path).name.startswith(
        "confirmation_"
    ):
        raise ValueError(
            "A confirmation config requires its finalized pre-run and "
            "pretraining receipt authority before config evaluation."
        )
    if (
        not has_scout_authority
        and Path(config_path).name in set(_SCOUT_CONFIG_BY_COMPONENT.values())
    ):
        raise ValueError(
            "An Ender20 Scout config requires its finalized pre-run receipt "
            "authority before config evaluation."
        )
    ender21_reservations = _ender21_output_reservations(
        Path(config_path), output_dir_override
    )
    authority = None
    authority_checkpoint = None
    receipt_leases = ()
    if has_receipt_authority:
        _require_frozen_python_runtime()
    if has_scout_authority:
        assert scout_component is not None
        assert scout_pre_run_receipt is not None
        assert scout_pre_run_receipt_sha256 is not None
        authority_checkpoint, receipt_leases = (
            _preflight_scout_training_authority(
                scout_component,
                scout_pre_run_receipt,
                scout_pre_run_receipt_sha256,
            )
        )
    if has_confirmation_authority:
        assert confirmation_component is not None
        assert confirmation_pre_run_receipt is not None
        assert confirmation_pre_run_receipt_sha256 is not None
        assert confirmation_pretraining_receipt is not None
        assert confirmation_pretraining_receipt_sha256 is not None
        authority_checkpoint, receipt_leases = (
            _preflight_confirmation_training_authority(
                confirmation_component,
                confirmation_pre_run_receipt,
                confirmation_pre_run_receipt_sha256,
                confirmation_pretraining_receipt,
                confirmation_pretraining_receipt_sha256,
            )
        )
    source_leases = ()
    try:
        if authority_checkpoint is not None:
            source_leases = _verify_frozen_training_source(
                Path(config_path), authority_checkpoint
            )
        if has_scout_authority:
            assert scout_component is not None
            assert scout_pre_run_receipt is not None
            assert scout_pre_run_receipt_sha256 is not None
            authority = _derive_scout_training_authority(
                Path(config_path),
                component_name=scout_component,
                pre_run_receipt_path=scout_pre_run_receipt,
                pre_run_receipt_sha256=scout_pre_run_receipt_sha256,
            )
        if has_confirmation_authority:
            assert confirmation_component is not None
            assert confirmation_pre_run_receipt is not None
            assert confirmation_pre_run_receipt_sha256 is not None
            assert confirmation_pretraining_receipt is not None
            assert confirmation_pretraining_receipt_sha256 is not None
            authority = _derive_confirmation_training_authority(
                Path(config_path),
                component_name=confirmation_component,
                pre_run_receipt_path=confirmation_pre_run_receipt,
                pre_run_receipt_sha256=confirmation_pre_run_receipt_sha256,
                pretraining_receipt_path=confirmation_pretraining_receipt,
                pretraining_receipt_sha256=(
                    confirmation_pretraining_receipt_sha256
                ),
            )
        if authority is not None and authority.checkpoint != authority_checkpoint:
            raise ValueError("Training authority checkpoint changed after leasing.")
        reservations = ender21_reservations
        if authority is not None:
            reservations = _ExclusiveOutputReservations(
                authority.component.predictions,
                authority.component.result,
            )
        reservation_scope = (
            reservations if reservations is not None else nullcontext(None)
        )
        with reservation_scope as reserved_outputs:
            if ender21_reservations is not None:
                _verify_ender21_round1_manifest()
            marker_lease = None
            completion_claim_lease = None
            completion_receipt_lease = None
            try:
                if authority is not None:
                    from agents.code.analysis import (
                        evaluate_ender20_aux_target_rank_ensemble as evaluator,
                    )
                    from .disk_feature_store import _ReadOnlyFileLease

                    claim_path, claim_payload = (
                        evaluator.claim_component_training_consumption(
                            authority.protocol,
                            authority.component,
                            authority.pre_run_receipt_path,
                            authority.pre_run_receipt_sha256,
                            reserved_outputs.identities(),
                            confirmation=authority.mode == "confirmation",
                            confirmation_pretraining_receipt_path=(
                                authority.confirmation_pretraining_receipt_path
                            ),
                            confirmation_pretraining_receipt_sha256=(
                                authority.confirmation_pretraining_receipt_sha256
                            ),
                        )
                    )
                    marker_lease = _ReadOnlyFileLease(
                        claim_path,
                        f"{authority.component_name} training consumption claim",
                    )
                    if marker_lease.read_bytes() != evaluator._receipt_bytes(
                        claim_payload
                    ):
                        raise ValueError(
                            "Training consumption claim changed before its lease."
                        )
                    completion_claim_path = (
                        evaluator.claim_component_training_completion(
                            authority.protocol,
                            authority.component,
                            confirmation=authority.mode == "confirmation",
                        )
                    )
                    completion_prefix = (
                        evaluator._component_training_completion_prefix(
                            authority.component_name,
                            confirmation=authority.mode == "confirmation",
                        )
                    )
                    completion_claim_lease = _ReadOnlyFileLease(
                        completion_claim_path,
                        f"{authority.component_name} training completion claim",
                    )
                    if completion_claim_lease.read_bytes() != evaluator._receipt_bytes(
                        evaluator._claim_payload(completion_prefix)
                    ):
                        raise ValueError(
                            "Training completion claim changed before fitting."
                        )
                with _frozen_source_module_scope(source_leases):
                    training_outputs = _run_training_impl(
                        config_path,
                        output_dir_override,
                        frozen_git_commit=(
                            authority.checkpoint if authority is not None else None
                        ),
                        frozen_inventory_blob=(
                            authority.inventory_blob if authority is not None else None
                        ),
                        reserved_outputs=reserved_outputs,
                    )
                    if authority is not None:
                        completion_path, completion_payload = (
                            evaluator.complete_component_training_consumption(
                                authority.protocol,
                                authority.component,
                                authority.pre_run_receipt_path,
                                authority.pre_run_receipt_sha256,
                                reserved_outputs.completion_identities(),
                                completion_claim_path,
                                confirmation=authority.mode == "confirmation",
                                confirmation_pretraining_receipt_path=(
                                    authority.confirmation_pretraining_receipt_path
                                ),
                                confirmation_pretraining_receipt_sha256=(
                                    authority.confirmation_pretraining_receipt_sha256
                                ),
                            )
                        )
                        completion_receipt_lease = _ReadOnlyFileLease(
                            completion_path,
                            f"{authority.component_name} training completion receipt",
                        )
                        completion_bytes = evaluator._receipt_bytes(
                            completion_payload
                        )
                        if completion_receipt_lease.read_bytes() != completion_bytes:
                            raise ValueError(
                                "Training completion receipt changed before return."
                            )
                        print(
                            json.dumps(
                                {
                                    "training_completion_receipt": str(
                                        completion_path
                                    ),
                                    "training_completion_receipt_sha256": (
                                        hashlib.sha256(completion_bytes).hexdigest()
                                    ),
                                },
                                sort_keys=True,
                            )
                        )
                    return training_outputs
            finally:
                if completion_receipt_lease is not None:
                    completion_receipt_lease.close()
                if completion_claim_lease is not None:
                    completion_claim_lease.close()
                if marker_lease is not None:
                    marker_lease.close()
    finally:
        for lease in reversed(source_leases):
            lease.close()
        if authority is not None:
            for lease in reversed(authority.data_leases):
                lease.close()
        for lease in reversed(receipt_leases):
            lease.close()


def _run_training_impl(
    config_path: Path,
    output_dir_override: Path | None,
    *,
    frozen_git_commit: str | None,
    frozen_inventory_blob: str | None,
    reserved_outputs: _ExclusiveOutputReservations | None = None,
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
    allowed_eras, era_allowlist_receipt = _load_era_allowlist(
        data_config.get("era_allowlist_path")
    )
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

    raw_max_train_samples = training_config.get("max_train_samples")
    max_train_samples = raw_max_train_samples
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)
    sample_seed = int(training_config.get("sample_seed", 1337))
    data_mode = str(training_config.get("data_mode", "eager")).lower()
    if data_mode not in {"eager", "disk_feature_store"}:
        raise ValueError(
            "training.data_mode must be 'eager' or 'disk_feature_store'."
        )
    inventory_value = data_config.get("disk_feature_store_inventory_path")
    if inventory_value is not None and (
        frozen_git_commit is None or frozen_inventory_blob is None
    ):
        raise ValueError(
            "A committed feature-store inventory requires its frozen training "
            "commit and Git blob before any data access."
        )
    if inventory_value is None and frozen_inventory_blob is not None:
        raise ValueError(
            "Frozen inventory authority was supplied for a config without an inventory."
        )

    output_dir, baselines_dir, results_dir, predictions_dir = resolve_output_locations(
        config, output_dir_override
    )
    configured_predictions = Path(
        os.path.abspath(resolve_predictions_path(config, config_path, predictions_dir))
    )
    configured_results = Path(
        os.path.abspath(resolve_results_path(config, config_path, results_dir))
    )
    if reserved_outputs is not None:
        if configured_predictions != reserved_outputs.predictions_path:
            raise ValueError(
                "Configured prediction output differs from its consumed authorization."
            )
        if configured_results != reserved_outputs.results_path:
            raise ValueError(
                "Configured result output differs from its consumed authorization."
            )
    else:
        _require_safe_unreserved_output(
            configured_predictions, "prediction"
        )
        _require_safe_unreserved_output(configured_results, "result")
        if {
            configured_predictions,
            configured_results,
        } & _governed_output_paths():
            raise ValueError(
                "Governed experiment outputs require an authorized exclusive run."
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
        disk_model_types = {"LGBMRegressor", "TorchTabularRegressor"}
        if model_type not in disk_model_types:
            raise ValueError(
                "training.data_mode='disk_feature_store' currently requires "
                "model.type to be 'LGBMRegressor' or 'TorchTabularRegressor'."
            )
        if model_type == "LGBMRegressor" and (
            isinstance(raw_max_train_samples, bool)
            or not isinstance(raw_max_train_samples, int)
            or max_train_samples <= 0
        ):
            raise ValueError(
                "Disk LGBMRegressor training requires training.max_train_samples "
                "to be an explicit positive integer materialization cap."
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

        expected_store_receipt = None
        expected_inventory_identity = None
        if inventory_value is not None:
            assert frozen_git_commit is not None
            assert frozen_inventory_blob is not None
            expected_store_receipt, expected_inventory_identity = (
                _load_committed_feature_store_identity(
                    inventory_value,
                    target_col=target_col,
                    store_path=store_path,
                    expected_commit=frozen_git_commit,
                    expected_blob=frozen_inventory_blob,
                )
            )
        configured_features = load_features(napi, data_version, feature_set)
        data_loader = DiskFeatureStoreLoader(
            store_path,
            era_col=era_col,
            target_col=target_col,
            id_col=id_col,
            benchmark_col=benchmark_model,
            expected_store_receipt=expected_store_receipt,
            expected_receipt_root=(
                REPO_DIR if expected_store_receipt is not None else None
            ),
            expected_inventory_identity=expected_inventory_identity,
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

    full = _filter_to_era_allowlist(full, era_col, allowed_eras)
    if data_mode == "disk_feature_store":
        scoring_benchmark_data = full

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
        reserved_stream=(
            reserved_outputs.predictions_stream
            if reserved_outputs is not None
            else None
        ),
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
        era_allowlist_receipt=era_allowlist_receipt,
    )
    save_results(
        results,
        results_path,
        reserved_stream=(
            reserved_outputs.results_stream
            if reserved_outputs is not None
            else None
        ),
    )

    return predictions_path, results_path
