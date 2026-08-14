"""Stdlib-only custody boundary for one exact Ender27 Round-1 process."""

from __future__ import annotations

import argparse
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


REPO_DIR = Path(__file__).resolve().parents[4]
PREFIX = "numerai/agents/experiments/ender27_tempered_gaussian_rank_residual_v53"
ROUND1_SEQUENCE = (
    "r1_control_rawresid_seed1337",
    "r1_tempered_grank_resid_seed1337",
    "r1_control_rawresid_seed2027",
    "r1_tempered_grank_resid_seed2027",
)
ROUND1_NAMES = frozenset(ROUND1_SEQUENCE)
ALL_CONFIG_NAMES = ROUND1_NAMES
EXTERNALS = frozenset(
    {
        "numerai/v5.3/ender21_discovery_full_through_0861.parquet",
        "numerai/v5.3/ender21_discovery_benchmark_models_through_0861.parquet",
    }
)
BOOTSTRAP_SOURCES = frozenset(
    {
        "numerai/agents/code/modeling/__main__.py",
        "numerai/agents/code/modeling/models/torch_tabular_regressor.py",
        "numerai/agents/code/modeling/utils/cli.py",
        "numerai/agents/code/modeling/utils/config.py",
        "numerai/agents/code/modeling/utils/constants.py",
        "numerai/agents/code/modeling/utils/data.py",
        "numerai/agents/code/modeling/utils/model_data.py",
        "numerai/agents/code/modeling/utils/model_factory.py",
        "numerai/agents/code/modeling/utils/numerai_cv.py",
        "numerai/agents/code/modeling/utils/pipeline.py",
        "numerai/agents/code/modeling/utils/target_transforms.py",
        "numerai/agents/code/metrics/numerai_metrics.py",
        f"{PREFIX}/configs/base_r1.py",
        f"{PREFIX}/training_bootstrap.py",
        f"{PREFIX}/run_round1.py",
        f"{PREFIX}/evaluation_common.py",
        f"{PREFIX}/evaluate_round1.py",
        f"{PREFIX}/evaluate_round1_impl.py",
    }
)
STATIC_MANIFEST_FILES = frozenset(
    {
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "protocol/discovery_eras_through_0861.json",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "protocol/feature_columns_all_v53.json",
        f"{PREFIX}/experiment.md",
        f"{PREFIX}/gate.md",
        f"{PREFIX}/protocol/discovery_data_authority.json",
        "numerai/agents/tests/test_target_transforms.py",
        "numerai/agents/tests/test_ender27_tempered_gaussian_rank_residual.py",
        "numerai/agents/tests/test_ender27_pipeline_custody.py",
        "numerai/agents/tests/test_ender27_evaluator.py",
    }
)
EXPECTED_RUNTIME = {
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
        "scipy": "1.18.0",
        "scikit-learn": "1.9.0",
    },
}


def _expected_manifest_files(round_number: int) -> frozenset[str]:
    if round_number != 1:
        raise ValueError("Ender27 bootstrap supports Round 1 only.")
    files = {
        *BOOTSTRAP_SOURCES,
        *STATIC_MANIFEST_FILES,
        *(f"{PREFIX}/configs/{name}.py" for name in ALL_CONFIG_NAMES),
    }
    expected = frozenset(files)
    if len(expected) != 31:
        raise AssertionError("Ender27 Round-1 manifest set is not exactly 31 paths.")
    return expected


def _plain_directory_chain(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    for directory in reversed([absolute, *absolute.parents]):
        if directory == directory.parent or not os.path.lexists(directory):
            continue
        inspected = directory.lstat()
        attributes = getattr(inspected, "st_file_attributes", 0)
        if directory.is_symlink() or bool(
            attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise ValueError(
                f"Ender27 bootstrap directory is a reparse point: {directory}"
            )
        if not stat.S_ISDIR(inspected.st_mode):
            raise ValueError(
                f"Ender27 bootstrap directory is not plain: {directory}"
            )


def _plain_file(path: Path, label: str) -> None:
    inspected = path.lstat()
    attributes = getattr(inspected, "st_file_attributes", 0)
    if (
        path.is_symlink()
        or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
        or not stat.S_ISREG(inspected.st_mode)
        or inspected.st_nlink != 1
    ):
        raise ValueError(
            f"Ender27 bootstrap {label} is not a unique plain file: {path}"
        )


class _Lease:
    def __init__(self, path: Path, label: str) -> None:
        self.path = Path(path)
        self.label = label
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
                str(path), 0x80000000, 0x00000001, None, 3, 0x80, None
            )
            if handle == ctypes.c_void_p(-1).value:
                raise ValueError(
                    f"Cannot lease Ender27 {label}: {path}"
                ) from ctypes.WinError(ctypes.get_last_error())
            descriptor = msvcrt.open_osfhandle(
                int(handle), os.O_RDONLY | getattr(os, "O_BINARY", 0)
            )
            self.stream = os.fdopen(descriptor, "rb", buffering=0)
        else:
            self.stream = path.open("rb", buffering=0)
        try:
            opened = os.fstat(self.stream.fileno())
            lexical = path.lstat()
            attributes = getattr(lexical, "st_file_attributes", 0)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or path.is_symlink()
                or bool(
                    attributes
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                )
                or not stat.S_ISREG(lexical.st_mode)
                or lexical.st_nlink != 1
                or (int(opened.st_dev), int(opened.st_ino))
                != (int(lexical.st_dev), int(lexical.st_ino))
            ):
                raise ValueError(
                    f"Leased Ender27 {label} identity is malformed or changed."
                )
        except BaseException:
            self.close()
            raise

    def read_bytes(self) -> bytes:
        self.stream.seek(0)
        value = self.stream.read()
        self.stream.seek(0)
        return value

    def sha256(self) -> str:
        digest = hashlib.sha256()
        self.stream.seek(0)
        while chunk := self.stream.read(8 * 1024 * 1024):
            digest.update(chunk)
        self.stream.seek(0)
        return digest.hexdigest()

    def size_bytes(self) -> int:
        return int(os.fstat(self.stream.fileno()).st_size)

    def close(self) -> None:
        self.stream.close()


class _DecisionReservation:
    """CREATE_NEW evaluation decision reserved before manifest access."""

    def __init__(self, path: Path) -> None:
        self.path = Path(os.path.abspath(path))
        self.stream = None
        self.identity = None
        self.committed = False

    def __enter__(self):
        _plain_directory_chain(self.path.parent)
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
                str(self.path), 0xC0000000, 0x00000001, None, 1, 0x80, None
            )
            if handle == ctypes.c_void_p(-1).value:
                raise ValueError(
                    f"Cannot reserve Ender27 evaluation decision: {self.path}"
                ) from ctypes.WinError(ctypes.get_last_error())
            descriptor = msvcrt.open_osfhandle(
                int(handle), os.O_RDWR | getattr(os, "O_BINARY", 0)
            )
        else:
            descriptor = os.open(
                self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600
            )
        self.stream = os.fdopen(descriptor, "w+b", buffering=0)
        opened = os.fstat(self.stream.fileno())
        lexical = self.path.lstat()
        self.identity = (int(opened.st_dev), int(opened.st_ino))
        if (
            opened.st_nlink != 1
            or opened.st_size != 0
            or self.identity != (int(lexical.st_dev), int(lexical.st_ino))
        ):
            self.__exit__(None, None, None)
            raise ValueError("Reserved Ender27 evaluation decision is malformed.")
        return self

    def commit_json(self, payload: dict) -> bytes:
        if self.stream is None or self.committed:
            raise RuntimeError("Evaluation decision is not writable.")
        if os.fstat(self.stream.fileno()).st_size != 0:
            raise ValueError("Reserved evaluation decision is no longer empty.")
        payload_bytes = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode(
                "utf-8"
            )
            + b"\n"
        )
        self.stream.write(payload_bytes)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        opened = os.fstat(self.stream.fileno())
        lexical = self.path.lstat()
        if self.identity != (
            int(opened.st_dev),
            int(opened.st_ino),
        ) or self.identity != (int(lexical.st_dev), int(lexical.st_ino)):
            raise ValueError("Evaluation decision path identity changed before fsync.")
        self.committed = True
        return payload_bytes

    def __exit__(self, *_args) -> None:
        stream = self.stream
        identity = self.identity
        self.stream = None
        if stream is not None:
            stream.close()
        if self.committed or identity is None:
            return
        if not os.path.lexists(self.path):
            raise ValueError("Uncommitted evaluation decision path disappeared.")
        lexical = self.path.lstat()
        attributes = getattr(lexical, "st_file_attributes", 0)
        if (
            not stat.S_ISREG(lexical.st_mode)
            or lexical.st_nlink != 1
            or self.path.is_symlink()
            or bool(
                attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
            or (int(lexical.st_dev), int(lexical.st_ino)) != identity
        ):
            raise ValueError("Refusing to remove changed evaluation decision.")
        # Any uncommitted invocation is terminal evidence. Preserve even an
        # untouched zero-byte reservation so CREATE_NEW rejects every retry.


class _TrainingOutputReservations:
    """CREATE_NEW training outputs held before any manifest or data read."""

    def __init__(self, config_name: str) -> None:
        experiment = Path(os.path.abspath(REPO_DIR / PREFIX))
        _plain_directory_chain(experiment)
        paths = {
            "predictions": experiment / "predictions",
            "results": experiment / "results",
            "receipts": experiment / "receipts",
        }
        for directory in paths.values():
            directory.mkdir(exist_ok=True)
            _plain_directory_chain(directory)
        self.predictions_path = paths["predictions"] / f"{config_name}.parquet"
        self.results_path = paths["results"] / f"{config_name}.json"
        self.completion_path = paths["receipts"] / f"{config_name}.completion.json"
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
                str(path), 0xC0000000, 0x00000001, None, 1, 0x80, None
            )
            if handle == ctypes.c_void_p(-1).value:
                raise ValueError(
                    f"Cannot reserve Ender27 {label}: {path}"
                ) from ctypes.WinError(ctypes.get_last_error())
            descriptor = msvcrt.open_osfhandle(
                int(handle), os.O_RDWR | getattr(os, "O_BINARY", 0)
            )
        else:
            descriptor = os.open(
                path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600
            )
        return os.fdopen(descriptor, "w+b", buffering=0)

    def __enter__(self):
        opened = []
        try:
            for attribute, path, label in (
                ("predictions_stream", self.predictions_path, "prediction"),
                ("results_stream", self.results_path, "result"),
                ("completion_stream", self.completion_path, "completion"),
            ):
                stream = self._open(path, label)
                setattr(self, attribute, stream)
                opened.append(stream)
            self.identities()
            return self
        except BaseException:
            for stream in reversed(opened):
                stream.close()
            raise

    def __exit__(self, *_args) -> None:
        for attribute in (
            "completion_stream",
            "results_stream",
            "predictions_stream",
        ):
            stream = getattr(self, attribute)
            setattr(self, attribute, None)
            if stream is not None:
                stream.close()

    def _identity(self, label: str, path: Path, stream) -> dict:
        opened = os.fstat(stream.fileno())
        lexical = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (int(opened.st_dev), int(opened.st_ino))
            != (int(lexical.st_dev), int(lexical.st_ino))
        ):
            raise ValueError(f"Reserved Ender27 {label} identity changed.")
        return {
            "path": str(path),
            "device": int(opened.st_dev),
            "inode": int(opened.st_ino),
        }

    def identities(self) -> dict[str, dict[str, object]]:
        return {
            "predictions": self._identity(
                "prediction", self.predictions_path, self.predictions_stream
            ),
            "result": self._identity(
                "result", self.results_path, self.results_stream
            ),
        }

    def completion_identities(self) -> dict[str, dict[str, object]]:
        values = {}
        for label, path, stream in (
            ("predictions", self.predictions_path, self.predictions_stream),
            ("result", self.results_path, self.results_stream),
        ):
            stream.flush()
            os.fsync(stream.fileno())
            identity = self._identity(label, path, stream)
            inspected = os.fstat(stream.fileno())
            if inspected.st_size <= 0:
                raise ValueError(f"Completed Ender27 {label} is empty.")
            digest = hashlib.sha256()
            stream.seek(0)
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
            stream.seek(0)
            values[label] = {
                **identity,
                "size_bytes": int(inspected.st_size),
                "sha256": digest.hexdigest(),
            }
        return values

    def write_completion(self, payload: dict) -> bytes:
        if os.fstat(self.completion_stream.fileno()).st_size != 0:
            raise ValueError("Reserved Ender27 completion is no longer empty.")
        payload_bytes = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode(
                "utf-8"
            )
            + b"\n"
        )
        self.completion_stream.write(payload_bytes)
        self.completion_stream.flush()
        os.fsync(self.completion_stream.fileno())
        self._identity("completion", self.completion_path, self.completion_stream)
        return payload_bytes


def _require_launch() -> Path:
    preloaded = sorted(
        name for name in sys.modules if name == "agents" or name.startswith("agents.")
    )
    if preloaded:
        raise ValueError(f"Ender27 governed modules were pre-imported: {preloaded}")
    option = getattr(sys, "_xoptions", {}).get("pycache_prefix")
    if (
        sys.flags.dont_write_bytecode != 1
        or sys.dont_write_bytecode is not True
        or sys.flags.isolated != 1
        or sys.flags.safe_path != 1
    ):
        raise ValueError("Ender27 bootstrap requires isolated Python -I -B -P.")
    if type(option) is not str or not option or sys.pycache_prefix != option:
        raise ValueError("Ender27 bootstrap requires exact -X pycache_prefix.")
    prefix = Path(option)
    if not prefix.is_absolute() or prefix != Path(os.path.abspath(prefix)):
        raise ValueError("Ender27 pycache prefix must be absolute and canonical.")
    try:
        prefix.relative_to(REPO_DIR)
    except ValueError:
        pass
    else:
        raise ValueError("Ender27 pycache prefix must be outside the repository.")
    _plain_directory_chain(prefix)
    if next(prefix.iterdir(), None) is not None:
        raise ValueError("Ender27 pycache prefix must be freshly empty.")
    return prefix


def _git(*args: str, allow_one: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in ({0, 1} if allow_one else {0}):
        raise ValueError(
            result.stderr.strip() or "Ender27 bootstrap Git verification failed."
        )
    return result


def _acquire(round_number: int) -> tuple[dict, bytes, tuple[_Lease, ...]]:
    if round_number != 1:
        raise ValueError("Ender27 bootstrap supports Round 1 only.")
    filename = "source_manifest_round1.json"
    relative = f"{PREFIX}/{filename}"
    path = Path(os.path.abspath(REPO_DIR / relative))
    _plain_directory_chain(path.parent)
    _plain_file(path, "manifest")
    leases: list[_Lease] = []
    try:
        manifest_lease = _Lease(path, "manifest")
        leases.append(manifest_lease)
        manifest_bytes = manifest_lease.read_bytes()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "frozen_at",
            "git_head",
            "hash_algorithm",
            "files",
            "external_artifacts",
            "runtime",
        }:
            raise ValueError("Ender27 bootstrap manifest schema differs.")
        if manifest["schema_version"] != 1 or manifest["hash_algorithm"] != "sha256":
            raise ValueError("Ender27 bootstrap manifest version differs.")
        files = manifest["files"]
        external = manifest["external_artifacts"]
        required = _expected_manifest_files(round_number)
        if not isinstance(files, dict) or set(files) != required:
            raise ValueError("Ender27 bootstrap manifest file set differs.")
        if not isinstance(external, dict) or set(external) != EXTERNALS:
            raise ValueError("Ender27 bootstrap external set differs.")
        if (
            manifest["runtime"] != EXPECTED_RUNTIME
            or platform.python_version() != EXPECTED_RUNTIME["python"]
        ):
            raise ValueError("Ender27 bootstrap runtime differs.")
        for package, version in EXPECTED_RUNTIME["packages"].items():
            if importlib_metadata.version(package) != version:
                raise ValueError(f"Ender27 bootstrap package drifted: {package}")
        commit = manifest["git_head"]
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise ValueError("Ender27 bootstrap git_head is malformed.")
        if (
            _git("rev-parse", "--verify", f"{commit}^{{commit}}").stdout.strip()
            != commit
        ):
            raise ValueError("Ender27 bootstrap git_head does not resolve.")
        if _git(
            "merge-base", "--is-ancestor", commit, "HEAD", allow_one=True
        ).returncode:
            raise ValueError("Ender27 bootstrap git_head is not an ancestor of HEAD.")
        _git("cat-file", "-e", f"HEAD:{relative}")
        governed = [relative, *sorted(files)]
        if _git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *governed,
        ).stdout:
            raise ValueError("Ender27 bootstrap source is not committed and clean.")
        for relative_text, expected in sorted(files.items()):
            if (
                not isinstance(expected, str)
                or len(expected) != 64
                or any(
                    character not in "0123456789abcdef" for character in expected
                )
            ):
                raise ValueError(
                    f"Ender27 bootstrap source digest is malformed: {relative_text}"
                )
            source = Path(os.path.abspath(REPO_DIR / relative_text))
            _plain_directory_chain(source.parent)
            _plain_file(source, f"source {relative_text}")
            lease = _Lease(source, f"source {relative_text}")
            leases.append(lease)
            if lease.sha256() != expected:
                raise ValueError(f"Ender27 bootstrap source drifted: {relative_text}")
            if _git(
                "diff", "--quiet", commit, "--", relative_text, allow_one=True
            ).returncode:
                raise ValueError(
                    f"Ender27 bootstrap source differs from git_head: {relative_text}"
                )
        for relative_text, artifact_receipt in sorted(external.items()):
            if (
                not isinstance(artifact_receipt, dict)
                or set(artifact_receipt) != {"size_bytes", "sha256", "last_era"}
                or artifact_receipt.get("last_era") != "0861"
            ):
                raise ValueError(
                    f"Ender27 bootstrap external receipt differs: {relative_text}"
                )
            artifact = Path(os.path.abspath(REPO_DIR / relative_text))
            _plain_directory_chain(artifact.parent)
            _plain_file(artifact, f"artifact {relative_text}")
            lease = _Lease(artifact, f"artifact {relative_text}")
            leases.append(lease)
            if (
                lease.sha256() != artifact_receipt.get("sha256")
                or lease.size_bytes() != artifact_receipt.get("size_bytes")
            ):
                raise ValueError(f"Ender27 bootstrap artifact drifted: {relative_text}")
        return manifest, manifest_bytes, tuple(leases)
    except BaseException:
        for lease in reversed(leases):
            lease.close()
        raise


def _round1_output_paths(name: str) -> tuple[Path, Path, Path]:
    if name not in ROUND1_NAMES:
        raise ValueError(f"Unknown Ender27 Round-1 component: {name}")
    experiment = Path(os.path.abspath(REPO_DIR / PREFIX))
    return (
        experiment / f"predictions/{name}.parquet",
        experiment / f"results/{name}.json",
        experiment / f"receipts/{name}.completion.json",
    )


def _preflight_round1_sequence(name: str) -> tuple[str, ...]:
    """Prove the fixed cohort position before reservation or governed reads."""

    try:
        index = ROUND1_SEQUENCE.index(name)
    except ValueError as error:
        raise ValueError(f"Unknown Ender27 Round-1 component: {name}") from error
    experiment = Path(os.path.abspath(REPO_DIR / PREFIX))
    decision = experiment / "receipts/round1_tempered_alignment.json"
    if os.path.lexists(decision):
        raise ValueError("Ender27 decision destination is already consumed.")

    predecessors = ROUND1_SEQUENCE[:index]
    for predecessor in predecessors:
        for path in _round1_output_paths(predecessor):
            if not os.path.lexists(path):
                raise ValueError(
                    f"Ender27 predecessor output is missing: {path}"
                )
            _plain_directory_chain(path.parent)
            _plain_file(path, f"predecessor output {path.name}")
            if path.stat().st_size <= 0:
                raise ValueError(
                    f"Ender27 predecessor output is empty: {path}"
                )

    for pending in ROUND1_SEQUENCE[index:]:
        for path in _round1_output_paths(pending):
            if os.path.lexists(path):
                raise ValueError(
                    f"Ender27 current or future output is already consumed: {path}"
                )
    return predecessors


def _acquire_predecessor_custody(
    name: str,
    manifest: dict,
    manifest_bytes: bytes,
) -> tuple[_Lease, ...]:
    """Lease and opaque-validate every predecessor under the current manifest."""

    index = ROUND1_SEQUENCE.index(name)
    predecessors = ROUND1_SEQUENCE[:index]
    if not predecessors:
        return ()
    experiment = Path(os.path.abspath(REPO_DIR / PREFIX))
    leases: list[_Lease] = []
    try:
        by_path: dict[Path, _Lease] = {}
        for predecessor in predecessors:
            for path in _round1_output_paths(predecessor):
                lease = _Lease(path, f"predecessor {predecessor} {path.name}")
                leases.append(lease)
                by_path[Path(os.path.abspath(path))] = lease

        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        for predecessor in predecessors:
            prediction_path, result_path, completion_path = (
                _round1_output_paths(predecessor)
            )
            completion_lease = by_path[Path(os.path.abspath(completion_path))]
            try:
                completion = json.loads(
                    completion_lease.read_bytes().decode("utf-8")
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Ender27 predecessor completion is malformed: {predecessor}"
                ) from error
            if not isinstance(completion, dict) or set(completion) != {
                "schema_version",
                "stage",
                "state",
                "component",
                "manifest",
                "config",
                "outputs",
            }:
                raise ValueError(
                    f"Ender27 predecessor completion schema differs: {predecessor}"
                )
            config_relative = f"{PREFIX}/configs/{predecessor}.py"
            if (
                completion["schema_version"] != 1
                or completion["stage"]
                != "ender27-round1-training-completion"
                or completion["state"] != "OUTPUTS_FINALIZED"
                or completion["component"] != predecessor
                or completion["manifest"]
                != {
                    "path": f"{PREFIX}/source_manifest_round1.json",
                    "sha256": manifest_digest,
                    "git_head": manifest["git_head"],
                }
                or completion["config"]
                != {
                    "path": config_relative,
                    "sha256": manifest["files"][config_relative],
                }
                or not isinstance(completion["outputs"], dict)
                or set(completion["outputs"]) != {"predictions", "result"}
            ):
                raise ValueError(
                    f"Ender27 predecessor completion envelope differs: {predecessor}"
                )
            for label, path in (
                ("predictions", prediction_path),
                ("result", result_path),
            ):
                lease = by_path[Path(os.path.abspath(path))]
                inspected = os.fstat(lease.stream.fileno())
                expected = {
                    "path": str(path),
                    "device": int(inspected.st_dev),
                    "inode": int(inspected.st_ino),
                    "size_bytes": int(inspected.st_size),
                    "sha256": lease.sha256(),
                }
                if completion["outputs"].get(label) != expected:
                    raise ValueError(
                        "Ender27 predecessor output identity differs: "
                        f"{predecessor} {label}"
                    )
        return tuple(leases)
    except BaseException:
        for lease in reversed(leases):
            lease.close()
        raise


def _training_main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("round", type=int, choices=(1,))
    parser.add_argument("config", choices=sorted(ROUND1_NAMES))
    args = parser.parse_args(arguments)
    _require_launch()
    _preflight_round1_sequence(args.config)
    config = REPO_DIR / f"{PREFIX}/configs/{args.config}.py"
    # Output custody is consumed before any source manifest or target-bearing
    # artifact is opened. A failed attempt leaves terminal zero-byte evidence.
    with _TrainingOutputReservations(args.config) as reservations:
        manifest, manifest_bytes, leases = _acquire(args.round)
        predecessor_leases = ()
        try:
            predecessor_leases = _acquire_predecessor_custody(
                args.config, manifest, manifest_bytes
            )
            numerai_source = str(REPO_DIR / "numerai")
            if numerai_source not in sys.path:
                sys.path.insert(0, numerai_source)
            pipeline = importlib.import_module("agents.code.modeling.utils.pipeline")
            pipeline.run_ender27_training_from_bootstrap(
                config,
                round_number=args.round,
                manifest=manifest,
                manifest_bytes=manifest_bytes,
                leases=leases,
                reservations=reservations,
            )
            return 0
        finally:
            for lease in reversed(predecessor_leases):
                lease.close()
            for lease in reversed(leases):
                lease.close()


def _load_governed_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load governed Ender27 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _evaluation_main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("round", type=int, choices=(1,))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--numerai-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(arguments)
    _require_launch()
    experiment = Path(os.path.abspath(args.experiment))
    expected_experiment = Path(os.path.abspath(REPO_DIR / PREFIX))
    numerai_dir = Path(os.path.abspath(args.numerai_dir))
    output = Path(os.path.abspath(args.output))
    expected_output = experiment / "receipts/round1_tempered_alignment.json"
    if experiment != expected_experiment:
        raise ValueError("Ender27 evaluator requires the canonical experiment.")
    if numerai_dir != Path(os.path.abspath(REPO_DIR / "numerai")):
        raise ValueError("Ender27 evaluator requires the canonical data root.")
    if output != expected_output:
        raise ValueError("Ender27 evaluator output path differs from the freeze.")

    # CREATE_NEW happens before the source manifest is opened. All bootstrap
    # and evaluator leases remain held until the decision handle is fsynced.
    with _DecisionReservation(output) as decision:
        manifest, manifest_bytes, leases = _acquire(args.round)
        try:
            numerai_source = str(REPO_DIR / "numerai")
            if numerai_source not in sys.path:
                sys.path.insert(0, numerai_source)
            common_path = REPO_DIR / PREFIX / "evaluation_common.py"
            _load_governed_module("evaluation_common", common_path)
            implementation = _load_governed_module(
                "ender27_evaluate_round1_impl",
                REPO_DIR / PREFIX / "evaluate_round1_impl.py",
            )
            implementation.run_bootstrapped(
                REPO_DIR,
                experiment,
                numerai_dir,
                decision,
                manifest=manifest,
                manifest_bytes=manifest_bytes,
                bootstrap_leases=leases,
            )
        finally:
            for lease in reversed(leases):
                lease.close()
    return 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "evaluate":
        return _evaluation_main(sys.argv[2:])
    return _training_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
