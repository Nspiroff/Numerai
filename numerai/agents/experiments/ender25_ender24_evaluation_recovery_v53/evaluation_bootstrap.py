"""Stdlib-only immutable bootstrap for the Ender25 recovery evaluator."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
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
FAMILY = "ender25_ender24_evaluation_recovery_v53"
PREFIX = f"numerai/agents/experiments/{FAMILY}"
ENDER24_PREFIX = "numerai/agents/experiments/ender24_ema_seed_stability_v53"
MANIFEST_NAME = "source_manifest_evaluation_recovery.json"
DECISION_RELATIVE = f"{PREFIX}/receipts/ender24_round1_recovery_decision.json"
ROUND1_NAMES = (
    "r1_control_seed1337",
    "r1_ema995_seed1337",
    "r1_control_seed2027",
    "r1_ema995_seed2027",
)
PORTABLE_TEXT_PATHS = frozenset(
    {
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "protocol/discovery_eras_through_0861.json",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "protocol/feature_columns_all_v53.json",
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
        "scipy": "1.18.0",
        "scikit-learn": "1.9.0",
    },
}
EXPECTED_MANIFEST_FILES = frozenset(
    {
        f"{PREFIX}/evaluate_recovery.py",
        f"{PREFIX}/evaluation_bootstrap.py",
        f"{PREFIX}/evaluation_common.py",
        f"{PREFIX}/evaluate_recovery_impl.py",
        f"{PREFIX}/experiment.md",
        f"{PREFIX}/gate.md",
        f"{PREFIX}/protocol/ender24_input_authority.json",
        f"{PREFIX}/receipts/.gitkeep",
        "numerai/agents/tests/test_ender25_ema_evaluation_recovery.py",
        f"{ENDER24_PREFIX}/evaluation_common.py",
        f"{ENDER24_PREFIX}/configs/base_r1.py",
        *(
            f"{ENDER24_PREFIX}/configs/{name}.py"
            for name in ROUND1_NAMES
        ),
        "numerai/agents/code/metrics/numerai_metrics.py",
        "numerai/agents/code/modeling/utils/constants.py",
        *PORTABLE_TEXT_PATHS,
    }
)

ENDER24_MANIFEST_AUTHORITY = {
    "path": f"{ENDER24_PREFIX}/source_manifest_round1.json",
    "size_bytes": 5_491,
    "sha256": "bd55280e4a99a1b45be87cc5af73aea2615da14a4de0e0662d1ac6c008ab1b35",
    "git_blob": "c5654ca7a860d7381eadf06f4dfe033713a5d703",
    "git_head": "a2bfe0fce7ac1a6a6b075a65b8538aa32165e3c6",
    "manifest_commit": "5a1a75d1b00f639fb04a522dda6c390d5535732f",
}
ENDER24_POSTMORTEM_AUTHORITY = {
    "path": f"{ENDER24_PREFIX}/receipts/round1_execution_postmortem.json",
    "size_bytes": 5_922,
    "sha256": "60acca8896fab9e97d2c559b951019da901fc033fb2b2e65d35d0e7b5101ed03",
    "git_blob": "d23b7b27d85c3b57debb320213d3e3f4f09e67e7",
}
ENDER24_TEXT_AUTHORITY = {
    "era_allowlist": {
        "path": (
            "numerai/agents/experiments/ender21_residual_stability_v53/"
            "protocol/discovery_eras_through_0861.json"
        ),
        "sealed_physical": {
            "size_bytes": 1_941,
            "sha256": "4ffd0ef68092d935c121b45c83a89ef67afe832b48fc259e425d3fe3f51deae7",
        },
        "canonical_lf": {
            "size_bytes": 1_763,
            "sha256": "be0c212a8e910f56dbdae4e1e134fa36ce7e5e1a95e43faa1ccc9e6330f544ca",
        },
    },
    "feature_columns": {
        "path": (
            "numerai/agents/experiments/ender21_residual_stability_v53/"
            "protocol/feature_columns_all_v53.json"
        ),
        "sealed_physical": {
            "size_bytes": 151_736,
            "sha256": "e4df25383aff5ddf9446df275f55a8a93ca64f926a842f4cf84a68280adf769d",
        },
        "canonical_lf": {
            "size_bytes": 148_179,
            "sha256": "663184191e17d2fa4fac6dae017890f0e762368e638d46cfaa489297b9b2049b",
        },
    },
}


def _component_authority(kind: str, values: dict[str, tuple[int, str, str]]) -> dict:
    directory = "receipts" if kind == "completion" else f"{kind}s"
    suffix = ".completion.json" if kind == "completion" else (
        ".parquet" if kind == "prediction" else ".json"
    )
    result = {}
    for name in ROUND1_NAMES:
        size_bytes, sha256, git_blob = values[name]
        receipt = {
            "path": f"{ENDER24_PREFIX}/{directory}/{name}{suffix}",
            "size_bytes": size_bytes,
            "sha256": sha256,
        }
        if git_blob:
            receipt["git_blob"] = git_blob
        result[name] = receipt
    return result


ENDER24_COMPLETION_AUTHORITY = _component_authority(
    "completion",
    {
        "r1_control_seed1337": (1_370, "5a38b9f7211155b7ce9ea71db8c6815a72940f8c999c53e7b2f0ef6d4bd65b4e", "358b6e7bdfc481f294eecaacd7a9aea0168ff5fe"),
        "r1_ema995_seed1337": (1_365, "0b6d500c571abe376ae388ace4abc56430ac8df9c5bce71be7a41657a149431f", "2ddd2d0e2c96d8767f98f0c19cf6dcae5d2624d5"),
        "r1_control_seed2027": (1_369, "f33f40f537413fdd8fc80cb7d656192fbf02d906b6885723894e9fc1776653e4", "58841af05d8f318a6cc1b5a1102228d5fba87dc0"),
        "r1_ema995_seed2027": (1_363, "c63793b84ef0e3fa03a9dc1a8c1aa167a1fb6a29863313e82259301f12f512ed", "6d7aba2a6d45d2e26140fd3466adcf8da5c1f7fb"),
    },
)
ENDER24_RESULT_AUTHORITY = _component_authority(
    "result",
    {
        "r1_control_seed1337": (6_148, "fecf817835e2794c3d9caae16a15a244ff6976636295530a34c59b5772c9b5e0", "8d10219d3799e152035499148d9679bab3750fa1"),
        "r1_ema995_seed1337": (7_678, "0aba98130c6d3f80b91731f59b130531f7792266338fe230ae02f581df43c194", "f8efc4f912643ff3b4917b51e5999e22177efa0a"),
        "r1_control_seed2027": (6_144, "4be1fc26f223275ea7dd9526706d348ca84c66aee5ecd69e3578ee11c5fa807c", "9a754c910032d853f897784ad7b86c41b68ba9f7"),
        "r1_ema995_seed2027": (7_673, "4852ad7788c13197f8a0c9ac6fc4a075974471a0db031d024a417d46332a80df", "d15e9b3fdf1e94a51b14d096222b37f75c77043c"),
    },
)
ENDER24_PREDICTION_AUTHORITY = _component_authority(
    "prediction",
    {
        "r1_control_seed1337": (15_779_944, "3e5285e3d5b9e3dea7df8da3d121a0157fa83e91bbec371c30526a8539aced73", ""),
        "r1_ema995_seed1337": (15_786_841, "373110812795dc907e7ee8346b209204a73a3e242a6665cf14863aeb0abb467f", ""),
        "r1_control_seed2027": (15_786_216, "183e9f2d74f44e34a8b0bd36b34d86d76808ae501a1d06e46b3d33cc89ce0df0", ""),
        "r1_ema995_seed2027": (15_787_673, "763d8030056af4e83828ce952c86ec499452014dd9a14ba361e0daccd7f501f6", ""),
    },
)
EXPECTED_EXTERNAL_ARTIFACTS = {
    "numerai/v5.3/ender21_discovery_full_through_0861.parquet": {
        "size_bytes": 1_302_848_771,
        "sha256": "476d561ba8515a0066e892c5489a5ae1db6443587e9d5d06a9a6280400a701b9",
        "last_era": "0861",
    },
    "numerai/v5.3/ender21_discovery_benchmark_models_through_0861.parquet": {
        "size_bytes": 23_325_224,
        "sha256": "c2db9a77811390e9b9c47926b62fbfb7a6c7af24bb9c4db63137798e61b955b6",
        "last_era": "0861",
    },
    **{
        receipt["path"]: {
            "size_bytes": receipt["size_bytes"],
            "sha256": receipt["sha256"],
            "git_ignored": True,
        }
        for receipt in ENDER24_PREDICTION_AUTHORITY.values()
    },
}
EXPECTED_ENDER24_AUTHORITY = {
    "source_commit": "aebc577249d202ab9f32e4dac2bc939f496a6ddc",
    "mechanical_receipt_commit": "a2bfe0fce7ac1a6a6b075a65b8538aa32165e3c6",
    "manifest_commit": "5a1a75d1b00f639fb04a522dda6c390d5535732f",
    "launch_commit": "7adc6724bd41689e34e8d21effa088b0ff606022",
    "evidence_commit": "e3c7b28be06e175b2d784082cd8606972140010a",
    "terminal_commit": "d12f75552d76ebceb7c73fa3ff0ef9c608105599",
    "terminal_tag": "numerai-ender24-terminal",
    "source_manifest": ENDER24_MANIFEST_AUTHORITY,
    "postmortem": ENDER24_POSTMORTEM_AUTHORITY,
    "text_authority": ENDER24_TEXT_AUTHORITY,
    "completions": ENDER24_COMPLETION_AUTHORITY,
    "results": ENDER24_RESULT_AUTHORITY,
    "predictions": ENDER24_PREDICTION_AUTHORITY,
    "old_decision": {
        "path": f"{ENDER24_PREFIX}/receipts/round1_ema_stability.json",
        "required_absent": True,
    },
}


def _valid_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_recovery_manifest(manifest: object) -> dict:
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "frozen_at",
        "git_head",
        "hash_algorithm",
        "files",
        "external_artifacts",
        "runtime",
        "ender24_authority",
    }:
        raise ValueError("Ender25 recovery manifest schema differs.")
    if manifest["schema_version"] != 1 or manifest["hash_algorithm"] != "sha256":
        raise ValueError("Ender25 recovery manifest version differs.")
    if not _valid_hex(manifest["git_head"], 40):
        raise ValueError("Ender25 recovery source checkpoint is malformed.")
    if not isinstance(manifest["files"], dict) or set(manifest["files"]) != EXPECTED_MANIFEST_FILES:
        raise ValueError("Ender25 recovery source set differs.")
    if any(not _valid_hex(value, 64) for value in manifest["files"].values()):
        raise ValueError("Ender25 recovery source digest is malformed.")
    if manifest["external_artifacts"] != EXPECTED_EXTERNAL_ARTIFACTS:
        raise ValueError("Ender25 recovery external artifact set differs.")
    if manifest["runtime"] != EXPECTED_RUNTIME:
        raise ValueError("Ender25 recovery runtime contract differs.")
    if manifest["ender24_authority"] != EXPECTED_ENDER24_AUTHORITY:
        raise ValueError("Ender25 immutable Ender24 authority differs.")
    return manifest


def expected_recovery_manifest_template() -> dict:
    """Return a schema-valid template whose placeholder hashes must be sealed."""

    return {
        "schema_version": 1,
        "frozen_at": "1970-01-01T00:00:00Z",
        "git_head": "0" * 40,
        "hash_algorithm": "sha256",
        "files": {relative: "0" * 64 for relative in EXPECTED_MANIFEST_FILES},
        "external_artifacts": EXPECTED_EXTERNAL_ARTIFACTS,
        "runtime": EXPECTED_RUNTIME,
        "ender24_authority": EXPECTED_ENDER24_AUTHORITY,
    }


def _require_plain_directory_chain(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    for directory in reversed([absolute, *absolute.parents]):
        if directory == directory.parent or not os.path.lexists(directory):
            continue
        inspected = directory.lstat()
        attributes = getattr(inspected, "st_file_attributes", 0)
        if directory.is_symlink() or bool(
            attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ) or not stat.S_ISDIR(inspected.st_mode):
            raise ValueError(f"Ender25 custody directory is not plain: {directory}")


def _require_plain_file(path: Path, label: str) -> os.stat_result:
    try:
        inspected = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing {label}: {path}") from error
    attributes = getattr(inspected, "st_file_attributes", 0)
    if path.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ) or not stat.S_ISREG(inspected.st_mode) or inspected.st_nlink != 1:
        raise ValueError(f"Ender25 {label} must be a unique plain file: {path}")
    return inspected


class HeldReadLease:
    """Pin a unique regular file without write/delete sharing on Windows."""

    def __init__(self, path: Path, label: str) -> None:
        self.path = Path(os.path.abspath(path))
        self.label = label
        self.stream = None
        _require_plain_directory_chain(self.path.parent)
        expected = _require_plain_file(self.path, label)
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
            handle = create_file(str(self.path), 0x80000000, 0x1, None, 3, 0x80, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ValueError(f"Cannot lease {label}: {self.path}") from ctypes.WinError(ctypes.get_last_error())
            try:
                descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | getattr(os, "O_BINARY", 0))
            except BaseException:
                kernel32.CloseHandle(handle)
                raise
        else:
            descriptor = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        self.stream = os.fdopen(descriptor, "rb", buffering=0)
        observed = os.fstat(self.stream.fileno())
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or (int(observed.st_dev), int(observed.st_ino), int(observed.st_size))
            != (int(expected.st_dev), int(expected.st_ino), int(expected.st_size))
        ):
            self.close()
            raise ValueError(f"Leased {label} identity changed during open.")

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def stat(self):
        return os.fstat(self.stream.fileno())

    def size_bytes(self) -> int:
        return int(self.stat().st_size)

    def read_bytes(self) -> bytes:
        self.stream.seek(0)
        payload = self.stream.read()
        self.stream.seek(0)
        return payload

    def sha256(self) -> str:
        digest = hashlib.sha256()
        self.stream.seek(0)
        while chunk := self.stream.read(8 * 1024 * 1024):
            digest.update(chunk)
        self.stream.seek(0)
        return digest.hexdigest()

    def close(self) -> None:
        stream = self.stream
        self.stream = None
        if stream is not None:
            stream.close()


class DecisionReservation:
    """CREATE_NEW decision removed on failure and retained only after fsync."""

    def __init__(self, path: Path) -> None:
        self.path = Path(os.path.abspath(path))
        self.stream = None
        self.identity = None
        self.committed = False

    def __enter__(self):
        _require_plain_directory_chain(self.path.parent)
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
            handle = create_file(str(self.path), 0xC0000000, 0x1, None, 1, 0x80, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ValueError(f"Cannot reserve Ender25 decision: {self.path}") from ctypes.WinError(ctypes.get_last_error())
            descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDWR | getattr(os, "O_BINARY", 0))
        else:
            try:
                descriptor = os.open(
                    self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600
                )
            except OSError as error:
                raise ValueError(
                    f"Cannot reserve Ender25 decision: {self.path}"
                ) from error
        self.stream = os.fdopen(descriptor, "w+b", buffering=0)
        opened = os.fstat(self.stream.fileno())
        lexical = self.path.lstat()
        self.identity = (int(opened.st_dev), int(opened.st_ino))
        if opened.st_nlink != 1 or opened.st_size != 0 or self.identity != (int(lexical.st_dev), int(lexical.st_ino)):
            self.__exit__()
            raise ValueError("Reserved Ender25 decision identity is malformed.")
        return self

    def commit_json(self, payload: dict) -> bytes:
        if self.stream is None or self.committed or self.stat().st_size != 0:
            raise RuntimeError("Ender25 decision reservation is not writable.")
        encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
        self.stream.write(encoded)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        opened = os.fstat(self.stream.fileno())
        lexical = self.path.lstat()
        if (
            opened.st_nlink != 1
            or lexical.st_nlink != 1
            or not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(lexical.st_mode)
            or self.identity != (int(opened.st_dev), int(opened.st_ino))
            or self.identity != (int(lexical.st_dev), int(lexical.st_ino))
        ):
            raise ValueError("Ender25 decision identity changed before durability.")
        self.committed = True
        return encoded

    def stat(self):
        return os.fstat(self.stream.fileno())

    def __exit__(self, *_args) -> None:
        stream = self.stream
        identity = self.identity
        self.stream = None
        if stream is not None:
            stream.close()
        if self.committed or identity is None or not os.path.lexists(self.path):
            return
        lexical = self.path.lstat()
        if (
            (int(lexical.st_dev), int(lexical.st_ino)) != identity
            or not stat.S_ISREG(lexical.st_mode)
            or lexical.st_nlink != 1
            or lexical.st_size != 0
        ):
            raise ValueError("Refusing to remove changed Ender25 reservation.")
        self.path.unlink()


def _git(*arguments: str, allow_one: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", *arguments], cwd=REPO_DIR, capture_output=True, text=True, check=False)
    if result.returncode not in ({0, 1} if allow_one else {0}):
        raise ValueError(result.stderr.strip() or "Ender25 Git verification failed.")
    return result


def _canonical_repo_file(relative: str, label: str) -> Path:
    if type(relative) is not str or not relative:
        raise ValueError(f"{label} path is malformed.")
    parsed = Path(relative)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError(f"{label} path is not canonical: {relative}")
    absolute = Path(os.path.abspath(REPO_DIR / parsed))
    try:
        absolute.relative_to(REPO_DIR)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository.") from error
    return absolute


def _require_launch() -> None:
    preloaded = sorted(name for name in sys.modules if name == "agents" or name.startswith("agents."))
    if preloaded:
        raise ValueError(f"Governed Numerai modules were pre-imported: {preloaded}")
    option = getattr(sys, "_xoptions", {}).get("pycache_prefix")
    if sys.flags.dont_write_bytecode != 1 or not sys.dont_write_bytecode or sys.flags.isolated != 1 or sys.flags.safe_path != 1:
        raise ValueError("Ender25 bootstrap requires isolated Python -I -B -P.")
    if type(option) is not str or not option or sys.pycache_prefix != option:
        raise ValueError("Ender25 bootstrap requires exact -X pycache_prefix.")
    prefix = Path(option)
    if not prefix.is_absolute() or prefix != Path(os.path.abspath(prefix)):
        raise ValueError("Ender25 pycache prefix must be absolute and canonical.")
    try:
        prefix.relative_to(REPO_DIR)
    except ValueError:
        pass
    else:
        raise ValueError("Ender25 pycache prefix must be outside the repository.")
    _require_plain_directory_chain(prefix)
    if next(prefix.iterdir(), None) is not None:
        raise ValueError("Ender25 pycache prefix must be freshly empty.")


def _verify_topology(manifest: dict) -> None:
    frozen_commits = (
        EXPECTED_ENDER24_AUTHORITY["source_commit"],
        EXPECTED_ENDER24_AUTHORITY["mechanical_receipt_commit"],
        EXPECTED_ENDER24_AUTHORITY["manifest_commit"],
        EXPECTED_ENDER24_AUTHORITY["launch_commit"],
        EXPECTED_ENDER24_AUTHORITY["evidence_commit"],
        EXPECTED_ENDER24_AUTHORITY["terminal_commit"],
    )
    for commit in (manifest["git_head"], *frozen_commits):
        if _git("rev-parse", "--verify", f"{commit}^{{commit}}").stdout.strip() != commit:
            raise ValueError(f"Ender25 checkpoint does not resolve exactly: {commit}")
    for ancestor, descendant in zip(frozen_commits, frozen_commits[1:]):
        if _git(
            "merge-base", "--is-ancestor", ancestor, descendant, allow_one=True
        ).returncode:
            raise ValueError(
                f"Ender24 checkpoint ancestry differs: {ancestor} -> {descendant}"
            )
    if _git(
        "merge-base",
        "--is-ancestor",
        EXPECTED_ENDER24_AUTHORITY["terminal_commit"],
        manifest["git_head"],
        allow_one=True,
    ).returncode:
        raise ValueError("Ender25 source checkpoint does not descend from Ender24.")
    tag_target = _git("rev-parse", f"{EXPECTED_ENDER24_AUTHORITY['terminal_tag']}^{{}}").stdout.strip()
    if tag_target != EXPECTED_ENDER24_AUTHORITY["terminal_commit"]:
        raise ValueError("Ender24 terminal tag target differs.")
    manifest_blob = _git("rev-parse", f"{ENDER24_MANIFEST_AUTHORITY['manifest_commit']}:{ENDER24_MANIFEST_AUTHORITY['path']}").stdout.strip()
    if manifest_blob != ENDER24_MANIFEST_AUTHORITY["git_blob"]:
        raise ValueError("Ender24 source-manifest Git binding differs.")
    for mapping in (ENDER24_COMPLETION_AUTHORITY, ENDER24_RESULT_AUTHORITY):
        for receipt in mapping.values():
            blob = _git("rev-parse", f"{EXPECTED_ENDER24_AUTHORITY['evidence_commit']}:{receipt['path']}").stdout.strip()
            if blob != receipt["git_blob"]:
                raise ValueError(f"Ender24 evidence Git binding differs: {receipt['path']}")
    evidence_paths = {
        receipt["path"]
        for mapping in (ENDER24_COMPLETION_AUTHORITY, ENDER24_RESULT_AUTHORITY)
        for receipt in mapping.values()
    }
    evidence_paths.update(
        {
            ".gitattributes",
            "numerai/agents/experiments/README.md",
            ENDER24_POSTMORTEM_AUTHORITY["path"],
            f"{ENDER24_PREFIX}/round1_execution_postmortem.md",
        }
    )
    changed_evidence = set(
        _git(
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            EXPECTED_ENDER24_AUTHORITY["evidence_commit"],
        ).stdout.splitlines()
    )
    if changed_evidence != evidence_paths:
        raise ValueError("Ender24 evidence checkpoint path set differs.")
    postmortem_blob = _git("rev-parse", f"{EXPECTED_ENDER24_AUTHORITY['terminal_commit']}:{ENDER24_POSTMORTEM_AUTHORITY['path']}").stdout.strip()
    if postmortem_blob != ENDER24_POSTMORTEM_AUTHORITY["git_blob"]:
        raise ValueError("Ender24 terminal postmortem Git binding differs.")
    head = _git("rev-parse", "HEAD").stdout.strip()
    expected_manifest_path = f"{PREFIX}/{MANIFEST_NAME}"
    seal_candidates = tuple(
        line
        for line in _git(
            "log",
            "--format=%H",
            "--diff-filter=A",
            "--",
            expected_manifest_path,
        ).stdout.splitlines()
        if line
    )
    if len(seal_candidates) != 1:
        raise ValueError("Ender25 manifest seal commit is not unique in HEAD history.")
    seal_commit = seal_candidates[0]
    seal_line = _git(
        "rev-list", "--parents", "-n", "1", seal_commit
    ).stdout.strip().split()
    seal_changed = set(
        _git(
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            seal_commit,
        ).stdout.splitlines()
    )
    if (
        len(seal_line) != 2
        or seal_line[1] != manifest["git_head"]
        or seal_changed != {expected_manifest_path}
        or _git(
            "merge-base", "--is-ancestor", seal_commit, head, allow_one=True
        ).returncode
    ):
        raise ValueError("Ender25 manifest-only seal topology differs.")
    _git("cat-file", "-e", f"HEAD:{expected_manifest_path}")
    seal_blob = _git(
        "rev-parse", f"{seal_commit}:{expected_manifest_path}"
    ).stdout.strip()
    head_blob = _git("rev-parse", f"HEAD:{expected_manifest_path}").stdout.strip()
    live_blob = _git(
        "hash-object", f"--path={expected_manifest_path}", expected_manifest_path
    ).stdout.strip()
    if len({seal_blob, head_blob, live_blob}) != 1:
        raise ValueError("Ender25 recovery manifest drifted after its seal.")


def _lease_receipt(stack: ExitStack, leases: dict[Path, HeldReadLease], path: Path, expected: dict, label: str) -> HeldReadLease:
    lease = stack.enter_context(HeldReadLease(path, label))
    leases[Path(os.path.abspath(path))] = lease
    if lease.size_bytes() != expected["size_bytes"] or lease.sha256() != expected["sha256"]:
        raise ValueError(f"Ender25 held receipt differs for {label}.")
    return lease


def _strict_json(raw: bytes, label: str) -> object:
    def reject_constant(value: str):
        raise ValueError(f"{label} contains non-finite JSON: {value}")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key: {key}")
            result[key] = value
        return result

    if raw.startswith(b"\xef\xbb\xbf") or b"\x00" in raw:
        raise ValueError(f"{label} has a forbidden text encoding.")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid strict JSON.") from error


def acquire_custody(stack: ExitStack, experiment: Path, numerai_dir: Path):
    """Lease and verify every source/input without parsing evidence payloads."""
    leases: dict[Path, HeldReadLease] = {}
    manifest_path = experiment / MANIFEST_NAME
    manifest_lease = stack.enter_context(HeldReadLease(manifest_path, "recovery manifest"))
    leases[Path(os.path.abspath(manifest_path))] = manifest_lease
    try:
        manifest = json.loads(manifest_lease.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Ender25 recovery manifest is invalid JSON.") from error
    validate_recovery_manifest(manifest)
    if platform.python_version() != EXPECTED_RUNTIME["python"]:
        raise ValueError("Ender25 Python runtime differs.")
    for package, version in EXPECTED_RUNTIME["packages"].items():
        try:
            actual = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError as error:
            raise ValueError(f"Ender25 runtime package is absent: {package}") from error
        if actual != version:
            raise ValueError(f"Ender25 runtime package drifted: {package}")
    _verify_topology(manifest)
    governed = [f"{PREFIX}/{MANIFEST_NAME}", *sorted(manifest["files"])]
    if _git("status", "--porcelain=v1", "--untracked-files=all", "--", *governed).stdout:
        raise ValueError("Ender25 recovery source is not committed and clean.")
    for relative, digest in sorted(manifest["files"].items()):
        path = _canonical_repo_file(relative, "recovery source")
        lease = stack.enter_context(HeldReadLease(path, f"source {relative}"))
        leases[Path(os.path.abspath(path))] = lease
        if lease.sha256() != digest:
            raise ValueError(f"Ender25 recovery source hash drifted: {relative}")
        _git("cat-file", "-e", f"{manifest['git_head']}:{relative}")
        if _git("diff", "--quiet", manifest["git_head"], "--", relative, allow_one=True).returncode:
            raise ValueError(f"Ender25 recovery source differs from checkpoint: {relative}")
    common = _load_module(
        "ender25_recovery_common", experiment / "evaluation_common.py"
    )
    _lease_receipt(stack, leases, REPO_DIR / ENDER24_MANIFEST_AUTHORITY["path"], ENDER24_MANIFEST_AUTHORITY, "Ender24 source manifest")
    old_manifest = common.strict_json(
        leases[
            Path(
                os.path.abspath(REPO_DIR / ENDER24_MANIFEST_AUTHORITY["path"])
            )
        ].read_bytes(),
        "Ender24 source manifest",
    )
    if not isinstance(old_manifest, dict) or old_manifest.get("git_head") != ENDER24_MANIFEST_AUTHORITY["git_head"]:
        raise ValueError("Ender24 source-manifest envelope differs.")
    for relative, digest in sorted(old_manifest.get("files", {}).items()):
        path = _canonical_repo_file(relative, "Ender24 source")
        canonical = Path(os.path.abspath(path))
        lease = leases.get(canonical)
        if lease is None:
            lease = stack.enter_context(HeldReadLease(path, f"Ender24 source {relative}"))
            leases[canonical] = lease
        # Ender25 deliberately supersedes only the physical-byte identity of
        # the two portable JSON authorities. Their held raw bytes were already
        # verified against the Ender25 manifest above; canonical LF identity
        # and exact semantics are enforced before any scoring import.
        if relative not in PORTABLE_TEXT_PATHS and lease.sha256() != digest:
            raise ValueError(f"Ender24 source raw bytes drifted: {relative}")
    for relative, expected in sorted(old_manifest.get("external_artifacts", {}).items()):
        _lease_receipt(stack, leases, _canonical_repo_file(relative, "Ender24 external"), expected, f"Ender24 external {relative}")
    for mapping, label in (
        (ENDER24_COMPLETION_AUTHORITY, "completion"),
        (ENDER24_RESULT_AUTHORITY, "result"),
        (ENDER24_PREDICTION_AUTHORITY, "prediction"),
    ):
        for name in ROUND1_NAMES:
            receipt = mapping[name]
            _lease_receipt(stack, leases, _canonical_repo_file(receipt["path"], f"Ender24 {label}"), receipt, f"Ender24 {name} {label}")
    _lease_receipt(stack, leases, _canonical_repo_file(ENDER24_POSTMORTEM_AUTHORITY["path"], "postmortem"), ENDER24_POSTMORTEM_AUTHORITY, "Ender24 terminal postmortem")
    old_decision = _canonical_repo_file(EXPECTED_ENDER24_AUTHORITY["old_decision"]["path"], "old decision")
    if os.path.lexists(old_decision):
        raise ValueError("The forbidden Ender24 decision path is not absent.")
    custody = common.RecoveryCustody(REPO_DIR, experiment, numerai_dir, manifest, leases)
    return manifest, common, custody


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load governed module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--numerai-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _require_launch()
    experiment = Path(os.path.abspath(args.experiment))
    numerai_dir = Path(os.path.abspath(args.numerai_dir))
    output = Path(os.path.abspath(args.output))
    if experiment != Path(os.path.abspath(REPO_DIR / PREFIX)):
        raise ValueError("Ender25 recovery experiment path differs.")
    if numerai_dir != Path(os.path.abspath(REPO_DIR / "numerai")):
        raise ValueError("Ender25 recovery Numerai root differs.")
    if output != Path(os.path.abspath(REPO_DIR / DECISION_RELATIVE)):
        raise ValueError("Ender25 recovery decision path differs.")
    if not output.parent.is_dir():
        raise ValueError("Ender25 decision parent must already exist.")

    # CREATE_NEW is consumed before the recovery manifest or any immutable
    # Ender24 input is opened. All leases survive until commit_json fsync.
    with DecisionReservation(output) as decision, ExitStack() as stack:
        old_decision = Path(
            os.path.abspath(
                REPO_DIR / EXPECTED_ENDER24_AUTHORITY["old_decision"]["path"]
            )
        )
        if os.path.lexists(old_decision):
            raise ValueError("The forbidden Ender24 decision path is not absent.")
        manifest, common, custody = acquire_custody(stack, experiment, numerai_dir)
        # Cross the complete opaque-evidence and canonical-authority barriers
        # before importing any third-party scoring dependency.
        completions = custody.preflight_completions()
        authority_bundle = custody.load_authority()
        numerai_source = str(REPO_DIR / "numerai")
        if numerai_source not in sys.path:
            sys.path.insert(0, numerai_source)
        numpy = importlib.import_module("numpy")
        pandas = importlib.import_module("pandas")
        parquet = importlib.import_module("pyarrow.parquet")
        metrics = importlib.import_module("agents.code.metrics.numerai_metrics")
        old_common = _load_module(
            "ender25_frozen_ender24_evaluation_common",
            REPO_DIR / f"{ENDER24_PREFIX}/evaluation_common.py",
        )
        old_common.np = numpy
        old_common.pd = pandas
        old_common.pq = parquet
        old_common.numerai_metrics = metrics
        old_common.era_cv_splits = common.exact_era_cv_splits
        old_common.REPO_DIR = REPO_DIR
        old_common.PREDICTION_SEMANTICS_METADATA_KEY = b"numerai.agents.prediction_semantics"
        implementation = _load_module(
            "ender25_evaluate_recovery_impl",
            experiment / "evaluate_recovery_impl.py",
        )
        payload = implementation.run_bootstrapped(
            experiment,
            numerai_dir,
            custody,
            old_common,
            decision,
            completions=completions,
            authority_bundle=authority_bundle,
        )
    print(f"state={payload['state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
