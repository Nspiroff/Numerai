"""Shared, fail-closed scoring rules for frozen Ender24 Round 1.

Only the standard library is imported at module load time.  The isolated
bootstrap first acquires the complete immutable envelope, then calls
``load_governed_dependencies`` before any scoring dependency is imported.
"""

from __future__ import annotations

import ast
import hashlib
from importlib import import_module
import json
import math
import os
from pathlib import Path
import runpy
import stat


np = None
pd = None
pq = None
numerai_metrics = None
era_cv_splits = None
REPO_DIR = None
PREDICTION_SEMANTICS_METADATA_KEY = None
_pipeline = None


EXPERIMENT_NAME = "ender24_ema_seed_stability_v53"
PREFIX = f"numerai/agents/experiments/{EXPERIMENT_NAME}"
TARGET = "target_ender_20"
BENCHMARK = "v53_lgbm_ender20"
ERA = "era"
ID = "id"
ROUND1_NAMES = (
    "r1_control_seed1337",
    "r1_ema995_seed1337",
    "r1_control_seed2027",
    "r1_ema995_seed2027",
)
PAIR_NAMES = {
    "1337": ("r1_control_seed1337", "r1_ema995_seed1337"),
    "2027": ("r1_control_seed2027", "r1_ema995_seed2027"),
}
RECENT_BLOCKS = {
    "0705-0741": tuple(f"{era:04d}" for era in range(705, 742, 4)),
    "0745-0781": tuple(f"{era:04d}" for era in range(745, 782, 4)),
    "0785-0821": tuple(f"{era:04d}" for era in range(785, 822, 4)),
    "0825-0861": tuple(f"{era:04d}" for era in range(825, 862, 4)),
}
EXPECTED_SEMANTICS = {
    "artifact_kind": "out_of_fold_validation",
    "column": "prediction",
    "era_column": ERA,
    "fold_column": "cv_fold",
    "fold_index_base": 0,
    "inverse_target_transform_applied": False,
    "pipeline_postprocess": {"type": "identity"},
    "producer": "model.predict",
    "schema_version": 1,
    "stored_target": {"column": TARGET, "transform": {"type": "identity"}},
    "training_target": {
        "column": TARGET,
        "transform": {
            "benchmark_col": BENCHMARK,
            "era_col": ERA,
            "fit_intercept": True,
            "per_era": True,
            "proportion": 1.0,
            "type": "residual_to_benchmark",
        },
    },
}
BASE_DIAGNOSTIC_KEYS = {
    "best_epoch",
    "n_parameters",
    "epochs_ran",
    "best_val_loss",
    "final_train_loss",
}
EMA_DIAGNOSTIC_KEYS = {
    "ema_decay",
    "ema_updates",
    "ema_live_state_sha256",
    "ema_shadow_state_sha256",
    "ema_inference_state_sha256",
}


def load_governed_dependencies() -> None:
    """Import scoring code only after the caller holds frozen leases."""

    global np, pd, pq, numerai_metrics, era_cv_splits, REPO_DIR
    global PREDICTION_SEMANTICS_METADATA_KEY, _pipeline
    if np is not None:
        return
    np = import_module("numpy")
    pd = import_module("pandas")
    pq = import_module("pyarrow.parquet")
    numerai_metrics = import_module("agents.code.metrics.numerai_metrics")
    constants = import_module("agents.code.modeling.utils.constants")
    cv = import_module("agents.code.modeling.utils.numerai_cv")
    pipeline = import_module("agents.code.modeling.utils.pipeline")
    REPO_DIR = constants.REPO_DIR
    era_cv_splits = cv.era_cv_splits
    PREDICTION_SEMANTICS_METADATA_KEY = pipeline.PREDICTION_SEMANTICS_METADATA_KEY
    _pipeline = pipeline


def verify_governed_manifest(custody: "EvaluationCustody") -> dict:
    if _pipeline is None:
        raise RuntimeError("Governed dependencies have not been loaded")
    verified = _pipeline._verify_ender24_round1_manifest()
    if verified != custody.manifest:
        raise ValueError("Held and independently verified manifests differ")
    return verified


class HeldReadLease:
    """Read-only, no-delete-share lease with path/handle identity proof."""

    def __init__(self, path: Path, label: str) -> None:
        self.path = Path(os.path.abspath(path))
        self.label = label
        require_plain_directory_chain(self.path.parent)
        require_plain_file(self.path, label)
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
                str(self.path), 0x80000000, 0x00000001, None, 3, 0x00000080, None
            )
            if handle == ctypes.c_void_p(-1).value:
                raise ValueError(f"Cannot lease {label}: {self.path}") from ctypes.WinError(
                    ctypes.get_last_error()
                )
            try:
                descriptor = msvcrt.open_osfhandle(
                    int(handle), os.O_RDONLY | getattr(os, "O_BINARY", 0)
                )
            except BaseException:
                kernel32.CloseHandle(handle)
                raise
            self.stream = os.fdopen(descriptor, "rb", buffering=0)
        else:
            descriptor = os.open(self.path, os.O_RDONLY)
            self.stream = os.fdopen(descriptor, "rb", buffering=0)
        opened = os.fstat(self.stream.fileno())
        lexical = self.path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (int(opened.st_dev), int(opened.st_ino))
            != (int(lexical.st_dev), int(lexical.st_ino))
        ):
            self.close()
            raise ValueError(f"Leased {label} identity is malformed or changed")

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

    def size_bytes(self) -> int:
        return int(os.fstat(self.stream.fileno()).st_size)

    def close(self) -> None:
        stream = getattr(self, "stream", None)
        self.stream = None
        if stream is not None:
            stream.close()


class DecisionReservation:
    """CREATE_NEW decision handle removed on failure, durable on commit."""

    def __init__(self, path: Path) -> None:
        self.path = Path(os.path.abspath(path))
        self.stream = None
        self.identity = None
        self.committed = False

    def __enter__(self):
        require_plain_directory_chain(self.path.parent)
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
                str(self.path), 0xC0000000, 0x00000001, None, 1, 0x00000080, None
            )
            if handle == ctypes.c_void_p(-1).value:
                raise ValueError(
                    f"Cannot reserve create-new decision: {self.path}"
                ) from ctypes.WinError(ctypes.get_last_error())
            try:
                descriptor = msvcrt.open_osfhandle(
                    int(handle), os.O_RDWR | getattr(os, "O_BINARY", 0)
                )
            except BaseException:
                kernel32.CloseHandle(handle)
                raise
        else:
            try:
                descriptor = os.open(
                    self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600
                )
            except OSError as error:
                raise ValueError(
                    f"Cannot reserve create-new decision: {self.path}"
                ) from error
        self.stream = os.fdopen(descriptor, "w+b", buffering=0)
        inspected = os.fstat(self.stream.fileno())
        lexical = self.path.lstat()
        self.identity = (int(inspected.st_dev), int(inspected.st_ino))
        if (
            inspected.st_nlink != 1
            or inspected.st_size != 0
            or self.identity != (int(lexical.st_dev), int(lexical.st_ino))
        ):
            self.__exit__(None, None, None)
            raise ValueError("Reserved decision identity is malformed")
        return self

    def commit_json(self, payload: dict) -> bytes:
        if self.stream is None or self.committed:
            raise RuntimeError("Decision reservation is not writable")
        if os.fstat(self.stream.fileno()).st_size != 0:
            raise ValueError("Reserved decision is no longer empty")
        payload_bytes = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode(
                "utf-8"
            )
            + b"\n"
        )
        self.stream.write(payload_bytes)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        lexical = self.path.lstat()
        inspected = os.fstat(self.stream.fileno())
        current = (int(inspected.st_dev), int(inspected.st_ino))
        if self.identity != (int(lexical.st_dev), int(lexical.st_ino)) or (
            self.identity != current
        ):
            raise ValueError("Decision path identity changed before durability")
        self.committed = True
        return payload_bytes

    def __exit__(self, *_args) -> None:
        stream = self.stream
        identity = self.identity
        self.stream = None
        if stream is not None:
            stream.close()
        if self.committed or identity is None or not os.path.lexists(self.path):
            return
        lexical = self.path.lstat()
        if (int(lexical.st_dev), int(lexical.st_ino)) != identity:
            raise ValueError("Refusing to remove changed decision reservation")
        self.path.unlink()


def require_plain_directory_chain(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    for directory in reversed([absolute, *absolute.parents]):
        if directory == directory.parent:
            continue
        inspected = directory.lstat()
        attributes = getattr(inspected, "st_file_attributes", 0)
        if directory.is_symlink() or bool(
            attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise ValueError(
                f"Evaluator directory may not be a reparse point: {directory}"
            )
        if not stat.S_ISDIR(inspected.st_mode):
            raise ValueError(f"Evaluator path is not a directory: {directory}")


def require_plain_file(path: Path, label: str) -> None:
    inspected = path.lstat()
    attributes = getattr(inspected, "st_file_attributes", 0)
    if path.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ):
        raise ValueError(f"Evaluator {label} may not be a reparse point: {path}")
    if not stat.S_ISREG(inspected.st_mode) or inspected.st_nlink != 1:
        raise ValueError(f"Evaluator {label} is not a unique regular file: {path}")


def require_frozen_python_runtime(repo_dir: Path) -> Path:
    import sys

    launch = getattr(sys, "_xoptions", {}).get("pycache_prefix")
    if (
        sys.flags.dont_write_bytecode != 1
        or not sys.dont_write_bytecode
        or sys.pycache_prefix is None
        or not isinstance(launch, str)
        or not launch
    ):
        raise ValueError(
            "Ender24 evaluation requires Python -B with an isolated "
            "-X pycache_prefix directory"
        )
    prefix = Path(os.path.abspath(sys.pycache_prefix))
    if prefix != Path(os.path.abspath(launch)) or not prefix.is_absolute():
        raise ValueError("Ender24 pycache_prefix differs from frozen launch state")
    try:
        prefix.relative_to(Path(os.path.abspath(repo_dir)))
    except ValueError:
        pass
    else:
        raise ValueError("Ender24 pycache_prefix must be outside the repository")
    require_plain_directory_chain(prefix)
    if next(prefix.iterdir(), None) is not None:
        raise ValueError("Ender24 pycache_prefix must be freshly empty")
    return prefix


def _manifest_file_set(_round_number: int = 1) -> set[str]:
    if _round_number != 1:
        raise ValueError("Ender24 currently defines only a Round-1 manifest")
    shared = {
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
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "protocol/discovery_eras_through_0861.json",
        "numerai/agents/experiments/ender21_residual_stability_v53/"
        "protocol/feature_columns_all_v53.json",
        f"{PREFIX}/experiment.md",
        f"{PREFIX}/gate.md",
        f"{PREFIX}/protocol/discovery_data_authority.json",
        f"{PREFIX}/protocol/mechanical_activity_receipt.json",
        "numerai/agents/tests/test_ender24_ema_seed_stability.py",
        f"{PREFIX}/configs/base_r1.py",
        f"{PREFIX}/training_bootstrap.py",
        f"{PREFIX}/run_round1.py",
        f"{PREFIX}/evaluation_common.py",
        f"{PREFIX}/evaluate_round1.py",
        f"{PREFIX}/evaluate_round1_impl.py",
    }
    shared.update(f"{PREFIX}/configs/{name}.py" for name in (
        *ROUND1_NAMES,
        "r2_control_seed7331",
        "r2_ema995_seed7331",
    ))
    return shared


class EvaluationCustody:
    """Held immutable envelope for one Ender24 Round-1 evaluation."""

    def __init__(self, repo_dir: Path, round_number: int) -> None:
        self.repo_dir = Path(os.path.abspath(repo_dir))
        self.round_number = round_number
        self.leases: dict[Path, HeldReadLease] = {}
        self.manifest = None

    def _hold(self, path: Path, label: str) -> HeldReadLease:
        canonical = Path(os.path.abspath(path))
        if canonical not in self.leases:
            self.leases[canonical] = HeldReadLease(canonical, label)
        return self.leases[canonical]

    def __enter__(self):
        try:
            return self._acquire()
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _acquire(self):
        if self.round_number != 1:
            raise ValueError("Unknown Ender24 evaluation round")
        require_frozen_python_runtime(self.repo_dir)
        experiment = self.repo_dir / PREFIX
        manifest_path = experiment / "source_manifest_round1.json"
        manifest_lease = self._hold(manifest_path, "source manifest")
        try:
            manifest = json.loads(manifest_lease.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Ender24 source manifest is not valid UTF-8 JSON") from error
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "frozen_at",
            "git_head",
            "hash_algorithm",
            "files",
            "external_artifacts",
            "runtime",
        }:
            raise ValueError("Ender24 source manifest schema differs")
        if manifest["schema_version"] != 1 or manifest["hash_algorithm"] != "sha256":
            raise ValueError("Ender24 source manifest version differs")
        files = manifest["files"]
        external = manifest["external_artifacts"]
        if not isinstance(files, dict) or set(files) != _manifest_file_set():
            raise ValueError("Ender24 source manifest file set differs")
        expected_external = {
            "numerai/v5.3/ender21_discovery_full_through_0861.parquet",
            "numerai/v5.3/ender21_discovery_benchmark_models_through_0861.parquet",
        }
        if not isinstance(external, dict) or set(external) != expected_external:
            raise ValueError("Ender24 external artifact set differs")
        for relative, expected in sorted(files.items()):
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"Malformed source digest: {relative}")
            lease = self._hold(self.repo_dir / relative, f"source {relative}")
            if lease.sha256() != expected:
                raise ValueError(f"Ender24 source hash drifted: {relative}")
        for relative, expected in sorted(external.items()):
            if not isinstance(expected, dict) or set(expected) != {
                "size_bytes",
                "sha256",
                "last_era",
            }:
                raise ValueError(f"Malformed external receipt: {relative}")
            lease = self._hold(self.repo_dir / relative, f"artifact {relative}")
            if (
                lease.size_bytes() != expected["size_bytes"]
                or lease.sha256() != expected["sha256"]
                or expected["last_era"] != "0861"
            ):
                raise ValueError(f"Ender24 external artifact drifted: {relative}")
        for name in ROUND1_NAMES:
            for kind, relative in (
                ("prediction", f"predictions/{name}.parquet"),
                ("result", f"results/{name}.json"),
                ("completion", f"receipts/{name}.completion.json"),
            ):
                self._hold(experiment / relative, f"{name} {kind}")
        self.manifest = manifest
        return self

    def __exit__(self, *_args) -> None:
        for lease in reversed(tuple(self.leases.values())):
            lease.close()
        self.leases.clear()

    def lease(self, path: Path) -> HeldReadLease:
        canonical = Path(os.path.abspath(path))
        try:
            return self.leases[canonical]
        except KeyError as error:
            raise ValueError(f"Governed path has no held lease: {canonical}") from error

    def read_bytes(self, path: Path) -> bytes:
        return self.lease(path).read_bytes()

    def read_json(self, path: Path) -> object:
        try:
            return json.loads(self.read_bytes(path).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Governed JSON is malformed: {path}") from error

    def receipt(self, path: Path) -> dict:
        lease = self.lease(path)
        return {
            "path": Path(path).as_posix(),
            "size_bytes": lease.size_bytes(),
            "sha256": lease.sha256(),
        }

    def load_config(self, path: Path) -> dict:
        source = self.read_bytes(path)
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as error:
            raise ValueError(f"Frozen config syntax is invalid: {path}") from error
        if len(tree.body) != 4 or [ast.unparse(node) for node in tree.body[:3]] != [
            "from pathlib import Path",
            "import runpy",
            "variant = runpy.run_path(str(Path(__file__).with_name('base_r1.py')))['variant']",
        ]:
            raise ValueError(f"Frozen config loader envelope differs: {path}")
        assignment = tree.body[3]
        if (
            not isinstance(assignment, ast.Assign)
            or len(assignment.targets) != 1
            or not isinstance(assignment.targets[0], ast.Name)
            or assignment.targets[0].id != "CONFIG"
            or not isinstance(assignment.value, ast.Call)
            or not isinstance(assignment.value.func, ast.Name)
            or assignment.value.func.id != "variant"
            or any(keyword.arg is None for keyword in assignment.value.keywords)
        ):
            raise ValueError(f"Frozen config variant expression differs: {path}")
        try:
            arguments = [ast.literal_eval(value) for value in assignment.value.args]
            keywords = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in assignment.value.keywords
            }
        except (ValueError, TypeError) as error:
            raise ValueError(f"Frozen config arguments are not literals: {path}") from error
        base_path = path.with_name("base_r1.py")
        base_namespace: dict[str, object] = {
            "__file__": str(base_path),
            "__name__": "__ender24_frozen_base__",
        }
        base_source = self.read_bytes(base_path)
        exec(compile(base_source, str(base_path), "exec"), base_namespace)
        variant = base_namespace.get("variant")
        if not callable(variant):
            raise ValueError("Frozen base config does not define variant")
        return variant(*arguments, **keywords)


def sha256_file(path: Path, custody: EvaluationCustody | None = None) -> str:
    if custody is not None:
        return custody.lease(path).sha256()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def receipt(path: Path, custody: EvaluationCustody | None = None) -> dict:
    if custody is not None:
        return custody.receipt(path)
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def finite(value: object, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} is not finite")
    return parsed


def expected_procedure(name: str) -> dict:
    mapping = {
        "r1_control_seed1337": (1337, None),
        "r1_ema995_seed1337": (1337, 0.995),
        "r1_control_seed2027": (2027, None),
        "r1_ema995_seed2027": (2027, 0.995),
    }
    try:
        model_seed, ema_decay = mapping[name]
    except KeyError as error:
        raise ValueError(f"Unknown Ender24 Round-1 component {name}") from error
    return {"model_seed": model_seed, "sample_seed": 1337, "ema_decay": ema_decay}


def validate_config(
    experiment: Path,
    name: str,
    custody: EvaluationCustody | None = None,
) -> tuple[dict, dict]:
    path = experiment / f"configs/{name}.py"
    config = (
        custody.load_config(path)
        if custody is not None
        else runpy.run_path(str(path))["CONFIG"]
    )
    procedure = expected_procedure(name)
    required_data = {
        "data_version": "v5.3",
        "embargo_eras": 13,
        "era_col": ERA,
        "feature_set": "all",
        "feature_columns_path": (
            "numerai/agents/experiments/ender21_residual_stability_v53/"
            "protocol/feature_columns_all_v53.json"
        ),
        "target_col": TARGET,
        "id_col": ID,
        "full_data_path": "v5.3/ender21_discovery_full_through_0861.parquet",
        "benchmark_data_path": (
            "v5.3/ender21_discovery_benchmark_models_through_0861.parquet"
        ),
        "benchmark_model": BENCHMARK,
        "require_benchmark_coverage": True,
        "era_allowlist_path": (
            "numerai/agents/experiments/ender21_residual_stability_v53/"
            "protocol/discovery_eras_through_0861.json"
        ),
    }
    if config.get("data") != required_data:
        raise ValueError(f"{name} data contract differs")
    if config.get("output") != {
        "output_dir": f"experiments/{EXPERIMENT_NAME}",
        "results_name": name,
    }:
        raise ValueError(f"{name} output contract differs")
    if config.get("preprocessing") != {
        "missing_value": 2.0,
        "nan_missing_all_twos": False,
    }:
        raise ValueError(f"{name} preprocessing contract differs")
    model = config.get("model", {})
    if (
        model.get("type") != "TorchTabularRegressor"
        or model.get("x_groups") != ["features", "era", "benchmark_models"]
        or model.get("target_transform")
        != EXPECTED_SEMANTICS["training_target"]["transform"]
    ):
        raise ValueError(f"{name} model/target contract differs")
    required_params = {
        "architecture": "tabm",
        "activation": "relu",
        "tabm_arch_type": "tabm",
        "tabm_k": 64,
        "tabm_width": 512,
        "tabm_blocks": 3,
        "dropout": 0.1,
        "batch_size": 1024,
        "prediction_batch_size": 2048,
        "learning_rate": 0.002,
        "weight_decay": 0.0003,
        "max_epochs": 30,
        "patience": 4,
        "val_fraction": 0.1,
        "val_split": "recent_eras",
        "internal_val_embargo": 13,
        "feature_center": 2.0,
        "feature_scale": 2.0,
        "device": "cuda",
        "amp": True,
        "seed": procedure["model_seed"],
        "loss_mode": "chronological_block_dro",
        "chronological_blocks": 4,
        "dro_temperature": 2.0,
    }
    if procedure["ema_decay"] is not None:
        required_params["ema_decay"] = procedure["ema_decay"]
    if model.get("params") != required_params:
        raise ValueError(f"{name} model parameters differ")
    if config.get("training") != {
        "max_train_samples": 500_000,
        "sample_seed": 1337,
        "cv": {
            "embargo": 13,
            "enabled": True,
            "max_train_eras": 78,
            "min_train_size": 0,
            "mode": "expanding",
            "n_splits": 5,
        },
    }:
        raise ValueError(f"{name} training/CV contract differs")
    return config, procedure


def validate_completion(
    experiment: Path,
    name: str,
    manifest: dict,
    custody: EvaluationCustody | None = None,
) -> dict:
    path = experiment / f"receipts/{name}.completion.json"
    payload = (
        custody.read_json(path)
        if custody is not None
        else json.loads(path.read_text(encoding="utf-8"))
    )
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "stage",
        "state",
        "component",
        "manifest",
        "config",
        "outputs",
    }:
        raise ValueError(f"{name} completion schema differs")
    if (
        payload["schema_version"] != 1
        or payload["stage"] != "ender24-round1-training-completion"
        or payload["state"] != "OUTPUTS_FINALIZED"
        or payload["component"] != name
    ):
        raise ValueError(f"{name} completion envelope differs")
    manifest_path = experiment / "source_manifest_round1.json"
    expected_manifest = {
        "path": manifest_path.relative_to(REPO_DIR).as_posix(),
        "sha256": sha256_file(manifest_path, custody),
        "git_head": manifest["git_head"],
    }
    config_path = experiment / f"configs/{name}.py"
    config_relative = config_path.relative_to(REPO_DIR).as_posix()
    if payload["manifest"] != expected_manifest or payload["config"] != {
        "path": config_relative,
        "sha256": manifest["files"][config_relative],
    }:
        raise ValueError(f"{name} completion provenance differs")
    if not isinstance(payload["outputs"], dict) or set(payload["outputs"]) != {
        "predictions",
        "result",
    }:
        raise ValueError(f"{name} completion output set differs")
    for label, artifact in (
        ("predictions", experiment / f"predictions/{name}.parquet"),
        ("result", experiment / f"results/{name}.json"),
    ):
        stored = payload["outputs"][label]
        stat_result = (
            os.fstat(custody.lease(artifact).stream.fileno())
            if custody
            else artifact.lstat()
        )
        if not isinstance(stored, dict) or set(stored) != {
            "path",
            "device",
            "inode",
            "size_bytes",
            "sha256",
        }:
            raise ValueError(f"{name} completion {label} schema differs")
        if stored != {
            "path": str(artifact),
            "device": int(stat_result.st_dev),
            "inode": int(stat_result.st_ino),
            "size_bytes": int(stat_result.st_size),
            "sha256": sha256_file(artifact, custody),
        }:
            raise ValueError(f"{name} completion {label} identity differs")
    return payload


def preflight_all_completions(
    experiment: Path,
    manifest: dict,
    custody: EvaluationCustody,
) -> dict[str, dict]:
    """Validate all opaque completions before any config/result is parsed."""

    return {
        name: validate_completion(experiment, name, manifest, custody)
        for name in ROUND1_NAMES
    }


def load_authority(
    experiment: Path,
    numerai_dir: Path,
    custody: EvaluationCustody | None = None,
) -> tuple[dict, list[str]]:
    authority_path = experiment / "protocol/discovery_data_authority.json"
    authority = (
        custody.read_json(authority_path)
        if custody is not None
        else json.loads(authority_path.read_text(encoding="utf-8"))
    )
    if (
        not isinstance(authority, dict)
        or authority.get("schema_version") != 1
        or authority.get("authority") != "ender24-discovery-only"
        or authority.get("forbidden_historical_confirmation")
        != {
            "first_era": "0865",
            "last_era": "1021",
            "rule": "never read, score, select, tune, or report in Ender24",
        }
        or authority.get("prospective_confirmation")
        != {
            "first_era": "1231",
            "last_era": "1282",
            "era_count": 52,
            "rule": "future resolved eras only; no local historical substitute",
        }
    ):
        raise ValueError("Ender24 discovery authority envelope differs from freeze")
    expected = {
        "full": (
            numerai_dir / "v5.3/ender21_discovery_full_through_0861.parquet",
            1_302_848_771,
            "476d561ba8515a0066e892c5489a5ae1db6443587e9d5d06a9a6280400a701b9",
        ),
        "benchmark": (
            numerai_dir
            / "v5.3/ender21_discovery_benchmark_models_through_0861.parquet",
            23_325_224,
            "c2db9a77811390e9b9c47926b62fbfb7a6c7af24bb9c4db63137798e61b955b6",
        ),
        "era_allowlist": (
            REPO_DIR
            / "numerai/agents/experiments/ender21_residual_stability_v53/"
            "protocol/discovery_eras_through_0861.json",
            1_763,
            "be0c212a8e910f56dbdae4e1e134fa36ce7e5e1a95e43faa1ccc9e6330f544ca",
        ),
        "feature_columns": (
            REPO_DIR
            / "numerai/agents/experiments/ender21_residual_stability_v53/"
            "protocol/feature_columns_all_v53.json",
            148_179,
            "663184191e17d2fa4fac6dae017890f0e762368e638d46cfaa489297b9b2049b",
        ),
    }
    for label, (path, size, digest) in expected.items():
        section = authority.get(label)
        if not isinstance(section, dict):
            raise ValueError(f"Missing authority section {label}")
        actual_size = custody.lease(path).size_bytes() if custody else path.stat().st_size
        if actual_size != size or sha256_file(path, custody) != digest:
            raise ValueError(f"Physical authority differs for {label}")
        if section.get("size_bytes") != size or section.get("sha256") != digest:
            raise ValueError(f"Recorded authority differs for {label}")
    allowlist_path = expected["era_allowlist"][0]
    allowed = (
        custody.read_json(allowlist_path)
        if custody is not None
        else json.loads(allowlist_path.read_text(encoding="utf-8"))
    )
    expected_allowed = [f"{era:04d}" for era in range(161, 862, 4)]
    if allowed != expected_allowed or len(set(allowed)) != 176:
        raise ValueError("Discovery era allowlist differs from freeze")
    features_path = expected["feature_columns"][0]
    features = (
        custody.read_json(features_path)
        if custody is not None
        else json.loads(features_path.read_text(encoding="utf-8"))
    )
    if not isinstance(features, list) or len(features) != 3_555 or len(set(features)) != 3_555:
        raise ValueError("Feature authority differs from exact 3,555-column freeze")
    return authority, allowed


def load_truth(
    numerai_dir: Path,
    allowed: list[str],
    custody: EvaluationCustody | None = None,
):
    allowed_set = set(allowed)
    full_path = numerai_dir / "v5.3/ender21_discovery_full_through_0861.parquet"
    benchmark_path = (
        numerai_dir / "v5.3/ender21_discovery_benchmark_models_through_0861.parquet"
    )
    full_source = custody.lease(full_path).stream if custody else full_path
    benchmark_source = custody.lease(benchmark_path).stream if custody else benchmark_path
    full = pd.read_parquet(full_source, columns=[ID, ERA, TARGET])
    benchmark = pd.read_parquet(benchmark_source, columns=[ID, ERA, BENCHMARK])
    full[ERA] = full[ERA].astype(str)
    benchmark[ERA] = benchmark[ERA].astype(str)
    if max(map(int, full[ERA])) > 861 or max(map(int, benchmark[ERA])) > 861:
        raise ValueError("A discovery physical input contains a forbidden later era")
    truth = full.loc[full[ERA].isin(allowed_set)].merge(
        benchmark.loc[benchmark[ERA].isin(allowed_set)],
        on=[ID, ERA],
        how="inner",
        validate="one_to_one",
    )
    if truth.empty or truth[ID].isna().any() or truth[ID].duplicated().any():
        raise ValueError("Frozen discovery truth has invalid IDs")
    if set(truth[ERA]) != allowed_set:
        raise ValueError("Frozen discovery truth does not cover every allowed era")
    return truth


def expected_fold_geometry(allowed: list[str], truth) -> tuple[dict, list[dict], list[str]]:
    era_to_fold: dict[str, int] = {}
    folds: list[dict] = []
    oof_ids: list[str] = []
    for fold, (available_train, val_eras) in enumerate(
        era_cv_splits(
            allowed, n_splits=5, embargo=13, mode="expanding", min_train_size=0
        )
    ):
        if not available_train:
            continue
        train_eras = available_train[-78:]
        era_to_fold.update({str(era): fold for era in val_eras})
        oof_ids.extend(truth.loc[truth[ERA].isin(val_eras), ID].tolist())
        folds.append(
            {
                "fold": fold,
                "train_eras": len(train_eras),
                "val_eras": len(val_eras),
                "train_rows": min(int(truth[ERA].isin(train_eras).sum()), 500_000),
                "val_rows": int(truth[ERA].isin(val_eras).sum()),
                "available_train_eras": len(available_train),
                "max_train_eras": 78,
                "first_train_era": str(train_eras[0]),
                "last_train_era": str(train_eras[-1]),
            }
        )
    return era_to_fold, folds, oof_ids


def compute_metrics(joined, label: str = "prediction") -> dict:
    bmc = numerai_metrics.per_era_bmc(joined, [label], BENCHMARK, TARGET)[label]
    corr = numerai_metrics.per_era_corr(joined, [label], TARGET)[label]
    similarity = numerai_metrics.per_era_pred_corr(joined, [label], BENCHMARK)[label]
    bmc.index = bmc.index.astype(str)
    corr.index = corr.index.astype(str)
    similarity.index = similarity.index.astype(str)
    full = numerai_metrics.score_summary(bmc)
    recent = bmc.loc[[f"{era:04d}" for era in range(705, 862, 4)]]
    fold_bmc = {}
    for fold, subset in joined.groupby("cv_fold", sort=True):
        scores = numerai_metrics.per_era_bmc(
            subset, [label], BENCHMARK, TARGET
        )[label]
        fold_bmc[str(int(fold))] = finite(
            scores.mean(), f"{label} fold {fold} BMC"
        )
    blocks = {
        block: finite(bmc.loc[list(eras)].mean(), f"{label} block {block} BMC")
        for block, eras in RECENT_BLOCKS.items()
    }
    return {
        "bmc": {
            key: finite(value, f"{label} full BMC {key}")
            for key, value in full.items()
        },
        "recent40_bmc_mean": finite(recent.mean(), f"{label} recent40 BMC"),
        "recent_blocks_bmc_mean": blocks,
        "fold_bmc_mean": fold_bmc,
        "corr_mean": finite(corr.mean(), f"{label} Corr"),
        "avg_corr_with_benchmark": finite(
            similarity.mean(), f"{label} benchmark Corr"
        ),
    }


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_model_diagnostics(
    diagnostics: object,
    *,
    name: str,
    is_ema: bool,
) -> int:
    expected = BASE_DIAGNOSTIC_KEYS | (EMA_DIAGNOSTIC_KEYS if is_ema else set())
    if not isinstance(diagnostics, dict) or set(diagnostics) != expected:
        raise ValueError(f"{name} model diagnostics schema differs")
    if type(diagnostics["best_epoch"]) is not int or type(diagnostics["epochs_ran"]) is not int:
        raise ValueError(f"{name} epoch diagnostics differ")
    if not 1 <= diagnostics["best_epoch"] <= diagnostics["epochs_ran"] <= 30:
        raise ValueError(f"{name} epoch diagnostics differ")
    if type(diagnostics["n_parameters"]) is not int or diagnostics["n_parameters"] <= 0:
        raise ValueError(f"{name} parameter diagnostics differ")
    finite(diagnostics["best_val_loss"], f"{name} best_val_loss")
    finite(diagnostics["final_train_loss"], f"{name} final_train_loss")
    if is_ema:
        if diagnostics["ema_decay"] != 0.995:
            raise ValueError(f"{name} EMA decay differs")
        if type(diagnostics["ema_updates"]) is not int or diagnostics["ema_updates"] <= 0:
            raise ValueError(f"{name} EMA updates are inactive")
        for key in (
            "ema_live_state_sha256",
            "ema_shadow_state_sha256",
            "ema_inference_state_sha256",
        ):
            if not _valid_sha256(diagnostics[key]):
                raise ValueError(f"{name} {key} is malformed")
        if (
            diagnostics["ema_live_state_sha256"]
            == diagnostics["ema_shadow_state_sha256"]
        ):
            raise ValueError(f"{name} terminal live and EMA shadow hashes match")
    return diagnostics["n_parameters"]


def validate_stored_result_schema(
    stored: dict,
    config: dict,
    experiment: Path,
    name: str,
    joined,
) -> None:
    if not isinstance(stored, dict) or set(stored) != {
        "model",
        "preprocessing",
        "data",
        "benchmark",
        "output",
        "metrics",
        "cv",
        "training",
    }:
        raise ValueError(f"{name} stored result top-level schema differs")
    if stored.get("model") != config["model"] or stored.get("preprocessing") != config["preprocessing"]:
        raise ValueError(f"{name} stored model/preprocessing differs")
    output = stored.get("output")
    expected_output = {
        "output_dir": str(Path(os.path.abspath(experiment))),
        "predictions_file": str(Path("predictions") / f"{name}.parquet"),
        "prediction_semantics": EXPECTED_SEMANTICS,
    }
    if output != expected_output:
        raise ValueError(f"{name} stored output contract differs")
    metrics = stored.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {
        "corr",
        "bmc",
        "bmc_last_200_eras",
    }:
        raise ValueError(f"{name} stored metrics schema differs")
    bmc = numerai_metrics.per_era_bmc(
        joined, ["prediction"], BENCHMARK, TARGET
    )["prediction"]
    corr = numerai_metrics.per_era_corr(joined, ["prediction"], TARGET)["prediction"]
    similarity = numerai_metrics.per_era_pred_corr(
        joined, ["prediction"], BENCHMARK
    )["prediction"]
    expected_sections = {
        "corr": numerai_metrics.score_summary(corr),
        "bmc": {
            **numerai_metrics.score_summary(bmc),
            "avg_corr_with_benchmark": finite(similarity.mean(), "benchmark Corr"),
        },
        "bmc_last_200_eras": {
            **numerai_metrics.score_summary(bmc.iloc[-200:]),
            "avg_corr_with_benchmark": finite(
                similarity.iloc[-200:].mean(), "last-200 benchmark Corr"
            ),
        },
    }
    for section, expected in expected_sections.items():
        actual = metrics.get(section)
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{name} stored metrics.{section} schema differs")
        for field, value in expected.items():
            recorded = finite(actual[field], f"stored {name} {section}.{field}")
            if not np.isclose(recorded, finite(value, field), rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"{name} stored metrics.{section}.{field} does not recompute"
                )


def score_candidate(
    experiment: Path,
    name: str,
    allowed: list[str],
    truth,
    completion: dict,
    custody: EvaluationCustody | None = None,
) -> tuple[dict, object]:
    config, procedure = validate_config(experiment, name, custody)
    prediction_path = experiment / f"predictions/{name}.parquet"
    result_path = experiment / f"results/{name}.json"
    parquet = pq.ParquetFile(
        custody.lease(prediction_path).stream if custody else prediction_path
    )
    semantics_raw = (parquet.schema_arrow.metadata or {}).get(
        PREDICTION_SEMANTICS_METADATA_KEY
    )
    if semantics_raw is None or json.loads(semantics_raw) != EXPECTED_SEMANTICS:
        raise ValueError(f"{name} prediction semantics differ")
    frame = parquet.read().to_pandas()
    if list(frame.columns) != [ID, ERA, TARGET, "prediction", "cv_fold"]:
        raise ValueError(f"{name} prediction columns differ")
    if frame.empty or frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError(f"{name} prediction IDs are invalid")
    frame[ERA] = frame[ERA].astype(str)
    if (
        min(frame[ERA], key=int) != "0301"
        or max(frame[ERA], key=int) != "0861"
        or set(frame[ERA]) - set(allowed)
    ):
        raise ValueError(f"{name} prediction cohort differs from exact OOF 0301-0861")
    for column in (TARGET, "prediction", "cv_fold"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype="float64"
        )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} has non-finite {column}")
    folds_raw = frame["cv_fold"].to_numpy(dtype="float64")
    if not np.array_equal(folds_raw, folds_raw.astype("int64")):
        raise ValueError(f"{name} has fractional folds")
    joined = frame.merge(
        truth,
        on=[ID, ERA],
        how="left",
        validate="one_to_one",
        suffixes=("", "_truth"),
    )
    if joined[[f"{TARGET}_truth", BENCHMARK]].isna().any().any() or not np.array_equal(
        joined[TARGET].to_numpy(dtype="float64"),
        joined[f"{TARGET}_truth"].to_numpy(dtype="float64"),
        equal_nan=True,
    ):
        raise ValueError(f"{name} target/benchmark alignment differs")
    joined = joined.drop(columns=[f"{TARGET}_truth"])
    era_to_fold, expected_folds, expected_ids = expected_fold_geometry(allowed, truth)
    if len(frame) != len(expected_ids) or set(frame[ID]) != set(expected_ids):
        raise ValueError(f"{name} OOF IDs do not exactly cover frozen folds")
    if not np.array_equal(
        frame["cv_fold"].to_numpy(dtype="int64"),
        frame[ERA].map(era_to_fold).to_numpy(dtype="int64"),
    ):
        raise ValueError(f"{name} fold assignment differs")
    stored = (
        custody.read_json(result_path)
        if custody
        else json.loads(result_path.read_text(encoding="utf-8"))
    )
    expected_training = {
        "data_sampling": {"max_train_samples": 500_000, "sample_seed": 1337},
        "data_mode": "eager",
        "cv": config["training"]["cv"],
    }
    if stored.get("training") != expected_training:
        raise ValueError(f"{name} stored training differs")
    cv_header = {
        "n_splits": 5,
        "embargo": 13,
        "mode": "expanding",
        "min_train_size": 0,
        "folds_used": 4,
        "max_train_eras": 78,
    }
    stored_cv = stored.get("cv", {})
    if not isinstance(stored_cv, dict) or set(stored_cv) != {*cv_header, "folds"}:
        raise ValueError(f"{name} stored CV schema differs")
    if any(stored_cv.get(key) != value for key, value in cv_header.items()):
        raise ValueError(f"{name} stored CV header differs")
    if not isinstance(stored_cv.get("folds"), list) or len(stored_cv["folds"]) != 4:
        raise ValueError(f"{name} stored CV folds differ")
    parameter_count = None
    for actual, expected in zip(stored_cv["folds"], expected_folds):
        if set(actual) != {*expected, "model_diagnostics"} or any(
            actual.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"{name} stored CV geometry differs")
        count = validate_model_diagnostics(
            actual["model_diagnostics"],
            name=name,
            is_ema=procedure["ema_decay"] is not None,
        )
        if parameter_count is None:
            parameter_count = count
        elif count != parameter_count:
            raise ValueError(f"{name} parameter count differs across folds")
    allowlist_path = REPO_DIR / config["data"]["era_allowlist_path"]
    expected_data = {
        "data_version": "v5.3",
        "feature_set": "all",
        "target": TARGET,
        "full_data_path": config["data"]["full_data_path"],
        "full_rows": len(truth),
        "full_eras": 176,
        "oof_rows": len(expected_ids),
        "oof_eras": 141,
        "embargo_eras": 13,
        "require_benchmark_coverage": True,
        "data_mode": "eager",
        "era_allowlist": {
            "path": config["data"]["era_allowlist_path"],
            "sha256": sha256_file(allowlist_path, custody),
            "size_bytes": (
                custody.lease(allowlist_path).size_bytes()
                if custody
                else allowlist_path.stat().st_size
            ),
            "era_count": 176,
            "first_era": "0161",
            "last_era": "0861",
        },
    }
    if stored.get("data") != expected_data or stored.get("benchmark") != {
        "model": BENCHMARK,
        "file": config["data"]["benchmark_data_path"],
    }:
        raise ValueError(f"{name} stored data/benchmark contract differs")
    validate_stored_result_schema(stored, config, experiment, name, joined)
    metrics = compute_metrics(joined)
    for section, field, actual in (
        ("bmc", "mean", metrics["bmc"]["mean"]),
        ("bmc", "sharpe", metrics["bmc"]["sharpe"]),
        ("bmc", "max_drawdown", metrics["bmc"]["max_drawdown"]),
        ("corr", "mean", metrics["corr_mean"]),
    ):
        expected = finite(
            stored["metrics"][section][field],
            f"stored {name} {section}.{field}",
        )
        if not np.isclose(expected, actual, rtol=0.0, atol=1e-12):
            raise ValueError(f"{name} stored {section}.{field} does not recompute")
    return (
        {
            "artifacts": {
                "config": receipt(experiment / f"configs/{name}.py", custody),
                "prediction": receipt(prediction_path, custody),
                "result": receipt(result_path, custody),
                "completion": receipt(
                    experiment / f"receipts/{name}.completion.json", custody
                ),
            },
            "provenance": completion,
            "procedure": procedure,
            "rows": len(joined),
            "eras": int(joined[ERA].nunique()),
            "first_era": min(joined[ERA], key=int),
            "last_era": max(joined[ERA], key=int),
            "parameter_count": parameter_count,
            "metrics": metrics,
        },
        joined,
    )


def cohort_sha256(frame) -> str:
    columns = [ID, ERA, TARGET, BENCHMARK, "cv_fold"]
    ordered = frame.loc[:, columns].sort_values(ID, kind="mergesort")
    digest = hashlib.sha256()
    for row in ordered.itertuples(index=False, name=None):
        encoded = json.dumps(
            [str(row[0]), str(row[1]), float(row[2]), float(row[3]), int(row[4])],
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def per_ema_checks(metrics: dict, control: dict) -> dict:
    blocks = list(metrics["recent_blocks_bmc_mean"].values())
    return {
        "full_bmc_positive": metrics["bmc"]["mean"] > 0.0,
        "all_used_folds_bmc_positive": all(
            value > 0.0 for value in metrics["fold_bmc_mean"].values()
        ),
        "three_of_four_recent_blocks_positive": sum(
            value > 0.0 for value in blocks
        )
        >= 3,
        "worst_recent_block_above_minus_0_001": min(blocks) > -0.001,
        "corr_at_least_0_005": metrics["corr_mean"] >= 0.005,
        "corr_below_0_04": metrics["corr_mean"] < 0.04,
        "benchmark_corr_below_0_25": metrics["avg_corr_with_benchmark"] < 0.25,
        "sharpe_not_below_matched_control_minus_0_05": (
            metrics["bmc"]["sharpe"] >= control["bmc"]["sharpe"] - 0.05
        ),
        "drawdown_no_greater_than_matched_control_plus_0_01": (
            metrics["bmc"]["max_drawdown"]
            <= control["bmc"]["max_drawdown"] + 0.01
        ),
    }


def procedure_aggregates(records: dict[str, dict], names: tuple[str, str]) -> dict:
    full = [finite(records[name]["metrics"]["bmc"]["mean"], name) for name in names]
    recent = [
        finite(records[name]["metrics"]["recent40_bmc_mean"], name) for name in names
    ]
    return {
        "full_bmc": {
            "mean": sum(full) / 2.0,
            "worst_seed": min(full),
            "absolute_two_seed_gap": abs(full[0] - full[1]),
        },
        "recent40_bmc": {
            "mean": sum(recent) / 2.0,
            "worst_seed": min(recent),
            "absolute_two_seed_gap": abs(recent[0] - recent[1]),
        },
    }


def aggregate_checks(control: dict, ema: dict) -> dict:
    return {
        "ema_mean_recent40_bmc_at_least_control": (
            ema["recent40_bmc"]["mean"] >= control["recent40_bmc"]["mean"]
        ),
        "ema_worst_seed_recent40_bmc_at_least_control": (
            ema["recent40_bmc"]["worst_seed"]
            >= control["recent40_bmc"]["worst_seed"]
        ),
        "ema_mean_full_bmc_retains_95pct_control": (
            ema["full_bmc"]["mean"] >= 0.95 * control["full_bmc"]["mean"]
        ),
        "ema_worst_seed_full_bmc_retains_95pct_control": (
            ema["full_bmc"]["worst_seed"]
            >= 0.95 * control["full_bmc"]["worst_seed"]
        ),
        "ema_full_bmc_two_seed_gap_at_most_75pct_control": (
            ema["full_bmc"]["absolute_two_seed_gap"]
            <= 0.75 * control["full_bmc"]["absolute_two_seed_gap"]
        ),
        "ema_recent40_bmc_two_seed_gap_at_most_75pct_control": (
            ema["recent40_bmc"]["absolute_two_seed_gap"]
            <= 0.75 * control["recent40_bmc"]["absolute_two_seed_gap"]
        ),
    }
