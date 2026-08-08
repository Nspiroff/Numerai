from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from agents.code.analysis import evaluate_ender20_aux_target_rank_ensemble as aux


def _repo_root() -> Path:
    return Path(aux.__file__).resolve().parents[4]


def _synthetic_consumption_file_receipt(
    component: str,
    *,
    confirmation: bool,
) -> dict[str, object]:
    family = "confirmation" if confirmation else "scout"
    return {
        "path": (
            f"numerai/agents/experiments/{aux.EXPERIMENT_NAME}/receipts/"
            f".{family}-train-{component}.consumed.json"
        ),
        "sha256": hashlib.sha256(
            f"{family}:{component}".encode("utf-8")
        ).hexdigest(),
        "size_bytes": 1,
    }


def _synthetic_completion_file_receipt(
    component: str,
    *,
    confirmation: bool,
) -> dict[str, object]:
    family = "confirmation" if confirmation else "scout"
    digest = hashlib.sha256(
        f"{family}:{component}:completed".encode("utf-8")
    ).hexdigest()
    return {
        "path": (
            f"numerai/agents/experiments/{aux.EXPERIMENT_NAME}/receipts/"
            f"{family}-train-{component}-completion-{digest}.json"
        ),
        "sha256": digest,
        "size_bytes": 1,
    }


def _manifest() -> dict:
    path = (
        _repo_root()
        / "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53/"
        "source_manifest.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(
    *,
    bmc_mean: float = 0.003,
    bmc_sharpe: float = 0.40,
    bmc_drawdown: float = 0.05,
    corr_mean: float = 0.02,
    ender20_similarity: float = 0.50,
    ender60_similarity: float = 0.50,
    tabm_similarity: float = 0.50,
) -> dict:
    return {
        "era_count": 10,
        "corr": {
            "mean": corr_mean,
            "std": 0.01,
            "sharpe": corr_mean / 0.01,
            "max_drawdown": 0.02,
        },
        "bmc": {
            "mean": bmc_mean,
            "std": 0.01,
            "sharpe": bmc_sharpe,
            "max_drawdown": bmc_drawdown,
        },
        "avg_ender20_similarity": ender20_similarity,
        "avg_ender60_similarity": ender60_similarity,
        "avg_tabm_similarity": tabm_similarity,
    }


def _write_prediction_artifact(
    path: Path,
    frame: pd.DataFrame,
    semantics: dict | None,
) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    if semantics is not None:
        metadata[aux.PREDICTION_SEMANTICS_METADATA_KEY] = json.dumps(
            semantics, sort_keys=True
        ).encode("utf-8")
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path)


def _synthetic_protocol(
    root: Path,
    *,
    canonical_experiment: bool = False,
) -> aux.FrozenProtocol:
    experiment = (
        root
        / "numerai/agents/experiments"
        / aux.EXPERIMENT_NAME
        if canonical_experiment
        else root / "experiment"
    )
    experiment.mkdir(parents=True, exist_ok=True)
    return aux.FrozenProtocol(
        repo_root=root,
        experiment_dir=experiment,
        source_manifest_path=root / "source_manifest.json",
        source_manifest={},
        scout_configs={},
        scout_config_paths={},
        medium_features=("feature_1",),
        pretraining_commit="1" * 40,
        gpu_runtime_path=root / "gpu_runtime.json",
        gpu_runtime_receipt={},
    )


def _tracking_read_only_lease_type(
    active_paths: set[Path],
) -> type:
    original_lease = aux._ReadOnlyFileLease

    class TrackingLease:
        def __init__(self, path: Path, label: str) -> None:
            self.path = Path(aux.os.path.abspath(path))
            self.inner = original_lease(path, label)
            if self.path in active_paths:
                raise AssertionError(f"DUPLICATE_LEASE:{self.path}")
            active_paths.add(self.path)

        def fileno(self) -> int:
            self._assert_open()
            return self.inner.fileno()

        def read_bytes(self) -> bytes:
            self._assert_open()
            return self.inner.read_bytes()

        def sha256(self) -> str:
            self._assert_open()
            return self.inner.sha256()

        def _assert_open(self) -> None:
            if self.path not in active_paths:
                raise AssertionError(f"LEASE_CLOSED_EARLY:{self.path}")

        def close(self) -> None:
            active_paths.discard(self.path)
            self.inner.close()

    return TrackingLease


def _stage_expected() -> aux.ExpectedCohort:
    calibration_eras = tuple(f"{index:04d}" for index in range(1, 164)) + (
        aux.SCOUT_LAST_CALIBRATION_ERA,
    )
    locked_eras = (
        aux.SCOUT_FIRST_LOCKED_ERA,
        *(f"{index:04d}" for index in range(1030, 1078)),
        aux.SCOUT_LAST_ERA,
    )
    eras = calibration_eras + locked_eras
    frame = pd.DataFrame(
        {
            aux.ID_COLUMN: [f"id-{index}" for index in range(len(eras))],
            aux.ERA_COLUMN: eras,
            aux.ENDER_TARGET: np.linspace(0.0, 1.0, len(eras)),
            aux.BENCHMARK_ENDER20: np.linspace(1.0, 0.0, len(eras)),
            aux.BENCHMARK_ENDER60: np.linspace(0.5, 0.9, len(eras)),
            aux.FOLD_COLUMN: np.repeat([1, 2], [164, 50]),
        }
    )
    folds = (
        {"fold": 1, "train_eras": 10, "val_eras": 164, "train_rows": 10, "val_rows": 164},
        {"fold": 2, "train_eras": 174, "val_eras": 50, "train_rows": 174, "val_rows": 50},
    )
    return aux.ExpectedCohort(
        frame=frame,
        full_rows=len(frame),
        full_eras=len(eras),
        eras=eras,
        folds=folds,
    )


def _confirmation_stage_expected() -> aux.ExpectedCohort:
    calibration_eras = tuple(f"{index:04d}" for index in range(371, 1026))
    locked_eras = tuple(f"{index:04d}" for index in range(1026, 1226))
    eras = calibration_eras + locked_eras
    frame = pd.DataFrame(
        {
            aux.ID_COLUMN: [f"confirmation-id-{index}" for index in range(len(eras))],
            aux.ERA_COLUMN: eras,
            aux.ENDER_TARGET: np.linspace(0.0, 1.0, len(eras)),
            aux.BENCHMARK_ENDER20: np.linspace(1.0, 0.0, len(eras)),
            aux.BENCHMARK_ENDER60: np.linspace(0.25, 0.75, len(eras)),
            aux.FOLD_COLUMN: np.repeat([1, 2, 3, 4], [214, 214, 214, 213]),
        }
    )
    folds = tuple(
        {
            "fold": fold,
            "train_eras": fold * 100,
            "val_eras": count,
            "train_rows": fold * 100,
            "val_rows": count,
        }
        for fold, count in enumerate((214, 214, 214, 213), start=1)
    )
    return aux.ExpectedCohort(
        frame=frame,
        full_rows=len(frame),
        full_eras=len(eras),
        eras=eras,
        folds=folds,
    )


def _stage_scoring_frame(expected: aux.ExpectedCohort) -> pd.DataFrame:
    frame = expected.frame.copy()
    position = np.arange(len(frame), dtype=float)
    denominator = max(len(frame) - 1, 1)
    for offset, component in enumerate(aux.ALL_COMPONENTS, start=1):
        frame[component] = ((position + offset) % len(frame)) / denominator
    frame["tabm_two_seed_residual"] = position / denominator
    return frame


def _mock_per_era_metrics(
    eras: tuple[str, ...],
    signals: tuple[str, ...] | list[str],
) -> dict[str, pd.DataFrame]:
    values = {
        "corr": 0.02,
        "bmc": 0.003,
        "ender20_similarity": 0.50,
        "ender60_similarity": 0.50,
        "tabm_similarity": 0.50,
    }
    return {
        metric: pd.DataFrame(
            {signal: [value] * len(eras) for signal in signals},
            index=list(eras),
        )
        for metric, value in values.items()
    }


def _serialized_synthetic_metrics(
    eras: tuple[str, ...],
    signals: tuple[str, ...] | list[str],
) -> dict[str, dict[str, list[list[object]]]]:
    values = {
        "corr": (0.015, 0.025),
        "bmc": (0.0025, 0.0035),
        "ender20_similarity": (0.50, 0.50),
        "ender60_similarity": (0.50, 0.50),
        "tabm_similarity": (0.50, 0.50),
    }
    return {
        metric: {
            signal: [
                [era, pair[index % 2]] for index, era in enumerate(eras)
            ]
            for signal in signals
        }
        for metric, pair in values.items()
    }


def _summary_from_serialized_metrics(
    per_era: dict[str, dict[str, list[list[object]]]],
    signal: str,
    eras: tuple[str, ...],
) -> dict[str, object]:
    frames = {
        metric: pd.DataFrame(
            {
                name: [float(row[1]) for row in rows]
                for name, rows in signals.items()
            },
            index=list(eras),
        )
        for metric, signals in per_era.items()
    }
    return aux.summarize_signal(frames, signal)


def _mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_mapping_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            keys.update(_mapping_keys(child))
    return keys


def _git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(arguments)} failed ({result.returncode}): {result.stderr}"
        )
    return result


def _init_git_repo(root: Path) -> None:
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "synthetic@example.invalid")
    _git(root, "config", "user.name", "Synthetic Test")


def _commit_all(root: Path, message: str) -> str:
    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "-m", message)
    return _git(root, "rev-parse", "HEAD").stdout.strip()


def _regular_stat(*, reparse: bool = False, links: int = 1) -> object:
    return type(
        "SyntheticFileStat",
        (),
        {
            "st_mode": aux.stat.S_IFREG,
            "st_nlink": links,
            "st_file_attributes": (
                getattr(aux.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                if reparse
                else 0
            ),
        },
    )()


def _directory_stat(*, reparse: bool = False) -> object:
    return type(
        "SyntheticDirectoryStat",
        (),
        {
            "st_mode": aux.stat.S_IFDIR,
            "st_nlink": 1,
            "st_file_attributes": (
                getattr(aux.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                if reparse
                else 0
            ),
        },
    )()


def _confirmation_result_provenance_fixture(root: Path) -> dict[str, object]:
    config_path = (
        _repo_root()
        / "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53/"
        "configs/r1_jasper_d8_t6000.py"
    )
    config = copy.deepcopy(runpy.run_path(str(config_path))["CONFIG"])
    config["data"].pop("full_data_path")
    config["data"].pop("benchmark_data_path")
    config["data"]["disk_feature_store_path"] = (
        aux._confirmation_store_relative("jasper")
    )
    config["data"]["disk_feature_store_inventory_path"] = (
        aux.CONFIRMATION_STORE_INVENTORY_PATH.removeprefix("numerai/")
    )
    config["data"]["embargo_eras"] = 52
    config["training"]["cv"]["embargo"] = 52
    config["training"]["data_mode"] = "disk_feature_store"
    config["output"]["results_name"] = "confirmation_jasper_d8_t6000"

    target = aux.COMPONENT_TARGETS["jasper"]
    frame = pd.DataFrame(
        {
            aux.ID_COLUMN: ["a", "b", "c", "d"],
            aux.ERA_COLUMN: ["0001", "0001", "0002", "0002"],
            target: [0.1, 0.2, 0.3, 0.4],
            aux.FOLD_COLUMN: [1, 1, 2, 2],
        }
    )
    folds = (
        {
            "fold": 1,
            "train_eras": 1,
            "val_eras": 1,
            "train_rows": 2,
            "val_rows": 2,
        },
        {
            "fold": 2,
            "train_eras": 2,
            "val_eras": 1,
            "train_rows": 4,
            "val_rows": 2,
        },
    )
    expected = aux.ExpectedCohort(
        frame=frame,
        full_rows=6,
        full_eras=3,
        eras=("0001", "0002"),
        folds=folds,
    )
    result_name = config["output"]["results_name"]
    component = aux.ComponentPaths(
        name="jasper",
        config=root / "configs" / "confirmation_jasper_d8_t6000.py",
        result=root / "results" / f"{result_name}.json",
        predictions=root / "predictions" / f"{result_name}.parquet",
    )
    store_metadata = {
        "generation_id": "1" * 32,
        "row_count": 6,
        "feature_count": 780,
        "feature_order_sha256": "2" * 64,
    }
    store_receipt = {
        "metadata": {
            "path": "stores/jasper/metadata.json",
            "sha256": "a" * 64,
            "size_bytes": 512,
        },
        "features": {
            "path": "stores/jasper/features.bin",
            "sha256": "b" * 64,
            "size_bytes": 1024,
        },
        "manifest": {
            "path": "stores/jasper/manifest.parquet",
            "sha256": "c" * 64,
            "size_bytes": 256,
        },
    }
    store_inventory_receipt = {
        "path": aux.CONFIRMATION_STORE_INVENTORY_PATH,
        "git_blob_id": "d" * 40,
        "checkpoint_commit": "e" * 40,
    }
    diagnostics = {
        "directory": str(root / "stores" / "jasper"),
        "feature_path": str(root / "stores" / "jasper" / "features.bin"),
        "manifest_path": str(
            root / "stores" / "jasper" / "manifest.parquet"
        ),
        "generation_id": store_metadata["generation_id"],
        "row_count": store_metadata["row_count"],
        "feature_count": store_metadata["feature_count"],
        "feature_bytes": store_receipt["features"]["size_bytes"],
        "manifest_bytes": store_receipt["manifest"]["size_bytes"],
        "feature_order_sha256": store_metadata["feature_order_sha256"],
        "metadata_sha256": store_receipt["metadata"]["sha256"],
        "feature_sha256": store_receipt["features"]["sha256"],
        "manifest_sha256": store_receipt["manifest"]["sha256"],
        "committed_inventory": {
            "path": store_inventory_receipt["path"],
            "git_blob_id": store_inventory_receipt["git_blob_id"],
            "checkpoint_commit": store_inventory_receipt["checkpoint_commit"],
        },
    }
    result = {
        "model": aux._expected_model_payload(config),
        "preprocessing": {
            "nan_missing_all_twos": False,
            "missing_value": 2.0,
        },
        "data": {
            "data_version": "v5.3",
            "feature_set": "medium",
            "target": target,
            "full_rows": 6,
            "full_eras": 3,
            "oof_rows": 4,
            "oof_eras": 2,
            "embargo_eras": 52,
            "require_benchmark_coverage": True,
            "data_mode": "disk_feature_store",
            "disk_feature_store": diagnostics,
        },
        "benchmark": {"model": aux.BENCHMARK_ENDER20},
        "training": {
            "data_sampling": {"max_train_samples": 500_000, "sample_seed": 1337},
            "data_mode": "disk_feature_store",
            "cv": {
                "enabled": True,
                "n_splits": 5,
                "embargo": 52,
                "mode": "expanding",
                "min_train_size": 0,
            },
        },
        "cv": {
            "n_splits": 5,
            "embargo": 52,
            "mode": "expanding",
            "min_train_size": 0,
            "folds_used": len(folds),
            "folds": [
                {
                    **fold,
                    "model_diagnostics": {
                        "effective_device_type": "gpu",
                        "gpu_fallback_used": False,
                    },
                }
                for fold in folds
            ],
        },
        "output": {
            "predictions_file": str(component.predictions),
            "prediction_semantics": aux._expected_semantics(config),
        },
        "metrics": {"corr": {}, "bmc": {}, "bmc_last_200_eras": {}},
    }
    return {
        "config": config,
        "component": component,
        "expected": expected,
        "store_metadata": store_metadata,
        "store_receipt": store_receipt,
        "store_inventory_receipt": store_inventory_receipt,
        "result": result,
    }


class CheckpointAndManifestTests(unittest.TestCase):
    def test_protocol_and_implementation_paths_use_distinct_checkpoints(self) -> None:
        calls: list[list[str]] = []

        def fake_git(_root: Path, arguments: list[str]) -> subprocess.CompletedProcess:
            calls.append(list(arguments))
            return subprocess.CompletedProcess(["git", *arguments], 0, "", "")

        pretraining_commit = "1" * 40
        with patch.object(aux, "_run_git", side_effect=fake_git):
            aux.verify_checkpoint_boundaries(Path("repo"), pretraining_commit)

        diff_calls = [call for call in calls if call[:2] == ["diff", "--quiet"]]
        self.assertEqual(len(diff_calls), 2)
        protocol_diff = next(
            call for call in diff_calls if call[2] == aux.PRE_SCORING_COMMIT
        )
        implementation_diff = next(
            call for call in diff_calls if call[2] == pretraining_commit
        )
        self.assertTrue(set(aux.PROTOCOL_CHECKPOINT_PATHS).issubset(protocol_diff))
        self.assertTrue(
            set(aux.TRAINING_CHECKPOINT_PATHS).issubset(implementation_diff)
        )
        self.assertFalse(
            set(aux.PROTOCOL_CHECKPOINT_PATHS).intersection(
                aux.TRAINING_CHECKPOINT_PATHS
            )
        )

    def test_checkpoint_rejects_noncanonical_commit_or_dirty_paths(self) -> None:
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError, "full lowercase 40-character"
        ):
            aux.verify_checkpoint_boundaries(Path("repo"), "ABC")

        def dirty_git(_root: Path, arguments: list[str]) -> subprocess.CompletedProcess:
            stdout = " M frozen.py\n" if arguments[0] == "status" else ""
            return subprocess.CompletedProcess(["git", *arguments], 0, stdout, "")

        with patch.object(aux, "_run_git", side_effect=dirty_git):
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "uncommitted or untracked"
            ):
                aux.verify_checkpoint_boundaries(Path("repo"), "1" * 40)

    def test_checkpoint_covers_imported_executables_and_rejects_dirty_dependency(
        self,
    ) -> None:
        imported_executables = {
            "numerai/agents/code/analysis/evaluate_ender20_hybrid_stability.py",
            "numerai/agents/code/analysis/evaluate_xerxes20_lgbm_challenger.py",
        }
        self.assertTrue(
            imported_executables.issubset(aux.TRAINING_CHECKPOINT_PATHS),
            "Every imported analysis evaluator that can affect scoring must be "
            "bound to the pretraining checkpoint.",
        )

        dirty_dependency = next(iter(imported_executables))

        def dirty_git(_root: Path, arguments: list[str]) -> subprocess.CompletedProcess:
            is_training_status = (
                arguments[:3] == ["status", "--porcelain", "--untracked-files=all"]
                and dirty_dependency in arguments
            )
            stdout = f" M {dirty_dependency}\n" if is_training_status else ""
            return subprocess.CompletedProcess(["git", *arguments], 0, stdout, "")

        with patch.object(aux, "_run_git", side_effect=dirty_git):
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "uncommitted or untracked"
            ):
                aux.verify_checkpoint_boundaries(Path("repo"), "1" * 40)

    def test_leased_checkpoint_file_receipt_rejects_live_checkpoint_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority = root / "authority.py"
            authority.write_bytes(b"live authority\n")

            def git_show(payload: bytes) -> subprocess.CompletedProcess:
                return subprocess.CompletedProcess(
                    ["git", "show"],
                    0,
                    payload,
                    b"",
                )

            with patch.object(
                aux.subprocess,
                "run",
                return_value=git_show(b"checkpoint authority\n"),
            ):
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "authority checkpoint bytes differs",
                ):
                    aux._leased_checkpoint_file_receipt(
                        authority,
                        root,
                        "1" * 40,
                        "authority",
                    )

            with patch.object(
                aux.subprocess,
                "run",
                return_value=git_show(authority.read_bytes()),
            ):
                self.assertEqual(
                    aux._leased_checkpoint_file_receipt(
                        authority,
                        root,
                        "1" * 40,
                        "authority",
                    ),
                    aux._file_receipt(authority, root),
                )

    def test_manifest_pins_protocol_configs_sources_and_output_paths(self) -> None:
        root = _repo_root()
        experiment = (
            root
            / "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53"
        )
        manifest_path = experiment / "source_manifest.json"
        manifest = _manifest()

        self.assertEqual(aux._sha256_file(manifest_path), aux.SOURCE_MANIFEST_SHA256)
        self.assertEqual(
            manifest["experiment_files"]["gate.md"]["sha256"], aux.GATE_SHA256
        )
        for relative, receipt in manifest["experiment_files"].items():
            path = experiment / relative
            self.assertEqual(path.stat().st_size, receipt["size_bytes"])
            self.assertEqual(aux._sha256_file(path), receipt["sha256"])

        source_paths = {item["path"] for item in manifest["confirmation_sources"]}
        self.assertEqual(
            source_paths,
            {
                "numerai/v5.3/train.parquet",
                "numerai/v5.3/validation.parquet",
                "numerai/v5.3/train_benchmark_models.parquet",
                "numerai/v5.3/validation_benchmark_models.parquet",
            },
        )
        for receipt in manifest["confirmation_sources"]:
            self.assertRegex(receipt["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue((root / receipt["path"]).is_file())

        scout_outputs = manifest["new_scout_outputs"]
        self.assertEqual(set(scout_outputs), set(aux.SCOUT_NEW_COMPONENTS))
        self.assertTrue(
            all(item["must_be_absent_before_run"] for item in scout_outputs.values())
        )
        confirmation_outputs = manifest["confirmation_output_contract"]
        self.assertEqual(set(confirmation_outputs), set(aux.ALL_COMPONENTS))
        self.assertEqual(
            len(
                {
                    item[key]
                    for item in confirmation_outputs.values()
                    for key in ("predictions_path", "results_path")
                }
            ),
            10,
        )
        deployment = manifest["deployment_output_contract"]
        self.assertTrue(deployment["must_be_absent_before_packaging"])
        self.assertFalse(deployment["overwrite_or_rerun_allowed"])
        self.assertEqual(
            len(
                {
                    deployment["final_pickle_path"],
                    deployment["final_fit_receipt_path"],
                    deployment["docker_predictions_path"],
                    deployment["docker_receipt_path"],
                }
            ),
            4,
        )

    def test_protocol_uses_leased_manifest_bytes_and_cached_authority_receipts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = (
                root
                / "numerai/agents/experiments"
                / aux.EXPERIMENT_NAME
            )
            experiment.mkdir(parents=True)
            gate_path = experiment / "gate.md"
            gate_path.write_bytes(b"frozen gate")
            gpu_path = root / "gpu-runtime.json"
            gpu_path.write_bytes(b"frozen gpu runtime")
            feature_path = root / "features.json"
            feature_path.write_bytes(b"features")
            medium = ["feature_1"]
            config_dir = experiment / "configs"
            config_dir.mkdir()
            base_config_path = config_dir / "base_d8.py"
            base_config_path.write_bytes(b"BASE_CONFIG = True\n")
            scout_config_paths: dict[str, Path] = {}
            for name in aux.SCOUT_NEW_COMPONENTS:
                path = config_dir / f"r1_{name}_d8_t6000.py"
                path.write_bytes(f"CONFIG_NAME = {name!r}\n".encode("utf-8"))
                scout_config_paths[name] = path

            confirmation_sources: list[dict[str, object]] = []
            fingerprints: dict[Path, dict[str, object]] = {}
            for index in range(4):
                source_path = root / "data" / f"source-{index}.parquet"
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_bytes(f"source-{index}".encode("utf-8"))
                receipt: dict[str, object] = {
                    "path": source_path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    "size_bytes": source_path.stat().st_size,
                    "mtime_ns": index + 1,
                    "num_rows": index + 10,
                    "num_row_groups": 1,
                    "schema_sha256": str(index + 1) * 64,
                    "footer_sha256": str(index + 5) * 64,
                }
                confirmation_sources.append(receipt)
                fingerprints[source_path] = {
                    key: receipt[key]
                    for key in (
                        "size_bytes",
                        "mtime_ns",
                        "num_rows",
                        "num_row_groups",
                        "schema_sha256",
                        "footer_sha256",
                    )
                }

            expected_blends = {
                name: {
                    "tyler_weight": weights["tyler"],
                    "core_weight_each": weights["core"],
                }
                for name, weights in aux.BLEND_WEIGHTS.items()
            }
            gate_receipt = aux._file_receipt(gate_path, root)
            experiment_files = {
                "gate.md": gate_receipt,
                "configs/base_d8.py": aux._file_receipt(
                    base_config_path, root
                ),
                **{
                    f"configs/r1_{name}_d8_t6000.py": aux._file_receipt(
                        path, root
                    )
                    for name, path in scout_config_paths.items()
                },
            }
            xerxes_evaluator_receipt = {
                "path": "authority/xerxes-evaluator.py",
                "sha256": hashlib.sha256(b"Xerxes evaluator").hexdigest(),
                "size_bytes": len(b"Xerxes evaluator"),
            }
            manifest = {
                "schema_version": 1,
                "as_of_date": aux.AS_OF_DATE,
                "data_version": "v5.3",
                "component_targets": aux.COMPONENT_TARGETS,
                "blend_candidates": expected_blends,
                "experiment_files": experiment_files,
                "scout_sources": {},
                "confirmation_sources": confirmation_sources,
                "feature_metadata": {
                    "path": feature_path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
                    "size_bytes": feature_path.stat().st_size,
                    "medium_feature_count": len(medium),
                    "medium_feature_order_sha256": aux.feature_order_sha256(
                        medium
                    ),
                },
                "gpu_runtime": aux._file_receipt(gpu_path, root),
                "reused_xerxes_component": {
                    "evaluator": xerxes_evaluator_receipt
                },
            }
            manifest_path = experiment / "source_manifest.json"
            manifest_bytes = aux._receipt_bytes(manifest)
            manifest_path.write_bytes(manifest_bytes)
            forged_bytes = manifest_bytes.replace(
                aux.AS_OF_DATE.encode("utf-8"),
                b"9999-99-99",
                1,
            )
            self.assertEqual(len(forged_bytes), len(manifest_bytes))
            self.assertNotEqual(forged_bytes, manifest_bytes)

            mutation = {"attempted": False, "blocked": False}
            parsed_manifests: list[dict[str, object]] = []
            active_leases: set[Path] = set()
            original_lease = aux._ReadOnlyFileLease

            class MutationLease:
                def __init__(self, path: Path, label: str) -> None:
                    self.path = Path(aux.os.path.abspath(path))
                    self.inner = original_lease(path, label)
                    active_leases.add(self.path)

                def read_bytes(self) -> bytes:
                    value = self.inner.read_bytes()
                    if self.path == Path(aux.os.path.abspath(manifest_path)):
                        mutation["attempted"] = True
                        try:
                            manifest_path.write_bytes(forged_bytes)
                        except OSError:
                            mutation["blocked"] = True
                    return value

                def fileno(self) -> int:
                    return self.inner.fileno()

                def sha256(self) -> str:
                    return self.inner.sha256()

                def close(self) -> None:
                    active_leases.discard(self.path)
                    self.inner.close()

            original_json_loads = aux.json.loads

            def parse_leased_json(value: str, *args, **kwargs):
                parsed = original_json_loads(value, *args, **kwargs)
                if isinstance(parsed, dict) and "as_of_date" in parsed:
                    parsed_manifests.append(parsed)
                return parsed

            def inspect_manifest_receipts(
                _repo_root: Path,
                parsed: dict[str, object],
            ) -> None:
                self.assertTrue(mutation["attempted"])
                self.assertEqual(parsed["as_of_date"], aux.AS_OF_DATE)
                self.assertIsNot(parsed, manifest)

            authority_calls: list[str] = []

            expected_config_leases = {
                Path(aux.os.path.abspath(base_config_path)),
                *(
                    Path(aux.os.path.abspath(path))
                    for path in scout_config_paths.values()
                ),
            }
            evaluated_configs: list[Path] = []

            def evaluate_scout_config(path: Path, _label: str):
                absolute = Path(aux.os.path.abspath(path))
                self.assertEqual(active_leases, expected_config_leases)
                self.assertIn(absolute, expected_config_leases)
                evaluated_configs.append(absolute)
                return {"component": absolute.stem}

            def authority_receipt(
                _path: Path,
                _repo_root: Path,
                _checkpoint: str,
                label: str,
            ) -> dict[str, object]:
                authority_calls.append(label)
                if label == "frozen gate":
                    return {
                        key: gate_receipt[key]
                        for key in ("path", "sha256", "size_bytes")
                    }
                if label == "Xerxes evaluator":
                    return copy.deepcopy(xerxes_evaluator_receipt)
                payload = label.encode("utf-8")
                return {
                    "path": f"authority/{label.replace(' ', '-')}.bin",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }

            with patch.object(aux, "_repo_root", return_value=root), patch.object(
                aux, "verify_checkpoint_boundaries"
            ), patch.object(
                aux,
                "SOURCE_MANIFEST_SHA256",
                hashlib.sha256(manifest_bytes).hexdigest(),
            ), patch.object(
                aux, "GATE_SHA256", gate_receipt["sha256"]
            ), patch.object(
                aux, "_ReadOnlyFileLease", MutationLease
            ), patch.object(
                aux.json, "loads", side_effect=parse_leased_json
            ), patch.object(
                aux,
                "_walk_explicit_path_receipts",
                side_effect=inspect_manifest_receipts,
            ), patch.object(
                aux,
                "parquet_source_fingerprint",
                side_effect=lambda path: fingerprints[Path(path)],
            ), patch.object(
                aux,
                "_load_json",
                return_value={"feature_sets": {"medium": medium}},
            ), patch.object(
                aux, "_load_config", side_effect=evaluate_scout_config
            ), patch.object(
                aux, "validate_component_config"
            ), patch.object(
                aux.xerxes,
                "verify_live_gpu_runtime",
                return_value={"verified": True},
            ), patch.object(
                aux,
                "_leased_checkpoint_file_receipt",
                side_effect=authority_receipt,
            ):
                protocol = aux.verify_frozen_protocol(
                    manifest_path,
                    pretraining_commit="1" * 40,
                )

            self.assertTrue(mutation["attempted"])
            self.assertTrue(
                mutation["blocked"]
                or manifest_path.read_bytes() == forged_bytes
            )
            self.assertEqual(len(parsed_manifests), 1)
            self.assertEqual(parsed_manifests[0], manifest)
            self.assertEqual(active_leases, set())
            self.assertEqual(
                set(evaluated_configs),
                {
                    Path(aux.os.path.abspath(path))
                    for path in scout_config_paths.values()
                },
            )
            self.assertEqual(
                authority_calls,
                [
                    "frozen gate",
                    "frozen evaluator",
                    "hybrid evaluator",
                    "Xerxes evaluator",
                ],
            )
            cached = copy.deepcopy(protocol.authority_file_receipts)
            self.assertIsNotNone(cached)
            assert cached is not None
            self.assertEqual(cached["gate"], gate_receipt)
            self.assertEqual(
                cached["imported_evaluators"]["xerxes"],
                xerxes_evaluator_receipt,
            )

            manifest_path.write_bytes(forged_bytes)
            gate_path.write_bytes(b"mutated gate")
            with patch.object(
                aux,
                "_file_receipt",
                side_effect=AssertionError("LIVE_AUTHORITY_REOPENED"),
            ) as live_receipt, patch.object(
                aux,
                "_leased_checkpoint_file_receipt",
                side_effect=AssertionError("LIVE_AUTHORITY_REHASHED"),
            ) as leased_receipt:
                binding = aux._protocol_binding(protocol)
            live_receipt.assert_not_called()
            leased_receipt.assert_not_called()
            self.assertEqual(
                {
                    key: value
                    for key, value in binding.items()
                    if key not in {"pre_scoring_commit", "pretraining_commit"}
                },
                cached,
            )
            self.assertEqual(
                binding["source_manifest"]["sha256"],
                hashlib.sha256(manifest_bytes).hexdigest(),
            )

    def test_scout_configs_are_exact_target_only_variants_of_xerxes_depth8(self) -> None:
        root = _repo_root()
        experiment = (
            root
            / "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53"
        )
        xerxes_path = (
            root
            / "numerai/agents/experiments/xerxes20_lgbm_challenger_v53/"
            "configs/r1_depth8.py"
        )
        xerxes_config = runpy.run_path(str(xerxes_path))["CONFIG"]

        for component in aux.SCOUT_NEW_COMPONENTS:
            name = f"r1_{component}_d8_t6000"
            path = experiment / "configs" / f"{name}.py"
            actual = runpy.run_path(str(path))["CONFIG"]
            expected = copy.deepcopy(xerxes_config)
            expected["data"]["target_col"] = aux.COMPONENT_TARGETS[component]
            expected["output"]["output_dir"] = (
                "experiments/ender20_aux_target_rank_ensemble_v53"
            )
            expected["output"]["results_name"] = name
            self.assertEqual(actual, expected)
            aux.validate_component_config(component, actual)

            altered = copy.deepcopy(actual)
            altered["model"]["params"]["learning_rate"] = 0.004
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "LightGBM parameters differs"
            ):
                aux.validate_component_config(component, altered)

    def test_confirmation_config_allows_only_frozen_data_mode_changes(self) -> None:
        root = _repo_root()
        scout_path = (
            root
            / "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53/"
            "configs/r1_jasper_d8_t6000.py"
        )
        confirmation = copy.deepcopy(runpy.run_path(str(scout_path))["CONFIG"])
        confirmation["data"].pop("full_data_path")
        confirmation["data"].pop("benchmark_data_path")
        confirmation["data"]["disk_feature_store_path"] = (
            "v5.3/target_jasper_20_feature_store"
        )
        confirmation["data"]["disk_feature_store_inventory_path"] = (
            aux.CONFIRMATION_STORE_INVENTORY_PATH.removeprefix("numerai/")
        )
        confirmation["data"]["embargo_eras"] = 52
        confirmation["training"]["data_mode"] = "disk_feature_store"
        confirmation["training"]["cv"]["embargo"] = 52
        confirmation["output"]["results_name"] = "confirmation_jasper_d8_t6000"

        aux.validate_component_config("jasper", confirmation, confirmation=True)

        confirmation["training"]["sample_seed"] = 2027
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError, "training.sample_seed differs"
        ):
            aux.validate_component_config("jasper", confirmation, confirmation=True)

    def test_confirmation_config_rejects_unexpected_behavior_keys(self) -> None:
        root = _repo_root()
        scout_path = (
            root
            / "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53/"
            "configs/r1_jasper_d8_t6000.py"
        )
        confirmation = copy.deepcopy(runpy.run_path(str(scout_path))["CONFIG"])
        confirmation["data"].pop("full_data_path")
        confirmation["data"].pop("benchmark_data_path")
        confirmation["data"]["disk_feature_store_path"] = (
            "v5.3/target_jasper_20_feature_store"
        )
        confirmation["data"]["disk_feature_store_inventory_path"] = (
            aux.CONFIRMATION_STORE_INVENTORY_PATH.removeprefix("numerai/")
        )
        confirmation["data"]["embargo_eras"] = 52
        confirmation["training"]["data_mode"] = "disk_feature_store"
        confirmation["training"]["cv"]["embargo"] = 52
        confirmation["output"]["results_name"] = "confirmation_jasper_d8_t6000"

        unexpected_model = copy.deepcopy(confirmation)
        unexpected_model["model"]["prediction_batch_size"] = 1
        with self.assertRaises(aux.EnderEnsembleEvaluationError):
            aux.validate_component_config(
                "jasper", unexpected_model, confirmation=True
            )

        unexpected_top_level = copy.deepcopy(confirmation)
        unexpected_top_level["behavior_override"] = {"enabled": True}
        with self.assertRaises(aux.EnderEnsembleEvaluationError):
            aux.validate_component_config(
                "jasper", unexpected_top_level, confirmation=True
            )


class ReceiptPrimitiveTests(unittest.TestCase):
    def test_artifact_receipt_binds_repo_path_size_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifacts" / "component.bin"
            artifact.parent.mkdir()
            artifact.write_bytes(b"frozen component")
            receipt = {
                "path": "artifacts/component.bin",
                "size_bytes": artifact.stat().st_size,
                "sha256": aux._sha256_file(artifact),
            }

            self.assertEqual(
                aux._validate_path_receipt(root, receipt, "component"),
                artifact,
            )

            wrong_hash = {**receipt, "sha256": "0" * 64}
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "component hash differs"
            ):
                aux._validate_path_receipt(root, wrong_hash, "component")

            escaping = {**receipt, "path": "../component.bin"}
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "escapes"
            ):
                aux._validate_path_receipt(root, escaping, "component")

    def test_receipt_is_content_addressed_and_write_once(self) -> None:
        receipt = {
            "schema_version": 1,
            "experiment": aux.EXPERIMENT_NAME,
            "stage": "calibrate",
            "state": "PASS",
            "passed": True,
            "protocol": {"gate": aux.GATE_SHA256},
            "inputs": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = aux._write_content_addressed_receipt(output, "calibration", receipt)
            expected = hashlib.sha256(aux._receipt_bytes(receipt)).hexdigest()
            self.assertEqual(path.name, f"calibration-{expected}.json")
            self.assertEqual(aux._sha256_file(path), expected)
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "already exists"
            ):
                aux._write_content_addressed_receipt(output, "calibration", receipt)

            conflicting = copy.deepcopy(receipt)
            conflicting["state"] = "FAIL"
            conflicting["passed"] = False
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "already exists"
            ):
                aux._write_content_addressed_receipt(
                    output, "calibration", conflicting
                )

    def test_bound_receipt_requires_exact_hash_filename_stage_and_experiment(self) -> None:
        receipt = {
            "schema_version": 1,
            "experiment": aux.EXPERIMENT_NAME,
            "stage": "calibrate",
            "state": "PASS",
            "passed": True,
            "protocol": {},
            "inputs": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = aux._write_content_addressed_receipt(
                Path(directory), "calibrate", receipt
            )
            digest = aux._sha256_file(path)
            loaded = aux._load_bound_receipt(
                path, digest, expected_stage="calibrate"
            )
            self.assertEqual(loaded, receipt)
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "receipt hash differs"
            ):
                aux._load_bound_receipt(path, "0" * 64, expected_stage="calibrate")
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "receipt stage differs"
            ):
                aux._load_bound_receipt(path, digest, expected_stage="locked")

    def test_production_stage_loader_applies_closed_schema_before_protocol_access(
        self,
    ) -> None:
        receipt = {
            "schema_version": 1,
            "experiment": aux.EXPERIMENT_NAME,
            "stage": "locked",
            "state": "PASS",
            "passed": True,
            "protocol": {},
            "input_receipt": {
                "path": "receipts/calibrate.json",
                "sha256": "a" * 64,
            },
            "selected_formula": {
                "name": "tyler_w10",
                "weights": dict(aux.BLEND_WEIGHTS["tyler_w10"]),
            },
            "locked": {},
            "metrics": {"forbidden": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            protocol = _synthetic_protocol(Path(directory))
            with patch.object(
                aux,
                "_load_bound_receipt",
                return_value=receipt,
            ), patch.object(
                aux,
                "_validate_protocol_binding",
                side_effect=AssertionError("PROTOCOL_ACCESSED_BEFORE_SCHEMA"),
            ) as protocol_access:
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "keys differs|forbidden pre-scoring keys",
                ):
                    aux._load_passing_stage_receipt(
                        protocol,
                        Path(directory) / "locked.json",
                        "a" * 64,
                        stage="locked",
                    )
            protocol_access.assert_not_called()

    def test_claimed_receipt_requires_canonical_parent_prefix_and_finalization(
        self,
    ) -> None:
        receipt = {
            "schema_version": 1,
            "experiment": aux.EXPERIMENT_NAME,
            "stage": "calibrate",
            "state": "PASS",
            "passed": True,
            "protocol": {},
            "inputs": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "experiment" / "receipts"
            claim = aux._claim_receipt_prefix(canonical, "calibrate")

            self.assertEqual(claim.parent, canonical.resolve())
            self.assertEqual(claim.name, ".calibrate.claimed.json")
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "already.*claim|already exists"
            ):
                aux._claim_receipt_prefix(canonical, "calibrate")

            path = aux._write_claimed_content_addressed_receipt(
                canonical,
                "calibrate",
                claim,
                receipt,
            )
            digest = aux._sha256_file(path)
            loaded = aux._load_bound_receipt(
                path,
                digest,
                expected_stage="calibrate",
                receipt_dir=canonical,
                expected_prefix="calibrate",
            )
            self.assertEqual(loaded, receipt)

            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "prefix|filename"
            ):
                aux._load_bound_receipt(
                    path,
                    digest,
                    expected_stage="calibrate",
                    receipt_dir=canonical,
                    expected_prefix="locked",
                )

            incomplete_dir = root / "incomplete" / "receipts"
            aux._claim_receipt_prefix(incomplete_dir, "calibrate")
            payload = aux._receipt_bytes(receipt)
            incomplete_digest = hashlib.sha256(payload).hexdigest()
            incomplete = incomplete_dir / f"calibrate-{incomplete_digest}.json"
            incomplete.write_bytes(payload)
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "finaliz"
            ):
                aux._load_bound_receipt(
                    incomplete,
                    incomplete_digest,
                    expected_stage="calibrate",
                    receipt_dir=incomplete_dir,
                    expected_prefix="calibrate",
                )

            outside = root / "outside"
            outside_claim = aux._claim_receipt_prefix(outside, "calibrate")
            outside_path = aux._write_claimed_content_addressed_receipt(
                outside,
                "calibrate",
                outside_claim,
                receipt,
            )
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError,
                "receipt directory|canonical|receipt parent",
            ):
                aux._load_bound_receipt(
                    outside_path,
                    aux._sha256_file(outside_path),
                    expected_stage="calibrate",
                    receipt_dir=canonical,
                    expected_prefix="calibrate",
                )

    def test_bound_receipt_rejects_duplicate_and_hardlinked_receipt_markers(
        self,
    ) -> None:
        receipt = {
            "schema_version": 1,
            "experiment": aux.EXPERIMENT_NAME,
            "stage": "calibrate",
            "state": "PASS",
            "passed": True,
            "protocol": {},
            "inputs": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            receipt_dir = Path(directory) / "receipts"
            claim = aux._claim_receipt_prefix(receipt_dir, "calibrate")
            path = aux._write_claimed_content_addressed_receipt(
                receipt_dir, "calibrate", claim, receipt
            )
            digest = aux._sha256_file(path)
            duplicate = receipt_dir / f"calibrate-{'0' * 64}.json"
            duplicate.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError,
                "canonical receipt set|duplicate",
            ):
                aux._load_bound_receipt(
                    path,
                    digest,
                    expected_stage="calibrate",
                    receipt_dir=receipt_dir,
                    expected_prefix="calibrate",
                )

        for marker_name in ("receipt", "claim", "finalization"):
            with self.subTest(marker=marker_name), tempfile.TemporaryDirectory() as directory:
                receipt_dir = Path(directory) / "receipts"
                claim = aux._claim_receipt_prefix(receipt_dir, "calibrate")
                path = aux._write_claimed_content_addressed_receipt(
                    receipt_dir, "calibrate", claim, receipt
                )
                markers = {
                    "receipt": path,
                    "claim": claim,
                    "finalization": receipt_dir / ".calibrate.finalized.json",
                }
                marker = markers[marker_name]
                hardlink = receipt_dir / f"{marker_name}.hardlink"
                aux.os.link(marker, hardlink)
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "hardlink|link count|regular unlinked|st_nlink",
                ):
                    aux._load_bound_receipt(
                        path,
                        aux._sha256_file(path),
                        expected_stage="calibrate",
                        receipt_dir=receipt_dir,
                        expected_prefix="calibrate",
                    )

    def test_regular_receipt_marker_guards_reject_reparse_points(self) -> None:
        reparse_flag = getattr(aux.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        regular_stat = type(
            "ReparseFileStat",
            (),
            {
                "st_mode": aux.stat.S_IFREG,
                "st_nlink": 1,
                "st_file_attributes": reparse_flag,
            },
        )()
        directory_stat = type(
            "ReparseDirectoryStat",
            (),
            {
                "st_mode": aux.stat.S_IFDIR,
                "st_nlink": 1,
                "st_file_attributes": reparse_flag,
            },
        )()
        with patch.object(Path, "is_symlink", return_value=False), patch.object(
            Path, "lstat", return_value=regular_stat
        ):
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "reparse point"
            ):
                aux._require_regular_unlinked_file(
                    Path("receipt.json"), "receipt"
                )
        with patch.object(Path, "is_symlink", return_value=False), patch.object(
            Path, "lstat", return_value=directory_stat
        ):
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "reparse point"
            ):
                aux._require_regular_directory(Path("receipts"), "receipt dir")
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "symbolic link"
            ):
                aux._require_regular_unlinked_file(
                    Path("receipt.json"), "receipt"
                )
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "symbolic link"
            ):
                aux._require_regular_directory(Path("receipts"), "receipt dir")

    def test_seal_bindings_require_each_unique_known_component(self) -> None:
        bindings = aux._parse_seal_bindings(
            [
                [component, f"{component}.json", str(index) * 64]
                for index, component in enumerate(aux.SCOUT_NEW_COMPONENTS, start=1)
            ]
        )
        self.assertEqual(set(bindings), set(aux.SCOUT_NEW_COMPONENTS))
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError, "Duplicate seal component"
        ):
            aux._parse_seal_bindings(
                [["jasper", "one.json", "1" * 64], ["jasper", "two.json", "2" * 64]]
            )
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError, "Unknown seal component"
        ):
            aux._parse_seal_bindings([["xerxes", "x.json", "1" * 64]])

    def test_cli_requires_exactly_four_explicit_seal_bindings(self) -> None:
        args = aux.parse_args(
            [
                "calibrate",
                "--pretraining-commit",
                "1" * 40,
                *sum(
                    (
                        [
                            "--seal-receipt",
                            component,
                            f"{component}.json",
                            str(index) * 64,
                        ]
                        for index, component in enumerate(
                            aux.SCOUT_NEW_COMPONENTS, start=1
                        )
                    ),
                    [],
                ),
            ]
        )
        bindings = aux._parse_seal_bindings(args.seal_receipt)
        self.assertEqual(set(bindings), set(aux.SCOUT_NEW_COMPONENTS))


class ScoutStageGuardTests(unittest.TestCase):
    def test_scout_claim_rejects_file_directory_and_dangling_link_destinations(
        self,
    ) -> None:
        for destination_name in ("result", "predictions"):
            for destination_kind in ("file", "directory"):
                with self.subTest(
                    destination=destination_name, kind=destination_kind
                ), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    protocol = _synthetic_protocol(root)
                    component = aux.default_scout_component_paths(
                        protocol, "jasper"
                    )
                    protocol.source_manifest["new_scout_outputs"] = {
                        "jasper": {
                            "must_be_absent_before_run": True,
                            "results_path": aux._relative_path(
                                component.result, root
                            ),
                            "predictions_path": aux._relative_path(
                                component.predictions, root
                            ),
                        }
                    }
                    destination = getattr(component, destination_name)
                    if destination_kind == "file":
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(b"existing")
                    else:
                        destination.mkdir(parents=True)
                    with self.assertRaisesRegex(
                        aux.EnderEnsembleEvaluationError, "already exists"
                    ):
                        aux.claim_scout_component_run(
                            protocol,
                            "jasper",
                            protocol.experiment_dir / "receipts",
                        )

        real_lexists = aux.os.path.lexists
        for destination_name in ("result", "predictions"):
            with self.subTest(
                destination=destination_name, kind="dangling-link"
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                protocol = _synthetic_protocol(root)
                component = aux.default_scout_component_paths(protocol, "jasper")
                protocol.source_manifest["new_scout_outputs"] = {
                    "jasper": {
                        "must_be_absent_before_run": True,
                        "results_path": aux._relative_path(component.result, root),
                        "predictions_path": aux._relative_path(
                            component.predictions, root
                        ),
                    }
                }
                dangling_path = getattr(component, destination_name)

                def lexists(path: str | Path) -> bool:
                    return (
                        Path(path) == dangling_path
                        or real_lexists(path)
                    )

                with patch.object(aux.os.path, "lexists", side_effect=lexists):
                    with self.assertRaisesRegex(
                        aux.EnderEnsembleEvaluationError, "already exists"
                    ):
                        aux.claim_scout_component_run(
                            protocol,
                            "jasper",
                            protocol.experiment_dir / "receipts",
                        )

        dangling = Path("synthetic-dangling-destination")
        with patch.object(aux.os.path, "lexists", return_value=True):
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError,
                "already exists|dangling|reparse|link",
            ):
                aux._require_absent_destination(dangling, "synthetic")

    def test_existing_scout_seal_claim_refuses_before_pre_run_or_artifact_access(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            protocol = _synthetic_protocol(Path(directory))
            receipt_dir = protocol.experiment_dir / "receipts"
            component = aux.default_scout_component_paths(protocol, "jasper")
            aux._claim_receipt_prefix(receipt_dir, "scout-seal-jasper")
            with patch.object(
                aux,
                "_validate_scout_pre_run_absence_receipt",
                side_effect=AssertionError("PRE_RUN_ACCESSED"),
            ) as pre_run_access, patch.object(
                aux,
                "build_scout_expected_cohort",
                side_effect=AssertionError("ARTIFACT_ACCESSED"),
            ) as artifact_access:
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "already exists|immutable prefix",
                ):
                    aux.seal_scout_component(
                        protocol,
                        component,
                        receipt_dir,
                        receipt_dir / "pre-run.json",
                        "0" * 64,
                        receipt_dir / "completion.json",
                        "1" * 64,
                    )
            pre_run_access.assert_not_called()
            artifact_access.assert_not_called()

    def test_scout_pre_run_absence_proof_is_durable_bound_and_metric_free(
        self,
    ) -> None:
        expected = _stage_expected()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(root)
            protocol.source_manifest["confirmation_output_contract"] = {
                name: {
                    "config_path": f"configs/{name}.py",
                    "results_path": f"results/{name}.json",
                    "predictions_path": f"predictions/{name}.parquet",
                    "must_be_absent_before_run": True,
                }
                for name in aux.CONFIRMATION_RUN_ORDER
            }
            receipt_dir = protocol.experiment_dir / "receipts"
            component = aux.default_scout_component_paths(protocol, "jasper")
            protocol.source_manifest["new_scout_outputs"] = {
                "jasper": {
                    "must_be_absent_before_run": True,
                    "results_path": aux._relative_path(component.result, root),
                    "predictions_path": aux._relative_path(
                        component.predictions, root
                    ),
                }
            }
            protocol_binding = {"checkpoint": "mocked"}

            with patch.object(
                aux, "_protocol_binding", return_value=protocol_binding
            ):
                pre_run_path, pre_run = aux.claim_scout_component_run(
                    protocol,
                    "jasper",
                    receipt_dir,
                )

            self.assertEqual(pre_run["state"], "ABSENCE_PROVEN")
            self.assertEqual(
                set(pre_run["destinations"]), {"result", "predictions"}
            )
            self.assertTrue(
                all(
                    destination["absent"] is True
                    for destination in pre_run["destinations"].values()
                )
            )
            pre_run_digest = aux._sha256_file(pre_run_path)
            self.assertEqual(
                aux._load_bound_receipt(
                    pre_run_path,
                    pre_run_digest,
                    expected_stage="claim-scout-component-run",
                    receipt_dir=receipt_dir,
                    expected_prefix="scout-pre-run-jasper",
                ),
                pre_run,
            )

            for path in (component.predictions, component.result):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            def reservation_identity(path: Path) -> dict[str, object]:
                inspected = path.lstat()
                return {
                    "path": aux._lexical_relative_path(path, root),
                    "device": int(inspected.st_dev),
                    "inode": int(inspected.st_ino),
                }

            reservations = {
                "predictions": reservation_identity(component.predictions),
                "result": reservation_identity(component.result),
            }
            with patch.object(
                aux, "_protocol_binding", return_value=protocol_binding
            ), patch.object(aux, "_require_frozen_python_runtime"):
                consumption_path, consumption = (
                    aux.claim_component_training_consumption(
                        protocol,
                        component,
                        pre_run_path,
                        pre_run_digest,
                        reservations,
                        confirmation=False,
                    )
                )
                original_consumption = consumption_path.read_bytes()
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "already consumed",
                ):
                    aux.claim_component_training_consumption(
                        protocol,
                        component,
                        pre_run_path,
                        pre_run_digest,
                        reservations,
                        confirmation=False,
                    )
            self.assertEqual(consumption_path.read_bytes(), original_consumption)
            self.assertEqual(consumption["state"], "CONSUMED")
            self.assertEqual(consumption["output_reservations"], reservations)
            self.assertTrue(
                {
                    "metrics",
                    "corr",
                    "bmc",
                    "summary",
                    "per_era",
                }.isdisjoint(_mapping_keys(consumption))
            )

            with patch.object(
                aux, "_protocol_binding", return_value=protocol_binding
            ):
                completion_claim_path = (
                    aux.claim_component_training_completion(
                        protocol,
                        component,
                        confirmation=False,
                    )
                )

            for path, payload in (
                (component.config, b"CONFIG = {}\n"),
                (component.result, b"sealed-result"),
                (component.predictions, b"sealed-predictions"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)

            def completion_identity(path: Path) -> dict[str, object]:
                inspected = path.lstat()
                return {
                    "path": aux._lexical_relative_path(path, root),
                    "device": int(inspected.st_dev),
                    "inode": int(inspected.st_ino),
                    "size_bytes": int(inspected.st_size),
                    "sha256": aux._sha256_file(path),
                }

            completed_outputs = {
                "predictions": completion_identity(component.predictions),
                "result": completion_identity(component.result),
            }
            with patch.object(
                aux, "_protocol_binding", return_value=protocol_binding
            ), patch.object(aux, "_require_frozen_python_runtime"):
                completion_path, completion = (
                    aux.complete_component_training_consumption(
                        protocol,
                        component,
                        pre_run_path,
                        pre_run_digest,
                        completed_outputs,
                        completion_claim_path,
                        confirmation=False,
                    )
                )
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "already exists|immutable prefix",
                ):
                    aux.complete_component_training_consumption(
                        protocol,
                        component,
                        pre_run_path,
                        pre_run_digest,
                        completed_outputs,
                        completion_claim_path,
                        confirmation=False,
                    )
            completion_digest = aux._sha256_file(completion_path)
            completion_prefix = "scout-train-jasper-completion"
            completion_finalization_path = receipt_dir / (
                f".{completion_prefix}.finalized.json"
            )
            artifact = {
                "component": "jasper",
                "target": aux.COMPONENT_TARGETS["jasper"],
                "config": aux._file_receipt(component.config, root),
                "result": aux._file_receipt(component.result, root),
                "predictions": aux._file_receipt(component.predictions, root),
            }
            active_leases: set[Path] = set()
            original_lease = aux._ReadOnlyFileLease

            class TrackingLease:
                def __init__(self, path: Path, label: str) -> None:
                    self.path = Path(aux.os.path.abspath(path))
                    self.inner = original_lease(path, label)
                    active_leases.add(self.path)

                def read_bytes(self) -> bytes:
                    return self.inner.read_bytes()

                def fileno(self) -> int:
                    return self.inner.fileno()

                def sha256(self) -> str:
                    return self.inner.sha256()

                def close(self) -> None:
                    active_leases.discard(self.path)
                    self.inner.close()

            artifact_reads = 0

            def validate_artifact(*_args, **_kwargs):
                nonlocal artifact_reads
                artifact_reads += 1
                self.assertEqual(
                    active_leases,
                    {
                        Path(aux.os.path.abspath(completion_claim_path)),
                        Path(aux.os.path.abspath(completion_path)),
                        Path(aux.os.path.abspath(completion_finalization_path)),
                        Path(aux.os.path.abspath(consumption_path)),
                        Path(aux.os.path.abspath(component.predictions)),
                        Path(aux.os.path.abspath(component.result)),
                    },
                )
                return np.zeros(len(expected.frame)), artifact

            with patch.object(
                aux, "_protocol_binding", return_value=protocol_binding
            ), patch.object(
                aux, "build_scout_expected_cohort", return_value=expected
            ), patch.object(
                aux,
                "_validate_scout_component",
                side_effect=validate_artifact,
            ), patch.object(
                aux,
                "_ReadOnlyFileLease",
                TrackingLease,
            ):
                seal_path, seal = aux.seal_scout_component(
                    protocol,
                    component,
                    receipt_dir,
                    pre_run_path,
                    pre_run_digest,
                    completion_path,
                    completion_digest,
                )

            self.assertEqual(artifact_reads, 1)
            self.assertEqual(active_leases, set())
            self.assertTrue(seal_path.is_file())
            self.assertEqual(
                seal["pre_run_absence_receipt"],
                {
                    "path": aux._relative_path(pre_run_path, root),
                    "sha256": pre_run_digest,
                },
            )
            self.assertEqual(
                seal["run_consumption_claim"],
                aux._file_receipt(consumption_path, root),
            )
            self.assertEqual(
                seal["run_completion_receipt"],
                aux._file_receipt(completion_path, root),
            )
            self.assertTrue(
                {
                    "metrics",
                    "corr",
                    "bmc",
                    "summary",
                    "per_era",
                    "calibration",
                    "locked",
                }.isdisjoint(_mapping_keys(seal))
            )

            result_inode = component.result.lstat().st_ino
            with component.result.open("r+b") as stream:
                stream.seek(0)
                stream.write(b"X")
                stream.flush()
            self.assertEqual(component.result.lstat().st_ino, result_inode)
            with patch.object(
                aux,
                "_validate_scout_component",
                side_effect=AssertionError("ARTIFACT_READ"),
            ) as artifact_read, patch.object(
                aux,
                "_protocol_binding",
                return_value=protocol_binding,
            ):
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "completed hash differs|hash differs",
                ):
                    with aux._lease_validated_component_outputs(
                        protocol,
                        component,
                        pre_run_path,
                        pre_run_digest,
                        completion_path,
                        completion_digest,
                        confirmation=False,
                    ):
                        aux._validate_scout_component(
                            protocol,
                            component,
                            expected,
                        )
            artifact_read.assert_not_called()

    def test_completion_receipt_rejects_missing_tamper_and_misbindings_before_artifact(
        self,
    ) -> None:
        variants = {
            "missing": "missing|does not exist|No such file|regular file|cannot be inspected",
            "tamper": "hash differs|invalid JSON|malformed",
            "marker": "training consumption claim differs|claim hash differs|malformed",
            "hash": "completed hash differs|hash differs",
            "size": "completed result size differs|completed size differs|size differs",
            "path": "completed result path differs|path differs",
            "id": "completed result inode differs|inode differs",
        }
        for variant, expected_error in variants.items():
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                protocol = _synthetic_protocol(root)
                receipt_dir = protocol.experiment_dir / "receipts"
                component = aux.default_scout_component_paths(protocol, "jasper")
                protocol.source_manifest["new_scout_outputs"] = {
                    "jasper": {
                        "must_be_absent_before_run": True,
                        "results_path": aux._relative_path(component.result, root),
                        "predictions_path": aux._relative_path(
                            component.predictions, root
                        ),
                    }
                }
                protocol_binding = {"checkpoint": "mocked"}
                with patch.object(
                    aux, "_protocol_binding", return_value=protocol_binding
                ):
                    pre_run_path, _ = aux.claim_scout_component_run(
                        protocol,
                        "jasper",
                        receipt_dir,
                    )
                pre_run_digest = aux._sha256_file(pre_run_path)

                for path in (component.predictions, component.result):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()

                def reserved(path: Path) -> dict[str, object]:
                    inspected = path.lstat()
                    return {
                        "path": aux._lexical_relative_path(path, root),
                        "device": int(inspected.st_dev),
                        "inode": int(inspected.st_ino),
                    }

                reservations = {
                    "predictions": reserved(component.predictions),
                    "result": reserved(component.result),
                }
                with patch.object(
                    aux, "_protocol_binding", return_value=protocol_binding
                ), patch.object(aux, "_require_frozen_python_runtime"):
                    consumption_path, _ = (
                        aux.claim_component_training_consumption(
                            protocol,
                            component,
                            pre_run_path,
                            pre_run_digest,
                            reservations,
                            confirmation=False,
                        )
                    )
                    completion_claim_path = (
                        aux.claim_component_training_completion(
                            protocol,
                            component,
                            confirmation=False,
                        )
                    )

                    component.predictions.write_bytes(b"prediction-output")
                    component.result.write_bytes(b"result-output")

                    def completed(path: Path) -> dict[str, object]:
                        inspected = path.lstat()
                        return {
                            "path": aux._lexical_relative_path(path, root),
                            "device": int(inspected.st_dev),
                            "inode": int(inspected.st_ino),
                            "size_bytes": int(inspected.st_size),
                            "sha256": aux._sha256_file(path),
                        }

                    completion_path, _ = (
                        aux.complete_component_training_consumption(
                            protocol,
                            component,
                            pre_run_path,
                            pre_run_digest,
                            {
                                "predictions": completed(component.predictions),
                                "result": completed(component.result),
                            },
                            completion_claim_path,
                            confirmation=False,
                        )
                    )
                completion_digest = aux._sha256_file(completion_path)
                selected_path = completion_path
                selected_digest = completion_digest
                completion_prefix = "scout-train-jasper-completion"
                finalization_path = receipt_dir / (
                    f".{completion_prefix}.finalized.json"
                )

                if variant == "missing":
                    completion_path.rename(completion_path.with_suffix(".missing"))
                elif variant == "tamper":
                    completion_path.write_bytes(
                        completion_path.read_bytes() + b"FORGED"
                    )
                elif variant == "marker":
                    forged_marker = json.loads(
                        consumption_path.read_text(encoding="utf-8")
                    )
                    forged_marker["state"] = "FORGED"
                    consumption_path.write_bytes(aux._receipt_bytes(forged_marker))
                else:
                    forged_completion = json.loads(
                        completion_path.read_text(encoding="utf-8")
                    )
                    result = forged_completion["outputs"]["result"]
                    if variant == "hash":
                        result["sha256"] = "0" * 64
                    elif variant == "size":
                        result["size_bytes"] += 1
                    elif variant == "path":
                        result["path"] = forged_completion["outputs"][
                            "predictions"
                        ]["path"]
                    elif variant == "id":
                        result["inode"] += 1
                    forged_bytes = aux._receipt_bytes(forged_completion)
                    selected_digest = hashlib.sha256(forged_bytes).hexdigest()
                    completion_path.rename(
                        completion_path.with_suffix(".original")
                    )
                    selected_path = receipt_dir / (
                        f"{completion_prefix}-{selected_digest}.json"
                    )
                    selected_path.write_bytes(forged_bytes)
                    finalization = aux._finalization_payload(
                        completion_prefix,
                        completion_claim_path,
                        aux._sha256_file(completion_claim_path),
                        selected_path,
                        selected_digest,
                    )
                    finalization_path.write_bytes(
                        aux._receipt_bytes(finalization)
                    )

                with patch.object(
                    aux, "_protocol_binding", return_value=protocol_binding
                ), patch.object(
                    aux,
                    "_validate_scout_component",
                    side_effect=AssertionError("ARTIFACT_READ"),
                ) as artifact_read:
                    with self.assertRaisesRegex(
                        (aux.EnderEnsembleEvaluationError, FileNotFoundError),
                        expected_error,
                    ):
                        with aux._lease_validated_component_outputs(
                            protocol,
                            component,
                            pre_run_path,
                            pre_run_digest,
                            selected_path,
                            selected_digest,
                            confirmation=False,
                        ):
                            aux._validate_scout_component(
                                protocol,
                                component,
                                _stage_expected(),
                            )
                artifact_read.assert_not_called()

    def test_scout_and_confirmation_seals_finalize_while_output_leases_are_open(
        self,
    ) -> None:
        for confirmation in (False, True):
            with self.subTest(confirmation=confirmation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                protocol = _synthetic_protocol(root)
                receipt_dir = protocol.experiment_dir / "receipts"
                component_name = "jasper"
                if confirmation:
                    confirmation_component = aux.ComponentPaths(
                        name=component_name,
                        config=protocol.experiment_dir
                        / "configs/confirmation_jasper.py",
                        result=protocol.experiment_dir
                        / "results/confirmation_jasper.json",
                        predictions=protocol.experiment_dir
                        / "predictions/confirmation_jasper.parquet",
                    )
                    protocol.source_manifest[
                        "confirmation_output_contract"
                    ] = {
                        component_name: {
                            "config_path": aux._relative_path(
                                confirmation_component.config, root
                            ),
                            "results_path": aux._relative_path(
                                confirmation_component.result, root
                            ),
                            "predictions_path": aux._relative_path(
                                confirmation_component.predictions, root
                            ),
                        }
                    }
                    component = aux.default_confirmation_component_paths(
                        protocol, component_name
                    )
                    expected = _confirmation_stage_expected()
                else:
                    component = aux.default_scout_component_paths(
                        protocol, component_name
                    )
                    expected = _stage_expected()

                pre_run_path = receipt_dir / "pre-run.json"
                pre_run_digest = "1" * 64
                completion_path = receipt_dir / "completion.json"
                completion_digest = "2" * 64
                pretraining_path = receipt_dir / "pretraining.json"
                pretraining_digest = "3" * 64
                consumption_receipt = _synthetic_consumption_file_receipt(
                    component_name,
                    confirmation=confirmation,
                )
                completion_receipt = _synthetic_completion_file_receipt(
                    component_name,
                    confirmation=confirmation,
                )
                state = {"held": False, "finalizations": 0}

                @contextmanager
                def leased_outputs(*_args, **_kwargs):
                    self.assertFalse(state["held"])
                    state["held"] = True
                    try:
                        yield (
                            copy.deepcopy(consumption_receipt),
                            copy.deepcopy(completion_receipt),
                        )
                    finally:
                        state["held"] = False

                original_writer = aux._write_claimed_content_addressed_receipt

                def finalize_while_held(*args, **kwargs):
                    self.assertTrue(state["held"])
                    state["finalizations"] += 1
                    return original_writer(*args, **kwargs)

                artifact = {"component": component_name, "sealed": True}
                common_patches = (
                    patch.object(aux, "_protocol_binding", return_value={}),
                    patch.object(
                        aux,
                        "_lease_validated_component_outputs",
                        side_effect=leased_outputs,
                    ),
                    patch.object(
                        aux,
                        "_write_claimed_content_addressed_receipt",
                        side_effect=finalize_while_held,
                    ),
                )
                with common_patches[0], common_patches[1], common_patches[2]:
                    if confirmation:
                        with patch.object(
                            aux,
                            "_validate_confirmation_pretraining_receipt",
                            return_value={},
                        ), patch.object(
                            aux,
                            "_validate_confirmation_pre_run_absence_receipt",
                            return_value={"prior_finalized_seal": None},
                        ), patch.object(
                            aux,
                            "build_confirmation_expected_cohort",
                            return_value=expected,
                        ), patch.object(
                            aux,
                            "_validate_confirmation_component",
                            return_value=(
                                np.zeros(len(expected.frame)),
                                artifact,
                            ),
                        ):
                            seal_path, seal = aux.seal_confirmation_component(
                                protocol,
                                component,
                                pre_run_path,
                                pre_run_digest,
                                completion_path,
                                completion_digest,
                                pretraining_path,
                                pretraining_digest,
                                receipt_dir,
                            )
                    else:
                        with patch.object(
                            aux,
                            "_validate_scout_pre_run_absence_receipt",
                            return_value={"prior_finalized_seal": None},
                        ), patch.object(
                            aux,
                            "build_scout_expected_cohort",
                            return_value=expected,
                        ), patch.object(
                            aux,
                            "_validate_scout_component",
                            return_value=(
                                np.zeros(len(expected.frame)),
                                artifact,
                            ),
                        ):
                            seal_path, seal = aux.seal_scout_component(
                                protocol,
                                component,
                                receipt_dir,
                                pre_run_path,
                                pre_run_digest,
                                completion_path,
                                completion_digest,
                            )

                self.assertFalse(state["held"])
                self.assertEqual(state["finalizations"], 1)
                self.assertTrue(seal_path.is_file())
                self.assertEqual(
                    seal["run_completion_receipt"],
                    completion_receipt,
                )

    def test_scout_seal_validation_rejects_missing_tampered_and_wrong_preflight(
        self,
    ) -> None:
        original_load = aux._load_bound_receipt
        for variant, expected_error in (
            ("missing", "receipt keys differs|pre-run absence binding is malformed"),
            ("tampered", "receipt hash differs|receipt filename differs"),
            ("wrong", "filename differs|prefix|component"),
        ):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                protocol = _synthetic_protocol(root)
                receipt_dir = protocol.experiment_dir / "receipts"
                contracts: dict[str, dict[str, object]] = {}
                for component_name in ("jasper", "teager2b"):
                    component = aux.default_scout_component_paths(
                        protocol, component_name
                    )
                    contracts[component_name] = {
                        "must_be_absent_before_run": True,
                        "results_path": aux._relative_path(component.result, root),
                        "predictions_path": aux._relative_path(
                            component.predictions, root
                        ),
                    }
                protocol.source_manifest["new_scout_outputs"] = contracts

                pre_run_path: Path | None = None
                pre_run_digest: str | None = None
                if variant in {"tampered", "wrong"}:
                    proof_component = "teager2b" if variant == "wrong" else "jasper"
                    if proof_component == "teager2b":
                        prior_digest = "f" * 64
                        prior_path = receipt_dir / (
                            f"scout-seal-jasper-{prior_digest}.json"
                        )
                        with patch.object(
                            aux,
                            "_protocol_binding",
                            return_value={"checkpoint": "mocked"},
                        ), patch.object(
                            aux,
                            "_validate_prior_finalized_seal",
                            return_value={
                                "component": "jasper",
                                "path": aux._lexical_relative_path(
                                    prior_path, root
                                ),
                                "sha256": prior_digest,
                            },
                        ):
                            pre_run_path, _ = aux.claim_scout_component_run(
                                protocol,
                                proof_component,
                                receipt_dir,
                                prior_path,
                                prior_digest,
                            )
                    else:
                        with patch.object(
                            aux,
                            "_protocol_binding",
                            return_value={"checkpoint": "mocked"},
                        ):
                            pre_run_path, _ = aux.claim_scout_component_run(
                                protocol,
                                proof_component,
                                receipt_dir,
                            )
                    pre_run_digest = aux._sha256_file(pre_run_path)
                    if variant == "tampered":
                        pre_run_digest = "0" * 64

                seal: dict[str, object] = {
                    "schema_version": 1,
                    "experiment": aux.EXPERIMENT_NAME,
                    "stage": "seal-scout-component",
                    "state": "SEALED",
                    "passed": True,
                    "protocol": {"checkpoint": "mocked"},
                    "component": "jasper",
                    "prior_finalized_seal": None,
                    "run_consumption_claim": (
                        _synthetic_consumption_file_receipt(
                            "jasper",
                            confirmation=False,
                        )
                    ),
                    "run_completion_receipt": (
                        _synthetic_completion_file_receipt(
                            "jasper",
                            confirmation=False,
                        )
                    ),
                    "cohort": aux._cohort_receipt(_stage_expected()),
                    "artifact": {
                        "component": "jasper",
                        "target": aux.COMPONENT_TARGETS["jasper"],
                        "config": {
                            "path": "artifacts/config.py",
                            "sha256": "a" * 64,
                            "size_bytes": 1,
                        },
                        "result": {
                            "path": "artifacts/result.json",
                            "sha256": "b" * 64,
                            "size_bytes": 1,
                        },
                        "predictions": {
                            "path": "artifacts/predictions.parquet",
                            "sha256": "c" * 64,
                            "size_bytes": 1,
                        },
                    },
                    "gpu_folds_verified": len(_stage_expected().folds),
                }
                if pre_run_path is not None and pre_run_digest is not None:
                    seal["pre_run_absence_receipt"] = {
                        "path": aux._relative_path(pre_run_path, root),
                        "sha256": pre_run_digest,
                    }

                def load_receipt(
                    path: Path,
                    digest: str,
                    *,
                    expected_stage: str,
                    receipt_dir: Path | None = None,
                    expected_prefix: str | None = None,
                ) -> dict:
                    if expected_stage == "seal-scout-component":
                        return seal
                    return original_load(
                        path,
                        digest,
                        expected_stage=expected_stage,
                        receipt_dir=receipt_dir,
                        expected_prefix=expected_prefix,
                    )

                bindings = {
                    component: (
                        receipt_dir / f"{component}.json",
                        str(index) * 64,
                    )
                    for index, component in enumerate(
                        aux.SCOUT_NEW_COMPONENTS,
                        start=1,
                    )
                }
                with patch.object(
                    aux, "_load_bound_receipt", side_effect=load_receipt
                ), patch.object(aux, "_validate_protocol_binding"):
                    with self.assertRaisesRegex(
                        aux.EnderEnsembleEvaluationError, expected_error
                    ):
                        aux.validate_seal_receipts(
                            protocol,
                            bindings,
                            expected=_stage_expected(),
                        )

    def test_existing_stage_claim_refuses_before_scout_or_receipt_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            protocol = _synthetic_protocol(Path(directory))
            receipt_dir = protocol.experiment_dir / "receipts"

            with patch.object(
                aux,
                "build_scout_expected_cohort",
                side_effect=AssertionError("SCOUT_ACCESSED"),
            ) as noncanonical_access:
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "canonical receipt directory|receipt directory differs",
                ):
                    aux.run_calibrate(protocol, {}, protocol.repo_root / "receipts")
            noncanonical_access.assert_not_called()

            aux._claim_receipt_prefix(receipt_dir, "calibrate")
            with patch.object(
                aux,
                "build_scout_expected_cohort",
                side_effect=AssertionError("SCOUT_ACCESSED"),
            ) as scout_access:
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "already.*claim|already exists|immutable prefix",
                ):
                    aux.run_calibrate(protocol, {}, receipt_dir)
            scout_access.assert_not_called()

            aux._claim_receipt_prefix(receipt_dir, "locked")
            with patch.object(
                aux,
                "_load_passing_stage_receipt",
                side_effect=AssertionError("RECEIPT_ACCESSED"),
            ) as receipt_access:
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "already.*claim|already exists|immutable prefix",
                ):
                    aux.run_locked(
                        protocol,
                        receipt_dir / "calibrate-does-not-matter.json",
                        "0" * 64,
                        receipt_dir,
                    )
            receipt_access.assert_not_called()

    def test_calibrate_slices_before_all_candidates_and_persists_no_locked_data(
        self,
    ) -> None:
        expected = _stage_expected()
        full_frame = _stage_scoring_frame(expected)
        calibration_eras = expected.eras[: aux.SCOUT_CALIBRATION_ERAS]
        locked_eras = set(expected.eras[-aux.SCOUT_LOCKED_ERAS :])

        def build_all(frame: pd.DataFrame) -> pd.DataFrame:
            actual_eras = tuple(
                sorted(frame[aux.ERA_COLUMN].astype(str).unique(), key=int)
            )
            self.assertEqual(actual_eras, calibration_eras)
            self.assertTrue(locked_eras.isdisjoint(actual_eras))
            scored = frame.copy()
            for index, candidate in enumerate(aux.CANDIDATE_NAMES, start=1):
                scored[candidate] = np.linspace(
                    float(index), float(index + 1), len(scored)
                )
            return scored

        def compute(
            frame: pd.DataFrame,
            signals: tuple[str, ...],
            eras: tuple[str, ...],
            *,
            tabm_column: str,
        ) -> dict[str, pd.DataFrame]:
            self.assertEqual(tuple(signals), aux.CANDIDATE_NAMES)
            self.assertEqual(tuple(eras), calibration_eras)
            self.assertEqual(tabm_column, "tabm_two_seed_residual")
            self.assertTrue(
                locked_eras.isdisjoint(frame[aux.ERA_COLUMN].astype(str))
            )
            return _mock_per_era_metrics(calibration_eras, list(signals))

        with tempfile.TemporaryDirectory() as directory:
            protocol = _synthetic_protocol(Path(directory))
            receipt_dir = protocol.experiment_dir / "receipts"
            with patch.object(
                aux, "build_scout_expected_cohort", return_value=expected
            ), patch.object(
                aux,
                "_build_scout_scoring_frame",
                return_value=(full_frame, {"sealed_inputs": "mocked"}),
            ), patch.object(
                aux, "build_rank_blends", side_effect=build_all
            ) as all_builder, patch.object(
                aux, "compute_per_era_metrics", side_effect=compute
            ), patch.object(
                aux, "summarize_signal", return_value=_summary()
            ), patch.object(
                aux, "_protocol_binding", return_value={"checkpoint": "mocked"}
            ):
                _, receipt = aux.run_calibrate(protocol, {}, receipt_dir)

        all_builder.assert_called_once()
        self.assertEqual(receipt["calibration"]["eras"], 164)
        self.assertEqual(
            receipt["calibration"]["last_era"],
            aux.SCOUT_LAST_CALIBRATION_ERA,
        )
        self.assertNotIn("locked", _mapping_keys(receipt))
        serialized = json.dumps(receipt, sort_keys=True)
        self.assertTrue(all(era not in serialized for era in locked_eras))

    def test_forged_calibration_selection_fails_before_locked_slice(self) -> None:
        expected = _stage_expected()
        full_frame = _stage_scoring_frame(expected)
        calibration_eras = expected.eras[: aux.SCOUT_CALIBRATION_ERAS]
        locked_eras = expected.eras[-aux.SCOUT_LOCKED_ERAS :]
        inputs = {
            "seal_receipts": {
                component: {
                    "path": f"receipts/scout-seal-{component}.json",
                    "sha256": str(index) * 64,
                }
                for index, component in enumerate(
                    aux.SCOUT_NEW_COMPONENTS, start=1
                )
            }
        }

        def compute(
            _frame: pd.DataFrame,
            signals: tuple[str, ...] | list[str],
            eras: tuple[str, ...],
            *,
            tabm_column: str,
        ) -> dict[str, pd.DataFrame]:
            self.assertEqual(tuple(signals), aux.CANDIDATE_NAMES)
            self.assertEqual(tuple(eras), calibration_eras)
            self.assertEqual(tabm_column, "tabm_two_seed_residual")
            return _mock_per_era_metrics(calibration_eras, list(signals))

        real_slice = aux._slice_eras
        locked_opened = False

        def guarded_slice(
            frame: pd.DataFrame, eras: tuple[str, ...]
        ) -> pd.DataFrame:
            nonlocal locked_opened
            if tuple(eras) == locked_eras:
                locked_opened = True
                raise AssertionError("LOCKED_SCOUT_OPENED")
            return real_slice(frame, eras)

        with patch.object(
            aux, "compute_per_era_metrics", side_effect=compute
        ), patch.object(aux, "summarize_signal", return_value=_summary()):
            derived = aux._derive_scout_calibration(full_frame, expected)
        self.assertEqual(derived["selected_formula"]["name"], "tyler_w00")
        forged = {
            "schema_version": 1,
            "experiment": aux.EXPERIMENT_NAME,
            "stage": "calibrate",
            "protocol": {"checkpoint": "mocked"},
            "inputs": inputs,
            **copy.deepcopy(derived),
        }
        forged["selected_formula"] = {
            "name": "tyler_w40",
            "weights": dict(aux.BLEND_WEIGHTS["tyler_w40"]),
        }

        with tempfile.TemporaryDirectory() as directory:
            protocol = _synthetic_protocol(Path(directory))
            receipt_dir = protocol.experiment_dir / "receipts"
            with patch.object(
                aux, "_load_passing_stage_receipt", return_value=forged
            ), patch.object(
                aux, "build_scout_expected_cohort", return_value=expected
            ), patch.object(
                aux,
                "_build_scout_scoring_frame",
                return_value=(full_frame, inputs),
            ), patch.object(
                aux, "compute_per_era_metrics", side_effect=compute
            ), patch.object(
                aux, "summarize_signal", return_value=_summary()
            ), patch.object(
                aux, "_slice_eras", side_effect=guarded_slice
            ), patch.object(
                aux, "build_selected_rank_blend"
            ) as selected_builder:
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "recomputed calibration selected_formula differs",
                ):
                    aux.run_locked(
                        protocol,
                        receipt_dir / "calibrate-forged.json",
                        "a" * 64,
                        receipt_dir,
                    )

        self.assertFalse(locked_opened)
        selected_builder.assert_not_called()

    def test_locked_rederives_calibration_then_builds_only_selected_formula(
        self,
    ) -> None:
        expected = _stage_expected()
        full_frame = _stage_scoring_frame(expected)
        calibration_eras = expected.eras[: aux.SCOUT_CALIBRATION_ERAS]
        locked_eras = expected.eras[-aux.SCOUT_LOCKED_ERAS :]
        inputs = {
            "seal_receipts": {
                component: {
                    "path": f"receipts/scout-seal-{component}.json",
                    "sha256": str(index) * 64,
                }
                for index, component in enumerate(
                    aux.SCOUT_NEW_COMPONENTS, start=1
                )
            }
        }

        def build_all(frame: pd.DataFrame) -> pd.DataFrame:
            actual_eras = tuple(
                sorted(frame[aux.ERA_COLUMN].astype(str).unique(), key=int)
            )
            self.assertEqual(actual_eras, calibration_eras)
            scored = frame.copy()
            for index, candidate in enumerate(aux.CANDIDATE_NAMES, start=1):
                scored[candidate] = np.linspace(
                    float(index), float(index + 1), len(scored)
                )
            return scored

        def build_selected(frame: pd.DataFrame, selected: str) -> pd.DataFrame:
            actual_eras = tuple(
                sorted(frame[aux.ERA_COLUMN].astype(str).unique(), key=int)
            )
            self.assertEqual(actual_eras, locked_eras)
            self.assertEqual(selected, "tyler_w00")
            self.assertTrue(
                (set(aux.CANDIDATE_NAMES) - {selected}).isdisjoint(frame.columns)
            )
            scored = frame.copy()
            scored[selected] = np.linspace(0.0, 1.0, len(scored))
            return scored

        def compute(
            frame: pd.DataFrame,
            signals: tuple[str, ...] | list[str],
            eras: tuple[str, ...],
            *,
            tabm_column: str,
        ) -> dict[str, pd.DataFrame]:
            self.assertEqual(tabm_column, "tabm_two_seed_residual")
            if tuple(eras) == calibration_eras:
                self.assertEqual(tuple(signals), aux.CANDIDATE_NAMES)
                self.assertTrue(
                    set(aux.CANDIDATE_NAMES).issubset(frame.columns)
                )
            else:
                self.assertEqual(tuple(eras), locked_eras)
                self.assertEqual(list(signals), ["tyler_w00"])
                self.assertTrue(
                    (set(aux.CANDIDATE_NAMES) - {"tyler_w00"}).isdisjoint(
                        frame.columns
                    )
                )
            return _mock_per_era_metrics(tuple(eras), list(signals))

        with patch.object(
            aux, "build_rank_blends", side_effect=build_all
        ), patch.object(
            aux, "compute_per_era_metrics", side_effect=compute
        ), patch.object(aux, "summarize_signal", return_value=_summary()):
            derived = aux._derive_scout_calibration(full_frame, expected)
        calibration_receipt = {
            "schema_version": 1,
            "experiment": aux.EXPERIMENT_NAME,
            "stage": "calibrate",
            "protocol": {"checkpoint": "mocked"},
            "inputs": inputs,
            **derived,
        }

        with tempfile.TemporaryDirectory() as directory:
            protocol = _synthetic_protocol(Path(directory))
            receipt_dir = protocol.experiment_dir / "receipts"
            calibration_path = receipt_dir / "calibrate-valid.json"
            with patch.object(
                aux,
                "_load_passing_stage_receipt",
                return_value=calibration_receipt,
            ), patch.object(
                aux, "build_scout_expected_cohort", return_value=expected
            ), patch.object(
                aux,
                "_build_scout_scoring_frame",
                return_value=(full_frame, inputs),
            ), patch.object(
                aux, "build_rank_blends", side_effect=build_all
            ) as all_builder, patch.object(
                aux, "build_selected_rank_blend", side_effect=build_selected
            ) as selected_builder, patch.object(
                aux, "compute_per_era_metrics", side_effect=compute
            ), patch.object(
                aux, "summarize_signal", return_value=_summary()
            ), patch.object(
                aux, "_protocol_binding", return_value={"checkpoint": "mocked"}
            ):
                _, receipt = aux.run_locked(
                    protocol,
                    calibration_path,
                    "a" * 64,
                    receipt_dir,
                )

        all_builder.assert_called_once()
        selected_builder.assert_called_once()
        selected = receipt["selected_formula"]["name"]
        self.assertEqual(selected, "tyler_w00")
        self.assertNotIn("calibration", receipt)
        for metric in receipt["locked"]["per_era"].values():
            self.assertEqual(set(metric), {selected})
        self.assertTrue(
            all(
                candidate not in json.dumps(receipt["locked"], sort_keys=True)
                for candidate in set(aux.CANDIDATE_NAMES) - {selected}
            )
        )


class ConfirmationStageGuardTests(unittest.TestCase):
    def test_checkpoint_destination_absence_uses_successful_empty_ls_tree_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(root)
            protocol.source_manifest["confirmation_output_contract"] = {
                name: {
                    "config_path": f"configs/confirmation_{name}.py",
                    "results_path": f"results/confirmation_{name}.json",
                    "predictions_path": f"predictions/confirmation_{name}.parquet",
                    "must_be_absent_before_run": True,
                }
                for name in aux.ALL_COMPONENTS
            }

            calls: list[list[str]] = []

            def empty_tree(
                _root: Path, arguments: list[str]
            ) -> subprocess.CompletedProcess[str]:
                calls.append(list(arguments))
                return subprocess.CompletedProcess(arguments, 0, "", "")

            with patch.object(aux, "_run_git", side_effect=empty_tree):
                receipts = aux._confirmation_destination_receipts(
                    protocol,
                    require_live_absence=False,
                )
            self.assertEqual(set(receipts), set(aux.ALL_COMPONENTS))
            self.assertTrue(calls)
            self.assertTrue(all(call[0] == "ls-tree" for call in calls))

            for label, result, expected_error in (
                (
                    "present",
                    subprocess.CompletedProcess(
                        ["git", "ls-tree"],
                        0,
                        "100644 blob deadbeef\tresults/file.json\n",
                        "",
                    ),
                    "existed|not absent|non-empty|checkpoint absence",
                ),
                (
                    "fatal",
                    subprocess.CompletedProcess(
                        ["git", "ls-tree"],
                        128,
                        "",
                        "fatal: bad object",
                    ),
                    "failed|cannot|fatal|query",
                ),
            ):
                with self.subTest(label=label), patch.object(
                    aux,
                    "_run_git",
                    return_value=result,
                ):
                    with self.assertRaisesRegex(
                        aux.EnderEnsembleEvaluationError,
                        expected_error,
                    ):
                        aux._confirmation_destination_receipts(
                            protocol,
                            require_live_absence=False,
                        )

    def test_destination_and_store_checks_reject_reparse_points_in_any_ancestor(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(root)
            redirected = root / "safe" / "redirected"
            results = redirected / "results"
            predictions = redirected / "predictions"
            results.mkdir(parents=True)
            predictions.mkdir(parents=True)
            protocol.source_manifest["confirmation_output_contract"] = {
                name: {
                    "config_path": f"configs/confirmation_{name}.py",
                    "results_path": (
                        f"safe/redirected/results/confirmation_{name}.json"
                    ),
                    "predictions_path": (
                        f"safe/redirected/predictions/confirmation_{name}.parquet"
                    ),
                    "must_be_absent_before_run": True,
                }
                for name in aux.ALL_COMPONENTS
            }
            store_dir = redirected / "stores" / "jasper"
            store_dir.mkdir(parents=True)
            store_file = store_dir / "metadata.json"
            store_file.write_text("{}\n", encoding="utf-8")
            store_receipt = {
                "path": aux._relative_path(store_file, root),
                "size_bytes": store_file.stat().st_size,
                "sha256": aux._sha256_file(store_file),
            }
            real_lstat = Path.lstat

            def fake_lstat(path: Path) -> object:
                if Path(path) == redirected:
                    return _directory_stat(reparse=True)
                return real_lstat(path)

            with patch.object(
                aux,
                "_run_git",
                return_value=subprocess.CompletedProcess(
                    ["git", "ls-tree"], 0, "", ""
                ),
            ), patch.object(Path, "lstat", fake_lstat), patch.object(
                Path, "is_symlink", return_value=False
            ):
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "reparse point|ancestor",
                ):
                    aux._confirmation_destination_receipts(
                        protocol,
                        require_live_absence=True,
                    )
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "reparse point|ancestor",
                ):
                    aux._require_regular_unlinked_receipt_file(
                        protocol,
                        store_receipt,
                        "synthetic store",
                    )

    def test_scout_and_confirmation_post_run_outputs_require_regular_unlinked_files(
        self,
    ) -> None:
        expected = _stage_expected()
        confirmation_expected = _confirmation_stage_expected()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(root)
            scout = aux.ComponentPaths(
                name="jasper",
                config=root / "scout.py",
                result=root / "scout.json",
                predictions=root / "scout.parquet",
            )
            confirmation = aux.ComponentPaths(
                name="jasper",
                config=root / "confirmation.py",
                result=root / "confirmation.json",
                predictions=root / "confirmation.parquet",
            )
            for label, callback in (
                (
                    "Scout",
                    lambda: aux._validate_scout_component(
                        protocol,
                        scout,
                        expected,
                    ),
                ),
                (
                    "confirmation",
                    lambda: aux._validate_confirmation_component(
                        protocol,
                        confirmation,
                        confirmation_expected,
                        {"input_layout": {"stores": {}}},
                    ),
                ),
            ):
                with self.subTest(stage=label), patch.object(
                    aux,
                    "_require_regular_unlinked_file",
                    side_effect=aux.EnderEnsembleEvaluationError(
                        f"{label} output is hardlinked"
                    ),
                ) as guard, patch.object(
                    aux,
                    "_load_config",
                    side_effect=AssertionError("OUTPUT_GUARD_BYPASSED"),
                ):
                    with self.assertRaisesRegex(
                        aux.EnderEnsembleEvaluationError,
                        "hardlinked",
                    ):
                        callback()
                    guard.assert_called_once()

    def test_checkpointed_config_and_loader_reject_untracked_or_blob_drift(
        self,
    ) -> None:
        committed_blob = "a" * 40
        for label, relative in (
            ("confirmation config", "configs/confirmation.py"),
            ("confirmation loader", "numerai/agents/code/modeling/utils/pipeline.py"),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                protocol = _synthetic_protocol(root)
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("frozen = True\n", encoding="utf-8")

                def fake_git(
                    _root: Path, arguments: list[str]
                ) -> subprocess.CompletedProcess:
                    if arguments[0] == "cat-file":
                        return subprocess.CompletedProcess(arguments, 0, "", "")
                    if arguments[0] == "rev-parse":
                        return subprocess.CompletedProcess(
                            arguments, 0, f"{committed_blob}\n", ""
                        )
                    if arguments[0] == "hash-object":
                        return subprocess.CompletedProcess(
                            arguments, 0, f"{committed_blob}\n", ""
                        )
                    if arguments[0] == "status":
                        return subprocess.CompletedProcess(arguments, 0, "", "")
                    raise AssertionError(arguments)

                with patch.object(aux, "_run_git", side_effect=fake_git):
                    receipt = aux._checkpointed_file_receipt(
                        protocol, path, label
                    )
                self.assertEqual(receipt["git_blob_id"], committed_blob)
                self.assertEqual(
                    receipt["checkpoint_commit"], protocol.pretraining_commit
                )

                def untracked_git(
                    _root: Path, arguments: list[str]
                ) -> subprocess.CompletedProcess:
                    result = fake_git(_root, arguments)
                    if arguments[0] == "status":
                        return subprocess.CompletedProcess(
                            arguments, 0, f"?? {relative}\n", ""
                        )
                    return result

                with patch.object(aux, "_run_git", side_effect=untracked_git):
                    with self.assertRaisesRegex(
                        aux.EnderEnsembleEvaluationError,
                        "uncommitted or untracked",
                    ):
                        aux._checkpointed_file_receipt(protocol, path, label)

    def test_checkpointed_config_inventory_and_loader_reject_links_before_git(
        self,
    ) -> None:
        for label in (
            "confirmation config",
            "confirmation store inventory",
            "confirmation loader",
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                protocol = _synthetic_protocol(root)
                source = root / "source.py"
                path = root / "checkpointed.py"
                source.write_text("CONFIG = {}\n", encoding="utf-8")
                aux.os.link(source, path)
                with patch.object(
                    aux,
                    "_run_git",
                    side_effect=AssertionError("GIT_ACCESSED_BEFORE_LINK_GUARD"),
                ) as git_access:
                    with self.assertRaisesRegex(
                        aux.EnderEnsembleEvaluationError,
                        "hard.?link|link count|regular unlinked",
                    ):
                        aux._checkpointed_file_receipt(protocol, path, label)
                    git_access.assert_not_called()

                path.unlink()
                path.write_text("CONFIG = {}\n", encoding="utf-8")
                with patch.object(Path, "is_symlink", return_value=False), patch.object(
                    Path,
                    "lstat",
                    return_value=_regular_stat(reparse=True),
                ), patch.object(
                    aux,
                    "_run_git",
                    side_effect=AssertionError("GIT_ACCESSED_BEFORE_LINK_GUARD"),
                ) as git_access:
                    with self.assertRaisesRegex(
                        aux.EnderEnsembleEvaluationError,
                        "reparse point",
                    ):
                        aux._checkpointed_file_receipt(protocol, path, label)
                    git_access.assert_not_called()

                committed_blob = "a" * 40

                def drifted_git(
                    _root: Path, arguments: list[str]
                ) -> subprocess.CompletedProcess[str]:
                    if arguments[0] == "cat-file":
                        return subprocess.CompletedProcess(arguments, 0, "", "")
                    if arguments[0] == "rev-parse":
                        return subprocess.CompletedProcess(
                            arguments,
                            0,
                            f"{committed_blob}\n",
                            "",
                        )
                    if arguments[0] == "hash-object":
                        return subprocess.CompletedProcess(
                            arguments,
                            0,
                            f"{'b' * 40}\n",
                            "",
                        )
                    if arguments[0] == "status":
                        return subprocess.CompletedProcess(arguments, 0, "", "")
                    raise AssertionError(arguments)

                with patch.object(aux, "_run_git", side_effect=drifted_git):
                    with self.assertRaisesRegex(
                        aux.EnderEnsembleEvaluationError,
                        "checkpoint blob differs",
                    ):
                        aux._checkpointed_file_receipt(protocol, path, label)

    def test_all_confirmation_wrappers_require_exact_source_contract_and_store(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(root)
            protocol.source_manifest["confirmation_output_contract"] = {
                name: {
                    "config_path": aux._confirmation_config_relative(name),
                    "must_be_absent_before_run": True,
                    "predictions_path": (
                        f"numerai/agents/experiments/{aux.EXPERIMENT_NAME}/"
                        f"predictions/confirmation_{name}_d8_t6000.parquet"
                    ),
                    "results_name": f"confirmation_{name}_d8_t6000",
                    "results_path": (
                        f"numerai/agents/experiments/{aux.EXPERIMENT_NAME}/"
                        f"results/confirmation_{name}_d8_t6000.json"
                    ),
                }
                for name in aux.ALL_COMPONENTS
            }
            config_dir = (
                root
                / "numerai/agents/experiments"
                / aux.EXPERIMENT_NAME
                / "configs"
            )
            config_dir.mkdir(parents=True)
            base = config_dir / "base_d8.py"
            base.write_text("def make_config(*args):\n    return {}\n", encoding="utf-8")
            protocol.source_manifest["experiment_files"] = {
                "configs/base_d8.py": aux._file_receipt(base, root)
            }
            sentinel = {"synthetic": True}

            for name in aux.ALL_COMPONENTS:
                with self.subTest(component=name, variant="exact"):
                    path = root / aux._confirmation_config_relative(name)
                    path.write_bytes(
                        aux._expected_confirmation_config_source(name).encode(
                            "utf-8"
                        )
                    )
                    with patch.object(
                        aux, "_load_config", return_value=sentinel
                    ) as evaluate, patch.object(
                        aux, "validate_component_config"
                    ) as validate:
                        loaded = aux._load_confirmation_config(
                            protocol, name, path
                        )
                        self.assertEqual(loaded, sentinel)
                        self.assertIsNot(loaded, sentinel)
                    evaluate.assert_called_once_with(
                        path, f"{name} confirmation config"
                    )
                    validate.assert_called_once_with(
                        name, sentinel, confirmation=True
                    )

                expected_source = aux._expected_confirmation_config_source(name)
                mutations = {
                    "transitive import": expected_source.replace(
                        'with_name("base_d8.py")',
                        'with_name("alternate_base.py")',
                    ),
                    "target": expected_source.replace(
                        aux.COMPONENT_TARGETS[name],
                        aux.COMPONENT_TARGETS[
                            aux.ALL_COMPONENTS[
                                (aux.ALL_COMPONENTS.index(name) + 1)
                                % len(aux.ALL_COMPONENTS)
                            ]
                        ],
                    ),
                    "results name": expected_source.replace(
                        f"confirmation_{name}_d8_t6000",
                        f"confirmation_{name}_alternate",
                    ),
                    "store": expected_source.replace(
                        aux._confirmation_store_relative(name),
                        "v5.3/shared_feature_store",
                    ),
                    "inventory": expected_source.replace(
                        aux.CONFIRMATION_STORE_INVENTORY_PATH.removeprefix(
                            "numerai/"
                        ),
                        "agents/experiments/forged/store_inventory.json",
                    ),
                }
                for variant, source in mutations.items():
                    with self.subTest(component=name, variant=variant):
                        self.assertNotEqual(source, expected_source)
                        path.write_bytes(source.encode("utf-8"))
                        with patch.object(
                            aux,
                            "_load_config",
                            side_effect=AssertionError(
                                "CONFIG_EVALUATED_BEFORE_SOURCE_VALIDATION"
                            ),
                        ) as evaluate:
                            with self.assertRaisesRegex(
                                aux.EnderEnsembleEvaluationError,
                                "config source differs",
                            ):
                                aux._load_confirmation_config(protocol, name, path)
                        evaluate.assert_not_called()

                with self.subTest(component=name, variant="contract"):
                    path.write_bytes(expected_source.encode("utf-8"))
                    contract = protocol.source_manifest[
                        "confirmation_output_contract"
                    ][name]
                    original = dict(contract)
                    contract["results_name"] = f"confirmation_{name}_alternate"
                    try:
                        with patch.object(
                            aux,
                            "_load_config",
                            side_effect=AssertionError(
                                "CONFIG_EVALUATED_BEFORE_CONTRACT_VALIDATION"
                            ),
                        ) as evaluate:
                            with self.assertRaisesRegex(
                                aux.EnderEnsembleEvaluationError,
                                "output contract differs",
                            ):
                                aux._load_confirmation_config(protocol, name, path)
                        evaluate.assert_not_called()
                    finally:
                        contract.clear()
                        contract.update(original)

    def test_confirmation_wrapper_and_transitive_helper_reject_links_before_eval(
        self,
    ) -> None:
        for linked_item in ("wrapper", "base helper"):
            with self.subTest(linked_item=linked_item), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                protocol = _synthetic_protocol(root)
                name = "jasper"
                relative = aux._confirmation_config_relative(name)
                protocol.source_manifest["confirmation_output_contract"] = {
                    name: {
                        "config_path": relative,
                        "must_be_absent_before_run": True,
                        "predictions_path": (
                            f"numerai/agents/experiments/{aux.EXPERIMENT_NAME}/"
                            "predictions/confirmation_jasper_d8_t6000.parquet"
                        ),
                        "results_name": "confirmation_jasper_d8_t6000",
                        "results_path": (
                            f"numerai/agents/experiments/{aux.EXPERIMENT_NAME}/"
                            "results/confirmation_jasper_d8_t6000.json"
                        ),
                    }
                }
                path = root / relative
                path.parent.mkdir(parents=True)
                base = path.with_name("base_d8.py")
                wrapper_source = path.parent / "wrapper-source.py"
                base_source = path.parent / "base-source.py"
                wrapper_source.write_text(
                    aux._expected_confirmation_config_source(name),
                    encoding="utf-8",
                )
                base_source.write_text(
                    "def make_config(*args):\n    return {}\n",
                    encoding="utf-8",
                )
                if linked_item == "wrapper":
                    aux.os.link(wrapper_source, path)
                    base.write_bytes(base_source.read_bytes())
                else:
                    path.write_bytes(wrapper_source.read_bytes())
                    aux.os.link(base_source, base)
                with patch.object(
                    aux,
                    "_load_config",
                    side_effect=AssertionError("LINKED_CONFIG_EVALUATED"),
                ) as evaluate:
                    with self.assertRaisesRegex(
                        aux.EnderEnsembleEvaluationError,
                        "hard.?link|link count|regular unlinked",
                    ):
                        aux._load_confirmation_config(protocol, name, path)
                evaluate.assert_not_called()

    def test_confirmation_store_files_are_lexically_and_physically_distinct(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(root)
            stores: dict[str, dict[str, object]] = {}
            for name in aux.ALL_COMPONENTS:
                store: dict[str, object] = {
                    "generation_id": name,
                    "row_count": 1,
                    "feature_count": 1,
                    "feature_order_sha256": "a" * 64,
                    "target_column": aux.COMPONENT_TARGETS[name],
                }
                for key in ("metadata", "manifest", "features"):
                    path = root / "stores" / name / f"{key}.bin"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"{name}:{key}".encode("utf-8"))
                    store[key] = aux._file_receipt(path, root)
                stores[name] = store

            aux._validate_distinct_confirmation_store_files(protocol, stores)

            duplicate = copy.deepcopy(stores)
            duplicate["teager2b"]["features"] = copy.deepcopy(
                duplicate["jasper"]["features"]
            )
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError,
                "lexical file paths|physically distinct",
            ):
                aux._validate_distinct_confirmation_store_files(
                    protocol,
                    duplicate,
                )

            with patch.object(
                aux.os.path,
                "samefile",
                return_value=True,
            ):
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "not physically distinct",
                ):
                    aux._validate_distinct_confirmation_store_files(
                        protocol,
                        stores,
                    )

    def test_confirmation_store_inventory_is_two_phase_and_checkpoint_bound(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _init_git_repo(root)
            inventory_path = root / aux.CONFIRMATION_STORE_INVENTORY_PATH
            experiment_dir = inventory_path.parent
            experiment_dir.mkdir(parents=True)
            receipt_dir = experiment_dir / "receipts"
            (root / ".gitignore").write_text(
                (
                    "numerai/agents/experiments/"
                    f"{aux.EXPERIMENT_NAME}/receipts/\n"
                ),
                encoding="utf-8",
            )

            stores: dict[str, dict[str, object]] = {}
            for name in aux.ALL_COMPONENTS:
                store: dict[str, object] = {
                    "generation_id": f"{aux.ALL_COMPONENTS.index(name) + 1}" * 32,
                    "row_count": 15,
                    "feature_count": 1,
                    "feature_order_sha256": "a" * 64,
                    "target_column": aux.COMPONENT_TARGETS[name],
                }
                for key in ("metadata", "manifest", "features"):
                    path = root / "numerai" / "v5.3" / name / f"{key}.bin"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"{name}:{key}".encode("utf-8"))
                    store[key] = aux._file_receipt(path, root)
                stores[name] = store

            manifest = {
                "confirmation_xerxes_medium_store_anchor": copy.deepcopy(
                    stores["xerxes"]
                ),
                "confirmation_output_contract": {
                    name: {
                        "config_path": aux._confirmation_config_relative(name),
                        "must_be_absent_before_run": True,
                        "predictions_path": f"predictions/{name}.parquet",
                        "results_name": f"confirmation_{name}_d8_t6000",
                        "results_path": f"results/{name}.json",
                    }
                    for name in aux.ALL_COMPONENTS
                },
            }
            source_manifest_path = root / "source_manifest.json"
            source_manifest_path.write_text("{}\n", encoding="utf-8")
            gpu_runtime_path = root / "gpu_runtime.json"
            gpu_runtime_path.write_text("{}\n", encoding="utf-8")
            (experiment_dir / "gate.md").write_text("synthetic\n", encoding="utf-8")
            commit_a = _commit_all(root, "Scout checkpoint A")

            def protocol_at(commit: str) -> aux.FrozenProtocol:
                return aux.FrozenProtocol(
                    repo_root=root,
                    experiment_dir=experiment_dir,
                    source_manifest_path=source_manifest_path,
                    source_manifest=manifest,
                    scout_configs={},
                    scout_config_paths={},
                    medium_features=("feature_1",),
                    pretraining_commit=commit,
                    gpu_runtime_path=gpu_runtime_path,
                    gpu_runtime_receipt={},
                )

            protocol_a = protocol_at(commit_a)
            selected = "tyler_w10"
            locked_metrics = _serialized_synthetic_metrics(
                aux.SCOUT_LOCKED_ERA_SEQUENCE,
                [selected],
            )
            locked_summary = _summary_from_serialized_metrics(
                locked_metrics,
                selected,
                aux.SCOUT_LOCKED_ERA_SEQUENCE,
            )
            scout_payload = {
                "schema_version": 1,
                "experiment": aux.EXPERIMENT_NAME,
                "stage": "locked",
                "state": "PASS",
                "passed": True,
                "protocol": {
                    "pretraining_commit": commit_a,
                    "stable": "synthetic-binding",
                },
                "input_receipt": {
                    "path": (
                        "numerai/agents/experiments/"
                        f"{aux.EXPERIMENT_NAME}/receipts/"
                        f"calibrate-{'9' * 64}.json"
                    ),
                    "sha256": "9" * 64,
                },
                "selected_formula": {
                    "name": selected,
                    "weights": dict(aux.BLEND_WEIGHTS[selected]),
                },
                "locked": {
                    "rows": len(aux.SCOUT_LOCKED_ERA_SEQUENCE),
                    "eras": len(aux.SCOUT_LOCKED_ERA_SEQUENCE),
                    "first_era": aux.SCOUT_LOCKED_ERA_SEQUENCE[0],
                    "last_era": aux.SCOUT_LOCKED_ERA_SEQUENCE[-1],
                    "summary": locked_summary,
                    "checks": aux.locked_checks(locked_summary),
                    "per_era": locked_metrics,
                },
            }
            scout_claim = aux._claim_receipt_prefix(receipt_dir, "locked")
            scout_path = aux._write_claimed_content_addressed_receipt(
                receipt_dir,
                "locked",
                scout_claim,
                scout_payload,
            )
            scout_digest = aux._sha256_file(scout_path)

            def protocol_binding(protocol: aux.FrozenProtocol) -> dict[str, str]:
                return {
                    "pretraining_commit": protocol.pretraining_commit,
                    "stable": "synthetic-binding",
                }

            common_patches = (
                patch.object(aux, "_protocol_binding", side_effect=protocol_binding),
                patch.object(aux, "_load_confirmation_config", return_value={}),
                patch.object(
                    aux,
                    "_read_full_confirmation_sources",
                    return_value=pd.DataFrame(),
                ),
                patch.object(
                    aux,
                    "_confirmation_store_receipt",
                    side_effect=lambda _protocol, name: copy.deepcopy(stores[name]),
                ),
                patch.object(aux, "_validate_confirmation_store_receipt"),
                patch.object(
                    aux,
                    "_load_passing_scout_locked_receipt",
                    return_value=scout_payload,
                ),
            )
            with common_patches[0], common_patches[1], common_patches[2], common_patches[3], common_patches[4], common_patches[5]:
                inventory_stage_path, inventory_stage = (
                    aux.create_confirmation_store_inventory(
                        protocol_a,
                        scout_path,
                        scout_digest,
                        receipt_dir,
                    )
                )

            self.assertEqual(
                inventory_stage["stage"],
                "create-confirmation-store-inventory",
            )
            self.assertTrue(inventory_stage_path.name.startswith(
                "confirmation-store-inventory-"
            ))
            static_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(static_inventory),
                {
                    "schema_version",
                    "experiment",
                    "artifact",
                    "scout_locked_receipt",
                    "selected_formula",
                    "canonical_store",
                    "input_layout",
                },
            )
            self.assertNotIn("checkpoint", _mapping_keys(static_inventory))
            self.assertNotIn("protocol", _mapping_keys(static_inventory))
            self.assertNotIn(commit_a, json.dumps(static_inventory, sort_keys=True))

            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError,
                "absent from the confirmation checkpoint",
            ):
                aux._checkpointed_file_receipt(
                    protocol_a,
                    inventory_path,
                    "confirmation store inventory",
                )

            _git(
                root,
                "add",
                "--force",
                "--",
                aux.CONFIRMATION_STORE_INVENTORY_PATH,
            )
            _git(root, "commit", "--quiet", "-m", "Inventory checkpoint C")
            commit_c = _git(root, "rev-parse", "HEAD").stdout.strip()
            self.assertNotEqual(commit_c, commit_a)
            protocol_c = protocol_at(commit_c)
            original_checkpointed = aux._checkpointed_file_receipt

            def checkpointed(
                protocol: aux.FrozenProtocol,
                path: Path,
                label: str,
            ) -> dict[str, object]:
                if Path(aux.os.path.abspath(path)) == inventory_path:
                    return original_checkpointed(protocol, path, label)
                return {
                    "path": aux._lexical_relative_path(path, root),
                    "size_bytes": 1,
                    "sha256": "8" * 64,
                    "checkpoint_commit": protocol.pretraining_commit,
                    "git_blob_id": "7" * 40,
                }

            with patch.object(
                aux, "_protocol_binding", side_effect=protocol_binding
            ), patch.object(
                aux, "_load_confirmation_config", return_value={}
            ), patch.object(
                aux,
                "_checkpointed_file_receipt",
                side_effect=checkpointed,
            ), patch.object(
                aux,
                "_read_full_confirmation_sources",
                return_value=pd.DataFrame(),
            ), patch.object(
                aux,
                "_confirmation_store_receipt",
                side_effect=lambda _protocol, name: copy.deepcopy(stores[name]),
            ), patch.object(
                aux, "_validate_confirmation_store_receipt"
            ), patch.object(
                aux, "_confirmation_destination_receipts", return_value={}
            ), patch.object(
                aux,
                "_load_passing_scout_locked_receipt",
                return_value=scout_payload,
            ):
                _, pretraining = aux.create_confirmation_pretraining_receipt(
                    protocol_c,
                    scout_path,
                    scout_digest,
                    receipt_dir,
                )

            self.assertEqual(
                pretraining["store_inventory"]["checkpoint_commit"],
                commit_c,
            )
            self.assertEqual(
                pretraining["store_inventory"]["path"],
                aux.CONFIRMATION_STORE_INVENTORY_PATH,
            )
            self.assertNotIn(commit_c, json.dumps(static_inventory, sort_keys=True))

            original_inventory = inventory_path.read_bytes()
            inventory_path.write_bytes(original_inventory + b" ")
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError,
                "checkpoint blob differs",
            ):
                original_checkpointed(
                    protocol_c,
                    inventory_path,
                    "confirmation store inventory",
                )
            inventory_path.write_bytes(original_inventory)

            hardlink = root / "inventory-hardlink.json"
            aux.os.link(inventory_path, hardlink)
            try:
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "hard.?link|link count|regular unlinked",
                ):
                    original_checkpointed(
                        protocol_c,
                        inventory_path,
                        "confirmation store inventory",
                    )
            finally:
                hardlink.unlink()

    def test_forged_scout_lock_is_rederived_before_confirmation_access(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(
                root,
                canonical_experiment=True,
            )
            receipt_dir = protocol.experiment_dir / "receipts"
            selected = "tyler_w00"
            locked_metrics = _serialized_synthetic_metrics(
                aux.SCOUT_LOCKED_ERA_SEQUENCE,
                [selected],
            )
            locked_summary = _summary_from_serialized_metrics(
                locked_metrics,
                selected,
                aux.SCOUT_LOCKED_ERA_SEQUENCE,
            )
            calibration_digest = "a" * 64
            locked_receipt = {
                "schema_version": 1,
                "experiment": aux.EXPERIMENT_NAME,
                "stage": "locked",
                "state": "PASS",
                "passed": True,
                "protocol": {
                    "pretraining_commit": protocol.pretraining_commit,
                },
                "input_receipt": {
                    "path": (
                        "numerai/agents/experiments/"
                        f"{aux.EXPERIMENT_NAME}/receipts/"
                        f"calibrate-{calibration_digest}.json"
                    ),
                    "sha256": calibration_digest,
                },
                "selected_formula": {
                    "name": selected,
                    "weights": dict(aux.BLEND_WEIGHTS[selected]),
                },
                "locked": {
                    "rows": len(aux.SCOUT_LOCKED_ERA_SEQUENCE),
                    "eras": len(aux.SCOUT_LOCKED_ERA_SEQUENCE),
                    "first_era": aux.SCOUT_LOCKED_ERA_SEQUENCE[0],
                    "last_era": aux.SCOUT_LOCKED_ERA_SEQUENCE[-1],
                    "summary": locked_summary,
                    "checks": aux.locked_checks(locked_summary),
                    "per_era": locked_metrics,
                },
            }
            aux._validate_stage_receipt_schema(locked_receipt, "locked")

            authorized_eras = (
                aux.SCOUT_CALIBRATION_ERA_SEQUENCE
                + aux.SCOUT_LOCKED_ERA_SEQUENCE
            )
            expected = aux.ExpectedCohort(
                frame=pd.DataFrame(
                    {
                        aux.ID_COLUMN: [
                            f"synthetic-{index}"
                            for index in range(len(authorized_eras))
                        ],
                        aux.ERA_COLUMN: authorized_eras,
                    }
                ),
                full_rows=len(authorized_eras),
                full_eras=len(authorized_eras),
                eras=authorized_eras,
                folds=(
                    {
                        "fold": 1,
                        "train_eras": len(aux.SCOUT_CALIBRATION_ERA_SEQUENCE),
                        "val_eras": len(aux.SCOUT_LOCKED_ERA_SEQUENCE),
                        "train_rows": len(aux.SCOUT_CALIBRATION_ERA_SEQUENCE),
                        "val_rows": len(aux.SCOUT_LOCKED_ERA_SEQUENCE),
                    },
                ),
            )
            calibration_inputs = {"seal_receipts": {}}
            calibration_receipt = {"inputs": calibration_inputs}
            full_frame = expected.frame.copy()
            live_serialized = _serialized_synthetic_metrics(
                aux.SCOUT_LOCKED_ERA_SEQUENCE,
                [selected],
            )
            for row in live_serialized["bmc"][selected]:
                row[1] = float(row[1]) + 0.001
            live_metrics = {
                metric: pd.DataFrame(
                    {
                        signal: [float(row[1]) for row in rows]
                        for signal, rows in signals.items()
                    },
                    index=list(aux.SCOUT_LOCKED_ERA_SEQUENCE),
                )
                for metric, signals in live_serialized.items()
            }

            def load_stage(
                _protocol: aux.FrozenProtocol,
                _path: Path,
                _digest: str,
                *,
                stage: str,
                allow_prior_pretraining_commit: bool = False,
                expected_prior_pretraining_commit: str | None = None,
            ) -> dict[str, object]:
                self.assertTrue(allow_prior_pretraining_commit)
                if stage == "locked":
                    self.assertIsNone(expected_prior_pretraining_commit)
                    return locked_receipt
                if stage == "calibrate":
                    self.assertEqual(
                        expected_prior_pretraining_commit,
                        protocol.pretraining_commit,
                    )
                    return calibration_receipt
                raise AssertionError(stage)

            locked_digest = "b" * 64
            locked_path = receipt_dir / f"locked-{locked_digest}.json"
            with patch.object(
                aux,
                "_load_passing_stage_receipt",
                side_effect=load_stage,
            ), patch.object(
                aux,
                "_seal_bindings_from_receipt",
                return_value={},
            ), patch.object(
                aux,
                "build_scout_expected_cohort",
                return_value=expected,
            ), patch.object(
                aux,
                "_build_scout_scoring_frame",
                return_value=(full_frame, calibration_inputs),
            ), patch.object(
                aux,
                "_validate_scout_calibration_derivation",
                return_value=selected,
            ), patch.object(
                aux,
                "build_selected_rank_blend",
                side_effect=lambda frame, _selected: frame,
            ), patch.object(
                aux,
                "compute_per_era_metrics",
                return_value=live_metrics,
            ), patch.object(
                aux,
                "_load_confirmation_config",
                side_effect=AssertionError("CONFIRMATION_CONFIG_ACCESSED"),
            ) as config_access, patch.object(
                aux,
                "_read_full_confirmation_sources",
                side_effect=AssertionError("CONFIRMATION_RAW_ACCESSED"),
            ) as raw_access, patch.object(
                aux,
                "_confirmation_store_receipt",
                side_effect=AssertionError("CONFIRMATION_STORE_ACCESSED"),
            ) as store_access:
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "Scout locked authorization derivation differs",
                ):
                    aux.create_confirmation_store_inventory(
                        protocol,
                        locked_path,
                        locked_digest,
                        receipt_dir,
                    )
            config_access.assert_not_called()
            raw_access.assert_not_called()
            store_access.assert_not_called()

    def test_prior_scout_checkpoint_must_be_an_ancestor_of_inventory_checkpoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(root)
            receipt = {
                "protocol": {
                    "pretraining_commit": "2" * 40,
                    "stable": "synthetic-binding",
                }
            }

            def expected_binding(_protocol: aux.FrozenProtocol) -> dict[str, str]:
                return {
                    "pretraining_commit": _protocol.pretraining_commit,
                    "stable": "synthetic-binding",
                }

            malformed = copy.deepcopy(receipt)
            malformed["protocol"]["pretraining_commit"] = "g" * 40
            with patch.object(
                aux,
                "_protocol_binding",
                side_effect=expected_binding,
            ), patch.object(aux, "_run_git") as git_access:
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "Prior stage pretraining commit is malformed",
                ):
                    aux._validate_protocol_binding(
                        malformed,
                        protocol,
                        allow_prior_pretraining_commit=True,
                    )
                git_access.assert_not_called()

            with patch.object(
                aux, "_protocol_binding", side_effect=expected_binding
            ), patch.object(
                aux,
                "_run_git",
                side_effect=(
                    subprocess.CompletedProcess(
                        ["git", "cat-file"],
                        0,
                        "",
                        "",
                    ),
                    subprocess.CompletedProcess(
                        ["git", "merge-base", "lower-bound"],
                        0,
                        "",
                        "",
                    ),
                    subprocess.CompletedProcess(
                        ["git", "merge-base", "upper-bound"],
                        1,
                        "",
                        "",
                    ),
                ),
            ) as git_access:
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "not an ancestor",
                ):
                    aux._validate_protocol_binding(
                        receipt,
                        protocol,
                        allow_prior_pretraining_commit=True,
                    )
            self.assertEqual(git_access.call_count, 3)
            self.assertEqual(
                git_access.call_args_list[0].args[1],
                ["cat-file", "-e", f"{'2' * 40}^{{commit}}"],
            )
            self.assertEqual(
                git_access.call_args_list[1].args[1],
                [
                    "merge-base",
                    "--is-ancestor",
                    aux.PRE_SCORING_COMMIT,
                    "2" * 40,
                ],
            )
            self.assertEqual(
                git_access.call_args_list[2].args[1],
                [
                    "merge-base",
                    "--is-ancestor",
                    "2" * 40,
                    protocol.pretraining_commit,
                ],
            )

            with patch.object(
                aux,
                "_protocol_binding",
                side_effect=expected_binding,
            ), patch.object(
                aux,
                "_run_git",
                side_effect=(
                    subprocess.CompletedProcess(
                        ["git", "cat-file"],
                        0,
                        "",
                        "",
                    ),
                    subprocess.CompletedProcess(
                        ["git", "merge-base", "lower-bound"],
                        1,
                        "",
                        "",
                    ),
                ),
            ) as git_access:
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "predates the frozen protocol checkpoint",
                ):
                    aux._validate_protocol_binding(
                        receipt,
                        protocol,
                        allow_prior_pretraining_commit=True,
                    )
            self.assertEqual(git_access.call_count, 2)
            self.assertEqual(
                git_access.call_args_list[1].args[1],
                [
                    "merge-base",
                    "--is-ancestor",
                    aux.PRE_SCORING_COMMIT,
                    "2" * 40,
                ],
            )

            mixed_checkpoint = copy.deepcopy(receipt)
            mixed_checkpoint["protocol"]["pretraining_commit"] = "3" * 40
            with patch.object(
                aux,
                "_protocol_binding",
                side_effect=expected_binding,
            ), patch.object(aux, "_run_git") as git_access:
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "shared prior stage pretraining checkpoint differs",
                ):
                    aux._validate_protocol_binding(
                        mixed_checkpoint,
                        protocol,
                        allow_prior_pretraining_commit=True,
                        expected_prior_pretraining_commit="2" * 40,
                    )
                git_access.assert_not_called()

            with patch.object(
                aux,
                "_protocol_binding",
                side_effect=expected_binding,
            ), patch.object(
                aux,
                "_run_git",
                side_effect=(
                    subprocess.CompletedProcess(
                        ["git", "cat-file"],
                        0,
                        "",
                        "",
                    ),
                    subprocess.CompletedProcess(
                        ["git", "merge-base", "lower-bound"],
                        0,
                        "",
                        "",
                    ),
                    subprocess.CompletedProcess(
                        ["git", "merge-base"],
                        0,
                        "",
                        "",
                    ),
                    subprocess.CompletedProcess(
                        ["git", "diff"],
                        1,
                        "",
                        "",
                    ),
                ),
            ) as git_access:
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "Training implementation changed",
                ):
                    aux._validate_protocol_binding(
                        receipt,
                        protocol,
                        allow_prior_pretraining_commit=True,
                    )
            self.assertEqual(git_access.call_count, 4)
            diff_arguments = git_access.call_args_list[3].args[1]
            self.assertIn("diff", diff_arguments)
            for path in aux.TRAINING_CHECKPOINT_PATHS:
                self.assertIn(path, diff_arguments)

    def test_closed_stage_receipt_schemas_reject_metric_and_holdout_injection(
        self,
    ) -> None:
        selected = "tyler_w00"
        receipt_root = (
            f"numerai/agents/experiments/{aux.EXPERIMENT_NAME}/receipts"
        )

        def canonical_binding(prefix: str, digest: str) -> dict[str, str]:
            return {
                "path": f"{receipt_root}/{prefix}-{digest}.json",
                "sha256": digest,
            }

        scout_locked_binding = canonical_binding("locked", "a" * 64)
        pretraining_binding = canonical_binding(
            "confirmation-pretraining", "b" * 64
        )
        calibration_binding = canonical_binding("calibrate", "c" * 64)
        confirmation_calibration_binding = canonical_binding(
            "confirmation-calibrate", "d" * 64
        )
        file_receipt = {
            "path": "artifacts/file.bin",
            "sha256": "b" * 64,
            "size_bytes": 1,
        }
        cohort = {
            "rows": 1,
            "eras": 1,
            "first_era": "0001",
            "last_era": "0001",
            "full_rows": 1,
            "full_eras": 1,
            "folds": [
                {
                    "fold": 1,
                    "train_eras": 0,
                    "val_eras": 1,
                    "train_rows": 0,
                    "val_rows": 1,
                }
            ],
        }
        def component_artifact(component: str) -> dict[str, object]:
            return {
                "component": component,
                "target": aux.COMPONENT_TARGETS[component],
                "config": copy.deepcopy(file_receipt),
                "result": copy.deepcopy(file_receipt),
                "predictions": copy.deepcopy(file_receipt),
            }

        def consumption_claim(
            component: str, *, confirmation: bool
        ) -> dict[str, object]:
            family = "confirmation" if confirmation else "scout"
            return {
                "path": (
                    f"numerai/agents/experiments/{aux.EXPERIMENT_NAME}/"
                    f"receipts/.{family}-train-{component}.consumed.json"
                ),
                "sha256": hashlib.sha256(
                    f"{family}:{component}".encode("utf-8")
                ).hexdigest(),
                "size_bytes": 1,
            }

        artifact = component_artifact("jasper")
        selected_formula = {
            "name": selected,
            "weights": dict(aux.BLEND_WEIGHTS[selected]),
        }
        summary = _summary()
        summary["era_count"] = 1
        summary["corr"]["std_valid"] = True
        summary["bmc"]["std_valid"] = True

        tabm_two_seed_residual = {
            seed: {
                "config": "c" * 64,
                "result": "d" * 64,
                "predictions": "e" * 64,
            }
            for seed in ("seed1337", "seed2027")
        }

        def normalized_seal_inputs(*, confirmation: bool) -> dict[str, object]:
            order = (
                aux.CONFIRMATION_RUN_ORDER
                if confirmation
                else aux.SCOUT_RUN_ORDER
            )
            seal_digest_chars = ("1", "2", "3", "4", "5")
            pre_run_digest_chars = ("6", "7", "8", "9", "a")
            seals: dict[str, object] = {}
            for index, component in enumerate(order):
                seal_digest = seal_digest_chars[index] * 64
                seal_prefix = (
                    f"confirmation-seal-{component}"
                    if confirmation
                    else f"scout-seal-{component}"
                )
                pre_run_prefix = (
                    f"confirmation-pre-run-{component}"
                    if confirmation
                    else f"scout-pre-run-{component}"
                )
                item: dict[str, object] = {
                    "path": canonical_binding(
                        seal_prefix,
                        seal_digest,
                    )["path"],
                    "sha256": seal_digest,
                    "prior_finalized_seal": (
                        None
                        if index == 0
                        else {
                            "component": order[index - 1],
                            "path": seals[order[index - 1]]["path"],
                            "sha256": seals[order[index - 1]]["sha256"],
                        }
                    ),
                    "pre_run_absence_receipt": canonical_binding(
                        pre_run_prefix,
                        pre_run_digest_chars[index] * 64,
                    ),
                    "run_consumption_claim": consumption_claim(
                        component,
                        confirmation=confirmation,
                    ),
                    "run_completion_receipt": (
                        _synthetic_completion_file_receipt(
                            component,
                            confirmation=confirmation,
                        )
                    ),
                    "artifact": component_artifact(component),
                }
                if confirmation:
                    item["confirmation_pretraining_receipt"] = copy.deepcopy(
                        pretraining_binding
                    )
                seals[component] = item
            return seals

        def scout_calibration_block() -> dict[str, object]:
            era_sequence = aux.SCOUT_CALIBRATION_ERA_SEQUENCE
            per_era = _serialized_synthetic_metrics(
                era_sequence,
                list(aux.CANDIDATE_NAMES),
            )
            summaries = {
                candidate: _summary_from_serialized_metrics(
                    per_era,
                    candidate,
                    era_sequence,
                )
                for candidate in aux.CANDIDATE_NAMES
            }
            derived_selected, evaluations = aux.select_scout_candidate(summaries)
            self.assertEqual(derived_selected, selected)
            return {
                "rows": len(era_sequence),
                "eras": len(era_sequence),
                "first_era": era_sequence[0],
                "last_era": era_sequence[-1],
                "candidates": {
                    candidate: {
                        "summary": copy.deepcopy(summaries[candidate]),
                        **copy.deepcopy(evaluations[candidate]),
                    }
                    for candidate in aux.CANDIDATE_NAMES
                },
                "per_era": per_era,
            }

        def confirmation_calibration_block() -> dict[str, object]:
            block = scoring_block(
                (
                    "calibration_bmc_mean",
                    "calibration_bmc_sharpe",
                    "calibration_bmc_max_drawdown",
                    "calibration_corr_mean",
                    "calibration_ender20_similarity",
                    "calibration_ender60_similarity",
                    "calibration_tabm_similarity",
                ),
                aux.CONFIRMATION_CALIBRATION_ERA_SEQUENCE,
                aux.confirmation_calibration_checks,
            )
            return block

        checkpointed_file_receipt = {
            **copy.deepcopy(file_receipt),
            "checkpoint_commit": "1" * 40,
            "git_blob_id": "2" * 40,
        }

        def store_receipt(component: str) -> dict[str, object]:
            return {
                "generation_id": "4" * 32,
                "row_count": 1,
                "feature_count": 1,
                "feature_order_sha256": "3" * 64,
                "target_column": aux.COMPONENT_TARGETS[component],
                "metadata": copy.deepcopy(file_receipt),
                "manifest": copy.deepcopy(file_receipt),
                "features": copy.deepcopy(file_receipt),
            }

        confirmation_stores = {
            component: store_receipt(component)
            for component in aux.ALL_COMPONENTS
        }

        def scoring_block(
            check_keys: tuple[str, ...],
            era_sequence: tuple[str, ...],
            check_function: object,
        ) -> dict[str, object]:
            per_era = _serialized_synthetic_metrics(
                era_sequence,
                [selected],
            )
            block_summary = _summary_from_serialized_metrics(
                per_era,
                selected,
                era_sequence,
            )
            checks = check_function(block_summary)
            self.assertEqual(set(checks), set(check_keys))
            return {
                "rows": len(era_sequence),
                "eras": len(era_sequence),
                "first_era": era_sequence[0],
                "last_era": era_sequence[-1],
                "summary": block_summary,
                "checks": checks,
                "per_era": per_era,
            }

        def envelope(
            stage: str,
            *,
            state: str = "PASS",
            passed: bool = True,
        ) -> dict[str, object]:
            return {
                "schema_version": 1,
                "experiment": aux.EXPERIMENT_NAME,
                "stage": stage,
                "state": state,
                "passed": passed,
                "protocol": {},
            }

        scout_seal = {
            **envelope("seal-scout-component", state="SEALED"),
            "component": "jasper",
            "prior_finalized_seal": None,
            "pre_run_absence_receipt": canonical_binding(
                "scout-pre-run-jasper",
                "e" * 64,
            ),
            "run_consumption_claim": consumption_claim(
                "jasper", confirmation=False
            ),
            "run_completion_receipt": _synthetic_completion_file_receipt(
                "jasper", confirmation=False
            ),
            "cohort": copy.deepcopy(cohort),
            "artifact": copy.deepcopy(artifact),
            "gpu_folds_verified": 1,
        }
        scout_calibrate = {
            **envelope("calibrate"),
            "inputs": {
                "seal_receipts": normalized_seal_inputs(confirmation=False),
                "reused_xerxes": component_artifact("xerxes"),
                "tabm_two_seed_residual": copy.deepcopy(
                    tabm_two_seed_residual
                ),
                "cohort": copy.deepcopy(cohort),
            },
            "selected_formula": copy.deepcopy(selected_formula),
            "calibration": scout_calibration_block(),
        }
        scout_locked = {
            **envelope("locked"),
            "input_receipt": copy.deepcopy(calibration_binding),
            "selected_formula": copy.deepcopy(selected_formula),
            "locked": scoring_block(
                ("bmc_mean", "bmc_sharpe", "bmc_max_drawdown", "corr_mean"),
                aux.SCOUT_LOCKED_ERA_SEQUENCE,
                aux.locked_checks,
            ),
        }
        confirmation_preflight = {
            **envelope(
                "claim-confirmation-component-run",
                state="ABSENCE_PROVEN",
            ),
            "component": "jasper",
            "confirmation_pretraining_receipt": copy.deepcopy(
                pretraining_binding
            ),
            "prior_finalized_seal": None,
            "destinations": {
                "result": {"path": "results/jasper.json", "absent": True},
                "predictions": {
                    "path": "predictions/jasper.parquet",
                    "absent": True,
                },
            },
        }
        confirmation_seal = {
            **envelope("seal-confirmation-component", state="SEALED"),
            "component": "jasper",
            "confirmation_pretraining_receipt": copy.deepcopy(
                pretraining_binding
            ),
            "prior_finalized_seal": None,
            "pre_run_absence_receipt": canonical_binding(
                "confirmation-pre-run-jasper",
                "e" * 64,
            ),
            "run_consumption_claim": consumption_claim(
                "jasper", confirmation=True
            ),
            "run_completion_receipt": _synthetic_completion_file_receipt(
                "jasper", confirmation=True
            ),
            "cohort": copy.deepcopy(cohort),
            "artifact": copy.deepcopy(artifact),
            "gpu_folds_verified": 1,
        }
        confirmation_calibrate = {
            **envelope("confirmation-calibrate"),
            "inputs": {
                "scout_locked_receipt": copy.deepcopy(
                    scout_locked_binding
                ),
                "confirmation_pretraining_receipt": copy.deepcopy(
                    pretraining_binding
                ),
                "confirmation_seal_receipts": normalized_seal_inputs(
                    confirmation=True
                ),
                "tabm_two_seed_residual": copy.deepcopy(
                    tabm_two_seed_residual
                ),
                "cohort": copy.deepcopy(cohort),
            },
            "selected_formula": copy.deepcopy(selected_formula),
            "calibration": confirmation_calibration_block(),
        }
        confirmation_locked = {
            **envelope(
                "confirmation-locked",
                state="STOP_CONFIRMATION_LOCKED_FAILED",
                passed=False,
            ),
            "input_receipt": copy.deepcopy(
                confirmation_calibration_binding
            ),
            "confirmation_pretraining_receipt": copy.deepcopy(
                pretraining_binding
            ),
            "confirmation_seal_receipts": normalized_seal_inputs(
                confirmation=True
            ),
            "selected_formula": copy.deepcopy(selected_formula),
            "locked": scoring_block(
                (
                    "locked_bmc_mean",
                    "locked_bmc_sharpe",
                    "locked_bmc_max_drawdown",
                    "locked_corr_mean",
                ),
                aux.CONFIRMATION_LOCKED_ERA_SEQUENCE,
                aux.confirmation_locked_checks,
            ),
        }
        failed_locked_metrics = confirmation_locked["locked"]["per_era"]
        for index, row in enumerate(failed_locked_metrics["bmc"][selected]):
            row[1] = (-0.0005, -0.0015)[index % 2]
        confirmation_locked["locked"]["summary"] = (
            _summary_from_serialized_metrics(
                failed_locked_metrics,
                selected,
                aux.CONFIRMATION_LOCKED_ERA_SEQUENCE,
            )
        )
        confirmation_locked["locked"]["checks"] = (
            aux.confirmation_locked_checks(
                confirmation_locked["locked"]["summary"]
            )
        )
        confirmation_pretraining = {
            **envelope("confirmation-pretraining"),
            "checkpoint": "1" * 40,
            "scout_locked_receipt": copy.deepcopy(scout_locked_binding),
            "configs": {
                component: copy.deepcopy(checkpointed_file_receipt)
                for component in aux.ALL_COMPONENTS
            },
            "config_helpers": [
                copy.deepcopy(checkpointed_file_receipt)
                for _ in aux.CONFIRMATION_CONFIG_HELPER_PATHS
            ],
            "loader": {
                "checkpoint": "1" * 40,
                "files": [
                    copy.deepcopy(checkpointed_file_receipt)
                    for _ in aux.CONFIRMATION_LOADER_PATHS
                ],
            },
            "store_inventory": copy.deepcopy(checkpointed_file_receipt),
            "canonical_store": copy.deepcopy(confirmation_stores["xerxes"]),
            "input_layout": {
                "type": "dedicated_target_stores",
                "stores": copy.deepcopy(confirmation_stores),
            },
            "output_destinations": {
                component: {
                    "results_path": f"results/{component}.json",
                    "predictions_path": f"predictions/{component}.parquet",
                    "results_absent_at_checkpoint": True,
                    "predictions_absent_at_checkpoint": True,
                }
                for component in aux.ALL_COMPONENTS
            },
        }
        receipts = {
            "seal-scout-component": scout_seal,
            "calibrate": scout_calibrate,
            "locked": scout_locked,
            "claim-confirmation-component-run": confirmation_preflight,
            "seal-confirmation-component": confirmation_seal,
            "confirmation-calibrate": confirmation_calibrate,
            "confirmation-locked": confirmation_locked,
            "confirmation-pretraining": confirmation_pretraining,
        }

        for stage, receipt in receipts.items():
            with self.subTest(stage=stage, variant="exact"):
                aux._validate_stage_receipt_schema(receipt, stage)
            for injected in ("metrics", "locked", "full"):
                if injected in receipt and not (
                    stage == "confirmation-locked" and injected == "full"
                ):
                    continue
                forged = copy.deepcopy(receipt)
                forged[injected] = (
                    scoring_block(
                        (
                            "full_bmc_mean",
                            "full_bmc_sharpe",
                            "full_bmc_max_drawdown",
                            "full_corr_mean",
                            "full_ender20_similarity",
                            "full_ender60_similarity",
                            "full_tabm_similarity",
                        ),
                        aux.CONFIRMATION_FULL_ERA_SEQUENCE,
                        aux.confirmation_full_checks,
                    )
                    if injected == "full"
                    else {}
                )
                with self.subTest(stage=stage, injected=injected):
                    with self.assertRaisesRegex(
                        aux.EnderEnsembleEvaluationError,
                        "keys differs|forbidden|may not contain|inconsistent",
                    ):
                        aux._validate_stage_receipt_schema(forged, stage)

        nested_cases = []
        forged_scout_seal = copy.deepcopy(scout_seal)
        forged_scout_seal["artifact"]["metrics"] = {}
        nested_cases.append(("Scout seal artifact", forged_scout_seal, "seal-scout-component"))
        forged_preflight = copy.deepcopy(confirmation_preflight)
        forged_preflight["destinations"]["result"]["metrics"] = {}
        nested_cases.append(
            (
                "confirmation preflight destination",
                forged_preflight,
                "claim-confirmation-component-run",
            )
        )
        forged_preflight_binding = copy.deepcopy(confirmation_preflight)
        forged_preflight_binding["confirmation_pretraining_receipt"][
            "unexpected"
        ] = True
        nested_cases.append(
            (
                "confirmation preflight pretraining binding schema",
                forged_preflight_binding,
                "claim-confirmation-component-run",
            )
        )
        forged_confirmation_seal_binding = copy.deepcopy(confirmation_seal)
        forged_confirmation_seal_binding["confirmation_pretraining_receipt"][
            "unexpected"
        ] = True
        nested_cases.append(
            (
                "confirmation seal pretraining binding schema",
                forged_confirmation_seal_binding,
                "seal-confirmation-component",
            )
        )
        forged_pretraining = copy.deepcopy(confirmation_pretraining)
        forged_pretraining["configs"]["jasper"] = {"metrics": {}}
        nested_cases.append(
            (
                "confirmation pretraining config",
                forged_pretraining,
                "confirmation-pretraining",
            )
        )
        forged_scout_calibrate = copy.deepcopy(scout_calibrate)
        forged_scout_calibrate["inputs"]["locked"] = {}
        nested_cases.append(
            (
                "Scout calibration holdout injection",
                forged_scout_calibrate,
                "calibrate",
            )
        )
        forged_confirmation_calibrate = copy.deepcopy(confirmation_calibrate)
        forged_confirmation_calibrate["inputs"]["full"] = {}
        nested_cases.append(
            (
                "confirmation calibration full injection",
                forged_confirmation_calibrate,
                "confirmation-calibrate",
            )
        )
        forged_scout_calibration_block = copy.deepcopy(scout_calibrate)
        forged_scout_calibration_block["calibration"]["locked"] = {}
        nested_cases.append(
            (
                "Scout calibration block holdout injection",
                forged_scout_calibration_block,
                "calibrate",
            )
        )
        forged_confirmation_calibration_block = copy.deepcopy(
            confirmation_calibrate
        )
        forged_confirmation_calibration_block["calibration"]["full"] = {}
        nested_cases.append(
            (
                "confirmation calibration block full injection",
                forged_confirmation_calibration_block,
                "confirmation-calibrate",
            )
        )
        for label, base, stage in (
            ("Scout selected formula schema", scout_calibrate, "calibrate"),
            (
                "confirmation selected formula schema",
                confirmation_calibrate,
                "confirmation-calibrate",
            ),
        ):
            forged_formula = copy.deepcopy(base)
            forged_formula["selected_formula"]["unexpected"] = True
            nested_cases.append((label, forged_formula, stage))
        forged_loader = copy.deepcopy(confirmation_pretraining)
        forged_loader["loader"]["unexpected"] = True
        nested_cases.append(
            (
                "confirmation pretraining loader schema",
                forged_loader,
                "confirmation-pretraining",
            )
        )
        for label, path in (
            (
                "confirmation pretraining canonical store schema",
                ("canonical_store",),
            ),
            (
                "confirmation pretraining component store schema",
                ("input_layout", "stores", "jasper"),
            ),
            (
                "confirmation pretraining output destination schema",
                ("output_destinations", "jasper"),
            ),
        ):
            forged_pretraining_nested = copy.deepcopy(confirmation_pretraining)
            target = forged_pretraining_nested
            for key in path:
                target = target[key]
            target["unexpected"] = True
            nested_cases.append(
                (label, forged_pretraining_nested, "confirmation-pretraining")
            )
        forged_confirmation_locked_seals = copy.deepcopy(confirmation_locked)
        forged_confirmation_locked_seals["confirmation_seal_receipts"][
            "metrics"
        ] = {}
        nested_cases.append(
            (
                "confirmation locked seal schema",
                forged_confirmation_locked_seals,
                "confirmation-locked",
            )
        )
        for label, forged, stage in nested_cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "forbidden pre-scoring keys|keys differs",
                ):
                    aux._validate_stage_receipt_schema(forged, stage)

        for stage, receipt in (
            ("locked", scout_locked),
            ("confirmation-locked", confirmation_locked),
        ):
            forged = copy.deepcopy(receipt)
            forged["locked"]["per_era"]["corr"]["tyler_w40"] = [
                ["0001", 0.02]
            ]
            with self.subTest(stage=stage, variant="unselected signal"):
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "per_era.corr keys differs",
                ):
                    aux._validate_stage_receipt_schema(forged, stage)

        replaced_interior = copy.deepcopy(scout_calibrate)
        interior = len(aux.SCOUT_CALIBRATION_ERA_SEQUENCE) // 2
        authorized_era = aux.SCOUT_CALIBRATION_ERA_SEQUENCE[interior]
        replaced_interior["calibration"]["per_era"]["corr"][selected][
            interior
        ][0] = f"{int(authorized_era) + 1:04d}"
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError,
            "authorized era sequence differs",
        ):
            aux._validate_stage_receipt_schema(replaced_interior, "calibrate")

        self_declared_one_era = copy.deepcopy(scout_locked)
        locked = self_declared_one_era["locked"]
        locked["rows"] = 1
        locked["eras"] = 1
        locked["last_era"] = locked["first_era"]
        locked["summary"]["era_count"] = 1
        for metric in locked["per_era"].values():
            metric[selected] = metric[selected][:1]
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError,
            "locked block era count differs",
        ):
            aux._validate_stage_receipt_schema(
                self_declared_one_era,
                "locked",
            )

        summary_mismatch = copy.deepcopy(scout_locked)
        summary_mismatch["locked"]["summary"]["corr"]["mean"] += 0.001
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError,
            "summary derivation differs",
        ):
            aux._validate_stage_receipt_schema(summary_mismatch, "locked")

        negative_with_true_checks = copy.deepcopy(scout_locked)
        negative_metrics = negative_with_true_checks["locked"]["per_era"]
        for index, row in enumerate(negative_metrics["bmc"][selected]):
            row[1] = (-0.0005, -0.0015)[index % 2]
        negative_with_true_checks["locked"]["summary"] = (
            _summary_from_serialized_metrics(
                negative_metrics,
                selected,
                aux.SCOUT_LOCKED_ERA_SEQUENCE,
            )
        )
        negative_with_true_checks["locked"]["checks"] = {
            key: True
            for key in negative_with_true_checks["locked"]["checks"]
        }
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError,
            "checks derivation differs",
        ):
            aux._validate_stage_receipt_schema(
                negative_with_true_checks,
                "locked",
            )

        duplicate_seal = copy.deepcopy(scout_calibrate)
        duplicate_digest = duplicate_seal["inputs"]["seal_receipts"][
            "jasper"
        ]["sha256"]
        duplicate_seal["inputs"]["seal_receipts"]["teager2b"][
            "sha256"
        ] = duplicate_digest
        duplicate_seal["inputs"]["seal_receipts"]["teager2b"]["path"] = (
            f"{receipt_root}/scout-seal-teager2b-{duplicate_digest}.json"
        )
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError,
            "reuses a seal digest",
        ):
            aux._validate_stage_receipt_schema(duplicate_seal, "calibrate")

        wrong_prior_binding = copy.deepcopy(scout_calibrate)
        wrong_prior_binding["inputs"]["seal_receipts"]["teager2b"][
            "prior_finalized_seal"
        ]["path"] = f"{receipt_root}/scout-seal-jasper-{'0' * 64}.json"
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError,
            "contiguous predecessor differs",
        ):
            aux._validate_stage_receipt_schema(
                wrong_prior_binding,
                "calibrate",
            )
        wrong_prior_hash = copy.deepcopy(scout_calibrate)
        wrong_prior_hash["inputs"]["seal_receipts"]["teager2b"][
            "prior_finalized_seal"
        ]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError,
            "contiguous predecessor differs",
        ):
            aux._validate_stage_receipt_schema(
                wrong_prior_hash,
                "calibrate",
            )

        wrong_seal_prefix = copy.deepcopy(scout_calibrate)
        jasper_seal = wrong_seal_prefix["inputs"]["seal_receipts"]["jasper"]
        jasper_seal["path"] = (
            f"{receipt_root}/confirmation-seal-jasper-"
            f"{jasper_seal['sha256']}.json"
        )
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError,
            "canonical seal path differs",
        ):
            aux._validate_stage_receipt_schema(wrong_seal_prefix, "calibrate")

        wrong_pre_run_prefix = copy.deepcopy(scout_calibrate)
        jasper_pre_run = wrong_pre_run_prefix["inputs"]["seal_receipts"][
            "jasper"
        ]["pre_run_absence_receipt"]
        jasper_pre_run["path"] = (
            f"{receipt_root}/confirmation-pre-run-jasper-"
            f"{jasper_pre_run['sha256']}.json"
        )
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError,
            "pre-run canonical path differs",
        ):
            aux._validate_stage_receipt_schema(
                wrong_pre_run_prefix,
                "calibrate",
            )

        wrong_pretraining_prefix = copy.deepcopy(confirmation_calibrate)
        wrong_pretraining = wrong_pretraining_prefix["inputs"][
            "confirmation_pretraining_receipt"
        ]
        wrong_pretraining["path"] = (
            f"{receipt_root}/locked-{wrong_pretraining['sha256']}.json"
        )
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError,
            "pretraining canonical path differs",
        ):
            aux._validate_stage_receipt_schema(
                wrong_pretraining_prefix,
                "confirmation-calibrate",
            )

    def test_confirmation_jit_chain_requires_each_immediate_finalized_predecessor(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(
                root,
                canonical_experiment=True,
            )
            protocol.source_manifest["confirmation_output_contract"] = {
                name: {
                    "config_path": f"configs/{name}.py",
                    "results_path": f"results/{name}.json",
                    "predictions_path": f"predictions/{name}.parquet",
                    "must_be_absent_before_run": True,
                }
                for name in aux.CONFIRMATION_RUN_ORDER
            }
            receipt_dir = protocol.experiment_dir / "receipts"
            pretraining_digest = "f" * 64
            pretraining_path = (
                receipt_dir
                / f"confirmation-pretraining-{pretraining_digest}.json"
            )
            pretraining_binding = aux._confirmation_pretraining_binding(
                protocol,
                pretraining_path,
                pretraining_digest,
            )
            file_receipt = {
                "path": "artifacts/file.bin",
                "sha256": "a" * 64,
                "size_bytes": 1,
            }
            cohort = {
                "rows": 1,
                "eras": 1,
                "first_era": "0001",
                "last_era": "0001",
                "full_rows": 1,
                "full_eras": 1,
                "folds": [
                    {
                        "fold": 1,
                        "train_eras": 0,
                        "val_eras": 1,
                        "train_rows": 0,
                        "val_rows": 1,
                    }
                ],
            }
            expected = aux.ExpectedCohort(
                frame=pd.DataFrame(
                    {
                        aux.ID_COLUMN: ["synthetic-id"],
                        aux.ERA_COLUMN: ["0001"],
                    }
                ),
                full_rows=1,
                full_eras=1,
                eras=("0001",),
                folds=tuple(copy.deepcopy(cohort["folds"])),
            )
            paths = {
                name: receipt_dir
                / (
                    f"confirmation-seal-{name}-"
                    f"{str(index + 1) * 64}.json"
                )
                for index, name in enumerate(aux.CONFIRMATION_RUN_ORDER)
            }
            digests = {
                name: str(index + 1) * 64
                for index, name in enumerate(aux.CONFIRMATION_RUN_ORDER)
            }
            preflight_digest_chars = ("6", "7", "8", "9", "a")
            preflight_paths = {
                name: receipt_dir
                / (
                    f"confirmation-pre-run-{name}-"
                    f"{preflight_digest_chars[index] * 64}.json"
                )
                for index, name in enumerate(aux.CONFIRMATION_RUN_ORDER)
            }
            preflight_digests = {
                name: preflight_digest_chars[index] * 64
                for index, name in enumerate(aux.CONFIRMATION_RUN_ORDER)
            }
            seals: dict[str, dict[str, object]] = {}
            for index, name in enumerate(aux.CONFIRMATION_RUN_ORDER):
                predecessor = (
                    None
                    if index == 0
                    else {
                        "component": aux.CONFIRMATION_RUN_ORDER[index - 1],
                        "path": aux._lexical_relative_path(
                            paths[aux.CONFIRMATION_RUN_ORDER[index - 1]],
                            root,
                        ),
                        "sha256": digests[
                            aux.CONFIRMATION_RUN_ORDER[index - 1]
                        ],
                    }
                )
                seals[name] = {
                    "schema_version": 1,
                    "experiment": aux.EXPERIMENT_NAME,
                    "stage": "seal-confirmation-component",
                    "state": "SEALED",
                    "passed": True,
                    "protocol": {},
                    "component": name,
                    "confirmation_pretraining_receipt": copy.deepcopy(
                        pretraining_binding
                    ),
                    "prior_finalized_seal": predecessor,
                    "pre_run_absence_receipt": {
                        "path": aux._lexical_relative_path(
                            preflight_paths[name],
                            root,
                        ),
                        "sha256": preflight_digests[name],
                    },
                    "run_consumption_claim": (
                        _synthetic_consumption_file_receipt(
                            name,
                            confirmation=True,
                        )
                    ),
                    "run_completion_receipt": (
                        _synthetic_completion_file_receipt(
                            name,
                            confirmation=True,
                        )
                    ),
                    "cohort": copy.deepcopy(cohort),
                    "artifact": {
                        "component": name,
                        "target": aux.COMPONENT_TARGETS[name],
                        "config": copy.deepcopy(file_receipt),
                        "result": copy.deepcopy(file_receipt),
                        "predictions": copy.deepcopy(file_receipt),
                    },
                    "gpu_folds_verified": 1,
                }
            preflights = {
                name: {
                    "schema_version": 1,
                    "experiment": aux.EXPERIMENT_NAME,
                    "stage": "claim-confirmation-component-run",
                    "state": "ABSENCE_PROVEN",
                    "passed": True,
                    "protocol": {},
                    "component": name,
                    "confirmation_pretraining_receipt": copy.deepcopy(
                        pretraining_binding
                    ),
                    "prior_finalized_seal": copy.deepcopy(
                        seals[name]["prior_finalized_seal"]
                    ),
                    "destinations": {
                        "result": {
                            "path": f"results/{name}.json",
                            "absent": True,
                        },
                        "predictions": {
                            "path": f"predictions/{name}.parquet",
                            "absent": True,
                        },
                    },
                }
                for name in aux.CONFIRMATION_RUN_ORDER
            }

            loaded: list[str] = []

            def load_seal(
                path: Path,
                digest: str,
                *,
                expected_stage: str,
                receipt_dir: Path,
                expected_prefix: str,
            ) -> dict[str, object]:
                del receipt_dir
                if expected_stage == "seal-confirmation-component":
                    name = expected_prefix.removeprefix("confirmation-seal-")
                    expected_path = paths.get(name)
                    expected_digest = digests.get(name)
                    payload = seals.get(name)
                elif expected_stage == "claim-confirmation-component-run":
                    name = expected_prefix.removeprefix(
                        "confirmation-pre-run-"
                    )
                    expected_path = preflight_paths.get(name)
                    expected_digest = preflight_digests.get(name)
                    payload = preflights.get(name)
                else:
                    raise AssertionError(expected_stage)
                if (
                    payload is None
                    or Path(aux.os.path.abspath(path)) != expected_path
                    or digest != expected_digest
                ):
                    raise aux.EnderEnsembleEvaluationError(
                        f"{name} predecessor seal path/hash differs"
                    )
                if expected_stage == "seal-confirmation-component":
                    loaded.append(name)
                return copy.deepcopy(payload)

            with patch.object(
                aux,
                "_load_bound_receipt",
                side_effect=load_seal,
            ), patch.object(
                aux,
                "_validate_confirmation_pretraining_receipt",
                return_value={},
            ), patch.object(
                aux,
                "_confirmation_component_destination_receipt",
                side_effect=lambda _protocol, component: copy.deepcopy(
                    preflights[component.name]["destinations"]
                ),
            ), patch.object(
                aux,
                "build_confirmation_expected_cohort",
                return_value=expected,
            ), patch.object(
                aux,
                "_validate_confirmation_component",
                side_effect=lambda _protocol, component, _expected, _pretraining: (
                    np.array([0.5]),
                    copy.deepcopy(seals[component.name]["artifact"]),
                ),
            ), patch.object(
                aux,
                "_lease_validated_component_outputs",
                side_effect=lambda _protocol, component, *_args, **_kwargs: (
                    nullcontext(
                        (
                            copy.deepcopy(_synthetic_consumption_file_receipt(
                                component.name,
                                confirmation=True,
                            )),
                            copy.deepcopy(_synthetic_completion_file_receipt(
                                component.name,
                                confirmation=True,
                            )),
                        )
                    )
                ),
            ), patch.object(aux, "_validate_protocol_binding"):
                binding = aux._validate_prior_finalized_seal(
                    protocol,
                    aux.CONFIRMATION_RUN_ORDER,
                    "tyler",
                    paths["xerxes"],
                    digests["xerxes"],
                    confirmation_pretraining_receipt_path=pretraining_path,
                    confirmation_pretraining_receipt_sha256=pretraining_digest,
                )
            self.assertEqual(
                loaded,
                ["xerxes", "victor", "teager2b", "jasper"],
            )
            self.assertEqual(
                binding,
                {
                    "component": "xerxes",
                    "path": aux._lexical_relative_path(paths["xerxes"], root),
                    "sha256": digests["xerxes"],
                },
            )

            for component, path, digest, expected_error in (
                (
                    "jasper",
                    paths["jasper"],
                    digests["jasper"],
                    "first.*may not bind",
                ),
                (
                    "teager2b",
                    None,
                    None,
                    "requires the immediately preceding",
                ),
                (
                    "tyler",
                    paths["victor"],
                    digests["victor"],
                    "predecessor seal path/hash differs",
                ),
            ):
                with self.subTest(component=component, path=path), patch.object(
                    aux,
                    "_load_bound_receipt",
                    side_effect=load_seal,
                ), patch.object(aux, "_validate_protocol_binding"):
                    with self.assertRaisesRegex(
                        aux.EnderEnsembleEvaluationError,
                        expected_error,
                    ):
                        aux._validate_prior_finalized_seal(
                            protocol,
                            aux.CONFIRMATION_RUN_ORDER,
                            component,
                            path,
                            digest,
                            confirmation_pretraining_receipt_path=pretraining_path,
                            confirmation_pretraining_receipt_sha256=(
                                pretraining_digest
                            ),
                        )

            mismatched = copy.deepcopy(seals["xerxes"])
            mismatched["confirmation_pretraining_receipt"]["sha256"] = "0" * 64
            with patch.object(
                aux,
                "_load_bound_receipt",
                return_value=mismatched,
            ), patch.object(aux, "_validate_protocol_binding"):
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "pretraining.*differs",
                ):
                    aux._validate_prior_finalized_seal(
                        protocol,
                        aux.CONFIRMATION_RUN_ORDER,
                        "tyler",
                        paths["xerxes"],
                        digests["xerxes"],
                        confirmation_pretraining_receipt_path=pretraining_path,
                        confirmation_pretraining_receipt_sha256=(
                            pretraining_digest
                        ),
                    )

    def test_predecessor_seal_rebinds_live_cohort_artifact_and_gpu_semantics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(
                root,
                canonical_experiment=True,
            )
            protocol.source_manifest["confirmation_output_contract"] = {
                name: {
                    "config_path": f"configs/{name}.py",
                    "results_path": f"results/{name}.json",
                    "predictions_path": f"predictions/{name}.parquet",
                    "must_be_absent_before_run": True,
                }
                for name in aux.CONFIRMATION_RUN_ORDER
            }
            receipt_dir = protocol.experiment_dir / "receipts"
            predecessor_digest = "a" * 64
            predecessor_path = receipt_dir / (
                f"confirmation-seal-jasper-{predecessor_digest}.json"
            )
            pretraining_digest = "b" * 64
            pretraining_path = receipt_dir / (
                f"confirmation-pretraining-{pretraining_digest}.json"
            )
            expected = aux.ExpectedCohort(
                frame=pd.DataFrame(
                    {
                        aux.ID_COLUMN: ["synthetic-id"],
                        aux.ERA_COLUMN: ["0001"],
                    }
                ),
                full_rows=1,
                full_eras=1,
                eras=("0001",),
                folds=(
                    {
                        "fold": 1,
                        "train_eras": 0,
                        "val_eras": 1,
                        "train_rows": 0,
                        "val_rows": 1,
                    },
                ),
            )
            file_receipt = {
                "path": "artifacts/file.bin",
                "sha256": "c" * 64,
                "size_bytes": 1,
            }
            artifact = {
                "component": "jasper",
                "target": aux.COMPONENT_TARGETS["jasper"],
                "config": copy.deepcopy(file_receipt),
                "result": copy.deepcopy(file_receipt),
                "predictions": copy.deepcopy(file_receipt),
            }
            seal = {
                "schema_version": 1,
                "experiment": aux.EXPERIMENT_NAME,
                "stage": "seal-confirmation-component",
                "state": "SEALED",
                "passed": True,
                "protocol": {},
                "component": "jasper",
                "confirmation_pretraining_receipt": (
                    aux._confirmation_pretraining_binding(
                        protocol,
                        pretraining_path,
                        pretraining_digest,
                    )
                ),
                "prior_finalized_seal": None,
                "pre_run_absence_receipt": {
                    "path": (
                        "numerai/agents/experiments/"
                        f"{aux.EXPERIMENT_NAME}/receipts/"
                        f"confirmation-pre-run-jasper-{'d' * 64}.json"
                    ),
                    "sha256": "d" * 64,
                },
                "run_consumption_claim": (
                    _synthetic_consumption_file_receipt(
                        "jasper",
                        confirmation=True,
                    )
                ),
                "run_completion_receipt": (
                    _synthetic_completion_file_receipt(
                        "jasper",
                        confirmation=True,
                    )
                ),
                "cohort": aux._cohort_receipt(expected),
                "artifact": copy.deepcopy(artifact),
                "gpu_folds_verified": 1,
            }
            cases: list[tuple[str, dict[str, object], str]] = []
            wrong_cohort = copy.deepcopy(seal)
            wrong_cohort["cohort"]["rows"] = 2
            cases.append(
                ("cohort", wrong_cohort, "predecessor cohort.*differs")
            )
            wrong_artifact = copy.deepcopy(seal)
            wrong_artifact["artifact"]["target"] = aux.COMPONENT_TARGETS[
                "victor"
            ]
            cases.append(
                ("artifact", wrong_artifact, "predecessor artifact.*differs")
            )
            wrong_gpu = copy.deepcopy(seal)
            wrong_gpu["gpu_folds_verified"] = 2
            cases.append(
                ("gpu", wrong_gpu, "predecessor GPU folds.*differs")
            )

            for label, forged, expected_error in cases:
                with self.subTest(label=label), patch.object(
                    aux,
                    "_load_bound_receipt",
                    return_value=forged,
                ), patch.object(
                    aux,
                    "_validate_protocol_binding",
                ), patch.object(
                    aux,
                    "_validate_confirmation_pre_run_absence_receipt",
                    return_value={"prior_finalized_seal": None},
                ), patch.object(
                    aux,
                    "_validate_confirmation_pretraining_receipt",
                    return_value={},
                ), patch.object(
                    aux,
                    "_lease_validated_component_outputs",
                    side_effect=lambda *_args, **_kwargs: nullcontext(
                        (
                            copy.deepcopy(seal["run_consumption_claim"]),
                            copy.deepcopy(seal["run_completion_receipt"]),
                        )
                    ),
                ), patch.object(
                    aux,
                    "build_confirmation_expected_cohort",
                    return_value=expected,
                ), patch.object(
                    aux,
                    "_validate_confirmation_component",
                    return_value=(np.array([0.5]), copy.deepcopy(artifact)),
                ) as live_validation:
                    with self.assertRaisesRegex(
                        aux.EnderEnsembleEvaluationError,
                        expected_error,
                    ):
                        aux._validate_prior_finalized_seal(
                            protocol,
                            aux.CONFIRMATION_RUN_ORDER,
                            "teager2b",
                            predecessor_path,
                            predecessor_digest,
                            confirmation_pretraining_receipt_path=(
                                pretraining_path
                            ),
                            confirmation_pretraining_receipt_sha256=(
                                pretraining_digest
                            ),
                        )
                    live_validation.assert_called_once()

    def test_scoring_block_binds_era_boundaries_order_and_summary_count(
        self,
    ) -> None:
        selected = "tyler_w10"
        expected_eras = ("0001", "0002")
        per_era = _serialized_synthetic_metrics(expected_eras, [selected])
        summary = _summary_from_serialized_metrics(
            per_era,
            selected,
            expected_eras,
        )
        check_keys = {
            "bmc_mean",
            "bmc_sharpe",
            "bmc_max_drawdown",
            "corr_mean",
        }
        block = {
            "rows": 2,
            "eras": 2,
            "first_era": expected_eras[0],
            "last_era": expected_eras[-1],
            "summary": summary,
            "checks": aux.locked_checks(summary),
            "per_era": per_era,
        }
        self.assertTrue(
            aux._validate_scoring_block_schema(
                block,
                selected,
                "synthetic block",
                check_keys,
                expected_eras,
                aux.locked_checks,
            )
        )

        cases: list[tuple[str, dict[str, object], str]] = []
        wrong_boundary = copy.deepcopy(block)
        wrong_boundary["first_era"] = "0000"
        cases.append(("boundary", wrong_boundary, "first era differs"))
        wrong_order = copy.deepcopy(block)
        wrong_order["per_era"]["corr"][selected] = [
            ["0002", 0.02],
            ["0001", 0.02],
        ]
        cases.append(("order", wrong_order, "not strictly increasing"))
        wrong_summary_count = copy.deepcopy(block)
        wrong_summary_count["summary"]["era_count"] = 1
        cases.append(
            (
                "summary count",
                wrong_summary_count,
                "summary era count differs",
            )
        )
        for label, forged, expected_error in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    expected_error,
                ):
                    aux._validate_scoring_block_schema(
                        forged,
                        selected,
                        "synthetic block",
                        check_keys,
                        expected_eras,
                        aux.locked_checks,
                    )

    def test_confirmation_pretraining_creation_binds_checkpoint_stores_and_outputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(root)
            receipt_dir = protocol.experiment_dir / "receipts"
            scout_locked_path = receipt_dir / "locked-authorizing.json"
            contracts = {
                name: {
                    "config_path": aux._confirmation_config_relative(name),
                    "results_path": (
                        f"numerai/agents/experiments/{aux.EXPERIMENT_NAME}/"
                        f"results/confirmation_{name}_d8_t6000.json"
                    ),
                    "predictions_path": (
                        f"numerai/agents/experiments/{aux.EXPERIMENT_NAME}/"
                        f"predictions/confirmation_{name}_d8_t6000.parquet"
                    ),
                    "results_name": f"confirmation_{name}_d8_t6000",
                    "must_be_absent_before_run": True,
                }
                for name in aux.ALL_COMPONENTS
            }
            canonical_store = {
                "generation_id": "a" * 32,
                "row_count": 6_195_697,
                "feature_count": 780,
                "feature_order_sha256": "b" * 64,
                "target_column": aux.COMPONENT_TARGETS["xerxes"],
                "metadata": {"path": "stores/xerxes/metadata.json"},
                "manifest": {"path": "stores/xerxes/manifest.parquet"},
                "features": {"path": "stores/xerxes/features.bin"},
            }
            protocol.source_manifest.update(
                {
                    "confirmation_output_contract": contracts,
                    "confirmation_xerxes_medium_store_anchor": canonical_store,
                }
            )
            scout_locked = {
                "selected_formula": {
                    "name": "tyler_w10",
                    "weights": dict(aux.BLEND_WEIGHTS["tyler_w10"]),
                }
            }
            checkpointed: list[str] = []

            def checkpoint_receipt(
                _protocol: aux.FrozenProtocol, path: Path, _label: str
            ) -> dict[str, object]:
                relative = aux._relative_path(path, root)
                checkpointed.append(relative)
                return {
                    "path": relative,
                    "size_bytes": 1,
                    "sha256": "c" * 64,
                    "checkpoint_commit": protocol.pretraining_commit,
                    "git_blob_id": "d" * 40,
                }

            def store_receipt(
                _protocol: aux.FrozenProtocol, name: str
            ) -> dict[str, object]:
                if name == "xerxes":
                    return canonical_store
                return {
                    **canonical_store,
                    "generation_id": str(aux.ALL_COMPONENTS.index(name) + 1) * 32,
                    "target_column": aux.COMPONENT_TARGETS[name],
                }

            stores = {
                name: copy.deepcopy(store_receipt(protocol, name))
                for name in aux.ALL_COMPONENTS
            }
            inventory_path = root / "confirmation_store_inventory.json"
            inventory = {
                "schema_version": 1,
                "experiment": aux.EXPERIMENT_NAME,
                "artifact": "confirmation-store-inventory-v1",
                "scout_locked_receipt": {
                    "path": aux._relative_path(scout_locked_path, root),
                    "sha256": "e" * 64,
                },
                "selected_formula": copy.deepcopy(
                    scout_locked["selected_formula"]
                ),
                "canonical_store": copy.deepcopy(canonical_store),
                "input_layout": {
                    "type": "dedicated_target_stores",
                    "stores": stores,
                },
            }
            inventory_path.write_text(
                json.dumps(inventory),
                encoding="utf-8",
            )

            destinations = {
                name: {
                    "results_path": contract["results_path"],
                    "predictions_path": contract["predictions_path"],
                    "results_absent_at_checkpoint": True,
                    "predictions_absent_at_checkpoint": True,
                }
                for name, contract in contracts.items()
            }
            with patch.object(
                aux,
                "_load_passing_scout_locked_receipt",
                return_value=scout_locked,
            ), patch.object(
                aux, "_checkpointed_file_receipt", side_effect=checkpoint_receipt
            ), patch.object(
                aux, "_load_confirmation_config", return_value={"data": {}}
            ), patch.object(
                aux,
                "_confirmation_store_inventory_file",
                return_value=inventory_path,
            ), patch.object(
                aux, "_read_full_confirmation_sources", return_value=pd.DataFrame()
            ), patch.object(
                aux, "_confirmation_store_receipt", side_effect=store_receipt
            ), patch.object(
                aux, "_validate_confirmation_store_receipt"
            ), patch.object(
                aux, "_validate_distinct_confirmation_store_files"
            ), patch.object(
                aux, "_confirmation_destination_receipts", return_value=destinations
            ), patch.object(
                aux, "_protocol_binding", return_value={"checkpoint": "mocked"}
            ):
                path, receipt = aux.create_confirmation_pretraining_receipt(
                    protocol,
                    scout_locked_path,
                    "e" * 64,
                    receipt_dir,
                )

        self.assertTrue(path.name.startswith("confirmation-pretraining-"))
        self.assertEqual(receipt["checkpoint"], protocol.pretraining_commit)
        self.assertEqual(set(receipt["configs"]), set(aux.ALL_COMPONENTS))
        self.assertEqual(
            len(receipt["config_helpers"]), len(aux.CONFIRMATION_CONFIG_HELPER_PATHS)
        )
        self.assertEqual(
            len(receipt["loader"]["files"]), len(aux.CONFIRMATION_LOADER_PATHS)
        )
        self.assertEqual(
            receipt["input_layout"]["type"], "dedicated_target_stores"
        )
        self.assertEqual(
            set(receipt["input_layout"]["stores"]), set(aux.ALL_COMPONENTS)
        )
        self.assertEqual(receipt["canonical_store"], canonical_store)
        self.assertEqual(receipt["output_destinations"], destinations)
        self.assertEqual(
            receipt["store_inventory"]["path"],
            aux._relative_path(inventory_path, root),
        )
        self.assertEqual(
            receipt["scout_locked_receipt"],
            {
                "path": aux._relative_path(scout_locked_path, root),
                "sha256": "e" * 64,
            },
        )
        self.assertTrue(
            set(aux.CONFIRMATION_CONFIG_HELPER_PATHS).issubset(checkpointed)
        )
        self.assertTrue(set(aux.CONFIRMATION_LOADER_PATHS).issubset(checkpointed))

    def test_confirmation_sidecar_layout_is_forbidden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            protocol = _synthetic_protocol(Path(directory))
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "sidecar is forbidden"
            ):
                aux._configured_confirmation_store_directory(
                    protocol,
                    "jasper",
                    {
                        "data": {
                            "disk_feature_store_path": "numerai/v5.3/jasper-store",
                            "label_sidecar_path": "numerai/v5.3/labels.parquet",
                        }
                    },
                )

    def test_confirmation_store_rejects_label_source_order_and_link_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(root)
            store_dir = root / "stores" / "jasper"
            store_dir.mkdir(parents=True)
            generation = "a" * 32
            canonical_feature_hash = "b" * 64
            canonical_feature_size = 6_195_697 * 780
            file_receipts = {
                "metadata": {
                    "path": "stores/jasper/metadata.json",
                    "size_bytes": 10,
                    "sha256": "c" * 64,
                },
                "manifest": {
                    "path": f"stores/jasper/manifest-{generation}.parquet",
                    "size_bytes": 20,
                    "sha256": "d" * 64,
                },
                "features": {
                    "path": f"stores/jasper/features-{generation}.int8.bin",
                    "size_bytes": canonical_feature_size,
                    "sha256": canonical_feature_hash,
                },
            }
            store = {
                "generation_id": generation,
                "row_count": 6_195_697,
                "feature_count": 780,
                "feature_order_sha256": "e" * 64,
                "target_column": aux.COMPONENT_TARGETS["jasper"],
                **file_receipts,
            }
            protocol.source_manifest[
                "confirmation_xerxes_medium_store_anchor"
            ] = {
                **store,
                "target_column": aux.COMPONENT_TARGETS["xerxes"],
            }
            source_fingerprints = [{"role": "data", "position": 0}]
            metadata = {
                "format": "numerai-v5.3-int8-feature-store",
                "format_version": 1,
                "complete": True,
                "generation_id": generation,
                "row_count": 6_195_697,
                "feature_count": 780,
                "feature_columns": list(protocol.medium_features),
                "feature_order_sha256": "e" * 64,
                "target_column": aux.COMPONENT_TARGETS["jasper"],
                "benchmark_column": aux.BENCHMARK_ENDER20,
                "source_fingerprints": source_fingerprints,
                "features": {
                    "filename": f"features-{generation}.int8.bin",
                    "dtype": "int8",
                    "layout": "C",
                    "size_bytes": canonical_feature_size,
                    "sha256": canonical_feature_hash,
                },
                "manifest": {
                    "filename": f"manifest-{generation}.parquet",
                    "columns": [
                        "row_offset",
                        aux.ID_COLUMN,
                        aux.ERA_COLUMN,
                        aux.COMPONENT_TARGETS["jasper"],
                        aux.BENCHMARK_ENDER20,
                    ],
                    "size_bytes": 20,
                    "sha256": "d" * 64,
                },
            }
            raw = pd.DataFrame(
                {
                    aux.ID_COLUMN: ["a", "b", "c"],
                    aux.ERA_COLUMN: ["0001", "0001", "0002"],
                    aux.COMPONENT_TARGETS["jasper"]: np.array(
                        [0.1, 0.2, 0.3], dtype=np.float32
                    ),
                    aux.BENCHMARK_ENDER20: np.array(
                        [0.4, 0.5, 0.6], dtype=np.float64
                    ),
                }
            )
            manifest = pd.DataFrame(
                {
                    "row_offset": np.arange(3, dtype=np.int64),
                    aux.ID_COLUMN: raw[aux.ID_COLUMN],
                    aux.ERA_COLUMN: raw[aux.ERA_COLUMN],
                    aux.COMPONENT_TARGETS["jasper"]: raw[
                        aux.COMPONENT_TARGETS["jasper"]
                    ],
                    aux.BENCHMARK_ENDER20: raw[aux.BENCHMARK_ENDER20],
                }
            )

            class FakeParquetFile:
                def __init__(self, _path: Path) -> None:
                    fields = [
                        pa.field("row_offset", pa.int64()),
                        pa.field(aux.ID_COLUMN, pa.string()),
                        pa.field(aux.ERA_COLUMN, pa.string()),
                        pa.field(aux.COMPONENT_TARGETS["jasper"], pa.float32()),
                        pa.field(aux.BENCHMARK_ENDER20, pa.float64()),
                    ]
                    self.schema_arrow = pa.schema(fields)
                    self.metadata = type(
                        "Metadata", (), {"num_rows": 6_195_697}
                    )()

                def close(self) -> None:
                    return None

            def current_file_receipt(path: Path, _root: Path) -> dict:
                if path.name == "metadata.json":
                    return file_receipts["metadata"]
                if path.suffix == ".parquet":
                    return file_receipts["manifest"]
                return file_receipts["features"]

            def validate(
                current_metadata: dict,
                current_manifest: pd.DataFrame,
                *,
                linked: bool = False,
            ) -> None:
                regular = (
                    aux.EnderEnsembleEvaluationError("store file is hardlinked")
                    if linked
                    else None
                )
                with patch.object(
                    aux,
                    "_require_regular_unlinked_receipt_file",
                    side_effect=regular,
                ), patch.object(
                    aux, "_load_json", return_value=current_metadata
                ), patch.object(
                    aux,
                    "_confirmation_source_fingerprints",
                    return_value=source_fingerprints,
                ), patch.object(
                    aux, "feature_order_sha256", return_value="e" * 64
                ), patch.object(
                    aux, "_file_receipt", side_effect=current_file_receipt
                ), patch.object(
                    aux,
                    "_configured_confirmation_store_directory",
                    return_value=store_dir,
                ), patch.object(
                    aux, "validate_component_config"
                ), patch.object(
                    aux.pq, "ParquetFile", FakeParquetFile
                ), patch.object(
                    aux.pd, "read_parquet", return_value=current_manifest
                ):
                    aux._validate_confirmation_store_receipt(
                        protocol,
                        "jasper",
                        store,
                        raw,
                        config={"data": {}},
                    )

            validate(copy.deepcopy(metadata), manifest.copy())

            wrong_label = manifest.copy()
            wrong_label.loc[1, aux.COMPONENT_TARGETS["jasper"]] = 0.9
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError,
                "target_jasper_20 differs from raw sources",
            ):
                validate(copy.deepcopy(metadata), wrong_label)

            wrong_source = copy.deepcopy(metadata)
            wrong_source["source_fingerprints"] = [{"role": "wrong"}]
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError,
                "metadata.source_fingerprints differs",
            ):
                validate(wrong_source, manifest.copy())

            wrong_order = manifest.iloc[[1, 0, 2]].reset_index(drop=True)
            wrong_order["row_offset"] = np.arange(3, dtype=np.int64)
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError,
                "manifest id order differs",
            ):
                validate(copy.deepcopy(metadata), wrong_order)

            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "hardlinked"
            ):
                validate(copy.deepcopy(metadata), manifest.copy(), linked=True)

    def test_confirmation_seal_bindings_require_all_five_unique_components(
        self,
    ) -> None:
        values = [
            [component, f"{component}.json", str(index) * 64]
            for index, component in enumerate(aux.ALL_COMPONENTS, start=1)
        ]
        bindings = aux._parse_confirmation_seal_bindings(values)
        self.assertEqual(set(bindings), set(aux.ALL_COMPONENTS))

        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError, "Duplicate confirmation seal"
        ):
            aux._parse_confirmation_seal_bindings(
                [values[0], ["jasper", "other.json", "9" * 64]]
            )
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError, "Unknown confirmation seal"
        ):
            aux._parse_confirmation_seal_bindings(
                [["unknown", "unknown.json", "9" * 64]]
            )
        incomplete = aux._parse_confirmation_seal_bindings(values[:-1])
        with tempfile.TemporaryDirectory() as directory:
            protocol = _synthetic_protocol(Path(directory))
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError,
                "confirmation seal components differs",
            ):
                aux.validate_confirmation_seal_receipts(
                    protocol,
                    incomplete,
                    protocol.experiment_dir / "receipts" / "pretraining.json",
                    "9" * 64,
                    _confirmation_stage_expected(),
                )

    def test_confirmation_seals_bind_own_preflight_and_same_pretraining(
        self,
    ) -> None:
        expected = _confirmation_stage_expected()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(
                root,
                canonical_experiment=True,
            )
            receipt_dir = protocol.experiment_dir / "receipts"
            pretraining_digest = "9" * 64
            pretraining_path = receipt_dir / (
                f"confirmation-pretraining-{pretraining_digest}.json"
            )
            expected_pretraining = {
                "path": aux._lexical_relative_path(pretraining_path, root),
                "sha256": pretraining_digest,
            }
            bindings = {
                name: (
                    receipt_dir
                    / f"confirmation-seal-{name}-{str(index) * 64}.json",
                    str(index) * 64,
                )
                for index, name in enumerate(aux.ALL_COMPONENTS, start=1)
            }
            components = {
                name: aux.ComponentPaths(
                    name=name,
                    config=root / f"{name}.py",
                    result=root / f"{name}.json",
                    predictions=root / f"{name}.parquet",
                )
                for name in aux.ALL_COMPONENTS
            }
            synthetic_file_receipt = {
                "path": "artifacts/file.bin",
                "sha256": "a" * 64,
                "size_bytes": 1,
            }
            artifacts = {
                name: {
                    "component": name,
                    "target": aux.COMPONENT_TARGETS[name],
                    "config": copy.deepcopy(synthetic_file_receipt),
                    "result": copy.deepcopy(synthetic_file_receipt),
                    "predictions": copy.deepcopy(synthetic_file_receipt),
                }
                for name in aux.ALL_COMPONENTS
            }
            preflight_digest_chars = ("6", "7", "8", "9", "a")
            seals = {
                name: {
                    "schema_version": 1,
                    "experiment": aux.EXPERIMENT_NAME,
                    "stage": "seal-confirmation-component",
                    "state": "SEALED",
                    "passed": True,
                    "protocol": {"checkpoint": "mocked"},
                    "component": name,
                    "confirmation_pretraining_receipt": expected_pretraining,
                    "prior_finalized_seal": (
                        None
                        if index == 0
                        else {
                            "component": aux.CONFIRMATION_RUN_ORDER[index - 1],
                            "path": aux._lexical_relative_path(
                                bindings[
                                    aux.CONFIRMATION_RUN_ORDER[index - 1]
                                ][0],
                                root,
                            ),
                            "sha256": bindings[
                                aux.CONFIRMATION_RUN_ORDER[index - 1]
                            ][1],
                        }
                    ),
                    "pre_run_absence_receipt": {
                        "path": aux._lexical_relative_path(
                            receipt_dir
                            / (
                                f"confirmation-pre-run-{name}-"
                                f"{preflight_digest_chars[index] * 64}.json"
                            ),
                            root,
                        ),
                        "sha256": preflight_digest_chars[index] * 64,
                    },
                    "run_consumption_claim": (
                        _synthetic_consumption_file_receipt(
                            name,
                            confirmation=True,
                        )
                    ),
                    "run_completion_receipt": (
                        _synthetic_completion_file_receipt(
                            name,
                            confirmation=True,
                        )
                    ),
                    "cohort": aux._cohort_receipt(expected),
                    "artifact": artifacts[name],
                    "gpu_folds_verified": len(expected.folds),
                }
                for index, name in enumerate(aux.ALL_COMPONENTS)
            }
            preflight_calls: list[
                tuple[str, Path, str, Path, str]
            ] = []

            def load_seal(
                _path: Path,
                _digest: str,
                *,
                expected_stage: str,
                receipt_dir: Path | None = None,
                expected_prefix: str | None = None,
            ) -> dict:
                self.assertEqual(expected_stage, "seal-confirmation-component")
                self.assertIsNotNone(receipt_dir)
                assert expected_prefix is not None
                name = expected_prefix.removeprefix("confirmation-seal-")
                return seals[name]

            def validate_preflight(
                _protocol: aux.FrozenProtocol,
                component: aux.ComponentPaths,
                path: Path,
                digest: str,
                bound_pretraining_path: Path,
                bound_pretraining_digest: str,
            ) -> dict:
                index = aux.CONFIRMATION_RUN_ORDER.index(component.name)
                expected_path = (
                    receipt_dir
                    / (
                        f"confirmation-pre-run-{component.name}-"
                        f"{preflight_digest_chars[index] * 64}.json"
                    )
                ).resolve()
                if path.resolve() != expected_path:
                    raise aux.EnderEnsembleEvaluationError(
                        f"{component.name} preflight component differs"
                    )
                preflight_calls.append(
                    (
                        component.name,
                        path.resolve(),
                        digest,
                        bound_pretraining_path.resolve(),
                        bound_pretraining_digest,
                    )
                )
                return {
                    "prior_finalized_seal": (
                        None
                        if index == 0
                        else {
                            "component": aux.CONFIRMATION_RUN_ORDER[index - 1],
                            "path": aux._lexical_relative_path(
                                bindings[
                                    aux.CONFIRMATION_RUN_ORDER[index - 1]
                                ][0],
                                root,
                            ),
                            "sha256": bindings[
                                aux.CONFIRMATION_RUN_ORDER[index - 1]
                            ][1],
                        }
                    )
                }

            def validate_component(
                _protocol: aux.FrozenProtocol,
                component: aux.ComponentPaths,
                _expected: aux.ExpectedCohort,
                _pretraining: dict,
            ) -> tuple[np.ndarray, dict]:
                return np.zeros(len(expected.frame)), artifacts[component.name]

            def run_validation() -> tuple[dict[str, np.ndarray], dict[str, dict]]:
                with patch.object(
                    aux,
                    "_validate_confirmation_pretraining_receipt",
                    return_value={},
                ), patch.object(
                    aux, "_load_bound_receipt", side_effect=load_seal
                ), patch.object(
                    aux, "_validate_protocol_binding"
                ), patch.object(
                    aux,
                    "default_confirmation_component_paths",
                    side_effect=lambda _protocol, name: components[name],
                ), patch.object(
                    aux,
                    "_validate_confirmation_pre_run_absence_receipt",
                    side_effect=validate_preflight,
                ), patch.object(
                    aux,
                    "_lease_validated_component_outputs",
                    side_effect=lambda _protocol, component, *_args, **_kwargs: (
                        nullcontext(
                            (
                                copy.deepcopy(_synthetic_consumption_file_receipt(
                                    component.name,
                                    confirmation=True,
                                )),
                                copy.deepcopy(_synthetic_completion_file_receipt(
                                    component.name,
                                    confirmation=True,
                                )),
                            )
                        )
                    ),
                ), patch.object(
                    aux,
                    "_validate_confirmation_component",
                    side_effect=validate_component,
                ):
                    return aux.validate_confirmation_seal_receipts(
                        protocol,
                        bindings,
                        pretraining_path,
                        pretraining_digest,
                        expected,
                    )

            signals, normalized = run_validation()
            self.assertEqual(set(signals), set(aux.ALL_COMPONENTS))
            self.assertEqual(set(normalized), set(aux.ALL_COMPONENTS))
            self.assertEqual(
                [call[0] for call in preflight_calls], list(aux.ALL_COMPONENTS)
            )
            self.assertTrue(
                all(call[3] == pretraining_path.resolve() for call in preflight_calls)
            )
            self.assertTrue(
                all(call[4] == pretraining_digest for call in preflight_calls)
            )

            preflight_calls.clear()
            seals["jasper"]["confirmation_pretraining_receipt"] = {
                "path": expected_pretraining["path"],
                "sha256": "0" * 64,
            }
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError,
                "confirmation seal pretraining.*differs",
            ):
                run_validation()
            seals["jasper"][
                "confirmation_pretraining_receipt"
            ] = expected_pretraining

            seals["jasper"]["pre_run_absence_receipt"] = {
                "path": "preflight-teager2b.json",
                "sha256": "7" * 64,
            }
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError,
                "preflight component differs",
            ):
                run_validation()

    def test_confirmation_calibrate_scores_only_selected_first_655(self) -> None:
        expected = _confirmation_stage_expected()
        full_frame = _stage_scoring_frame(expected)
        calibration_eras = expected.eras[: aux.CONFIRMATION_CALIBRATION_ERAS]
        locked_eras = set(expected.eras[-aux.CONFIRMATION_LOCKED_ERAS :])
        selected = "tyler_w10"

        def build_selected(frame: pd.DataFrame, candidate: str) -> pd.DataFrame:
            self.assertEqual(candidate, selected)
            actual_eras = tuple(
                sorted(frame[aux.ERA_COLUMN].astype(str).unique(), key=int)
            )
            self.assertEqual(actual_eras, calibration_eras)
            self.assertTrue(locked_eras.isdisjoint(actual_eras))
            self.assertTrue(
                (set(aux.CANDIDATE_NAMES) - {selected}).isdisjoint(frame.columns)
            )
            scored = frame.copy()
            scored[selected] = np.linspace(0.0, 1.0, len(scored))
            return scored

        def compute(
            frame: pd.DataFrame,
            signals: list[str],
            eras: tuple[str, ...],
            *,
            tabm_column: str,
        ) -> dict[str, pd.DataFrame]:
            self.assertEqual(signals, [selected])
            self.assertEqual(tuple(eras), calibration_eras)
            self.assertEqual(tabm_column, "tabm_two_seed_residual")
            self.assertTrue(locked_eras.isdisjoint(frame[aux.ERA_COLUMN]))
            return _mock_per_era_metrics(calibration_eras, signals)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(root)
            receipt_dir = protocol.experiment_dir / "receipts"
            scout_path = receipt_dir / "locked.json"
            scout_digest = "1" * 64
            pretraining_path = receipt_dir / "confirmation-pretraining.json"
            pretraining_digest = "2" * 64
            scout_binding = {
                "path": aux._relative_path(scout_path, root),
                "sha256": scout_digest,
            }
            pretraining_binding = {
                "path": aux._relative_path(pretraining_path, root),
                "sha256": pretraining_digest,
            }
            scout_locked = {
                "selected_formula": {
                    "name": selected,
                    "weights": dict(aux.BLEND_WEIGHTS[selected]),
                }
            }
            pretraining = {"scout_locked_receipt": scout_binding}
            seal_bindings = {
                name: (receipt_dir / f"seal-{name}.json", str(index) * 64)
                for index, name in enumerate(aux.ALL_COMPONENTS, start=1)
            }
            sealed_inputs = {
                "confirmation_pretraining_receipt": pretraining_binding,
                "confirmation_seal_receipts": {
                    name: {
                        "path": aux._relative_path(path, root),
                        "sha256": digest,
                    }
                    for name, (path, digest) in seal_bindings.items()
                },
                "tabm_two_seed_residual": {"sealed": True},
                "cohort": aux._cohort_receipt(expected),
            }
            with patch.object(
                aux,
                "_load_passing_scout_locked_receipt",
                return_value=scout_locked,
            ), patch.object(
                aux,
                "_validate_confirmation_pretraining_receipt",
                return_value=pretraining,
            ), patch.object(
                aux, "build_confirmation_expected_cohort", return_value=expected
            ), patch.object(
                aux,
                "_build_confirmation_scoring_frame",
                return_value=(full_frame, sealed_inputs),
            ), patch.object(
                aux, "build_selected_rank_blend", side_effect=build_selected
            ) as selected_builder, patch.object(
                aux,
                "build_rank_blends",
                side_effect=AssertionError("ALL_CANDIDATES_BUILT"),
            ) as all_builder, patch.object(
                aux, "compute_per_era_metrics", side_effect=compute
            ), patch.object(
                aux, "summarize_signal", return_value=_summary()
            ), patch.object(
                aux, "_protocol_binding", return_value={"checkpoint": "mocked"}
            ):
                _, receipt = aux.run_confirmation_calibrate(
                    protocol,
                    scout_path,
                    scout_digest,
                    pretraining_path,
                    pretraining_digest,
                    seal_bindings,
                    receipt_dir,
                )

        selected_builder.assert_called_once()
        all_builder.assert_not_called()
        self.assertEqual(receipt["calibration"]["eras"], 655)
        self.assertEqual(receipt["calibration"]["first_era"], "0371")
        self.assertEqual(receipt["calibration"]["last_era"], "1025")
        self.assertTrue(
            {"candidates", "locked", "full"}.isdisjoint(_mapping_keys(receipt))
        )
        for metric in receipt["calibration"]["per_era"].values():
            self.assertEqual(set(metric), {selected})
        self.assertTrue(
            all(
                candidate not in json.dumps(receipt["calibration"], sort_keys=True)
                for candidate in set(aux.CANDIDATE_NAMES) - {selected}
            )
        )

    def test_confirmation_calibrate_rejects_mismatched_scout_authorization_before_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(root)
            receipt_dir = protocol.experiment_dir / "receipts"
            scout_path = receipt_dir / "locked.json"
            pretraining_path = receipt_dir / "pretraining.json"
            scout_locked = {
                "selected_formula": {
                    "name": "tyler_w10",
                    "weights": dict(aux.BLEND_WEIGHTS["tyler_w10"]),
                }
            }
            pretraining = {
                "scout_locked_receipt": {
                    "path": aux._relative_path(scout_path, root),
                    "sha256": "0" * 64,
                }
            }
            with patch.object(
                aux,
                "_load_passing_scout_locked_receipt",
                return_value=scout_locked,
            ), patch.object(
                aux,
                "_validate_confirmation_pretraining_receipt",
                return_value=pretraining,
            ), patch.object(
                aux,
                "build_confirmation_expected_cohort",
                side_effect=AssertionError("CONFIRMATION_DATA_ACCESSED"),
            ) as data_access:
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "Scout authorization differs",
                ):
                    aux.run_confirmation_calibrate(
                        protocol,
                        scout_path,
                        "1" * 64,
                        pretraining_path,
                        "2" * 64,
                        {},
                        receipt_dir,
                    )
            data_access.assert_not_called()

    def test_confirmation_locked_failure_omits_full_and_pass_scores_200_then_855(
        self,
    ) -> None:
        expected = _confirmation_stage_expected()
        selected = "tyler_w10"

        def run_case(locked_passes: bool) -> tuple[dict, list[int]]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                protocol = _synthetic_protocol(root)
                receipt_dir = protocol.experiment_dir / "receipts"
                calibration_path = receipt_dir / "confirmation-calibrate.json"
                pretraining_path = receipt_dir / "confirmation-pretraining.json"
                scout_path = receipt_dir / "locked.json"
                pretraining_digest = "2" * 64
                scout_binding = {
                    "path": aux._relative_path(scout_path, root),
                    "sha256": "3" * 64,
                }
                pretraining_binding = {
                    "path": aux._relative_path(pretraining_path, root),
                    "sha256": pretraining_digest,
                }
                seal_receipts = {
                    name: {
                        "path": f"receipts/confirmation-seal-{name}.json",
                        "sha256": str(index) * 64,
                    }
                    for index, name in enumerate(aux.ALL_COMPONENTS, start=1)
                }
                current_inputs = {
                    "confirmation_pretraining_receipt": pretraining_binding,
                    "confirmation_seal_receipts": seal_receipts,
                    "tabm_two_seed_residual": {"sealed": True},
                    "cohort": aux._cohort_receipt(expected),
                }
                inputs = {"scout_locked_receipt": scout_binding, **current_inputs}
                full_frame = _stage_scoring_frame(expected)

                def materialize(
                    frame: pd.DataFrame, candidate: str
                ) -> pd.DataFrame:
                    self.assertEqual(candidate, selected)
                    self.assertTrue(
                        (set(aux.CANDIDATE_NAMES) - {selected}).isdisjoint(
                            frame.columns
                        )
                    )
                    scored = frame.copy()
                    scored[selected] = np.linspace(0.0, 1.0, len(scored))
                    return scored

                def compute(
                    _frame: pd.DataFrame,
                    signals: list[str],
                    eras: tuple[str, ...],
                    *,
                    tabm_column: str,
                ) -> dict[str, pd.DataFrame]:
                    self.assertEqual(signals, [selected])
                    self.assertEqual(tabm_column, "tabm_two_seed_residual")
                    return _mock_per_era_metrics(tuple(eras), signals)

                with patch.object(
                    aux, "build_selected_rank_blend", side_effect=materialize
                ), patch.object(
                    aux, "compute_per_era_metrics", side_effect=compute
                ), patch.object(
                    aux, "summarize_signal", return_value=_summary()
                ):
                    derived = aux._derive_confirmation_calibration(
                        full_frame, expected, selected
                    )
                calibration_receipt = {
                    "schema_version": 1,
                    "experiment": aux.EXPERIMENT_NAME,
                    "stage": "confirmation-calibrate",
                    "protocol": {"checkpoint": "mocked"},
                    "inputs": inputs,
                    "selected_formula": {
                        "name": selected,
                        "weights": dict(aux.BLEND_WEIGHTS[selected]),
                    },
                    **derived,
                }
                scout_locked = {
                    "selected_formula": {
                        "name": selected,
                        "weights": dict(aux.BLEND_WEIGHTS[selected]),
                    }
                }
                order: list[int] = []

                def ordered_materialize(
                    frame: pd.DataFrame, candidate: str
                ) -> pd.DataFrame:
                    order.append(frame[aux.ERA_COLUMN].nunique())
                    return materialize(frame, candidate)

                def summarize(
                    per_era: dict[str, pd.DataFrame], _signal: str
                ) -> dict:
                    era_count = len(per_era["bmc"])
                    if era_count == aux.CONFIRMATION_LOCKED_ERAS and not locked_passes:
                        return _summary(
                            bmc_mean=0.0,
                            bmc_sharpe=0.1,
                            bmc_drawdown=0.20,
                            corr_mean=0.007,
                        )
                    return _summary()

                def load_stage(
                    _protocol: aux.FrozenProtocol,
                    _path: Path,
                    _digest: str,
                    *,
                    stage: str,
                    allow_prior_pretraining_commit: bool = False,
                ) -> dict:
                    del allow_prior_pretraining_commit
                    if stage == "confirmation-calibrate":
                        return calibration_receipt
                    if stage == "locked":
                        return scout_locked
                    raise AssertionError(stage)

                with patch.object(
                    aux, "_load_passing_stage_receipt", side_effect=load_stage
                ), patch.object(
                    aux,
                    "_load_passing_scout_locked_receipt",
                    return_value=scout_locked,
                ), patch.object(
                    aux,
                    "_validate_confirmation_pretraining_receipt",
                    return_value={},
                ), patch.object(
                    aux, "build_confirmation_expected_cohort", return_value=expected
                ), patch.object(
                    aux,
                    "_build_confirmation_scoring_frame",
                    return_value=(full_frame, current_inputs),
                ), patch.object(
                    aux,
                    "build_selected_rank_blend",
                    side_effect=ordered_materialize,
                ), patch.object(
                    aux,
                    "build_rank_blends",
                    side_effect=AssertionError("ALL_CANDIDATES_BUILT"),
                ), patch.object(
                    aux, "compute_per_era_metrics", side_effect=compute
                ), patch.object(
                    aux, "summarize_signal", side_effect=summarize
                ), patch.object(
                    aux,
                    "_protocol_binding",
                    return_value={"checkpoint": "mocked"},
                ):
                    _, receipt = aux.run_confirmation_locked(
                        protocol,
                        calibration_path,
                        "4" * 64,
                        pretraining_path,
                        pretraining_digest,
                        receipt_dir,
                    )
                return receipt, order

        failed, failed_order = run_case(False)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["state"], "STOP_CONFIRMATION_LOCKED_FAILED")
        self.assertEqual(failed_order, [655, 200])
        self.assertNotIn("full", failed)

        passed, passed_order = run_case(True)
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["state"], "PASS")
        self.assertEqual(passed_order, [655, 200, 855])
        self.assertEqual(passed["locked"]["eras"], 200)
        self.assertEqual(passed["full"]["eras"], 855)
        for section in ("locked", "full"):
            for metric in passed[section]["per_era"].values():
                self.assertEqual(set(metric), {selected})

    def test_confirmation_locked_rejects_pretraining_selected_and_seal_mismatch(
        self,
    ) -> None:
        expected = _confirmation_stage_expected()
        for variant, expected_error in (
            ("pretraining", "confirmation locked pretraining binding differs"),
            ("selected", "confirmation selected Scout formula differs"),
            ("seals", "confirmation locked input revalidation differs"),
        ):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                protocol = _synthetic_protocol(root)
                receipt_dir = protocol.experiment_dir / "receipts"
                calibration_path = receipt_dir / "calibration.json"
                pretraining_path = receipt_dir / "pretraining.json"
                scout_path = receipt_dir / "scout-locked.json"
                selected = "tyler_w10"
                explicit_pretraining = {
                    "path": aux._relative_path(pretraining_path, root),
                    "sha256": "2" * 64,
                }
                scout_binding = {
                    "path": aux._relative_path(scout_path, root),
                    "sha256": "3" * 64,
                }
                seals = {
                    name: {
                        "path": f"receipts/confirmation-seal-{name}.json",
                        "sha256": str(index) * 64,
                    }
                    for index, name in enumerate(aux.ALL_COMPONENTS, start=1)
                }
                inputs = {
                    "scout_locked_receipt": scout_binding,
                    "confirmation_pretraining_receipt": explicit_pretraining,
                    "confirmation_seal_receipts": copy.deepcopy(seals),
                    "tabm_two_seed_residual": {"sealed": True},
                    "cohort": aux._cohort_receipt(expected),
                }
                if variant == "pretraining":
                    inputs["confirmation_pretraining_receipt"] = {
                        **explicit_pretraining,
                        "sha256": "0" * 64,
                    }
                calibration_receipt = {
                    "inputs": inputs,
                    "selected_formula": {
                        "name": selected,
                        "weights": dict(aux.BLEND_WEIGHTS[selected]),
                    },
                }
                scout_selected = "tyler_w40" if variant == "selected" else selected
                scout_locked = {
                    "selected_formula": {
                        "name": scout_selected,
                        "weights": dict(aux.BLEND_WEIGHTS[scout_selected]),
                    }
                }

                def load_stage(
                    _protocol: aux.FrozenProtocol,
                    _path: Path,
                    _digest: str,
                    *,
                    stage: str,
                    allow_prior_pretraining_commit: bool = False,
                ) -> dict:
                    del allow_prior_pretraining_commit
                    return (
                        calibration_receipt
                        if stage == "confirmation-calibrate"
                        else scout_locked
                    )

                current_inputs = {
                    key: copy.deepcopy(value)
                    for key, value in inputs.items()
                    if key != "scout_locked_receipt"
                }
                if variant == "seals":
                    current_inputs["confirmation_seal_receipts"]["jasper"][
                        "sha256"
                    ] = "9" * 64
                full_frame = _stage_scoring_frame(expected)
                with patch.object(
                    aux, "_load_passing_stage_receipt", side_effect=load_stage
                ), patch.object(
                    aux,
                    "_load_passing_scout_locked_receipt",
                    return_value=scout_locked,
                ), patch.object(
                    aux,
                    "_validate_confirmation_pretraining_receipt",
                    return_value={},
                ), patch.object(
                    aux, "build_confirmation_expected_cohort", return_value=expected
                ) as data_access, patch.object(
                    aux,
                    "_build_confirmation_scoring_frame",
                    return_value=(full_frame, current_inputs),
                ) as frame_access:
                    with self.assertRaisesRegex(
                        aux.EnderEnsembleEvaluationError, expected_error
                    ):
                        aux.run_confirmation_locked(
                            protocol,
                            calibration_path,
                            "4" * 64,
                            pretraining_path,
                            "2" * 64,
                            receipt_dir,
                        )
                if variant in {"pretraining", "selected"}:
                    data_access.assert_not_called()
                    frame_access.assert_not_called()
                else:
                    data_access.assert_called_once()
                    frame_access.assert_called_once()

    def test_cli_missing_stage_bindings_refuse_before_protocol_access(
        self,
    ) -> None:
        for stage in (
            "claim-scout-component-run",
            "seal-scout-component",
            "calibrate",
            "locked",
            "create-confirmation-pretraining",
            "claim-confirmation-component-run",
            "seal-confirmation-component",
            "confirmation-calibrate",
            "confirmation-locked",
        ):
            with self.subTest(stage=stage), patch.object(
                aux,
                "verify_frozen_protocol",
                side_effect=AssertionError("PROTOCOL_ACCESSED"),
            ) as protocol_access:
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "required|differs|components",
                ):
                    aux.main(
                        [
                            stage,
                            "--pretraining-commit",
                            "1" * 40,
                        ]
                    )
                protocol_access.assert_not_called()

    def test_cli_rejects_malformed_commit_hash_and_receipt_location_before_protocol_access(
        self,
    ) -> None:
        receipt_dir = (
            _repo_root()
            / "numerai/agents/experiments"
            / aux.EXPERIMENT_NAME
            / "receipts"
        )
        digest = "a" * 64
        canonical = receipt_dir / f"calibrate-{digest}.json"
        cases = (
            (
                [
                    "locked",
                    "--pretraining-commit",
                    "not-a-commit",
                    "--input-receipt",
                    str(canonical),
                    "--input-receipt-sha256",
                    digest,
                ],
                "pretraining commit|40-character|SHA",
            ),
            (
                [
                    "locked",
                    "--pretraining-commit",
                    "1" * 40,
                    "--input-receipt",
                    str(canonical),
                    "--input-receipt-sha256",
                    "bad",
                ],
                "receipt.*SHA|64|hash",
            ),
            (
                [
                    "locked",
                    "--pretraining-commit",
                    "1" * 40,
                    "--input-receipt",
                    str(receipt_dir.parent / canonical.name),
                    "--input-receipt-sha256",
                    digest,
                ],
                "canonical|parent|receipt directory",
            ),
            (
                [
                    "locked",
                    "--pretraining-commit",
                    "1" * 40,
                    "--input-receipt",
                    str(receipt_dir / f"calibrate-{'b' * 64}.json"),
                    "--input-receipt-sha256",
                    digest,
                ],
                "filename|content-addressed|digest",
            ),
        )
        for argv, expected_error in cases:
            with self.subTest(argv=argv), patch.object(
                aux,
                "verify_frozen_protocol",
                side_effect=AssertionError("PROTOCOL_ACCESSED"),
            ) as protocol_access:
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    expected_error,
                ):
                    aux.main(argv)
                protocol_access.assert_not_called()

    def test_cli_rejects_duplicate_confirmation_seal_path_and_digest_before_protocol_access(
        self,
    ) -> None:
        receipt_dir = (
            _repo_root()
            / "numerai/agents/experiments"
            / aux.EXPERIMENT_NAME
            / "receipts"
        )
        scout_digest = "a" * 64
        pretraining_digest = "b" * 64
        base = [
            "confirmation-calibrate",
            "--pretraining-commit",
            "1" * 40,
            "--input-receipt",
            str(receipt_dir / f"locked-{scout_digest}.json"),
            "--input-receipt-sha256",
            scout_digest,
            "--confirmation-pretraining-receipt",
            str(
                receipt_dir
                / f"confirmation-pretraining-{pretraining_digest}.json"
            ),
            "--confirmation-pretraining-receipt-sha256",
            pretraining_digest,
        ]
        values: list[tuple[str, str, str]] = []
        for index, component in enumerate(aux.ALL_COMPONENTS, start=1):
            digest = str(index) * 64
            values.append(
                (
                    component,
                    str(
                        receipt_dir
                        / f"confirmation-seal-{component}-{digest}.json"
                    ),
                    digest,
                )
            )

        duplicate_path = list(values)
        duplicate_path[-1] = (
            duplicate_path[-1][0],
            duplicate_path[0][1],
            duplicate_path[-1][2],
        )
        duplicate_digest = list(values)
        duplicate_digest[-1] = (
            duplicate_digest[-1][0],
            duplicate_digest[-1][1].replace(
                duplicate_digest[-1][2], duplicate_digest[0][2]
            ),
            duplicate_digest[0][2],
        )
        for label, bindings in (
            ("path", duplicate_path),
            ("digest", duplicate_digest),
        ):
            argv = list(base)
            for component, path, digest in bindings:
                argv.extend(
                    ["--confirmation-seal-receipt", component, path, digest]
                )
            with self.subTest(label=label), patch.object(
                aux,
                "verify_frozen_protocol",
                side_effect=AssertionError("PROTOCOL_ACCESSED"),
            ) as protocol_access:
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "unique|duplicate|path|digest",
                ):
                    aux.main(argv)
                protocol_access.assert_not_called()

    def test_cli_rejects_noncanonical_irrelevant_and_half_pair_arguments_early(
        self,
    ) -> None:
        experiment_dir = (
            _repo_root()
            / "numerai/agents/experiments"
            / aux.EXPERIMENT_NAME
        )
        receipt_dir = experiment_dir / "receipts"
        digest = "a" * 64
        locked_base = [
            "locked",
            "--pretraining-commit",
            "1" * 40,
            "--input-receipt",
            str(receipt_dir / f"calibrate-{digest}.json"),
            "--input-receipt-sha256",
            digest,
        ]
        pretraining_digest = "b" * 64
        confirmation_base = [
            "claim-confirmation-component-run",
            "--pretraining-commit",
            "1" * 40,
            "--confirmation-pretraining-receipt",
            str(
                receipt_dir
                / (
                    "confirmation-pretraining-"
                    f"{pretraining_digest}.json"
                )
            ),
            "--confirmation-pretraining-receipt-sha256",
            pretraining_digest,
        ]
        predecessor_digest = "c" * 64
        predecessor = receipt_dir / (
            f"confirmation-seal-jasper-{predecessor_digest}.json"
        )
        cases = (
            (
                [*locked_base, "--source-manifest", str(experiment_dir / "other.json")],
                "canonical source manifest",
            ),
            (
                [*locked_base, "--output-dir", str(experiment_dir / "other")],
                "canonical output receipt directory",
            ),
            (
                [*locked_base, "--receipt-dir", str(experiment_dir / "other")],
                "canonical receipt directory",
            ),
            (
                [
                    "locked",
                    "--pretraining-commit",
                    "1" * 40,
                    "--input-receipt",
                    str(
                        receipt_dir
                        / "alias"
                        / ".."
                        / f"calibrate-{digest}.json"
                    ),
                    "--input-receipt-sha256",
                    digest,
                ],
                "Lexical|canonical",
            ),
            (
                [*locked_base, "--component", "jasper"],
                "not valid for locked",
            ),
            (
                [
                    *locked_base,
                    "--pretraining-commit",
                    "A" * 40,
                ],
                "lowercase 40-character",
            ),
            (
                [
                    *confirmation_base,
                    "--component",
                    "jasper",
                    "--prior-seal-receipt",
                    str(predecessor),
                    "--prior-seal-receipt-sha256",
                    predecessor_digest,
                ],
                "first confirmation component may not bind",
            ),
            (
                [*confirmation_base, "--component", "teager2b"],
                "prior-seal-receipt.*required",
            ),
            (
                [
                    *confirmation_base,
                    "--component",
                    "teager2b",
                    "--prior-seal-receipt",
                    str(predecessor),
                ],
                "prior-seal-receipt-sha256.*required",
            ),
        )
        for argv, expected_error in cases:
            with self.subTest(argv=argv), patch.object(
                aux,
                "verify_frozen_protocol",
                side_effect=AssertionError("PROTOCOL_ACCESSED"),
            ) as protocol_access:
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    expected_error,
                ):
                    aux.main(argv)
                protocol_access.assert_not_called()

    def test_stage_receipt_alias_is_rejected_before_resolving_to_canonical_target(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(root)
            receipt_dir = protocol.experiment_dir / "receipts"
            component = aux.default_scout_component_paths(protocol, "jasper")
            destinations = {
                "result": {"path": "results/jasper.json", "absent": True},
                "predictions": {
                    "path": "predictions/jasper.parquet",
                    "absent": True,
                },
            }
            payload = {
                "schema_version": 1,
                "experiment": aux.EXPERIMENT_NAME,
                "stage": "claim-scout-component-run",
                "state": "ABSENCE_PROVEN",
                "passed": True,
                "protocol": {},
                "component": "jasper",
                "destinations": destinations,
            }
            prefix = "scout-pre-run-jasper"
            claim = aux._claim_receipt_prefix(receipt_dir, prefix)
            canonical = aux._write_claimed_content_addressed_receipt(
                receipt_dir,
                prefix,
                claim,
                payload,
            )
            digest = aux._sha256_file(canonical)
            alias = root / "alias" / canonical.name
            real_resolve = Path.resolve

            def fake_resolve(path: Path, *args: object, **kwargs: object) -> Path:
                if Path(path) == alias:
                    return canonical
                return real_resolve(path, *args, **kwargs)

            def fake_is_symlink(path: Path) -> bool:
                return Path(path) == alias

            with patch.object(Path, "resolve", fake_resolve), patch.object(
                Path, "is_symlink", fake_is_symlink
            ), patch.object(
                aux, "_validate_protocol_binding"
            ), patch.object(
                aux,
                "_scout_destination_receipt",
                return_value=destinations,
            ):
                with self.assertRaisesRegex(
                    aux.EnderEnsembleEvaluationError,
                    "symbolic link|reparse|canonical",
                ):
                    aux._validate_scout_pre_run_absence_receipt(
                        protocol,
                        component,
                        alias,
                        digest,
                    )


class RankAndMetricTests(unittest.TestCase):
    def test_rank_within_era_uses_average_ties_and_is_era_local(self) -> None:
        ranked = aux.rank_within_era(
            [1.0, 2.0, 2.0, 4.0, 100.0, 50.0],
            ["0001", "0001", "0001", "0001", "0002", "0002"],
        )
        np.testing.assert_allclose(ranked, [0.25, 0.625, 0.625, 1.0, 1.0, 0.5])

    def test_rank_blend_ranks_components_then_weighted_sum_then_reranks(self) -> None:
        frame = pd.DataFrame(
            {
                aux.ERA_COLUMN: ["0001"] * 4,
                "jasper": [1.0, 2.0, 2.0, 4.0],
                "teager2b": [4.0, 3.0, 2.0, 1.0],
                "victor": [1.0, 3.0, 2.0, 4.0],
                "xerxes": [2.0, 4.0, 1.0, 3.0],
                "tyler": [4.0, 1.0, 3.0, 2.0],
            }
        )
        blended = aux.build_rank_blends(frame)

        component_ranks = {
            name: aux.rank_within_era(frame[name], frame[aux.ERA_COLUMN])
            for name in aux.ALL_COMPONENTS
        }
        for candidate, weights in aux.BLEND_WEIGHTS.items():
            raw = weights["core"] * sum(
                component_ranks[name]
                for name in ("jasper", "teager2b", "victor", "xerxes")
            )
            raw += weights["tyler"] * component_ranks["tyler"]
            expected = aux.rank_within_era(raw, frame[aux.ERA_COLUMN])
            np.testing.assert_allclose(blended[candidate], expected)
        self.assertEqual(
            set(blended) - set(frame),
            set(aux.CANDIDATE_NAMES),
        )

    def test_locked_blend_materializes_only_the_selected_candidate(self) -> None:
        frame = pd.DataFrame(
            {
                aux.ERA_COLUMN: ["0001"] * 4,
                "jasper": [1.0, 2.0, 3.0, 4.0],
                "teager2b": [4.0, 3.0, 2.0, 1.0],
                "victor": [1.0, 3.0, 2.0, 4.0],
                "xerxes": [2.0, 4.0, 1.0, 3.0],
                "tyler": [4.0, 1.0, 3.0, 2.0],
            }
        )

        locked = aux.build_selected_rank_blend(frame, "tyler_w10")

        self.assertIn("tyler_w10", locked)
        self.assertFalse(
            (set(aux.CANDIDATE_NAMES) - {"tyler_w10"}).intersection(locked)
        )

    def test_symmetric_similarity_is_average_tie_spearman_in_each_era(self) -> None:
        eras = ["0001"] * 4 + ["0002"] * 4
        reference = [1, 2, 2, 4, 10, 30, 20, 40]
        same = [10, 20, 20, 40, 100, 300, 200, 400]
        reverse = [4, 2, 2, 1, 40, 20, 30, 10]

        forward = aux.symmetric_per_era_similarity(same, reference, eras)
        backward = aux.symmetric_per_era_similarity(reference, same, eras)
        negative = aux.symmetric_per_era_similarity(reverse, reference, eras)

        np.testing.assert_allclose(forward, [1.0, 1.0])
        np.testing.assert_allclose(backward, forward)
        np.testing.assert_allclose(negative, [-1.0, -1.0])

    def test_summary_uses_population_std_and_drawdown_without_initial_zero(self) -> None:
        signal = "candidate"
        index = ["0001", "0002"]
        per_era = {
            "corr": pd.DataFrame({signal: [0.1, 0.3]}, index=index),
            "bmc": pd.DataFrame({signal: [-1.0, 2.0]}, index=index),
            "ender20_similarity": pd.DataFrame(
                {signal: [0.1, 0.3]}, index=index
            ),
            "ender60_similarity": pd.DataFrame(
                {signal: [-0.2, 0.2]}, index=index
            ),
            "tabm_similarity": pd.DataFrame({signal: [0.4, 0.6]}, index=index),
        }

        summary = aux.summarize_signal(per_era, signal)

        self.assertAlmostEqual(summary["bmc"]["mean"], 0.5)
        self.assertAlmostEqual(summary["bmc"]["std"], 1.5)
        self.assertAlmostEqual(summary["bmc"]["sharpe"], 1.0 / 3.0)
        self.assertEqual(summary["bmc"]["max_drawdown"], 0.0)
        self.assertAlmostEqual(summary["corr"]["std"], 0.1)
        self.assertAlmostEqual(summary["avg_ender20_similarity"], 0.2)
        self.assertAlmostEqual(summary["avg_ender60_similarity"], 0.0)
        self.assertAlmostEqual(summary["avg_tabm_similarity"], 0.5)

    def test_compute_metrics_includes_all_three_similarity_guards(self) -> None:
        eras = ["0001"] * 4 + ["0002"] * 4
        frame = pd.DataFrame(
            {
                aux.ERA_COLUMN: eras,
                aux.ENDER_TARGET: [1, 2, 3, 4, 1, 2, 3, 4],
                aux.BENCHMARK_ENDER20: [1, 2, 3, 4, 4, 3, 2, 1],
                aux.BENCHMARK_ENDER60: [4, 3, 2, 1, 1, 2, 3, 4],
                "tabm": [1, 3, 2, 4, 4, 2, 3, 1],
                "candidate": [1, 2, 3, 4, 4, 3, 2, 1],
            }
        )
        corr = pd.DataFrame({"candidate": [0.1, 0.2]}, index=["0001", "0002"])
        bmc = pd.DataFrame({"candidate": [0.01, 0.02]}, index=["0001", "0002"])
        with patch.object(aux.numerai_metrics, "per_era_corr", return_value=corr), patch.object(
            aux.numerai_metrics, "per_era_bmc", return_value=bmc
        ):
            metrics = aux.compute_per_era_metrics(
                frame,
                ["candidate"],
                ["0001", "0002"],
                tabm_column="tabm",
            )

        self.assertEqual(
            set(metrics),
            {
                "corr",
                "bmc",
                "ender20_similarity",
                "ender60_similarity",
                "tabm_similarity",
            },
        )
        np.testing.assert_allclose(
            metrics["ender20_similarity"]["candidate"], [1.0, 1.0]
        )
        np.testing.assert_allclose(
            metrics["ender60_similarity"]["candidate"], [-1.0, -1.0]
        )
        self.assertTrue(
            np.isfinite(metrics["tabm_similarity"]["candidate"]).all()
        )


class CohortAndArtifactSemanticsTests(unittest.TestCase):
    def test_historical_manifest_lease_checks_size_and_hash_before_yield(
        self,
    ) -> None:
        for variant, expected_error in (
            ("size", "size differs"),
            ("hash", "hash differs"),
        ):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                protocol = _synthetic_protocol(root)
                artifact_path = root / "historical" / "artifact.bin"
                artifact_path.parent.mkdir(parents=True)
                artifact_path.write_bytes(b"historical-bytes")
                receipt = aux._file_receipt(artifact_path, root)
                if variant == "size":
                    receipt["size_bytes"] += 1
                else:
                    receipt["sha256"] = "0" * 64

                lease_events: list[str] = []
                original_lease = aux._ReadOnlyFileLease

                class TrackingLease:
                    def __init__(self, path: Path, label: str) -> None:
                        self.inner = original_lease(path, label)
                        self.open = True

                    def fileno(self) -> int:
                        self.assert_open("fstat")
                        return self.inner.fileno()

                    def sha256(self) -> str:
                        self.assert_open("sha256")
                        return self.inner.sha256()

                    def assert_open(self, event: str) -> None:
                        if not self.open:
                            raise AssertionError("LEASE_CLOSED_EARLY")
                        lease_events.append(event)

                    def close(self) -> None:
                        self.open = False
                        self.inner.close()

                with patch.object(aux, "_ReadOnlyFileLease", TrackingLease):
                    with self.assertRaisesRegex(
                        aux.EnderEnsembleEvaluationError,
                        expected_error,
                    ):
                        with aux._lease_frozen_manifest_artifacts(
                            protocol,
                            {"artifact": receipt},
                            "synthetic historical",
                        ):
                            self.fail("malformed historical receipt was yielded")
                self.assertIn("fstat", lease_events)
                if variant == "hash":
                    self.assertIn("sha256", lease_events)

    def test_new_scout_uses_cached_config_through_base_and_wrapper_leases(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(root)
            config_dir = protocol.experiment_dir / "configs"
            config_dir.mkdir()
            base_path = config_dir / "base_d8.py"
            base_path.write_bytes(b"BASE = True\n")
            component = aux.default_scout_component_paths(protocol, "jasper")
            component.config.write_bytes(b"CONFIG = {'cached': True}\n")
            cached_config = {"nested": {"sentinel": "cached"}}
            protocol.scout_config_paths[component.name] = component.config
            protocol.scout_configs[component.name] = cached_config
            protocol.source_manifest["experiment_files"] = {
                "configs/base_d8.py": aux._file_receipt(base_path, root),
                "configs/r1_jasper_d8_t6000.py": aux._file_receipt(
                    component.config,
                    root,
                ),
            }

            expected = _stage_expected()
            expected_leases = {
                Path(aux.os.path.abspath(base_path)),
                Path(aux.os.path.abspath(component.config)),
            }
            active_leases: set[Path] = set()
            TrackingLease = _tracking_read_only_lease_type(active_leases)
            config_objects: list[dict[str, object]] = []
            semantics = {"sentinel": "semantics"}
            artifact = {"sentinel": "artifact"}

            def assert_leases() -> None:
                self.assertEqual(active_leases, expected_leases)

            def validate_cached_config(
                name: str,
                config: dict[str, object],
                **_kwargs,
            ) -> None:
                assert_leases()
                self.assertEqual(name, component.name)
                self.assertEqual(config, cached_config)
                self.assertIsNot(config, cached_config)
                self.assertIsNot(config["nested"], cached_config["nested"])
                config_objects.append(config)

            def validate_result(
                current_component: aux.ComponentPaths,
                result: dict[str, object],
                config: dict[str, object],
                current_expected: aux.ExpectedCohort,
                **_kwargs,
            ) -> dict[str, str]:
                assert_leases()
                self.assertEqual(current_component, component)
                self.assertEqual(result, {"sentinel": "result"})
                self.assertIs(config, config_objects[0])
                self.assertIs(current_expected, expected)
                return semantics

            def validate_predictions(*_args, **_kwargs) -> np.ndarray:
                assert_leases()
                return np.zeros(len(expected.frame), dtype=np.float64)

            def artifact_receipt(
                current_protocol: aux.FrozenProtocol,
                current_component: aux.ComponentPaths,
            ) -> dict[str, str]:
                assert_leases()
                self.assertIs(current_protocol, protocol)
                self.assertEqual(current_component, component)
                return artifact

            with patch.object(
                aux, "_ReadOnlyFileLease", TrackingLease
            ), patch.object(
                aux, "_require_regular_output_file"
            ), patch.object(
                aux,
                "_load_config",
                side_effect=AssertionError("CACHED_SCOUT_CONFIG_REEVALUATED"),
            ), patch.object(
                aux,
                "validate_component_config",
                side_effect=validate_cached_config,
            ), patch.object(
                aux, "_load_json", return_value={"sentinel": "result"}
            ), patch.object(
                aux, "validate_result_json", side_effect=validate_result
            ), patch.object(
                aux,
                "validate_prediction_artifact",
                side_effect=validate_predictions,
            ), patch.object(
                aux, "_artifact_receipt", side_effect=artifact_receipt
            ):
                raw, current_artifact = aux._validate_scout_component(
                    protocol,
                    component,
                    expected,
                )

            self.assertEqual(len(raw), len(expected.frame))
            self.assertEqual(current_artifact, artifact)
            self.assertEqual(len(config_objects), 1)
            self.assertEqual(cached_config, {"nested": {"sentinel": "cached"}})
            self.assertEqual(active_leases, set())

    def test_reused_xerxes_config_and_artifacts_share_base_and_wrapper_leases(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(root)
            historical = root / "historical-xerxes"
            historical.mkdir()
            base_path = historical / "base.py"
            config_path = historical / "xerxes.py"
            base_path.write_bytes(b"BASE = True\n")
            config_path.write_bytes(b"CONFIG = {'xerxes': True}\n")
            component = aux.ComponentPaths(
                name="xerxes",
                config=config_path,
                result=root / "outputs" / "xerxes.json",
                predictions=root / "outputs" / "xerxes.parquet",
            )
            protocol.source_manifest["reused_xerxes_component"] = {
                "base_config": aux._file_receipt(base_path, root),
                "config": aux._file_receipt(config_path, root),
            }
            expected = _stage_expected()
            expected_leases = {
                Path(aux.os.path.abspath(base_path)),
                Path(aux.os.path.abspath(config_path)),
            }
            active_leases: set[Path] = set()
            TrackingLease = _tracking_read_only_lease_type(active_leases)
            config = {"sentinel": "xerxes config"}
            semantics = {"sentinel": "semantics"}
            artifact = {"sentinel": "artifact"}
            config_loads: list[Path] = []

            def assert_leases() -> None:
                self.assertEqual(active_leases, expected_leases)

            def load_config(path: Path, _label: str) -> dict[str, str]:
                assert_leases()
                self.assertEqual(path, config_path)
                config_loads.append(path)
                return config

            def validate_config(
                name: str,
                current_config: dict[str, str],
                **_kwargs,
            ) -> None:
                assert_leases()
                self.assertEqual(name, component.name)
                self.assertIs(current_config, config)

            def validate_result(*args, **_kwargs) -> dict[str, str]:
                assert_leases()
                self.assertIs(args[2], config)
                return semantics

            def validate_predictions(*_args, **_kwargs) -> np.ndarray:
                assert_leases()
                return np.zeros(len(expected.frame), dtype=np.float64)

            def artifact_receipt(*_args, **_kwargs) -> dict[str, str]:
                assert_leases()
                return artifact

            with patch.object(
                aux, "_ReadOnlyFileLease", TrackingLease
            ), patch.object(
                aux, "_require_regular_output_file"
            ), patch.object(
                aux, "_load_config", side_effect=load_config
            ), patch.object(
                aux, "validate_component_config", side_effect=validate_config
            ), patch.object(
                aux, "_load_json", return_value={"sentinel": "result"}
            ), patch.object(
                aux, "validate_result_json", side_effect=validate_result
            ), patch.object(
                aux,
                "validate_prediction_artifact",
                side_effect=validate_predictions,
            ), patch.object(
                aux, "_artifact_receipt", side_effect=artifact_receipt
            ):
                raw, current_artifact = aux._validate_scout_component(
                    protocol,
                    component,
                    expected,
                )

            self.assertEqual(len(raw), len(expected.frame))
            self.assertEqual(current_artifact, artifact)
            self.assertEqual(config_loads, [config_path])
            self.assertEqual(active_leases, set())

    def test_confirmation_component_holds_manifest_base_and_exact_wrapper_leases(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(root, canonical_experiment=True)
            name = "jasper"
            config_dir = protocol.experiment_dir / "configs"
            config_dir.mkdir()
            base_path = config_dir / "base_d8.py"
            wrapper_path = config_dir / f"confirmation_{name}_d8_t6000.py"
            base_path.write_bytes(b"BASE = True\n")
            wrapper_path.write_bytes(
                aux._expected_confirmation_config_source(name).encode("utf-8")
            )
            stem = f"confirmation_{name}_d8_t6000"
            protocol.source_manifest.update({
                "experiment_files": {
                    "configs/base_d8.py": aux._file_receipt(base_path, root),
                },
                "confirmation_output_contract": {
                    name: {
                        "config_path": aux._confirmation_config_relative(name),
                        "must_be_absent_before_run": True,
                        "predictions_path": (
                            f"numerai/agents/experiments/{aux.EXPERIMENT_NAME}/"
                            f"predictions/{stem}.parquet"
                        ),
                        "results_name": stem,
                        "results_path": (
                            f"numerai/agents/experiments/{aux.EXPERIMENT_NAME}/"
                            f"results/{stem}.json"
                        ),
                    }
                },
            })
            component = aux.default_confirmation_component_paths(protocol, name)

            store_dir = root / "stores" / name
            store_dir.mkdir(parents=True)
            metadata_path = store_dir / "metadata.json"
            feature_path = store_dir / "features.bin"
            manifest_path = store_dir / "manifest.parquet"
            store_metadata = {"sentinel": "metadata"}
            metadata_path.write_bytes(aux._receipt_bytes(store_metadata))
            feature_path.write_bytes(b"features")
            manifest_path.write_bytes(b"manifest")
            store = {
                "metadata": aux._file_receipt(metadata_path, root),
                "features": aux._file_receipt(feature_path, root),
                "manifest": aux._file_receipt(manifest_path, root),
            }
            store_inventory = {
                "path": "inventory.json",
                "sha256": "1" * 64,
                "size_bytes": 1,
            }
            pretraining = {
                "input_layout": {"stores": {name: store}},
                "store_inventory": store_inventory,
            }
            result = {
                "data": {
                    "disk_feature_store": {
                        "directory": str(metadata_path.parent),
                        "feature_path": str(feature_path),
                        "manifest_path": str(manifest_path),
                    }
                }
            }
            expected = _confirmation_stage_expected()
            config = {"sentinel": "confirmation config"}
            semantics = {"sentinel": "semantics"}
            artifact = {"sentinel": "artifact"}
            expected_leases = {
                Path(aux.os.path.abspath(base_path)),
                Path(aux.os.path.abspath(wrapper_path)),
            }
            active_leases: set[Path] = set()
            TrackingLease = _tracking_read_only_lease_type(active_leases)
            config_loads: list[Path] = []

            def assert_leases() -> None:
                self.assertEqual(active_leases, expected_leases)

            def load_config(path: Path, _label: str) -> dict[str, str]:
                assert_leases()
                self.assertEqual(path, wrapper_path)
                config_loads.append(path)
                return config

            def validate_config(
                current_name: str,
                current_config: dict[str, str],
                **kwargs,
            ) -> None:
                assert_leases()
                self.assertEqual(current_name, name)
                self.assertIs(current_config, config)
                self.assertIs(kwargs.get("confirmation"), True)

            def load_json(path: Path, _label: str) -> dict[str, object]:
                assert_leases()
                if path == metadata_path:
                    return store_metadata
                if path == component.result:
                    return result
                raise AssertionError(path)

            def validate_result(*args, **kwargs) -> dict[str, str]:
                assert_leases()
                self.assertIs(args[2], config)
                self.assertIs(kwargs["store_metadata"], store_metadata)
                self.assertIs(kwargs["store_receipt"], store)
                self.assertIs(kwargs["store_inventory_receipt"], store_inventory)
                return semantics

            def validate_predictions(*_args, **_kwargs) -> np.ndarray:
                assert_leases()
                return np.zeros(len(expected.frame), dtype=np.float64)

            def artifact_receipt(*_args, **_kwargs) -> dict[str, str]:
                assert_leases()
                return artifact

            with patch.object(
                aux, "_ReadOnlyFileLease", TrackingLease
            ), patch.object(
                aux, "_require_regular_output_file"
            ), patch.object(
                aux, "_load_config", side_effect=load_config
            ), patch.object(
                aux, "validate_component_config", side_effect=validate_config
            ), patch.object(
                aux, "_load_json", side_effect=load_json
            ), patch.object(
                aux, "validate_result_json", side_effect=validate_result
            ), patch.object(
                aux,
                "validate_prediction_artifact",
                side_effect=validate_predictions,
            ), patch.object(
                aux, "_artifact_receipt", side_effect=artifact_receipt
            ):
                raw, current_artifact = aux._validate_confirmation_component(
                    protocol,
                    component,
                    expected,
                    pretraining,
                )

            self.assertEqual(len(raw), len(expected.frame))
            self.assertEqual(current_artifact, artifact)
            self.assertEqual(config_loads, [wrapper_path])
            self.assertEqual(active_leases, set())

    def test_tabm_scout_and_confirmation_files_remain_leased_through_reads(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(root)
            historical_dir = root / "historical-tabm"
            historical_dir.mkdir()

            def artifact(name: str, payload: bytes) -> dict[str, object]:
                path = historical_dir / name
                path.write_bytes(payload)
                return aux._file_receipt(path, root)

            dependency_chain = {
                "config_loader": artifact("config_loader.py", b"loader"),
                "pipeline": artifact("pipeline.py", b"pipeline"),
            }
            source_manifest = artifact("source_manifest.json", b"manifest")
            sections: dict[str, dict[str, dict[str, object]]] = {}
            for section_name in ("scout", "confirmation"):
                sections[section_name] = {}
                for seed in ("seed1337", "seed2027"):
                    sections[section_name][seed] = {
                        "config": artifact(
                            f"{section_name}-{seed}-config.py", b"config"
                        ),
                        "result": artifact(
                            f"{section_name}-{seed}-result.json", b"result"
                        ),
                        "predictions": artifact(
                            f"{section_name}-{seed}-predictions.parquet",
                            b"predictions",
                        ),
                    }
            protocol.source_manifest["tabm_similarity_reference"] = {
                "config_dependency_chain": dependency_chain,
                "historical_source_manifest": source_manifest,
                **sections,
            }
            expected = _stage_expected()
            active_leases: set[Path] = set()
            hash_checks: list[Path] = []
            original_lease = aux._ReadOnlyFileLease

            class TrackingLease:
                def __init__(self, path: Path, label: str) -> None:
                    self.path = Path(aux.os.path.abspath(path))
                    self.inner = original_lease(path, label)
                    active_leases.add(self.path)

                def fileno(self) -> int:
                    return self.inner.fileno()

                def sha256(self) -> str:
                    self.assert_open()
                    hash_checks.append(self.path)
                    return self.inner.sha256()

                def assert_open(self) -> None:
                    if self.path not in active_leases:
                        raise AssertionError("HISTORICAL_LEASE_CLOSED_EARLY")

                def close(self) -> None:
                    active_leases.discard(self.path)
                    self.inner.close()

            expected_active: set[Path] = set()
            read_events: list[tuple[str, Path]] = []

            def assert_all_leased(path: Path, kind: str) -> None:
                absolute = Path(aux.os.path.abspath(path))
                self.assertEqual(active_leases, expected_active)
                self.assertIn(absolute, active_leases)
                read_events.append((kind, absolute))

            def load_config(path: Path, _label: str):
                assert_all_leased(path, "config")
                return {
                    "model": {
                        "target_transform": {
                            "type": "residual_to_benchmark"
                        }
                    }
                }

            def load_result(path: Path, _label: str):
                assert_all_leased(path, "result")
                return {}

            def load_predictions(path: Path, *_args, **_kwargs):
                assert_all_leased(path, "predictions")
                return np.linspace(0.0, 1.0, len(expected.frame))

            with patch.object(
                aux, "_ReadOnlyFileLease", TrackingLease
            ), patch.object(
                aux, "_independent_fold_map", return_value={}
            ), patch.object(
                aux, "_load_config", side_effect=load_config
            ), patch.object(
                aux, "_load_json", side_effect=load_result
            ), patch.object(
                aux, "_validate_tabm_result", return_value={}
            ), patch.object(
                aux,
                "validate_prediction_artifact",
                side_effect=load_predictions,
            ):
                for confirmation in (False, True):
                    section_name = "confirmation" if confirmation else "scout"
                    receipts = [
                        *dependency_chain.values(),
                        source_manifest,
                        *(
                            sections[section_name][seed][name]
                            for seed in ("seed1337", "seed2027")
                            for name in ("config", "result", "predictions")
                        ),
                    ]
                    expected_active = {
                        Path(aux.os.path.abspath(root / str(receipt["path"])))
                        for receipt in receipts
                    }
                    blended, frozen_receipts = aux.load_frozen_two_seed_residual(
                        protocol,
                        expected,
                        confirmation=confirmation,
                    )
                    self.assertEqual(len(blended), len(expected.frame))
                    self.assertEqual(
                        set(frozen_receipts), {"seed1337", "seed2027"}
                    )
                    self.assertEqual(active_leases, set())

            self.assertEqual(
                [kind for kind, _ in read_events].count("config"), 4
            )
            self.assertEqual(
                [kind for kind, _ in read_events].count("result"), 4
            )
            self.assertEqual(
                [kind for kind, _ in read_events].count("predictions"), 4
            )
            self.assertEqual(len(hash_checks), 18)

            if os.name == "nt":
                for confirmation in (False, True):
                    section_name = "confirmation" if confirmation else "scout"
                    mutation_target = root / str(
                        sections[section_name]["seed1337"]["predictions"][
                            "path"
                        ]
                    )

                    def mutate_after_verification(*_args, **_kwargs):
                        mutation_target.write_bytes(b"post-verify mutation")
                        raise AssertionError("MUTATION_WAS_NOT_BLOCKED")

                    with patch.object(
                        aux, "_ReadOnlyFileLease", TrackingLease
                    ), patch.object(
                        aux, "_independent_fold_map", return_value={}
                    ), patch.object(
                        aux,
                        "_load_config",
                        side_effect=mutate_after_verification,
                    ), patch.object(
                        aux,
                        "validate_prediction_artifact",
                        side_effect=AssertionError("SCORING_REACHED"),
                    ) as scoring:
                        with self.assertRaises(OSError):
                            aux.load_frozen_two_seed_residual(
                                protocol,
                                expected,
                                confirmation=confirmation,
                            )
                    scoring.assert_not_called()
                    self.assertEqual(active_leases, set())

    def test_reused_xerxes_files_remain_leased_through_scout_scoring(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(root)
            historical_dir = root / "historical-xerxes"
            historical_dir.mkdir()

            def artifact(name: str) -> dict[str, object]:
                path = historical_dir / name
                path.write_bytes(name.encode("utf-8"))
                return aux._file_receipt(path, root)

            frozen_xerxes = {
                name: artifact(f"{name}.bin")
                for name in (
                    "base_config",
                    "config",
                    "evaluator",
                    "predictions",
                    "result",
                    "source_manifest",
                )
            }
            protocol.source_manifest["reused_xerxes_component"] = frozen_xerxes
            expected = _stage_expected()
            active_leases: set[Path] = set()
            original_lease = aux._ReadOnlyFileLease

            class TrackingLease:
                def __init__(self, path: Path, label: str) -> None:
                    self.path = Path(aux.os.path.abspath(path))
                    self.inner = original_lease(path, label)
                    active_leases.add(self.path)

                def fileno(self) -> int:
                    return self.inner.fileno()

                def sha256(self) -> str:
                    return self.inner.sha256()

                def close(self) -> None:
                    active_leases.discard(self.path)
                    self.inner.close()

            expected_active = {
                Path(aux.os.path.abspath(root / str(receipt["path"])))
                for receipt in frozen_xerxes.values()
            }
            signals = {
                name: np.zeros(len(expected.frame))
                for name in aux.SCOUT_NEW_COMPONENTS
            }
            xerxes_artifact = {
                name: copy.deepcopy(frozen_xerxes[name])
                for name in ("config", "result", "predictions")
            }

            def validate_xerxes(*_args, **_kwargs):
                self.assertEqual(active_leases, expected_active)
                return np.zeros(len(expected.frame)), xerxes_artifact

            def load_tabm(*_args, **_kwargs):
                self.assertEqual(active_leases, expected_active)
                return np.zeros(len(expected.frame)), {"sealed": True}

            with patch.object(
                aux, "_ReadOnlyFileLease", TrackingLease
            ), patch.object(
                aux,
                "validate_seal_receipts",
                return_value=(signals, {}),
            ), patch.object(
                aux, "_validate_scout_component", side_effect=validate_xerxes
            ), patch.object(
                aux, "load_frozen_two_seed_residual", side_effect=load_tabm
            ):
                frame, evidence = aux._build_scout_scoring_frame(
                    protocol,
                    expected,
                    {},
                )
            self.assertEqual(len(frame), len(expected.frame))
            self.assertEqual(evidence["reused_xerxes"], xerxes_artifact)
            self.assertEqual(active_leases, set())

            if os.name == "nt":
                mutation_target = root / str(frozen_xerxes["predictions"]["path"])

                def mutate_after_verification(*_args, **_kwargs):
                    mutation_target.write_bytes(b"post-verify mutation")
                    raise AssertionError("MUTATION_WAS_NOT_BLOCKED")

                with patch.object(
                    aux, "_ReadOnlyFileLease", TrackingLease
                ), patch.object(
                    aux,
                    "validate_seal_receipts",
                    return_value=(signals, {}),
                ), patch.object(
                    aux,
                    "_validate_scout_component",
                    side_effect=mutate_after_verification,
                ), patch.object(
                    aux,
                    "load_frozen_two_seed_residual",
                    side_effect=AssertionError("SCORING_REACHED"),
                ) as scoring:
                    with self.assertRaises(OSError):
                        aux._build_scout_scoring_frame(
                            protocol,
                            expected,
                            {},
                        )
                scoring.assert_not_called()
                self.assertEqual(active_leases, set())

    def test_source_join_is_by_id_and_era_not_incidental_row_order(self) -> None:
        data = pd.DataFrame(
            {
                aux.ID_COLUMN: ["a", "b"],
                aux.ERA_COLUMN: ["0001", "0002"],
                aux.ENDER_TARGET: [0.1, 0.2],
            }
        )
        benchmark = pd.DataFrame(
            {
                aux.ID_COLUMN: ["b", "a"],
                aux.ERA_COLUMN: ["0002", "0001"],
                aux.BENCHMARK_ENDER20: [0.4, 0.3],
            }
        )

        merged = aux._merge_sources_one_to_one(data, benchmark, label="synthetic")

        self.assertEqual(merged[aux.ID_COLUMN].tolist(), ["a", "b"])
        self.assertEqual(merged[aux.BENCHMARK_ENDER20].tolist(), [0.3, 0.4])

        wrong_era = benchmark.copy()
        wrong_era.loc[wrong_era[aux.ID_COLUMN] == "a", aux.ERA_COLUMN] = "9999"
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError, "source eras differ by ID"
        ):
            aux._merge_sources_one_to_one(data, wrong_era, label="synthetic")

    def test_expected_folds_are_derived_from_the_producers_frozen_chronology(self) -> None:
        full = pd.DataFrame(
            {
                aux.ID_COLUMN: [f"id-{index}" for index in range(1, 11)],
                aux.ERA_COLUMN: [f"{index:04d}" for index in range(1, 11)],
            }
        )

        expected = aux._derive_expected_oof(
            full,
            embargo=1,
            expected_rows=8,
            expected_eras=8,
            first_era="0003",
            last_era="0010",
            calibration_eras=4,
            last_calibration_era="0006",
            first_locked_era="0007",
        )

        self.assertEqual(expected.eras, tuple(f"{index:04d}" for index in range(3, 11)))
        self.assertEqual(expected.frame[aux.FOLD_COLUMN].tolist(), [1, 1, 2, 2, 3, 3, 4, 4])
        self.assertEqual([fold["fold"] for fold in expected.folds], [1, 2, 3, 4])

    def test_prediction_artifact_aligns_by_id_and_era_but_uses_own_fold_map(
        self,
    ) -> None:
        target = aux.COMPONENT_TARGETS["jasper"]
        expected = pd.DataFrame(
            {
                aux.ID_COLUMN: ["a", "b", "c", "d"],
                aux.ERA_COLUMN: ["0001", "0001", "0002", "0002"],
                target: [0.1, 0.2, 0.3, 0.4],
                # This deliberately represents some other producer's fold receipt.
                aux.FOLD_COLUMN: [8, 8, 9, 9],
            }
        )
        artifact = pd.DataFrame(
            {
                aux.ID_COLUMN: ["d", "b", "a", "c"],
                aux.ERA_COLUMN: ["0002", "0001", "0001", "0002"],
                target: [0.4, 0.2, 0.1, 0.3],
                aux.PREDICTION_COLUMN: [40.0, 20.0, 10.0, 30.0],
                aux.FOLD_COLUMN: [2, 1, 1, 2],
            }
        )
        semantics = {"producer": "jasper", "rank": "raw"}
        own_fold_map = {"0001": 1, "0002": 2}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jasper.parquet"
            _write_prediction_artifact(path, artifact, semantics)
            predictions = aux.validate_prediction_artifact(
                path,
                expected,
                semantics,
                target_column=target,
                expected_fold_by_era=own_fold_map,
            )
            np.testing.assert_allclose(predictions, [10.0, 20.0, 30.0, 40.0])

            wrong_fold = artifact.copy()
            wrong_fold.loc[0, aux.FOLD_COLUMN] = 1
            _write_prediction_artifact(path, wrong_fold, semantics)
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "frozen producer"
            ):
                aux.validate_prediction_artifact(
                    path,
                    expected,
                    semantics,
                    target_column=target,
                    expected_fold_by_era=own_fold_map,
                )

            fractional_fold = artifact.copy()
            fractional_fold[aux.FOLD_COLUMN] = fractional_fold[
                aux.FOLD_COLUMN
            ].astype(float)
            # Row 1 belongs to producer fold 1, so an unsafe int cast would
            # silently turn this malformed 1.5 provenance value into 1.
            fractional_fold.loc[1, aux.FOLD_COLUMN] = 1.5
            _write_prediction_artifact(path, fractional_fold, semantics)
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "fold provenance"
            ):
                aux.validate_prediction_artifact(
                    path,
                    expected,
                    semantics,
                    target_column=target,
                    expected_fold_by_era=own_fold_map,
                )

    def test_prediction_semantics_enforce_legacy_and_full_contracts(self) -> None:
        target = aux.ENDER_TARGET
        expected = pd.DataFrame(
            {
                aux.ID_COLUMN: ["a", "b"],
                aux.ERA_COLUMN: ["0001", "0001"],
                target: [0.1, 0.2],
                aux.FOLD_COLUMN: [1, 1],
            }
        )
        artifact = pd.DataFrame(
            {
                aux.ID_COLUMN: ["a", "b"],
                aux.ERA_COLUMN: ["0001", "0001"],
                target: [0.1, 0.2],
                aux.PREDICTION_COLUMN: [0.3, 0.4],
                aux.FOLD_COLUMN: [1, 1],
            }
        )
        semantics = {
            "training_target": {"transform": {"type": "residual_to_benchmark"}}
        }

        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "legacy.parquet"
            _write_prediction_artifact(legacy, artifact, None)
            aux.validate_prediction_artifact(
                legacy,
                expected,
                None,
                target_column=target,
                require_semantics=False,
            )
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "Parquet semantics differs"
            ):
                aux.validate_prediction_artifact(
                    legacy,
                    expected,
                    semantics,
                    target_column=target,
                    require_semantics=True,
                )

            full = Path(directory) / "full.parquet"
            _write_prediction_artifact(full, artifact, semantics)
            aux.validate_prediction_artifact(
                full,
                expected,
                semantics,
                target_column=target,
                require_semantics=True,
            )

    def test_result_receipt_requires_gpu_folds_and_exact_automatic_metrics(self) -> None:
        root = _repo_root()
        config_path = (
            root
            / "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53/"
            "configs/r1_jasper_d8_t6000.py"
        )
        config = runpy.run_path(str(config_path))["CONFIG"]
        target = aux.COMPONENT_TARGETS["jasper"]
        frame = pd.DataFrame(
            {
                aux.ID_COLUMN: ["a", "b", "c", "d"],
                aux.ERA_COLUMN: ["0001", "0001", "0002", "0002"],
                target: [0.1, 0.2, 0.3, 0.4],
                aux.FOLD_COLUMN: [1, 1, 2, 2],
            }
        )
        folds = (
            {"fold": 1, "train_eras": 1, "val_eras": 1, "train_rows": 2, "val_rows": 2},
            {"fold": 2, "train_eras": 2, "val_eras": 1, "train_rows": 4, "val_rows": 2},
        )
        expected = aux.ExpectedCohort(
            frame=frame,
            full_rows=6,
            full_eras=3,
            eras=("0001", "0002"),
            folds=folds,
        )
        result_name = config["output"]["results_name"]
        component = aux.ComponentPaths(
            name="jasper",
            config=Path("config.py"),
            result=Path(f"{result_name}.json"),
            predictions=Path(f"{result_name}.parquet"),
        )
        result = {
            "model": aux._expected_model_payload(config),
            "preprocessing": {
                "nan_missing_all_twos": False,
                "missing_value": 2.0,
            },
            "data": {
                "data_version": "v5.3",
                "feature_set": "medium",
                "target": target,
                "full_rows": 6,
                "full_eras": 3,
                "oof_rows": 4,
                "oof_eras": 2,
                "embargo_eras": 13,
                "require_benchmark_coverage": True,
                "data_mode": "eager",
                "full_data_path": "v5.3/downsampled_full.parquet",
            },
            "benchmark": {
                "model": aux.BENCHMARK_ENDER20,
                "file": "v5.3/downsampled_full_benchmark_models.parquet",
            },
            "training": {
                "data_sampling": {"max_train_samples": 500_000, "sample_seed": 1337},
                "data_mode": "eager",
                "cv": {
                    "enabled": True,
                    "n_splits": 5,
                    "embargo": 13,
                    "mode": "expanding",
                    "min_train_size": 0,
                },
            },
            "cv": {
                "n_splits": 5,
                "embargo": 13,
                "mode": "expanding",
                "min_train_size": 0,
                "folds_used": 2,
                "folds": [
                    {
                        **fold,
                        "model_diagnostics": {
                            "effective_device_type": "gpu",
                            "gpu_fallback_used": False,
                        },
                    }
                    for fold in folds
                ],
            },
            "output": {
                "predictions_file": f"some/path/{result_name}.parquet",
                "prediction_semantics": aux._expected_semantics(config),
            },
            "metrics": {"corr": {}, "bmc": {}, "bmc_last_200_eras": {}},
        }

        self.assertEqual(
            aux.validate_result_json(component, result, config, expected),
            aux._expected_semantics(config),
        )

        fallback = copy.deepcopy(result)
        fallback["cv"]["folds"][0]["model_diagnostics"]["gpu_fallback_used"] = True
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError, "GPU fallback differs"
        ):
            aux.validate_result_json(component, fallback, config, expected)

        extra_metric = copy.deepcopy(result)
        extra_metric["metrics"]["unfrozen"] = 1.0
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError, "automatic metric schema differs"
        ):
            aux.validate_result_json(component, extra_metric, config, expected)

    def test_confirmation_result_binds_immutable_store_hashes_and_inventory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _confirmation_result_provenance_fixture(Path(directory))
            component = fixture["component"]
            result = fixture["result"]
            config = fixture["config"]
            expected = fixture["expected"]
            store_metadata = fixture["store_metadata"]
            store_receipt = fixture["store_receipt"]
            store_inventory_receipt = fixture["store_inventory_receipt"]
            assert isinstance(component, aux.ComponentPaths)
            assert isinstance(result, dict)
            assert isinstance(config, dict)
            assert isinstance(expected, aux.ExpectedCohort)
            assert isinstance(store_metadata, dict)
            assert isinstance(store_receipt, dict)
            assert isinstance(store_inventory_receipt, dict)

            validation_arguments = {
                "confirmation": True,
                "store_metadata": store_metadata,
                "store_receipt": store_receipt,
                "store_inventory_receipt": store_inventory_receipt,
            }
            self.assertEqual(
                aux.validate_result_json(
                    component,
                    result,
                    config,
                    expected,
                    **validation_arguments,
                ),
                aux._expected_semantics(config),
            )

            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError,
                "immutable store provenance",
            ):
                aux.validate_result_json(
                    component,
                    result,
                    config,
                    expected,
                    confirmation=True,
                )

            diagnostics_tampers = (
                ("metadata_sha256", "0" * 64),
                ("feature_sha256", "1" * 64),
                ("manifest_sha256", "2" * 64),
            )
            for key, forged_value in diagnostics_tampers:
                with self.subTest(field=key):
                    forged = copy.deepcopy(result)
                    forged["data"]["disk_feature_store"][key] = forged_value
                    with self.assertRaisesRegex(
                        aux.EnderEnsembleEvaluationError,
                        f"store.{key} differs",
                    ):
                        aux.validate_result_json(
                            component,
                            forged,
                            config,
                            expected,
                            **validation_arguments,
                        )

            for key, forged_value in (
                ("path", "inventories/forged.json"),
                ("git_blob_id", "e" * 40),
                ("checkpoint_commit", "f" * 40),
            ):
                with self.subTest(committed_inventory=key):
                    forged = copy.deepcopy(result)
                    forged["data"]["disk_feature_store"][
                        "committed_inventory"
                    ][key] = forged_value
                    with self.assertRaisesRegex(
                        aux.EnderEnsembleEvaluationError,
                        "committed inventory diagnostics differs",
                    ):
                        aux.validate_result_json(
                            component,
                            forged,
                            config,
                            expected,
                            **validation_arguments,
                        )

            injected = copy.deepcopy(result)
            injected["data"]["disk_feature_store"]["unfrozen"] = True
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError,
                "store diagnostics keys differs",
            ):
                aux.validate_result_json(
                    component,
                    injected,
                    config,
                    expected,
                    **validation_arguments,
                )

    def test_confirmation_component_binds_reported_store_paths_before_result_validation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = _synthetic_protocol(root)
            fixture = _confirmation_result_provenance_fixture(root)
            component = fixture["component"]
            config = fixture["config"]
            expected = fixture["expected"]
            store_metadata = fixture["store_metadata"]
            store_receipt = fixture["store_receipt"]
            store_inventory_receipt = fixture["store_inventory_receipt"]
            result = fixture["result"]
            assert isinstance(component, aux.ComponentPaths)
            assert isinstance(config, dict)
            assert isinstance(expected, aux.ExpectedCohort)
            assert isinstance(store_metadata, dict)
            assert isinstance(store_receipt, dict)
            assert isinstance(store_inventory_receipt, dict)
            assert isinstance(result, dict)
            pretraining = {
                "input_layout": {"stores": {"jasper": store_receipt}},
                "store_inventory": store_inventory_receipt,
            }
            metadata_path = root / "stores" / "jasper" / "metadata.json"
            semantics = aux._expected_semantics(config)
            artifact = {"synthetic": "artifact"}

            def invoke(
                current_result: dict[str, object],
                *,
                forbid_result_validation: bool,
            ) -> tuple[tuple[np.ndarray, dict[str, object]], object]:
                def load_json(path: Path, _label: str) -> dict[str, object]:
                    if path == metadata_path:
                        return store_metadata
                    if path == component.result:
                        return current_result
                    raise AssertionError(path)

                result_side_effect = (
                    AssertionError("RESULT_VALIDATED_BEFORE_STORE_PATHS")
                    if forbid_result_validation
                    else None
                )
                with patch.object(
                    aux, "_require_regular_output_file"
                ), patch.object(
                    aux,
                    "_lease_confirmation_config",
                    side_effect=lambda *_args, **_kwargs: nullcontext(config),
                ), patch.object(
                    aux,
                    "_require_regular_unlinked_receipt_file",
                    return_value=metadata_path,
                ), patch.object(
                    aux, "_load_json", side_effect=load_json
                ), patch.object(
                    aux,
                    "validate_result_json",
                    return_value=semantics,
                    side_effect=result_side_effect,
                ) as result_validation, patch.object(
                    aux,
                    "validate_prediction_artifact",
                    return_value=np.zeros(len(expected.frame), dtype=np.float64),
                ), patch.object(
                    aux, "_artifact_receipt", return_value=artifact
                ):
                    validated = aux._validate_confirmation_component(
                        protocol,
                        component,
                        expected,
                        pretraining,
                    )
                return validated, result_validation

            (raw, current_artifact), result_validation = invoke(
                result,
                forbid_result_validation=False,
            )
            self.assertEqual(len(raw), len(expected.frame))
            self.assertEqual(current_artifact, artifact)
            result_validation.assert_called_once()
            self.assertIs(
                result_validation.call_args.kwargs["store_metadata"],
                store_metadata,
            )
            self.assertIs(
                result_validation.call_args.kwargs["store_receipt"],
                store_receipt,
            )
            self.assertIs(
                result_validation.call_args.kwargs["store_inventory_receipt"],
                store_inventory_receipt,
            )

            for key, forged_path in (
                ("directory", root / "stores" / "forged"),
                ("feature_path", root / "stores" / "jasper" / "forged.bin"),
                (
                    "manifest_path",
                    root / "stores" / "jasper" / "forged.parquet",
                ),
            ):
                with self.subTest(path=key):
                    forged = copy.deepcopy(result)
                    forged["data"]["disk_feature_store"][key] = str(forged_path)
                    with self.assertRaisesRegex(
                        aux.EnderEnsembleEvaluationError,
                        f"store.{key} differs",
                    ):
                        invoke(forged, forbid_result_validation=True)

    def test_tabm_result_contract_distinguishes_legacy_and_full_metadata(self) -> None:
        target_transform = {
            "type": "residual_to_benchmark",
            "benchmark_col": aux.BENCHMARK_ENDER20,
            "era_col": aux.ERA_COLUMN,
            "per_era": True,
            "fit_intercept": True,
        }
        legacy = {"model": {"target_transform": target_transform}, "output": {}}
        self.assertIsNone(
            aux._validate_tabm_result(legacy, "legacy", metadata_required=False)
        )
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError, "lacks prediction semantics"
        ):
            aux._validate_tabm_result(legacy, "full", metadata_required=True)

        semantics = {
            "training_target": {"transform": {"type": "residual_to_benchmark"}}
        }
        full = copy.deepcopy(legacy)
        full["output"]["prediction_semantics"] = semantics
        self.assertEqual(
            aux._validate_tabm_result(full, "full", metadata_required=True),
            semantics,
        )


class SelectionAndThresholdTests(unittest.TestCase):
    def test_calibration_boundaries_preserve_inclusive_and_strict_operators(self) -> None:
        boundary = _summary(
            bmc_mean=0.002,
            bmc_sharpe=0.25,
            bmc_drawdown=0.10,
            corr_mean=0.012,
            ender20_similarity=0.75,
            ender60_similarity=0.75,
            tabm_similarity=0.75,
        )
        checks = aux.calibration_checks(boundary)

        self.assertTrue(checks["bmc_mean"])
        self.assertTrue(checks["corr_mean"])
        self.assertFalse(checks["bmc_sharpe"])
        self.assertFalse(checks["bmc_max_drawdown"])
        self.assertFalse(checks["ender20_similarity"])
        self.assertFalse(checks["ender60_similarity"])
        self.assertFalse(checks["tabm_similarity"])

    def test_anchored_tie_set_does_not_chain_pairwise_neighbors(self) -> None:
        summaries = {name: _summary(bmc_mean=0.0021) for name in aux.CANDIDATE_NAMES}
        summaries["tyler_w00"] = _summary(bmc_mean=0.00300, bmc_sharpe=0.30)
        summaries["tyler_w10"] = _summary(bmc_mean=0.00295, bmc_sharpe=0.80)
        summaries["tyler_w20_equal5"] = _summary(
            bmc_mean=0.00286, bmc_sharpe=10.0
        )

        selected, evaluations = aux.select_scout_candidate(summaries)

        self.assertEqual(selected, "tyler_w10")
        self.assertTrue(evaluations["tyler_w00"]["in_tie_set"])
        self.assertTrue(evaluations["tyler_w10"]["in_tie_set"])
        self.assertFalse(evaluations["tyler_w20_equal5"]["in_tie_set"])

    def test_tie_breaks_by_sharpe_drawdown_weight_then_name(self) -> None:
        summaries = {name: _summary(bmc_mean=0.0021) for name in aux.CANDIDATE_NAMES}
        for name in ("tyler_w00", "tyler_w10"):
            summaries[name] = _summary(bmc_mean=0.003, bmc_sharpe=0.5)
        summaries["tyler_w10"]["bmc"]["max_drawdown"] = 0.04
        self.assertEqual(aux.select_scout_candidate(summaries)[0], "tyler_w10")

        summaries["tyler_w10"]["bmc"]["max_drawdown"] = 0.05
        self.assertEqual(aux.select_scout_candidate(summaries)[0], "tyler_w00")

    def test_locked_and_confirmation_strict_boundaries_fail(self) -> None:
        locked_boundary = _summary(
            bmc_mean=0.0,
            bmc_sharpe=0.20,
            bmc_drawdown=0.10,
            corr_mean=0.008,
        )
        self.assertFalse(any(aux.locked_checks(locked_boundary).values()))

        confirmation_boundary = _summary(
            bmc_mean=0.002,
            bmc_sharpe=0.35,
            bmc_drawdown=0.15,
            corr_mean=0.012,
            ender20_similarity=0.75,
            ender60_similarity=0.75,
            tabm_similarity=0.75,
        )
        locked_confirmation = _summary(
            bmc_mean=0.0,
            bmc_sharpe=0.20,
            bmc_drawdown=0.15,
            corr_mean=0.008,
        )
        checks = aux.confirmation_checks(
            confirmation_boundary,
            locked_confirmation,
            confirmation_boundary,
        )
        self.assertTrue(checks["calibration_bmc_mean"])
        self.assertTrue(checks["calibration_corr_mean"])
        self.assertTrue(checks["full_bmc_mean"])
        self.assertTrue(checks["full_corr_mean"])
        for name, passed in checks.items():
            if name not in {
                "calibration_bmc_mean",
                "calibration_corr_mean",
                "full_bmc_mean",
                "full_corr_mean",
            }:
                self.assertFalse(passed, name)

    def test_confirmation_calibration_checks_need_no_locked_metrics_object(self) -> None:
        calibration_only = _summary(
            bmc_mean=0.002,
            bmc_sharpe=0.36,
            bmc_drawdown=0.14,
            corr_mean=0.012,
            ender20_similarity=0.74,
            ender60_similarity=0.74,
            tabm_similarity=0.74,
        )

        checks = aux.confirmation_calibration_checks(calibration_only)

        self.assertTrue(all(checks.values()))
        self.assertTrue(all(name.startswith("calibration_") for name in checks))
        self.assertFalse(any("locked" in name or "full" in name for name in checks))


if __name__ == "__main__":
    unittest.main()
