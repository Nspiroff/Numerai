from __future__ import annotations

from contextlib import nullcontext
import hashlib
import importlib
import json
import os
import py_compile
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from agents.code.data.build_full_datasets import build_disk_feature_store
from agents.code.analysis import evaluate_ender20_aux_target_rank_ensemble as evaluator_module
from agents.code.metrics import numerai_metrics
from agents.code.modeling.utils.disk_feature_store import (
    DiskFeatureStoreLoader,
    DiskFeatureView,
)
from agents.code.modeling.utils import disk_feature_store as disk_store_module
from agents.code.modeling.utils import cli as cli_module
from agents.code.modeling.utils import pipeline as pipeline_module
from agents.code.modeling.utils.model_factory import build_model
from agents.code.modeling.utils.model_data import ModelDataLoader
from agents.code.modeling.utils.numerai_cv import build_oof_predictions
from agents.code.modeling.utils.pipeline import (
    _load_committed_feature_store_identity,
    _resolve_feature_store_dir,
    run_training,
)


TARGET = "target_ender_20"
BENCHMARK = "v53_lgbm_ender20"
FEATURES = ["feature_a", "feature_b", "feature_c"]


def _build_fixture(root: Path, *, eras: int = 8, rows_per_era: int = 4):
    total = eras * rows_per_era
    era_values = np.repeat([f"{era:04d}" for era in range(1, eras + 1)], rows_per_era)
    ids = [f"n{row:06d}" for row in range(total)]
    feature_a = (np.arange(total) % 5).astype(np.int8)
    feature_b = ((np.arange(total) * 2 + 1) % 5).astype(np.int8)
    feature_c = ((np.arange(total) * 3 + 2) % 5).astype(np.int8)
    benchmark = np.linspace(0.05, 0.95, total, dtype=np.float64)
    target = (
        0.4
        + 0.03 * (feature_a.astype(np.float32) - 2.0)
        + 0.02 * benchmark.astype(np.float32)
    ).astype(np.float32)
    split = (eras // 2) * rows_per_era

    data = pd.DataFrame(
        {
            "id": ids,
            "era": era_values,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "feature_c": feature_c,
            TARGET: target,
        }
    )
    benchmarks = pd.DataFrame(
        {"id": ids, "era": era_values, BENCHMARK: benchmark}
    )
    train = data.iloc[:split].copy()
    train["data_type"] = "train"
    validation = data.iloc[split:].copy()
    validation["data_type"] = "validation"

    train_path = root / "train.parquet"
    validation_path = root / "validation.parquet"
    train_benchmark_path = root / "train_benchmark_models.parquet"
    validation_benchmark_path = root / "validation_benchmark_models.parquet"
    train.to_parquet(train_path, index=False)
    validation.to_parquet(validation_path, index=False)
    benchmarks.iloc[:split].to_parquet(train_benchmark_path, index=False)
    benchmarks.iloc[split:].to_parquet(validation_benchmark_path, index=False)
    store = build_disk_feature_store(
        root / "store",
        [train_path, validation_path],
        [train_benchmark_path, validation_benchmark_path],
        FEATURES,
        batch_size=5,
        reuse_existing=False,
    )
    eager = data.copy()
    eager[BENCHMARK] = benchmark
    return store, eager


def _loader(store) -> DiskFeatureStoreLoader:
    loader = DiskFeatureStoreLoader(
        store.directory,
        era_col="era",
        target_col=TARGET,
        id_col="id",
        benchmark_col=BENCHMARK,
    )
    loader.configure_x_cols([*FEATURES, "era", BENCHMARK])
    return loader


def _store_receipt(store, receipt_root: Path) -> dict:
    metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))

    def file_receipt(path: Path) -> dict:
        return {
            "path": path.relative_to(receipt_root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }

    return {
        "generation_id": metadata["generation_id"],
        "row_count": metadata["row_count"],
        "feature_count": metadata["feature_count"],
        "feature_order_sha256": metadata["feature_order_sha256"],
        "target_column": metadata["target_column"],
        "metadata": file_receipt(store.metadata_path),
        "manifest": file_receipt(store.manifest_path),
        "features": file_receipt(store.feature_path),
    }


def _inventory_bound_loader(
    store,
    receipt_root: Path,
    receipt: dict,
) -> DiskFeatureStoreLoader:
    return DiskFeatureStoreLoader(
        store.directory,
        era_col="era",
        target_col=TARGET,
        id_col="id",
        benchmark_col=BENCHMARK,
        expected_store_receipt=receipt,
        expected_receipt_root=receipt_root,
        expected_inventory_identity={
            "path": "numerai/agents/experiments/test/confirmation_store_inventory.json",
            "git_blob_id": "a" * 40,
            "checkpoint_commit": "c" * 40,
        },
    )


def _torch_params(*, batch_size: int) -> dict:
    return {
        "architecture": "mlp",
        "hidden_layer_sizes": (12, 6),
        "activation": "gelu",
        "dropout": 0.0,
        "batch_size": batch_size,
        "prediction_batch_size": batch_size,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 1,
        "patience": 1,
        "val_split": "none",
        "device": "cpu",
        "amp": False,
        "seed": 19,
        "num_workers": 0,
        "deterministic": True,
        "verbose": False,
    }


def _synthetic_training_authority(
    root: Path,
    *,
    checkpoint: str = "c" * 40,
    inventory_blob: str | None = "b" * 40,
    mode: str = "confirmation",
    predictions_path: Path | None = None,
    result_path: Path | None = None,
) -> pipeline_module._TrainingAuthority:
    component_name = "jasper"
    return pipeline_module._TrainingAuthority(
        mode=mode,
        component_name=component_name,
        checkpoint=checkpoint,
        protocol=SimpleNamespace(repo_root=root),
        component=SimpleNamespace(
            name=component_name,
            config=root / "config.py",
            predictions=(predictions_path or root / "reserved-predictions.parquet"),
            result=(result_path or root / "reserved-result.json"),
        ),
        pre_run_receipt_path=root / "pre-run.json",
        pre_run_receipt_sha256="d" * 64,
        inventory_blob=inventory_blob,
        confirmation_pretraining_receipt_path=(
            root / "pretraining.json" if mode == "confirmation" else None
        ),
        confirmation_pretraining_receipt_sha256=(
            "e" * 64 if mode == "confirmation" else None
        ),
    )


def _synthetic_reservation_scope(
    predictions_path: Path,
    result_path: Path,
):
    reserved = SimpleNamespace(
        predictions_path=Path(os.path.abspath(predictions_path)),
        results_path=Path(os.path.abspath(result_path)),
        predictions_stream=None,
        results_stream=None,
        identities=mock.Mock(return_value={}),
        completion_identities=mock.Mock(return_value={}),
    )
    scope = mock.MagicMock()
    scope.__enter__.return_value = reserved
    scope.__exit__.return_value = None
    return scope


class TestDiskFeatureStoreModeling(unittest.TestCase):
    def test_ordered_and_block_shuffled_batches_visit_each_row_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, eager = _build_fixture(Path(tmp))
            loader = _loader(store)
            try:
                view = loader.load(loader.eras.unique()).X
                self.assertIsInstance(view, DiskFeatureView)

                ordered_values = []
                ordered_positions = []
                for values, positions in view.iter_feature_batches(
                    3, shuffle_blocks=False
                ):
                    ordered_values.append(values)
                    ordered_positions.extend(positions.tolist())
                self.assertEqual(ordered_positions, list(range(len(eager))))
                np.testing.assert_array_equal(
                    np.concatenate(ordered_values), eager[FEATURES].to_numpy()
                )

                shuffled_positions = []
                for _, positions in view.iter_feature_batches(
                    2, shuffle_blocks=True, seed=7, block_rows=4
                ):
                    shuffled_positions.extend(positions.tolist())
                    physical = view.row_offsets[positions]
                    self.assertTrue(np.all(np.diff(physical) >= 0))
                    self.assertEqual(len(set((physical // 4).tolist())), 1)
                self.assertEqual(sorted(shuffled_positions), list(range(len(eager))))
                self.assertNotEqual(shuffled_positions, list(range(len(eager))))

                subset = loader.load(["0002", "0006"])
                expected = eager[eager["era"].isin(["0002", "0006"])]
                self.assertEqual(subset.id.tolist(), expected["id"].tolist())
                self.assertEqual(subset.X.row_offsets.tolist(), expected.index.tolist())
                duplicate_request = loader.load(["0002", "0002", "0006"])
                self.assertEqual(duplicate_request.id.tolist(), expected["id"].tolist())
            finally:
                loader.close()

    def test_target_residualization_and_training_match_eager_for_full_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, eager = _build_fixture(Path(tmp))
            loader = _loader(store)
            try:
                disk_batch = loader.load(loader.eras.unique())
                eager_X = eager[[*FEATURES, "era", BENCHMARK]]
                model_config = {
                    "target_transform": {
                        "type": "residual_to_benchmark",
                        "benchmark_col": BENCHMARK,
                        "era_col": "era",
                        "per_era": True,
                        "fit_intercept": True,
                    }
                }
                eager_model = build_model(
                    "TorchTabularRegressor",
                    _torch_params(batch_size=len(eager)),
                    model_config,
                    feature_cols=FEATURES,
                )
                disk_model = build_model(
                    "TorchTabularRegressor",
                    _torch_params(batch_size=len(eager)),
                    model_config,
                    feature_cols=FEATURES,
                )
                eager_model.fit(eager_X, eager[TARGET])
                disk_model.fit(disk_batch.X, disk_batch.y)
                eager_predictions = eager_model.predict(eager_X)
                disk_predictions = disk_model.predict(disk_batch.X)
                np.testing.assert_allclose(
                    disk_predictions, eager_predictions, rtol=1e-6, atol=1e-6
                )
                self.assertEqual(disk_model.data_mode_, "disk_feature_store")
                self.assertEqual(disk_model.disk_train_rows_, len(eager))
                self.assertEqual(disk_model.disk_rows_per_epoch_, [len(eager)])
            finally:
                loader.close()

    def test_capped_cv_preserves_oof_order_and_reports_disk_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, eager = _build_fixture(Path(tmp))
            loader = _loader(store)
            try:
                oof, cv_meta = build_oof_predictions(
                    loader.eras,
                    loader,
                    "TorchTabularRegressor",
                    _torch_params(batch_size=16),
                    {},
                    {
                        "n_splits": 4,
                        "embargo": 0,
                        "mode": "expanding",
                        "min_train_size": 0,
                    },
                    5,
                    23,
                    "id",
                    "era",
                    TARGET,
                    feature_cols=FEATURES,
                )
                expected = eager[eager["era"].isin([f"{era:04d}" for era in range(3, 9)])]
                self.assertEqual(oof["id"].tolist(), expected["id"].tolist())
                self.assertEqual(oof["era"].tolist(), expected["era"].tolist())
                self.assertTrue(np.isfinite(oof["prediction"]).all())
                self.assertEqual(len(cv_meta["folds"]), 3)
                for fold in cv_meta["folds"]:
                    self.assertEqual(fold["train_rows"], 5)
                    diagnostics = fold["model_diagnostics"]
                    self.assertEqual(diagnostics["data_mode"], "disk_feature_store")
                    self.assertEqual(diagnostics["disk_rows_per_epoch"], [5])
            finally:
                loader.close()

    def test_disk_training_rejects_worker_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = _build_fixture(Path(tmp))
            loader = _loader(store)
            try:
                batch = loader.load(loader.eras.unique())
                params = _torch_params(batch_size=16)
                params["num_workers"] = 1
                model = build_model(
                    "TorchTabularRegressor", params, feature_cols=FEATURES
                )
                with self.assertRaisesRegex(ValueError, "num_workers=0"):
                    model.fit(batch.X, batch.y)
            finally:
                loader.close()

    def test_retired_generation_is_cleaned_after_reader_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, _ = _build_fixture(root)
            loader = _loader(store)
            old_feature_path = store.feature_path
            try:
                rebuilt = build_disk_feature_store(
                    store.directory,
                    [root / "train.parquet", root / "validation.parquet"],
                    [
                        root / "train_benchmark_models.parquet",
                        root / "validation_benchmark_models.parquet",
                    ],
                    FEATURES,
                    batch_size=5,
                    reuse_existing=False,
                )
                self.assertNotEqual(rebuilt.generation_id, store.generation_id)
                metadata = json.loads(
                    rebuilt.metadata_path.read_text(encoding="utf-8")
                )
                self.assertIn(
                    store.generation_id,
                    metadata["retired_generation_ids"],
                )
            finally:
                loader.close()
            self.assertFalse(old_feature_path.exists())

    def test_close_does_not_delete_artifacts_named_by_corrupt_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = _build_fixture(Path(tmp))
            loader = _loader(store)
            metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
            metadata["generation_id"] = uuid.uuid4().hex
            store.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            loader.close()
            self.assertTrue(store.feature_path.is_file())
            self.assertTrue(store.manifest_path.is_file())

    def test_disk_internal_validation_honors_early_stopping(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = _build_fixture(Path(tmp))
            loader = _loader(store)
            try:
                batch = loader.load(loader.eras.unique())
                params = _torch_params(batch_size=16)
                params.update(
                    {
                        "learning_rate": 0.0,
                        "max_epochs": 5,
                        "patience": 1,
                        "val_split": "random_rows",
                        "val_fraction": 0.25,
                    }
                )
                model = build_model(
                    "TorchTabularRegressor", params, feature_cols=FEATURES
                )
                model.fit(batch.X, batch.y)
                self.assertEqual(model.best_epoch_, 1)
                self.assertEqual(model.epochs_trained_, 2)
                self.assertEqual(len(model.training_history_), 2)
                self.assertGreater(model.disk_validation_rows_, 0)
            finally:
                loader.close()

    def test_malformed_store_metadata_and_offsets_fail_closed(self):
        mutations = (
            "shape",
            "filename",
            "payload_hash",
            "manifest_hash",
            "offsets",
            "truncated",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                store, _ = _build_fixture(Path(tmp))
                metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
                if mutation == "shape":
                    metadata["feature_count"] += 1
                    store.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                elif mutation == "filename":
                    metadata["features"]["filename"] = "unrelated.bin"
                    store.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                elif mutation == "manifest_hash":
                    with store.manifest_path.open("r+b") as stream:
                        stream.seek(store.manifest_path.stat().st_size // 2)
                        original = stream.read(1)
                        stream.seek(-1, 1)
                        stream.write(bytes([original[0] ^ 1]))
                elif mutation == "payload_hash":
                    with store.feature_path.open("r+b") as stream:
                        stream.seek(store.feature_path.stat().st_size // 2)
                        original = stream.read(1)
                        stream.seek(-1, 1)
                        stream.write(bytes([original[0] ^ 1]))
                elif mutation == "offsets":
                    manifest = pd.read_parquet(store.manifest_path)
                    manifest.loc[1, "row_offset"] = 0
                    manifest.to_parquet(store.manifest_path, index=False)
                    metadata["manifest"]["size_bytes"] = store.manifest_path.stat().st_size
                    metadata["manifest"]["sha256"] = hashlib.sha256(
                        store.manifest_path.read_bytes()
                    ).hexdigest()
                    store.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                else:
                    with store.feature_path.open("r+b") as stream:
                        stream.truncate(store.feature_path.stat().st_size - 1)
                with self.assertRaises(ValueError):
                    DiskFeatureStoreLoader(
                        store.directory,
                        era_col="era",
                        target_col=TARGET,
                        id_col="id",
                        benchmark_col=BENCHMARK,
                    )

    def test_store_loader_rejects_linked_artifacts_and_reparse_directories(self):
        with self.assertRaisesRegex(ValueError, "parent traversal"):
            _resolve_feature_store_dir(
                Path("v5.3") / ".." / "alternate-store",
                data_version="v5.3",
                target_col=TARGET,
            )

        with tempfile.TemporaryDirectory() as tmp:
            store, _ = _build_fixture(Path(tmp))
            hardlink = store.directory / "payload-hardlink.bin"
            hardlink.hardlink_to(store.feature_path)
            with self.assertRaisesRegex(ValueError, "hard linked"):
                _loader(store)

        with tempfile.TemporaryDirectory() as tmp:
            store, _ = _build_fixture(Path(tmp))

            def fake_is_symlink(path: Path) -> bool:
                return path == store.feature_path

            with mock.patch.object(Path, "is_symlink", fake_is_symlink):
                with self.assertRaisesRegex(ValueError, "link or reparse"):
                    _loader(store)

        with tempfile.TemporaryDirectory() as tmp:
            store, _ = _build_fixture(Path(tmp))
            actual_lstat = Path.lstat
            reparse_flag = 0x400

            def fake_lstat(path: Path):
                inspected = actual_lstat(path)
                if path == store.directory:
                    return SimpleNamespace(
                        st_mode=inspected.st_mode,
                        st_nlink=inspected.st_nlink,
                        st_file_attributes=reparse_flag,
                    )
                return inspected

            with mock.patch.object(Path, "is_symlink", return_value=False), mock.patch.object(
                Path, "lstat", fake_lstat
            ):
                with self.assertRaisesRegex(ValueError, "link or reparse"):
                    _loader(store)

    def test_committed_inventory_identity_rejects_coordinated_store_rewrites(self):
        mutations = ("payload_and_metadata", "manifest_and_metadata")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                store, _ = _build_fixture(root)
                receipt = _store_receipt(store, root)
                metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
                if mutation == "payload_and_metadata":
                    with store.feature_path.open("r+b") as stream:
                        stream.seek(store.feature_path.stat().st_size // 2)
                        original = stream.read(1)
                        stream.seek(-1, 1)
                        stream.write(bytes([original[0] ^ 1]))
                    metadata["features"]["sha256"] = hashlib.sha256(
                        store.feature_path.read_bytes()
                    ).hexdigest()
                else:
                    manifest = pd.read_parquet(store.manifest_path)
                    manifest.loc[0, BENCHMARK] += 0.01
                    manifest.to_parquet(store.manifest_path, index=False)
                    metadata["manifest"]["size_bytes"] = store.manifest_path.stat().st_size
                    metadata["manifest"]["sha256"] = hashlib.sha256(
                        store.manifest_path.read_bytes()
                    ).hexdigest()
                store.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

                with mock.patch(
                    "agents.code.modeling.utils.disk_feature_store.np.memmap",
                    side_effect=AssertionError("MODEL_INPUT_ACCESSED"),
                ) as memmap:
                    with self.assertRaisesRegex(
                        ValueError,
                        "metadata (size|SHA-256) differs from inventory",
                    ):
                        _inventory_bound_loader(store, root, receipt)
                memmap.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "requires Windows no-write sharing")
    def test_store_file_leases_block_mutation_after_hash_before_use(self):
        for target_label in ("metadata", "payload", "manifest"):
            with self.subTest(target=target_label), tempfile.TemporaryDirectory() as tmp:
                store, _ = _build_fixture(Path(tmp))
                original_sha256 = disk_store_module._ReadOnlyFileLease.sha256

                def sha256_then_mutate(lease, *args, **kwargs):
                    digest = original_sha256(lease, *args, **kwargs)
                    if lease.label == target_label:
                        with lease.path.open("r+b") as stream:
                            original = stream.read(1)
                            stream.seek(0)
                            stream.write(bytes([original[0] ^ 1]))
                    return digest

                with mock.patch.object(
                    disk_store_module._ReadOnlyFileLease,
                    "sha256",
                    sha256_then_mutate,
                ), mock.patch.object(
                    disk_store_module.np,
                    "memmap",
                    wraps=np.memmap,
                ) as memmap, mock.patch.object(
                    disk_store_module.pq,
                    "ParquetFile",
                    wraps=disk_store_module.pq.ParquetFile,
                ) as parquet:
                    with self.assertRaises((OSError, ValueError)):
                        _loader(store)
                if target_label in {"metadata", "payload"}:
                    memmap.assert_not_called()
                if target_label == "manifest":
                    parquet.assert_not_called()

    def test_committed_inventory_identity_is_persisted_in_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, _ = _build_fixture(root)
            receipt = _store_receipt(store, root)
            loader = _inventory_bound_loader(store, root, receipt)
            try:
                diagnostics = loader.diagnostics
                self.assertEqual(
                    diagnostics["metadata_sha256"],
                    receipt["metadata"]["sha256"],
                )
                self.assertEqual(
                    diagnostics["feature_sha256"],
                    receipt["features"]["sha256"],
                )
                self.assertEqual(
                    diagnostics["manifest_sha256"],
                    receipt["manifest"]["sha256"],
                )
                self.assertEqual(
                    diagnostics["committed_inventory"],
                    {
                        "path": (
                            "numerai/agents/experiments/test/"
                            "confirmation_store_inventory.json"
                        ),
                        "git_blob_id": "a" * 40,
                        "checkpoint_commit": "c" * 40,
                    },
                )
            finally:
                loader.close()

    def test_pipeline_loads_the_clean_head_inventory_blob(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            numerai_root = root / "numerai"
            agents_root = numerai_root / "agents"
            store, _ = _build_fixture(root)
            receipt = _store_receipt(store, root)
            inventory_relative = Path(
                "numerai/agents/experiments/test/confirmation_store_inventory.json"
            )
            inventory_path = root / inventory_relative
            inventory_path.parent.mkdir(parents=True)
            inventory = {
                "input_layout": {
                    "type": "dedicated_target_stores",
                    "stores": {"jasper": receipt},
                }
            }
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

            git_commands = (
                ("init",),
                ("config", "user.email", "test@example.com"),
                ("config", "user.name", "Ender Test"),
                ("add", inventory_relative.as_posix()),
                ("commit", "-m", "freeze inventory"),
            )
            for arguments in git_commands:
                subprocess.run(
                    ["git", *arguments],
                    cwd=root,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            blob = subprocess.run(
                ["git", "rev-parse", f"HEAD:{inventory_relative.as_posix()}"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).stdout.strip()

            with mock.patch(
                "agents.code.modeling.utils.pipeline.REPO_DIR", root
            ), mock.patch(
                "agents.code.modeling.utils.pipeline.NUMERAI_DIR", numerai_root
            ), mock.patch(
                "agents.code.modeling.utils.pipeline.BASE_DIR", agents_root
            ):
                actual_receipt, identity = _load_committed_feature_store_identity(
                    "agents/experiments/test/confirmation_store_inventory.json",
                    target_col=TARGET,
                    store_path=store.directory,
                    expected_commit=commit,
                    expected_blob=blob,
                )
                self.assertEqual(actual_receipt, receipt)
                self.assertEqual(
                    identity,
                    {
                        "path": inventory_relative.as_posix(),
                        "git_blob_id": blob,
                        "checkpoint_commit": commit,
                    },
                )

                original_run_git = pipeline_module._run_git
                forged_inventory = {
                    "input_layout": {
                        "type": "dedicated_target_stores",
                        "stores": {
                            "forged": {
                                **receipt,
                                "target_column": "target_forged_20",
                            }
                        },
                    }
                }
                mutated_after_hash = False

                def mutate_after_hash(*arguments, text=True):
                    nonlocal mutated_after_hash
                    completed = original_run_git(*arguments, text=text)
                    if arguments and arguments[0] == "hash-object":
                        inventory_path.write_text(
                            json.dumps(forged_inventory), encoding="utf-8"
                        )
                        mutated_after_hash = True
                    return completed

                with mock.patch.object(
                    pipeline_module,
                    "_run_git",
                    side_effect=mutate_after_hash,
                ):
                    frozen_receipt, frozen_identity = (
                        _load_committed_feature_store_identity(
                            "agents/experiments/test/confirmation_store_inventory.json",
                            target_col=TARGET,
                            store_path=store.directory,
                            expected_commit=commit,
                            expected_blob=blob,
                        )
                    )
                self.assertTrue(mutated_after_hash)
                self.assertEqual(frozen_receipt, receipt)
                self.assertEqual(frozen_identity["git_blob_id"], blob)
                self.assertEqual(frozen_identity["checkpoint_commit"], commit)

                with self.assertRaisesRegex(ValueError, "not a clean committed file"):
                    _load_committed_feature_store_identity(
                        "agents/experiments/test/confirmation_store_inventory.json",
                        target_col=TARGET,
                        store_path=store.directory,
                        expected_commit=commit,
                        expected_blob=blob,
                    )

    def test_changed_head_is_rejected_before_config_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            inventory_path = root / "inventory.json"
            config_path.write_text("{}", encoding="utf-8")
            inventory_path.write_text("{}", encoding="utf-8")
            commands = (
                ("init",),
                ("config", "user.email", "test@example.com"),
                ("config", "user.name", "Ender Test"),
                ("add", "config.json", "inventory.json"),
                ("commit", "-m", "checkpoint A"),
            )
            for arguments in commands:
                subprocess.run(
                    ["git", *arguments],
                    cwd=root,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            commit_a = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            blob_a = subprocess.run(
                ["git", "rev-parse", "HEAD:inventory.json"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            inventory_path.write_text('{"changed":true}', encoding="utf-8")
            subprocess.run(
                ["git", "add", "inventory.json"], cwd=root, check=True
            )
            subprocess.run(
                ["git", "commit", "-m", "checkpoint B"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            with mock.patch.object(
                pipeline_module, "REPO_DIR", root
            ), mock.patch.object(
                pipeline_module, "_require_frozen_python_runtime"
            ), mock.patch.object(
                pipeline_module,
                "_preflight_confirmation_training_authority",
                return_value=(commit_a, ()),
            ), mock.patch.object(
                pipeline_module,
                "_derive_confirmation_training_authority",
                return_value=_synthetic_training_authority(
                    root,
                    checkpoint=commit_a,
                    inventory_blob=blob_a,
                ),
            ) as derive_authority, mock.patch.object(
                pipeline_module,
                "load_config",
                side_effect=AssertionError("CONFIG_EVALUATED"),
            ) as load_config:
                with self.assertRaisesRegex(
                    ValueError,
                    "HEAD is not the frozen training checkpoint",
                ):
                    run_training(
                        config_path,
                        confirmation_component="jasper",
                        confirmation_pre_run_receipt=Path("pre-run.json"),
                        confirmation_pre_run_receipt_sha256="d" * 64,
                        confirmation_pretraining_receipt=Path("pretraining.json"),
                        confirmation_pretraining_receipt_sha256="e" * 64,
                    )

            load_config.assert_not_called()
            derive_authority.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "requires Windows no-write sharing")
    def test_frozen_config_lease_blocks_mutation_during_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            modeling_source = (
                root / "numerai/agents/code/modeling/model_factory.py"
            )
            metrics_source = (
                root / "numerai/agents/code/metrics/numerai_metrics.py"
            )
            modeling_source.parent.mkdir(parents=True)
            metrics_source.parent.mkdir(parents=True)
            modeling_source.write_text("VALUE = 1\n", encoding="utf-8")
            metrics_source.write_text("VALUE = 1\n", encoding="utf-8")
            commands = (
                ("init",),
                ("config", "user.email", "test@example.com"),
                ("config", "user.name", "Ender Test"),
                (
                    "add",
                    "config.json",
                    "numerai/agents/code/modeling/model_factory.py",
                    "numerai/agents/code/metrics/numerai_metrics.py",
                ),
                ("commit", "-m", "frozen config"),
            )
            for arguments in commands:
                subprocess.run(
                    ["git", *arguments],
                    cwd=root,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            marker_path = root / "consumption.json"
            marker_path.write_bytes(
                json.dumps({}, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            )
            completion_claim_path = root / "completion.claimed.json"
            completion_claim_path.write_bytes(
                evaluator_module._receipt_bytes(
                    evaluator_module._claim_payload(
                        "confirmation-train-jasper-completion"
                    )
                )
            )

            def mutate_config(_path):
                config_path.write_text('{"changed":true}', encoding="utf-8")
                raise AssertionError("unreachable")

            with mock.patch.object(
                pipeline_module, "REPO_DIR", root
            ), mock.patch.object(
                pipeline_module, "_require_frozen_python_runtime"
            ), mock.patch.object(
                pipeline_module,
                "_preflight_confirmation_training_authority",
                return_value=(commit, ()),
            ), mock.patch.object(
                pipeline_module,
                "_derive_confirmation_training_authority",
                return_value=_synthetic_training_authority(
                    root,
                    checkpoint=commit,
                ),
            ), mock.patch.object(
                pipeline_module,
                "_frozen_source_module_scope",
                side_effect=lambda _leases: nullcontext(),
            ), mock.patch(
                "agents.code.analysis.evaluate_ender20_aux_target_rank_ensemble."
                "claim_component_training_consumption",
                return_value=(marker_path, {}),
            ), mock.patch.object(
                evaluator_module,
                "claim_component_training_completion",
                return_value=completion_claim_path,
            ), mock.patch.object(
                pipeline_module,
                "_ExclusiveOutputReservations",
                return_value=_synthetic_reservation_scope(
                    root / "reserved-predictions.parquet",
                    root / "reserved-result.json",
                ),
            ), mock.patch.object(
                pipeline_module, "load_config", side_effect=mutate_config
            ):
                with self.assertRaises(OSError):
                    run_training(
                        config_path,
                        confirmation_component="jasper",
                        confirmation_pre_run_receipt=Path("pre-run.json"),
                        confirmation_pre_run_receipt_sha256="d" * 64,
                        confirmation_pretraining_receipt=Path("pretraining.json"),
                        confirmation_pretraining_receipt_sha256="e" * 64,
                    )

            def mutate_lazy_model_source(*_args, **_kwargs):
                modeling_source.write_text("VALUE = 2\n", encoding="utf-8")
                raise AssertionError("MODEL_SOURCE_MUTATION_SUCCEEDED")

            with mock.patch.object(
                pipeline_module, "REPO_DIR", root
            ), mock.patch.object(
                pipeline_module, "_require_frozen_python_runtime"
            ), mock.patch.object(
                pipeline_module,
                "_preflight_confirmation_training_authority",
                return_value=(commit, ()),
            ), mock.patch.object(
                pipeline_module,
                "_derive_confirmation_training_authority",
                return_value=_synthetic_training_authority(
                    root,
                    checkpoint=commit,
                ),
            ), mock.patch.object(
                pipeline_module,
                "_frozen_source_module_scope",
                side_effect=lambda _leases: nullcontext(),
            ), mock.patch(
                "agents.code.analysis.evaluate_ender20_aux_target_rank_ensemble."
                "claim_component_training_consumption",
                return_value=(marker_path, {}),
            ), mock.patch.object(
                evaluator_module,
                "claim_component_training_completion",
                return_value=completion_claim_path,
            ), mock.patch.object(
                pipeline_module,
                "_ExclusiveOutputReservations",
                return_value=_synthetic_reservation_scope(
                    root / "reserved-predictions.parquet",
                    root / "reserved-result.json",
                ),
            ), mock.patch.object(
                pipeline_module,
                "_run_training_impl",
                side_effect=mutate_lazy_model_source,
            ) as implementation:
                with self.assertRaises(OSError):
                    run_training(
                        config_path,
                        confirmation_component="jasper",
                        confirmation_pre_run_receipt=Path("pre-run.json"),
                        confirmation_pre_run_receipt_sha256="d" * 64,
                        confirmation_pretraining_receipt=Path("pretraining.json"),
                        confirmation_pretraining_receipt_sha256="e" * 64,
                    )
            implementation.assert_called_once()

    def test_frozen_source_scope_ignores_valid_timestamp_poison_pyc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relative = Path(
                "numerai/agents/code/modeling/models/lgbm_regressor.py"
            )
            source_path = root / relative
            source_path.parent.mkdir(parents=True)

            def padded_source(value: str) -> bytes:
                prefix = f'SENTINEL = "{value}"\n'.encode("utf-8")
                return prefix + (b"#" * (255 - len(prefix))) + b"\n"

            verified_source = padded_source("verified-source")
            poison_source = padded_source("poisoned-cache")
            self.assertEqual(len(verified_source), len(poison_source))
            source_path.write_bytes(verified_source)
            poison_path = root / "poison.py"
            poison_path.write_bytes(poison_source)
            timestamp = 1_700_000_000
            os.utime(source_path, (timestamp, timestamp))
            os.utime(poison_path, (timestamp, timestamp))

            cache_path = Path(importlib.util.cache_from_source(str(source_path)))
            py_compile.compile(
                str(poison_path),
                cfile=str(cache_path),
                dfile=str(source_path),
                doraise=True,
                invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
            )
            cache_bytes = cache_path.read_bytes()
            self.assertEqual(cache_bytes[4:8], b"\x00" * 4)
            self.assertEqual(
                int.from_bytes(cache_bytes[8:12], "little"), timestamp
            )
            self.assertEqual(
                int.from_bytes(cache_bytes[12:16], "little"),
                len(verified_source),
            )

            (root / ".gitignore").write_text(
                "__pycache__/\n*.pyc\n", encoding="utf-8"
            )
            subprocess.run(
                ["git", "init", "--quiet"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            ignored = subprocess.run(
                [
                    "git",
                    "check-ignore",
                    "--quiet",
                    cache_path.relative_to(root).as_posix(),
                ],
                cwd=root,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(ignored.returncode, 0)

            probe_spec = importlib.util.spec_from_file_location(
                f"poison_probe_{uuid.uuid4().hex}", source_path
            )
            self.assertIsNotNone(probe_spec)
            assert probe_spec is not None and probe_spec.loader is not None
            probe = importlib.util.module_from_spec(probe_spec)
            probe_spec.loader.exec_module(probe)
            self.assertEqual(probe.SENTINEL, "poisoned-cache")

            module_name = "agents.code.modeling.models.lgbm_regressor"
            lease = disk_store_module._ReadOnlyFileLease(
                source_path, "synthetic verified source"
            )
            try:
                with mock.patch.object(
                    pipeline_module, "REPO_DIR", root
                ), mock.patch.object(
                    pipeline_module,
                    "_FROZEN_SOURCE_MODULES",
                    {module_name: relative.as_posix()},
                ):
                    with pipeline_module._frozen_source_module_scope((lease,)):
                        loaded = importlib.import_module(module_name)
                        self.assertEqual(loaded.SENTINEL, "verified-source")
                        self.assertEqual(loaded.__file__, str(source_path))
                        self.assertIsNone(loaded.__cached__)
            finally:
                lease.close()

    def test_modeling_cli_forwards_frozen_inventory_authority(self):
        arguments = SimpleNamespace(
            config=Path("confirmation.py"),
            output_dir=Path("output"),
            scout_component=None,
            scout_pre_run_receipt=None,
            scout_pre_run_receipt_sha256=None,
            confirmation_component="jasper",
            confirmation_pre_run_receipt=Path("pre-run.json"),
            confirmation_pre_run_receipt_sha256="a" * 64,
            confirmation_pretraining_receipt=Path("pretraining.json"),
            confirmation_pretraining_receipt_sha256="b" * 64,
        )
        with mock.patch.object(
            cli_module, "parse_args", return_value=arguments
        ), mock.patch.object(cli_module, "run_training") as training:
            cli_module.main()
        training.assert_called_once_with(
            arguments.config,
            arguments.output_dir,
            scout_component=arguments.scout_component,
            scout_pre_run_receipt=arguments.scout_pre_run_receipt,
            scout_pre_run_receipt_sha256=(
                arguments.scout_pre_run_receipt_sha256
            ),
            confirmation_component=arguments.confirmation_component,
            confirmation_pre_run_receipt=arguments.confirmation_pre_run_receipt,
            confirmation_pre_run_receipt_sha256=(
                arguments.confirmation_pre_run_receipt_sha256
            ),
            confirmation_pretraining_receipt=(
                arguments.confirmation_pretraining_receipt
            ),
            confirmation_pretraining_receipt_sha256=(
                arguments.confirmation_pretraining_receipt_sha256
            ),
        )

    def test_modeling_cli_forwards_scout_receipt_authority(self):
        arguments = SimpleNamespace(
            config=Path("r1_jasper_d8_t6000.py"),
            output_dir=None,
            scout_component="jasper",
            scout_pre_run_receipt=Path("scout-pre-run.json"),
            scout_pre_run_receipt_sha256="a" * 64,
            confirmation_component=None,
            confirmation_pre_run_receipt=None,
            confirmation_pre_run_receipt_sha256=None,
            confirmation_pretraining_receipt=None,
            confirmation_pretraining_receipt_sha256=None,
        )
        with mock.patch.object(
            cli_module, "parse_args", return_value=arguments
        ), mock.patch.object(cli_module, "run_training") as training:
            cli_module.main()
        training.assert_called_once_with(
            arguments.config,
            arguments.output_dir,
            scout_component=arguments.scout_component,
            scout_pre_run_receipt=arguments.scout_pre_run_receipt,
            scout_pre_run_receipt_sha256=(
                arguments.scout_pre_run_receipt_sha256
            ),
            confirmation_component=arguments.confirmation_component,
            confirmation_pre_run_receipt=arguments.confirmation_pre_run_receipt,
            confirmation_pre_run_receipt_sha256=(
                arguments.confirmation_pre_run_receipt_sha256
            ),
            confirmation_pretraining_receipt=(
                arguments.confirmation_pretraining_receipt
            ),
            confirmation_pretraining_receipt_sha256=(
                arguments.confirmation_pretraining_receipt_sha256
            ),
        )

    def test_confirmation_training_rejects_output_override_before_authority(
        self,
    ):
        with mock.patch.object(
            pipeline_module,
            "_derive_confirmation_training_authority",
            side_effect=AssertionError("AUTHORITY_DERIVED"),
        ) as authority, mock.patch.object(
            pipeline_module,
            "_run_training_impl",
            side_effect=AssertionError("TRAINING_STARTED"),
        ) as implementation:
            with self.assertRaisesRegex(
                ValueError,
                "may not override its bound output directory",
            ):
                run_training(
                    Path("confirmation_jasper.py"),
                    Path("override-output"),
                    confirmation_component="jasper",
                    confirmation_pre_run_receipt=Path("pre-run.json"),
                    confirmation_pre_run_receipt_sha256="a" * 64,
                    confirmation_pretraining_receipt=Path("pretraining.json"),
                    confirmation_pretraining_receipt_sha256="b" * 64,
                )
        authority.assert_not_called()
        implementation.assert_not_called()

    def test_changed_consumption_marker_bytes_reject_before_training_impl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            authority = _synthetic_training_authority(root)
            marker_path = root / "consumption.json"
            changed_bytes = (
                json.dumps(
                    {"state": "TAMPERED"},
                    indent=2,
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            )
            marker_path.write_bytes(changed_bytes)
            claim_payload = {"state": "CONSUMED"}

            with mock.patch.object(
                pipeline_module, "_require_frozen_python_runtime"
            ), mock.patch.object(
                pipeline_module,
                "_preflight_confirmation_training_authority",
                return_value=(authority.checkpoint, ()),
            ), mock.patch.object(
                pipeline_module,
                "_verify_frozen_training_source",
                return_value=(),
            ), mock.patch.object(
                pipeline_module,
                "_derive_confirmation_training_authority",
                return_value=authority,
            ), mock.patch.object(
                pipeline_module,
                "_ExclusiveOutputReservations",
                return_value=_synthetic_reservation_scope(
                    authority.component.predictions,
                    authority.component.result,
                ),
            ), mock.patch(
                "agents.code.analysis.evaluate_ender20_aux_target_rank_ensemble."
                "claim_component_training_consumption",
                return_value=(marker_path, claim_payload),
            ), mock.patch.object(
                pipeline_module,
                "_frozen_source_module_scope",
                side_effect=AssertionError("SOURCE_SCOPE_ENTERED"),
            ) as source_scope, mock.patch.object(
                pipeline_module,
                "_run_training_impl",
                side_effect=AssertionError("TRAINING_STARTED"),
            ) as implementation:
                with self.assertRaisesRegex(
                    ValueError,
                    "Training consumption claim changed before its lease",
                ):
                    run_training(
                        root / "confirmation.py",
                        confirmation_component="jasper",
                        confirmation_pre_run_receipt=root / "pre-run.json",
                        confirmation_pre_run_receipt_sha256="d" * 64,
                        confirmation_pretraining_receipt=(
                            root / "pretraining.json"
                        ),
                        confirmation_pretraining_receipt_sha256="e" * 64,
                    )
            source_scope.assert_not_called()
            implementation.assert_not_called()
            self.assertEqual(marker_path.read_bytes(), changed_bytes)

    def test_pipeline_finalizes_outputs_while_marker_claim_and_reservations_are_held(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            authority = _synthetic_training_authority(root)
            marker_path = root / "consumption.json"
            completion_claim_path = root / "completion.claimed.json"
            completion_path = root / "completion.json"
            claim_payload = {"state": "CONSUMED"}
            completion_payload = {"state": "OUTPUTS_FINALIZED"}
            completion_prefix = "confirmation-train-jasper-completion"
            events: list[str] = []
            active_leases: set[Path] = set()
            original_lease = disk_store_module._ReadOnlyFileLease

            class TrackingLease:
                def __init__(self, path: Path, label: str) -> None:
                    self.path = Path(os.path.abspath(path))
                    self.inner = original_lease(path, label)
                    active_leases.add(self.path)

                def read_bytes(self) -> bytes:
                    return self.inner.read_bytes()

                def close(self) -> None:
                    active_leases.discard(self.path)
                    self.inner.close()

            captured: dict[str, object] = {}

            def claim_consumption(
                _protocol,
                _component,
                _pre_run_path,
                _pre_run_sha,
                reservations,
                **_kwargs,
            ):
                events.append("consumption-claimed")
                captured["reservations"] = reservations
                marker_path.write_bytes(
                    evaluator_module._receipt_bytes(claim_payload)
                )
                return marker_path, claim_payload

            def claim_completion(_protocol, _component, **_kwargs):
                events.append("completion-claimed")
                completion_claim_path.write_bytes(
                    evaluator_module._receipt_bytes(
                        evaluator_module._claim_payload(completion_prefix)
                    )
                )
                return completion_claim_path

            def train(*_args, reserved_outputs=None, **_kwargs):
                events.append("outputs-written")
                self.assertIsNotNone(reserved_outputs)
                captured["reserved"] = reserved_outputs
                reserved_outputs.predictions_stream.write(b"prediction-output")
                reserved_outputs.results_stream.write(b"result-output")
                self.assertIn(Path(os.path.abspath(marker_path)), active_leases)
                self.assertIn(
                    Path(os.path.abspath(completion_claim_path)), active_leases
                )
                return (
                    authority.component.predictions,
                    authority.component.result,
                )

            def complete(
                _protocol,
                _component,
                _pre_run_path,
                _pre_run_sha,
                output_artifacts,
                claim_path,
                **_kwargs,
            ):
                events.append("completion-finalized")
                reserved = captured["reserved"]
                self.assertIsNotNone(reserved.predictions_stream)
                self.assertIsNotNone(reserved.results_stream)
                self.assertEqual(claim_path, completion_claim_path)
                self.assertIn(Path(os.path.abspath(marker_path)), active_leases)
                self.assertIn(
                    Path(os.path.abspath(completion_claim_path)), active_leases
                )
                self.assertEqual(
                    output_artifacts["predictions"]["size_bytes"],
                    len(b"prediction-output"),
                )
                self.assertEqual(
                    output_artifacts["predictions"]["sha256"],
                    hashlib.sha256(b"prediction-output").hexdigest(),
                )
                self.assertEqual(
                    output_artifacts["result"]["size_bytes"],
                    len(b"result-output"),
                )
                self.assertEqual(
                    output_artifacts["result"]["sha256"],
                    hashlib.sha256(b"result-output").hexdigest(),
                )
                completion_path.write_bytes(
                    evaluator_module._receipt_bytes(completion_payload)
                )
                return completion_path, completion_payload

            def observe_completion_report(*_args, **_kwargs):
                events.append("completion-reported")
                self.assertIn(Path(os.path.abspath(marker_path)), active_leases)
                self.assertIn(
                    Path(os.path.abspath(completion_claim_path)), active_leases
                )
                self.assertIn(
                    Path(os.path.abspath(completion_path)), active_leases
                )
                reserved = captured["reserved"]
                self.assertIsNotNone(reserved.predictions_stream)
                self.assertIsNotNone(reserved.results_stream)

            with mock.patch.object(
                pipeline_module, "_require_frozen_python_runtime"
            ), mock.patch.object(
                pipeline_module,
                "_preflight_confirmation_training_authority",
                return_value=(authority.checkpoint, ()),
            ), mock.patch.object(
                pipeline_module,
                "_verify_frozen_training_source",
                return_value=(),
            ), mock.patch.object(
                pipeline_module,
                "_derive_confirmation_training_authority",
                return_value=authority,
            ), mock.patch.object(
                pipeline_module,
                "_frozen_source_module_scope",
                side_effect=lambda _leases: nullcontext(),
            ), mock.patch.object(
                evaluator_module,
                "claim_component_training_consumption",
                side_effect=claim_consumption,
            ), mock.patch.object(
                evaluator_module,
                "claim_component_training_completion",
                side_effect=claim_completion,
            ), mock.patch.object(
                evaluator_module,
                "complete_component_training_consumption",
                side_effect=complete,
            ), mock.patch.object(
                disk_store_module,
                "_ReadOnlyFileLease",
                TrackingLease,
            ), mock.patch.object(
                pipeline_module,
                "_run_training_impl",
                side_effect=train,
            ), mock.patch("builtins.print", side_effect=observe_completion_report):
                outputs = run_training(
                    root / "confirmation.py",
                    confirmation_component="jasper",
                    confirmation_pre_run_receipt=root / "pre-run.json",
                    confirmation_pre_run_receipt_sha256="d" * 64,
                    confirmation_pretraining_receipt=root / "pretraining.json",
                    confirmation_pretraining_receipt_sha256="e" * 64,
                )

            self.assertEqual(
                outputs,
                (
                    authority.component.predictions,
                    authority.component.result,
                ),
            )
            self.assertEqual(
                events,
                [
                    "consumption-claimed",
                    "completion-claimed",
                    "outputs-written",
                    "completion-finalized",
                    "completion-reported",
                ],
            )
            self.assertEqual(active_leases, set())
            reserved = captured["reserved"]
            self.assertIsNone(reserved.predictions_stream)
            self.assertIsNone(reserved.results_stream)

    def test_failed_training_leaves_terminal_marker_and_outputs_without_completion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            authority = _synthetic_training_authority(root)
            marker_path = root / "consumption.json"
            completion_claim_path = root / "completion.claimed.json"
            completion_path = root / "completion.json"
            claim_payload = {"state": "CONSUMED"}
            completion_prefix = "confirmation-train-jasper-completion"

            def claim_consumption(*_args, **_kwargs):
                marker_path.write_bytes(
                    evaluator_module._receipt_bytes(claim_payload)
                )
                return marker_path, claim_payload

            def claim_completion(*_args, **_kwargs):
                completion_claim_path.write_bytes(
                    evaluator_module._receipt_bytes(
                        evaluator_module._claim_payload(completion_prefix)
                    )
                )
                return completion_claim_path

            def fail_training(*_args, reserved_outputs=None, **_kwargs):
                reserved_outputs.predictions_stream.write(b"partial-prediction")
                reserved_outputs.results_stream.write(b"partial-result")
                raise RuntimeError("SYNTHETIC_TRAINING_FAILURE")

            with mock.patch.object(
                pipeline_module, "_require_frozen_python_runtime"
            ), mock.patch.object(
                pipeline_module,
                "_preflight_confirmation_training_authority",
                return_value=(authority.checkpoint, ()),
            ), mock.patch.object(
                pipeline_module,
                "_verify_frozen_training_source",
                return_value=(),
            ), mock.patch.object(
                pipeline_module,
                "_derive_confirmation_training_authority",
                return_value=authority,
            ), mock.patch.object(
                pipeline_module,
                "_frozen_source_module_scope",
                side_effect=lambda _leases: nullcontext(),
            ), mock.patch.object(
                evaluator_module,
                "claim_component_training_consumption",
                side_effect=claim_consumption,
            ), mock.patch.object(
                evaluator_module,
                "claim_component_training_completion",
                side_effect=claim_completion,
            ), mock.patch.object(
                evaluator_module,
                "complete_component_training_consumption",
            ) as complete, mock.patch.object(
                pipeline_module,
                "_run_training_impl",
                side_effect=fail_training,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "SYNTHETIC_TRAINING_FAILURE",
                ):
                    run_training(
                        root / "confirmation.py",
                        confirmation_component="jasper",
                        confirmation_pre_run_receipt=root / "pre-run.json",
                        confirmation_pre_run_receipt_sha256="d" * 64,
                        confirmation_pretraining_receipt=root / "pretraining.json",
                        confirmation_pretraining_receipt_sha256="e" * 64,
                    )

            complete.assert_not_called()
            self.assertEqual(
                marker_path.read_bytes(),
                evaluator_module._receipt_bytes(claim_payload),
            )
            self.assertTrue(completion_claim_path.is_file())
            self.assertEqual(
                authority.component.predictions.read_bytes(),
                b"partial-prediction",
            )
            self.assertEqual(
                authority.component.result.read_bytes(),
                b"partial-result",
            )
            self.assertFalse(completion_path.exists())

    def test_confirmation_config_requires_receipts_before_evaluation(self):
        config_path = Path("confirmation_jasper_d8_t6000.py")
        with mock.patch.object(
            pipeline_module,
            "load_config",
            side_effect=AssertionError("CONFIG_EVALUATED"),
        ) as load_config:
            with self.assertRaisesRegex(
                ValueError,
                "requires its finalized pre-run and pretraining receipt authority",
            ):
                run_training(config_path)
        load_config.assert_not_called()

    def test_scout_config_requires_receipt_authority_before_evaluation(self):
        config_path = Path("r1_jasper_d8_t6000.py")
        with mock.patch.object(
            pipeline_module,
            "load_config",
            side_effect=AssertionError("CONFIG_EVALUATED"),
        ) as load_config, mock.patch.object(
            pipeline_module,
            "_derive_scout_training_authority",
            side_effect=AssertionError("AUTHORITY_DERIVED"),
        ) as authority:
            with self.assertRaisesRegex(
                ValueError,
                "Scout config requires its finalized pre-run receipt authority",
            ):
                run_training(config_path)
        load_config.assert_not_called()
        authority.assert_not_called()

    def test_scout_runtime_policy_rejects_missing_or_dirty_prefix_before_authority(
        self,
    ):
        arguments = {
            "scout_component": "jasper",
            "scout_pre_run_receipt": Path("scout-pre-run.json"),
            "scout_pre_run_receipt_sha256": "a" * 64,
        }
        with mock.patch.object(
            pipeline_module,
            "_derive_scout_training_authority",
            side_effect=AssertionError("AUTHORITY_DERIVED"),
        ) as authority, mock.patch.object(
            pipeline_module,
            "_run_training_impl",
            side_effect=AssertionError("TRAINING_STARTED"),
        ) as implementation, mock.patch.object(
            pipeline_module.sys,
            "flags",
            SimpleNamespace(dont_write_bytecode=0),
        ), mock.patch.object(
            pipeline_module.sys, "dont_write_bytecode", False
        ), mock.patch.object(
            pipeline_module.sys, "pycache_prefix", None
        ), mock.patch.object(
            pipeline_module.sys, "_xoptions", {}
        ):
            with self.assertRaisesRegex(
                ValueError,
                "requires Python -B with an isolated",
            ):
                run_training(Path("r1_jasper_d8_t6000.py"), **arguments)
        authority.assert_not_called()
        implementation.assert_not_called()

        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "isolated-pycache"
            prefix.mkdir()
            injected = prefix / "injected.pyc"
            injected.write_bytes(b"ignored cache")
            with mock.patch.object(
                pipeline_module,
                "_derive_scout_training_authority",
                side_effect=AssertionError("AUTHORITY_DERIVED"),
            ) as authority, mock.patch.object(
                pipeline_module,
                "_run_training_impl",
                side_effect=AssertionError("TRAINING_STARTED"),
            ) as implementation, mock.patch.object(
                pipeline_module.sys,
                "flags",
                SimpleNamespace(dont_write_bytecode=1),
            ), mock.patch.object(
                pipeline_module.sys, "dont_write_bytecode", True
            ), mock.patch.object(
                pipeline_module.sys, "pycache_prefix", str(prefix)
            ), mock.patch.object(
                pipeline_module.sys,
                "_xoptions",
                {"pycache_prefix": str(prefix)},
            ), mock.patch.object(
                disk_store_module, "_require_plain_directory_chain"
            ):
                with self.assertRaisesRegex(ValueError, "freshly empty"):
                    run_training(
                        Path("r1_jasper_d8_t6000.py"),
                        **arguments,
                    )
            authority.assert_not_called()
            implementation.assert_not_called()
            self.assertEqual(injected.read_bytes(), b"ignored cache")

    def test_exclusive_output_reservations_refuse_existing_and_injected_paths(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.parquet"
            results = root / "results.json"
            predictions.write_bytes(b"existing prediction")
            with self.assertRaisesRegex(
                ValueError,
                "Cannot reserve exclusive prediction output",
            ):
                with pipeline_module._ExclusiveOutputReservations(
                    predictions,
                    results,
                ):
                    self.fail("preexisting output was reserved")
            self.assertEqual(predictions.read_bytes(), b"existing prediction")
            self.assertFalse(results.exists())

            predictions.unlink()
            original_open = pipeline_module._ExclusiveOutputReservations._open

            def inject_result(path: Path, label: str):
                if label == "result":
                    path.write_bytes(b"injected result")
                return original_open(path, label)

            with mock.patch.object(
                pipeline_module._ExclusiveOutputReservations,
                "_open",
                side_effect=inject_result,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "Cannot reserve exclusive result output",
                ):
                    with pipeline_module._ExclusiveOutputReservations(
                        predictions,
                        results,
                    ):
                        self.fail("injected output was reserved")
            self.assertEqual(results.read_bytes(), b"injected result")
            self.assertEqual(predictions.read_bytes(), b"")

    def test_generic_config_cannot_overwrite_governed_ender20_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = (
                root
                / "numerai/agents/experiments"
                / "ender20_aux_target_rank_ensemble_v53"
            )
            predictions = (
                experiment
                / "predictions"
                / "r1_jasper_d8_t6000.parquet"
            )
            results = experiment / "results" / "r1_jasper_d8_t6000.json"
            predictions.parent.mkdir(parents=True)
            results.parent.mkdir(parents=True)
            predictions.write_bytes(b"protected prediction")
            results.write_bytes(b"protected result")
            config_path = root / "renamed-generic.json"
            config_path.write_text(
                json.dumps(
                    {
                        "data": {"target_col": TARGET, "id_col": "id"},
                        "output": {
                            "output_dir": str(experiment),
                            "results_name": "r1_jasper_d8_t6000",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                pipeline_module, "REPO_DIR", root
            ), mock.patch.object(
                pipeline_module,
                "NumerAPI",
                side_effect=AssertionError("DATA_CLIENT_OPENED"),
            ) as data_client:
                with self.assertRaisesRegex(
                    ValueError,
                    "Governed Ender20 outputs require a consumed receipt|"
                    "aliases a governed artifact",
                ):
                    run_training(config_path)
            data_client.assert_not_called()
            self.assertEqual(predictions.read_bytes(), b"protected prediction")
            self.assertEqual(results.read_bytes(), b"protected result")

    def test_generic_outputs_reject_hardlink_aliases_to_historical_artifacts(
        self,
    ) -> None:
        cases = (
            (
                "prediction",
                "xerxes",
                Path(
                    "numerai/agents/experiments/"
                    "xerxes20_lgbm_challenger_v53/predictions/"
                    "r1_depth8.parquet"
                ),
            ),
            (
                "result",
                "xerxes",
                Path(
                    "numerai/agents/experiments/"
                    "xerxes20_lgbm_challenger_v53/results/r1_depth8.json"
                ),
            ),
            (
                "prediction",
                "tabm",
                Path(
                    "numerai/agents/experiments/"
                    "ender20_nn_architecture_v53/predictions/"
                    "r5_tabm_k64_train500k.parquet"
                ),
            ),
            (
                "result",
                "tabm",
                Path(
                    "numerai/agents/experiments/"
                    "ender20_nn_architecture_v53/results/"
                    "r5_tabm_k64_train500k.json"
                ),
            ),
        )
        for output_kind, family, governed_relative in cases:
            with self.subTest(
                output_kind=output_kind,
                family=family,
            ), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                governed = root / governed_relative
                governed.parent.mkdir(parents=True)
                original_bytes = (
                    f"protected-{family}-{output_kind}".encode("utf-8")
                )
                governed.write_bytes(original_bytes)

                output_dir = root / "unreserved-output"
                results_name = f"generic-{family}-{output_kind}"
                prediction_path = (
                    output_dir
                    / "predictions"
                    / f"{results_name}.parquet"
                )
                result_path = (
                    output_dir / "results" / f"{results_name}.json"
                )
                alias = (
                    prediction_path
                    if output_kind == "prediction"
                    else result_path
                )
                alias.parent.mkdir(parents=True)
                os.link(governed, alias)
                self.assertEqual(governed.stat().st_nlink, 2)
                config_path = root / "generic-alias.json"
                config_path.write_text(
                    json.dumps(
                        {
                            "data": {"target_col": TARGET, "id_col": "id"},
                            "output": {
                                "output_dir": str(output_dir),
                                "results_name": results_name,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

                with mock.patch.object(
                    pipeline_module, "REPO_DIR", root
                ), mock.patch.object(
                    pipeline_module,
                    "NumerAPI",
                    side_effect=AssertionError("DATA_CLIENT_OPENED"),
                ) as data_client, mock.patch.object(
                    pipeline_module,
                    "load_and_prepare_data",
                    side_effect=AssertionError("DATA_LOADED"),
                ) as data_load, mock.patch.object(
                    pipeline_module,
                    "build_oof_predictions",
                    side_effect=AssertionError("MODEL_RAN"),
                ) as model, mock.patch.object(
                    pipeline_module,
                    "save_predictions",
                    side_effect=AssertionError("PREDICTION_WRITTEN"),
                ) as prediction_write, mock.patch.object(
                    pipeline_module,
                    "save_results",
                    side_effect=AssertionError("RESULT_WRITTEN"),
                ) as result_write:
                    with self.assertRaisesRegex(
                        ValueError,
                        "hardlink|aliases a governed artifact",
                    ):
                        run_training(config_path)

                data_client.assert_not_called()
                data_load.assert_not_called()
                model.assert_not_called()
                prediction_write.assert_not_called()
                result_write.assert_not_called()
                self.assertEqual(governed.read_bytes(), original_bytes)
                self.assertEqual(alias.read_bytes(), original_bytes)
                other = (
                    result_path
                    if output_kind == "prediction"
                    else prediction_path
                )
                self.assertFalse(other.exists())

    def test_generic_output_rejects_parent_reparse_before_data_or_writes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_parent = root / "synthetic-reparse-parent"
            output_dir = output_parent / "output"
            output_dir.mkdir(parents=True)
            config_path = root / "generic-reparse.json"
            config_path.write_text(
                json.dumps(
                    {
                        "data": {"target_col": TARGET, "id_col": "id"},
                        "output": {
                            "output_dir": str(output_dir),
                            "results_name": "reparse-guard",
                        },
                    }
                ),
                encoding="utf-8",
            )
            real_is_symlink = Path.is_symlink

            def synthetic_reparse(path: Path) -> bool:
                candidate = Path(os.path.abspath(path))
                if candidate == Path(os.path.abspath(output_parent)):
                    return True
                return real_is_symlink(path)

            with mock.patch.object(
                pipeline_module, "REPO_DIR", root
            ), mock.patch.object(
                Path, "is_symlink", synthetic_reparse
            ), mock.patch.object(
                pipeline_module,
                "NumerAPI",
                side_effect=AssertionError("DATA_CLIENT_OPENED"),
            ) as data_client, mock.patch.object(
                pipeline_module,
                "load_and_prepare_data",
                side_effect=AssertionError("DATA_LOADED"),
            ) as data_load, mock.patch.object(
                pipeline_module,
                "build_oof_predictions",
                side_effect=AssertionError("MODEL_RAN"),
            ) as model, mock.patch.object(
                pipeline_module,
                "save_predictions",
                side_effect=AssertionError("PREDICTION_WRITTEN"),
            ) as prediction_write, mock.patch.object(
                pipeline_module,
                "save_results",
                side_effect=AssertionError("RESULT_WRITTEN"),
            ) as result_write:
                with self.assertRaisesRegex(
                    ValueError,
                    "parent may not be a reparse point",
                ):
                    run_training(config_path)

            data_client.assert_not_called()
            data_load.assert_not_called()
            model.assert_not_called()
            prediction_write.assert_not_called()
            result_write.assert_not_called()
            self.assertFalse(
                (output_dir / "predictions/reparse-guard.parquet").exists()
            )
            self.assertFalse(
                (output_dir / "results/reparse-guard.json").exists()
            )

    def test_confirmation_training_authority_is_derived_from_finalized_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "confirmation_jasper_d8_t6000.py"
            result_path = root / "results.json"
            predictions_path = root / "predictions.parquet"
            pre_run_path = root / "receipts" / "pre-run.json"
            pretraining_path = root / "receipts" / "pretraining.json"
            checkpoint = "a" * 40
            inventory_blob = "b" * 40
            protocol = SimpleNamespace(repo_root=root)
            component = SimpleNamespace(
                config=config_path,
                result=result_path,
                predictions=predictions_path,
            )
            pretraining = {
                "checkpoint": checkpoint,
                "store_inventory": {"git_blob_id": inventory_blob},
            }
            pre_run = {
                "passed": True,
                "state": "ABSENCE_PROVEN",
                "component": "jasper",
                "confirmation_pretraining_receipt": {
                    "path": pretraining_path.relative_to(root).as_posix(),
                    "sha256": "d" * 64,
                },
            }

            from agents.code.analysis import (
                evaluate_ender20_aux_target_rank_ensemble as evaluator,
            )

            with mock.patch.object(
                pipeline_module, "REPO_DIR", root
            ), mock.patch.object(
                evaluator, "_require_lexical_directory_chain"
            ), mock.patch.object(
                evaluator,
                "_load_bound_receipt",
                side_effect=(pretraining, pre_run),
            ) as bound, mock.patch.object(
                evaluator, "_validate_stage_receipt_schema"
            ), mock.patch.object(
                evaluator, "verify_frozen_protocol", return_value=protocol
            ), mock.patch.object(
                evaluator,
                "default_confirmation_component_paths",
                return_value=component,
            ), mock.patch.object(
                evaluator,
                "_validate_confirmation_pretraining_receipt",
                return_value=pretraining,
            ) as validate_pretraining, mock.patch.object(
                evaluator, "_validate_confirmation_pre_run_absence_receipt"
            ) as validate_pre_run, mock.patch.object(
                evaluator, "_prepare_output_destination_parent"
            ), mock.patch.object(
                evaluator, "_require_absent_destination"
            ):
                authority = pipeline_module._derive_confirmation_training_authority(
                    config_path,
                    component_name="jasper",
                    pre_run_receipt_path=pre_run_path,
                    pre_run_receipt_sha256="c" * 64,
                    pretraining_receipt_path=pretraining_path,
                    pretraining_receipt_sha256="d" * 64,
                )

            self.assertIsInstance(authority, pipeline_module._TrainingAuthority)
            self.assertEqual(authority.mode, "confirmation")
            self.assertEqual(authority.component_name, "jasper")
            self.assertEqual(authority.checkpoint, checkpoint)
            self.assertEqual(authority.inventory_blob, inventory_blob)
            self.assertIs(authority.protocol, protocol)
            self.assertIs(authority.component, component)
            self.assertEqual(authority.pre_run_receipt_path, pre_run_path)
            self.assertEqual(authority.pre_run_receipt_sha256, "c" * 64)
            self.assertEqual(
                authority.confirmation_pretraining_receipt_path,
                pretraining_path,
            )
            self.assertEqual(
                authority.confirmation_pretraining_receipt_sha256,
                "d" * 64,
            )
            self.assertEqual(bound.call_count, 2)
            validate_pretraining.assert_called_once()
            validate_pre_run.assert_called_once()

    def test_pipeline_rejects_inventory_drift_before_oof_or_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, _ = _build_fixture(root)
            receipt = _store_receipt(store, root)
            metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
            with store.feature_path.open("r+b") as stream:
                original = stream.read(1)
                stream.seek(0)
                stream.write(bytes([original[0] ^ 1]))
            metadata["features"]["sha256"] = hashlib.sha256(
                store.feature_path.read_bytes()
            ).hexdigest()
            store.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            output = root / "output"
            config = {
                "data": {
                    "data_version": "v5.3",
                    "feature_set": "all",
                    "target_col": TARGET,
                    "era_col": "era",
                    "id_col": "id",
                    "benchmark_model": BENCHMARK,
                    "disk_feature_store_path": str(store.directory),
                    "disk_feature_store_inventory_path": "frozen-inventory.json",
                },
                "model": {
                    "type": "TorchTabularRegressor",
                    "x_groups": ["features", "era", "benchmark_models"],
                    "params": _torch_params(batch_size=16),
                },
                "training": {
                    "data_mode": "disk_feature_store",
                    "cv": {"enabled": True, "n_splits": 2, "embargo": 0},
                },
                "preprocessing": {"nan_missing_all_twos": False},
                "output": {"output_dir": str(output), "results_name": "drift"},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            identity = {
                "path": "numerai/agents/experiments/test/inventory.json",
                "git_blob_id": "b" * 40,
                "checkpoint_commit": "c" * 40,
            }
            predictions_path = output / "predictions" / "drift.parquet"
            result_path = output / "results" / "drift.json"
            authority = _synthetic_training_authority(
                root,
                predictions_path=predictions_path,
                result_path=result_path,
            )
            marker_path = root / "consumption.json"
            marker_path.write_bytes(
                json.dumps({}, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            )
            completion_claim_path = root / "completion.claimed.json"
            completion_claim_path.write_bytes(
                evaluator_module._receipt_bytes(
                    evaluator_module._claim_payload(
                        "confirmation-train-jasper-completion"
                    )
                )
            )
            with mock.patch(
                "agents.code.modeling.utils.pipeline.load_features",
                side_effect=AssertionError("DATA_ACCESSED"),
            ) as load_features, mock.patch(
                "agents.code.modeling.utils.pipeline._load_committed_feature_store_identity"
            ) as load_inventory:
                with self.assertRaisesRegex(
                    ValueError,
                    "requires its frozen training commit and Git blob before any data access",
                ):
                    run_training(config_path)
            load_features.assert_not_called()
            load_inventory.assert_not_called()

            with mock.patch(
                "agents.code.modeling.utils.pipeline._derive_confirmation_training_authority",
                return_value=authority,
            ), mock.patch(
                "agents.code.modeling.utils.pipeline._require_frozen_python_runtime"
            ), mock.patch(
                "agents.code.modeling.utils.pipeline."
                "_preflight_confirmation_training_authority",
                return_value=(authority.checkpoint, ()),
            ), mock.patch(
                "agents.code.modeling.utils.pipeline._verify_frozen_training_source",
                return_value=(),
            ), mock.patch(
                "agents.code.analysis.evaluate_ender20_aux_target_rank_ensemble."
                "claim_component_training_consumption",
                return_value=(marker_path, {}),
            ), mock.patch.object(
                evaluator_module,
                "claim_component_training_completion",
                return_value=completion_claim_path,
            ), mock.patch(
                "agents.code.modeling.utils.pipeline._ExclusiveOutputReservations",
                return_value=_synthetic_reservation_scope(
                    predictions_path,
                    result_path,
                ),
            ), mock.patch(
                "agents.code.modeling.utils.pipeline.REPO_DIR",
                root,
            ), mock.patch(
                "agents.code.modeling.utils.pipeline._load_committed_feature_store_identity",
                side_effect=ValueError("FROZEN_INVENTORY_REJECTED"),
            ), mock.patch(
                "agents.code.modeling.utils.pipeline.load_features",
                side_effect=AssertionError("FEATURE_METADATA_ACCESSED"),
            ) as frozen_load_features:
                with self.assertRaisesRegex(
                    ValueError,
                    "FROZEN_INVENTORY_REJECTED",
                ):
                    run_training(
                        config_path,
                        confirmation_component="jasper",
                        confirmation_pre_run_receipt=Path("pre-run.json"),
                        confirmation_pre_run_receipt_sha256="d" * 64,
                        confirmation_pretraining_receipt=Path("pretraining.json"),
                        confirmation_pretraining_receipt_sha256="e" * 64,
                    )
            frozen_load_features.assert_not_called()

            with mock.patch(
                "agents.code.modeling.utils.pipeline.load_features",
                return_value=FEATURES,
            ), mock.patch(
                "agents.code.modeling.utils.pipeline._derive_confirmation_training_authority",
                return_value=authority,
            ), mock.patch(
                "agents.code.modeling.utils.pipeline._require_frozen_python_runtime"
            ), mock.patch(
                "agents.code.modeling.utils.pipeline."
                "_preflight_confirmation_training_authority",
                return_value=(authority.checkpoint, ()),
            ), mock.patch(
                "agents.code.modeling.utils.pipeline._verify_frozen_training_source",
                return_value=(),
            ), mock.patch(
                "agents.code.analysis.evaluate_ender20_aux_target_rank_ensemble."
                "claim_component_training_consumption",
                return_value=(marker_path, {}),
            ), mock.patch.object(
                evaluator_module,
                "claim_component_training_completion",
                return_value=completion_claim_path,
            ), mock.patch(
                "agents.code.modeling.utils.pipeline._ExclusiveOutputReservations",
                return_value=_synthetic_reservation_scope(
                    predictions_path,
                    result_path,
                ),
            ), mock.patch(
                "agents.code.modeling.utils.pipeline.REPO_DIR",
                root,
            ), mock.patch(
                "agents.code.modeling.utils.pipeline._load_committed_feature_store_identity",
                return_value=(receipt, identity),
            ), mock.patch(
                "agents.code.modeling.utils.pipeline.build_oof_predictions",
                side_effect=AssertionError("OOF_ACCESSED"),
            ) as oof:
                with self.assertRaisesRegex(
                    ValueError,
                    "metadata (size|SHA-256) differs from inventory",
                ):
                    run_training(
                        config_path,
                        confirmation_component="jasper",
                        confirmation_pre_run_receipt=Path("pre-run.json"),
                        confirmation_pre_run_receipt_sha256="d" * 64,
                        confirmation_pretraining_receipt=Path("pretraining.json"),
                        confirmation_pretraining_receipt_sha256="e" * 64,
                    )
            oof.assert_not_called()
            self.assertFalse((output / "predictions" / "drift.parquet").exists())
            self.assertFalse((output / "results" / "drift.json").exists())

    def test_pipeline_dispatches_explicit_disk_mode_and_scores_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, _ = _build_fixture(root)
            output = root / "output"
            config = {
                "data": {
                    "data_version": "v5.3",
                    "feature_set": "all",
                    "target_col": TARGET,
                    "era_col": "era",
                    "id_col": "id",
                    "benchmark_model": BENCHMARK,
                    "disk_feature_store_path": str(store.directory),
                },
                "model": {
                    "type": "TorchTabularRegressor",
                    "x_groups": ["features", "era", "benchmark_models"],
                    "params": _torch_params(batch_size=16),
                },
                "training": {
                    "data_mode": "disk_feature_store",
                    "cv": {"enabled": True, "n_splits": 2, "embargo": 0},
                },
                "preprocessing": {"nan_missing_all_twos": False},
                "output": {"output_dir": str(output), "results_name": "disk"},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            captured = {}

            def fake_oof(eras, data_loader, *args, **kwargs):
                captured["loader"] = data_loader
                batch = data_loader.load(["0007", "0008"])
                predictions = pd.DataFrame(
                    {
                        "id": batch.id.to_numpy(),
                        "era": batch.era.to_numpy(),
                        TARGET: batch.y.to_numpy(),
                        "prediction": np.linspace(0.1, 0.9, len(batch.y)),
                        "cv_fold": 1,
                    }
                )
                return predictions, {
                    "n_splits": 2,
                    "embargo": 0,
                    "mode": "expanding",
                    "min_train_size": 0,
                    "folds_used": 1,
                    "folds": [],
                }

            summary = pd.DataFrame(
                {"mean": [0.01], "avg_corr_with_benchmark": [0.1]},
                index=["prediction"],
            )
            with mock.patch(
                "agents.code.modeling.utils.pipeline.load_features",
                return_value=FEATURES,
            ), mock.patch(
                "agents.code.modeling.utils.pipeline.build_oof_predictions",
                side_effect=fake_oof,
            ), mock.patch(
                "agents.code.modeling.utils.pipeline.summarize_predictions",
                return_value={
                    "corr": summary,
                    "bmc": summary,
                    "bmc_last_200_eras": summary,
                },
            ) as summarize:
                _, results_path = run_training(config_path)

            self.assertIsInstance(captured["loader"], DiskFeatureStoreLoader)
            self.assertEqual(
                Path(summarize.call_args.args[4]), store.manifest_path
            )
            scoring_manifest = summarize.call_args.kwargs["benchmark_data"]
            self.assertIsInstance(scoring_manifest, pd.DataFrame)
            self.assertEqual(scoring_manifest["id"].tolist(), captured["loader"].manifest["id"].tolist())
            results = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(results["data"]["data_mode"], "disk_feature_store")
            self.assertEqual(
                results["data"]["disk_feature_store"]["generation_id"],
                store.generation_id,
            )

    def test_manifest_is_a_valid_benchmark_scoring_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, eager = _build_fixture(root)
            loader = _loader(store)
            benchmark_manifest = loader.manifest
            loader.close()
            predictions_path = root / "predictions.parquet"
            pd.DataFrame(
                {
                    "id": eager["id"],
                    "era": eager["era"],
                    TARGET: eager[TARGET],
                    "prediction": eager["feature_a"].astype(np.float64),
                }
            ).to_parquet(predictions_path, index=False)
            summaries = numerai_metrics.summarize_prediction_file_with_bmc(
                predictions_path,
                ["prediction"],
                TARGET,
                "v5.3",
                benchmark_model=BENCHMARK,
                benchmark_data_path=root / "retired-generation.parquet",
                era_col="era",
                id_col="id",
                benchmark_data=benchmark_manifest,
            )
            self.assertEqual(
                set(summaries), {"corr", "bmc", "bmc_last_200_eras"}
            )
            self.assertIn("prediction", summaries["bmc"].index)

    def test_pipeline_default_mode_remains_eager(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, eager = _build_fixture(root)
            output = root / "eager-output"
            config = {
                "data": {
                    "data_version": "v5.3",
                    "feature_set": "all",
                    "target_col": TARGET,
                    "era_col": "era",
                    "id_col": "id",
                    "benchmark_model": BENCHMARK,
                },
                "model": {
                    "type": "LGBMRegressor",
                    "x_groups": ["features", "era", "benchmark_models"],
                    "params": {},
                },
                "training": {
                    "cv": {"enabled": True, "n_splits": 2, "embargo": 0}
                },
                "output": {"output_dir": str(output), "results_name": "eager"},
            }
            config_path = root / "eager-config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            full_without_benchmark = eager.drop(columns=[BENCHMARK])
            captured = {}

            def fake_oof(eras, data_loader, *args, **kwargs):
                captured["loader"] = data_loader
                batch = data_loader.load(["0007", "0008"])
                return pd.DataFrame(
                    {
                        "id": batch.id.to_numpy(),
                        "era": batch.era.to_numpy(),
                        TARGET: batch.y.to_numpy(),
                        "prediction": np.linspace(0.1, 0.9, len(batch.y)),
                        "cv_fold": 1,
                    }
                ), {
                    "n_splits": 2,
                    "embargo": 0,
                    "mode": "expanding",
                    "min_train_size": 0,
                    "folds_used": 1,
                    "folds": [],
                }

            summary = pd.DataFrame(
                {"mean": [0.01], "avg_corr_with_benchmark": [0.1]},
                index=["prediction"],
            )
            with mock.patch(
                "agents.code.modeling.utils.pipeline.load_and_prepare_data",
                return_value=(full_without_benchmark, FEATURES),
            ), mock.patch(
                "agents.code.modeling.utils.pipeline.attach_benchmark_models",
                return_value=(eager, [BENCHMARK]),
            ), mock.patch(
                "agents.code.modeling.utils.pipeline.build_oof_predictions",
                side_effect=fake_oof,
            ), mock.patch(
                "agents.code.modeling.utils.pipeline.summarize_predictions",
                return_value={
                    "corr": summary,
                    "bmc": summary,
                    "bmc_last_200_eras": summary,
                },
            ):
                _, results_path = run_training(config_path)

            self.assertIsInstance(captured["loader"], ModelDataLoader)
            results = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(results["data"]["data_mode"], "eager")


if __name__ == "__main__":
    unittest.main()
