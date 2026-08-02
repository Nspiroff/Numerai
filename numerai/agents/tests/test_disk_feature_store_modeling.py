from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from agents.code.data.build_full_datasets import build_disk_feature_store
from agents.code.metrics import numerai_metrics
from agents.code.modeling.utils.disk_feature_store import (
    DiskFeatureStoreLoader,
    DiskFeatureView,
)
from agents.code.modeling.utils.model_factory import build_model
from agents.code.modeling.utils.model_data import ModelDataLoader
from agents.code.modeling.utils.numerai_cv import build_oof_predictions
from agents.code.modeling.utils.pipeline import run_training


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
