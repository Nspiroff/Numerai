from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from agents.code.data.build_full_datasets import build_disk_feature_store
from agents.code.modeling.models.lgbm_regressor import LGBMRegressor
from agents.code.modeling.utils.disk_feature_store import DiskFeatureStoreLoader
from agents.code.modeling.utils.model_data import ModelDataLoader
from agents.code.modeling.utils.model_factory import build_model
from agents.code.modeling.utils.numerai_cv import build_oof_predictions
from agents.code.modeling.utils.pipeline import run_training


TARGET = "target_xerxes_20"
BENCHMARK = "v53_lgbm_ender20"
FEATURES = ["feature_a", "feature_b", "feature_c"]


class _FakeLightGBMError(Exception):
    pass


class _RecordingEstimator:
    def __init__(self, params: dict, *, fail_fit: bool = False) -> None:
        self.params = dict(params)
        self.fail_fit = fail_fit
        self.fit_inputs: list[object] = []
        self.fit_targets: list[np.ndarray] = []
        self.predict_inputs: list[np.ndarray] = []

    def fit(self, X, y, **kwargs):
        del kwargs
        self.fit_inputs.append(X)
        self.fit_targets.append(np.asarray(y).copy())
        if self.fail_fit:
            raise _FakeLightGBMError("GPU Tree Learner was not enabled")
        return self

    def predict(self, X):
        values = np.asarray(X)
        self.predict_inputs.append(values)
        predictions = values[:, 0].astype(np.float64)
        if values.shape[1] > 1:
            predictions += values[:, 1].astype(np.float64) / 100.0
        return predictions


def _fake_lightgbm(
    *, fail_first_gpu_fit: bool = False, fail_all_gpu_fits: bool = False
):
    module = types.ModuleType("lightgbm")
    module.basic = types.SimpleNamespace(LightGBMError=_FakeLightGBMError)
    module.created = []

    def build_estimator(**params):
        fail_fit = bool(
            (fail_all_gpu_fits or (fail_first_gpu_fit and not module.created))
            and params.get("device_type") == "gpu"
        )
        estimator = _RecordingEstimator(params, fail_fit=fail_fit)
        module.created.append(estimator)
        return estimator

    module.LGBMRegressor = build_estimator
    return module


def _build_fixture(root: Path, *, eras: int = 8, rows_per_era: int = 3):
    row_count = eras * rows_per_era
    era_values = np.repeat(
        [f"{era:04d}" for era in range(1, eras + 1)], rows_per_era
    )
    row_positions = np.arange(row_count)
    data = pd.DataFrame(
        {
            "id": [f"row-{row:06d}" for row in row_positions],
            "era": era_values,
            "feature_a": (row_positions % 5).astype(np.int8),
            "feature_b": ((row_positions * 2 + 1) % 5).astype(np.int8),
            "feature_c": ((row_positions * 3 + 2) % 5).astype(np.int8),
            TARGET: (0.2 + row_positions * 0.001).astype(np.float32),
        }
    )
    benchmark = pd.DataFrame(
        {
            "id": data["id"],
            "era": data["era"],
            BENCHMARK: np.linspace(0.05, 0.95, row_count),
        }
    )
    split = (eras // 2) * rows_per_era
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
    benchmark.iloc[:split].to_parquet(train_benchmark_path, index=False)
    benchmark.iloc[split:].to_parquet(validation_benchmark_path, index=False)
    store = build_disk_feature_store(
        root / "store",
        [train_path, validation_path],
        [train_benchmark_path, validation_benchmark_path],
        FEATURES,
        target_column=TARGET,
        benchmark_column=BENCHMARK,
        batch_size=5,
        reuse_existing=False,
    )
    eager = data.copy()
    eager[BENCHMARK] = benchmark[BENCHMARK]
    return store, eager


def _open_loader(store) -> DiskFeatureStoreLoader:
    loader = DiskFeatureStoreLoader(
        store.directory,
        era_col="era",
        target_col=TARGET,
        id_col="id",
        benchmark_col=BENCHMARK,
    )
    loader.configure_x_cols([*FEATURES, "era", BENCHMARK])
    return loader


class _SyntheticDiskFeatureView:
    is_disk_feature_view = True

    def __init__(
        self,
        row_count: int,
        feature_columns=FEATURES,
        *,
        state: dict | None = None,
    ) -> None:
        self._row_count = row_count
        self.feature_columns = tuple(feature_columns)
        self.columns = pd.Index(self.feature_columns)
        self.state = state if state is not None else {"batch_sizes": []}

    def __len__(self) -> int:
        return self._row_count

    def __getitem__(self, columns):
        requested = list(columns)
        missing = [name for name in requested if name not in self.feature_columns]
        if missing:
            raise KeyError(missing[0])
        return _SyntheticDiskFeatureView(
            self._row_count,
            requested,
            state=self.state,
        )

    def iter_feature_batches(self, batch_size, *, shuffle_blocks=False):
        if shuffle_blocks:
            raise AssertionError("LightGBM prediction must use ordered disk batches.")
        self.state["batch_sizes"].append(batch_size)
        ranges = [
            (start, min(start + batch_size, self._row_count))
            for start in range(0, self._row_count, batch_size)
        ]
        for start, stop in reversed(ranges):
            positions = np.arange(start, stop, dtype=np.int64)
            columns = []
            for name in self.feature_columns:
                offset = FEATURES.index(name)
                columns.append(((positions + offset * 7) % 97).astype(np.int8))
            yield np.column_stack(columns).astype(np.int8, copy=False), positions


class TestLGBMDiskFeatureStore(unittest.TestCase):
    def test_fit_materializes_selected_features_in_local_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, eager = _build_fixture(Path(tmp))
            loader = _open_loader(store)
            try:
                batch = loader.load(loader.eras.unique())
                indices = np.array([7, 1, 19, 0, 13], dtype=np.int64)
                selected_features = ["feature_c", "feature_a"]
                fake_lgb = _fake_lightgbm()
                with mock.patch.dict(sys.modules, {"lightgbm": fake_lgb}):
                    model = LGBMRegressor(
                        feature_cols=selected_features,
                        disk_materialization_max_rows=len(indices),
                        device_type="cpu",
                    )

                model.fit(batch.X.take(indices), batch.y.iloc[indices])

                fit_values = fake_lgb.created[0].fit_inputs[0]
                self.assertIsInstance(fit_values, np.ndarray)
                self.assertEqual(fit_values.dtype, np.int8)
                self.assertTrue(fit_values.flags.c_contiguous)
                np.testing.assert_array_equal(
                    fit_values,
                    eager.iloc[indices][selected_features].to_numpy(dtype=np.int8),
                )
                self.assertEqual(model.data_mode_, "disk_feature_store")
                self.assertEqual(model.disk_train_rows_, len(indices))
                self.assertEqual(model.disk_batches_per_epoch_, [1])
                self.assertEqual(model.disk_rows_per_epoch_, [len(indices)])
            finally:
                loader.close()

    def test_fit_rejects_missing_or_exceeded_materialization_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = _build_fixture(Path(tmp))
            loader = _open_loader(store)
            try:
                batch = loader.load(loader.eras.unique())
                view = batch.X.take(np.arange(5))
                target = batch.y.iloc[:5]
                for cap, message in (
                    (None, "positive training.max_train_samples"),
                    (4, "exceeding the materialization cap"),
                ):
                    with self.subTest(cap=cap):
                        fake_lgb = _fake_lightgbm()
                        with mock.patch.dict(
                            sys.modules, {"lightgbm": fake_lgb}
                        ):
                            model = LGBMRegressor(
                                feature_cols=FEATURES,
                                disk_materialization_max_rows=cap,
                            )
                        with self.assertRaisesRegex(ValueError, message):
                            model.fit(view, target)
                        self.assertEqual(fake_lgb.created[0].fit_inputs, [])
            finally:
                loader.close()

    def test_gpu_fallback_reuses_materialized_matrix_and_hides_wrapper_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = _build_fixture(Path(tmp))
            loader = _open_loader(store)
            try:
                batch = loader.load(loader.eras.unique())
                indices = np.array([8, 2, 11, 4], dtype=np.int64)
                fake_lgb = _fake_lightgbm(fail_first_gpu_fit=True)
                with mock.patch.dict(sys.modules, {"lightgbm": fake_lgb}):
                    model = LGBMRegressor(
                        feature_cols=FEATURES,
                        disk_materialization_max_rows=4,
                        device_type="gpu",
                        n_estimators=2,
                    )

                model.fit(batch.X.take(indices), batch.y.iloc[indices])

                self.assertEqual(len(fake_lgb.created), 2)
                gpu_model, cpu_model = fake_lgb.created
                self.assertEqual(gpu_model.params["device_type"], "gpu")
                self.assertEqual(cpu_model.params["device_type"], "cpu")
                self.assertNotIn("disk_materialization_max_rows", gpu_model.params)
                self.assertNotIn("disk_materialization_max_rows", cpu_model.params)
                self.assertIs(gpu_model.fit_inputs[0], cpu_model.fit_inputs[0])
                self.assertEqual(model.effective_device_type_, "cpu")
                self.assertTrue(model.gpu_fallback_used_)
            finally:
                loader.close()

    def test_prediction_uses_65536_row_batches_and_restores_positions(self):
        fake_lgb = _fake_lightgbm()
        with mock.patch.dict(sys.modules, {"lightgbm": fake_lgb}):
            model = LGBMRegressor(feature_cols=FEATURES[:2])
        view = _SyntheticDiskFeatureView(65_539)

        predictions = model.predict(view)

        positions = np.arange(len(view), dtype=np.int64)
        expected = (positions % 97) + ((positions + 7) % 97) / 100.0
        np.testing.assert_allclose(predictions, expected)
        self.assertEqual(view.state["batch_sizes"], [65_536])
        self.assertEqual(
            [len(values) for values in fake_lgb.created[0].predict_inputs],
            [3, 65_536],
        )
        self.assertEqual(model.disk_validation_rows_, len(view))
        self.assertEqual(model.disk_prediction_batch_size_, 65_536)
        self.assertEqual(model.disk_prediction_batches_, 2)

    def test_model_factory_configures_wrapper_only_prediction_batch_size(self):
        fake_lgb = _fake_lightgbm()
        with mock.patch.dict(sys.modules, {"lightgbm": fake_lgb}):
            model = build_model(
                "LGBMRegressor",
                {"device_type": "cpu", "n_estimators": 2},
                {"prediction_batch_size": 3},
                feature_cols=FEATURES[:2],
            )
        view = _SyntheticDiskFeatureView(8)

        predictions = model.predict(view)

        self.assertEqual(len(predictions), 8)
        self.assertEqual(view.state["batch_sizes"], [3])
        self.assertEqual(
            [len(values) for values in fake_lgb.created[0].predict_inputs],
            [2, 3, 3],
        )
        self.assertEqual(model.prediction_batch_size, 3)
        self.assertEqual(model.disk_prediction_batch_size_, 3)
        self.assertEqual(model.disk_prediction_batches_, 3)
        self.assertNotIn("prediction_batch_size", fake_lgb.created[0].params)

    def test_prediction_batch_size_must_be_a_positive_integer(self):
        for value in (0, -1, True, 1.5, "3"):
            with self.subTest(value=value):
                fake_lgb = _fake_lightgbm()
                with mock.patch.dict(sys.modules, {"lightgbm": fake_lgb}):
                    with self.assertRaisesRegex(
                        ValueError, "prediction_batch_size must be a positive integer"
                    ):
                        build_model(
                            "LGBMRegressor",
                            {},
                            {"prediction_batch_size": value},
                            feature_cols=FEATURES,
                        )

    def test_eager_fit_and_predict_do_not_require_disk_cap(self):
        frame = pd.DataFrame(
            {
                "feature_a": [0, 1, 2],
                "feature_b": [2, 1, 0],
                "feature_c": [1, 1, 1],
                "era": ["0001", "0001", "0002"],
            }
        )
        fake_lgb = _fake_lightgbm()
        with mock.patch.dict(sys.modules, {"lightgbm": fake_lgb}):
            model = LGBMRegressor(feature_cols=FEATURES)

        model.fit(frame, np.array([0.1, 0.2, 0.3]))
        predictions = model.predict(frame)

        self.assertEqual(model.data_mode_, "eager")
        self.assertEqual(model.effective_device_type_, "cpu")
        self.assertFalse(model.gpu_fallback_used_)
        self.assertIsInstance(fake_lgb.created[0].fit_inputs[0], pd.DataFrame)
        np.testing.assert_allclose(predictions, [0.02, 1.01, 2.0])

    def test_eager_oof_reports_successful_gpu_device_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, eager = _build_fixture(Path(tmp))
            loader = ModelDataLoader(
                full=eager,
                era_col="era",
                target_col=TARGET,
                id_col="id",
                x_cols=tuple(FEATURES),
            )
            fake_lgb = _fake_lightgbm()
            with mock.patch.dict(sys.modules, {"lightgbm": fake_lgb}):
                _, cv_meta = build_oof_predictions(
                    eager["era"],
                    loader,
                    "LGBMRegressor",
                    {"device_type": "gpu"},
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

            self.assertEqual(len(cv_meta["folds"]), 3)
            for fold in cv_meta["folds"]:
                diagnostics = fold["model_diagnostics"]
                self.assertEqual(diagnostics["effective_device_type"], "gpu")
                self.assertFalse(diagnostics["gpu_fallback_used"])
                self.assertNotIn("data_mode", diagnostics)

    def test_eager_oof_reports_cpu_fallback_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, eager = _build_fixture(Path(tmp))
            loader = ModelDataLoader(
                full=eager,
                era_col="era",
                target_col=TARGET,
                id_col="id",
                x_cols=tuple(FEATURES),
            )
            fake_lgb = _fake_lightgbm(fail_all_gpu_fits=True)
            with mock.patch.dict(sys.modules, {"lightgbm": fake_lgb}):
                _, cv_meta = build_oof_predictions(
                    eager["era"],
                    loader,
                    "LGBMRegressor",
                    {"device_type": "gpu"},
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

            self.assertEqual(len(cv_meta["folds"]), 3)
            for fold in cv_meta["folds"]:
                diagnostics = fold["model_diagnostics"]
                self.assertEqual(diagnostics["effective_device_type"], "cpu")
                self.assertTrue(diagnostics["gpu_fallback_used"])

    def test_oof_forwards_cap_and_reports_disk_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = _build_fixture(Path(tmp))
            loader = _open_loader(store)
            fake_lgb = _fake_lightgbm()
            try:
                with mock.patch.dict(sys.modules, {"lightgbm": fake_lgb}):
                    oof, cv_meta = build_oof_predictions(
                        loader.eras,
                    loader,
                    "LGBMRegressor",
                    {"device_type": "gpu"},
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

                self.assertTrue(np.isfinite(oof["prediction"]).all())
                self.assertEqual(len(cv_meta["folds"]), 3)
                for fold in cv_meta["folds"]:
                    self.assertEqual(fold["train_rows"], 5)
                    diagnostics = fold["model_diagnostics"]
                    self.assertEqual(diagnostics["data_mode"], "disk_feature_store")
                    self.assertEqual(diagnostics["disk_train_rows"], 5)
                    self.assertEqual(diagnostics["disk_batches_per_epoch"], [1])
                    self.assertEqual(diagnostics["disk_rows_per_epoch"], [5])
                    self.assertEqual(
                        diagnostics["disk_prediction_batch_size"], 65_536
                    )
                    self.assertEqual(diagnostics["disk_prediction_batches"], 1)
                    self.assertEqual(diagnostics["effective_device_type"], "gpu")
                    self.assertFalse(diagnostics["gpu_fallback_used"])
            finally:
                loader.close()

    def test_pipeline_fails_closed_without_explicit_positive_integer_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cap in (None, 0, -1, True, "5"):
                with self.subTest(cap=cap):
                    config = {
                        "data": {
                            "data_version": "v5.3",
                            "feature_set": "medium",
                            "target_col": TARGET,
                            "era_col": "era",
                            "id_col": "id",
                            "benchmark_model": BENCHMARK,
                            "disk_feature_store_path": str(root / "missing-store"),
                        },
                        "model": {
                            "type": "LGBMRegressor",
                            "x_groups": ["features"],
                            "params": {},
                        },
                        "training": {
                            "data_mode": "disk_feature_store",
                            "cv": {"enabled": True, "n_splits": 2, "embargo": 0},
                        },
                        "preprocessing": {"nan_missing_all_twos": False},
                        "output": {
                            "output_dir": str(root / "output"),
                            "results_name": "invalid-cap",
                        },
                    }
                    if cap is not None:
                        config["training"]["max_train_samples"] = cap
                    config_path = root / f"invalid-cap-{cap!s}.json"
                    config_path.write_text(json.dumps(config), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError, "explicit positive integer materialization cap"
                    ):
                        run_training(config_path)

    def test_pipeline_accepts_capped_disk_lgbm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, _ = _build_fixture(root)
            output = root / "output"
            config = {
                "data": {
                    "data_version": "v5.3",
                    "feature_set": "medium",
                    "target_col": TARGET,
                    "era_col": "era",
                    "id_col": "id",
                    "benchmark_model": BENCHMARK,
                    "disk_feature_store_path": str(store.directory),
                },
                "model": {
                    "type": "LGBMRegressor",
                    "x_groups": ["features"],
                    "params": {"device_type": "cpu"},
                },
                "training": {
                    "data_mode": "disk_feature_store",
                    "max_train_samples": 5,
                    "cv": {"enabled": True, "n_splits": 2, "embargo": 0},
                },
                "preprocessing": {"nan_missing_all_twos": False},
                "output": {
                    "output_dir": str(output),
                    "results_name": "capped-disk-lgbm",
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            captured = {}

            def fake_oof(
                eras,
                data_loader,
                model_type,
                model_params,
                model_config,
                cv_config,
                max_train_samples,
                *args,
                **kwargs,
            ):
                del eras, model_params, model_config, cv_config, args, kwargs
                captured["model_type"] = model_type
                captured["max_train_samples"] = max_train_samples
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
            ):
                run_training(config_path)

            self.assertEqual(captured["model_type"], "LGBMRegressor")
            self.assertEqual(captured["max_train_samples"], 5)


if __name__ == "__main__":
    unittest.main()
