from __future__ import annotations

import hashlib
import json
import runpy
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from agents.code.modeling.models.torch_tabular_regressor import (
    TorchTabularRegressor,
)


class TestEnder24EMASeedStability(unittest.TestCase):
    def _model(self, *, seed: int = 1337, ema_decay=None, **overrides):
        params = {
            "feature_cols": ["feature_a", "feature_b"],
            "architecture": "mlp",
            "hidden_layer_sizes": (8,),
            "device": "cpu",
            "amp": False,
            "batch_size": 8,
            "prediction_batch_size": 8,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "max_epochs": 3,
            "patience": 3,
            "val_fraction": 0.25,
            "val_split": "recent_eras",
            "internal_val_embargo": 1,
            "feature_center": None,
            "feature_scale": None,
            "max_grad_norm": None,
            "seed": seed,
            "deterministic": True,
            "verbose": False,
            "ema_decay": ema_decay,
        }
        params.update(overrides)
        return TorchTabularRegressor(**params)

    def _frame(self):
        rng = np.random.default_rng(24)
        rows = 48
        X = pd.DataFrame(
            {
                "feature_a": rng.normal(size=rows).astype(np.float32),
                "feature_b": rng.normal(size=rows).astype(np.float32),
                "era": np.repeat([f"{era:04d}" for era in range(12)], 4),
            }
        )
        y = pd.Series(
            0.4 * X["feature_a"] - 0.2 * X["feature_b"]
            + rng.normal(0.0, 0.05, size=rows)
        )
        return X, y.astype(np.float32)

    def test_ema_decay_validation_and_disabled_default(self):
        self.assertIsNone(self._model().ema_decay)
        for value in (True, False, 0.0, 1.0, -0.1, 1.1, np.nan, np.inf, "x"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "ema_decay"
            ):
                self._model(ema_decay=value)

    def test_exact_recurrence_and_first_step_initialization(self):
        model = self._model(ema_decay=0.5)
        model._model = model._torch.nn.Linear(1, 1, bias=False)
        with model._torch.no_grad():
            model._model.weight.fill_(2.0)
        model._update_ema_after_step()
        self.assertEqual(model.ema_updates_, 1)
        self.assertEqual(float(model._ema_state["weight"]), 2.0)

        with model._torch.no_grad():
            model._model.weight.fill_(6.0)
        model._update_ema_after_step()
        self.assertEqual(model.ema_updates_, 2)
        self.assertEqual(float(model._ema_state["weight"]), 4.0)

    def test_control_optimizer_path_never_queries_scale(self):
        model = self._model()
        events = []

        class Scaler:
            def step(self, optimizer):
                events.append(("step", optimizer))

            def update(self):
                events.append(("update", None))

            def get_scale(self):
                raise AssertionError("control path must not query AMP scale")

        optimizer = object()
        model._step_optimizer(optimizer, Scaler())
        self.assertEqual(events, [("step", optimizer), ("update", None)])
        self.assertEqual(model.ema_updates_, 0)

    def test_skipped_amp_step_does_not_update_shadow(self):
        model = self._model(ema_decay=0.995)
        model._model = model._torch.nn.Linear(1, 1)
        model._update_ema_after_step()
        prior_updates = model.ema_updates_
        prior_shadow_hash = model._state_sha256(model._ema_state)
        model.ema_shadow_state_sha256_ = prior_shadow_hash
        with model._torch.no_grad():
            model._model.weight.add_(3.0)

        class OverflowScaler:
            def __init__(self):
                self.scales = iter((8.0, 4.0))

            def get_scale(self):
                return next(self.scales)

            def step(self, _optimizer):
                return None

            def update(self):
                return None

        model._step_optimizer(object(), OverflowScaler())
        self.assertEqual(model.ema_updates_, prior_updates)
        self.assertEqual(model._state_sha256(model._ema_state), prior_shadow_hash)
        self.assertEqual(model.ema_shadow_state_sha256_, prior_shadow_hash)

        empty_model = self._model(ema_decay=0.995)
        empty_model._model = empty_model._torch.nn.Linear(1, 1)

        class SaturatedOverflowScaler(OverflowScaler):
            def __init__(self):
                self.scales = iter((0.0, 0.0))

        with self.assertRaisesRegex(RuntimeError, "positive AMP scale before"):
            empty_model._step_optimizer(object(), SaturatedOverflowScaler())
        self.assertEqual(empty_model.ema_updates_, 0)
        self.assertIsNone(empty_model._ema_state)

        underflow_model = self._model(ema_decay=0.995)
        underflow_model._model = underflow_model._torch.nn.Linear(1, 1)

        class UnderflowScaler(OverflowScaler):
            def __init__(self):
                self.scales = iter((1e-45, 0.0))

        with self.assertRaisesRegex(RuntimeError, "positive AMP scale after"):
            underflow_model._step_optimizer(object(), UnderflowScaler())
        self.assertEqual(underflow_model.ema_updates_, 0)
        self.assertIsNone(underflow_model._ema_state)

    def test_evaluation_scope_uses_shadow_and_restores_live_on_error(self):
        model = self._model(ema_decay=0.5)
        model._model = model._torch.nn.Linear(1, 1, bias=False)
        with model._torch.no_grad():
            model._model.weight.fill_(2.0)
        model._update_ema_after_step()
        with model._torch.no_grad():
            model._model.weight.fill_(7.0)

        with self.assertRaisesRegex(RuntimeError, "probe"):
            with model._ema_evaluation_scope():
                self.assertEqual(float(model._model.weight.detach()), 2.0)
                raise RuntimeError("probe")
        self.assertEqual(float(model._model.weight.detach()), 7.0)

    def test_fit_records_active_terminal_and_inference_hashes(self):
        X, y = self._frame()
        model = self._model(ema_decay=0.995)
        model.fit(X, y)

        self.assertGreater(model.ema_updates_, 0)
        self.assertRegex(model.ema_live_state_sha256_, r"^[0-9a-f]{64}$")
        self.assertRegex(model.ema_shadow_state_sha256_, r"^[0-9a-f]{64}$")
        self.assertRegex(model.ema_inference_state_sha256_, r"^[0-9a-f]{64}$")
        self.assertNotEqual(
            model.ema_live_state_sha256_, model.ema_shadow_state_sha256_
        )
        self.assertEqual(
            model.ema_inference_state_sha256_,
            model._state_sha256(model.cpu_state_dict()),
        )
        self.assertEqual(
            model.ema_inference_state_sha256_,
            model._state_sha256(model._ema_state),
        )
        self.assertTrue(np.isfinite(model.predict(X.iloc[:5])).all())

    def test_validation_selects_best_shadow_not_terminal_shadow(self):
        X, y = self._frame()
        model = self._model(
            ema_decay=0.5,
            max_epochs=2,
            patience=2,
        )
        shadow_hashes = []

        def train_epoch(_loader, _optimizer, _scaler):
            with model._torch.no_grad():
                for parameter in model._model.parameters():
                    parameter.add_(1.0)
            model._update_ema_after_step()
            return float(model.ema_updates_)

        validation_losses = iter((0.1, 0.2))

        def validation_loss(_loader):
            shadow_hashes.append(model._state_sha256(model._model.state_dict()))
            return next(validation_losses)

        model._train_epoch = train_epoch
        model._validation_loss = validation_loss
        model.fit(X, y)

        self.assertEqual(model.best_epoch_, 1)
        self.assertEqual(len(shadow_hashes), 2)
        self.assertNotEqual(shadow_hashes[0], shadow_hashes[1])
        self.assertEqual(model.ema_shadow_state_sha256_, shadow_hashes[1])
        self.assertEqual(model.ema_inference_state_sha256_, shadow_hashes[0])
        self.assertNotEqual(
            model.ema_inference_state_sha256_, model.ema_shadow_state_sha256_
        )

    def test_full_disk_fit_validation_and_prediction_match_eager_ema(self):
        from agents.tests.test_disk_feature_store_modeling import (
            BENCHMARK,
            FEATURES,
            TARGET,
            _build_fixture,
            _loader,
            _torch_params,
        )

        with tempfile.TemporaryDirectory() as tmp:
            store, eager = _build_fixture(Path(tmp), eras=8, rows_per_era=4)
            loader = _loader(store)
            try:
                disk_batch = loader.load(loader.eras.unique())
                eager_X = eager[[*FEATURES, "era", BENCHMARK]]
                params = _torch_params(batch_size=len(eager))
                params.update(
                    {
                        "ema_decay": 0.995,
                        "max_epochs": 2,
                        "patience": 2,
                        "val_split": "recent_eras",
                        "val_fraction": 0.25,
                        "internal_val_embargo": 1,
                    }
                )
                eager_model = TorchTabularRegressor(
                    feature_cols=FEATURES, **params
                )
                disk_model = TorchTabularRegressor(
                    feature_cols=FEATURES, **params
                )

                eager_model.fit(eager_X, eager[TARGET])
                disk_model.fit(disk_batch.X, disk_batch.y)

                np.testing.assert_allclose(
                    disk_model.predict(disk_batch.X),
                    eager_model.predict(eager_X),
                    rtol=1e-6,
                    atol=1e-6,
                )
                self.assertEqual(disk_model.data_mode_, "disk_feature_store")
                self.assertGreater(disk_model.disk_validation_rows_, 0)
                np.testing.assert_allclose(
                    [
                        (row["train_loss"], row["val_loss"])
                        for row in disk_model.training_history_
                    ],
                    [
                        (row["train_loss"], row["val_loss"])
                        for row in eager_model.training_history_
                    ],
                    rtol=1e-6,
                    atol=1e-7,
                )
                self.assertRegex(
                    disk_model.ema_inference_state_sha256_, r"^[0-9a-f]{64}$"
                )
                self.assertRegex(
                    eager_model.ema_inference_state_sha256_, r"^[0-9a-f]{64}$"
                )
                self.assertGreater(disk_model.ema_updates_, 0)
                self.assertNotEqual(
                    disk_model.ema_live_state_sha256_,
                    disk_model.ema_shadow_state_sha256_,
                )
            finally:
                loader.close()

    def test_disk_epoch_updates_ema_after_completed_optimizer_step(self):
        model = self._model(
            ema_decay=0.995,
            val_split="none",
            val_fraction=0.0,
            max_epochs=1,
        )
        model._model = model._build_model(2).to(model._device)
        optimizer = model._torch.optim.AdamW(
            model._model.parameters(), lr=0.01, weight_decay=0.0
        )
        scaler = model._torch.amp.GradScaler("cpu", enabled=False)
        values = np.array(
            [[0.0, 1.0], [1.0, 0.0], [0.5, -0.5], [-0.5, 0.5]],
            dtype=np.float32,
        )
        targets = np.array([0.25, 0.75, 0.5, -0.25], dtype=np.float32)

        class View:
            def iter_feature_batches(self, *_args, **_kwargs):
                yield values, np.arange(len(values), dtype=np.int64)

        _, rows, batches = model._train_epoch_disk(
            View(), targets, optimizer, scaler, epoch=1
        )
        self.assertEqual((rows, batches), (4, 1))
        self.assertEqual(model.ema_updates_, 1)
        self.assertIsNotNone(model._ema_state)

    def test_model_seeds_are_active_and_matched_initialization_is_exact(self):
        root = Path(__file__).resolve().parents[1]
        config = runpy.run_path(
            str(
                root
                / "experiments"
                / "ender24_ema_seed_stability_v53"
                / "configs"
                / "r1_control_seed1337.py"
            )
        )["CONFIG"]
        feature_path = root.parent.parent / config["data"]["feature_columns_path"]
        feature_columns = json.loads(feature_path.read_text(encoding="utf-8"))
        self.assertEqual(len(feature_columns), 3_555)

        def exact_activity(seed, *, ema_decay=None):
            params = deepcopy(config["model"]["params"])
            params["seed"] = seed
            params["device"] = "cpu"
            params["amp"] = False
            if ema_decay is not None:
                params["ema_decay"] = ema_decay
            model = TorchTabularRegressor(
                feature_cols=feature_columns,
                **params,
            )
            model._seed_everything()
            dataset = model._torch.utils.data.TensorDataset(
                model._torch.arange(64)
            )
            loader = model._make_loader(
                dataset, np.arange(64), batch_size=16, shuffle=True
            )
            built = model._build_model(len(feature_columns))
            initial_hash = model._state_sha256(built.state_dict())
            batch_order = np.asarray(
                tuple(
                    value for (batch,) in loader for value in batch.tolist()
                ),
                dtype="<i8",
            )
            batch_order_hash = hashlib.sha256(batch_order.tobytes()).hexdigest()
            return initial_hash, batch_order_hash

        hashes = {}
        orders = {}
        for seed in (1337, 2027, 7331):
            hashes[seed], orders[seed] = exact_activity(seed)
        self.assertEqual(len(set(hashes.values())), 3)
        self.assertEqual(len(set(orders.values())), 3)

        ema_hash, ema_order = exact_activity(1337, ema_decay=0.995)
        self.assertEqual(hashes[1337], ema_hash)
        self.assertEqual(orders[1337], ema_order)

    def test_oof_diagnostics_bind_ema_activity(self):
        from agents.code.modeling.utils import numerai_cv

        rng = np.random.default_rng(240)
        rows = 96
        frame = pd.DataFrame(
            {
                "feature_a": rng.normal(size=rows).astype(np.float32),
                "feature_b": rng.normal(size=rows).astype(np.float32),
                "era": np.repeat([f"{era:04d}" for era in range(24)], 4),
            }
        )
        frame["target"] = (
            0.4 * frame["feature_a"]
            - 0.2 * frame["feature_b"]
            + rng.normal(0.0, 0.05, size=rows)
        ).astype(np.float32)
        frame["id"] = [f"id_{index}" for index in range(len(frame))]
        model = self._model(ema_decay=0.995, max_epochs=1)
        from agents.code.modeling.utils.model_data import build_model_data_loader

        loader = build_model_data_loader(
            full=frame,
            x_cols=["feature_a", "feature_b", "era"],
            era_col="era",
            target_col="target",
            id_col="id",
        )

        def build(*_args, **_kwargs):
            return model

        with mock.patch.object(numerai_cv, "build_model", side_effect=build):
            predictions, cv = numerai_cv.build_oof_predictions(
                frame["era"],
                loader,
                "TorchTabularRegressor",
                {},
                {},
                {
                    "n_splits": 4,
                    "embargo": 0,
                    "mode": "expanding",
                    "min_train_size": 0,
                },
                None,
                1337,
                "id",
                "era",
                "target",
                feature_cols=["feature_a", "feature_b"],
            )
        self.assertFalse(predictions.empty)
        self.assertEqual(len(cv["folds"]), 3)
        for fold in cv["folds"]:
            diagnostics = fold["model_diagnostics"]
            self.assertEqual(diagnostics["ema_decay"], 0.995)
            self.assertGreater(diagnostics["ema_updates"], 0)
            self.assertNotEqual(
                diagnostics["ema_live_state_sha256"],
                diagnostics["ema_shadow_state_sha256"],
            )

    def test_ender24_config_matrix_is_exact(self):
        root = Path(__file__).resolve().parents[1]
        experiment = root / "experiments" / "ender24_ema_seed_stability_v53"
        configs = experiment / "configs"
        expected = {
            "r1_control_seed1337.py": (1337, None),
            "r1_ema995_seed1337.py": (1337, 0.995),
            "r1_control_seed2027.py": (2027, None),
            "r1_ema995_seed2027.py": (2027, 0.995),
            "r2_control_seed7331.py": (7331, None),
            "r2_ema995_seed7331.py": (7331, 0.995),
        }
        loaded = {}
        for filename, (seed, decay) in expected.items():
            path = configs / filename
            self.assertTrue(path.is_file())
            config = runpy.run_path(str(path))["CONFIG"]
            loaded[filename] = config
            self.assertEqual(config["output"]["results_name"], path.stem)
            self.assertEqual(config["model"]["params"]["seed"], seed)
            self.assertEqual(config["training"]["sample_seed"], 1337)
            self.assertEqual(config["training"]["cv"]["max_train_eras"], 78)
            self.assertEqual(config["training"]["max_train_samples"], 500_000)
            if decay is None:
                self.assertNotIn("ema_decay", config["model"]["params"])
            else:
                self.assertEqual(config["model"]["params"]["ema_decay"], decay)

        for seed in (1337, 2027, 7331):
            prefix = "r1" if seed != 7331 else "r2"
            control = deepcopy(loaded[f"{prefix}_control_seed{seed}.py"])
            ema = deepcopy(loaded[f"{prefix}_ema995_seed{seed}.py"])
            control["output"].pop("results_name")
            ema["output"].pop("results_name")
            self.assertEqual(ema["model"]["params"].pop("ema_decay"), 0.995)
            self.assertEqual(control, ema)

        prior_experiment = root / "experiments" / "ender23_temporal_retention_v53"
        prior_config = runpy.run_path(
            str(prior_experiment / "configs" / "r1_recent_window78.py")
        )["CONFIG"]
        current_control = deepcopy(loaded["r1_control_seed1337.py"])
        prior_config = deepcopy(prior_config)
        current_control.pop("output")
        prior_config.pop("output")
        self.assertEqual(current_control, prior_config)

        prior_result_path = (
            prior_experiment / "results" / "r1_recent_window78.json"
        )
        prior_result_bytes = prior_result_path.read_bytes()
        prior_result_bytes = prior_result_bytes.replace(b"\r\n", b"\n")
        self.assertNotIn(b"\r", prior_result_bytes)
        self.assertEqual(
            hashlib.sha256(prior_result_bytes).hexdigest(),
            "399d3bf16c9efae734a13653ee4b7f744080350fd774a4c94e8e6e02c2012d1c",
        )
        prior_cv = json.loads(prior_result_bytes)["cv"]
        folds = prior_cv["folds"]
        self.assertEqual([fold["fold"] for fold in folds], [1, 2, 3, 4])
        self.assertEqual(
            [fold["train_rows"] for fold in folds],
            [117_820, 290_284, 394_883, 415_481],
        )
        self.assertEqual(
            [fold["train_eras"] for fold in folds], [22, 57, 78, 78]
        )
        self.assertTrue(
            all(
                fold["train_rows"]
                < loaded["r1_control_seed1337.py"]["training"][
                    "max_train_samples"
                ]
                for fold in folds
            )
        )

        self.assertFalse((experiment / "source_manifest_round1.json").exists())
        self.assertFalse((experiment / "source_manifest_round2.json").exists())
        self.assertFalse((experiment / "receipts").exists())
        authority = (
            root.parent.parent
            / loaded["r1_control_seed1337.py"]["data"]["era_allowlist_path"]
        )
        eras = json.loads(authority.read_text(encoding="utf-8"))
        self.assertEqual((len(eras), eras[-1]), (176, "0861"))


if __name__ == "__main__":
    unittest.main()
