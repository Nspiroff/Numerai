from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from agents.code.modeling.models.torch_tabular_regressor import (
    TorchTabularRegressor,
)
from agents.code.modeling.utils.model_factory import build_model


class TestTorchTabularRegressor(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        rows = 96
        self.features = ["feature_a", "feature_b", "feature_c"]
        self.X = pd.DataFrame(
            {
                "feature_a": rng.integers(0, 5, rows, dtype=np.int8),
                "feature_b": rng.integers(0, 5, rows, dtype=np.int8),
                "feature_c": rng.integers(0, 5, rows, dtype=np.int8),
                "era": np.repeat([f"{era:04d}" for era in range(12)], 8),
                "v53_lgbm_ender20": rng.random(rows),
            }
        )
        self.y = pd.Series(
            0.5
            + 0.03 * (self.X["feature_a"].astype(float) - 2.0)
            + rng.normal(0.0, 0.01, rows)
        )
        self.y.iloc[0] = np.nan

    def _build(self, architecture: str):
        params = {
            "architecture": architecture,
            "device": "cpu",
            "amp": False,
            "batch_size": 32,
            "max_epochs": 2,
            "patience": 1,
            "val_fraction": 0.2,
            "internal_val_embargo": 1,
            "verbose": False,
            "hidden_layer_sizes": (16, 8),
            "resnet_width": 16,
            "resnet_hidden_width": 32,
            "resnet_blocks": 2,
            "tabm_k": 4,
            "tabm_width": 16,
            "tabm_blocks": 2,
        }
        return TorchTabularRegressor(feature_cols=self.features, **params)

    def test_mlp_and_resnet_fit_predict(self) -> None:
        for architecture in ("mlp", "resnet"):
            with self.subTest(architecture=architecture):
                model = self._build(architecture)
                model.fit(self.X, self.y)
                predictions = model.predict(self.X.iloc[:11])
                self.assertEqual(predictions.shape, (11,))
                self.assertTrue(np.isfinite(predictions).all())
                self.assertGreater(model.n_parameters_, 0)
                self.assertEqual(model.epochs_trained_, len(model.training_history_))

    def test_tabm_trains_individual_members_and_averages_predictions(self) -> None:
        try:
            import tabm  # noqa: F401
        except ImportError:
            self.skipTest("tabm is not installed")
        model = self._build("tabm")
        model.fit(self.X, self.y)
        predictions = model.predict(self.X.iloc[:11])
        self.assertEqual(predictions.shape, (11,))
        self.assertTrue(np.isfinite(predictions).all())

    def test_factory_filters_non_feature_inputs(self) -> None:
        model = build_model(
            "TorchTabularRegressor",
            {
                "architecture": "mlp",
                "hidden_layer_sizes": (8,),
                "device": "cpu",
                "amp": False,
                "max_epochs": 1,
                "val_split": "none",
                "verbose": False,
            },
            feature_cols=self.features,
        )
        model.fit(self.X, self.y)
        self.assertEqual(model._input_cols, self.features)

    def test_no_validation_retains_terminal_epoch_and_exports_cpu_clones(self) -> None:
        model = self._build("mlp")
        with self.assertRaisesRegex(RuntimeError, "fitted before exporting"):
            model.cpu_state_dict()

        model.val_split = "none"
        model.val_fraction = 0.0
        model.max_epochs = 3
        model.best_epoch_ = 99
        losses = iter((0.1, 0.2, 0.3))
        epochs = []

        def train_terminal_epoch(_loader, _optimizer, _scaler):
            epoch = len(epochs) + 1
            epochs.append(epoch)
            with model._torch.no_grad():
                for value in model._model.state_dict().values():
                    value.fill_(float(epoch))
            return next(losses)

        model._train_epoch = train_terminal_epoch
        model.fit(self.X, self.y)

        self.assertEqual(epochs, [1, 2, 3])
        self.assertEqual(model.epochs_trained_, 3)
        self.assertEqual(len(model.training_history_), 3)
        self.assertIsNone(model.best_epoch_)

        exported = model.cpu_state_dict()
        self.assertFalse(model._model.training)
        self.assertTrue(exported)
        for value in exported.values():
            self.assertEqual(value.device.type, "cpu")
            self.assertFalse(value.requires_grad)
            self.assertTrue(model._torch.all(value == 3.0))

        first_key = next(iter(exported))
        exported[first_key].zero_()
        self.assertTrue(model._torch.all(model._model.state_dict()[first_key] == 3.0))


if __name__ == "__main__":
    unittest.main()
