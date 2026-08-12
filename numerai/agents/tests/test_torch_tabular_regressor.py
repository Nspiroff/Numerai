from __future__ import annotations

import json
import runpy
import unittest
from pathlib import Path

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

    def _build_loss_probe(
        self,
        loss_mode: str,
        *,
        architecture: str = "mlp",
        val_split: str = "none",
        val_fraction: float = 0.0,
    ):
        model = TorchTabularRegressor(
            feature_cols=["feature_a"],
            architecture=architecture,
            hidden_layer_sizes=(4,),
            tabm_k=2,
            tabm_width=4,
            tabm_blocks=1,
            device="cpu",
            amp=False,
            batch_size=128,
            prediction_batch_size=128,
            learning_rate=0.0,
            weight_decay=0.0,
            max_epochs=1,
            patience=1,
            val_split=val_split,
            val_fraction=val_fraction,
            internal_val_embargo=0,
            feature_center=None,
            feature_scale=None,
            deterministic=True,
            verbose=False,
            loss_mode=loss_mode,
            chronological_blocks=2,
            dro_temperature=2.0,
        )

        torch = model._torch

        def build_fixed_output(input_dim: int):
            if architecture == "tabm":
                class FixedTabM(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.member_bias = torch.nn.Parameter(
                            torch.tensor([0.0, 2.0], dtype=torch.float32)
                        )

                    def forward(self, x):
                        return self.member_bias[None, :, None].expand(
                            x.shape[0], -1, -1
                        )

                return FixedTabM()

            layer = torch.nn.Linear(input_dim, 1)
            with torch.no_grad():
                layer.weight.zero_()
                layer.bias.zero_()
            return layer

        model._build_model = build_fixed_output
        return model

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

    def test_mse_and_era_balanced_mse_match_their_exact_objectives(self) -> None:
        X = pd.DataFrame(
            {
                "feature_a": np.zeros(8, dtype=np.float32),
                "era": ["0001", *(["0002"] * 3), *(["0003"] * 4)],
            }
        )
        y = pd.Series([1.0, *([3.0] * 3), *([2.0] * 4)])

        expected = {
            "mse": float(np.mean(np.square(y.to_numpy()))),
            "era_balanced_mse": float(np.mean([1.0, 9.0, 4.0])),
        }
        for loss_mode, expected_loss in expected.items():
            with self.subTest(loss_mode=loss_mode):
                model = self._build_loss_probe(loss_mode)
                model.fit(X, y)
                self.assertAlmostEqual(
                    model.training_history_[0]["train_loss"],
                    expected_loss,
                    places=6,
                )

    def test_chronological_block_dro_uses_member_mse_and_frozen_weights(self) -> None:
        X = pd.DataFrame(
            {
                "feature_a": np.zeros(8, dtype=np.float32),
                "era": [
                    "0001",
                    "0002",
                    "0003",
                    "0004",
                    "0005",
                    "0006",
                    "0007",
                    "0008",
                ],
            }
        )
        y = pd.Series([1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0])

        # The fixed two-member TabM emits [0, 2]. Member-averaged row MSE is
        # therefore 1 for target=1 and 5 for target=3.
        block_losses = np.array([1.0, 5.0], dtype=np.float64)
        logits = 2.0 * (block_losses / block_losses.mean() - 1.0)
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        expected = float(np.dot(weights, block_losses))

        model = self._build_loss_probe(
            "chronological_block_dro", architecture="tabm"
        )
        model.fit(X, y)
        np.testing.assert_allclose(
            model.training_history_[0]["train_loss"],
            expected,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_dro_blocks_are_derived_after_inner_split_and_validation_is_mse(self) -> None:
        X = pd.DataFrame(
            {
                "feature_a": np.zeros(8, dtype=np.float32),
                "era": [f"{era:04d}" for era in range(1, 9)],
            }
        )
        y = pd.Series([1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 2.0, 4.0])

        # The inner split leaves eras 0001-0006 for training. Splitting those
        # six eras into two near-equal blocks yields losses [1, 9]. If block
        # membership were incorrectly derived before the split, it would yield
        # represented training losses [3, 9] instead.
        block_losses = np.array([1.0, 9.0], dtype=np.float64)
        logits = 2.0 * (block_losses / block_losses.mean() - 1.0)
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        expected_train = float(np.dot(weights, block_losses))
        expected_validation = float(np.mean([2.0**2, 4.0**2]))

        model = self._build_loss_probe(
            "chronological_block_dro",
            val_split="recent_eras",
            val_fraction=0.25,
        )
        model.fit(X, y)
        history = model.training_history_[0]
        self.assertAlmostEqual(history["train_loss"], expected_train, places=6)
        self.assertAlmostEqual(
            history["val_loss"], expected_validation, places=6
        )

    def test_loss_configuration_rejects_invalid_values(self) -> None:
        common = {
            "feature_cols": ["feature_a"],
            "device": "cpu",
            "amp": False,
            "verbose": False,
        }
        with self.assertRaisesRegex(ValueError, "loss_mode"):
            TorchTabularRegressor(loss_mode="not-a-loss", **common)

        for value in (0, -1, 1.5, True):
            with self.subTest(chronological_blocks=value), self.assertRaisesRegex(
                (TypeError, ValueError), "chronological_blocks"
            ):
                TorchTabularRegressor(
                    loss_mode="chronological_block_dro",
                    chronological_blocks=value,
                    **common,
                )

        for value in (0.0, -1.0, np.inf, np.nan):
            with self.subTest(dro_temperature=value), self.assertRaisesRegex(
                ValueError, "dro_temperature"
            ):
                TorchTabularRegressor(
                    loss_mode="chronological_block_dro",
                    dro_temperature=value,
                    **common,
                )

    def test_ender21_round_one_configs_match_the_frozen_contract(self) -> None:
        agents_dir = Path(__file__).resolve().parents[1]
        experiment = agents_dir / "experiments" / "ender21_residual_stability_v53"
        expected = {
            "r1_control_tabm_k64.py": ("tabm", "mse"),
            "r1_tabm_mini_k64.py": ("tabm-mini", "mse"),
            "r1_tabm_k64_era_balanced.py": ("tabm", "era_balanced_mse"),
            "r1_tabm_k64_block_dro.py": ("tabm", "chronological_block_dro"),
            "r1_tabm_mini_k64_block_dro.py": (
                "tabm-mini",
                "chronological_block_dro",
            ),
        }

        loaded = {}
        for filename, contract in expected.items():
            path = experiment / "configs" / filename
            self.assertTrue(path.is_file(), f"Missing frozen Ender21 config: {path}")
            config = runpy.run_path(str(path))["CONFIG"]
            loaded[filename] = config
            params = config["model"]["params"]
            self.assertEqual(params["tabm_arch_type"], contract[0])
            self.assertEqual(params["loss_mode"], contract[1])
            self.assertEqual(params["tabm_k"], 64)
            self.assertEqual(params["tabm_width"], 512)
            self.assertEqual(params["tabm_blocks"], 3)
            self.assertEqual(params["seed"], 1337)
            self.assertEqual(config["training"]["sample_seed"], 1337)
            self.assertEqual(config["training"]["max_train_samples"], 500_000)
            self.assertEqual(config["output"]["results_name"], path.stem)

            if contract[1] == "chronological_block_dro":
                self.assertEqual(params["chronological_blocks"], 4)
                self.assertEqual(params["dro_temperature"], 2.0)

        control = loaded["r1_control_tabm_k64.py"]
        data = control["data"]
        self.assertEqual(data["data_version"], "v5.3")
        self.assertEqual(data["feature_set"], "all")
        self.assertEqual(data["target_col"], "target_ender_20")
        self.assertEqual(data["benchmark_model"], "v53_lgbm_ender20")
        self.assertTrue(data["require_benchmark_coverage"])
        self.assertEqual(control["training"]["cv"]["n_splits"], 5)
        self.assertEqual(control["training"]["cv"]["embargo"], 13)
        transform = control["model"]["target_transform"]
        self.assertEqual(transform["type"], "residual_to_benchmark")
        self.assertEqual(transform["benchmark_col"], "v53_lgbm_ender20")
        self.assertTrue(transform["per_era"])
        self.assertTrue(transform["fit_intercept"])
        self.assertEqual(transform.get("proportion", 1.0), 1.0)

        expected_allowlist = (
            "numerai/agents/experiments/ender21_residual_stability_v53/"
            "protocol/discovery_eras_through_0861.json"
        )
        for config in loaded.values():
            self.assertEqual(config["data"]["era_allowlist_path"], expected_allowlist)
        allowlist = agents_dir.parent.parent / data["era_allowlist_path"]
        eras = json.loads(allowlist.read_text(encoding="utf-8"))
        self.assertEqual(len(eras), 176)
        self.assertEqual(len(eras), len(set(eras)))
        self.assertTrue(all(isinstance(era, str) for era in eras))
        self.assertEqual(eras[-1], "0861")
        self.assertIn("discovery_full_through_0861", data["full_data_path"])
        self.assertIn(
            "discovery_benchmark_models_through_0861",
            data["benchmark_data_path"],
        )


if __name__ == "__main__":
    unittest.main()
