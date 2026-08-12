from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agents.code.modeling.models.torch_tabular_regressor import (
    TorchTabularRegressor,
)
from agents.code.modeling.utils.model_data import ModelDataBatch
from agents.code.modeling.utils.numerai_cv import (
    build_oof_predictions,
    era_cv_splits,
)


class _RecordingLoader:
    def __init__(self, eras: list[str]) -> None:
        self.requests: list[tuple[str, ...]] = []
        self.full = pd.DataFrame(
            {
                "id": [f"id_{era}" for era in eras],
                "era": eras,
                "feature_a": np.arange(len(eras), dtype=np.float32),
                "target_ender_20": np.linspace(0.0, 1.0, len(eras)),
            }
        )

    def load(self, eras) -> ModelDataBatch:
        requested = tuple(eras)
        self.requests.append(requested)
        subset = self.full[self.full["era"].isin(requested)]
        return ModelDataBatch(
            X=subset[["feature_a", "era"]],
            y=subset["target_ender_20"],
            era=subset["era"],
            id=subset["id"],
        )


class _NoOpModel:
    def fit(self, X, y):
        if len(X) != len(y):
            raise AssertionError("Synthetic fit inputs lost row alignment.")
        return self

    @staticmethod
    def predict(X):
        return np.zeros(len(X), dtype=np.float32)


class TestEnder22MaxTrainEras(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The frozen Ender21 discovery cohort contains 176 retained four-week
        # eras.  Reproducing that exact geometry makes this a direct contract
        # test for Ender22's proposed 78-era rolling training window.
        cls.eras = [f"{161 + 4 * index:04d}" for index in range(176)]

    def _run(self, max_train_eras):
        loader = _RecordingLoader(self.eras)
        cv_config = {
            "n_splits": 5,
            "embargo": 13,
            "mode": "expanding",
            "min_train_size": 0,
        }
        if max_train_eras is not None:
            cv_config["max_train_eras"] = max_train_eras
        with patch(
            "agents.code.modeling.utils.numerai_cv.build_model",
            return_value=_NoOpModel(),
        ):
            oof, metadata = build_oof_predictions(
                self.eras,
                loader,
                model_type="TorchTabularRegressor",
                model_params={},
                model_config={},
                cv_config=cv_config,
                max_train_samples=None,
                sample_seed=1337,
                id_col="id",
                era_col="era",
                target_col="target_ender_20",
                feature_cols=["feature_a"],
            )
        return loader, oof, metadata

    def test_exact_78_era_geometry_and_validation_are_unchanged(self) -> None:
        unlimited_loader, unlimited_oof, unlimited = self._run(None)
        limited_loader, limited_oof, limited = self._run(78)

        self.assertEqual(
            [fold["train_eras"] for fold in unlimited["folds"]],
            [22, 57, 92, 127],
        )
        self.assertEqual(
            [fold["train_eras"] for fold in limited["folds"]],
            [22, 57, 78, 78],
        )
        self.assertEqual(
            [fold["available_train_eras"] for fold in limited["folds"]],
            [22, 57, 92, 127],
        )
        self.assertEqual(limited["max_train_eras"], 78)
        self.assertTrue(
            all(fold["max_train_eras"] == 78 for fold in limited["folds"])
        )

        # Every successful fold requests training first and validation second.
        # A rolling train window must not move, shrink, or otherwise change the
        # outer validation cohorts.
        self.assertEqual(unlimited_loader.requests[1::2], limited_loader.requests[1::2])
        self.assertEqual(
            unlimited_oof[["id", "era", "cv_fold"]].to_dict("records"),
            limited_oof[["id", "era", "cv_fold"]].to_dict("records"),
        )

    def test_train_eras_are_trimmed_before_the_loader_is_called(self) -> None:
        loader, _, metadata = self._run(78)
        outer_splits = era_cv_splits(
            self.eras,
            n_splits=5,
            embargo=13,
            mode="expanding",
            min_train_size=0,
        )
        expected_train_requests = [
            tuple(train_eras[-78:])
            for train_eras, validation_eras in outer_splits
            if train_eras and validation_eras
        ]
        self.assertEqual(loader.requests[0::2], expected_train_requests)
        for request, receipt in zip(loader.requests[0::2], metadata["folds"]):
            self.assertLessEqual(len(request), 78)
            self.assertEqual(receipt["first_train_era"], request[0])
            self.assertEqual(receipt["last_train_era"], request[-1])

    def test_invalid_window_fails_before_any_loader_or_model_access(self) -> None:
        for invalid in (True, np.bool_(False), 0, -1, 78.0, "78"):
            with self.subTest(invalid=invalid):
                loader = _RecordingLoader(self.eras)
                with patch(
                    "agents.code.modeling.utils.numerai_cv.build_model"
                ) as build_model_mock:
                    with self.assertRaisesRegex(ValueError, "max_train_eras"):
                        build_oof_predictions(
                            self.eras,
                            loader,
                            model_type="TorchTabularRegressor",
                            model_params={},
                            model_config={},
                            cv_config={
                                "n_splits": 5,
                                "embargo": 13,
                                "mode": "expanding",
                                "min_train_size": 0,
                                "max_train_eras": invalid,
                            },
                            max_train_samples=None,
                            sample_seed=1337,
                            id_col="id",
                            era_col="era",
                            target_col="target_ender_20",
                            feature_cols=["feature_a"],
                        )
                self.assertEqual(loader.requests, [])
                build_model_mock.assert_not_called()

    def test_large_window_is_an_explicit_no_op(self) -> None:
        unlimited_loader, unlimited_oof, unlimited = self._run(None)
        limited_loader, limited_oof, limited = self._run(10_000)

        self.assertEqual(unlimited_loader.requests, limited_loader.requests)
        self.assertEqual(
            unlimited_oof[["id", "era", "cv_fold"]].to_dict("records"),
            limited_oof[["id", "era", "cv_fold"]].to_dict("records"),
        )
        self.assertEqual(
            [fold["train_eras"] for fold in unlimited["folds"]],
            [fold["train_eras"] for fold in limited["folds"]],
        )


class TestEnder22RecencyHalfLife(unittest.TestCase):
    @staticmethod
    def _build(recency_half_life_eras):
        return TorchTabularRegressor(
            feature_cols=["feature_a"],
            architecture="mlp",
            hidden_layer_sizes=(4,),
            device="cpu",
            amp=False,
            batch_size=128,
            prediction_batch_size=128,
            learning_rate=0.0,
            weight_decay=0.0,
            max_epochs=1,
            patience=1,
            val_split="none",
            val_fraction=0.0,
            feature_center=None,
            feature_scale=None,
            deterministic=True,
            verbose=False,
            loss_mode="chronological_block_dro",
            chronological_blocks=2,
            dro_temperature=2.0,
            recency_half_life_eras=recency_half_life_eras,
        )

    @staticmethod
    def _training_rows():
        # The non-chronological row order is intentional: auxiliary metadata
        # must remain aligned to train_indices rather than being sorted by era.
        X = pd.DataFrame(
            {
                "feature_a": np.zeros(6, dtype=np.float32),
                "era": ["0004", "0001", "0003", "0002", "0004", "0002"],
            }
        )
        train_indices = np.asarray([1, 2, 3, 4], dtype=np.int64)
        return X, train_indices

    def test_half_life_formula_and_row_alignment(self) -> None:
        model = self._build(2.0)
        X, train_indices = self._training_rows()
        auxiliary = model._training_loss_aux(X, train_indices)

        self.assertEqual(auxiliary.shape, (4, 2))
        np.testing.assert_array_equal(auxiliary[:, 0], [0.0, 1.0, 0.0, 1.0])
        expected_weights = np.asarray(
            [
                2.0 ** (-3.0 / 2.0),  # era 0001
                2.0 ** (-1.0 / 2.0),  # era 0003
                2.0 ** (-2.0 / 2.0),  # era 0002
                1.0,  # era 0004, newest retained training era
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            auxiliary[:, 1], expected_weights, rtol=1e-6, atol=1e-7
        )

        # Confirm the aligned weights flow into each weighted block mean before
        # the existing detached Block-DRO softmax is applied.
        torch = model._torch
        predictions = torch.zeros((4, 1), dtype=torch.float32)
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        actual = float(
            model._training_loss(
                predictions,
                targets,
                torch.from_numpy(auxiliary),
            )
        )
        row_losses = np.square(targets.numpy()).astype(np.float64)
        weights = expected_weights.astype(np.float64)
        block_losses = np.asarray(
            [
                np.dot(row_losses[[0, 2]], weights[[0, 2]])
                / weights[[0, 2]].sum(),
                np.dot(row_losses[[1, 3]], weights[[1, 3]])
                / weights[[1, 3]].sum(),
            ]
        )
        logits = 2.0 * (block_losses / block_losses.mean() - 1.0)
        dro_weights = np.exp(logits - logits.max())
        dro_weights /= dro_weights.sum()
        recency_prior = np.asarray(
            [weights[[0, 2]].sum(), weights[[1, 3]].sum()]
        )
        recency_prior /= recency_prior.sum()
        dro_weights *= recency_prior
        dro_weights /= dro_weights.sum()
        expected = float(np.dot(dro_weights, block_losses))
        self.assertAlmostEqual(actual, expected, places=5)

    def test_none_preserves_the_legacy_one_dimensional_block_contract(self) -> None:
        model = self._build(None)
        X, train_indices = self._training_rows()
        auxiliary = model._training_loss_aux(X, train_indices)

        self.assertEqual(auxiliary.shape, (4,))
        self.assertEqual(auxiliary.dtype, np.dtype(np.int64))
        np.testing.assert_array_equal(auxiliary, [0, 1, 0, 1])

        torch = model._torch
        predictions = torch.zeros((4, 1), dtype=torch.float32)
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        actual = float(
            model._training_loss(
                predictions,
                targets,
                torch.from_numpy(auxiliary),
            )
        )
        block_losses = np.asarray([5.0, 10.0], dtype=np.float64)
        logits = 2.0 * (block_losses / block_losses.mean() - 1.0)
        dro_weights = np.exp(logits - logits.max())
        dro_weights /= dro_weights.sum()
        self.assertAlmostEqual(
            actual,
            float(np.dot(dro_weights, block_losses)),
            places=5,
        )

    def test_invalid_or_incompatible_half_life_is_rejected(self) -> None:
        for invalid in (True, 0.0, -1.0, float("inf"), float("nan"), "bad"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "recency_half_life_eras"):
                    self._build(invalid)

        with self.assertRaisesRegex(ValueError, "chronological_block_dro"):
            TorchTabularRegressor(
                feature_cols=["feature_a"],
                architecture="mlp",
                device="cpu",
                amp=False,
                loss_mode="mse",
                recency_half_life_eras=52.0,
            )


if __name__ == "__main__":
    unittest.main()
