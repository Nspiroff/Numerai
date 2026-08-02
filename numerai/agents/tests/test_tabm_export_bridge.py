from __future__ import annotations

from collections import OrderedDict
import unittest

import numpy as np
import pandas as pd

from agents.code.modeling.deployment.tabm_export import (
    build_tabm_numpy_predictor_from_fitted,
    extract_tabm_numpy_predictor_spec,
)
from agents.code.modeling.deployment.tabm_numpy import build_tabm_numpy_forward
from agents.code.modeling.models.torch_tabular_regressor import (
    TorchTabularRegressor,
)
from agents.code.modeling.utils.target_transforms import TargetTransformWrapper


class _StateModule:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


class TestTabMExportBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import tabm  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("torch/tabm are not installed") from exc

    def _constructed_regressor(self):
        import torch

        feature_names = ["feature_z", "feature_a", "feature_m"]
        torch.manual_seed(41)
        regressor = TorchTabularRegressor(
            feature_cols=feature_names,
            architecture="tabm",
            activation="relu",
            dropout=0.1,
            tabm_arch_type="tabm",
            tabm_k=3,
            tabm_width=7,
            tabm_blocks=2,
            feature_center=2.0,
            feature_scale=2.0,
            prediction_batch_size=4,
            device="cpu",
            amp=False,
            verbose=False,
        )
        regressor._input_cols = list(feature_names)
        regressor._model = regressor._build_model(len(feature_names))
        return regressor

    def test_wrapper_export_matches_torch_and_freezes_feature_order(self) -> None:
        regressor = self._constructed_regressor()
        wrapped = TargetTransformWrapper(
            regressor,
            {"name": "residualize_column", "benchmark_col": "benchmark"},
        )
        spec = extract_tabm_numpy_predictor_spec(wrapped, batch_size=5)

        self.assertEqual(
            spec["feature_names"], ("feature_z", "feature_a", "feature_m")
        )
        self.assertEqual(spec["feature_center"], 2.0)
        self.assertEqual(spec["feature_scale"], 2.0)
        self.assertEqual(spec["era_column"], "era")
        self.assertEqual(len(spec["blocks"]), 2)
        self.assertTrue(spec["blocks"][0]["weight"].flags.c_contiguous)
        self.assertEqual(spec["blocks"][0]["weight"].dtype, np.float32)

        frame = pd.DataFrame(
            {
                "feature_a": [1.0, np.inf, 4.0, 2.0, 0.0, 3.0, 1.0],
                "era": ["0001", "0001", "0001", "0001", "0002", "0002", "0002"],
                "feature_m": [4.0, 3.0, 2.0, np.nan, 1.0, 0.0, 4.0],
                "feature_z": [0.0, 1.0, 2.0, 3.0, -np.inf, 4.0, 1.0],
            },
            index=["id-6", "id-2", "id-4", "id-1", "id-7", "id-3", "id-5"],
        )
        forward = build_tabm_numpy_forward(
            blocks=spec["blocks"],
            output_weight=spec["output_weight"],
            output_bias=spec["output_bias"],
            feature_center=spec["feature_center"],
            feature_scale=spec["feature_scale"],
            batch_size=spec["batch_size"],
            activation=spec["activation"],
        )

        torch_raw = wrapped.predict(frame)
        numpy_raw = forward(frame.loc[:, list(spec["feature_names"])].to_numpy())
        np.testing.assert_allclose(numpy_raw, torch_raw, rtol=1e-5, atol=1e-6)

        predictor = build_tabm_numpy_predictor_from_fitted(wrapped, batch_size=5)
        actual = predictor(frame, pd.DataFrame(index=frame.index))
        expected = (
            pd.Series(torch_raw, index=frame.index)
            .groupby(frame["era"], sort=False, dropna=False)
            .rank(method="average", pct=True)
        )
        self.assertTrue(actual.index.equals(frame.index))
        np.testing.assert_allclose(actual["prediction"], expected)

    def test_export_copies_state_and_maps_identity_preprocessing(self) -> None:
        regressor = self._constructed_regressor()
        regressor.feature_center = None
        regressor.feature_scale = None
        source_weight = regressor._model.state_dict()[
            "backbone.blocks.0.0.weight"
        ]
        spec = extract_tabm_numpy_predictor_spec(regressor)

        self.assertEqual(spec["feature_center"], 0.0)
        self.assertEqual(spec["feature_scale"], 1.0)
        exported_before = spec["blocks"][0]["weight"].copy()
        with regressor._torch.no_grad():
            source_weight.add_(10.0)
        np.testing.assert_array_equal(
            spec["blocks"][0]["weight"], exported_before
        )

    def test_rejects_wrong_wrapper_and_unfitted_or_unsupported_models(self) -> None:
        with self.assertRaisesRegex(TypeError, "TorchTabularRegressor"):
            extract_tabm_numpy_predictor_spec(object())
        with self.assertRaisesRegex(TypeError, "must contain"):
            extract_tabm_numpy_predictor_spec(
                TargetTransformWrapper(object(), "unused")
            )

        regressor = self._constructed_regressor()
        regressor._model = None
        with self.assertRaisesRegex(RuntimeError, "fitted before export"):
            extract_tabm_numpy_predictor_spec(regressor)

        checks = (
            ("architecture", "mlp", "architecture='tabm'"),
            ("tabm_arch_type", "tabm-mini", "tabm_arch_type='tabm'"),
            ("activation", "gelu", "activation='relu'"),
        )
        for attribute, value, message in checks:
            with self.subTest(attribute=attribute):
                regressor = self._constructed_regressor()
                setattr(regressor, attribute, value)
                with self.assertRaisesRegex(ValueError, message):
                    extract_tabm_numpy_predictor_spec(regressor)

    def test_rejects_invalid_feature_and_preprocessing_metadata(self) -> None:
        cases = (
            ("_input_cols", None, RuntimeError, "frozen fitted feature order"),
            ("_input_cols", ["feature_a", "feature_a"], ValueError, "unique"),
            ("feature_center", np.inf, ValueError, "finite"),
            ("feature_scale", 0.0, ValueError, "non-zero"),
        )
        for attribute, value, error, message in cases:
            with self.subTest(attribute=attribute, value=value):
                regressor = self._constructed_regressor()
                setattr(regressor, attribute, value)
                with self.assertRaisesRegex(error, message):
                    extract_tabm_numpy_predictor_spec(regressor)

    def test_rejects_missing_unexpected_and_inconsistent_state(self) -> None:
        base = self._constructed_regressor()
        state = OrderedDict(base._model.state_dict())

        missing = self._constructed_regressor()
        missing_state = OrderedDict(state)
        del missing_state["output.bias"]
        missing._model = _StateModule(missing_state)
        with self.assertRaisesRegex(ValueError, "missing=.*output.bias"):
            extract_tabm_numpy_predictor_spec(missing)

        unexpected = self._constructed_regressor()
        unexpected_state = OrderedDict(state)
        unexpected_state["unsupported.buffer"] = unexpected_state["output.bias"]
        unexpected._model = _StateModule(unexpected_state)
        with self.assertRaisesRegex(ValueError, "unexpected=.*unsupported.buffer"):
            extract_tabm_numpy_predictor_spec(unexpected)

        bad_k = self._constructed_regressor()
        bad_k.tabm_k = 4
        with self.assertRaisesRegex(ValueError, "configuration declares tabm_k=4"):
            extract_tabm_numpy_predictor_spec(bad_k)

        bad_width = self._constructed_regressor()
        bad_width.tabm_width = 8
        with self.assertRaisesRegex(ValueError, "tabm_width=8"):
            extract_tabm_numpy_predictor_spec(bad_width)


if __name__ == "__main__":
    unittest.main()
