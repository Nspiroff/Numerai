from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from agents.code.modeling.deployment.tabm_numpy import (
    build_tabm_numpy_forward,
    build_tabm_numpy_predictor,
)


def _identity_weights(*, k: int = 1):
    return {
        "blocks": [
            {
                "weight": np.ones((1, 1), dtype=np.float32),
                "r": np.ones((k, 1), dtype=np.float32),
                "s": np.ones((k, 1), dtype=np.float32),
                "bias": np.zeros((k, 1), dtype=np.float32),
            }
        ],
        "output_weight": np.ones((k, 1, 1), dtype=np.float32),
        "output_bias": np.zeros((k, 1), dtype=np.float32),
    }


class TestTabMNumpyDeployment(unittest.TestCase):
    def test_numpy_forward_matches_tabm_eval(self) -> None:
        try:
            import torch
            from tabm import TabM
        except ImportError:
            self.skipTest("torch/tabm are not installed")

        torch.manual_seed(19)
        model = TabM.make(
            n_num_features=5,
            d_out=1,
            arch_type="tabm",
            k=3,
            d_block=7,
            n_blocks=2,
            dropout=0.1,
            activation="ReLU",
        ).eval()
        state = model.state_dict()
        blocks = []
        for index in range(2):
            prefix = f"backbone.blocks.{index}.0"
            blocks.append(
                {
                    name: state[f"{prefix}.{name}"].detach().cpu().numpy()
                    for name in ("weight", "r", "s", "bias")
                }
            )
        forward = build_tabm_numpy_forward(
            blocks=blocks,
            output_weight=state["output.weight"].detach().cpu().numpy(),
            output_bias=state["output.bias"].detach().cpu().numpy(),
            feature_center=2.0,
            feature_scale=2.0,
            batch_size=4,
        )

        features = np.random.default_rng(23).normal(size=(11, 5)).astype(np.float32)
        features[0, 0] = np.nan
        features[1, 1] = np.inf
        features[2, 2] = -np.inf
        prepared = np.nan_to_num(
            (features - np.float32(2.0)) / np.float32(2.0),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        with torch.inference_mode():
            expected = (
                model(torch.from_numpy(prepared)).squeeze(-1).mean(dim=1).numpy()
            )
        actual = forward(features)

        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_predict_ranks_within_each_era_and_preserves_index(self) -> None:
        predictor = build_tabm_numpy_predictor(
            feature_names=["feature_a"],
            feature_center=0.0,
            feature_scale=1.0,
            batch_size=2,
            **_identity_weights(),
        )
        index = pd.Index(["id-3", "id-1", "id-2", "id-5", "id-4"], name="id")
        live = pd.DataFrame(
            {
                "era": ["0001", "0001", "0001", "0002", "0002"],
                "ignored": [9, 8, 7, 6, 5],
                "feature_a": [0.0, 2.0, 1.0, 4.0, 4.0],
            },
            index=index,
        )
        output = predictor(live, pd.DataFrame(index=index))

        self.assertEqual(output.columns.tolist(), ["prediction"])
        self.assertTrue(output.index.equals(index))
        np.testing.assert_allclose(
            output["prediction"].to_numpy(),
            np.array([1 / 3, 1.0, 2 / 3, 0.75, 0.75]),
        )
        self.assertTrue(output["prediction"].between(0.0, 1.0).all())

    def test_predict_uses_frozen_feature_order_not_frame_column_order(self) -> None:
        predictor = build_tabm_numpy_predictor(
            feature_names=["feature_a", "feature_b"],
            blocks=[
                {
                    "weight": np.array([[1.0, 10.0]], dtype=np.float32),
                    "r": np.ones((1, 2), dtype=np.float32),
                    "s": np.ones((1, 1), dtype=np.float32),
                    "bias": np.zeros((1, 1), dtype=np.float32),
                }
            ],
            output_weight=np.ones((1, 1, 1), dtype=np.float32),
            output_bias=np.zeros((1, 1), dtype=np.float32),
            feature_center=0.0,
            feature_scale=1.0,
        )
        live = pd.DataFrame(
            {
                "feature_b": [0.0, 1.0],
                "era": ["0001", "0001"],
                "feature_a": [1.0, 0.0],
            },
            index=["a", "b"],
        )

        output = predictor(live, pd.DataFrame(index=live.index))

        self.assertEqual(output["prediction"].tolist(), [0.5, 1.0])

    def test_predictor_rejects_invalid_features_eras_and_index(self) -> None:
        predictor = build_tabm_numpy_predictor(
            feature_names=["feature_a"],
            feature_center=0.0,
            feature_scale=1.0,
            **_identity_weights(),
        )
        valid = pd.DataFrame(
            {"era": ["0001", "0001"], "feature_a": [1.0, 2.0]},
            index=["a", "b"],
        )
        cases = {
            "missing era": valid.drop(columns="era"),
            "missing feature": valid.drop(columns="feature_a"),
            "missing era value": valid.assign(era=["0001", None]),
            "duplicate index": valid.set_axis(["a", "a"]),
            "missing index value": valid.set_axis(["a", None]),
            "nonnumeric feature": valid.assign(feature_a=[1.0, "bad"]),
        }
        for name, frame in cases.items():
            with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                predictor(frame, pd.DataFrame(index=frame.index))

        duplicate_columns = valid.copy()
        duplicate_columns.columns = ["era", "era"]
        with self.assertRaisesRegex(ValueError, "column names must be unique"):
            predictor(
                duplicate_columns,
                pd.DataFrame(index=duplicate_columns.index),
            )

    def test_weight_validation_rejects_shape_and_nonfinite_state(self) -> None:
        bad_shape = _identity_weights(k=2)
        bad_shape["blocks"][0]["s"] = np.ones((1, 1), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "must have shape"):
            build_tabm_numpy_forward(**bad_shape)

        nonfinite = _identity_weights()
        nonfinite["output_bias"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_tabm_numpy_forward(**nonfinite)

    def test_forward_rejects_nonfinite_predictions(self) -> None:
        weights = _identity_weights()
        weights["output_weight"][0, 0, 0] = np.finfo(np.float32).max
        forward = build_tabm_numpy_forward(
            feature_center=0.0,
            feature_scale=1.0,
            **weights,
        )
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            forward(np.array([[np.finfo(np.float32).max]], dtype=np.float32))

    def test_cloudpickled_callable_loads_without_repo_or_ml_packages(self) -> None:
        try:
            import cloudpickle
        except ImportError:
            self.skipTest("cloudpickle is not installed")

        predictor = build_tabm_numpy_predictor(
            feature_names=["feature_a"],
            feature_center=0.0,
            feature_scale=1.0,
            **_identity_weights(k=2),
        )
        with tempfile.TemporaryDirectory() as tmp:
            pickle_path = Path(tmp) / "predict.pkl"
            with pickle_path.open("wb") as handle:
                cloudpickle.dump(predictor, handle)

            script = r'''
import sys

class BlockLocalAndMLPackages:
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("agents", "tabm", "torch", "rtdl_num_embeddings")
        if fullname in blocked or fullname.startswith(tuple(x + "." for x in blocked)):
            raise ModuleNotFoundError(f"blocked portability dependency: {fullname}")
        return None

sys.meta_path.insert(0, BlockLocalAndMLPackages())
import cloudpickle
import pandas as pd

with open(sys.argv[1], "rb") as handle:
    predict = cloudpickle.load(handle)
features = pd.DataFrame(
    {"feature_a": [1.0, 3.0], "era": ["0001", "0001"]},
    index=["a", "b"],
)
output = predict(features, pd.DataFrame(index=features.index))
assert output.index.equals(features.index)
assert output.columns.tolist() == ["prediction"]
assert output["prediction"].tolist() == [0.5, 1.0]
assert not any(name == "agents" or name.startswith("agents.") for name in sys.modules)
'''
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            completed = subprocess.run(
                [sys.executable, "-I", "-c", script, str(pickle_path)],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
