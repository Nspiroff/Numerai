from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from agents.code.metrics import numerai_metrics
from agents.code.modeling.utils.numerai_cv import era_cv_splits
from agents.code.modeling.utils.pipeline import PREDICTION_SEMANTICS_METADATA_KEY


def _load_evaluator():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/ender21_residual_stability_v53/evaluate_round1.py"
    )
    spec = importlib.util.spec_from_file_location("ender21_round1_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the Ender21 Round-1 evaluator.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load_evaluator()


def _prediction_semantics() -> dict:
    return {
        "artifact_kind": "out_of_fold_validation",
        "column": "prediction",
        "era_column": "era",
        "fold_column": "cv_fold",
        "fold_index_base": 0,
        "inverse_target_transform_applied": False,
        "pipeline_postprocess": {"type": "identity"},
        "producer": "model.predict",
        "schema_version": 1,
        "stored_target": {
            "column": "target_ender_20",
            "transform": {"type": "identity"},
        },
        "training_target": {
            "column": "target_ender_20",
            "transform": {
                "benchmark_col": "v53_lgbm_ender20",
                "era_col": "era",
                "fit_intercept": True,
                "per_era": True,
                "proportion": 1.0,
                "type": "residual_to_benchmark",
            },
        },
    }


class TestEnder21Round1Evaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.experiment = Path(self.temporary.name)
        for name in ("configs", "predictions", "results", "protocol"):
            (self.experiment / name).mkdir()

        self.name = "r1_control_tabm_k64"
        self.allowed = [f"{era:04d}" for era in range(161, 862, 4)]
        allowlist_path = self.experiment / "protocol/discovery_eras_through_0861.json"
        allowlist_path.write_text(
            json.dumps(self.allowed, indent=2) + "\n", encoding="utf-8"
        )

        rows = []
        target_values = (0.0, 0.25, 0.5, 0.75, 1.0)
        benchmark_values = (0.8, 0.1, 0.6, 0.2, 0.4)
        prediction_values = (0.2, 0.9, 0.4, 0.7, 0.1)
        for era in self.allowed:
            for position in range(5):
                rows.append(
                    {
                        "id": f"{era}_{position}",
                        "era": era,
                        "target_ender_20": target_values[position],
                        "v53_lgbm_ender20": benchmark_values[position],
                        "prediction": prediction_values[position],
                    }
                )
        complete = pd.DataFrame(rows)
        self.truth = complete[
            ["id", "era", "target_ender_20", "v53_lgbm_ender20"]
        ].copy()

        expected_folds = {}
        fold_contract = []
        for fold, (train_eras, val_eras) in enumerate(
            era_cv_splits(
                self.allowed,
                n_splits=5,
                embargo=13,
                mode="expanding",
                min_train_size=0,
            )
        ):
            if not train_eras:
                continue
            expected_folds.update({era: fold for era in val_eras})
            fold_contract.append(
                {
                    "fold": fold,
                    "train_eras": len(train_eras),
                    "val_eras": len(val_eras),
                    "train_rows": min(
                        int(self.truth["era"].isin(train_eras).sum()), 500_000
                    ),
                    "val_rows": int(self.truth["era"].isin(val_eras).sum()),
                    "model_diagnostics": {},
                }
            )
        self.frame = complete.loc[complete["era"].isin(expected_folds)].copy()
        self.frame["cv_fold"] = self.frame["era"].map(expected_folds).astype("int64")
        self.frame = self.frame[
            ["id", "era", "target_ender_20", "prediction", "cv_fold"]
        ]

        model = {
            "type": "TorchTabularRegressor",
            "params": {"architecture": "tabm", "seed": 1337},
        }
        preprocessing = {"missing_value": 2.0, "nan_missing_all_twos": False}
        cv_config = {
            "embargo": 13,
            "enabled": True,
            "min_train_size": 0,
            "mode": "expanding",
            "n_splits": 5,
        }
        allowlist_relative = (
            "numerai/agents/experiments/ender21_residual_stability_v53/"
            "protocol/discovery_eras_through_0861.json"
        )
        config = {
            "data": {
                "era_allowlist_path": allowlist_relative,
                "full_data_path": "v5.3/ender21_discovery_full_through_0861.parquet",
                "benchmark_data_path": (
                    "v5.3/ender21_discovery_benchmark_models_through_0861.parquet"
                ),
            },
            "model": model,
            "preprocessing": preprocessing,
            "training": {"cv": cv_config},
        }
        (self.experiment / f"configs/{self.name}.py").write_text(
            "CONFIG = " + repr(config) + "\n", encoding="utf-8"
        )
        (self.experiment / f"predictions/{self.name}.parquet").write_bytes(b"fixture")

        scored = self.frame.merge(
            self.truth[["id", "era", "v53_lgbm_ender20"]],
            on=["id", "era"],
            validate="one_to_one",
        )
        bmc = numerai_metrics.per_era_bmc(
            scored,
            ["prediction"],
            "v53_lgbm_ender20",
            "target_ender_20",
        )["prediction"]
        corr = numerai_metrics.per_era_corr(
            scored, ["prediction"], "target_ender_20"
        )["prediction"]
        bmc_summary = numerai_metrics.score_summary(bmc)
        corr_summary = numerai_metrics.score_summary(corr)
        allowlist_bytes = allowlist_path.read_bytes()
        allowlist_receipt = {
            "path": allowlist_relative,
            "sha256": hashlib.sha256(allowlist_bytes).hexdigest(),
            "size_bytes": len(allowlist_bytes),
            "era_count": 176,
            "first_era": "0161",
            "last_era": "0861",
        }
        self.stored = {
            "model": model,
            "preprocessing": preprocessing,
            "training": {
                "data_sampling": {
                    "max_train_samples": 500_000,
                    "sample_seed": 1337,
                },
                "data_mode": "eager",
                "cv": cv_config,
            },
            "cv": {
                "n_splits": 5,
                "embargo": 13,
                "mode": "expanding",
                "min_train_size": 0,
                "folds_used": 4,
                "folds": fold_contract,
            },
            "data": {
                "data_version": "v5.3",
                "feature_set": "all",
                "target": "target_ender_20",
                "full_data_path": config["data"]["full_data_path"],
                "full_rows": len(self.truth),
                "full_eras": len(self.allowed),
                "oof_rows": len(self.frame),
                "oof_eras": len(expected_folds),
                "embargo_eras": 13,
                "require_benchmark_coverage": True,
                "data_mode": "eager",
                "era_allowlist": allowlist_receipt,
            },
            "benchmark": {
                "model": "v53_lgbm_ender20",
                "file": config["data"]["benchmark_data_path"],
            },
            "output": {"prediction_semantics": _prediction_semantics()},
            "metrics": {
                "bmc": {
                    "mean": bmc_summary["mean"],
                    "sharpe": bmc_summary["sharpe"],
                    "max_drawdown": bmc_summary["max_drawdown"],
                },
                "corr": {"mean": corr_summary["mean"]},
            },
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _assert_rejected(self, mutation, message: str) -> None:
        stored = deepcopy(self.stored)
        mutation(stored)
        (self.experiment / f"results/{self.name}.json").write_text(
            json.dumps(stored), encoding="utf-8"
        )
        fake_parquet = SimpleNamespace(
            schema_arrow=SimpleNamespace(
                metadata={
                    PREDICTION_SEMANTICS_METADATA_KEY: json.dumps(
                        _prediction_semantics(), sort_keys=True
                    ).encode("utf-8")
                }
            ),
            read=lambda: SimpleNamespace(to_pandas=lambda: self.frame.copy()),
        )
        with mock.patch.object(
            EVALUATOR.pq, "ParquetFile", return_value=fake_parquet
        ):
            with self.assertRaisesRegex(ValueError, message):
                EVALUATOR._score_candidate(
                    self.experiment, self.name, self.allowed, self.truth
                )

    def test_rejects_blocked_cv_contract(self) -> None:
        self._assert_rejected(
            lambda stored: stored["training"]["cv"].update(mode="blocked"),
            "stored training contract differs",
        )

    def test_rejects_wrong_training_sample_cap(self) -> None:
        self._assert_rejected(
            lambda stored: stored["training"]["data_sampling"].update(
                max_train_samples=250_000
            ),
            "stored training contract differs",
        )

    def test_rejects_wrong_training_sample_seed(self) -> None:
        self._assert_rejected(
            lambda stored: stored["training"]["data_sampling"].update(
                sample_seed=2027
            ),
            "stored training contract differs",
        )

    def test_rejects_wrong_allowlist_binding(self) -> None:
        self._assert_rejected(
            lambda stored: stored["data"]["era_allowlist"].update(
                sha256="0" * 64
            ),
            "stored allowlist binding is absent or wrong",
        )


if __name__ == "__main__":
    unittest.main()
