from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from agents.code.metrics import numerai_metrics
from agents.code.modeling.utils import pipeline


BENCHMARK = "v53_test_benchmark"
TARGET = "target_test"


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["c", "a", "b"],
            "era": ["0003", "0001", "0002"],
            TARGET: [0.7, 0.2, 0.5],
            "prediction": [0.8, 0.1, 0.4],
            "cv_fold": [1, 0, 0],
        }
    )


def _benchmark() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["a", "b", "c", "extra"],
            "era": ["0001", "0002", "0003", "0004"],
            BENCHMARK: [0.11, 0.22, 0.33, 0.44],
        }
    )


class TestBenchmarkAlignment(unittest.TestCase):
    def test_left_one_to_one_join_preserves_prediction_order_and_allows_extras(self):
        attached = numerai_metrics.attach_benchmark_predictions(
            _predictions(), _benchmark(), BENCHMARK
        )
        self.assertEqual(attached["id"].tolist(), ["c", "a", "b"])
        self.assertEqual(attached[BENCHMARK].tolist(), [0.33, 0.11, 0.22])

        indexed = numerai_metrics.attach_benchmark_predictions(
            _predictions().set_index("id"),
            _benchmark().set_index("id"),
            BENCHMARK,
        )
        self.assertEqual(indexed["id"].tolist(), ["c", "a", "b"])

    def test_missing_named_id_or_era_is_rejected_without_positional_fallback(self):
        cases = {
            "prediction id": (_predictions().drop(columns="id"), _benchmark()),
            "benchmark id": (_predictions(), _benchmark().drop(columns="id")),
            "prediction era": (_predictions().drop(columns="era"), _benchmark()),
            "benchmark era": (_predictions(), _benchmark().drop(columns="era")),
        }
        for name, (predictions, benchmark) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, "missing named column"):
                numerai_metrics.attach_benchmark_predictions(
                    predictions, benchmark, BENCHMARK
                )

    def test_null_and_duplicate_alignment_keys_are_rejected(self):
        cases = []
        for source in ("prediction", "benchmark"):
            base = _predictions() if source == "prediction" else _benchmark()
            for column in ("id", "era"):
                mutated = base.copy()
                mutated.loc[0, column] = None
                cases.append((f"{source} null {column}", source, mutated, "null"))
            duplicated = base.copy()
            duplicated.loc[1, "id"] = duplicated.loc[0, "id"]
            cases.append((f"{source} duplicate id", source, duplicated, "duplicate ids"))

        for name, source, mutated, message in cases:
            predictions = mutated if source == "prediction" else _predictions()
            benchmark = mutated if source == "benchmark" else _benchmark()
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                numerai_metrics.attach_benchmark_predictions(
                    predictions, benchmark, BENCHMARK
                )

    def test_complete_benchmark_coverage_is_required(self):
        benchmark = _benchmark()[lambda frame: frame["id"] != "c"]
        with self.assertRaisesRegex(ValueError, "completely cover"):
            numerai_metrics.attach_benchmark_predictions(
                _predictions(), benchmark, BENCHMARK
            )

    def test_era_values_and_dtypes_must_match_exactly(self):
        wrong_value = _benchmark()
        wrong_value.loc[wrong_value["id"] == "c", "era"] = "9999"
        with self.assertRaisesRegex(ValueError, "exactly match"):
            numerai_metrics.attach_benchmark_predictions(
                _predictions(), wrong_value, BENCHMARK
            )

        integer_eras = _predictions()
        integer_eras["era"] = [3, 1, 2]
        with self.assertRaisesRegex(ValueError, "exactly match"):
            numerai_metrics.attach_benchmark_predictions(
                integer_eras, _benchmark(), BENCHMARK
            )

    def test_attached_benchmark_must_be_finite_numeric(self):
        for value in (np.inf, np.nan, "not-numeric"):
            benchmark = _benchmark()
            if isinstance(value, str):
                benchmark[BENCHMARK] = benchmark[BENCHMARK].astype(object)
            benchmark.loc[benchmark["id"] == "c", BENCHMARK] = value
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "finite numeric"
            ):
                numerai_metrics.attach_benchmark_predictions(
                    _predictions(), benchmark, BENCHMARK
                )


class TestValidatedScoringCohort(unittest.TestCase):
    def test_summary_loads_and_attaches_once_then_reuses_the_same_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            predictions_path = Path(tmp) / "predictions.parquet"
            _predictions().to_parquet(predictions_path, index=False)
            cohorts: list[pd.DataFrame] = []
            score = pd.DataFrame(
                {"prediction": [0.01, 0.02, 0.03]},
                index=["0001", "0002", "0003"],
            )

            def capture(frame, *args, **kwargs):
                cohorts.append(frame)
                return score.copy()

            original_read_parquet = pd.read_parquet
            with mock.patch.object(
                numerai_metrics.pd,
                "read_parquet",
                wraps=original_read_parquet,
            ) as read_parquet, mock.patch.object(
                numerai_metrics,
                "attach_benchmark_predictions",
                wraps=numerai_metrics.attach_benchmark_predictions,
            ) as attach, mock.patch.object(
                numerai_metrics, "per_era_corr", side_effect=capture
            ), mock.patch.object(
                numerai_metrics, "per_era_bmc", side_effect=capture
            ), mock.patch.object(
                numerai_metrics, "per_era_pred_corr", side_effect=capture
            ):
                summaries = numerai_metrics.summarize_prediction_file_with_bmc(
                    predictions_path,
                    ["prediction"],
                    TARGET,
                    "v5.3",
                    benchmark_model=BENCHMARK,
                    benchmark_data=_benchmark(),
                )

            self.assertEqual(read_parquet.call_count, 1)
            self.assertEqual(attach.call_count, 1)
            self.assertEqual(len(cohorts), 3)
            self.assertIs(cohorts[0], cohorts[1])
            self.assertIs(cohorts[1], cohorts[2])
            self.assertEqual(cohorts[0]["id"].tolist(), ["c", "a", "b"])
            self.assertEqual(
                set(summaries), {"corr", "bmc", "bmc_last_200_eras"}
            )

    def test_prediction_target_and_selected_benchmark_must_all_be_finite_numeric(self):
        cases = (
            ("prediction infinite", "prediction", np.inf, None),
            ("prediction nonnumeric", "prediction", "bad", None),
            ("target nan", TARGET, np.nan, None),
            ("target nonnumeric", TARGET, "bad", None),
            ("benchmark infinite", None, None, np.inf),
            ("benchmark nonnumeric", None, None, "bad"),
        )
        for name, column, value, benchmark_value in cases:
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                predictions = _predictions()
                benchmark = _benchmark()
                if column is not None:
                    if isinstance(value, str):
                        predictions[column] = value
                    else:
                        predictions.loc[0, column] = value
                if benchmark_value is not None:
                    if isinstance(benchmark_value, str):
                        benchmark[BENCHMARK] = benchmark[BENCHMARK].astype(object)
                    benchmark.loc[benchmark["id"] == "c", BENCHMARK] = benchmark_value
                predictions_path = Path(tmp) / "predictions.parquet"
                predictions.to_parquet(predictions_path, index=False)
                with self.assertRaisesRegex(ValueError, "finite numeric"):
                    numerai_metrics.summarize_prediction_file_with_bmc(
                        predictions_path,
                        ["prediction"],
                        TARGET,
                        "v5.3",
                        benchmark_model=BENCHMARK,
                        benchmark_data=benchmark,
                    )

    def test_prediction_file_without_id_never_uses_range_index_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            predictions_path = Path(tmp) / "predictions.parquet"
            _predictions().drop(columns="id").to_parquet(
                predictions_path, index=False
            )
            with self.assertRaisesRegex(ValueError, "missing required named columns"):
                numerai_metrics.summarize_prediction_file_with_bmc(
                    predictions_path,
                    ["prediction"],
                    TARGET,
                    "v5.3",
                    benchmark_model=BENCHMARK,
                    benchmark_data=_benchmark(),
                )


class TestPredictionSemantics(unittest.TestCase):
    def test_identity_and_residual_training_targets_have_canonical_semantics(self):
        identity = pipeline.build_prediction_semantics({}, TARGET, "era")
        self.assertEqual(
            identity,
            {
                "schema_version": 1,
                "column": "prediction",
                "artifact_kind": "out_of_fold_validation",
                "producer": "model.predict",
                "training_target": {
                    "column": TARGET,
                    "transform": {"type": "identity"},
                },
                "stored_target": {
                    "column": TARGET,
                    "transform": {"type": "identity"},
                },
                "inverse_target_transform_applied": False,
                "pipeline_postprocess": {"type": "identity"},
                "era_column": "era",
                "fold_column": "cv_fold",
                "fold_index_base": 0,
            },
        )

        transform = {
            "type": "residual_to_benchmark",
            "benchmark_col": BENCHMARK,
            "era_col": "era",
            "per_era": True,
        }
        residual = pipeline.build_prediction_semantics(
            {"target_transform": transform}, TARGET, "era"
        )
        self.assertEqual(residual["training_target"]["transform"], transform)
        self.assertEqual(
            residual["stored_target"]["transform"], {"type": "identity"}
        )
        self.assertFalse(residual["inverse_target_transform_applied"])
        self.assertEqual(residual["producer"], "model.predict")

    def test_nonempty_prediction_transform_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not implemented"):
            pipeline.build_prediction_semantics(
                {"prediction_transform": {"type": "rank"}}, TARGET, "era"
            )

    def test_select_prediction_columns_requires_every_oof_column(self):
        required = ["id", "era", TARGET, "prediction", "cv_fold"]
        complete = _predictions()
        for missing in required:
            with self.subTest(missing), self.assertRaisesRegex(
                ValueError, "missing required columns"
            ):
                pipeline.select_prediction_columns(
                    complete.drop(columns=missing), "id", "era", TARGET
                )
        with self.assertRaisesRegex(ValueError, "column names are required"):
            pipeline.select_prediction_columns(complete, None, "era", TARGET)

    def test_pipeline_writes_identical_json_and_parquet_semantics_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            transform = {
                "type": "residual_to_benchmark",
                "benchmark_col": BENCHMARK,
                "era_col": "era",
                "per_era": True,
            }
            config = {
                "data": {
                    "data_version": "v5.3",
                    "feature_set": "small",
                    "target_col": TARGET,
                    "era_col": "era",
                    "id_col": "id",
                    "benchmark_model": BENCHMARK,
                },
                "model": {
                    "type": "LGBMRegressor",
                    "params": {},
                    "target_transform": transform,
                },
                "training": {
                    "cv": {"enabled": True, "n_splits": 2, "embargo": 0}
                },
                "output": {"output_dir": str(output), "results_name": "semantics"},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            full = pd.DataFrame(
                {
                    "id": ["a", "b", "c", "d"],
                    "era": ["0001", "0001", "0002", "0002"],
                    TARGET: [0.1, 0.4, 0.6, 0.9],
                    "feature": [0.0, 1.0, 2.0, 3.0],
                    BENCHMARK: [0.2, 0.3, 0.7, 0.8],
                }
            )
            oof = full[["id", "era", TARGET]].copy()
            oof["prediction"] = [0.15, 0.35, 0.65, 0.85]
            oof["cv_fold"] = [0, 0, 1, 1]
            cv_meta = {
                "n_splits": 2,
                "embargo": 0,
                "mode": "expanding",
                "min_train_size": 0,
                "folds_used": 2,
                "folds": [],
            }
            corr = pd.DataFrame({"mean": [0.01]}, index=["prediction"])
            bmc = pd.DataFrame(
                {"mean": [0.001], "avg_corr_with_benchmark": [0.2]},
                index=["prediction"],
            )

            with mock.patch.object(pipeline, "NumerAPI"), mock.patch.object(
                pipeline,
                "load_and_prepare_data",
                return_value=(full.drop(columns=BENCHMARK), ["feature"]),
            ), mock.patch.object(
                pipeline,
                "attach_benchmark_models",
                return_value=(full, [BENCHMARK]),
            ), mock.patch.object(
                pipeline,
                "build_oof_predictions",
                return_value=(oof, cv_meta),
            ), mock.patch.object(
                pipeline,
                "summarize_predictions",
                return_value={
                    "corr": corr,
                    "bmc": bmc,
                    "bmc_last_200_eras": bmc,
                },
            ):
                predictions_path, results_path = pipeline.run_training(config_path)

            results = json.loads(results_path.read_text(encoding="utf-8"))
            semantics = results["output"]["prediction_semantics"]
            metadata = pq.ParquetFile(predictions_path).schema_arrow.metadata
            self.assertIsNotNone(metadata)
            self.assertIn(b"pandas", metadata)
            self.assertIn(pipeline.PREDICTION_SEMANTICS_METADATA_KEY, metadata)
            encoded = metadata[pipeline.PREDICTION_SEMANTICS_METADATA_KEY].decode(
                "utf-8"
            )
            self.assertEqual(json.loads(encoded), semantics)
            self.assertEqual(
                encoded,
                json.dumps(semantics, sort_keys=True, separators=(",", ":")),
            )
            stored = pd.read_parquet(predictions_path)
            self.assertEqual(stored[TARGET].tolist(), oof[TARGET].tolist())
            self.assertEqual(stored["prediction"].tolist(), oof["prediction"].tolist())


if __name__ == "__main__":
    unittest.main()
