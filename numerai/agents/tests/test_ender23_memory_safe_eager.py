from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from agents.code.modeling.utils import pipeline
from agents.code.modeling.utils.data import attach_benchmark_models, load_full_data
from agents.code.modeling.utils.numerai_cv import era_cv_splits
from agents.code.modeling.utils.target_transforms import apply_target_transform


TARGET = "target_ender_20"
BENCHMARK = "v53_lgbm_ender20"


class TestEnder23MemorySafeEager(unittest.TestCase):
    def test_full_parquet_applies_frozen_eras_during_the_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            full_path = Path(tmp) / "full.parquet"
            pd.DataFrame(
                {
                    "id": ["d", "a", "c", "b"],
                    "era": ["0004", "0001", "0003", "0002"],
                    TARGET: pd.Series([0.4, 0.1, 0.3, 0.2], dtype="float32"),
                    "feature": [4, 1, 3, 2],
                }
            ).to_parquet(full_path, index=False)

            loaded = load_full_data(
                mock.Mock(),
                "v5.3",
                ["feature"],
                "era",
                TARGET,
                "id",
                full_data_path=full_path,
                allowed_eras=("0002", "0004"),
            )

        self.assertEqual(loaded["id"].tolist(), ["d", "b"])
        self.assertEqual(loaded["era"].tolist(), ["0004", "0002"])
        self.assertEqual(loaded.columns.tolist(), ["era", TARGET, "feature", "id"])
        self.assertEqual(str(loaded[TARGET].dtype), "float32")

    def test_memory_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            full_path = Path(tmp) / "full.parquet"
            pd.DataFrame(
                {"id": ["a"], "era": ["0001"], TARGET: [0.1], "feature": [1]}
            ).to_parquet(full_path, index=False)
            with mock.patch(
                "agents.code.modeling.utils.data.pd.read_parquet",
                side_effect=MemoryError("synthetic allocation failure"),
            ) as read_mock:
                with self.assertRaisesRegex(MemoryError, "synthetic allocation"):
                    load_full_data(
                        mock.Mock(),
                        "v5.3",
                        ["feature"],
                        "era",
                        TARGET,
                        "id",
                        full_data_path=full_path,
                        allowed_eras=("0001",),
                    )
        self.assertEqual(read_mock.call_count, 1)

    def test_filtered_read_rejects_synthetic_positional_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            full_path = Path(tmp) / "full.parquet"
            pd.DataFrame(
                {"era": ["0001", "0002"], TARGET: [0.1, 0.2], "feature": [1, 2]}
            ).to_parquet(full_path, index=False)
            with self.assertRaisesRegex(ValueError, "stable ids"):
                load_full_data(
                    mock.Mock(),
                    "v5.3",
                    ["feature"],
                    "era",
                    TARGET,
                    "id",
                    full_data_path=full_path,
                    allowed_eras=("0002",),
                )

    def test_required_benchmark_uses_one_inner_join_and_preserves_left_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark_path = Path(tmp) / "benchmark.parquet"
            full = pd.DataFrame(
                {
                    "id": ["d", "c", "a", "b"],
                    "era": ["0002", "0001", "0001", "0002"],
                    "feature": [4, 3, 1, 2],
                }
            )
            pd.DataFrame(
                {
                    "id": ["a", "c", "d"],
                    "era": ["0001", "0001", "0002"],
                    BENCHMARK: [0.1, np.nan, 0.4],
                    "other_benchmark": [0.2, 0.3, 0.5],
                }
            ).to_parquet(benchmark_path, index=False)

            joined, columns = attach_benchmark_models(
                full,
                mock.Mock(),
                "v5.3",
                benchmark_path,
                "era",
                "id",
                required_benchmark_model=BENCHMARK,
            )

        self.assertEqual(joined["id"].tolist(), ["d", "a"])
        self.assertEqual(joined["feature"].tolist(), [4, 1])
        self.assertEqual(columns, [BENCHMARK, "other_benchmark"])
        self.assertFalse(joined[BENCHMARK].isna().any())

    def test_required_benchmark_name_must_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark_path = Path(tmp) / "benchmark.parquet"
            full = pd.DataFrame({"id": ["a"], "era": ["0001"], "feature": [1]})
            pd.DataFrame(
                {"id": ["a"], "era": ["0001"], "other": [0.2]}
            ).to_parquet(benchmark_path, index=False)
            with self.assertRaisesRegex(ValueError, "Required benchmark"):
                attach_benchmark_models(
                    full,
                    mock.Mock(),
                    "v5.3",
                    benchmark_path,
                    "era",
                    "id",
                    required_benchmark_model=BENCHMARK,
                )

    def test_optional_coverage_preserves_legacy_left_join_and_missing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark_path = Path(tmp) / "benchmark.parquet"
            full = pd.DataFrame(
                {
                    "id": ["d", "a", "b"],
                    "era": ["0002", "0001", "0002"],
                    "feature": [4, 1, 2],
                }
            )
            pd.DataFrame(
                {
                    "id": ["a", "d"],
                    "era": ["0001", "0002"],
                    BENCHMARK: [0.1, np.nan],
                }
            ).to_parquet(benchmark_path, index=False)

            joined, _ = attach_benchmark_models(
                full,
                mock.Mock(),
                "v5.3",
                benchmark_path,
                "era",
                "id",
            )

        self.assertEqual(joined["id"].tolist(), ["d", "a", "b"])
        self.assertTrue(pd.isna(joined.loc[0, BENCHMARK]))
        self.assertEqual(joined.loc[1, BENCHMARK], 0.1)
        self.assertTrue(pd.isna(joined.loc[2, BENCHMARK]))

    def test_repaired_path_is_semantically_equal_to_legacy_path(self):
        allowlist = ("0001", "0003", "0004", "0005")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_path = root / "full.parquet"
            benchmark_path = root / "benchmark.parquet"
            source = pd.DataFrame(
                {
                    "id": ["e", "a", "h", "b", "f", "c", "g", "d", "x"],
                    "era": ["0005", "0001", "0004", "0001", "0003", "0003", "0004", "0005", "0002"],
                    TARGET: pd.Series([0.7, 0.1, 0.6, 0.2, 0.4, 0.3, 0.5, 0.8, 0.9], dtype="float32"),
                    "feature_a": [7, 1, 6, 2, 4, 3, 5, 8, 9],
                    "feature_b": [17, 11, 16, 12, 14, 13, 15, 18, 19],
                }
            )
            source.to_parquet(full_path, index=False)
            pd.DataFrame(
                {
                    "id": ["a", "b", "c", "d", "e", "f", "g", "h", "x"],
                    "era": ["0001", "0001", "0003", "0005", "0005", "0003", "0004", "0004", "0002"],
                    BENCHMARK: [0.11, 0.12, 0.31, 0.51, 0.52, 0.32, 0.41, 0.42, 0.21],
                }
            ).to_parquet(benchmark_path, index=False)

            legacy_loaded = load_full_data(
                mock.Mock(), "v5.3", ["feature_a", "feature_b"], "era", TARGET, "id", full_data_path=full_path
            )
            legacy_joined, _ = attach_benchmark_models(
                legacy_loaded, mock.Mock(), "v5.3", benchmark_path, "era", "id"
            )
            legacy = legacy_joined.loc[
                legacy_joined[BENCHMARK].notna()
                & legacy_joined["era"].astype(str).isin(set(allowlist))
            ].reset_index(drop=True)

            repaired = load_full_data(
                mock.Mock(), "v5.3", ["feature_a", "feature_b"], "era", TARGET, "id", full_data_path=full_path, allowed_eras=allowlist
            )
            repaired = pipeline._filter_to_era_allowlist(repaired, "era", allowlist)
            repaired, _ = attach_benchmark_models(
                repaired, mock.Mock(), "v5.3", benchmark_path, "era", "id", required_benchmark_model=BENCHMARK
            )
            repaired = pipeline._filter_to_era_allowlist(repaired, "era", allowlist).reset_index(drop=True)

        pd.testing.assert_frame_equal(repaired, legacy, check_dtype=True)
        self.assertEqual(
            era_cv_splits(repaired["era"], n_splits=2, embargo=0, min_train_size=0),
            era_cv_splits(legacy["era"], n_splits=2, embargo=0, min_train_size=0),
        )
        transform = {
            "type": "residual_to_benchmark",
            "benchmark_col": BENCHMARK,
            "era_col": "era",
            "per_era": True,
            "fit_intercept": True,
            "proportion": 1.0,
        }
        repaired_y = apply_target_transform(repaired[TARGET], repaired, transform)
        legacy_y = apply_target_transform(legacy[TARGET], legacy, transform)
        np.testing.assert_array_equal(repaired_y.to_numpy(), legacy_y.to_numpy())

    def test_missing_coverage_for_an_entire_allowed_era_fails_after_join(self):
        full = pd.DataFrame(
            {"id": ["a", "b"], "era": ["0001", "0002"], "feature": [1, 2]}
        )
        with tempfile.TemporaryDirectory() as tmp:
            benchmark_path = Path(tmp) / "benchmark.parquet"
            pd.DataFrame(
                {"id": ["a"], "era": ["0001"], BENCHMARK: [0.1]}
            ).to_parquet(benchmark_path, index=False)
            joined, _ = attach_benchmark_models(
                full, mock.Mock(), "v5.3", benchmark_path, "era", "id", required_benchmark_model=BENCHMARK
            )
        with self.assertRaisesRegex(ValueError, "absent from modeling data"):
            pipeline._filter_to_era_allowlist(joined, "era", ("0001", "0002"))

    def test_exact_allowlist_reuses_frame_but_subset_remains_exact_copy(self):
        full = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "era": ["0001", "0002", "0003"],
                "feature": [1, 2, 3],
            }
        )
        exact = pipeline._filter_to_era_allowlist(
            full, "era", ("0001", "0002", "0003")
        )
        self.assertIs(exact, full)

        subset = pipeline._filter_to_era_allowlist(
            full, "era", ("0001", "0003")
        )
        self.assertIsNot(subset, full)
        self.assertEqual(subset["id"].tolist(), ["a", "c"])
        subset.loc[subset.index[0], "feature"] = 99
        self.assertEqual(full.loc[0, "feature"], 1)

    def test_pipeline_forwards_frozen_required_benchmark_before_oof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            config = {
                "data": {
                    "data_version": "v5.3",
                    "feature_set": "all",
                    "target_col": TARGET,
                    "era_col": "era",
                    "id_col": "id",
                    "benchmark_model": BENCHMARK,
                    "require_benchmark_coverage": True,
                },
                "model": {
                    "type": "LGBMRegressor",
                    "x_groups": ["features", "benchmark_models"],
                    "params": {},
                },
                "training": {
                    "cv": {"enabled": True, "n_splits": 2, "embargo": 0}
                },
                "output": {"output_dir": str(output), "results_name": "memory_safe"},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            full = pd.DataFrame(
                {
                    "id": ["a", "b", "c", "d"],
                    "era": ["0001", "0001", "0002", "0002"],
                    TARGET: [0.1, 0.2, 0.3, 0.4],
                    "feature": [1, 2, 3, 4],
                    BENCHMARK: [0.2, 0.3, 0.4, 0.5],
                }
            )
            oof = full[["id", "era", TARGET]].copy()
            oof["prediction"] = [0.15, 0.25, 0.35, 0.45]
            oof["cv_fold"] = [0, 0, 1, 1]
            cv_meta = {
                "n_splits": 2,
                "embargo": 0,
                "mode": "expanding",
                "min_train_size": 0,
                "folds_used": 2,
                "folds": [],
            }
            summary = pd.DataFrame(
                {"mean": [0.01], "avg_corr_with_benchmark": [0.1]},
                index=["prediction"],
            )

            def attach(frame, *_args, **kwargs):
                self.assertEqual(kwargs["required_benchmark_model"], BENCHMARK)
                self.assertEqual(frame.columns.tolist(), ["id", "era", TARGET, "feature"])
                return full, [BENCHMARK]

            def load_data(*_args, **kwargs):
                self.assertIsNone(kwargs["allowed_eras"])
                return full.drop(columns=BENCHMARK), ["feature"]

            with mock.patch.object(pipeline, "NumerAPI"), mock.patch.object(
                pipeline,
                "load_and_prepare_data",
                side_effect=load_data,
            ), mock.patch.object(
                pipeline, "attach_benchmark_models", side_effect=attach
            ) as attach_mock, mock.patch.object(
                pipeline, "build_oof_predictions", return_value=(oof, cv_meta)
            ), mock.patch.object(
                pipeline,
                "summarize_predictions",
                return_value={
                    "corr": summary,
                    "bmc": summary,
                    "bmc_last_200_eras": summary,
                },
            ):
                pipeline.run_training(config_path)

            attach_mock.assert_called_once()

    def test_pipeline_rejects_an_uncovered_allowed_era_before_oof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowlist = root / "eras.json"
            allowlist.write_text('["0001", "0002"]\n', encoding="utf-8")
            config = {
                "data": {
                    "data_version": "v5.3",
                    "feature_set": "all",
                    "target_col": TARGET,
                    "era_col": "era",
                    "id_col": "id",
                    "benchmark_model": BENCHMARK,
                    "require_benchmark_coverage": True,
                    "era_allowlist_path": "eras.json",
                },
                "model": {
                    "type": "LGBMRegressor",
                    "x_groups": ["features", "benchmark_models"],
                    "params": {},
                },
                "training": {
                    "cv": {"enabled": True, "n_splits": 2, "embargo": 0}
                },
                "output": {
                    "output_dir": str(root / "output"),
                    "results_name": "coverage_gap",
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            loaded = pd.DataFrame(
                {
                    "id": ["a", "b"],
                    "era": ["0001", "0002"],
                    TARGET: [0.1, 0.2],
                    "feature": [1, 2],
                }
            )
            covered = loaded.loc[loaded["era"] == "0001"].copy()
            covered[BENCHMARK] = 0.25

            with mock.patch.object(pipeline, "REPO_DIR", root), mock.patch.object(
                pipeline, "NumerAPI"
            ), mock.patch.object(
                pipeline,
                "load_and_prepare_data",
                return_value=(loaded, ["feature"]),
            ), mock.patch.object(
                pipeline,
                "attach_benchmark_models",
                return_value=(covered, [BENCHMARK]),
            ), mock.patch.object(
                pipeline, "build_oof_predictions"
            ) as build_oof:
                with self.assertRaisesRegex(ValueError, "absent from modeling data"):
                    pipeline.run_training(config_path)

            build_oof.assert_not_called()


if __name__ == "__main__":
    unittest.main()
