from __future__ import annotations

import copy
import json
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from agents.code.analysis import evaluate_xerxes20_lgbm_challenger as xerxes


def _metric_summary(
    *,
    bmc_mean: float = 0.002,
    bmc_sharpe: float = 0.30,
    bmc_drawdown: float = 0.10,
    corr_mean: float = 0.015,
    benchmark_similarity: float = 0.50,
    tabm_similarity: float | None = None,
) -> dict:
    result = {
        "era_count": 10,
        "corr": {
            "mean": corr_mean,
            "std": 0.01,
            "sharpe": corr_mean / 0.01,
            "max_drawdown": 0.02,
        },
        "bmc": {
            "mean": bmc_mean,
            "std": 0.01,
            "sharpe": bmc_sharpe,
            "max_drawdown": bmc_drawdown,
        },
        "avg_benchmark_similarity": benchmark_similarity,
    }
    if tabm_similarity is not None:
        result["avg_tabm_similarity"] = tabm_similarity
    return result


def _passing_scout_summaries() -> dict[str, dict]:
    return {
        name: _metric_summary(bmc_mean=0.002 + index * 0.0001)
        for index, name in enumerate(xerxes.SCOUT_NAMES)
    }


def _write_prediction_artifact(
    path: Path,
    frame: pd.DataFrame,
    semantics: dict,
) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[xerxes.PREDICTION_SEMANTICS_METADATA_KEY] = json.dumps(
        semantics, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path)


class ScoutDecisionTests(unittest.TestCase):
    def test_selection_uses_bmc_then_drawdown_depth_and_name(self) -> None:
        summaries = _passing_scout_summaries()
        for metrics in summaries.values():
            metrics["bmc"]["mean"] = 0.003
            metrics["bmc"]["max_drawdown"] = 0.10

        selected, evaluations = xerxes.select_scout_candidate(summaries)

        self.assertEqual(selected, "r1_depth5")
        self.assertTrue(all(row["eligible"] for row in evaluations.values()))

        summaries["r1_depth5"]["bmc"]["max_drawdown"] = 0.11
        selected, _ = xerxes.select_scout_candidate(summaries)
        self.assertEqual(selected, "r1_base_d6_t6000")

        summaries["r1_trees2k"]["bmc"]["mean"] = 0.0031
        selected, _ = xerxes.select_scout_candidate(summaries)
        self.assertEqual(selected, "r1_trees2k")

    def test_no_eligible_candidate_stops_without_selection(self) -> None:
        summaries = _passing_scout_summaries()
        for metrics in summaries.values():
            metrics["bmc"]["mean"] = 0.001

        selected, evaluations = xerxes.select_scout_candidate(summaries)

        self.assertIsNone(selected)
        self.assertFalse(any(row["eligible"] for row in evaluations.values()))

    def test_scout_calibration_thresholds_are_strict(self) -> None:
        boundary = _metric_summary(
            bmc_mean=0.001,
            bmc_sharpe=0.20,
            bmc_drawdown=0.15,
            corr_mean=0.010,
            benchmark_similarity=0.85,
        )

        checks = xerxes.scout_calibration_checks(boundary)

        self.assertFalse(any(checks.values()))

    def test_scout_holdout_thresholds_are_strict(self) -> None:
        boundary = _metric_summary(
            bmc_mean=0.0,
            bmc_sharpe=0.20,
            bmc_drawdown=0.10,
            corr_mean=0.008,
        )

        checks = xerxes.scout_holdout_checks(boundary)

        self.assertFalse(any(checks.values()))


class CheckpointBoundaryTests(unittest.TestCase):
    def test_protocol_and_training_paths_use_distinct_checkpoints(self) -> None:
        calls: list[list[str]] = []

        def fake_git(_root: Path, arguments: list[str]) -> subprocess.CompletedProcess:
            calls.append(list(arguments))
            return subprocess.CompletedProcess(["git", *arguments], 0, "", "")

        training_commit = "1" * 40
        with patch.object(xerxes, "_run_git", side_effect=fake_git):
            xerxes.verify_checkpoint_boundaries(Path("repo"), training_commit)

        diff_calls = [call for call in calls if call[:2] == ["diff", "--quiet"]]
        self.assertEqual(len(diff_calls), 2)
        protocol_diff = next(
            call for call in diff_calls if call[2] == xerxes.PRE_SCORING_COMMIT
        )
        training_diff = next(call for call in diff_calls if call[2] == training_commit)
        self.assertTrue(
            set(xerxes.PROTOCOL_CHECKPOINT_PATHS).issubset(protocol_diff)
        )
        self.assertFalse(
            set(xerxes.TRAINING_CHECKPOINT_PATHS).intersection(protocol_diff)
        )
        self.assertTrue(
            set(xerxes.TRAINING_CHECKPOINT_PATHS).issubset(training_diff)
        )
        self.assertFalse(
            set(xerxes.PROTOCOL_CHECKPOINT_PATHS).intersection(training_diff)
        )
        gpu_runtime_path = (
            "numerai/agents/experiments/xerxes20_lgbm_challenger_v53/"
            "gpu_runtime.json"
        )
        self.assertIn(gpu_runtime_path, xerxes.TRAINING_CHECKPOINT_PATHS)
        self.assertNotIn(gpu_runtime_path, xerxes.PROTOCOL_CHECKPOINT_PATHS)
        for analysis_path in (
            "numerai/agents/code/analysis/evaluate_ender20_hybrid_stability.py",
            "numerai/agents/code/analysis/evaluate_xerxes20_lgbm_challenger.py",
        ):
            self.assertIn(analysis_path, xerxes.TRAINING_CHECKPOINT_PATHS)
            self.assertNotIn(analysis_path, xerxes.PROTOCOL_CHECKPOINT_PATHS)


class GpuRuntimeProvenanceTests(unittest.TestCase):
    @staticmethod
    def _runtime_path() -> Path:
        return (
            Path(xerxes.__file__).resolve().parents[2]
            / "experiments/xerxes20_lgbm_challenger_v53/gpu_runtime.json"
        )

    def test_committed_gpu_runtime_hash_and_schema_are_exact(self) -> None:
        receipt = xerxes.validate_gpu_runtime_receipt(self._runtime_path())

        self.assertEqual(receipt["schema_version"], 1)
        self.assertEqual(receipt["lightgbm"]["version"], "4.7.0")
        self.assertEqual(
            receipt["lightgbm"]["dll"]["sha256"],
            xerxes.EXPECTED_LIGHTGBM_DLL_SHA256,
        )

    def test_gpu_runtime_schema_fails_closed_even_with_mocked_file_hash(self) -> None:
        receipt = json.loads(self._runtime_path().read_text(encoding="utf-8"))
        del receipt["proof"]["wrapper_gpu_fallback_used"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu_runtime.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with patch.object(
                xerxes, "_sha256_file", return_value=xerxes.GPU_RUNTIME_SHA256
            ):
                with self.assertRaisesRegex(
                    xerxes.XerxesEvaluationError, "GPU proof schema"
                ):
                    xerxes.validate_gpu_runtime_receipt(path)

    def test_live_probe_binds_python_lightgbm_and_loaded_dll(self) -> None:
        live = {
            "python_major_minor": [3, 12],
            "python_version": "3.12.13",
            "python_executable": "C:/env/python.exe",
            "lightgbm_version": "4.7.0",
            "dll_path": "C:/env/lib_lightgbm.dll",
            "dll_size_bytes": xerxes.EXPECTED_LIGHTGBM_DLL_SIZE,
            "dll_sha256": xerxes.EXPECTED_LIGHTGBM_DLL_SHA256,
        }
        with patch.object(xerxes, "validate_gpu_runtime_receipt", return_value={}), patch.object(
            xerxes, "_probe_live_gpu_runtime", return_value=live
        ):
            receipt = xerxes.verify_live_gpu_runtime(Path("gpu_runtime.json"))

        self.assertEqual(receipt["runtime_receipt_sha256"], xerxes.GPU_RUNTIME_SHA256)
        self.assertEqual(receipt["dll_sha256"], xerxes.EXPECTED_LIGHTGBM_DLL_SHA256)

        wrong = {**live, "python_major_minor": [3, 13]}
        with patch.object(xerxes, "validate_gpu_runtime_receipt", return_value={}), patch.object(
            xerxes, "_probe_live_gpu_runtime", return_value=wrong
        ):
            with self.assertRaisesRegex(
                xerxes.XerxesEvaluationError, "Python major/minor"
            ):
                xerxes.verify_live_gpu_runtime(Path("gpu_runtime.json"))

class ConfirmationGateTests(unittest.TestCase):
    def _summaries(self) -> dict[str, dict]:
        return {
            "confirmation_calibration": _metric_summary(
                bmc_mean=0.0015,
                bmc_sharpe=0.36,
                bmc_drawdown=0.14,
                corr_mean=0.012,
                benchmark_similarity=0.74,
                tabm_similarity=0.74,
            ),
            "confirmation_holdout": _metric_summary(
                bmc_mean=0.0001,
                bmc_sharpe=0.01,
                bmc_drawdown=0.14,
                corr_mean=0.0001,
                benchmark_similarity=0.8,
                tabm_similarity=0.8,
            ),
            "confirmation_full": _metric_summary(
                bmc_mean=0.0015,
                bmc_sharpe=0.36,
                bmc_drawdown=0.14,
                corr_mean=0.012,
                benchmark_similarity=0.8,
                tabm_similarity=0.8,
            ),
        }

    def test_inclusive_means_pass_at_exact_boundary(self) -> None:
        checks = xerxes.confirmation_promotion_checks(
            self._summaries(), provenance_ok=True
        )

        self.assertTrue(all(checks.values()))

    def test_strict_confirmation_limits_fail_at_exact_boundary(self) -> None:
        summaries = self._summaries()
        summaries["confirmation_calibration"]["bmc"]["sharpe"] = 0.35
        summaries["confirmation_calibration"]["bmc"]["max_drawdown"] = 0.15
        summaries["confirmation_calibration"]["avg_benchmark_similarity"] = 0.75
        summaries["confirmation_calibration"]["avg_tabm_similarity"] = 0.75
        summaries["confirmation_holdout"]["bmc"]["mean"] = 0.0
        summaries["confirmation_holdout"]["bmc"]["max_drawdown"] = 0.15
        summaries["confirmation_holdout"]["corr"]["mean"] = 0.0
        summaries["confirmation_full"]["bmc"]["sharpe"] = 0.35
        summaries["confirmation_full"]["bmc"]["max_drawdown"] = 0.15

        checks = xerxes.confirmation_promotion_checks(
            summaries, provenance_ok=True
        )

        strict_names = {
            "calibration_bmc_sharpe",
            "calibration_bmc_max_drawdown",
            "calibration_benchmark_similarity",
            "calibration_tabm_similarity",
            "holdout_bmc_mean",
            "holdout_bmc_max_drawdown",
            "holdout_corr_mean",
            "full_bmc_sharpe",
            "full_bmc_max_drawdown",
        }
        self.assertTrue(all(not checks[name] for name in strict_names))

    def test_confirmation_config_rejects_unlisted_changes(self) -> None:
        path = (
            Path(xerxes.__file__).resolve().parents[2]
            / "experiments/xerxes20_lgbm_challenger_v53/configs/r1_depth5.py"
        )
        base = runpy.run_path(str(path))["CONFIG"]
        confirmation = copy.deepcopy(base)
        confirmation["data"].pop("full_data_path")
        confirmation["data"].pop("benchmark_data_path")
        confirmation["data"]["disk_feature_store_path"] = (
            "v5.3/target_xerxes_20_feature_store"
        )
        confirmation["data"]["embargo_eras"] = 52
        confirmation["model"]["prediction_batch_size"] = 65_536
        confirmation["training"]["data_mode"] = "disk_feature_store"
        confirmation["training"]["cv"]["embargo"] = 52
        confirmation["output"]["results_name"] = "r1_depth5_confirmation"

        xerxes.validate_confirmation_config(
            "r1_depth5", base, confirmation
        )

        confirmation["preprocessing"]["missing_value"] = 1.0
        with self.assertRaisesRegex(
            xerxes.XerxesEvaluationError, "outside authorized changes"
        ):
            xerxes.validate_confirmation_config(
                "r1_depth5", base, confirmation
            )


class ArtifactValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = pd.DataFrame(
            {
                xerxes.ID_COLUMN: ["a", "b", "c", "d"],
                xerxes.ERA_COLUMN: ["0001", "0001", "0002", "0002"],
                xerxes.XERXES_TARGET: [0.1, 0.2, 0.3, 0.4],
                xerxes.ENDER_TARGET: [0.5, 0.6, 0.7, 0.8],
                xerxes.BENCHMARK_COLUMN: [0.2, 0.4, 0.6, 0.8],
                xerxes.FOLD_COLUMN: [1, 1, 2, 2],
            }
        )
        self.semantics = {
            "schema_version": 1,
            "training_target": {
                "column": xerxes.XERXES_TARGET,
                "transform": {"type": "identity"},
            },
            "stored_target": {
                "column": xerxes.XERXES_TARGET,
                "transform": {"type": "identity"},
            },
        }

    def _artifact(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                xerxes.ID_COLUMN: ["d", "b", "a", "c"],
                xerxes.ERA_COLUMN: ["0002", "0001", "0001", "0002"],
                xerxes.XERXES_TARGET: [0.4, 0.2, 0.1, 0.3],
                xerxes.PREDICTION_COLUMN: [40.0, 20.0, 10.0, 30.0],
                xerxes.FOLD_COLUMN: [2, 1, 1, 2],
            }
        )

    def test_prediction_alignment_is_by_id_not_file_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.parquet"
            _write_prediction_artifact(path, self._artifact(), self.semantics)

            raw = xerxes.validate_prediction_artifact(
                path, self.expected, self.semantics
            )

        np.testing.assert_allclose(raw, [10.0, 20.0, 30.0, 40.0])

    def test_target_or_fold_mismatch_fails_closed(self) -> None:
        artifact = self._artifact()
        artifact.loc[artifact[xerxes.ID_COLUMN] == "b", xerxes.XERXES_TARGET] = 9.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.parquet"
            _write_prediction_artifact(path, artifact, self.semantics)

            with self.assertRaisesRegex(
                xerxes.XerxesEvaluationError, "stored target_xerxes_20 differs"
            ):
                xerxes.validate_prediction_artifact(
                    path, self.expected, self.semantics
                )

        artifact = self._artifact()
        artifact.loc[artifact[xerxes.ID_COLUMN] == "c", xerxes.FOLD_COLUMN] = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.parquet"
            _write_prediction_artifact(path, artifact, self.semantics)

            with self.assertRaisesRegex(
                xerxes.XerxesEvaluationError, "fold assignments differ"
            ):
                xerxes.validate_prediction_artifact(
                    path, self.expected, self.semantics
                )

    def test_result_receipt_requires_gpu_and_exact_disk_batch_diagnostics(self) -> None:
        config = {
            "data": {
                "data_version": "v5.3",
                "feature_set": "medium",
                "target_col": xerxes.XERXES_TARGET,
                "era_col": xerxes.ERA_COLUMN,
                "id_col": xerxes.ID_COLUMN,
                "benchmark_model": xerxes.BENCHMARK_COLUMN,
                "require_benchmark_coverage": True,
                "embargo_eras": 52,
                "disk_feature_store_path": "v5.3/target_xerxes_20_feature_store",
            },
            "model": {
                "type": "LGBMRegressor",
                "x_groups": ["features", "era", "benchmark_models"],
                "prediction_batch_size": 2,
                "params": {"device_type": "gpu", "random_state": 1337},
            },
            "training": {
                "max_train_samples": 500_000,
                "sample_seed": 1337,
                "data_mode": "disk_feature_store",
                "cv": {
                    "enabled": True,
                    "n_splits": 5,
                    "embargo": 52,
                    "mode": "expanding",
                    "min_train_size": 0,
                },
            },
            "preprocessing": {
                "nan_missing_all_twos": False,
                "missing_value": 2.0,
            },
            "output": {"results_name": "confirmation"},
        }
        expected = xerxes.ExpectedCohort(
            frame=self.expected.iloc[:2].copy(),
            full_rows=2,
            full_eras=1,
            eras=("0001",),
            folds=(
                {
                    "fold": 1,
                    "train_eras": 1,
                    "val_eras": 1,
                    "train_rows": 1,
                    "val_rows": 2,
                },
            ),
        )
        store_metadata = {
            "generation_id": "generation",
            "row_count": 2,
            "feature_count": 780,
            "feature_order_sha256": "order",
            "features": {
                "filename": "features.bin",
                "size_bytes": 10,
                "sha256": "features",
            },
            "manifest": {
                "filename": "manifest.parquet",
                "size_bytes": 20,
                "sha256": "manifest",
            },
        }
        semantics = xerxes._expected_semantics(config)
        diagnostics = {
            "effective_device_type": "gpu",
            "gpu_fallback_used": False,
            "data_mode": "disk_feature_store",
            "disk_train_rows": 1,
            "disk_validation_rows": 2,
            "disk_prediction_batches": 1,
            "disk_prediction_batch_size": 2,
            "disk_rows_per_epoch": [1],
            "disk_batches_per_epoch": [1],
        }
        result = {
            "model": xerxes._expected_model_payload(config),
            "preprocessing": config["preprocessing"],
            "data": {
                "data_version": "v5.3",
                "feature_set": "medium",
                "target": xerxes.XERXES_TARGET,
                "full_rows": 2,
                "full_eras": 1,
                "oof_rows": 2,
                "oof_eras": 1,
                "embargo_eras": 52,
                "require_benchmark_coverage": True,
                "data_mode": "disk_feature_store",
                "disk_feature_store": {
                    "generation_id": "generation",
                    "row_count": 2,
                    "feature_count": 780,
                    "feature_bytes": 10,
                    "manifest_bytes": 20,
                    "feature_order_sha256": "order",
                    "feature_sha256": "features",
                    "manifest_sha256": "manifest",
                    "feature_path": "features.bin",
                    "manifest_path": "manifest.parquet",
                },
            },
            "benchmark": {
                "model": xerxes.BENCHMARK_COLUMN,
                "file": "manifest.parquet",
            },
            "training": {
                "data_sampling": {
                    "max_train_samples": 500_000,
                    "sample_seed": 1337,
                },
                "data_mode": "disk_feature_store",
                "cv": config["training"]["cv"],
            },
            "cv": {
                "n_splits": 5,
                "embargo": 52,
                "mode": "expanding",
                "min_train_size": 0,
                "folds_used": 1,
                "folds": [
                    {
                        **expected.folds[0],
                        "model_diagnostics": diagnostics,
                    }
                ],
            },
            "output": {
                "predictions_file": "confirmation.parquet",
                "prediction_semantics": semantics,
            },
            "metrics": {"corr": {}, "bmc": {}, "bmc_last_200_eras": {}},
        }
        run = xerxes.RunPaths(
            "confirmation",
            Path("confirmation.py"),
            Path("confirmation.json"),
            Path("confirmation.parquet"),
        )

        xerxes.validate_result_json(
            run, result, config, expected, store_metadata=store_metadata
        )

        diagnostics["gpu_fallback_used"] = True
        with self.assertRaisesRegex(
            xerxes.XerxesEvaluationError, "GPU fallback receipt"
        ):
            xerxes.validate_result_json(
                run, result, config, expected, store_metadata=store_metadata
            )

    def test_semantics_mismatch_fails_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.parquet"
            altered = copy.deepcopy(self.semantics)
            altered["training_target"]["column"] = xerxes.ENDER_TARGET
            _write_prediction_artifact(path, self._artifact(), altered)

            with self.assertRaisesRegex(
                xerxes.XerxesEvaluationError, "Parquet semantics"
            ):
                xerxes.validate_prediction_artifact(
                    path, self.expected, self.semantics
                )


class SimilarityTests(unittest.TestCase):
    def test_symmetric_spearman_uses_average_tie_ranks_per_era(self) -> None:
        frame = pd.DataFrame(
            {
                xerxes.ERA_COLUMN: ["0001"] * 4 + ["0002"] * 4,
                "reference": [1, 2, 2, 4, 10, 30, 20, 40],
                "same": [10, 20, 20, 40, 100, 300, 200, 400],
                "reverse": [4, 2, 2, 1, 40, 20, 30, 10],
            }
        )

        forward = xerxes.symmetric_per_era_similarity(
            frame, ["same", "reverse"], "reference"
        )
        reverse = xerxes.symmetric_per_era_similarity(
            frame, ["reference"], "same"
        )

        np.testing.assert_allclose(forward["same"], [1.0, 1.0])
        np.testing.assert_allclose(forward["reverse"], [-1.0, -1.0])
        np.testing.assert_allclose(reverse["reference"], forward["same"])


class OutputIntegrityTests(unittest.TestCase):
    def test_content_addressed_output_is_immutable_and_stable_pointer_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final = root / "immutable.csv"
            first = root / "first.tmp"
            first.write_bytes(b"one\n")
            xerxes._install_content_addressed(first, final)
            self.assertEqual(final.read_bytes(), b"one\n")

            same = root / "same.tmp"
            same.write_bytes(b"one\n")
            xerxes._install_content_addressed(same, final)
            self.assertFalse(same.exists())

            different = root / "different.tmp"
            different.write_bytes(b"two\n")
            with self.assertRaisesRegex(
                xerxes.XerxesEvaluationError, "content-addressed output differs"
            ):
                xerxes._install_content_addressed(different, final)

            pointer = root / "result.json"
            xerxes._atomic_replace_bytes(pointer, b'{"generation":1}\n')
            xerxes._atomic_replace_bytes(pointer, b'{"generation":2}\n')
            self.assertEqual(pointer.read_bytes(), b'{"generation":2}\n')


if __name__ == "__main__":
    unittest.main()
