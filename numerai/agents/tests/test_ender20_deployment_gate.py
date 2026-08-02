from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from agents.code.analysis import evaluate_ender20_deployment_gate as gate


def _semantics() -> dict:
    return {
        "schema_version": 1,
        "column": "prediction",
        "artifact_kind": "out_of_fold_validation",
        "producer": "model.predict",
        "training_target": {
            "column": gate.TARGET_COLUMN,
            "transform": {
                "type": "residual_to_benchmark",
                "benchmark_col": gate.BENCHMARK_COLUMN,
                "era_col": "era",
                "fit_intercept": True,
                "per_era": True,
            },
        },
        "stored_target": {
            "column": gate.TARGET_COLUMN,
            "transform": {"type": "identity"},
        },
        "inverse_target_transform_applied": False,
        "pipeline_postprocess": {"type": "identity"},
        "era_column": "era",
        "fold_column": "cv_fold",
        "fold_index_base": 0,
    }


def _expected() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "era": ["0002", "0002", "0002", "0003", "0003"],
            gate.TARGET_COLUMN: [0.1, 0.4, 0.8, 0.2, 0.9],
            gate.BENCHMARK_COLUMN: [0.2, 0.5, 0.7, 0.3, 0.8],
            "cv_fold": [1, 1, 1, 2, 2],
        }
    )


def _write_prediction(path: Path, frame: pd.DataFrame, semantics: dict) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[gate.PREDICTION_SEMANTICS_METADATA_KEY] = gate._canonical_json(
        semantics
    ).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path)


def _passing_metrics() -> dict:
    return {
        "corr": {
            "mean": 0.012,
            "std": 0.01,
            "sharpe": 1.2,
            "max_drawdown": 0.02,
        },
        "bmc": {
            "mean": 0.004,
            "std": 0.008,
            "sharpe": 0.5,
            "max_drawdown": 0.04,
            "avg_corr_with_benchmark": 0.1,
        },
        "bmc_last_200_eras": {
            "mean": 0.003,
            "std": 0.009,
            "sharpe": 0.33,
            "max_drawdown": 0.04,
            "avg_corr_with_benchmark": 0.1,
        },
    }


class TestFrozenRankTransform(unittest.TestCase):
    def test_average_percentile_rank_is_exact_with_ties_and_era_order(self):
        predictions = pd.Series([0.2, 0.2, 0.8, 2.0, 1.0])
        eras = pd.Series(["0002", "0002", "0002", "0001", "0001"])
        ranked = gate.rank_predictions_exact(predictions, eras)
        np.testing.assert_array_equal(ranked, [0.5, 0.5, 1.0, 1.0, 0.5])

    def test_nonfinite_raw_prediction_is_rejected(self):
        with self.assertRaisesRegex(gate.GateEvaluationError, "non-finite"):
            gate.rank_predictions_exact(
                pd.Series([0.1, np.inf]), pd.Series(["0001", "0001"])
            )


class TestFrozenSourceIntegrity(unittest.TestCase):
    def test_manifest_anchor_rejects_any_manifest_edit(self):
        repo_root = Path(__file__).resolve().parents[2]
        source = repo_root / gate.DEFAULT_SOURCE_MANIFEST
        with tempfile.TemporaryDirectory() as tmp:
            edited = Path(tmp) / "gate_source_manifest.json"
            edited.write_bytes(source.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                gate.GateEvaluationError, "predeclared anchor"
            ):
                gate.verify_frozen_source(repo_root, edited)

    def test_listed_source_hash_mismatch_is_rejected(self):
        repo_root = Path(__file__).resolve().parents[2]
        source = repo_root / gate.DEFAULT_SOURCE_MANIFEST
        manifest = json.loads(source.read_text(encoding="utf-8"))
        first_file = sorted(manifest["files"])[0]
        manifest["files"][first_file] = "0" * 64
        with tempfile.TemporaryDirectory() as tmp:
            edited = Path(tmp) / "gate_source_manifest.json"
            edited.write_text(json.dumps(manifest), encoding="utf-8")
            edited_sha = gate._sha256_file(edited)
            with mock.patch.object(gate, "SOURCE_MANIFEST_SHA256", edited_sha):
                with self.assertRaisesRegex(
                    gate.GateEvaluationError, "Frozen source hash mismatch"
                ):
                    gate.verify_frozen_source(repo_root, edited)


class TestInternalRecentEraSplit(unittest.TestCase):
    def test_uncapped_recent_era_split_accounts_for_train_embargo_and_validation(self):
        outer_eras = np.repeat([f"{era:04d}" for era in range(1, 11)], 3)
        counts = gate.derive_internal_split_counts(
            outer_eras,
            sample_seed=7,
            max_train_samples=100,
            val_fraction=0.2,
            internal_val_embargo=2,
        )
        self.assertEqual(
            counts,
            {
                "outer_sample_rows": 30,
                "sampled_eras": 10,
                "internal_train_eras": 6,
                "internal_validation_eras": 2,
                "internal_embargo_eras": 2,
                "disk_train_rows": 18,
                "disk_validation_rows": 6,
                "internal_embargo_rows": 6,
            },
        )

    def test_capped_outer_sampling_is_recreated_from_fresh_seeded_generator(self):
        outer_eras = np.concatenate(
            [np.repeat(f"{era:04d}", era) for era in range(1, 13)]
        )
        counts = gate.derive_internal_split_counts(
            outer_eras,
            sample_seed=7,
            max_train_samples=40,
            val_fraction=0.25,
            internal_val_embargo=2,
        )
        self.assertEqual(
            counts,
            {
                "outer_sample_rows": 40,
                "sampled_eras": 11,
                "internal_train_eras": 7,
                "internal_validation_eras": 2,
                "internal_embargo_eras": 2,
                "disk_train_rows": 19,
                "disk_validation_rows": 9,
                "internal_embargo_rows": 12,
            },
        )
        self.assertEqual(
            counts,
            gate.derive_internal_split_counts(
                outer_eras,
                sample_seed=7,
                max_train_samples=40,
                val_fraction=0.25,
                internal_val_embargo=2,
            ),
        )


class TestPredictionArtifactValidation(unittest.TestCase):
    def test_exact_artifact_passes_and_returns_frozen_ranks(self):
        expected = _expected()
        artifact = expected.drop(columns=gate.BENCHMARK_COLUMN).copy()
        artifact.insert(3, "prediction", [0.2, 0.2, 0.8, 0.9, 0.1])
        artifact = artifact[["id", "era", gate.TARGET_COLUMN, "prediction", "cv_fold"]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.parquet"
            _write_prediction(path, artifact, _semantics())
            raw, ranked, diagnostics = gate.validate_prediction_artifact(
                path, expected, _semantics()
            )
        np.testing.assert_array_equal(raw, [0.2, 0.2, 0.8, 0.9, 0.1])
        np.testing.assert_array_equal(ranked, [0.5, 0.5, 1.0, 1.0, 0.5])
        self.assertTrue(diagnostics["unique_ids"])

    def test_duplicate_ids_fail_closed_before_scoring(self):
        expected = _expected()
        artifact = expected.drop(columns=gate.BENCHMARK_COLUMN).copy()
        artifact["prediction"] = np.linspace(0.1, 0.9, len(artifact))
        artifact.loc[1, "id"] = artifact.loc[0, "id"]
        artifact = artifact[["id", "era", gate.TARGET_COLUMN, "prediction", "cv_fold"]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.parquet"
            _write_prediction(path, artifact, _semantics())
            with self.assertRaisesRegex(gate.GateEvaluationError, "duplicate ids"):
                gate.validate_prediction_artifact(path, expected, _semantics())

    def test_alignment_target_fold_and_semantics_mutations_are_rejected(self):
        expected = _expected()
        base = expected.drop(columns=gate.BENCHMARK_COLUMN).copy()
        base["prediction"] = np.linspace(0.1, 0.9, len(base))
        base = base[["id", "era", gate.TARGET_COLUMN, "prediction", "cv_fold"]]
        cases = []
        wrong_id = base.copy()
        wrong_id.loc[0, "id"] = "wrong"
        cases.append(("id", wrong_id, _semantics(), "ids or row order"))
        wrong_era = base.copy()
        wrong_era.loc[0, "era"] = "9999"
        cases.append(("era", wrong_era, _semantics(), "eras do not exactly align"))
        wrong_target = base.copy()
        wrong_target.loc[0, gate.TARGET_COLUMN] += 0.01
        cases.append(("target", wrong_target, _semantics(), "targets do not exactly match"))
        wrong_fold = base.copy()
        wrong_fold.loc[0, "cv_fold"] = 4
        cases.append(("fold", wrong_fold, _semantics(), "CV folds"))
        wrong_semantics = _semantics()
        wrong_semantics["inverse_target_transform_applied"] = True
        cases.append(("semantics", base, wrong_semantics, "semantics metadata"))
        for name, artifact, semantics, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "predictions.parquet"
                _write_prediction(path, artifact, semantics)
                with self.assertRaisesRegex(gate.GateEvaluationError, message):
                    gate.validate_prediction_artifact(path, expected, _semantics())


class TestQualityThresholds(unittest.TestCase):
    def test_three_run_and_ensemble_pass(self):
        runs = {name: _passing_metrics() for name in gate.RUN_SPECS}
        result = gate.evaluate_quality_thresholds(runs, _passing_metrics())
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(check["passed"] for check in result["checks"]))
        self.assertEqual(result["aggregate"]["median_bmc_sharpe"], 0.5)

    def test_strict_boundary_and_ensemble_failure_are_not_promoted(self):
        runs = {name: _passing_metrics() for name in gate.RUN_SPECS}
        first = next(iter(gate.RUN_SPECS))
        runs[first] = json.loads(json.dumps(_passing_metrics()))
        runs[first]["bmc"]["sharpe"] = 0.25
        ensemble = json.loads(json.dumps(_passing_metrics()))
        ensemble["bmc_last_200_eras"]["mean"] = 0.0
        result = gate.evaluate_quality_thresholds(runs, ensemble)
        self.assertEqual(result["status"], "FAIL")
        failures = {
            (check["scope"], check["name"])
            for check in result["checks"]
            if not check["passed"]
        }
        self.assertIn((first, "bmc_sharpe"), failures)
        self.assertIn(("rank_mean_ensemble", "positive_last_200_bmc"), failures)

    def test_corr_range_is_inclusive_but_other_thresholds_are_strict(self):
        runs = {name: json.loads(json.dumps(_passing_metrics())) for name in gate.RUN_SPECS}
        values = [0.005, 0.02, 0.04]
        for metrics, corr in zip(runs.values(), values, strict=True):
            metrics["corr"]["mean"] = corr
        result = gate.evaluate_quality_thresholds(runs, _passing_metrics())
        self.assertEqual(result["status"], "PASS")


class TestFailClosedWrapper(unittest.TestCase):
    def test_incomplete_matrix_returns_machine_readable_failure(self):
        report = gate.evaluate_gate_fail_closed([], repo_root=Path("."))
        self.assertEqual(report["status"], "FAIL_CLOSED")
        self.assertFalse(report["deployment_approved"])
        self.assertEqual(report["quality_gate"]["status"], "NOT_EVALUATED")
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
