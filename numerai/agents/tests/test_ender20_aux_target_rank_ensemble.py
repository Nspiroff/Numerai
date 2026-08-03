from __future__ import annotations

import copy
import hashlib
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

from agents.code.analysis import evaluate_ender20_aux_target_rank_ensemble as aux


def _repo_root() -> Path:
    return Path(aux.__file__).resolve().parents[4]


def _manifest() -> dict:
    path = (
        _repo_root()
        / "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53/"
        "source_manifest.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(
    *,
    bmc_mean: float = 0.003,
    bmc_sharpe: float = 0.40,
    bmc_drawdown: float = 0.05,
    corr_mean: float = 0.02,
    ender20_similarity: float = 0.50,
    ender60_similarity: float = 0.50,
    tabm_similarity: float = 0.50,
) -> dict:
    return {
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
        "avg_ender20_similarity": ender20_similarity,
        "avg_ender60_similarity": ender60_similarity,
        "avg_tabm_similarity": tabm_similarity,
    }


def _write_prediction_artifact(
    path: Path,
    frame: pd.DataFrame,
    semantics: dict | None,
) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    if semantics is not None:
        metadata[aux.PREDICTION_SEMANTICS_METADATA_KEY] = json.dumps(
            semantics, sort_keys=True
        ).encode("utf-8")
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path)


def _synthetic_protocol(root: Path) -> aux.FrozenProtocol:
    experiment = root / "experiment"
    experiment.mkdir(parents=True, exist_ok=True)
    return aux.FrozenProtocol(
        repo_root=root,
        experiment_dir=experiment,
        source_manifest_path=root / "source_manifest.json",
        source_manifest={},
        scout_configs={},
        scout_config_paths={},
        medium_features=("feature_1",),
        pretraining_commit="1" * 40,
        gpu_runtime_path=root / "gpu_runtime.json",
        gpu_runtime_receipt={},
    )


def _stage_expected() -> aux.ExpectedCohort:
    calibration_eras = tuple(f"{index:04d}" for index in range(1, 164)) + (
        aux.SCOUT_LAST_CALIBRATION_ERA,
    )
    locked_eras = (
        aux.SCOUT_FIRST_LOCKED_ERA,
        *(f"{index:04d}" for index in range(1030, 1078)),
        aux.SCOUT_LAST_ERA,
    )
    eras = calibration_eras + locked_eras
    frame = pd.DataFrame(
        {
            aux.ID_COLUMN: [f"id-{index}" for index in range(len(eras))],
            aux.ERA_COLUMN: eras,
            aux.ENDER_TARGET: np.linspace(0.0, 1.0, len(eras)),
            aux.BENCHMARK_ENDER20: np.linspace(1.0, 0.0, len(eras)),
            aux.BENCHMARK_ENDER60: np.linspace(0.5, 0.9, len(eras)),
            aux.FOLD_COLUMN: np.repeat([1, 2], [164, 50]),
        }
    )
    folds = (
        {"fold": 1, "train_eras": 10, "val_eras": 164, "train_rows": 10, "val_rows": 164},
        {"fold": 2, "train_eras": 174, "val_eras": 50, "train_rows": 174, "val_rows": 50},
    )
    return aux.ExpectedCohort(
        frame=frame,
        full_rows=len(frame),
        full_eras=len(eras),
        eras=eras,
        folds=folds,
    )


class CheckpointAndManifestTests(unittest.TestCase):
    def test_protocol_and_implementation_paths_use_distinct_checkpoints(self) -> None:
        calls: list[list[str]] = []

        def fake_git(_root: Path, arguments: list[str]) -> subprocess.CompletedProcess:
            calls.append(list(arguments))
            return subprocess.CompletedProcess(["git", *arguments], 0, "", "")

        pretraining_commit = "1" * 40
        with patch.object(aux, "_run_git", side_effect=fake_git):
            aux.verify_checkpoint_boundaries(Path("repo"), pretraining_commit)

        diff_calls = [call for call in calls if call[:2] == ["diff", "--quiet"]]
        self.assertEqual(len(diff_calls), 2)
        protocol_diff = next(
            call for call in diff_calls if call[2] == aux.PRE_SCORING_COMMIT
        )
        implementation_diff = next(
            call for call in diff_calls if call[2] == pretraining_commit
        )
        self.assertTrue(set(aux.PROTOCOL_CHECKPOINT_PATHS).issubset(protocol_diff))
        self.assertTrue(
            set(aux.TRAINING_CHECKPOINT_PATHS).issubset(implementation_diff)
        )
        self.assertFalse(
            set(aux.PROTOCOL_CHECKPOINT_PATHS).intersection(
                aux.TRAINING_CHECKPOINT_PATHS
            )
        )

    def test_checkpoint_rejects_noncanonical_commit_or_dirty_paths(self) -> None:
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError, "full lowercase 40-character"
        ):
            aux.verify_checkpoint_boundaries(Path("repo"), "ABC")

        def dirty_git(_root: Path, arguments: list[str]) -> subprocess.CompletedProcess:
            stdout = " M frozen.py\n" if arguments[0] == "status" else ""
            return subprocess.CompletedProcess(["git", *arguments], 0, stdout, "")

        with patch.object(aux, "_run_git", side_effect=dirty_git):
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "uncommitted or untracked"
            ):
                aux.verify_checkpoint_boundaries(Path("repo"), "1" * 40)

    def test_checkpoint_covers_imported_executables_and_rejects_dirty_dependency(
        self,
    ) -> None:
        imported_executables = {
            "numerai/agents/code/analysis/evaluate_ender20_hybrid_stability.py",
            "numerai/agents/code/analysis/evaluate_xerxes20_lgbm_challenger.py",
        }
        self.assertTrue(
            imported_executables.issubset(aux.TRAINING_CHECKPOINT_PATHS),
            "Every imported analysis evaluator that can affect scoring must be "
            "bound to the pretraining checkpoint.",
        )

        dirty_dependency = next(iter(imported_executables))

        def dirty_git(_root: Path, arguments: list[str]) -> subprocess.CompletedProcess:
            is_training_status = (
                arguments[:3] == ["status", "--porcelain", "--untracked-files=all"]
                and dirty_dependency in arguments
            )
            stdout = f" M {dirty_dependency}\n" if is_training_status else ""
            return subprocess.CompletedProcess(["git", *arguments], 0, stdout, "")

        with patch.object(aux, "_run_git", side_effect=dirty_git):
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "uncommitted or untracked"
            ):
                aux.verify_checkpoint_boundaries(Path("repo"), "1" * 40)

    def test_manifest_pins_protocol_configs_sources_and_output_paths(self) -> None:
        root = _repo_root()
        experiment = (
            root
            / "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53"
        )
        manifest_path = experiment / "source_manifest.json"
        manifest = _manifest()

        self.assertEqual(aux._sha256_file(manifest_path), aux.SOURCE_MANIFEST_SHA256)
        self.assertEqual(
            manifest["experiment_files"]["gate.md"]["sha256"], aux.GATE_SHA256
        )
        for relative, receipt in manifest["experiment_files"].items():
            path = experiment / relative
            self.assertEqual(path.stat().st_size, receipt["size_bytes"])
            self.assertEqual(aux._sha256_file(path), receipt["sha256"])

        source_paths = {item["path"] for item in manifest["confirmation_sources"]}
        self.assertEqual(
            source_paths,
            {
                "numerai/v5.3/train.parquet",
                "numerai/v5.3/validation.parquet",
                "numerai/v5.3/train_benchmark_models.parquet",
                "numerai/v5.3/validation_benchmark_models.parquet",
            },
        )
        for receipt in manifest["confirmation_sources"]:
            self.assertRegex(receipt["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue((root / receipt["path"]).is_file())

        scout_outputs = manifest["new_scout_outputs"]
        self.assertEqual(set(scout_outputs), set(aux.SCOUT_NEW_COMPONENTS))
        self.assertTrue(
            all(item["must_be_absent_before_run"] for item in scout_outputs.values())
        )
        confirmation_outputs = manifest["confirmation_output_contract"]
        self.assertEqual(set(confirmation_outputs), set(aux.ALL_COMPONENTS))
        self.assertEqual(
            len(
                {
                    item[key]
                    for item in confirmation_outputs.values()
                    for key in ("predictions_path", "results_path")
                }
            ),
            10,
        )
        deployment = manifest["deployment_output_contract"]
        self.assertTrue(deployment["must_be_absent_before_packaging"])
        self.assertFalse(deployment["overwrite_or_rerun_allowed"])
        self.assertEqual(
            len(
                {
                    deployment["final_pickle_path"],
                    deployment["final_fit_receipt_path"],
                    deployment["docker_predictions_path"],
                    deployment["docker_receipt_path"],
                }
            ),
            4,
        )

    def test_scout_configs_are_exact_target_only_variants_of_xerxes_depth8(self) -> None:
        root = _repo_root()
        experiment = (
            root
            / "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53"
        )
        xerxes_path = (
            root
            / "numerai/agents/experiments/xerxes20_lgbm_challenger_v53/"
            "configs/r1_depth8.py"
        )
        xerxes_config = runpy.run_path(str(xerxes_path))["CONFIG"]

        for component in aux.SCOUT_NEW_COMPONENTS:
            name = f"r1_{component}_d8_t6000"
            path = experiment / "configs" / f"{name}.py"
            actual = runpy.run_path(str(path))["CONFIG"]
            expected = copy.deepcopy(xerxes_config)
            expected["data"]["target_col"] = aux.COMPONENT_TARGETS[component]
            expected["output"]["output_dir"] = (
                "experiments/ender20_aux_target_rank_ensemble_v53"
            )
            expected["output"]["results_name"] = name
            self.assertEqual(actual, expected)
            aux.validate_component_config(component, actual)

            altered = copy.deepcopy(actual)
            altered["model"]["params"]["learning_rate"] = 0.004
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "LightGBM parameters differs"
            ):
                aux.validate_component_config(component, altered)

    def test_confirmation_config_allows_only_frozen_data_mode_changes(self) -> None:
        root = _repo_root()
        scout_path = (
            root
            / "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53/"
            "configs/r1_jasper_d8_t6000.py"
        )
        confirmation = copy.deepcopy(runpy.run_path(str(scout_path))["CONFIG"])
        confirmation["data"].pop("full_data_path")
        confirmation["data"].pop("benchmark_data_path")
        confirmation["data"]["disk_feature_store_path"] = (
            "v5.3/target_jasper_20_feature_store"
        )
        confirmation["data"]["embargo_eras"] = 52
        confirmation["training"]["data_mode"] = "disk_feature_store"
        confirmation["training"]["cv"]["embargo"] = 52
        confirmation["output"]["results_name"] = "confirmation_jasper_d8_t6000"

        aux.validate_component_config("jasper", confirmation, confirmation=True)

        confirmation["training"]["sample_seed"] = 2027
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError, "training.sample_seed differs"
        ):
            aux.validate_component_config("jasper", confirmation, confirmation=True)

    def test_confirmation_config_rejects_unexpected_behavior_keys(self) -> None:
        root = _repo_root()
        scout_path = (
            root
            / "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53/"
            "configs/r1_jasper_d8_t6000.py"
        )
        confirmation = copy.deepcopy(runpy.run_path(str(scout_path))["CONFIG"])
        confirmation["data"].pop("full_data_path")
        confirmation["data"].pop("benchmark_data_path")
        confirmation["data"]["disk_feature_store_path"] = (
            "v5.3/target_jasper_20_feature_store"
        )
        confirmation["data"]["embargo_eras"] = 52
        confirmation["training"]["data_mode"] = "disk_feature_store"
        confirmation["training"]["cv"]["embargo"] = 52
        confirmation["output"]["results_name"] = "confirmation_jasper_d8_t6000"

        unexpected_model = copy.deepcopy(confirmation)
        unexpected_model["model"]["prediction_batch_size"] = 1
        with self.assertRaises(aux.EnderEnsembleEvaluationError):
            aux.validate_component_config(
                "jasper", unexpected_model, confirmation=True
            )

        unexpected_top_level = copy.deepcopy(confirmation)
        unexpected_top_level["behavior_override"] = {"enabled": True}
        with self.assertRaises(aux.EnderEnsembleEvaluationError):
            aux.validate_component_config(
                "jasper", unexpected_top_level, confirmation=True
            )


class ReceiptPrimitiveTests(unittest.TestCase):
    def test_artifact_receipt_binds_repo_path_size_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifacts" / "component.bin"
            artifact.parent.mkdir()
            artifact.write_bytes(b"frozen component")
            receipt = {
                "path": "artifacts/component.bin",
                "size_bytes": artifact.stat().st_size,
                "sha256": aux._sha256_file(artifact),
            }

            self.assertEqual(
                aux._validate_path_receipt(root, receipt, "component"),
                artifact,
            )

            wrong_hash = {**receipt, "sha256": "0" * 64}
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "component hash differs"
            ):
                aux._validate_path_receipt(root, wrong_hash, "component")

            escaping = {**receipt, "path": "../component.bin"}
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "escapes"
            ):
                aux._validate_path_receipt(root, escaping, "component")

    def test_receipt_is_content_addressed_and_write_once(self) -> None:
        receipt = {
            "schema_version": 1,
            "experiment": aux.EXPERIMENT_NAME,
            "stage": "calibrate",
            "state": "PASS",
            "passed": True,
            "protocol": {"gate": aux.GATE_SHA256},
            "inputs": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = aux._write_content_addressed_receipt(output, "calibration", receipt)
            expected = hashlib.sha256(aux._receipt_bytes(receipt)).hexdigest()
            self.assertEqual(path.name, f"calibration-{expected}.json")
            self.assertEqual(aux._sha256_file(path), expected)
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "already exists"
            ):
                aux._write_content_addressed_receipt(output, "calibration", receipt)

            conflicting = copy.deepcopy(receipt)
            conflicting["state"] = "FAIL"
            conflicting["passed"] = False
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "already exists"
            ):
                aux._write_content_addressed_receipt(
                    output, "calibration", conflicting
                )

    def test_bound_receipt_requires_exact_hash_filename_stage_and_experiment(self) -> None:
        receipt = {
            "schema_version": 1,
            "experiment": aux.EXPERIMENT_NAME,
            "stage": "calibrate",
            "state": "PASS",
            "passed": True,
            "protocol": {},
            "inputs": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = aux._write_content_addressed_receipt(
                Path(directory), "calibrate", receipt
            )
            digest = aux._sha256_file(path)
            loaded = aux._load_bound_receipt(
                path, digest, expected_stage="calibrate"
            )
            self.assertEqual(loaded, receipt)
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "receipt hash differs"
            ):
                aux._load_bound_receipt(path, "0" * 64, expected_stage="calibrate")
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "receipt stage differs"
            ):
                aux._load_bound_receipt(path, digest, expected_stage="locked")

    def test_seal_bindings_require_each_unique_known_component(self) -> None:
        bindings = aux._parse_seal_bindings(
            [
                [component, f"{component}.json", str(index) * 64]
                for index, component in enumerate(aux.SCOUT_NEW_COMPONENTS, start=1)
            ]
        )
        self.assertEqual(set(bindings), set(aux.SCOUT_NEW_COMPONENTS))
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError, "Duplicate seal component"
        ):
            aux._parse_seal_bindings(
                [["jasper", "one.json", "1" * 64], ["jasper", "two.json", "2" * 64]]
            )
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError, "Unknown seal component"
        ):
            aux._parse_seal_bindings([["xerxes", "x.json", "1" * 64]])

    def test_cli_requires_exactly_four_explicit_seal_bindings(self) -> None:
        args = aux.parse_args(
            [
                "calibrate",
                "--pretraining-commit",
                "1" * 40,
                *sum(
                    (
                        [
                            "--seal-receipt",
                            component,
                            f"{component}.json",
                            str(index) * 64,
                        ]
                        for index, component in enumerate(
                            aux.SCOUT_NEW_COMPONENTS, start=1
                        )
                    ),
                    [],
                ),
            ]
        )
        bindings = aux._parse_seal_bindings(args.seal_receipt)
        self.assertEqual(set(bindings), set(aux.SCOUT_NEW_COMPONENTS))


class RankAndMetricTests(unittest.TestCase):
    def test_rank_within_era_uses_average_ties_and_is_era_local(self) -> None:
        ranked = aux.rank_within_era(
            [1.0, 2.0, 2.0, 4.0, 100.0, 50.0],
            ["0001", "0001", "0001", "0001", "0002", "0002"],
        )
        np.testing.assert_allclose(ranked, [0.25, 0.625, 0.625, 1.0, 1.0, 0.5])

    def test_rank_blend_ranks_components_then_weighted_sum_then_reranks(self) -> None:
        frame = pd.DataFrame(
            {
                aux.ERA_COLUMN: ["0001"] * 4,
                "jasper": [1.0, 2.0, 2.0, 4.0],
                "teager2b": [4.0, 3.0, 2.0, 1.0],
                "victor": [1.0, 3.0, 2.0, 4.0],
                "xerxes": [2.0, 4.0, 1.0, 3.0],
                "tyler": [4.0, 1.0, 3.0, 2.0],
            }
        )
        blended = aux.build_rank_blends(frame)

        component_ranks = {
            name: aux.rank_within_era(frame[name], frame[aux.ERA_COLUMN])
            for name in aux.ALL_COMPONENTS
        }
        for candidate, weights in aux.BLEND_WEIGHTS.items():
            raw = weights["core"] * sum(
                component_ranks[name]
                for name in ("jasper", "teager2b", "victor", "xerxes")
            )
            raw += weights["tyler"] * component_ranks["tyler"]
            expected = aux.rank_within_era(raw, frame[aux.ERA_COLUMN])
            np.testing.assert_allclose(blended[candidate], expected)
        self.assertEqual(
            set(blended) - set(frame),
            set(aux.CANDIDATE_NAMES),
        )

    def test_locked_blend_materializes_only_the_selected_candidate(self) -> None:
        frame = pd.DataFrame(
            {
                aux.ERA_COLUMN: ["0001"] * 4,
                "jasper": [1.0, 2.0, 3.0, 4.0],
                "teager2b": [4.0, 3.0, 2.0, 1.0],
                "victor": [1.0, 3.0, 2.0, 4.0],
                "xerxes": [2.0, 4.0, 1.0, 3.0],
                "tyler": [4.0, 1.0, 3.0, 2.0],
            }
        )

        locked = aux.build_selected_rank_blend(frame, "tyler_w10")

        self.assertIn("tyler_w10", locked)
        self.assertFalse(
            (set(aux.CANDIDATE_NAMES) - {"tyler_w10"}).intersection(locked)
        )

    def test_symmetric_similarity_is_average_tie_spearman_in_each_era(self) -> None:
        eras = ["0001"] * 4 + ["0002"] * 4
        reference = [1, 2, 2, 4, 10, 30, 20, 40]
        same = [10, 20, 20, 40, 100, 300, 200, 400]
        reverse = [4, 2, 2, 1, 40, 20, 30, 10]

        forward = aux.symmetric_per_era_similarity(same, reference, eras)
        backward = aux.symmetric_per_era_similarity(reference, same, eras)
        negative = aux.symmetric_per_era_similarity(reverse, reference, eras)

        np.testing.assert_allclose(forward, [1.0, 1.0])
        np.testing.assert_allclose(backward, forward)
        np.testing.assert_allclose(negative, [-1.0, -1.0])

    def test_summary_uses_population_std_and_drawdown_without_initial_zero(self) -> None:
        signal = "candidate"
        index = ["0001", "0002"]
        per_era = {
            "corr": pd.DataFrame({signal: [0.1, 0.3]}, index=index),
            "bmc": pd.DataFrame({signal: [-1.0, 2.0]}, index=index),
            "ender20_similarity": pd.DataFrame(
                {signal: [0.1, 0.3]}, index=index
            ),
            "ender60_similarity": pd.DataFrame(
                {signal: [-0.2, 0.2]}, index=index
            ),
            "tabm_similarity": pd.DataFrame({signal: [0.4, 0.6]}, index=index),
        }

        summary = aux.summarize_signal(per_era, signal)

        self.assertAlmostEqual(summary["bmc"]["mean"], 0.5)
        self.assertAlmostEqual(summary["bmc"]["std"], 1.5)
        self.assertAlmostEqual(summary["bmc"]["sharpe"], 1.0 / 3.0)
        self.assertEqual(summary["bmc"]["max_drawdown"], 0.0)
        self.assertAlmostEqual(summary["corr"]["std"], 0.1)
        self.assertAlmostEqual(summary["avg_ender20_similarity"], 0.2)
        self.assertAlmostEqual(summary["avg_ender60_similarity"], 0.0)
        self.assertAlmostEqual(summary["avg_tabm_similarity"], 0.5)

    def test_compute_metrics_includes_all_three_similarity_guards(self) -> None:
        eras = ["0001"] * 4 + ["0002"] * 4
        frame = pd.DataFrame(
            {
                aux.ERA_COLUMN: eras,
                aux.ENDER_TARGET: [1, 2, 3, 4, 1, 2, 3, 4],
                aux.BENCHMARK_ENDER20: [1, 2, 3, 4, 4, 3, 2, 1],
                aux.BENCHMARK_ENDER60: [4, 3, 2, 1, 1, 2, 3, 4],
                "tabm": [1, 3, 2, 4, 4, 2, 3, 1],
                "candidate": [1, 2, 3, 4, 4, 3, 2, 1],
            }
        )
        corr = pd.DataFrame({"candidate": [0.1, 0.2]}, index=["0001", "0002"])
        bmc = pd.DataFrame({"candidate": [0.01, 0.02]}, index=["0001", "0002"])
        with patch.object(aux.numerai_metrics, "per_era_corr", return_value=corr), patch.object(
            aux.numerai_metrics, "per_era_bmc", return_value=bmc
        ):
            metrics = aux.compute_per_era_metrics(
                frame,
                ["candidate"],
                ["0001", "0002"],
                tabm_column="tabm",
            )

        self.assertEqual(
            set(metrics),
            {
                "corr",
                "bmc",
                "ender20_similarity",
                "ender60_similarity",
                "tabm_similarity",
            },
        )
        np.testing.assert_allclose(
            metrics["ender20_similarity"]["candidate"], [1.0, 1.0]
        )
        np.testing.assert_allclose(
            metrics["ender60_similarity"]["candidate"], [-1.0, -1.0]
        )
        self.assertTrue(
            np.isfinite(metrics["tabm_similarity"]["candidate"]).all()
        )


class CohortAndArtifactSemanticsTests(unittest.TestCase):
    def test_source_join_is_by_id_and_era_not_incidental_row_order(self) -> None:
        data = pd.DataFrame(
            {
                aux.ID_COLUMN: ["a", "b"],
                aux.ERA_COLUMN: ["0001", "0002"],
                aux.ENDER_TARGET: [0.1, 0.2],
            }
        )
        benchmark = pd.DataFrame(
            {
                aux.ID_COLUMN: ["b", "a"],
                aux.ERA_COLUMN: ["0002", "0001"],
                aux.BENCHMARK_ENDER20: [0.4, 0.3],
            }
        )

        merged = aux._merge_sources_one_to_one(data, benchmark, label="synthetic")

        self.assertEqual(merged[aux.ID_COLUMN].tolist(), ["a", "b"])
        self.assertEqual(merged[aux.BENCHMARK_ENDER20].tolist(), [0.3, 0.4])

        wrong_era = benchmark.copy()
        wrong_era.loc[wrong_era[aux.ID_COLUMN] == "a", aux.ERA_COLUMN] = "9999"
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError, "source eras differ by ID"
        ):
            aux._merge_sources_one_to_one(data, wrong_era, label="synthetic")

    def test_expected_folds_are_derived_from_the_producers_frozen_chronology(self) -> None:
        full = pd.DataFrame(
            {
                aux.ID_COLUMN: [f"id-{index}" for index in range(1, 11)],
                aux.ERA_COLUMN: [f"{index:04d}" for index in range(1, 11)],
            }
        )

        expected = aux._derive_expected_oof(
            full,
            embargo=1,
            expected_rows=8,
            expected_eras=8,
            first_era="0003",
            last_era="0010",
            calibration_eras=4,
            last_calibration_era="0006",
            first_locked_era="0007",
        )

        self.assertEqual(expected.eras, tuple(f"{index:04d}" for index in range(3, 11)))
        self.assertEqual(expected.frame[aux.FOLD_COLUMN].tolist(), [1, 1, 2, 2, 3, 3, 4, 4])
        self.assertEqual([fold["fold"] for fold in expected.folds], [1, 2, 3, 4])

    def test_prediction_artifact_aligns_by_id_and_era_but_uses_own_fold_map(
        self,
    ) -> None:
        target = aux.COMPONENT_TARGETS["jasper"]
        expected = pd.DataFrame(
            {
                aux.ID_COLUMN: ["a", "b", "c", "d"],
                aux.ERA_COLUMN: ["0001", "0001", "0002", "0002"],
                target: [0.1, 0.2, 0.3, 0.4],
                # This deliberately represents some other producer's fold receipt.
                aux.FOLD_COLUMN: [8, 8, 9, 9],
            }
        )
        artifact = pd.DataFrame(
            {
                aux.ID_COLUMN: ["d", "b", "a", "c"],
                aux.ERA_COLUMN: ["0002", "0001", "0001", "0002"],
                target: [0.4, 0.2, 0.1, 0.3],
                aux.PREDICTION_COLUMN: [40.0, 20.0, 10.0, 30.0],
                aux.FOLD_COLUMN: [2, 1, 1, 2],
            }
        )
        semantics = {"producer": "jasper", "rank": "raw"}
        own_fold_map = {"0001": 1, "0002": 2}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jasper.parquet"
            _write_prediction_artifact(path, artifact, semantics)
            predictions = aux.validate_prediction_artifact(
                path,
                expected,
                semantics,
                target_column=target,
                expected_fold_by_era=own_fold_map,
            )
            np.testing.assert_allclose(predictions, [10.0, 20.0, 30.0, 40.0])

            wrong_fold = artifact.copy()
            wrong_fold.loc[0, aux.FOLD_COLUMN] = 1
            _write_prediction_artifact(path, wrong_fold, semantics)
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "frozen producer"
            ):
                aux.validate_prediction_artifact(
                    path,
                    expected,
                    semantics,
                    target_column=target,
                    expected_fold_by_era=own_fold_map,
                )

            fractional_fold = artifact.copy()
            fractional_fold[aux.FOLD_COLUMN] = fractional_fold[
                aux.FOLD_COLUMN
            ].astype(float)
            # Row 1 belongs to producer fold 1, so an unsafe int cast would
            # silently turn this malformed 1.5 provenance value into 1.
            fractional_fold.loc[1, aux.FOLD_COLUMN] = 1.5
            _write_prediction_artifact(path, fractional_fold, semantics)
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "fold provenance"
            ):
                aux.validate_prediction_artifact(
                    path,
                    expected,
                    semantics,
                    target_column=target,
                    expected_fold_by_era=own_fold_map,
                )

    def test_prediction_semantics_enforce_legacy_and_full_contracts(self) -> None:
        target = aux.ENDER_TARGET
        expected = pd.DataFrame(
            {
                aux.ID_COLUMN: ["a", "b"],
                aux.ERA_COLUMN: ["0001", "0001"],
                target: [0.1, 0.2],
                aux.FOLD_COLUMN: [1, 1],
            }
        )
        artifact = pd.DataFrame(
            {
                aux.ID_COLUMN: ["a", "b"],
                aux.ERA_COLUMN: ["0001", "0001"],
                target: [0.1, 0.2],
                aux.PREDICTION_COLUMN: [0.3, 0.4],
                aux.FOLD_COLUMN: [1, 1],
            }
        )
        semantics = {
            "training_target": {"transform": {"type": "residual_to_benchmark"}}
        }

        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "legacy.parquet"
            _write_prediction_artifact(legacy, artifact, None)
            aux.validate_prediction_artifact(
                legacy,
                expected,
                None,
                target_column=target,
                require_semantics=False,
            )
            with self.assertRaisesRegex(
                aux.EnderEnsembleEvaluationError, "Parquet semantics differs"
            ):
                aux.validate_prediction_artifact(
                    legacy,
                    expected,
                    semantics,
                    target_column=target,
                    require_semantics=True,
                )

            full = Path(directory) / "full.parquet"
            _write_prediction_artifact(full, artifact, semantics)
            aux.validate_prediction_artifact(
                full,
                expected,
                semantics,
                target_column=target,
                require_semantics=True,
            )

    def test_result_receipt_requires_gpu_folds_and_exact_automatic_metrics(self) -> None:
        root = _repo_root()
        config_path = (
            root
            / "numerai/agents/experiments/ender20_aux_target_rank_ensemble_v53/"
            "configs/r1_jasper_d8_t6000.py"
        )
        config = runpy.run_path(str(config_path))["CONFIG"]
        target = aux.COMPONENT_TARGETS["jasper"]
        frame = pd.DataFrame(
            {
                aux.ID_COLUMN: ["a", "b", "c", "d"],
                aux.ERA_COLUMN: ["0001", "0001", "0002", "0002"],
                target: [0.1, 0.2, 0.3, 0.4],
                aux.FOLD_COLUMN: [1, 1, 2, 2],
            }
        )
        folds = (
            {"fold": 1, "train_eras": 1, "val_eras": 1, "train_rows": 2, "val_rows": 2},
            {"fold": 2, "train_eras": 2, "val_eras": 1, "train_rows": 4, "val_rows": 2},
        )
        expected = aux.ExpectedCohort(
            frame=frame,
            full_rows=6,
            full_eras=3,
            eras=("0001", "0002"),
            folds=folds,
        )
        result_name = config["output"]["results_name"]
        component = aux.ComponentPaths(
            name="jasper",
            config=Path("config.py"),
            result=Path(f"{result_name}.json"),
            predictions=Path(f"{result_name}.parquet"),
        )
        result = {
            "model": aux._expected_model_payload(config),
            "preprocessing": {
                "nan_missing_all_twos": False,
                "missing_value": 2.0,
            },
            "data": {
                "data_version": "v5.3",
                "feature_set": "medium",
                "target": target,
                "full_rows": 6,
                "full_eras": 3,
                "oof_rows": 4,
                "oof_eras": 2,
                "embargo_eras": 13,
                "require_benchmark_coverage": True,
                "data_mode": "eager",
                "full_data_path": "v5.3/downsampled_full.parquet",
            },
            "benchmark": {
                "model": aux.BENCHMARK_ENDER20,
                "file": "v5.3/downsampled_full_benchmark_models.parquet",
            },
            "training": {
                "data_sampling": {"max_train_samples": 500_000, "sample_seed": 1337},
                "data_mode": "eager",
                "cv": {
                    "enabled": True,
                    "n_splits": 5,
                    "embargo": 13,
                    "mode": "expanding",
                    "min_train_size": 0,
                },
            },
            "cv": {
                "n_splits": 5,
                "embargo": 13,
                "mode": "expanding",
                "min_train_size": 0,
                "folds_used": 2,
                "folds": [
                    {
                        **fold,
                        "model_diagnostics": {
                            "effective_device_type": "gpu",
                            "gpu_fallback_used": False,
                        },
                    }
                    for fold in folds
                ],
            },
            "output": {
                "predictions_file": f"some/path/{result_name}.parquet",
                "prediction_semantics": aux._expected_semantics(config),
            },
            "metrics": {"corr": {}, "bmc": {}, "bmc_last_200_eras": {}},
        }

        self.assertEqual(
            aux.validate_result_json(component, result, config, expected),
            aux._expected_semantics(config),
        )

        fallback = copy.deepcopy(result)
        fallback["cv"]["folds"][0]["model_diagnostics"]["gpu_fallback_used"] = True
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError, "GPU fallback differs"
        ):
            aux.validate_result_json(component, fallback, config, expected)

        extra_metric = copy.deepcopy(result)
        extra_metric["metrics"]["unfrozen"] = 1.0
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError, "automatic metric schema differs"
        ):
            aux.validate_result_json(component, extra_metric, config, expected)

    def test_tabm_result_contract_distinguishes_legacy_and_full_metadata(self) -> None:
        target_transform = {
            "type": "residual_to_benchmark",
            "benchmark_col": aux.BENCHMARK_ENDER20,
            "era_col": aux.ERA_COLUMN,
            "per_era": True,
            "fit_intercept": True,
        }
        legacy = {"model": {"target_transform": target_transform}, "output": {}}
        self.assertIsNone(
            aux._validate_tabm_result(legacy, "legacy", metadata_required=False)
        )
        with self.assertRaisesRegex(
            aux.EnderEnsembleEvaluationError, "lacks prediction semantics"
        ):
            aux._validate_tabm_result(legacy, "full", metadata_required=True)

        semantics = {
            "training_target": {"transform": {"type": "residual_to_benchmark"}}
        }
        full = copy.deepcopy(legacy)
        full["output"]["prediction_semantics"] = semantics
        self.assertEqual(
            aux._validate_tabm_result(full, "full", metadata_required=True),
            semantics,
        )


class SelectionAndThresholdTests(unittest.TestCase):
    def test_calibration_boundaries_preserve_inclusive_and_strict_operators(self) -> None:
        boundary = _summary(
            bmc_mean=0.002,
            bmc_sharpe=0.25,
            bmc_drawdown=0.10,
            corr_mean=0.012,
            ender20_similarity=0.75,
            ender60_similarity=0.75,
            tabm_similarity=0.75,
        )
        checks = aux.calibration_checks(boundary)

        self.assertTrue(checks["bmc_mean"])
        self.assertTrue(checks["corr_mean"])
        self.assertFalse(checks["bmc_sharpe"])
        self.assertFalse(checks["bmc_max_drawdown"])
        self.assertFalse(checks["ender20_similarity"])
        self.assertFalse(checks["ender60_similarity"])
        self.assertFalse(checks["tabm_similarity"])

    def test_anchored_tie_set_does_not_chain_pairwise_neighbors(self) -> None:
        summaries = {name: _summary(bmc_mean=0.0021) for name in aux.CANDIDATE_NAMES}
        summaries["tyler_w00"] = _summary(bmc_mean=0.00300, bmc_sharpe=0.30)
        summaries["tyler_w10"] = _summary(bmc_mean=0.00295, bmc_sharpe=0.80)
        summaries["tyler_w20_equal5"] = _summary(
            bmc_mean=0.00286, bmc_sharpe=10.0
        )

        selected, evaluations = aux.select_scout_candidate(summaries)

        self.assertEqual(selected, "tyler_w10")
        self.assertTrue(evaluations["tyler_w00"]["in_tie_set"])
        self.assertTrue(evaluations["tyler_w10"]["in_tie_set"])
        self.assertFalse(evaluations["tyler_w20_equal5"]["in_tie_set"])

    def test_tie_breaks_by_sharpe_drawdown_weight_then_name(self) -> None:
        summaries = {name: _summary(bmc_mean=0.0021) for name in aux.CANDIDATE_NAMES}
        for name in ("tyler_w00", "tyler_w10"):
            summaries[name] = _summary(bmc_mean=0.003, bmc_sharpe=0.5)
        summaries["tyler_w10"]["bmc"]["max_drawdown"] = 0.04
        self.assertEqual(aux.select_scout_candidate(summaries)[0], "tyler_w10")

        summaries["tyler_w10"]["bmc"]["max_drawdown"] = 0.05
        self.assertEqual(aux.select_scout_candidate(summaries)[0], "tyler_w00")

    def test_locked_and_confirmation_strict_boundaries_fail(self) -> None:
        locked_boundary = _summary(
            bmc_mean=0.0,
            bmc_sharpe=0.20,
            bmc_drawdown=0.10,
            corr_mean=0.008,
        )
        self.assertFalse(any(aux.locked_checks(locked_boundary).values()))

        confirmation_boundary = _summary(
            bmc_mean=0.002,
            bmc_sharpe=0.35,
            bmc_drawdown=0.15,
            corr_mean=0.012,
            ender20_similarity=0.75,
            ender60_similarity=0.75,
            tabm_similarity=0.75,
        )
        locked_confirmation = _summary(
            bmc_mean=0.0,
            bmc_sharpe=0.20,
            bmc_drawdown=0.15,
            corr_mean=0.008,
        )
        checks = aux.confirmation_checks(
            confirmation_boundary,
            locked_confirmation,
            confirmation_boundary,
        )
        self.assertTrue(checks["calibration_bmc_mean"])
        self.assertTrue(checks["calibration_corr_mean"])
        self.assertTrue(checks["full_bmc_mean"])
        self.assertTrue(checks["full_corr_mean"])
        for name, passed in checks.items():
            if name not in {
                "calibration_bmc_mean",
                "calibration_corr_mean",
                "full_bmc_mean",
                "full_corr_mean",
            }:
                self.assertFalse(passed, name)

    def test_confirmation_calibration_checks_need_no_locked_metrics_object(self) -> None:
        calibration_only = _summary(
            bmc_mean=0.002,
            bmc_sharpe=0.36,
            bmc_drawdown=0.14,
            corr_mean=0.012,
            ender20_similarity=0.74,
            ender60_similarity=0.74,
            tabm_similarity=0.74,
        )

        checks = aux.confirmation_calibration_checks(calibration_only)

        self.assertTrue(all(checks.values()))
        self.assertTrue(all(name.startswith("calibration_") for name in checks))
        self.assertFalse(any("locked" in name or "full" in name for name in checks))


if __name__ == "__main__":
    unittest.main()
