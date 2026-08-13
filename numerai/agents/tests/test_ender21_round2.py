from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest import mock

from agents.code.modeling.utils import pipeline as pipeline_module
from agents.code.modeling.utils.pipeline import run_training


EXPERIMENT = (
    Path(__file__).resolve().parents[1]
    / "experiments/ender21_residual_stability_v53"
)
CONFIGS = EXPERIMENT / "configs"


def _load_config(name: str) -> dict:
    return runpy.run_path(str(CONFIGS / f"{name}.py"))["CONFIG"]


def _load_evaluator():
    path = EXPERIMENT / "evaluate_round2.py"
    spec = importlib.util.spec_from_file_location("ender21_round2_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the Ender21 Round-2 evaluator.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestEnder21Round2Configs(unittest.TestCase):
    def test_four_configs_change_only_the_declared_seed_dimension(self) -> None:
        contracts = {
            "r2_control_tabm_k64_model_seed2027": (
                "r1_control_tabm_k64",
                "model",
            ),
            "r2_control_tabm_k64_sample_seed2027": (
                "r1_control_tabm_k64",
                "sample",
            ),
            "r2_selected_tabm_k64_block_dro_model_seed2027": (
                "r1_tabm_k64_block_dro",
                "model",
            ),
            "r2_selected_tabm_k64_block_dro_sample_seed2027": (
                "r1_tabm_k64_block_dro",
                "sample",
            ),
        }
        for name, (base_name, seed_dimension) in contracts.items():
            with self.subTest(name=name):
                self.assertTrue((CONFIGS / f"{name}.py").is_file())
                actual = _load_config(name)
                expected = deepcopy(_load_config(base_name))
                expected["output"]["results_name"] = name
                if seed_dimension == "model":
                    expected["model"]["params"]["seed"] = 2027
                else:
                    expected["training"]["sample_seed"] = 2027
                self.assertEqual(actual, expected)

    def test_control_and_selected_seed_pairings_are_matched(self) -> None:
        pairs = (
            (
                "r2_control_tabm_k64_model_seed2027",
                "r2_selected_tabm_k64_block_dro_model_seed2027",
                2027,
                1337,
            ),
            (
                "r2_control_tabm_k64_sample_seed2027",
                "r2_selected_tabm_k64_block_dro_sample_seed2027",
                1337,
                2027,
            ),
        )
        for control_name, selected_name, model_seed, sample_seed in pairs:
            with self.subTest(control=control_name, selected=selected_name):
                control = _load_config(control_name)
                selected = _load_config(selected_name)
                self.assertEqual(control["model"]["params"]["seed"], model_seed)
                self.assertEqual(selected["model"]["params"]["seed"], model_seed)
                self.assertEqual(control["training"]["sample_seed"], sample_seed)
                self.assertEqual(selected["training"]["sample_seed"], sample_seed)

                normalized_control = deepcopy(control)
                normalized_selected = deepcopy(selected)
                normalized_control["output"]["results_name"] = "paired"
                normalized_selected["output"]["results_name"] = "paired"
                normalized_control["model"]["params"]["loss_mode"] = "paired"
                normalized_selected["model"]["params"]["loss_mode"] = "paired"
                self.assertEqual(normalized_control, normalized_selected)


class TestEnder21Round2Evaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluator = _load_evaluator()

    @staticmethod
    def _metrics(
        *,
        bmc: float,
        recent: float,
        drawdown: float,
        sharpe: float,
        corr: float,
        folds: tuple[float, ...] = (0.001, 0.002, 0.003, 0.004),
    ) -> dict:
        return {
            "bmc": {
                "mean": bmc,
                "max_drawdown": drawdown,
                "sharpe": sharpe,
            },
            "recent_fold_bmc_mean": recent,
            "corr": {"mean": corr},
            "fold_bmc_mean": {
                str(index + 1): value for index, value in enumerate(folds)
            },
        }

    def test_matched_checks_pass_at_frozen_inclusive_boundaries(self) -> None:
        control = self._metrics(
            bmc=0.010,
            recent=0.008,
            drawdown=0.100,
            sharpe=0.50,
            corr=0.020,
        )
        selected = self._metrics(
            bmc=0.009001,
            recent=0.007201,
            drawdown=0.08499,
            sharpe=0.4501,
            corr=0.005,
        )
        checks = self.evaluator._matched_checks(selected, control)
        self.assertEqual(
            set(checks),
            {
                "positive_full_bmc",
                "positive_recent_fold_bmc",
                "corr_guardrail",
                "full_bmc_retention",
                "recent_bmc_retention",
                "drawdown_improvement",
                "sharpe_retention",
                "all_folds_positive",
            },
        )
        self.assertTrue(all(checks.values()))

    def test_two_of_three_matched_realizations_pass(self) -> None:
        decision = self.evaluator._decide(
            {
                "base_seed1337": {"passed": True},
                "model_seed2027": {"passed": False},
                "sample_seed2027": {"passed": True},
            }
        )
        self.assertEqual(
            decision,
            {
                "passed_count": 2,
                "required_count": 2,
                "passed": True,
                "state": "SEED_REPLICATION_PASS",
            },
        )

    def test_one_of_three_matched_realizations_is_negative(self) -> None:
        decision = self.evaluator._decide(
            {
                "base_seed1337": {"passed": False},
                "model_seed2027": {"passed": True},
                "sample_seed2027": {"passed": False},
            }
        )
        self.assertEqual(
            decision,
            {
                "passed_count": 1,
                "required_count": 2,
                "passed": False,
                "state": "NEGATIVE",
            },
        )

    def test_decide_rejects_missing_extra_or_non_boolean_realizations(self) -> None:
        cases = (
            {
                "base_seed1337": {"passed": True},
                "model_seed2027": {"passed": True},
            },
            {
                "base_seed1337": {"passed": True},
                "model_seed2027": {"passed": True},
                "sample_seed2027": {"passed": False},
                "extra_seed": {"passed": False},
            },
            {
                "base_seed1337": {"passed": True},
                "model_seed2027": {"passed": 1},
                "sample_seed2027": {"passed": False},
            },
        )
        for realizations in cases:
            with self.subTest(realizations=realizations):
                with self.assertRaisesRegex(
                    ValueError, "realization|passed|boolean"
                ):
                    self.evaluator._decide(realizations)

    @staticmethod
    def _paired_config(*, loss_mode: str) -> dict:
        return {
            "data": {
                "data_version": "v5.3",
                "full_data_path": (
                    "v5.3/ender21_discovery_full_through_0861.parquet"
                ),
                "benchmark_data_path": (
                    "v5.3/ender21_discovery_benchmark_models_through_0861.parquet"
                ),
                "era_allowlist_path": (
                    "numerai/agents/experiments/ender21_residual_stability_v53/"
                    "protocol/discovery_eras_through_0861.json"
                ),
                "require_benchmark_coverage": True,
            },
            "model": {
                "params": {
                    "loss_mode": loss_mode,
                    "seed": 2027,
                }
            },
            "training": {
                "max_train_samples": 500_000,
                "sample_seed": 1337,
                "cv": {
                    "enabled": True,
                    "n_splits": 5,
                    "embargo": 13,
                    "mode": "expanding",
                    "min_train_size": 0,
                },
            },
        }

    def _write_model_seed_pair(self, root: Path, mutate=None) -> None:
        configs = root / "configs"
        configs.mkdir(parents=True)
        control = self._paired_config(loss_mode="mse")
        selected = self._paired_config(loss_mode="chronological_block_dro")
        if mutate is not None:
            mutate(control, selected)
        paths = {
            "r2_control_tabm_k64_model_seed2027": control,
            "r2_selected_tabm_k64_block_dro_model_seed2027": selected,
        }
        for name, config in paths.items():
            (configs / f"{name}.py").write_text(
                "CONFIG = " + repr(config) + "\n", encoding="utf-8"
            )

    def test_validate_pair_rejects_wrong_seed_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def mismatch(_control, selected):
                selected["model"]["params"]["seed"] = 1337

            self._write_model_seed_pair(root, mismatch)
            with self.assertRaisesRegex(ValueError, "frozen loss|seeds"):
                self.evaluator._validate_config_pair(root, "model_seed2027")

    def test_validate_pair_rejects_blocked_cv_even_when_pair_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def blocked(control, selected):
                control["training"]["cv"]["mode"] = "blocked"
                selected["training"]["cv"]["mode"] = "blocked"

            self._write_model_seed_pair(root, blocked)
            with self.assertRaisesRegex(ValueError, "CV|training"):
                self.evaluator._validate_config_pair(root, "model_seed2027")

    def test_validate_pair_rejects_disabled_benchmark_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def uncovered(control, selected):
                control["data"]["require_benchmark_coverage"] = False
                selected["data"]["require_benchmark_coverage"] = False

            self._write_model_seed_pair(root, uncovered)
            with self.assertRaisesRegex(ValueError, "coverage|data"):
                self.evaluator._validate_config_pair(root, "model_seed2027")


class TestEnder21Round2Completion(unittest.TestCase):
    NAME = "r2_control_tabm_k64_model_seed2027"

    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluator = _load_evaluator()

    @staticmethod
    def _experiment(root: Path) -> Path:
        return (
            root
            / "numerai/agents/experiments/ender21_residual_stability_v53"
        )

    def _write_completion_fixture(
        self,
        root: Path,
    ) -> tuple[Path, dict, dict, Path]:
        experiment = self._experiment(root)
        for relative in ("configs", "predictions", "results", "receipts"):
            (experiment / relative).mkdir(parents=True, exist_ok=True)

        config_path = experiment / f"configs/{self.NAME}.py"
        config_bytes = b"CONFIG = {'synthetic': True}\n"
        config_path.write_bytes(config_bytes)
        config_relative = config_path.relative_to(root).as_posix()
        manifest = {
            "git_head": "a" * 40,
            "files": {
                config_relative: hashlib.sha256(config_bytes).hexdigest(),
            },
        }
        manifest_path = experiment / "source_manifest_round2.json"
        manifest_path.write_bytes(
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )

        output_paths = {
            "predictions": experiment / f"predictions/{self.NAME}.parquet",
            "result": experiment / f"results/{self.NAME}.json",
        }
        output_paths["predictions"].write_bytes(b"synthetic parquet bytes")
        output_paths["result"].write_bytes(b'{"synthetic": true}\n')
        outputs = {}
        for label, path in output_paths.items():
            inspected = path.lstat()
            outputs[label] = {
                "path": str(path),
                "device": int(inspected.st_dev),
                "inode": int(inspected.st_ino),
                "size_bytes": int(inspected.st_size),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        payload = {
            "schema_version": 1,
            "stage": "ender21-round2-training-completion",
            "state": "OUTPUTS_FINALIZED",
            "component": self.NAME,
            "manifest": {
                "path": manifest_path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "git_head": manifest["git_head"],
            },
            "config": {
                "path": config_relative,
                "sha256": manifest["files"][config_relative],
            },
            "outputs": outputs,
        }
        completion_path = experiment / f"receipts/{self.NAME}.completion.json"
        completion_path.write_bytes(
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        return experiment, manifest, payload, completion_path

    def test_writer_accepts_relative_config_and_hashes_actual_pretty_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self._experiment(root)
            for relative in ("configs", "predictions", "results", "receipts"):
                (experiment / relative).mkdir(parents=True, exist_ok=True)
            config_path = experiment / f"configs/{self.NAME}.py"
            config_bytes = b"CONFIG = {'synthetic': True}\n"
            config_path.write_bytes(config_bytes)
            relative_config = config_path.relative_to(root)
            config_relative = relative_config.as_posix()
            manifest = {
                "git_head": "a" * 40,
                "files": {
                    config_relative: hashlib.sha256(config_bytes).hexdigest(),
                },
            }
            manifest_path = experiment / "source_manifest_round2.json"
            pretty_manifest = (
                json.dumps(manifest, indent=4, sort_keys=True).encode("utf-8")
                + b"\n"
            )
            manifest_path.write_bytes(pretty_manifest)
            predictions = experiment / f"predictions/{self.NAME}.parquet"
            result = experiment / f"results/{self.NAME}.json"
            completion = experiment / f"receipts/{self.NAME}.completion.json"

            previous_cwd = Path.cwd()
            prediction_bytes = b"prediction bytes"
            result_bytes = b"result bytes"
            try:
                os.chdir(root)
                with mock.patch.object(pipeline_module, "REPO_DIR", root):
                    with pipeline_module._ExclusiveOutputReservations(
                        predictions,
                        result,
                        completion,
                    ) as reservations:
                        reservations.predictions_stream.write(prediction_bytes)
                        reservations.results_stream.write(result_bytes)
                        path, payload, payload_bytes = (
                            pipeline_module._write_ender21_round2_completion(
                                relative_config,
                                manifest,
                                reservations,
                            )
                        )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(path, completion)
            self.assertEqual(completion.read_bytes(), payload_bytes)
            self.assertEqual(
                payload_bytes,
                json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
                + b"\n",
            )
            self.assertEqual(
                payload["outputs"]["predictions"]["sha256"],
                hashlib.sha256(prediction_bytes).hexdigest(),
            )
            self.assertEqual(
                payload["outputs"]["predictions"]["size_bytes"],
                len(prediction_bytes),
            )
            self.assertEqual(
                payload["outputs"]["result"]["sha256"],
                hashlib.sha256(result_bytes).hexdigest(),
            )
            self.assertEqual(
                payload["outputs"]["result"]["size_bytes"],
                len(result_bytes),
            )
            self.assertEqual(
                payload["manifest"]["sha256"],
                hashlib.sha256(pretty_manifest).hexdigest(),
            )
            canonical_manifest = json.dumps(
                manifest,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            self.assertNotEqual(
                payload["manifest"]["sha256"],
                hashlib.sha256(canonical_manifest).hexdigest(),
            )
            self.assertEqual(payload["config"]["path"], config_relative)

    def test_preexisting_completion_stops_third_reservation_before_inputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self._experiment(root)
            configs = experiment / "configs"
            receipts = experiment / "receipts"
            configs.mkdir(parents=True)
            receipts.mkdir()
            config = configs / f"{self.NAME}.py"
            config.write_text("CONFIG = {}\n", encoding="utf-8")
            completion = receipts / f"{self.NAME}.completion.json"
            original_completion = b"existing completion receipt"
            completion.write_bytes(original_completion)

            with mock.patch.object(
                pipeline_module, "REPO_DIR", root
            ), mock.patch.object(
                pipeline_module, "_verify_ender21_round2_manifest"
            ) as manifest_verifier, mock.patch.object(
                pipeline_module,
                "load_config",
                side_effect=AssertionError("CONFIG_EVALUATED"),
            ) as config_loader, mock.patch.object(
                pipeline_module,
                "NumerAPI",
                side_effect=AssertionError("DATA_CLIENT_OPENED"),
            ) as data_client:
                with self.assertRaisesRegex(
                    ValueError,
                    "Cannot reserve exclusive completion receipt output",
                ):
                    run_training(config)

            manifest_verifier.assert_not_called()
            config_loader.assert_not_called()
            data_client.assert_not_called()
            self.assertEqual(completion.read_bytes(), original_completion)
            prediction = experiment / f"predictions/{self.NAME}.parquet"
            result = experiment / f"results/{self.NAME}.json"
            self.assertTrue(prediction.is_file())
            self.assertTrue(result.is_file())
            self.assertEqual(prediction.stat().st_size, 0)
            self.assertEqual(result.stat().st_size, 0)

    def test_validator_rejects_cross_manifest_and_payload_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment, manifest, payload, completion = (
                self._write_completion_fixture(root)
            )
            with mock.patch.object(self.evaluator, "REPO_DIR", root):
                self.assertEqual(
                    self.evaluator._validate_completion(
                        experiment,
                        self.NAME,
                        manifest,
                    ),
                    payload,
                )

                cross_manifest = deepcopy(manifest)
                cross_manifest["git_head"] = "b" * 40
                with self.assertRaisesRegex(ValueError, "provenance differs"):
                    self.evaluator._validate_completion(
                        experiment,
                        self.NAME,
                        cross_manifest,
                    )

                cases = (
                    (
                        "manifest_hash",
                        lambda value: value["manifest"].__setitem__(
                            "sha256", "0" * 64
                        ),
                        "provenance differs",
                    ),
                    (
                        "config_hash",
                        lambda value: value["config"].__setitem__(
                            "sha256", "1" * 64
                        ),
                        "provenance differs",
                    ),
                    (
                        "prediction_hash",
                        lambda value: value["outputs"]["predictions"].__setitem__(
                            "sha256", "2" * 64
                        ),
                        "predictions differs from its artifact",
                    ),
                    (
                        "result_size",
                        lambda value: value["outputs"]["result"].__setitem__(
                            "size_bytes",
                            value["outputs"]["result"]["size_bytes"] + 1,
                        ),
                        "result differs from its artifact",
                    ),
                )
                for label, mutate, message in cases:
                    with self.subTest(label=label):
                        tampered = deepcopy(payload)
                        mutate(tampered)
                        completion.write_bytes(
                            json.dumps(tampered, indent=2, sort_keys=True).encode(
                                "utf-8"
                            )
                            + b"\n"
                        )
                        with self.assertRaisesRegex(ValueError, message):
                            self.evaluator._validate_completion(
                                experiment,
                                self.NAME,
                                manifest,
                            )


if __name__ == "__main__":
    unittest.main()
