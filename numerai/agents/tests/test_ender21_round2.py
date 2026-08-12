from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import runpy
import tempfile
import unittest


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


if __name__ == "__main__":
    unittest.main()
