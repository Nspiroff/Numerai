from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from agents.code.analysis import evaluate_ender20_seed_ensemble_stability as seed_gate
from agents.code.analysis import evaluate_ender20_hybrid_stability as single


def _metrics(
    *,
    corr_mean: float = 0.02,
    bmc_mean: float = 0.006,
    bmc_sharpe: float = 0.60,
    bmc_drawdown: float = 0.10,
    benchmark_similarity: float = 0.80,
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
        "avg_corr_with_benchmark": benchmark_similarity,
    }


def _summaries() -> dict:
    calibration = {
        "benchmark_only": _metrics(corr_mean=0.02, bmc_mean=0.0, benchmark_similarity=1.0),
        "seed1337_only": _metrics(corr_mean=0.01, bmc_mean=0.009),
        "seed2027_only": _metrics(corr_mean=0.011, bmc_mean=0.008),
        "two_seed_residual": _metrics(corr_mean=0.012, bmc_mean=0.01),
    }
    for candidate in seed_gate.CANDIDATE_COLUMNS:
        calibration[candidate] = _metrics(bmc_drawdown=0.20)
    calibration["two_seed_hybrid_w35"] = _metrics(
        corr_mean=0.018, bmc_mean=0.006, bmc_sharpe=0.60, bmc_drawdown=0.10
    )
    calibration["two_seed_hybrid_w45"] = _metrics(
        corr_mean=0.019, bmc_mean=0.007, bmc_sharpe=0.65, bmc_drawdown=0.16
    )
    holdout = copy.deepcopy(calibration)
    holdout["benchmark_only"] = _metrics(corr_mean=0.02, benchmark_similarity=1.0)
    holdout["two_seed_residual"] = _metrics(bmc_mean=0.01)
    holdout["two_seed_hybrid_w35"] = _metrics(
        corr_mean=0.018, bmc_mean=0.004, bmc_sharpe=0.40, bmc_drawdown=0.10
    )
    full = copy.deepcopy(calibration)
    full["two_seed_hybrid_w35"] = _metrics(
        corr_mean=0.01,
        bmc_mean=0.005,
        bmc_sharpe=0.50,
        bmc_drawdown=0.14,
    )
    return {"calibration": calibration, "holdout": holdout, "full": full}


class TwoSeedSignalTests(unittest.TestCase):
    def test_equal_rank_ensemble_and_final_rerank(self):
        frame = pd.DataFrame(
            {
                single.ERA_COLUMN: ["0001", "0001", "0001"],
                single.BENCHMARK_COLUMN: [1.0, 2.0, 3.0],
                seed_gate.SEED_1337_RAW: [1.0, 2.0, 3.0],
                seed_gate.SEED_2027_RAW: [3.0, 1.0, 2.0],
            }
        )

        result = seed_gate.rank_two_seed_signals(
            frame, {"two_seed_hybrid_w50": 0.50}
        )

        np.testing.assert_allclose(result["seed1337_only"], [1 / 3, 2 / 3, 1])
        np.testing.assert_allclose(result["seed2027_only"], [1, 1 / 3, 2 / 3])
        np.testing.assert_allclose(result["two_seed_residual"], [2 / 3, 1 / 3, 1])
        self.assertTrue(result["two_seed_hybrid_w50"].between(0.0, 1.0).all())

    def test_reference_similarity_is_exactly_one(self):
        frame = pd.DataFrame(
            {
                single.ERA_COLUMN: ["0001", "0001", "0001", "0001"],
                single.BENCHMARK_COLUMN: [4.0, 1.0, 3.0, 2.0],
                "benchmark_only": [4.0, 1.0, 3.0, 2.0],
            }
        )
        similarity = single.per_era_rank_similarity(
            frame, ["benchmark_only"], single.BENCHMARK_COLUMN
        )
        np.testing.assert_allclose(similarity["benchmark_only"], [1.0])


class TwoSeedGateTests(unittest.TestCase):
    def test_selection_uses_two_seed_residual_reference(self):
        selected, evaluations = seed_gate.select_calibration_candidate(_summaries())

        self.assertEqual(selected, "two_seed_hybrid_w35")
        self.assertTrue(evaluations["two_seed_hybrid_w35"]["eligible"])
        self.assertFalse(evaluations["two_seed_hybrid_w45"]["eligible"])
        self.assertFalse(
            evaluations["two_seed_hybrid_w45"]["checks"]["bmc_max_drawdown"]
        )

    def test_promotion_gate_obeys_strict_full_drawdown(self):
        summaries = _summaries()
        checks = seed_gate.promotion_checks(
            "two_seed_hybrid_w35", summaries, coverage_ok=True
        )
        self.assertTrue(all(checks.values()))

        summaries["full"]["two_seed_hybrid_w35"]["bmc"]["max_drawdown"] = 0.15
        checks = seed_gate.promotion_checks(
            "two_seed_hybrid_w35", summaries, coverage_ok=True
        )
        self.assertFalse(checks["full_bmc_max_drawdown"])

    def test_no_eligible_candidate_stops_before_holdout(self):
        summaries = _summaries()
        for candidate in seed_gate.CANDIDATE_COLUMNS:
            summaries["calibration"][candidate]["bmc"]["mean"] = -0.001

        selected, evaluations = seed_gate.select_calibration_candidate(summaries)

        self.assertIsNone(selected)
        self.assertFalse(any(row["eligible"] for row in evaluations.values()))

    def test_exact_tie_break_prefers_drawdown_then_lower_weight(self):
        summaries = _summaries()
        w35 = summaries["calibration"]["two_seed_hybrid_w35"]
        w45 = summaries["calibration"]["two_seed_hybrid_w45"]
        w45["bmc"]["mean"] = w35["bmc"]["mean"]
        w45["bmc"]["max_drawdown"] = 0.09
        w45["corr"]["mean"] = 0.019

        selected, _ = seed_gate.select_calibration_candidate(summaries)
        self.assertEqual(selected, "two_seed_hybrid_w45")

        w45["bmc"]["max_drawdown"] = w35["bmc"]["max_drawdown"]
        selected, _ = seed_gate.select_calibration_candidate(summaries)
        self.assertEqual(selected, "two_seed_hybrid_w35")

    def test_shell_exit_code_enforces_stop_rule(self):
        self.assertEqual(seed_gate.decision_exit_code({"promotion_eligible": True}), 0)
        self.assertEqual(seed_gate.decision_exit_code({"promotion_eligible": False}), 2)
        self.assertEqual(seed_gate.decision_exit_code({}), 2)

    def test_frozen_gate_and_wrapper_hashes_are_pinned(self):
        repo_root = single._repo_root()
        gate_path = (
            repo_root
            / "numerai/agents/experiments/ender20_seed_ensemble_stability_v53/gate.md"
        )
        config_path = (
            repo_root
            / "numerai/agents/experiments/ender20_nn_architecture_v53/configs/"
            "scale_disk_tabm_k64_train500k_seed2027.py"
        )
        seed_gate.validate_frozen_gate_and_config(gate_path, config_path)

        with tempfile.TemporaryDirectory() as directory:
            altered_gate = Path(directory) / "gate.md"
            altered_gate.write_text("altered\n", encoding="utf-8")
            with self.assertRaises(single.HybridEvaluationError):
                seed_gate.validate_frozen_gate_and_config(altered_gate, config_path)


if __name__ == "__main__":
    unittest.main()
