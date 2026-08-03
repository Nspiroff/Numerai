from __future__ import annotations

import copy
import unittest

import numpy as np
import pandas as pd

from agents.code.analysis import evaluate_ender20_hybrid_stability as hybrid


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
        "benchmark_only": _metrics(
            corr_mean=0.02, bmc_mean=0.0, benchmark_similarity=1.0
        ),
        "residual_only": _metrics(
            corr_mean=0.01, bmc_mean=0.01, benchmark_similarity=0.10
        ),
    }
    for candidate in hybrid.CANDIDATE_COLUMNS:
        calibration[candidate] = _metrics(bmc_drawdown=0.20)
    calibration["hybrid_w55"] = _metrics(
        corr_mean=0.018, bmc_mean=0.006, bmc_sharpe=0.60, bmc_drawdown=0.10
    )
    calibration["hybrid_w65"] = _metrics(
        corr_mean=0.019, bmc_mean=0.007, bmc_sharpe=0.65, bmc_drawdown=0.16
    )

    holdout = copy.deepcopy(calibration)
    holdout["residual_only"] = _metrics(bmc_mean=0.01)
    holdout["benchmark_only"] = _metrics(corr_mean=0.02)
    holdout["hybrid_w55"] = _metrics(
        corr_mean=0.018, bmc_mean=0.004, bmc_sharpe=0.40, bmc_drawdown=0.10
    )

    full = copy.deepcopy(calibration)
    full["hybrid_w55"] = _metrics(
        corr_mean=0.01,
        bmc_mean=0.005,
        bmc_sharpe=0.50,
        bmc_drawdown=0.14,
        benchmark_similarity=0.80,
    )
    return {"calibration": calibration, "holdout": holdout, "full": full}


class HybridSignalTests(unittest.TestCase):
    def test_rank_blend_uses_average_percentile_ranks_and_final_rerank(self):
        frame = pd.DataFrame(
            {
                hybrid.ERA_COLUMN: ["0001", "0001", "0001"],
                hybrid.BENCHMARK_COLUMN: [10.0, 20.0, 30.0],
                hybrid.RESIDUAL_COLUMN: [30.0, 10.0, 20.0],
            }
        )

        result = hybrid.rank_hybrid_signals(frame, {"hybrid_w50": 0.50})

        np.testing.assert_allclose(
            result["benchmark_only"], [1.0 / 3.0, 2.0 / 3.0, 1.0]
        )
        np.testing.assert_allclose(
            result["residual_only"], [1.0, 1.0 / 3.0, 2.0 / 3.0]
        )
        np.testing.assert_allclose(
            result["hybrid_w50"], [2.0 / 3.0, 1.0 / 3.0, 1.0]
        )

    def test_rank_blend_is_independent_by_era(self):
        frame = pd.DataFrame(
            {
                hybrid.ERA_COLUMN: ["0001", "0001", "0002", "0002"],
                hybrid.BENCHMARK_COLUMN: [0.0, 1.0, 100.0, 200.0],
                hybrid.RESIDUAL_COLUMN: [1.0, 0.0, 200.0, 100.0],
            }
        )

        result = hybrid.rank_hybrid_signals(frame, {"hybrid_w50": 0.50})

        self.assertEqual(result.groupby(hybrid.ERA_COLUMN)["benchmark_only"].min().tolist(), [0.5, 0.5])
        self.assertEqual(result.groupby(hybrid.ERA_COLUMN)["benchmark_only"].max().tolist(), [1.0, 1.0])

    def test_prediction_similarity_is_symmetric_and_self_similarity_is_one(self):
        frame = pd.DataFrame(
            {
                hybrid.ERA_COLUMN: ["0001"] * 4 + ["0002"] * 4,
                hybrid.BENCHMARK_COLUMN: [1, 2, 3, 4, 10, 30, 20, 40],
                "benchmark_only": [1, 2, 3, 4, 10, 30, 20, 40],
                "same_order": [2, 4, 6, 8, 100, 300, 200, 400],
                "reverse_order": [4, 3, 2, 1, 40, 20, 30, 10],
            }
        )

        scores = hybrid.per_era_rank_similarity(
            frame,
            ["benchmark_only", "same_order", "reverse_order"],
            hybrid.BENCHMARK_COLUMN,
        )

        np.testing.assert_allclose(scores["benchmark_only"], [1.0, 1.0])
        np.testing.assert_allclose(scores["same_order"], [1.0, 1.0])
        np.testing.assert_allclose(scores["reverse_order"], [-1.0, -1.0])


class HybridGateTests(unittest.TestCase):
    def test_selection_uses_calibration_only_and_rejects_drawdown_failure(self):
        summaries = _summaries()

        selected, evaluations = hybrid.select_calibration_candidate(summaries)

        self.assertEqual(selected, "hybrid_w55")
        self.assertTrue(evaluations["hybrid_w55"]["eligible"])
        self.assertFalse(evaluations["hybrid_w65"]["eligible"])
        self.assertFalse(
            evaluations["hybrid_w65"]["checks"]["bmc_max_drawdown"]
        )

    def test_promotion_gate_passes_inclusive_retention_and_strict_limits(self):
        summaries = _summaries()

        checks = hybrid.promotion_checks("hybrid_w55", summaries, coverage_ok=True)

        self.assertTrue(all(checks.values()))
        self.assertTrue(checks["holdout_benchmark_corr_retention"])

        summaries["full"]["hybrid_w55"]["bmc"]["max_drawdown"] = 0.15
        checks = hybrid.promotion_checks("hybrid_w55", summaries, coverage_ok=True)
        self.assertFalse(checks["full_bmc_max_drawdown"])

    def test_no_calibration_candidate_returns_no_selection(self):
        summaries = _summaries()
        for candidate in hybrid.CANDIDATE_COLUMNS:
            summaries["calibration"][candidate]["bmc"]["mean"] = -0.001

        selected, evaluations = hybrid.select_calibration_candidate(summaries)

        self.assertIsNone(selected)
        self.assertFalse(any(row["eligible"] for row in evaluations.values()))


if __name__ == "__main__":
    unittest.main()
