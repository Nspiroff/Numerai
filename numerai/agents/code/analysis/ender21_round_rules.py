"""Frozen metric rules shared by Ender21 discovery rounds."""

from __future__ import annotations


def matched_eligibility_checks(
    selected_metrics: dict,
    control_metrics: dict,
) -> dict:
    """Apply the exact predeclared Round-1 checks to one matched pair."""

    return {
        "positive_full_bmc": selected_metrics["bmc"]["mean"] > 0.0,
        "positive_recent_fold_bmc": selected_metrics["recent_fold_bmc_mean"] > 0.0,
        "corr_guardrail": 0.005 <= selected_metrics["corr"]["mean"] < 0.04,
        "full_bmc_retention": selected_metrics["bmc"]["mean"]
        >= 0.90 * control_metrics["bmc"]["mean"],
        "recent_bmc_retention": selected_metrics["recent_fold_bmc_mean"]
        >= 0.90 * control_metrics["recent_fold_bmc_mean"],
        "drawdown_improvement": selected_metrics["bmc"]["max_drawdown"]
        <= 0.85 * control_metrics["bmc"]["max_drawdown"],
        "sharpe_retention": selected_metrics["bmc"]["sharpe"]
        >= control_metrics["bmc"]["sharpe"] - 0.05,
        "all_folds_positive": all(
            value > 0.0
            for value in selected_metrics["fold_bmc_mean"].values()
        ),
    }
