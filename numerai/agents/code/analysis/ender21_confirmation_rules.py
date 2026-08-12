"""Exact frozen eligibility rules for Ender21 historical confirmation."""

from __future__ import annotations

import math


DISCOVERY_BMC_BASELINE = 0.006876950492356912
BMC_MIN = 0.0020
BMC_SHARPE_MIN = 0.25
BMC_MAX_DRAWDOWN_MAX = 0.10
CORR_MIN = 0.008
BENCHMARK_CORR_MAX = 0.25
POSITIVE_BLOCKS_MIN = 3
WORST_BLOCK_BMC_MIN = -0.001
DISCOVERY_RETENTION_MIN = 0.60
EXPECTED_BLOCK_KEYS = frozenset({"0", "1", "2", "3"})


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number, not a boolean.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def confirmation_checks(metrics: dict, discovery_bmc: float) -> dict[str, bool]:
    """Apply all frozen confirmation thresholds without tolerance or rounding."""

    if not isinstance(metrics, dict) or set(metrics) != {
        "bmc",
        "corr",
        "avg_corr_with_benchmark",
        "chronological_block_bmc",
    }:
        raise ValueError("Confirmation metrics have an unexpected schema.")
    bmc = metrics["bmc"]
    corr = metrics["corr"]
    blocks = metrics["chronological_block_bmc"]
    if not isinstance(bmc, dict) or set(bmc) != {
        "mean",
        "sharpe",
        "max_drawdown",
    }:
        raise ValueError("Confirmation BMC summary has an unexpected schema.")
    if not isinstance(corr, dict) or set(corr) != {"mean"}:
        raise ValueError("Confirmation Corr summary has an unexpected schema.")
    if not isinstance(blocks, dict) or set(blocks) != EXPECTED_BLOCK_KEYS:
        raise ValueError("Confirmation requires exactly chronological blocks 0-3.")

    bmc_values = {
        key: _finite_number(value, f"bmc.{key}") for key, value in bmc.items()
    }
    corr_values = {
        key: _finite_number(value, f"corr.{key}") for key, value in corr.items()
    }
    benchmark_corr = _finite_number(
        metrics["avg_corr_with_benchmark"], "avg_corr_with_benchmark"
    )
    block_values = {
        key: _finite_number(value, f"chronological_block_bmc.{key}")
        for key, value in blocks.items()
    }
    discovery = _finite_number(discovery_bmc, "discovery_bmc")
    if discovery <= 0.0:
        raise ValueError("discovery_bmc must be positive.")

    return {
        "bmc_floor": bmc_values["mean"] >= BMC_MIN,
        "sharpe_floor": bmc_values["sharpe"] > BMC_SHARPE_MIN,
        "drawdown_ceiling": bmc_values["max_drawdown"] < BMC_MAX_DRAWDOWN_MAX,
        "corr_floor": corr_values["mean"] >= CORR_MIN,
        "benchmark_corr_ceiling": benchmark_corr < BENCHMARK_CORR_MAX,
        "positive_block_count": sum(value > 0.0 for value in block_values.values())
        >= POSITIVE_BLOCKS_MIN,
        "worst_block_floor": min(block_values.values()) > WORST_BLOCK_BMC_MIN,
        "discovery_bmc_retention": bmc_values["mean"]
        >= DISCOVERY_RETENTION_MIN * discovery,
    }


__all__ = [
    "DISCOVERY_BMC_BASELINE",
    "confirmation_checks",
]
