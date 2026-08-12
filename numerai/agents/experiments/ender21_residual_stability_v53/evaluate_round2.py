"""Evaluate the frozen Ender21 Round-2 seed replications exactly once."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import runpy

from agents.code.modeling.utils.constants import REPO_DIR
from agents.code.modeling.utils.pipeline import _verify_ender21_round2_manifest


EXPERIMENT_NAME = "ender21_residual_stability_v53"
REALIZATIONS = {
    "base_seed1337": (
        "r1_control_tabm_k64",
        "r1_tabm_k64_block_dro",
    ),
    "model_seed2027": (
        "r2_control_tabm_k64_model_seed2027",
        "r2_selected_tabm_k64_block_dro_model_seed2027",
    ),
    "sample_seed2027": (
        "r2_control_tabm_k64_sample_seed2027",
        "r2_selected_tabm_k64_block_dro_sample_seed2027",
    ),
}


def _matched_checks(selected_metrics: dict, control_metrics: dict) -> dict:
    def at_least(actual: float, threshold: float) -> bool:
        return actual > threshold or math.isclose(
            actual, threshold, rel_tol=1e-12, abs_tol=1e-15
        )

    def at_most(actual: float, threshold: float) -> bool:
        return actual < threshold or math.isclose(
            actual, threshold, rel_tol=1e-12, abs_tol=1e-15
        )

    return {
        "positive_full_bmc": selected_metrics["bmc"]["mean"] > 0.0,
        "positive_recent_fold_bmc": selected_metrics["recent_fold_bmc_mean"] > 0.0,
        "corr_guardrail": 0.005 <= selected_metrics["corr"]["mean"] < 0.04,
        "full_bmc_retention": at_least(
            selected_metrics["bmc"]["mean"],
            0.90 * control_metrics["bmc"]["mean"],
        ),
        "recent_bmc_retention": at_least(
            selected_metrics["recent_fold_bmc_mean"],
            0.90 * control_metrics["recent_fold_bmc_mean"],
        ),
        "drawdown_improvement": at_most(
            selected_metrics["bmc"]["max_drawdown"],
            0.85 * control_metrics["bmc"]["max_drawdown"],
        ),
        "sharpe_retention": at_least(
            selected_metrics["bmc"]["sharpe"],
            control_metrics["bmc"]["sharpe"] - 0.05,
        ),
        "all_folds_positive": all(
            value > 0.0
            for value in selected_metrics["fold_bmc_mean"].values()
        ),
    }


def _decide(realizations: dict) -> dict:
    if not isinstance(realizations, dict) or set(realizations) != set(REALIZATIONS):
        raise ValueError("Round-2 decision requires exactly three named realizations.")
    if any(
        not isinstance(item, dict) or type(item.get("passed")) is not bool
        for item in realizations.values()
    ):
        raise ValueError("Round-2 realization pass states must be exact booleans.")
    passed_count = sum(
        bool(item["passed"]) for item in realizations.values()
    )
    passed = passed_count >= 2
    return {
        "passed_count": passed_count,
        "required_count": 2,
        "passed": passed,
        "state": "SEED_REPLICATION_PASS" if passed else "NEGATIVE",
    }


def _validate_config_pair(experiment: Path, realization: str) -> None:
    control_name, selected_name = REALIZATIONS[realization]
    control = runpy.run_path(str(experiment / f"configs/{control_name}.py"))["CONFIG"]
    selected = runpy.run_path(str(experiment / f"configs/{selected_name}.py"))["CONFIG"]
    expected_data = {
        "data_version": "v5.3",
        "full_data_path": "v5.3/ender21_discovery_full_through_0861.parquet",
        "benchmark_data_path": (
            "v5.3/ender21_discovery_benchmark_models_through_0861.parquet"
        ),
        "era_allowlist_path": (
            "numerai/agents/experiments/ender21_residual_stability_v53/"
            "protocol/discovery_eras_through_0861.json"
        ),
        "require_benchmark_coverage": True,
    }
    for label, config in (("control", control), ("selected", selected)):
        data = config.get("data", {})
        if any(data.get(key) != value for key, value in expected_data.items()):
            raise ValueError(f"{realization} {label} data/coverage contract differs")
        training = config.get("training", {})
        if training.get("max_train_samples") != 500_000 or training.get("cv") != {
            "enabled": True,
            "n_splits": 5,
            "embargo": 13,
            "mode": "expanding",
            "min_train_size": 0,
        }:
            raise ValueError(f"{realization} {label} CV/training contract differs")
    comparable_control_training = dict(control["training"])
    comparable_selected_training = dict(selected["training"])
    comparable_control_training.pop("sample_seed", None)
    comparable_selected_training.pop("sample_seed", None)
    if comparable_control_training != comparable_selected_training:
        raise ValueError(f"{realization} matched training settings differ")
    control_params = control["model"]["params"]
    selected_params = selected["model"]["params"]
    expected_differences = {
        "loss_mode": ("mse", "chronological_block_dro")
    }
    differences = {
        key: (control_params.get(key), selected_params.get(key))
        for key in set(control_params) | set(selected_params)
        if control_params.get(key) != selected_params.get(key)
    }
    if differences != expected_differences:
        raise ValueError(f"{realization} model pair differs beyond the frozen loss")
    expected_model_seed = 2027 if realization == "model_seed2027" else 1337
    expected_sample_seed = 2027 if realization == "sample_seed2027" else 1337
    if (
        control_params["seed"] != expected_model_seed
        or selected_params["seed"] != expected_model_seed
        or control["training"]["sample_seed"] != expected_sample_seed
        or selected["training"]["sample_seed"] != expected_sample_seed
    ):
        raise ValueError(f"{realization} seeds differ from the frozen pairing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--numerai-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected_experiment = Path(
        os.path.abspath(REPO_DIR / "numerai/agents/experiments" / EXPERIMENT_NAME)
    )
    experiment = Path(os.path.abspath(args.experiment))
    if experiment != expected_experiment:
        raise ValueError("Round-2 evaluator requires the canonical experiment.")
    numerai_dir = Path(os.path.abspath(args.numerai_dir))
    if numerai_dir != Path(os.path.abspath(REPO_DIR / "numerai")):
        raise ValueError("Round-2 evaluator requires the canonical data root.")
    output = Path(os.path.abspath(args.output))
    if output != experiment / "receipts/round2_seed_replication.json":
        raise ValueError("Round-2 evaluator output path differs from the freeze.")
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to overwrite Round-2 receipt: {output}")

    _verify_ender21_round2_manifest()
    round1 = json.loads(
        (experiment / "receipts/round1_discovery.json").read_text(encoding="utf-8")
    )
    if round1.get("state") != "SCOUT_WINNER" or round1.get("selected") != (
        "r1_tabm_k64_block_dro"
    ):
        raise ValueError("Round-1 authority does not select the frozen family.")

    round1_module = runpy.run_path(str(experiment / "evaluate_round1.py"))
    score_candidate = round1_module["_score_candidate"]
    allowed = json.loads(
        (experiment / "protocol/discovery_eras_through_0861.json").read_text(
            encoding="utf-8"
        )
    )
    import pandas as pd

    full = pd.read_parquet(
        numerai_dir / "v5.3/ender21_discovery_full_through_0861.parquet",
        columns=["id", "era", "target_ender_20"],
    )
    benchmark = pd.read_parquet(
        numerai_dir
        / "v5.3/ender21_discovery_benchmark_models_through_0861.parquet",
        columns=["id", "era", "v53_lgbm_ender20"],
    )
    full["era"] = full["era"].astype(str)
    benchmark["era"] = benchmark["era"].astype(str)
    truth = full.loc[full["era"].isin(set(allowed))].merge(
        benchmark.loc[benchmark["era"].isin(set(allowed))],
        on=["id", "era"],
        how="inner",
        validate="one_to_one",
    )

    realizations = {}
    for realization, (control_name, selected_name) in REALIZATIONS.items():
        _validate_config_pair(experiment, realization)
        if realization == "base_seed1337":
            control = round1["candidates"][control_name]
            selected = round1["candidates"][selected_name]
        else:
            control = score_candidate(experiment, control_name, allowed, truth)
            selected = score_candidate(experiment, selected_name, allowed, truth)
        checks = _matched_checks(selected["metrics"], control["metrics"])
        realizations[realization] = {
            "control": control,
            "selected": selected,
            "checks": checks,
            "passed": all(checks.values()),
        }
    decision = _decide(realizations)
    payload = {
        "schema_version": 1,
        "stage": "ender21-round2-seed-replication",
        **decision,
        "round1_receipt": round1,
        "realizations": realizations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: payload[key] for key in ("state", "passed_count")}))


if __name__ == "__main__":
    main()
