"""Evaluate the frozen Ender21 Round-2 seed replications exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy

from agents.code.modeling.utils.constants import REPO_DIR
from agents.code.modeling.utils.pipeline import _verify_ender21_round2_manifest
from agents.code.analysis.ender21_round_rules import matched_eligibility_checks


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
    return matched_eligibility_checks(selected_metrics, control_metrics)


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


def _validate_completion(
    experiment: Path,
    name: str,
    manifest: dict,
) -> dict:
    path = experiment / f"receipts/{name}.completion.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "stage",
        "state",
        "component",
        "manifest",
        "config",
        "outputs",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError(f"{name} completion schema differs")
    if (
        payload["schema_version"] != 1
        or payload["stage"] != "ender21-round2-training-completion"
        or payload["state"] != "OUTPUTS_FINALIZED"
        or payload["component"] != name
    ):
        raise ValueError(f"{name} completion envelope differs")
    manifest_path = experiment / "source_manifest_round2.json"
    expected_manifest = {
        "path": manifest_path.relative_to(REPO_DIR).as_posix(),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "git_head": manifest["git_head"],
    }
    config_relative = (
        experiment / f"configs/{name}.py"
    ).relative_to(REPO_DIR).as_posix()
    if payload["manifest"] != expected_manifest or payload["config"] != {
        "path": config_relative,
        "sha256": manifest["files"][config_relative],
    }:
        raise ValueError(f"{name} completion provenance differs")
    outputs = payload["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != {"predictions", "result"}:
        raise ValueError(f"{name} completion output set differs")
    for label, expected_path in (
        ("predictions", experiment / f"predictions/{name}.parquet"),
        ("result", experiment / f"results/{name}.json"),
    ):
        receipt = outputs[label]
        if not isinstance(receipt, dict) or set(receipt) != {
            "path",
            "device",
            "inode",
            "size_bytes",
            "sha256",
        }:
            raise ValueError(f"{name} completion {label} schema differs")
        inspected = expected_path.lstat()
        if (
            receipt["path"] != str(expected_path)
            or int(receipt["device"]) != int(inspected.st_dev)
            or int(receipt["inode"]) != int(inspected.st_ino)
            or int(receipt["size_bytes"]) != int(inspected.st_size)
            or receipt["sha256"] != _sha256_file(expected_path)
        ):
            raise ValueError(f"{name} completion {label} differs from its artifact")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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

    manifest = _verify_ender21_round2_manifest()
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
        control_completion = None
        selected_completion = None
        if realization == "base_seed1337":
            control = round1["candidates"][control_name]
            selected = round1["candidates"][selected_name]
        else:
            control_completion = _validate_completion(
                experiment, control_name, manifest
            )
            selected_completion = _validate_completion(
                experiment, selected_name, manifest
            )
            control = score_candidate(experiment, control_name, allowed, truth)
            selected = score_candidate(experiment, selected_name, allowed, truth)
        checks = _matched_checks(selected["metrics"], control["metrics"])
        realizations[realization] = {
            "control": control,
            "selected": selected,
            "control_completion": control_completion,
            "selected_completion": selected_completion,
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
