"""Governed implementation for Ender23 Round-2 evaluation."""

from __future__ import annotations

from pathlib import Path

import evaluation_common as common

from evaluation_common import (
    CONTROL,
    DecisionReservation,
    EvaluationCustody,
    ROUND1_CANDIDATES,
    ROUND2_BY_SELECTED,
    challenger_checks,
    compute_metrics,
    load_governed_dependencies,
    load_authority,
    load_truth,
    receipt,
    replication_checks,
    score_candidate,
    verify_governed_manifest,
)


def evaluate(
    experiment: Path,
    numerai_dir: Path,
    custody: EvaluationCustody,
) -> dict:
    manifest = custody.manifest
    round1_manifest = custody.read_json(experiment / "source_manifest_round1.json")
    round1_path = experiment / "receipts/round1_discovery.json"
    round1 = custody.read_json(round1_path)
    _, allowed = load_authority(experiment, numerai_dir, custody)
    truth = load_truth(numerai_dir, allowed, custody)

    round1_records = {}
    round1_frames = {}
    for name in ROUND1_CANDIDATES:
        round1_records[name], round1_frames[name] = score_candidate(
            experiment, name, allowed, truth, round1_manifest, 1, custody
        )
    control = round1_records[CONTROL]
    round1_decisions = {}
    for name in ROUND1_CANDIDATES[1:]:
        checks = challenger_checks(round1_records[name]["metrics"], control["metrics"])
        round1_decisions[name] = {"checks": checks, "eligible": all(checks.values())}
    eligible = [name for name, decision in round1_decisions.items() if decision["eligible"]]
    eligible.sort(
        key=lambda name: (
            -round1_records[name]["metrics"]["recent40_bmc_mean"],
            -min(round1_records[name]["metrics"]["recent_blocks_bmc_mean"].values()),
            -round1_records[name]["metrics"]["bmc"]["mean"],
            round1_records[name]["metrics"]["bmc"]["max_drawdown"],
            name,
        )
    )
    selected = eligible[0] if eligible else None
    expected_inputs = {
        "authority": receipt(experiment / "protocol/discovery_data_authority.json", custody),
        "full": receipt(numerai_dir / "v5.3/ender21_discovery_full_through_0861.parquet", custody),
        "benchmark": receipt(numerai_dir / "v5.3/ender21_discovery_benchmark_models_through_0861.parquet", custody),
    }
    if (
        set(round1) != {"schema_version", "stage", "state", "selected", "inputs", "candidates", "decisions"}
        or round1["schema_version"] != 1
        or round1["stage"] != "ender23-round1-discovery"
        or round1["state"] != "SCOUT_WINNER"
        or selected not in ROUND2_BY_SELECTED
        or round1["selected"] != selected
        or round1["inputs"] != expected_inputs
        or round1["candidates"] != round1_records
        or round1["decisions"] != round1_decisions
    ):
        raise ValueError("Round-1 decision does not exactly match independent recomputation")
    base_selected = round1_records[selected]

    realization_names = (selected, *ROUND2_BY_SELECTED[selected])
    records = {selected: base_selected}
    frames = {selected: round1_frames[selected]}
    for name in ROUND2_BY_SELECTED[selected]:
        records[name], frames[name] = score_candidate(
            experiment, name, allowed, truth, manifest, 2, custody
        )
    base_frame = frames[selected]
    for name, frame in frames.items():
        if (
            len(frame) != len(base_frame)
            or not frame[["id", "era", "target_ender_20", "cv_fold", "v53_lgbm_ender20"]].equals(
                base_frame[["id", "era", "target_ender_20", "cv_fold", "v53_lgbm_ender20"]]
            )
        ):
            raise ValueError(f"{name} is not on the exact matched selected cohort")

    realizations = {}
    for name in realization_names:
        metrics = records[name]["metrics"]
        checks = replication_checks(metrics, control["metrics"])
        realizations[name] = {
            "record": records[name],
            "checks": checks,
            "passed": all(checks.values()),
        }
    passed_count = sum(item["passed"] for item in realizations.values())

    ensemble = base_frame[["id", "era", "target_ender_20", "cv_fold", "v53_lgbm_ender20"]].copy()
    ranked_columns = []
    for index, name in enumerate(realization_names):
        column = f"rank_{index}"
        predictions = frames[name].set_index("id").loc[ensemble["id"], "prediction"].to_numpy()
        temporary = common.pd.DataFrame({"era": ensemble["era"].to_numpy(), "prediction": predictions})
        ensemble[column] = temporary.groupby("era", sort=False)["prediction"].rank(
            method="average", pct=True
        ).to_numpy()
        ranked_columns.append(column)
    ensemble["prediction"] = ensemble[ranked_columns].mean(axis=1)
    ensemble_metrics = compute_metrics(ensemble)
    ensemble_checks = challenger_checks(ensemble_metrics, control["metrics"])
    ensemble_passed = all(ensemble_checks.values())
    passed = passed_count >= 2 and ensemble_passed
    return {
        "schema_version": 1,
        "stage": "ender23-round2-seed-replication",
        "state": (
            "HISTORICAL_RESEARCH_PASS_FORWARD_VALIDATION_REQUIRED"
            if passed
            else "NEGATIVE_SEED_INSTABILITY"
        ),
        "selected": selected,
        "passed_count": passed_count,
        "required_count": 2,
        "individual_requirement_passed": passed_count >= 2,
        "ensemble": {
            "definition": "equal mean of three within-era average percentile ranks; no rerank",
            "metrics": ensemble_metrics,
            "checks": ensemble_checks,
            "passed": ensemble_passed,
        },
        "round1_receipt": receipt(round1_path, custody),
        "realizations": realizations,
    }


def run_bootstrapped(
    repo_dir: Path,
    experiment: Path,
    numerai_dir: Path,
    decision: DecisionReservation,
) -> dict:
    """Evaluate only inside the already-reserved, verified bootstrap envelope."""

    with EvaluationCustody(repo_dir, 2) as custody:
        load_governed_dependencies()
        verify_governed_manifest(custody)
        payload = evaluate(experiment, numerai_dir, custody)
        decision.commit_json(payload)
    print(f"state={payload['state']} passed_count={payload['passed_count']}")
    return payload
