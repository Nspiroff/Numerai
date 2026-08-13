"""Governed implementation for Ender23 Round-1 evaluation."""

from __future__ import annotations

from pathlib import Path

from evaluation_common import (
    CONTROL,
    DecisionReservation,
    EvaluationCustody,
    ROUND1_CANDIDATES,
    challenger_checks,
    load_governed_dependencies,
    load_authority,
    load_truth,
    receipt,
    score_candidate,
    verify_governed_manifest,
)


def evaluate(
    experiment: Path,
    numerai_dir: Path,
    custody: EvaluationCustody,
) -> dict:
    manifest = custody.manifest
    _, allowed = load_authority(experiment, numerai_dir, custody)
    truth = load_truth(numerai_dir, allowed, custody)
    records = {}
    frames = {}
    for name in ROUND1_CANDIDATES:
        records[name], frames[name] = score_candidate(
            experiment, name, allowed, truth, manifest, 1, custody
        )
    base = frames[CONTROL]
    for name, frame in frames.items():
        if (
            len(frame) != len(base)
            or not frame[["id", "era", "target_ender_20", "cv_fold"]].equals(
                base[["id", "era", "target_ender_20", "cv_fold"]]
            )
        ):
            raise ValueError(f"{name} is not on the exact matched control cohort")
    control_metrics = records[CONTROL]["metrics"]
    decisions = {}
    for name in ROUND1_CANDIDATES[1:]:
        checks = challenger_checks(records[name]["metrics"], control_metrics)
        decisions[name] = {"checks": checks, "eligible": all(checks.values())}
    eligible = [name for name, decision in decisions.items() if decision["eligible"]]
    eligible.sort(
        key=lambda name: (
            -records[name]["metrics"]["recent40_bmc_mean"],
            -min(records[name]["metrics"]["recent_blocks_bmc_mean"].values()),
            -records[name]["metrics"]["bmc"]["mean"],
            records[name]["metrics"]["bmc"]["max_drawdown"],
            name,
        )
    )
    selected = eligible[0] if eligible else None
    return {
        "schema_version": 1,
        "stage": "ender23-round1-discovery",
        "state": "SCOUT_WINNER" if selected else "NEGATIVE_NO_TEMPORAL_RETENTION_GAIN",
        "selected": selected,
        "inputs": {
            "authority": receipt(experiment / "protocol/discovery_data_authority.json", custody),
            "full": receipt(numerai_dir / "v5.3/ender21_discovery_full_through_0861.parquet", custody),
            "benchmark": receipt(numerai_dir / "v5.3/ender21_discovery_benchmark_models_through_0861.parquet", custody),
        },
        "candidates": records,
        "decisions": decisions,
    }


def run_bootstrapped(
    repo_dir: Path,
    experiment: Path,
    numerai_dir: Path,
    decision: DecisionReservation,
) -> dict:
    """Evaluate only inside the already-reserved, verified bootstrap envelope."""

    with EvaluationCustody(repo_dir, 1) as custody:
        load_governed_dependencies()
        verify_governed_manifest(custody)
        payload = evaluate(experiment, numerai_dir, custody)
        decision.commit_json(payload)
    print(f"state={payload['state']} selected={payload['selected']}")
    return payload
