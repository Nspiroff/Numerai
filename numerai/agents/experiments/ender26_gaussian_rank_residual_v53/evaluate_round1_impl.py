"""Governed implementation for Ender26 Round-1 evaluation."""

from __future__ import annotations

from pathlib import Path

from evaluation_common import (
    DecisionReservation,
    EvaluationCustody,
    PAIR_NAMES,
    ROUND1_NAMES,
    aggregate_checks,
    candidate_run_checks,
    cohort_sha256,
    load_authority,
    load_governed_dependencies,
    load_truth,
    preflight_all_completions,
    procedure_aggregates,
    receipt,
    score_candidate,
    validate_config,
    verify_governed_manifest,
)


def _exact_config_pair_delta(
    experiment: Path,
    control_name: str,
    challenger_name: str,
    custody: EvaluationCustody | None,
) -> bool:
    """Prove the pair differs only by benchmark transform and output name."""

    control, _ = validate_config(experiment, control_name, custody)
    challenger, _ = validate_config(experiment, challenger_name, custody)
    control = {
        **control,
        "output": {**control["output"], "results_name": "<matched>"},
        "model": {
            **control["model"],
            "target_transform": dict(control["model"]["target_transform"]),
        },
    }
    challenger = {
        **challenger,
        "output": {**challenger["output"], "results_name": "<matched>"},
        "model": {
            **challenger["model"],
            "target_transform": dict(challenger["model"]["target_transform"]),
        },
    }
    control_mode = control["model"]["target_transform"].pop(
        "benchmark_transform", "identity"
    )
    challenger_mode = challenger["model"]["target_transform"].pop(
        "benchmark_transform", None
    )
    return (
        control_mode == "identity"
        and challenger_mode == "tie_kept_rank_gaussian"
        and control == challenger
    )


def evaluate(
    experiment: Path,
    numerai_dir: Path,
    custody: EvaluationCustody,
) -> dict:
    manifest = custody.manifest

    # All four completions are still opaque here. Nothing target-bearing or
    # parseable beyond those completion envelopes is opened before this call.
    completions = preflight_all_completions(experiment, manifest, custody)
    _, allowed = load_authority(experiment, numerai_dir, custody)
    truth = load_truth(numerai_dir, allowed, custody)
    records = {}
    frames = {}
    for name in ROUND1_NAMES:
        records[name], frames[name] = score_candidate(
            experiment,
            name,
            allowed,
            truth,
            completions[name],
            custody,
        )

    if len({record["parameter_count"] for record in records.values()}) != 1:
        raise ValueError("Ender26 parameter count differs across runs")

    matched_pairs = {}
    for seed, (control_name, challenger_name) in PAIR_NAMES.items():
        control_frame = frames[control_name]
        challenger_frame = frames[challenger_name]
        columns = ["id", "era", "target_ender_20", "v53_lgbm_ender20", "cv_fold"]
        control_cohort = control_frame.loc[:, columns].sort_values(
            "id", kind="mergesort"
        )
        challenger_cohort = challenger_frame.loc[:, columns].sort_values(
            "id", kind="mergesort"
        )
        exact_cohort = control_cohort.reset_index(drop=True).equals(
            challenger_cohort.reset_index(drop=True)
        )
        exact_config_delta = _exact_config_pair_delta(
            experiment, control_name, challenger_name, custody
        )
        common_provenance = (
            records[control_name]["provenance"]["manifest"]
            == records[challenger_name]["provenance"]["manifest"]
        )
        checks = {
            "exact_cohort": exact_cohort,
            "exact_common_provenance": common_provenance,
            "exact_config_delta": exact_config_delta,
        }
        if not all(checks.values()):
            raise ValueError(f"Seed {seed} matched-pair contract differs")
        matched_pairs[seed] = {
            "control": control_name,
            "challenger": challenger_name,
            "model_seed": int(seed),
            "sample_seed": 1337,
            "cohort": {
                "rows": len(control_cohort),
                "eras": int(control_cohort["era"].nunique()),
                "first_era": min(control_cohort["era"], key=int),
                "last_era": max(control_cohort["era"], key=int),
                "sha256": cohort_sha256(control_frame),
            },
            "config_delta": [
                "model.target_transform.benchmark_transform",
                "output.results_name",
            ],
            "checks": checks,
        }

    control_names = tuple(pair[0] for pair in PAIR_NAMES.values())
    challenger_names = tuple(pair[1] for pair in PAIR_NAMES.values())
    aggregates = {
        "control": procedure_aggregates(records, control_names),
        "gaussian_rank_residual": procedure_aggregates(
            records, challenger_names
        ),
    }
    aggregate_decisions = aggregate_checks(
        aggregates["control"], aggregates["gaussian_rank_residual"]
    )
    challenger_checks = {
        challenger_name: candidate_run_checks(
            records[challenger_name]["metrics"],
            records[control_name]["metrics"],
        )
        for control_name, challenger_name in PAIR_NAMES.values()
    }
    passed = all(aggregate_decisions.values()) and all(
        all(checks.values()) for checks in challenger_checks.values()
    )
    return {
        "schema_version": 1,
        "stage": "ender26-round1-bmc-alignment",
        "state": (
            "ENDER26_ROUND2_SOURCE_GATE_ELIGIBLE"
            if passed
            else "ENDER26_NEGATIVE_NO_BMC_ALIGNMENT_GAIN"
        ),
        "round2_source_gate_eligible": passed,
        "continuation_authorized": False,
        "round2_source_gate_authorized": False,
        "round2_authorized": False,
        "training_authorized": False,
        "scoring_authorized": False,
        "deployment_authorized": False,
        "submission_authorized": False,
        "account_actions_authorized": False,
        "inputs": {
            "manifest": receipt(
                experiment / "source_manifest_round1.json", custody
            ),
            "authority": receipt(
                experiment / "protocol/discovery_data_authority.json", custody
            ),
            "full": receipt(
                numerai_dir / "v5.3/ender21_discovery_full_through_0861.parquet",
                custody,
            ),
            "benchmark": receipt(
                numerai_dir
                / "v5.3/ender21_discovery_benchmark_models_through_0861.parquet",
                custody,
            ),
        },
        "runs": records,
        "matched_pairs": matched_pairs,
        "aggregates": aggregates,
        "aggregate_checks": aggregate_decisions,
        "challenger_run_checks": challenger_checks,
        "passed": passed,
    }


def run_bootstrapped(
    repo_dir: Path,
    experiment: Path,
    numerai_dir: Path,
    decision: DecisionReservation,
    *,
    manifest: dict,
    manifest_bytes: bytes,
    bootstrap_leases: tuple,
) -> dict:
    with EvaluationCustody(
        repo_dir,
        1,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        bootstrap_leases=bootstrap_leases,
    ) as custody:
        load_governed_dependencies()
        verify_governed_manifest(custody)
        payload = evaluate(experiment, numerai_dir, custody)
        decision.commit_json(payload)
    print(f"state={payload['state']}")
    return payload
