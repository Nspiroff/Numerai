"""Governed implementation for Ender24 Round-1 evaluation."""

from __future__ import annotations

from pathlib import Path

from evaluation_common import (
    DecisionReservation,
    EvaluationCustody,
    PAIR_NAMES,
    ROUND1_NAMES,
    aggregate_checks,
    cohort_sha256,
    load_authority,
    load_governed_dependencies,
    load_truth,
    per_ema_checks,
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
    ema_name: str,
    custody: EvaluationCustody,
) -> bool:
    control, _ = validate_config(experiment, control_name, custody)
    ema, _ = validate_config(experiment, ema_name, custody)
    control = {
        **control,
        "output": {**control["output"], "results_name": "<matched>"},
    }
    ema = {
        **ema,
        "output": {**ema["output"], "results_name": "<matched>"},
        "model": {
            **ema["model"],
            "params": {
                key: value
                for key, value in ema["model"]["params"].items()
                if key != "ema_decay"
            },
        },
    }
    return control == ema


def evaluate(
    experiment: Path,
    numerai_dir: Path,
    custody: EvaluationCustody,
) -> dict:
    manifest = custody.manifest
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
        raise ValueError("Ender24 parameter count differs across runs")

    matched_pairs = {}
    for seed, (control_name, ema_name) in PAIR_NAMES.items():
        control_frame = frames[control_name]
        ema_frame = frames[ema_name]
        columns = ["id", "era", "target_ender_20", "v53_lgbm_ender20", "cv_fold"]
        control_cohort = control_frame.loc[:, columns].sort_values("id", kind="mergesort")
        ema_cohort = ema_frame.loc[:, columns].sort_values("id", kind="mergesort")
        exact_cohort = control_cohort.reset_index(drop=True).equals(
            ema_cohort.reset_index(drop=True)
        )
        exact_config_delta = _exact_config_pair_delta(
            experiment, control_name, ema_name, custody
        )
        common_provenance = (
            records[control_name]["provenance"]["manifest"]
            == records[ema_name]["provenance"]["manifest"]
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
            "ema": ema_name,
            "model_seed": int(seed),
            "sample_seed": 1337,
            "cohort": {
                "rows": len(control_cohort),
                "eras": int(control_cohort["era"].nunique()),
                "first_era": min(control_cohort["era"], key=int),
                "last_era": max(control_cohort["era"], key=int),
                "sha256": cohort_sha256(control_frame),
            },
            "config_delta": ["model.params.ema_decay", "output.results_name"],
            "checks": checks,
        }

    control_names = tuple(pair[0] for pair in PAIR_NAMES.values())
    ema_names = tuple(pair[1] for pair in PAIR_NAMES.values())
    aggregates = {
        "control": procedure_aggregates(records, control_names),
        "ema995": procedure_aggregates(records, ema_names),
    }
    aggregate_decisions = aggregate_checks(aggregates["control"], aggregates["ema995"])
    ema_run_checks = {
        ema_name: per_ema_checks(
            records[ema_name]["metrics"], records[control_name]["metrics"]
        )
        for control_name, ema_name in PAIR_NAMES.values()
    }
    passed = all(aggregate_decisions.values()) and all(
        all(checks.values()) for checks in ema_run_checks.values()
    )
    return {
        "schema_version": 1,
        "stage": "ender24-round1-ema-seed-stability",
        "state": (
            "ROUND2_AUTHORIZED" if passed else "NEGATIVE_NO_EMA_STABILITY_GAIN"
        ),
        "round2_authorized": passed,
        "inputs": {
            "manifest": receipt(experiment / "source_manifest_round1.json", custody),
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
        "ema_run_checks": ema_run_checks,
        "passed": passed,
    }


def run_bootstrapped(
    repo_dir: Path,
    experiment: Path,
    numerai_dir: Path,
    decision: DecisionReservation,
) -> dict:
    with EvaluationCustody(repo_dir, 1) as custody:
        load_governed_dependencies()
        verify_governed_manifest(custody)
        payload = evaluate(experiment, numerai_dir, custody)
        decision.commit_json(payload)
    print(f"state={payload['state']}")
    return payload
