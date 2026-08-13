"""Governed scientific orchestration for Ender25's Ender24 recovery."""

from __future__ import annotations

from pathlib import Path

from ender25_recovery_common import (
    FAMILY,
    NEGATIVE_STATE,
    PAIR_NAMES,
    POSITIVE_STATE,
    ROUND1_NAMES,
    STAGE,
)


def authorization_fields(passed: bool) -> dict:
    """Return the recovery's deliberately narrow authority envelope."""

    if type(passed) is not bool:
        raise TypeError("Ender25 passed must be a bool.")
    return {
        "state": POSITIVE_STATE if passed else NEGATIVE_STATE,
        "round2_source_gate_authorized": passed,
        "round2_authorized": False,
        "training_authorized": False,
        "scoring_authorized": False,
        "deployment_authorized": False,
        "account_actions_authorized": False,
    }


def _exact_config_pair_delta(
    old_experiment: Path,
    control_name: str,
    ema_name: str,
    custody,
    frozen_common,
) -> bool:
    control, _ = frozen_common.validate_config(
        old_experiment, control_name, custody
    )
    ema, _ = frozen_common.validate_config(old_experiment, ema_name, custody)
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
    custody,
    frozen_common,
    *,
    completions: dict | None = None,
    authority_bundle: tuple | None = None,
) -> dict:
    """Recover the one Ender24 decision with an explicit opaque barrier.

    `preflight_completions` is intentionally the first evidence parser. Only
    after all four completion envelopes pass may authority, truth, results, or
    prediction payloads be parsed.
    """

    if completions is None:
        completions = custody.preflight_completions(frozen_common)
    if authority_bundle is None:
        authority_bundle = custody.load_authority(frozen_common)
    authority, allowed, _features = authority_bundle
    truth = custody.load_truth(allowed, frozen_common)
    records = {}
    frames = {}
    for name in ROUND1_NAMES:
        records[name], frames[name] = custody.score_candidate(
            name,
            allowed,
            truth,
            completions[name],
            frozen_common,
        )

    if len({record["parameter_count"] for record in records.values()}) != 1:
        raise ValueError("Ender24 parameter count differs across recovered runs.")

    old_experiment = experiment.parent / "ender24_ema_seed_stability_v53"
    matched_pairs = {}
    for seed, (control_name, ema_name) in PAIR_NAMES.items():
        control_frame = frames[control_name]
        ema_frame = frames[ema_name]
        columns = ["id", "era", "target_ender_20", "v53_lgbm_ender20", "cv_fold"]
        control_cohort = control_frame.loc[:, columns].sort_values(
            "id", kind="mergesort"
        )
        ema_cohort = ema_frame.loc[:, columns].sort_values(
            "id", kind="mergesort"
        )
        checks = {
            "exact_cohort": control_cohort.reset_index(drop=True).equals(
                ema_cohort.reset_index(drop=True)
            ),
            "exact_common_provenance": (
                records[control_name]["provenance"]["manifest"]
                == records[ema_name]["provenance"]["manifest"]
            ),
            "exact_config_delta": _exact_config_pair_delta(
                old_experiment,
                control_name,
                ema_name,
                custody,
                frozen_common,
            ),
        }
        if not all(checks.values()):
            raise ValueError(f"Seed {seed} recovered matched-pair contract differs.")
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
                "sha256": frozen_common.cohort_sha256(control_frame),
            },
            "config_delta": ["model.params.ema_decay", "output.results_name"],
            "checks": checks,
        }

    control_names = tuple(pair[0] for pair in PAIR_NAMES.values())
    ema_names = tuple(pair[1] for pair in PAIR_NAMES.values())
    aggregates = {
        "control": frozen_common.procedure_aggregates(records, control_names),
        "ema995": frozen_common.procedure_aggregates(records, ema_names),
    }
    aggregate_decisions = frozen_common.aggregate_checks(
        aggregates["control"], aggregates["ema995"]
    )
    ema_run_checks = {
        ema_name: frozen_common.per_ema_checks(
            records[ema_name]["metrics"], records[control_name]["metrics"]
        )
        for control_name, ema_name in PAIR_NAMES.values()
    }
    passed = all(aggregate_decisions.values()) and all(
        all(checks.values()) for checks in ema_run_checks.values()
    )
    return {
        "schema_version": 1,
        "family": FAMILY,
        "stage": STAGE,
        **authorization_fields(passed),
        "scientific_decision": True,
        "reused_training": True,
        "evaluator_retry": False,
        "inputs": {
            "recovery_manifest": custody.receipt(
                experiment / "source_manifest_evaluation_recovery.json"
            ),
            "ender24_manifest": custody.receipt(
                old_experiment / "source_manifest_round1.json"
            ),
            "ender24_postmortem": custody.receipt(
                old_experiment / "receipts/round1_execution_postmortem.json"
            ),
            "ender24_discovery_authority": custody.receipt(
                old_experiment / "protocol/discovery_data_authority.json"
            ),
            "discovery_full": custody.receipt(
                numerai_dir / "v5.3/ender21_discovery_full_through_0861.parquet"
            ),
            "discovery_benchmark": custody.receipt(
                numerai_dir
                / "v5.3/ender21_discovery_benchmark_models_through_0861.parquet"
            ),
            "canonical_text_authority": {
                "era_allowlist": authority["era_allowlist"],
                "feature_columns": authority["feature_columns"],
            },
        },
        "runs": records,
        "matched_pairs": matched_pairs,
        "aggregates": aggregates,
        "aggregate_checks": aggregate_decisions,
        "ema_run_checks": ema_run_checks,
        "passed": passed,
    }


def run_bootstrapped(
    experiment: Path,
    numerai_dir: Path,
    custody,
    frozen_common,
    decision,
    *,
    completions: dict,
    authority_bundle: tuple,
) -> dict:
    payload = evaluate(
        experiment,
        numerai_dir,
        custody,
        frozen_common,
        completions=completions,
        authority_bundle=authority_bundle,
    )
    decision.commit_json(payload)
    return payload
