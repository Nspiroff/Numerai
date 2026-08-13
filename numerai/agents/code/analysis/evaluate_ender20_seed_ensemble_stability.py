"""Evaluate the frozen Ender20 two-seed stability experiment.

The evaluator is training-free. It validates the seed-1337 and seed-2027 OOF
artifacts against the exact feature-store manifest, builds only the equal-rank
ensemble and five benchmark blends frozen in the experiment gate, selects on
the first 655 eras, and evaluates the locked final 200 eras.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from agents.code.analysis import evaluate_ender20_deployment_gate as frozen
from agents.code.analysis import evaluate_ender20_hybrid_stability as single
from agents.code.metrics import numerai_metrics


EXPERIMENT_NAME = "ender20_seed_ensemble_stability_v53"
AS_OF_DATE = "2026-08-03"

SEED_1337_RAW = "_seed1337_raw"
SEED_2027_RAW = "_seed2027_raw"
REFERENCE_COLUMNS = (
    "benchmark_only",
    "seed1337_only",
    "seed2027_only",
    "two_seed_residual",
)
WEIGHTS: dict[str, float] = {
    "two_seed_hybrid_w35": 0.35,
    "two_seed_hybrid_w45": 0.45,
    "two_seed_hybrid_w55": 0.55,
    "two_seed_hybrid_w65": 0.65,
    "two_seed_hybrid_w75": 0.75,
}
CANDIDATE_COLUMNS = tuple(WEIGHTS)
ALL_SIGNAL_COLUMNS = (*REFERENCE_COLUMNS, *CANDIDATE_COLUMNS)

EXPECTED_SEED_CONFIG_SHA256 = (
    "3f952a926136f1d810fbf0b3cacb59e026b1156ddf3402b384c9d12b895f98dd"
)
EXPECTED_GATE_SHA256 = (
    "3097219be90c4fb49d07a461c129e8121942e49a74d9af75397cfe7eec841cc4"
)
PRETRAINING_GIT_COMMIT = "8964a1b51813a352a526fccd7270d23fed834d0d"
CHECKPOINT_TRAINING_PATHS = (
    "numerai/agents/code/modeling",
    "numerai/agents/code/metrics/numerai_metrics.py",
    "numerai/agents/experiments/ender20_nn_architecture_v53/configs",
    "numerai/agents/experiments/ender20_seed_ensemble_stability_v53/gate.md",
)


def load_checkpoint_frozen_source(
    repo_root: Path,
    numerai_root: Path,
) -> frozen.FrozenSource:
    """Bind current training code to the pre-run commit and data to its old anchor."""

    commit_exists = subprocess.run(
        ["git", "cat-file", "-e", f"{PRETRAINING_GIT_COMMIT}^{{commit}}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    single._require(
        commit_exists.returncode == 0,
        "Pre-training Git checkpoint is not present locally.",
    )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PRETRAINING_GIT_COMMIT, "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    single._require(
        ancestor.returncode == 0,
        "Pre-training Git checkpoint is not an ancestor of current HEAD.",
    )
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", PRETRAINING_GIT_COMMIT, "--", *CHECKPOINT_TRAINING_PATHS],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    single._require(
        unchanged.returncode == 0,
        "Tracked training/config/gate paths differ from the pre-training checkpoint.",
    )

    source_manifest_path = (numerai_root / frozen.DEFAULT_SOURCE_MANIFEST).resolve()
    single._require(
        frozen._sha256_file(source_manifest_path) == frozen.SOURCE_MANIFEST_SHA256,
        "Historical data source manifest does not match its anchored hash.",
    )
    source_manifest = frozen._load_json_object(
        source_manifest_path, "historical data source manifest"
    )
    metadata_path = (numerai_root / frozen.STORE_METADATA_RELATIVE).resolve()
    metadata_key = str(frozen.STORE_METADATA_RELATIVE).replace("\\", "/")
    single._require(
        frozen._sha256_file(metadata_path) == source_manifest["files"][metadata_key],
        "Feature-store metadata does not match the predeclared data anchor.",
    )
    metadata = frozen._load_json_object(metadata_path, "feature-store metadata")
    single._require(metadata.get("complete") is True, "Feature-store generation is incomplete.")
    single._require(
        metadata.get("target_column") == frozen.TARGET_COLUMN
        and metadata.get("benchmark_column") == frozen.BENCHMARK_COLUMN,
        "Feature-store target or benchmark differs from the frozen contract.",
    )
    manifest_meta = metadata.get("manifest")
    feature_meta = metadata.get("features")
    single._require(
        isinstance(manifest_meta, dict) and isinstance(feature_meta, dict),
        "Feature-store artifact metadata is malformed.",
    )
    store_manifest_path = (metadata_path.parent / str(manifest_meta.get("filename", ""))).resolve()
    feature_path = (metadata_path.parent / str(feature_meta.get("filename", ""))).resolve()
    single._require(
        store_manifest_path.parent == metadata_path.parent
        and feature_path.parent == metadata_path.parent,
        "Feature-store artifact filename escapes the anchored directory.",
    )
    single._require(
        store_manifest_path.is_file()
        and store_manifest_path.stat().st_size == int(manifest_meta.get("size_bytes", -1))
        and frozen._sha256_file(store_manifest_path) == manifest_meta.get("sha256"),
        "Feature-store manifest payload differs from anchored metadata.",
    )
    single._require(
        feature_path.is_file()
        and feature_path.stat().st_size == int(feature_meta.get("size_bytes", -1)),
        "Feature-store feature payload size differs from anchored metadata.",
    )

    configs: dict[str, dict[str, Any]] = {}
    for run_name, spec in frozen.RUN_SPECS.items():
        config_path = frozen._safe_repo_path(numerai_root, spec["config"])
        namespace = runpy.run_path(str(config_path))
        config = namespace.get("CONFIG", namespace.get("config"))
        single._require(
            isinstance(config, dict),
            f"Checkpoint config {run_name} has no CONFIG mapping.",
        )
        configs[run_name] = config
    return frozen.FrozenSource(
        repo_root=numerai_root,
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        source_manifest_sha256=frozen.SOURCE_MANIFEST_SHA256,
        store_metadata_path=metadata_path,
        store_metadata=metadata,
        store_manifest_path=store_manifest_path,
        configs=configs,
        recorded_commit_present=True,
    )


def load_and_validate_two_seed_cohort(
    seed1337_path: Path,
    seed1337_result_path: Path,
    seed2027_path: Path,
    seed2027_result_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Build both signals on the independently derived frozen OOF cohort."""

    repo_root = single._repo_root()
    numerai_root = repo_root / "numerai"
    source = load_checkpoint_frozen_source(repo_root, numerai_root)
    single._require(
        source.store_manifest_path.resolve() == manifest_path.resolve(),
        "Supplied manifest is not the source-manifest-pinned feature-store artifact.",
    )
    single._require(
        source.source_manifest["files"][
            "agents/experiments/ender20_nn_architecture_v53/configs/"
            "scale_disk_tabm_k64_train500k_seed2027.py"
        ]
        == EXPECTED_SEED_CONFIG_SHA256,
        "Anchored source manifest has an unexpected seed-2027 config hash.",
    )
    expected = frozen.build_expected_oof_cohort(source)
    run_specs = (
        (
            "scale_disk_tabm_k64_train500k",
            seed1337_result_path,
            seed1337_path,
            SEED_1337_RAW,
        ),
        (
            "scale_disk_tabm_k64_train500k_seed2027",
            seed2027_result_path,
            seed2027_path,
            SEED_2027_RAW,
        ),
    )
    raw_signals: dict[str, np.ndarray] = {}
    semantics: dict[str, dict[str, Any]] = {}
    artifact_reports: dict[str, dict[str, Any]] = {}
    for run_name, result_path, prediction_path, output_name in run_specs:
        run = frozen.RunArtifact(run_name, result_path.resolve(), prediction_path.resolve())
        result = frozen._load_json_object(result_path, f"{run_name} result JSON")
        expected_semantics = frozen.validate_result_json(
            run,
            result,
            source.configs[run_name],
            source,
            expected,
        )
        raw, _ranked, report = frozen.validate_prediction_artifact(
            prediction_path, expected.frame, expected_semantics
        )
        raw_signals[output_name] = raw
        semantics[run_name] = expected_semantics
        artifact_reports[run_name] = report

    cohort = expected.frame.copy()
    cohort[SEED_1337_RAW] = raw_signals[SEED_1337_RAW]
    cohort[SEED_2027_RAW] = raw_signals[SEED_2027_RAW]
    eras = sorted(cohort[single.ERA_COLUMN].astype(str).unique().tolist(), key=int)
    single._require(
        len(cohort) == single.EXPECTED_ROWS
        and len(eras) == single.EXPECTED_ERAS
        and eras[0] == single.EXPECTED_FIRST_ERA
        and eras[-1] == single.EXPECTED_LAST_ERA,
        "Anchored OOF cohort does not match the frozen 5,112,039-row era range.",
    )
    single._require(
        eras[single.CALIBRATION_ERAS - 1] == "1025"
        and eras[single.CALIBRATION_ERAS] == "1026",
        "Anchored OOF cohort does not preserve the frozen 1025/1026 split boundary.",
    )
    del raw_signals
    gc.collect()
    return cohort, eras, {
        "source_manifest_sha256": source.source_manifest_sha256,
        "historical_data_git_head": source.source_manifest["git_head"],
        "pretraining_git_commit": PRETRAINING_GIT_COMMIT,
        "store_generation_id": source.store_metadata["generation_id"],
        "semantics": semantics,
        "artifacts": artifact_reports,
    }


def rank_two_seed_signals(
    frame: pd.DataFrame,
    weights: Mapping[str, float] = WEIGHTS,
) -> pd.DataFrame:
    """Build the frozen two-seed rank ensemble and benchmark hybrids."""

    required = {
        single.ERA_COLUMN,
        single.BENCHMARK_COLUMN,
        SEED_1337_RAW,
        SEED_2027_RAW,
    }
    single._require(
        required.issubset(frame.columns),
        f"Missing ensemble columns: {sorted(required - set(frame.columns))}",
    )
    result = frame.copy()
    grouped = result.groupby(single.ERA_COLUMN, sort=False, observed=True)
    result["benchmark_only"] = grouped[single.BENCHMARK_COLUMN].rank(
        method="average", pct=True
    )
    result["seed1337_only"] = grouped[SEED_1337_RAW].rank(
        method="average", pct=True
    )
    result["seed2027_only"] = grouped[SEED_2027_RAW].rank(
        method="average", pct=True
    )
    residual_average = 0.5 * (
        result["seed1337_only"] + result["seed2027_only"]
    )
    result["two_seed_residual"] = residual_average.groupby(
        result[single.ERA_COLUMN], sort=False
    ).rank(method="average", pct=True)
    for name, residual_weight in weights.items():
        single._require(
            0.0 < float(residual_weight) < 1.0,
            f"Invalid residual weight: {residual_weight}",
        )
        raw_score = (
            (1.0 - float(residual_weight)) * result["benchmark_only"]
            + float(residual_weight) * result["two_seed_residual"]
        )
        result[name] = raw_score.groupby(
            result[single.ERA_COLUMN], sort=False
        ).rank(method="average", pct=True)
    generated = [*REFERENCE_COLUMNS, *tuple(weights)]
    values = result[generated].to_numpy(dtype=np.float64)
    single._require(np.isfinite(values).all(), "Generated signals contain non-finite values.")
    single._require(
        ((values >= 0.0) & (values <= 1.0)).all(),
        "Generated signals leave [0, 1].",
    )
    return result


def _validate_scores(
    scores: pd.DataFrame,
    eras: Sequence[str],
    label: str,
) -> pd.DataFrame:
    result = scores.copy()
    result.index = result.index.astype(str)
    result = result.loc[list(eras), list(ALL_SIGNAL_COLUMNS)]
    single._require(result.index.tolist() == list(eras), f"{label} eras are out of order.")
    single._require(
        np.isfinite(result.to_numpy(dtype=np.float64)).all(),
        f"{label} contains non-finite values.",
    )
    return result


def compute_per_era_metrics(
    cohort: pd.DataFrame,
    eras: Sequence[str],
) -> dict[str, pd.DataFrame]:
    corr = numerai_metrics.per_era_corr(
        cohort, ALL_SIGNAL_COLUMNS, single.TARGET_COLUMN, single.ERA_COLUMN
    )
    bmc = numerai_metrics.per_era_bmc(
        cohort,
        ALL_SIGNAL_COLUMNS,
        single.BENCHMARK_COLUMN,
        single.TARGET_COLUMN,
        single.ERA_COLUMN,
    )
    similarity = single.per_era_rank_similarity(
        cohort,
        ALL_SIGNAL_COLUMNS,
        single.BENCHMARK_COLUMN,
        single.ERA_COLUMN,
    )
    benchmark_self_similarity = similarity["benchmark_only"].to_numpy(
        dtype=np.float64
    )
    single._require(
        np.allclose(
            benchmark_self_similarity,
            np.ones_like(benchmark_self_similarity),
            rtol=0.0,
            atol=1e-12,
        ),
        "Benchmark self-similarity is not exactly one within tolerance.",
    )
    return {
        "corr": _validate_scores(corr, eras, "Per-era Corr"),
        "bmc": _validate_scores(bmc, eras, "Per-era BMC"),
        "benchmark_corr": _validate_scores(
            similarity, eras, "Per-era benchmark similarity"
        ),
    }


def summarize_segments(
    per_era: Mapping[str, pd.DataFrame],
    eras: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    segments = {
        "calibration": list(eras[: single.CALIBRATION_ERAS]),
        "holdout": list(eras[single.CALIBRATION_ERAS :]),
        "full": list(eras),
    }
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for segment_name, segment_eras in segments.items():
        output[segment_name] = {}
        for candidate in ALL_SIGNAL_COLUMNS:
            output[segment_name][candidate] = {
                "era_count": len(segment_eras),
                "corr": single._score_summary(
                    per_era["corr"].loc[segment_eras, candidate]
                ),
                "bmc": single._score_summary(
                    per_era["bmc"].loc[segment_eras, candidate]
                ),
                "avg_corr_with_benchmark": float(
                    per_era["benchmark_corr"].loc[segment_eras, candidate].mean()
                ),
            }
    return output


def calibration_checks(
    candidate_metrics: Mapping[str, Any],
    residual_bmc_mean: float,
    benchmark_corr_mean: float,
) -> dict[str, bool]:
    bmc = candidate_metrics["bmc"]
    return {
        "bmc_mean_positive": float(bmc["mean"]) > 0.0,
        "residual_bmc_retention": single._at_least(
            float(bmc["mean"]),
            single.CALIBRATION_THRESHOLDS[
                "residual_bmc_retention_min_inclusive"
            ]
            * residual_bmc_mean,
        ),
        "bmc_sharpe": float(bmc["sharpe"])
        > single.CALIBRATION_THRESHOLDS["bmc_sharpe_min_exclusive"],
        "bmc_max_drawdown": float(bmc["max_drawdown"])
        < single.CALIBRATION_THRESHOLDS["bmc_max_drawdown_max_exclusive"],
        "benchmark_corr_retention": single._at_least(
            float(candidate_metrics["corr"]["mean"]),
            single.CALIBRATION_THRESHOLDS[
                "benchmark_corr_retention_min_inclusive"
            ]
            * benchmark_corr_mean,
        ),
        "benchmark_similarity": float(candidate_metrics["avg_corr_with_benchmark"])
        < single.CALIBRATION_THRESHOLDS[
            "avg_corr_with_benchmark_max_exclusive"
        ],
    }


def select_calibration_candidate(
    summaries: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    calibration = summaries["calibration"]
    residual_bmc_mean = float(calibration["two_seed_residual"]["bmc"]["mean"])
    benchmark_corr_mean = float(calibration["benchmark_only"]["corr"]["mean"])
    evaluations: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    for candidate in CANDIDATE_COLUMNS:
        checks = calibration_checks(
            calibration[candidate], residual_bmc_mean, benchmark_corr_mean
        )
        is_eligible = all(checks.values())
        evaluations[candidate] = {"eligible": is_eligible, "checks": checks}
        if is_eligible:
            eligible.append(candidate)
    if not eligible:
        return None, evaluations
    selected = sorted(
        eligible,
        key=lambda name: (
            -float(calibration[name]["bmc"]["mean"]),
            float(calibration[name]["bmc"]["max_drawdown"]),
            float(WEIGHTS[name]),
        ),
    )[0]
    return selected, evaluations


def promotion_checks(
    selected: str,
    summaries: Mapping[str, Mapping[str, Mapping[str, Any]]],
    coverage_ok: bool,
) -> dict[str, bool]:
    holdout = summaries["holdout"]
    full = summaries["full"]
    selected_holdout = holdout[selected]
    selected_full = full[selected]
    residual_holdout_bmc = float(holdout["two_seed_residual"]["bmc"]["mean"])
    benchmark_holdout_corr = float(holdout["benchmark_only"]["corr"]["mean"])
    return {
        "exact_finite_coverage": bool(coverage_ok),
        "holdout_bmc_mean_positive": float(selected_holdout["bmc"]["mean"]) > 0.0,
        "holdout_residual_bmc_retention": single._at_least(
            float(selected_holdout["bmc"]["mean"]),
            single.PROMOTION_THRESHOLDS[
                "holdout_residual_bmc_retention_min_inclusive"
            ]
            * residual_holdout_bmc,
        ),
        "holdout_bmc_sharpe": float(selected_holdout["bmc"]["sharpe"])
        > single.PROMOTION_THRESHOLDS["holdout_bmc_sharpe_min_exclusive"],
        "holdout_bmc_max_drawdown": float(selected_holdout["bmc"]["max_drawdown"])
        < single.PROMOTION_THRESHOLDS[
            "holdout_bmc_max_drawdown_max_exclusive"
        ],
        "holdout_benchmark_corr_retention": single._at_least(
            float(selected_holdout["corr"]["mean"]),
            single.PROMOTION_THRESHOLDS[
                "holdout_benchmark_corr_retention_min_inclusive"
            ]
            * benchmark_holdout_corr,
        ),
        "full_bmc_mean_positive": float(selected_full["bmc"]["mean"]) > 0.0,
        "last_200_bmc_mean_positive": float(selected_holdout["bmc"]["mean"]) > 0.0,
        "full_bmc_sharpe": float(selected_full["bmc"]["sharpe"])
        > single.PROMOTION_THRESHOLDS["full_bmc_sharpe_min_exclusive"],
        "full_bmc_max_drawdown": float(selected_full["bmc"]["max_drawdown"])
        < single.PROMOTION_THRESHOLDS["full_bmc_max_drawdown_max_exclusive"],
        "full_corr_lower_bound": single._at_least(
            float(selected_full["corr"]["mean"]),
            single.PROMOTION_THRESHOLDS["full_corr_mean_min_inclusive"],
        ),
        "full_corr_upper_bound": single._at_most(
            float(selected_full["corr"]["mean"]),
            single.PROMOTION_THRESHOLDS["full_corr_mean_max_inclusive"],
        ),
        "full_benchmark_similarity": float(selected_full["avg_corr_with_benchmark"])
        < single.PROMOTION_THRESHOLDS[
            "full_avg_corr_with_benchmark_max_exclusive"
        ],
    }


def _write_summary_csv(
    path: Path,
    summaries: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    rows: list[dict[str, Any]] = []
    for segment, candidates in summaries.items():
        for candidate, metrics in candidates.items():
            rows.append(
                {
                    "segment": segment,
                    "candidate": candidate,
                    "selectable": candidate in CANDIDATE_COLUMNS,
                    "residual_weight": WEIGHTS.get(candidate),
                    "era_count": metrics["era_count"],
                    "corr_mean": metrics["corr"]["mean"],
                    "corr_std": metrics["corr"]["std"],
                    "corr_sharpe": metrics["corr"]["sharpe"],
                    "bmc_mean": metrics["bmc"]["mean"],
                    "bmc_std": metrics["bmc"]["std"],
                    "bmc_sharpe": metrics["bmc"]["sharpe"],
                    "bmc_max_drawdown": metrics["bmc"]["max_drawdown"],
                    "avg_corr_with_benchmark": metrics[
                        "avg_corr_with_benchmark"
                    ],
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_per_era_csv(path: Path, per_era: Mapping[str, pd.DataFrame]) -> None:
    rows: list[pd.DataFrame] = []
    for candidate in ALL_SIGNAL_COLUMNS:
        rows.append(
            pd.DataFrame(
                {
                    "era": per_era["bmc"].index.astype(str),
                    "candidate": candidate,
                    "corr": per_era["corr"][candidate].to_numpy(),
                    "bmc": per_era["bmc"][candidate].to_numpy(),
                    "benchmark_corr": per_era["benchmark_corr"][candidate].to_numpy(),
                }
            )
        )
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)


def _new_pending_path(output_dir: Path, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=".two_seed_stability_",
        suffix=suffix,
        dir=output_dir,
        delete=False,
    ) as stream:
        return Path(stream.name)


def _install_immutable_output(pending: Path, final: Path) -> None:
    """Install a content-addressed output without mutating an existing payload."""

    if final.exists():
        single._require(
            single._sha256_file(final) == single._sha256_file(pending),
            f"Existing immutable output differs: {final.name}",
        )
        pending.unlink()
        return
    pending.replace(final)


def validate_frozen_gate_and_config(gate_path: Path, seed_config_path: Path) -> None:
    single._require(
        single._sha256_file(gate_path) == EXPECTED_GATE_SHA256,
        "Two-seed gate hash differs from the committed pre-training gate.",
    )
    single._require(
        single._sha256_file(seed_config_path) == EXPECTED_SEED_CONFIG_SHA256,
        "Seed-2027 wrapper config hash differs from the anchored configuration.",
    )


def evaluate(
    seed1337_path: Path,
    seed1337_result_path: Path,
    seed2027_path: Path,
    seed2027_result_path: Path,
    manifest_path: Path,
    seed_config_path: Path,
    gate_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    repo_root = single._repo_root()
    validate_frozen_gate_and_config(gate_path, seed_config_path)
    cohort, eras, provenance = load_and_validate_two_seed_cohort(
        seed1337_path,
        seed1337_result_path,
        seed2027_path,
        seed2027_result_path,
        manifest_path,
    )
    cohort = rank_two_seed_signals(cohort)
    coverage_ok = (
        len(cohort) == single.EXPECTED_ROWS
        and cohort[single.ID_COLUMN].is_unique
        and np.isfinite(
            cohort[list(ALL_SIGNAL_COLUMNS)].to_numpy(dtype=np.float64)
        ).all()
    )
    per_era = compute_per_era_metrics(cohort, eras)
    summaries = summarize_segments(per_era, eras)
    selected, calibration_candidates = select_calibration_candidate(summaries)
    checks = promotion_checks(selected, summaries, coverage_ok) if selected else {}
    promotion_eligible = selected is not None and all(checks.values())

    input_hashes = {
        "seed1337_predictions": single._sha256_file(seed1337_path),
        "seed1337_result": single._sha256_file(seed1337_result_path),
        "seed2027_predictions": single._sha256_file(seed2027_path),
        "seed2027_result": single._sha256_file(seed2027_result_path),
        "seed2027_config": single._sha256_file(seed_config_path),
        "manifest": single._sha256_file(manifest_path),
        "gate": single._sha256_file(gate_path),
        "evaluator": single._sha256_file(Path(__file__).resolve()),
    }
    generation_payload = {
        **input_hashes,
        "source_manifest": provenance["source_manifest_sha256"],
        "evaluator_schema": 1,
    }
    generation_id = hashlib.sha256(
        json.dumps(generation_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:20]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"two_seed_stability_summary-{generation_id}.csv"
    per_era_path = output_dir / f"two_seed_stability_per_era-{generation_id}.csv"
    result_path = output_dir / "two_seed_stability_result.json"
    pending_summary = _new_pending_path(output_dir, ".summary.csv")
    pending_per_era = _new_pending_path(output_dir, ".per_era.csv")
    pending_result = _new_pending_path(output_dir, ".result.json")
    _write_summary_csv(pending_summary, summaries)
    _write_per_era_csv(pending_per_era, per_era)
    summary_sha256 = single._sha256_file(pending_summary)
    per_era_sha256 = single._sha256_file(pending_per_era)
    _install_immutable_output(pending_summary, summary_path)
    _install_immutable_output(pending_per_era, per_era_path)

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "as_of_date": AS_OF_DATE,
        "generation_id": generation_id,
        "state": (
            "PROMOTION_ELIGIBLE_LOCAL_PACKAGING_ONLY"
            if promotion_eligible
            else "NOT_PROMOTION_ELIGIBLE"
        ),
        "promotion_eligible": promotion_eligible,
        "selected_candidate": selected,
        "selected_residual_weight": WEIGHTS.get(selected) if selected else None,
        "deployment_boundary": "No upload or staking is authorized by this result.",
        "inputs": {
            "seed1337_predictions": {
                "path": single._relative_path(seed1337_path, repo_root),
                "sha256": input_hashes["seed1337_predictions"],
            },
            "seed1337_result": {
                "path": single._relative_path(seed1337_result_path, repo_root),
                "sha256": input_hashes["seed1337_result"],
            },
            "seed2027_predictions": {
                "path": single._relative_path(seed2027_path, repo_root),
                "sha256": input_hashes["seed2027_predictions"],
            },
            "seed2027_result": {
                "path": single._relative_path(seed2027_result_path, repo_root),
                "sha256": input_hashes["seed2027_result"],
            },
            "seed2027_config": {
                "path": single._relative_path(seed_config_path, repo_root),
                "sha256": input_hashes["seed2027_config"],
            },
            "manifest": {
                "path": single._relative_path(manifest_path, repo_root),
                "sha256": input_hashes["manifest"],
            },
            "gate": {
                "path": single._relative_path(gate_path, repo_root),
                "sha256": input_hashes["gate"],
            },
            "evaluator": {
                "path": single._relative_path(Path(__file__).resolve(), repo_root),
                "sha256": input_hashes["evaluator"],
            },
            "frozen_source": {
                "source_manifest_sha256": provenance["source_manifest_sha256"],
                "historical_data_git_head": provenance["historical_data_git_head"],
                "pretraining_git_commit": provenance["pretraining_git_commit"],
                "store_generation_id": provenance["store_generation_id"],
            },
            "prediction_semantics": provenance["semantics"],
            "artifact_validation": provenance["artifacts"],
        },
        "cohort": {
            "rows": len(cohort),
            "eras": len(eras),
            "first_era": eras[0],
            "last_era": eras[-1],
            "calibration_eras": single.CALIBRATION_ERAS,
            "holdout_eras": single.HOLDOUT_ERAS,
            "first_holdout_era": eras[single.CALIBRATION_ERAS],
        },
        "weights": WEIGHTS,
        "calibration_thresholds": single.CALIBRATION_THRESHOLDS,
        "promotion_thresholds": single.PROMOTION_THRESHOLDS,
        "calibration_candidates": calibration_candidates,
        "promotion_checks": checks,
        "summaries": summaries,
        "outputs": {
            "summary_csv": single._relative_path(summary_path, repo_root),
            "summary_csv_sha256": summary_sha256,
            "per_era_csv": single._relative_path(per_era_path, repo_root),
            "per_era_csv_sha256": per_era_sha256,
        },
    }
    pending_result.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pending_result.replace(result_path)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "state": result["state"],
                "selected_candidate": selected,
                "promotion_eligible": promotion_eligible,
            },
            sort_keys=True,
        )
    )
    return result


def decision_exit_code(result: Mapping[str, Any]) -> int:
    """Signal the frozen stop rule to shell automation."""

    return 0 if result.get("promotion_eligible") is True else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = single._repo_root()
    architecture_dir = (
        repo_root / "numerai/agents/experiments/ender20_nn_architecture_v53"
    )
    experiment_dir = repo_root / "numerai/agents/experiments" / EXPERIMENT_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed1337-predictions",
        type=Path,
        default=architecture_dir
        / "predictions/scale_disk_tabm_k64_train500k.parquet",
    )
    parser.add_argument(
        "--seed1337-result",
        type=Path,
        default=architecture_dir / "results/scale_disk_tabm_k64_train500k.json",
    )
    parser.add_argument(
        "--seed2027-predictions",
        type=Path,
        default=architecture_dir
        / "predictions/scale_disk_tabm_k64_train500k_seed2027.parquet",
    )
    parser.add_argument(
        "--seed2027-result",
        type=Path,
        default=architecture_dir
        / "results/scale_disk_tabm_k64_train500k_seed2027.json",
    )
    parser.add_argument(
        "--seed2027-config",
        type=Path,
        default=architecture_dir
        / "configs/scale_disk_tabm_k64_train500k_seed2027.py",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root
        / "numerai/v5.3/target_ender_20_feature_store/manifest-*.parquet",
    )
    parser.add_argument("--gate", type=Path, default=experiment_dir / "gate.md")
    parser.add_argument("--output-dir", type=Path, default=experiment_dir / "results")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = single.resolve_manifest(args.manifest)
    result = evaluate(
        args.seed1337_predictions.resolve(),
        args.seed1337_result.resolve(),
        args.seed2027_predictions.resolve(),
        args.seed2027_result.resolve(),
        manifest,
        args.seed2027_config.resolve(),
        args.gate.resolve(),
        args.output_dir.resolve(),
    )
    return decision_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
