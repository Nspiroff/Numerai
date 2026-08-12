"""Evaluate the frozen Ender21 Round-1 discovery candidates exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from agents.code.metrics import numerai_metrics
from agents.code.modeling.utils.numerai_cv import era_cv_splits
from agents.code.modeling.utils.constants import REPO_DIR
from agents.code.modeling.utils.pipeline import (
    PREDICTION_SEMANTICS_METADATA_KEY,
    _verify_ender21_round1_manifest,
)

import pyarrow.parquet as pq


TARGET = "target_ender_20"
BENCHMARK = "v53_lgbm_ender20"
ERA = "era"
ID = "id"
CONTROL = "r1_control_tabm_k64"
CANDIDATES = (
    "r1_control_tabm_k64",
    "r1_tabm_mini_k64",
    "r1_tabm_k64_era_balanced",
    "r1_tabm_k64_block_dro",
    "r1_tabm_mini_k64_block_dro",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt(path: Path) -> dict:
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Non-finite {label}")
    return number


def _score_candidate(
    experiment: Path,
    name: str,
    allowed_list: list[str],
    truth: pd.DataFrame,
) -> dict:
    allowed_eras = set(allowed_list)
    prediction_path = experiment / "predictions" / f"{name}.parquet"
    result_path = experiment / "results" / f"{name}.json"
    if not prediction_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"Missing single-run artifact for {name}")
    parquet = pq.ParquetFile(prediction_path)
    metadata = parquet.schema_arrow.metadata or {}
    semantics_raw = metadata.get(PREDICTION_SEMANTICS_METADATA_KEY)
    if semantics_raw is None:
        raise ValueError(f"{name} has no frozen prediction-semantics metadata")
    semantics = json.loads(semantics_raw)
    expected_semantics = {
        "artifact_kind": "out_of_fold_validation",
        "column": "prediction",
        "era_column": ERA,
        "fold_column": "cv_fold",
        "fold_index_base": 0,
        "inverse_target_transform_applied": False,
        "pipeline_postprocess": {"type": "identity"},
        "producer": "model.predict",
        "schema_version": 1,
        "stored_target": {"column": TARGET, "transform": {"type": "identity"}},
        "training_target": {
            "column": TARGET,
            "transform": {
                "benchmark_col": BENCHMARK,
                "era_col": ERA,
                "fit_intercept": True,
                "per_era": True,
                "proportion": 1.0,
                "type": "residual_to_benchmark",
            },
        },
    }
    if semantics != expected_semantics:
        raise ValueError(f"{name} prediction semantics differ from the protocol")
    frame = parquet.read().to_pandas()
    expected_columns = [ID, ERA, TARGET, "prediction", "cv_fold"]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"{name} prediction columns differ: {list(frame.columns)}")
    if frame.empty or frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError(f"{name} has empty/null/duplicate prediction ids")
    frame[ERA] = frame[ERA].astype(str)
    if not set(frame[ERA]).issubset(allowed_eras):
        raise ValueError(f"{name} contains an era outside frozen discovery")
    if max(map(int, frame[ERA])) > 861:
        raise ValueError(f"{name} reached Ender21 confirmation eras")
    for column in (TARGET, "prediction", "cv_fold"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} has non-finite {column}")
    folds = frame["cv_fold"].to_numpy(dtype="float64")
    if not np.array_equal(folds, folds.astype("int64")):
        raise ValueError(f"{name} has fractional CV folds")

    joined = frame.merge(
        truth,
        on=[ID, ERA],
        how="left",
        validate="one_to_one",
        suffixes=("", "_truth"),
    )
    if joined[[f"{TARGET}_truth", BENCHMARK]].isna().any().any():
        raise ValueError(f"{name} has unmatched target or benchmark rows")
    if not np.array_equal(
        joined[TARGET].to_numpy(dtype="float64"),
        joined[f"{TARGET}_truth"].to_numpy(dtype="float64"),
        equal_nan=True,
    ):
        raise ValueError(f"{name} target values differ from frozen development data")
    joined = joined.drop(columns=[f"{TARGET}_truth"])

    expected_rows = []
    expected_folds = {}
    expected_fold_contract = []
    for fold, (train_eras, val_eras) in enumerate(
        era_cv_splits(
            allowed_list,
            n_splits=5,
            embargo=13,
            mode="expanding",
            min_train_size=0,
        )
    ):
        if not train_eras:
            continue
        expected_folds.update({era: fold for era in val_eras})
        expected_rows.extend(truth.loc[truth[ERA].isin(val_eras), ID].tolist())
        expected_fold_contract.append(
            {
                "fold": fold,
                "train_eras": len(train_eras),
                "val_eras": len(val_eras),
                "train_rows": min(
                    int(truth[ERA].isin(train_eras).sum()), 500_000
                ),
                "val_rows": int(truth[ERA].isin(val_eras).sum()),
            }
        )
    if set(frame[ID]) != set(expected_rows) or len(frame) != len(expected_rows):
        raise ValueError(f"{name} OOF ids do not exactly cover frozen validation folds")
    expected_fold_values = frame[ERA].map(expected_folds).to_numpy(dtype="int64")
    if not np.array_equal(frame["cv_fold"].to_numpy(dtype="int64"), expected_fold_values):
        raise ValueError(f"{name} CV fold assignment differs from the frozen split")

    bmc = numerai_metrics.per_era_bmc(joined, ["prediction"], BENCHMARK, TARGET)
    corr = numerai_metrics.per_era_corr(joined, ["prediction"], TARGET)
    similarity = numerai_metrics.per_era_pred_corr(
        joined, ["prediction"], BENCHMARK
    )
    bmc_summary = numerai_metrics.score_summary(bmc["prediction"])
    corr_summary = numerai_metrics.score_summary(corr["prediction"])
    recent_fold = int(joined["cv_fold"].max())
    recent_eras = set(joined.loc[joined["cv_fold"] == recent_fold, ERA])
    recent_bmc = bmc.loc[bmc.index.astype(str).isin(recent_eras), "prediction"]
    fold_bmc = {}
    for fold, subset in joined.groupby("cv_fold", sort=True):
        fold_scores = numerai_metrics.per_era_bmc(
            subset, ["prediction"], BENCHMARK, TARGET
        )["prediction"]
        fold_bmc[str(int(fold))] = _finite(fold_scores.mean(), f"{name} fold BMC")

    stored = json.loads(result_path.read_text(encoding="utf-8"))
    config = __import__("runpy").run_path(
        str(experiment / "configs" / f"{name}.py")
    )["CONFIG"]
    if stored["model"] != config["model"]:
        raise ValueError(f"{name} stored model contract differs from its frozen config")
    if stored["preprocessing"] != config["preprocessing"]:
        raise ValueError(f"{name} stored preprocessing contract differs")
    expected_training = {
        "data_sampling": {"max_train_samples": 500_000, "sample_seed": 1337},
        "data_mode": "eager",
        "cv": config["training"]["cv"],
    }
    if stored["training"] != expected_training:
        raise ValueError(f"{name} stored training contract differs")
    expected_cv_header = {
        "n_splits": 5,
        "embargo": 13,
        "mode": "expanding",
        "min_train_size": 0,
        "folds_used": 4,
    }
    if any(stored["cv"].get(key) != value for key, value in expected_cv_header.items()):
        raise ValueError(f"{name} stored CV contract differs")
    stored_folds = stored["cv"].get("folds")
    if not isinstance(stored_folds, list) or len(stored_folds) != 4:
        raise ValueError(f"{name} stored fold count differs")
    for actual, expected in zip(stored_folds, expected_fold_contract):
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{name} stored fold geometry differs")
        if not isinstance(actual.get("model_diagnostics"), dict):
            raise ValueError(f"{name} has no model diagnostics for a used fold")
    allowlist_receipt = stored["data"].get("era_allowlist")
    allowlist_path = experiment / "protocol/discovery_eras_through_0861.json"
    allowlist_bytes = allowlist_path.read_bytes()
    expected_allowlist_receipt = {
        "path": config["data"]["era_allowlist_path"],
        "sha256": hashlib.sha256(allowlist_bytes).hexdigest(),
        "size_bytes": len(allowlist_bytes),
        "era_count": 176,
        "first_era": "0161",
        "last_era": "0861",
    }
    if allowlist_receipt != expected_allowlist_receipt:
        raise ValueError(f"{name} stored allowlist binding is absent or wrong")
    expected_data = {
        "data_version": "v5.3",
        "feature_set": "all",
        "target": TARGET,
        "full_data_path": config["data"]["full_data_path"],
        "full_rows": len(truth),
        "full_eras": len(allowed_list),
        "oof_rows": len(expected_rows),
        "oof_eras": len(expected_folds),
        "embargo_eras": 13,
        "require_benchmark_coverage": True,
        "data_mode": "eager",
        "era_allowlist": expected_allowlist_receipt,
    }
    if stored["data"] != expected_data:
        raise ValueError(f"{name} stored data contract differs")
    if stored["benchmark"] != {
        "model": BENCHMARK,
        "file": config["data"]["benchmark_data_path"],
    }:
        raise ValueError(f"{name} stored benchmark contract differs")
    if stored["output"].get("prediction_semantics") != expected_semantics:
        raise ValueError(f"{name} stored prediction semantics differ")
    for key, actual in (
        ("bmc.mean", bmc_summary["mean"]),
        ("bmc.sharpe", bmc_summary["sharpe"]),
        ("bmc.max_drawdown", bmc_summary["max_drawdown"]),
        ("corr.mean", corr_summary["mean"]),
    ):
        section, field = key.split(".")
        expected = _finite(stored["metrics"][section][field], f"stored {name} {key}")
        if not np.isclose(expected, actual, rtol=0.0, atol=1e-12):
            raise ValueError(f"{name} stored {key} does not recompute exactly")

    return {
        "artifacts": {
            "config": _receipt(experiment / "configs" / f"{name}.py"),
            "prediction": _receipt(prediction_path),
            "result": _receipt(result_path),
        },
        "rows": len(joined),
        "eras": int(joined[ERA].nunique()),
        "first_era": min(joined[ERA], key=int),
        "last_era": max(joined[ERA], key=int),
        "recent_fold": recent_fold,
        "metrics": {
            "bmc": {key: _finite(value, f"{name} BMC {key}") for key, value in bmc_summary.items()},
            "recent_fold_bmc_mean": _finite(recent_bmc.mean(), f"{name} recent BMC"),
            "corr": {key: _finite(value, f"{name} Corr {key}") for key, value in corr_summary.items()},
            "avg_corr_with_benchmark": _finite(
                similarity["prediction"].mean(), f"{name} benchmark correlation"
            ),
            "fold_bmc_mean": fold_bmc,
        },
        "per_era": {
            str(era): {
                "bmc": _finite(bmc.loc[era, "prediction"], f"{name} era BMC"),
                "corr": _finite(corr.loc[era, "prediction"], f"{name} era Corr"),
                "corr_with_benchmark": _finite(
                    similarity.loc[era, "prediction"], f"{name} era similarity"
                ),
            }
            for era in bmc.index
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--numerai-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_experiment = Path(
        os.path.abspath(
            REPO_DIR
            / "numerai/agents/experiments/ender21_residual_stability_v53"
        )
    )
    experiment = Path(os.path.abspath(args.experiment))
    if experiment != expected_experiment:
        raise ValueError("Round-1 evaluator requires the canonical Ender21 experiment.")
    expected_numerai_dir = Path(os.path.abspath(REPO_DIR / "numerai"))
    numerai_dir = Path(os.path.abspath(args.numerai_dir))
    if numerai_dir != expected_numerai_dir:
        raise ValueError("Round-1 evaluator requires the canonical Numerai data root.")
    output = Path(os.path.abspath(args.output))
    expected_output = experiment / "receipts/round1_discovery.json"
    if output != expected_output:
        raise ValueError("Round-1 evaluator output path differs from the freeze.")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite Round-1 receipt: {output}")
    _verify_ender21_round1_manifest()

    allowlist_path = experiment / "protocol/discovery_eras_through_0861.json"
    allowed_list = json.loads(allowlist_path.read_text(encoding="utf-8"))
    if (
        not isinstance(allowed_list, list)
        or len(allowed_list) != 176
        or len(set(allowed_list)) != 176
        or allowed_list[0] != "0161"
        or allowed_list[-1] != "0861"
        or allowed_list != sorted(allowed_list, key=int)
    ):
        raise ValueError("Round-1 discovery allowlist differs from the freeze.")
    allowed_eras = set(allowed_list)
    full_path = (
        numerai_dir
        / "v5.3/ender21_discovery_full_through_0861.parquet"
    )
    benchmark_path = (
        numerai_dir
        / "v5.3/ender21_discovery_benchmark_models_through_0861.parquet"
    )
    full = pd.read_parquet(full_path, columns=[ID, ERA, TARGET])
    benchmark = pd.read_parquet(benchmark_path, columns=[ID, ERA, BENCHMARK])
    full[ERA] = full[ERA].astype(str)
    benchmark[ERA] = benchmark[ERA].astype(str)
    truth = full.loc[full[ERA].isin(allowed_eras)].merge(
        benchmark.loc[benchmark[ERA].isin(allowed_eras)],
        on=[ID, ERA],
        how="inner",
        validate="one_to_one",
    )
    results = {
        name: _score_candidate(experiment, name, allowed_list, truth)
        for name in CANDIDATES
    }
    control = results[CONTROL]["metrics"]
    decisions = {}
    for name in CANDIDATES[1:]:
        metrics = results[name]["metrics"]
        checks = {
            "positive_full_bmc": metrics["bmc"]["mean"] > 0.0,
            "positive_recent_fold_bmc": metrics["recent_fold_bmc_mean"] > 0.0,
            "corr_guardrail": 0.005 <= metrics["corr"]["mean"] < 0.04,
            "full_bmc_retention": metrics["bmc"]["mean"] >= 0.90 * control["bmc"]["mean"],
            "recent_bmc_retention": metrics["recent_fold_bmc_mean"]
            >= 0.90 * control["recent_fold_bmc_mean"],
            "drawdown_improvement": metrics["bmc"]["max_drawdown"]
            <= 0.85 * control["bmc"]["max_drawdown"],
            "sharpe_retention": metrics["bmc"]["sharpe"]
            >= control["bmc"]["sharpe"] - 0.05,
            "all_folds_positive": all(
                value > 0.0 for value in metrics["fold_bmc_mean"].values()
            ),
        }
        decisions[name] = {"eligible": all(checks.values()), "checks": checks}
    eligible = [name for name, item in decisions.items() if item["eligible"]]
    eligible.sort(
        key=lambda name: (
            -results[name]["metrics"]["bmc"]["mean"],
            -results[name]["metrics"]["recent_fold_bmc_mean"],
            results[name]["metrics"]["bmc"]["max_drawdown"],
            name,
        )
    )
    payload = {
        "schema_version": 1,
        "stage": "ender21-round1-discovery",
        "state": "SCOUT_WINNER" if eligible else "NEGATIVE",
        "selected": eligible[0] if eligible else None,
        "inputs": {
            "allowlist": _receipt(allowlist_path),
            "full": _receipt(full_path),
            "benchmark": _receipt(benchmark_path),
        },
        "candidates": results,
        "decisions": decisions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"state": payload["state"], "selected": payload["selected"]}))


if __name__ == "__main__":
    main()
