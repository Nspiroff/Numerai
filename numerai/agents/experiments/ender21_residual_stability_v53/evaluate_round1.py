"""Evaluate the frozen Ender21 Round-1 discovery candidates exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents.code.metrics import numerai_metrics


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
    allowed_eras: set[str],
    truth: pd.DataFrame,
) -> dict:
    prediction_path = experiment / "predictions" / f"{name}.parquet"
    result_path = experiment / "results" / f"{name}.json"
    if not prediction_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"Missing single-run artifact for {name}")
    frame = pd.read_parquet(prediction_path)
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
    experiment = args.experiment.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite Round-1 receipt: {output}")

    allowlist_path = experiment / "protocol/discovery_eras_through_0861.json"
    allowed_list = json.loads(allowlist_path.read_text(encoding="utf-8"))
    allowed_eras = set(allowed_list)
    full_path = args.numerai_dir.resolve() / "v5.3/ender21_dev_full_through_1021.parquet"
    benchmark_path = (
        args.numerai_dir.resolve()
        / "v5.3/ender21_dev_benchmark_models_through_1021.parquet"
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
        name: _score_candidate(experiment, name, allowed_eras, truth)
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
