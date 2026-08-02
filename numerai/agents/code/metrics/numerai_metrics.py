"""Reusable Numerai scoring helpers built on numerai_tools."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numerapi import NumerAPI
from numerai_tools.scoring import correlation_contribution, numerai_corr

from agents.code.modeling.utils.constants import NUMERAI_DIR, REPO_DIR


def _resolve_data_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] == NUMERAI_DIR.name:
        return (REPO_DIR / candidate).resolve()
    return (NUMERAI_DIR / candidate).resolve()


def _as_list(cols: Iterable) -> List:
    if isinstance(cols, (list, tuple, pd.Index, np.ndarray)):
        return list(cols)
    return [cols]


def _sort_era_index(scores: pd.DataFrame | pd.Series):
    if scores.index.dtype == object:
        try:
            order = sorted(scores.index, key=lambda x: int(x))
            return scores.loc[order]
        except (TypeError, ValueError):
            return scores.sort_index()
    return scores.sort_index()


def _normalize_per_era(per_era, pred_cols: Sequence) -> pd.DataFrame:
    if isinstance(per_era, pd.DataFrame):
        return _sort_era_index(per_era)
    if isinstance(per_era, pd.Series):
        if per_era.index.nlevels > 1:
            per_era = per_era.unstack()
            return _sort_era_index(per_era)
        return _sort_era_index(per_era.to_frame(pred_cols[0]))
    raise TypeError("Unexpected per-era score type")


def _last_n_eras(per_era_scores: pd.DataFrame | pd.Series, n: int):
    if n <= 0:
        raise ValueError("n must be > 0")
    per_era_scores = _sort_era_index(per_era_scores)
    return per_era_scores.tail(n)


def per_era_corr(
    df: pd.DataFrame,
    pred_cols: Sequence,
    target_col: str,
    era_col: str = "era",
) -> pd.DataFrame:
    """Compute per-era Numerai correlation for one or more prediction columns."""
    pred_cols = _as_list(pred_cols)
    df = df.dropna(subset=pred_cols + [target_col])

    def _corr(group):
        return numerai_corr(group[pred_cols], group[target_col])

    per_era = df.groupby(era_col).apply(_corr)
    return _normalize_per_era(per_era, pred_cols)


def per_era_mmc(
    df: pd.DataFrame,
    pred_cols: Sequence,
    meta_col: str,
    target_col: str,
    era_col: str = "era",
) -> pd.DataFrame:
    """Compute per-era MMC using Numerai's meta model predictions."""
    pred_cols = _as_list(pred_cols)
    df = df.dropna(subset=pred_cols + [meta_col, target_col])

    def _mmc(group):
        return correlation_contribution(
            group[pred_cols], group[meta_col], group[target_col]
        )

    per_era = df.groupby(era_col).apply(_mmc)
    return _normalize_per_era(per_era, pred_cols)


def per_era_pred_corr(
    df: pd.DataFrame,
    pred_cols: Sequence,
    benchmark_col: str,
    era_col: str = "era",
) -> pd.DataFrame:
    """Compute per-era correlation between predictions and benchmark predictions."""
    pred_cols = _as_list(pred_cols)
    df = df.dropna(subset=pred_cols + [benchmark_col])

    def _corr(group):
        return numerai_corr(group[pred_cols], group[benchmark_col])

    per_era = df.groupby(era_col).apply(_corr)
    return _normalize_per_era(per_era, pred_cols)


def per_era_bmc(
    df: pd.DataFrame,
    pred_cols: Sequence,
    benchmark_col: str,
    target_col: str,
    era_col: str = "era",
) -> pd.DataFrame:
    """Compute per-era benchmark contribution using the benchmark predictions."""
    pred_cols = _as_list(pred_cols)
    df = df.dropna(subset=pred_cols + [benchmark_col, target_col])

    def _bmc(group):
        return correlation_contribution(
            group[pred_cols], group[benchmark_col], group[target_col]
        )

    per_era = df.groupby(era_col).apply(_bmc)
    return _normalize_per_era(per_era, pred_cols)


def max_drawdown(scores: pd.Series) -> float:
    scores = scores.dropna()
    if scores.empty:
        return np.nan
    cumsum = scores.cumsum()
    running_max = cumsum.expanding(min_periods=1).max()
    return (running_max - cumsum).max()


def score_summary(scores: pd.Series) -> dict:
    scores = scores.dropna()
    if scores.empty:
        return {
            "mean": np.nan,
            "std": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }
    mean = scores.mean()
    std = scores.std(ddof=0)
    sharpe = mean / std if std != 0 else np.nan
    return {
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(scores),
    }


def summarize_scores(per_era_scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-era score DataFrame into mean/std/sharpe/drawdown."""
    per_era_scores = _sort_era_index(per_era_scores)
    summary = {col: score_summary(per_era_scores[col]) for col in per_era_scores}
    return pd.DataFrame(summary).T


def _parquet_columns(path: str | Path) -> list[str]:
    parquet = pq.ParquetFile(path)
    try:
        return parquet.schema.names
    finally:
        parquet.close()


def _require_parquet_columns(
    columns: Iterable[str], required: Sequence[str], *, source: str | Path
) -> None:
    available = set(columns)
    missing = [col for col in required if col not in available]
    if missing:
        raise ValueError(
            f"Parquet data at {source} is missing required named columns: {missing}."
        )


def _materialize_named_column(
    frame: pd.DataFrame, column: str, *, source: str
) -> pd.DataFrame:
    if column in frame.columns:
        return frame
    if column in frame.index.names:
        return frame.reset_index(level=column)
    raise ValueError(f"{source} data missing named column '{column}'.")


def _validate_alignment_keys(
    frame: pd.DataFrame,
    *,
    source: str,
    id_col: str,
    era_col: str,
) -> pd.DataFrame:
    aligned = _materialize_named_column(frame.copy(), id_col, source=source)
    aligned = _materialize_named_column(aligned, era_col, source=source)
    if aligned[id_col].isna().any():
        raise ValueError(f"{source} data contains null ids in '{id_col}'.")
    if aligned[era_col].isna().any():
        raise ValueError(f"{source} data contains null eras in '{era_col}'.")
    if aligned[id_col].duplicated().any():
        raise ValueError(f"{source} data contains duplicate ids in '{id_col}'.")
    return aligned.reset_index(drop=True)


def _require_finite_numeric_columns(
    frame: pd.DataFrame, columns: Sequence[str], *, source: str
) -> None:
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"{source} data missing required column '{column}'.")
        values = frame[column]
        if pd.api.types.is_bool_dtype(values.dtype) or not pd.api.types.is_numeric_dtype(
            values.dtype
        ):
            raise ValueError(
                f"{source} column '{column}' must contain finite numeric values."
            )
        if pd.api.types.is_complex_dtype(values.dtype):
            raise ValueError(
                f"{source} column '{column}' must contain finite real numeric values."
            )
        numeric = values.to_numpy(dtype="float64", copy=False)
        if not np.isfinite(numeric).all():
            raise ValueError(
                f"{source} column '{column}' must contain finite numeric values."
            )


def _unique_temporary_column(frame: pd.DataFrame, base: str) -> str:
    column = base
    suffix = 0
    while column in frame.columns:
        suffix += 1
        column = f"{base}_{suffix}"
    return column


def _resolve_benchmark_column(columns: Iterable[str], benchmark_model: str) -> str:
    if benchmark_model in columns:
        return benchmark_model
    suffix = f"_{benchmark_model}"
    matches = [col for col in columns if col.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Benchmark model '{benchmark_model}' not found in columns.")
    raise ValueError(
        f"Benchmark model '{benchmark_model}' matched multiple columns: {matches}"
    )


def load_custom_benchmark_predictions(
    predictions_path: str | Path,
    benchmark_name: str,
    pred_col: str = "prediction",
    era_col: str = "era",
    id_col: str = "id",
) -> tuple[pd.DataFrame, str]:
    """Load predictions from a local file to use as a benchmark column."""
    columns = _parquet_columns(predictions_path)
    required = [id_col, era_col, pred_col]
    _require_parquet_columns(columns, required, source=predictions_path)
    benchmark = pd.read_parquet(predictions_path, columns=required)
    if pred_col != benchmark_name:
        benchmark = benchmark.rename(columns={pred_col: benchmark_name})
    return benchmark, benchmark_name


def ensure_full_benchmark_models(napi: NumerAPI, data_version: str) -> Path:
    full_path = (NUMERAI_DIR / data_version / "full_benchmark_models.parquet").resolve()
    if full_path.exists():
        return full_path

    train_path = (NUMERAI_DIR / data_version / "train_benchmark_models.parquet").resolve()
    validation_path = (
        NUMERAI_DIR / data_version / "validation_benchmark_models.parquet"
    ).resolve()
    validation_data_path = (NUMERAI_DIR / data_version / "validation.parquet").resolve()
    if not train_path.exists():
        train_path.parent.mkdir(parents=True, exist_ok=True)
        napi.download_dataset(
            f"{data_version}/train_benchmark_models.parquet",
            dest_path=str(train_path),
        )
    if not validation_path.exists():
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        napi.download_dataset(
            f"{data_version}/validation_benchmark_models.parquet",
            dest_path=str(validation_path),
        )
    if not validation_data_path.exists():
        validation_data_path.parent.mkdir(parents=True, exist_ok=True)
        napi.download_dataset(
            f"{data_version}/validation.parquet", dest_path=str(validation_data_path)
        )

    validation_meta = pd.read_parquet(validation_data_path, columns=["data_type"])
    validation_meta = validation_meta[validation_meta["data_type"] == "validation"]
    validation_ids = validation_meta.index

    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)
    if "id" in train.columns:
        train = train.set_index("id")
    if "id" in validation.columns:
        validation = validation.set_index("id")
    validation = validation.loc[validation.index.intersection(validation_ids)]

    full = pd.concat([train, validation], axis=0)
    full.to_parquet(full_path)
    return full_path


def load_benchmark_predictions(
    data_version: str,
    split: str = "full",
    benchmark_model: str = "v53_lgbm_ender20",
    era_col: str = "era",
    id_col: str = "id",
) -> tuple[pd.DataFrame, str]:
    """Download and load benchmark predictions for the given data split."""
    napi = NumerAPI()
    if split == "full":
        dataset = str(ensure_full_benchmark_models(napi, data_version))
    else:
        remote_name = f"{data_version}/{split}_benchmark_models.parquet"
        local_path = (NUMERAI_DIR / data_version / f"{split}_benchmark_models.parquet").resolve()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            napi.download_dataset(remote_name, dest_path=str(local_path))
        dataset = str(local_path)
    columns = _parquet_columns(dataset)
    benchmark_col = _resolve_benchmark_column(columns, benchmark_model)
    required = [id_col, era_col, benchmark_col]
    _require_parquet_columns(columns, required, source=dataset)
    benchmark = pd.read_parquet(dataset, columns=required)
    return benchmark, benchmark_col


def load_benchmark_predictions_from_path(
    benchmark_path: str | Path,
    benchmark_model: str,
    era_col: str = "era",
    id_col: str = "id",
) -> tuple[pd.DataFrame, str]:
    """Load benchmark predictions from a local parquet path."""
    path = _resolve_data_path(benchmark_path)
    columns = _parquet_columns(path)
    benchmark_col = _resolve_benchmark_column(columns, benchmark_model)
    read_cols = [id_col, era_col, benchmark_col]
    _require_parquet_columns(columns, read_cols, source=path)
    benchmark = pd.read_parquet(path, columns=read_cols)
    return benchmark, benchmark_col


def attach_benchmark_predictions(
    predictions: pd.DataFrame,
    benchmark: pd.DataFrame,
    benchmark_col: str,
    era_col: str = "era",
    id_col: str = "id",
) -> pd.DataFrame:
    """Attach a complete, one-to-one benchmark alignment by explicit id and era."""
    if not isinstance(id_col, str) or not id_col:
        raise ValueError("id_col must be a non-empty named column.")
    if not isinstance(era_col, str) or not era_col:
        raise ValueError("era_col must be a non-empty named column.")
    if id_col == era_col:
        raise ValueError("id_col and era_col must name different columns.")

    preds = _validate_alignment_keys(
        predictions, source="Prediction", id_col=id_col, era_col=era_col
    )
    bench = _validate_alignment_keys(
        benchmark, source="Benchmark", id_col=id_col, era_col=era_col
    )
    if preds.empty:
        raise ValueError("Prediction data is empty.")
    if benchmark_col not in bench.columns:
        raise ValueError(f"Benchmark data missing selected column '{benchmark_col}'.")
    if benchmark_col in preds.columns:
        raise ValueError(
            f"Prediction data already contains benchmark column '{benchmark_col}'."
        )

    benchmark_era_col = _unique_temporary_column(preds, "__benchmark_era")
    benchmark_value_col = _unique_temporary_column(preds, "__benchmark_value")
    prediction_order_col = _unique_temporary_column(preds, "__prediction_order")
    merge_status_col = _unique_temporary_column(preds, "__benchmark_merge")
    benchmark_for_join = bench[[id_col, era_col, benchmark_col]].rename(
        columns={
            era_col: benchmark_era_col,
            benchmark_col: benchmark_value_col,
        }
    )
    left = preds.copy()
    left[prediction_order_col] = np.arange(len(left), dtype=np.int64)
    try:
        enriched = left.merge(
            benchmark_for_join,
            on=id_col,
            how="left",
            sort=False,
            validate="one_to_one",
            indicator=merge_status_col,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Prediction and benchmark ids could not be joined one-to-one."
        ) from exc

    missing = enriched[merge_status_col] != "both"
    if missing.any():
        missing_ids = enriched.loc[missing, id_col].head(5).tolist()
        raise ValueError(
            "Benchmark data does not completely cover prediction ids; "
            f"missing examples: {missing_ids}."
        )

    prediction_eras = enriched[era_col].reset_index(drop=True)
    benchmark_eras = enriched[benchmark_era_col].reset_index(drop=True)
    same_dtype = pd.api.types.is_dtype_equal(
        prediction_eras.dtype, benchmark_eras.dtype
    )
    same_values = np.array_equal(
        prediction_eras.to_numpy(), benchmark_eras.to_numpy()
    )
    if not same_dtype or not same_values:
        raise ValueError(
            "Benchmark eras do not exactly match prediction eras by id."
        )

    enriched = enriched.sort_values(prediction_order_col, kind="stable")
    enriched[benchmark_col] = enriched[benchmark_value_col]
    enriched = enriched[[*preds.columns, benchmark_col]].reset_index(drop=True)
    _require_finite_numeric_columns(
        enriched, [benchmark_col], source="Attached benchmark"
    )
    return enriched


def summarize_prediction_file(
    predictions_path: str | Path,
    pred_cols: Sequence,
    target_col: str,
    era_col: str = "era",
) -> dict[str, pd.DataFrame]:
    """Load a predictions parquet file and summarize per-era correlation metrics."""
    pred_cols = _as_list(pred_cols)
    required_cols = []
    columns = _parquet_columns(predictions_path)
    for col in [era_col, target_col, *pred_cols]:
        if col not in required_cols:
            required_cols.append(col)
    if "id" in columns:
        required_cols.append("id")
    predictions = pd.read_parquet(predictions_path, columns=required_cols)
    per_era = per_era_corr(predictions, pred_cols, target_col, era_col)
    return {"corr": summarize_scores(per_era)}


def summarize_prediction_file_with_bmc(
    predictions_path: str | Path,
    pred_cols: Sequence,
    target_col: str,
    data_version: str,
    benchmark_model: str = "v53_lgbm_ender20",
    benchmark_data_path: str | Path | None = None,
    era_col: str = "era",
    id_col: str = "id",
    benchmark_data: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Load predictions, attach benchmark model, and summarize corr and BMC metrics.

    ``benchmark_data`` is an in-memory alternative to ``benchmark_data_path``.
    It is useful when a validated generation-specific manifest must remain
    scoreable even if its artifact is concurrently retired.
    """
    pred_cols = _as_list(pred_cols)
    columns = _parquet_columns(predictions_path)
    required_cols = list(dict.fromkeys([id_col, era_col, target_col, *pred_cols]))
    _require_parquet_columns(columns, required_cols, source=predictions_path)
    predictions = pd.read_parquet(predictions_path, columns=required_cols)
    if benchmark_data is not None:
        columns = list(benchmark_data.columns)
        benchmark_col = _resolve_benchmark_column(columns, benchmark_model)
        read_cols = [benchmark_col]
        if id_col in columns:
            read_cols.append(id_col)
        if era_col in columns:
            read_cols.append(era_col)
        benchmark = benchmark_data.loc[:, read_cols].copy()
    elif benchmark_data_path is not None:
        benchmark_path = _resolve_data_path(benchmark_data_path)
        if not benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark data file not found: {benchmark_path}")
        columns = _parquet_columns(benchmark_path)
        benchmark_col = _resolve_benchmark_column(columns, benchmark_model)
        read_cols = [id_col, era_col, benchmark_col]
        _require_parquet_columns(columns, read_cols, source=benchmark_path)
        benchmark = pd.read_parquet(benchmark_path, columns=read_cols)
    else:
        benchmark, benchmark_col = load_benchmark_predictions(
            data_version,
            benchmark_model=benchmark_model,
            era_col=era_col,
            id_col=id_col,
        )
    predictions = attach_benchmark_predictions(
        predictions,
        benchmark,
        benchmark_col,
        era_col=era_col,
        id_col=id_col,
    )
    _require_finite_numeric_columns(
        predictions,
        [*pred_cols, target_col, benchmark_col],
        source="Scoring cohort",
    )
    corr_per_era = per_era_corr(predictions, pred_cols, target_col, era_col)
    per_era = per_era_bmc(predictions, pred_cols, benchmark_col, target_col, era_col)
    benchmark_corr = per_era_pred_corr(
        predictions, pred_cols, benchmark_col, era_col=era_col
    )

    summaries = {"corr": summarize_scores(corr_per_era)}
    bmc_summary = summarize_scores(per_era)
    benchmark_corr_mean = benchmark_corr.mean()
    for col in bmc_summary.index:
        bmc_summary.loc[col, "avg_corr_with_benchmark"] = float(
            benchmark_corr_mean.get(col, np.nan)
        )
    summaries["bmc"] = bmc_summary

    per_era_recent = _last_n_eras(per_era, 200)
    bmc_recent_summary = summarize_scores(per_era_recent)
    benchmark_corr_recent = _last_n_eras(benchmark_corr, 200)
    benchmark_corr_recent_mean = benchmark_corr_recent.mean()
    for col in bmc_recent_summary.index:
        bmc_recent_summary.loc[col, "avg_corr_with_benchmark"] = float(
            benchmark_corr_recent_mean.get(col, np.nan)
        )
    summaries["bmc_last_200_eras"] = bmc_recent_summary
    return summaries
