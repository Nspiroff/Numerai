"""Keystone Round-0 authority-explicit scoring harness (KA28).

Lean reusable interface for official-parity per-era scoring:

* per-era CORR against the configured payout target via
  ``numerai_tools.scoring.numerai_corr`` (official implementation, not rederived);
* per-era MMC against the published Meta Model via
  ``numerai_tools.scoring.correlation_contribution`` (official implementation);
* the weighted model score
  ``corr * corr_multiplier + mmc * mmc_multiplier`` (the model-selection
  authority; staking settlement is a separate concern recorded in the
  score-authority record and never influences selection);
* deterministic summary statistics reproducible from the per-era values.

Every scoring-authority value (payout target, multipliers, meta-model column,
score names, BMC aggregate authority) is explicit configuration supplied by the
caller through :class:`ScoreAuthority`, typically loaded from a dated
machine-readable score-authority record with :meth:`ScoreAuthority.from_json`.
Nothing payout-relevant is hard-coded here, and the bare ``target`` dataset
alias is rejected because it names a dataset default, not a payout objective.

The harness fails loudly instead of silently repairing data: duplicate ids,
unmatched ids, era disagreements between joined frames, missing target values
inside scored eras, non-finite predictions, and unexpected era sets raise
``ValueError``. Meta-model coverage gaps either fail or cleanly exclude whole
eras according to an explicitly declared policy; partially covered eras always
fail. Benchmark Meta Model contribution (BMC) is computed only when the exact
official historical BMM aggregate authority is configured; otherwise its status
is reported as unavailable instead of being substituted with a single benchmark
or an unofficial average.

Inputs are keyed by named ``id`` and ``era`` columns. Before any official
scoring call the joined cohort is sorted by ``(era, id)``, so results are
independent of input row order and repeated executions are numerically
identical.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from numerai_tools.scoring import correlation_contribution, numerai_corr

BMC_AGGREGATE_NOT_REPRODUCIBLE_FROM_PUBLISHED_FILES = (
    "BMC_AGGREGATE_NOT_REPRODUCIBLE_FROM_PUBLISHED_FILES"
)
META_MODEL_POLICY_FAIL = "fail_on_missing_meta_model_era"
META_MODEL_POLICY_EXCLUDE = "exclude_missing_meta_model_eras"
_META_MODEL_POLICIES = (META_MODEL_POLICY_FAIL, META_MODEL_POLICY_EXCLUDE)

SHARPE_CONVENTION = (
    "era Sharpe = mean(per-era score) / population std(per-era score, ddof=0); "
    "no annualization; NaN when the std is 0"
)
DRAWDOWN_CONVENTION = (
    "max drawdown = max(running_max(equity) - equity) where equity = "
    "concatenate([0.0], cumsum(per-era score)) in chronological order; the "
    "zero-equity starting baseline is included so an initial losing streak "
    "from zero is counted; positive magnitude in score units"
)
PREDICTION_CORR_CONVENTION = (
    "correlation with the Meta Model and with benchmark columns uses the "
    "official Numerai Corr transform with the reference predictions in the "
    "target slot (repo per_era_pred_corr convention); it is diagnostic, not a "
    "payout score"
)
ERA_ORDER_CONVENTION = (
    "eras are uniform-width zero-padded digit strings sorted lexicographically "
    "(equal to chronological order); the joined cohort is sorted by (era, id) "
    "before scoring"
)


@dataclass(frozen=True)
class ScoreAuthority:
    """Explicit scoring authority. All values must be supplied by the caller.

    ``bmm_aggregate_authority`` is the exact official historical Benchmark
    Meta Model aggregate prediction column, or ``None`` when no such official
    aggregate is available from published files (BMC is then unavailable).
    ``retrieved_utc`` and ``documentation_authority`` bind the record to the
    audit that produced it; authority values change over time and must be
    re-audited before deployment-relevant use.
    """

    payout_target: str
    corr_multiplier: float
    mmc_multiplier: float
    meta_model_column: str
    corr_score_name: str
    mmc_score_name: str
    bmm_aggregate_authority: str | None
    retrieved_utc: str
    documentation_authority: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "payout_target",
            "meta_model_column",
            "corr_score_name",
            "mmc_score_name",
            "retrieved_utc",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ScoreAuthority.{field_name} must be a non-empty string.")
        if self.payout_target.strip().lower() == "target":
            raise ValueError(
                "ScoreAuthority.payout_target must be the explicit payout target "
                "column; the bare dataset alias 'target' is a dataset default, "
                "not a payout authority."
            )
        for field_name in ("corr_multiplier", "mmc_multiplier"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"ScoreAuthority.{field_name} must be a finite number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"ScoreAuthority.{field_name} must be a finite number.")
        if self.bmm_aggregate_authority is not None and (
            not isinstance(self.bmm_aggregate_authority, str)
            or not self.bmm_aggregate_authority.strip()
        ):
            raise ValueError(
                "ScoreAuthority.bmm_aggregate_authority must be None or the exact "
                "official BMM aggregate column name."
            )
        docs = self.documentation_authority
        if (
            not isinstance(docs, tuple)
            or not docs
            or not all(isinstance(item, str) and item.strip() for item in docs)
        ):
            raise ValueError(
                "ScoreAuthority.documentation_authority must be a non-empty tuple "
                "of source strings."
            )

    @classmethod
    def from_json(cls, path: str | Path) -> "ScoreAuthority":
        """Load the authority from a machine-readable score-authority record."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(raw, source=str(path))

    @classmethod
    def from_mapping(
        cls, record: Mapping, *, source: str = "<mapping>"
    ) -> "ScoreAuthority":
        def require(key: str):
            if key not in record:
                raise ValueError(
                    f"Score-authority record {source} is missing required key '{key}'."
                )
            return record[key]

        bmc = require("bmc")
        if not isinstance(bmc, Mapping) or "bmm_aggregate_authority" not in bmc:
            raise ValueError(
                f"Score-authority record {source} must contain "
                "bmc.bmm_aggregate_authority (exact official aggregate column or null)."
            )
        docs = require("documentation_authority")
        if not isinstance(docs, (list, tuple)):
            raise ValueError(
                f"Score-authority record {source} documentation_authority must be a list."
            )
        return cls(
            payout_target=require("payout_target"),
            corr_multiplier=require("corr_multiplier"),
            mmc_multiplier=require("mmc_multiplier"),
            meta_model_column=require("meta_model_column"),
            corr_score_name=require("corr_score_version"),
            mmc_score_name=require("mmc_score_version"),
            bmm_aggregate_authority=bmc["bmm_aggregate_authority"],
            retrieved_utc=require("retrieved_utc"),
            documentation_authority=tuple(docs),
        )


@dataclass(frozen=True)
class Round0Result:
    """Per-era official scores plus a summary reproducible from them."""

    per_era: pd.DataFrame
    summary: dict
    excluded_eras: tuple[str, ...]
    authority: ScoreAuthority


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, source: str) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{source} data is missing required named columns: {missing}.")


def _validate_era_strings(values: pd.Series, *, source: str) -> None:
    as_list = values.unique().tolist()
    if not all(isinstance(v, str) and v.isdigit() for v in as_list):
        raise ValueError(
            f"{source} era values must be zero-padded digit strings "
            "(e.g. '0575'); got a non-digit or non-string era."
        )
    widths = {len(v) for v in as_list}
    if len(widths) > 1:
        raise ValueError(
            f"{source} era values must have uniform width for deterministic "
            f"chronological ordering; got widths {sorted(widths)}."
        )


def _validate_keys(
    frame: pd.DataFrame, *, source: str, id_col: str, era_col: str | None
) -> None:
    if frame.empty:
        raise ValueError(f"{source} data is empty.")
    if frame[id_col].isna().any():
        raise ValueError(f"{source} data contains null ids in '{id_col}'.")
    if frame[id_col].duplicated().any():
        raise ValueError(f"{source} data contains duplicate ids in '{id_col}'.")
    if era_col is not None:
        if frame[era_col].isna().any():
            raise ValueError(f"{source} data contains null eras in '{era_col}'.")
        _validate_era_strings(frame[era_col], source=source)


def _validate_finite(
    frame: pd.DataFrame, columns: Sequence[str], *, source: str
) -> None:
    for column in columns:
        values = frame[column]
        if pd.api.types.is_bool_dtype(values.dtype) or not pd.api.types.is_numeric_dtype(
            values.dtype
        ):
            raise ValueError(
                f"{source} column '{column}' must contain finite numeric values."
            )
        numeric = values.to_numpy(dtype="float64", copy=False)
        if not np.isfinite(numeric).all():
            bad = int((~np.isfinite(numeric)).sum())
            raise ValueError(
                f"{source} column '{column}' must contain finite numeric values; "
                f"found {bad} non-finite value(s)."
            )


def _merge_by_id(
    left: pd.DataFrame,
    right: pd.DataFrame,
    value_cols: Sequence[str],
    *,
    id_col: str,
    era_col: str,
    right_has_era: bool,
    right_source: str,
    require_full_coverage: bool,
) -> pd.DataFrame:
    """Strict one-to-one id join; matched rows must agree on era when present."""
    right_cols = [id_col, *value_cols]
    rename = {col: f"__right_{col}" for col in value_cols}
    if right_has_era:
        right_cols.append(era_col)
        rename[era_col] = "__right_era"
    right_view = right[right_cols].rename(columns=rename)
    try:
        merged = left.merge(
            right_view,
            on=id_col,
            how="left",
            sort=False,
            validate="one_to_one",
            indicator="__merge_status",
        )
    except Exception as exc:
        raise ValueError(
            f"Prediction and {right_source} ids could not be joined one-to-one."
        ) from exc
    if require_full_coverage:
        missing = merged["__merge_status"] != "both"
        if bool(missing.any()):
            examples = merged.loc[missing, id_col].head(5).tolist()
            raise ValueError(
                f"{right_source} data does not cover all prediction ids; "
                f"missing examples: {examples}."
            )
    if right_has_era:
        matched = merged["__merge_status"] == "both"
        left_eras = merged.loc[matched, era_col]
        right_eras = merged.loc[matched, "__right_era"]
        if not np.array_equal(left_eras.to_numpy(), right_eras.to_numpy()):
            raise ValueError(
                f"{right_source} eras do not exactly match prediction eras by id."
            )
        merged = merged.drop(columns="__right_era")
    merged = merged.drop(columns="__merge_status")
    return merged.rename(columns={f"__right_{col}": f"__joined_{col}" for col in value_cols})


def _score_one_era(
    sub: pd.DataFrame,
    *,
    prediction_col: str,
    target_col: str,
    reference_cols: Mapping[str, str],
    contribution_cols: Mapping[str, str],
) -> dict[str, float]:
    """Score a single era slice (already sorted; unique ids as values)."""
    indexed = sub.set_index("__scoring_id")
    preds = indexed[[prediction_col]]
    target = indexed[target_col]
    row: dict[str, float] = {}
    row["corr"] = float(numerai_corr(preds, target)[prediction_col])
    for out_name, column in contribution_cols.items():
        row[out_name] = float(
            correlation_contribution(preds, indexed[column], target)[prediction_col]
        )
    for out_name, column in reference_cols.items():
        row[out_name] = float(numerai_corr(preds, indexed[column])[prediction_col])
    return row


def score_round0(
    predictions: pd.DataFrame,
    scoring_data: pd.DataFrame,
    meta_model: pd.DataFrame,
    authority: ScoreAuthority,
    *,
    prediction_col: str = "prediction",
    id_col: str = "id",
    era_col: str = "era",
    benchmark_data: pd.DataFrame | None = None,
    benchmark_cols: Sequence[str] | None = None,
    meta_model_policy: str = META_MODEL_POLICY_FAIL,
    expected_eras: Sequence[str] | None = None,
    recent_window: int = 40,
    block_size: int = 52,
) -> Round0Result:
    """Compute per-era official CORR/MMC/weighted scores for one prediction vector.

    ``predictions`` needs named ``id_col``/``era_col``/``prediction_col``
    columns; ``scoring_data`` needs ``id_col``/``era_col`` and the authority's
    payout-target column; ``meta_model`` needs ``id_col`` and the authority's
    meta-model column (``era_col`` is verified when present). ``benchmark_data``
    optionally supplies diagnostic benchmark prediction columns and, when the
    authority declares an official BMM aggregate column, the BMC reference.
    """
    if not isinstance(authority, ScoreAuthority):
        raise ValueError("score_round0 requires an explicit ScoreAuthority.")
    if meta_model_policy not in _META_MODEL_POLICIES:
        raise ValueError(
            f"meta_model_policy must be one of {_META_MODEL_POLICIES}; "
            f"got '{meta_model_policy}'."
        )
    target_col = authority.payout_target
    mm_col = authority.meta_model_column

    _require_columns(
        predictions, [id_col, era_col, prediction_col], source="Prediction"
    )
    preds = predictions[[id_col, era_col, prediction_col]].copy()
    _validate_keys(preds, source="Prediction", id_col=id_col, era_col=era_col)
    _validate_finite(preds, [prediction_col], source="Prediction")

    _require_columns(scoring_data, [id_col, era_col, target_col], source="Scoring")
    scoring = scoring_data[[id_col, era_col, target_col]].copy()
    _validate_keys(scoring, source="Scoring", id_col=id_col, era_col=era_col)
    _validate_finite(scoring, [target_col], source="Scoring")

    if expected_eras is not None:
        expected = set(expected_eras)
        actual = set(preds[era_col].unique().tolist())
        if actual != expected:
            raise ValueError(
                "Prediction eras do not match the expected era set; "
                f"missing={sorted(expected - actual)} unexpected={sorted(actual - expected)}."
            )

    joined = _merge_by_id(
        preds,
        scoring,
        [target_col],
        id_col=id_col,
        era_col=era_col,
        right_has_era=True,
        right_source="Scoring",
        require_full_coverage=True,
    )

    _require_columns(meta_model, [id_col, mm_col], source="Meta Model")
    mm_frame_cols = [id_col, mm_col] + ([era_col] if era_col in meta_model.columns else [])
    mm_frame = meta_model[mm_frame_cols].copy()
    _validate_keys(
        mm_frame,
        source="Meta Model",
        id_col=id_col,
        era_col=era_col if era_col in mm_frame.columns else None,
    )
    _validate_finite(mm_frame, [mm_col], source="Meta Model")
    joined = _merge_by_id(
        joined,
        mm_frame,
        [mm_col],
        id_col=id_col,
        era_col=era_col,
        right_has_era=era_col in mm_frame.columns,
        right_source="Meta Model",
        require_full_coverage=False,
    )

    mm_joined = f"__joined_{mm_col}"
    coverage = joined[mm_joined].notna().groupby(joined[era_col]).mean()
    partially_covered = sorted(coverage[(coverage > 0.0) & (coverage < 1.0)].index)
    if partially_covered:
        raise ValueError(
            "Meta Model coverage is partial inside eras "
            f"{partially_covered}; partially covered eras cannot be scored or "
            "cleanly excluded."
        )
    uncovered = tuple(sorted(coverage[coverage == 0.0].index))
    if uncovered:
        if meta_model_policy == META_MODEL_POLICY_FAIL:
            raise ValueError(
                f"Meta Model does not cover eras {list(uncovered)}; rerun with "
                f"meta_model_policy='{META_MODEL_POLICY_EXCLUDE}' to cleanly "
                "exclude whole uncovered eras."
            )
        joined = joined[~joined[era_col].isin(uncovered)]
        if joined.empty:
            raise ValueError(
                "Meta Model covers none of the prediction eras; nothing to score."
            )

    contribution_cols: dict[str, str] = {"mmc": mm_joined}
    reference_cols: dict[str, str] = {"corr_with_meta_model": mm_joined}

    if benchmark_cols is not None and benchmark_data is None:
        raise ValueError("benchmark_cols were requested without benchmark_data.")
    bmm_col = authority.bmm_aggregate_authority
    if bmm_col is not None and benchmark_data is None:
        raise ValueError(
            "ScoreAuthority declares an official BMM aggregate column "
            f"'{bmm_col}' but no benchmark_data was provided."
        )
    if benchmark_data is not None:
        requested = list(benchmark_cols) if benchmark_cols is not None else []
        bench_value_cols = list(dict.fromkeys(requested + ([bmm_col] if bmm_col else [])))
        if not bench_value_cols:
            raise ValueError(
                "benchmark_data was provided but no benchmark_cols were requested "
                "and the authority declares no official BMM aggregate column."
            )
        _require_columns(
            benchmark_data, [id_col, *bench_value_cols], source="Benchmark"
        )
        bench_cols = [id_col, *bench_value_cols] + (
            [era_col] if era_col in benchmark_data.columns else []
        )
        bench = benchmark_data[bench_cols].copy()
        _validate_keys(
            bench,
            source="Benchmark",
            id_col=id_col,
            era_col=era_col if era_col in bench.columns else None,
        )
        _validate_finite(bench, bench_value_cols, source="Benchmark")
        joined = _merge_by_id(
            joined,
            bench,
            bench_value_cols,
            id_col=id_col,
            era_col=era_col,
            right_has_era=era_col in bench.columns,
            right_source="Benchmark",
            require_full_coverage=True,
        )
        for column in requested:
            reference_cols[f"corr_with_{column}"] = f"__joined_{column}"
        if bmm_col is not None:
            contribution_cols["bmc"] = f"__joined_{bmm_col}"

    joined = joined.rename(columns={id_col: "__scoring_id"})
    joined = joined.sort_values(
        [era_col, "__scoring_id"], kind="stable"
    ).reset_index(drop=True)

    target_joined = f"__joined_{target_col}"
    rows: dict[str, dict[str, float]] = {}
    for era in sorted(joined[era_col].unique().tolist()):
        sub = joined[joined[era_col] == era]
        rows[era] = _score_one_era(
            sub,
            prediction_col=prediction_col,
            target_col=target_joined,
            reference_cols=reference_cols,
            contribution_cols=contribution_cols,
        )

    per_era = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    per_era.index.name = era_col
    per_era["weighted_score"] = (
        per_era["corr"] * float(authority.corr_multiplier)
        + per_era["mmc"] * float(authority.mmc_multiplier)
    )
    ordered = ["corr", "mmc", "weighted_score"] + [
        col for col in per_era.columns if col not in ("corr", "mmc", "weighted_score")
    ]
    per_era = per_era[ordered]

    summary = summarize_round0(
        per_era,
        authority,
        recent_window=recent_window,
        block_size=block_size,
        excluded_eras=uncovered if meta_model_policy == META_MODEL_POLICY_EXCLUDE else (),
    )
    return Round0Result(
        per_era=per_era,
        summary=summary,
        excluded_eras=uncovered
        if meta_model_policy == META_MODEL_POLICY_EXCLUDE
        else (),
        authority=authority,
    )


def _series_summary(scores: pd.Series, *, recent_window: int, block_size: int) -> dict:
    values = scores.to_numpy(dtype="float64")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Per-era scores for '{scores.name}' must be finite and non-empty.")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    sharpe = float(mean / std) if std != 0.0 else float("nan")
    equity_curve = np.concatenate(([0.0], np.cumsum(values)))
    running_max = np.maximum.accumulate(equity_curve)
    max_drawdown = float(np.max(running_max - equity_curve))
    recent_used = int(min(recent_window, values.size))
    recent_mean = float(np.mean(values[-recent_used:]))
    blocks = []
    eras = scores.index.tolist()
    for start in range(0, values.size, block_size):
        chunk = values[start : start + block_size]
        blocks.append(
            {
                "start_era": eras[start],
                "end_era": eras[min(start + block_size, values.size) - 1],
                "n_eras": int(chunk.size),
                "mean": float(np.mean(chunk)),
            }
        )
    return {
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "recent_window_used": recent_used,
        "recent_mean": recent_mean,
        "per_block_means": blocks,
    }


def summarize_round0(
    per_era: pd.DataFrame,
    authority: ScoreAuthority,
    *,
    recent_window: int = 40,
    block_size: int = 52,
    excluded_eras: Sequence[str] = (),
) -> dict:
    """Summarize a per-era score frame. Reproducible from ``per_era`` alone."""
    if recent_window <= 0 or block_size <= 0:
        raise ValueError("recent_window and block_size must be positive.")
    for column in ("corr", "mmc", "weighted_score"):
        if column not in per_era.columns:
            raise ValueError(f"per_era frame is missing required column '{column}'.")
    per_era = per_era.sort_index()
    summary: dict = {
        "n_eras": int(len(per_era)),
        "first_era": per_era.index[0],
        "last_era": per_era.index[-1],
        "recent_window": int(recent_window),
        "block_size": int(block_size),
        "excluded_eras_missing_meta_model": list(excluded_eras),
        "scores": {},
        "diagnostics": {},
        "authority": {
            "payout_target": authority.payout_target,
            "corr_score_name": authority.corr_score_name,
            "mmc_score_name": authority.mmc_score_name,
            "corr_multiplier": float(authority.corr_multiplier),
            "mmc_multiplier": float(authority.mmc_multiplier),
            "meta_model_column": authority.meta_model_column,
            "weighted_model_score_formula": (
                "weighted_model_score = corr * corr_multiplier + mmc * "
                "mmc_multiplier (model-selection score only; staking "
                "settlement mechanics are recorded separately in the "
                "score-authority record and never influence selection)"
            ),
        },
        "conventions": {
            "sharpe": SHARPE_CONVENTION,
            "max_drawdown": DRAWDOWN_CONVENTION,
            "prediction_reference_corr": PREDICTION_CORR_CONVENTION,
            "era_ordering": ERA_ORDER_CONVENTION,
        },
    }
    score_cols = ["corr", "mmc", "weighted_score"] + (
        ["bmc"] if "bmc" in per_era.columns else []
    )
    for column in score_cols:
        summary["scores"][column] = _series_summary(
            per_era[column], recent_window=recent_window, block_size=block_size
        )
    for column in per_era.columns:
        if column.startswith("corr_with_"):
            summary["diagnostics"][column] = {
                "mean": float(np.mean(per_era[column].to_numpy(dtype="float64")))
            }
    if authority.bmm_aggregate_authority is None:
        summary["bmc"] = {
            "status": "unavailable",
            "reason": BMC_AGGREGATE_NOT_REPRODUCIBLE_FROM_PUBLISHED_FILES,
            "detail": (
                "No official historical Benchmark Meta Model aggregate (or its "
                "stake weights) is published; a single benchmark column or an "
                "equal-weight average is not the official BMM and is not "
                "reported as BMC."
            ),
        }
    else:
        summary["bmc"] = {
            "status": "official",
            "bmm_aggregate_authority": authority.bmm_aggregate_authority,
        }
    return summary
