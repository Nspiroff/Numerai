"""Keystone Round-1 walk-forward laws (KW33).

Pure, deterministic functions that define the frozen Round-1 protocol:
fold boundaries, the eight-era embargo, training-era eligibility, the
era-balanced deterministic sampling law, fit-spec generation, score-frame
projection, forbidden-era guards, and the pre-registered decision law.

Everything here is synthetic-testable without any dataset. The numeric
constants mirror ``round1_protocol.json``; the protocol record is the
authority, this module is the executable form. No bare ``target`` alias is
used anywhere; the payout target arrives via the Round-0 ``ScoreAuthority``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from agents.code.metrics.keystone_round0 import ScoreAuthority

# ---------------------------------------------------------------- frozen zone
SCORE_ZONE_START = 1133
SCORE_ZONE_END = 1219
WALK_FORWARD_BLOCKS: tuple[tuple[int, int], ...] = (
    (1133, 1147),
    (1148, 1162),
    (1163, 1177),
    (1178, 1192),
    (1193, 1207),
    (1208, 1219),
)
EMBARGO_ERAS = 8
GAP_START = 1223
GAP_END = 1230
HOLDOUT_START = 1231
VALIDATION_FIRST_ERA = 575

SAMPLING_SEED = 20260817
MAX_SAMPLED_ROWS = 1_000_000
MODEL_SEEDS = (42, 1337, 2024)

CONTROL_T = "control_t"
CANDIDATE_V = "candidate_v"

ENDER60_AUX_LABEL = "HYPOTHETICAL_CUTOVER_WEIGHTED_DIAGNOSTIC"


def _era(value: int) -> str:
    return f"{value:04d}"


def score_zone_eras() -> list[str]:
    """All 87 scored eras, chronological."""
    return [_era(e) for e in range(SCORE_ZONE_START, SCORE_ZONE_END + 1)]


def block_eras(block_index: int) -> list[str]:
    start, end = WALK_FORWARD_BLOCKS[block_index]
    return [_era(e) for e in range(start, end + 1)]


def latest_eligible_validation_era(scored_start_era: str) -> str:
    """Latest validation era a fit may train on: ``scored_start - 9``.

    This leaves the eight eras ``scored_start-8 .. scored_start-1`` embargoed
    (excluded from both training and scoring for that fold).
    """
    return _era(int(scored_start_era) - (EMBARGO_ERAS + 1))


def embargo_eras_for_block(block_index: int) -> list[str]:
    start = WALK_FORWARD_BLOCKS[block_index][0]
    return [_era(e) for e in range(start - EMBARGO_ERAS, start)]


def eligible_validation_eras(
    block_index: int, available_validation_eras: Sequence[str]
) -> list[str]:
    """Validation eras a Candidate-V fit for ``block_index`` may train on."""
    latest = latest_eligible_validation_era(_era(WALK_FORWARD_BLOCKS[block_index][0]))
    eligible = sorted(
        era
        for era in available_validation_eras
        if _era(VALIDATION_FIRST_ERA) <= era <= latest
    )
    assert_no_forbidden_eras(eligible, context=f"candidate_v block {block_index} training")
    return eligible


def assert_no_forbidden_eras(eras: Sequence[str], *, context: str) -> None:
    """Fail loudly if any GAP (1223-1230) or HOLDOUT (>=1231) era appears."""
    forbidden = sorted(e for e in eras if int(e) >= GAP_START)
    if forbidden:
        raise ValueError(
            f"{context}: forbidden GAP/HOLDOUT eras present: {forbidden}. "
            "Eras >= 1223 may never be loaded for training, scoring, metric "
            "calculation, or model selection in Round 1."
        )


def project_to_scored_eras(era_values: Sequence[str]) -> list[bool]:
    """Row mask restricting any frame to exactly the scored zone."""
    zone = set(score_zone_eras())
    return [era in zone for era in era_values]


# ------------------------------------------------------------------- sampling
def era_balanced_sample_positions(
    era_of_row: Sequence[str],
    cap: int = MAX_SAMPLED_ROWS,
    seed: int = SAMPLING_SEED,
) -> np.ndarray:
    """Deterministic era-balanced row sampling law.

    Input: the era value of every candidate training row, in a fixed caller
    order (the caller must present rows sorted by ``(era, id)`` so the law is
    order-independent of storage layout). Output: sorted positional indices of
    the selected rows.

    Law: if the cohort fits under ``cap``, take every row. Otherwise give each
    era an equal quota ``cap // n_eras``; eras with fewer rows contribute all
    of theirs; remaining capacity is then granted one era at a time in
    ascending era order from each era's remaining rows, in that era's fixed
    random permutation order. Per-era permutations come from
    ``SeedSequence([seed, int(era)])`` — a function of the sampling seed and
    the era only, never of the model seed.
    """
    eras = np.asarray(era_of_row)
    order = np.arange(len(eras))
    unique_eras = sorted(set(eras.tolist()))
    total = len(eras)
    if total <= cap:
        return order
    per_era_positions = {
        era: order[eras == era] for era in unique_eras
    }
    per_era_perm = {}
    for era in unique_eras:
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(era)]))
        positions = per_era_positions[era]
        per_era_perm[era] = positions[rng.permutation(len(positions))]
    quota = cap // len(unique_eras)
    selected: list[np.ndarray] = []
    remainder_pools: dict[str, np.ndarray] = {}
    taken = 0
    for era in unique_eras:
        perm = per_era_perm[era]
        take = min(quota, len(perm))
        selected.append(perm[:take])
        remainder_pools[era] = perm[take:]
        taken += take
    remaining = cap - taken
    for era in unique_eras:
        if remaining <= 0:
            break
        pool = remainder_pools[era]
        take = min(remaining, len(pool))
        if take:
            selected.append(pool[:take])
            remaining -= take
    result = np.sort(np.concatenate(selected))
    if len(result) != min(cap, total):
        raise AssertionError("sampling law failed to hit the deterministic cap")
    return result


def validate_prediction_vector(ids: Sequence[str], eras: Sequence[str]) -> None:
    """A scored vector must cover the exact 87-era zone with unique ids."""
    id_list = list(ids)
    if len(set(id_list)) != len(id_list):
        raise ValueError("prediction vector contains duplicate ids")
    got = sorted(set(eras))
    zone = score_zone_eras()
    if got != zone:
        raise ValueError(
            "prediction vector eras do not equal the exact 87-era score zone; "
            f"missing={sorted(set(zone) - set(got))} "
            f"unexpected={sorted(set(got) - set(zone))}"
        )
    assert_no_forbidden_eras(got, context="prediction vector")


def zero_baseline_stats(values: Sequence[float]) -> dict:
    """Mean, population std, era Sharpe, and zero-baseline max drawdown.

    Identical semantics to the corrected Round-0 summary conventions
    (``keystone_round0``): the equity curve starts at zero so an initial
    losing streak is counted.
    """
    arr = np.asarray(values, dtype="float64")
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("scores must be finite and non-empty")
    equity = np.concatenate(([0.0], np.cumsum(arr)))
    running = np.maximum.accumulate(equity)
    std = float(np.std(arr, ddof=0))
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "sharpe": float(np.mean(arr) / std) if std else float("nan"),
        "max_drawdown_zero_baseline": float(np.max(running - equity)),
    }


# ------------------------------------------------------------------ fit specs
@dataclass(frozen=True)
class FitSpec:
    fit_id: str
    procedure: str          # CONTROL_T | CANDIDATE_V
    model_seed: int
    block_index: int | None  # None for CONTROL_T (predicts all 87 eras)
    scored_eras: tuple[str, ...]
    uses_validation_history: bool
    latest_validation_train_era: str | None  # None = trains on no validation row

    @property
    def sampling_seed(self) -> int:
        return SAMPLING_SEED  # never varies with the model seed


def build_fit_specs() -> list[FitSpec]:
    """The exact 21-fit Round-1 cohort: 3 CONTROL-T + 18 CANDIDATE-V."""
    specs: list[FitSpec] = []
    for seed in MODEL_SEEDS:
        specs.append(
            FitSpec(
                fit_id=f"{CONTROL_T}_seed{seed}",
                procedure=CONTROL_T,
                model_seed=seed,
                block_index=None,
                scored_eras=tuple(score_zone_eras()),
                uses_validation_history=False,
                latest_validation_train_era=None,
            )
        )
    for block_index in range(len(WALK_FORWARD_BLOCKS)):
        for seed in MODEL_SEEDS:
            start = _era(WALK_FORWARD_BLOCKS[block_index][0])
            specs.append(
                FitSpec(
                    fit_id=f"{CANDIDATE_V}_block{block_index + 1}_seed{seed}",
                    procedure=CANDIDATE_V,
                    model_seed=seed,
                    block_index=block_index,
                    scored_eras=tuple(block_eras(block_index)),
                    uses_validation_history=True,
                    latest_validation_train_era=latest_eligible_validation_era(start),
                )
            )
    assert len(specs) == 21
    return specs


def lightgbm_params(profile: Mapping, model_seed: int) -> dict:
    """Frozen profile -> exact LightGBM parameter dict for one fit.

    Only the model seed varies between matched fits; every other field is the
    frozen profile. No early stopping and no eval on scored folds exists
    anywhere in the training path.
    """
    return {
        "objective": "regression",
        "boosting": "gbdt",
        "learning_rate": profile["learning_rate"],
        "max_depth": profile["max_depth"],
        "num_leaves": profile["num_leaves"],
        "min_data_in_leaf": profile["min_data_in_leaf"],
        "feature_fraction": profile["feature_fraction"],
        "seed": model_seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": profile["num_threads"],
        "verbosity": -1,
    }


# -------------------------------------------------------------- decision law
DECISION_THRESHOLDS = {
    "seed_mean_weighted_uplift_min": 0.00025,
    "positive_block_differences_min": 4,
    "worst_seed_mean_max_deficit": 0.00050,
    "worst_seed_sharpe_max_deficit": 0.05,
    "worst_seed_drawdown_max_ratio": 1.25,
    "pipeline_parity_min_corr_fraction": 0.70,
}


def decide_round1(
    candidate: Mapping,
    control: Mapping,
    thresholds: Mapping = DECISION_THRESHOLDS,
) -> dict:
    """Pre-registered Round-1 decision law (Ender20 weighted model score only).

    ``candidate``/``control`` each carry: ``seed_mean_weighted_mean`` (float),
    ``block_means`` (six seed-mean block means, chronological),
    ``worst_seed_weighted_mean``, ``worst_seed_sharpe``,
    ``worst_seed_drawdown``. Ender60 quantities have no input path here and
    can never select the winner.
    """
    diffs = [c - t for c, t in zip(candidate["block_means"], control["block_means"])]
    if len(diffs) != len(WALK_FORWARD_BLOCKS):
        raise ValueError("decision law requires exactly six matched block means")
    conditions = {
        "1_seed_mean_weighted_uplift": {
            "threshold": f">= +{thresholds['seed_mean_weighted_uplift_min']}",
            "value": candidate["seed_mean_weighted_mean"] - control["seed_mean_weighted_mean"],
            "passed": (
                candidate["seed_mean_weighted_mean"] - control["seed_mean_weighted_mean"]
                >= thresholds["seed_mean_weighted_uplift_min"]
            ),
        },
        "2_positive_block_differences": {
            "threshold": f">= {thresholds['positive_block_differences_min']} of 6",
            "value": sum(1 for d in diffs if d > 0),
            "passed": sum(1 for d in diffs if d > 0)
            >= thresholds["positive_block_differences_min"],
        },
        "3_worst_seed_mean_guard": {
            "threshold": f">= control worst-seed mean - {thresholds['worst_seed_mean_max_deficit']}",
            "value": candidate["worst_seed_weighted_mean"] - control["worst_seed_weighted_mean"],
            "passed": (
                candidate["worst_seed_weighted_mean"]
                >= control["worst_seed_weighted_mean"] - thresholds["worst_seed_mean_max_deficit"]
            ),
        },
        "4_worst_seed_sharpe_guard": {
            "threshold": f">= control worst-seed sharpe - {thresholds['worst_seed_sharpe_max_deficit']}",
            "value": candidate["worst_seed_sharpe"] - control["worst_seed_sharpe"],
            "passed": (
                candidate["worst_seed_sharpe"]
                >= control["worst_seed_sharpe"] - thresholds["worst_seed_sharpe_max_deficit"]
            ),
        },
        "5_worst_seed_drawdown_guard": {
            "threshold": f"<= {thresholds['worst_seed_drawdown_max_ratio']} * control worst-seed drawdown",
            "value": candidate["worst_seed_drawdown"],
            "passed": (
                candidate["worst_seed_drawdown"]
                <= thresholds["worst_seed_drawdown_max_ratio"] * control["worst_seed_drawdown"]
            ),
        },
    }
    promoted = all(c["passed"] for c in conditions.values())
    return {
        "promoted": promoted,
        "terminal_state": (
            "KEYSTONE_R1_RECENCY_AUGMENTATION_PROMOTED"
            if promoted
            else "KEYSTONE_R1_NEGATIVE_NO_RECENCY_GAIN"
        ),
        "conditions": conditions,
        "block_differences": diffs,
    }


def moving_block_bootstrap_ci(
    per_era_difference: Sequence[float],
    n_resamples: int = 10_000,
    block_length: int = 8,
    seed: int = SAMPLING_SEED,
    ci: tuple[float, float] = (5.0, 95.0),
) -> dict:
    """Report-only fixed-seed moving-block bootstrap of the mean difference."""
    values = np.asarray(per_era_difference, dtype="float64")
    n = len(values)
    if n < block_length:
        raise ValueError("series shorter than the bootstrap block length")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n - block_length + 1, size=(n_resamples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_length)[None, None, :]).reshape(
        n_resamples, -1
    )[:, :n]
    means = values[idx].mean(axis=1)
    low, high = np.percentile(means, ci)
    return {
        "observed_mean_difference": float(values.mean()),
        "n_resamples": n_resamples,
        "block_length": block_length,
        "seed": seed,
        "ci_percentiles": list(ci),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def pipeline_parity_ok(
    control_mean_corr: float,
    benchmark_mean_corr: float,
    thresholds: Mapping = DECISION_THRESHOLDS,
) -> bool:
    """Sanity gate: CONTROL-T must reach 70% of the benchmark's mean CORR."""
    return control_mean_corr >= thresholds["pipeline_parity_min_corr_fraction"] * benchmark_mean_corr


def aux_authority_ender60(base: ScoreAuthority) -> ScoreAuthority:
    """Ender60 auxiliary scoring authority (cutover-readiness diagnostics only).

    The weighted output under this authority is the
    ``HYPOTHETICAL_CUTOVER_WEIGHTED_DIAGNOSTIC`` — never a payout score and
    never a selection gate; ``decide_round1`` has no input path for it.
    """
    return ScoreAuthority(
        payout_target="target_ender_60",
        corr_multiplier=base.corr_multiplier,
        mmc_multiplier=base.mmc_multiplier,
        meta_model_column=base.meta_model_column,
        corr_score_name="CORR60_RAW_AUXILIARY",
        mmc_score_name="MMC60_RAW_AUXILIARY",
        bmm_aggregate_authority=None,
        retrieved_utc=base.retrieved_utc,
        documentation_authority=base.documentation_authority
        + (
            "Ender60 auxiliary: current 60D round score configs are "
            "non-payout (multiplier 0); any combined weighted number is the "
            + ENDER60_AUX_LABEL
            + " and must never select a winner.",
        ),
    )
