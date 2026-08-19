"""Keystone Round-2 parity-calibration ladder laws (KP35).

Pure, deterministic, dataset-free functions defining the frozen KP35 protocol:
the documented benchmark information boundary, the eight-era purge, the score
zone, the two stage profiles (P1 FALLBACK, P2 documented v5 deep), stage
ordering and eligibility, the one-seed screening law, the two-part final
confirmation law, the exact-row-universe contract, canonical key hashing, the
sample-manifest schema and its frozen composition, attempt/retry custody, the
strict prior-result authority, fit-log provenance validation, runtime-version
binding, the terminal-state transition law, and the forbidden-era guards.

Scope. KP35 is *parity calibration only*. It asks whether either of the two
PROVEN mismatches between the KW33 static CONTROL backbone and the documented
v5 benchmark procedure -- the training-history boundary, or the LightGBM
parameter profile -- restores benchmark-plausible CORR on eras 1133-1219. It
does not test Candidate-V, validation recency promotion, MMC specialists,
feature ensembles, target ensembles, blending, deployment, or live performance.

Invariants enforced here rather than merely documented:

* there is no Candidate-V stage and no code path that can construct one;
* Ender60 has no input path to any decision function;
* the bare ``target`` dataset alias is rejected as a payout objective;
* GAP (1223-1230) and HOLDOUT (>=1231) eras are refused everywhere;
* CORR alone selects; MMC, weighted score, Sharpe, drawdown, recent-window and
  benchmark correlations are carried as explicitly non-selecting diagnostics;
* every scalar entering a decision must be finite -- NaN and infinities raise
  before a terminal scientific state can be produced, and never masquerade as
  an ordinary screen or confirmation failure;
* the sample identity is a function of the data, era range, feature list,
  sampling law, seed and cap only -- never of the model seed or model profile,
  so P1 and P2 are provably fitted on the identical sampled rows;
* a prior result authorises a successor fit only as a complete, canonically
  located, internally consistent KP35 result envelope -- never because a file
  happens to contain a plausible ``terminal_state`` string;
* at most two attempts exist per (stage, seed), each with its own preserved
  failure path, and a retry requires a validated first failure;
* terminal states are absorbing: no transition may run backward.

This module is the executable form of ``round2_parity_protocol.json``; the
protocol record is the authority and the trainer and evaluator revalidate
agreement between the two before doing anything. Nothing here imports a
gradient-boosting library, a Parquet reader, an API client, a dataset, or a
network client, so every law below is synthetically testable on a bare CPU
runner.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------- frozen zone
SCORE_ZONE_START = 1133
SCORE_ZONE_END = 1219
N_SCORE_ERAS = SCORE_ZONE_END - SCORE_ZONE_START + 1  # 87

#: The complete authoritative scoring universe frozen by Round 1 and
#: independently recomputed at KP35 source freeze from the current data files.
SCORING_UNIVERSE_ROWS = 575_597
SCORING_UNIVERSE_CANON_SHA256 = (
    "91e519aff5c656c9acf7cc6fe74daebfc034650bae47ee4e3889a98ec8fac033"
)

# Documented v5 benchmark walk-forward construction (numerai/docs,
# numerai-tournament/models.md): 156-era prediction chunks; purge 8 eras for
# 20D targets; window n trains on 1 .. 148+156*(n-1) and predicts
# 157+156*(n-1) .. 312+156*(n-1). Window 7 predicts 1093-1248, which is the
# chunk containing the 1133-1219 score zone.
BENCHMARK_CHUNK_START = 1093
BENCHMARK_CHUNK_END = 1248
PURGE_START = 1085
PURGE_END = 1092
PURGE_ERAS = 8
HISTORY_BOUNDARY_END = 1084  # last benchmark-eligible training era

TRAIN_PARQUET_FIRST_ERA = 1
TRAIN_PARQUET_LAST_ERA = 574
VALIDATION_FIRST_ERA = 575

GAP_START = 1223
GAP_END = 1230
HOLDOUT_START = 1231

PAYOUT_TARGET = "target_ender_20"
BARE_TARGET_ALIAS = "target"
FEATURE_SET = "medium"
N_FEATURES = 780
FEATURE_LIST_SHA256 = (
    "dd03cd099eb2c2283786eb123fe13a374460251b49f4717ec5ad34cabede80ba"
)

BENCHMARK_COLUMN = "v53_lgbm_ender20"
BENCHMARK_60_COLUMN = "v53_lgbm_ender60"
META_MODEL_COLUMN = "numerai_meta_model"

SAMPLING_SEED = 20260817
MAX_SAMPLED_ROWS = 1_000_000
SAMPLING_LAW_VERSION = "kw33_era_balanced_v1"

SCREENING_SEED = 42
CONFIRMATION_SEEDS = (1337, 2024)
ALL_SEEDS = (SCREENING_SEED,) + CONFIRMATION_SEEDS

# ---------------------------------------------------------- frozen thresholds
#: KW33's published benchmark mean CORR over the exact 87-era zone. Every
#: future evaluation recomputes this from the published benchmark column and
#: requires identity with this value within ``BENCHMARK_MEAN_CORR_TOLERANCE``.
BENCHMARK_MEAN_CORR = 0.02094843151562169
BENCHMARK_MEAN_CORR_TOLERANCE = 1e-12

#: Final parity fraction. Unchanged and not weakened by the screen.
FINAL_PARITY_FRACTION = 0.70

#: One-seed screen factor = 0.70 * (1 - 0.0350), where 0.0350 is the worst
#: single-seed deviation below the three-seed mean ever observed in the KW33
#: CONTROL-T cohort (seed 2024, -3.50%). The screen therefore errs low by
#: exactly the observed dispersion and by no more.
SCREEN_SEED_DISPERSION_ALLOWANCE = 0.0350
SCREEN_FACTOR = 0.6755

#: Exact frozen thresholds (IEEE-754 doubles, reproduced at source freeze).
SCREEN_THRESHOLD = 0.014150665488802451
FINAL_THREE_SEED_THRESHOLD = 0.014663902060935183
UNTOUCHED_PAIR_THRESHOLD = 0.014663902060935183

# ------------------------------------------------------------------- profiles
#: The exact KW33 FALLBACK profile. P1 changes nothing about the model.
P1_PROFILE: Mapping[str, object] = {
    "name": "FALLBACK",
    "objective": "regression",
    "num_trees": 6000,
    "learning_rate": 0.005,
    "max_depth": 8,
    "num_leaves": 255,
    "min_data_in_leaf": 10000,
    "feature_fraction": 0.1,
    "num_threads": 8,
    "deterministic": True,
    "force_row_wise": True,
    "device": "cpu",
    "no_early_stopping": True,
    "no_evaluation_set": True,
}

#: The documented v5 deep profile (numerai/docs, models.md ``deep_lgbm_params``).
P2_PROFILE: Mapping[str, object] = {
    "name": "DOCUMENTED_V5_DEEP",
    "objective": "regression",
    "num_trees": 30000,
    "learning_rate": 0.001,
    "max_depth": 10,
    "num_leaves": 1024,
    "min_data_in_leaf": 10000,
    "feature_fraction": 0.1,
    "num_threads": 8,
    "deterministic": True,
    "force_row_wise": True,
    "device": "cpu",
    "no_early_stopping": True,
    "no_evaluation_set": True,
}

#: The only fields in which P2 is permitted to differ from P1. Any other
#: divergence is a protocol violation, not a configuration choice.
DECLARED_PROFILE_DIFFERENCE_FIELDS: frozenset[str] = frozenset(
    {"name", "num_trees", "learning_rate", "max_depth", "num_leaves"}
)

# --------------------------------------------------------------------- stages
P1 = "P1_HISTORY_BOUNDARY_1084"
P2 = "P2_DEEP_PROFILE"
STAGES: tuple[str, ...] = (P1, P2)
STAGE_PROFILES: Mapping[str, Mapping[str, object]] = {P1: P1_PROFILE, P2: P2_PROFILE}

MODE_SCREEN = "screen"
MODE_CONFIRMATION = "confirmation"
MODES: tuple[str, ...] = (MODE_SCREEN, MODE_CONFIRMATION)

RECORD_SCREEN = "kp35_screen_result"
RECORD_CONFIRMATION = "kp35_confirmation_result"
RECORD_FIT_LOG = "kp35_fit_log"
RECORD_SAMPLE = "kp35_sample_identity"
MODE_RECORDS: Mapping[str, str] = {
    MODE_SCREEN: RECORD_SCREEN,
    MODE_CONFIRMATION: RECORD_CONFIRMATION,
}

#: Substring guard: any stage name containing this token is refused outright,
#: so a near-miss spelling cannot slip past the exact-name list below.
FORBIDDEN_STAGE_SUBSTRING = "candidate"

#: The confirmation record's explicit denial field. Even a confirmed parity
#: backbone authorises no return to Candidate-V; that requires a separately
#: reviewed recency experiment.
CANDIDATE_V_RETURN_DENIAL_KEY = "candidate_v_return_authorized"

#: Names that must never resolve to a KP35 stage. Candidate-V is a Round-1
#: procedure; no parity-calibration path may reach it, and no future parity
#: pass authorises a return to it.
FORBIDDEN_STAGE_NAMES: frozenset[str] = frozenset(
    {
        "candidate_v",
        "CANDIDATE_V",
        "candidate-v",
        "CandidateV",
        "P3_FEATURE_UNIVERSE_ALL",
        "P4_ROW_BUDGET_FULL",
        "MAXIMALLY_RECENT_STATIC_OOS_BACKBONE",
    }
)

# ------------------------------------------------------------ terminal states
KP35_SOURCE_FROZEN = "KP35_SOURCE_FROZEN_AWAITING_INDEPENDENT_REVIEW"
KP35_P1_SCREEN_PASSED = "KP35_P1_SCREEN_PASSED_AWAITING_CONFIRMATION"
KP35_P1_SCREEN_FAILED = "KP35_P1_SCREEN_FAILED_P2_AUTHORIZABLE"
KP35_P1_CONFIRMATION_FAILED = "KP35_P1_CONFIRMATION_FAILED_P2_AUTHORIZABLE"
KP35_P2_SCREEN_PASSED = "KP35_P2_SCREEN_PASSED_AWAITING_CONFIRMATION"
KP35_PARITY_NOT_RESTORED = "KP35_PARITY_NOT_RESTORED_BY_PROVEN_MISMATCHES"
KP35_P2_CONFIRMATION_FAILED = "KP35_P2_SCREEN_PASS_CONFIRMATION_FAILED"
KP35_PARITY_CONFIRMED = "KP35_PARITY_BACKBONE_CONFIRMED"

#: (screen pass state, screen failure state) per stage.
STAGE_STATES: Mapping[str, tuple[str, str]] = {
    P1: (KP35_P1_SCREEN_PASSED, KP35_P1_SCREEN_FAILED),
    P2: (KP35_P2_SCREEN_PASSED, KP35_PARITY_NOT_RESTORED),
}

#: (confirmation pass state, confirmation failure state) per stage. The
#: failure states are deliberately stage-specific: a P1 confirmation failure
#: leaves the deep profile untested and must therefore authorise P2, while a
#: P2 confirmation failure ends the ladder.
CONFIRMATION_STATES: Mapping[str, tuple[str, str]] = {
    P1: (KP35_PARITY_CONFIRMED, KP35_P1_CONFIRMATION_FAILED),
    P2: (KP35_PARITY_CONFIRMED, KP35_P2_CONFIRMATION_FAILED),
}

#: States from which P2 screening may be separately authorised. Both arise
#: from a P1 outcome that leaves the documented deep profile untested, and the
#: registered question asks whether *either* proven mismatch restores parity.
P2_AUTHORIZING_STATES: frozenset[str] = frozenset(
    {KP35_P1_SCREEN_FAILED, KP35_P1_CONFIRMATION_FAILED}
)

#: Absorbing states. Nothing in this gate may transition out of one.
ABSORBING_STATES: frozenset[str] = frozenset(
    {
        KP35_PARITY_NOT_RESTORED,
        KP35_PARITY_CONFIRMED,
        KP35_P2_CONFIRMATION_FAILED,
    }
)

#: The complete forward transition law. A state absent from a key's value set
#: is unreachable from that key, which makes every backward move an error.
FORWARD_TRANSITIONS: Mapping[str, frozenset[str]] = {
    KP35_SOURCE_FROZEN: frozenset({KP35_P1_SCREEN_PASSED, KP35_P1_SCREEN_FAILED}),
    KP35_P1_SCREEN_PASSED: frozenset(
        {KP35_PARITY_CONFIRMED, KP35_P1_CONFIRMATION_FAILED}
    ),
    KP35_P1_SCREEN_FAILED: frozenset(
        {KP35_P2_SCREEN_PASSED, KP35_PARITY_NOT_RESTORED}
    ),
    KP35_P1_CONFIRMATION_FAILED: frozenset(
        {KP35_P2_SCREEN_PASSED, KP35_PARITY_NOT_RESTORED}
    ),
    KP35_P2_SCREEN_PASSED: frozenset(
        {KP35_PARITY_CONFIRMED, KP35_P2_CONFIRMATION_FAILED}
    ),
    KP35_PARITY_NOT_RESTORED: frozenset(),
    KP35_PARITY_CONFIRMED: frozenset(),
    KP35_P2_CONFIRMATION_FAILED: frozenset(),
}

# ------------------------------------------------------ non-selecting outputs
#: Quantities that are reported for interpretation but may never enter a
#: parity decision. Ender60 in every form lives here permanently.
NON_SELECTING_DIAGNOSTICS: frozenset[str] = frozenset(
    {
        "mmc",
        "mmc60",
        "corr60",
        "weighted_score",
        "sharpe",
        "max_drawdown_zero_baseline",
        "recent_20",
        "benchmark_correlation",
        "bmc",
        "ender60",
        "hypothetical_cutover_weighted_diagnostic",
    }
)
ENDER60_AUX_LABEL = "HYPOTHETICAL_CUTOVER_WEIGHTED_DIAGNOSTIC"


# --------------------------------------------------------------- error types
class ExactRowUniverseError(ValueError):
    """Raised when two frames do not cover the identical ``(era, id)`` universe."""


class StageAuthorityError(ValueError):
    """Raised when a stage/seed/state combination is not authorised."""


class PriorResultAuthorityError(ValueError):
    """Raised when a claimed prior result is not a valid KP35 result envelope."""


class FitProvenanceError(ValueError):
    """Raised when a fit log or prediction artifact fails provenance validation."""


class SampleCustodyError(ValueError):
    """Raised when a sample manifest does not reproduce the frozen composition.

    This is an infrastructure, data, or implementation stop. It is never a
    model result and must never be reported as a screen or confirmation
    failure.
    """


class EnvironmentBindingError(ValueError):
    """Raised when the runtime environment differs from the frozen protocol."""


class NonFiniteValueError(ValueError):
    """Raised when a NaN or infinite value reaches a decision input."""


# ------------------------------------------------------- finite-value discipline
def assert_finite_scalar(value: object, *, name: str) -> float:
    """Return ``value`` as a finite float, or raise.

    Every scalar that can move the state machine passes through here first.
    A NaN or infinity must never pass a screen, become an ordinary screen
    failure, become a confirmation failure, or otherwise advance the ladder --
    it is a computation fault and is raised as one.
    """
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise NonFiniteValueError(f"{name} is not a real number: {value!r}") from exc
    if not math.isfinite(number):
        raise NonFiniteValueError(
            f"{name} is not finite ({number!r}); a non-finite value may never "
            "pass a screen, become a screen or confirmation failure, or move "
            "the KP35 state machine"
        )
    return number


# ------------------------------------------------------------------- era laws
def _era(value: int) -> str:
    return f"{value:04d}"


def score_zone_eras() -> list[str]:
    """The exact 87 scored eras, chronological."""
    return [_era(e) for e in range(SCORE_ZONE_START, SCORE_ZONE_END + 1)]


def purge_eras() -> list[str]:
    """The exact eight purged eras 1085-1092, excluded from training and scoring."""
    return [_era(e) for e in range(PURGE_START, PURGE_END + 1)]


def benchmark_chunk_eras() -> list[str]:
    """The documented 156-era benchmark prediction chunk containing the zone."""
    return [_era(e) for e in range(BENCHMARK_CHUNK_START, BENCHMARK_CHUNK_END + 1)]


def eligible_training_eras() -> list[str]:
    """Every era a KP35 fit may train on: 0001-1084, nothing later."""
    return [_era(e) for e in range(TRAIN_PARQUET_FIRST_ERA, HISTORY_BOUNDARY_END + 1)]


def training_era_source(era: str) -> str:
    """Which parquet an eligible training era comes from."""
    value = int(era)
    if TRAIN_PARQUET_FIRST_ERA <= value <= TRAIN_PARQUET_LAST_ERA:
        return "train"
    if VALIDATION_FIRST_ERA <= value <= HISTORY_BOUNDARY_END:
        return "validation"
    raise ValueError(f"era {era} is not eligible KP35 training history")


def derive_history_boundary(score_zone_start: int = SCORE_ZONE_START) -> dict:
    """Re-derive the documented boundary from the published window arithmetic.

    Window ``n`` trains on eras ``1 .. 148 + 156*(n-1)``, purges the following
    eight eras for a 20D target, and predicts
    ``157 + 156*(n-1) .. 312 + 156*(n-1)``. This function finds the window
    whose prediction chunk contains ``score_zone_start`` and returns its
    boundary, rather than trusting the constants above.
    """
    for n in range(1, 64):
        k = n - 1
        train_end = 148 + 156 * k
        predict_start = 157 + 156 * k
        predict_end = 312 + 156 * k
        if predict_start <= score_zone_start <= predict_end:
            return {
                "window": n,
                "train_end": train_end,
                "purge_start": train_end + 1,
                "purge_end": train_end + PURGE_ERAS,
                "predict_start": predict_start,
                "predict_end": predict_end,
            }
    raise ValueError(f"no documented window contains era {score_zone_start}")


def assert_no_forbidden_eras(eras: Sequence[str], *, context: str) -> None:
    """Fail loudly if any GAP (1223-1230) or HOLDOUT (>=1231) era appears."""
    forbidden = sorted({e for e in eras if int(e) >= GAP_START})
    if forbidden:
        raise ValueError(
            f"{context}: forbidden GAP/HOLDOUT eras present: {forbidden}. "
            f"Eras >= {GAP_START} may never be loaded for training, scoring, "
            "metric calculation, or model selection in KP35."
        )


def assert_training_eras_authorized(eras: Sequence[str], *, context: str) -> None:
    """Training may use 0001-1084 only: no purge era, no zone era, no GAP/HOLDOUT."""
    assert_no_forbidden_eras(eras, context=context)
    late = sorted({e for e in eras if int(e) > HISTORY_BOUNDARY_END})
    if late:
        raise ValueError(
            f"{context}: eras beyond the documented boundary "
            f"{HISTORY_BOUNDARY_END} present in training history: {late}"
        )
    early = sorted({e for e in eras if int(e) < TRAIN_PARQUET_FIRST_ERA})
    if early:
        raise ValueError(f"{context}: eras before 0001 present: {early}")


def assert_scoring_zone_exact(eras: Iterable[str], *, context: str) -> None:
    """Scoring must cover exactly eras 1133-1219 -- no subset, no superset."""
    got = sorted(set(eras))
    zone = score_zone_eras()
    if got != zone:
        raise ValueError(
            f"{context}: scored eras are not the exact 87-era zone; "
            f"missing={sorted(set(zone) - set(got))} "
            f"unexpected={sorted(set(got) - set(zone))}"
        )
    assert_no_forbidden_eras(got, context=context)


def assert_payout_target(target: str) -> str:
    """Reject the bare ``target`` dataset alias; require the payout objective."""
    if target == BARE_TARGET_ALIAS:
        raise ValueError(
            "bare `target` names a dataset default, not a payout objective; "
            f"KP35 requires the explicit payout target {PAYOUT_TARGET!r}"
        )
    if target != PAYOUT_TARGET:
        raise ValueError(f"KP35 payout target is {PAYOUT_TARGET!r}, got {target!r}")
    return target


def assert_non_selecting(name: str) -> None:
    """Refuse to let a diagnostic quantity reach a selection path."""
    if name.lower() in NON_SELECTING_DIAGNOSTICS:
        raise ValueError(
            f"{name!r} is a non-selecting KP35 diagnostic and may never enter a "
            "parity decision; parity selection uses CORR only"
        )


# ---------------------------------------------------------------- canonical keys
def canonical_keys(
    eras: Sequence[str], ids: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    """The ``(era, id)`` universe in canonical order: stable sort by era then id."""
    if len(eras) != len(ids):
        raise ValueError("era and id sequences have different lengths")
    return tuple(sorted(zip((str(e) for e in eras), (str(i) for i in ids))))


def canonical_key_hash(eras: Sequence[str], ids: Sequence[str]) -> str:
    """SHA-256 over canonically ordered ``"era,id"`` lines, UTF-8 encoded.

    Order-invariant by construction (the input is sorted first) and
    content-sensitive (any changed, added or removed pair changes the digest).
    This is the identical canonicalisation used by the KP34 audit, whose
    published universe digest was independently reproduced at source freeze.
    """
    blob = "\n".join(f"{era},{ident}" for era, ident in canonical_keys(eras, ids))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def per_era_counts(eras: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for era in eras:
        key = str(era)
        counts[key] = counts.get(key, 0) + 1
    return counts


def canonical_json_sha256(payload: object) -> str:
    """SHA-256 of a canonical JSON serialisation of a parsed object.

    Computed from the *parsed* structure rather than file bytes, so indentation
    and key order in the on-disk file are irrelevant while any change to a key
    or value changes the digest.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def protocol_semantic_sha256(protocol: Mapping) -> str:
    """The binding identity of a protocol record."""
    return canonical_json_sha256(protocol)


def normalize_relpath(value: str) -> str:
    """Normalise a relative artifact path for canonical-location comparison."""
    text = str(value).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")


# ------------------------------------------------------- exact-row contract (J)
@dataclass(frozen=True)
class RowUniverse:
    """A named ``(era, id)`` universe carried by one frame."""

    name: str
    eras: tuple[str, ...]
    ids: tuple[str, ...]

    @classmethod
    def from_columns(cls, name: str, eras: Sequence[str], ids: Sequence[str]):
        return cls(
            name=name,
            eras=tuple(str(e) for e in eras),
            ids=tuple(str(i) for i in ids),
        )

    @property
    def n_rows(self) -> int:
        return len(self.ids)

    @property
    def era_set(self) -> frozenset[str]:
        return frozenset(self.eras)

    @property
    def keys(self) -> tuple[tuple[str, str], ...]:
        return canonical_keys(self.eras, self.ids)

    @property
    def canon_sha256(self) -> str:
        return canonical_key_hash(self.eras, self.ids)

    @property
    def per_era_rows(self) -> dict[str, int]:
        return per_era_counts(self.eras)

    def duplicate_ids(self) -> list[str]:
        seen: dict[str, int] = {}
        for ident in self.ids:
            seen[ident] = seen.get(ident, 0) + 1
        return sorted(k for k, v in seen.items() if v > 1)


def assert_exact_row_universe(
    reference: RowUniverse,
    *others: RowUniverse,
    expected_eras: Sequence[str] | None = None,
    expected_rows: int | None = None,
    expected_canon_sha256: str | None = None,
) -> dict:
    """Strict equality of the complete canonical ``(era, id)`` universe.

    This is the prospective repair of the KW33 source-contract gap. Round 1's
    frozen sources checked era-set coverage and left row-level identity to a
    join, so a strict subset of the scoring universe would have scored without
    complaint. KP35 refuses any of: a strict subset, a strict superset, a
    missing row, an extra row, a duplicate id, an era disagreement, or a row
    carrying an unexpected era.

    Compared for every frame: total rows, era set, per-era row counts, the
    complete sorted ``(era, id)`` pairs, and the canonical SHA-256.

    No Round-1 file is modified; this contract lives only in the KP35 packet.
    """
    frames = (reference,) + others
    if (
        len(frames) < 2
        and expected_canon_sha256 is None
        and expected_eras is None
        and expected_rows is None
    ):
        raise ValueError("nothing to compare: pass another frame or an expectation")

    for frame in frames:
        duplicates = frame.duplicate_ids()
        if duplicates:
            raise ExactRowUniverseError(
                f"{frame.name}: duplicate ids present "
                f"({len(duplicates)}), e.g. {duplicates[:5]}"
            )
        if expected_eras is not None:
            unexpected = sorted(frame.era_set - set(expected_eras))
            if unexpected:
                raise ExactRowUniverseError(
                    f"{frame.name}: rows carry unexpected eras {unexpected}"
                )
            missing = sorted(set(expected_eras) - frame.era_set)
            if missing:
                raise ExactRowUniverseError(
                    f"{frame.name}: expected eras absent {missing}"
                )
        assert_no_forbidden_eras(frame.eras, context=frame.name)

    ref_keys = set(reference.keys)
    for frame in others:
        if frame.n_rows != reference.n_rows:
            raise ExactRowUniverseError(
                f"{frame.name}: row count {frame.n_rows} != "
                f"{reference.name} row count {reference.n_rows}"
            )
        if frame.era_set != reference.era_set:
            raise ExactRowUniverseError(
                f"{frame.name}: era set disagrees with {reference.name}; "
                f"missing={sorted(reference.era_set - frame.era_set)} "
                f"extra={sorted(frame.era_set - reference.era_set)}"
            )
        if frame.per_era_rows != reference.per_era_rows:
            differing = sorted(
                era
                for era in reference.era_set
                if frame.per_era_rows.get(era) != reference.per_era_rows.get(era)
            )
            raise ExactRowUniverseError(
                f"{frame.name}: per-era row counts disagree with "
                f"{reference.name} in eras {differing[:10]}"
            )
        frame_keys = set(frame.keys)
        missing = ref_keys - frame_keys
        extra = frame_keys - ref_keys
        if missing or extra:
            raise ExactRowUniverseError(
                f"{frame.name}: (era,id) universe differs from {reference.name}; "
                f"{len(missing)} missing e.g. {sorted(missing)[:3]}; "
                f"{len(extra)} extra e.g. {sorted(extra)[:3]}"
            )
        if frame.canon_sha256 != reference.canon_sha256:
            raise ExactRowUniverseError(
                f"{frame.name}: canonical hash {frame.canon_sha256} != "
                f"{reference.name} {reference.canon_sha256}"
            )

    if expected_rows is not None and reference.n_rows != expected_rows:
        raise ExactRowUniverseError(
            f"{reference.name}: {reference.n_rows} rows != expected {expected_rows}"
        )
    if (
        expected_canon_sha256 is not None
        and reference.canon_sha256 != expected_canon_sha256
    ):
        raise ExactRowUniverseError(
            f"{reference.name}: canonical hash {reference.canon_sha256} != "
            f"expected {expected_canon_sha256}"
        )
    return {
        "frames": [f.name for f in frames],
        "n_rows": reference.n_rows,
        "n_eras": len(reference.era_set),
        "canon_sha256": reference.canon_sha256,
        "identical": True,
    }


def assert_finite_predictions(values: Sequence[float], *, context: str) -> None:
    """Any non-finite prediction is a hard failure, never a silently dropped row."""
    array = np.asarray(values, dtype="float64")
    if array.size == 0:
        raise ValueError(f"{context}: empty prediction vector")
    if not np.isfinite(array).all():
        bad = int((~np.isfinite(array)).sum())
        raise ValueError(f"{context}: {bad} non-finite prediction value(s)")


# ------------------------------------------------------------ sampling law (K)
def era_balanced_sample_positions(
    era_of_row: Sequence[str],
    cap: int = MAX_SAMPLED_ROWS,
    seed: int = SAMPLING_SEED,
) -> np.ndarray:
    """The exact frozen KW33 deterministic era-balanced sampling law.

    Reproduced verbatim from ``round1_lib.era_balanced_sample_positions`` so
    KP35 samples under an identical law without importing or mutating any
    merged Round-1 file. The caller must present rows sorted by ``(era, id)``.

    Law: under the cap, take every row. Otherwise grant each era an equal quota
    ``cap // n_eras``; eras with fewer rows contribute all of theirs; remaining
    capacity is granted one era at a time in ascending era order from that
    era's fixed random permutation. Per-era permutations come from
    ``SeedSequence([seed, int(era)])`` -- a function of the sampling seed and
    the era only, never of the model seed or the model profile.
    """
    eras = np.asarray(era_of_row)
    order = np.arange(len(eras))
    unique_eras = sorted(set(eras.tolist()))
    total = len(eras)
    if total <= cap:
        return order
    per_era_positions = {era: order[eras == era] for era in unique_eras}
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


# --------------------------------------------------------- sample manifest (K)
SAMPLE_MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "record",
    "data_identities",
    "eligible_era_range",
    "n_eligible_eras",
    "feature_set",
    "feature_list_sha256",
    "n_features",
    "sampling_law_version",
    "sampling_seed",
    "row_cap",
    "rows_before_sampling",
    "selected_row_count",
    "selected_rows_per_era",
    "source_split_rows",
    "sample_canon_sha256",
    "sample_identity_sha256",
)

#: Fields that must never contribute to the parameter-independent sample
#: identity. If any of these could change the identity, P1 and P2 could not be
#: proven to share a sample.
SAMPLE_IDENTITY_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {"model_seed", "stage", "profile", "num_trees", "learning_rate",
     "max_depth", "num_leaves", "params_sha256"}
)

#: The complete sample composition computed at source freeze by a keys-only,
#: explicitly non-scientific audit (no model feature loaded, no training, no
#: prediction). Every future fit must reproduce all of it before LightGBM is
#: called. A mismatch is an infrastructure/data/implementation stop.
FROZEN_SAMPLE: Mapping[str, object] = {
    "eligible_era_range": ["0001", "1084"],
    "n_eligible_eras": 1084,
    "rows_before_sampling": 5_890_287,
    "selected_row_count": 1_000_000,
    "source_split_rows": {"train": 529_780, "validation": 470_220},
    "sample_canon_sha256": (
        "e555e848770f4acd276020aca833541e8b0702a2f1b7c3ebc8068b657d101350"
    ),
    "sampling_seed": SAMPLING_SEED,
    "row_cap": MAX_SAMPLED_ROWS,
    "sampling_law_version": SAMPLING_LAW_VERSION,
    "feature_set": FEATURE_SET,
    "feature_list_sha256": FEATURE_LIST_SHA256,
    "n_features": N_FEATURES,
}

#: The custody envelope compared field-for-field when an existing manifest is
#: reloaded -- not merely two hash fields.
SAMPLE_CUSTODY_ENVELOPE_FIELDS: tuple[str, ...] = (
    "data_identities",
    "eligible_era_range",
    "n_eligible_eras",
    "feature_set",
    "feature_list_sha256",
    "n_features",
    "sampling_law_version",
    "sampling_seed",
    "row_cap",
    "rows_before_sampling",
    "selected_row_count",
    "selected_rows_per_era",
    "source_split_rows",
    "sample_canon_sha256",
    "sample_identity_sha256",
)


def sample_identity(
    *,
    data_identities: Mapping[str, str],
    eligible_era_range: Sequence[str],
    feature_list_sha256: str,
    sampling_law_version: str,
    sampling_seed: int,
    row_cap: int,
    sample_canon_sha256: str,
) -> str:
    """Parameter-independent sample identity.

    Deliberately a function of the data, the eligible era range, the feature
    list, the sampling law, its seed, the row cap and the resulting canonical
    ``(era, id)`` universe -- and of nothing else. The model seed and the model
    profile are structurally excluded, which is what lets P2 prove it reused
    P1's exact sampled rows instead of regenerating an allegedly equivalent
    sample.
    """
    parts = [
        "kp35_sample_identity_v1",
        "|".join(f"{k}={data_identities[k]}" for k in sorted(data_identities)),
        f"eras={eligible_era_range[0]}-{eligible_era_range[-1]}",
        f"features={feature_list_sha256}",
        f"law={sampling_law_version}",
        f"seed={sampling_seed}",
        f"cap={row_cap}",
        f"keys={sample_canon_sha256}",
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def validate_sample_manifest(manifest: Mapping) -> dict:
    """Structural and arithmetic validation of a sample-identity manifest."""
    missing = [f for f in SAMPLE_MANIFEST_REQUIRED_FIELDS if f not in manifest]
    if missing:
        raise ValueError(f"sample manifest missing required fields: {missing}")
    leaked = sorted(SAMPLE_IDENTITY_EXCLUDED_FIELDS & set(manifest))
    if leaked:
        raise ValueError(
            "sample manifest carries parameter-dependent fields that would make "
            f"the sample identity model-dependent: {leaked}"
        )
    if manifest["sampling_seed"] != SAMPLING_SEED:
        raise ValueError(f"sample manifest sampling seed != {SAMPLING_SEED}")
    if manifest["row_cap"] != MAX_SAMPLED_ROWS:
        raise ValueError(f"sample manifest row cap != {MAX_SAMPLED_ROWS}")
    if manifest["sampling_law_version"] != SAMPLING_LAW_VERSION:
        raise ValueError("sample manifest sampling law version is not the frozen law")
    if manifest["feature_list_sha256"] != FEATURE_LIST_SHA256:
        raise ValueError("sample manifest feature-list hash is not the frozen list")
    if manifest["n_features"] != N_FEATURES:
        raise ValueError(f"sample manifest feature count != {N_FEATURES}")
    per_era = manifest["selected_rows_per_era"]
    if sum(per_era.values()) != manifest["selected_row_count"]:
        raise ValueError("selected_rows_per_era does not sum to selected_row_count")
    split = manifest["source_split_rows"]
    if split["train"] + split["validation"] != manifest["selected_row_count"]:
        raise ValueError("source_split_rows does not sum to selected_row_count")
    assert_training_eras_authorized(list(per_era), context="sample manifest")
    expected = sample_identity(
        data_identities=manifest["data_identities"],
        eligible_era_range=manifest["eligible_era_range"],
        feature_list_sha256=manifest["feature_list_sha256"],
        sampling_law_version=manifest["sampling_law_version"],
        sampling_seed=manifest["sampling_seed"],
        row_cap=manifest["row_cap"],
        sample_canon_sha256=manifest["sample_canon_sha256"],
    )
    if expected != manifest["sample_identity_sha256"]:
        raise ValueError(
            "sample_identity_sha256 does not match its own recomputed identity"
        )
    return {"valid": True, "sample_identity_sha256": expected}


def assert_frozen_sample(manifest: Mapping) -> dict:
    """The manifest must reproduce every value frozen at source freeze.

    Called before any LightGBM invocation and again by the evaluator. A
    mismatch here means the data, the sampling implementation, or the
    environment changed -- it is an infrastructure stop, not a model result,
    and it must never be reported as a parity outcome.
    """
    validate_sample_manifest(manifest)
    mismatches: list[str] = []
    for key, expected in FROZEN_SAMPLE.items():
        actual = manifest.get(key)
        if isinstance(expected, list):
            actual = list(actual) if actual is not None else None
        if actual != expected:
            mismatches.append(f"{key}: {actual!r} != frozen {expected!r}")
    if mismatches:
        raise SampleCustodyError(
            "sample manifest does not reproduce the frozen KP35 composition -- "
            "this is an infrastructure, data, or implementation stop and is NOT "
            "a model result: " + "; ".join(mismatches)
        )
    return {
        "frozen_sample_reproduced": True,
        "sample_canon_sha256": manifest["sample_canon_sha256"],
        "sample_identity_sha256": manifest["sample_identity_sha256"],
    }


def assert_sample_envelope_equal(
    existing: Mapping, fresh: Mapping, *, context: str = "sample manifest"
) -> str:
    """Compare the complete custody envelope, not merely two hash fields."""
    for manifest in (existing, fresh):
        validate_sample_manifest(manifest)
    mismatches = [
        f"{field}: recorded {existing.get(field)!r} != recomputed {fresh.get(field)!r}"
        for field in SAMPLE_CUSTODY_ENVELOPE_FIELDS
        if existing.get(field) != fresh.get(field)
    ]
    if mismatches:
        raise SampleCustodyError(
            f"{context}: the recorded sample custody envelope does not match the "
            "freshly computed one: " + "; ".join(mismatches)
        )
    return str(existing["sample_identity_sha256"])


def assert_shared_sample_identity(p1_manifest: Mapping, p2_manifest: Mapping) -> str:
    """P1 and P2 must reference one identical sample, proven by hash."""
    for manifest in (p1_manifest, p2_manifest):
        validate_sample_manifest(manifest)
    if p1_manifest["sample_canon_sha256"] != p2_manifest["sample_canon_sha256"]:
        raise SampleCustodyError(
            "P2 sampled a different (era,id) universe than P1; regenerating an "
            "allegedly equivalent sample is not permitted -- the identities must match"
        )
    if p1_manifest["sample_identity_sha256"] != p2_manifest["sample_identity_sha256"]:
        raise SampleCustodyError("P1 and P2 sample identities differ")
    return str(p1_manifest["sample_identity_sha256"])


# --------------------------------------------------------------- profile laws
def profile_for(stage: str) -> Mapping[str, object]:
    assert_stage(stage)
    return STAGE_PROFILES[stage]


def profile_difference(
    a: Mapping[str, object] = P1_PROFILE, b: Mapping[str, object] = P2_PROFILE
) -> dict[str, tuple[object, object]]:
    """Every field in which two profiles differ."""
    keys = set(a) | set(b)
    return {k: (a.get(k), b.get(k)) for k in sorted(keys) if a.get(k) != b.get(k)}


def assert_declared_profile_difference() -> dict:
    """P2 may differ from P1 only in the declared model-profile fields."""
    difference = profile_difference()
    undeclared = sorted(set(difference) - DECLARED_PROFILE_DIFFERENCE_FIELDS)
    if undeclared:
        raise ValueError(
            f"P2 differs from P1 in undeclared fields {undeclared}; the only "
            "permitted change is the documented v5 deep model profile"
        )
    return difference


def lightgbm_params(profile: Mapping[str, object], model_seed: int) -> dict:
    """Frozen profile plus one model seed -> the exact LightGBM parameter dict.

    Only the seed varies between matched fits. There is no early-stopping and
    no evaluation-set parameter anywhere in this dict or its call sites.
    """
    if model_seed not in ALL_SEEDS:
        raise StageAuthorityError(f"model seed {model_seed} is not a KP35 seed")
    return {
        "objective": profile["objective"],
        "boosting": "gbdt",
        "learning_rate": profile["learning_rate"],
        "max_depth": profile["max_depth"],
        "num_leaves": profile["num_leaves"],
        "min_data_in_leaf": profile["min_data_in_leaf"],
        "feature_fraction": profile["feature_fraction"],
        "seed": model_seed,
        "deterministic": profile["deterministic"],
        "force_row_wise": profile["force_row_wise"],
        "num_threads": profile["num_threads"],
        "verbosity": -1,
    }


def params_sha256(stage: str, model_seed: int) -> str:
    """The canonical parameter digest a fit log must reproduce."""
    profile = profile_for(stage)
    params = lightgbm_params(profile, model_seed)
    return canonical_json_sha256({**params, "num_trees": profile["num_trees"]})


# ------------------------------------------------- stage ordering/eligibility
def assert_stage(stage: str) -> str:
    if stage in FORBIDDEN_STAGE_NAMES or FORBIDDEN_STAGE_SUBSTRING in stage.lower():
        raise StageAuthorityError(
            f"{stage!r} is not a KP35 parity stage. The parity ladder contains "
            f"{STAGES} only; Candidate-V and every later branch require a "
            "separately reviewed gate."
        )
    if stage not in STAGES:
        raise StageAuthorityError(f"unknown KP35 stage {stage!r}; expected one of {STAGES}")
    return stage


def assert_mode(mode: str) -> str:
    if mode not in MODES:
        raise StageAuthorityError(f"unknown KP35 mode {mode!r}; expected one of {MODES}")
    return mode


def stage_index(stage: str) -> int:
    return STAGES.index(assert_stage(stage))


def assert_stage_seed(stage: str, model_seed: int, *, screening: bool = True) -> None:
    """Seed 42 is the only screening seed; 1337/2024 are the only extra seeds."""
    assert_stage(stage)
    if screening:
        if model_seed != SCREENING_SEED:
            raise StageAuthorityError(
                f"stage {stage} screens with seed {SCREENING_SEED} only, got {model_seed}"
            )
    elif model_seed not in CONFIRMATION_SEEDS:
        raise StageAuthorityError(
            f"confirmation runs seeds {CONFIRMATION_SEEDS} only, got {model_seed}"
        )


def assert_stage_executable(stage: str, prior_state: str | None) -> None:
    """Stage eligibility. P2 requires a recorded P1 failure -- screen or confirmation.

    Nothing here starts a fit; this is the refusal law that a runner consults.
    There is no automatic chaining anywhere in the packet: a human must invoke
    each stage explicitly after independent review of the prior artifact.
    """
    assert_stage(stage)
    if stage == P1:
        if prior_state not in (None, KP35_SOURCE_FROZEN):
            raise StageAuthorityError(
                f"P1 is the first stage; it cannot run after state {prior_state!r}"
            )
        return
    if prior_state not in P2_AUTHORIZING_STATES:
        raise StageAuthorityError(
            f"{P2} becomes executable only when a valid P1 artifact records one of "
            f"{sorted(P2_AUTHORIZING_STATES)}; got {prior_state!r}. A P1 screen "
            "pass goes to P1 confirmation first, never straight to P2."
        )


def assert_confirmation_authorized(stage: str, prior_state: str | None) -> None:
    """Confirmation of a stage requires that stage's recorded screen pass."""
    assert_stage(stage)
    expected = STAGE_STATES[stage][0]
    if prior_state != expected:
        raise StageAuthorityError(
            f"confirmation of {stage} requires a recorded {expected}, "
            f"got {prior_state!r}"
        )


def assert_forward_transition(from_state: str | None, to_state: str) -> None:
    """Terminal states are absorbing; no transition may run backward."""
    origin = KP35_SOURCE_FROZEN if from_state is None else from_state
    if origin in ABSORBING_STATES:
        raise StageAuthorityError(
            f"{origin!r} is a terminal KP35 state; no further transition "
            f"(attempted {to_state!r}) is authorised by this gate"
        )
    allowed = FORWARD_TRANSITIONS.get(origin)
    if allowed is None:
        raise StageAuthorityError(f"unknown KP35 state {origin!r}")
    if to_state not in allowed:
        raise StageAuthorityError(
            f"transition {origin!r} -> {to_state!r} is not a forward KP35 "
            f"transition; allowed: {sorted(allowed)}"
        )


# --------------------------------------------------- strict prior authority (C)
@dataclass(frozen=True)
class PriorRequirement:
    """One acceptable predecessor envelope for a (stage, mode) invocation."""

    relpath: str
    record: str
    mode: str
    stage: str
    terminal_state: str


#: The complete map from an intended (stage, mode) invocation to the exact set
#: of prior result envelopes that may authorise it. A prior result is accepted
#: only at its canonical path under the same ``--out-root``; an arbitrary JSON
#: file elsewhere containing a plausible ``terminal_state`` authorises nothing.
PRIOR_AUTHORITY: Mapping[tuple[str, str], tuple[PriorRequirement, ...]] = {
    (P1, MODE_SCREEN): (),
    (P1, MODE_CONFIRMATION): (
        PriorRequirement(
            relpath=f"results/{P1}_screen.json",
            record=RECORD_SCREEN,
            mode=MODE_SCREEN,
            stage=P1,
            terminal_state=KP35_P1_SCREEN_PASSED,
        ),
    ),
    (P2, MODE_SCREEN): (
        PriorRequirement(
            relpath=f"results/{P1}_screen.json",
            record=RECORD_SCREEN,
            mode=MODE_SCREEN,
            stage=P1,
            terminal_state=KP35_P1_SCREEN_FAILED,
        ),
        PriorRequirement(
            relpath=f"results/{P1}_confirmation.json",
            record=RECORD_CONFIRMATION,
            mode=MODE_CONFIRMATION,
            stage=P1,
            terminal_state=KP35_P1_CONFIRMATION_FAILED,
        ),
    ),
    (P2, MODE_CONFIRMATION): (
        PriorRequirement(
            relpath=f"results/{P2}_screen.json",
            record=RECORD_SCREEN,
            mode=MODE_SCREEN,
            stage=P2,
            terminal_state=KP35_P2_SCREEN_PASSED,
        ),
    ),
}

#: Fields a KP35 result envelope must carry for a successor to authenticate it.
RESULT_ENVELOPE_REQUIRED_FIELDS: tuple[str, ...] = (
    "record",
    "mode",
    "stage",
    "terminal_state",
    "prior_state",
    "protocol_semantic_sha256",
    "benchmark",
    "scoring_universe",
    "sample_custody",
    "fit_provenance",
    "authorizes_next_fit",
)


def canonical_prior_relpaths(stage: str, mode: str) -> tuple[str, ...]:
    """The canonical predecessor locations for one (stage, mode) invocation."""
    assert_stage(stage)
    assert_mode(mode)
    return tuple(r.relpath for r in PRIOR_AUTHORITY[(stage, mode)])


def validate_result_envelope(envelope: Mapping, *, context: str) -> None:
    """Structural completeness of any KP35 result envelope."""
    if not isinstance(envelope, Mapping):
        raise PriorResultAuthorityError(f"{context}: result envelope is not an object")
    missing = [f for f in RESULT_ENVELOPE_REQUIRED_FIELDS if f not in envelope]
    if missing:
        raise PriorResultAuthorityError(
            f"{context}: result envelope missing required fields {missing}; a bare "
            "terminal_state is not a KP35 result and authorises nothing"
        )


def validate_prior_result(
    *,
    stage: str,
    mode: str,
    prior_relpath: str | None,
    prior_envelope: Mapping | None,
    protocol_semantic: str,
    sample_identity_sha256: str | None = None,
    sample_canon_sha256: str | None = None,
) -> str | None:
    """Authenticate the artifact claiming to authorise this invocation.

    Phase 1 (before any model feature is loaded) validates everything that does
    not require the sample manifest. Phase 2 re-invokes with
    ``sample_identity_sha256``/``sample_canon_sha256`` once the frozen manifest
    is in hand, proving the predecessor was produced against the identical
    sample.

    Returns the validated predecessor terminal state, or ``None`` when the
    invocation legitimately has no predecessor (the P1 screen).

    A valid artifact is a necessary condition, never a sufficient one: a human
    execution authorisation is separately required and nothing here starts a fit.
    """
    assert_stage(stage)
    assert_mode(mode)
    requirements = PRIOR_AUTHORITY[(stage, mode)]

    if not requirements:
        if prior_envelope is not None or prior_relpath is not None:
            raise PriorResultAuthorityError(
                f"{stage} {mode} is the first invocation and accepts no prior result"
            )
        return None

    if prior_envelope is None or prior_relpath is None:
        raise PriorResultAuthorityError(
            f"{stage} {mode} requires a prior result at one of "
            f"{[r.relpath for r in requirements]} under the same --out-root"
        )

    normalized = normalize_relpath(prior_relpath)
    matched = [r for r in requirements if normalize_relpath(r.relpath) == normalized]
    if not matched:
        raise PriorResultAuthorityError(
            f"prior result at {normalized!r} is not a canonical predecessor for "
            f"{stage} {mode}; expected one of "
            f"{[normalize_relpath(r.relpath) for r in requirements]}. A result at "
            "an arbitrary location authorises nothing regardless of its contents."
        )

    validate_result_envelope(prior_envelope, context=f"prior result {normalized}")
    requirement = matched[0]

    for field_name, expected in (
        ("record", requirement.record),
        ("mode", requirement.mode),
        ("stage", requirement.stage),
        ("terminal_state", requirement.terminal_state),
    ):
        actual = prior_envelope.get(field_name)
        if actual != expected:
            raise PriorResultAuthorityError(
                f"prior result {normalized}: {field_name} {actual!r} != required "
                f"{expected!r} for authorising {stage} {mode}"
            )

    # The predecessor's own recorded transition must itself be legal.
    try:
        assert_forward_transition(
            prior_envelope.get("prior_state"), requirement.terminal_state
        )
    except StageAuthorityError as exc:
        raise PriorResultAuthorityError(
            f"prior result {normalized}: its own recorded transition "
            f"{prior_envelope.get('prior_state')!r} -> {requirement.terminal_state!r} "
            f"is not legal, so the envelope cannot be authentic: {exc}"
        ) from exc

    if prior_envelope.get("protocol_semantic_sha256") != protocol_semantic:
        raise PriorResultAuthorityError(
            f"prior result {normalized}: protocol identity "
            f"{prior_envelope.get('protocol_semantic_sha256')!r} != current "
            f"{protocol_semantic!r}; the predecessor was produced under a "
            "different protocol"
        )

    benchmark = prior_envelope.get("benchmark") or {}
    if benchmark.get("frozen_kw33_mean_corr") != BENCHMARK_MEAN_CORR:
        raise PriorResultAuthorityError(
            f"prior result {normalized}: frozen benchmark value mismatch"
        )
    if benchmark.get("tolerance") != BENCHMARK_MEAN_CORR_TOLERANCE:
        raise PriorResultAuthorityError(
            f"prior result {normalized}: benchmark tolerance mismatch"
        )
    try:
        assert_benchmark_identity(benchmark.get("recomputed_mean_corr"))
    except ValueError as exc:
        # A predecessor carrying a drifted or non-finite benchmark is an invalid
        # envelope, so it is refused with the uniform prior-authority error.
        raise PriorResultAuthorityError(
            f"prior result {normalized}: recorded benchmark mean CORR is not the "
            f"frozen identity: {exc}"
        ) from exc

    universe = prior_envelope.get("scoring_universe") or {}
    if universe.get("rows") != SCORING_UNIVERSE_ROWS:
        raise PriorResultAuthorityError(
            f"prior result {normalized}: scoring universe row count mismatch"
        )
    if universe.get("canon_sha256") != SCORING_UNIVERSE_CANON_SHA256:
        raise PriorResultAuthorityError(
            f"prior result {normalized}: scoring universe canonical hash mismatch"
        )

    custody = prior_envelope.get("sample_custody") or {}
    if sample_identity_sha256 is not None:
        if custody.get("sample_identity_sha256") != sample_identity_sha256:
            raise PriorResultAuthorityError(
                f"prior result {normalized}: sample identity "
                f"{custody.get('sample_identity_sha256')!r} != current "
                f"{sample_identity_sha256!r}"
            )
    if sample_canon_sha256 is not None:
        if custody.get("sample_canon_sha256") != sample_canon_sha256:
            raise PriorResultAuthorityError(
                f"prior result {normalized}: sample canonical hash mismatch"
            )
    if custody.get("sample_canon_sha256") != FROZEN_SAMPLE["sample_canon_sha256"]:
        raise PriorResultAuthorityError(
            f"prior result {normalized}: sample canonical hash is not the frozen sample"
        )

    provenance = prior_envelope.get("fit_provenance") or {}
    expected_seeds = (
        [SCREENING_SEED]
        if requirement.mode == MODE_SCREEN
        else [SCREENING_SEED, *CONFIRMATION_SEEDS]
    )
    for seed in expected_seeds:
        entry = provenance.get(str(seed))
        if not entry:
            raise PriorResultAuthorityError(
                f"prior result {normalized}: missing fit provenance for seed {seed}"
            )
        if entry.get("stage") != requirement.stage:
            raise PriorResultAuthorityError(
                f"prior result {normalized}: seed {seed} provenance names stage "
                f"{entry.get('stage')!r}, expected {requirement.stage!r}"
            )
        if entry.get("model_seed") != seed:
            raise PriorResultAuthorityError(
                f"prior result {normalized}: seed {seed} provenance seed mismatch"
            )
        for required in ("params_sha256", "prediction_sha256", "prediction_canon_sha256"):
            if not entry.get(required):
                raise PriorResultAuthorityError(
                    f"prior result {normalized}: seed {seed} provenance missing "
                    f"{required}"
                )
        if entry.get("params_sha256") != params_sha256(requirement.stage, seed):
            raise PriorResultAuthorityError(
                f"prior result {normalized}: seed {seed} parameter digest does not "
                "match the frozen stage recipe"
            )
        if entry.get("prediction_canon_sha256") != SCORING_UNIVERSE_CANON_SHA256:
            raise PriorResultAuthorityError(
                f"prior result {normalized}: seed {seed} prediction universe hash "
                "is not the frozen scoring universe"
            )
    unexpected = sorted(set(provenance) - {str(s) for s in expected_seeds})
    if unexpected:
        raise PriorResultAuthorityError(
            f"prior result {normalized}: unexpected fit provenance seeds {unexpected}"
        )

    if prior_envelope.get("authorizes_next_fit") is not False:
        raise PriorResultAuthorityError(
            f"prior result {normalized}: a KP35 result never self-authorises a "
            "successor fit; authorizes_next_fit must be false"
        )

    return requirement.terminal_state


# ------------------------------------------------- fit-log provenance law (D)
FIT_LOG_REQUIRED_FIELDS: tuple[str, ...] = (
    "record",
    "stage",
    "model_seed",
    "role",
    "attempt",
    "payout_target",
    "feature_set",
    "n_features",
    "feature_list_sha256",
    "profile_name",
    "num_trees",
    "params",
    "params_sha256",
    "protocol_semantic_sha256",
    "data_identities",
    "sample_identity_sha256",
    "sample_canon_sha256",
    "rows_before_sampling",
    "rows_after_sampling",
    "source_split_rows",
    "selected_rows_per_era",
    "eligible_era_range",
    "n_eligible_eras",
    "scored_eras",
    "no_early_stopping",
    "no_evaluation_set",
    "model_artifact_written",
    "exit_status",
    "prediction_rows",
    "prediction_sha256",
    "prediction_canon_sha256",
)


def expected_role(model_seed: int) -> str:
    return "screening" if model_seed == SCREENING_SEED else "confirmation"


def validate_fit_log(
    log: Mapping,
    *,
    stage: str,
    model_seed: int,
    protocol_semantic: str,
    data_identities: Mapping[str, str],
    sample_manifest: Mapping,
    actual_prediction_sha256: str | None = None,
    actual_prediction_canon_sha256: str | None = None,
    actual_prediction_rows: int | None = None,
) -> dict:
    """Independently reconstruct and validate one fit envelope.

    Nothing is trusted because it appears in the log: the profile, the exact
    LightGBM parameter dictionary and its digest are recomputed from the frozen
    stage recipe, the sample fields are compared against the separately loaded
    external manifest, and the prediction digests are compared against the
    artifact actually on disk when the caller supplies them.

    This runs for a one-seed screen too, where there is no second seed to
    compare against, so the frozen recipe is the only available reference.
    """
    assert_stage(stage)
    if model_seed not in ALL_SEEDS:
        raise StageAuthorityError(f"model seed {model_seed} is not a KP35 seed")

    missing = [f for f in FIT_LOG_REQUIRED_FIELDS if f not in log]
    if missing:
        raise FitProvenanceError(
            f"{stage} seed {model_seed}: fit log missing required fields {missing}"
        )

    profile = profile_for(stage)
    expected_params = lightgbm_params(profile, model_seed)

    checks: list[tuple[str, object, object]] = [
        ("record", log.get("record"), RECORD_FIT_LOG),
        ("stage", log.get("stage"), stage),
        ("model_seed", log.get("model_seed"), model_seed),
        ("role", log.get("role"), expected_role(model_seed)),
        ("payout_target", log.get("payout_target"), PAYOUT_TARGET),
        ("feature_set", log.get("feature_set"), FEATURE_SET),
        ("n_features", log.get("n_features"), N_FEATURES),
        ("feature_list_sha256", log.get("feature_list_sha256"), FEATURE_LIST_SHA256),
        ("profile_name", log.get("profile_name"), profile["name"]),
        ("num_trees", log.get("num_trees"), profile["num_trees"]),
        ("params", log.get("params"), expected_params),
        ("params_sha256", log.get("params_sha256"), params_sha256(stage, model_seed)),
        ("protocol_semantic_sha256", log.get("protocol_semantic_sha256"), protocol_semantic),
        ("no_early_stopping", log.get("no_early_stopping"), True),
        ("no_evaluation_set", log.get("no_evaluation_set"), True),
        ("model_artifact_written", log.get("model_artifact_written"), False),
        ("exit_status", log.get("exit_status"), "success"),
        (
            "sample_identity_sha256",
            log.get("sample_identity_sha256"),
            sample_manifest["sample_identity_sha256"],
        ),
        (
            "sample_canon_sha256",
            log.get("sample_canon_sha256"),
            sample_manifest["sample_canon_sha256"],
        ),
        (
            "rows_before_sampling",
            log.get("rows_before_sampling"),
            sample_manifest["rows_before_sampling"],
        ),
        (
            "rows_after_sampling",
            log.get("rows_after_sampling"),
            sample_manifest["selected_row_count"],
        ),
        (
            "source_split_rows",
            log.get("source_split_rows"),
            sample_manifest["source_split_rows"],
        ),
        (
            "selected_rows_per_era",
            log.get("selected_rows_per_era"),
            sample_manifest["selected_rows_per_era"],
        ),
        (
            "eligible_era_range",
            list(log.get("eligible_era_range") or []),
            list(sample_manifest["eligible_era_range"]),
        ),
        (
            "n_eligible_eras",
            log.get("n_eligible_eras"),
            sample_manifest["n_eligible_eras"],
        ),
        (
            "scored_eras",
            list(log.get("scored_eras") or []),
            [score_zone_eras()[0], score_zone_eras()[-1]],
        ),
        ("prediction_rows", log.get("prediction_rows"), SCORING_UNIVERSE_ROWS),
        (
            "prediction_canon_sha256",
            log.get("prediction_canon_sha256"),
            SCORING_UNIVERSE_CANON_SHA256,
        ),
    ]
    mismatches = [
        f"{name}: {actual!r} != expected {expected!r}"
        for name, actual, expected in checks
        if actual != expected
    ]
    if mismatches:
        raise FitProvenanceError(
            f"{stage} seed {model_seed}: fit log does not match the frozen recipe: "
            + "; ".join(mismatches)
        )

    logged_identities = dict(log.get("data_identities") or {})
    for name, digest in data_identities.items():
        if logged_identities.get(name) != digest:
            raise FitProvenanceError(
                f"{stage} seed {model_seed}: data identity for {name} "
                f"{logged_identities.get(name)!r} != revalidated {digest!r}"
            )

    if actual_prediction_sha256 is not None:
        if log.get("prediction_sha256") != actual_prediction_sha256:
            raise FitProvenanceError(
                f"{stage} seed {model_seed}: prediction file digest "
                f"{actual_prediction_sha256!r} != logged "
                f"{log.get('prediction_sha256')!r}"
            )
    if actual_prediction_canon_sha256 is not None:
        if actual_prediction_canon_sha256 != SCORING_UNIVERSE_CANON_SHA256:
            raise FitProvenanceError(
                f"{stage} seed {model_seed}: prediction universe is not the frozen "
                "scoring universe"
            )
    if actual_prediction_rows is not None and actual_prediction_rows != SCORING_UNIVERSE_ROWS:
        raise FitProvenanceError(
            f"{stage} seed {model_seed}: prediction has {actual_prediction_rows} rows "
            f"!= {SCORING_UNIVERSE_ROWS}"
        )

    return {
        "stage": stage,
        "model_seed": model_seed,
        "role": expected_role(model_seed),
        "attempt": log.get("attempt"),
        "params_sha256": log.get("params_sha256"),
        "prediction_sha256": log.get("prediction_sha256"),
        "prediction_canon_sha256": log.get("prediction_canon_sha256"),
        "sample_identity_sha256": log.get("sample_identity_sha256"),
        "sample_canon_sha256": log.get("sample_canon_sha256"),
        "validated": True,
    }


def assert_cohort_identical_except_seed(logs: Mapping[int, Mapping]) -> dict:
    """Every fit in a cohort shares one sample identity and one parameter set."""
    if not logs:
        raise FitProvenanceError("empty cohort")
    identities = {log["sample_identity_sha256"] for log in logs.values()}
    if len(identities) != 1:
        raise FitProvenanceError(
            f"cohort spans multiple sample identities: {sorted(identities)}"
        )
    canon = {log["sample_canon_sha256"] for log in logs.values()}
    if len(canon) != 1:
        raise FitProvenanceError(
            "cohort fits were trained on different sampled (era,id) universes"
        )
    stages = {log["stage"] for log in logs.values()}
    if len(stages) != 1:
        raise FitProvenanceError(f"cohort spans multiple stages: {sorted(stages)}")
    stripped = set()
    for log in logs.values():
        params = dict(log["params"])
        params.pop("seed", None)
        stripped.add(canonical_json_sha256({**params, "num_trees": log["num_trees"]}))
    if len(stripped) != 1:
        raise FitProvenanceError(
            "cohort fits differ in a model parameter other than the seed"
        )
    return {
        "stage": next(iter(stages)),
        "sample_identity_sha256": next(iter(identities)),
        "sample_canon_sha256": next(iter(canon)),
        "seeds": sorted(logs),
        "identical_except_seed": True,
    }


# ------------------------------------------------------------- screening law
def assert_benchmark_identity(
    recomputed_mean_corr: object,
    frozen: float = BENCHMARK_MEAN_CORR,
    tolerance: float = BENCHMARK_MEAN_CORR_TOLERANCE,
) -> float:
    """The benchmark mean CORR must reproduce the frozen KW33 value exactly.

    A drifting benchmark would silently move every threshold, so identity is
    required within a strict declared numerical tolerance rather than assumed.
    A non-finite recomputation is a computation fault, raised before any state
    can be produced.
    """
    value = assert_finite_scalar(recomputed_mean_corr, name="recomputed benchmark mean CORR")
    delta = abs(value - frozen)
    if delta > tolerance:
        raise ValueError(
            f"recomputed benchmark mean CORR {value!r} differs "
            f"from the frozen KW33 value {frozen!r} by {delta!r} > {tolerance!r}"
        )
    return value


def screen_threshold(benchmark_mean_corr: float = BENCHMARK_MEAN_CORR) -> float:
    """``SCREEN_FACTOR * benchmark_mean_corr``."""
    return SCREEN_FACTOR * assert_finite_scalar(
        benchmark_mean_corr, name="benchmark mean CORR"
    )


def final_threshold(benchmark_mean_corr: float = BENCHMARK_MEAN_CORR) -> float:
    """``FINAL_PARITY_FRACTION * benchmark_mean_corr``."""
    return FINAL_PARITY_FRACTION * assert_finite_scalar(
        benchmark_mean_corr, name="benchmark mean CORR"
    )


def screen_stage(
    stage: str,
    seed42_corr: object,
    benchmark_mean_corr: float = BENCHMARK_MEAN_CORR,
) -> dict:
    """The one-seed screen. CORR only; nothing else has an input path.

    A pass means only that the *stage's own confirmation* is authorisable. It
    is not final parity, not model promotion, not recency promotion, and not
    deployment authority. A P1 screen failure means only that P2 becomes
    authorisable. This function never starts the next stage.
    """
    assert_stage(stage)
    corr = assert_finite_scalar(seed42_corr, name=f"{stage} seed-42 screen CORR")
    benchmark = assert_finite_scalar(benchmark_mean_corr, name="benchmark mean CORR")
    threshold = screen_threshold(benchmark)
    passed = corr >= threshold
    state = STAGE_STATES[stage][0] if passed else STAGE_STATES[stage][1]
    return {
        "stage": stage,
        "screening_seed": SCREENING_SEED,
        "seed42_corr": corr,
        "benchmark_mean_corr": benchmark,
        "screen_factor": SCREEN_FACTOR,
        "threshold": threshold,
        "law": "seed42_corr >= SCREEN_FACTOR * benchmark_mean_corr",
        "passed": passed,
        "terminal_state": state,
        "means": (
            f"{stage} confirmation is authorisable; this is NOT final parity, "
            "model promotion, recency promotion, or deployment authority"
            if passed
            else (
                "P2 screening may be separately authorised"
                if stage == P1
                else "both proven mismatches failed to restore parity"
            )
        ),
        "next_stage_started": False,
        "selection_input": "CORR only",
    }


def final_confirmation(
    stage: str,
    corr_42: object,
    corr_1337: object,
    corr_2024: object,
    benchmark_mean_corr: float = BENCHMARK_MEAN_CORR,
) -> dict:
    """The two-part final parity law. Both conditions are independently required.

    1. three-seed mean CORR >= 0.70 * benchmark mean CORR
    2. untouched confirmation-pair mean CORR >= 0.70 * benchmark mean CORR

    The pair gate exists because the seed-42 fit is the one that was selected
    *on*; requiring the two seeds that had no part in the selection to clear
    the same bar on their own is what stops a lucky screening draw from
    carrying a recipe through confirmation.

    The failure state is stage-specific. A P1 confirmation failure leaves the
    documented deep profile untested and therefore authorises P2 screening; a
    P2 confirmation failure ends the ladder.
    """
    assert_stage(stage)
    c42 = assert_finite_scalar(corr_42, name=f"{stage} seed-42 confirmation CORR")
    c1337 = assert_finite_scalar(corr_1337, name=f"{stage} seed-1337 confirmation CORR")
    c2024 = assert_finite_scalar(corr_2024, name=f"{stage} seed-2024 confirmation CORR")
    benchmark = assert_finite_scalar(benchmark_mean_corr, name="benchmark mean CORR")
    threshold = final_threshold(benchmark)

    three_seed_mean = assert_finite_scalar(
        (c42 + c1337 + c2024) / 3.0, name=f"{stage} three-seed mean CORR"
    )
    pair_mean = assert_finite_scalar(
        (c1337 + c2024) / 2.0, name=f"{stage} untouched-pair mean CORR"
    )
    three_seed_pass = three_seed_mean >= threshold
    pair_pass = pair_mean >= threshold
    confirmed = three_seed_pass and pair_pass
    passed_state, failed_state = CONFIRMATION_STATES[stage]
    return {
        "stage": stage,
        "seeds": {"42": c42, "1337": c1337, "2024": c2024},
        "benchmark_mean_corr": benchmark,
        "final_parity_fraction": FINAL_PARITY_FRACTION,
        "threshold": threshold,
        "three_seed_mean": three_seed_mean,
        "three_seed_gate_passed": three_seed_pass,
        "untouched_pair_mean": pair_mean,
        "untouched_pair_gate_passed": pair_pass,
        "both_required": True,
        "confirmed": confirmed,
        "terminal_state": passed_state if confirmed else failed_state,
        "means": (
            "parity backbone confirmed; this grants no promotion and no "
            "deployment authority"
            if confirmed
            else (
                "P2 screening may be separately authorised; the documented deep "
                "profile remains untested"
                if stage == P1
                else "the ladder ends; neither proven mismatch restored parity"
            )
        ),
        "selection_input": "CORR only",
        "non_selecting": sorted(NON_SELECTING_DIAGNOSTICS),
        "next_stage_started": False,
        "promotion_granted": False,
        CANDIDATE_V_RETURN_DENIAL_KEY: False,
    }


# --------------------------------------------- one-shot artifact + attempt law
ARTIFACT_KINDS: tuple[str, ...] = (
    "prediction",
    "fit_log",
    "failure_record",
    "screen_result",
    "confirmation_result",
    "sample_identity",
    "final_report",
)

#: Normal invocation is attempt 1; ``--retry`` is attempt 2; there is no
#: attempt 3. Each attempt owns a distinct failure path so a second failure can
#: never be masked by the first failure file already existing.
FIRST_ATTEMPT = 1
RETRY_ATTEMPT = 2
MAX_ATTEMPTS = RETRY_ATTEMPT


def artifact_relpath(
    kind: str,
    *,
    stage: str | None = None,
    model_seed: int | None = None,
    attempt: int | None = None,
) -> str:
    """The unique create-new-only relative path for one future artifact.

    Every scientific artifact path is a pure function of its kind, stage, seed
    and (for failure records) attempt, so two different fits can never collide
    on one path and one fit can never be written twice under two names.
    """
    if kind not in ARTIFACT_KINDS:
        raise ValueError(f"unknown artifact kind {kind!r}; expected one of {ARTIFACT_KINDS}")
    if kind in ("prediction", "fit_log", "failure_record"):
        assert_stage(stage or "")
        if model_seed not in ALL_SEEDS:
            raise StageAuthorityError(f"model seed {model_seed} is not a KP35 seed")
        if kind == "failure_record":
            if attempt not in (FIRST_ATTEMPT, RETRY_ATTEMPT):
                raise StageAuthorityError(
                    f"failure records exist for attempts "
                    f"{(FIRST_ATTEMPT, RETRY_ATTEMPT)} only, got {attempt!r}"
                )
            return f"failures/{stage}_seed{model_seed}_attempt{attempt}.json"
        suffix = "parquet" if kind == "prediction" else "json"
        folder = {"prediction": "predictions", "fit_log": "logs"}[kind]
        return f"{folder}/{stage}_seed{model_seed}.{suffix}"
    if kind == "screen_result":
        assert_stage(stage or "")
        return f"results/{stage}_screen.json"
    if kind == "confirmation_result":
        assert_stage(stage or "")
        return f"results/{stage}_confirmation.json"
    if kind == "sample_identity":
        return "sample/kp35_sample_identity.json"
    return "results/kp35_final_report.md"


def assert_create_new_only(exists: bool, path: str, *, kind: str) -> None:
    """A final artifact path is never replaced, only created."""
    if exists:
        raise FileExistsError(
            f"refusing to overwrite an existing {kind} artifact at {path}. "
            "KP35 is create-new-only: completed predictions are never "
            "overwritten, completed fits are never rerun, and failure records "
            "are preserved. Atomic temp-to-final replacement is permitted for "
            "first-time creation only."
        )


def resolve_attempt(*, retry_requested: bool) -> int:
    """Normal invocation is attempt 1; ``--retry`` is attempt 2."""
    return RETRY_ATTEMPT if retry_requested else FIRST_ATTEMPT


def assert_attempt_authorized(
    *,
    stage: str,
    model_seed: int,
    attempt: int,
    prediction_exists: bool,
    success_log_exists: bool,
    attempt1_failure_exists: bool,
    attempt2_failure_exists: bool,
) -> dict:
    """The complete attempt-posture law, derived from artifacts on disk.

    No caller supplies a constant retry count: the posture is read from which
    artifacts exist. A retry is authorised only when a first failure was
    actually preserved, and a third attempt does not exist.
    """
    assert_stage(stage)
    if model_seed not in ALL_SEEDS:
        raise StageAuthorityError(f"model seed {model_seed} is not a KP35 seed")
    if attempt not in (FIRST_ATTEMPT, RETRY_ATTEMPT):
        raise StageAuthorityError(
            f"{stage} seed {model_seed}: attempt {attempt!r} does not exist; "
            f"KP35 permits attempts {(FIRST_ATTEMPT, RETRY_ATTEMPT)} only"
        )
    prefix = f"{stage} seed {model_seed}"
    if prediction_exists:
        raise StageAuthorityError(
            f"{prefix}: a valid prediction artifact already exists; a completed "
            "scientific fit is never rerun"
        )
    if success_log_exists:
        raise StageAuthorityError(
            f"{prefix}: a success fit log already exists; a completed scientific "
            "fit is never rerun"
        )
    if attempt2_failure_exists:
        raise StageAuthorityError(
            f"{prefix}: an attempt-2 failure record is already preserved; no "
            "further attempt is authorised"
        )
    if attempt == FIRST_ATTEMPT:
        if attempt1_failure_exists:
            raise StageAuthorityError(
                f"{prefix}: an attempt-1 failure record already exists; the only "
                "authorised continuation is a single explicit retry (attempt 2)"
            )
    else:
        if not attempt1_failure_exists:
            raise StageAuthorityError(
                f"{prefix}: a retry may not be requested without a preserved "
                "attempt-1 failure record"
            )
    return {
        "stage": stage,
        "model_seed": model_seed,
        "attempt": attempt,
        "authorized": True,
        "max_attempts": MAX_ATTEMPTS,
    }


def validate_attempt1_failure_record(
    record: Mapping,
    *,
    stage: str,
    model_seed: int,
    protocol_semantic: str,
) -> None:
    """A retry must inherit an unchanged first-failure envelope."""
    assert_stage(stage)
    checks = [
        ("stage", record.get("stage"), stage),
        ("model_seed", record.get("model_seed"), model_seed),
        ("attempt", record.get("attempt"), FIRST_ATTEMPT),
        ("protocol_semantic_sha256", record.get("protocol_semantic_sha256"), protocol_semantic),
        ("params_sha256", record.get("params_sha256"), params_sha256(stage, model_seed)),
        ("payout_target", record.get("payout_target"), PAYOUT_TARGET),
        ("feature_list_sha256", record.get("feature_list_sha256"), FEATURE_LIST_SHA256),
    ]
    mismatches = [
        f"{name}: {actual!r} != expected {expected!r}"
        for name, actual, expected in checks
        if actual != expected
    ]
    if mismatches:
        raise StageAuthorityError(
            f"{stage} seed {model_seed}: the preserved attempt-1 failure record does "
            "not match this invocation, so a retry would not be the same fit: "
            + "; ".join(mismatches)
        )
    if str(record.get("exit_status", "")).startswith("success"):
        raise StageAuthorityError(
            f"{stage} seed {model_seed}: the attempt-1 record is not a failure"
        )


# --------------------------------------------- authority + environment binding
#: Public authority observed at source freeze. This is a dated snapshot, not a
#: substitute for live revalidation: the execution assignment must independently
#: re-query current public authority immediately before P1.
AUTHORITY_SNAPSHOT: Mapping[str, object] = {
    "retrieved_utc": "2026-08-19T03:10:55+00:00",
    "round_number": 1335,
    "payout_target": PAYOUT_TARGET,
    "corr_config": {
        "name": "correlation",
        "display_name": "v2_corr20",
        "version": "6",
        "multiplier": 0.75,
        "is_payout": True,
    },
    "mmc_config": {
        "name": "meta_model_contribution",
        "display_name": "mmc",
        "version": "5",
        "multiplier": 2.25,
        "is_payout": True,
    },
    "ender60_payout_active": False,
    "ender60_note": (
        "corr60 v3 and mmc60 v3 are isPayout false with multiplier 0.0; no "
        "Ender60 payout cutover has occurred"
    ),
    "dataset_version": "v5.3",
    "query": (
        "query { rounds(tournament: 8, number: 1335) { number openTime closeTime "
        "target v3Staking payoutFactor roundScoreConfigs { name displayName "
        "version isPayout defaultMultiplier minMultiplier maxMultiplier "
        "totalScoreDays returnsLagDays roundNumberStart } } }"
    ),
    "docs_commit": "5bf294adbac78d0cde497a7d1589694ee9951169",
    "live_revalidation_required_before_p1": True,
}

CORR_MULTIPLIER = 0.75
MMC_MULTIPLIER = 2.25


def assert_score_authority(
    *,
    payout_target: str,
    corr_multiplier: object,
    mmc_multiplier: object,
    meta_model_column: str,
) -> dict:
    """Bind the evaluator's local ScoreAuthority to the frozen snapshot.

    Pure and dataset-free: the caller passes the fields of whatever authority
    it loaded. No network client is added to this packet -- live public
    revalidation is a separate, mandatory step of the execution assignment.
    """
    assert_payout_target(payout_target)
    corr = assert_finite_scalar(corr_multiplier, name="CORR multiplier")
    mmc = assert_finite_scalar(mmc_multiplier, name="MMC multiplier")
    if corr != CORR_MULTIPLIER:
        raise EnvironmentBindingError(
            f"CORR multiplier {corr!r} != frozen {CORR_MULTIPLIER!r}"
        )
    if mmc != MMC_MULTIPLIER:
        raise EnvironmentBindingError(
            f"MMC multiplier {mmc!r} != frozen {MMC_MULTIPLIER!r}"
        )
    if meta_model_column != META_MODEL_COLUMN:
        raise EnvironmentBindingError(
            f"Meta Model column {meta_model_column!r} != frozen {META_MODEL_COLUMN!r}"
        )
    return {
        "payout_target": payout_target,
        "corr_multiplier": corr,
        "mmc_multiplier": mmc,
        "meta_model_column": meta_model_column,
        "matches_frozen_snapshot": True,
        "live_revalidation_still_required_before_p1": True,
    }


#: The exact runtime the protocol freezes. A version mismatch stops before
#: training rather than producing a result under an unfrozen toolchain.
FROZEN_ENVIRONMENT: Mapping[str, str] = {
    "python": "3.13.14",
    "lightgbm": "4.7.0",
    "numpy": "2.5.1",
    "pandas": "3.0.5",
    "pyarrow": "25.0.1",
    "numerai_tools": "0.6.0",
    "psutil": "7.2.2",
}

#: The packages whose identity determines a produced score. The evaluator must
#: verify these even when it does not train.
SCORE_PRODUCING_PACKAGES: tuple[str, ...] = ("python", "numpy", "pandas", "numerai_tools")


def assert_runtime_versions(
    observed: Mapping[str, str], *, required: Sequence[str] | None = None
) -> dict:
    """Every required package must match the frozen version exactly."""
    names = tuple(required) if required is not None else tuple(FROZEN_ENVIRONMENT)
    missing = [n for n in names if n not in observed]
    if missing:
        raise EnvironmentBindingError(
            f"runtime environment does not report {missing}; refusing to proceed "
            "under an unverified toolchain"
        )
    mismatches = [
        f"{name}: observed {observed[name]!r} != frozen {FROZEN_ENVIRONMENT[name]!r}"
        for name in names
        if observed[name] != FROZEN_ENVIRONMENT.get(name)
    ]
    if mismatches:
        raise EnvironmentBindingError(
            "runtime environment differs from the frozen protocol: "
            + "; ".join(mismatches)
        )
    return {"verified": sorted(names), "matches_frozen_environment": True}


# ------------------------------------------------------------------- summary
@dataclass(frozen=True)
class FrozenDesign:
    """The complete frozen KP35 scientific design, for record emission."""

    question: str = (
        "Can either the documented benchmark information boundary or the "
        "documented v5 deep LightGBM profile restore a static Keystone CONTROL "
        "backbone to benchmark-plausible CORR on eras 1133-1219?"
    )
    scope: str = "parity calibration only"
    excludes: tuple[str, ...] = (
        "Candidate-V",
        "validation recency promotion",
        "MMC specialist models",
        "feature ensembles",
        "target ensembles",
        "blending",
        "deployment",
        "live performance",
    )
    stages: tuple[str, ...] = STAGES
    terminal_states: tuple[str, ...] = field(
        default=(
            KP35_P1_SCREEN_PASSED,
            KP35_P1_SCREEN_FAILED,
            KP35_P1_CONFIRMATION_FAILED,
            KP35_P2_SCREEN_PASSED,
            KP35_PARITY_NOT_RESTORED,
            KP35_P2_CONFIRMATION_FAILED,
            KP35_PARITY_CONFIRMED,
        )
    )
    ends_if_both_fail: str = (
        "If both P1 and P2 fail, the experiment ends. Feature-universe and "
        "row-budget expansion require a later, separately reviewed gate."
    )


def frozen_constants() -> dict:
    """The exact constants the protocol record must mirror."""
    return {
        "score_zone": [f"{SCORE_ZONE_START:04d}", f"{SCORE_ZONE_END:04d}"],
        "n_score_eras": N_SCORE_ERAS,
        "scoring_universe_rows": SCORING_UNIVERSE_ROWS,
        "scoring_universe_canon_sha256": SCORING_UNIVERSE_CANON_SHA256,
        "benchmark_chunk": [BENCHMARK_CHUNK_START, BENCHMARK_CHUNK_END],
        "purge": [PURGE_START, PURGE_END],
        "history_boundary_end": HISTORY_BOUNDARY_END,
        "gap": [GAP_START, GAP_END],
        "holdout_start": HOLDOUT_START,
        "payout_target": PAYOUT_TARGET,
        "feature_set": FEATURE_SET,
        "n_features": N_FEATURES,
        "feature_list_sha256": FEATURE_LIST_SHA256,
        "sampling_seed": SAMPLING_SEED,
        "sampling_law_version": SAMPLING_LAW_VERSION,
        "max_sampled_rows": MAX_SAMPLED_ROWS,
        "screening_seed": SCREENING_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "benchmark_mean_corr": BENCHMARK_MEAN_CORR,
        "benchmark_mean_corr_tolerance": BENCHMARK_MEAN_CORR_TOLERANCE,
        "screen_factor": SCREEN_FACTOR,
        "screen_threshold": SCREEN_THRESHOLD,
        "final_parity_fraction": FINAL_PARITY_FRACTION,
        "final_three_seed_threshold": FINAL_THREE_SEED_THRESHOLD,
        "untouched_pair_threshold": UNTOUCHED_PAIR_THRESHOLD,
        "stages": list(STAGES),
        "frozen_sample": dict(FROZEN_SAMPLE),
        "max_attempts": MAX_ATTEMPTS,
        "corr_multiplier": CORR_MULTIPLIER,
        "mmc_multiplier": MMC_MULTIPLIER,
        "meta_model_column": META_MODEL_COLUMN,
        "frozen_environment": dict(FROZEN_ENVIRONMENT),
    }
