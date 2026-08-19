"""Keystone Round-2 parity-calibration evaluator (KP35): one decision per call.

One invocation evaluates exactly one thing — a single stage screen, or a single
confirmation cohort — and writes exactly one result. It never starts a fit and
never authorises the next one; a screen result records what became *authorisable*,
and a human decides whether to act on it after independent review.

What is enforced before any number is computed:

* the completed artifacts named by the protocol must all exist, and every
  prediction parquet must reproduce the SHA-256 recorded in its fit log;
* the complete canonical ``(era, id)`` universe must be identical across the
  prediction vector, the scoring target frame, the Meta Model frame, the
  published Ender20 benchmark frame, and the published Ender60 benchmark frame
  whenever auxiliary diagnostics are loaded — this is the prospective repair of
  the KW33 source-contract gap, and it rejects strict subsets, strict supersets,
  missing or extra rows, duplicate ids, era disagreements and unexpected eras;
* every scored fit must share the recorded sample identity, and a confirmation
  cohort must additionally share every model parameter except the seed;
* GAP (1223-1230) and HOLDOUT (>=1231) eras are filtered at the Parquet scan
  boundary and refused if they somehow appear;
* the benchmark mean CORR is recomputed from the published column on the
  identical rows and must reproduce the frozen KW33 value within the declared
  tolerance, because a drifting benchmark would silently move every threshold.

Selection uses CORR alone. MMC, the weighted model score, Sharpe, zero-baseline
drawdown, the recent-20 window, the benchmark correlations and every Ender60
quantity are computed and reported as explicitly non-selecting diagnostics; the
decision functions in ``round2_parity_lib`` have no input path for any of them.

CORR and MMC are delegated to pinned official ``numerai_tools`` through the
audited Round-0 harness and are never rederived here.

Result paths are create-new-only: a second invocation against an existing
result path is refused rather than allowed to overwrite a recorded decision.

Running this module produces a scientific result and is authorised only by an
explicit, separately reviewed execution gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from agents.code.metrics.keystone_round0 import ScoreAuthority, score_round0
from agents.experiments.keystone28_corr_backbone_v53 import round2_parity_lib as kp

PACKET_DIR = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PACKET_DIR / "round2_parity_protocol.json"
DEFAULT_AUTHORITY = PACKET_DIR / "round0_score_authority.json"

RECENT_WINDOW = 20
BLOCK_SIZE = 15

MODE_SCREEN = "screen"
MODE_CONFIRMATION = "confirmation"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024 * 8), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_json(path: Path, payload: dict, *, kind: str) -> None:
    kp.assert_create_new_only(path.exists(), str(path), kind=kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------- data intake
def _load_zone_frame(data_root: Path, name: str, columns: list[str]) -> pd.DataFrame:
    zone = kp.score_zone_eras()
    frame = pq.read_table(
        data_root / name, columns=columns, filters=[("era", "in", zone)]
    ).to_pandas()
    if frame.index.name == "id":
        frame = frame.reset_index()
    kp.assert_scoring_zone_exact(frame["era"].unique(), context=f"{name} load")
    return frame.sort_values(["era", "id"], kind="stable").reset_index(drop=True)


def _universe(name: str, frame: pd.DataFrame) -> kp.RowUniverse:
    return kp.RowUniverse.from_columns(name, frame["era"], frame["id"])


def enforce_exact_row_universe(
    predictions: dict[str, pd.DataFrame],
    scoring: pd.DataFrame,
    meta_model: pd.DataFrame,
    benchmarks: pd.DataFrame,
    *,
    auxiliary_loaded: bool,
) -> dict:
    """Complete row-universe equality across every frame that touches a score.

    Round 1 checked era-set coverage and delegated row identity to a join, so a
    strict subset would have scored silently. Here the full canonical universe
    of every frame must be identical before a single metric is computed.
    """
    zone = kp.score_zone_eras()
    reference = _universe("scoring_target_frame", scoring)
    frames = [
        _universe("meta_model_frame", meta_model),
        _universe("benchmark_ender20_frame", benchmarks),
    ]
    if auxiliary_loaded:
        frames.append(_universe("benchmark_ender60_frame", benchmarks))
    for name, frame in predictions.items():
        kp.assert_finite_predictions(frame["prediction"], context=name)
        frames.append(_universe(name, frame))
    return kp.assert_exact_row_universe(
        reference,
        *frames,
        expected_eras=zone,
        expected_rows=kp.SCORING_UNIVERSE_ROWS,
        expected_canon_sha256=kp.SCORING_UNIVERSE_CANON_SHA256,
    )


# -------------------------------------------------------------------- scoring
def _score_vector(
    frame: pd.DataFrame,
    scoring: pd.DataFrame,
    meta_model: pd.DataFrame,
    benchmarks: pd.DataFrame,
    authority: ScoreAuthority,
    aux_authority: ScoreAuthority,
) -> dict:
    """Delegate CORR/MMC to pinned official numerai_tools via the Round-0 harness."""
    primary = score_round0(
        frame,
        scoring,
        meta_model,
        authority,
        benchmark_data=benchmarks,
        benchmark_cols=[kp.BENCHMARK_COLUMN, kp.BENCHMARK_60_COLUMN],
        expected_eras=kp.score_zone_eras(),
        recent_window=RECENT_WINDOW,
        block_size=BLOCK_SIZE,
    )
    aux = score_round0(
        frame,
        scoring,
        meta_model,
        aux_authority,
        expected_eras=kp.score_zone_eras(),
        recent_window=RECENT_WINDOW,
        block_size=BLOCK_SIZE,
    )
    per_era_corr = [float(v) for v in primary.per_era["corr"].tolist()]
    return {
        "selecting": {"per_era_corr": per_era_corr, "mean_corr": float(np.mean(per_era_corr))},
        "non_selecting_diagnostics": {
            "per_era_mmc": [float(v) for v in primary.per_era["mmc"].tolist()],
            "per_era_weighted_score": [
                float(v) for v in primary.per_era["weighted_score"].tolist()
            ],
            "summary": primary.summary,
            "ender60_auxiliary": {
                "per_era_corr60": [float(v) for v in aux.per_era["corr"].tolist()],
                "per_era_mmc60": [float(v) for v in aux.per_era["mmc"].tolist()],
                "summary": aux.summary,
                "combined_weighted_label": kp.ENDER60_AUX_LABEL,
                "selects": False,
            },
            "note": (
                "None of these quantities may change a parity decision. Parity "
                "selection uses CORR only."
            ),
        },
    }


def aux_authority_ender60(base: ScoreAuthority) -> ScoreAuthority:
    """Ender60 auxiliary authority: diagnostics only, never a selection input."""
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
            "KP35 Ender60 auxiliary: current 60D round score configs are "
            "non-payout (multiplier 0); any combined weighted number is the "
            + kp.ENDER60_AUX_LABEL
            + " and has no input path to any KP35 decision function.",
        ),
    )


# ---------------------------------------------------------------- artifact load
def load_fit(out_root: Path, stage: str, model_seed: int) -> tuple[pd.DataFrame, dict]:
    """Load one completed fit and prove its prediction matches its recorded hash."""
    pred_path = out_root / kp.artifact_relpath("prediction", stage=stage, model_seed=model_seed)
    log_path = out_root / kp.artifact_relpath("fit_log", stage=stage, model_seed=model_seed)
    for path in (pred_path, log_path):
        if not path.exists():
            raise FileNotFoundError(f"expected KP35 artifact missing: {path}")
    log = json.loads(log_path.read_text(encoding="utf-8"))
    if log.get("exit_status") != "success":
        raise ValueError(f"{stage} seed {model_seed} did not succeed; refusing to evaluate")
    digest = _sha256_file(pred_path)
    if digest != log.get("prediction_sha256"):
        raise ValueError(
            f"{pred_path}: sha256 {digest} != recorded {log.get('prediction_sha256')}"
        )
    if log.get("stage") != stage or log.get("model_seed") != model_seed:
        raise ValueError(f"{log_path}: fit log does not describe {stage} seed {model_seed}")
    frame = pd.read_parquet(pred_path)
    frame = frame.sort_values(["era", "id"], kind="stable").reset_index(drop=True)
    return frame, log


def assert_cohort_consistency(logs: dict[int, dict]) -> dict:
    """Every fit in a cohort shares one sample identity and one parameter set."""
    identities = {log["sample_identity_sha256"] for log in logs.values()}
    if len(identities) != 1:
        raise ValueError(f"cohort spans multiple sample identities: {sorted(identities)}")
    canon = {log["sample_canon_sha256"] for log in logs.values()}
    if len(canon) != 1:
        raise ValueError("cohort fits were trained on different sampled (era,id) universes")
    stripped = set()
    for log in logs.values():
        params = dict(log["params"])
        params.pop("seed", None)
        stripped.add(json.dumps({**params, "num_trees": log["num_trees"]}, sort_keys=True))
    if len(stripped) != 1:
        raise ValueError("cohort fits differ in a model parameter other than the seed")
    return {
        "sample_identity_sha256": next(iter(identities)),
        "sample_canon_sha256": next(iter(canon)),
        "seeds": sorted(logs),
        "identical_except_seed": True,
    }


# ------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=[MODE_SCREEN, MODE_CONFIRMATION])
    parser.add_argument("--stage", required=True, choices=list(kp.STAGES))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    parser.add_argument(
        "--prior-result",
        type=Path,
        default=None,
        help="Recorded prior result whose terminal_state authorises this evaluation.",
    )
    args = parser.parse_args(argv)

    stage = kp.assert_stage(args.stage)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    for key, value in kp.frozen_constants().items():
        if protocol["frozen_constants"].get(key) != value:
            raise ValueError(f"protocol disagrees with the frozen law on {key!r}")
    kp.assert_payout_target(protocol["payout_target"])

    prior_state = None
    if args.prior_result is not None:
        prior_state = json.loads(args.prior_result.read_text(encoding="utf-8"))["terminal_state"]

    if args.mode == MODE_SCREEN:
        kp.assert_stage_executable(stage, prior_state)
        seeds = [kp.SCREENING_SEED]
        result_path = args.out_root / kp.artifact_relpath("screen_result", stage=stage)
        kind = "screen_result"
    else:
        kp.assert_confirmation_authorized(prior_state)
        if prior_state != kp.STAGE_STATES[stage][0]:
            raise kp.StageAuthorityError(
                f"confirmation of {stage} requires {kp.STAGE_STATES[stage][0]}, "
                f"got {prior_state!r}"
            )
        seeds = [kp.SCREENING_SEED, *kp.CONFIRMATION_SEEDS]
        result_path = args.out_root / kp.artifact_relpath("confirmation_result", stage=stage)
        kind = "confirmation_result"

    # Refuse a second result write before doing any work at all.
    kp.assert_create_new_only(result_path.exists(), str(result_path), kind=kind)

    authority = ScoreAuthority.from_json(args.authority)
    kp.assert_payout_target(authority.payout_target)
    aux_authority = aux_authority_ender60(authority)

    scoring = _load_zone_frame(
        args.data_root, "validation.parquet", ["id", "era", authority.payout_target,
                                              aux_authority.payout_target]
    )
    meta_model = _load_zone_frame(
        args.data_root, "meta_model.parquet", ["id", "era", authority.meta_model_column]
    )
    benchmarks = _load_zone_frame(
        args.data_root,
        "validation_benchmark_models.parquet",
        ["id", "era", kp.BENCHMARK_COLUMN, kp.BENCHMARK_60_COLUMN],
    )

    frames: dict[str, pd.DataFrame] = {}
    logs: dict[int, dict] = {}
    for seed in seeds:
        frame, log = load_fit(args.out_root, stage, seed)
        frames[f"{stage}_seed{seed}"] = frame
        logs[seed] = log
    cohort = assert_cohort_consistency(logs)

    benchmark_vector = benchmarks[["id", "era", kp.BENCHMARK_COLUMN]].rename(
        columns={kp.BENCHMARK_COLUMN: "prediction"}
    )
    row_contract = enforce_exact_row_universe(
        {**frames, "benchmark_reference_vector": benchmark_vector},
        scoring,
        meta_model,
        benchmarks,
        auxiliary_loaded=True,
    )

    scored = {
        name: _score_vector(frame, scoring, meta_model, benchmarks, authority, aux_authority)
        for name, frame in {**frames, "benchmark_reference_vector": benchmark_vector}.items()
    }
    benchmark_mean_corr = kp.assert_benchmark_identity(
        scored["benchmark_reference_vector"]["selecting"]["mean_corr"]
    )

    if args.mode == MODE_SCREEN:
        decision = kp.screen_stage(
            stage,
            scored[f"{stage}_seed{kp.SCREENING_SEED}"]["selecting"]["mean_corr"],
            benchmark_mean_corr,
        )
    else:
        decision = kp.final_confirmation(
            scored[f"{stage}_seed{kp.SCREENING_SEED}"]["selecting"]["mean_corr"],
            scored[f"{stage}_seed{kp.CONFIRMATION_SEEDS[0]}"]["selecting"]["mean_corr"],
            scored[f"{stage}_seed{kp.CONFIRMATION_SEEDS[1]}"]["selecting"]["mean_corr"],
            benchmark_mean_corr,
        )
    kp.assert_forward_transition(prior_state, decision["terminal_state"])

    result = {
        "record": f"kp35_{args.mode}_result",
        "generated_utc": _now(),
        "stage": stage,
        "mode": args.mode,
        "prior_state": prior_state,
        "terminal_state": decision["terminal_state"],
        "decision": decision,
        "benchmark": {
            "column": kp.BENCHMARK_COLUMN,
            "recomputed_mean_corr": benchmark_mean_corr,
            "frozen_kw33_mean_corr": kp.BENCHMARK_MEAN_CORR,
            "tolerance": kp.BENCHMARK_MEAN_CORR_TOLERANCE,
            "identity_enforced": True,
        },
        "exact_row_contract": row_contract,
        "cohort": cohort,
        "score_zone": {
            "eras": [kp.score_zone_eras()[0], kp.score_zone_eras()[-1]],
            "n_eras": kp.N_SCORE_ERAS,
            "no_gap_or_holdout_loaded": True,
        },
        "scored_vectors": scored,
        "fit_logs": {
            str(seed): {
                k: log.get(k)
                for k in (
                    "stage", "model_seed", "role", "params_sha256",
                    "sample_identity_sha256", "sample_canon_sha256",
                    "rows_before_sampling", "rows_after_sampling",
                    "source_split_rows", "duration_seconds",
                    "peak_working_set_bytes", "prediction_sha256",
                    "prediction_canon_sha256", "exit_status", "retry_of",
                    "started_utc", "ended_utc",
                )
            }
            for seed, log in logs.items()
        },
        "authorizes_next_fit": False,
        "next_step": (
            "This evaluator does not authorise any further fit. Obtain "
            "independent review of this recorded result before invoking a "
            "successor stage or a confirmation cohort."
        ),
        "non_actions": (
            "No GAP/HOLDOUT access; no Candidate-V path; no Ender60 selection "
            "path; no upload, model creation, submission, staking, deployment, "
            "or Numerai account action."
        ),
    }
    _write_new_json(result_path, result, kind=kind)
    print(f"[{_now()}] terminal_state={decision['terminal_state']}")
    print(f"wrote {result_path}")
    print("no successor fit was started and none is authorised by this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
