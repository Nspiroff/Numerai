"""Keystone Round-2 parity-calibration trainer (KP35): exactly one fit per call.

One invocation trains one stage at one model seed and stops. There is no cohort
loop, no stage list, and no code path that starts a successor fit: chaining is
deliberately a human decision taken after independent review of the completed
artifact, which is why every eligibility check below reads a *recorded* prior
result rather than an in-process variable.

Refusal law enforced before any data is touched:

* the stage must be one of the two frozen parity stages (Candidate-V and every
  later ladder branch are structurally unreachable);
* seed 42 is the only screening seed; 1337 and 2024 are the only confirmation
  seeds, and a confirmation fit requires a recorded screen pass;
* P2 requires a recorded ``KP35_P1_SCREEN_FAILED_P2_AUTHORIZABLE`` artifact;
* the protocol record and ``round2_parity_lib`` must agree constant for constant;
* every data file must reproduce its frozen SHA-256;
* the payout target is explicit and the bare ``target`` alias is rejected;
* the feature list must match the frozen 780-feature medium hash;
* training loads eras 0001-1084 only — the purge 1085-1092, the benchmark
  prediction chunk 1093-1248, GAP 1223-1230 and HOLDOUT >=1231 are excluded at
  the Parquet scan boundary, so a forbidden row is never materialised;
* P2 must reuse P1's exact sampled ``(era, id)`` universe, proven by identity
  comparison rather than by regenerating an allegedly equivalent sample.

Artifacts. Predictions, logs, failure records and the sample manifest are
written outside Git under ``--out-root``. Paths are create-new-only: a final
path is never replaced, a completed prediction is never overwritten, a
completed fit is never rerun, and a failure record is preserved. Atomic
temp-to-final replacement is used for first-time creation only. No model
artifact is written; no future stage is started; nothing is uploaded.

Running this module performs a scientific fit and is authorised only by an
explicit, separately reviewed execution gate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

from agents.experiments.keystone28_corr_backbone_v53 import round2_parity_lib as kp

PACKET_DIR = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PACKET_DIR / "round2_parity_protocol.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024 * 8), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _peak_working_set_bytes() -> int:
    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", getattr(info, "rss", 0)))


def _cpu_posture() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "total_ram_bytes": psutil.virtual_memory().total,
        "device": "cpu",
        "deterministic": True,
        "force_row_wise": True,
        "gpu_used": False,
    }


def _write_new_json(path: Path, payload: dict, *, kind: str) -> None:
    """Atomic first-time creation only; a final path is never replaced."""
    kp.assert_create_new_only(path.exists(), str(path), kind=kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------- protocol revalidation
def load_protocol(path: Path) -> dict:
    """Load the protocol record and prove it agrees with the executable law."""
    protocol = json.loads(path.read_text(encoding="utf-8"))
    expected = kp.frozen_constants()
    actual = protocol["frozen_constants"]
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(
                "round2_parity_protocol.json disagrees with round2_parity_lib on "
                f"{key!r}: {actual.get(key)!r} != {value!r}"
            )
    for stage in kp.STAGES:
        declared = protocol["stages"][stage]["lightgbm_profile"]
        frozen = dict(kp.profile_for(stage))
        if declared != frozen:
            raise ValueError(f"protocol {stage} profile disagrees with the frozen law")
    kp.assert_payout_target(protocol["payout_target"])
    if protocol["feature_list_sha256"] != kp.FEATURE_LIST_SHA256:
        raise ValueError("protocol feature-list hash is not the frozen list")
    kp.assert_declared_profile_difference()
    return protocol


def revalidate_data_identities(data_root: Path, protocol: dict) -> dict:
    """Every declared data file must reproduce its frozen SHA-256."""
    verified: dict[str, str] = {}
    for name, declared in protocol["data_identities"].items():
        path = data_root / name
        if not path.exists():
            raise FileNotFoundError(f"declared data file missing: {path}")
        size = path.stat().st_size
        if size != declared["bytes"]:
            raise ValueError(f"{name}: {size} bytes != declared {declared['bytes']}")
        digest = _sha256_file(path)
        if digest != declared["sha256"]:
            raise ValueError(
                f"{name}: sha256 {digest} != frozen {declared['sha256']}; the "
                "protocol is anchored to different data and must not be executed"
            )
        verified[name] = digest
    return verified


def load_feature_list(data_root: Path, protocol: dict) -> list[str]:
    features = json.loads((data_root / "features.json").read_text(encoding="utf-8"))
    medium = list(features["feature_sets"][protocol["feature_set"]])
    if len(medium) != kp.N_FEATURES:
        raise ValueError(f"medium feature set has {len(medium)} features != {kp.N_FEATURES}")
    digest = _sha256_text("\n".join(medium))
    if digest != kp.FEATURE_LIST_SHA256:
        raise ValueError("features.json medium list hash does not match the frozen protocol")
    return medium


def read_prior_state(path: Path | None) -> str | None:
    """Stage eligibility is read from a recorded artifact, never asserted inline."""
    if path is None:
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    state = record.get("terminal_state")
    if state is None:
        raise ValueError(f"{path} carries no terminal_state")
    return state


# ------------------------------------------------------------------ data load
def load_training_frames(data_root: Path, feature_list: list[str], target: str):
    """Load only authorised training history: eras 0001-1084, nothing later.

    Era filters are applied at the Parquet scan boundary so no purge, benchmark
    chunk, GAP or HOLDOUT row is ever materialised in memory.
    """
    columns = ["id", "era", target, *feature_list]
    train_eras = [f"{e:04d}" for e in range(kp.TRAIN_PARQUET_FIRST_ERA, kp.TRAIN_PARQUET_LAST_ERA + 1)]
    history_eras = [f"{e:04d}" for e in range(kp.VALIDATION_FIRST_ERA, kp.HISTORY_BOUNDARY_END + 1)]

    train = pq.read_table(
        data_root / "train.parquet", columns=columns, filters=[("era", "in", train_eras)]
    ).to_pandas()
    if train.index.name == "id":
        train = train.reset_index()
    history = pq.read_table(
        data_root / "validation.parquet", columns=columns, filters=[("era", "in", history_eras)]
    ).to_pandas()
    if history.index.name == "id":
        history = history.reset_index()

    for frame, label in ((train, "train.parquet"), (history, "validation history")):
        kp.assert_training_eras_authorized(
            sorted(frame["era"].unique()), context=f"{label} load"
        )
        if frame[target].isna().any():
            raise ValueError(f"{label}: {target} contains nulls; refusing to fit")

    train = train.assign(_source="train")
    history = history.assign(_source="validation")
    cohort = pd.concat([train, history], ignore_index=True)
    # The sampling law is order-independent only if rows arrive sorted by (era, id).
    cohort = cohort.sort_values(["era", "id"], kind="stable").reset_index(drop=True)
    del train, history
    gc.collect()
    return cohort


def load_scoring_frame(data_root: Path, feature_list: list[str]) -> pd.DataFrame:
    """Load exactly the 87-era scoring zone: keys and features only, no target."""
    zone = kp.score_zone_eras()
    frame = pq.read_table(
        data_root / "validation.parquet",
        columns=["id", "era", *feature_list],
        filters=[("era", "in", zone)],
    ).to_pandas()
    if frame.index.name == "id":
        frame = frame.reset_index()
    kp.assert_scoring_zone_exact(frame["era"].unique(), context="scoring frame load")
    frame = frame.sort_values(["era", "id"], kind="stable").reset_index(drop=True)
    universe = kp.RowUniverse.from_columns("scoring_frame", frame["era"], frame["id"])
    kp.assert_exact_row_universe(
        universe,
        expected_eras=zone,
        expected_rows=kp.SCORING_UNIVERSE_ROWS,
        expected_canon_sha256=kp.SCORING_UNIVERSE_CANON_SHA256,
    )
    return frame


# -------------------------------------------------------------- sample custody
def build_or_load_sample(
    cohort: pd.DataFrame, data_identities: dict, out_root: Path
) -> tuple[np.ndarray, dict]:
    """Construct the canonical sample on first use; afterwards load and reuse it.

    P2 never regenerates a fresh sample: it loads P1's manifest and proves the
    identity matches, which is what makes "the only change is the model
    profile" a checked fact rather than a claim.
    """
    manifest_path = out_root / kp.artifact_relpath("sample_identity")
    positions = kp.era_balanced_sample_positions(cohort["era"].to_numpy())
    selected = cohort.iloc[positions]
    canon = kp.canonical_key_hash(selected["era"].tolist(), selected["id"].tolist())

    eligible_eras = sorted(cohort["era"].unique().tolist())
    per_era = kp.per_era_counts(selected["era"].tolist())
    split = selected["_source"].value_counts().to_dict()
    identities = {name: digest for name, digest in sorted(data_identities.items())}

    manifest = {
        "record": "kp35_sample_identity",
        "non_scientific": False,
        "generated_utc": _now(),
        "data_identities": identities,
        "eligible_era_range": [eligible_eras[0], eligible_eras[-1]],
        "n_eligible_eras": len(eligible_eras),
        "feature_set": kp.FEATURE_SET,
        "feature_list_sha256": kp.FEATURE_LIST_SHA256,
        "n_features": kp.N_FEATURES,
        "sampling_law_version": kp.SAMPLING_LAW_VERSION,
        "sampling_seed": kp.SAMPLING_SEED,
        "row_cap": kp.MAX_SAMPLED_ROWS,
        "rows_before_sampling": int(len(cohort)),
        "selected_row_count": int(len(selected)),
        "selected_rows_per_era": per_era,
        "source_split_rows": {
            "train": int(split.get("train", 0)),
            "validation": int(split.get("validation", 0)),
        },
        "sample_canon_sha256": canon,
        "sample_identity_sha256": kp.sample_identity(
            data_identities=identities,
            eligible_era_range=[eligible_eras[0], eligible_eras[-1]],
            feature_list_sha256=kp.FEATURE_LIST_SHA256,
            sampling_law_version=kp.SAMPLING_LAW_VERSION,
            sampling_seed=kp.SAMPLING_SEED,
            row_cap=kp.MAX_SAMPLED_ROWS,
            sample_canon_sha256=canon,
        ),
    }
    kp.validate_sample_manifest(manifest)

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        kp.assert_shared_sample_identity(existing, manifest)
        print(f"[{_now()}] reusing recorded sample identity "
              f"{existing['sample_identity_sha256'][:16]}...", flush=True)
        return positions, existing

    _write_new_json(manifest_path, manifest, kind="sample_identity")
    print(f"[{_now()}] wrote sample identity {manifest['sample_identity_sha256'][:16]}...",
          flush=True)
    return positions, manifest


# --------------------------------------------------------------------- the fit
def run_fit(
    *,
    stage: str,
    model_seed: int,
    cohort: pd.DataFrame,
    positions: np.ndarray,
    manifest: dict,
    scoring: pd.DataFrame,
    feature_list: list[str],
    target: str,
    protocol: dict,
    data_identities: dict,
    out_root: Path,
    retry_of: str | None,
) -> dict:
    profile = kp.profile_for(stage)
    params = kp.lightgbm_params(profile, model_seed)
    num_trees = int(profile["num_trees"])

    pred_path = out_root / kp.artifact_relpath("prediction", stage=stage, model_seed=model_seed)
    log_path = out_root / kp.artifact_relpath("fit_log", stage=stage, model_seed=model_seed)
    fail_path = out_root / kp.artifact_relpath("failure_record", stage=stage, model_seed=model_seed)

    kp.assert_create_new_only(pred_path.exists(), str(pred_path), kind="prediction")
    kp.assert_create_new_only(log_path.exists(), str(log_path), kind="fit_log")

    record = {
        "record": "kp35_fit_log",
        "stage": stage,
        "model_seed": model_seed,
        "role": "screening" if model_seed == kp.SCREENING_SEED else "confirmation",
        "payout_target": target,
        "feature_set": kp.FEATURE_SET,
        "n_features": len(feature_list),
        "feature_list_sha256": kp.FEATURE_LIST_SHA256,
        "profile_name": profile["name"],
        "num_trees": num_trees,
        "params": params,
        "params_sha256": _sha256_text(json.dumps({**params, "num_trees": num_trees}, sort_keys=True)),
        "protocol_sha256": _sha256_text(json.dumps(protocol, sort_keys=True)),
        "data_identities": data_identities,
        "sample_identity_sha256": manifest["sample_identity_sha256"],
        "sample_canon_sha256": manifest["sample_canon_sha256"],
        "rows_before_sampling": manifest["rows_before_sampling"],
        "rows_after_sampling": manifest["selected_row_count"],
        "source_split_rows": manifest["source_split_rows"],
        "selected_rows_per_era": manifest["selected_rows_per_era"],
        "eligible_era_range": manifest["eligible_era_range"],
        "n_eligible_eras": manifest["n_eligible_eras"],
        "scored_eras": [kp.score_zone_eras()[0], kp.score_zone_eras()[-1]],
        "no_early_stopping": True,
        "no_evaluation_set": True,
        "model_artifact_written": False,
        "cpu_posture": _cpu_posture(),
        "started_utc": _now(),
        "retry_of": retry_of,
        "next_stage_started": False,
    }

    start = time.time()
    try:
        sampled = cohort.iloc[positions]
        kp.assert_training_eras_authorized(
            sorted(sampled["era"].unique()), context=f"{stage} sampled training rows"
        )
        dataset = lgb.Dataset(
            sampled[feature_list],
            label=sampled[target].to_numpy(dtype="float32"),
            free_raw_data=True,
        )
        # No `valid_sets`, no `callbacks`, no early stopping anywhere.
        booster = lgb.train(params, dataset, num_boost_round=num_trees)
        del dataset, sampled
        gc.collect()

        preds = booster.predict(scoring[feature_list])
        kp.assert_finite_predictions(preds, context=f"{stage} seed {model_seed}")
        frame = pd.DataFrame(
            {"id": scoring["id"], "era": scoring["era"], "prediction": preds}
        ).reset_index(drop=True)
        universe = kp.RowUniverse.from_columns(
            f"{stage}_seed{model_seed}", frame["era"], frame["id"]
        )
        kp.assert_exact_row_universe(
            universe,
            expected_eras=kp.score_zone_eras(),
            expected_rows=kp.SCORING_UNIVERSE_ROWS,
            expected_canon_sha256=kp.SCORING_UNIVERSE_CANON_SHA256,
        )

        pred_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = pred_path.with_suffix(".parquet.tmp-%d" % os.getpid())
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, pred_path)

        record["prediction_path"] = str(pred_path)
        record["prediction_sha256"] = _sha256_file(pred_path)
        record["prediction_rows"] = int(len(frame))
        record["prediction_canon_sha256"] = universe.canon_sha256
        record["exit_status"] = "success"
        del booster, frame, preds
        gc.collect()
    except Exception as exc:  # noqa: BLE001 - recorded as a preserved failure, then re-raised
        record["exit_status"] = f"FAILED: {type(exc).__name__}: {exc}"
        record["ended_utc"] = _now()
        record["duration_seconds"] = round(time.time() - start, 1)
        record["peak_working_set_bytes"] = _peak_working_set_bytes()
        _write_new_json(fail_path, record, kind="failure_record")
        raise

    record["ended_utc"] = _now()
    record["duration_seconds"] = round(time.time() - start, 1)
    record["peak_working_set_bytes"] = _peak_working_set_bytes()
    record["peak_vram"] = "not_applicable_cpu_training"
    _write_new_json(log_path, record, kind="fit_log")
    print(
        f"[{_now()}] {stage} seed {model_seed}: OK "
        f"rows={record['rows_after_sampling']} dur={record['duration_seconds']}s",
        flush=True,
    )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=list(kp.STAGES))
    parser.add_argument("--model-seed", required=True, type=int, choices=list(kp.ALL_SEEDS))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--prior-result",
        type=Path,
        default=None,
        help="Recorded prior stage result whose terminal_state authorises this fit.",
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Single infrastructural retry; permitted only when no valid prediction exists.",
    )
    args = parser.parse_args(argv)

    stage = kp.assert_stage(args.stage)
    seed = args.model_seed
    screening = seed == kp.SCREENING_SEED
    kp.assert_stage_seed(stage, seed, screening=screening)
    prior_state = read_prior_state(args.prior_result)

    if screening:
        kp.assert_stage_executable(stage, prior_state)
    else:
        kp.assert_confirmation_authorized(prior_state)
        expected_pass = kp.STAGE_STATES[stage][0]
        if prior_state != expected_pass:
            raise kp.StageAuthorityError(
                f"confirmation seed {seed} for {stage} requires a recorded "
                f"{expected_pass}; got {prior_state!r}"
            )

    pred_path = args.out_root / kp.artifact_relpath("prediction", stage=stage, model_seed=seed)
    if args.retry:
        kp.assert_retry_authorized(
            prediction_exists=pred_path.exists(),
            prior_retries=0,
            stage=stage,
            model_seed=seed,
        )
    kp.assert_create_new_only(pred_path.exists(), str(pred_path), kind="prediction")

    protocol = load_protocol(args.protocol)
    target = kp.assert_payout_target(protocol["payout_target"])
    data_identities = revalidate_data_identities(args.data_root, protocol)
    feature_list = load_feature_list(args.data_root, protocol)

    print(f"[{_now()}] loading authorised training history 0001-{kp.HISTORY_BOUNDARY_END}...",
          flush=True)
    cohort = load_training_frames(args.data_root, feature_list, target)
    positions, manifest = build_or_load_sample(cohort, data_identities, args.out_root)
    scoring = load_scoring_frame(args.data_root, feature_list)

    run_fit(
        stage=stage,
        model_seed=seed,
        cohort=cohort,
        positions=positions,
        manifest=manifest,
        scoring=scoring,
        feature_list=feature_list,
        target=target,
        protocol=protocol,
        data_identities=data_identities,
        out_root=args.out_root,
        retry_of=f"{stage}_seed{seed}" if args.retry else None,
    )
    print(
        f"[{_now()}] one fit complete. No successor stage was started and none "
        "is authorised by this run; evaluate the artifact and obtain separate "
        "review before any further fit.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
