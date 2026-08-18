"""Keystone Round-1 trainer (KW33): the exact 21-fit walk-forward cohort.

Serial execution of the frozen cohort defined by ``round1_lib.build_fit_specs``
and ``round1_protocol.json``: 3 CONTROL-T fits (train.parquet only) and 18
CANDIDATE-V fits (train.parquet plus strictly-prior, embargoed validation
history). Every fit writes a prediction parquet and a complete execution
record. Predictions, logs, and models live outside Git under ``--out-root``.

Guards: no GAP/HOLDOUT era is ever loaded; scored rows are never trained on;
the bare ``target`` alias is never referenced; sampling is deterministic,
era-balanced, capped, and independent of the model seed; there is no early
stopping and no eval set. A completed scientific fit is never rerun — reruns
require deleting nothing: the runner refuses unless ``--retry-fit-id`` names
a fit whose prediction artifact is absent (infrastructure retry, once).

``--probe N`` runs a non-scientific throughput probe (N trees, CONTROL-T
seed-42 cohort, no prediction written) used only to choose between the two
predeclared compute profiles before the protocol freeze.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

from agents.experiments.keystone28_corr_backbone_v53 import round1_lib as r1

PACKET_DIR = Path(__file__).resolve().parent
TARGET = None  # resolved from the protocol's authority reference at runtime


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024 * 8), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_working_set_bytes() -> int:
    return int(psutil.Process().memory_info().peak_wset)


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    lib_constants = {
        "score_zone": [r1.score_zone_eras()[0], r1.score_zone_eras()[-1]],
        "blocks": [list(b) for b in r1.WALK_FORWARD_BLOCKS],
        "embargo_eras": r1.EMBARGO_ERAS,
        "sampling_seed": r1.SAMPLING_SEED,
        "max_sampled_rows": r1.MAX_SAMPLED_ROWS,
        "model_seeds": list(r1.MODEL_SEEDS),
    }
    for key, expected in lib_constants.items():
        if protocol["frozen_constants"][key] != expected:
            raise ValueError(
                f"round1_protocol.json disagrees with round1_lib on '{key}': "
                f"{protocol['frozen_constants'][key]} != {expected}"
            )
    return protocol


def load_feature_list(data_root: Path, protocol: dict) -> list[str]:
    features = json.loads((data_root / "features.json").read_text(encoding="utf-8"))
    medium = features["feature_sets"][protocol["feature_set"]]
    digest = hashlib.sha256("\n".join(medium).encode("utf-8")).hexdigest()
    if digest != protocol["feature_list_sha256"]:
        raise ValueError(
            "features.json medium list hash does not match the frozen protocol."
        )
    return list(medium)


def load_frames(data_root: Path, feature_list: list[str], target: str):
    columns = ["id", "era", target, *feature_list]
    train = pq.read_table(data_root / "train.parquet", columns=columns).to_pandas()
    if train.index.name == "id":
        train = train.reset_index()
    val_eras = [f"{e:04d}" for e in range(r1.VALIDATION_FIRST_ERA, r1.SCORE_ZONE_END + 1)]
    validation = pq.read_table(
        data_root / "validation.parquet",
        columns=columns,
        filters=[("era", "in", val_eras)],
    ).to_pandas()
    if validation.index.name == "id":
        validation = validation.reset_index()
    r1.assert_no_forbidden_eras(
        sorted(validation["era"].unique()), context="validation load"
    )
    train = train.sort_values(["era", "id"], kind="stable").reset_index(drop=True)
    validation = validation.sort_values(["era", "id"], kind="stable").reset_index(
        drop=True
    )
    if train[target].isna().any():
        raise ValueError("train.parquet target contains nulls; refusing to fit.")
    return train, validation


def _training_cohort(spec, train, validation, target):
    if spec.procedure == r1.CONTROL_T:
        cohort = train
    else:
        eligible = r1.eligible_validation_eras(
            spec.block_index, sorted(validation["era"].unique())
        )
        val_slice = validation[validation["era"].isin(set(eligible))]
        if val_slice[target].isna().any():
            raise ValueError(
                f"{spec.fit_id}: eligible validation rows contain null targets."
            )
        overlap = set(val_slice["era"].unique()) & set(spec.scored_eras)
        if overlap:
            raise AssertionError(f"{spec.fit_id}: scored eras in training: {overlap}")
        cohort = pd.concat([train, val_slice], ignore_index=True)
    positions = r1.era_balanced_sample_positions(cohort["era"].to_numpy())
    return cohort, positions


def run_fit(spec, train, validation, feature_list, target, params, num_trees, out_root, retry_of=None):
    pred_path = out_root / "predictions" / f"{spec.fit_id}.parquet"
    log_path = out_root / "logs" / f"{spec.fit_id}.json"
    record = {
        "fit_id": spec.fit_id,
        "procedure": spec.procedure,
        "model_seed": spec.model_seed,
        "block_index": spec.block_index,
        "scored_eras": [spec.scored_eras[0], spec.scored_eras[-1]],
        "latest_validation_train_era": spec.latest_validation_train_era,
        "sampling_seed": spec.sampling_seed,
        "target": target,
        "n_features": len(feature_list),
        "num_trees": num_trees,
        "params_sha256": hashlib.sha256(
            json.dumps({**params, "num_trees": num_trees}, sort_keys=True).encode()
        ).hexdigest(),
        "started_utc": _now(),
        "retry_of": retry_of,
    }
    start = time.time()
    try:
        cohort, positions = _training_cohort(spec, train, validation, target)
        eligible_eras = sorted(cohort["era"].unique())
        record["eligible_train_eras"] = [eligible_eras[0], eligible_eras[-1]]
        record["n_eligible_train_eras"] = len(eligible_eras)
        record["rows_before_sampling"] = int(len(cohort))
        sampled = cohort.iloc[positions]
        record["rows_after_sampling"] = int(len(sampled))
        dataset = lgb.Dataset(
            sampled[feature_list],
            label=sampled[target].to_numpy(dtype="float32"),
            free_raw_data=True,
        )
        booster = lgb.train(params, dataset, num_boost_round=num_trees)
        del dataset, sampled, cohort
        gc.collect()
        scored = validation[validation["era"].isin(set(spec.scored_eras))]
        preds = booster.predict(scored[feature_list])
        frame = pd.DataFrame(
            {"id": scored["id"], "era": scored["era"], "prediction": preds}
        ).reset_index(drop=True)
        if not np.isfinite(frame["prediction"].to_numpy()).all():
            raise ValueError("non-finite prediction produced")
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(pred_path, index=False)
        record["prediction_path"] = str(pred_path)
        record["prediction_sha256"] = _sha256(pred_path)
        record["prediction_rows"] = int(len(frame))
        record["exit_status"] = "success"
        del booster, frame, scored, preds
        gc.collect()
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
        record["exit_status"] = f"FAILED: {type(exc).__name__}: {exc}"
        record["ended_utc"] = _now()
        record["duration_seconds"] = round(time.time() - start, 1)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        raise
    record["ended_utc"] = _now()
    record["duration_seconds"] = round(time.time() - start, 1)
    record["peak_working_set_bytes"] = peak_working_set_bytes()
    record["peak_vram"] = "not_applicable_cpu_training"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(
        f"[{_now()}] {spec.fit_id}: OK rows={record['rows_after_sampling']} "
        f"dur={record['duration_seconds']}s"
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument(
        "--protocol", type=Path, default=PACKET_DIR / "round1_protocol.json"
    )
    parser.add_argument("--probe", type=int, default=None)
    parser.add_argument("--retry-fit-id", type=str, default=None)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    target = protocol["payout_target"]
    if target == "target":
        raise ValueError("bare `target` alias is not a payout objective")
    feature_list = load_feature_list(args.data_root, protocol)
    profile = protocol["lightgbm_profile"]
    num_trees = profile["num_trees"]

    print(f"[{_now()}] loading frames...")
    train, validation = load_frames(args.data_root, feature_list, target)
    print(
        f"[{_now()}] train rows={len(train)} validation rows={len(validation)} "
        f"(eras {validation['era'].min()}-{validation['era'].max()})"
    )

    if args.probe is not None:
        spec = r1.build_fit_specs()[0]  # CONTROL-T seed 42 cohort
        params = r1.lightgbm_params(profile, spec.model_seed)
        cohort, positions = _training_cohort(spec, train, validation, target)
        sampled = cohort.iloc[positions]
        start = time.time()
        booster = lgb.train(
            params,
            lgb.Dataset(sampled[feature_list], label=sampled[target].to_numpy("float32")),
            num_boost_round=args.probe,
        )
        elapsed = time.time() - start
        del booster
        print(
            json.dumps(
                {
                    "probe_trees": args.probe,
                    "probe_seconds": round(elapsed, 1),
                    "rows": int(len(sampled)),
                    "seconds_per_tree": round(elapsed / args.probe, 4),
                    "peak_working_set_gb": round(peak_working_set_bytes() / 1024**3, 2),
                    "non_scientific": True,
                }
            )
        )
        return 0

    specs = r1.build_fit_specs()
    if args.retry_fit_id is not None:
        specs = [s for s in specs if s.fit_id == args.retry_fit_id]
        if not specs:
            raise ValueError(f"unknown fit id {args.retry_fit_id}")
        if (args.out_root / "predictions" / f"{args.retry_fit_id}.parquet").exists():
            raise ValueError(
                "refusing to retry: a valid prediction artifact already exists "
                "(completed scientific fits are never rerun)."
            )
    results = []
    for spec in specs:
        pred_path = args.out_root / "predictions" / f"{spec.fit_id}.parquet"
        if pred_path.exists() and args.retry_fit_id is None:
            print(f"[{_now()}] {spec.fit_id}: already complete, skipping")
            continue
        params = r1.lightgbm_params(profile, spec.model_seed)
        results.append(
            run_fit(
                spec,
                train,
                validation,
                feature_list,
                target,
                params,
                num_trees,
                args.out_root,
                retry_of=args.retry_fit_id,
            )
        )
    print(f"[{_now()}] cohort pass complete: {len(results)} fits executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
