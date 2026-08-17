"""Keystone Round-0 real-data smoke validation (KA28).

Deterministic plumbing proof for the Round-0 scoring harness on a small,
pre-holdout development slice of the current v5.3 data:

* the prediction vector is an existing published benchmark prediction column
  (no model is trained);
* joins are strict and one-to-one (the harness fails loudly otherwise);
* per-era CORR (CORR20V2 via ``numerai_tools``), MMC against the published
  Meta Model, and the authority-weighted score must all be finite;
* the era slice is frozen in ``round0_partition.json`` (``smoke_slice``) and is
  enforced with ``expected_eras``; holdout and embargo eras are refused;
* the output JSON contains no timestamps and is byte-identical across runs.

Offline by construction: reads only local parquet/json files below
``--data-root``. No numerapi import, no network, no credential, no account
action. Smoke metrics are plumbing evidence, not a candidate result; they rank
nothing and select nothing.

Usage:
    python -m agents.experiments.keystone28_corr_backbone_v53.round0_smoke \
        --data-root D:/numerai-data/keystone28/v5.3 --out round0_smoke_result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from agents.code.metrics.keystone_round0 import ScoreAuthority, score_round0

PACKET_DIR = Path(__file__).resolve().parent
SMOKE_PREDICTION_COLUMN = "v53_lgbm_ender20"


def _load(data_root: Path, name: str, columns: list[str], eras: list[str]):
    frame = pq.read_table(
        data_root / name, columns=columns, filters=[("era", "in", eras)]
    ).to_pandas()
    if frame.index.name == "id":
        frame = frame.reset_index()
    return frame


def run_smoke(data_root: Path, authority_path: Path, partition_path: Path) -> dict:
    authority = ScoreAuthority.from_json(authority_path)
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    smoke_eras = list(partition["smoke_slice"]["eras"])
    dev_end = partition["dev"]["end_era"]
    holdout_start = partition["holdout"]["start_era"]
    gap_start = partition["gap"]["start_era"]
    if not smoke_eras:
        raise ValueError("Partition smoke_slice declares no eras.")
    if max(smoke_eras) > dev_end or max(smoke_eras) >= min(gap_start, holdout_start):
        raise ValueError(
            "Smoke slice must stay strictly inside DEV, before the embargo gap "
            f"and the holdout; got {smoke_eras} with dev end {dev_end}, gap "
            f"start {gap_start}, holdout start {holdout_start}."
        )

    benchmarks = _load(
        data_root,
        "validation_benchmark_models.parquet",
        ["id", "era", SMOKE_PREDICTION_COLUMN],
        smoke_eras,
    )
    predictions = benchmarks.rename(columns={SMOKE_PREDICTION_COLUMN: "prediction"})
    scoring = _load(
        data_root, "validation.parquet", ["id", "era", authority.payout_target], smoke_eras
    )
    meta_model = _load(
        data_root,
        "meta_model.parquet",
        ["id", "era", authority.meta_model_column],
        smoke_eras,
    )

    result = score_round0(
        predictions,
        scoring,
        meta_model,
        authority,
        expected_eras=smoke_eras,
        recent_window=len(smoke_eras),
        block_size=len(smoke_eras),
    )

    per_era = {
        era: {col: float(result.per_era.loc[era, col]) for col in result.per_era.columns}
        for era in result.per_era.index
    }
    finite = all(
        all(_is_finite(v) for v in row.values()) for row in per_era.values()
    )
    if not finite:
        raise ValueError("Smoke produced a non-finite score; refusing to report.")

    return {
        "record": "keystone28_round0_smoke",
        "data_version": "v5.3",
        "purpose": (
            "Plumbing-only smoke of the Round-0 harness on real historical "
            "pre-holdout data. Not a candidate result; ranks and selects "
            "nothing. No model was trained; no holdout era was scored; no "
            "network or account action occurred."
        ),
        "prediction_vector": (
            f"published benchmark predictions '{SMOKE_PREDICTION_COLUMN}' from "
            "v5.3/validation_benchmark_models.parquet"
        ),
        "authority": {
            "payout_target": authority.payout_target,
            "corr_score_name": authority.corr_score_name,
            "mmc_score_name": authority.mmc_score_name,
            "corr_multiplier": authority.corr_multiplier,
            "mmc_multiplier": authority.mmc_multiplier,
            "meta_model_column": authority.meta_model_column,
        },
        "smoke_eras": smoke_eras,
        "inputs": {
            "prediction_rows": int(len(predictions)),
            "scoring_rows": int(len(scoring)),
            "meta_model_rows": int(len(meta_model)),
            "distinct_ids": int(predictions["id"].nunique()),
        },
        "checks": {
            "one_to_one_joins": True,
            "all_scores_finite": True,
            "expected_eras_enforced": True,
            "holdout_untouched": True,
            "excluded_eras": list(result.excluded_eras),
        },
        "per_era_scores": per_era,
        "summary": result.summary,
    }


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--authority", type=Path, default=PACKET_DIR / "round0_score_authority.json"
    )
    parser.add_argument(
        "--partition", type=Path, default=PACKET_DIR / "round0_partition.json"
    )
    args = parser.parse_args()
    payload = run_smoke(args.data_root, args.authority, args.partition)
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    args.out.write_bytes(text.encode("utf-8"))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"wrote {args.out} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
