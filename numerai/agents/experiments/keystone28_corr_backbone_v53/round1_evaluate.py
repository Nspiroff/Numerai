"""Keystone Round-1 frozen evaluator (KW33).

Runs once over the completed external prediction artifacts, scores every
vector on the identical 87-era zone under the revalidated Ender20 authority
(plus the Ender60 auxiliary authority), applies the pipeline-parity sanity
gate and the pre-registered decision law, and emits the compact
``round1_result.json`` and ``round1_report.md``.

The decision uses only Ender20 weighted model scores. Ender60 numbers are
auxiliary cutover-readiness diagnostics; any combined Ender60 value is the
``HYPOTHETICAL_CUTOVER_WEIGHTED_DIAGNOSTIC`` and cannot select the winner.
No GAP (1223-1230) or HOLDOUT (>=1231) row is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from agents.code.metrics.keystone_round0 import ScoreAuthority, score_round0
from agents.experiments.keystone28_corr_backbone_v53 import round1_lib as r1

PACKET_DIR = Path(__file__).resolve().parent
BENCHMARK_20 = "v53_lgbm_ender20"
BENCHMARK_60 = "v53_lgbm_ender60"
RECENT_WINDOW = 20
BLOCK_SIZE = 15  # summarize blocks == the six walk-forward blocks (5x15 + 12)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_zone_frame(data_root: Path, name: str, columns: list[str]) -> pd.DataFrame:
    zone = r1.score_zone_eras()
    frame = pq.read_table(
        data_root / name, columns=columns, filters=[("era", "in", zone)]
    ).to_pandas()
    if frame.index.name == "id":
        frame = frame.reset_index()
    r1.assert_no_forbidden_eras(sorted(frame["era"].unique()), context=f"{name} load")
    if sorted(frame["era"].unique()) != zone:
        raise ValueError(f"{name} does not cover the exact 87-era score zone")
    return frame


def _assemble_predictions(out_root: Path) -> dict[str, pd.DataFrame]:
    """One 87-era vector per (procedure, seed); exactly one row per id."""
    vectors: dict[str, pd.DataFrame] = {}
    zone = r1.score_zone_eras()
    for seed in r1.MODEL_SEEDS:
        control = pd.read_parquet(out_root / "predictions" / f"control_t_seed{seed}.parquet")
        blocks = [
            pd.read_parquet(
                out_root / "predictions" / f"candidate_v_block{i + 1}_seed{seed}.parquet"
            )
            for i in range(len(r1.WALK_FORWARD_BLOCKS))
        ]
        candidate = pd.concat(blocks, ignore_index=True)
        for name, frame in ((f"control_t_seed{seed}", control), (f"candidate_v_seed{seed}", candidate)):
            r1.validate_prediction_vector(frame["id"].tolist(), frame["era"].tolist())
            vectors[name] = frame.sort_values(["era", "id"], kind="stable").reset_index(drop=True)
    del zone
    return vectors


def _score(prediction_frame, scoring, meta_model, benchmarks, authority, aux_authority):
    primary = score_round0(
        prediction_frame,
        scoring,
        meta_model,
        authority,
        benchmark_data=benchmarks,
        benchmark_cols=[BENCHMARK_20, BENCHMARK_60],
        expected_eras=r1.score_zone_eras(),
        recent_window=RECENT_WINDOW,
        block_size=BLOCK_SIZE,
    )
    aux = score_round0(
        prediction_frame,
        scoring,
        meta_model,
        aux_authority,
        expected_eras=r1.score_zone_eras(),
        recent_window=RECENT_WINDOW,
        block_size=BLOCK_SIZE,
    )
    return primary, aux


def _series(result, column) -> list[float]:
    return [float(v) for v in result.per_era[column].tolist()]


def _seed_mean_series(per_seed: list[list[float]]) -> list[float]:
    return [float(np.mean(vals)) for vals in zip(*per_seed)]


_stats = r1.zero_baseline_stats


def _block_means(values: list[float]) -> list[float]:
    means, pos = [], 0
    for start, end in r1.WALK_FORWARD_BLOCKS:
        n = end - start + 1
        means.append(float(np.mean(values[pos : pos + n])))
        pos += n
    return means


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    args = parser.parse_args()

    authority = ScoreAuthority.from_json(PACKET_DIR / "round0_score_authority.json")
    aux_authority = r1.aux_authority_ender60(authority)
    protocol = json.loads((PACKET_DIR / "round1_protocol.json").read_text(encoding="utf-8"))

    scoring = _load_zone_frame(
        args.data_root,
        "validation.parquet",
        ["id", "era", authority.payout_target, aux_authority.payout_target],
    )
    meta_model = _load_zone_frame(
        args.data_root, "meta_model.parquet", ["id", "era", authority.meta_model_column]
    )
    benchmarks = _load_zone_frame(
        args.data_root,
        "validation_benchmark_models.parquet",
        ["id", "era", BENCHMARK_20, BENCHMARK_60],
    )
    for column in (authority.payout_target, aux_authority.payout_target):
        if scoring[column].isna().any():
            raise ValueError(f"scoring frame has null {column} inside the zone")

    vectors = _assemble_predictions(args.out_root)
    benchmark_vector = benchmarks[["id", "era", BENCHMARK_20]].rename(
        columns={BENCHMARK_20: "prediction"}
    )

    scored: dict[str, dict] = {}
    for name, frame in {**vectors, "benchmark_v53_lgbm_ender20": benchmark_vector}.items():
        primary, aux = _score(
            frame, scoring, meta_model, benchmarks, authority, aux_authority
        )
        scored[name] = {
            "ender20": {
                "per_era_corr": _series(primary, "corr"),
                "per_era_mmc": _series(primary, "mmc"),
                "per_era_weighted": _series(primary, "weighted_score"),
                "summary": primary.summary,
            },
            "ender60_auxiliary": {
                "per_era_corr60": _series(aux, "corr"),
                "per_era_mmc60": _series(aux, "mmc"),
                "summary": aux.summary,
                "combined_weighted_label": r1.ENDER60_AUX_LABEL,
            },
        }

    def proc_view(procedure: str) -> dict:
        seeds = list(r1.MODEL_SEEDS)
        per_seed_weighted = [
            scored[f"{procedure}_seed{s}"]["ender20"]["per_era_weighted"] for s in seeds
        ]
        per_seed_stats = {
            str(s): _stats(scored[f"{procedure}_seed{s}"]["ender20"]["per_era_weighted"])
            for s in seeds
        }
        seed_mean = _seed_mean_series(per_seed_weighted)
        means = [per_seed_stats[str(s)]["mean"] for s in seeds]
        return {
            "per_seed_weighted_stats": per_seed_stats,
            "seed_mean_per_era_weighted": seed_mean,
            "seed_mean_weighted_mean": float(np.mean(means)),
            "seed_dispersion_of_means": float(np.std(means, ddof=0)),
            "block_means": _block_means(seed_mean),
            "worst_seed_weighted_mean": float(min(means)),
            "worst_seed_sharpe": float(
                min(per_seed_stats[str(s)]["sharpe"] for s in seeds)
            ),
            "worst_seed_drawdown": float(
                max(
                    per_seed_stats[str(s)]["max_drawdown_zero_baseline"] for s in seeds
                )
            ),
        }

    control_view = proc_view("control_t")
    candidate_view = proc_view("candidate_v")

    control_mean_corr = float(
        np.mean(
            [
                np.mean(scored[f"control_t_seed{s}"]["ender20"]["per_era_corr"])
                for s in r1.MODEL_SEEDS
            ]
        )
    )
    benchmark_mean_corr = float(
        np.mean(scored["benchmark_v53_lgbm_ender20"]["ender20"]["per_era_corr"])
    )
    parity_ok = r1.pipeline_parity_ok(control_mean_corr, benchmark_mean_corr)

    if not parity_ok:
        terminal = "STOPPED_AT_KW33_PIPELINE_PARITY_FAILURE"
        decision = None
        bootstrap = None
    else:
        decision = r1.decide_round1(candidate_view, control_view)
        terminal = decision["terminal_state"]
        diff = [
            c - t
            for c, t in zip(
                candidate_view["seed_mean_per_era_weighted"],
                control_view["seed_mean_per_era_weighted"],
            )
        ]
        bootstrap = r1.moving_block_bootstrap_ci(diff)

    fit_logs = []
    for spec in r1.build_fit_specs():
        log = json.loads(
            (args.out_root / "logs" / f"{spec.fit_id}.json").read_text(encoding="utf-8")
        )
        fit_logs.append(
            {
                k: log.get(k)
                for k in (
                    "fit_id",
                    "procedure",
                    "model_seed",
                    "block_index",
                    "eligible_train_eras",
                    "n_eligible_train_eras",
                    "rows_before_sampling",
                    "rows_after_sampling",
                    "duration_seconds",
                    "peak_working_set_bytes",
                    "params_sha256",
                    "prediction_sha256",
                    "exit_status",
                    "retry_of",
                    "started_utc",
                    "ended_utc",
                )
            }
        )
        if log["exit_status"] != "success":
            raise ValueError(f"fit {spec.fit_id} did not succeed; refusing to evaluate")

    result = {
        "record": "keystone28_round1_result",
        "generated_utc": _now(),
        "terminal_state": terminal,
        "authority": {
            "payout_target": authority.payout_target,
            "corr_multiplier": authority.corr_multiplier,
            "mmc_multiplier": authority.mmc_multiplier,
            "revalidation": protocol["authority_revalidation"],
        },
        "score_zone": {
            "eras": [r1.score_zone_eras()[0], r1.score_zone_eras()[-1]],
            "n_eras": len(r1.score_zone_eras()),
            "blocks": [list(b) for b in r1.WALK_FORWARD_BLOCKS],
            "no_gap_or_holdout_loaded": True,
        },
        "pipeline_parity_gate": {
            "control_seed_mean_corr": control_mean_corr,
            "benchmark_mean_corr": benchmark_mean_corr,
            "required_fraction": r1.DECISION_THRESHOLDS["pipeline_parity_min_corr_fraction"],
            "passed": parity_ok,
        },
        "control_t": control_view,
        "candidate_v": candidate_view,
        "decision": decision,
        "bootstrap_report_only": bootstrap,
        "scored_vectors": scored,
        "fit_matrix": fit_logs,
        "non_actions": (
            "No GAP/HOLDOUT access; no training beyond the frozen 21-fit "
            "cohort; no upload, model creation, submission, staking, "
            "deployment, or Numerai account action."
        ),
    }
    result_path = PACKET_DIR / "round1_result.json"
    result_path.write_bytes((json.dumps(result, indent=1) + "\n").encode("utf-8"))

    report = _render_report(result)
    (PACKET_DIR / "round1_report.md").write_bytes(report.encode("utf-8"))
    print(f"[{_now()}] terminal_state={terminal}")
    print(f"wrote {result_path}")
    return 0


def _render_report(result: dict) -> str:
    c, v = result["control_t"], result["candidate_v"]
    b = result["scored_vectors"]["benchmark_v53_lgbm_ender20"]["ender20"]["summary"]
    lines = [
        "# Keystone Round-1 report (KW33)",
        "",
        f"Generated {result['generated_utc']}. Terminal state: **{result['terminal_state']}**.",
        "",
        "Primary comparison: CANDIDATE-V vs CONTROL-T on the Ender20 seed-mean",
        "weighted model score (corr x 0.75 + mmc x 2.25) over the 87-era zone",
        "1133-1219 (six walk-forward blocks, eight-era embargo).",
        "",
        "| quantity | CONTROL-T | CANDIDATE-V |",
        "| --- | --- | --- |",
        f"| seed-mean weighted mean | {c['seed_mean_weighted_mean']:.6f} | {v['seed_mean_weighted_mean']:.6f} |",
        f"| worst-seed weighted mean | {c['worst_seed_weighted_mean']:.6f} | {v['worst_seed_weighted_mean']:.6f} |",
        f"| worst-seed sharpe | {c['worst_seed_sharpe']:.4f} | {v['worst_seed_sharpe']:.4f} |",
        f"| worst-seed zero-baseline drawdown | {c['worst_seed_drawdown']:.6f} | {v['worst_seed_drawdown']:.6f} |",
        f"| seed dispersion of means | {c['seed_dispersion_of_means']:.6f} | {v['seed_dispersion_of_means']:.6f} |",
        "",
        f"Benchmark v53_lgbm_ender20 mean CORR {b['scores']['corr']['mean']:.6f}, "
        f"mean weighted {b['scores']['weighted_score']['mean']:.6f} on the same eras.",
        "",
        "## Decision reconstruction",
        "",
    ]
    if result["decision"] is None:
        lines.append("Pipeline parity gate FAILED; no scientific decision was taken.")
    else:
        lines.append("| condition | threshold | value | result |")
        lines.append("| --- | --- | --- | --- |")
        for name, cond in result["decision"]["conditions"].items():
            value = cond["value"]
            shown = f"{value:.6f}" if isinstance(value, float) else str(value)
            lines.append(
                f"| {name} | {cond['threshold']} | {shown} | "
                f"{'PASS' if cond['passed'] else 'FAIL'} |"
            )
        boot = result["bootstrap_report_only"]
        lines += [
            "",
            f"Report-only moving-block bootstrap (n={boot['n_resamples']}, block "
            f"{boot['block_length']}, seed {boot['seed']}): mean difference "
            f"{boot['observed_mean_difference']:.6f}, 90% CI "
            f"[{boot['ci_low']:.6f}, {boot['ci_high']:.6f}].",
        ]
    aux_c = result["scored_vectors"]["control_t_seed42"]["ender60_auxiliary"]
    lines += [
        "",
        "## Ender60 auxiliary (cutover readiness, non-selecting)",
        "",
        "Raw CORR60/MMC60 are reported per vector in round1_result.json. Any",
        f"combined number is the {aux_c['combined_weighted_label']} and never",
        "selects a winner. Current 60D round score configs are non-payout.",
        "",
        "## Non-actions",
        "",
        result["non_actions"],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
