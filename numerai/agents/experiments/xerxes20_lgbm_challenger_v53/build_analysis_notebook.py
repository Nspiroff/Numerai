"""Build and optionally execute the frozen Xerxes20 scout reader notebook."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
EXPECTED_RESULT_SHA256 = "c5939fc19c57688788fc2fdd2e28e8a49e99394ecab5aac019ddf1069cd62c6d"
EXPECTED_GENERATION_ID = "92389e16ab7f6fe244c1"
EXPECTED_CANDIDATES = (
    "r1_base_d6_t6000",
    "r1_trees2k",
    "r1_depth5",
    "r1_depth8",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _notebook_path_literal(source: Path, notebook_dir: Path) -> str:
    relative = Path(os.path.relpath(source.resolve(), notebook_dir.resolve())).as_posix()
    return repr(relative)


def _metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in EXPECTED_CANDIDATES:
        candidate_rows = frame.loc[frame["candidate"] == candidate].sort_values("era")
        bmc = candidate_rows["bmc"]
        corr = candidate_rows["corr"]
        cumulative_bmc = bmc.cumsum()
        rows.append(
            {
                "segment": "scout_calibration",
                "candidate": candidate,
                "era_count": len(candidate_rows),
                "corr_mean": corr.mean(),
                "corr_std": corr.std(ddof=0),
                "corr_sharpe": corr.mean() / corr.std(ddof=0),
                "bmc_mean": bmc.mean(),
                "bmc_std": bmc.std(ddof=0),
                "bmc_sharpe": bmc.mean() / bmc.std(ddof=0),
                "bmc_max_drawdown": (cumulative_bmc.cummax() - cumulative_bmc).max(),
                "avg_benchmark_similarity": candidate_rows[
                    "benchmark_similarity"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def _load_and_verify_frozen_outputs(
    result_json: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if _sha256_file(result_json) != EXPECTED_RESULT_SHA256:
        raise ValueError("The canonical evaluator result does not match its frozen hash.")

    result = json.loads(result_json.read_text(encoding="utf-8"))
    if result.get("experiment") != "xerxes20_lgbm_challenger_v53":
        raise ValueError("Result JSON is not for the Xerxes20 challenger experiment.")
    if result.get("generation_id") != EXPECTED_GENERATION_ID:
        raise ValueError("Unexpected evaluator generation ID.")
    if result.get("state") != "STOP_NO_SCOUT_CALIBRATION_WINNER":
        raise ValueError("The frozen evaluator decision changed unexpectedly.")

    content_result = result_json.with_name(
        f"xerxes20_result-{EXPECTED_GENERATION_ID}.json"
    )
    if _sha256_file(content_result) != EXPECTED_RESULT_SHA256:
        raise ValueError("The content-addressed evaluator result differs from canonical.")

    summary_path = result_json.parent / Path(result["outputs"]["summary_csv"]).name
    per_era_path = result_json.parent / Path(result["outputs"]["per_era_csv"]).name
    if _sha256_file(summary_path) != result["outputs"]["summary_csv_sha256"]:
        raise ValueError("Summary CSV hash does not match the evaluator receipt.")
    if _sha256_file(per_era_path) != result["outputs"]["per_era_csv_sha256"]:
        raise ValueError("Per-era CSV hash does not match the evaluator receipt.")

    summary = pd.read_csv(summary_path)
    per_era = pd.read_csv(per_era_path, dtype={"era": str})
    if set(per_era["phase"]) != {"scout_calibration"}:
        raise ValueError("Reader output unexpectedly contains a locked phase.")
    if set(per_era["candidate"]) != set(EXPECTED_CANDIDATES):
        raise ValueError("Reader output has an unexpected scout candidate set.")
    if per_era.duplicated(["phase", "candidate", "era"]).any():
        raise ValueError("Reader output contains duplicate candidate-era rows.")
    if not np.isfinite(
        per_era[["corr", "bmc", "benchmark_similarity"]].to_numpy()
    ).all():
        raise ValueError("Reader output contains non-finite metrics.")

    expected_eras: list[str] | None = None
    for candidate in EXPECTED_CANDIDATES:
        eras = per_era.loc[per_era["candidate"] == candidate, "era"].tolist()
        if len(eras) != 164 or eras[0] != "0373" or eras[-1] != "1025":
            raise ValueError(f"Unexpected calibration coverage for {candidate}.")
        if expected_eras is None:
            expected_eras = eras
        elif eras != expected_eras:
            raise ValueError("Scout candidates do not share exact calibration eras.")

    recomputed = _metric_summary(per_era)
    comparable_columns = [
        "era_count",
        "corr_mean",
        "corr_std",
        "corr_sharpe",
        "bmc_mean",
        "bmc_std",
        "bmc_sharpe",
        "bmc_max_drawdown",
        "avg_benchmark_similarity",
    ]
    reported = summary.set_index("candidate").loc[list(EXPECTED_CANDIDATES)]
    independently_computed = recomputed.set_index("candidate").loc[
        list(EXPECTED_CANDIDATES)
    ]
    if not np.allclose(
        reported[comparable_columns].to_numpy(dtype=float),
        independently_computed[comparable_columns].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Summary CSV does not reconcile to the per-era calibration rows.")

    return result, summary, per_era


def _markdown_cell(source: str, cell_id: str) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_markdown_cell(source)
    cell["id"] = cell_id
    return cell


def _code_cell(source: str, cell_id: str) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_code_cell(source)
    cell["id"] = cell_id
    return cell


def build_notebook(result_json: Path, output_path: Path) -> nbformat.NotebookNode:
    result, summary, _ = _load_and_verify_frozen_outputs(result_json)
    output_path = output_path.resolve()
    result_literal = _notebook_path_literal(result_json, output_path.parent)
    best = summary.sort_values("bmc_mean", ascending=False).iloc[0]

    cells = [
        _markdown_cell(
            f"""# Xerxes20 LightGBM challenger analysis

## tl;dr

Frozen-gate result: **`{result['state']}`**. None of the four direct
`target_xerxes_20` LightGBM scouts cleared calibration eligibility. Every scout
missed the strict BMC Sharpe `> 0.20` requirement; `r1_trees2k` also missed the
BMC mean `> 0.0010` requirement. The highest calibration BMC mean was
`{best['bmc_mean']:.6f}` from `{best['candidate']}`, with BMC Sharpe
`{best['bmc_sharpe']:.4f}`.

Per the frozen stop rule, the locked final 50 scout eras were not scored, no
confirmation was run, and no model was packaged or uploaded.
""",
            "tldr",
        ),
        _markdown_cell(
            """## Context & Methods

This experiment tested four fixed LightGBM capacity profiles trained directly
on `target_xerxes_20` with Numerai's 780 medium features. Selection was based on
`target_ender_20` performance against `v53_lgbm_ender20`. The scout produced
1,279,658 OOF prediction rows across 214 retained eras, but the predeclared
selection decision used only the first 164 eras (`0373`-`1025`).

### Key Assumptions

- `gate.md`, the four configs, and their source/runtime receipts were frozen
  before scoring.
- The evaluator's final JSON and calibration CSVs are the authoritative outputs.
- This notebook verifies those content hashes and independently reconciles the
  saved calibration summaries; it does not retrain or rescore predictions.
- The prediction parquets and per-run result JSONs are intentionally not opened
  here because no scout earned access to the locked 50-era slice.
""",
            "context-methods",
        ),
        _markdown_cell("## Data & Integrity", "data-integrity"),
        _code_cell(
            f"""from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

RESULT_JSON = Path({result_literal}).resolve()
RESULTS_DIR = RESULT_JSON.parent
EXPERIMENT_DIR = RESULT_JSON.parent.parent
REPO_ROOT = EXPERIMENT_DIR.parents[3]
EXPECTED_RESULT_SHA256 = {EXPECTED_RESULT_SHA256!r}
EXPECTED_GENERATION_ID = {EXPECTED_GENERATION_ID!r}
EXPECTED_CANDIDATES = {EXPECTED_CANDIDATES!r}

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

assert sys.version_info[:2] == (3, 12), sys.version
assert sha256_file(RESULT_JSON) == EXPECTED_RESULT_SHA256
result = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
assert result["generation_id"] == EXPECTED_GENERATION_ID
assert result["state"] == "STOP_NO_SCOUT_CALIBRATION_WINNER"

content_result = RESULTS_DIR / f"xerxes20_result-{{EXPECTED_GENERATION_ID}}.json"
assert content_result.read_bytes() == RESULT_JSON.read_bytes()
print(f"Python {{sys.version.split()[0]}} | evaluator generation {{EXPECTED_GENERATION_ID}}")
print(f"Canonical result SHA-256: {{EXPECTED_RESULT_SHA256}}")
""",
            "load-result",
        ),
        _code_cell(
            """summary_path = RESULTS_DIR / Path(result["outputs"]["summary_csv"]).name
per_era_path = RESULTS_DIR / Path(result["outputs"]["per_era_csv"]).name

verified_files = [
    ("canonical result", RESULT_JSON, EXPECTED_RESULT_SHA256),
    ("content result", content_result, EXPECTED_RESULT_SHA256),
    ("summary CSV", summary_path, result["outputs"]["summary_csv_sha256"]),
    ("per-era CSV", per_era_path, result["outputs"]["per_era_csv_sha256"]),
]
for label, relative_path, expected_hash in [
    ("GPU runtime", result["inputs"]["gpu_runtime"]["path"], result["inputs"]["gpu_runtime"]["runtime_receipt_sha256"]),
    ("source manifest", result["inputs"]["source_manifest"]["path"], result["inputs"]["source_manifest"]["sha256"]),
    ("evaluator", result["evaluator"]["path"], result["evaluator"]["sha256"]),
]:
    verified_files.append((label, REPO_ROOT / relative_path, expected_hash))
for candidate in EXPECTED_CANDIDATES:
    verified_files.append((
        f"config {candidate}",
        EXPERIMENT_DIR / "configs" / f"{candidate}.py",
        result["inputs"]["scout_runs"][candidate]["config_sha256"],
    ))

verification_rows = []
for label, path, expected_hash in verified_files:
    actual_hash = sha256_file(path)
    assert actual_hash == expected_hash, label
    verification_rows.append({"artifact": label, "sha256": actual_hash})

print(pd.DataFrame(verification_rows).to_string(index=False))
print("\\nPrediction and per-run result files: evaluator receipts retained; files not opened.")
""",
            "verify-hashes",
        ),
        _markdown_cell("## Results", "results"),
        _code_cell(
            """summary = pd.read_csv(summary_path)
per_era = pd.read_csv(per_era_path, dtype={"era": str})

assert set(per_era["phase"]) == {"scout_calibration"}
assert set(per_era["candidate"]) == set(EXPECTED_CANDIDATES)
assert not per_era.duplicated(["phase", "candidate", "era"]).any()
assert np.isfinite(per_era[["corr", "bmc", "benchmark_similarity"]].to_numpy()).all()
assert len(per_era) == 4 * 164

reference_eras = None
recomputed_rows = []
for candidate in EXPECTED_CANDIDATES:
    candidate_rows = per_era.loc[per_era["candidate"] == candidate].sort_values("era")
    eras = candidate_rows["era"].tolist()
    assert len(eras) == 164 and eras[0] == "0373" and eras[-1] == "1025"
    if reference_eras is None:
        reference_eras = eras
    else:
        assert eras == reference_eras
    bmc = candidate_rows["bmc"]
    corr = candidate_rows["corr"]
    cumulative_bmc = bmc.cumsum()
    recomputed_rows.append({
        "candidate": candidate,
        "era_count": len(candidate_rows),
        "corr_mean": corr.mean(),
        "corr_std": corr.std(ddof=0),
        "corr_sharpe": corr.mean() / corr.std(ddof=0),
        "bmc_mean": bmc.mean(),
        "bmc_std": bmc.std(ddof=0),
        "bmc_sharpe": bmc.mean() / bmc.std(ddof=0),
        "bmc_max_drawdown": (cumulative_bmc.cummax() - cumulative_bmc).max(),
        "avg_benchmark_similarity": candidate_rows["benchmark_similarity"].mean(),
    })

recomputed = pd.DataFrame(recomputed_rows).set_index("candidate")
reported = summary.set_index("candidate").loc[list(EXPECTED_CANDIDATES)]
metric_columns = [
    "era_count", "corr_mean", "corr_std", "corr_sharpe", "bmc_mean",
    "bmc_std", "bmc_sharpe", "bmc_max_drawdown", "avg_benchmark_similarity",
]
assert np.allclose(
    reported[metric_columns].to_numpy(dtype=float),
    recomputed[metric_columns].to_numpy(dtype=float),
    rtol=0.0,
    atol=1e-12,
)
print("656 calibration rows reconcile to the evaluator summary within 1e-12.")
""",
            "reconcile-calibration",
        ),
        _code_cell(
            """thresholds = result["thresholds"]["scout_calibration"]
decision_rows = []
for candidate in EXPECTED_CANDIDATES:
    metrics = reported.loc[candidate]
    independent_checks = {
        "bmc_mean": metrics["bmc_mean"] > thresholds["bmc_mean_min_exclusive"],
        "bmc_sharpe": metrics["bmc_sharpe"] > thresholds["bmc_sharpe_min_exclusive"],
        "bmc_max_drawdown": metrics["bmc_max_drawdown"] < thresholds["bmc_max_drawdown_max_exclusive"],
        "corr_mean": metrics["corr_mean"] > thresholds["corr_mean_min_exclusive"],
        "benchmark_similarity": metrics["avg_benchmark_similarity"] < thresholds["benchmark_similarity_max_exclusive"],
    }
    recorded = result["scout_calibration_candidates"][candidate]
    assert independent_checks == recorded["checks"]
    assert recorded["eligible"] == all(independent_checks.values())
    failed = [name for name, passed in independent_checks.items() if not passed]
    decision_rows.append({
        "candidate": candidate,
        "BMC mean": metrics["bmc_mean"],
        "BMC Sharpe": metrics["bmc_sharpe"],
        "BMC max DD": metrics["bmc_max_drawdown"],
        "Ender Corr": metrics["corr_mean"],
        "benchmark Spearman": metrics["avg_benchmark_similarity"],
        "failed checks": ", ".join(failed),
    })

decision_table = pd.DataFrame(decision_rows)
numeric_columns = ["BMC mean", "BMC Sharpe", "BMC max DD", "Ender Corr", "benchmark Spearman"]
decision_table[numeric_columns] = decision_table[numeric_columns].round(6)
print(decision_table.to_string(index=False))

assert not any(item["eligible"] for item in result["scout_calibration_candidates"].values())
assert result["selected_scout"] is None
assert result["scout_holdout_checks"] == {}
assert result["confirmation_checks"] == {}
assert result["offline_gate_passed"] is False
assert result["promotion_eligible"] is False
print(f"\\nFinal decision: {result['state']}")
""",
            "verify-decision",
        ),
        _markdown_cell(
            """## Takeaways

`r1_depth8` had the strongest calibration BMC mean and Ender Corr, but its BMC
Sharpe was `0.1751`, below the frozen strict `> 0.20` gate. The other three
scouts also failed BMC Sharpe; the 2,000-tree scout additionally failed minimum
BMC mean. Capacity changed signal strength, but none produced sufficiently
stable unique signal under the predeclared screen.

The correct decision is **not promotion eligible**. The experiment stops at
calibration without opening the locked 50-era metrics, running a consecutive
confirmation, packaging a pickle, uploading a model, submitting predictions,
or staking.
""",
            "takeaways",
        ),
    ]

    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3.12 (numerai-lgbm-gpu312)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12.13"},
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-json",
        type=Path,
        default=RESULTS_DIR / "xerxes20_result.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_DIR / "xerxes20_lgbm_challenger_analysis.ipynb",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result_json = args.result_json.resolve()
    output_path = args.output.resolve()
    notebook = build_notebook(result_json, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.execute:
        if os.name == "nt":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(output_path.parent)}},
            record_timing=False,
        )
        notebook = client.execute()
    nbformat.write(notebook, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
