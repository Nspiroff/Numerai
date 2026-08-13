"""Build and optionally execute the two-seed stability reader notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import nbformat
import pandas as pd
from nbclient import NotebookClient


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise ValueError(f"Could not find repository root above {start}")


def _notebook_path_literal(source: Path, notebook_dir: Path) -> str:
    return repr(Path(os.path.relpath(source.resolve(), notebook_dir.resolve())).as_posix())


def _takeaway_markdown(result: dict) -> str:
    selected = result.get("selected_candidate")
    if selected:
        headline = (
            f"The frozen gate selected `{selected}`. Local packaging would still require "
            "a separate runtime-validation step and would not authorize an upload."
        )
    else:
        headline = (
            "No frozen blend cleared calibration eligibility. Per the predeclared stop "
            "rule, this benchmark-blend line ends without another weight or seed search."
        )
    return f"""## Takeaways

{headline}

The two-seed residual was more stable than either seed alone, and the 35% blend
looked favorable on the locked holdout. Neither observation can override the
calibration gate: the 35% blend retained 38.47% of residual BMC against a 40%
minimum, while the 45% blend retained 85.12% of benchmark target Corr against a
90% minimum. The existing single-seed pickle remains experimental and should
not be uploaded as a production model.

The final 200 eras are formula-new but not fully data-new because prior research
already summarized recent performance for the residual reference. This is a
robustness check, not live validation, and offline metrics cannot prove future
returns or Numerai-hosted runtime compatibility.
"""


def build_notebook(result_json: Path, output_path: Path) -> nbformat.NotebookNode:
    result = json.loads(result_json.read_text(encoding="utf-8"))
    if result.get("experiment") != "ender20_seed_ensemble_stability_v53":
        raise ValueError("Result JSON is not for the two-seed stability experiment.")

    repo_root = _repo_root(result_json.resolve())
    outputs = result.get("outputs", {})
    summary_csv = repo_root / outputs["summary_csv"]
    per_era_csv = repo_root / outputs["per_era_csv"]
    expected_summary_hash = outputs["summary_csv_sha256"]
    expected_per_era_hash = outputs["per_era_csv_sha256"]
    for path, expected_hash in (
        (summary_csv, expected_summary_hash),
        (per_era_csv, expected_per_era_hash),
    ):
        if _sha256(path) != expected_hash:
            raise ValueError(f"Output hash mismatch: {path}")

    summary = pd.read_csv(summary_csv)
    required_summary = {
        "segment",
        "candidate",
        "bmc_mean",
        "bmc_sharpe",
        "bmc_max_drawdown",
        "corr_mean",
        "avg_corr_with_benchmark",
    }
    missing = sorted(required_summary - set(summary.columns))
    if missing:
        raise ValueError(f"Summary CSV is missing columns: {missing}")

    output_path = output_path.resolve()
    notebook_dir = output_path.parent
    result_literal = _notebook_path_literal(result_json, notebook_dir)
    summary_literal = _notebook_path_literal(summary_csv, notebook_dir)
    per_era_literal = _notebook_path_literal(per_era_csv, notebook_dir)
    result_hash = _sha256(result_json)
    state = result.get("state", "UNKNOWN")
    selected = result.get("selected_candidate") or "none"

    cells = [
        nbformat.v4.new_markdown_cell(
            f"""# Ender20 two-seed stability analysis

## tl;dr

Frozen-gate result: **`{state}`**. Calibration selection: **`{selected}`**.
The equal-rank seed ensemble improved stability, but no predeclared benchmark
blend met every calibration requirement. No model was packaged or uploaded.
"""
        ),
        nbformat.v4.new_code_cell(
            f"""from pathlib import Path
import hashlib
import json
import pandas as pd
import matplotlib.pyplot as plt

RESULT_JSON = Path({result_literal})
SUMMARY_CSV = Path({summary_literal})
PER_ERA_CSV = Path({per_era_literal})
EXPECTED_RESULT_SHA256 = {result_hash!r}
EXPECTED_SUMMARY_SHA256 = {expected_summary_hash!r}
EXPECTED_PER_ERA_SHA256 = {expected_per_era_hash!r}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

assert sha256(RESULT_JSON) == EXPECTED_RESULT_SHA256
assert sha256(SUMMARY_CSV) == EXPECTED_SUMMARY_SHA256
assert sha256(PER_ERA_CSV) == EXPECTED_PER_ERA_SHA256

result = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
summary = pd.read_csv(SUMMARY_CSV)
per_era = pd.read_csv(PER_ERA_CSV, dtype={{"era": str}})

assert result["cohort"]["rows"] == 5_112_039
assert result["cohort"]["eras"] == 855
assert per_era["era"].nunique() == 855
print(result["state"], "| selected:", result["selected_candidate"])
print("generation:", result["generation_id"], "| output hashes verified")
"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Context & Methods

The original K64 TabM residual retained unique Ender20 signal but failed its
production stability gate. A first benchmark-blend experiment narrowly missed
its frozen Sharpe threshold. This final iteration trained only the predeclared
second model seed, ranked both seeds within each era, averaged them equally,
reranked the ensemble, and tested the same five benchmark weights.

The first 655 chronological OOF eras are calibration. The final 200 eras are a
locked holdout. Candidate selection uses calibration only; because none passed,
the holdout is context and cannot choose a substitute winner.

### Key Assumptions

- Both OOF artifacts are exact model predictions for the independently derived
  ID, era, target, and fold cohort and use the same residual-target semantics.
- The historical source manifest is only the immutable feature-store data
  anchor; current training code is anchored to the committed pre-training Git
  checkpoint recorded in the result receipt.
- Per-era Corr, BMC, benchmark similarity, Sharpe, and additive maximum drawdown
  are useful offline diagnostics, not guarantees of live performance.
- Seeds, blend weights, thresholds, split, and stop rule in `gate.md` were
  frozen before the second-seed model was trained or scored.
"""
        ),
        nbformat.v4.new_markdown_cell("## Data"),
        nbformat.v4.new_code_cell(
            """direct_inputs = []
for name, details in result["inputs"].items():
    if isinstance(details, dict) and "path" in details and "sha256" in details:
        direct_inputs.append({"input": name, "path": details["path"], "sha256": details["sha256"]})

display(pd.Series(result["cohort"], name="cohort"))
display(pd.DataFrame(direct_inputs))
display(pd.Series(result["inputs"]["frozen_source"], name="source anchors"))
"""
        ),
        nbformat.v4.new_markdown_cell("## Results"),
        nbformat.v4.new_code_cell(
            """columns = [
    "segment", "candidate", "bmc_mean", "bmc_sharpe",
    "bmc_max_drawdown", "corr_mean", "avg_corr_with_benchmark",
]
candidate_table = summary.loc[
    summary["candidate"].str.startswith("two_seed_hybrid_"), columns
].copy()
display(candidate_table.round(6).reset_index(drop=True))
"""
        ),
        nbformat.v4.new_code_cell(
            """plot_data = candidate_table.pivot(
    index="candidate", columns="segment", values=["bmc_mean", "bmc_max_drawdown"]
)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_data["bmc_mean"][["calibration", "holdout", "full"]].plot.bar(
    ax=axes[0], title="BMC mean by frozen segment"
)
axes[0].axhline(0.0, color="black", linewidth=0.8)
axes[0].set_ylabel("mean BMC")
plot_data["bmc_max_drawdown"][["calibration", "holdout", "full"]].plot.bar(
    ax=axes[1], title="BMC maximum drawdown"
)
axes[1].axhline(0.15, color="red", linestyle="--", label="gate: < 0.15")
axes[1].set_ylabel("additive drawdown")
axes[1].legend()
for axis in axes:
    axis.tick_params(axis="x", rotation=35)
fig.tight_layout()
plt.show()
"""
        ),
        nbformat.v4.new_code_cell(
            """calibration = pd.DataFrame(result["calibration_candidates"]).T
calibration["failed_checks"] = calibration["checks"].map(
    lambda checks: ", ".join(name for name, passed in checks.items() if not passed) or "none"
)
display(calibration[["eligible", "failed_checks"]])

cal_summary = summary.loc[summary["segment"].eq("calibration")].set_index("candidate")
residual_bmc = cal_summary.loc["two_seed_residual", "bmc_mean"]
benchmark_corr = cal_summary.loc["benchmark_only", "corr_mean"]
near_misses = pd.DataFrame(
    {
        "observed_retention": [
            cal_summary.loc["two_seed_hybrid_w35", "bmc_mean"] / residual_bmc,
            cal_summary.loc["two_seed_hybrid_w45", "corr_mean"] / benchmark_corr,
        ],
        "required_retention": [0.40, 0.90],
    },
    index=["w35 residual BMC", "w45 benchmark Corr"],
)
display(near_misses.style.format("{:.2%}"))
print("Final state:", result["state"], "| promotion checks:", result["promotion_checks"])
"""
        ),
        nbformat.v4.new_markdown_cell(_takeaway_markdown(result)),
    ]
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-json",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "two_seed_stability_result.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_DIR / "ender20_seed_ensemble_stability_analysis.ipynb",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    notebook = build_notebook(args.result_json.resolve(), args.output.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.execute:
        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(args.output.parent.resolve())}},
        )
        notebook = client.execute()
    nbformat.write(notebook, args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
