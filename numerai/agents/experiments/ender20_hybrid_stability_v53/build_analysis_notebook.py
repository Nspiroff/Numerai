"""Build and optionally execute the hybrid-stability reader notebook."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import nbformat
import pandas as pd
from nbclient import NotebookClient


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results"


def _notebook_path_literal(source: Path, notebook_dir: Path) -> str:
    return repr(Path(os.path.relpath(source.resolve(), notebook_dir.resolve())).as_posix())


def _takeaway_markdown(result: dict) -> str:
    selected = result.get("selected_candidate")
    eligible = bool(result.get("promotion_eligible"))
    if eligible:
        headline = (
            f"The frozen gate passed for `{selected}`. This permits local packaging and "
            "runtime validation, but it does not authorize a Numerai upload or staking."
        )
    elif selected:
        failed = [
            name for name, passed in result.get("promotion_checks", {}).items() if not passed
        ]
        headline = (
            f"Calibration selected `{selected}`, but the promotion gate failed on: "
            f"{', '.join(f'`{name}`' for name in failed)}. No upload candidate should be produced."
        )
    else:
        headline = (
            "No frozen weight cleared calibration eligibility, so the experiment stops "
            "without inspecting holdout weights for a replacement winner."
        )
    return f"""## Takeaways

{headline}

The final 200 eras are formula-new but not fully data-new: prior research already
reported aggregate recent performance for the residual reference. These results
are therefore a robustness check, not a substitute for live validation. Offline
metrics also cannot prove Numerai-hosted Docker compatibility or future returns.
"""


def build_notebook(
    result_json: Path,
    summary_csv: Path,
    per_era_csv: Path,
    output_path: Path,
) -> nbformat.NotebookNode:
    result = json.loads(result_json.read_text(encoding="utf-8"))
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
    if result.get("experiment") != "ender20_hybrid_stability_v53":
        raise ValueError("Result JSON is not for the hybrid stability experiment.")

    output_path = output_path.resolve()
    notebook_dir = output_path.parent
    result_literal = _notebook_path_literal(result_json, notebook_dir)
    summary_literal = _notebook_path_literal(summary_csv, notebook_dir)
    per_era_literal = _notebook_path_literal(per_era_csv, notebook_dir)
    state = result.get("state", "UNKNOWN")
    selected = result.get("selected_candidate") or "none"

    cells = [
        nbformat.v4.new_markdown_cell(
            f"""# Ender20 hybrid stability analysis

## tl;dr

Frozen-gate result: **`{state}`**. Calibration selection: **`{selected}`**.
The analysis uses 5,112,039 out-of-fold rows across 855 consecutive eras and
does not retrain or upload a model.
"""
        ),
        nbformat.v4.new_code_cell(
            f"""from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

RESULT_JSON = Path({result_literal})
SUMMARY_CSV = Path({summary_literal})
PER_ERA_CSV = Path({per_era_literal})

result = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
summary = pd.read_csv(SUMMARY_CSV)
per_era = pd.read_csv(PER_ERA_CSV, dtype={{"era": str}})

assert result["cohort"]["rows"] == 5_112_039
assert result["cohort"]["eras"] == 855
assert len(per_era["era"].unique()) == 855
print(result["state"], "| selected:", result["selected_candidate"])
"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Context & Methods

The prior TabM residual retained positive unique Ender20 signal but failed its
production gate because full-period BMC maximum drawdown was about 0.301. This
experiment combines that fixed residual with Numerai's official
`v53_lgbm_ender20` benchmark. Both inputs are percentile-ranked within era,
combined at five weights frozen before scoring, and percentile-ranked again.

The first 655 chronological OOF eras are calibration. The final 200 eras are a
locked holdout. Candidate selection uses calibration only; the holdout can
accept or reject that one selection but cannot substitute another weight.

### Key Assumptions

- The OOF parquet's prediction-semantics metadata correctly identifies a raw
  benchmark-residual model output.
- Manifest IDs, eras, targets, and official benchmark predictions are the exact
  generation used by the earlier validated feature store.
- Per-era Corr, BMC, benchmark similarity, Sharpe, and additive maximum drawdown
  are suitable offline diagnostics, not guarantees of live performance.
- Thresholds and blend weights in `gate.md` were frozen before any hybrid result
  was computed.
"""
        ),
        nbformat.v4.new_markdown_cell("## Data"),
        nbformat.v4.new_code_cell(
            """provenance = pd.DataFrame([
    {
        "input": name,
        "path": details["path"],
        "sha256": details["sha256"],
    }
    for name, details in result["inputs"].items()
    if isinstance(details, dict) and "sha256" in details
])
display(pd.Series(result["cohort"], name="cohort"))
display(provenance)
"""
        ),
        nbformat.v4.new_markdown_cell("## Results"),
        nbformat.v4.new_code_cell(
            """columns = [
    "segment", "candidate", "bmc_mean", "bmc_sharpe",
    "bmc_max_drawdown", "corr_mean", "avg_corr_with_benchmark",
]
candidate_table = summary.loc[
    summary["candidate"].str.startswith("hybrid_"), columns
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

promotion = pd.Series(result["promotion_checks"], name="passed")
display(promotion.to_frame())
print("Final state:", result["state"])
"""
        ),
        nbformat.v4.new_markdown_cell(_takeaway_markdown(result)),
    ]
    notebook = nbformat.v4.new_notebook(
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
    return notebook


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-json",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "hybrid_stability_result.json",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "hybrid_stability_summary.csv",
    )
    parser.add_argument(
        "--per-era-csv",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "hybrid_stability_per_era.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_DIR / "ender20_hybrid_stability_analysis.ipynb",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    notebook = build_notebook(
        args.result_json.resolve(),
        args.summary_csv.resolve(),
        args.per_era_csv.resolve(),
        args.output.resolve(),
    )
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
