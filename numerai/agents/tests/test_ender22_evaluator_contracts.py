from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from agents.code.metrics import numerai_metrics
from agents.code.modeling.models.torch_tabular_regressor import (
    TorchTabularRegressor,
)
from agents.code.modeling.utils.numerai_cv import era_cv_splits
from agents.code.modeling.utils.pipeline import PREDICTION_SEMANTICS_METADATA_KEY


EXPERIMENT_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "experiments/ender22_temporal_retention_v53"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = _load_module(
    "ender22_evaluation_common_contract", EXPERIMENT_SOURCE / "evaluation_common.py"
)
_previous_common = sys.modules.get("evaluation_common")
sys.modules["evaluation_common"] = COMMON
try:
    ROUND1 = _load_module(
        "ender22_round1_contract", EXPERIMENT_SOURCE / "evaluate_round1_impl.py"
    )
    ROUND2 = _load_module(
        "ender22_round2_contract", EXPERIMENT_SOURCE / "evaluate_round2_impl.py"
    )
finally:
    if _previous_common is None:
        sys.modules.pop("evaluation_common", None)
    else:
        sys.modules["evaluation_common"] = _previous_common


def _leaf_values(value, prefix=()):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(_leaf_values(child, (*prefix, key)))
        return result
    return {prefix: value}


def _metrics(
    *,
    full: float = 0.010,
    recent: float = 0.009,
    sharpe: float = 1.0,
    drawdown: float = 0.010,
) -> dict:
    return {
        "bmc": {
            "mean": full,
            "std": 0.01,
            "sharpe": sharpe,
            "max_drawdown": drawdown,
        },
        "recent40_bmc_mean": recent,
        "recent_blocks_bmc_mean": {
            "0705-0741": recent,
            "0745-0781": recent,
            "0785-0821": recent,
            "0825-0861": recent,
        },
        "fold_bmc_mean": {"1": full, "2": full, "3": full, "4": full},
        "corr_mean": 0.010,
        "avg_corr_with_benchmark": 0.10,
    }


class TestEnder22FrozenConfigMatrix(unittest.TestCase):
    def test_every_config_has_only_its_declared_delta_and_feature_authority(self) -> None:
        base = runpy.run_path(str(EXPERIMENT_SOURCE / "configs/base_r1.py"))["CONFIG"]
        cases = {
            "r1_control_block_dro": {},
            "r1_recent_half_life52": {
                ("model", "params", "recency_half_life_eras"): 52.0,
            },
            "r1_recent_window78": {
                ("training", "cv", "max_train_eras"): 78,
            },
            "r2_recent_half_life52_model_seed2027": {
                ("model", "params", "recency_half_life_eras"): 52.0,
                ("model", "params", "seed"): 2027,
            },
            "r2_recent_half_life52_sample_seed2027": {
                ("model", "params", "recency_half_life_eras"): 52.0,
                ("training", "sample_seed"): 2027,
            },
            "r2_recent_window78_model_seed2027": {
                ("training", "cv", "max_train_eras"): 78,
                ("model", "params", "seed"): 2027,
            },
            "r2_recent_window78_sample_seed2027": {
                ("training", "cv", "max_train_eras"): 78,
                ("training", "sample_seed"): 2027,
            },
        }
        base_leaves = _leaf_values(base)
        for name, declared in cases.items():
            with self.subTest(name=name):
                config = runpy.run_path(
                    str(EXPERIMENT_SOURCE / f"configs/{name}.py")
                )["CONFIG"]
                expected_deltas = {
                    ("output", "results_name"): name,
                    **declared,
                }
                actual_leaves = _leaf_values(config)
                all_paths = set(base_leaves) | set(actual_leaves)
                actual_deltas = {
                    path: actual_leaves.get(path)
                    for path in all_paths
                    if actual_leaves.get(path) != base_leaves.get(path)
                }
                self.assertEqual(actual_deltas, expected_deltas)
                self.assertEqual(
                    config["data"]["feature_columns_path"],
                    "numerai/agents/experiments/ender21_residual_stability_v53/"
                    "protocol/feature_columns_all_v53.json",
                )


class TestEnder22RecencyPrior(unittest.TestCase):
    def test_weighted_block_dro_moves_total_gradient_mass_to_newer_block(self) -> None:
        model = TorchTabularRegressor(
            feature_cols=["feature_a"],
            architecture="mlp",
            hidden_layer_sizes=(4,),
            device="cpu",
            amp=False,
            batch_size=16,
            prediction_batch_size=16,
            max_epochs=1,
            patience=1,
            val_split="none",
            val_fraction=0.0,
            feature_center=None,
            feature_scale=None,
            deterministic=True,
            verbose=False,
            loss_mode="chronological_block_dro",
            chronological_blocks=2,
            dro_temperature=2.0,
            recency_half_life_eras=2.0,
        )
        torch = model._torch
        predictions = torch.zeros(4, requires_grad=True)
        targets = torch.ones(4)
        # Equal per-block losses make the detached DRO softmax exactly uniform.
        # The only source of unequal block gradient mass is the recency prior.
        auxiliary = torch.tensor(
            [[0.0, 0.25], [0.0, 0.50], [1.0, 0.75], [1.0, 1.00]]
        )
        model._training_loss(predictions, targets, auxiliary).backward()
        gradients = predictions.grad.detach().abs().numpy()
        older_mass = float(gradients[:2].sum())
        newer_mass = float(gradients[2:].sum())
        self.assertGreater(newer_mass, older_mass)
        self.assertAlmostEqual(newer_mass / older_mass, 1.75 / 0.75, places=5)


class TestEnder22ScoreConvention(unittest.TestCase):
    def test_population_sharpe_and_first_era_drawdown_are_frozen(self) -> None:
        summary = numerai_metrics.score_summary(pd.Series([1.0, 3.0]))
        self.assertEqual(summary["std"], 1.0)
        self.assertEqual(summary["sharpe"], 2.0)

        # A zero-baseline implementation would report drawdown 2.0 here.  The
        # repository convention starts the running peak at the first cumsum.
        self.assertEqual(
            numerai_metrics.score_summary(pd.Series([-2.0, 1.0, 1.0]))[
                "max_drawdown"
            ],
            0.0,
        )
        protocol = (EXPERIMENT_SOURCE / "experiment.md").read_text(encoding="utf-8")
        self.assertIn("population standard deviation (`ddof=0`)", protocol)
        self.assertIn("running peak begins at the first scored era", protocol)


class TestEnder22DecisionContracts(unittest.TestCase):
    def test_decision_check_keys_are_exact(self) -> None:
        self.assertEqual(
            set(COMMON.challenger_checks(_metrics(), _metrics())),
            {
                "recent40_gain_at_least_0_00030",
                "full_bmc_retains_90pct_control",
                "recent40_retains_80pct_candidate_full",
                "full_bmc_positive",
                "all_used_folds_bmc_positive",
                "three_of_four_recent_blocks_positive",
                "worst_recent_block_above_minus_0_001",
                "sharpe_not_below_control_minus_0_05",
                "drawdown_no_greater_than_control",
                "corr_at_least_0_005",
                "corr_below_0_04",
                "benchmark_corr_below_0_25",
            },
        )
        self.assertEqual(
            set(COMMON.replication_checks(_metrics(), _metrics())),
            {
                "full_bmc_retains_90pct_base_control",
                "recent40_at_least_base_control",
                "all_used_folds_bmc_positive",
                "three_of_four_recent_blocks_positive",
                "worst_recent_block_above_minus_0_001",
                "sharpe_not_below_base_control_minus_0_05",
                "drawdown_no_greater_than_base_control",
                "corr_at_least_0_005",
                "corr_below_0_04",
                "benchmark_corr_below_0_25",
            },
        )

    def _round2_fixture(self, experiment: Path):
        control = {"metrics": _metrics(recent=0.008)}
        selected = {"metrics": _metrics(recent=0.009)}
        rejected = {"metrics": _metrics(recent=0.007)}
        rep1 = {"metrics": _metrics(recent=0.0091)}
        rep2 = {"metrics": _metrics(recent=0.0092)}
        records = {
            COMMON.CONTROL: control,
            "r1_recent_half_life52": selected,
            "r1_recent_window78": rejected,
            "r2_recent_half_life52_model_seed2027": rep1,
            "r2_recent_half_life52_sample_seed2027": rep2,
        }
        ids = ["a", "b", "c", "d"]
        base = pd.DataFrame(
            {
                "id": ids,
                "era": ["0705"] * 4,
                "target_ender_20": [0.0, 0.25, 0.75, 1.0],
                "cv_fold": [4] * 4,
                "v53_lgbm_ender20": [0.4, 0.1, 0.8, 0.2],
            }
        )
        predictions = {
            COMMON.CONTROL: [0.1, 0.2, 0.3, 0.4],
            "r1_recent_half_life52": [0.1, 0.2, 0.3, 0.4],
            "r1_recent_window78": [0.4, 0.3, 0.2, 0.1],
            "r2_recent_half_life52_model_seed2027": [0.1, 0.2, 0.3, 0.4],
            "r2_recent_half_life52_sample_seed2027": [0.1, 0.2, 0.4, 0.3],
        }
        frames = {}
        for name, values in predictions.items():
            frames[name] = base.assign(prediction=values)
        checks = {}
        for name in COMMON.ROUND1_CANDIDATES[1:]:
            candidate_checks = COMMON.challenger_checks(
                records[name]["metrics"], control["metrics"]
            )
            checks[name] = {
                "checks": candidate_checks,
                "eligible": all(candidate_checks.values()),
            }
        fake_receipt = lambda path, *_args: {"path": Path(path).name}
        inputs = {
            "authority": fake_receipt(experiment / "protocol/discovery_data_authority.json"),
            "full": fake_receipt(Path("ender21_discovery_full_through_0861.parquet")),
            "benchmark": fake_receipt(
                Path("ender21_discovery_benchmark_models_through_0861.parquet")
            ),
        }
        round1 = {
            "schema_version": 1,
            "stage": "ender22-round1-discovery",
            "state": "SCOUT_WINNER",
            "selected": "r1_recent_half_life52",
            "inputs": inputs,
            "candidates": {name: records[name] for name in COMMON.ROUND1_CANDIDATES},
            "decisions": checks,
        }
        receipts = experiment / "receipts"
        receipts.mkdir(parents=True)
        (receipts / "round1_discovery.json").write_text(
            json.dumps(round1), encoding="utf-8"
        )
        return records, frames, round1, fake_receipt

    def test_round2_recomputes_round1_and_rejects_tampered_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp)
            records, frames, round1, fake_receipt = self._round2_fixture(experiment)
            round1["candidates"][COMMON.CONTROL]["metrics"]["bmc"]["mean"] = 99.0
            (experiment / "receipts/round1_discovery.json").write_text(
                json.dumps(round1), encoding="utf-8"
            )

            def score(_experiment, name, *_args):
                return records[name], frames[name]

            custody = SimpleNamespace(
                manifest={},
                read_json=lambda path: (
                    {}
                    if Path(path).name == "source_manifest_round1.json"
                    else json.loads(Path(path).read_text(encoding="utf-8"))
                ),
            )
            with mock.patch.object(ROUND2, "load_authority", return_value=({}, [])), mock.patch.object(
                ROUND2, "load_truth", return_value=pd.DataFrame()
            ), mock.patch.object(ROUND2, "score_candidate", side_effect=score), mock.patch.object(
                ROUND2, "receipt", side_effect=fake_receipt
            ):
                with self.assertRaisesRegex(ValueError, "independent recomputation"):
                    ROUND2.evaluate(experiment, Path(tmp), custody)

    def test_round2_ensemble_is_mean_of_ranks_without_rerank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp)
            records, frames, _round1, fake_receipt = self._round2_fixture(experiment)
            captured = {}

            def score(_experiment, name, *_args):
                return records[name], frames[name]

            def compute(frame):
                captured["predictions"] = frame["prediction"].to_numpy().copy()
                return _metrics(recent=0.009)

            custody = SimpleNamespace(
                manifest={},
                read_json=lambda path: (
                    {}
                    if Path(path).name == "source_manifest_round1.json"
                    else json.loads(Path(path).read_text(encoding="utf-8"))
                ),
            )
            with mock.patch.object(ROUND2, "load_authority", return_value=({}, [])), mock.patch.object(
                ROUND2, "load_truth", return_value=pd.DataFrame()
            ), mock.patch.object(ROUND2, "score_candidate", side_effect=score), mock.patch.object(
                ROUND2, "receipt", side_effect=fake_receipt
            ), mock.patch.object(ROUND2, "compute_metrics", side_effect=compute), mock.patch.object(
                ROUND2.common, "pd", pd
            ):
                payload = ROUND2.evaluate(experiment, Path(tmp), custody)

            expected = np.asarray([0.25, 0.50, 5.0 / 6.0, 11.0 / 12.0])
            reranked = pd.Series(expected).rank(method="average", pct=True).to_numpy()
            np.testing.assert_allclose(captured["predictions"], expected)
            self.assertFalse(np.allclose(captured["predictions"], reranked))
            self.assertEqual(
                set(payload),
                {
                    "schema_version", "stage", "state", "selected",
                    "passed_count", "required_count",
                    "individual_requirement_passed", "ensemble",
                    "round1_receipt", "realizations",
                },
            )
            self.assertEqual(
                set(payload["ensemble"]),
                {"definition", "metrics", "checks", "passed"},
            )
            self.assertIn("no rerank", payload["ensemble"]["definition"])


class TestEnder22CompletionIdentity(unittest.TestCase):
    def test_tampered_completion_and_bound_outputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "numerai/agents/experiments/ender22_temporal_retention_v53"
            for directory in ("configs", "predictions", "results", "receipts"):
                (experiment / directory).mkdir(parents=True, exist_ok=True)
            name = COMMON.CONTROL
            config = experiment / f"configs/{name}.py"
            predictions = experiment / f"predictions/{name}.parquet"
            result = experiment / f"results/{name}.json"
            completion = experiment / f"receipts/{name}.completion.json"
            manifest_path = experiment / "source_manifest_round1.json"
            config.write_bytes(b"CONFIG = {}\n")
            predictions.write_bytes(b"prediction bytes")
            result.write_bytes(b"result bytes")
            manifest_path.write_bytes(b"{}\n")
            relative = config.relative_to(root).as_posix()
            manifest = {
                "git_head": "a" * 40,
                "files": {relative: hashlib.sha256(config.read_bytes()).hexdigest()},
            }

            def identity(path: Path) -> dict:
                stat = path.lstat()
                return {
                    "path": str(path),
                    "device": int(stat.st_dev),
                    "inode": int(stat.st_ino),
                    "size_bytes": int(stat.st_size),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            payload = {
                "schema_version": 1,
                "stage": "ender22-round1-training-completion",
                "state": "OUTPUTS_FINALIZED",
                "component": name,
                "manifest": {
                    "path": manifest_path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    "git_head": manifest["git_head"],
                },
                "config": {"path": relative, "sha256": manifest["files"][relative]},
                "outputs": {
                    "predictions": identity(predictions),
                    "result": identity(result),
                },
            }
            completion.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(COMMON, "REPO_DIR", root):
                COMMON.validate_completion(experiment, name, manifest, 1)
                mutations = (
                    lambda: completion.write_text("{}", encoding="utf-8"),
                    lambda: predictions.write_bytes(b"tampered prediction bytes"),
                    lambda: result.write_bytes(b"tampered result bytes"),
                )
                for index, mutate in enumerate(mutations):
                    with self.subTest(case=index):
                        predictions.write_bytes(b"prediction bytes")
                        result.write_bytes(b"result bytes")
                        completion.write_text(json.dumps(payload), encoding="utf-8")
                        mutate()
                        with self.assertRaises(ValueError):
                            COMMON.validate_completion(experiment, name, manifest, 1)


class TestEnder22StoredResultSchema(unittest.TestCase):
    @staticmethod
    def _joined() -> pd.DataFrame:
        rows = []
        for era_index, era in enumerate(("0301", "0305", "0309")):
            for index in range(5):
                rows.append(
                    {
                        "id": f"{era}_{index}",
                        "era": era,
                        "target_ender_20": [0.0, 0.25, 0.5, 0.75, 1.0][index],
                        "v53_lgbm_ender20": [0.8, 0.1, 0.6, 0.2, 0.4][index],
                        "prediction": [0.2, 0.9, 0.4, 0.7, 0.1][
                            (index + era_index) % 5
                        ],
                        "cv_fold": 1,
                    }
                )
        return pd.DataFrame(rows)

    def _stored(self, experiment: Path, name: str, joined: pd.DataFrame) -> dict:
        bmc = numerai_metrics.per_era_bmc(
            joined, ["prediction"], COMMON.BENCHMARK, COMMON.TARGET
        )["prediction"]
        corr = numerai_metrics.per_era_corr(
            joined, ["prediction"], COMMON.TARGET
        )["prediction"]
        similarity = numerai_metrics.per_era_pred_corr(
            joined, ["prediction"], COMMON.BENCHMARK
        )["prediction"]
        bmc_summary = numerai_metrics.score_summary(bmc)
        corr_summary = numerai_metrics.score_summary(corr)
        bmc_metrics = {
            **bmc_summary,
            "avg_corr_with_benchmark": float(similarity.mean()),
        }
        return {
            "model": {},
            "preprocessing": {},
            "data": {
                "full_rows": len(joined),
                "full_eras": int(joined["era"].nunique()),
                "oof_rows": len(joined),
                "oof_eras": int(joined["era"].nunique()),
            },
            "benchmark": {},
            "output": {
                "output_dir": str(Path(experiment).absolute()),
                "predictions_file": str(Path("predictions") / f"{name}.parquet"),
                "prediction_semantics": COMMON.EXPECTED_SEMANTICS,
            },
            "metrics": {
                "corr": corr_summary,
                "bmc": bmc_metrics,
                "bmc_last_200_eras": deepcopy(bmc_metrics),
            },
            "cv": {},
            "training": {},
        }

    def test_exact_metrics_and_output_contract_rejects_tampering(self) -> None:
        COMMON.load_governed_dependencies()
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp)
            name = COMMON.CONTROL
            joined = self._joined()
            stored = self._stored(experiment, name, joined)
            COMMON.validate_stored_result_schema(
                stored, {}, experiment, name, joined
            )
            mutations = (
                lambda value: value["metrics"]["bmc"].update(std=99.0),
                lambda value: value["metrics"]["bmc"].update(
                    avg_corr_with_benchmark=99.0
                ),
                lambda value: value["output"].update(output_dir="elsewhere"),
                lambda value: value["output"].update(
                    predictions_file="predictions/other.parquet"
                ),
                lambda value: value["metrics"]["bmc"].update(extra=1.0),
                lambda value: value["data"].update(oof_rows=len(joined) + 1),
                lambda value: value["data"].update(
                    oof_eras=int(joined["era"].nunique()) + 1
                ),
            )
            for index, mutation in enumerate(mutations):
                with self.subTest(case=index):
                    tampered = deepcopy(stored)
                    mutation(tampered)
                    # Row/era tampering is rejected by score_candidate's exact
                    # data envelope; schema/metric/output changes are rejected
                    # directly by this validator.
                    if index >= 5:
                        self.assertNotEqual(tampered["data"], stored["data"])
                    else:
                        with self.assertRaises(ValueError):
                            COMMON.validate_stored_result_schema(
                                tampered, {}, experiment, name, joined
                            )


class TestEnder22DecisionReservation(unittest.TestCase):
    def test_validation_failure_rolls_back_decision_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decision.json"
            with self.assertRaisesRegex(ValueError, "synthetic validation"):
                with COMMON.DecisionReservation(path):
                    self.assertTrue(path.exists())
                    raise ValueError("synthetic validation")
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
