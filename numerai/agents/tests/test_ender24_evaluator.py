from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import pandas as pd


EXPERIMENT = (
    Path(__file__).resolve().parents[1]
    / "experiments/ender24_ema_seed_stability_v53"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = _load_module("ender24_evaluation_common_test", EXPERIMENT / "evaluation_common.py")
_previous_common = sys.modules.get("evaluation_common")
sys.modules["evaluation_common"] = COMMON
try:
    ROUND1 = _load_module(
        "ender24_evaluate_round1_test", EXPERIMENT / "evaluate_round1_impl.py"
    )
finally:
    if _previous_common is None:
        sys.modules.pop("evaluation_common", None)
    else:
        sys.modules["evaluation_common"] = _previous_common


def _metrics(
    *,
    full: float = 0.010,
    recent: float = 0.012,
    sharpe: float = 1.0,
    drawdown: float = 0.02,
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


def _diagnostics(*, ema: bool) -> dict:
    result = {
        "best_epoch": 2,
        "n_parameters": 123,
        "epochs_ran": 4,
        "best_val_loss": 0.1,
        "final_train_loss": 0.2,
    }
    if ema:
        result.update(
            {
                "ema_decay": 0.995,
                "ema_updates": 8,
                "ema_live_state_sha256": "a" * 64,
                "ema_shadow_state_sha256": "b" * 64,
                "ema_inference_state_sha256": "c" * 64,
            }
        )
    return result


class TestEnder24FrozenEvaluatorContracts(unittest.TestCase):
    def test_manifest_set_is_exact_31_file_round1_envelope(self):
        files = COMMON._manifest_file_set(1)
        self.assertEqual(len(files), 31)
        self.assertIn(f"{COMMON.PREFIX}/protocol/mechanical_activity_receipt.json", files)
        self.assertIn("numerai/agents/tests/test_ender24_ema_seed_stability.py", files)
        self.assertNotIn("numerai/agents/tests/test_ender24_evaluator.py", files)
        self.assertFalse(any("round2.py" in path for path in files))
        self.assertFalse(any("evaluate_round2" in path for path in files))
        with self.assertRaisesRegex(ValueError, "only a Round-1 manifest"):
            COMMON._manifest_file_set(2)

    def test_all_four_round1_configs_validate_with_exact_declared_procedure(self):
        for name in COMMON.ROUND1_NAMES:
            with self.subTest(name=name):
                config, procedure = COMMON.validate_config(EXPERIMENT, name)
                self.assertEqual(config["output"]["results_name"], name)
                self.assertEqual(config["training"]["sample_seed"], 1337)
                if "ema995" in name:
                    self.assertEqual(procedure["ema_decay"], 0.995)
                    self.assertEqual(config["model"]["params"]["ema_decay"], 0.995)
                else:
                    self.assertIsNone(procedure["ema_decay"])
                    self.assertNotIn("ema_decay", config["model"]["params"])

    def test_completion_preflight_is_exact_and_ordered(self):
        calls = []

        def validate(_experiment, name, _manifest, _custody):
            calls.append(name)
            if name == COMMON.ROUND1_NAMES[-1]:
                raise ValueError("fourth completion")
            return {"component": name}

        with mock.patch.object(COMMON, "validate_completion", side_effect=validate):
            with self.assertRaisesRegex(ValueError, "fourth completion"):
                COMMON.preflight_all_completions(
                    EXPERIMENT, {}, SimpleNamespace()
                )
        self.assertEqual(calls, list(COMMON.ROUND1_NAMES))

    def test_evaluator_reads_no_authority_or_result_after_failed_preflight(self):
        custody = SimpleNamespace(manifest={})
        with mock.patch.object(
            ROUND1,
            "preflight_all_completions",
            side_effect=ValueError("completion invalid"),
        ), mock.patch.object(ROUND1, "load_authority") as authority, mock.patch.object(
            ROUND1, "load_truth"
        ) as truth, mock.patch.object(ROUND1, "score_candidate") as score:
            with self.assertRaisesRegex(ValueError, "completion invalid"):
                ROUND1.evaluate(EXPERIMENT, Path("numerai"), custody)
        authority.assert_not_called()
        truth.assert_not_called()
        score.assert_not_called()

    def test_failed_decision_validation_removes_only_its_empty_reservation(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "round1_ema_stability.json"
            with self.assertRaisesRegex(ValueError, "synthetic invalid evidence"):
                with COMMON.DecisionReservation(path):
                    self.assertTrue(path.exists())
                    raise ValueError("synthetic invalid evidence")
            self.assertFalse(path.exists())

            existing = b'{"prior":"evidence"}\n'
            path.write_bytes(existing)
            with self.assertRaisesRegex(ValueError, "create-new decision"):
                with COMMON.DecisionReservation(path):
                    pass
            self.assertEqual(path.read_bytes(), existing)


class TestEnder24Diagnostics(unittest.TestCase):
    def test_exact_control_and_ema_schemas_pass(self):
        self.assertEqual(
            COMMON.validate_model_diagnostics(
                _diagnostics(ema=False), name="control", is_ema=False
            ),
            123,
        )
        self.assertEqual(
            COMMON.validate_model_diagnostics(
                _diagnostics(ema=True), name="ema", is_ema=True
            ),
            123,
        )

    def test_diagnostic_mutations_fail_closed(self):
        cases = []
        control_extra = _diagnostics(ema=False)
        control_extra["ema_updates"] = 1
        cases.append((control_extra, False))
        for key, value in (
            ("ema_decay", 0.99),
            ("ema_updates", 0),
            ("ema_updates", True),
            ("ema_live_state_sha256", "A" * 64),
            ("ema_inference_state_sha256", "x" * 64),
            ("best_val_loss", float("nan")),
        ):
            mutated = _diagnostics(ema=True)
            mutated[key] = value
            cases.append((mutated, True))
        equal_hashes = _diagnostics(ema=True)
        equal_hashes["ema_shadow_state_sha256"] = equal_hashes[
            "ema_live_state_sha256"
        ]
        cases.append((equal_hashes, True))
        missing = _diagnostics(ema=True)
        missing.pop("ema_inference_state_sha256")
        cases.append((missing, True))
        for index, (diagnostics, is_ema) in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                COMMON.validate_model_diagnostics(
                    diagnostics, name="run", is_ema=is_ema
                )


class TestEnder24DecisionRules(unittest.TestCase):
    def test_check_key_sets_are_exact(self):
        per_run = COMMON.per_ema_checks(_metrics(), _metrics())
        self.assertEqual(
            set(per_run),
            {
                "full_bmc_positive",
                "all_used_folds_bmc_positive",
                "three_of_four_recent_blocks_positive",
                "worst_recent_block_above_minus_0_001",
                "corr_at_least_0_005",
                "corr_below_0_04",
                "benchmark_corr_below_0_25",
                "sharpe_not_below_matched_control_minus_0_05",
                "drawdown_no_greater_than_matched_control_plus_0_01",
            },
        )
        aggregates = {
            "full_bmc": {
                "mean": 1.0,
                "worst_seed": 1.0,
                "absolute_two_seed_gap": 0.0,
            },
            "recent40_bmc": {
                "mean": 1.0,
                "worst_seed": 1.0,
                "absolute_two_seed_gap": 0.0,
            },
        }
        self.assertEqual(
            set(COMMON.aggregate_checks(aggregates, aggregates)),
            {
                "ema_mean_recent40_bmc_at_least_control",
                "ema_worst_seed_recent40_bmc_at_least_control",
                "ema_mean_full_bmc_retains_95pct_control",
                "ema_worst_seed_full_bmc_retains_95pct_control",
                "ema_full_bmc_two_seed_gap_at_most_75pct_control",
                "ema_recent40_bmc_two_seed_gap_at_most_75pct_control",
            },
        )

    def test_aggregate_thresholds_accept_equality_and_reject_beyond(self):
        control = {
            "full_bmc": {
                "mean": 0.010,
                "worst_seed": 0.008,
                "absolute_two_seed_gap": 0.004,
            },
            "recent40_bmc": {
                "mean": 0.012,
                "worst_seed": 0.010,
                "absolute_two_seed_gap": 0.004,
            },
        }
        ema = {
            "full_bmc": {
                "mean": 0.0095,
                "worst_seed": 0.0076,
                "absolute_two_seed_gap": 0.003,
            },
            "recent40_bmc": {
                "mean": 0.012,
                "worst_seed": 0.010,
                "absolute_two_seed_gap": 0.003,
            },
        }
        self.assertTrue(all(COMMON.aggregate_checks(control, ema).values()))
        for section, field in (
            ("recent40_bmc", "mean"),
            ("recent40_bmc", "worst_seed"),
            ("full_bmc", "mean"),
            ("full_bmc", "worst_seed"),
        ):
            mutated = deepcopy(ema)
            mutated[section][field] -= 1e-12
            self.assertFalse(all(COMMON.aggregate_checks(control, mutated).values()))
        for section in ("full_bmc", "recent40_bmc"):
            mutated = deepcopy(ema)
            mutated[section]["absolute_two_seed_gap"] += 1e-12
            self.assertFalse(all(COMMON.aggregate_checks(control, mutated).values()))

    def test_zero_control_gap_requires_zero_ema_gap(self):
        control_records = {
            "a": {"metrics": _metrics(full=0.01, recent=0.01)},
            "b": {"metrics": _metrics(full=0.01, recent=0.01)},
        }
        control = COMMON.procedure_aggregates(control_records, ("a", "b"))
        ema = deepcopy(control)
        self.assertTrue(all(COMMON.aggregate_checks(control, ema).values()))
        ema["full_bmc"]["absolute_two_seed_gap"] = 1e-15
        self.assertFalse(
            COMMON.aggregate_checks(control, ema)[
                "ema_full_bmc_two_seed_gap_at_most_75pct_control"
            ]
        )

    def test_all_nine_per_run_boundaries(self):
        control = _metrics(sharpe=1.0, drawdown=0.02)
        ema = _metrics(sharpe=0.95, drawdown=0.03)
        ema["corr_mean"] = 0.005
        ema["avg_corr_with_benchmark"] = 0.249999
        ema["recent_blocks_bmc_mean"] = {
            "a": 0.01,
            "b": 0.01,
            "c": 0.01,
            "d": -0.000999,
        }
        self.assertTrue(all(COMMON.per_ema_checks(ema, control).values()))
        mutations = (
            lambda value: value["bmc"].update(mean=0.0),
            lambda value: value["fold_bmc_mean"].update({"1": 0.0}),
            lambda value: value["recent_blocks_bmc_mean"].update(
                {"b": 0.0}
            ),
            lambda value: value["recent_blocks_bmc_mean"].update(
                {"d": -0.001}
            ),
            lambda value: value.update(corr_mean=0.004999),
            lambda value: value.update(corr_mean=0.04),
            lambda value: value.update(avg_corr_with_benchmark=0.25),
            lambda value: value["bmc"].update(sharpe=0.949999),
            lambda value: value["bmc"].update(max_drawdown=0.030001),
        )
        for index, mutation in enumerate(mutations):
            changed = deepcopy(ema)
            mutation(changed)
            with self.subTest(case=index):
                self.assertFalse(all(COMMON.per_ema_checks(changed, control).values()))


class TestEnder24DecisionEnvelope(unittest.TestCase):
    @staticmethod
    def _frame():
        return pd.DataFrame(
            {
                "id": ["a", "b"],
                "era": ["0301", "0861"],
                "target_ender_20": [0.25, 0.75],
                "v53_lgbm_ender20": [0.4, 0.6],
                "cv_fold": [1, 4],
                "prediction": [0.2, 0.8],
            }
        )

    def _records(self):
        values = {
            "r1_control_seed1337": _metrics(full=0.012, recent=0.014),
            "r1_ema995_seed1337": _metrics(full=0.0108, recent=0.013),
            "r1_control_seed2027": _metrics(full=0.008, recent=0.010),
            "r1_ema995_seed2027": _metrics(full=0.009, recent=0.0115),
        }
        return {
            name: {
                "metrics": metrics,
                "parameter_count": 123,
                "provenance": {"manifest": {"git_head": "a" * 40}},
            }
            for name, metrics in values.items()
        }

    def _evaluate(self, records):
        frames = {name: self._frame() for name in COMMON.ROUND1_NAMES}

        def score(_experiment, name, *_args):
            return records[name], frames[name]

        custody = SimpleNamespace(manifest={})
        with mock.patch.object(
            ROUND1,
            "preflight_all_completions",
            return_value={name: {} for name in COMMON.ROUND1_NAMES},
        ), mock.patch.object(ROUND1, "load_authority", return_value=({}, [])), mock.patch.object(
            ROUND1, "load_truth", return_value=pd.DataFrame()
        ), mock.patch.object(ROUND1, "score_candidate", side_effect=score), mock.patch.object(
            ROUND1, "_exact_config_pair_delta", return_value=True
        ), mock.patch.object(
            ROUND1, "receipt", side_effect=lambda path, *_args: {"path": str(path)}
        ):
            return ROUND1.evaluate(EXPERIMENT, Path("numerai"), custody)

    def test_exact_passing_decision_envelope(self):
        payload = self._evaluate(self._records())
        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "stage",
                "state",
                "round2_authorized",
                "inputs",
                "runs",
                "matched_pairs",
                "aggregates",
                "aggregate_checks",
                "ema_run_checks",
                "passed",
            },
        )
        self.assertEqual(payload["stage"], "ender24-round1-ema-seed-stability")
        self.assertEqual(payload["state"], "ROUND2_AUTHORIZED")
        self.assertTrue(payload["round2_authorized"])
        self.assertEqual(set(payload["runs"]), set(COMMON.ROUND1_NAMES))
        self.assertEqual(set(payload["matched_pairs"]), {"1337", "2027"})
        self.assertEqual(
            payload["matched_pairs"]["1337"]["config_delta"],
            ["model.params.ema_decay", "output.results_name"],
        )

    def test_metric_failure_is_terminal_negative(self):
        records = self._records()
        records["r1_ema995_seed2027"]["metrics"]["corr_mean"] = 0.04
        payload = self._evaluate(records)
        self.assertEqual(payload["state"], "NEGATIVE_NO_EMA_STABILITY_GAIN")
        self.assertFalse(payload["round2_authorized"])
        self.assertFalse(payload["passed"])

    def test_pair_cohort_and_provenance_mismatch_raise(self):
        records = self._records()
        records["r1_ema995_seed1337"]["provenance"] = {
            "manifest": {"git_head": "b" * 40}
        }
        with self.assertRaisesRegex(ValueError, "matched-pair contract"):
            self._evaluate(records)

        records = self._records()
        original = self._frame
        calls = {"count": 0}

        def mismatched():
            calls["count"] += 1
            frame = original()
            if calls["count"] == 2:
                frame.loc[0, "target_ender_20"] = 0.5
            return frame

        self._frame = mismatched
        try:
            with self.assertRaisesRegex(ValueError, "matched-pair contract"):
                self._evaluate(records)
        finally:
            self._frame = original


if __name__ == "__main__":
    unittest.main()
