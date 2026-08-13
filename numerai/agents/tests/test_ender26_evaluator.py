from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd


EXPERIMENT = (
    Path(__file__).resolve().parents[1]
    / "experiments/ender26_gaussian_rank_residual_v53"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = _load_module("ender26_evaluation_common_test", EXPERIMENT / "evaluation_common.py")
BOOTSTRAP = _load_module(
    "ender26_training_bootstrap_evaluator_test",
    EXPERIMENT / "training_bootstrap.py",
)
_previous_common = sys.modules.get("evaluation_common")
sys.modules["evaluation_common"] = COMMON
try:
    ROUND1 = _load_module(
        "ender26_evaluate_round1_test", EXPERIMENT / "evaluate_round1_impl.py"
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


class TestEnder26PreflightAndCustody(unittest.TestCase):
    def test_completion_preflight_is_all_four_and_ordered(self) -> None:
        calls = []

        def validate(_experiment, name, _manifest, _custody):
            calls.append(name)
            if name == COMMON.ROUND1_NAMES[-1]:
                raise ValueError("fourth completion")
            return {"component": name}

        with mock.patch.object(COMMON, "validate_completion", side_effect=validate):
            with self.assertRaisesRegex(ValueError, "fourth completion"):
                COMMON.preflight_all_completions(EXPERIMENT, {}, SimpleNamespace())
        self.assertEqual(calls, list(COMMON.ROUND1_NAMES))

    def test_failed_preflight_reads_no_authority_truth_config_or_result(self) -> None:
        custody = SimpleNamespace(manifest={})
        with mock.patch.object(
            ROUND1,
            "preflight_all_completions",
            side_effect=ValueError("completion invalid"),
        ), mock.patch.object(ROUND1, "load_authority") as authority, mock.patch.object(
            ROUND1, "load_truth"
        ) as truth, mock.patch.object(ROUND1, "score_candidate") as score, mock.patch.object(
            ROUND1, "_exact_config_pair_delta"
        ) as config_delta:
            with self.assertRaisesRegex(ValueError, "completion invalid"):
                ROUND1.evaluate(EXPERIMENT, Path("numerai"), custody)
        authority.assert_not_called()
        truth.assert_not_called()
        score.assert_not_called()
        config_delta.assert_not_called()

    def test_failed_decision_is_terminal_zero_byte_create_new_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "round1_bmc_alignment.json"
            with self.assertRaisesRegex(ValueError, "synthetic invalid evidence"):
                with COMMON.DecisionReservation(path):
                    self.assertTrue(path.exists())
                    raise ValueError("synthetic invalid evidence")
            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes(), b"")
            with self.assertRaisesRegex(ValueError, "create-new decision"):
                with COMMON.DecisionReservation(path):
                    pass
            self.assertEqual(path.read_bytes(), b"")

    def test_nonempty_failed_decision_evidence_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "round1_bmc_alignment.json"
            reservation = COMMON.DecisionReservation(path)
            with self.assertRaisesRegex(RuntimeError, "post-write validation failure"):
                with reservation:
                    reservation.stream.write(b'{"partial":true}\n')
                    reservation.stream.flush()
                    raise RuntimeError("post-write validation failure")
            self.assertEqual(path.read_bytes(), b'{"partial":true}\n')

    def test_replaced_failed_decision_path_is_preserved_and_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "round1_bmc_alignment.json"
            replacement = Path(temporary) / "replacement.json"
            replacement.write_bytes(b'{"foreign":true}\n')
            reservation = COMMON.DecisionReservation(path)
            reservation.__enter__()
            reservation.stream.close()
            reservation.stream = None
            path.unlink()
            replacement.replace(path)
            with self.assertRaisesRegex(ValueError, "changed decision reservation"):
                reservation.__exit__(RuntimeError, RuntimeError("identity swap"), None)
            self.assertEqual(path.read_bytes(), b'{"foreign":true}\n')

    def test_bootstrap_decision_preserves_nonempty_failed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "round1_bmc_alignment.json"
            reservation = BOOTSTRAP._DecisionReservation(path)
            with self.assertRaisesRegex(RuntimeError, "post-write validation failure"):
                with reservation:
                    reservation.stream.write(b'{"partial":true}\n')
                    reservation.stream.flush()
                    raise RuntimeError("post-write validation failure")
            self.assertEqual(path.read_bytes(), b'{"partial":true}\n')

    def test_bootstrap_failed_decision_is_terminal_zero_byte_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "round1_bmc_alignment.json"
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                with BOOTSTRAP._DecisionReservation(path):
                    raise RuntimeError("synthetic failure")
            self.assertEqual(path.read_bytes(), b"")
            with self.assertRaisesRegex(ValueError, "evaluation decision"):
                with BOOTSTRAP._DecisionReservation(path):
                    pass

    def test_bootstrap_decision_refuses_replaced_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "round1_bmc_alignment.json"
            replacement = Path(temporary) / "replacement.json"
            replacement.write_bytes(b'{"foreign":true}\n')
            reservation = BOOTSTRAP._DecisionReservation(path)
            reservation.__enter__()
            reservation.stream.close()
            reservation.stream = None
            path.unlink()
            replacement.replace(path)
            with self.assertRaisesRegex(ValueError, "changed evaluation decision"):
                reservation.__exit__(RuntimeError, RuntimeError("identity swap"), None)
            self.assertEqual(path.read_bytes(), b'{"foreign":true}\n')

    def test_semantic_authority_accepts_lf_or_crlf_only(self) -> None:
        lf = b'[\n  "0161",\n  "0165"\n]\n'
        crlf = lf.replace(b"\n", b"\r\n")
        self.assertEqual(
            COMMON.canonical_lf_json_bytes(lf, "allowlist"),
            COMMON.canonical_lf_json_bytes(crlf, "allowlist"),
        )
        for invalid in (
            b"\xef\xbb\xbf[]\n",
            b"[\r]\n",
            b"[\r\n]\n",
            b"",
        ):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                COMMON.canonical_lf_json_bytes(invalid, "allowlist")


class TestEnder26FrozenContracts(unittest.TestCase):
    def test_truth_rejects_malformed_or_nonfinite_before_scoring(self) -> None:
        allowed = ["0161"]
        full_base = pd.DataFrame(
            {
                "id": ["a"],
                "era": ["0161"],
                "target_ender_20": [0.5],
            }
        )
        benchmark_base = pd.DataFrame(
            {
                "id": ["a"],
                "era": ["0161"],
                "v53_lgbm_ender20": [0.4],
            }
        )
        cases = (
            ("target", "bad"),
            ("target", float("nan")),
            ("target", float("inf")),
            ("target", float("-inf")),
            ("benchmark", "bad"),
            ("benchmark", float("nan")),
            ("benchmark", float("inf")),
            ("benchmark", float("-inf")),
        )
        for column, value in cases:
            full = full_base.copy()
            benchmark = benchmark_base.copy()
            if column == "target":
                full["target_ender_20"] = pd.Series([value], dtype="object")
            else:
                benchmark["v53_lgbm_ender20"] = pd.Series(
                    [value], dtype="object"
                )
            reads = iter((full, benchmark))
            fake_pd = SimpleNamespace(
                read_parquet=mock.Mock(side_effect=lambda *_args, **_kwargs: next(reads)),
                to_numeric=pd.to_numeric,
            )
            with self.subTest(column=column, value=value), mock.patch.object(
                COMMON, "pd", fake_pd
            ), mock.patch.object(COMMON, "np", __import__("numpy")):
                with self.assertRaisesRegex(
                    ValueError, "malformed|non-finite"
                ):
                    COMMON.load_truth(Path("numerai"), allowed)

    def test_all_configs_validate_with_exact_target_semantics(self) -> None:
        for name in COMMON.ROUND1_NAMES:
            with self.subTest(name=name):
                config, procedure = COMMON.validate_config(EXPERIMENT, name)
                transform = config["model"]["target_transform"]
                self.assertTrue(transform["fit_intercept"])
                self.assertEqual(transform["proportion"], 1.0)
                self.assertEqual(config["training"]["cv"]["max_train_eras"], 78)
                if "grank" in name:
                    self.assertEqual(
                        procedure["benchmark_transform"],
                        "tie_kept_rank_gaussian",
                    )
                    self.assertEqual(
                        transform["benchmark_transform"],
                        "tie_kept_rank_gaussian",
                    )
                else:
                    self.assertEqual(procedure["benchmark_transform"], "identity")
                    self.assertNotIn("benchmark_transform", transform)

    def test_exact_pair_delta_rejects_any_second_scientific_change(self) -> None:
        for control, challenger in COMMON.PAIR_NAMES.values():
            self.assertTrue(
                ROUND1._exact_config_pair_delta(
                    EXPERIMENT, control, challenger, None
                )
            )

        original = ROUND1.validate_config

        def changed(experiment, name, custody):
            config, procedure = original(experiment, name, custody)
            config = deepcopy(config)
            if "grank" in name:
                config["model"]["params"]["dropout"] = 0.2
            return config, procedure

        with mock.patch.object(ROUND1, "validate_config", side_effect=changed):
            self.assertFalse(
                ROUND1._exact_config_pair_delta(
                    EXPERIMENT,
                    "r1_control_rawresid_seed1337",
                    "r1_grank_resid_seed1337",
                    None,
                )
            )

    def test_diagnostics_schema_is_exact(self) -> None:
        diagnostics = {
            "best_epoch": 2,
            "n_parameters": 123,
            "epochs_ran": 4,
            "best_val_loss": 0.1,
            "final_train_loss": 0.2,
        }
        self.assertEqual(
            COMMON.validate_model_diagnostics(diagnostics, name="run"), 123
        )
        for mutation in (
            lambda value: value.update(ema_updates=1),
            lambda value: value.update(best_epoch=True),
            lambda value: value.update(n_parameters=0),
            lambda value: value.update(best_val_loss=float("nan")),
        ):
            changed = deepcopy(diagnostics)
            mutation(changed)
            with self.assertRaises(ValueError):
                COMMON.validate_model_diagnostics(changed, name="run")


class TestEnder26DecisionRules(unittest.TestCase):
    def test_nonfinite_missing_extra_or_misaligned_era_cannot_be_dropped(self) -> None:
        eras = [f"{era:04d}" for era in range(301, 862, 4)]
        joined = pd.DataFrame(
            {
                "era": eras,
                "cv_fold": [1] * len(eras),
            }
        )
        finite_series = pd.Series([0.01] * len(eras), index=eras)
        cases = {}
        nonfinite = finite_series.copy()
        nonfinite.iloc[17] = float("nan")
        cases["nonfinite"] = (nonfinite, finite_series, finite_series)
        cases["missing"] = (
            finite_series.iloc[:-1], finite_series, finite_series
        )
        extra = pd.concat(
            [finite_series, pd.Series([0.01], index=["0865"])]
        )
        cases["extra"] = (extra, finite_series, finite_series)
        misaligned = finite_series.sort_index(ascending=False)
        cases["misaligned"] = (finite_series, misaligned, finite_series)
        for label, series in cases.items():
            fake_metrics = SimpleNamespace(
                per_era_bmc=lambda *_args, value=series[0]: {
                    "prediction": value.copy()
                },
                per_era_corr=lambda *_args, value=series[1]: {
                    "prediction": value.copy()
                },
                per_era_pred_corr=lambda *_args, value=series[2]: {
                    "prediction": value.copy()
                },
                score_summary=mock.Mock(
                    side_effect=AssertionError("bad era was aggregated")
                ),
            )
            with self.subTest(case=label), mock.patch.object(
                COMMON, "np", __import__("numpy")
            ), mock.patch.object(COMMON, "numerai_metrics", fake_metrics):
                with self.assertRaisesRegex(
                    ValueError, "finite for every exact OOF era|indices differ"
                ):
                    COMMON.compute_metrics(joined)
            fake_metrics.score_summary.assert_not_called()

    def test_aggregate_thresholds_accept_equality_and_reject_below(self) -> None:
        control = {
            "full_bmc": {"mean": 0.010, "worst_seed": 0.009, "absolute_two_seed_gap": 0.002},
            "recent40_bmc": {"mean": 0.012, "worst_seed": 0.011, "absolute_two_seed_gap": 0.002},
        }
        challenger = deepcopy(control)
        challenger["full_bmc"]["mean"] += 0.00020
        challenger["recent40_bmc"]["mean"] += 0.00030
        checks = COMMON.aggregate_checks(control, challenger)
        self.assertEqual(
            set(checks),
            {
                "mean_full_bmc_at_least_control_plus_0_00020",
                "mean_recent40_bmc_at_least_control_plus_0_00030",
            },
        )
        self.assertTrue(all(checks.values()))
        for section in ("full_bmc", "recent40_bmc"):
            changed = deepcopy(challenger)
            changed[section]["mean"] -= 1e-12
            self.assertFalse(all(COMMON.aggregate_checks(control, changed).values()))

    def test_all_ten_per_run_boundaries(self) -> None:
        control = _metrics(full=0.010, recent=0.012, sharpe=1.0, drawdown=0.02)
        candidate = _metrics(full=0.0095, recent=0.012, sharpe=0.95, drawdown=0.03)
        candidate["corr_mean"] = 0.008
        candidate["avg_corr_with_benchmark"] = 0.249999
        candidate["recent_blocks_bmc_mean"] = {
            "a": 0.01,
            "b": 0.01,
            "c": 0.01,
            "d": -0.000999,
        }
        checks = COMMON.candidate_run_checks(candidate, control)
        self.assertEqual(len(checks), 10)
        self.assertTrue(all(checks.values()))
        mutations = (
            lambda value: value.update(recent40_bmc_mean=0.011999),
            lambda value: value["bmc"].update(mean=0.009499),
            lambda value: value["fold_bmc_mean"].update({"1": 0.0}),
            lambda value: value["recent_blocks_bmc_mean"].update({"b": 0.0}),
            lambda value: value["recent_blocks_bmc_mean"].update({"d": -0.001}),
            lambda value: value.update(corr_mean=0.007999),
            lambda value: value.update(corr_mean=0.04),
            lambda value: value.update(avg_corr_with_benchmark=0.25),
            lambda value: value["bmc"].update(sharpe=0.949999),
            lambda value: value["bmc"].update(max_drawdown=0.030001),
        )
        for index, mutation in enumerate(mutations):
            changed = deepcopy(candidate)
            mutation(changed)
            with self.subTest(case=index):
                self.assertFalse(
                    all(COMMON.candidate_run_checks(changed, control).values())
                )


class TestEnder26DecisionEnvelope(unittest.TestCase):
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

    @staticmethod
    def _records():
        values = {
            "r1_control_rawresid_seed1337": _metrics(full=0.0100, recent=0.0120),
            "r1_grank_resid_seed1337": _metrics(full=0.0103, recent=0.0125),
            "r1_control_rawresid_seed2027": _metrics(full=0.0100, recent=0.0120),
            "r1_grank_resid_seed2027": _metrics(full=0.0102, recent=0.0121),
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

    def test_positive_is_source_eligible_but_authorizes_nothing(self) -> None:
        payload = self._evaluate(self._records())
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["stage"], "ender26-round1-bmc-alignment")
        self.assertEqual(payload["state"], "ENDER26_ROUND2_SOURCE_GATE_ELIGIBLE")
        self.assertTrue(payload["round2_source_gate_eligible"])
        for key in (
            "continuation_authorized",
            "round2_source_gate_authorized",
            "round2_authorized",
            "training_authorized",
            "scoring_authorized",
            "deployment_authorized",
            "submission_authorized",
            "account_actions_authorized",
        ):
            self.assertIs(payload[key], False)
        self.assertEqual(
            payload["matched_pairs"]["1337"]["config_delta"],
            [
                "model.target_transform.benchmark_transform",
                "output.results_name",
            ],
        )

    def test_any_gate_failure_is_exact_terminal_negative(self) -> None:
        records = self._records()
        records["r1_grank_resid_seed2027"]["metrics"]["corr_mean"] = 0.04
        payload = self._evaluate(records)
        self.assertFalse(payload["passed"])
        self.assertFalse(payload["round2_source_gate_eligible"])
        self.assertEqual(
            payload["state"], "ENDER26_NEGATIVE_NO_BMC_ALIGNMENT_GAIN"
        )

    def test_pair_cohort_or_provenance_mismatch_fails_closed(self) -> None:
        records = self._records()
        records["r1_grank_resid_seed1337"]["provenance"] = {
            "manifest": {"git_head": "b" * 40}
        }
        with self.assertRaisesRegex(ValueError, "matched-pair contract"):
            self._evaluate(records)


if __name__ == "__main__":
    unittest.main()
