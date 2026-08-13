from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from agents.code.analysis import ender21_confirmation_rules as rules


EXPERIMENT = (
    Path(__file__).resolve().parents[1]
    / "experiments/ender21_residual_stability_v53"
)
CONFIGS = EXPERIMENT / "configs"
CONFIRMATION_NAME = "c1_selected_tabm_k64_block_dro"
EXPECTED_CONFIRMATION_ERAS = tuple(
    f"{era:04d}" for era in range(865, 1022, 4)
)


def _load_script(filename: str, module_name: str):
    path = EXPERIMENT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Ender21 script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_script("run_confirmation.py", "ender21_confirmation_runner")
EVALUATOR = _load_script(
    "evaluate_confirmation.py", "ender21_confirmation_evaluator"
)


def _metrics(
    *,
    bmc: float,
    sharpe: float = 0.250001,
    drawdown: float = 0.099999,
    corr: float = 0.008,
    benchmark_corr: float = 0.249999,
    blocks: tuple[float, float, float, float] = (
        0.001,
        0.001,
        0.001,
        -0.000999,
    ),
) -> dict:
    return {
        "bmc": {
            "mean": bmc,
            "sharpe": sharpe,
            "max_drawdown": drawdown,
        },
        "corr": {"mean": corr},
        "avg_corr_with_benchmark": benchmark_corr,
        "chronological_block_bmc": {
            str(index): value for index, value in enumerate(blocks)
        },
    }


class TestEnder21ConfirmationProtocol(unittest.TestCase):
    def test_runner_top_level_import_loads_no_governed_or_ml_modules(self) -> None:
        script = r'''
import importlib.util, json, sys
before = set(sys.modules)
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("ender21_runner_import_audit", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
loaded = set(sys.modules) - before
forbidden = sorted(
    name for name in loaded
    if name == "agents" or name.startswith("agents.")
    or name.split(".", 1)[0] in {"numpy", "pandas", "pyarrow", "torch", "tabm"}
)
print(json.dumps(forbidden))
'''
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-X",
                    f"pycache_prefix={tmp}",
                    "-c",
                    script,
                    str(EXPERIMENT / "run_confirmation.py"),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        self.assertEqual(json.loads(completed.stdout), [])

    def test_runner_manifest_and_leases_precede_governed_runtime(self) -> None:
        events = []

        class Reservation:
            def __init__(self, *_args):
                events.append("reserve_init")

            def __enter__(self):
                events.append("reserve_enter")
                return self

            def __exit__(self, *_args):
                events.append("reserve_exit")

        def manifest(*_args):
            events.append("manifest")
            return {}

        def leases(*_args):
            events.append("leases")
            return {}

        def runtime():
            events.append("governed_runtime")
            raise RuntimeError("STOP_AFTER_GOVERNED_IMPORT")

        with mock.patch.object(
            RUNNER,
            "_require_frozen_launch_policy",
            side_effect=lambda: events.append("launch_policy"),
        ), mock.patch.object(
            RUNNER, "_ConfirmationOutputReservations", Reservation
        ), mock.patch.object(
            RUNNER, "verify_confirmation_manifest", side_effect=manifest
        ), mock.patch.object(
            RUNNER, "_acquire_confirmation_input_leases", side_effect=leases
        ), mock.patch.object(
            RUNNER, "_load_governed_runtime", side_effect=runtime
        ):
            with self.assertRaisesRegex(RuntimeError, "STOP_AFTER_GOVERNED_IMPORT"):
                RUNNER.run_confirmation(Path("experiment"), Path("numerai"))

        self.assertEqual(
            events[:5],
            [
                "launch_policy",
                "reserve_init",
                "reserve_enter",
                "manifest",
                "leases",
            ],
        )
        self.assertEqual(events[5], "governed_runtime")

    def test_confirmation_era_contract_is_exactly_four_ordered_blocks(self) -> None:
        import json

        path = EXPERIMENT / "protocol/confirmation_eras_0865_through_1021.json"
        actual = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(tuple(actual), EXPECTED_CONFIRMATION_ERAS)
        self.assertEqual(len(actual), 40)
        self.assertEqual(
            tuple(tuple(actual[start : start + 10]) for start in range(0, 40, 10)),
            tuple(
                EXPECTED_CONFIRMATION_ERAS[start : start + 10]
                for start in range(0, 40, 10)
            ),
        )

    def test_runner_era_contract_keeps_fit_embargo_and_holdout_disjoint(self) -> None:
        contract = RUNNER._load_era_contract(EXPERIMENT)
        self.assertEqual(set(contract), {"fit", "embargo", "confirmation", "receipts"})
        self.assertEqual(
            contract["fit"], tuple(f"{era:04d}" for era in range(161, 810, 4))
        )
        self.assertEqual(
            contract["embargo"], tuple(f"{era:04d}" for era in range(813, 862, 4))
        )
        self.assertEqual(contract["confirmation"], EXPECTED_CONFIRMATION_ERAS)
        self.assertEqual(
            (len(contract["fit"]), len(contract["embargo"]), len(contract["confirmation"])),
            (163, 13, 40),
        )
        self.assertFalse(set(contract["fit"]) & set(contract["embargo"]))
        self.assertFalse(set(contract["fit"]) & set(contract["confirmation"]))
        self.assertFalse(set(contract["embargo"]) & set(contract["confirmation"]))
        self.assertEqual(set(contract["receipts"]), {"fit", "embargo", "confirmation"})
        for label, eras in (
            ("fit", contract["fit"]),
            ("embargo", contract["embargo"]),
            ("confirmation", contract["confirmation"]),
        ):
            receipt = contract["receipts"][label]
            self.assertEqual(receipt["era_count"], len(eras))
            self.assertEqual(receipt["first_era"], eras[0])
            self.assertEqual(receipt["last_era"], eras[-1])
            self.assertRegex(receipt["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(receipt["size_bytes"], 0)

    def test_confirmation_reader_never_requests_target_column(self) -> None:
        full_path = Path("synthetic-full.parquet")
        benchmark_path = Path("synthetic-benchmark.parquet")
        full = pd.DataFrame(
            {
                "id": ["a", "b"],
                "era": ["0865", "0869"],
                "feature_a": [0.1, 0.2],
                "feature_b": [0.3, 0.4],
            }
        )
        benchmark = pd.DataFrame(
            {
                "id": ["a", "b"],
                "era": ["0865", "0869"],
                "v53_lgbm_ender20": [0.5, 0.6],
            }
        )
        requested_columns = []

        def read_parquet(path, *, columns, filters):
            requested_columns.append((Path(path), tuple(columns)))
            self.assertEqual(filters, [("era", "in", ["0865", "0869"])])
            if Path(path) == full_path:
                return full.copy()
            if Path(path) == benchmark_path:
                return benchmark.copy()
            raise AssertionError(f"unexpected path: {path}")

        fake_pd = mock.Mock(read_parquet=mock.Mock(side_effect=read_parquet))
        with mock.patch.object(RUNNER, "CONFIRMATION_ROWS", 2):
            frame = RUNNER._read_confirmation_predictors(
                full_path,
                benchmark_path,
                ("0865", "0869"),
                ["feature_a", "feature_b"],
                pd_module=fake_pd,
            )

        self.assertEqual(
            requested_columns,
            [
                (full_path, ("id", "era", "feature_a", "feature_b")),
                (
                    benchmark_path,
                    ("id", "era", "v53_lgbm_ender20"),
                ),
            ],
        )
        self.assertNotIn("target_ender_20", frame.columns)
        self.assertEqual(
            list(frame.columns),
            [
                "id",
                "era",
                "feature_a",
                "feature_b",
                "v53_lgbm_ender20",
            ],
        )

    def test_confirmation_config_is_the_exact_selected_family(self) -> None:
        selected = runpy.run_path(
            str(CONFIGS / "r1_tabm_k64_block_dro.py")
        )["CONFIG"]
        actual = runpy.run_path(
            str(CONFIGS / f"{CONFIRMATION_NAME}.py")
        )["CONFIG"]
        expected = deepcopy(selected)
        expected["output"]["results_name"] = CONFIRMATION_NAME
        self.assertEqual(actual, expected)

        self.assertEqual(actual["model"]["params"]["seed"], 1337)
        self.assertEqual(
            actual["model"]["params"]["loss_mode"],
            "chronological_block_dro",
        )
        self.assertEqual(actual["model"]["params"]["internal_val_embargo"], 13)
        self.assertEqual(actual["training"]["max_train_samples"], 500_000)
        self.assertEqual(actual["training"]["sample_seed"], 1337)
        RUNNER._validate_config(
            actual,
            EXPERIMENT,
            load_config_fn=lambda _path: selected,
        )

    def test_confirmation_config_validator_rejects_family_drift(self) -> None:
        canonical = runpy.run_path(
            str(CONFIGS / f"{CONFIRMATION_NAME}.py")
        )["CONFIG"]
        mutations = {
            "model seed": lambda value: value["model"]["params"].__setitem__(
                "seed", 2027
            ),
            "sample seed": lambda value: value["training"].__setitem__(
                "sample_seed", 2027
            ),
            "sample cap": lambda value: value["training"].__setitem__(
                "max_train_samples", 250_000
            ),
            "loss": lambda value: value["model"]["params"].__setitem__(
                "loss_mode", "mse"
            ),
            "inner embargo": lambda value: value["model"]["params"].__setitem__(
                "internal_val_embargo", 12
            ),
            "target transform": lambda value: value["model"].__setitem__(
                "target_transform", {"type": "identity"}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                changed = deepcopy(canonical)
                mutate(changed)
                with self.assertRaises((TypeError, ValueError)):
                    RUNNER._validate_config(
                        changed,
                        EXPERIMENT,
                        load_config_fn=lambda _path: canonical,
                    )

    def test_runner_freezes_one_canonical_output_family(self) -> None:
        self.assertEqual(RUNNER.CONFIRMATION_NAME, CONFIRMATION_NAME)
        paths = RUNNER._output_paths(EXPERIMENT)
        expected = {
            "predictions": EXPERIMENT
            / f"predictions/{CONFIRMATION_NAME}.parquet",
            "result": EXPERIMENT / f"results/{CONFIRMATION_NAME}.json",
            "completion": EXPERIMENT
            / f"receipts/{CONFIRMATION_NAME}.completion.json",
            "bundle": EXPERIMENT / f"models/{CONFIRMATION_NAME}",
        }
        self.assertEqual(paths, expected)

    def test_create_new_reservation_rejects_every_existing_destination(self) -> None:
        for occupied in ("predictions", "result", "completion", "bundle"):
            with self.subTest(occupied=occupied), tempfile.TemporaryDirectory() as tmp:
                experiment = Path(tmp)
                paths = RUNNER._output_paths(experiment)
                path = paths[occupied]
                path.parent.mkdir(parents=True, exist_ok=True)
                if occupied == "bundle":
                    path.mkdir()
                else:
                    path.write_bytes(b"preexisting evidence")

                with self.assertRaisesRegex(
                    (FileExistsError, ValueError),
                    "Cannot reserve exclusive (confirmation|prediction|result|completion)",
                ):
                    with RUNNER._ConfirmationOutputReservations(
                        experiment,
                        CONFIRMATION_NAME,
                    ):
                        self.fail("reservation unexpectedly opened")

                if occupied == "bundle":
                    self.assertTrue(path.is_dir())
                else:
                    self.assertEqual(path.read_bytes(), b"preexisting evidence")

    def test_runner_reserves_outputs_before_manifest_config_or_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp) / "experiment"
            numerai_dir = Path(tmp) / "numerai"
            paths = RUNNER._output_paths(experiment)
            paths["completion"].parent.mkdir(parents=True, exist_ok=True)
            paths["completion"].write_bytes(b"prior terminal receipt")

            with mock.patch.object(
                RUNNER, "_require_frozen_launch_policy"
            ), mock.patch.object(
                RUNNER,
                "verify_confirmation_manifest",
                side_effect=AssertionError("MANIFEST_OPENED"),
            ) as manifest_verifier, mock.patch.object(
                RUNNER,
                "_load_governed_runtime",
                side_effect=AssertionError("GOVERNED_RUNTIME_LOADED"),
            ) as runtime_loader:
                with self.assertRaisesRegex(
                    (FileExistsError, ValueError),
                    "Cannot reserve exclusive (confirmation|completion)",
                ):
                    RUNNER.run_confirmation(experiment, numerai_dir)

            manifest_verifier.assert_not_called()
            runtime_loader.assert_not_called()
            self.assertEqual(
                paths["completion"].read_bytes(), b"prior terminal receipt"
            )

    def test_prediction_projection_is_target_free_and_raw(self) -> None:
        confirmation = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "era": ["0865", "0865", "0869"],
                "feature_a": [0.1, 0.2, 0.3],
                "feature_b": [0.4, 0.5, 0.6],
                "v53_lgbm_ender20": [0.7, 0.8, 0.9],
            }
        )

        class RecordingModel:
            def __init__(self) -> None:
                self.seen = None

            def predict(self, frame):
                self.seen = frame.copy()
                return np.array([0.125, -0.5, 1.75])

        model = RecordingModel()
        output = RUNNER._predict_target_free(
            model,
            confirmation,
            ["feature_a", "feature_b"],
            "v53_lgbm_ender20",
            np_module=np,
            pd_module=pd,
        )
        self.assertIsNotNone(model.seen)
        self.assertEqual(
            list(model.seen.columns),
            ["feature_a", "feature_b", "era", "v53_lgbm_ender20"],
        )
        self.assertNotIn("target_ender_20", model.seen.columns)
        self.assertEqual(list(output.columns), ["id", "era", "prediction"])
        np.testing.assert_array_equal(
            output["prediction"].to_numpy(),
            np.array([0.125, -0.5, 1.75]),
        )

    def test_prediction_projection_rejects_materialized_target_before_predict(self) -> None:
        confirmation = pd.DataFrame(
            {
                "id": ["a"],
                "era": ["0865"],
                "feature_a": [0.1],
                "v53_lgbm_ender20": [0.7],
                "target_ender_20": [0.9],
            }
        )
        model = mock.Mock()
        with self.assertRaisesRegex(ValueError, "target|Target"):
            RUNNER._predict_target_free(
                model,
                confirmation,
                ["feature_a"],
                "v53_lgbm_ender20",
                np_module=np,
                pd_module=pd,
            )
        model.predict.assert_not_called()


class TestEnder21ConfirmationRules(unittest.TestCase):
    def test_all_frozen_checks_pass_at_inclusive_boundaries(self) -> None:
        discovery_bmc = 0.004
        metrics = _metrics(bmc=discovery_bmc * 0.60)
        checks = rules.confirmation_checks(metrics, discovery_bmc)
        self.assertEqual(
            set(checks),
            {
                "bmc_floor",
                "sharpe_floor",
                "drawdown_ceiling",
                "corr_floor",
                "benchmark_corr_ceiling",
                "positive_block_count",
                "worst_block_floor",
                "discovery_bmc_retention",
            },
        )
        self.assertTrue(all(checks.values()))

    def test_bmc_and_corr_floors_are_inclusive(self) -> None:
        checks = rules.confirmation_checks(
            _metrics(bmc=0.002, corr=0.008),
            0.003,
        )
        self.assertTrue(checks["bmc_floor"])
        self.assertTrue(checks["corr_floor"])
        self.assertTrue(checks["discovery_bmc_retention"])

    def test_strict_limits_fail_at_their_exact_boundaries(self) -> None:
        checks = rules.confirmation_checks(
            _metrics(
                bmc=0.004,
                sharpe=0.25,
                drawdown=0.10,
                benchmark_corr=0.25,
                blocks=(0.001, 0.001, 0.001, -0.001),
            ),
            0.004,
        )
        self.assertFalse(checks["sharpe_floor"])
        self.assertFalse(checks["drawdown_ceiling"])
        self.assertFalse(checks["benchmark_corr_ceiling"])
        self.assertFalse(checks["worst_block_floor"])

    def test_three_positive_blocks_pass_but_two_do_not(self) -> None:
        three = rules.confirmation_checks(
            _metrics(bmc=0.004, blocks=(0.001, 0.001, 0.001, -0.0005)),
            0.004,
        )
        two = rules.confirmation_checks(
            _metrics(bmc=0.004, blocks=(0.001, 0.001, 0.0, -0.0005)),
            0.004,
        )
        self.assertTrue(three["positive_block_count"])
        self.assertFalse(two["positive_block_count"])

    def test_rules_reject_bad_block_geometry_and_nonfinite_metrics(self) -> None:
        bad_blocks = _metrics(bmc=0.004)
        bad_blocks["chronological_block_bmc"].pop("3")
        nonfinite = _metrics(bmc=np.nan)
        for label, metrics in (("blocks", bad_blocks), ("finite", nonfinite)):
            with self.subTest(label=label):
                with self.assertRaises((TypeError, ValueError)):
                    rules.confirmation_checks(metrics, 0.004)

    def test_terminal_decision_has_only_the_frozen_research_states(self) -> None:
        passing_checks = rules.confirmation_checks(_metrics(bmc=0.004), 0.004)
        self.assertEqual(
            EVALUATOR._decision(passing_checks),
            {
                "passed": True,
                "state": "HISTORICAL_RESEARCH_PASS_FORWARD_VALIDATION_REQUIRED",
            },
        )
        failing_checks = dict(passing_checks)
        failing_checks["corr_floor"] = False
        self.assertEqual(
            EVALUATOR._decision(failing_checks),
            {"passed": False, "state": "NEGATIVE"},
        )

    def test_terminal_decision_rejects_schema_or_non_boolean_drift(self) -> None:
        passing_checks = rules.confirmation_checks(_metrics(bmc=0.004), 0.004)
        missing = dict(passing_checks)
        missing.pop("corr_floor")
        non_boolean = dict(passing_checks)
        non_boolean["corr_floor"] = 1
        for checks in (missing, non_boolean):
            with self.subTest(checks=checks):
                with self.assertRaises(ValueError):
                    EVALUATOR._decision(checks)


class TestEnder21ConfirmationCohort(unittest.TestCase):
    @staticmethod
    def _frame() -> pd.DataFrame:
        rows = []
        for era_index, era in enumerate(EXPECTED_CONFIRMATION_ERAS):
            for row_index in range(5):
                rows.append(
                    {
                        "id": f"{era}-{row_index}",
                        "era": era,
                        "target_ender_20": row_index / 4,
                        "v53_lgbm_ender20": ((row_index * 3) % 5) / 4,
                        "prediction": ((row_index + era_index) % 5) / 4,
                    }
                )
        return pd.DataFrame(rows)

    def test_validator_accepts_only_the_exact_finite_40_era_cohort(self) -> None:
        frame = self._frame()
        with mock.patch.object(EVALUATOR, "CONFIRMATION_ROWS", len(frame)):
            validated = EVALUATOR._validate_confirmation_frame(
                frame,
                EXPECTED_CONFIRMATION_ERAS,
            )
        self.assertEqual(len(validated), len(frame))
        self.assertEqual(
            tuple(validated["era"].drop_duplicates()),
            EXPECTED_CONFIRMATION_ERAS,
        )

    def test_validator_rejects_missing_extra_duplicate_or_nonfinite_rows(self) -> None:
        base = self._frame()
        cases = {
            "missing era": base[base["era"] != EXPECTED_CONFIRMATION_ERAS[-1]],
            "extra era": pd.concat(
                [
                    base,
                    base.iloc[[0]].assign(id="extra", era="1025"),
                ],
                ignore_index=True,
            ),
            "duplicate id": pd.concat([base, base.iloc[[0]]], ignore_index=True),
            "nonfinite target": base.assign(
                target_ender_20=lambda value: value["target_ender_20"].mask(
                    value.index == 0, np.nan
                )
            ),
        }
        for label, frame in cases.items():
            with self.subTest(label=label):
                with mock.patch.object(
                    EVALUATOR,
                    "CONFIRMATION_ROWS",
                    len(frame),
                ):
                    with self.assertRaises((TypeError, ValueError)):
                        EVALUATOR._validate_confirmation_frame(
                            frame,
                            EXPECTED_CONFIRMATION_ERAS,
                        )


class TestEnder21ConfirmationEvidenceOrder(unittest.TestCase):
    def test_evaluation_manifest_contract_is_exact_and_fail_closed(self) -> None:
        files = {
            relative: "a" * 64
            for relative in EVALUATOR.EXPECTED_EVALUATION_MANIFEST_FILES
        }
        canonical = {
            "schema_version": 1,
            "frozen_at": "2026-08-11",
            "git_head": "b" * 40,
            "hash_algorithm": "sha256",
            "files": files,
            "external_artifacts": EVALUATOR.EXPECTED_EVALUATION_EXTERNAL_ARTIFACTS,
            "runtime": EVALUATOR.EXPECTED_EVALUATION_RUNTIME,
            "training_authority": EVALUATOR.EXPECTED_TRAINING_AUTHORITY,
        }
        self.assertIs(
            EVALUATOR._validate_evaluation_manifest_schema(canonical), canonical
        )
        mutations = []
        wrong_files = deepcopy(canonical)
        wrong_files["files"].pop(next(iter(wrong_files["files"])))
        mutations.append(wrong_files)
        wrong_external = deepcopy(canonical)
        wrong_external["external_artifacts"] = {}
        mutations.append(wrong_external)
        wrong_authority = deepcopy(canonical)
        wrong_authority["training_authority"]["evidence_commit"] = "c" * 40
        mutations.append(wrong_authority)
        wrong_runtime = deepcopy(canonical)
        wrong_runtime["runtime"]["python"] = "0.0.0"
        mutations.append(wrong_runtime)
        for mutated in mutations:
            with self.subTest(keys=mutated.keys()), self.assertRaises(ValueError):
                EVALUATOR._validate_evaluation_manifest_schema(mutated)

    def test_actual_sealed_training_evidence_schema_passes_before_target(self) -> None:
        manifest = json.loads(
            (EXPERIMENT / "source_manifest_confirmation.json").read_text(
                encoding="utf-8"
            )
        )
        target_reader = mock.Mock(
            side_effect=AssertionError("CONFIRMATION_TARGET_OPENED")
        )
        with mock.patch.object(
            EVALUATOR,
            "_load_confirmation_truth",
            target_reader,
        ):
            completion = EVALUATOR._validate_training_evidence(
                EXPERIMENT, manifest
            )
        self.assertEqual(
            set(completion["era_contract"]),
            {"fit", "embargo", "confirmation"},
        )
        target_reader.assert_not_called()

    def test_evaluator_reserves_receipt_and_holds_data_leases_through_fsync(
        self,
    ) -> None:
        events = []
        leases = {}

        class Lease:
            def __init__(self, name: str, payload: bytes = b"") -> None:
                self.name = name
                self.payload = payload
                self.open = False
                leases[name] = self

            def __enter__(self):
                self.open = True
                events.append(f"lease_open:{self.name}")
                return self

            def __exit__(self, *_args):
                self.open = False
                events.append(f"lease_close:{self.name}")

            def read_bytes(self):
                self.assert_open()
                return self.payload

            def assert_open(self):
                if not self.open:
                    raise AssertionError(f"lease closed early: {self.name}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = (
                root
                / "numerai/agents/experiments/ender21_residual_stability_v53"
            )
            receipts = experiment / "receipts"
            receipts.mkdir(parents=True)
            (experiment / EVALUATOR.EVALUATION_MANIFEST_NAME).write_bytes(b"{}")
            numerai_dir = root / "numerai"
            output = receipts / "confirmation_research.json"
            expected_eras = ("0865", "0869")
            protocol_key = (
                "numerai/agents/experiments/ender21_residual_stability_v53/"
                "protocol/confirmation_eras_0865_through_1021.json"
            )
            evaluation_manifest = {"git_head": "a" * 40}
            training_manifest = {"git_head": "b" * 40}

            def bootstrap(stack, _experiment):
                events.append("bootstrap_sources")
                source = stack.enter_context(
                    Lease("source", json.dumps(expected_eras).encode("utf-8"))
                )
                return evaluation_manifest, training_manifest, {protocol_key: source}

            def acquire(stack, _experiment, _numerai_dir):
                events.append("acquire_evidence_data")
                return {
                    name: stack.enter_context(Lease(name))
                    for name in (
                        "completion",
                        "result",
                        "prediction",
                        "confirmation_benchmark",
                        "confirmation_full",
                    )
                }

            def require_open(stage):
                for name in (
                    "prediction",
                    "confirmation_benchmark",
                    "confirmation_full",
                ):
                    leases[name].assert_open()
                events.append(stage)

            prediction = pd.DataFrame(
                {
                    "id": ["a", "b"],
                    "era": list(expected_eras),
                    "prediction": [0.2, 0.8],
                }
            )
            benchmark = pd.DataFrame(
                {
                    "id": ["a", "b"],
                    "era": list(expected_eras),
                    "v53_lgbm_ender20": [0.3, 0.7],
                }
            )
            truth = pd.DataFrame(
                {
                    "id": ["a", "b"],
                    "era": list(expected_eras),
                    "target_ender_20": [0.1, 0.9],
                }
            )

            parquet_reads = iter((prediction, benchmark))

            def read_parquet(*_args, **_kwargs):
                require_open("prediction_or_benchmark_read")
                return next(parquet_reads).copy()

            def validate_evidence(*_args):
                require_open("evidence_validation")
                return {"sealed": True}

            def read_truth(*_args):
                require_open("target_read")
                return truth.copy()

            checks = {
                "bmc_floor": True,
                "sharpe_floor": True,
                "drawdown_ceiling": True,
                "corr_floor": True,
                "benchmark_corr_ceiling": True,
                "positive_block_count": True,
                "worst_block_floor": True,
                "discovery_bmc_retention": True,
            }

            def score(*_args):
                require_open("scoring")
                return {"metrics": {}, "checks": checks}, {}

            def validate_frame(frame, *_args):
                require_open("frame_validation")
                return frame

            def launch_policy():
                self.assertTrue(output.exists())
                events.append("receipt_reserved")

            def fsync(_fileno):
                require_open("receipt_fsync")

            def load_scoring():
                events.append("governed_scoring")
                return mock.Mock(), mock.Mock()

            with mock.patch.object(EVALUATOR, "REPO_DIR", root), mock.patch.object(
                EVALUATOR,
                "_require_frozen_launch_policy",
                side_effect=launch_policy,
            ), mock.patch.object(
                EVALUATOR,
                "_bootstrap_verify_and_lease_sources",
                side_effect=bootstrap,
            ), mock.patch.object(
                EVALUATOR,
                "_acquire_evaluation_leases",
                side_effect=acquire,
            ), mock.patch.object(
                EVALUATOR,
                "_validate_evidence_commit_leases",
            ), mock.patch.object(
                EVALUATOR,
                "_validate_evaluation_external_leases",
            ), mock.patch.object(
                EVALUATOR,
                "_load_governed_scoring",
                side_effect=load_scoring,
            ), mock.patch.object(
                EVALUATOR,
                "_validate_training_evidence",
                side_effect=validate_evidence,
            ), mock.patch.object(
                EVALUATOR.pd, "read_parquet", side_effect=read_parquet
            ), mock.patch.object(
                EVALUATOR,
                "_load_confirmation_truth",
                side_effect=read_truth,
            ), mock.patch.object(
                EVALUATOR,
                "_validate_confirmation_frame",
                side_effect=validate_frame,
            ), mock.patch.object(
                EVALUATOR, "_score_confirmation", side_effect=score
            ), mock.patch.object(
                EVALUATOR, "CONFIRMATION_ROWS", 2
            ), mock.patch.object(
                EVALUATOR, "EXPECTED_CONFIRMATION_ERAS", expected_eras
            ), mock.patch.object(EVALUATOR.os, "fsync", side_effect=fsync):
                EVALUATOR.main(
                    [
                        "--experiment",
                        str(experiment),
                        "--numerai-dir",
                        str(numerai_dir),
                        "--output",
                        str(output),
                    ]
                )

            fsync_index = events.index("receipt_fsync")
            for name in (
                "prediction",
                "confirmation_benchmark",
                "confirmation_full",
            ):
                self.assertGreater(events.index(f"lease_close:{name}"), fsync_index)
            self.assertLess(events.index("receipt_reserved"), events.index("bootstrap_sources"))
            self.assertLess(events.index("bootstrap_sources"), events.index("acquire_evidence_data"))
            self.assertLess(events.index("acquire_evidence_data"), events.index("governed_scoring"))
            self.assertLess(events.index("governed_scoring"), events.index("evidence_validation"))
            self.assertLess(events.index("evidence_validation"), events.index("target_read"))
            self.assertLess(events.index("target_read"), events.index("scoring"))

    def test_uncommitted_training_evidence_fails_before_target_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = (
                root
                / "numerai/agents/experiments/ender21_residual_stability_v53"
            )
            experiment.mkdir(parents=True)
            target_reader = mock.Mock(
                side_effect=AssertionError("CONFIRMATION_TARGET_OPENED")
            )
            fake_evaluation_manifest = {"git_head": "a" * 40}
            fake_training_manifest = {"git_head": "b" * 40}
            fake_leases = {"completion": mock.Mock()}
            with mock.patch.object(EVALUATOR, "REPO_DIR", root), mock.patch.object(
                EVALUATOR,
                "_bootstrap_verify_and_lease_sources",
                return_value=(fake_evaluation_manifest, fake_training_manifest, {}),
            ), mock.patch.object(
                EVALUATOR,
                "_acquire_evaluation_leases",
                return_value=fake_leases,
            ), mock.patch.object(
                EVALUATOR,
                "_validate_evidence_commit_leases",
            ), mock.patch.object(
                EVALUATOR,
                "_validate_evaluation_external_leases",
            ), mock.patch.object(
                EVALUATOR,
                "_load_governed_scoring",
                return_value=(mock.Mock(), mock.Mock()),
            ), mock.patch.object(
                EVALUATOR,
                "_validate_training_evidence",
                side_effect=ValueError("confirmation evidence is not committed"),
            ) as evidence_validator, mock.patch.object(
                EVALUATOR,
                "_load_confirmation_truth",
                target_reader,
            ):
                with self.assertRaisesRegex(ValueError, "not committed"):
                    EVALUATOR._evaluate_confirmation(
                        experiment,
                        root / "numerai",
                    )

            evidence_validator.assert_called_once_with(
                experiment, fake_training_manifest, fake_leases
            )
            target_reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
