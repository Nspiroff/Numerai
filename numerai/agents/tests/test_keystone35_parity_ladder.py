"""Protected source-contract tests for the Keystone parity-calibration ladder (KP35).

Deterministic, CPU-only and dataset-free. Every test below runs from synthetic
fixtures, the frozen law module, the protocol record and the packet source text
alone: no LightGBM, no PyArrow, no NumerAPI, no network, no local dataset, no
GPU and no external artifact is required, which is what lets the full suite run
on a bare protected CI runner.

The suite proves the frozen scientific design (boundary, purge, score zone, the
two profiles and their single declared difference), the stage ordering and
eligibility law, the exact screening and two-part confirmation arithmetic, the
exact-row universe contract that prospectively repairs the KW33 source-contract
gap, sample custody and its parameter independence, and the structural absences
that matter: no Candidate-V stage, no Ender60 selection path, no bare ``target``
alias, no backward terminal transition, and no automatic next-stage execution.
"""

from __future__ import annotations

import ast
import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from agents.experiments.keystone28_corr_backbone_v53 import round2_parity_lib as kp

PACKET_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "keystone28_corr_backbone_v53"
)
PROTOCOL_PATH = PACKET_DIR / "round2_parity_protocol.json"
TRAIN_SRC = (PACKET_DIR / "round2_parity_train.py").read_text(encoding="utf-8")
EVAL_SRC = (PACKET_DIR / "round2_parity_evaluate.py").read_text(encoding="utf-8")
LIB_SRC = (PACKET_DIR / "round2_parity_lib.py").read_text(encoding="utf-8")
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

TREES = {
    "lib": ast.parse(LIB_SRC),
    "train": ast.parse(TRAIN_SRC),
    "evaluate": ast.parse(EVAL_SRC),
}


def _imported_roots(tree: ast.AST) -> set[str]:
    """Top-level package names this module actually imports."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _identifiers(tree: ast.AST) -> set[str]:
    """Every name, attribute, function, class and keyword argument identifier.

    Structural rather than textual: prose in a docstring can never satisfy
    these tests, and an actual call or import always will.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def _short_string_constants(tree: ast.AST, limit: int = 64) -> set[str]:
    """String literals short enough to be labels rather than prose."""
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and len(node.value) <= limit
    }


def _calls_inside_loops(tree: ast.AST) -> set[str]:
    """Names of functions invoked anywhere inside a ``for``/``while`` body."""
    inside: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    func = inner.func
                    if isinstance(func, ast.Name):
                        inside.add(func.id)
                    elif isinstance(func, ast.Attribute):
                        inside.add(func.attr)
    return inside


def _call_count(tree: ast.AST, name: str) -> int:
    total = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == name:
                total += 1
            elif isinstance(func, ast.Attribute) and func.attr == name:
                total += 1
    return total


def _universe(name, pairs):
    return kp.RowUniverse.from_columns(name, [e for e, _ in pairs], [i for _, i in pairs])


def _zone_pairs(per_era=3):
    return [
        (era, f"id_{era}_{k}")
        for era in kp.score_zone_eras()
        for k in range(per_era)
    ]


def _manifest(**overrides):
    identities = {"train.parquet": "a" * 64, "validation.parquet": "b" * 64}
    canon = overrides.pop("sample_canon_sha256", "c" * 64)
    per_era = overrides.pop("selected_rows_per_era", {"0001": 4, "1084": 6})
    base = {
        "record": "kp35_sample_identity",
        "data_identities": identities,
        "eligible_era_range": ["0001", "1084"],
        "n_eligible_eras": 1084,
        "feature_set": kp.FEATURE_SET,
        "feature_list_sha256": kp.FEATURE_LIST_SHA256,
        "n_features": kp.N_FEATURES,
        "sampling_law_version": kp.SAMPLING_LAW_VERSION,
        "sampling_seed": kp.SAMPLING_SEED,
        "row_cap": kp.MAX_SAMPLED_ROWS,
        "rows_before_sampling": 5_890_287,
        "selected_row_count": sum(per_era.values()),
        "selected_rows_per_era": per_era,
        "source_split_rows": {"train": 4, "validation": 6},
        "sample_canon_sha256": canon,
    }
    base["sample_identity_sha256"] = kp.sample_identity(
        data_identities=base["data_identities"],
        eligible_era_range=base["eligible_era_range"],
        feature_list_sha256=base["feature_list_sha256"],
        sampling_law_version=base["sampling_law_version"],
        sampling_seed=base["sampling_seed"],
        row_cap=base["row_cap"],
        sample_canon_sha256=base["sample_canon_sha256"],
    )
    base.update(overrides)
    return base


# ============================================================ frozen boundaries
class TestFrozenBoundaries(unittest.TestCase):
    def test_1_p1_training_eras_end_exactly_at_1084(self):
        eligible = kp.eligible_training_eras()
        self.assertEqual(eligible[0], "0001")
        self.assertEqual(eligible[-1], "1084")
        self.assertEqual(len(eligible), 1084)
        self.assertEqual(kp.HISTORY_BOUNDARY_END, 1084)
        kp.assert_training_eras_authorized(eligible, context="test")
        with self.assertRaises(ValueError):
            kp.assert_training_eras_authorized(eligible + ["1085"], context="test")

    def test_1b_boundary_is_rederived_from_documented_window_arithmetic(self):
        derived = kp.derive_history_boundary()
        self.assertEqual(derived["window"], 7)
        self.assertEqual(derived["train_end"], 1084)
        self.assertEqual(derived["purge_start"], 1085)
        self.assertEqual(derived["purge_end"], 1092)
        self.assertEqual(derived["predict_start"], 1093)
        self.assertEqual(derived["predict_end"], 1248)
        self.assertLessEqual(derived["predict_start"], kp.SCORE_ZONE_START)
        self.assertGreaterEqual(derived["predict_end"], kp.SCORE_ZONE_END)

    def test_2_purge_is_exactly_1085_through_1092(self):
        purge = kp.purge_eras()
        self.assertEqual(purge, [f"{e:04d}" for e in range(1085, 1093)])
        self.assertEqual(len(purge), kp.PURGE_ERAS)
        self.assertEqual(kp.PURGE_ERAS, 8)
        for era in purge:
            with self.assertRaises(ValueError):
                kp.assert_training_eras_authorized([era], context="purge")

    def test_3_scoring_zone_is_exactly_1133_through_1219(self):
        zone = kp.score_zone_eras()
        self.assertEqual(zone[0], "1133")
        self.assertEqual(zone[-1], "1219")
        self.assertEqual(len(zone), 87)
        self.assertEqual(kp.N_SCORE_ERAS, 87)
        kp.assert_scoring_zone_exact(zone, context="test")
        with self.assertRaises(ValueError):
            kp.assert_scoring_zone_exact(zone[:-1], context="subset")
        with self.assertRaises(ValueError):
            kp.assert_scoring_zone_exact(zone + ["1220"], context="superset")

    def test_3b_benchmark_chunk_contains_the_zone(self):
        chunk = kp.benchmark_chunk_eras()
        self.assertEqual(chunk[0], "1093")
        self.assertEqual(chunk[-1], "1248")
        self.assertEqual(len(chunk), 156)
        self.assertTrue(set(kp.score_zone_eras()).issubset(set(chunk)))

    def test_4_gap_and_holdout_are_rejected_everywhere(self):
        for era in ("1223", "1227", "1230", "1231", "1400"):
            with self.assertRaises(ValueError):
                kp.assert_no_forbidden_eras([era], context="guard")
            with self.assertRaises(ValueError):
                kp.assert_training_eras_authorized([era], context="guard")
            with self.assertRaises(ValueError):
                kp.assert_scoring_zone_exact(kp.score_zone_eras() + [era], context="guard")
        # The forbidden-era guard fires before any universe comparison, so it
        # raises the base ValueError rather than the comparison subclass.
        with self.assertRaises(ValueError):
            kp.assert_exact_row_universe(
                _universe("gap", [("1223", "x")]),
                _universe("gap2", [("1223", "x")]),
            )

    def test_4b_training_era_source_split_is_exact(self):
        self.assertEqual(kp.training_era_source("0001"), "train")
        self.assertEqual(kp.training_era_source("0574"), "train")
        self.assertEqual(kp.training_era_source("0575"), "validation")
        self.assertEqual(kp.training_era_source("1084"), "validation")
        for era in ("1085", "1133", "1223", "1231"):
            with self.assertRaises(ValueError):
                kp.training_era_source(era)


# ================================================================== profiles
class TestProfiles(unittest.TestCase):
    def test_5_p1_is_the_exact_kw33_fallback_profile(self):
        p1 = kp.profile_for(kp.P1)
        self.assertEqual(p1["name"], "FALLBACK")
        self.assertEqual(p1["objective"], "regression")
        self.assertEqual(p1["num_trees"], 6000)
        self.assertEqual(p1["learning_rate"], 0.005)
        self.assertEqual(p1["max_depth"], 8)
        self.assertEqual(p1["num_leaves"], 255)
        self.assertEqual(p1["min_data_in_leaf"], 10000)
        self.assertEqual(p1["feature_fraction"], 0.1)
        self.assertEqual(p1["num_threads"], 8)
        self.assertIs(p1["deterministic"], True)
        self.assertIs(p1["force_row_wise"], True)
        self.assertEqual(p1["device"], "cpu")
        self.assertIs(p1["no_early_stopping"], True)
        self.assertIs(p1["no_evaluation_set"], True)

    def test_6_p2_is_the_exact_documented_v5_deep_profile(self):
        p2 = kp.profile_for(kp.P2)
        self.assertEqual(p2["num_trees"], 30000)
        self.assertEqual(p2["learning_rate"], 0.001)
        self.assertEqual(p2["max_depth"], 10)
        self.assertEqual(p2["num_leaves"], 1024)
        self.assertEqual(p2["min_data_in_leaf"], 10000)
        self.assertEqual(p2["feature_fraction"], 0.1)
        self.assertEqual(p2["num_threads"], 8)
        self.assertIs(p2["deterministic"], True)
        self.assertIs(p2["force_row_wise"], True)
        self.assertEqual(p2["device"], "cpu")
        self.assertIs(p2["no_early_stopping"], True)
        self.assertIs(p2["no_evaluation_set"], True)

    def test_7_p2_differs_from_p1_only_in_declared_profile_fields(self):
        difference = kp.assert_declared_profile_difference()
        self.assertEqual(
            set(difference),
            {"name", "num_trees", "learning_rate", "max_depth", "num_leaves"},
        )
        self.assertEqual(difference["num_trees"], (6000, 30000))
        self.assertEqual(difference["learning_rate"], (0.005, 0.001))
        self.assertEqual(difference["max_depth"], (8, 10))
        self.assertEqual(difference["num_leaves"], (255, 1024))
        for shared in (
            "objective", "min_data_in_leaf", "feature_fraction", "num_threads",
            "deterministic", "force_row_wise", "device", "no_early_stopping",
            "no_evaluation_set",
        ):
            self.assertNotIn(shared, difference)

    def test_7b_undeclared_profile_divergence_is_refused(self):
        tampered = dict(kp.P2_PROFILE)
        tampered["feature_fraction"] = 0.2
        difference = kp.profile_difference(kp.P1_PROFILE, tampered)
        self.assertIn("feature_fraction", difference)
        self.assertFalse(
            set(difference).issubset(kp.DECLARED_PROFILE_DIFFERENCE_FIELDS)
        )

    def test_7c_params_carry_no_early_stopping_or_eval_set(self):
        for stage in kp.STAGES:
            params = kp.lightgbm_params(kp.profile_for(stage), kp.SCREENING_SEED)
            self.assertEqual(params["seed"], kp.SCREENING_SEED)
            self.assertIs(params["deterministic"], True)
            self.assertIs(params["force_row_wise"], True)
            for forbidden in (
                "early_stopping_round", "early_stopping_rounds",
                "valid_sets", "eval_set", "first_metric_only",
            ):
                self.assertNotIn(forbidden, params)

    def test_7d_only_the_seed_varies_between_matched_fits(self):
        a = kp.lightgbm_params(kp.profile_for(kp.P1), 42)
        b = kp.lightgbm_params(kp.profile_for(kp.P1), 1337)
        self.assertNotEqual(a["seed"], b["seed"])
        a.pop("seed"), b.pop("seed")
        self.assertEqual(a, b)


# ==================================================== stage ordering/eligibility
class TestStageAuthority(unittest.TestCase):
    def test_8_p2_requires_a_recorded_p1_failure_state(self):
        kp.assert_stage_executable(kp.P2, kp.KP35_P1_SCREEN_FAILED)
        for bad in (
            None,
            kp.KP35_SOURCE_FROZEN,
            kp.KP35_P1_SCREEN_PASSED,
            kp.KP35_P2_SCREEN_PASSED,
            kp.KP35_PARITY_NOT_RESTORED,
            kp.KP35_PARITY_CONFIRMED,
        ):
            with self.assertRaises(kp.StageAuthorityError):
                kp.assert_stage_executable(kp.P2, bad)

    def test_8b_p1_is_the_first_stage(self):
        kp.assert_stage_executable(kp.P1, None)
        kp.assert_stage_executable(kp.P1, kp.KP35_SOURCE_FROZEN)
        for bad in (kp.KP35_P1_SCREEN_FAILED, kp.KP35_P1_SCREEN_PASSED):
            with self.assertRaises(kp.StageAuthorityError):
                kp.assert_stage_executable(kp.P1, bad)

    def test_9_confirmation_requires_a_screen_pass(self):
        kp.assert_confirmation_authorized(kp.KP35_P1_SCREEN_PASSED)
        kp.assert_confirmation_authorized(kp.KP35_P2_SCREEN_PASSED)
        for bad in (
            None,
            kp.KP35_SOURCE_FROZEN,
            kp.KP35_P1_SCREEN_FAILED,
            kp.KP35_PARITY_NOT_RESTORED,
            kp.KP35_CONFIRMATION_FAILED,
        ):
            with self.assertRaises(kp.StageAuthorityError):
                kp.assert_confirmation_authorized(bad)

    def test_10_seed_42_is_the_only_screening_seed(self):
        self.assertEqual(kp.SCREENING_SEED, 42)
        for stage in kp.STAGES:
            kp.assert_stage_seed(stage, 42, screening=True)
            for bad in (1337, 2024, 0, 7, 2027):
                with self.assertRaises(kp.StageAuthorityError):
                    kp.assert_stage_seed(stage, bad, screening=True)

    def test_11_only_1337_and_2024_are_confirmation_seeds(self):
        self.assertEqual(kp.CONFIRMATION_SEEDS, (1337, 2024))
        self.assertEqual(kp.ALL_SEEDS, (42, 1337, 2024))
        for stage in kp.STAGES:
            for good in kp.CONFIRMATION_SEEDS:
                kp.assert_stage_seed(stage, good, screening=False)
            for bad in (42, 2027, 1, 20260817):
                with self.assertRaises(kp.StageAuthorityError):
                    kp.assert_stage_seed(stage, bad, screening=False)

    def test_11b_unknown_seeds_cannot_produce_params_or_paths(self):
        for bad in (0, 7, 2027):
            with self.assertRaises(kp.StageAuthorityError):
                kp.lightgbm_params(kp.profile_for(kp.P1), bad)
            with self.assertRaises(kp.StageAuthorityError):
                kp.artifact_relpath("prediction", stage=kp.P1, model_seed=bad)


# ================================================================ threshold law
class TestThresholdArithmetic(unittest.TestCase):
    def test_12_screen_threshold_arithmetic_is_exact(self):
        self.assertEqual(kp.BENCHMARK_MEAN_CORR, 0.02094843151562169)
        self.assertEqual(kp.SCREEN_FACTOR, 0.6755)
        self.assertEqual(kp.SCREEN_THRESHOLD, 0.014150665488802451)
        self.assertEqual(
            kp.screen_threshold(kp.BENCHMARK_MEAN_CORR), kp.SCREEN_THRESHOLD
        )
        self.assertEqual(
            kp.SCREEN_FACTOR * kp.BENCHMARK_MEAN_CORR, 0.014150665488802451
        )

    def test_12b_screen_factor_derivation_is_exact(self):
        self.assertEqual(kp.SCREEN_SEED_DISPERSION_ALLOWANCE, 0.0350)
        self.assertEqual(
            kp.FINAL_PARITY_FRACTION * (1 - kp.SCREEN_SEED_DISPERSION_ALLOWANCE),
            kp.SCREEN_FACTOR,
        )
        self.assertLess(kp.SCREEN_THRESHOLD, kp.FINAL_THREE_SEED_THRESHOLD)

    def test_12c_screen_law_is_a_closed_inequality_on_corr_only(self):
        threshold = kp.SCREEN_THRESHOLD
        for stage in kp.STAGES:
            self.assertTrue(kp.screen_stage(stage, threshold)["passed"])
            self.assertTrue(kp.screen_stage(stage, threshold + 1e-9)["passed"])
            self.assertFalse(kp.screen_stage(stage, threshold - 1e-9)["passed"])
        self.assertEqual(kp.screen_stage(kp.P1, 0.0)["terminal_state"],
                         kp.KP35_P1_SCREEN_FAILED)
        self.assertEqual(kp.screen_stage(kp.P1, 1.0)["terminal_state"],
                         kp.KP35_P1_SCREEN_PASSED)
        self.assertEqual(kp.screen_stage(kp.P2, 0.0)["terminal_state"],
                         kp.KP35_PARITY_NOT_RESTORED)
        self.assertEqual(kp.screen_stage(kp.P2, 1.0)["terminal_state"],
                         kp.KP35_P2_SCREEN_PASSED)

    def test_12d_p2_uses_the_same_screen_threshold_as_p1(self):
        self.assertEqual(
            kp.screen_stage(kp.P1, 0.01)["threshold"],
            kp.screen_stage(kp.P2, 0.01)["threshold"],
        )

    def test_13_three_seed_final_threshold_arithmetic_is_exact(self):
        self.assertEqual(kp.FINAL_PARITY_FRACTION, 0.70)
        self.assertEqual(kp.FINAL_THREE_SEED_THRESHOLD, 0.014663902060935183)
        self.assertEqual(
            kp.FINAL_PARITY_FRACTION * kp.BENCHMARK_MEAN_CORR, 0.014663902060935183
        )
        self.assertEqual(kp.final_threshold(), kp.FINAL_THREE_SEED_THRESHOLD)
        self.assertEqual(kp.UNTOUCHED_PAIR_THRESHOLD, kp.FINAL_THREE_SEED_THRESHOLD)
        out = kp.final_confirmation(0.02, 0.02, 0.02)
        self.assertEqual(out["threshold"], 0.014663902060935183)
        self.assertAlmostEqual(out["three_seed_mean"], 0.02, places=15)

    def test_14_untouched_pair_gate_is_independently_required(self):
        t = kp.FINAL_THREE_SEED_THRESHOLD
        # Three-seed mean clears the bar only because seed 42 is very strong;
        # the untouched pair does not, so confirmation must fail.
        out = kp.final_confirmation(t * 2.0, t * 0.6, t * 0.6)
        self.assertTrue(out["three_seed_gate_passed"])
        self.assertFalse(out["untouched_pair_gate_passed"])
        self.assertFalse(out["confirmed"])
        self.assertEqual(out["terminal_state"], kp.KP35_CONFIRMATION_FAILED)
        self.assertIs(out["both_required"], True)

    def test_15_either_final_gate_failing_prevents_confirmation(self):
        t = kp.FINAL_THREE_SEED_THRESHOLD
        both = kp.final_confirmation(t, t, t)
        self.assertTrue(both["confirmed"])
        self.assertEqual(both["terminal_state"], kp.KP35_PARITY_CONFIRMED)

        # Pair passes, three-seed mean fails (seed 42 drags it under).
        pair_only = kp.final_confirmation(0.0, t, t)
        self.assertTrue(pair_only["untouched_pair_gate_passed"])
        self.assertFalse(pair_only["three_seed_gate_passed"])
        self.assertFalse(pair_only["confirmed"])

        # Three-seed passes, pair fails.
        three_only = kp.final_confirmation(t * 2.0, t * 0.6, t * 0.6)
        self.assertTrue(three_only["three_seed_gate_passed"])
        self.assertFalse(three_only["untouched_pair_gate_passed"])
        self.assertFalse(three_only["confirmed"])

        # Neither passes.
        neither = kp.final_confirmation(0.0, 0.0, 0.0)
        self.assertFalse(neither["confirmed"])
        self.assertEqual(neither["terminal_state"], kp.KP35_CONFIRMATION_FAILED)

    def test_15b_confirmation_grants_no_promotion_and_no_candidate_v_return(self):
        out = kp.final_confirmation(1.0, 1.0, 1.0)
        self.assertTrue(out["confirmed"])
        self.assertIs(out["promotion_granted"], False)
        self.assertIs(out["candidate_v_return_authorized"], False)

    def test_15c_benchmark_identity_is_enforced_within_tolerance(self):
        self.assertEqual(
            kp.assert_benchmark_identity(kp.BENCHMARK_MEAN_CORR), kp.BENCHMARK_MEAN_CORR
        )
        kp.assert_benchmark_identity(kp.BENCHMARK_MEAN_CORR + 1e-13)
        with self.assertRaises(ValueError):
            kp.assert_benchmark_identity(kp.BENCHMARK_MEAN_CORR + 1e-9)
        with self.assertRaises(ValueError):
            kp.assert_benchmark_identity(kp.BENCHMARK_MEAN_CORR * 1.01)


# ========================================================= exact-row contract (J)
class TestExactRowUniverse(unittest.TestCase):
    def setUp(self):
        self.pairs = _zone_pairs(2)
        self.reference = _universe("scoring", self.pairs)

    def test_16_accepts_a_complete_shuffled_universe(self):
        shuffled = list(self.pairs)
        rng = np.random.default_rng(7)
        rng.shuffle(shuffled)
        self.assertNotEqual(shuffled, self.pairs)
        other = _universe("prediction", shuffled)
        out = kp.assert_exact_row_universe(
            self.reference, other, expected_eras=kp.score_zone_eras()
        )
        self.assertTrue(out["identical"])
        self.assertEqual(out["n_rows"], len(self.pairs))
        self.assertEqual(other.canon_sha256, self.reference.canon_sha256)

    def test_17_rejects_a_strict_subset(self):
        subset = _universe("prediction", self.pairs[:-1])
        with self.assertRaises(kp.ExactRowUniverseError) as ctx:
            kp.assert_exact_row_universe(self.reference, subset)
        self.assertIn("row count", str(ctx.exception))

    def test_17b_rejects_a_same_size_subset_with_a_swapped_row(self):
        swapped = list(self.pairs[:-1]) + [(self.pairs[-1][0], "id_not_in_universe")]
        other = _universe("prediction", swapped)
        self.assertEqual(other.n_rows, self.reference.n_rows)
        with self.assertRaises(kp.ExactRowUniverseError) as ctx:
            kp.assert_exact_row_universe(self.reference, other)
        self.assertIn("universe differs", str(ctx.exception))

    def test_18_rejects_a_strict_superset(self):
        superset = _universe("prediction", self.pairs + [("1133", "extra_row")])
        with self.assertRaises(kp.ExactRowUniverseError):
            kp.assert_exact_row_universe(self.reference, superset)

    def test_19_rejects_duplicate_ids(self):
        duplicated = _universe("prediction", self.pairs + [self.pairs[0]])
        with self.assertRaises(kp.ExactRowUniverseError) as ctx:
            kp.assert_exact_row_universe(self.reference, duplicated)
        self.assertIn("duplicate ids", str(ctx.exception))
        self.assertEqual(duplicated.duplicate_ids(), [self.pairs[0][1]])

    def test_20_rejects_era_disagreement(self):
        moved = [(("1134" if era == "1133" else era), ident) for era, ident in self.pairs]
        other = _universe("prediction", moved)
        self.assertEqual(other.n_rows, self.reference.n_rows)
        with self.assertRaises(kp.ExactRowUniverseError):
            kp.assert_exact_row_universe(self.reference, other)

    def test_20b_rejects_a_row_carrying_an_unexpected_era(self):
        other = _universe("prediction", self.pairs + [("1220", "x")])
        with self.assertRaises(kp.ExactRowUniverseError) as ctx:
            kp.assert_exact_row_universe(
                self.reference, other, expected_eras=kp.score_zone_eras()
            )
        self.assertIn("unexpected eras", str(ctx.exception))

    def test_20c_rejects_per_era_count_disagreement_at_equal_totals(self):
        rebalanced = [p for p in self.pairs if p[0] != "1133"]
        rebalanced += [("1134", "moved_a"), ("1134", "moved_b")]
        other = _universe("prediction", rebalanced)
        self.assertEqual(other.n_rows, self.reference.n_rows)
        with self.assertRaises(kp.ExactRowUniverseError):
            kp.assert_exact_row_universe(self.reference, other)

    def test_20d_rejects_non_finite_predictions(self):
        kp.assert_finite_predictions([0.1, -0.2, 0.0], context="ok")
        for bad in ([0.1, float("nan")], [float("inf"), 0.2], [float("-inf")]):
            with self.assertRaises(ValueError):
                kp.assert_finite_predictions(bad, context="bad")
        with self.assertRaises(ValueError):
            kp.assert_finite_predictions([], context="empty")

    def test_20e_expected_rows_and_hash_expectations_are_enforced(self):
        with self.assertRaises(kp.ExactRowUniverseError):
            kp.assert_exact_row_universe(self.reference, expected_rows=999999)
        with self.assertRaises(kp.ExactRowUniverseError):
            kp.assert_exact_row_universe(self.reference, expected_canon_sha256="0" * 64)

    def test_20f_frozen_scoring_universe_identity_is_recorded(self):
        self.assertEqual(kp.SCORING_UNIVERSE_ROWS, 575_597)
        self.assertEqual(
            kp.SCORING_UNIVERSE_CANON_SHA256,
            "91e519aff5c656c9acf7cc6fe74daebfc034650bae47ee4e3889a98ec8fac033",
        )


# ================================================================ canonical keys
class TestCanonicalHashing(unittest.TestCase):
    def test_21_canonical_hashes_are_order_invariant_and_content_sensitive(self):
        eras = ["1133", "1133", "1134"]
        ids = ["b", "a", "c"]
        base = kp.canonical_key_hash(eras, ids)
        permuted = kp.canonical_key_hash(["1134", "1133", "1133"], ["c", "a", "b"])
        self.assertEqual(base, permuted)

        self.assertNotEqual(base, kp.canonical_key_hash(eras, ["b", "a", "d"]))
        self.assertNotEqual(base, kp.canonical_key_hash(["1133", "1134", "1134"], ids))
        self.assertNotEqual(base, kp.canonical_key_hash(eras + ["1135"], ids + ["e"]))
        self.assertNotEqual(base, kp.canonical_key_hash(eras[:-1], ids[:-1]))

        self.assertEqual(
            kp.canonical_keys(eras, ids),
            (("1133", "a"), ("1133", "b"), ("1134", "c")),
        )
        self.assertEqual(len(base), 64)
        with self.assertRaises(ValueError):
            kp.canonical_key_hash(["1133"], ["a", "b"])

    def test_21b_per_era_counts_are_exact(self):
        counts = kp.per_era_counts(["1133", "1133", "1134"])
        self.assertEqual(counts, {"1133": 2, "1134": 1})


# ============================================================== sample custody
class TestSampleCustody(unittest.TestCase):
    def test_22_sample_identity_is_independent_of_model_seed(self):
        signature = set(inspect.signature(kp.sample_identity).parameters)
        for excluded in ("model_seed", "stage", "profile", "num_trees", "learning_rate"):
            self.assertNotIn(excluded, signature)
        kwargs = dict(
            data_identities={"train.parquet": "a" * 64},
            eligible_era_range=["0001", "1084"],
            feature_list_sha256=kp.FEATURE_LIST_SHA256,
            sampling_law_version=kp.SAMPLING_LAW_VERSION,
            sampling_seed=kp.SAMPLING_SEED,
            row_cap=kp.MAX_SAMPLED_ROWS,
            sample_canon_sha256="c" * 64,
        )
        self.assertEqual(kp.sample_identity(**kwargs), kp.sample_identity(**kwargs))

    def test_22b_manifests_carrying_model_fields_are_refused(self):
        for leaked in ("model_seed", "stage", "num_trees", "params_sha256"):
            with self.assertRaises(ValueError) as ctx:
                kp.validate_sample_manifest(_manifest(**{leaked: 42}))
            self.assertIn("parameter-dependent", str(ctx.exception))

    def test_22c_sample_identity_changes_with_every_bound_input(self):
        base = dict(
            data_identities={"train.parquet": "a" * 64},
            eligible_era_range=["0001", "1084"],
            feature_list_sha256=kp.FEATURE_LIST_SHA256,
            sampling_law_version=kp.SAMPLING_LAW_VERSION,
            sampling_seed=kp.SAMPLING_SEED,
            row_cap=kp.MAX_SAMPLED_ROWS,
            sample_canon_sha256="c" * 64,
        )
        reference = kp.sample_identity(**base)
        for field, value in (
            ("data_identities", {"train.parquet": "z" * 64}),
            ("eligible_era_range", ["0001", "1124"]),
            ("feature_list_sha256", "f" * 64),
            ("sampling_law_version", "other_law"),
            ("sampling_seed", 1),
            ("row_cap", 500_000),
            ("sample_canon_sha256", "d" * 64),
        ):
            self.assertNotEqual(reference, kp.sample_identity(**{**base, field: value}))

    def test_23_p1_and_p2_require_identical_sample_hashes(self):
        p1 = _manifest()
        p2 = _manifest()
        self.assertEqual(kp.assert_shared_sample_identity(p1, p2), p1["sample_identity_sha256"])
        divergent = _manifest(sample_canon_sha256="e" * 64)
        with self.assertRaises(ValueError) as ctx:
            kp.assert_shared_sample_identity(p1, divergent)
        self.assertIn("different (era,id) universe", str(ctx.exception))

    def test_23b_manifest_schema_and_arithmetic_are_validated(self):
        self.assertTrue(kp.validate_sample_manifest(_manifest())["valid"])
        for field in kp.SAMPLE_MANIFEST_REQUIRED_FIELDS:
            broken = _manifest()
            broken.pop(field)
            with self.assertRaises(ValueError):
                kp.validate_sample_manifest(broken)
        with self.assertRaises(ValueError):
            kp.validate_sample_manifest(_manifest(selected_row_count=999))
        with self.assertRaises(ValueError):
            kp.validate_sample_manifest(
                _manifest(source_split_rows={"train": 1, "validation": 1})
            )
        with self.assertRaises(ValueError):
            kp.validate_sample_manifest(_manifest(sampling_seed=1))
        with self.assertRaises(ValueError):
            kp.validate_sample_manifest(_manifest(row_cap=42))
        with self.assertRaises(ValueError):
            kp.validate_sample_manifest(_manifest(sample_identity_sha256="0" * 64))

    def test_23c_manifest_rejects_forbidden_training_eras(self):
        with self.assertRaises(ValueError):
            kp.validate_sample_manifest(
                _manifest(selected_rows_per_era={"0001": 4, "1085": 6})
            )
        with self.assertRaises(ValueError):
            kp.validate_sample_manifest(
                _manifest(selected_rows_per_era={"0001": 4, "1231": 6})
            )

    def test_23d_sampling_law_is_deterministic_capped_and_seed_independent(self):
        eras = np.array([f"{e:04d}" for e in range(1, 11) for _ in range(50)])
        under_cap = kp.era_balanced_sample_positions(eras, cap=10_000)
        self.assertEqual(len(under_cap), len(eras))

        first = kp.era_balanced_sample_positions(eras, cap=100)
        second = kp.era_balanced_sample_positions(eras, cap=100)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 100)
        counts = kp.per_era_counts(eras[first].tolist())
        self.assertEqual(sorted(set(counts.values())), [10])
        # The law reads only the sampling seed and the era, never a model seed.
        law = inspect.signature(kp.era_balanced_sample_positions)
        self.assertEqual(set(law.parameters), {"era_of_row", "cap", "seed"})
        self.assertEqual(law.parameters["seed"].default, kp.SAMPLING_SEED)

    def test_23e_remainder_is_granted_in_ascending_era_order(self):
        eras = np.array(["0001"] * 10 + ["0002"] * 10 + ["0003"] * 10)
        selected = kp.era_balanced_sample_positions(eras, cap=11)
        counts = kp.per_era_counts(eras[selected].tolist())
        self.assertEqual(counts, {"0001": 5, "0002": 3, "0003": 3})


# ================================================= structural absences (24-28)
class TestStructuralAbsences(unittest.TestCase):
    def test_24_no_candidate_v_stage_exists(self):
        self.assertEqual(kp.STAGES, (kp.P1, kp.P2))
        self.assertEqual(len(kp.STAGES), 2)
        self.assertEqual(set(kp.STAGE_PROFILES), set(kp.STAGES))
        for name in (
            "candidate_v", "CANDIDATE_V", "candidate_v_block1_seed42",
            "CandidateV", "P3_FEATURE_UNIVERSE_ALL", "P4_ROW_BUDGET_FULL",
            "MAXIMALLY_RECENT_STATIC_OOS_BACKBONE",
        ):
            with self.assertRaises(kp.StageAuthorityError):
                kp.assert_stage(name)
            with self.assertRaises(kp.StageAuthorityError):
                kp.artifact_relpath("prediction", stage=name, model_seed=42)

    def test_24b_no_candidate_v_identifier_exists_in_packet_source(self):
        """Structurally: no Candidate-V name, and no live Candidate-V label.

        The only Candidate-V strings permitted anywhere in the packet are the
        entries of ``FORBIDDEN_STAGE_NAMES``, the substring guard, and the
        declared exclusion labels — every one of which exists to refuse or to
        disclaim, never to select.
        """
        refusal_tokens = (
            set(kp.FORBIDDEN_STAGE_NAMES)
            | {kp.FORBIDDEN_STAGE_SUBSTRING, kp.CANDIDATE_V_RETURN_DENIAL_KEY}
            | set(kp.FrozenDesign().excludes)
        )
        # The only three names allowed to mention Candidate-V are the two
        # refusal constants and the confirmation record's denial-field name.
        refusal_identifiers = {
            "FORBIDDEN_STAGE_NAMES",
            "FORBIDDEN_STAGE_SUBSTRING",
            "CANDIDATE_V_RETURN_DENIAL_KEY",
        }
        for label, tree in TREES.items():
            for identifier in _identifiers(tree) - refusal_identifiers:
                self.assertNotIn(
                    "candidate", identifier.lower(),
                    f"{label}: Candidate-V identifier {identifier!r}",
                )
            for literal in _short_string_constants(tree):
                if "candidate" in literal.lower():
                    self.assertIn(
                        literal, refusal_tokens,
                        f"{label}: live Candidate-V label {literal!r}",
                    )
        for stage in kp.STAGES:
            self.assertNotIn("candidate", stage.lower())

    def test_24c_confirmation_result_denies_candidate_v_return(self):
        for corr in (0.0, kp.FINAL_THREE_SEED_THRESHOLD, 1.0):
            record = kp.final_confirmation(corr, corr, corr)
            self.assertIs(record[kp.CANDIDATE_V_RETURN_DENIAL_KEY], False)
            self.assertIs(record["promotion_granted"], False)

    def test_25_ender60_cannot_affect_any_decision(self):
        screen_params = set(inspect.signature(kp.screen_stage).parameters)
        final_params = set(inspect.signature(kp.final_confirmation).parameters)
        self.assertEqual(screen_params, {"stage", "seed42_corr", "benchmark_mean_corr"})
        self.assertEqual(
            final_params,
            {"corr_42", "corr_1337", "corr_2024", "benchmark_mean_corr"},
        )
        for name in ("ender60", "corr60", "mmc60", "mmc", "weighted_score",
                     "sharpe", "bmc", "recent_20"):
            self.assertIn(name, kp.NON_SELECTING_DIAGNOSTICS)
            with self.assertRaises(ValueError):
                kp.assert_non_selecting(name)
        self.assertEqual(kp.screen_stage(kp.P1, 0.02)["selection_input"], "CORR only")
        self.assertEqual(
            kp.final_confirmation(0.02, 0.02, 0.02)["selection_input"], "CORR only"
        )

    def test_25b_decision_outputs_carry_no_ender60_key(self):
        screen = kp.screen_stage(kp.P1, 0.02)
        final = kp.final_confirmation(0.02, 0.02, 0.02)
        for payload in (screen, final):
            for key in payload:
                self.assertNotIn("60", key)
                self.assertNotIn("ender", key.lower())

    def test_26_bare_target_is_rejected(self):
        self.assertEqual(kp.assert_payout_target("target_ender_20"), "target_ender_20")
        with self.assertRaises(ValueError) as ctx:
            kp.assert_payout_target("target")
        self.assertIn("not a payout objective", str(ctx.exception))
        for bad in ("target_ender_60", "target_cyrusd_20", ""):
            with self.assertRaises(ValueError):
                kp.assert_payout_target(bad)

    def test_26b_the_only_bare_target_literal_is_the_refusal_guard(self):
        """No string literal equal to ``"target"`` outside the rejection constant."""
        for label in ("train", "evaluate"):
            self.assertNotIn(
                kp.BARE_TARGET_ALIAS, _short_string_constants(TREES[label]),
                f"{label}: bare 'target' literal present",
            )
        guarded = [
            node
            for node in ast.walk(TREES["lib"])
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "BARE_TARGET_ALIAS"
                for t in node.targets
            )
        ]
        self.assertEqual(len(guarded), 1)
        self.assertEqual(guarded[0].value.value, "target")
        occurrences = [
            node
            for node in ast.walk(TREES["lib"])
            if isinstance(node, ast.Constant) and node.value == "target"
        ]
        self.assertEqual(len(occurrences), 1, "lib: 'target' literal outside the guard")

    def test_27_terminal_states_cannot_transition_backward(self):
        kp.assert_forward_transition(None, kp.KP35_P1_SCREEN_FAILED)
        kp.assert_forward_transition(kp.KP35_P1_SCREEN_FAILED, kp.KP35_P2_SCREEN_PASSED)
        kp.assert_forward_transition(kp.KP35_P2_SCREEN_PASSED, kp.KP35_PARITY_CONFIRMED)

        for absorbing in kp.ABSORBING_STATES:
            self.assertEqual(kp.FORWARD_TRANSITIONS[absorbing], frozenset())
            for target in (
                kp.KP35_P1_SCREEN_FAILED,
                kp.KP35_P1_SCREEN_PASSED,
                kp.KP35_P2_SCREEN_PASSED,
                kp.KP35_PARITY_CONFIRMED,
            ):
                with self.assertRaises(kp.StageAuthorityError):
                    kp.assert_forward_transition(absorbing, target)

        # Explicit backward moves are refused.
        for origin, target in (
            (kp.KP35_P2_SCREEN_PASSED, kp.KP35_P1_SCREEN_FAILED),
            (kp.KP35_P1_SCREEN_PASSED, kp.KP35_P1_SCREEN_FAILED),
            (kp.KP35_P1_SCREEN_FAILED, kp.KP35_SOURCE_FROZEN),
            (kp.KP35_P1_SCREEN_PASSED, kp.KP35_P2_SCREEN_PASSED),
        ):
            with self.assertRaises(kp.StageAuthorityError):
                kp.assert_forward_transition(origin, target)

    def test_27b_a_p1_pass_can_never_reach_p2(self):
        self.assertNotIn(
            kp.KP35_P2_SCREEN_PASSED, kp.FORWARD_TRANSITIONS[kp.KP35_P1_SCREEN_PASSED]
        )
        with self.assertRaises(kp.StageAuthorityError):
            kp.assert_stage_executable(kp.P2, kp.KP35_P1_SCREEN_PASSED)

    def test_28_no_automatic_next_stage_execution_exists(self):
        self.assertIs(kp.screen_stage(kp.P1, 1.0)["next_stage_started"], False)
        self.assertIs(kp.screen_stage(kp.P1, 0.0)["next_stage_started"], False)

        # The trainer takes exactly one stage and one seed, both required.
        self.assertIn('"--stage", required=True', TRAIN_SRC)
        self.assertIn('"--model-seed", required=True', TRAIN_SRC)
        self.assertIn('"authorizes_next_fit": False', EVAL_SRC)

        # Exactly one fit is executed per trainer invocation, and that call is
        # not reachable from inside any loop.
        self.assertEqual(_call_count(TREES["train"], "run_fit"), 1)
        self.assertNotIn("run_fit", _calls_inside_loops(TREES["train"]))

        # Neither runner can spawn the other or shell out to anything.
        self.assertNotIn("round2_parity_evaluate", _imported_roots(TREES["train"]))
        self.assertNotIn("round2_parity_train", _imported_roots(TREES["evaluate"]))
        for label in ("train", "evaluate"):
            roots = _imported_roots(TREES[label])
            for forbidden in ("subprocess", "multiprocessing", "socket", "shutil"):
                self.assertNotIn(forbidden, roots, f"{label} imports {forbidden}")
            self.assertNotIn("system", _identifiers(TREES[label]), label)

    def test_28b_packet_source_contains_no_account_action(self):
        forbidden_imports = {
            "numerapi", "requests", "urllib", "http", "socket", "ftplib", "smtplib",
        }
        forbidden_identifiers = {
            "NumerAPI", "upload_predictions", "upload_diagnostics",
            "set_submission_webhook", "create_model", "submission_id",
            "public_id", "secret_key", "post", "put", "urlopen",
        }
        for label, tree in TREES.items():
            self.assertEqual(
                _imported_roots(tree) & forbidden_imports, set(),
                f"{label}: forbidden network/account import",
            )
            self.assertEqual(
                _identifiers(tree) & forbidden_identifiers, set(),
                f"{label}: forbidden account-action identifier",
            )

    def test_28c_pure_law_module_imports_only_stdlib_and_numpy(self):
        self.assertEqual(
            _imported_roots(TREES["lib"]),
            {"__future__", "hashlib", "dataclasses", "typing", "numpy"},
        )
        for forbidden in ("lightgbm", "pyarrow", "numerapi", "pandas"):
            self.assertNotIn(forbidden, _imported_roots(TREES["lib"]))

    def test_28d_no_gap_or_holdout_training_path_exists(self):
        # Training loads are bounded by the documented boundary constant only.
        self.assertIn("kp.HISTORY_BOUNDARY_END + 1", TRAIN_SRC)
        self.assertIn("assert_training_eras_authorized", TRAIN_SRC)
        for token in ("1223", "1230", "1231"):
            self.assertNotIn(
                f"range({token}", TRAIN_SRC, "trainer constructs a forbidden era range"
            )
        self.assertEqual(kp.GAP_START, 1223)
        self.assertEqual(kp.GAP_END, 1230)
        self.assertEqual(kp.HOLDOUT_START, 1231)


# ==================================================== one-shot artifact law (L)
class TestOneShotArtifactLaw(unittest.TestCase):
    def test_artifact_paths_are_unique_per_stage_and_seed(self):
        seen = set()
        for kind in ("prediction", "fit_log", "failure_record"):
            for stage in kp.STAGES:
                for seed in kp.ALL_SEEDS:
                    path = kp.artifact_relpath(kind, stage=stage, model_seed=seed)
                    self.assertNotIn(path, seen)
                    seen.add(path)
        for kind in ("screen_result", "confirmation_result"):
            for stage in kp.STAGES:
                path = kp.artifact_relpath(kind, stage=stage)
                self.assertNotIn(path, seen)
                seen.add(path)
        for kind in ("sample_identity", "final_report"):
            path = kp.artifact_relpath(kind)
            self.assertNotIn(path, seen)
            seen.add(path)
        self.assertEqual(len(seen), 3 * 2 * 3 + 2 * 2 + 2)

    def test_unknown_artifact_kind_is_refused(self):
        with self.assertRaises(ValueError):
            kp.artifact_relpath("model", stage=kp.P1, model_seed=42)

    def test_create_new_only_refuses_to_replace_a_final_path(self):
        kp.assert_create_new_only(False, "results/x.json", kind="screen_result")
        with self.assertRaises(FileExistsError) as ctx:
            kp.assert_create_new_only(True, "results/x.json", kind="screen_result")
        self.assertIn("create-new-only", str(ctx.exception))

    def test_retry_is_permitted_at_most_once_and_never_over_a_valid_prediction(self):
        kp.assert_retry_authorized(
            prediction_exists=False, prior_retries=0, stage=kp.P1, model_seed=42
        )
        with self.assertRaises(kp.StageAuthorityError):
            kp.assert_retry_authorized(
                prediction_exists=True, prior_retries=0, stage=kp.P1, model_seed=42
            )
        with self.assertRaises(kp.StageAuthorityError):
            kp.assert_retry_authorized(
                prediction_exists=False, prior_retries=1, stage=kp.P1, model_seed=42
            )

    def test_evaluator_refuses_a_second_result_write(self):
        self.assertIn("assert_create_new_only(result_path.exists()", EVAL_SRC)

    def test_trainer_writes_no_model_artifact(self):
        self.assertIn('"model_artifact_written": False', TRAIN_SRC)
        for token in ("save_model", "booster.save", "joblib.dump", "pickle.dump"):
            self.assertNotIn(token, TRAIN_SRC)


# ===================================================== protocol/source agreement
class TestProtocolAgreement(unittest.TestCase):
    def test_protocol_mirrors_every_frozen_constant(self):
        for key, value in kp.frozen_constants().items():
            self.assertEqual(PROTOCOL["frozen_constants"][key], value, key)

    def test_protocol_profiles_match_the_frozen_law(self):
        for stage in kp.STAGES:
            self.assertEqual(
                PROTOCOL["stages"][stage]["lightgbm_profile"], dict(kp.profile_for(stage))
            )

    def test_protocol_declares_the_payout_target_and_feature_identity(self):
        self.assertEqual(PROTOCOL["payout_target"], "target_ender_20")
        self.assertEqual(PROTOCOL["feature_set"], "medium")
        self.assertEqual(PROTOCOL["n_features"], 780)
        self.assertEqual(PROTOCOL["feature_list_sha256"], kp.FEATURE_LIST_SHA256)

    def test_protocol_records_the_documented_window_derivation(self):
        window = PROTOCOL["documented_benchmark_construction"]["window_containing_score_zone"]
        self.assertEqual(window, kp.derive_history_boundary())
        self.assertEqual(PROTOCOL["documented_benchmark_construction"]["purge_eras_20d"], 8)
        self.assertEqual(PROTOCOL["documented_benchmark_construction"]["chunk_eras"], 156)

    def test_protocol_states_the_honest_row_replacement_caveat(self):
        statement = PROTOCOL["stages"][kp.P1]["honest_composition_statement"]
        self.assertIn("REPLACES", statement)
        self.assertIn("NOT a pure additive-data experiment", statement)
        audit = PROTOCOL["stages"][kp.P1]["keys_only_composition_audit"]
        self.assertEqual(audit["rows_before_sampling"], 5_890_287)
        self.assertEqual(audit["sampled_from_train"], 529_780)
        self.assertEqual(audit["sampled_from_validation"], 470_220)
        self.assertEqual(audit["selected_row_count"] if "selected_row_count" in audit
                         else audit["rows_after_sampling"], 1_000_000)

    def test_protocol_scope_excludes_every_untested_mechanism(self):
        excludes = set(PROTOCOL["scope"]["does_not_test"])
        for mechanism in (
            "Candidate-V", "validation recency promotion", "MMC specialist models",
            "feature ensembles", "target ensembles", "blending", "deployment",
            "live performance",
        ):
            self.assertIn(mechanism, excludes)
        self.assertEqual(PROTOCOL["scope"]["ladder"], list(kp.STAGES))
        self.assertIs(PROTOCOL["scope"]["no_p3_rescue_in_this_gate"], True)

    def test_protocol_records_the_independently_recomputed_universe(self):
        recomputed = PROTOCOL["exact_row_contract"]["independent_recomputation_at_source_freeze"]
        self.assertTrue(recomputed["all_three_identical"])
        self.assertTrue(recomputed["reproduces_kp34_claim"])
        self.assertEqual(recomputed["canon_sha256"], kp.SCORING_UNIVERSE_CANON_SHA256)

    def test_protocol_terminal_states_match_the_transition_law(self):
        self.assertEqual(
            set(PROTOCOL["terminal_states"]["absorbing"]), set(kp.ABSORBING_STATES)
        )
        for state, allowed in PROTOCOL["terminal_states"]["forward_transitions"].items():
            self.assertEqual(set(allowed), set(kp.FORWARD_TRANSITIONS[state]), state)


if __name__ == "__main__":
    unittest.main()
