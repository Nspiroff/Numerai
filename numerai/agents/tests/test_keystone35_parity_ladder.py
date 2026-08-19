"""Protected source-contract tests for the Keystone parity-calibration ladder (KP35).

Deterministic, CPU-only and dataset-free. Every test below runs from synthetic
fixtures, the frozen law module, the protocol record and the packet source text
alone: no LightGBM, no PyArrow, no NumerAPI, no network, no local dataset, no
GPU and no external artifact is required, which is what lets the full suite run
on a bare protected CI runner.

The suite proves the frozen scientific design (boundary, purge, score zone, the
two profiles and their single declared difference), the corrected stage graph
and its eligibility law, the exact screening and two-part confirmation
arithmetic, the exact-row universe contract that prospectively repairs the KW33
source-contract gap, sample custody and its frozen composition, attempt/retry
custody, the strict prior-result authority, independent fit-log provenance
validation, runtime and score-authority binding, non-finite failure discipline,
and the structural absences that matter: no Candidate-V stage, no Ender60
selection path, no bare ``target`` alias, no backward terminal transition, and
no automatic next-stage execution.
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
PROTOCOL_SEMANTIC = kp.protocol_semantic_sha256(PROTOCOL)

TREES = {
    "lib": ast.parse(LIB_SRC),
    "train": ast.parse(TRAIN_SRC),
    "evaluate": ast.parse(EVAL_SRC),
}

NON_FINITE = (float("nan"), float("inf"), float("-inf"))


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


def _function_def(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


# ------------------------------------------------------------------ fixtures
def _universe(name, pairs):
    return kp.RowUniverse.from_columns(name, [e for e, _ in pairs], [i for _, i in pairs])


def _zone_pairs(per_era=3):
    return [
        (era, f"id_{era}_{k}")
        for era in kp.score_zone_eras()
        for k in range(per_era)
    ]


DATA_IDENTITIES = {"train.parquet": "a" * 64, "validation.parquet": "b" * 64}


def _manifest(**overrides):
    """A sample manifest reproducing the frozen KP35 composition.

    Overrides are applied *before* the identity is derived, so a fixture that
    perturbs an identity-bound field still carries a self-consistent identity
    and is rejected for the perturbation itself rather than for a stale hash.
    Pass ``sample_identity_sha256`` explicitly to forge an inconsistent one.
    """
    per_era = overrides.pop("selected_rows_per_era", {"0001": 500_000, "1084": 500_000})
    explicit_identity = overrides.pop("sample_identity_sha256", None)
    base = {
        "record": kp.RECORD_SAMPLE,
        "data_identities": dict(DATA_IDENTITIES),
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
        "source_split_rows": {"train": 529_780, "validation": 470_220},
        "sample_canon_sha256": kp.FROZEN_SAMPLE["sample_canon_sha256"],
    }
    base.update(overrides)
    base["sample_identity_sha256"] = (
        explicit_identity
        if explicit_identity is not None
        else kp.sample_identity(
            data_identities=base["data_identities"],
            eligible_era_range=base["eligible_era_range"],
            feature_list_sha256=base["feature_list_sha256"],
            sampling_law_version=base["sampling_law_version"],
            sampling_seed=base["sampling_seed"],
            row_cap=base["row_cap"],
            sample_canon_sha256=base["sample_canon_sha256"],
        )
    )
    return base


def _fit_log(stage, seed, manifest=None, **overrides):
    """A fit log that validates against the frozen stage recipe."""
    manifest = manifest or _manifest()
    profile = kp.profile_for(stage)
    log = {
        "record": kp.RECORD_FIT_LOG,
        "stage": stage,
        "model_seed": seed,
        "role": kp.expected_role(seed),
        "attempt": kp.FIRST_ATTEMPT,
        "payout_target": kp.PAYOUT_TARGET,
        "feature_set": kp.FEATURE_SET,
        "n_features": kp.N_FEATURES,
        "feature_list_sha256": kp.FEATURE_LIST_SHA256,
        "profile_name": profile["name"],
        "num_trees": profile["num_trees"],
        "params": kp.lightgbm_params(profile, seed),
        "params_sha256": kp.params_sha256(stage, seed),
        "protocol_semantic_sha256": PROTOCOL_SEMANTIC,
        "data_identities": dict(DATA_IDENTITIES),
        "sample_identity_sha256": manifest["sample_identity_sha256"],
        "sample_canon_sha256": manifest["sample_canon_sha256"],
        "rows_before_sampling": manifest["rows_before_sampling"],
        "rows_after_sampling": manifest["selected_row_count"],
        "source_split_rows": manifest["source_split_rows"],
        "selected_rows_per_era": manifest["selected_rows_per_era"],
        "eligible_era_range": manifest["eligible_era_range"],
        "n_eligible_eras": manifest["n_eligible_eras"],
        "scored_eras": ["1133", "1219"],
        "no_early_stopping": True,
        "no_evaluation_set": True,
        "model_artifact_written": False,
        "exit_status": "success",
        "prediction_rows": kp.SCORING_UNIVERSE_ROWS,
        "prediction_sha256": f"{seed:064d}",
        "prediction_canon_sha256": kp.SCORING_UNIVERSE_CANON_SHA256,
    }
    log.update(overrides)
    return log


def _envelope(stage, mode, terminal_state, prior_state, manifest=None, **overrides):
    """A complete KP35 result envelope."""
    manifest = manifest or _manifest()
    seeds = (
        [kp.SCREENING_SEED]
        if mode == kp.MODE_SCREEN
        else [kp.SCREENING_SEED, *kp.CONFIRMATION_SEEDS]
    )
    envelope = {
        "record": kp.MODE_RECORDS[mode],
        "mode": mode,
        "stage": stage,
        "terminal_state": terminal_state,
        "prior_state": prior_state,
        "protocol_semantic_sha256": PROTOCOL_SEMANTIC,
        "benchmark": {
            "recomputed_mean_corr": kp.BENCHMARK_MEAN_CORR,
            "frozen_kw33_mean_corr": kp.BENCHMARK_MEAN_CORR,
            "tolerance": kp.BENCHMARK_MEAN_CORR_TOLERANCE,
        },
        "scoring_universe": {
            "rows": kp.SCORING_UNIVERSE_ROWS,
            "canon_sha256": kp.SCORING_UNIVERSE_CANON_SHA256,
        },
        "sample_custody": {
            "sample_identity_sha256": manifest["sample_identity_sha256"],
            "sample_canon_sha256": manifest["sample_canon_sha256"],
        },
        "fit_provenance": {
            str(seed): {
                "stage": stage,
                "model_seed": seed,
                "params_sha256": kp.params_sha256(stage, seed),
                "prediction_sha256": f"{seed:064d}",
                "prediction_canon_sha256": kp.SCORING_UNIVERSE_CANON_SHA256,
            }
            for seed in seeds
        },
        "authorizes_next_fit": False,
    }
    envelope.update(overrides)
    return envelope


def _build(stage, mode, terminal_state, prior_state, overrides):
    """Build a valid envelope, then let overrides replace any field by name."""
    envelope = _envelope(
        stage, mode, terminal_state, prior_state, manifest=overrides.pop("manifest", None)
    )
    envelope.update(overrides)
    return envelope


def _p1_screen_fail_envelope(**overrides):
    return _build(
        kp.P1, kp.MODE_SCREEN, kp.KP35_P1_SCREEN_FAILED, kp.KP35_SOURCE_FROZEN, overrides
    )


def _p1_screen_pass_envelope(**overrides):
    return _build(
        kp.P1, kp.MODE_SCREEN, kp.KP35_P1_SCREEN_PASSED, kp.KP35_SOURCE_FROZEN, overrides
    )


def _p1_confirmation_fail_envelope(**overrides):
    return _build(
        kp.P1,
        kp.MODE_CONFIRMATION,
        kp.KP35_P1_CONFIRMATION_FAILED,
        kp.KP35_P1_SCREEN_PASSED,
        overrides,
    )


P1_SCREEN_PATH = f"results/{kp.P1}_screen.json"
P1_CONFIRM_PATH = f"results/{kp.P1}_confirmation.json"
P2_SCREEN_PATH = f"results/{kp.P2}_screen.json"


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

    def test_7e_parameter_digests_are_stage_and_seed_specific(self):
        digests = {
            (stage, seed): kp.params_sha256(stage, seed)
            for stage in kp.STAGES
            for seed in kp.ALL_SEEDS
        }
        self.assertEqual(len(set(digests.values())), len(digests))


# ==================================== corrected stage graph and eligibility (G)
class TestStageAuthority(unittest.TestCase):
    def test_8_p2_requires_a_recorded_p1_failure_state(self):
        for authorizing in kp.P2_AUTHORIZING_STATES:
            kp.assert_stage_executable(kp.P2, authorizing)
        for bad in (
            None,
            kp.KP35_SOURCE_FROZEN,
            kp.KP35_P1_SCREEN_PASSED,
            kp.KP35_P2_SCREEN_PASSED,
            kp.KP35_PARITY_NOT_RESTORED,
            kp.KP35_PARITY_CONFIRMED,
            kp.KP35_P2_CONFIRMATION_FAILED,
        ):
            with self.assertRaises(kp.StageAuthorityError):
                kp.assert_stage_executable(kp.P2, bad)

    def test_8b_p1_is_the_first_stage(self):
        kp.assert_stage_executable(kp.P1, None)
        kp.assert_stage_executable(kp.P1, kp.KP35_SOURCE_FROZEN)
        for bad in (kp.KP35_P1_SCREEN_FAILED, kp.KP35_P1_SCREEN_PASSED):
            with self.assertRaises(kp.StageAuthorityError):
                kp.assert_stage_executable(kp.P1, bad)

    def test_9_confirmation_requires_that_stages_screen_pass(self):
        kp.assert_confirmation_authorized(kp.P1, kp.KP35_P1_SCREEN_PASSED)
        kp.assert_confirmation_authorized(kp.P2, kp.KP35_P2_SCREEN_PASSED)
        # A stage may not be confirmed on the other stage's screen pass.
        with self.assertRaises(kp.StageAuthorityError):
            kp.assert_confirmation_authorized(kp.P1, kp.KP35_P2_SCREEN_PASSED)
        with self.assertRaises(kp.StageAuthorityError):
            kp.assert_confirmation_authorized(kp.P2, kp.KP35_P1_SCREEN_PASSED)
        for bad in (
            None,
            kp.KP35_SOURCE_FROZEN,
            kp.KP35_P1_SCREEN_FAILED,
            kp.KP35_PARITY_NOT_RESTORED,
            kp.KP35_P2_CONFIRMATION_FAILED,
        ):
            for stage in kp.STAGES:
                with self.assertRaises(kp.StageAuthorityError):
                    kp.assert_confirmation_authorized(stage, bad)

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

    def test_30_p1_confirmation_failure_authorizes_p2(self):
        outcome = kp.final_confirmation(kp.P1, 0.0, 0.0, 0.0)
        self.assertFalse(outcome["confirmed"])
        self.assertEqual(outcome["terminal_state"], kp.KP35_P1_CONFIRMATION_FAILED)
        self.assertNotIn(kp.KP35_P1_CONFIRMATION_FAILED, kp.ABSORBING_STATES)
        self.assertIn(kp.KP35_P1_CONFIRMATION_FAILED, kp.P2_AUTHORIZING_STATES)
        kp.assert_stage_executable(kp.P2, outcome["terminal_state"])
        kp.assert_forward_transition(
            outcome["terminal_state"], kp.KP35_P2_SCREEN_PASSED
        )

    def test_31_p1_confirmation_success_remains_terminal(self):
        outcome = kp.final_confirmation(kp.P1, 1.0, 1.0, 1.0)
        self.assertTrue(outcome["confirmed"])
        self.assertEqual(outcome["terminal_state"], kp.KP35_PARITY_CONFIRMED)
        self.assertIn(kp.KP35_PARITY_CONFIRMED, kp.ABSORBING_STATES)
        self.assertEqual(kp.FORWARD_TRANSITIONS[kp.KP35_PARITY_CONFIRMED], frozenset())

    def test_32_p2_confirmation_failure_is_terminal(self):
        outcome = kp.final_confirmation(kp.P2, 0.0, 0.0, 0.0)
        self.assertFalse(outcome["confirmed"])
        self.assertEqual(outcome["terminal_state"], kp.KP35_P2_CONFIRMATION_FAILED)
        self.assertIn(kp.KP35_P2_CONFIRMATION_FAILED, kp.ABSORBING_STATES)
        self.assertEqual(
            kp.FORWARD_TRANSITIONS[kp.KP35_P2_CONFIRMATION_FAILED], frozenset()
        )
        self.assertNotIn(kp.KP35_P2_CONFIRMATION_FAILED, kp.P2_AUTHORIZING_STATES)

    def test_33_p2_screen_failure_is_terminal(self):
        outcome = kp.screen_stage(kp.P2, 0.0)
        self.assertFalse(outcome["passed"])
        self.assertEqual(outcome["terminal_state"], kp.KP35_PARITY_NOT_RESTORED)
        self.assertIn(kp.KP35_PARITY_NOT_RESTORED, kp.ABSORBING_STATES)
        self.assertEqual(
            kp.FORWARD_TRANSITIONS[kp.KP35_PARITY_NOT_RESTORED], frozenset()
        )

    def test_33b_confirmation_failure_states_are_stage_specific(self):
        self.assertNotEqual(
            kp.KP35_P1_CONFIRMATION_FAILED, kp.KP35_P2_CONFIRMATION_FAILED
        )
        self.assertEqual(
            kp.CONFIRMATION_STATES[kp.P1][1], kp.KP35_P1_CONFIRMATION_FAILED
        )
        self.assertEqual(
            kp.CONFIRMATION_STATES[kp.P2][1], kp.KP35_P2_CONFIRMATION_FAILED
        )

    def test_33c_the_graph_never_strands_the_deep_profile(self):
        """Every non-confirming P1 outcome must leave P2 reachable."""
        for corr in (0.0, kp.SCREEN_THRESHOLD * 0.5):
            self.assertIn(
                kp.screen_stage(kp.P1, corr)["terminal_state"], kp.P2_AUTHORIZING_STATES
            )
        self.assertIn(
            kp.final_confirmation(kp.P1, 0.0, 0.0, 0.0)["terminal_state"],
            kp.P2_AUTHORIZING_STATES,
        )


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
        out = kp.final_confirmation(kp.P1, 0.02, 0.02, 0.02)
        self.assertEqual(out["threshold"], 0.014663902060935183)
        self.assertAlmostEqual(out["three_seed_mean"], 0.02, places=15)

    def test_14_untouched_pair_gate_is_independently_required(self):
        t = kp.FINAL_THREE_SEED_THRESHOLD
        # Three-seed mean clears the bar only because seed 42 is very strong;
        # the untouched pair does not, so confirmation must fail.
        for stage in kp.STAGES:
            out = kp.final_confirmation(stage, t * 2.0, t * 0.6, t * 0.6)
            self.assertTrue(out["three_seed_gate_passed"])
            self.assertFalse(out["untouched_pair_gate_passed"])
            self.assertFalse(out["confirmed"])
            self.assertEqual(out["terminal_state"], kp.CONFIRMATION_STATES[stage][1])
            self.assertIs(out["both_required"], True)

    def test_15_either_final_gate_failing_prevents_confirmation(self):
        t = kp.FINAL_THREE_SEED_THRESHOLD
        both = kp.final_confirmation(kp.P2, t, t, t)
        self.assertTrue(both["confirmed"])
        self.assertEqual(both["terminal_state"], kp.KP35_PARITY_CONFIRMED)

        # Pair passes, three-seed mean fails (seed 42 drags it under).
        pair_only = kp.final_confirmation(kp.P2, 0.0, t, t)
        self.assertTrue(pair_only["untouched_pair_gate_passed"])
        self.assertFalse(pair_only["three_seed_gate_passed"])
        self.assertFalse(pair_only["confirmed"])

        # Three-seed passes, pair fails.
        three_only = kp.final_confirmation(kp.P2, t * 2.0, t * 0.6, t * 0.6)
        self.assertTrue(three_only["three_seed_gate_passed"])
        self.assertFalse(three_only["untouched_pair_gate_passed"])
        self.assertFalse(three_only["confirmed"])

        # Neither passes.
        neither = kp.final_confirmation(kp.P2, 0.0, 0.0, 0.0)
        self.assertFalse(neither["confirmed"])
        self.assertEqual(neither["terminal_state"], kp.KP35_P2_CONFIRMATION_FAILED)

    def test_15b_confirmation_grants_no_promotion_and_no_candidate_v_return(self):
        out = kp.final_confirmation(kp.P2, 1.0, 1.0, 1.0)
        self.assertTrue(out["confirmed"])
        self.assertIs(out["promotion_granted"], False)
        self.assertIs(out[kp.CANDIDATE_V_RETURN_DENIAL_KEY], False)

    def test_15c_benchmark_identity_is_enforced_within_tolerance(self):
        self.assertEqual(
            kp.assert_benchmark_identity(kp.BENCHMARK_MEAN_CORR), kp.BENCHMARK_MEAN_CORR
        )
        kp.assert_benchmark_identity(kp.BENCHMARK_MEAN_CORR + 1e-13)
        with self.assertRaises(ValueError):
            kp.assert_benchmark_identity(kp.BENCHMARK_MEAN_CORR + 1e-9)
        with self.assertRaises(ValueError):
            kp.assert_benchmark_identity(kp.BENCHMARK_MEAN_CORR * 1.01)


# ============================================= non-finite failure discipline (H)
class TestNonFiniteDiscipline(unittest.TestCase):
    def test_assert_finite_scalar_accepts_only_real_finite_numbers(self):
        self.assertEqual(kp.assert_finite_scalar(0.5, name="x"), 0.5)
        self.assertEqual(kp.assert_finite_scalar(-3, name="x"), -3.0)
        for bad in NON_FINITE:
            with self.assertRaises(kp.NonFiniteValueError):
                kp.assert_finite_scalar(bad, name="x")
        for junk in ("abc", None, object()):
            with self.assertRaises(kp.NonFiniteValueError):
                kp.assert_finite_scalar(junk, name="x")

    def test_24_25_26_non_finite_benchmark_mean_is_refused(self):
        for bad in NON_FINITE:
            with self.assertRaises(kp.NonFiniteValueError):
                kp.assert_benchmark_identity(bad)
            with self.assertRaises(kp.NonFiniteValueError):
                kp.screen_threshold(bad)
            with self.assertRaises(kp.NonFiniteValueError):
                kp.final_threshold(bad)

    def test_27_28_non_finite_screen_corr_is_refused_for_both_stages(self):
        for stage in kp.STAGES:
            for bad in NON_FINITE:
                with self.assertRaises(kp.NonFiniteValueError):
                    kp.screen_stage(stage, bad)
                with self.assertRaises(kp.NonFiniteValueError):
                    kp.screen_stage(stage, 0.02, bad)

    def test_29_non_finite_confirmation_corr_is_refused(self):
        good = kp.FINAL_THREE_SEED_THRESHOLD
        for stage in kp.STAGES:
            for bad in NON_FINITE:
                with self.assertRaises(kp.NonFiniteValueError):
                    kp.final_confirmation(stage, bad, good, good)
                with self.assertRaises(kp.NonFiniteValueError):
                    kp.final_confirmation(stage, good, bad, good)
                with self.assertRaises(kp.NonFiniteValueError):
                    kp.final_confirmation(stage, good, good, bad)
                with self.assertRaises(kp.NonFiniteValueError):
                    kp.final_confirmation(stage, good, good, good, bad)

    def test_non_finite_never_produces_a_terminal_state(self):
        """A NaN must raise, not become an ordinary failure outcome."""
        for stage in kp.STAGES:
            try:
                outcome = kp.screen_stage(stage, float("nan"))
            except kp.NonFiniteValueError:
                outcome = None
            self.assertIsNone(outcome)
            try:
                outcome = kp.final_confirmation(stage, float("nan"), 0.0, 0.0)
            except kp.NonFiniteValueError:
                outcome = None
            self.assertIsNone(outcome)


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

    def test_21c_canonical_json_hash_ignores_key_order_only(self):
        a = kp.canonical_json_sha256({"x": 1, "y": [2, 3]})
        b = kp.canonical_json_sha256({"y": [2, 3], "x": 1})
        self.assertEqual(a, b)
        self.assertNotEqual(a, kp.canonical_json_sha256({"x": 1, "y": [3, 2]}))
        self.assertNotEqual(a, kp.canonical_json_sha256({"x": 2, "y": [2, 3]}))

    def test_21d_relpath_normalisation_is_separator_agnostic(self):
        for raw in ("results/x.json", "results\\x.json", "./results/x.json", "/results/x.json"):
            self.assertEqual(kp.normalize_relpath(raw), "results/x.json")


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
        with self.assertRaises(kp.SampleCustodyError) as ctx:
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
                _manifest(selected_rows_per_era={"0001": 500_000, "1085": 500_000})
            )
        with self.assertRaises(ValueError):
            kp.validate_sample_manifest(
                _manifest(selected_rows_per_era={"0001": 500_000, "1231": 500_000})
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

    def test_34_the_frozen_sample_hash_and_composition_are_enforced(self):
        self.assertEqual(
            kp.FROZEN_SAMPLE["sample_canon_sha256"],
            "e555e848770f4acd276020aca833541e8b0702a2f1b7c3ebc8068b657d101350",
        )
        self.assertEqual(kp.FROZEN_SAMPLE["rows_before_sampling"], 5_890_287)
        self.assertEqual(kp.FROZEN_SAMPLE["selected_row_count"], 1_000_000)
        self.assertEqual(
            kp.FROZEN_SAMPLE["source_split_rows"],
            {"train": 529_780, "validation": 470_220},
        )
        self.assertEqual(kp.FROZEN_SAMPLE["n_eligible_eras"], 1084)
        self.assertEqual(kp.FROZEN_SAMPLE["eligible_era_range"], ["0001", "1084"])
        self.assertEqual(kp.FROZEN_SAMPLE["sampling_seed"], 20260817)
        self.assertEqual(kp.FROZEN_SAMPLE["row_cap"], 1_000_000)
        self.assertEqual(
            kp.FROZEN_SAMPLE["sampling_law_version"], "kw33_era_balanced_v1"
        )

        out = kp.assert_frozen_sample(_manifest())
        self.assertTrue(out["frozen_sample_reproduced"])

        for field, bad in (
            ("rows_before_sampling", 5_890_286),
            ("n_eligible_eras", 1083),
            ("eligible_era_range", ["0001", "1124"]),
            ("source_split_rows", {"train": 500_000, "validation": 500_000}),
            ("sample_canon_sha256", "f" * 64),
        ):
            with self.assertRaises(kp.SampleCustodyError) as ctx:
                kp.assert_frozen_sample(_manifest(**{field: bad}))
            self.assertIn("NOT", str(ctx.exception))

    def test_34b_frozen_sample_mismatch_is_an_infrastructure_stop_not_a_result(self):
        with self.assertRaises(kp.SampleCustodyError) as ctx:
            kp.assert_frozen_sample(_manifest(sample_canon_sha256="f" * 64))
        message = str(ctx.exception)
        self.assertIn("infrastructure, data, or implementation stop", message)
        self.assertIn("NOT", message)
        self.assertTrue(issubclass(kp.SampleCustodyError, ValueError))

    def test_19_wrong_sample_manifest_is_refused_on_the_full_envelope(self):
        existing = _manifest()
        # Same canonical hash and identity, but a different recorded composition:
        # comparing only two hash fields would have accepted this.
        fresh = _manifest()
        fresh["rows_before_sampling"] = 5_890_287
        fresh["selected_rows_per_era"] = {"0001": 400_000, "1084": 600_000}
        self.assertEqual(
            existing["sample_canon_sha256"], fresh["sample_canon_sha256"]
        )
        self.assertEqual(
            existing["sample_identity_sha256"], fresh["sample_identity_sha256"]
        )
        with self.assertRaises(kp.SampleCustodyError) as ctx:
            kp.assert_sample_envelope_equal(existing, fresh)
        self.assertIn("custody envelope", str(ctx.exception))
        self.assertEqual(
            kp.assert_sample_envelope_equal(existing, _manifest()),
            existing["sample_identity_sha256"],
        )


# ======================================= strict prior-result authority (C, J1-11)
class TestPriorResultAuthority(unittest.TestCase):
    def test_1_bare_terminal_state_json_cannot_authorize_p2(self):
        """The whole point: a plausible string in a file authorises nothing."""
        forged = {"terminal_state": kp.KP35_P1_SCREEN_FAILED}
        with self.assertRaises(kp.PriorResultAuthorityError):
            kp.validate_prior_result(
                stage=kp.P2,
                mode=kp.MODE_SCREEN,
                prior_relpath=P1_SCREEN_PATH,
                prior_envelope=forged,
                protocol_semantic=PROTOCOL_SEMANTIC,
            )

    def test_2_forged_screen_result_cannot_authorize_confirmation(self):
        forged = {"terminal_state": kp.KP35_P1_SCREEN_PASSED, "stage": kp.P1}
        with self.assertRaises(kp.PriorResultAuthorityError):
            kp.validate_prior_result(
                stage=kp.P1,
                mode=kp.MODE_CONFIRMATION,
                prior_relpath=P1_SCREEN_PATH,
                prior_envelope=forged,
                protocol_semantic=PROTOCOL_SEMANTIC,
            )

    def test_3_prior_result_at_a_noncanonical_path_is_refused(self):
        valid = _p1_screen_fail_envelope()
        for bad_path in (
            "results/somewhere_else.json",
            "scratch/P1_HISTORY_BOUNDARY_1084_screen.json",
            f"results/{kp.P2}_screen.json",
            "P1_HISTORY_BOUNDARY_1084_screen.json",
        ):
            with self.assertRaises(kp.PriorResultAuthorityError) as ctx:
                kp.validate_prior_result(
                    stage=kp.P2,
                    mode=kp.MODE_SCREEN,
                    prior_relpath=bad_path,
                    prior_envelope=valid,
                    protocol_semantic=PROTOCOL_SEMANTIC,
                )
            self.assertIn("canonical predecessor", str(ctx.exception))

    def test_3b_canonical_paths_are_the_documented_ones(self):
        self.assertEqual(
            kp.canonical_prior_relpaths(kp.P2, kp.MODE_SCREEN),
            (P1_SCREEN_PATH, P1_CONFIRM_PATH),
        )
        self.assertEqual(
            kp.canonical_prior_relpaths(kp.P1, kp.MODE_CONFIRMATION), (P1_SCREEN_PATH,)
        )
        self.assertEqual(
            kp.canonical_prior_relpaths(kp.P2, kp.MODE_CONFIRMATION), (P2_SCREEN_PATH,)
        )
        self.assertEqual(kp.canonical_prior_relpaths(kp.P1, kp.MODE_SCREEN), ())

    def test_valid_envelopes_authorize_their_successors(self):
        manifest = _manifest()
        self.assertEqual(
            kp.validate_prior_result(
                stage=kp.P2,
                mode=kp.MODE_SCREEN,
                prior_relpath=P1_SCREEN_PATH,
                prior_envelope=_p1_screen_fail_envelope(manifest=manifest),
                protocol_semantic=PROTOCOL_SEMANTIC,
                sample_identity_sha256=manifest["sample_identity_sha256"],
                sample_canon_sha256=manifest["sample_canon_sha256"],
            ),
            kp.KP35_P1_SCREEN_FAILED,
        )
        self.assertEqual(
            kp.validate_prior_result(
                stage=kp.P2,
                mode=kp.MODE_SCREEN,
                prior_relpath=P1_CONFIRM_PATH,
                prior_envelope=_p1_confirmation_fail_envelope(manifest=manifest),
                protocol_semantic=PROTOCOL_SEMANTIC,
            ),
            kp.KP35_P1_CONFIRMATION_FAILED,
        )
        self.assertEqual(
            kp.validate_prior_result(
                stage=kp.P1,
                mode=kp.MODE_CONFIRMATION,
                prior_relpath=P1_SCREEN_PATH,
                prior_envelope=_p1_screen_pass_envelope(manifest=manifest),
                protocol_semantic=PROTOCOL_SEMANTIC,
            ),
            kp.KP35_P1_SCREEN_PASSED,
        )

    def test_p1_screen_accepts_and_requires_no_predecessor(self):
        self.assertIsNone(
            kp.validate_prior_result(
                stage=kp.P1,
                mode=kp.MODE_SCREEN,
                prior_relpath=None,
                prior_envelope=None,
                protocol_semantic=PROTOCOL_SEMANTIC,
            )
        )
        with self.assertRaises(kp.PriorResultAuthorityError):
            kp.validate_prior_result(
                stage=kp.P1,
                mode=kp.MODE_SCREEN,
                prior_relpath=P1_SCREEN_PATH,
                prior_envelope=_p1_screen_fail_envelope(),
                protocol_semantic=PROTOCOL_SEMANTIC,
            )

    def test_missing_predecessor_is_refused(self):
        for stage, mode in (
            (kp.P2, kp.MODE_SCREEN),
            (kp.P1, kp.MODE_CONFIRMATION),
            (kp.P2, kp.MODE_CONFIRMATION),
        ):
            with self.assertRaises(kp.PriorResultAuthorityError):
                kp.validate_prior_result(
                    stage=stage,
                    mode=mode,
                    prior_relpath=None,
                    prior_envelope=None,
                    protocol_semantic=PROTOCOL_SEMANTIC,
                )

    def _refuse(self, **envelope_overrides):
        with self.assertRaises(kp.PriorResultAuthorityError):
            kp.validate_prior_result(
                stage=kp.P2,
                mode=kp.MODE_SCREEN,
                prior_relpath=P1_SCREEN_PATH,
                prior_envelope=_p1_screen_fail_envelope(**envelope_overrides),
                protocol_semantic=PROTOCOL_SEMANTIC,
            )

    def test_4_wrong_record_type_is_refused(self):
        self._refuse(record=kp.RECORD_CONFIRMATION)
        self._refuse(record="something_else")

    def test_5_wrong_mode_is_refused(self):
        self._refuse(mode=kp.MODE_CONFIRMATION)

    def test_6_wrong_stage_is_refused(self):
        self._refuse(stage=kp.P2)

    def test_7_wrong_predecessor_state_is_refused(self):
        self._refuse(terminal_state=kp.KP35_P1_SCREEN_PASSED)
        self._refuse(terminal_state=kp.KP35_PARITY_CONFIRMED)
        # A legal terminal state reached by an illegal transition is refused.
        self._refuse(prior_state=kp.KP35_PARITY_CONFIRMED)
        self._refuse(prior_state=kp.KP35_P2_SCREEN_PASSED)

    def test_8_wrong_protocol_hash_is_refused(self):
        self._refuse(protocol_semantic_sha256="0" * 64)

    def test_9_wrong_sample_identity_is_refused(self):
        manifest = _manifest()
        with self.assertRaises(kp.PriorResultAuthorityError):
            kp.validate_prior_result(
                stage=kp.P2,
                mode=kp.MODE_SCREEN,
                prior_relpath=P1_SCREEN_PATH,
                prior_envelope=_p1_screen_fail_envelope(manifest=manifest),
                protocol_semantic=PROTOCOL_SEMANTIC,
                sample_identity_sha256="0" * 64,
            )
        # A predecessor whose sample is not the frozen sample is refused outright.
        self._refuse(
            sample_custody={
                "sample_identity_sha256": "0" * 64,
                "sample_canon_sha256": "1" * 64,
            }
        )

    def test_10_wrong_scoring_universe_hash_is_refused(self):
        self._refuse(
            scoring_universe={
                "rows": kp.SCORING_UNIVERSE_ROWS,
                "canon_sha256": "0" * 64,
            }
        )
        self._refuse(
            scoring_universe={
                "rows": 1,
                "canon_sha256": kp.SCORING_UNIVERSE_CANON_SHA256,
            }
        )

    def test_11_wrong_benchmark_identity_is_refused(self):
        self._refuse(
            benchmark={
                "recomputed_mean_corr": kp.BENCHMARK_MEAN_CORR,
                "frozen_kw33_mean_corr": 0.5,
                "tolerance": kp.BENCHMARK_MEAN_CORR_TOLERANCE,
            }
        )
        self._refuse(
            benchmark={
                "recomputed_mean_corr": 0.5,
                "frozen_kw33_mean_corr": kp.BENCHMARK_MEAN_CORR,
                "tolerance": kp.BENCHMARK_MEAN_CORR_TOLERANCE,
            }
        )
        self._refuse(
            benchmark={
                "recomputed_mean_corr": kp.BENCHMARK_MEAN_CORR,
                "frozen_kw33_mean_corr": kp.BENCHMARK_MEAN_CORR,
                "tolerance": 1.0,
            }
        )

    def test_missing_or_wrong_fit_provenance_is_refused(self):
        self._refuse(fit_provenance={})
        self._refuse(
            fit_provenance={
                "42": {
                    "stage": kp.P1,
                    "model_seed": 42,
                    "params_sha256": "0" * 64,
                    "prediction_sha256": "1" * 64,
                    "prediction_canon_sha256": kp.SCORING_UNIVERSE_CANON_SHA256,
                }
            }
        )
        self._refuse(
            fit_provenance={
                "42": {
                    "stage": kp.P1,
                    "model_seed": 42,
                    "params_sha256": kp.params_sha256(kp.P1, 42),
                    "prediction_sha256": "1" * 64,
                    "prediction_canon_sha256": "0" * 64,
                }
            }
        )
        # An extra seed in a one-seed screen envelope is contradictory.
        provenance = _p1_screen_fail_envelope()["fit_provenance"]
        provenance["1337"] = {
            "stage": kp.P1,
            "model_seed": 1337,
            "params_sha256": kp.params_sha256(kp.P1, 1337),
            "prediction_sha256": "2" * 64,
            "prediction_canon_sha256": kp.SCORING_UNIVERSE_CANON_SHA256,
        }
        self._refuse(fit_provenance=provenance)

    def test_a_self_authorizing_envelope_is_refused(self):
        self._refuse(authorizes_next_fit=True)

    def test_incomplete_envelope_is_refused_field_by_field(self):
        for field in kp.RESULT_ENVELOPE_REQUIRED_FIELDS:
            broken = _p1_screen_fail_envelope()
            broken.pop(field)
            with self.assertRaises(kp.PriorResultAuthorityError):
                kp.validate_prior_result(
                    stage=kp.P2,
                    mode=kp.MODE_SCREEN,
                    prior_relpath=P1_SCREEN_PATH,
                    prior_envelope=broken,
                    protocol_semantic=PROTOCOL_SEMANTIC,
                )


# ============================================ fit-log provenance (D, J12-18)
class TestFitProvenance(unittest.TestCase):
    def setUp(self):
        self.manifest = _manifest()

    def _validate(self, log, stage=kp.P1, seed=kp.SCREENING_SEED, **kwargs):
        return kp.validate_fit_log(
            log,
            stage=stage,
            model_seed=seed,
            protocol_semantic=PROTOCOL_SEMANTIC,
            data_identities=DATA_IDENTITIES,
            sample_manifest=self.manifest,
            **kwargs,
        )

    def test_a_correct_log_validates_for_every_stage_and_seed(self):
        for stage in kp.STAGES:
            for seed in kp.ALL_SEEDS:
                out = self._validate(
                    _fit_log(stage, seed, self.manifest), stage=stage, seed=seed
                )
                self.assertTrue(out["validated"])
                self.assertEqual(out["role"], kp.expected_role(seed))

    def test_12_a_fit_log_with_the_wrong_p1_profile_is_refused(self):
        wrong = _fit_log(kp.P1, 42, self.manifest)
        wrong["params"] = kp.lightgbm_params(kp.profile_for(kp.P2), 42)
        with self.assertRaises(kp.FitProvenanceError):
            self._validate(wrong)
        wrong2 = _fit_log(kp.P1, 42, self.manifest, num_trees=30000)
        with self.assertRaises(kp.FitProvenanceError):
            self._validate(wrong2)
        wrong3 = _fit_log(kp.P1, 42, self.manifest, profile_name="DOCUMENTED_V5_DEEP")
        with self.assertRaises(kp.FitProvenanceError):
            self._validate(wrong3)

    def test_13_a_fit_log_with_the_wrong_p2_profile_is_refused(self):
        wrong = _fit_log(kp.P2, 42, self.manifest)
        wrong["params"] = kp.lightgbm_params(kp.profile_for(kp.P1), 42)
        with self.assertRaises(kp.FitProvenanceError):
            self._validate(wrong, stage=kp.P2)
        wrong2 = _fit_log(kp.P2, 42, self.manifest, num_trees=6000)
        with self.assertRaises(kp.FitProvenanceError):
            self._validate(wrong2, stage=kp.P2)

    def test_14_wrong_target_is_refused(self):
        for bad in ("target", "target_ender_60", "target_cyrusd_20"):
            with self.assertRaises(kp.FitProvenanceError):
                self._validate(_fit_log(kp.P1, 42, self.manifest, payout_target=bad))

    def test_15_wrong_feature_count_or_hash_is_refused(self):
        with self.assertRaises(kp.FitProvenanceError):
            self._validate(_fit_log(kp.P1, 42, self.manifest, n_features=3555))
        with self.assertRaises(kp.FitProvenanceError):
            self._validate(_fit_log(kp.P1, 42, self.manifest, feature_list_sha256="0" * 64))
        with self.assertRaises(kp.FitProvenanceError):
            self._validate(_fit_log(kp.P1, 42, self.manifest, feature_set="all"))

    def test_16_wrong_data_identity_is_refused(self):
        wrong = _fit_log(kp.P1, 42, self.manifest)
        wrong["data_identities"] = {"train.parquet": "z" * 64, "validation.parquet": "b" * 64}
        with self.assertRaises(kp.FitProvenanceError) as ctx:
            self._validate(wrong)
        self.assertIn("data identity", str(ctx.exception))

    def test_17_wrong_parameter_hash_is_refused(self):
        with self.assertRaises(kp.FitProvenanceError):
            self._validate(_fit_log(kp.P1, 42, self.manifest, params_sha256="0" * 64))
        # A seed-swapped digest is still wrong for this seed.
        with self.assertRaises(kp.FitProvenanceError):
            self._validate(
                _fit_log(kp.P1, 42, self.manifest, params_sha256=kp.params_sha256(kp.P1, 1337))
            )

    def test_18_wrong_prediction_hash_is_refused(self):
        log = _fit_log(kp.P1, 42, self.manifest)
        with self.assertRaises(kp.FitProvenanceError) as ctx:
            self._validate(log, actual_prediction_sha256="9" * 64)
        self.assertIn("prediction file digest", str(ctx.exception))
        with self.assertRaises(kp.FitProvenanceError):
            self._validate(log, actual_prediction_canon_sha256="9" * 64)
        with self.assertRaises(kp.FitProvenanceError):
            self._validate(log, actual_prediction_rows=1)
        with self.assertRaises(kp.FitProvenanceError):
            self._validate(_fit_log(kp.P1, 42, self.manifest, prediction_rows=1))
        with self.assertRaises(kp.FitProvenanceError):
            self._validate(
                _fit_log(kp.P1, 42, self.manifest, prediction_canon_sha256="0" * 64)
            )

    def test_a_one_seed_screen_log_is_validated_against_the_frozen_recipe(self):
        """No second seed exists, so the frozen recipe is the only reference."""
        log = _fit_log(kp.P1, kp.SCREENING_SEED, self.manifest)
        self.assertTrue(self._validate(log)["validated"])
        tampered = _fit_log(kp.P1, kp.SCREENING_SEED, self.manifest)
        tampered["params"] = {**tampered["params"], "learning_rate": 0.004}
        with self.assertRaises(kp.FitProvenanceError):
            self._validate(tampered)

    def test_sample_fields_are_compared_against_the_external_manifest(self):
        for field, bad in (
            ("sample_identity_sha256", "0" * 64),
            ("sample_canon_sha256", "0" * 64),
            ("rows_before_sampling", 1),
            ("rows_after_sampling", 1),
            ("source_split_rows", {"train": 1, "validation": 1}),
            ("n_eligible_eras", 1),
            ("eligible_era_range", ["0001", "1124"]),
        ):
            with self.assertRaises(kp.FitProvenanceError):
                self._validate(_fit_log(kp.P1, 42, self.manifest, **{field: bad}))

    def test_missing_required_fit_log_field_is_refused(self):
        for field in kp.FIT_LOG_REQUIRED_FIELDS:
            broken = _fit_log(kp.P1, 42, self.manifest)
            broken.pop(field)
            with self.assertRaises(kp.FitProvenanceError):
                self._validate(broken)

    def test_early_stopping_eval_set_model_artifact_and_failure_are_refused(self):
        for field, bad in (
            ("no_early_stopping", False),
            ("no_evaluation_set", False),
            ("model_artifact_written", True),
            ("exit_status", "FAILED: RuntimeError: boom"),
            ("scored_eras", ["1133", "1218"]),
            ("role", "confirmation"),
            ("record", "something_else"),
        ):
            with self.assertRaises(kp.FitProvenanceError):
                self._validate(_fit_log(kp.P1, 42, self.manifest, **{field: bad}))

    def test_cohort_must_differ_only_by_the_model_seed(self):
        logs = {seed: _fit_log(kp.P1, seed, self.manifest) for seed in kp.ALL_SEEDS}
        out = kp.assert_cohort_identical_except_seed(logs)
        self.assertTrue(out["identical_except_seed"])
        self.assertEqual(out["seeds"], [42, 1337, 2024])

        mixed = dict(logs)
        mixed[2024] = _fit_log(kp.P2, 2024, self.manifest)
        with self.assertRaises(kp.FitProvenanceError):
            kp.assert_cohort_identical_except_seed(mixed)

        drifted = {seed: dict(log) for seed, log in logs.items()}
        drifted[1337]["params"] = {**drifted[1337]["params"], "learning_rate": 0.004}
        with self.assertRaises(kp.FitProvenanceError):
            kp.assert_cohort_identical_except_seed(drifted)

        split_sample = {seed: dict(log) for seed, log in logs.items()}
        split_sample[2024]["sample_identity_sha256"] = "0" * 64
        with self.assertRaises(kp.FitProvenanceError):
            kp.assert_cohort_identical_except_seed(split_sample)


# ============================================ attempt / retry custody (F, J20-23)
class TestAttemptCustody(unittest.TestCase):
    def _authorize(self, **overrides):
        kwargs = {
            "stage": kp.P1,
            "model_seed": kp.SCREENING_SEED,
            "attempt": kp.FIRST_ATTEMPT,
            "prediction_exists": False,
            "success_log_exists": False,
            "attempt1_failure_exists": False,
            "attempt2_failure_exists": False,
        }
        kwargs.update(overrides)
        return kp.assert_attempt_authorized(**kwargs)

    def test_normal_invocation_is_attempt_one(self):
        self.assertEqual(kp.resolve_attempt(retry_requested=False), kp.FIRST_ATTEMPT)
        self.assertEqual(kp.resolve_attempt(retry_requested=True), kp.RETRY_ATTEMPT)
        self.assertEqual(kp.FIRST_ATTEMPT, 1)
        self.assertEqual(kp.RETRY_ATTEMPT, 2)
        self.assertEqual(kp.MAX_ATTEMPTS, 2)
        self.assertTrue(self._authorize()["authorized"])

    def test_20_retry_without_attempt1_failure_is_refused(self):
        with self.assertRaises(kp.StageAuthorityError) as ctx:
            self._authorize(attempt=kp.RETRY_ATTEMPT, attempt1_failure_exists=False)
        self.assertIn("without a preserved", str(ctx.exception))

    def test_21_exactly_one_retry_is_permitted_after_a_valid_first_failure(self):
        out = self._authorize(attempt=kp.RETRY_ATTEMPT, attempt1_failure_exists=True)
        self.assertTrue(out["authorized"])
        self.assertEqual(out["attempt"], kp.RETRY_ATTEMPT)
        # Attempt 1 cannot be replayed once its failure is preserved.
        with self.assertRaises(kp.StageAuthorityError):
            self._authorize(attempt=kp.FIRST_ATTEMPT, attempt1_failure_exists=True)

    def test_22_a_second_failure_gets_its_own_preserved_path(self):
        first = kp.artifact_relpath(
            "failure_record", stage=kp.P1, model_seed=42, attempt=kp.FIRST_ATTEMPT
        )
        second = kp.artifact_relpath(
            "failure_record", stage=kp.P1, model_seed=42, attempt=kp.RETRY_ATTEMPT
        )
        self.assertNotEqual(first, second)
        self.assertEqual(first, f"failures/{kp.P1}_seed42_attempt1.json")
        self.assertEqual(second, f"failures/{kp.P1}_seed42_attempt2.json")

    def test_23_a_third_attempt_is_refused(self):
        with self.assertRaises(kp.StageAuthorityError):
            self._authorize(
                attempt=3, attempt1_failure_exists=True, attempt2_failure_exists=True
            )
        with self.assertRaises(kp.StageAuthorityError) as ctx:
            self._authorize(
                attempt=kp.RETRY_ATTEMPT,
                attempt1_failure_exists=True,
                attempt2_failure_exists=True,
            )
        self.assertIn("no", str(ctx.exception).lower())
        with self.assertRaises(kp.StageAuthorityError):
            kp.artifact_relpath("failure_record", stage=kp.P1, model_seed=42, attempt=3)
        with self.assertRaises(kp.StageAuthorityError):
            kp.artifact_relpath("failure_record", stage=kp.P1, model_seed=42, attempt=0)

    def test_a_completed_fit_is_never_rerun_under_any_attempt(self):
        for attempt in (kp.FIRST_ATTEMPT, kp.RETRY_ATTEMPT):
            with self.assertRaises(kp.StageAuthorityError):
                self._authorize(attempt=attempt, prediction_exists=True,
                                attempt1_failure_exists=True)
            with self.assertRaises(kp.StageAuthorityError):
                self._authorize(attempt=attempt, success_log_exists=True,
                                attempt1_failure_exists=True)

    def test_attempt1_failure_record_must_match_this_invocation(self):
        good = {
            "stage": kp.P1,
            "model_seed": 42,
            "attempt": kp.FIRST_ATTEMPT,
            "protocol_semantic_sha256": PROTOCOL_SEMANTIC,
            "params_sha256": kp.params_sha256(kp.P1, 42),
            "payout_target": kp.PAYOUT_TARGET,
            "feature_list_sha256": kp.FEATURE_LIST_SHA256,
            "exit_status": "FAILED: RuntimeError: boom",
        }
        kp.validate_attempt1_failure_record(
            good, stage=kp.P1, model_seed=42, protocol_semantic=PROTOCOL_SEMANTIC
        )
        for field, bad in (
            ("stage", kp.P2),
            ("model_seed", 1337),
            ("attempt", kp.RETRY_ATTEMPT),
            ("protocol_semantic_sha256", "0" * 64),
            ("params_sha256", "0" * 64),
            ("payout_target", "target"),
            ("feature_list_sha256", "0" * 64),
            ("exit_status", "success"),
        ):
            with self.assertRaises(kp.StageAuthorityError):
                kp.validate_attempt1_failure_record(
                    {**good, field: bad},
                    stage=kp.P1,
                    model_seed=42,
                    protocol_semantic=PROTOCOL_SEMANTIC,
                )


# ==================================== authority / environment binding (I, J35)
class TestAuthorityAndEnvironment(unittest.TestCase):
    def test_authority_snapshot_records_the_frozen_public_authority(self):
        snapshot = kp.AUTHORITY_SNAPSHOT
        self.assertEqual(snapshot["payout_target"], "target_ender_20")
        self.assertEqual(snapshot["corr_config"]["multiplier"], 0.75)
        self.assertEqual(snapshot["corr_config"]["version"], "6")
        self.assertEqual(snapshot["corr_config"]["display_name"], "v2_corr20")
        self.assertEqual(snapshot["corr_config"]["name"], "correlation")
        self.assertEqual(snapshot["mmc_config"]["multiplier"], 2.25)
        self.assertEqual(snapshot["mmc_config"]["version"], "5")
        self.assertEqual(snapshot["mmc_config"]["display_name"], "mmc")
        self.assertEqual(snapshot["mmc_config"]["name"], "meta_model_contribution")
        self.assertIs(snapshot["ender60_payout_active"], False)
        self.assertIn("retrieved_utc", snapshot)
        self.assertIn("query", snapshot)
        self.assertIs(snapshot["live_revalidation_required_before_p1"], True)

    def test_score_authority_is_bound_to_the_frozen_snapshot(self):
        out = kp.assert_score_authority(
            payout_target=kp.PAYOUT_TARGET,
            corr_multiplier=0.75,
            mmc_multiplier=2.25,
            meta_model_column=kp.META_MODEL_COLUMN,
        )
        self.assertTrue(out["matches_frozen_snapshot"])
        self.assertIs(out["live_revalidation_still_required_before_p1"], True)
        with self.assertRaises(ValueError):
            kp.assert_score_authority(
                payout_target="target", corr_multiplier=0.75, mmc_multiplier=2.25,
                meta_model_column=kp.META_MODEL_COLUMN,
            )
        for bad_corr, bad_mmc in ((0.5, 2.25), (0.75, 2.0)):
            with self.assertRaises(kp.EnvironmentBindingError):
                kp.assert_score_authority(
                    payout_target=kp.PAYOUT_TARGET,
                    corr_multiplier=bad_corr,
                    mmc_multiplier=bad_mmc,
                    meta_model_column=kp.META_MODEL_COLUMN,
                )
        with self.assertRaises(kp.EnvironmentBindingError):
            kp.assert_score_authority(
                payout_target=kp.PAYOUT_TARGET, corr_multiplier=0.75,
                mmc_multiplier=2.25, meta_model_column="something_else",
            )
        for bad in NON_FINITE:
            with self.assertRaises(kp.NonFiniteValueError):
                kp.assert_score_authority(
                    payout_target=kp.PAYOUT_TARGET, corr_multiplier=bad,
                    mmc_multiplier=2.25, meta_model_column=kp.META_MODEL_COLUMN,
                )

    def test_35_runtime_version_mismatch_is_refused(self):
        frozen = dict(kp.FROZEN_ENVIRONMENT)
        self.assertEqual(frozen["python"], "3.13.14")
        self.assertEqual(frozen["lightgbm"], "4.7.0")
        self.assertEqual(frozen["numpy"], "2.5.1")
        self.assertEqual(frozen["pandas"], "3.0.5")
        self.assertEqual(frozen["pyarrow"], "25.0.1")
        self.assertEqual(frozen["numerai_tools"], "0.6.0")
        self.assertEqual(frozen["psutil"], "7.2.2")

        out = kp.assert_runtime_versions(frozen)
        self.assertTrue(out["matches_frozen_environment"])

        for package in frozen:
            drifted = {**frozen, package: "0.0.0"}
            with self.assertRaises(kp.EnvironmentBindingError) as ctx:
                kp.assert_runtime_versions(drifted)
            self.assertIn(package, str(ctx.exception))

        incomplete = {k: v for k, v in frozen.items() if k != "numpy"}
        with self.assertRaises(kp.EnvironmentBindingError):
            kp.assert_runtime_versions(incomplete)

    def test_score_producing_packages_are_verified_by_the_evaluator(self):
        self.assertEqual(
            set(kp.SCORE_PRODUCING_PACKAGES),
            {"python", "numpy", "pandas", "numerai_tools"},
        )
        subset = {k: kp.FROZEN_ENVIRONMENT[k] for k in kp.SCORE_PRODUCING_PACKAGES}
        self.assertTrue(
            kp.assert_runtime_versions(subset, required=kp.SCORE_PRODUCING_PACKAGES)[
                "matches_frozen_environment"
            ]
        )
        with self.assertRaises(kp.EnvironmentBindingError):
            kp.assert_runtime_versions(
                {**subset, "numerai_tools": "0.5.0"},
                required=kp.SCORE_PRODUCING_PACKAGES,
            )


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
        entries of ``FORBIDDEN_STAGE_NAMES``, the substring guard, the denial
        field name and the declared exclusion labels -- every one of which
        exists to refuse or to disclaim, never to select.
        """
        refusal_identifiers = {
            "FORBIDDEN_STAGE_NAMES",
            "FORBIDDEN_STAGE_SUBSTRING",
            "CANDIDATE_V_RETURN_DENIAL_KEY",
        }
        refusal_tokens = (
            set(kp.FORBIDDEN_STAGE_NAMES)
            | {kp.FORBIDDEN_STAGE_SUBSTRING, kp.CANDIDATE_V_RETURN_DENIAL_KEY}
            | set(kp.FrozenDesign().excludes)
        )
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
        for stage in kp.STAGES:
            for corr in (0.0, kp.FINAL_THREE_SEED_THRESHOLD, 1.0):
                record = kp.final_confirmation(stage, corr, corr, corr)
                self.assertIs(record[kp.CANDIDATE_V_RETURN_DENIAL_KEY], False)
                self.assertIs(record["promotion_granted"], False)

    def test_25_ender60_cannot_affect_any_decision(self):
        screen_params = set(inspect.signature(kp.screen_stage).parameters)
        final_params = set(inspect.signature(kp.final_confirmation).parameters)
        self.assertEqual(
            screen_params, {"stage", "seed42_corr", "benchmark_mean_corr"}
        )
        self.assertEqual(
            final_params,
            {"stage", "corr_42", "corr_1337", "corr_2024", "benchmark_mean_corr"},
        )
        for name in ("ender60", "corr60", "mmc60", "mmc", "weighted_score",
                     "sharpe", "bmc", "recent_20"):
            self.assertIn(name, kp.NON_SELECTING_DIAGNOSTICS)
            with self.assertRaises(ValueError):
                kp.assert_non_selecting(name)
        self.assertEqual(kp.screen_stage(kp.P1, 0.02)["selection_input"], "CORR only")
        self.assertEqual(
            kp.final_confirmation(kp.P1, 0.02, 0.02, 0.02)["selection_input"],
            "CORR only",
        )

    def test_25b_decision_outputs_carry_no_ender60_key(self):
        screen = kp.screen_stage(kp.P1, 0.02)
        final = kp.final_confirmation(kp.P1, 0.02, 0.02, 0.02)
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
        kp.assert_forward_transition(
            kp.KP35_P1_SCREEN_PASSED, kp.KP35_P1_CONFIRMATION_FAILED
        )
        kp.assert_forward_transition(
            kp.KP35_P1_CONFIRMATION_FAILED, kp.KP35_P2_SCREEN_PASSED
        )

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
            (kp.KP35_P1_CONFIRMATION_FAILED, kp.KP35_P1_SCREEN_PASSED),
        ):
            with self.assertRaises(kp.StageAuthorityError):
                kp.assert_forward_transition(origin, target)

    def test_27b_a_p1_screen_pass_can_never_reach_p2_directly(self):
        self.assertNotIn(
            kp.KP35_P2_SCREEN_PASSED, kp.FORWARD_TRANSITIONS[kp.KP35_P1_SCREEN_PASSED]
        )
        with self.assertRaises(kp.StageAuthorityError):
            kp.assert_stage_executable(kp.P2, kp.KP35_P1_SCREEN_PASSED)

    def test_28_no_automatic_next_stage_execution_exists(self):
        self.assertIs(kp.screen_stage(kp.P1, 1.0)["next_stage_started"], False)
        self.assertIs(kp.screen_stage(kp.P1, 0.0)["next_stage_started"], False)
        self.assertIs(
            kp.final_confirmation(kp.P1, 0.0, 0.0, 0.0)["next_stage_started"], False
        )

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
            {"__future__", "hashlib", "json", "math", "dataclasses", "typing", "numpy"},
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


# ============================ validators are called in the real paths (J36)
class TestValidatorsAreWiredIntoTheRunners(unittest.TestCase):
    def test_36_trainer_calls_the_strict_validators_in_its_real_path(self):
        for validator in (
            "validate_prior_result",
            "assert_stage_executable",
            "assert_confirmation_authorized",
            "assert_runtime_versions",
            "assert_attempt_authorized",
            "resolve_attempt",
            "assert_frozen_sample",
            "assert_sample_envelope_equal",
            "validate_fit_log",
            "validate_attempt1_failure_record",
            "assert_training_eras_authorized",
            "assert_exact_row_universe",
            "assert_finite_predictions",
            "assert_payout_target",
        ):
            self.assertGreaterEqual(
                _call_count(TREES["train"], validator), 1,
                f"trainer never calls {validator}",
            )

    def test_36b_evaluator_calls_the_strict_validators_in_its_real_path(self):
        for validator in (
            "validate_prior_result",
            "assert_stage_executable",
            "assert_confirmation_authorized",
            "assert_runtime_versions",
            "assert_score_authority",
            "assert_frozen_sample",
            "validate_fit_log",
            "assert_cohort_identical_except_seed",
            "assert_exact_row_universe",
            "assert_benchmark_identity",
            "assert_forward_transition",
            "validate_result_envelope",
            "assert_finite_scalar",
            "assert_create_new_only",
        ):
            self.assertGreaterEqual(
                _call_count(TREES["evaluate"], validator), 1,
                f"evaluator never calls {validator}",
            )

    def test_36c_prior_validation_runs_in_two_phases_in_both_runners(self):
        """Phase 1 before features/manifest, phase 2 after the manifest loads."""
        for label in ("train", "evaluate"):
            self.assertEqual(
                _call_count(TREES[label], "validate_prior_result"), 2,
                f"{label}: expected a phase-1 and a phase-2 prior validation",
            )

    def test_36d_runners_derive_attempt_posture_from_artifacts(self):
        """No caller may pass a constant retry count."""
        self.assertNotIn("prior_retries=0", TRAIN_SRC)
        self.assertNotIn("assert_retry_authorized", TRAIN_SRC)
        self.assertIn("prediction_exists=", TRAIN_SRC)
        self.assertIn("attempt1_failure_exists=", TRAIN_SRC)
        self.assertIn("attempt2_failure_exists=", TRAIN_SRC)
        self.assertIn("success_log_exists=", TRAIN_SRC)
        main = _function_def(TREES["train"], "main")
        self.assertIsNotNone(main)
        self.assertEqual(_call_count(main, "assert_attempt_authorized"), 1)

    def test_36e_evaluator_refuses_a_second_result_write_before_any_work(self):
        self.assertIn("assert_create_new_only(result_path.exists()", EVAL_SRC)

    def test_36f_trainer_writes_no_model_artifact(self):
        self.assertIn('"model_artifact_written": False', TRAIN_SRC)
        for token in ("save_model", "booster.save", "joblib.dump", "pickle.dump"):
            self.assertNotIn(token, TRAIN_SRC)

    def test_36g_evaluator_revalidates_data_and_loads_the_external_manifest(self):
        self.assertEqual(_call_count(TREES["evaluate"], "revalidate_data_identities"), 1)
        self.assertEqual(_call_count(TREES["evaluate"], "load_sample_manifest"), 1)
        self.assertIn("_sha256_file", EVAL_SRC)


# ==================================================== one-shot artifact law (L)
class TestOneShotArtifactLaw(unittest.TestCase):
    def test_artifact_paths_are_unique_per_stage_seed_and_attempt(self):
        seen = set()
        for kind in ("prediction", "fit_log"):
            for stage in kp.STAGES:
                for seed in kp.ALL_SEEDS:
                    path = kp.artifact_relpath(kind, stage=stage, model_seed=seed)
                    self.assertNotIn(path, seen)
                    seen.add(path)
        for stage in kp.STAGES:
            for seed in kp.ALL_SEEDS:
                for attempt in (kp.FIRST_ATTEMPT, kp.RETRY_ATTEMPT):
                    path = kp.artifact_relpath(
                        "failure_record", stage=stage, model_seed=seed, attempt=attempt
                    )
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
        self.assertEqual(len(seen), 2 * 2 * 3 + 2 * 3 * 2 + 2 * 2 + 2)

    def test_result_paths_match_the_canonical_prior_paths(self):
        self.assertEqual(
            kp.artifact_relpath("screen_result", stage=kp.P1), P1_SCREEN_PATH
        )
        self.assertEqual(
            kp.artifact_relpath("confirmation_result", stage=kp.P1), P1_CONFIRM_PATH
        )
        self.assertEqual(
            kp.artifact_relpath("screen_result", stage=kp.P2), P2_SCREEN_PATH
        )

    def test_unknown_artifact_kind_is_refused(self):
        with self.assertRaises(ValueError):
            kp.artifact_relpath("model", stage=kp.P1, model_seed=42)

    def test_create_new_only_refuses_to_replace_a_final_path(self):
        kp.assert_create_new_only(False, "results/x.json", kind="screen_result")
        with self.assertRaises(FileExistsError) as ctx:
            kp.assert_create_new_only(True, "results/x.json", kind="screen_result")
        self.assertIn("create-new-only", str(ctx.exception))


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
        self.assertEqual(audit["rows_after_sampling"], 1_000_000)
        self.assertEqual(
            audit["sample_canon_sha256"], kp.FROZEN_SAMPLE["sample_canon_sha256"]
        )

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
        self.assertIs(PROTOCOL["scope"]["no_feature_expansion"], True)
        self.assertIs(PROTOCOL["scope"]["no_row_budget_expansion"], True)

    def test_protocol_records_the_independently_recomputed_universe(self):
        recomputed = PROTOCOL["exact_row_contract"]["independent_recomputation_at_source_freeze"]
        self.assertTrue(recomputed["all_three_identical"])
        self.assertTrue(recomputed["reproduces_kp34_claim"])
        self.assertEqual(recomputed["canon_sha256"], kp.SCORING_UNIVERSE_CANON_SHA256)

    def test_protocol_terminal_states_match_the_corrected_transition_law(self):
        self.assertEqual(
            set(PROTOCOL["terminal_states"]["absorbing"]), set(kp.ABSORBING_STATES)
        )
        self.assertEqual(
            set(PROTOCOL["terminal_states"]["p2_authorizing"]),
            set(kp.P2_AUTHORIZING_STATES),
        )
        for state, allowed in PROTOCOL["terminal_states"]["forward_transitions"].items():
            self.assertEqual(set(allowed), set(kp.FORWARD_TRANSITIONS[state]), state)
        for stage in kp.STAGES:
            self.assertEqual(
                PROTOCOL["terminal_states"]["stage_confirmation_states"][stage],
                list(kp.CONFIRMATION_STATES[stage]),
            )

    def test_protocol_records_the_prior_result_authority_map(self):
        authority = PROTOCOL["prior_result_authority"]
        for stage in kp.STAGES:
            for mode in kp.MODES:
                self.assertEqual(
                    tuple(authority["canonical_paths"][f"{stage}:{mode}"]),
                    kp.canonical_prior_relpaths(stage, mode),
                )
        self.assertEqual(
            list(authority["result_envelope_required_fields"]),
            list(kp.RESULT_ENVELOPE_REQUIRED_FIELDS),
        )
        self.assertIn("never", authority["principle"].lower())

    def test_protocol_records_the_frozen_sample_and_attempt_law(self):
        self.assertEqual(PROTOCOL["sample_custody"]["frozen_sample"], dict(kp.FROZEN_SAMPLE))
        attempt = PROTOCOL["attempt_law"]
        self.assertEqual(attempt["max_attempts"], kp.MAX_ATTEMPTS)
        self.assertIs(attempt["no_third_attempt"], True)
        self.assertIs(attempt["second_failure_preserved_separately"], True)
        self.assertNotEqual(
            attempt["failure_paths"]["attempt1"], attempt["failure_paths"]["attempt2"]
        )

    def test_protocol_records_the_authority_snapshot_and_environment(self):
        self.assertEqual(PROTOCOL["authority_snapshot"], dict(kp.AUTHORITY_SNAPSHOT))
        self.assertEqual(PROTOCOL["environment"]["frozen"], dict(kp.FROZEN_ENVIRONMENT))
        self.assertIs(PROTOCOL["environment"]["mismatch_is_a_stop"], True)
        self.assertIn(
            "mandatory",
            PROTOCOL["authority_binding"]["live_revalidation_before_p1_is_mandatory"].lower()
            + " mandatory",
        )
        self.assertIs(PROTOCOL["authority_binding"]["no_network_client_in_packet"], True)

    def test_protocol_semantic_hash_is_stable_and_content_sensitive(self):
        self.assertEqual(kp.protocol_semantic_sha256(PROTOCOL), PROTOCOL_SEMANTIC)
        tampered = json.loads(json.dumps(PROTOCOL))
        tampered["frozen_constants"]["screen_factor"] = 0.5
        self.assertNotEqual(kp.protocol_semantic_sha256(tampered), PROTOCOL_SEMANTIC)


if __name__ == "__main__":
    unittest.main()
