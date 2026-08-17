"""Focused synthetic tests for the Keystone Round-1 walk-forward laws (KW33).

Deterministic, CPU-only, dataset-free: numpy/pandas fixtures plus the Round-0
harness only. Proves the frozen fold boundaries, the eight-era embargo, the
no-scored-row-training law, the deterministic era-balanced sampling law and
its cap, prediction-vector assembly rules, score-frame filtering, GAP/HOLDOUT
refusal, the pre-registered decision law and its reconstruction, zero-baseline
drawdown reuse, and Ender60's inability to select a winner.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from agents.code.metrics import keystone_round0 as kr0
from agents.experiments.keystone28_corr_backbone_v53 import round1_lib as r1


def _authority() -> kr0.ScoreAuthority:
    return kr0.ScoreAuthority(
        payout_target="target_ender_20",
        corr_multiplier=0.75,
        mmc_multiplier=2.25,
        meta_model_column="numerai_meta_model",
        corr_score_name="CORR20V2",
        mmc_score_name="MMC20",
        bmm_aggregate_authority=None,
        retrieved_utc="2026-08-17T00:00:00+00:00",
        documentation_authority=("test",),
    )


class TestFoldBoundaries(unittest.TestCase):
    def test_blocks_partition_the_exact_87_era_zone(self):
        zone = r1.score_zone_eras()
        self.assertEqual(len(zone), 87)
        self.assertEqual(zone[0], "1133")
        self.assertEqual(zone[-1], "1219")
        concatenated = []
        for i in range(len(r1.WALK_FORWARD_BLOCKS)):
            concatenated.extend(r1.block_eras(i))
        self.assertEqual(concatenated, zone)
        self.assertEqual(
            [len(r1.block_eras(i)) for i in range(6)], [15, 15, 15, 15, 15, 12]
        )

    def test_block_definitions_are_the_frozen_six(self):
        self.assertEqual(
            r1.WALK_FORWARD_BLOCKS,
            ((1133, 1147), (1148, 1162), (1163, 1177), (1178, 1192), (1193, 1207), (1208, 1219)),
        )


class TestEmbargoAndEligibility(unittest.TestCase):
    def test_eight_era_embargo_before_every_scored_block(self):
        for i, (start, _end) in enumerate(r1.WALK_FORWARD_BLOCKS):
            latest = r1.latest_eligible_validation_era(f"{start:04d}")
            self.assertEqual(int(latest), start - 9)
            embargo = r1.embargo_eras_for_block(i)
            self.assertEqual(len(embargo), 8)
            self.assertEqual(embargo[0], f"{start - 8:04d}")
            self.assertEqual(embargo[-1], f"{start - 1:04d}")
            self.assertTrue(all(int(latest) < int(e) for e in embargo))

    def test_no_scored_row_and_no_embargo_row_is_ever_eligible(self):
        available = [f"{e:04d}" for e in range(575, 1220)]
        for i in range(6):
            eligible = set(r1.eligible_validation_eras(i, available))
            self.assertFalse(eligible & set(r1.block_eras(i)))
            self.assertFalse(eligible & set(r1.embargo_eras_for_block(i)))
            zone_after = {e for e in r1.score_zone_eras() if int(e) >= r1.WALK_FORWARD_BLOCKS[i][0]}
            self.assertFalse(eligible & zone_after)

    def test_control_t_never_uses_validation_history(self):
        for spec in r1.build_fit_specs():
            if spec.procedure == r1.CONTROL_T:
                self.assertFalse(spec.uses_validation_history)
                self.assertIsNone(spec.latest_validation_train_era)

    def test_gap_and_holdout_are_refused_everywhere(self):
        with self.assertRaisesRegex(ValueError, "forbidden GAP/HOLDOUT"):
            r1.assert_no_forbidden_eras(["1222", "1223"], context="test")
        with self.assertRaisesRegex(ValueError, "forbidden GAP/HOLDOUT"):
            r1.assert_no_forbidden_eras(["1231"], context="test")
        r1.assert_no_forbidden_eras(["1222"], context="test")  # dev end is fine
        # Even if GAP/HOLDOUT eras are offered as available, eligibility never
        # returns them.
        available = [f"{e:04d}" for e in range(575, 1240)]
        for i in range(6):
            eligible = r1.eligible_validation_eras(i, available)
            self.assertTrue(all(int(e) < 1223 for e in eligible))
            self.assertTrue(all(int(e) <= 1199 for e in eligible) or i < 5)


class TestSamplingLaw(unittest.TestCase):
    def _cohort(self, rows_per_era, n_eras=10, start=100):
        eras = []
        for k in range(n_eras):
            eras.extend([f"{start + k:04d}"] * rows_per_era)
        return np.array(eras)

    def test_under_cap_takes_every_row(self):
        eras = self._cohort(50)
        positions = r1.era_balanced_sample_positions(eras, cap=1000, seed=7)
        self.assertEqual(len(positions), 500)
        self.assertTrue(np.array_equal(positions, np.arange(500)))

    def test_cap_is_exact_and_era_balanced(self):
        eras = self._cohort(100)
        positions = r1.era_balanced_sample_positions(eras, cap=250, seed=7)
        self.assertEqual(len(positions), 250)
        counts = pd.Series(eras[positions]).value_counts()
        self.assertEqual(int(counts.min()), 25)
        self.assertEqual(int(counts.max()), 25)

    def test_small_eras_contribute_everything_and_remainder_is_deterministic(self):
        eras = np.array(["0001"] * 10 + ["0002"] * 100 + ["0003"] * 100)
        positions = r1.era_balanced_sample_positions(eras, cap=90, seed=7)
        counts = pd.Series(eras[positions]).value_counts().sort_index()
        self.assertEqual(int(counts["0001"]), 10)
        self.assertEqual(len(positions), 90)

    def test_sampling_is_deterministic_and_independent_of_model_seed(self):
        eras = self._cohort(100)
        first = r1.era_balanced_sample_positions(eras, cap=300, seed=r1.SAMPLING_SEED)
        second = r1.era_balanced_sample_positions(eras, cap=300, seed=r1.SAMPLING_SEED)
        self.assertTrue(np.array_equal(first, second))
        for spec in r1.build_fit_specs():
            self.assertEqual(spec.sampling_seed, r1.SAMPLING_SEED)
        different = r1.era_balanced_sample_positions(eras, cap=300, seed=1)
        self.assertFalse(np.array_equal(first, different))

    def test_frozen_cap_constant(self):
        self.assertEqual(r1.MAX_SAMPLED_ROWS, 1_000_000)


class TestFitSpecs(unittest.TestCase):
    def test_exact_21_fit_cohort(self):
        specs = r1.build_fit_specs()
        self.assertEqual(len(specs), 21)
        self.assertEqual(
            sum(1 for s in specs if s.procedure == r1.CONTROL_T), 3
        )
        self.assertEqual(
            sum(1 for s in specs if s.procedure == r1.CANDIDATE_V), 18
        )
        ids = [s.fit_id for s in specs]
        self.assertEqual(len(set(ids)), 21)

    def test_candidate_blocks_cover_zone_exactly_once_per_seed(self):
        specs = r1.build_fit_specs()
        for seed in r1.MODEL_SEEDS:
            eras = []
            for s in specs:
                if s.procedure == r1.CANDIDATE_V and s.model_seed == seed:
                    eras.extend(s.scored_eras)
            self.assertEqual(sorted(eras), r1.score_zone_eras())

    def test_lightgbm_params_identical_between_procedures_and_seeded(self):
        profile = {
            "learning_rate": 0.005,
            "max_depth": 8,
            "num_leaves": 255,
            "min_data_in_leaf": 10000,
            "feature_fraction": 0.1,
            "num_threads": 8,
        }
        a = r1.lightgbm_params(profile, 42)
        b = r1.lightgbm_params(profile, 42)
        self.assertEqual(a, b)
        c = r1.lightgbm_params(profile, 1337)
        self.assertEqual(
            {k: v for k, v in a.items() if k != "seed"},
            {k: v for k, v in c.items() if k != "seed"},
        )
        self.assertEqual(a["objective"], "regression")
        self.assertTrue(a["deterministic"])
        self.assertNotIn("early_stopping_round", a)
        self.assertNotIn("early_stopping_rounds", a)


class TestVectorAndFrameLaws(unittest.TestCase):
    def test_prediction_vector_requires_exact_zone_and_unique_ids(self):
        zone = r1.score_zone_eras()
        ids = [f"id{i}" for i in range(len(zone))]
        r1.validate_prediction_vector(ids, zone)  # exact zone passes
        with self.assertRaisesRegex(ValueError, "duplicate"):
            r1.validate_prediction_vector(["a", "a"], zone[:2])
        with self.assertRaisesRegex(ValueError, "87-era"):
            r1.validate_prediction_vector(ids[:-1], zone[:-1])
        with self.assertRaisesRegex(ValueError, "87-era"):
            r1.validate_prediction_vector(ids + ["extra"], zone + ["1220"])

    def test_score_frame_projection_mask(self):
        eras = ["1132", "1133", "1219", "1220", "1223", "1231"]
        mask = r1.project_to_scored_eras(eras)
        self.assertEqual(mask, [False, True, True, False, False, False])


class TestDecisionLaw(unittest.TestCase):
    def _views(self, uplift, blocks_positive, worst_gap, sharpe_gap, dd_ratio):
        control = {
            "seed_mean_weighted_mean": 0.010,
            "block_means": [0.010] * 6,
            "worst_seed_weighted_mean": 0.008,
            "worst_seed_sharpe": 0.50,
            "worst_seed_drawdown": 0.020,
        }
        blocks = [
            0.010 + (0.002 if i < blocks_positive else -0.002) for i in range(6)
        ]
        candidate = {
            "seed_mean_weighted_mean": 0.010 + uplift,
            "block_means": blocks,
            "worst_seed_weighted_mean": 0.008 - worst_gap,
            "worst_seed_sharpe": 0.50 - sharpe_gap,
            "worst_seed_drawdown": 0.020 * dd_ratio,
        }
        return candidate, control

    def test_all_conditions_pass_promotes(self):
        candidate, control = self._views(0.0003, 5, 0.0, 0.0, 1.0)
        decision = r1.decide_round1(candidate, control)
        self.assertTrue(decision["promoted"])
        self.assertEqual(
            decision["terminal_state"], "KEYSTONE_R1_RECENCY_AUGMENTATION_PROMOTED"
        )
        self.assertTrue(all(c["passed"] for c in decision["conditions"].values()))

    def test_each_condition_fails_independently(self):
        cases = {
            "1_seed_mean_weighted_uplift": self._views(0.0002, 5, 0.0, 0.0, 1.0),
            "2_positive_block_differences": self._views(0.0003, 3, 0.0, 0.0, 1.0),
            "3_worst_seed_mean_guard": self._views(0.0003, 5, 0.00051, 0.0, 1.0),
            "4_worst_seed_sharpe_guard": self._views(0.0003, 5, 0.0, 0.051, 1.0),
            "5_worst_seed_drawdown_guard": self._views(0.0003, 5, 0.0, 0.0, 1.26),
        }
        for failing, (candidate, control) in cases.items():
            decision = r1.decide_round1(candidate, control)
            self.assertFalse(decision["promoted"], failing)
            self.assertFalse(decision["conditions"][failing]["passed"], failing)
            self.assertEqual(
                decision["terminal_state"], "KEYSTONE_R1_NEGATIVE_NO_RECENCY_GAIN"
            )

    def test_decision_reconstructs_from_per_era_values(self):
        rng = np.random.default_rng(3)
        control_series = rng.normal(0.004, 0.02, 87)
        candidate_series = control_series + 0.0004
        def view(series):
            stats = r1.zero_baseline_stats(series)
            means, pos = [], 0
            for start, end in r1.WALK_FORWARD_BLOCKS:
                n = end - start + 1
                means.append(float(np.mean(series[pos : pos + n])))
                pos += n
            return {
                "seed_mean_weighted_mean": stats["mean"],
                "block_means": means,
                "worst_seed_weighted_mean": stats["mean"],
                "worst_seed_sharpe": stats["sharpe"],
                "worst_seed_drawdown": stats["max_drawdown_zero_baseline"],
            }
        first = r1.decide_round1(view(candidate_series), view(control_series))
        second = r1.decide_round1(view(candidate_series), view(control_series))
        self.assertEqual(first, second)
        self.assertTrue(first["promoted"])

    def test_bootstrap_is_seed_fixed_and_reproducible(self):
        rng = np.random.default_rng(5)
        diff = rng.normal(0.0003, 0.001, 87)
        a = r1.moving_block_bootstrap_ci(diff, n_resamples=500)
        b = r1.moving_block_bootstrap_ci(diff, n_resamples=500)
        self.assertEqual(a, b)
        self.assertLess(a["ci_low"], a["ci_high"])
        self.assertEqual(a["block_length"], 8)
        self.assertEqual(a["seed"], r1.SAMPLING_SEED)

    def test_pipeline_parity_gate(self):
        self.assertTrue(r1.pipeline_parity_ok(0.014, 0.020))
        self.assertFalse(r1.pipeline_parity_ok(0.0139, 0.020))


class TestZeroBaselineDrawdownReuse(unittest.TestCase):
    def test_round1_stats_match_round0_summary_convention_exactly(self):
        values = [-0.02, 0.05, -0.04, -0.01, 0.03]
        eras = [f"{i + 1:04d}" for i in range(len(values))]
        frame = pd.DataFrame(
            {"corr": values, "mmc": values, "weighted_score": values}, index=eras
        )
        frame.index.name = "era"
        summary = kr0.summarize_round0(frame, _authority())
        stats = r1.zero_baseline_stats(values)
        ref = summary["scores"]["weighted_score"]
        self.assertEqual(stats["mean"], ref["mean"])
        self.assertEqual(stats["std"], ref["std"])
        self.assertEqual(stats["sharpe"], ref["sharpe"])
        self.assertEqual(stats["max_drawdown_zero_baseline"], ref["max_drawdown"])


class TestEnder60CannotSelect(unittest.TestCase):
    def test_decision_law_has_no_ender60_input_path(self):
        candidate = {
            "seed_mean_weighted_mean": 0.011,
            "block_means": [0.012] * 6,
            "worst_seed_weighted_mean": 0.009,
            "worst_seed_sharpe": 0.6,
            "worst_seed_drawdown": 0.01,
            "ender60_anything": 999.0,
        }
        control = {
            "seed_mean_weighted_mean": 0.010,
            "block_means": [0.010] * 6,
            "worst_seed_weighted_mean": 0.008,
            "worst_seed_sharpe": 0.5,
            "worst_seed_drawdown": 0.02,
        }
        base = r1.decide_round1(candidate, control)
        mutated = dict(candidate)
        mutated["ender60_anything"] = -999.0
        self.assertEqual(r1.decide_round1(mutated, control), base)

    def test_aux_authority_is_labeled_and_non_payout(self):
        aux = r1.aux_authority_ender60(_authority())
        self.assertEqual(aux.payout_target, "target_ender_60")
        self.assertIn("AUXILIARY", aux.corr_score_name)
        self.assertIn("AUXILIARY", aux.mmc_score_name)
        self.assertIsNone(aux.bmm_aggregate_authority)
        self.assertTrue(
            any(r1.ENDER60_AUX_LABEL in doc for doc in aux.documentation_authority)
        )
        self.assertEqual(r1.ENDER60_AUX_LABEL, "HYPOTHETICAL_CUTOVER_WEIGHTED_DIAGNOSTIC")


if __name__ == "__main__":
    unittest.main()
