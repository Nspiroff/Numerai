"""Focused source-contract tests for the Keystone Round-0 scoring harness.

Synthetic, deterministic, CPU-only. Proves exact parity with the official
``numerai_tools`` implementations, explicit (non-implicit) scoring authority,
deterministic alignment, exact weighted-score arithmetic, reproducible summary
statistics, and loud failure on malformed inputs. No dataset, no network, no
pyarrow/numerapi imports.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from numerai_tools.scoring import correlation_contribution, numerai_corr

from agents.code.metrics import keystone_round0 as kr0


ERAS = ["0101", "0102", "0103", "0104"]
TARGET = "target_synth_20"
MM_COL = "numerai_meta_model"
ROWS_PER_ERA = 48


def _authority(**overrides) -> kr0.ScoreAuthority:
    base = dict(
        payout_target=TARGET,
        corr_multiplier=0.75,
        mmc_multiplier=2.25,
        meta_model_column=MM_COL,
        corr_score_name="CORR20V2",
        mmc_score_name="MMC20",
        bmm_aggregate_authority=None,
        retrieved_utc="2026-08-17T00:00:00+00:00",
        documentation_authority=("https://docs.numer.ai/numerai-tournament/scoring",),
    )
    base.update(overrides)
    return kr0.ScoreAuthority(**base)


def _cohort(seed: int = 11):
    rng = np.random.default_rng(seed)
    ids, eras = [], []
    for era in ERAS:
        for i in range(ROWS_PER_ERA):
            ids.append(f"{era}_id{i:03d}")
            eras.append(era)
    n = len(ids)
    predictions = pd.DataFrame(
        {"id": ids, "era": eras, "prediction": rng.random(n)}
    )
    scoring = pd.DataFrame(
        {
            "id": ids,
            "era": eras,
            TARGET: rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=n),
            # Decoy bare alias with different values: the harness must never use it.
            "target": rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=n),
        }
    )
    meta_model = pd.DataFrame({"id": ids, "era": eras, MM_COL: rng.random(n)})
    benchmarks = pd.DataFrame(
        {
            "id": ids,
            "era": eras,
            "bench_a": rng.random(n),
            "bmm_official_synth": rng.random(n),
        }
    )
    return predictions, scoring, meta_model, benchmarks


def _era_slice(frame: pd.DataFrame, era: str) -> pd.DataFrame:
    return frame[frame["era"] == era].set_index("id").sort_index()


class TestOfficialParity(unittest.TestCase):
    """The harness must reproduce numerai_tools outputs exactly."""

    def setUp(self):
        self.predictions, self.scoring, self.meta_model, self.benchmarks = _cohort()
        self.result = kr0.score_round0(
            self.predictions, self.scoring, self.meta_model, _authority()
        )

    def test_per_era_corr_matches_numerai_tools_exactly(self):
        for era in ERAS:
            preds = _era_slice(self.predictions, era)[["prediction"]]
            target = _era_slice(self.scoring, era)[TARGET]
            expected = float(numerai_corr(preds, target)["prediction"])
            self.assertEqual(self.result.per_era.loc[era, "corr"], expected)

    def test_per_era_mmc_matches_correlation_contribution_exactly(self):
        for era in ERAS:
            preds = _era_slice(self.predictions, era)[["prediction"]]
            target = _era_slice(self.scoring, era)[TARGET]
            mm = _era_slice(self.meta_model, era)[MM_COL]
            expected = float(correlation_contribution(preds, mm, target)["prediction"])
            self.assertEqual(self.result.per_era.loc[era, "mmc"], expected)

    def test_weighted_score_arithmetic_is_exact(self):
        for era in ERAS:
            corr = self.result.per_era.loc[era, "corr"]
            mmc = self.result.per_era.loc[era, "mmc"]
            self.assertEqual(
                self.result.per_era.loc[era, "weighted_score"],
                corr * 0.75 + mmc * 2.25,
            )

    def test_corr_with_meta_model_uses_official_transform(self):
        for era in ERAS:
            preds = _era_slice(self.predictions, era)[["prediction"]]
            mm = _era_slice(self.meta_model, era)[MM_COL]
            expected = float(numerai_corr(preds, mm)["prediction"])
            self.assertEqual(
                self.result.per_era.loc[era, "corr_with_meta_model"], expected
            )


class TestDeterministicAlignment(unittest.TestCase):
    def test_row_order_of_all_inputs_is_irrelevant(self):
        predictions, scoring, meta_model, benchmarks = _cohort()
        base = kr0.score_round0(
            predictions,
            scoring,
            meta_model,
            _authority(),
            benchmark_data=benchmarks,
            benchmark_cols=["bench_a"],
        )
        rng = np.random.default_rng(99)
        shuffled = kr0.score_round0(
            predictions.sample(frac=1.0, random_state=1).reset_index(drop=True),
            scoring.sample(frac=1.0, random_state=2).reset_index(drop=True),
            meta_model.sample(frac=1.0, random_state=3).reset_index(drop=True),
            _authority(),
            benchmark_data=benchmarks.sample(frac=1.0, random_state=4).reset_index(
                drop=True
            ),
            benchmark_cols=["bench_a"],
        )
        del rng
        pd.testing.assert_frame_equal(base.per_era, shuffled.per_era)
        self.assertEqual(base.summary, shuffled.summary)

    def test_repeated_execution_is_identical(self):
        predictions, scoring, meta_model, _ = _cohort()
        first = kr0.score_round0(predictions, scoring, meta_model, _authority())
        second = kr0.score_round0(predictions, scoring, meta_model, _authority())
        pd.testing.assert_frame_equal(first.per_era, second.per_era)
        self.assertEqual(first.summary, second.summary)


class TestExplicitAuthority(unittest.TestCase):
    def test_bare_target_alias_is_rejected_as_payout_authority(self):
        with self.assertRaisesRegex(ValueError, "bare dataset alias"):
            _authority(payout_target="target")

    def test_scoring_uses_exactly_the_configured_target_not_the_alias(self):
        predictions, scoring, meta_model, _ = _cohort()
        result = kr0.score_round0(predictions, scoring, meta_model, _authority())
        era = ERAS[0]
        preds = _era_slice(predictions, era)[["prediction"]]
        configured = float(
            numerai_corr(preds, _era_slice(scoring, era)[TARGET])["prediction"]
        )
        decoy = float(
            numerai_corr(preds, _era_slice(scoring, era)["target"])["prediction"]
        )
        self.assertEqual(result.per_era.loc[era, "corr"], configured)
        self.assertNotEqual(result.per_era.loc[era, "corr"], decoy)

    def test_multipliers_and_target_have_no_defaults(self):
        with self.assertRaises(TypeError):
            kr0.ScoreAuthority(  # type: ignore[call-arg]
                payout_target=TARGET,
                meta_model_column=MM_COL,
                corr_score_name="CORR20V2",
                mmc_score_name="MMC20",
                bmm_aggregate_authority=None,
                retrieved_utc="2026-08-17T00:00:00+00:00",
                documentation_authority=("doc",),
            )
        with self.assertRaises(TypeError):
            kr0.ScoreAuthority(  # type: ignore[call-arg]
                corr_multiplier=0.75,
                mmc_multiplier=2.25,
                meta_model_column=MM_COL,
                corr_score_name="CORR20V2",
                mmc_score_name="MMC20",
                bmm_aggregate_authority=None,
                retrieved_utc="2026-08-17T00:00:00+00:00",
                documentation_authority=("doc",),
            )

    def test_non_finite_multiplier_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            _authority(mmc_multiplier=float("nan"))

    def test_score_authority_record_roundtrip_and_missing_keys(self):
        record = {
            "retrieved_utc": "2026-08-17T00:00:00+00:00",
            "documentation_authority": ["https://docs.numer.ai"],
            "payout_target": TARGET,
            "corr_score_version": "CORR20V2",
            "mmc_score_version": "MMC20",
            "corr_multiplier": 0.75,
            "mmc_multiplier": 2.25,
            "meta_model_column": MM_COL,
            "bmc": {"bmm_aggregate_authority": None},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "score_authority.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            authority = kr0.ScoreAuthority.from_json(path)
        self.assertEqual(authority.payout_target, TARGET)
        self.assertEqual(authority.corr_multiplier, 0.75)
        self.assertEqual(authority.mmc_multiplier, 2.25)
        self.assertIsNone(authority.bmm_aggregate_authority)

        for missing in ("payout_target", "corr_multiplier", "mmc_multiplier", "bmc"):
            broken = {k: v for k, v in record.items() if k != missing}
            with self.assertRaisesRegex(ValueError, missing):
                kr0.ScoreAuthority.from_mapping(broken)


class TestFailLoudValidation(unittest.TestCase):
    def setUp(self):
        self.predictions, self.scoring, self.meta_model, self.benchmarks = _cohort()

    def _score(self, predictions=None, scoring=None, meta_model=None, **kwargs):
        return kr0.score_round0(
            self.predictions if predictions is None else predictions,
            self.scoring if scoring is None else scoring,
            self.meta_model if meta_model is None else meta_model,
            _authority(),
            **kwargs,
        )

    def test_duplicate_prediction_ids_fail(self):
        broken = self.predictions.copy()
        broken.loc[1, "id"] = broken.loc[0, "id"]
        with self.assertRaisesRegex(ValueError, "duplicate ids"):
            self._score(predictions=broken)

    def test_missing_scoring_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "does not cover all prediction ids"):
            self._score(scoring=self.scoring.iloc[1:])

    def test_era_disagreement_by_id_fails(self):
        broken = self.scoring.copy()
        broken.loc[0, "era"] = ERAS[1]
        with self.assertRaisesRegex(ValueError, "eras do not exactly match"):
            self._score(scoring=broken)

    def test_non_finite_predictions_fail(self):
        broken = self.predictions.copy()
        broken.loc[3, "prediction"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            self._score(predictions=broken)

    def test_missing_target_values_in_scored_eras_fail(self):
        broken = self.scoring.copy()
        broken.loc[5, TARGET] = np.nan
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            self._score(scoring=broken)

    def test_non_string_and_mixed_width_eras_fail(self):
        broken = self.predictions.copy()
        broken["era"] = np.arange(len(broken))
        with self.assertRaisesRegex(ValueError, "digit strings"):
            self._score(predictions=broken)
        mixed = self.predictions.copy()
        mixed.loc[mixed["era"] == ERAS[0], "era"] = "101"
        with self.assertRaisesRegex(ValueError, "uniform width"):
            self._score(predictions=mixed)

    def test_expected_era_set_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "expected era set"):
            self._score(expected_eras=ERAS + ["0105"])
        with self.assertRaisesRegex(ValueError, "expected era set"):
            self._score(expected_eras=ERAS[:-1])

    def test_benchmark_cols_without_benchmark_data_fail(self):
        with self.assertRaisesRegex(ValueError, "without benchmark_data"):
            self._score(benchmark_cols=["bench_a"])


class TestMetaModelCoveragePolicy(unittest.TestCase):
    def setUp(self):
        self.predictions, self.scoring, self.meta_model, _ = _cohort()

    def test_uncovered_era_fails_under_default_policy(self):
        uncovered = self.meta_model[self.meta_model["era"] != ERAS[-1]]
        with self.assertRaisesRegex(ValueError, "does not cover eras"):
            kr0.score_round0(self.predictions, self.scoring, uncovered, _authority())

    def test_uncovered_era_is_cleanly_excluded_under_declared_policy(self):
        uncovered = self.meta_model[self.meta_model["era"] != ERAS[-1]]
        result = kr0.score_round0(
            self.predictions,
            self.scoring,
            uncovered,
            _authority(),
            meta_model_policy=kr0.META_MODEL_POLICY_EXCLUDE,
        )
        self.assertEqual(result.excluded_eras, (ERAS[-1],))
        self.assertEqual(result.per_era.index.tolist(), ERAS[:-1])
        self.assertEqual(
            result.summary["excluded_eras_missing_meta_model"], [ERAS[-1]]
        )

    def test_partially_covered_era_always_fails(self):
        partial = self.meta_model.drop(index=self.meta_model.index[-1])
        for policy in (kr0.META_MODEL_POLICY_FAIL, kr0.META_MODEL_POLICY_EXCLUDE):
            with self.assertRaisesRegex(ValueError, "partial"):
                kr0.score_round0(
                    self.predictions,
                    self.scoring,
                    partial,
                    _authority(),
                    meta_model_policy=policy,
                )

    def test_unknown_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "meta_model_policy"):
            kr0.score_round0(
                self.predictions,
                self.scoring,
                self.meta_model,
                _authority(),
                meta_model_policy="repair_silently",
            )


class TestBmcGating(unittest.TestCase):
    def setUp(self):
        self.predictions, self.scoring, self.meta_model, self.benchmarks = _cohort()

    def test_bmc_is_unavailable_without_official_bmm_authority(self):
        result = kr0.score_round0(
            self.predictions,
            self.scoring,
            self.meta_model,
            _authority(),
            benchmark_data=self.benchmarks,
            benchmark_cols=["bench_a"],
        )
        self.assertNotIn("bmc", result.per_era.columns)
        self.assertEqual(result.summary["bmc"]["status"], "unavailable")
        self.assertEqual(
            result.summary["bmc"]["reason"],
            kr0.BMC_AGGREGATE_NOT_REPRODUCIBLE_FROM_PUBLISHED_FILES,
        )

    def test_single_benchmark_is_reported_as_diagnostic_corr_not_bmc(self):
        result = kr0.score_round0(
            self.predictions,
            self.scoring,
            self.meta_model,
            _authority(),
            benchmark_data=self.benchmarks,
            benchmark_cols=["bench_a"],
        )
        self.assertIn("corr_with_bench_a", result.per_era.columns)
        era = ERAS[0]
        preds = _era_slice(self.predictions, era)[["prediction"]]
        bench = _era_slice(self.benchmarks, era)["bench_a"]
        self.assertEqual(
            result.per_era.loc[era, "corr_with_bench_a"],
            float(numerai_corr(preds, bench)["prediction"]),
        )

    def test_bmc_computed_only_with_declared_official_aggregate(self):
        authority = _authority(bmm_aggregate_authority="bmm_official_synth")
        result = kr0.score_round0(
            self.predictions,
            self.scoring,
            self.meta_model,
            authority,
            benchmark_data=self.benchmarks,
        )
        self.assertEqual(result.summary["bmc"]["status"], "official")
        for era in ERAS:
            preds = _era_slice(self.predictions, era)[["prediction"]]
            target = _era_slice(self.scoring, era)[TARGET]
            bmm = _era_slice(self.benchmarks, era)["bmm_official_synth"]
            expected = float(correlation_contribution(preds, bmm, target)["prediction"])
            self.assertEqual(result.per_era.loc[era, "bmc"], expected)

    def test_declared_bmm_authority_requires_benchmark_data(self):
        authority = _authority(bmm_aggregate_authority="bmm_official_synth")
        with self.assertRaisesRegex(ValueError, "no benchmark_data"):
            kr0.score_round0(
                self.predictions, self.scoring, self.meta_model, authority
            )


def _per_era_frame(values) -> pd.DataFrame:
    eras = [f"{i + 1:04d}" for i in range(len(values))]
    frame = pd.DataFrame(
        {"corr": values, "mmc": values, "weighted_score": values}, index=eras
    )
    frame.index.name = "era"
    return frame


class TestZeroBaselineDrawdown(unittest.TestCase):
    """The drawdown convention includes the zero-equity starting baseline."""

    def _drawdown(self, values) -> float:
        summary = kr0.summarize_round0(_per_era_frame(values), _authority())
        return summary["scores"]["weighted_score"]["max_drawdown"]

    def test_all_negative_series_draws_down_the_full_cumulative_loss(self):
        values = [-0.02, -0.05, -0.01, -0.03]
        self.assertEqual(self._drawdown(values), abs(sum(values)))
        self.assertEqual(self._drawdown(values), 0.11)

    def test_initial_loss_is_not_silently_discarded(self):
        # Under a no-baseline convention this series would report only the
        # later 0.04 dip; the initial 0.10 loss from zero equity must win.
        values = [-0.10, 0.08, -0.04]
        self.assertEqual(self._drawdown(values), 0.10)

    def test_positive_then_negative_path_matches_hand_calculation(self):
        values = [0.30, -0.10, -0.15, 0.05]
        # equity = [0, 0.30, 0.20, 0.05, 0.10]; running max peaks at 0.30;
        # deepest drawdown = 0.30 - 0.05 = 0.25.
        self.assertAlmostEqual(self._drawdown(values), 0.25, places=15)
        self.assertEqual(
            kr0.summarize_round0(_per_era_frame(values), _authority())[
                "conventions"
            ]["max_drawdown"],
            kr0.DRAWDOWN_CONVENTION,
        )


class TestSummaryReproducibility(unittest.TestCase):
    def test_summary_reproduces_from_per_era_values(self):
        predictions, scoring, meta_model, _ = _cohort()
        result = kr0.score_round0(
            predictions,
            scoring,
            meta_model,
            _authority(),
            recent_window=3,
            block_size=3,
        )
        recomputed = kr0.summarize_round0(
            result.per_era, result.authority, recent_window=3, block_size=3
        )
        self.assertEqual(result.summary, recomputed)

    def test_summary_statistics_match_hand_computation(self):
        predictions, scoring, meta_model, _ = _cohort()
        result = kr0.score_round0(
            predictions, scoring, meta_model, _authority(), recent_window=2, block_size=3
        )
        values = result.per_era["weighted_score"].to_numpy(dtype="float64")
        stats = result.summary["scores"]["weighted_score"]
        self.assertEqual(stats["mean"], float(np.mean(values)))
        self.assertEqual(stats["std"], float(np.std(values, ddof=0)))
        self.assertEqual(stats["sharpe"], float(np.mean(values) / np.std(values, ddof=0)))
        equity_curve = np.concatenate(([0.0], np.cumsum(values)))
        self.assertEqual(
            stats["max_drawdown"],
            float(np.max(np.maximum.accumulate(equity_curve) - equity_curve)),
        )
        self.assertEqual(stats["recent_mean"], float(np.mean(values[-2:])))
        self.assertEqual(stats["recent_window_used"], 2)
        blocks = stats["per_block_means"]
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["n_eras"], 3)
        self.assertEqual(blocks[0]["mean"], float(np.mean(values[:3])))
        self.assertEqual(blocks[1]["n_eras"], 1)
        self.assertEqual(blocks[1]["mean"], float(np.mean(values[3:])))
        self.assertEqual(
            result.summary["conventions"]["sharpe"], kr0.SHARPE_CONVENTION
        )


if __name__ == "__main__":
    unittest.main()
