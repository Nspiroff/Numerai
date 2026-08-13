from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from numerai_tools.scoring import (
    correlation_contribution,
    gaussian,
    orthogonalize,
    tie_kept_rank,
)
from scipy.special import ndtri

from agents.code.modeling.utils.target_transforms import (
    _tie_kept_rank_gaussian,
    apply_target_transform,
)


class TestTargetTransforms(unittest.TestCase):
    @staticmethod
    def _legacy_residual(
        y: pd.Series,
        benchmark: pd.Series,
        eras: pd.Series,
    ) -> pd.Series:
        expected = np.full(len(y), np.nan, dtype="float64")
        for era in pd.unique(eras.dropna()):
            mask = (eras == era).to_numpy() & np.isfinite(y) & np.isfinite(
                benchmark
            )
            yy = y.to_numpy(dtype="float64", copy=False)[mask]
            xx = benchmark.to_numpy(dtype="float64", copy=False)[mask]
            yy = yy - yy.mean()
            xx = xx - xx.mean()
            denominator = float(np.dot(xx, xx))
            alpha = float(np.dot(xx, yy) / denominator) if denominator else 0.0
            expected[mask] = yy - alpha * xx
        return pd.Series(expected, index=y.index, name=y.name)

    def test_residual_to_benchmark_is_orthogonal_per_era(self) -> None:
        rng = np.random.default_rng(7)
        n = 2000
        eras = np.array(["0001"] * (n // 2) + ["0002"] * (n - n // 2))
        benchmark = rng.normal(size=n)

        # Different slopes per era + intercept
        y = np.empty(n, dtype="float64")
        y[: n // 2] = 2.0 * benchmark[: n // 2] + 1.0 + rng.normal(scale=0.1, size=n // 2)
        y[n // 2 :] = -3.0 * benchmark[n // 2 :] - 0.5 + rng.normal(
            scale=0.1, size=n - n // 2
        )

        X = pd.DataFrame({"era": eras, "v53_lgbm_ender20": benchmark})
        y = pd.Series(y, name="target")

        transformed = apply_target_transform(
            y,
            X,
            {
                "type": "residual_to_benchmark",
                "benchmark_col": "v53_lgbm_ender20",
                "era_col": "era",
                "per_era": True,
                "fit_intercept": True,
                "proportion": 1.0,
            },
        )

        for era in ("0001", "0002"):
            mask = X["era"].to_numpy() == era
            corr = np.corrcoef(transformed[mask], benchmark[mask])[0, 1]
            self.assertLess(abs(float(corr)), 1e-10)

    def test_subtract_benchmark_zscore_is_per_era_and_scaled(self) -> None:
        eras = np.array(["0001", "0001", "0002", "0002"])
        benchmark = np.array([1.0, 3.0, 10.0, 14.0])
        X = pd.DataFrame({"era": eras, "v53_lgbm_ender20": benchmark})
        y = pd.Series([0.5, 0.5, 0.5, 0.5], name="target")

        transformed = apply_target_transform(
            y,
            X,
            {"type": "subtract_benchmark", "benchmark_col": "v53_lgbm_ender20"},
        )

        expected = pd.Series([0.57, 0.43, 0.57, 0.43], name="target")
        self.assertTrue(np.allclose(transformed.to_numpy(), expected.to_numpy()))

    def test_subtract_benchmark_zscore_zero_std_is_noop(self) -> None:
        eras = np.array(["0001", "0001", "0001", "0001"])
        benchmark = np.array([2.0, 2.0, 2.0, 2.0])
        X = pd.DataFrame({"era": eras, "v53_lgbm_ender20": benchmark})
        y = pd.Series([0.1, 0.2, 0.3, 0.4], name="target")

        transformed = apply_target_transform(
            y,
            X,
            {
                "type": "subtract_benchmark_zscore",
                "benchmark_col": "v53_lgbm_ender20",
                "scale": 0.07,
            },
        )
        self.assertTrue(np.allclose(transformed.to_numpy(), y.to_numpy()))

    def test_proportion_blends_with_original_target(self) -> None:
        rng = np.random.default_rng(1337)
        n = 1000
        eras = np.array(["0001"] * n)
        benchmark = rng.normal(size=n)
        y = 1.5 * benchmark + 0.2 + rng.normal(scale=0.3, size=n)

        X = pd.DataFrame({"era": eras, "v53_lgbm_ender20": benchmark})
        y = pd.Series(y, name="target")

        y_same = apply_target_transform(
            y,
            X,
            {
                "type": "residual_to_benchmark",
                "benchmark_col": "v53_lgbm_ender20",
                "era_col": "era",
                "per_era": True,
                "fit_intercept": True,
                "proportion": 0.0,
            },
        )
        self.assertTrue(np.allclose(y_same.to_numpy(), y.to_numpy()))

        y_resid = apply_target_transform(
            y,
            X,
            {
                "type": "residual_to_benchmark",
                "benchmark_col": "v53_lgbm_ender20",
                "era_col": "era",
                "per_era": True,
                "fit_intercept": True,
                "proportion": 1.0,
            },
        )
        y_half = apply_target_transform(
            y,
            X,
            {
                "type": "residual_to_benchmark",
                "benchmark_col": "v53_lgbm_ender20",
                "era_col": "era",
                "per_era": True,
                "fit_intercept": True,
                "proportion": 0.5,
            },
        )
        expected = 0.5 * y + 0.5 * y_resid
        self.assertTrue(np.allclose(y_half.to_numpy(), expected.to_numpy()))

    def test_default_and_explicit_identity_exactly_match_legacy_residual(self) -> None:
        X = pd.DataFrame(
            {
                "era": ["0002", "0001", "0002", "0001", "0002", "0001"],
                "benchmark": [0.9, 0.1, 0.5, 0.3, 0.2, 0.8],
            },
            index=[7, 2, 9, 1, 8, 3],
        )
        y = pd.Series([0.7, 0.1, 0.4, 0.6, 0.2, 0.9], index=X.index, name="target")
        legacy = self._legacy_residual(y, X["benchmark"], X["era"])
        base = {
            "type": "residual_to_benchmark",
            "benchmark_col": "benchmark",
            "era_col": "era",
            "per_era": True,
            "fit_intercept": True,
            "proportion": 1.0,
        }

        omitted = apply_target_transform(y, X, base)
        explicit = apply_target_transform(
            y, X, {**base, "benchmark_transform": "identity"}
        )

        np.testing.assert_array_equal(omitted.to_numpy(), legacy.to_numpy())
        np.testing.assert_array_equal(explicit.to_numpy(), legacy.to_numpy())
        self.assertTrue(omitted.index.equals(y.index))
        self.assertEqual(omitted.name, y.name)

    def test_tie_kept_rank_gaussian_is_per_era_finite_and_order_preserving(self) -> None:
        benchmark = pd.Series(
            [4.0, 1.0, np.nan, 1.0, 9.0, 4.0, 9.0, 5.0, 5.0],
            index=[90, 10, 70, 20, 80, 30, 60, 40, 50],
            name="benchmark",
        )
        eras = pd.Series(
            ["b", "a", "a", "a", "b", "b", "b", "c", "c"],
            index=benchmark.index,
        )

        transformed = _tie_kept_rank_gaussian(benchmark, groups=eras)

        expected = pd.Series(np.nan, index=benchmark.index, name="benchmark")
        for era in ("a", "b", "c"):
            values = benchmark[eras == era].dropna()
            ranks = values.rank(method="average")
            expected.loc[values.index] = ndtri(
                ((ranks - 0.5) / len(values)).to_numpy()
            )
        np.testing.assert_allclose(
            transformed.to_numpy(), expected.to_numpy(), rtol=0.0, atol=0.0,
            equal_nan=True,
        )
        self.assertTrue(transformed.index.equals(benchmark.index))
        self.assertEqual(transformed.name, benchmark.name)
        self.assertTrue(np.isfinite(transformed.dropna()).all())
        self.assertTrue(np.isnan(transformed.loc[70]))
        np.testing.assert_array_equal(
            transformed[eras == "c"].to_numpy(), np.zeros(2, dtype="float64")
        )

    def test_tie_kept_rank_gaussian_global_mode_ignores_era_boundaries(self) -> None:
        benchmark = pd.Series(
            [3.0, np.nan, 1.0, 3.0, 2.0],
            index=[4, 1, 5, 2, 3],
            name="benchmark",
        )
        values = benchmark.dropna()
        expected = pd.Series(np.nan, index=benchmark.index, name=benchmark.name)
        expected.loc[values.index] = ndtri(
            ((values.rank(method="average") - 0.5) / len(values)).to_numpy()
        )

        transformed = _tie_kept_rank_gaussian(benchmark, groups=None)

        np.testing.assert_allclose(
            transformed.to_numpy(), expected.to_numpy(), rtol=0.0, atol=0.0,
            equal_nan=True,
        )
        self.assertTrue(np.isfinite(transformed.dropna()).all())

    def test_bmc_aligned_residual_is_exact_target_side_score_projection(self) -> None:
        index = pd.Index([f"id-{value:02d}" for value in range(12)])
        benchmark = pd.Series(
            [0.1, 0.2, 0.2, 0.4, 0.7, 0.9, 0.05, 0.3, 0.3, 0.6, 0.8, 0.95],
            index=index,
            name="benchmark",
        )
        prediction = pd.DataFrame(
            {
                "prediction": [
                    0.8, 0.1, 0.4, 0.3, 0.9, 0.6,
                    0.2, 0.7, 0.5, 0.95, 0.35, 0.55,
                ]
            },
            index=index,
        )
        target = pd.Series(
            [0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.25, 0.0, 0.75, 1.0, 0.5, 0.25],
            index=index,
            name="target",
        )
        X = pd.DataFrame(
            {"era": ["0001"] * len(index), "benchmark": benchmark}, index=index
        )
        residual = apply_target_transform(
            target,
            X,
            {
                "type": "residual_to_benchmark",
                "benchmark_col": "benchmark",
                "era_col": "era",
                "per_era": True,
                "fit_intercept": True,
                "proportion": 1.0,
                "benchmark_transform": "tie_kept_rank_gaussian",
            },
        )

        prediction_gaussian = gaussian(tie_kept_rank(prediction))[
            "prediction"
        ].to_numpy()
        target_side_score = (
            4.0 * float(np.dot(residual.to_numpy(), prediction_gaussian)) / len(index)
        )
        direct_score = float(
            correlation_contribution(
                prediction, benchmark, target.copy()
            )["prediction"]
        )

        self.assertAlmostEqual(target_side_score, direct_score, places=14)
        benchmark_gaussian = _tie_kept_rank_gaussian(benchmark, groups=X["era"])
        centered_target = 4.0 * target.to_numpy() - float((4.0 * target).mean())
        scorer_side_target_projection = orthogonalize(
            centered_target[:, None], benchmark_gaussian.to_numpy()
        ).ravel()
        np.testing.assert_allclose(
            4.0 * residual.to_numpy(),
            scorer_side_target_projection,
            rtol=0.0,
            atol=1e-15,
        )
        self.assertAlmostEqual(
            float(np.dot(residual.to_numpy(), benchmark_gaussian.to_numpy())),
            0.0,
            places=14,
        )

    def test_bmc_aligned_residual_handles_multiple_eras_and_restores_order(self) -> None:
        X = pd.DataFrame(
            {
                "era": ["0002", "0001", "0002", "0001", "0003", "0003"],
                "benchmark": [0.9, 0.1, 0.2, 0.8, 0.4, 0.6],
            },
            index=[5, 0, 4, 1, 3, 2],
        )
        y = pd.Series([0.8, 0.2, 0.4, 0.7, 0.1, 0.9], index=X.index)
        transformed_benchmark = _tie_kept_rank_gaussian(
            X["benchmark"], groups=X["era"]
        )

        residual = apply_target_transform(
            y,
            X,
            {
                "type": "residual_to_benchmark",
                "benchmark_col": "benchmark",
                "era_col": "era",
                "per_era": True,
                "fit_intercept": True,
                "benchmark_transform": "tie_kept_rank_gaussian",
            },
        )

        self.assertTrue(residual.index.equals(y.index))
        for era in ("0001", "0002", "0003"):
            mask = X["era"] == era
            self.assertAlmostEqual(float(residual[mask].sum()), 0.0, places=14)
            self.assertAlmostEqual(
                float(
                    np.dot(
                        residual[mask].to_numpy(),
                        transformed_benchmark[mask].to_numpy(),
                    )
                ),
                0.0,
                places=14,
            )

    def test_bmc_aligned_residual_rejects_invalid_candidate_inputs(self) -> None:
        base_x = pd.DataFrame(
            {
                "era": ["0001", "0001", "0002", "0002"],
                "benchmark": [0.1, 0.9, 0.2, 0.8],
            }
        )
        base_y = pd.Series([0.2, 0.7, 0.4, 0.9])
        transform = {
            "type": "residual_to_benchmark",
            "benchmark_col": "benchmark",
            "era_col": "era",
            "per_era": True,
            "fit_intercept": True,
            "benchmark_transform": "tie_kept_rank_gaussian",
        }

        invalid_cases = (
            (
                "malformed target",
                base_y.astype(object).where(base_y.index != 1, "bad"),
                base_x,
            ),
            ("NaN target", base_y.where(base_y.index != 1, np.nan), base_x),
            (
                "positive infinite target",
                base_y.where(base_y.index != 1, np.inf),
                base_x,
            ),
            (
                "negative infinite target",
                base_y.where(base_y.index != 1, -np.inf),
                base_x,
            ),
            (
                "malformed benchmark",
                base_y,
                base_x.assign(benchmark=[0.1, "bad", 0.2, 0.8]),
            ),
            (
                "NaN benchmark",
                base_y,
                base_x.assign(benchmark=[0.1, np.nan, 0.2, 0.8]),
            ),
            (
                "positive infinite benchmark",
                base_y,
                base_x.assign(benchmark=[0.1, np.inf, 0.2, 0.8]),
            ),
            (
                "negative infinite benchmark",
                base_y,
                base_x.assign(benchmark=[0.1, -np.inf, 0.2, 0.8]),
            ),
            (
                "missing era",
                base_y,
                base_x.assign(era=["0001", None, "0002", "0002"]),
            ),
            (
                "constant group",
                base_y,
                base_x.assign(benchmark=[0.5, 0.5, 0.2, 0.8]),
            ),
            (
                "singleton group",
                base_y.iloc[:3],
                pd.DataFrame(
                    {
                        "era": ["0001", "0002", "0002"],
                        "benchmark": [0.1, 0.2, 0.8],
                    }
                ),
            ),
        )
        for label, y, X in invalid_cases:
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError,
                "numeric-convertible and finite|non-missing era|strictly positive",
            ):
                apply_target_transform(y, X, transform)

    def test_benchmark_transform_rejects_unknown_and_non_string_values(self) -> None:
        X = pd.DataFrame(
            {"era": ["0001", "0001"], "benchmark": [0.2, 0.8]}
        )
        y = pd.Series([0.1, 0.9])
        for value in ("gaussian", "", None, 1, True):
            with self.subTest(value=value), self.assertRaisesRegex(
                (TypeError, ValueError), "benchmark_transform must be one of"
            ):
                apply_target_transform(
                    y,
                    X,
                    {
                        "type": "residual_to_benchmark",
                        "benchmark_col": "benchmark",
                        "benchmark_transform": value,
                    },
                )
