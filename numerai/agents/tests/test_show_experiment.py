from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from agents.code.analysis import show_experiment


class TestPerEraBmcCompatibility(unittest.TestCase):
    def test_preattached_benchmark_is_authoritatively_reattached(self):
        predictions = pd.DataFrame(
            {
                "id": ["c", "a", "b"],
                "era": ["0003", "0001", "0002"],
                "target": [0.7, 0.2, 0.5],
                "prediction": [0.33, 0.11, 0.22],
                "benchmark": [-3.0, -1.0, -2.0],
            }
        )
        benchmark = pd.DataFrame(
            {
                "id": ["a", "b", "c", "extra"],
                "era": ["0001", "0002", "0003", "0004"],
                "benchmark": [0.11, 0.22, 0.33, 0.44],
            }
        )
        captured = {}

        def fake_per_era_bmc(enriched, *args, **kwargs):
            captured["enriched"] = enriched.copy()
            return pd.DataFrame(
                {"prediction": [0.0, 0.0, 0.0]},
                index=["0001", "0002", "0003"],
            )

        with mock.patch.object(
            show_experiment.numerai_metrics,
            "per_era_bmc",
            side_effect=fake_per_era_bmc,
        ):
            result = show_experiment._per_era_bmc(
                predictions,
                "prediction",
                "target",
                "era",
                "id",
                benchmark,
                "benchmark",
            )

        self.assertEqual(captured["enriched"]["id"].tolist(), ["c", "a", "b"])
        self.assertEqual(
            captured["enriched"]["benchmark"].tolist(), [0.33, 0.11, 0.22]
        )
        self.assertEqual(predictions["benchmark"].tolist(), [-3.0, -1.0, -2.0])
        self.assertEqual(result.index.tolist(), ["0001", "0002", "0003"])

        mismatched = benchmark.copy()
        mismatched.loc[mismatched["id"] == "c", "era"] = "9999"
        with self.assertRaisesRegex(ValueError, "exactly match"):
            show_experiment._per_era_bmc(
                predictions,
                "prediction",
                "target",
                "era",
                "id",
                mismatched,
                "benchmark",
            )


if __name__ == "__main__":
    unittest.main()
