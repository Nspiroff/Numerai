from __future__ import annotations

import json
from pathlib import Path
import runpy
import unittest


AGENTS_DIR = Path(__file__).resolve().parents[1]
ENDER22 = AGENTS_DIR / "experiments" / "ender22_temporal_retention_v53"
ENDER23 = AGENTS_DIR / "experiments" / "ender23_temporal_retention_v53"

MECHANICAL_FILES = (
    "configs/base_r1.py",
    "configs/r1_control_block_dro.py",
    "configs/r1_recent_half_life52.py",
    "configs/r1_recent_window78.py",
    "configs/r2_recent_half_life52_model_seed2027.py",
    "configs/r2_recent_half_life52_sample_seed2027.py",
    "configs/r2_recent_window78_model_seed2027.py",
    "configs/r2_recent_window78_sample_seed2027.py",
    "evaluation_common.py",
    "evaluate_round1.py",
    "evaluate_round1_impl.py",
    "evaluate_round2.py",
    "evaluate_round2_impl.py",
    "protocol/discovery_data_authority.json",
    "run_round1.py",
    "run_round2.py",
    "training_bootstrap.py",
)


def _load_config(name: str) -> dict:
    return runpy.run_path(str(ENDER23 / "configs" / f"{name}.py"))["CONFIG"]


class TestEnder23ProtocolScaffold(unittest.TestCase):
    def test_governed_sources_are_mechanical_ender23_copies(self) -> None:
        for relative in MECHANICAL_FILES:
            with self.subTest(relative=relative):
                frozen = (ENDER22 / relative).read_text(encoding="utf-8")
                expected = (
                    frozen.replace("Ender22", "Ender23")
                    .replace("ender22", "ender23")
                    .replace("ENDER22", "ENDER23")
                )
                actual = (ENDER23 / relative).read_text(encoding="utf-8")
                self.assertEqual(actual.rstrip("\n"), expected.rstrip("\n"))

    def test_round1_is_the_exact_three_way_clean_rerun(self) -> None:
        base_module = runpy.run_path(str(ENDER23 / "configs" / "base_r1.py"))
        variant = base_module["variant"]

        expected = {
            "r1_control_block_dro": variant("r1_control_block_dro"),
            "r1_recent_half_life52": variant(
                "r1_recent_half_life52",
                recency_half_life_eras=52.0,
            ),
            "r1_recent_window78": variant(
                "r1_recent_window78",
                max_train_eras=78,
            ),
        }
        self.assertEqual(
            {name: _load_config(name) for name in expected},
            expected,
        )
        for config in expected.values():
            self.assertEqual(
                config["output"]["output_dir"],
                "experiments/ender23_temporal_retention_v53",
            )
            self.assertEqual(
                config["data"]["full_data_path"],
                "v5.3/ender21_discovery_full_through_0861.parquet",
            )
            self.assertEqual(config["model"]["params"]["seed"], 1337)
            self.assertEqual(config["training"]["sample_seed"], 1337)

    def test_round2_pairs_and_seeds_are_unchanged(self) -> None:
        variant = runpy.run_path(
            str(ENDER23 / "configs" / "base_r1.py")
        )["variant"]
        expected = {
            "r2_recent_half_life52_model_seed2027": variant(
                "r2_recent_half_life52_model_seed2027",
                recency_half_life_eras=52.0,
                model_seed=2027,
            ),
            "r2_recent_half_life52_sample_seed2027": variant(
                "r2_recent_half_life52_sample_seed2027",
                recency_half_life_eras=52.0,
                sample_seed=2027,
            ),
            "r2_recent_window78_model_seed2027": variant(
                "r2_recent_window78_model_seed2027",
                max_train_eras=78,
                model_seed=2027,
            ),
            "r2_recent_window78_sample_seed2027": variant(
                "r2_recent_window78_sample_seed2027",
                max_train_eras=78,
                sample_seed=2027,
            ),
        }
        self.assertEqual(
            {name: _load_config(name) for name in expected},
            expected,
        )

    def test_bootstrap_and_evaluator_manifest_sets_agree(self) -> None:
        bootstrap = runpy.run_path(str(ENDER23 / "training_bootstrap.py"))
        common = runpy.run_path(str(ENDER23 / "evaluation_common.py"))

        for round_number, expected_count in ((1, 33), (2, 35)):
            with self.subTest(round_number=round_number):
                bootstrap_set = bootstrap["_expected_manifest_files"](
                    round_number
                )
                evaluator_set = common["_manifest_file_set"](round_number)
                self.assertEqual(bootstrap_set, evaluator_set)
                self.assertEqual(len(bootstrap_set), expected_count)

    def test_discovery_and_confirmation_boundaries_are_unchanged(self) -> None:
        authority = json.loads(
            (ENDER23 / "protocol" / "discovery_data_authority.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(authority["authority"], "ender23-discovery-only")
        self.assertEqual(authority["full"]["last_era"], "0861")
        self.assertEqual(authority["benchmark"]["last_era"], "0861")
        self.assertEqual(
            authority["forbidden_historical_confirmation"]["first_era"],
            "0865",
        )
        self.assertEqual(
            authority["forbidden_historical_confirmation"]["last_era"],
            "1021",
        )
        self.assertEqual(
            authority["prospective_confirmation"],
            {
                "first_era": "1231",
                "last_era": "1282",
                "era_count": 52,
                "rule": "future resolved eras only; no local historical substitute",
            },
        )

    def test_docs_require_new_manifests_and_all_three_fresh_runs(self) -> None:
        experiment = (ENDER23 / "experiment.md").read_text(encoding="utf-8")
        gate = (ENDER23 / "gate.md").read_text(encoding="utf-8")

        self.assertIn("new matched experiment cohort", experiment)
        self.assertIn("memory-path repair", experiment)
        self.assertIn("must rerun A, B, and C from scratch, once each", experiment)
        self.assertIn("mixing fresh B with Ender22 A/C artifacts", experiment)
        self.assertIn("All three runs are one indivisible matched cohort", experiment)
        self.assertIn("copying or reusing an\n   Ender22 manifest is forbidden", gate)
        self.assertIn("all nine new Ender23 Round-1", gate)
        self.assertIn(
            "Run A, B, and C from scratch exactly once under the same Ender23 manifest",
            gate,
        )


if __name__ == "__main__":
    unittest.main()
