from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


EXPERIMENT = (
    Path(__file__).resolve().parents[1]
    / "experiments/ender25_ender24_evaluation_recovery_v53"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = _load_module("ender25_evaluation_common_test", EXPERIMENT / "evaluation_common.py")

_previous_common = sys.modules.get("evaluation_common")
_previous_recovery_common = sys.modules.get("ender25_recovery_common")
sys.modules["evaluation_common"] = COMMON
sys.modules["ender25_recovery_common"] = COMMON
try:
    BOOTSTRAP = _load_module(
        "ender25_evaluation_bootstrap_test", EXPERIMENT / "evaluation_bootstrap.py"
    )
    IMPLEMENTATION = _load_module(
        "ender25_evaluate_recovery_impl_test", EXPERIMENT / "evaluate_recovery_impl.py"
    )
finally:
    if _previous_common is None:
        sys.modules.pop("evaluation_common", None)
    else:
        sys.modules["evaluation_common"] = _previous_common
    if _previous_recovery_common is None:
        sys.modules.pop("ender25_recovery_common", None)
    else:
        sys.modules["ender25_recovery_common"] = _previous_recovery_common


ROUND1_NAMES = (
    "r1_control_seed1337",
    "r1_ema995_seed1337",
    "r1_control_seed2027",
    "r1_ema995_seed2027",
)
EXPECTED_COMPLETION_SHA256 = {
    "r1_control_seed1337": (
        "5a38b9f7211155b7ce9ea71db8c6815a72940f8c999c53e7b2f0ef6d4bd65b4e"
    ),
    "r1_ema995_seed1337": (
        "0b6d500c571abe376ae388ace4abc56430ac8df9c5bce71be7a41657a149431f"
    ),
    "r1_control_seed2027": (
        "f33f40f537413fdd8fc80cb7d656192fbf02d906b6885723894e9fc1776653e4"
    ),
    "r1_ema995_seed2027": (
        "c63793b84ef0e3fa03a9dc1a8c1aa167a1fb6a29863313e82259301f12f512ed"
    ),
}


def _receipt(raw: bytes) -> dict[str, object]:
    return {
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


class TestEnder25CanonicalTextAuthority(unittest.TestCase):
    def test_lf_and_crlf_share_canonical_identity_but_not_raw_identity(self) -> None:
        lf = b'[\n  "0161",\n  "0165"\n]\n'
        crlf = lf.replace(b"\n", b"\r\n")

        self.assertNotEqual(_receipt(lf), _receipt(crlf))
        COMMON.verify_bytes_receipt(lf, _receipt(lf), "LF raw authority")
        COMMON.verify_bytes_receipt(crlf, _receipt(crlf), "CRLF raw authority")

        canonical_lf = COMMON.canonical_json_bytes(lf, "LF authority")
        canonical_crlf = COMMON.canonical_json_bytes(crlf, "CRLF authority")
        self.assertEqual(canonical_lf, lf)
        self.assertEqual(canonical_crlf, lf)
        COMMON.verify_bytes_receipt(
            canonical_lf, _receipt(lf), "canonical LF authority"
        )
        COMMON.verify_bytes_receipt(
            canonical_crlf, _receipt(lf), "canonical CRLF authority"
        )
        self.assertEqual(
            COMMON.strict_json(canonical_lf, "LF authority"),
            COMMON.strict_json(canonical_crlf, "CRLF authority"),
        )

    def test_wrong_raw_receipt_rejects_before_canonical_equivalence(self) -> None:
        lf = b'[\n  "0161"\n]\n'
        crlf = lf.replace(b"\n", b"\r\n")
        with self.assertRaises(ValueError):
            COMMON.verify_bytes_receipt(crlf, _receipt(lf), "physical authority")

    def test_bom_lone_cr_mixed_endings_nul_and_invalid_utf8_reject(self) -> None:
        cases = {
            "BOM": b'\xef\xbb\xbf["0161"]\n',
            "lone CR": b'["0161"]\r',
            "mixed": b'[\r\n  "0161",\n  "0165"\r\n]\r\n',
            "NUL": b'["0161"]\x00\n',
            "UTF-8": b'["\xff"]\n',
        }
        for label, raw in cases.items():
            with self.subTest(label=label), self.assertRaises((TypeError, ValueError)):
                COMMON.canonical_json_bytes(raw, label)

    def test_duplicate_json_keys_and_nonfinite_numbers_reject(self) -> None:
        cases = (
            b'{"authority":1,"authority":2}\n',
            b'{"value":NaN}\n',
            b'{"value":Infinity}\n',
        )
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises((TypeError, ValueError)):
                COMMON.strict_json(raw, "strict JSON")

    def test_same_length_mutation_reorder_and_duplicate_fail_frozen_hash(self) -> None:
        frozen = b'[\n  "0161",\n  "0165"\n]\n'
        mutations = {
            "semantic mutation": b'[\n  "0161",\n  "0169"\n]\n',
            "reorder": b'[\n  "0165",\n  "0161"\n]\n',
            "duplicate": b'[\n  "0161",\n  "0161"\n]\n',
        }
        frozen_receipt = _receipt(frozen)
        for label, raw in mutations.items():
            with self.subTest(label=label):
                self.assertEqual(len(raw), len(frozen))
                parsed = COMMON.strict_json(
                    COMMON.canonical_json_bytes(raw, label), label
                )
                self.assertIsInstance(parsed, list)
                with self.assertRaises(ValueError):
                    COMMON.verify_bytes_receipt(raw, frozen_receipt, label)

    def test_alternate_json_formatting_is_not_a_canonicalization_escape(self) -> None:
        frozen = b'[\n  "0161",\n  "0165"\n]\n'
        compact = b'["0161","0165"]\n'
        self.assertEqual(
            COMMON.strict_json(frozen, "pretty"),
            COMMON.strict_json(compact, "compact"),
        )
        self.assertNotEqual(
            COMMON.canonical_json_bytes(frozen, "pretty"),
            COMMON.canonical_json_bytes(compact, "compact"),
        )
        with self.assertRaises(ValueError):
            COMMON.verify_bytes_receipt(compact, _receipt(frozen), "compact")


class TestEnder25FrozenEnder24Authority(unittest.TestCase):
    def test_protocol_authority_matches_runtime_literals(self) -> None:
        authority = json.loads(
            (EXPERIMENT / "protocol/ender24_input_authority.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(authority["family"], COMMON.FAMILY)
        self.assertEqual(authority["stage"], COMMON.STAGE)
        self.assertEqual(
            tuple(authority["cohort_law"]["exact_members_in_preflight_order"]),
            ROUND1_NAMES,
        )
        self.assertEqual(
            authority["ender24_source_manifest"]["sha256"],
            BOOTSTRAP.ENDER24_MANIFEST_AUTHORITY["sha256"],
        )
        self.assertEqual(
            tuple(authority["future_execution"]["scientific_terminal_states"]),
            (COMMON.POSITIVE_STATE, COMMON.NEGATIVE_STATE),
        )
        self.assertTrue(
            all(value is False for value in authority["current_authority"].values())
        )

    def test_scaffold_or_seal_has_no_decision_and_valid_manifest(self) -> None:
        manifest_path = EXPERIMENT / "source_manifest_evaluation_recovery.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIs(BOOTSTRAP.validate_recovery_manifest(manifest), manifest)
        self.assertFalse(
            (EXPERIMENT / "receipts/ender24_round1_recovery_decision.json").exists()
        )

    def test_manifest_and_four_completion_hashes_are_exact_literals(self) -> None:
        self.assertEqual(COMMON.ROUND1_NAMES, ROUND1_NAMES)
        self.assertEqual(
            BOOTSTRAP.ENDER24_MANIFEST_AUTHORITY["sha256"],
            "bd55280e4a99a1b45be87cc5af73aea2615da14a4de0e0662d1ac6c008ab1b35",
        )
        self.assertEqual(
            tuple(BOOTSTRAP.ENDER24_COMPLETION_AUTHORITY),
            ROUND1_NAMES,
        )
        self.assertEqual(
            {
                name: BOOTSTRAP.ENDER24_COMPLETION_AUTHORITY[name]["sha256"]
                for name in ROUND1_NAMES
            },
            EXPECTED_COMPLETION_SHA256,
        )

    def test_terminal_evidence_commit_path_set_is_exactly_twelve(self) -> None:
        source = (EXPERIMENT / "evaluation_bootstrap.py").read_text(encoding="utf-8")
        for relative in (
            ".gitattributes",
            "numerai/agents/experiments/README.md",
            "round1_execution_postmortem.json",
            "round1_execution_postmortem.md",
        ):
            with self.subTest(relative=relative):
                self.assertIn(relative, source)

    def test_future_manifest_binds_every_executed_ender24_config(self) -> None:
        expected = BOOTSTRAP.EXPECTED_MANIFEST_FILES
        prefix = BOOTSTRAP.ENDER24_PREFIX
        self.assertIn(f"{prefix}/configs/base_r1.py", expected)
        for name in ROUND1_NAMES:
            with self.subTest(name=name):
                self.assertIn(f"{prefix}/configs/{name}.py", expected)

    def test_future_manifest_source_set_is_exact_nineteen_paths(self) -> None:
        prefix = (
            "numerai/agents/experiments/"
            "ender25_ender24_evaluation_recovery_v53"
        )
        old = BOOTSTRAP.ENDER24_PREFIX
        expected = {
            f"{prefix}/evaluate_recovery.py",
            f"{prefix}/evaluation_bootstrap.py",
            f"{prefix}/evaluation_common.py",
            f"{prefix}/evaluate_recovery_impl.py",
            f"{prefix}/experiment.md",
            f"{prefix}/gate.md",
            f"{prefix}/protocol/ender24_input_authority.json",
            f"{prefix}/receipts/.gitkeep",
            "numerai/agents/tests/test_ender25_ema_evaluation_recovery.py",
            f"{old}/evaluation_common.py",
            f"{old}/configs/base_r1.py",
            *(f"{old}/configs/{name}.py" for name in ROUND1_NAMES),
            "numerai/agents/code/metrics/numerai_metrics.py",
            "numerai/agents/code/modeling/utils/constants.py",
            *BOOTSTRAP.PORTABLE_TEXT_PATHS,
        }
        self.assertEqual(BOOTSTRAP.EXPECTED_MANIFEST_FILES, frozenset(expected))
        self.assertEqual(len(expected), 19)

    def test_scientific_scoring_runtime_closure_is_exactly_pinned(self) -> None:
        self.assertEqual(
            BOOTSTRAP.EXPECTED_RUNTIME["packages"],
            {
                "numpy": "2.5.1",
                "pandas": "3.0.5",
                "pyarrow": "25.0.0",
                "numerai-tools": "0.6.0",
                "numerapi": "2.23.3",
                "scipy": "1.18.0",
                "scikit-learn": "1.9.0",
            },
        )

    def test_production_raw_custody_defers_only_portable_text_to_new_manifest(self) -> None:
        source = (EXPERIMENT / "evaluation_bootstrap.py").read_text(encoding="utf-8")
        self.assertEqual(
            BOOTSTRAP.PORTABLE_TEXT_PATHS,
            {
                "numerai/agents/experiments/ender21_residual_stability_v53/"
                "protocol/discovery_eras_through_0861.json",
                "numerai/agents/experiments/ender21_residual_stability_v53/"
                "protocol/feature_columns_all_v53.json",
            },
        )
        self.assertIn("relative not in PORTABLE_TEXT_PATHS", source)

    def test_text_authority_literals_bind_exact_canonical_receipts(self) -> None:
        expected = {
            "era_allowlist": {
                "canonical_lf": (
                    1_763,
                    "be0c212a8e910f56dbdae4e1e134fa36ce7e5e1a95e43faa1ccc9e6330f544ca",
                ),
            },
            "feature_columns": {
                "canonical_lf": (
                    148_179,
                    "663184191e17d2fa4fac6dae017890f0e762368e638d46cfaa489297b9b2049b",
                ),
            },
        }
        for label, identities in expected.items():
            for identity, (size_bytes, sha256) in identities.items():
                with self.subTest(label=label, identity=identity):
                    actual = COMMON.TEXT_AUTHORITY[label][identity]
                    self.assertEqual(actual["size_bytes"], size_bytes)
                    self.assertEqual(actual["sha256"], sha256)

    def test_frozen_authority_is_rejected_on_literal_hash_drift(self) -> None:
        manifest = BOOTSTRAP.expected_recovery_manifest_template()
        manifest.update(
            {
                "frozen_at": "2026-08-13",
                "git_head": "a" * 40,
                "files": {
                    path: "b" * 64 for path in BOOTSTRAP.EXPECTED_MANIFEST_FILES
                },
            }
        )
        self.assertIs(BOOTSTRAP.validate_recovery_manifest(manifest), manifest)

        mutations = []
        wrong_manifest = deepcopy(manifest)
        wrong_manifest["ender24_authority"]["source_manifest"]["sha256"] = "0" * 64
        mutations.append(wrong_manifest)
        wrong_completion = deepcopy(manifest)
        wrong_completion["ender24_authority"]["completions"][ROUND1_NAMES[0]][
            "sha256"
        ] = "0" * 64
        mutations.append(wrong_completion)
        wrong_canonical = deepcopy(manifest)
        wrong_canonical["ender24_authority"]["source_manifest"]["size_bytes"] += 1
        mutations.append(wrong_canonical)
        for changed in mutations:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                BOOTSTRAP.validate_recovery_manifest(changed)


class TestEnder25PreflightAndReachability(unittest.TestCase):
    def test_all_four_completions_preflight_in_frozen_order(self) -> None:
        calls: list[str] = []

        def validate(name, *_args, **_kwargs):
            calls.append(name)
            return {"component": name}

        with mock.patch.object(
            COMMON, "validate_completion_envelope", side_effect=validate
        ):
            result = COMMON.preflight_all_completions(
                {name: {} for name in ROUND1_NAMES},
                {},
                {name: {} for name in ROUND1_NAMES},
            )
        self.assertEqual(calls, list(ROUND1_NAMES))
        self.assertEqual(tuple(result), ROUND1_NAMES)

    def test_failed_fourth_preflight_blocks_authority_truth_results_and_predictions(
        self,
    ) -> None:
        events: list[str] = []

        class Custody:
            manifest = {}

            def preflight_completions(self, *_args):
                for name in ROUND1_NAMES:
                    events.append(f"completion:{name}")
                raise ValueError("fourth completion invalid")

            def load_authority(self):
                events.append("authority")
                raise AssertionError("authority reached")

            def load_truth(self, *_args):
                events.append("truth")
                raise AssertionError("truth reached")

            def score_candidate(self, *_args):
                events.append("result-or-prediction")
                raise AssertionError("result or prediction reached")

        frozen_common = mock.Mock()
        with self.assertRaisesRegex(ValueError, "fourth completion invalid"):
            IMPLEMENTATION.evaluate(Path("experiment"), Path("numerai"), Custody(), frozen_common)
        self.assertEqual(events, [f"completion:{name}" for name in ROUND1_NAMES])
        frozen_common.load_truth.assert_not_called()
        frozen_common.score_candidate.assert_not_called()

    def test_bootstrap_crosses_preflight_and_authority_before_scoring_imports(self) -> None:
        source = (EXPERIMENT / "evaluation_bootstrap.py").read_text(encoding="utf-8")
        runtime = source.index("def main(")
        preflight = source.index("custody.preflight_completions()", runtime)
        authority = source.index("custody.load_authority()", preflight)
        numpy_import = source.index('importlib.import_module("numpy")', authority)
        self.assertLess(preflight, authority)
        self.assertLess(authority, numpy_import)

    def test_outer_launcher_requires_isolated_safe_python(self) -> None:
        source = (EXPERIMENT / "evaluate_recovery.py").read_text(encoding="utf-8")
        for requirement in (
            "sys.flags.isolated != 1",
            "sys.flags.safe_path != 1",
            "sys.flags.dont_write_bytecode != 1",
            "sys.dont_write_bytecode",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, source)

    def test_python_entrypoints_cannot_launch_old_evaluator_training_or_accounts(self) -> None:
        forbidden_import_roots = {
            "http",
            "httpx",
            "numerapi",
            "requests",
            "socket",
            "urllib",
        }
        forbidden_markers = (
            "evaluate_round1.py",
            "run_round1.py",
            "run_round2.py",
            "training_bootstrap",
            "agents.code.modeling",
            "upload_model",
            "create_model",
            "numerai_mcp_auth",
        )
        for path in sorted(EXPERIMENT.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".", 1)[0])
            with self.subTest(path=path.name):
                self.assertFalse(imported & forbidden_import_roots)
                lowered = source.lower()
                for marker in forbidden_markers:
                    self.assertNotIn(marker, lowered)


class TestEnder25DecisionReservation(unittest.TestCase):
    def test_failure_rolls_back_only_the_new_empty_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            new_decision = root / COMMON.DECISION_RELATIVE
            new_decision.parent.mkdir(parents=True)
            old_decision = root / "round1_ema_stability.json"
            old_bytes = b'{"state":"ENDER24_SENTINEL"}\n'
            old_decision.write_bytes(old_bytes)

            with self.assertRaisesRegex(ValueError, "synthetic recovery failure"):
                with BOOTSTRAP.DecisionReservation(new_decision):
                    self.assertTrue(new_decision.exists())
                    raise ValueError("synthetic recovery failure")

            self.assertFalse(new_decision.exists())
            self.assertEqual(old_decision.read_bytes(), old_bytes)

    def test_preexisting_new_decision_is_preserved_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            decision = Path(temporary) / "ender24_round1_recovery_decision.json"
            existing = b'{"prior":"evidence"}\n'
            decision.write_bytes(existing)
            with self.assertRaises(ValueError):
                with BOOTSTRAP.DecisionReservation(decision):
                    self.fail("preexisting decision was entered")
            self.assertEqual(decision.read_bytes(), existing)

    def test_failure_cleanup_preserves_same_inode_nonempty_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            decision = Path(temporary) / "ender24_round1_recovery_decision.json"
            changed = b"unexpected evidence"
            with self.assertRaisesRegex(ValueError, "changed"):
                with BOOTSTRAP.DecisionReservation(decision) as reservation:
                    reservation.stream.write(changed)
                    raise RuntimeError("synthetic producer failure")
            self.assertEqual(decision.read_bytes(), changed)

    def test_commit_targets_exact_new_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision = root / COMMON.DECISION_RELATIVE
            decision.parent.mkdir(parents=True)
            payload = {"schema_version": 1, "state": "synthetic"}
            with BOOTSTRAP.DecisionReservation(decision) as reservation:
                reservation.commit_json(payload)
            self.assertEqual(json.loads(decision.read_text(encoding="utf-8")), payload)
            self.assertEqual(
                decision.relative_to(root).as_posix(),
                "numerai/agents/experiments/ender25_ender24_evaluation_recovery_v53/receipts/ender24_round1_recovery_decision.json",
            )


class TestEnder25DecisionAuthority(unittest.TestCase):
    def test_exact_stage_states_and_source_gate_only_positive_authority(self) -> None:
        positive = IMPLEMENTATION.authorization_fields(True)
        negative = IMPLEMENTATION.authorization_fields(False)
        self.assertEqual(COMMON.STAGE, "ender25-ender24-round1-evaluation-recovery")
        self.assertEqual(
            COMMON.POSITIVE_STATE, "ENDER25_ROUND2_SOURCE_GATE_AUTHORIZED"
        )
        self.assertEqual(
            COMMON.NEGATIVE_STATE, "ENDER25_NEGATIVE_NO_EMA_STABILITY_GAIN"
        )
        self.assertEqual(
            positive,
            {
                "state": COMMON.POSITIVE_STATE,
                "round2_source_gate_authorized": True,
                "round2_authorized": False,
                "training_authorized": False,
                "scoring_authorized": False,
                "deployment_authorized": False,
                "account_actions_authorized": False,
            },
        )
        self.assertEqual(
            negative,
            {
                "state": COMMON.NEGATIVE_STATE,
                "round2_source_gate_authorized": False,
                "round2_authorized": False,
                "training_authorized": False,
                "scoring_authorized": False,
                "deployment_authorized": False,
                "account_actions_authorized": False,
            },
        )

    def test_decision_namespace_is_new_and_never_old_ender24_output(self) -> None:
        self.assertEqual(
            COMMON.DECISION_RELATIVE,
            "numerai/agents/experiments/ender25_ender24_evaluation_recovery_v53/"
            "receipts/ender24_round1_recovery_decision.json",
        )
        self.assertNotEqual(
            Path(COMMON.DECISION_RELATIVE).name,
            "round1_ema_stability.json",
        )


if __name__ == "__main__":
    unittest.main()
