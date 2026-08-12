from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock


EXPERIMENT = (
    Path(__file__).resolve().parents[1]
    / "experiments/ender22_temporal_retention_v53"
)


def _load_common():
    path = EXPERIMENT / "evaluation_common.py"
    spec = importlib.util.spec_from_file_location("ender22_evaluation_common_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Ender22 evaluation common module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = _load_common()


class TestEnder22EvaluatorCustody(unittest.TestCase):
    def test_evaluator_module_loads_use_only_stdlib(self) -> None:
        forbidden = {"agents", "numpy", "pandas", "pyarrow"}
        for filename in ("evaluation_common.py", "evaluate_round1.py", "evaluate_round2.py"):
            tree = ast.parse((EXPERIMENT / filename).read_text(encoding="utf-8"))
            imported = set()
            for node in tree.body:
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertTrue(forbidden.isdisjoint(imported), (filename, imported))

    def test_decision_reservation_rolls_back_validation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "decision.json"
            with self.assertRaisesRegex(ValueError, "synthetic validation"):
                with COMMON.DecisionReservation(output):
                    raise ValueError("synthetic validation")
            self.assertFalse(output.exists())

    def test_decision_reservation_commits_exact_durable_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "decision.json"
            payload = {"state": "SYNTHETIC", "schema_version": 1}
            with COMMON.DecisionReservation(output) as reservation:
                written = reservation.commit_json(payload)
            self.assertEqual(output.read_bytes(), written)
            self.assertEqual(json.loads(written), payload)
            with self.assertRaisesRegex(ValueError, "Cannot reserve"):
                with COMMON.DecisionReservation(output):
                    pass

    def test_pycache_contract_rejects_nonempty_external_prefix(self) -> None:
        repo = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            (prefix / "stale.pyc").write_bytes(b"stale")
            fake_flags = SimpleNamespace(dont_write_bytecode=1)
            with mock.patch.object(sys, "flags", fake_flags), mock.patch.object(
                sys, "dont_write_bytecode", True
            ), mock.patch.object(sys, "pycache_prefix", str(prefix)), mock.patch.object(
                sys, "_xoptions", {"pycache_prefix": str(prefix)}
            ):
                with self.assertRaisesRegex(ValueError, "freshly empty"):
                    COMMON.require_frozen_python_runtime(repo)


if __name__ == "__main__":
    unittest.main()
