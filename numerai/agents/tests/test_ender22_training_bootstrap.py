from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


BOOTSTRAP_PATH = (
    Path(__file__).parents[1]
    / "experiments/ender22_temporal_retention_v53/training_bootstrap.py"
)
EXPERIMENT_DIR = BOOTSTRAP_PATH.parent
SPEC = importlib.util.spec_from_file_location("ender22_training_bootstrap_test", BOOTSTRAP_PATH)
bootstrap = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bootstrap)


class TestEnder22TrainingBootstrap(unittest.TestCase):
    def test_windows_launchers_wait_and_propagate_bootstrap_failure(self) -> None:
        for filename in (
            "run_round1.py",
            "run_round2.py",
            "evaluate_round1.py",
            "evaluate_round2.py",
        ):
            source = (EXPERIMENT_DIR / filename).read_text(encoding="utf-8")
            self.assertIn("subprocess.run", source)
            self.assertNotIn("os.execv", source)

        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-X",
                    f"pycache_prefix={tmp}",
                    str(EXPERIMENT_DIR / "run_round1.py"),
                    "__invalid_config__",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("invalid choice", completed.stderr)

    def test_rejects_poisoned_preimport_before_manifest(self) -> None:
        modules = dict(bootstrap.sys.modules)
        modules["agents.code.modeling.utils.pipeline"] = object()
        with mock.patch.object(bootstrap.sys, "modules", modules):
            with self.assertRaisesRegex(ValueError, "pre-imported"):
                bootstrap._require_launch()

    def test_fresh_child_uses_source_not_adjacent_poisoned_pyc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = root / "victim.py"
            module.write_text("VALUE = 'poison'\n", encoding="utf-8")
            cache = root / "__pycache__/victim.pyc"
            cache.parent.mkdir()
            py_compile.compile(
                str(module),
                cfile=str(cache),
                invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
            )
            module.write_text("VALUE = 'source'\n", encoding="utf-8")
            isolated = root / "isolated-cache"
            isolated.mkdir()
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-X",
                    f"pycache_prefix={isolated}",
                    "-c",
                    f"import sys; sys.path.insert(0, {str(root)!r}); import victim; print(victim.VALUE)",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(completed.stdout.strip(), "source")

    def test_acquire_rejects_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = "numerai/agents/experiments/ender22_temporal_retention_v53"
            experiment = root / prefix
            (experiment / "configs").mkdir(parents=True)
            source = root / "src.py"
            source.write_bytes(b"drifted")
            config = experiment / "configs/cfg.py"
            config.write_bytes(b"CONFIG = {}\n")
            artifact = root / "data.bin"
            artifact.write_bytes(b"data")
            commit = "a" * 40
            manifest = {
                "schema_version": 1,
                "frozen_at": "synthetic",
                "git_head": commit,
                "hash_algorithm": "sha256",
                "files": {
                    "src.py": "0" * 64,
                    f"{prefix}/configs/cfg.py": __import__("hashlib").sha256(config.read_bytes()).hexdigest(),
                },
                "external_artifacts": {
                    "data.bin": {
                        "sha256": __import__("hashlib").sha256(b"data").hexdigest(),
                        "size_bytes": 4,
                        "last_era": "0861",
                    }
                },
                "runtime": {"python": sys.version.split()[0], "packages": {}},
            }
            (experiment / "source_manifest_round1.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            git_calls = []

            def git(*args, **_kwargs):
                git_calls.append(args)
                if args[:2] == ("rev-parse", "--verify"):
                    return subprocess.CompletedProcess(args, 0, commit, "")
                return subprocess.CompletedProcess(args, 0, "", "")

            with mock.patch.object(bootstrap, "REPO_DIR", root), mock.patch.object(
                bootstrap, "PREFIX", prefix
            ), mock.patch.object(
                bootstrap, "ROUND_NAMES", {1: frozenset({"cfg"}), 2: frozenset()}
            ), mock.patch.object(
                bootstrap, "BOOTSTRAP_SOURCES", frozenset({"src.py"})
            ), mock.patch.object(
                bootstrap, "STATIC_MANIFEST_FILES", frozenset()
            ), mock.patch.object(
                bootstrap, "EXTERNALS", frozenset({"data.bin"})
            ), mock.patch.object(
                bootstrap, "EXPECTED_RUNTIME", {"python": sys.version.split()[0], "packages": {}}
            ), mock.patch.object(bootstrap, "_git", side_effect=git):
                with self.assertRaisesRegex(ValueError, "source drifted"):
                    bootstrap._acquire(1)
            self.assertIn(
                ("cat-file", "-e", f"HEAD:{prefix}/source_manifest_round1.json"),
                git_calls,
            )

    def test_reservation_precedes_acquire_and_in_process_training(self) -> None:
        events = []

        class Lease:
            def close(self):
                events.append("close")

        class Reservations:
            def __init__(self, name):
                self.name = name

            def __enter__(self):
                events.append("reserve")
                return self

            def __exit__(self, *_args):
                events.append("reservation_close")

        def acquire(round_number):
            events.append(f"acquire:{round_number}")
            return {"held": True}, b"held manifest", (Lease(),)

        pipeline = SimpleNamespace(
            run_ender22_training_from_bootstrap=lambda *_args, **_kwargs: events.append(
                "train"
            )
        )

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            bootstrap, "_require_launch", return_value=Path(tmp)
        ), mock.patch.object(
            bootstrap, "_TrainingOutputReservations", Reservations
        ), mock.patch.object(
            bootstrap, "_acquire", side_effect=acquire
        ), mock.patch.object(
            bootstrap.importlib, "import_module", return_value=pipeline
        ), mock.patch.object(
            bootstrap.sys, "argv", ["training_bootstrap.py", "1", "r1_control_block_dro"]
        ):
            self.assertEqual(bootstrap.main(), 0)
        self.assertEqual(
            events,
            ["reserve", "acquire:1", "train", "close", "reservation_close"],
        )

    def test_existing_completion_stops_before_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / bootstrap.PREFIX
            (experiment / "predictions").mkdir(parents=True)
            (experiment / "results").mkdir()
            (experiment / "receipts").mkdir()
            name = "r1_control_block_dro"
            completion = experiment / f"receipts/{name}.completion.json"
            completion.write_bytes(b"existing")
            with mock.patch.object(bootstrap, "REPO_DIR", root), mock.patch.object(
                bootstrap, "_require_launch", return_value=root / "cache"
            ), mock.patch.object(bootstrap, "_acquire") as acquire:
                with self.assertRaisesRegex(ValueError, "completion"):
                    bootstrap._training_main(["1", name])
            acquire.assert_not_called()
            self.assertEqual(completion.read_bytes(), b"existing")

    def test_evaluation_reserves_before_manifest_and_holds_lease_through_commit(self) -> None:
        events = []

        class Decision:
            def __init__(self, _path):
                pass

            def __enter__(self):
                events.append("reserve")
                return self

            def __exit__(self, *_args):
                events.append("decision_close")

        class Lease:
            def close(self):
                events.append("lease_close")

        def acquire(round_number):
            events.append(f"acquire:{round_number}")
            return {}, b"manifest", (Lease(),)

        common = object()

        def load_module(name, _path):
            events.append(f"load:{name}")
            if name == "evaluation_common":
                return common
            return SimpleNamespace(
                run_bootstrapped=lambda *_args: events.append("evaluate_and_fsync")
            )

        experiment = bootstrap.REPO_DIR / bootstrap.PREFIX
        arguments = [
            "1",
            "--experiment", str(experiment),
            "--numerai-dir", str(bootstrap.REPO_DIR / "numerai"),
            "--output", str(experiment / "receipts/round1_discovery.json"),
        ]
        with mock.patch.object(
            bootstrap, "_require_launch", return_value=Path("C:/isolated")
        ), mock.patch.object(
            bootstrap, "_DecisionReservation", Decision
        ), mock.patch.object(
            bootstrap, "_acquire", side_effect=acquire
        ), mock.patch.object(
            bootstrap, "_load_governed_module", side_effect=load_module
        ):
            self.assertEqual(bootstrap._evaluation_main(arguments), 0)
        self.assertEqual(
            events,
            [
                "reserve",
                "acquire:1",
                "load:evaluation_common",
                "load:ender22_evaluate_round1_impl",
                "evaluate_and_fsync",
                "lease_close",
                "decision_close",
            ],
        )

    def test_round2_unselected_family_rejected_in_parent_bootstrap(self) -> None:
        decision_path = (
            bootstrap.REPO_DIR
            / bootstrap.PREFIX
            / "receipts/round1_discovery.json"
        )

        class Lease:
            path = decision_path

            @staticmethod
            def read_bytes():
                return json.dumps(
                    {
                        "schema_version": 1,
                        "stage": "ender22-round1-discovery",
                        "state": "SCOUT_WINNER",
                        "selected": "r1_recent_window78",
                    }
                ).encode("utf-8")

        with self.assertRaisesRegex(ValueError, "does not authorize"):
            bootstrap._require_round2_selection(
                "r2_recent_half_life52_model_seed2027", (Lease(),)
            )


if __name__ == "__main__":
    unittest.main()
