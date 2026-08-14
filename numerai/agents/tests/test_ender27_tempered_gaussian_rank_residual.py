from __future__ import annotations

from copy import deepcopy
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from agents.code.modeling.utils import pipeline


REPO = Path(__file__).resolve().parents[3]
EXPERIMENT = REPO / "numerai/agents/experiments/ender27_tempered_gaussian_rank_residual_v53"
ENDER24 = REPO / "numerai/agents/experiments/ender24_ema_seed_stability_v53"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BOOTSTRAP = _load_module(
    "ender27_training_bootstrap_source_test", EXPERIMENT / "training_bootstrap.py"
)
COMMON = _load_module(
    "ender27_evaluation_common_source_test", EXPERIMENT / "evaluation_common.py"
)


class TestEnder27FrozenSource(unittest.TestCase):
    def test_exact_four_run_cohort_and_matched_pair_delta(self) -> None:
        expected = (
            "r1_control_rawresid_seed1337",
            "r1_tempered_grank_resid_seed1337",
            "r1_control_rawresid_seed2027",
            "r1_tempered_grank_resid_seed2027",
        )
        self.assertEqual(COMMON.ROUND1_NAMES, expected)
        self.assertEqual(tuple(pipeline._ENDER27_ROUND1_NAMES), expected)
        self.assertEqual(BOOTSTRAP.ROUND1_SEQUENCE, expected)
        self.assertEqual(set(BOOTSTRAP.ROUND1_NAMES), set(expected))
        self.assertEqual(
            COMMON.PAIR_NAMES,
            {
                "1337": expected[:2],
                "2027": expected[2:],
            },
        )

        for seed, (control_name, challenger_name) in COMMON.PAIR_NAMES.items():
            control = runpy.run_path(
                str(EXPERIMENT / f"configs/{control_name}.py")
            )["CONFIG"]
            challenger = runpy.run_path(
                str(EXPERIMENT / f"configs/{challenger_name}.py")
            )["CONFIG"]
            self.assertEqual(control["model"]["params"]["seed"], int(seed))
            self.assertEqual(challenger["model"]["params"]["seed"], int(seed))
            self.assertEqual(control["training"]["sample_seed"], 1337)
            self.assertEqual(challenger["training"]["sample_seed"], 1337)
            self.assertNotIn(
                "benchmark_transform", control["model"]["target_transform"]
            )
            self.assertNotIn(
                "benchmark_transform_strength",
                control["model"]["target_transform"],
            )
            self.assertEqual(
                challenger["model"]["target_transform"].pop(
                    "benchmark_transform"
                ),
                "tie_kept_rank_gaussian",
            )
            self.assertEqual(
                challenger["model"]["target_transform"].pop(
                    "benchmark_transform_strength"
                ),
                0.5,
            )
            control = deepcopy(control)
            control["output"]["results_name"] = "<matched>"
            challenger["output"]["results_name"] = "<matched>"
            self.assertEqual(control, challenger)

    def test_control_procedure_is_exact_ender24_window78_non_ema(self) -> None:
        current = runpy.run_path(
            str(EXPERIMENT / "configs/r1_control_rawresid_seed1337.py")
        )["CONFIG"]
        prior = runpy.run_path(
            str(ENDER24 / "configs/r1_control_seed1337.py")
        )["CONFIG"]
        current["output"] = {"normalized": True}
        prior["output"] = {"normalized": True}
        self.assertEqual(current, prior)
        self.assertEqual(current["training"]["cv"]["max_train_eras"], 78)
        self.assertNotIn("ema_decay", current["model"]["params"])

    def test_manifest_inventory_is_exact_and_equal_everywhere(self) -> None:
        bootstrap = set(BOOTSTRAP._expected_manifest_files(1))
        evaluator = set(COMMON._manifest_file_set(1))
        production = set(pipeline._ENDER27_ROUND1_MANIFEST_FILES)
        self.assertEqual(len(bootstrap), 31)
        self.assertEqual(bootstrap, evaluator)
        self.assertEqual(bootstrap, production)
        self.assertEqual(set(BOOTSTRAP.EXTERNALS), set(pipeline._ENDER27_EXTERNAL_ARTIFACTS))
        self.assertFalse(any("round2" in path.lower() for path in bootstrap))
        with self.assertRaisesRegex(ValueError, "Round 1 only"):
            BOOTSTRAP._expected_manifest_files(2)
        with self.assertRaisesRegex(ValueError, "Round-1 manifest"):
            COMMON._manifest_file_set(2)

    def test_manifest_lifecycle_is_source_safe_when_absent_or_present(self) -> None:
        path = EXPERIMENT / "source_manifest_round1.json"
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["files"]), COMMON._manifest_file_set(1))
            self.assertEqual(set(manifest["external_artifacts"]), BOOTSTRAP.EXTERNALS)
            self.assertEqual(manifest["runtime"], BOOTSTRAP.EXPECTED_RUNTIME)
        else:
            self.assertNotIn(
                f"{COMMON.PREFIX}/source_manifest_round1.json",
                COMMON._manifest_file_set(1),
            )
            for module in (BOOTSTRAP, COMMON):
                self.assertIsNotNone(module)

    def test_launchers_pin_isolation_and_exact_bootstrap(self) -> None:
        for filename in ("run_round1.py", "evaluate_round1.py"):
            source = (EXPERIMENT / filename).read_text(encoding="utf-8")
            tree = ast.parse(source)
            top_imports = [
                node
                for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            self.assertEqual(ast.unparse(top_imports[0]), "from __future__ import annotations")
            self.assertEqual(ast.unparse(top_imports[1]), "import sys")
            self.assertEqual(len(top_imports), 2)
            for flag in ('"-I"', '"-B"', '"-P"'):
                self.assertIn(flag, source)
            for assertion in (
                "sys.flags.isolated != 1",
                "sys.flags.dont_write_bytecode != 1",
                "sys.dont_write_bytecode is not True",
                "sys.flags.safe_path != 1",
            ):
                self.assertIn(assertion, source)
            self.assertIn("training_bootstrap.py", source)
            self.assertIn("pycache_prefix", source)
        self.assertEqual(
            BOOTSTRAP.EXPECTED_RUNTIME,
            {
                "python": "3.13.14",
                "packages": {
                    "numpy": "2.5.1",
                    "pandas": "3.0.5",
                    "pyarrow": "25.0.0",
                    "numerai-tools": "0.6.0",
                    "numerapi": "2.23.3",
                    "torch": "2.13.0+cu130",
                    "tabm": "0.0.3",
                    "rtdl-num-embeddings": "0.0.12",
                    "scipy": "1.18.0",
                    "scikit-learn": "1.9.0",
                },
            },
        )

    def test_source_freeze_has_only_round1_and_no_artifacts(self) -> None:
        self.assertFalse(any(EXPERIMENT.glob("configs/r2_*.py")))
        if not (EXPERIMENT / "source_manifest_round1.json").exists():
            for name in ("predictions", "results", "receipts"):
                path = EXPERIMENT / name
                if path.exists():
                    self.assertEqual(list(path.iterdir()), [])
        text = (EXPERIMENT / "experiment.md").read_text(encoding="utf-8")
        self.assertIn("ENDER27_ROUND2_SOURCE_GATE_ELIGIBLE", text)
        self.assertIn("ENDER27_NEGATIVE_NO_TEMPERED_ALIGNMENT_GAIN", text)
        self.assertIn("y_c - m * (m.T @ y_c) / (m.T @ m)", text)


class TestEnder27SequentialBootstrap(unittest.TestCase):
    @staticmethod
    def _manifest(predecessors: tuple[str, ...]) -> tuple[dict, bytes]:
        manifest = {
            "git_head": "a" * 40,
            "files": {
                f"{BOOTSTRAP.PREFIX}/configs/{name}.py": "b" * 64
                for name in predecessors
            },
        }
        raw = (json.dumps(manifest, sort_keys=True) + "\n").encode()
        return manifest, raw

    @staticmethod
    def _write_predecessor(
        root: Path,
        name: str,
        manifest: dict,
        manifest_bytes: bytes,
    ) -> tuple[Path, Path, Path]:
        experiment = root / BOOTSTRAP.PREFIX
        prediction = experiment / f"predictions/{name}.parquet"
        result = experiment / f"results/{name}.json"
        completion = experiment / f"receipts/{name}.completion.json"
        for path in (prediction, result, completion):
            path.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_bytes(b"synthetic-prediction")
        result.write_bytes(b'{"synthetic":"result"}\n')

        def output_receipt(path: Path) -> dict:
            inspected = path.stat()
            return {
                "path": str(path),
                "device": int(inspected.st_dev),
                "inode": int(inspected.st_ino),
                "size_bytes": int(inspected.st_size),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        config_relative = f"{BOOTSTRAP.PREFIX}/configs/{name}.py"
        payload = {
            "schema_version": 1,
            "stage": "ender27-round1-training-completion",
            "state": "OUTPUTS_FINALIZED",
            "component": name,
            "manifest": {
                "path": f"{BOOTSTRAP.PREFIX}/source_manifest_round1.json",
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "git_head": manifest["git_head"],
            },
            "config": {
                "path": config_relative,
                "sha256": manifest["files"][config_relative],
            },
            "outputs": {
                "predictions": output_receipt(prediction),
                "result": output_receipt(result),
            },
        }
        completion.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return prediction, result, completion

    def test_first_run_proves_all_twelve_outputs_and_decision_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checked = []
            original = BOOTSTRAP.os.path.lexists

            def observe(path):
                checked.append(Path(path))
                return original(path)

            with mock.patch.object(BOOTSTRAP, "REPO_DIR", root), mock.patch.object(
                BOOTSTRAP.os.path, "lexists", side_effect=observe
            ):
                self.assertEqual(
                    BOOTSTRAP._preflight_round1_sequence(
                        BOOTSTRAP.ROUND1_SEQUENCE[0]
                    ),
                    (),
                )
            self.assertEqual(len(checked), 13)
            self.assertEqual(len(set(checked)), 13)

    def test_out_of_order_seed2027_stops_before_reservation_or_acquire(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(BOOTSTRAP, "REPO_DIR", root), mock.patch.object(
                BOOTSTRAP, "_require_launch"
            ), mock.patch.object(
                BOOTSTRAP, "_TrainingOutputReservations"
            ) as reservations, mock.patch.object(BOOTSTRAP, "_acquire") as acquire:
                with self.assertRaisesRegex(ValueError, "predecessor output is missing"):
                    BOOTSTRAP._training_main(
                        ["1", "r1_control_rawresid_seed2027"]
                    )
            reservations.assert_not_called()
            acquire.assert_not_called()

    def test_preexisting_future_or_decision_path_stops_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / BOOTSTRAP.PREFIX
            future = (
                experiment
                / "results/r1_tempered_grank_resid_seed2027.json"
            )
            future.parent.mkdir(parents=True)
            future.write_bytes(b"occupied")
            with mock.patch.object(BOOTSTRAP, "REPO_DIR", root):
                with self.assertRaisesRegex(ValueError, "already consumed"):
                    BOOTSTRAP._preflight_round1_sequence(
                        BOOTSTRAP.ROUND1_SEQUENCE[0]
                    )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision = (
                root
                / BOOTSTRAP.PREFIX
                / "receipts/round1_tempered_alignment.json"
            )
            decision.parent.mkdir(parents=True)
            decision.write_bytes(b"")
            with mock.patch.object(BOOTSTRAP, "REPO_DIR", root):
                with self.assertRaisesRegex(ValueError, "decision destination"):
                    BOOTSTRAP._preflight_round1_sequence(
                        BOOTSTRAP.ROUND1_SEQUENCE[0]
                    )

    def test_valid_predecessor_is_held_and_allows_next_pipeline_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predecessor = BOOTSTRAP.ROUND1_SEQUENCE[0]
            current = BOOTSTRAP.ROUND1_SEQUENCE[1]
            manifest, manifest_bytes = self._manifest((predecessor,))
            self._write_predecessor(
                root, predecessor, manifest, manifest_bytes
            )
            pipeline = SimpleNamespace(
                run_ender27_training_from_bootstrap=mock.Mock(return_value=None)
            )
            with mock.patch.object(BOOTSTRAP, "REPO_DIR", root), mock.patch.object(
                BOOTSTRAP, "_require_launch"
            ), mock.patch.object(
                BOOTSTRAP,
                "_acquire",
                return_value=(manifest, manifest_bytes, ()),
            ), mock.patch.object(
                BOOTSTRAP.importlib, "import_module", return_value=pipeline
            ):
                self.assertEqual(BOOTSTRAP._training_main(["1", current]), 0)
            pipeline.run_ender27_training_from_bootstrap.assert_called_once()

    def test_tampered_predecessor_stops_before_pipeline(self) -> None:
        for tamper in ("completion", "output"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                predecessor = BOOTSTRAP.ROUND1_SEQUENCE[0]
                current = BOOTSTRAP.ROUND1_SEQUENCE[1]
                manifest, manifest_bytes = self._manifest((predecessor,))
                prediction, _, completion = self._write_predecessor(
                    root, predecessor, manifest, manifest_bytes
                )
                if tamper == "completion":
                    payload = json.loads(completion.read_text(encoding="utf-8"))
                    payload["manifest"]["git_head"] = "c" * 40
                    completion.write_text(json.dumps(payload) + "\n", encoding="utf-8")
                else:
                    prediction.write_bytes(b"tampered-prediction-output")
                importer = mock.Mock()
                with mock.patch.object(BOOTSTRAP, "REPO_DIR", root), mock.patch.object(
                    BOOTSTRAP, "_require_launch"
                ), mock.patch.object(
                    BOOTSTRAP,
                    "_acquire",
                    return_value=(manifest, manifest_bytes, ()),
                ), mock.patch.object(BOOTSTRAP.importlib, "import_module", importer):
                    with self.assertRaisesRegex(
                        ValueError, "predecessor completion|predecessor output"
                    ):
                        BOOTSTRAP._training_main(["1", current])
                importer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
