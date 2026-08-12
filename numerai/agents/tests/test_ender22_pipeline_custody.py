from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.code.modeling.utils import pipeline


class TestEnder22PipelineCustody(unittest.TestCase):
    @staticmethod
    def _experiment(root: Path) -> Path:
        return root / "numerai/agents/experiments/ender22_temporal_retention_v53"

    def _config(self, root: Path, name: str) -> Path:
        path = self._experiment(root) / f"configs/{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("CONFIG = {}\n", encoding="utf-8")
        return path

    def test_direct_generic_cli_rejected_before_output_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root, pipeline._ENDER22_ROUND1_NAMES[0])
            with mock.patch.object(pipeline, "REPO_DIR", root), mock.patch.object(
                pipeline, "_run_training_impl"
            ) as implementation:
                with self.assertRaisesRegex(ValueError, "isolated run_round bootstrap"):
                    pipeline.run_training(config)
            implementation.assert_not_called()

    def test_reservation_precedes_manifest_config_and_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = pipeline._ENDER22_ROUND1_NAMES[0]
            config = self._config(root, name)
            experiment = self._experiment(root)
            events: list[str] = []

            class Reservation:
                predictions_path = experiment / f"predictions/{name}.parquet"
                results_path = experiment / f"results/{name}.json"
                completion_path = experiment / f"receipts/{name}.completion.json"
                predictions_stream = object()
                results_stream = object()
                completion_stream = object()

            reservation = Reservation()
            custody = mock.Mock(
                manifest={"files": {}, "external_artifacts": {}},
                manifest_bytes=b"{}",
                leases=(),
                reservations=reservation,
            )

            def implementation(*_args, **_kwargs):
                events.append("config")
                raise AssertionError("CONFIG_EVALUATED")

            def validate(identity, supplied):
                events.append("custody")
                self.assertEqual(identity, (1, name))
                self.assertIs(supplied, custody)
                return custody

            with mock.patch.object(pipeline, "REPO_DIR", root), mock.patch.object(
                pipeline, "_validate_ender22_bootstrap_custody", side_effect=validate
            ), mock.patch.object(
                pipeline, "_require_frozen_python_runtime", side_effect=lambda: events.append("runtime")
            ), mock.patch.object(
                pipeline, "_run_training_impl", side_effect=implementation
            ), mock.patch.object(
                pipeline, "NumerAPI", side_effect=AssertionError("DATA_OPENED")
            ) as data_client:
                with self.assertRaisesRegex(AssertionError, "CONFIG_EVALUATED"):
                    pipeline.run_training(
                        config, _ender22_bootstrap_custody=custody
                    )
            self.assertEqual(events, ["custody", "runtime", "config"])
            data_client.assert_not_called()

    def test_copied_config_cannot_target_governed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            copied = root / "copied.py"
            copied.write_text("CONFIG = {}\n", encoding="utf-8")
            experiment = self._experiment(root)
            config = {
                "data": {"target_col": "target", "id_col": "id"},
                "model": {"params": {}},
                "output": {
                    "output_dir": str(experiment),
                    "results_name": pipeline._ENDER22_ROUND1_NAMES[0],
                },
            }
            with mock.patch.object(pipeline, "REPO_DIR", root), mock.patch.object(
                pipeline, "load_config", return_value=config
            ), mock.patch.object(
                pipeline, "NumerAPI", side_effect=AssertionError("DATA_OPENED")
            ) as data_client:
                with self.assertRaisesRegex(ValueError, "Governed experiment outputs"):
                    pipeline.run_training(copied)
            data_client.assert_not_called()

    def test_canonical_config_rejects_output_override_before_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root, pipeline._ENDER22_ROUND1_NAMES[0])
            with mock.patch.object(pipeline, "REPO_DIR", root), mock.patch.object(
                pipeline, "_verify_ender22_round1_manifest"
            ) as verifier, mock.patch.object(pipeline, "load_config") as loader:
                with self.assertRaisesRegex(ValueError, "may not be redirected"):
                    pipeline.run_training(config, root / "bypass")
            verifier.assert_not_called()
            loader.assert_not_called()

    def test_every_r1_r2_completion_hash_binds_reserved_handles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self._experiment(root)
            for directory in ("configs", "predictions", "results", "receipts"):
                (experiment / directory).mkdir(parents=True, exist_ok=True)
            for name in pipeline._ENDER22_TRAINING_NAMES:
                round_number = 1 if name in pipeline._ENDER22_ROUND1_NAMES else 2
                config = experiment / f"configs/{name}.py"
                config_bytes = b"CONFIG = {'frozen': True}\n"
                config.write_bytes(config_bytes)
                relative = config.relative_to(root).as_posix()
                manifest = {
                    "git_head": "a" * 40,
                    "files": {relative: hashlib.sha256(config_bytes).hexdigest()},
                }
                manifest_path = experiment / f"source_manifest_round{round_number}.json"
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                predictions = experiment / f"predictions/{name}.parquet"
                result = experiment / f"results/{name}.json"
                completion = experiment / f"receipts/{name}.completion.json"
                with mock.patch.object(pipeline, "REPO_DIR", root):
                    with pipeline._ExclusiveOutputReservations(
                        predictions, result, completion
                    ) as reservations:
                        prediction_bytes = f"prediction:{name}".encode()
                        result_bytes = f"result:{name}".encode()
                        reservations.predictions_stream.write(prediction_bytes)
                        reservations.results_stream.write(result_bytes)
                        manifest_bytes = manifest_path.read_bytes()
                        path, payload, payload_bytes = pipeline._write_ender22_completion(
                            config, manifest, manifest_bytes, reservations
                        )
                self.assertEqual(path, completion)
                self.assertEqual(completion.read_bytes(), payload_bytes)
                self.assertEqual(
                    payload["outputs"]["predictions"]["sha256"],
                    hashlib.sha256(prediction_bytes).hexdigest(),
                )
                self.assertEqual(
                    payload["outputs"]["result"]["sha256"],
                    hashlib.sha256(result_bytes).hexdigest(),
                )
                self.assertEqual(
                    payload["stage"],
                    f"ender22-round{round_number}-training-completion",
                )


if __name__ == "__main__":
    unittest.main()
