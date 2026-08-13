from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.code.modeling.utils import pipeline


class TestEnder24PipelineCustody(unittest.TestCase):
    @staticmethod
    def _experiment(root: Path) -> Path:
        return root / "numerai/agents/experiments/ender24_ema_seed_stability_v53"

    def _config(self, root: Path, name: str) -> Path:
        path = self._experiment(root) / f"configs/{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("CONFIG = {}\n", encoding="utf-8")
        return path

    def test_manifest_set_and_governed_paths_are_exact(self) -> None:
        self.assertEqual(len(pipeline._ENDER24_ROUND1_MANIFEST_FILES), 31)
        self.assertEqual(
            pipeline._ENDER24_EXTERNAL_ARTIFACTS,
            {
                "numerai/v5.3/ender21_discovery_full_through_0861.parquet",
                (
                    "numerai/v5.3/"
                    "ender21_discovery_benchmark_models_through_0861.parquet"
                ),
            },
        )
        for required in (
            f"{pipeline._ENDER24_PREFIX}/protocol/mechanical_activity_receipt.json",
            f"{pipeline._ENDER24_PREFIX}/training_bootstrap.py",
            f"{pipeline._ENDER24_PREFIX}/evaluate_round1_impl.py",
            "numerai/agents/tests/test_ender24_ema_seed_stability.py",
        ):
            self.assertIn(required, pipeline._ENDER24_ROUND1_MANIFEST_FILES)

        governed = pipeline._governed_output_paths()
        experiment = Path(pipeline.REPO_DIR) / pipeline._ENDER24_PREFIX
        expected = {
            *(
                experiment / "predictions" / f"{name}.parquet"
                for name in pipeline._ENDER24_TRAINING_NAMES
            ),
            *(
                experiment / "results" / f"{name}.json"
                for name in pipeline._ENDER24_TRAINING_NAMES
            ),
            *(
                experiment / "receipts" / f"{name}.completion.json"
                for name in pipeline._ENDER24_TRAINING_NAMES
            ),
            experiment / "receipts/round1_ema_stability.json",
            experiment / "receipts/round2_seed_replication.json",
        }
        self.assertTrue(expected.issubset(governed))

    def test_all_six_configs_are_recognized_but_only_round1_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(pipeline, "REPO_DIR", root):
                for name in pipeline._ENDER24_ROUND1_NAMES:
                    config = self._config(root, name)
                    self.assertEqual(
                        pipeline._ender24_config_identity(config, None),
                        (1, name),
                    )
                for name in pipeline._ENDER24_ROUND2_NAMES:
                    config = self._config(root, name)
                    self.assertEqual(
                        pipeline._ender24_config_identity(config, None),
                        (2, name),
                    )
                    custody = pipeline._Ender24BootstrapCustody(
                        round_number=2,
                        config_name=name,
                        manifest={},
                        manifest_bytes=b"{}",
                        manifest_sha256=hashlib.sha256(b"{}").hexdigest(),
                        leases=(),
                        reservations=object(),
                    )
                    with self.assertRaisesRegex(ValueError, "Round 2 is not authorized"):
                        pipeline._validate_ender24_bootstrap_custody(
                            (2, name), custody
                        )

    def test_direct_generic_cli_rejected_before_config_or_output_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root, pipeline._ENDER24_ROUND1_NAMES[0])
            with mock.patch.object(pipeline, "REPO_DIR", root), mock.patch.object(
                pipeline, "_run_training_impl"
            ) as implementation, mock.patch.object(pipeline, "load_config") as loader:
                with self.assertRaisesRegex(ValueError, "isolated run_round bootstrap"):
                    pipeline.run_training(config)
            implementation.assert_not_called()
            loader.assert_not_called()

    def test_reservation_custody_precedes_config_and_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = pipeline._ENDER24_ROUND1_NAMES[0]
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

            def validate(identity, supplied):
                events.append("custody")
                self.assertEqual(identity, (1, name))
                self.assertIs(supplied, custody)
                return custody

            def implementation(*_args, **_kwargs):
                events.append("config")
                raise AssertionError("CONFIG_EVALUATED")

            with mock.patch.object(pipeline, "REPO_DIR", root), mock.patch.object(
                pipeline,
                "_validate_ender24_bootstrap_custody",
                side_effect=validate,
            ), mock.patch.object(
                pipeline,
                "_require_frozen_python_runtime",
                side_effect=lambda: events.append("runtime"),
            ), mock.patch.object(
                pipeline, "_run_training_impl", side_effect=implementation
            ), mock.patch.object(
                pipeline, "NumerAPI", side_effect=AssertionError("DATA_OPENED")
            ) as data_client:
                with self.assertRaisesRegex(AssertionError, "CONFIG_EVALUATED"):
                    pipeline.run_training(
                        config, _ender24_bootstrap_custody=custody
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
                    "results_name": pipeline._ENDER24_ROUND1_NAMES[0],
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

    def test_output_override_is_rejected_before_manifest_or_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root, pipeline._ENDER24_ROUND1_NAMES[0])
            with mock.patch.object(pipeline, "REPO_DIR", root), mock.patch.object(
                pipeline, "_verify_ender24_round1_manifest"
            ) as verifier, mock.patch.object(pipeline, "load_config") as loader:
                with self.assertRaisesRegex(ValueError, "may not be redirected"):
                    pipeline.run_training(config, root / "bypass")
            verifier.assert_not_called()
            loader.assert_not_called()

    def test_bootstrap_entry_binds_exact_round_and_custody(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = pipeline._ENDER24_ROUND1_NAMES[0]
            config = self._config(root, name)
            manifest_bytes = b'{"frozen":true}'
            leases = (object(),)
            reservations = object()
            with mock.patch.object(pipeline, "REPO_DIR", root), mock.patch.object(
                pipeline, "run_training", return_value=(Path("p"), Path("r"))
            ) as runner:
                outputs = pipeline.run_ender24_training_from_bootstrap(
                    config,
                    round_number=1,
                    manifest={"frozen": True},
                    manifest_bytes=manifest_bytes,
                    leases=leases,
                    reservations=reservations,
                )
            self.assertEqual(outputs, (Path("p"), Path("r")))
            custody = runner.call_args.kwargs["_ender24_bootstrap_custody"]
            self.assertEqual((custody.round_number, custody.config_name), (1, name))
            self.assertEqual(
                custody.manifest_sha256,
                hashlib.sha256(manifest_bytes).hexdigest(),
            )
            self.assertIs(custody.reservations, reservations)
            with mock.patch.object(pipeline, "REPO_DIR", root):
                with self.assertRaisesRegex(ValueError, "config/round identity differs"):
                    pipeline.run_ender24_training_from_bootstrap(
                        config,
                        round_number=2,
                        manifest={},
                        manifest_bytes=b"{}",
                        leases=(),
                        reservations=object(),
                    )

    def test_custody_requires_the_exact_held_lease_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = pipeline._ENDER24_ROUND1_NAMES[0]
            experiment = self._experiment(root)
            source_digest = "1" * 64
            artifact_digest = "2" * 64
            manifest = {
                "files": {
                    relative: source_digest
                    for relative in pipeline._ENDER24_ROUND1_MANIFEST_FILES
                },
                "external_artifacts": {
                    relative: {
                        "size_bytes": 17,
                        "sha256": artifact_digest,
                        "last_era": "0861",
                    }
                    for relative in pipeline._ENDER24_EXTERNAL_ARTIFACTS
                },
            }
            manifest_bytes = json.dumps(manifest, sort_keys=True).encode()

            class Lease:
                def __init__(self, path, digest, *, payload=b"", size=0):
                    self.path = Path(path)
                    self.digest = digest
                    self.payload = payload
                    self.size = size

                def read_bytes(self):
                    return self.payload

                def sha256(self):
                    return self.digest

                def size_bytes(self):
                    return self.size

            manifest_path = experiment / "source_manifest_round1.json"
            leases = [Lease(manifest_path, "0" * 64, payload=manifest_bytes)]
            leases.extend(
                Lease(root / relative, source_digest)
                for relative in pipeline._ENDER24_ROUND1_MANIFEST_FILES
            )
            leases.extend(
                Lease(root / relative, artifact_digest, size=17)
                for relative in pipeline._ENDER24_EXTERNAL_ARTIFACTS
            )

            class Reservations:
                predictions_path = experiment / f"predictions/{name}.parquet"
                results_path = experiment / f"results/{name}.json"
                completion_path = experiment / f"receipts/{name}.completion.json"
                predictions_stream = object()
                results_stream = object()
                completion_stream = object()

            custody = pipeline._Ender24BootstrapCustody(
                round_number=1,
                config_name=name,
                manifest=manifest,
                manifest_bytes=manifest_bytes,
                manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                leases=tuple(leases),
                reservations=Reservations(),
            )
            with mock.patch.object(pipeline, "REPO_DIR", root):
                self.assertIs(
                    pipeline._validate_ender24_bootstrap_custody(
                        (1, name), custody
                    ),
                    custody,
                )
                shortened = pipeline._Ender24BootstrapCustody(
                    **{**custody.__dict__, "leases": tuple(leases[:-1])}
                )
                with self.assertRaisesRegex(ValueError, "lease set differs"):
                    pipeline._validate_ender24_bootstrap_custody(
                        (1, name), shortened
                    )

    def test_round1_completion_hash_binds_reserved_handles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self._experiment(root)
            for directory in ("configs", "predictions", "results", "receipts"):
                (experiment / directory).mkdir(parents=True, exist_ok=True)
            for name in pipeline._ENDER24_ROUND1_NAMES:
                config = experiment / f"configs/{name}.py"
                config_bytes = b"CONFIG = {'frozen': True}\n"
                config.write_bytes(config_bytes)
                relative = config.relative_to(root).as_posix()
                manifest = {
                    "git_head": "a" * 40,
                    "files": {
                        relative: hashlib.sha256(config_bytes).hexdigest()
                    },
                }
                manifest_bytes = (
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                ).encode()
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
                        path, payload, payload_bytes = pipeline._write_ender24_completion(
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
                    payload["stage"], "ender24-round1-training-completion"
                )


if __name__ == "__main__":
    unittest.main()
