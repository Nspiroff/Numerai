from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.code.modeling.utils import pipeline


class TestEnder27PipelineCustody(unittest.TestCase):
    @staticmethod
    def _experiment(root: Path) -> Path:
        return root / pipeline._ENDER27_PREFIX

    def _config(self, root: Path, name: str) -> Path:
        path = self._experiment(root) / f"configs/{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("CONFIG = {}\n", encoding="utf-8")
        return path

    @staticmethod
    def _authority(external: dict[str, dict[str, object]]) -> bytes:
        full_path = (
            "numerai/v5.3/ender21_discovery_full_through_0861.parquet"
        )
        benchmark_path = (
            "numerai/v5.3/"
            "ender21_discovery_benchmark_models_through_0861.parquet"
        )
        payload = {
            "schema_version": 1,
            "authority": "ender27-discovery-only",
            "full": {"path": full_path, **external[full_path]},
            "benchmark": {
                "path": benchmark_path,
                **external[benchmark_path],
            },
        }
        return (json.dumps(payload, sort_keys=True) + "\n").encode()

    def _custody(self, root: Path, name: str):
        experiment = self._experiment(root)
        source_digest = "1" * 64
        artifact_digest = "2" * 64
        external = {
            relative: {
                "size_bytes": 17,
                "sha256": artifact_digest,
                "last_era": "0861",
            }
            for relative in pipeline._ENDER27_EXTERNAL_ARTIFACTS
        }
        manifest = {
            "schema_version": 1,
            "git_head": "a" * 40,
            "files": {
                relative: source_digest
                for relative in pipeline._ENDER27_ROUND1_MANIFEST_FILES
            },
            "external_artifacts": external,
        }
        manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
        authority_bytes = self._authority(external)

        class Lease:
            def __init__(self, path, digest, *, payload=b"", size=0):
                self.path = Path(path)
                self.digest = digest
                self.payload = payload
                self.size = size
                self.closed = False

            def read_bytes(self):
                if self.closed:
                    raise RuntimeError("closed lease")
                return self.payload

            def sha256(self):
                if self.closed:
                    raise RuntimeError("closed lease")
                return self.digest

            def size_bytes(self):
                if self.closed:
                    raise RuntimeError("closed lease")
                return self.size

            def close(self):
                self.closed = True

        manifest_path = experiment / "source_manifest_round1.json"
        leases = [
            Lease(
                manifest_path,
                hashlib.sha256(manifest_bytes).hexdigest(),
                payload=manifest_bytes,
            )
        ]
        authority_relative = (
            f"{pipeline._ENDER27_PREFIX}/protocol/discovery_data_authority.json"
        )
        for relative in pipeline._ENDER27_ROUND1_MANIFEST_FILES:
            leases.append(
                Lease(
                    root / relative,
                    source_digest,
                    payload=(
                        authority_bytes if relative == authority_relative else b""
                    ),
                )
            )
        leases.extend(
            Lease(root / relative, artifact_digest, size=17)
            for relative in pipeline._ENDER27_EXTERNAL_ARTIFACTS
        )

        for directory in ("predictions", "results", "receipts"):
            (experiment / directory).mkdir(parents=True, exist_ok=True)
        reservations = pipeline._ExclusiveOutputReservations(
            experiment / f"predictions/{name}.parquet",
            experiment / f"results/{name}.json",
            experiment / f"receipts/{name}.completion.json",
        )
        reservations.__enter__()

        custody = pipeline._Ender27BootstrapCustody(
            round_number=1,
            config_name=name,
            manifest=manifest,
            manifest_bytes=manifest_bytes,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            leases=tuple(leases),
            reservations=reservations,
        )
        return custody, leases

    def test_manifest_and_governed_output_sets_are_exact(self) -> None:
        self.assertEqual(len(pipeline._ENDER27_ROUND1_MANIFEST_FILES), 31)
        self.assertEqual(
            pipeline._ENDER27_EXTERNAL_ARTIFACTS,
            {
                "numerai/v5.3/ender21_discovery_full_through_0861.parquet",
                (
                    "numerai/v5.3/"
                    "ender21_discovery_benchmark_models_through_0861.parquet"
                ),
            },
        )
        self.assertEqual(
            pipeline._ENDER27_DISCOVERY_AUTHORITY,
            "ender27-discovery-only",
        )
        required = {
            "numerai/agents/code/modeling/__main__.py",
            "numerai/agents/code/modeling/models/torch_tabular_regressor.py",
            "numerai/agents/code/modeling/utils/cli.py",
            "numerai/agents/code/modeling/utils/config.py",
            "numerai/agents/code/modeling/utils/constants.py",
            "numerai/agents/code/modeling/utils/data.py",
            "numerai/agents/code/modeling/utils/model_data.py",
            "numerai/agents/code/modeling/utils/model_factory.py",
            "numerai/agents/code/modeling/utils/numerai_cv.py",
            "numerai/agents/code/modeling/utils/pipeline.py",
            "numerai/agents/code/modeling/utils/target_transforms.py",
            "numerai/agents/code/metrics/numerai_metrics.py",
            (
                "numerai/agents/experiments/ender21_residual_stability_v53/"
                "protocol/discovery_eras_through_0861.json"
            ),
            (
                "numerai/agents/experiments/ender21_residual_stability_v53/"
                "protocol/feature_columns_all_v53.json"
            ),
            f"{pipeline._ENDER27_PREFIX}/experiment.md",
            f"{pipeline._ENDER27_PREFIX}/gate.md",
            f"{pipeline._ENDER27_PREFIX}/protocol/discovery_data_authority.json",
            f"{pipeline._ENDER27_PREFIX}/configs/base_r1.py",
            *(
                f"{pipeline._ENDER27_PREFIX}/configs/{name}.py"
                for name in pipeline._ENDER27_ROUND1_NAMES
            ),
            f"{pipeline._ENDER27_PREFIX}/run_round1.py",
            f"{pipeline._ENDER27_PREFIX}/training_bootstrap.py",
            f"{pipeline._ENDER27_PREFIX}/evaluation_common.py",
            f"{pipeline._ENDER27_PREFIX}/evaluate_round1.py",
            f"{pipeline._ENDER27_PREFIX}/evaluate_round1_impl.py",
            "numerai/agents/tests/test_target_transforms.py",
            "numerai/agents/tests/test_ender27_tempered_gaussian_rank_residual.py",
            "numerai/agents/tests/test_ender27_pipeline_custody.py",
            "numerai/agents/tests/test_ender27_evaluator.py",
        }
        self.assertEqual(pipeline._ENDER27_ROUND1_MANIFEST_FILES, required)

        experiment = Path(pipeline.REPO_DIR) / pipeline._ENDER27_PREFIX
        expected_outputs = {
            *(
                experiment / "predictions" / f"{name}.parquet"
                for name in pipeline._ENDER27_ROUND1_NAMES
            ),
            *(
                experiment / "results" / f"{name}.json"
                for name in pipeline._ENDER27_ROUND1_NAMES
            ),
            *(
                experiment / "receipts" / f"{name}.completion.json"
                for name in pipeline._ENDER27_ROUND1_NAMES
            ),
            experiment / "receipts/round1_tempered_alignment.json",
        }
        governed = pipeline._governed_output_paths()
        self.assertTrue(expected_outputs.issubset(governed))
        self.assertFalse(
            any(
                pipeline._ENDER27_PREFIX in str(path) and "round2" in str(path).lower()
                for path in governed
            )
        )

    def test_only_four_canonical_round1_configs_are_recognized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(pipeline, "REPO_DIR", root):
                for name in pipeline._ENDER27_ROUND1_NAMES:
                    config = self._config(root, name)
                    self.assertEqual(
                        pipeline._ender27_config_identity(config, None),
                        (1, name),
                    )
                base = self._config(root, "base_r1")
                with self.assertRaisesRegex(ValueError, "frozen named Ender27"):
                    pipeline._ender27_config_identity(base, None)
                round2 = self._config(root, "r2_grank_resid_seed7331")
                with self.assertRaisesRegex(ValueError, "frozen named Ender27"):
                    pipeline._ender27_config_identity(round2, None)
                config = self._config(root, pipeline._ENDER27_ROUND1_NAMES[0])
                with self.assertRaisesRegex(ValueError, "may not be redirected"):
                    pipeline._ender27_config_identity(config, root / "bypass")

    def test_direct_generic_cli_rejects_before_config_or_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root, pipeline._ENDER27_ROUND1_NAMES[0])
            with mock.patch.object(pipeline, "REPO_DIR", root), mock.patch.object(
                pipeline, "_run_training_impl"
            ) as implementation, mock.patch.object(
                pipeline, "load_config"
            ) as loader, mock.patch.object(
                pipeline, "NumerAPI", side_effect=AssertionError("DATA_OPENED")
            ) as data_client:
                with self.assertRaisesRegex(ValueError, "isolated run_round bootstrap"):
                    pipeline.run_training(config)
            implementation.assert_not_called()
            loader.assert_not_called()
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
                    "results_name": pipeline._ENDER27_ROUND1_NAMES[0],
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

    def test_create_new_triple_has_exact_paths_and_rejects_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = pipeline._ENDER27_ROUND1_NAMES[0]
            config = self._config(root, name)
            with mock.patch.object(pipeline, "REPO_DIR", root):
                reservations = pipeline._ender27_output_reservations(config, None)
                self.assertIsNotNone(reservations)
                assert reservations is not None
                experiment = self._experiment(root)
                self.assertEqual(
                    (
                        reservations.predictions_path,
                        reservations.results_path,
                        reservations.completion_path,
                    ),
                    (
                        experiment / f"predictions/{name}.parquet",
                        experiment / f"results/{name}.json",
                        experiment / f"receipts/{name}.completion.json",
                    ),
                )
                with reservations:
                    self.assertIsNotNone(reservations.predictions_stream)
                    self.assertIsNotNone(reservations.results_stream)
                    self.assertIsNotNone(reservations.completion_stream)
                second = pipeline._ender27_output_reservations(config, None)
                assert second is not None
                with self.assertRaisesRegex(ValueError, "Cannot reserve exclusive"):
                    with second:
                        self.fail("CREATE_NEW unexpectedly reused Ender27 outputs")

    def test_custody_requires_exact_held_envelope_and_authority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = pipeline._ENDER27_ROUND1_NAMES[0]
            custody, leases = self._custody(root, name)
            try:
                with mock.patch.object(pipeline, "REPO_DIR", root):
                    self.assertIs(
                        pipeline._validate_ender27_bootstrap_custody(
                            (1, name), custody
                        ),
                        custody,
                    )
                    shortened = pipeline._Ender27BootstrapCustody(
                        **{**custody.__dict__, "leases": tuple(leases[:-1])}
                    )
                    with self.assertRaisesRegex(ValueError, "lease set differs"):
                        pipeline._validate_ender27_bootstrap_custody(
                            (1, name), shortened
                        )

                    authority_relative = (
                        f"{pipeline._ENDER27_PREFIX}/"
                        "protocol/discovery_data_authority.json"
                    )
                    authority_path = root / authority_relative
                    authority_lease = next(
                        lease for lease in leases if lease.path == authority_path
                    )
                    authority = json.loads(authority_lease.payload)
                    authority["authority"] = "ender24-discovery-only"
                    authority_lease.payload = json.dumps(authority).encode()
                    with self.assertRaisesRegex(
                        ValueError, "authority envelope differs"
                    ):
                        pipeline._validate_ender27_bootstrap_custody(
                            (1, name), custody
                        )
            finally:
                custody.reservations.__exit__(None, None, None)

    def test_reservation_and_leases_remain_held_through_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = pipeline._ENDER27_ROUND1_NAMES[0]
            config = self._config(root, name)
            custody, leases = self._custody(root, name)
            events: list[str] = []

            def validate(identity, supplied):
                events.append("custody")
                self.assertEqual(identity, (1, name))
                self.assertIs(supplied, custody)
                self.assertTrue(all(not lease.closed for lease in leases))
                self.assertTrue(
                    all(
                        getattr(custody.reservations, stream) is not None
                        for stream in (
                            "predictions_stream",
                            "results_stream",
                            "completion_stream",
                        )
                    )
                )
                return custody

            def implementation(*_args, **_kwargs):
                events.append("config/data")
                self.assertTrue(all(not lease.closed for lease in leases))
                return Path("predictions"), Path("results")

            def completion(*_args, **_kwargs):
                events.append("completion-fsync")
                self.assertTrue(all(not lease.closed for lease in leases))
                return Path("completion"), {}, b"completion\n"

            try:
                with mock.patch.object(
                    pipeline, "REPO_DIR", root
                ), mock.patch.object(
                    pipeline,
                    "_validate_ender27_bootstrap_custody",
                    side_effect=validate,
                ), mock.patch.object(
                    pipeline,
                    "_require_frozen_python_runtime",
                    side_effect=lambda: events.append("runtime"),
                ), mock.patch.object(
                    pipeline, "_run_training_impl", side_effect=implementation
                ), mock.patch.object(
                    pipeline, "_write_ender27_completion", side_effect=completion
                ):
                    outputs = pipeline.run_training(
                        config, _ender27_bootstrap_custody=custody
                    )
                self.assertEqual(outputs, (Path("predictions"), Path("results")))
                self.assertEqual(
                    events,
                    ["custody", "runtime", "config/data", "completion-fsync"],
                )
                self.assertTrue(all(not lease.closed for lease in leases))
            finally:
                custody.reservations.__exit__(None, None, None)

    def test_bootstrap_entry_binds_exact_round_and_custody(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = pipeline._ENDER27_ROUND1_NAMES[0]
            config = self._config(root, name)
            manifest_bytes = b'{"frozen":true}'
            leases = (object(),)
            reservations = object()
            with mock.patch.object(pipeline, "REPO_DIR", root), mock.patch.object(
                pipeline, "run_training", return_value=(Path("p"), Path("r"))
            ) as runner:
                outputs = pipeline.run_ender27_training_from_bootstrap(
                    config,
                    round_number=1,
                    manifest={"frozen": True},
                    manifest_bytes=manifest_bytes,
                    leases=leases,
                    reservations=reservations,
                )
            self.assertEqual(outputs, (Path("p"), Path("r")))
            custody = runner.call_args.kwargs["_ender27_bootstrap_custody"]
            self.assertEqual((custody.round_number, custody.config_name), (1, name))
            self.assertEqual(
                custody.manifest_sha256,
                hashlib.sha256(manifest_bytes).hexdigest(),
            )
            self.assertIs(custody.reservations, reservations)
            with mock.patch.object(pipeline, "REPO_DIR", root):
                with self.assertRaisesRegex(
                    ValueError, "config/round identity differs"
                ):
                    pipeline.run_ender27_training_from_bootstrap(
                        config,
                        round_number=2,
                        manifest={},
                        manifest_bytes=b"{}",
                        leases=(),
                        reservations=object(),
                    )

    def test_round1_completion_hash_binds_reserved_handles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = self._experiment(root)
            for directory in ("configs", "predictions", "results", "receipts"):
                (experiment / directory).mkdir(parents=True, exist_ok=True)
            for name in pipeline._ENDER27_ROUND1_NAMES:
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
                        path, payload, payload_bytes = (
                            pipeline._write_ender27_completion(
                                config, manifest, manifest_bytes, reservations
                            )
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
                    payload["stage"], "ender27-round1-training-completion"
                )


if __name__ == "__main__":
    unittest.main()
