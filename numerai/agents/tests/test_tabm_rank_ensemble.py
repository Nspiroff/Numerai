from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from agents.code.modeling.deployment.tabm_rank_ensemble import (
    build_frozen_ender20_rank_ensemble_from_bundles,
    build_three_tabm_rank_ensemble_predictor,
)
from agents.code.modeling.deployment.final_fit_export import (
    GATE_CONFIG_RELATIVE_PREFIX,
    GATE_STORE_METADATA_RELATIVE_PATH,
    SUPPORTED_RUNS,
    _split_predictor_spec,
    load_gate_source_manifest_pin,
    write_intermediate_bundle,
)


def _selector_spec(feature_index: int, *, feature_names=None):
    names = feature_names or ["feature_a", "feature_b", "feature_c"]
    weight = np.zeros((1, len(names)), dtype=np.float32)
    weight[0, feature_index] = 1.0
    return {
        "feature_names": names,
        "blocks": [
            {
                "weight": weight,
                "r": np.ones((1, len(names)), dtype=np.float32),
                "s": np.ones((1, 1), dtype=np.float32),
                "bias": np.zeros((1, 1), dtype=np.float32),
            }
        ],
        "output_weight": np.ones((1, 1, 1), dtype=np.float32),
        "output_bias": np.zeros((1, 1), dtype=np.float32),
        "feature_center": 0.0,
        "feature_scale": 1.0,
        "batch_size": 2,
        "activation": "relu",
        "era_column": "era",
        "prediction_column": "prediction",
    }


def _three_specs():
    return [_selector_spec(index) for index in range(3)]


def _positions_sha256(positions):
    canonical = np.ascontiguousarray(positions, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _write_lineage_fixture(root: Path, *, spec_indices=(0, 1, 2), positions=None):
    config_hashes = {
        name: hashlib.sha256(name.encode("utf-8")).hexdigest()
        for name in SUPPORTED_RUNS
    }
    store_metadata_sha256 = "d" * 64
    gate_path = root / "gate_source_manifest.json"
    gate_path.write_text(
        json.dumps(
            {
                "hash_algorithm": "sha256",
                "files": {
                    GATE_STORE_METADATA_RELATIVE_PATH: store_metadata_sha256,
                    **{
                        f"{GATE_CONFIG_RELATIVE_PREFIX}/{name}": digest
                        for name, digest in config_hashes.items()
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    gate_source = load_gate_source_manifest_pin(gate_path)
    if positions is None:
        positions = (
            np.array([1, 3], dtype=np.int64),
            np.array([1, 3], dtype=np.int64),
            np.array([2, 4], dtype=np.int64),
        )

    bundle_dirs = []
    for index, (config_name, (model_seed, sample_seed)) in enumerate(
        SUPPORTED_RUNS.items()
    ):
        raw_spec = _selector_spec(spec_indices[index])
        predictor_spec, arrays = _split_predictor_spec(
            raw_spec, tuple(raw_spec["feature_names"])
        )
        sample_positions = np.asarray(positions[index], dtype=np.int64)
        sample_hash = _positions_sha256(sample_positions)
        config_relative_path = f"{GATE_CONFIG_RELATIVE_PREFIX}/{config_name}"
        provenance = {
            "format": "numerai-ender20-final-fit-provenance",
            "format_version": 1,
            "artifact_state": "intermediates_only_no_pickle_no_upload",
            "gate_source": {
                "manifest_path": str(gate_path),
                "manifest_sha256": gate_source["manifest_sha256"],
                "store_metadata_relative_path": GATE_STORE_METADATA_RELATIVE_PATH,
                "expected_store_metadata_sha256": store_metadata_sha256,
                "config_relative_path": config_relative_path,
                "expected_config_sha256": config_hashes[config_name],
            },
            "config": {
                "name": config_name,
                "file_sha256": config_hashes[config_name],
                "canonical_sha256": "c" * 64,
            },
            "store": {
                "generation_id": "0" * 32,
                "metadata_sha256": store_metadata_sha256,
                "manifest_sha256": "a" * 64,
                "feature_sha256": "b" * 64,
                "feature_order_sha256": "c" * 64,
                "row_count": 100,
                "feature_count": 3,
                "era_start": "0001",
                "era_end": "0010",
                "era_count": 10,
            },
            "sample": {
                "seed": sample_seed,
                "manifest_positions_sha256": sample_hash,
            },
            "training": {
                "model_seed": model_seed,
                "selected_best_epoch": 2,
                "target_transform": {
                    "type": "residual_to_benchmark",
                    "benchmark_col": "v53_lgbm_ender20",
                    "era_col": "era",
                    "per_era": True,
                    "fit_intercept": True,
                },
            },
        }
        bundle_dir = root / f"bundle-{index}"
        write_intermediate_bundle(
            bundle_dir,
            predictor_spec=predictor_spec,
            arrays=arrays,
            sample_manifest_positions=sample_positions,
            provenance=provenance,
        )
        bundle_dirs.append(bundle_dir)
    return gate_path, bundle_dirs


def _mutate_provenance(bundle_dir: Path, mutate):
    path = bundle_dir / "provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    mutate(provenance)
    path.write_text(json.dumps(provenance), encoding="utf-8")


class TestTabMRankEnsemble(unittest.TestCase):
    def test_averages_each_models_rank_once_without_second_rank(self) -> None:
        predictor = build_three_tabm_rank_ensemble_predictor(
            model_specs=_three_specs()
        )
        frame = pd.DataFrame(
            {
                "feature_c": [1.0, 3.0, 4.0, 2.0],
                "era": ["0001"] * 4,
                "feature_a": [1.0, 2.0, 3.0, 4.0],
                "feature_b": [1.0, 2.0, 4.0, 3.0],
            },
            index=["row-4", "row-1", "row-3", "row-2"],
        )

        output = predictor(frame, object())

        expected = np.array([0.25, 7.0 / 12.0, 11.0 / 12.0, 0.75])
        np.testing.assert_allclose(output["prediction"], expected)
        second_rank = output["prediction"].rank(method="average", pct=True)
        self.assertFalse(np.allclose(output["prediction"], second_rank))
        self.assertEqual(
            predictor.ensemble_method,
            "mean_of_three_per_era_average_percentile_ranks",
        )
        self.assertFalse(predictor.reranks_ensemble_mean)
        self.assertTrue(output.index.equals(frame.index))

    def test_ranks_with_average_ties_per_era_and_ignores_benchmarks(self) -> None:
        predictor = build_three_tabm_rank_ensemble_predictor(
            model_specs=_three_specs()
        )
        frame = pd.DataFrame(
            {
                "feature_a": [1.0, 1.0, 3.0, 3.0, 2.0],
                "feature_b": [3.0, 1.0, 2.0, 1.0, 1.0],
                "feature_c": [2.0, 3.0, 1.0, 2.0, 2.0],
                "era": ["0001", "0001", "0001", "0002", "0002"],
            },
            index=["x-5", "x-1", "x-4", "x-3", "x-2"],
        )

        output = predictor(frame, object())

        expected_ranks = []
        for name in ("feature_a", "feature_b", "feature_c"):
            expected_ranks.append(
                frame[name]
                .groupby(frame["era"], sort=False, dropna=False)
                .rank(method="average", pct=True)
            )
        expected = sum(expected_ranks) / 3.0
        np.testing.assert_allclose(output["prediction"], expected)
        self.assertTrue(output["prediction"].between(0.0, 1.0).all())
        self.assertTrue(output.index.equals(frame.index))

    def test_build_and_live_input_validation_is_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly three"):
            build_three_tabm_rank_ensemble_predictor(
                model_specs=_three_specs()[:2]
            )

        mismatched = _three_specs()
        mismatched[2] = _selector_spec(
            2, feature_names=["feature_a", "feature_c", "feature_b"]
        )
        with self.assertRaisesRegex(ValueError, "same frozen feature order"):
            build_three_tabm_rank_ensemble_predictor(model_specs=mismatched)

        invalid_era = _three_specs()
        invalid_era[2]["era_column"] = "other_era"
        with self.assertRaisesRegex(ValueError, "same era_column"):
            build_three_tabm_rank_ensemble_predictor(model_specs=invalid_era)

        predictor = build_three_tabm_rank_ensemble_predictor(
            model_specs=_three_specs()
        )
        valid = pd.DataFrame(
            {
                "era": ["0001", "0001"],
                "feature_a": [1.0, 2.0],
                "feature_b": [2.0, 1.0],
                "feature_c": [1.0, 1.0],
            },
            index=["a", "b"],
        )
        cases = {
            "missing era": valid.drop(columns="era"),
            "missing feature": valid.drop(columns="feature_b"),
            "missing era value": valid.assign(era=["0001", None]),
            "duplicate index": valid.set_axis(["a", "a"]),
            "nonnumeric feature": valid.assign(feature_a=[1.0, "bad"]),
        }
        for name, frame in cases.items():
            with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                predictor(frame, object())

    def test_frozen_bundle_assembler_binds_lineage_and_orders_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate_path, bundle_dirs = _write_lineage_fixture(Path(tmp))
            predictor = build_frozen_ender20_rank_ensemble_from_bundles(
                bundle_dirs=[bundle_dirs[2], bundle_dirs[0], bundle_dirs[1]],
                gate_source_manifest_path=gate_path,
            )
            self.assertEqual(predictor.bundle_config_names, tuple(SUPPORTED_RUNS))
            self.assertEqual(len(set(predictor.bundle_weight_sha256)), 3)
            self.assertEqual(predictor.store_generation_id, "0" * 32)
            self.assertEqual(
                predictor.gate_source_manifest_sha256,
                load_gate_source_manifest_pin(gate_path)["manifest_sha256"],
            )

            frame = pd.DataFrame(
                {
                    "feature_c": [1.0, 3.0, 4.0, 2.0],
                    "era": ["0001"] * 4,
                    "feature_a": [1.0, 2.0, 3.0, 4.0],
                    "feature_b": [1.0, 2.0, 4.0, 3.0],
                },
                index=["row-4", "row-1", "row-3", "row-2"],
            )
            expected = np.array([0.25, 7.0 / 12.0, 11.0 / 12.0, 0.75])
            np.testing.assert_allclose(
                predictor(frame, object())["prediction"], expected
            )

            try:
                import cloudpickle
            except ImportError:
                return
            pickle_path = Path(tmp) / "frozen-ensemble.pkl"
            with pickle_path.open("wb") as stream:
                cloudpickle.dump(predictor, stream)
            script = r'''
import sys

class BlockLocalAndMLPackages:
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("agents", "tabm", "torch", "rtdl_num_embeddings")
        if fullname in blocked or fullname.startswith(tuple(x + "." for x in blocked)):
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockLocalAndMLPackages())
import cloudpickle
import pandas as pd
with open(sys.argv[1], "rb") as stream:
    predict = cloudpickle.load(stream)
frame = pd.DataFrame(
    {
        "feature_c": [1.0, 3.0, 4.0, 2.0],
        "era": ["0001"] * 4,
        "feature_a": [1.0, 2.0, 3.0, 4.0],
        "feature_b": [1.0, 2.0, 4.0, 3.0],
    },
    index=["row-4", "row-1", "row-3", "row-2"],
)
output = predict(frame, object())
assert output.index.equals(frame.index)
assert output.columns.tolist() == ["prediction"]
assert not any(name == "agents" or name.startswith("agents.") for name in sys.modules)
'''
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            completed = subprocess.run(
                [sys.executable, "-I", "-c", script, str(pickle_path)],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )

    def test_frozen_bundle_assembler_rejects_wrong_bundle_lineage(self) -> None:
        cases = (
            "duplicate config",
            "wrong model seed",
            "wrong sample seed",
            "store mismatch",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                gate_path, bundle_dirs = _write_lineage_fixture(Path(tmp))
                if case == "duplicate config":
                    def mutate(provenance):
                        provenance["config"] = json.loads(
                            (bundle_dirs[0] / "provenance.json").read_text("utf-8")
                        )["config"]

                    _mutate_provenance(bundle_dirs[1], mutate)
                elif case == "wrong model seed":
                    _mutate_provenance(
                        bundle_dirs[1],
                        lambda provenance: provenance["training"].update(
                            {"model_seed": 999}
                        ),
                    )
                elif case == "wrong sample seed":
                    _mutate_provenance(
                        bundle_dirs[2],
                        lambda provenance: provenance["sample"].update(
                            {"seed": 999}
                        ),
                    )
                else:
                    _mutate_provenance(
                        bundle_dirs[2],
                        lambda provenance: provenance["store"].update(
                            {"generation_id": "1" * 32}
                        ),
                    )
                with self.assertRaises(ValueError):
                    build_frozen_ender20_rank_ensemble_from_bundles(
                        bundle_dirs=bundle_dirs,
                        gate_source_manifest_path=gate_path,
                    )

    def test_frozen_bundle_assembler_rejects_duplicate_weights_and_sample_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate_path, bundle_dirs = _write_lineage_fixture(
                Path(tmp), spec_indices=(0, 0, 2)
            )
            with self.assertRaisesRegex(ValueError, "three distinct models"):
                build_frozen_ender20_rank_ensemble_from_bundles(
                    bundle_dirs=bundle_dirs,
                    gate_source_manifest_path=gate_path,
                )

        with tempfile.TemporaryDirectory() as tmp:
            gate_path, bundle_dirs = _write_lineage_fixture(
                Path(tmp),
                positions=(
                    np.array([1, 3]),
                    np.array([1, 4]),
                    np.array([2, 4]),
                ),
            )
            with self.assertRaisesRegex(ValueError, "sample-seed-1337"):
                build_frozen_ender20_rank_ensemble_from_bundles(
                    bundle_dirs=bundle_dirs,
                    gate_source_manifest_path=gate_path,
                )

        with tempfile.TemporaryDirectory() as tmp:
            gate_path, bundle_dirs = _write_lineage_fixture(
                Path(tmp),
                positions=(
                    np.array([1, 3]),
                    np.array([1, 3]),
                    np.array([1, 3]),
                ),
            )
            with self.assertRaisesRegex(ValueError, "sample-seed-2027"):
                build_frozen_ender20_rank_ensemble_from_bundles(
                    bundle_dirs=bundle_dirs,
                    gate_source_manifest_path=gate_path,
                )

    def test_cloudpickle_loads_without_repo_or_ml_packages(self) -> None:
        try:
            import cloudpickle
        except ImportError:
            self.skipTest("cloudpickle is not installed")

        predictor = build_three_tabm_rank_ensemble_predictor(
            model_specs=_three_specs()
        )
        with tempfile.TemporaryDirectory() as tmp:
            pickle_path = Path(tmp) / "ensemble.pkl"
            with pickle_path.open("wb") as handle:
                cloudpickle.dump(predictor, handle)

            script = r'''
import sys

class BlockLocalAndMLPackages:
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("agents", "tabm", "torch", "rtdl_num_embeddings")
        if fullname in blocked or fullname.startswith(tuple(x + "." for x in blocked)):
            raise ModuleNotFoundError(f"blocked portability dependency: {fullname}")
        return None

sys.meta_path.insert(0, BlockLocalAndMLPackages())
import cloudpickle
import pandas as pd

with open(sys.argv[1], "rb") as handle:
    predict = cloudpickle.load(handle)
features = pd.DataFrame(
    {
        "era": ["0001", "0001", "0001", "0001"],
        "feature_a": [1.0, 2.0, 3.0, 4.0],
        "feature_b": [1.0, 2.0, 4.0, 3.0],
        "feature_c": [1.0, 3.0, 4.0, 2.0],
    },
    index=["a", "b", "c", "d"],
)
output = predict(features, object())
expected = [0.25, 7.0 / 12.0, 11.0 / 12.0, 0.75]
assert output.index.equals(features.index)
assert output.columns.tolist() == ["prediction"]
assert all(abs(x - y) < 1e-12 for x, y in zip(output["prediction"], expected))
assert not any(name == "agents" or name.startswith("agents.") for name in sys.modules)
'''
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            completed = subprocess.run(
                [sys.executable, "-I", "-c", script, str(pickle_path)],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
