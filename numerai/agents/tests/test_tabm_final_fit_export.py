from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from agents.code.modeling.deployment.final_fit_export import (
    FinalFitProtocol,
    GATE_CONFIG_RELATIVE_PREFIX,
    GATE_STORE_METADATA_RELATIVE_PATH,
    SUPPORTED_RUNS,
    _best_epoch,
    load_gate_source_manifest_pin,
    load_intermediate_predictor_spec,
    run_final_fit_export,
    write_intermediate_bundle,
)
from agents.code.modeling.utils.model_data import ModelDataBatch
from agents.code.modeling.utils.target_transforms import (
    TargetTransformWrapper,
    apply_target_transform,
)


FEATURES = ("feature_a", "feature_b")
TARGET = "target_ender_20"
BENCHMARK = "v53_lgbm_ender20"


def _feature_order_hash(names):
    payload = json.dumps(list(names), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _FakeDiskView:
    is_disk_feature_view = True

    def __init__(self, manifest: pd.DataFrame, positions: np.ndarray, x_cols):
        self.manifest = manifest
        self._positions = np.asarray(positions, dtype=np.int64)
        self.columns = pd.Index(x_cols)

    def __len__(self):
        return len(self._positions)

    @property
    def manifest_positions(self):
        return self._positions.copy()

    def take(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return _FakeDiskView(
            self.manifest, self._positions[indices], tuple(self.columns)
        )

    def __getitem__(self, key):
        if key not in self.columns:
            raise KeyError(key)
        return self.manifest[key].iloc[self._positions]


class _FakeLoader:
    def __init__(self, directory: Path, records: dict, **kwargs):
        self.directory = Path(directory)
        self.records = records
        self.records["loader_kwargs"] = kwargs
        self.feature_columns = FEATURES
        rows = 10
        self.manifest = pd.DataFrame(
            {
                "row_offset": np.arange(rows, dtype=np.int64),
                "id": [f"id-{index}" for index in range(rows)],
                "era": [f"{index // 2 + 1:04d}" for index in range(rows)],
                TARGET: np.linspace(0.05, 0.95, rows),
                BENCHMARK: np.array(
                    [0.2, 0.7, 0.1, 0.8, 0.3, 0.6, 0.4, 0.9, 0.15, 0.85]
                ),
            }
        )
        (self.directory / "metadata.json").write_text(
            '{"complete":true}', encoding="utf-8"
        )
        self.diagnostics = {
            "directory": str(self.directory),
            "feature_path": str(self.directory / "features.bin"),
            "manifest_path": str(self.directory / "manifest.parquet"),
            "generation_id": "0" * 32,
            "row_count": rows,
            "feature_count": len(FEATURES),
            "feature_bytes": rows * len(FEATURES),
            "manifest_bytes": 123,
            "feature_order_sha256": _feature_order_hash(FEATURES),
            "feature_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
        }
        self._x_cols = None
        self.closed = False

    @property
    def eras(self):
        return self.manifest["era"]

    def configure_x_cols(self, x_cols):
        self._x_cols = tuple(x_cols)
        self.records["x_cols"] = self._x_cols

    def load(self, eras):
        self.records["loaded_eras"] = tuple(eras)
        positions = np.flatnonzero(self.manifest["era"].isin(eras).to_numpy())
        return ModelDataBatch(
            X=_FakeDiskView(self.manifest, positions, self._x_cols),
            y=self.manifest[TARGET].iloc[positions],
            era=self.manifest["era"].iloc[positions],
            id=self.manifest["id"].iloc[positions],
        )

    def close(self):
        self.closed = True
        self.records["loader_closed"] = True


class _RecordingRegressor:
    def __init__(self, role, params, feature_cols, records):
        self.role = role
        self.params = dict(params)
        self.feature_cols = tuple(feature_cols)
        self.records = records
        self.val_split = params["val_split"]
        self.val_fraction = params["val_fraction"]
        self.max_epochs = params["max_epochs"]
        self.best_epoch_ = 3 if role == "selector" else self.max_epochs
        self.training_history_ = []

    def fit(self, X, y, **kwargs):
        self.records["fits"].append(
            {
                "role": self.role,
                "x_id": id(X),
                "positions": X.manifest_positions,
                "target": np.asarray(y, dtype=np.float64),
                "params": dict(self.params),
            }
        )
        epochs = self.max_epochs if self.role == "fixed" else self.best_epoch_
        self.training_history_ = [
            {"epoch": epoch, "train_loss": 1.0, "val_loss": 1.0}
            for epoch in range(1, epochs + 1)
        ]
        return self


def _export_spec(model, *, batch_size, era_column, prediction_column):
    if not isinstance(model, TargetTransformWrapper):
        raise AssertionError("The fixed pass lost its residual target wrapper.")
    feature_names = tuple(model.feature_cols)
    return {
        "feature_names": feature_names,
        "blocks": [
            {
                "weight": np.eye(2, dtype=np.float32),
                "r": np.ones((1, 2), dtype=np.float32),
                "s": np.ones((1, 2), dtype=np.float32),
                "bias": np.zeros((1, 2), dtype=np.float32),
            }
        ],
        "output_weight": np.ones((1, 2, 1), dtype=np.float32),
        "output_bias": np.zeros((1, 1), dtype=np.float32),
        "feature_center": 2.0,
        "feature_scale": 2.0,
        "batch_size": batch_size,
        "activation": "relu",
        "era_column": era_column,
        "prediction_column": prediction_column,
    }


def _config(model_seed: int, sample_seed: int, sample_size: int, protocol):
    return {
        "data": {
            "data_version": "v5.3",
            "feature_set": "all",
            "target_col": TARGET,
            "era_col": "era",
            "id_col": "id",
            "benchmark_model": BENCHMARK,
            "require_benchmark_coverage": True,
            "embargo_eras": protocol.internal_val_embargo,
            "disk_feature_store_path": "unused-in-test",
        },
        "model": {
            "type": "TorchTabularRegressor",
            "x_groups": ["features", "era", "benchmark_models"],
            "target_transform": {
                "type": "residual_to_benchmark",
                "benchmark_col": BENCHMARK,
                "era_col": "era",
                "per_era": True,
                "fit_intercept": True,
            },
            "params": {
                "architecture": "tabm",
                "activation": "relu",
                "tabm_arch_type": "tabm",
                "tabm_k": 64,
                "tabm_width": 512,
                "tabm_blocks": 3,
                "dropout": 0.1,
                "batch_size": 1024,
                "prediction_batch_size": 2048,
                "learning_rate": 0.002,
                "weight_decay": 0.0003,
                "max_epochs": protocol.selector_max_epochs,
                "patience": protocol.selector_patience,
                "val_fraction": protocol.selector_val_fraction,
                "val_split": "recent_eras",
                "internal_val_embargo": protocol.internal_val_embargo,
                "feature_center": 2.0,
                "feature_scale": 2.0,
                "device": "cuda",
                "amp": True,
                "seed": model_seed,
            },
        },
        "preprocessing": {
            "missing_value": 2.0,
            "nan_missing_all_twos": False,
        },
        "training": {
            "max_train_samples": sample_size,
            "sample_seed": sample_seed,
            "data_mode": "disk_feature_store",
            "cv": {
                "embargo": protocol.internal_val_embargo,
                "enabled": True,
                "mode": "expanding",
                "n_splits": 5,
            },
        },
    }


class FinalFitExportTests(unittest.TestCase):
    def setUp(self):
        self.protocol = FinalFitProtocol(
            sample_size=6,
            selector_max_epochs=5,
            selector_patience=2,
            selector_val_fraction=0.25,
            internal_val_embargo=1,
            predictor_batch_size=3,
        )

    def _run(
        self,
        root: Path,
        config_name: str,
        *,
        gate_store_sha256: str | None = None,
    ):
        model_seed, sample_seed = SUPPORTED_RUNS[config_name]
        config_path = root / config_name
        config_path.write_text(
            "CONFIG = " + repr(_config(model_seed, sample_seed, 6, self.protocol)),
            encoding="utf-8",
        )
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        metadata_bytes = b'{"complete":true}'
        actual_metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
        gate_path = root / "gate_source_manifest.json"
        gate_path.write_text(
            json.dumps(
                {
                    "hash_algorithm": "sha256",
                    "files": {
                        GATE_STORE_METADATA_RELATIVE_PATH: (
                            gate_store_sha256 or actual_metadata_sha256
                        ),
                        f"{GATE_CONFIG_RELATIVE_PREFIX}/{config_name}": config_sha256,
                    },
                }
            ),
            encoding="utf-8",
        )
        store_dir = root / "store"
        store_dir.mkdir()
        records = {"fits": [], "builder_calls": []}
        loader_holder = {}
        self._last_records = records
        self._last_loader_holder = loader_holder

        def loader_factory(_path, **kwargs):
            loader = _FakeLoader(store_dir, records, **kwargs)
            loader_holder["loader"] = loader
            return loader

        def builder(model_type, params, model_config, *, feature_cols):
            self.assertEqual(model_type, "TorchTabularRegressor")
            self.assertEqual(
                model_config["target_transform"]["type"],
                "residual_to_benchmark",
            )
            role = "selector" if not records["builder_calls"] else "fixed"
            records["builder_calls"].append(
                {"role": role, "params": dict(params), "model_config": model_config}
            )
            inner = _RecordingRegressor(role, params, feature_cols, records)
            return TargetTransformWrapper(inner, model_config["target_transform"])

        output = root / "bundle"
        result = run_final_fit_export(
            config_path,
            output,
            gate_source_manifest_path=gate_path,
            protocol=self.protocol,
            loader_factory=loader_factory,
            feature_order_loader=lambda _config, _path: FEATURES,
            model_builder=builder,
            spec_exporter=_export_spec,
            utcnow=lambda: datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc),
        )
        return result, records, loader_holder["loader"]

    def test_all_frozen_configs_propagate_seeds_and_reuse_one_sample(self):
        for config_name, (model_seed, sample_seed) in SUPPORTED_RUNS.items():
            with self.subTest(config_name=config_name), tempfile.TemporaryDirectory() as tmp:
                result, records, loader = self._run(Path(tmp), config_name)
                self.assertTrue(loader.closed)
                self.assertEqual(
                    records["loaded_eras"],
                    ("0001", "0002", "0003", "0004", "0005"),
                )
                self.assertEqual(len(records["fits"]), 2)
                selector_fit, fixed_fit = records["fits"]
                self.assertEqual(selector_fit["x_id"], fixed_fit["x_id"])
                np.testing.assert_array_equal(
                    selector_fit["positions"], fixed_fit["positions"]
                )
                np.testing.assert_allclose(
                    selector_fit["target"], fixed_fit["target"]
                )

                sampled_view = _FakeDiskView(
                    loader.manifest,
                    selector_fit["positions"],
                    records["x_cols"],
                )
                sampled_y = loader.manifest[TARGET].iloc[selector_fit["positions"]]
                expected_residual = apply_target_transform(
                    sampled_y,
                    sampled_view,
                    records["builder_calls"][0]["model_config"]["target_transform"],
                )
                np.testing.assert_allclose(
                    selector_fit["target"], expected_residual.to_numpy()
                )

                for call in records["builder_calls"]:
                    self.assertEqual(call["params"]["seed"], model_seed)
                    self.assertEqual(
                        call["model_config"]["target_transform"]["type"],
                        "residual_to_benchmark",
                    )
                fixed_params = records["builder_calls"][1]["params"]
                self.assertEqual(fixed_params["val_split"], "none")
                self.assertEqual(fixed_params["val_fraction"], 0.0)
                self.assertEqual(fixed_params["max_epochs"], 3)

                positions = np.load(result.sample_positions_path, allow_pickle=False)
                np.testing.assert_array_equal(positions, selector_fit["positions"])
                digest = hashlib.sha256(
                    np.ascontiguousarray(positions, dtype="<i8").tobytes()
                ).hexdigest()
                self.assertEqual(digest, result.sample_positions_sha256)

                provenance = json.loads(result.provenance_path.read_text("utf-8"))
                self.assertEqual(provenance["sample"]["seed"], sample_seed)
                self.assertEqual(provenance["training"]["model_seed"], model_seed)
                self.assertEqual(provenance["training"]["selected_best_epoch"], 3)
                self.assertEqual(
                    provenance["training"]["fixed_pass"][
                        "same_sample_manifest_positions_sha256"
                    ],
                    digest,
                )
                self.assertTrue(provenance["store"]["consecutive_eras"])
                self.assertEqual(
                    provenance["artifact_state"],
                    "intermediates_only_no_pickle_no_upload",
                )
                self.assertEqual(
                    provenance["gate_source"]["manifest_sha256"],
                    hashlib.sha256(
                        Path(provenance["gate_source"]["manifest_path"]).read_bytes()
                    ).hexdigest(),
                )
                self.assertEqual(
                    provenance["gate_source"]["expected_store_metadata_sha256"],
                    provenance["store"]["metadata_sha256"],
                )
                self.assertEqual(
                    provenance["gate_source"]["expected_config_sha256"],
                    provenance["config"]["file_sha256"],
                )
                self.assertFalse(provenance["export"]["pickle_created"])
                self.assertFalse(provenance["export"]["upload_performed"])
                self.assertEqual(
                    provenance["export"]["intended_rebuild_runtime"][
                        "python_major_minor"
                    ],
                    "3.12",
                )
                self.assertEqual(
                    set(provenance["intermediates"]),
                    {"weights", "predictor_spec", "sample_manifest_positions"},
                )

                rebuilt = load_intermediate_predictor_spec(result.output_dir)
                self.assertEqual(rebuilt["feature_names"], FEATURES)
                self.assertEqual(rebuilt["batch_size"], 3)
                np.testing.assert_array_equal(
                    rebuilt["blocks"][0]["weight"], np.eye(2, dtype=np.float32)
                )

    def test_bundle_publish_failure_leaves_no_partial_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "bundle"
            arrays = {"output_bias": np.zeros((1, 1), dtype=np.float32)}
            positions = np.array([1, 3], dtype=np.int64)
            positions_hash = hashlib.sha256(
                np.ascontiguousarray(positions, dtype="<i8").tobytes()
            ).hexdigest()
            provenance = {
                "training": {"selected_best_epoch": 2},
                "sample": {"manifest_positions_sha256": positions_hash},
            }

            def fail_replace(_source, _destination):
                raise OSError("injected rename failure")

            with self.assertRaisesRegex(OSError, "injected rename failure"):
                write_intermediate_bundle(
                    destination,
                    predictor_spec={"format": "test"},
                    arrays=arrays,
                    sample_manifest_positions=positions,
                    provenance=provenance,
                    replace=fail_replace,
                )
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".bundle.tmp-*")), [])

    def test_selector_epoch_must_be_inside_frozen_range(self):
        class Selector:
            pass

        selector = Selector()
        for invalid in (None, False, 0, 6, 2.5):
            with self.subTest(invalid=invalid):
                selector.best_epoch_ = invalid
                with self.assertRaises(ValueError):
                    _best_epoch(selector, 5)
        selector.best_epoch_ = 5
        self.assertEqual(_best_epoch(selector, 5), 5)

    def test_gate_store_mismatch_fails_before_model_build_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(
                ValueError, "feature-store metadata SHA-256 pinned by the gate"
            ):
                self._run(
                    root,
                    "scale_disk_tabm_k64_train500k.py",
                    gate_store_sha256="f" * 64,
                )
            self.assertEqual(self._last_records["builder_calls"], [])
            self.assertTrue(self._last_loader_holder["loader"].closed)
            self.assertFalse((root / "bundle").exists())

    def test_gate_manifest_missing_or_malformed_store_pin_fails_closed(self):
        cases = {
            "missing files": {"hash_algorithm": "sha256"},
            "missing store": {
                "hash_algorithm": "sha256",
                "files": {"some/file": "a" * 64},
            },
            "malformed digest": {
                "hash_algorithm": "sha256",
                "files": {GATE_STORE_METADATA_RELATIVE_PATH: "not-a-digest"},
            },
        }
        for name, payload in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "gate.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_gate_source_manifest_pin(path)


if __name__ == "__main__":
    unittest.main()
