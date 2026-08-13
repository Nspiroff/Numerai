from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from agents.code.data.build_full_datasets import (
    build_disk_feature_store,
    feature_order_sha256,
    validate_disk_feature_store,
)


TARGET = "target_ender_20"
BENCHMARK = "v53_lgbm_ender20"
FEATURES = ["feature_a", "feature_b"]


def _write_fixture(
    root: Path, *, train_b_era: str = "0002"
) -> tuple[list[Path], list[Path]]:
    train_path = root / "train.parquet"
    validation_path = root / "validation.parquet"
    train_benchmark_path = root / "train_benchmark_models.parquet"
    validation_benchmark_path = root / "validation_benchmark_models.parquet"

    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "era": ["0001", "0002", "0003", "0004"],
            "data_type": ["train"] * 4,
            "feature_a": np.array([0, 1, 2, 3], dtype=np.int8),
            "feature_b": np.array([9, 8, 7, 6], dtype=np.int8),
            TARGET: np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        }
    ).set_index("id").to_parquet(train_path)
    pd.DataFrame(
        {
            "id": ["e", "live", "f"],
            "era": ["0005", "0006", "0007"],
            "data_type": ["validation", "live", "validation"],
            "feature_a": np.array([4, 5, 6], dtype=np.int8),
            "feature_b": np.array([5, 4, 3], dtype=np.int8),
            TARGET: np.array([0.5, np.nan, 0.7], dtype=np.float32),
        }
    ).set_index("id").to_parquet(validation_path)
    pd.DataFrame(
        {
            "id": ["b", "d"],
            "era": [train_b_era, "0004"],
            BENCHMARK: np.array([0.11, 0.22], dtype=np.float64),
        }
    ).set_index("id").to_parquet(train_benchmark_path)
    pd.DataFrame(
        {
            "id": ["e", "live"],
            "era": ["0005", "0006"],
            BENCHMARK: np.array([0.33, 0.44], dtype=np.float64),
        }
    ).set_index("id").to_parquet(validation_benchmark_path)
    return (
        [train_path, validation_path],
        [train_benchmark_path, validation_benchmark_path],
    )


class TestDiskFeatureStoreBuilder(unittest.TestCase):
    def test_writes_source_ordered_memmap_and_thin_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_paths, benchmark_paths = _write_fixture(root)
            output = root / "store"

            store = build_disk_feature_store(
                output,
                data_paths,
                benchmark_paths,
                FEATURES,
                batch_size=2,
            )

            self.assertFalse(store.reused)
            self.assertEqual(store.row_count, 3)
            self.assertEqual(store.feature_count, 2)
            feature_values = np.memmap(
                store.feature_path,
                dtype=np.int8,
                mode="r",
                shape=(store.row_count, store.feature_count),
            )
            np.testing.assert_array_equal(
                feature_values,
                np.array([[1, 8], [3, 6], [4, 5]], dtype=np.int8),
            )
            feature_values._mmap.close()

            manifest = pd.read_parquet(store.manifest_path)
            self.assertEqual(
                manifest.columns.tolist(),
                ["row_offset", "id", "era", TARGET, BENCHMARK],
            )
            self.assertEqual(manifest["row_offset"].tolist(), [0, 1, 2])
            self.assertEqual(manifest["id"].tolist(), ["b", "d", "e"])
            self.assertEqual(manifest["era"].tolist(), ["0002", "0004", "0005"])
            np.testing.assert_allclose(
                manifest[BENCHMARK].to_numpy(), [0.11, 0.22, 0.33]
            )

            metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
            expected_feature_hash = hashlib.sha256(
                b'["feature_a","feature_b"]'
            ).hexdigest()
            self.assertEqual(feature_order_sha256(FEATURES), expected_feature_hash)
            self.assertEqual(metadata["feature_order_sha256"], expected_feature_hash)
            self.assertEqual(
                [
                    (item["role"], item["position"])
                    for item in metadata["source_fingerprints"]
                ],
                [("data", 0), ("data", 1), ("benchmark", 0), ("benchmark", 1)],
            )
            self.assertTrue(
                validate_disk_feature_store(
                    output,
                    data_paths,
                    benchmark_paths,
                    FEATURES,
                    verify_artifact_hashes=True,
                )
            )
            self.assertFalse(
                any(path.name.endswith(".tmp") for path in output.iterdir())
            )

    def test_reuses_only_matching_source_and_feature_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_paths, benchmark_paths = _write_fixture(root)
            output = root / "store"
            first = build_disk_feature_store(
                output,
                data_paths,
                benchmark_paths,
                FEATURES,
                batch_size=2,
            )
            retired_id = uuid.uuid4().hex
            retired_feature = output / f"features-{retired_id}.int8.bin"
            retired_manifest = output / f"manifest-{retired_id}.parquet"
            retired_feature.write_bytes(b"retired")
            retired_manifest.write_bytes(b"retired")
            metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
            metadata["retired_generation_ids"] = [retired_id]
            first.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            second = build_disk_feature_store(
                output,
                data_paths,
                benchmark_paths,
                FEATURES,
                batch_size=1,
            )
            self.assertTrue(second.reused)
            self.assertEqual(second.generation_id, first.generation_id)
            self.assertEqual(second.feature_path, first.feature_path)
            self.assertFalse(retired_feature.exists())
            self.assertFalse(retired_manifest.exists())
            self.assertFalse(
                validate_disk_feature_store(
                    output,
                    data_paths,
                    benchmark_paths,
                    list(reversed(FEATURES)),
                )
            )

            train_benchmark = pd.read_parquet(benchmark_paths[0])
            train_benchmark.loc["b", BENCHMARK] = 0.99
            train_benchmark.to_parquet(benchmark_paths[0])
            self.assertFalse(
                validate_disk_feature_store(
                    output,
                    data_paths,
                    benchmark_paths,
                    FEATURES,
                )
            )
            rebuilt = build_disk_feature_store(
                output,
                data_paths,
                benchmark_paths,
                FEATURES,
                batch_size=2,
            )
            self.assertFalse(rebuilt.reused)
            self.assertNotEqual(rebuilt.generation_id, first.generation_id)
            rebuilt_metadata = json.loads(
                rebuilt.metadata_path.read_text(encoding="utf-8")
            )
            self.assertIn(
                first.generation_id,
                rebuilt_metadata["retired_generation_ids"],
            )
            self.assertFalse(first.feature_path.exists())
            self.assertFalse(first.manifest_path.exists())
            rebuilt_manifest = pd.read_parquet(rebuilt.manifest_path)
            self.assertAlmostEqual(rebuilt_manifest.loc[0, BENCHMARK], 0.99)

    def test_failed_rebuild_preserves_previous_commit_and_cleans_temp_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_paths, benchmark_paths = _write_fixture(root)
            output = root / "store"
            committed = build_disk_feature_store(
                output,
                data_paths,
                benchmark_paths,
                FEATURES,
                batch_size=2,
            )
            committed_metadata = committed.metadata_path.read_bytes()

            _write_fixture(root, train_b_era="9999")
            with self.assertRaisesRegex(ValueError, "Era mismatch"):
                build_disk_feature_store(
                    output,
                    data_paths,
                    benchmark_paths,
                    FEATURES,
                    batch_size=2,
                    reuse_existing=False,
                )

            self.assertEqual(committed.metadata_path.read_bytes(), committed_metadata)
            self.assertTrue(committed.feature_path.is_file())
            self.assertTrue(committed.manifest_path.is_file())
            self.assertFalse(
                any(path.name.endswith(".tmp") for path in output.iterdir())
            )

    def test_corrupt_metadata_cannot_nominate_unrelated_file_for_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_paths, benchmark_paths = _write_fixture(root)
            output = root / "store"
            committed = build_disk_feature_store(
                output,
                data_paths,
                benchmark_paths,
                FEATURES,
                batch_size=2,
            )
            sentinel = output / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")
            metadata = json.loads(committed.metadata_path.read_text(encoding="utf-8"))
            metadata["features"]["filename"] = sentinel.name
            committed.metadata_path.write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            rebuilt = build_disk_feature_store(
                output,
                data_paths,
                benchmark_paths,
                FEATURES,
                batch_size=2,
            )

            self.assertFalse(rebuilt.reused)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
