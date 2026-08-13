from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from agents.code.data import build_full_datasets as builder


class _NoDownloadNumerAPI:
    def download_dataset(self, *args, **kwargs):
        raise AssertionError(
            "The temp-dir fixture should satisfy every source request."
        )


class TestStreamingDownsampleBuilder(unittest.TestCase):
    data_version = "vtest"

    def _write_sources(self, root: Path) -> Path:
        dataset_dir = root / self.data_version
        dataset_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "id": ["a", "b", "c", "d"],
                "era": ["0001", "0002", "0003", "0004"],
                "data_type": ["train"] * 4,
                "feature_x": pd.Series([0, 1, 2, 3], dtype="int8"),
                "target_ender_20": [0.0, 0.25, 0.5, 0.75],
            }
        ).set_index("id").to_parquet(dataset_dir / "train.parquet")
        pd.DataFrame(
            {
                "id": ["e", "f", "live"],
                "era": ["0005", "0006", "0007"],
                "data_type": ["validation", "validation", "live"],
                "feature_x": pd.Series([4, 3, 2], dtype="int8"),
                "target_ender_20": [1.0, 0.75, None],
            }
        ).set_index("id").to_parquet(dataset_dir / "validation.parquet")
        pd.DataFrame(
            {
                "id": ["a", "b", "c", "d"],
                "era": ["0001", "0002", "0003", "0004"],
                "prediction": [0.1, 0.2, 0.3, 0.4],
            }
        ).set_index("id").to_parquet(
            dataset_dir / "train_benchmark_models.parquet"
        )
        pd.DataFrame(
            {
                "id": ["e", "f", "live"],
                "era": ["0005", "0006", "0007"],
                "prediction": [0.5, 0.6, 0.7],
            }
        ).set_index("id").to_parquet(
            dataset_dir / "validation_benchmark_models.parquet"
        )
        return dataset_dir

    def _build(
        self,
        root: Path,
        *,
        era_step: int = 2,
        era_offset: int = 0,
        reuse_existing: bool = True,
    ) -> tuple[Path, Path]:
        with mock.patch.object(builder, "NUMERAI_DIR", root):
            return builder.build_downsampled_direct(
                _NoDownloadNumerAPI(),
                self.data_version,
                era_step,
                era_offset,
                reuse_existing=reuse_existing,
            )

    def _pair_bytes(self, dataset_dir: Path) -> dict[str, bytes]:
        names = (
            builder.DOWNSAMPLE_DATA_FILENAME,
            builder.DOWNSAMPLE_BENCHMARK_FILENAME,
            builder.DOWNSAMPLE_PAIR_MANIFEST_FILENAME,
        )
        return {name: (dataset_dir / name).read_bytes() for name in names}

    def _assert_no_transaction_files(self, dataset_dir: Path) -> None:
        leftovers = [
            path.name
            for path in dataset_dir.iterdir()
            if ".staging-" in path.name
            or ".backup-" in path.name
            or path.name.endswith(".temp")
        ]
        self.assertEqual(leftovers, [])

    def test_filters_eras_and_validation_rows_without_full_materialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = self._write_sources(root)
            output_path = dataset_dir / "downsampled.parquet"

            keep_eras = builder.select_downsample_eras(
                dataset_dir / "train.parquet",
                dataset_dir / "validation.parquet",
                era_step=2,
                era_offset=0,
            )
            self.assertEqual(keep_eras, ["0001", "0003", "0005"])
            builder.write_filtered_parquets(
                [
                    dataset_dir / "train.parquet",
                    dataset_dir / "validation.parquet",
                ],
                output_path,
                keep_eras,
                validation_data=True,
                drop_data_type=True,
                batch_size=2,
            )

            result = pd.read_parquet(output_path)
            self.assertTrue(builder.is_valid_parquet(output_path))
            self.assertFalse((dataset_dir / "downsampled.parquet.temp").exists())
            self.assertEqual(result["id"].tolist(), ["a", "c", "e"])
            self.assertNotIn("data_type", result.columns)
            self.assertEqual(str(result["feature_x"].dtype), "int8")

            broken_path = dataset_dir / "broken.parquet"
            broken_path.write_bytes(b"PAR1")
            self.assertFalse(builder.is_valid_parquet(broken_path))

    def test_invalid_args_are_rejected_before_any_cache_or_source_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(builder, "NUMERAI_DIR", root), mock.patch.object(
                builder, "ensure_source_datasets"
            ) as ensure_sources:
                for era_step, era_offset in ((1, 0), (2, -1), (2, 2)):
                    with self.subTest(era_step=era_step, era_offset=era_offset):
                        with self.assertRaises(ValueError):
                            builder.build_downsampled_direct(
                                _NoDownloadNumerAPI(),
                                self.data_version,
                                era_step,
                                era_offset,
                            )
                ensure_sources.assert_not_called()

    def test_manifest_reuses_only_the_complete_matching_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = self._write_sources(root)
            data_path, benchmark_path = self._build(root)
            before = self._pair_bytes(dataset_dir)

            real_writer = builder.write_filtered_parquets
            with mock.patch.object(
                builder, "write_filtered_parquets", wraps=real_writer
            ) as writer:
                self.assertEqual(self._build(root), (data_path, benchmark_path))
                writer.assert_not_called()

            self.assertEqual(self._pair_bytes(dataset_dir), before)
            with (
                dataset_dir / builder.DOWNSAMPLE_PAIR_MANIFEST_FILENAME
            ).open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            self.assertEqual(manifest["format"], builder.DOWNSAMPLE_PAIR_FORMAT)
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["keep_eras"], ["0001", "0003", "0005"])
            self.assertEqual(len(manifest["source_fingerprints"]), 4)
            self.assertEqual(
                set(manifest["outputs"]), {"data", "benchmark"}
            )

    def test_changed_args_and_changed_source_each_rebuild_both_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = self._write_sources(root)
            self._build(root)
            real_writer = builder.write_filtered_parquets

            with mock.patch.object(
                builder, "write_filtered_parquets", wraps=real_writer
            ) as writer:
                self._build(root, era_offset=1)
                self.assertEqual(writer.call_count, 2)

            benchmark_source = dataset_dir / "train_benchmark_models.parquet"
            benchmark = pd.read_parquet(benchmark_source)
            benchmark.loc["b", "prediction"] = 0.91
            benchmark.to_parquet(benchmark_source)
            with mock.patch.object(
                builder, "write_filtered_parquets", wraps=real_writer
            ) as writer:
                self._build(root, era_offset=1)
                self.assertEqual(writer.call_count, 2)

    def test_missing_or_corrupt_member_rebuilds_both_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_sources(root)
            data_path, benchmark_path = self._build(root)
            real_writer = builder.write_filtered_parquets

            data_path.unlink()
            with mock.patch.object(
                builder, "write_filtered_parquets", wraps=real_writer
            ) as writer:
                self._build(root)
                self.assertEqual(writer.call_count, 2)
            self.assertTrue(builder.is_valid_parquet(data_path))

            benchmark_path.write_bytes(b"not a parquet file")
            with mock.patch.object(
                builder, "write_filtered_parquets", wraps=real_writer
            ) as writer:
                self._build(root)
                self.assertEqual(writer.call_count, 2)
            self.assertTrue(builder.is_valid_parquet(benchmark_path))

    def test_second_artifact_build_failure_preserves_prior_pair_and_cleans_temps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = self._write_sources(root)
            self._build(root)
            before = self._pair_bytes(dataset_dir)
            real_writer = builder.write_filtered_parquets
            call_count = 0

            def fail_second_write(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise RuntimeError("injected benchmark build failure")
                return real_writer(*args, **kwargs)

            with mock.patch.object(
                builder,
                "write_filtered_parquets",
                side_effect=fail_second_write,
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    self._build(root, reuse_existing=False)

            self.assertEqual(self._pair_bytes(dataset_dir), before)
            self._assert_no_transaction_files(dataset_dir)

    def test_publish_failure_rolls_back_prior_pair_and_cleans_temps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = self._write_sources(root)
            _, benchmark_path = self._build(root)
            before = self._pair_bytes(dataset_dir)
            real_replace = builder.os.replace

            def fail_benchmark_publish(source, destination):
                source_path = Path(source)
                if (
                    Path(destination) == benchmark_path
                    and ".staging-" in source_path.name
                ):
                    raise OSError("injected publish failure")
                return real_replace(source, destination)

            with mock.patch.object(
                builder.os, "replace", side_effect=fail_benchmark_publish
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    self._build(root, reuse_existing=False)

            self.assertEqual(self._pair_bytes(dataset_dir), before)
            self._assert_no_transaction_files(dataset_dir)

    def test_writer_error_removes_its_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = self._write_sources(root)
            broken_path = dataset_dir / "broken_source.parquet"
            broken_path.write_bytes(b"not parquet")
            output_path = dataset_dir / "failed.parquet"

            with self.assertRaises(Exception):
                builder.write_filtered_parquets(
                    [dataset_dir / "train.parquet", broken_path],
                    output_path,
                    ["0001"],
                    validation_data=True,
                    drop_data_type=True,
                )
            self.assertFalse(output_path.exists())
            self.assertFalse((dataset_dir / "failed.parquet.temp").exists())


if __name__ == "__main__":
    unittest.main()
