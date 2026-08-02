"""Build full dataset and benchmark parquet files for Numerai."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numerapi import NumerAPI

from agents.code.modeling.utils.constants import NUMERAI_DIR


FEATURE_STORE_FORMAT = "numerai-v5.3-int8-feature-store"
FEATURE_STORE_FORMAT_VERSION = 1
FEATURE_STORE_METADATA_FILENAME = "metadata.json"

DOWNSAMPLE_PAIR_FORMAT = "numerai-downsampled-parquet-pair"
DOWNSAMPLE_PAIR_FORMAT_VERSION = 1
DOWNSAMPLE_PAIR_MANIFEST_FILENAME = "downsampled_full_pair_manifest.json"
DOWNSAMPLE_DATA_FILENAME = "downsampled_full.parquet"
DOWNSAMPLE_BENCHMARK_FILENAME = "downsampled_full_benchmark_models.parquet"


@dataclass(frozen=True)
class DiskFeatureStore:
    """Paths and shape for one committed feature-store generation."""

    directory: Path
    feature_path: Path
    manifest_path: Path
    metadata_path: Path
    row_count: int
    feature_count: int
    generation_id: str
    reused: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full.parquet and full_benchmark_models.parquet."
    )
    parser.add_argument(
        "--data-version",
        type=str,
        default="v5.3",
        help="Numerai data version (default: v5.3).",
    )
    parser.add_argument(
        "--downsample-eras-step",
        type=int,
        default=4,
        help="Keep every Nth era when building downsampled_full (default: 4).",
    )
    parser.add_argument(
        "--downsample-eras-offset",
        type=int,
        default=0,
        help="Offset when selecting every Nth era (default: 0).",
    )
    parser.add_argument(
        "--skip-downsample",
        action="store_true",
        help="Skip building downsampled_full datasets.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild full datasets even if they already exist.",
    )
    parser.add_argument(
        "--downsample-only",
        action="store_true",
        help=(
            "Build downsampled files directly from source parquets without "
            "materializing full.parquet. This is the memory-safe scout path."
        ),
    )
    return parser.parse_args()


def ensure_source_datasets(napi: NumerAPI, data_version: str) -> dict[str, Path]:
    names = (
        "train.parquet",
        "validation.parquet",
        "train_benchmark_models.parquet",
        "validation_benchmark_models.parquet",
    )
    paths = {}
    for name in names:
        path = (NUMERAI_DIR / data_version / name).resolve()
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            napi.download_dataset(f"{data_version}/{name}", dest_path=str(path))
        paths[name] = path
    return paths


def feature_order_sha256(feature_columns: Sequence[str]) -> str:
    """Return a stable digest that changes when feature names or order change."""

    encoded = json.dumps(
        list(feature_columns), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _parquet_footer_sha256(path: Path) -> str:
    """Hash only the Parquet footer, which captures schema and row-group metadata."""

    size = path.stat().st_size
    if size < 8:
        raise ValueError(f"{path} is too small to be a parquet file.")
    with path.open("rb") as stream:
        stream.seek(-8, os.SEEK_END)
        trailer = stream.read(8)
        if len(trailer) != 8 or trailer[4:] != b"PAR1":
            raise ValueError(f"{path} has an invalid parquet trailer.")
        footer_size = struct.unpack("<I", trailer[:4])[0]
        footer_start = size - footer_size - 8
        if footer_start < 4:
            raise ValueError(f"{path} has an invalid parquet footer length.")
        stream.seek(footer_start)
        footer = stream.read(footer_size + 8)
    if len(footer) != footer_size + 8:
        raise ValueError(f"{path} has a truncated parquet footer.")
    return hashlib.sha256(footer).hexdigest()


def parquet_source_fingerprint(path: Path) -> dict[str, object]:
    """Create a cheap but source-specific fingerprint for reuse decisions."""

    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    parquet = pq.ParquetFile(resolved)
    try:
        schema_bytes = parquet.schema_arrow.serialize().to_pybytes()
        num_rows = parquet.metadata.num_rows
        num_row_groups = parquet.metadata.num_row_groups
    finally:
        parquet.close()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "num_rows": num_rows,
        "num_row_groups": num_row_groups,
        "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "footer_sha256": _parquet_footer_sha256(resolved),
    }


def _source_fingerprints(
    data_paths: Sequence[Path], benchmark_paths: Sequence[Path]
) -> list[dict[str, object]]:
    fingerprints: list[dict[str, object]] = []
    for role, paths in (("data", data_paths), ("benchmark", benchmark_paths)):
        for position, path in enumerate(paths):
            fingerprint = parquet_source_fingerprint(path)
            fingerprints.append(
                {"role": role, "position": position, **fingerprint}
            )
    return fingerprints


def _safe_artifact_path(directory: Path, filename: object) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError("Feature-store artifact filename must be a non-empty string.")
    candidate = Path(filename)
    if (
        filename in {".", ".."}
        or candidate.is_absolute()
        or candidate.name != filename
    ):
        raise ValueError("Feature-store artifact filename must not contain a path.")
    return directory / filename


def _read_store_metadata(directory: Path) -> dict[str, object]:
    metadata_path = directory / FEATURE_STORE_METADATA_FILENAME
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError("Feature-store metadata must be a JSON object.")
    return metadata


def _retirement_journal(
    directory: Path, *, include_current: bool
) -> list[str]:
    """Read only canonical generation ids from trusted committed metadata."""

    try:
        metadata = _read_store_metadata(directory)
        if metadata.get("format") != FEATURE_STORE_FORMAT:
            return []
        if metadata.get("format_version") != FEATURE_STORE_FORMAT_VERSION:
            return []
        if metadata.get("complete") is not True:
            return []
        generation_id = metadata["generation_id"]
        if not isinstance(generation_id, str):
            return []
        if uuid.UUID(hex=generation_id).hex != generation_id:
            return []
        features = metadata["features"]
        manifest = metadata["manifest"]
        if not isinstance(features, dict) or not isinstance(manifest, dict):
            return []
        if features.get("filename") != f"features-{generation_id}.int8.bin":
            return []
        if manifest.get("filename") != f"manifest-{generation_id}.parquet":
            return []

        retired = metadata.get("retired_generation_ids", [])
        if not isinstance(retired, list):
            return []
        validated: list[str] = []
        for value in retired:
            if not isinstance(value, str) or uuid.UUID(hex=value).hex != value:
                return []
            if value != generation_id and value not in validated:
                validated.append(value)
        if include_current and generation_id not in validated:
            validated.append(generation_id)
        return validated
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ):
        return []


def _cleanup_retired_generations(
    directory: Path,
    generation_ids: Sequence[str],
    *,
    active_generation_id: str,
) -> None:
    """Best-effort cleanup derived from exact, canonical generation ids."""

    for generation_id in dict.fromkeys(generation_ids):
        if generation_id == active_generation_id:
            continue
        try:
            if uuid.UUID(hex=generation_id).hex != generation_id:
                continue
        except (TypeError, ValueError):
            continue
        for filename in (
            f"features-{generation_id}.int8.bin",
            f"manifest-{generation_id}.parquet",
        ):
            try:
                (directory / filename).unlink(missing_ok=True)
            except OSError:
                # An active Windows reader can keep a retired generation open.
                pass


def validate_disk_feature_store(
    output_dir: Path,
    data_paths: Sequence[Path],
    benchmark_paths: Sequence[Path],
    feature_columns: Sequence[str],
    *,
    target_column: str = "target_ender_20",
    benchmark_column: str = "v53_lgbm_ender20",
    verify_artifact_hashes: bool = False,
) -> bool:
    """Return whether a committed store exactly matches its requested inputs.

    Normal reuse validates immutable generation filenames, source fingerprints,
    schema, row counts, and byte sizes without rereading the potentially 20+ GiB
    feature payload. ``verify_artifact_hashes`` enables a slower full audit.
    """

    directory = Path(output_dir).resolve()
    expected_features = list(feature_columns)
    try:
        metadata = _read_store_metadata(directory)
        if metadata.get("format") != FEATURE_STORE_FORMAT:
            return False
        if metadata.get("format_version") != FEATURE_STORE_FORMAT_VERSION:
            return False
        if metadata.get("complete") is not True:
            return False
        if metadata.get("target_column") != target_column:
            return False
        if metadata.get("benchmark_column") != benchmark_column:
            return False
        if metadata.get("feature_columns") != expected_features:
            return False
        if metadata.get("feature_order_sha256") != feature_order_sha256(
            expected_features
        ):
            return False
        if metadata.get("source_fingerprints") != _source_fingerprints(
            data_paths, benchmark_paths
        ):
            return False

        features = metadata["features"]
        manifest = metadata["manifest"]
        if not isinstance(features, dict) or not isinstance(manifest, dict):
            return False
        row_count = int(metadata["row_count"])
        feature_count = int(metadata["feature_count"])
        if row_count <= 0 or feature_count != len(expected_features):
            return False
        if features.get("dtype") != "int8" or features.get("layout") != "C":
            return False
        feature_path = _safe_artifact_path(directory, features.get("filename"))
        manifest_path = _safe_artifact_path(directory, manifest.get("filename"))
        if not feature_path.is_file() or not manifest_path.is_file():
            return False
        expected_bytes = row_count * feature_count * np.dtype(np.int8).itemsize
        if features.get("size_bytes") != expected_bytes:
            return False
        if feature_path.stat().st_size != expected_bytes:
            return False

        manifest_columns = [
            "row_offset",
            "id",
            "era",
            target_column,
            benchmark_column,
        ]
        if manifest.get("columns") != manifest_columns:
            return False
        parquet = pq.ParquetFile(manifest_path)
        try:
            manifest_num_rows = parquet.metadata.num_rows
            manifest_schema = parquet.schema_arrow
        finally:
            parquet.close()
        if manifest_num_rows != row_count:
            return False
        if manifest_schema.names != manifest_columns:
            return False
        expected_types = [
            pa.int64(),
            pa.string(),
            pa.string(),
            pa.float32(),
            pa.float64(),
        ]
        if [field.type for field in manifest_schema] != expected_types:
            return False
        if manifest.get("size_bytes") != manifest_path.stat().st_size:
            return False
        if verify_artifact_hashes:
            if features.get("sha256") != _file_sha256(feature_path):
                return False
            if manifest.get("sha256") != _file_sha256(manifest_path):
                return False
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        pa.ArrowInvalid,
    ):
        return False
    return True


def _artifacts_from_metadata(output_dir: Path, *, reused: bool) -> DiskFeatureStore:
    directory = Path(output_dir).resolve()
    metadata = _read_store_metadata(directory)
    features = metadata["features"]
    manifest = metadata["manifest"]
    if not isinstance(features, dict) or not isinstance(manifest, dict):
        raise ValueError("Invalid feature-store artifact metadata.")
    return DiskFeatureStore(
        directory=directory,
        feature_path=_safe_artifact_path(directory, features.get("filename")),
        manifest_path=_safe_artifact_path(directory, manifest.get("filename")),
        metadata_path=directory / FEATURE_STORE_METADATA_FILENAME,
        row_count=int(metadata["row_count"]),
        feature_count=int(metadata["feature_count"]),
        generation_id=str(metadata["generation_id"]),
        reused=reused,
    )


def _validate_feature_store_sources(
    data_paths: Sequence[Path],
    benchmark_paths: Sequence[Path],
    feature_columns: Sequence[str],
    target_column: str,
    benchmark_column: str,
) -> None:
    if not data_paths:
        raise ValueError("At least one data parquet is required.")
    if not benchmark_paths:
        raise ValueError("At least one benchmark parquet is required.")
    if not feature_columns:
        raise ValueError("At least one feature column is required.")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Feature columns must be unique.")

    for path in data_paths:
        parquet = pq.ParquetFile(path)
        try:
            schema = parquet.schema_arrow
        finally:
            parquet.close()
        required = ["id", "era", target_column, *feature_columns]
        missing = [column for column in required if column not in schema.names]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        wrong_types = [
            column
            for column in feature_columns
            if schema.field(column).type != pa.int8()
        ]
        if wrong_types:
            raise ValueError(
                f"{path} feature columns must be int8; invalid: {wrong_types[:5]}"
            )
    for path in benchmark_paths:
        parquet = pq.ParquetFile(path)
        try:
            schema = parquet.schema_arrow
        finally:
            parquet.close()
        required = ["id", "era", benchmark_column]
        missing = [column for column in required if column not in schema.names]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")


def _build_benchmark_index(
    connection: sqlite3.Connection,
    benchmark_paths: Sequence[Path],
    benchmark_column: str,
    batch_size: int,
) -> int:
    connection.execute(
        "CREATE TABLE benchmark ("
        "id TEXT PRIMARY KEY, era TEXT NOT NULL, prediction REAL NOT NULL, "
        "consumed INTEGER NOT NULL DEFAULT 0"
        ") WITHOUT ROWID"
    )
    row_count = 0
    insert_sql = (
        "INSERT INTO benchmark (id, era, prediction, consumed) VALUES (?, ?, ?, 0)"
    )
    try:
        for path in benchmark_paths:
            parquet = pq.ParquetFile(path)
            try:
                for batch in parquet.iter_batches(
                    batch_size=batch_size,
                    columns=["id", "era", benchmark_column],
                ):
                    ids = batch.column("id").to_pylist()
                    eras = batch.column("era").to_pylist()
                    predictions = batch.column(benchmark_column).to_pylist()
                    records: list[tuple[str, str, float]] = []
                    for identifier, era, prediction in zip(ids, eras, predictions):
                        if identifier is None or era is None or prediction is None:
                            raise ValueError(
                                f"{path} contains null benchmark metadata."
                            )
                        prediction_value = float(prediction)
                        if not math.isfinite(prediction_value):
                            raise ValueError(
                                f"{path} contains non-finite predictions."
                            )
                        records.append(
                            (str(identifier), str(era), prediction_value)
                        )
                    connection.executemany(insert_sql, records)
                    row_count += len(records)
            finally:
                parquet.close()
        connection.commit()
    except sqlite3.IntegrityError as error:
        raise ValueError("Benchmark parquet files contain duplicate ids.") from error
    return row_count


def _chunks(values: Sequence[str], size: int = 800):
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _fetch_benchmark_rows(
    connection: sqlite3.Connection, identifiers: Sequence[str]
) -> dict[str, tuple[str, float, int]]:
    rows: dict[str, tuple[str, float, int]] = {}
    for chunk in _chunks(list(dict.fromkeys(identifiers))):
        placeholders = ",".join("?" for _ in chunk)
        query = (
            "SELECT id, era, prediction, consumed FROM benchmark "
            f"WHERE id IN ({placeholders})"
        )
        for identifier, era, prediction, consumed in connection.execute(query, chunk):
            rows[str(identifier)] = (str(era), float(prediction), int(consumed))
    return rows


def _mark_benchmark_rows_consumed(
    connection: sqlite3.Connection, identifiers: Sequence[str]
) -> None:
    for chunk in _chunks(list(dict.fromkeys(identifiers))):
        placeholders = ",".join("?" for _ in chunk)
        connection.execute(
            f"UPDATE benchmark SET consumed = 1 WHERE id IN ({placeholders})",
            chunk,
        )


def _fsync_file(path: Path) -> None:
    # Windows rejects fsync on a read-only descriptor.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def build_disk_feature_store(
    output_dir: Path,
    data_paths: Sequence[Path],
    benchmark_paths: Sequence[Path],
    feature_columns: Sequence[str],
    *,
    target_column: str = "target_ender_20",
    benchmark_column: str = "v53_lgbm_ender20",
    batch_size: int = 8_192,
    reuse_existing: bool = True,
) -> DiskFeatureStore:
    """Build a bounded-memory, source-ordered int8 feature store.

    Benchmark metadata is streamed into a temporary disk-backed SQLite index.
    Wide source parquets are then processed in bounded Arrow batches, retaining
    only benchmark-covered train/validation rows. Immutable generation files
    are finalized first; ``metadata.json`` is atomically replaced last and is
    therefore the sole commit marker.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    data_sources = [Path(path).resolve() for path in data_paths]
    benchmark_sources = [Path(path).resolve() for path in benchmark_paths]
    features = list(feature_columns)
    _validate_feature_store_sources(
        data_sources,
        benchmark_sources,
        features,
        target_column,
        benchmark_column,
    )

    directory = Path(output_dir).resolve()
    if reuse_existing and validate_disk_feature_store(
        directory,
        data_sources,
        benchmark_sources,
        features,
        target_column=target_column,
        benchmark_column=benchmark_column,
    ):
        existing = _artifacts_from_metadata(directory, reused=True)
        _cleanup_retired_generations(
            directory,
            _retirement_journal(directory, include_current=False),
            active_generation_id=existing.generation_id,
        )
        return existing

    source_fingerprints = _source_fingerprints(
        data_sources, benchmark_sources
    )
    directory.mkdir(parents=True, exist_ok=True)
    retired_generation_ids = _retirement_journal(
        directory, include_current=True
    )
    generation_id = uuid.uuid4().hex
    feature_filename = f"features-{generation_id}.int8.bin"
    manifest_filename = f"manifest-{generation_id}.parquet"
    feature_path = directory / feature_filename
    manifest_path = directory / manifest_filename
    metadata_path = directory / FEATURE_STORE_METADATA_FILENAME
    feature_temp = directory / f".{feature_filename}.tmp"
    manifest_temp = directory / f".{manifest_filename}.tmp"
    metadata_temp = directory / f".metadata-{generation_id}.json.tmp"
    index_temp = directory / f".benchmark-index-{generation_id}.sqlite.tmp"
    temporary_paths = [feature_temp, manifest_temp, metadata_temp, index_temp]
    generation_paths = [feature_path, manifest_path]

    connection: sqlite3.Connection | None = None
    data_parquet: pq.ParquetFile | None = None
    manifest_writer: pq.ParquetWriter | None = None
    binary_stream = None
    committed = False
    row_count = 0
    benchmark_row_count = 0
    feature_digest = hashlib.sha256()
    manifest_columns = [
        "row_offset",
        "id",
        "era",
        target_column,
        benchmark_column,
    ]
    manifest_schema = pa.schema(
        [
            pa.field("row_offset", pa.int64()),
            pa.field("id", pa.string()),
            pa.field("era", pa.string()),
            pa.field(target_column, pa.float32()),
            pa.field(benchmark_column, pa.float64()),
        ]
    )

    try:
        connection = sqlite3.connect(index_temp)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-65536")
        benchmark_row_count = _build_benchmark_index(
            connection,
            benchmark_sources,
            benchmark_column,
            batch_size,
        )

        binary_stream = feature_temp.open("xb")
        manifest_writer = pq.ParquetWriter(
            manifest_temp,
            manifest_schema,
            compression="zstd",
            compression_level=3,
        )
        for data_path in data_sources:
            if data_parquet is not None:
                data_parquet.close()
            data_parquet = pq.ParquetFile(data_path)
            columns = ["id", "era", target_column, *features]
            if "data_type" in data_parquet.schema_arrow.names:
                columns.append("data_type")
            for batch in data_parquet.iter_batches(
                batch_size=batch_size,
                columns=columns,
            ):
                table = pa.Table.from_batches([batch])
                if "data_type" in table.column_names:
                    data_types = pc.cast(table["data_type"], pa.string())
                    mask = pc.is_in(
                        data_types,
                        value_set=pa.array(["train", "validation"]),
                    )
                    table = table.filter(pc.fill_null(mask, False))
                if table.num_rows == 0:
                    continue

                identifiers = [
                    str(value) if value is not None else ""
                    for value in table["id"].to_pylist()
                ]
                if "" in identifiers:
                    raise ValueError(f"{data_path} contains null ids.")
                benchmark_rows = _fetch_benchmark_rows(
                    connection, identifiers
                )
                retained_indices: list[int] = []
                retained_ids: list[str] = []
                retained_predictions: list[float] = []
                retained_eras: list[str] = []
                data_eras = table["era"].to_pylist()
                for index, (identifier, data_era) in enumerate(
                    zip(identifiers, data_eras)
                ):
                    benchmark_row = benchmark_rows.get(identifier)
                    if benchmark_row is None:
                        continue
                    benchmark_era, prediction, consumed = benchmark_row
                    if consumed:
                        raise ValueError(
                            f"Benchmark id {identifier!r} matched multiple data rows."
                        )
                    if data_era is None or str(data_era) != benchmark_era:
                        raise ValueError(
                            f"Era mismatch for benchmark id {identifier!r}: "
                            f"data={data_era!r}, benchmark={benchmark_era!r}."
                        )
                    retained_indices.append(index)
                    retained_ids.append(identifier)
                    retained_predictions.append(prediction)
                    retained_eras.append(benchmark_era)
                if not retained_indices:
                    continue
                if len(set(retained_ids)) != len(retained_ids):
                    raise ValueError("A data batch contains duplicate benchmark ids.")

                retained = table.take(pa.array(retained_indices, type=pa.int64()))
                target = pc.cast(retained[target_column], pa.float32(), safe=True)
                if target.null_count:
                    raise ValueError(
                        f"{data_path} contains null {target_column} values for "
                        "benchmark-covered rows."
                    )
                matrix = np.empty(
                    (retained.num_rows, len(features)), dtype=np.int8
                )
                for feature_index, feature in enumerate(features):
                    values = retained[feature].combine_chunks()
                    if values.null_count:
                        raise ValueError(
                            f"{data_path} contains null values in {feature}."
                        )
                    matrix[:, feature_index] = values.to_numpy(
                        zero_copy_only=False
                    )
                matrix_bytes = memoryview(matrix).cast("B")
                bytes_written = binary_stream.write(matrix_bytes)
                if bytes_written != matrix.nbytes:
                    raise OSError("Short write while building feature payload.")
                feature_digest.update(matrix_bytes)

                batch_rows = retained.num_rows
                manifest_batch = pa.Table.from_arrays(
                    [
                        pa.array(
                            np.arange(
                                row_count,
                                row_count + batch_rows,
                                dtype=np.int64,
                            )
                        ),
                        pa.array(retained_ids, type=pa.string()),
                        pa.array(retained_eras, type=pa.string()),
                        target.combine_chunks(),
                        pa.array(retained_predictions, type=pa.float64()),
                    ],
                    schema=manifest_schema,
                )
                manifest_writer.write_table(manifest_batch)
                _mark_benchmark_rows_consumed(connection, retained_ids)
                row_count += batch_rows

        if data_parquet is not None:
            data_parquet.close()
            data_parquet = None

        if row_count == 0:
            raise ValueError("No benchmark-covered train/validation rows were found.")
        connection.commit()
        binary_stream.flush()
        os.fsync(binary_stream.fileno())
        binary_stream.close()
        binary_stream = None
        manifest_writer.close()
        manifest_writer = None
        _fsync_file(manifest_temp)

        expected_size = row_count * len(features) * np.dtype(np.int8).itemsize
        if feature_temp.stat().st_size != expected_size:
            raise OSError("Feature payload byte size does not match its shape.")
        manifest_size = manifest_temp.stat().st_size
        manifest_digest = _file_sha256(manifest_temp)
        metadata: dict[str, object] = {
            "format": FEATURE_STORE_FORMAT,
            "format_version": FEATURE_STORE_FORMAT_VERSION,
            "complete": True,
            "generation_id": generation_id,
            "row_count": row_count,
            "feature_count": len(features),
            "feature_columns": features,
            "feature_order_sha256": feature_order_sha256(features),
            "target_column": target_column,
            "benchmark_column": benchmark_column,
            "source_fingerprints": source_fingerprints,
            "retired_generation_ids": retired_generation_ids,
            "benchmark_index_rows": benchmark_row_count,
            "build_batch_size": batch_size,
            "features": {
                "filename": feature_filename,
                "dtype": "int8",
                "layout": "C",
                "size_bytes": expected_size,
                "sha256": feature_digest.hexdigest(),
            },
            "manifest": {
                "filename": manifest_filename,
                "columns": manifest_columns,
                "size_bytes": manifest_size,
                "sha256": manifest_digest,
            },
        }
        with metadata_temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(feature_temp, feature_path)
        os.replace(manifest_temp, manifest_path)
        os.replace(metadata_temp, metadata_path)
        committed = True
        _cleanup_retired_generations(
            directory,
            retired_generation_ids,
            active_generation_id=generation_id,
        )
        return DiskFeatureStore(
            directory=directory,
            feature_path=feature_path,
            manifest_path=manifest_path,
            metadata_path=metadata_path,
            row_count=row_count,
            feature_count=len(features),
            generation_id=generation_id,
            reused=False,
        )
    finally:
        if binary_stream is not None:
            binary_stream.close()
        if manifest_writer is not None:
            manifest_writer.close()
        if data_parquet is not None:
            data_parquet.close()
        if connection is not None:
            connection.close()
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        for suffix in ("-journal", "-wal", "-shm"):
            Path(f"{index_temp}{suffix}").unlink(missing_ok=True)
        if not committed:
            for path in generation_paths:
                path.unlink(missing_ok=True)


def build_v53_disk_feature_store(
    dataset_dir: Path,
    *,
    feature_columns: Sequence[str] | None = None,
    feature_set: str = "all",
    output_dir: Path | None = None,
    target_column: str = "target_ender_20",
    benchmark_column: str = "v53_lgbm_ender20",
    batch_size: int = 8_192,
    reuse_existing: bool = True,
) -> DiskFeatureStore:
    """Build the standard v5.3 target-Ender store without downloading data."""

    root = Path(dataset_dir).resolve()
    if feature_columns is None:
        features_path = root / "features.json"
        with features_path.open("r", encoding="utf-8") as stream:
            feature_metadata = json.load(stream)
        try:
            feature_columns = feature_metadata["feature_sets"][feature_set]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Feature set {feature_set!r} is not present in {features_path}."
            ) from error
    destination = (
        Path(output_dir)
        if output_dir is not None
        else root / f"{target_column}_feature_store"
    )
    return build_disk_feature_store(
        destination,
        [root / "train.parquet", root / "validation.parquet"],
        [
            root / "train_benchmark_models.parquet",
            root / "validation_benchmark_models.parquet",
        ],
        feature_columns,
        target_column=target_column,
        benchmark_column=benchmark_column,
        batch_size=batch_size,
        reuse_existing=reuse_existing,
    )


def select_downsample_eras(
    train_path: Path,
    validation_path: Path,
    era_step: int,
    era_offset: int,
) -> list[str]:
    _validate_downsample_args(era_step, era_offset)

    train_eras = pd.read_parquet(train_path, columns=["era"])["era"]
    validation_meta = pd.read_parquet(
        validation_path, columns=["era", "data_type"]
    )
    validation_eras = validation_meta.loc[
        validation_meta["data_type"] == "validation", "era"
    ]
    unique_eras = sorted(
        set(train_eras.astype(str)) | set(validation_eras.astype(str)),
        key=lambda era: int(era),
    )
    return [
        era
        for index, era in enumerate(unique_eras)
        if index % era_step == era_offset
    ]


def _validate_downsample_args(era_step: int, era_offset: int) -> None:
    if isinstance(era_step, bool) or not isinstance(era_step, int) or era_step < 2:
        raise ValueError("downsample-eras-step must be >= 2.")
    if (
        isinstance(era_offset, bool)
        or not isinstance(era_offset, int)
        or era_offset < 0
        or era_offset >= era_step
    ):
        raise ValueError("downsample-eras-offset must be in [0, downsample-eras-step).")


def _keep_eras_sha256(keep_eras: Sequence[str]) -> str:
    encoded = json.dumps(
        list(keep_eras), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _downsample_artifact_fingerprint(
    path: Path, *, published_filename: str | None = None
) -> dict[str, object]:
    """Fully fingerprint one staged or published downsample parquet."""

    resolved = path.resolve(strict=True)
    parquet = pq.ParquetFile(resolved)
    try:
        row_count = parquet.metadata.num_rows
        schema_bytes = parquet.schema_arrow.serialize().to_pybytes()
    finally:
        parquet.close()
    if row_count <= 0:
        raise ValueError(f"{resolved} contains no rows.")
    return {
        "filename": published_filename or resolved.name,
        "num_rows": row_count,
        "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def _downsample_pair_metadata(
    *,
    data_version: str,
    era_step: int,
    era_offset: int,
    keep_eras: Sequence[str],
    source_fingerprints: Sequence[dict[str, object]],
    data_fingerprint: dict[str, object],
    benchmark_fingerprint: dict[str, object],
) -> dict[str, object]:
    return {
        "format": DOWNSAMPLE_PAIR_FORMAT,
        "format_version": DOWNSAMPLE_PAIR_FORMAT_VERSION,
        "complete": True,
        "data_version": data_version,
        "era_step": era_step,
        "era_offset": era_offset,
        "keep_eras": list(keep_eras),
        "keep_eras_sha256": _keep_eras_sha256(keep_eras),
        "source_fingerprints": list(source_fingerprints),
        "outputs": {
            "data": data_fingerprint,
            "benchmark": benchmark_fingerprint,
        },
    }


def _validate_downsample_pair(
    manifest_path: Path,
    data_path: Path,
    benchmark_path: Path,
    *,
    data_version: str,
    era_step: int,
    era_offset: int,
    keep_eras: Sequence[str],
    source_fingerprints: Sequence[dict[str, object]],
) -> bool:
    """Return whether the manifest and both fixed artifacts match exactly."""

    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            actual = json.load(stream)
        if not isinstance(actual, dict):
            return False
        expected = _downsample_pair_metadata(
            data_version=data_version,
            era_step=era_step,
            era_offset=era_offset,
            keep_eras=keep_eras,
            source_fingerprints=source_fingerprints,
            data_fingerprint=_downsample_artifact_fingerprint(data_path),
            benchmark_fingerprint=_downsample_artifact_fingerprint(benchmark_path),
        )
        return actual == expected
    except (
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
        pa.ArrowInvalid,
    ):
        return False


def _write_json_fsynced(path: Path, value: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _publish_downsample_pair(
    *,
    staged_data_path: Path,
    staged_benchmark_path: Path,
    staged_manifest_path: Path,
    data_path: Path,
    benchmark_path: Path,
    manifest_path: Path,
    transaction_id: str,
) -> None:
    """Publish two fixed outputs atomically from the manifest reader's view."""

    replacements = (
        (staged_data_path, data_path),
        (staged_benchmark_path, benchmark_path),
        (staged_manifest_path, manifest_path),
    )
    backup_paths = {
        fixed_path: fixed_path.with_name(
            f".{fixed_path.name}.backup-{transaction_id}"
        )
        for _, fixed_path in replacements
    }
    collisions = [path for path in backup_paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(f"Downsample backup path already exists: {collisions[0]}")

    moved_backups: list[tuple[Path, Path]] = []
    published_paths: list[Path] = []
    committed = False
    try:
        # Retire the old manifest first so readers cannot accept a mixed pair.
        for fixed_path in (manifest_path, data_path, benchmark_path):
            if fixed_path.exists():
                backup_path = backup_paths[fixed_path]
                os.replace(fixed_path, backup_path)
                moved_backups.append((backup_path, fixed_path))

        for staged_path, fixed_path in replacements:
            os.replace(staged_path, fixed_path)
            published_paths.append(fixed_path)
        committed = True
    except Exception as publish_error:
        rollback_errors: list[Exception] = []
        for fixed_path in reversed(published_paths):
            try:
                fixed_path.unlink(missing_ok=True)
            except Exception as error:
                rollback_errors.append(error)
        for backup_path, fixed_path in reversed(moved_backups):
            try:
                os.replace(backup_path, fixed_path)
            except Exception as error:
                rollback_errors.append(error)
        if rollback_errors:
            raise RuntimeError(
                "Downsample pair publication failed and rollback was incomplete."
            ) from publish_error
        raise
    finally:
        if committed:
            for backup_path in backup_paths.values():
                try:
                    backup_path.unlink(missing_ok=True)
                except OSError:
                    # The new manifest is already committed; retain a locked backup
                    # rather than turning a successful publication into a false error.
                    pass


def write_filtered_parquets(
    source_paths: list[Path],
    output_path: Path,
    keep_eras: list[str],
    *,
    validation_data: bool,
    drop_data_type: bool,
    batch_size: int = 65_536,
) -> Path:
    """Stream selected eras from wide parquet files with bounded host memory."""

    keep_values = pa.array(keep_eras)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".temp")
    temporary_path.unlink(missing_ok=True)
    writer = None
    try:
        for source_path in source_paths:
            is_validation_source = source_path.name.startswith("validation")
            parquet = pq.ParquetFile(source_path)
            try:
                for batch in parquet.iter_batches(batch_size=batch_size):
                    table = pa.Table.from_batches([batch])
                    mask = pc.is_in(table["era"], value_set=keep_values)
                    if (
                        validation_data
                        and is_validation_source
                        and "data_type" in table.column_names
                    ):
                        mask = pc.and_(
                            mask, pc.equal(table["data_type"], "validation")
                        )
                    table = table.filter(mask)
                    if table.num_rows == 0:
                        continue
                    if drop_data_type and "data_type" in table.column_names:
                        table = table.drop_columns(["data_type"])
                    # Source files encode ``id`` as a pandas index in schema
                    # metadata even though it is a physical parquet field. The
                    # modeling pipeline requires it to reload as a normal column.
                    table = table.replace_schema_metadata()
                    if writer is None:
                        writer = pq.ParquetWriter(
                            temporary_path,
                            table.schema,
                            compression="zstd",
                            compression_level=3,
                        )
                    writer.write_table(table)
            finally:
                parquet.close()
        if writer is None:
            raise ValueError("No rows matched the requested downsample eras.")
        writer.close()
        writer = None
        _fsync_file(temporary_path)
        os.replace(temporary_path, output_path)
        return output_path
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def is_valid_parquet(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        parquet = pq.ParquetFile(path)
        try:
            return parquet.metadata.num_rows > 0
        finally:
            parquet.close()
    except (OSError, pa.ArrowInvalid):
        return False


def build_downsampled_direct(
    napi: NumerAPI,
    data_version: str,
    era_step: int,
    era_offset: int,
    *,
    reuse_existing: bool = True,
) -> tuple[Path, Path]:
    # Invalid selection parameters must never be hidden by a warm cache.
    _validate_downsample_args(era_step, era_offset)

    sources = ensure_source_datasets(napi, data_version)
    data_sources = [
        Path(sources[name]).resolve(strict=True)
        for name in ("train.parquet", "validation.parquet")
    ]
    benchmark_sources = [
        Path(sources[name]).resolve(strict=True)
        for name in (
            "train_benchmark_models.parquet",
            "validation_benchmark_models.parquet",
        )
    ]
    source_fingerprints = _source_fingerprints(data_sources, benchmark_sources)
    keep_eras = select_downsample_eras(
        data_sources[0],
        data_sources[1],
        era_step,
        era_offset,
    )

    output_directory = (NUMERAI_DIR / data_version).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    data_path = output_directory / DOWNSAMPLE_DATA_FILENAME
    benchmark_path = output_directory / DOWNSAMPLE_BENCHMARK_FILENAME
    manifest_path = output_directory / DOWNSAMPLE_PAIR_MANIFEST_FILENAME
    if reuse_existing and _validate_downsample_pair(
        manifest_path,
        data_path,
        benchmark_path,
        data_version=data_version,
        era_step=era_step,
        era_offset=era_offset,
        keep_eras=keep_eras,
        source_fingerprints=source_fingerprints,
    ):
        return data_path, benchmark_path

    transaction_id = uuid.uuid4().hex
    staged_data_path = output_directory / (
        f".{DOWNSAMPLE_DATA_FILENAME}.staging-{transaction_id}"
    )
    staged_benchmark_path = output_directory / (
        f".{DOWNSAMPLE_BENCHMARK_FILENAME}.staging-{transaction_id}"
    )
    staged_manifest_path = output_directory / (
        f".{DOWNSAMPLE_PAIR_MANIFEST_FILENAME}.staging-{transaction_id}"
    )
    staging_paths = (
        staged_data_path,
        staged_data_path.with_suffix(staged_data_path.suffix + ".temp"),
        staged_benchmark_path,
        staged_benchmark_path.with_suffix(
            staged_benchmark_path.suffix + ".temp"
        ),
        staged_manifest_path,
    )
    collisions = [path for path in staging_paths if path.exists()]
    if collisions:
        raise FileExistsError(
            f"Downsample staging path already exists: {collisions[0]}"
        )

    try:
        write_filtered_parquets(
            data_sources,
            staged_data_path,
            keep_eras,
            validation_data=True,
            drop_data_type=True,
        )
        write_filtered_parquets(
            benchmark_sources,
            staged_benchmark_path,
            keep_eras,
            validation_data=False,
            drop_data_type=False,
        )

        _fsync_file(staged_data_path)
        _fsync_file(staged_benchmark_path)
        data_fingerprint = _downsample_artifact_fingerprint(
            staged_data_path, published_filename=DOWNSAMPLE_DATA_FILENAME
        )
        benchmark_fingerprint = _downsample_artifact_fingerprint(
            staged_benchmark_path,
            published_filename=DOWNSAMPLE_BENCHMARK_FILENAME,
        )
        if _source_fingerprints(data_sources, benchmark_sources) != source_fingerprints:
            raise RuntimeError("A downsample source changed while the pair was built.")

        metadata = _downsample_pair_metadata(
            data_version=data_version,
            era_step=era_step,
            era_offset=era_offset,
            keep_eras=keep_eras,
            source_fingerprints=source_fingerprints,
            data_fingerprint=data_fingerprint,
            benchmark_fingerprint=benchmark_fingerprint,
        )
        _write_json_fsynced(staged_manifest_path, metadata)
        with staged_manifest_path.open("r", encoding="utf-8") as stream:
            if json.load(stream) != metadata:
                raise OSError("Downsample pair manifest verification failed.")

        _publish_downsample_pair(
            staged_data_path=staged_data_path,
            staged_benchmark_path=staged_benchmark_path,
            staged_manifest_path=staged_manifest_path,
            data_path=data_path,
            benchmark_path=benchmark_path,
            manifest_path=manifest_path,
            transaction_id=transaction_id,
        )
    finally:
        for path in staging_paths:
            path.unlink(missing_ok=True)
    return data_path, benchmark_path


def build_full_dataset(
    napi: NumerAPI, data_version: str, reuse_existing: bool = True
) -> Path:
    full_path = (NUMERAI_DIR / data_version / "full.parquet").resolve()
    if reuse_existing and full_path.exists():
        return full_path
    train_path = (NUMERAI_DIR / data_version / "train.parquet").resolve()
    validation_path = (NUMERAI_DIR / data_version / "validation.parquet").resolve()
    if not train_path.exists():
        train_path.parent.mkdir(parents=True, exist_ok=True)
        napi.download_dataset(f"{data_version}/train.parquet", dest_path=str(train_path))
    if not validation_path.exists():
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        napi.download_dataset(
            f"{data_version}/validation.parquet", dest_path=str(validation_path)
        )

    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)
    validation = validation[validation["data_type"] == "validation"].copy()

    full = pd.concat([train, validation], ignore_index=False)
    full = full.drop(columns=["data_type"], errors="ignore")
    if full.index.name and full.index.name not in full.columns:
        full = full.reset_index()
    full.to_parquet(full_path, index=False)
    return full_path


def build_full_benchmark(
    napi: NumerAPI, data_version: str, reuse_existing: bool = True
) -> Path:
    full_path = (NUMERAI_DIR / data_version / "full_benchmark_models.parquet").resolve()
    if reuse_existing and full_path.exists():
        return full_path
    train_path = (NUMERAI_DIR / data_version / "train_benchmark_models.parquet").resolve()
    validation_path = (
        NUMERAI_DIR / data_version / "validation_benchmark_models.parquet"
    ).resolve()
    validation_data_path = (NUMERAI_DIR / data_version / "validation.parquet").resolve()
    if not train_path.exists():
        train_path.parent.mkdir(parents=True, exist_ok=True)
        napi.download_dataset(
            f"{data_version}/train_benchmark_models.parquet",
            dest_path=str(train_path),
        )
    if not validation_path.exists():
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        napi.download_dataset(
            f"{data_version}/validation_benchmark_models.parquet",
            dest_path=str(validation_path),
        )
    if not validation_data_path.exists():
        validation_data_path.parent.mkdir(parents=True, exist_ok=True)
        napi.download_dataset(
            f"{data_version}/validation.parquet", dest_path=str(validation_data_path)
        )

    validation_meta = pd.read_parquet(validation_data_path, columns=["data_type"])
    validation_meta = validation_meta[validation_meta["data_type"] == "validation"]
    validation_ids = validation_meta.index

    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)
    if "id" in train.columns:
        train = train.set_index("id")
    if "id" in validation.columns:
        validation = validation.set_index("id")
    validation = validation.loc[validation.index.intersection(validation_ids)]

    full = pd.concat([train, validation], axis=0)
    full.to_parquet(full_path)
    return full_path


def build_downsampled_full_dataset(
    full_path: Path,
    data_version: str,
    era_step: int,
    era_offset: int,
) -> Path:
    if era_step < 2:
        raise ValueError("downsample-eras-step must be >= 2.")
    if era_offset < 0 or era_offset >= era_step:
        raise ValueError("downsample-eras-offset must be in [0, downsample-eras-step).")
    downsampled_path = (NUMERAI_DIR / data_version / "downsampled_full.parquet").resolve()
    full = pd.read_parquet(full_path)
    era_col = "era"
    if era_col not in full.columns:
        raise ValueError(f"{full_path} missing '{era_col}' column.")
    unique_eras = sorted(full[era_col].unique(), key=lambda x: int(x))
    keep_eras = {
        era for idx, era in enumerate(unique_eras) if idx % era_step == era_offset
    }
    downsampled = full[full[era_col].isin(keep_eras)].copy()
    downsampled.to_parquet(downsampled_path, index=False)
    return downsampled_path


def build_downsampled_full_benchmark(
    full_benchmark_path: Path,
    downsampled_full_path: Path,
    data_version: str,
) -> Path:
    downsampled_path = (
        NUMERAI_DIR / data_version / "downsampled_full_benchmark_models.parquet"
    ).resolve()
    ids = pd.read_parquet(downsampled_full_path, columns=["id"])
    if "id" not in ids.columns:
        raise ValueError(f"{downsampled_full_path} missing 'id' column.")
    id_values = ids["id"].dropna().unique()
    benchmark = pd.read_parquet(full_benchmark_path)
    if "id" in benchmark.columns:
        benchmark = benchmark.set_index("id")
    benchmark = benchmark.loc[benchmark.index.intersection(id_values)]
    benchmark.to_parquet(downsampled_path)
    return downsampled_path


def main() -> None:
    args = parse_args()
    data_version = args.data_version
    napi = NumerAPI()
    reuse_existing = not args.rebuild

    if args.downsample_only:
        downsampled_full, downsampled_benchmark = build_downsampled_direct(
            napi,
            data_version,
            args.downsample_eras_step,
            args.downsample_eras_offset,
            reuse_existing=reuse_existing,
        )
        print(f"Built {downsampled_full}")
        print(f"Built {downsampled_benchmark}")
        return

    full_data = build_full_dataset(napi, data_version, reuse_existing=reuse_existing)
    full_benchmark = build_full_benchmark(
        napi, data_version, reuse_existing=reuse_existing
    )

    print(f"Built {full_data}")
    print(f"Built {full_benchmark}")
    if not args.skip_downsample:
        downsampled_full = build_downsampled_full_dataset(
            full_data,
            data_version,
            args.downsample_eras_step,
            args.downsample_eras_offset,
        )
        downsampled_benchmark = build_downsampled_full_benchmark(
            full_benchmark,
            downsampled_full,
            data_version,
        )
        print(f"Built {downsampled_full}")
        print(f"Built {downsampled_benchmark}")


if __name__ == "__main__":
    main()
