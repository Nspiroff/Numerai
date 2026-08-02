from __future__ import annotations

from collections.abc import Iterator, Sequence
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


FEATURE_STORE_FORMAT = "numerai-v5.3-int8-feature-store"
FEATURE_STORE_FORMAT_VERSION = 1
FEATURE_STORE_METADATA_FILENAME = "metadata.json"


def _feature_order_sha256(feature_columns: Sequence[str]) -> str:
    encoded = json.dumps(
        list(feature_columns), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_artifact_path(directory: Path, filename: object) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError("Feature-store artifact filename must be a non-empty string.")
    if Path(filename).name != filename:
        raise ValueError("Feature-store artifact filename must not contain a path.")
    return directory / filename


def _canonical_generation_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Feature-store generation_id must be a UUID hex string.")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise ValueError("Feature-store generation_id must be a UUID hex string.") from error
    if parsed.hex != value:
        raise ValueError("Feature-store generation_id must use canonical UUID hex.")
    return value


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class _DiskFeatureStoreState:
    def __init__(
        self,
        *,
        directory: Path,
        feature_path: Path,
        manifest_path: Path,
        metadata_path: Path,
        metadata: dict,
        manifest: pd.DataFrame,
        feature_columns: Sequence[str],
        row_offsets: np.ndarray,
        feature_memmap: np.memmap,
    ) -> None:
        self.directory = directory
        self.feature_path = feature_path
        self.manifest_path = manifest_path
        self.metadata_path = metadata_path
        self.metadata = metadata
        self.manifest = manifest
        self.feature_columns = tuple(feature_columns)
        self.feature_index = {
            name: position for position, name in enumerate(self.feature_columns)
        }
        self.row_offsets = np.asarray(row_offsets, dtype=np.int64)
        self.row_count = len(self.row_offsets)
        self.feature_count = len(self.feature_columns)
        self._memmap: np.memmap | None = feature_memmap
        self._closed = False

    def _features(self) -> np.memmap:
        if self._closed or self._memmap is None:
            raise RuntimeError("Disk feature-store loader is closed.")
        return self._memmap

    def read_features(
        self, manifest_positions: np.ndarray, feature_indices: np.ndarray
    ) -> np.ndarray:
        manifest_positions = np.asarray(manifest_positions, dtype=np.int64)
        if manifest_positions.ndim != 1:
            raise ValueError("manifest_positions must be one-dimensional.")
        if manifest_positions.size == 0:
            return np.empty((0, len(feature_indices)), dtype=np.int8)
        if manifest_positions.min() < 0 or manifest_positions.max() >= self.row_count:
            raise IndexError("Disk feature-store position is out of range.")

        offsets = self.row_offsets[manifest_positions]
        feature_indices = np.asarray(feature_indices, dtype=np.int64)
        mmap = self._features()
        contiguous_rows = offsets.size == 1 or np.all(np.diff(offsets) == 1)
        all_features = (
            feature_indices.size == self.feature_count
            and np.array_equal(
                feature_indices, np.arange(self.feature_count, dtype=np.int64)
            )
        )
        if contiguous_rows:
            rows = mmap[int(offsets[0]) : int(offsets[-1]) + 1]
            values = rows if all_features else rows[:, feature_indices]
        elif all_features:
            values = mmap[offsets]
        else:
            values = mmap[np.ix_(offsets, feature_indices)]
        # Torch must never receive a read-only view backed by the long-lived
        # Windows memmap. Copy only this bounded batch.
        return np.array(values, dtype=np.int8, order="C", copy=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        mmap = self._memmap
        self._memmap = None
        if mmap is not None:
            mmap._mmap.close()


class DiskFeatureView:
    """A row subset over an int8 feature memmap plus in-memory metadata.

    Feature access is deliberately batch-only. Era, id, target, and benchmark
    metadata remain pandas Series so existing target transforms and OOF output
    handling retain their eager semantics.
    """

    is_disk_feature_view = True

    def __init__(
        self,
        state: _DiskFeatureStoreState,
        manifest_positions: Sequence[int] | np.ndarray,
        *,
        feature_columns: Sequence[str],
        metadata_columns: Sequence[str],
    ) -> None:
        self._state = state
        self._positions = np.asarray(manifest_positions, dtype=np.int64)
        if self._positions.ndim != 1:
            raise ValueError("DiskFeatureView positions must be one-dimensional.")
        if self._positions.size and (
            self._positions.min() < 0 or self._positions.max() >= state.row_count
        ):
            raise IndexError("DiskFeatureView position is out of range.")
        self._feature_columns = tuple(feature_columns)
        missing_features = [
            name for name in self._feature_columns if name not in state.feature_index
        ]
        if missing_features:
            raise ValueError(
                f"Unknown feature columns in DiskFeatureView: {missing_features[:5]}"
            )
        self._feature_indices = np.asarray(
            [state.feature_index[name] for name in self._feature_columns],
            dtype=np.int64,
        )
        self._metadata_columns = tuple(metadata_columns)
        missing_metadata = [
            name
            for name in self._metadata_columns
            if name not in state.manifest.columns
        ]
        if missing_metadata:
            raise ValueError(
                f"Unknown metadata columns in DiskFeatureView: {missing_metadata}"
            )

    def __len__(self) -> int:
        return len(self._positions)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self), len(self.columns)

    @property
    def columns(self) -> pd.Index:
        return pd.Index([*self._feature_columns, *self._metadata_columns])

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self._feature_columns

    @property
    def manifest_positions(self) -> np.ndarray:
        return self._positions.copy()

    @property
    def row_offsets(self) -> np.ndarray:
        return self._state.row_offsets[self._positions].copy()

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self._metadata_columns:
                return self._state.manifest[key].iloc[self._positions]
            if key in self._feature_columns:
                raise TypeError(
                    "Disk feature columns are batch-only; use iter_feature_batches()."
                )
            raise KeyError(key)

        requested = list(key)
        if all(name in self._feature_columns for name in requested):
            return self.select_features(requested)
        if all(name in self._metadata_columns for name in requested):
            return self._state.manifest[requested].iloc[self._positions]
        raise TypeError(
            "Mixed feature/metadata column materialization is disabled for "
            "DiskFeatureView."
        )

    def select_features(self, feature_columns: Sequence[str]) -> DiskFeatureView:
        requested = tuple(feature_columns)
        missing = [name for name in requested if name not in self._feature_columns]
        if missing:
            raise ValueError(f"Missing disk feature columns: {missing[:5]}")
        return DiskFeatureView(
            self._state,
            self._positions,
            feature_columns=requested,
            metadata_columns=(),
        )

    def take(self, indices: Sequence[int] | np.ndarray) -> DiskFeatureView:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("DiskFeatureView take indices must be one-dimensional.")
        if indices.size and (indices.min() < 0 or indices.max() >= len(self)):
            raise IndexError("DiskFeatureView take index is out of range.")
        return DiskFeatureView(
            self._state,
            self._positions[indices],
            feature_columns=self._feature_columns,
            metadata_columns=self._metadata_columns,
        )

    def iter_feature_batches(
        self,
        batch_size: int,
        *,
        shuffle_blocks: bool = False,
        seed: int = 0,
        block_rows: int = 65_536,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if block_rows <= 0:
            raise ValueError("block_rows must be positive.")
        if not len(self):
            return

        if not shuffle_blocks:
            batches = [
                np.arange(start, min(start + batch_size, len(self)), dtype=np.int64)
                for start in range(0, len(self), batch_size)
            ]
        else:
            offsets = self._state.row_offsets[self._positions]
            source_order = np.argsort(offsets, kind="stable")
            block_ids = offsets[source_order] // int(block_rows)
            split_points = np.flatnonzero(np.diff(block_ids)) + 1
            blocks = list(np.split(source_order, split_points))
            rng = np.random.default_rng(seed)
            rng.shuffle(blocks)
            batches = []
            for block in blocks:
                block_batches = [
                    block[start : start + batch_size]
                    for start in range(0, len(block), batch_size)
                ]
                rng.shuffle(block_batches)
                batches.extend(block_batches)

        for local_positions in batches:
            yield (
                self._state.read_features(
                    self._positions[local_positions], self._feature_indices
                ),
                np.asarray(local_positions, dtype=np.int64),
            )


class DiskFeatureStoreLoader:
    """Load a committed feature store without materializing its feature matrix."""

    def __init__(
        self,
        directory: str | Path,
        *,
        era_col: str,
        target_col: str,
        id_col: str | None,
        benchmark_col: str,
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.era_col = str(era_col)
        self.target_col = str(target_col)
        self.id_col = str(id_col) if id_col else None
        self.benchmark_col = str(benchmark_col)
        self._x_cols: tuple[str, ...] | None = None
        self._era_positions: dict[str, np.ndarray] | None = None
        self._loading_generation_id: str | None = None
        for attempt in range(3):
            try:
                self._state = self._load_and_validate()
                break
            except Exception:
                if attempt == 2 or not self._metadata_generation_changed():
                    raise

    def _metadata_generation_changed(self) -> bool:
        previous = self._loading_generation_id
        if previous is None:
            return False
        metadata_path = self.directory / FEATURE_STORE_METADATA_FILENAME
        try:
            with metadata_path.open("r", encoding="utf-8") as stream:
                current = json.load(stream)
            return (
                isinstance(current, dict)
                and current.get("format") == FEATURE_STORE_FORMAT
                and current.get("format_version") == FEATURE_STORE_FORMAT_VERSION
                and current.get("complete") is True
                and _canonical_generation_id(current.get("generation_id")) != previous
            )
        except (OSError, json.JSONDecodeError, ValueError):
            return False

    def _load_and_validate(self) -> _DiskFeatureStoreState:
        metadata_path = self.directory / FEATURE_STORE_METADATA_FILENAME
        try:
            with metadata_path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Cannot read feature-store metadata: {metadata_path}"
            ) from error
        if not isinstance(metadata, dict):
            raise ValueError("Feature-store metadata must be a JSON object.")
        if metadata.get("format") != FEATURE_STORE_FORMAT:
            raise ValueError("Unsupported feature-store format.")
        if metadata.get("format_version") != FEATURE_STORE_FORMAT_VERSION:
            raise ValueError("Unsupported feature-store format_version.")
        if metadata.get("complete") is not True:
            raise ValueError("Feature-store metadata is not committed.")
        generation_id = _canonical_generation_id(metadata.get("generation_id"))
        self._loading_generation_id = generation_id
        if metadata.get("target_column") != self.target_col:
            raise ValueError(
                f"Feature-store target is {metadata.get('target_column')!r}, "
                f"not {self.target_col!r}."
            )
        if metadata.get("benchmark_column") != self.benchmark_col:
            raise ValueError(
                f"Feature-store benchmark is {metadata.get('benchmark_column')!r}, "
                f"not {self.benchmark_col!r}."
            )

        feature_columns = metadata.get("feature_columns")
        if (
            not isinstance(feature_columns, list)
            or not feature_columns
            or not all(isinstance(name, str) and name for name in feature_columns)
            or len(set(feature_columns)) != len(feature_columns)
        ):
            raise ValueError("Feature-store feature_columns must be unique names.")
        if metadata.get("feature_order_sha256") != _feature_order_sha256(
            feature_columns
        ):
            raise ValueError("Feature-store feature-order hash does not match metadata.")

        try:
            row_count = int(metadata["row_count"])
            feature_count = int(metadata["feature_count"])
            feature_meta = metadata["features"]
            manifest_meta = metadata["manifest"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Feature-store shape/artifact metadata is malformed.") from error
        if row_count <= 0 or feature_count != len(feature_columns):
            raise ValueError("Feature-store shape does not match feature metadata.")
        if not isinstance(feature_meta, dict) or not isinstance(manifest_meta, dict):
            raise ValueError("Feature-store artifact metadata must be JSON objects.")
        if feature_meta.get("dtype") != "int8" or feature_meta.get("layout") != "C":
            raise ValueError("Feature-store payload must be C-order int8.")
        if not _is_sha256(feature_meta.get("sha256")) or not _is_sha256(
            manifest_meta.get("sha256")
        ):
            raise ValueError("Feature-store artifact hashes are malformed.")

        expected_feature_name = f"features-{generation_id}.int8.bin"
        expected_manifest_name = f"manifest-{generation_id}.parquet"
        if feature_meta.get("filename") != expected_feature_name:
            raise ValueError("Feature-store payload filename does not match generation_id.")
        if manifest_meta.get("filename") != expected_manifest_name:
            raise ValueError("Feature-store manifest filename does not match generation_id.")
        feature_path = _safe_artifact_path(
            self.directory, feature_meta.get("filename")
        )
        manifest_path = _safe_artifact_path(
            self.directory, manifest_meta.get("filename")
        )
        if not feature_path.is_file() or not manifest_path.is_file():
            raise ValueError("Feature-store referenced artifacts do not exist.")
        expected_feature_bytes = row_count * feature_count * np.dtype(np.int8).itemsize
        if feature_meta.get("size_bytes") != expected_feature_bytes:
            raise ValueError("Feature-store payload size metadata is inconsistent.")
        if feature_path.stat().st_size != expected_feature_bytes:
            raise ValueError("Feature-store payload byte size does not match its shape.")
        if manifest_meta.get("size_bytes") != manifest_path.stat().st_size:
            raise ValueError("Feature-store manifest byte size does not match metadata.")
        source_fingerprints = metadata.get("source_fingerprints")
        if not isinstance(source_fingerprints, list) or not source_fingerprints:
            raise ValueError("Feature-store source fingerprints are missing.")

        expected_manifest_columns = [
            "row_offset",
            self.id_col,
            self.era_col,
            self.target_col,
            self.benchmark_col,
        ]
        expected_manifest_columns = [
            name for name in expected_manifest_columns if name is not None
        ]
        if manifest_meta.get("columns") != expected_manifest_columns:
            raise ValueError("Feature-store manifest columns do not match the request.")

        # Acquire the generation lease before hashing or parsing the much larger
        # manifest. A read-only memmap reserves address space but does not load
        # the feature payload into RAM.
        try:
            feature_memmap = np.memmap(
                feature_path,
                dtype=np.int8,
                mode="r",
                shape=(row_count, feature_count),
                order="C",
            )
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Cannot open feature-store payload: {feature_path}"
            ) from error
        try:
            manifest, offsets = self._load_validated_manifest(
                manifest_path,
                manifest_meta,
                expected_manifest_columns,
                row_count,
            )
            return _DiskFeatureStoreState(
                directory=self.directory,
                feature_path=feature_path,
                manifest_path=manifest_path,
                metadata_path=metadata_path,
                metadata=metadata,
                manifest=manifest,
                feature_columns=feature_columns,
                row_offsets=offsets,
                feature_memmap=feature_memmap,
            )
        except BaseException:
            feature_memmap._mmap.close()
            raise

    def _load_validated_manifest(
        self,
        manifest_path: Path,
        manifest_meta: dict,
        expected_manifest_columns: list[str],
        row_count: int,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        if _sha256_file(manifest_path) != manifest_meta["sha256"]:
            raise ValueError("Feature-store manifest SHA-256 does not match metadata.")

        parquet = pq.ParquetFile(manifest_path)
        try:
            if parquet.metadata.num_rows != row_count:
                raise ValueError("Feature-store manifest row count is inconsistent.")
            schema = parquet.schema_arrow
        finally:
            parquet.close()
        if schema.names != expected_manifest_columns:
            raise ValueError("Feature-store manifest schema columns are inconsistent.")
        if schema.field("row_offset").type != pa.int64():
            raise ValueError("Feature-store row_offset must be int64.")

        manifest = pd.read_parquet(manifest_path, columns=expected_manifest_columns)
        manifest = manifest.reset_index(drop=True)
        offsets = manifest["row_offset"].to_numpy(dtype=np.int64, copy=False)
        if not np.array_equal(offsets, np.arange(row_count, dtype=np.int64)):
            raise ValueError("Feature-store row_offset must be contiguous and ordered.")
        if self.id_col:
            if manifest[self.id_col].isna().any():
                raise ValueError("Feature-store manifest contains null ids.")
            if manifest[self.id_col].duplicated().any():
                raise ValueError("Feature-store manifest contains duplicate ids.")
        if manifest[self.era_col].isna().any():
            raise ValueError("Feature-store manifest contains null eras.")
        manifest[self.era_col] = manifest[self.era_col].astype(str)
        for column in (self.target_col, self.benchmark_col):
            values = pd.to_numeric(manifest[column], errors="coerce").to_numpy()
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Feature-store manifest contains non-finite {column} values."
                )
        return manifest, offsets

    @property
    def manifest(self) -> pd.DataFrame:
        return self._state.manifest

    @property
    def manifest_path(self) -> Path:
        return self._state.manifest_path

    @property
    def feature_path(self) -> Path:
        return self._state.feature_path

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self._state.feature_columns

    @property
    def eras(self) -> pd.Series:
        return self.manifest[self.era_col]

    @property
    def diagnostics(self) -> dict[str, object]:
        metadata = self._state.metadata
        return {
            "directory": str(self.directory),
            "feature_path": str(self.feature_path),
            "manifest_path": str(self.manifest_path),
            "generation_id": metadata["generation_id"],
            "row_count": self._state.row_count,
            "feature_count": self._state.feature_count,
            "feature_bytes": int(metadata["features"]["size_bytes"]),
            "manifest_bytes": int(metadata["manifest"]["size_bytes"]),
            "feature_order_sha256": metadata["feature_order_sha256"],
            "feature_sha256": metadata["features"]["sha256"],
            "manifest_sha256": metadata["manifest"]["sha256"],
        }

    def configure_x_cols(self, x_cols: Sequence[str]) -> None:
        requested = tuple(x_cols)
        if not requested:
            raise ValueError("x_cols must be a non-empty list.")
        available = set(self.feature_columns) | set(self.manifest.columns)
        missing = [name for name in requested if name not in available]
        if missing:
            raise ValueError(f"Feature-store x_cols are unavailable: {missing[:5]}")
        self._x_cols = requested

    def _positions_by_era(self) -> dict[str, np.ndarray]:
        if self._era_positions is None:
            grouped = self.manifest.groupby(self.era_col, sort=False).indices
            self._era_positions = {
                str(era): np.asarray(positions, dtype=np.int64)
                for era, positions in grouped.items()
            }
        return self._era_positions

    def load(self, eras: Sequence) -> "ModelDataBatch":
        if self._x_cols is None:
            raise RuntimeError("configure_x_cols() must be called before load().")
        positions_by_era = self._positions_by_era()
        requested_eras = dict.fromkeys(str(era) for era in eras)
        selected = [
            positions_by_era[era]
            for era in requested_eras
            if era in positions_by_era
        ]
        positions = (
            np.sort(np.concatenate(selected), kind="stable")
            if selected
            else np.empty(0, dtype=np.int64)
        )
        feature_columns = [
            name for name in self._x_cols if name in self._state.feature_index
        ]
        metadata_columns = [
            name for name in self._x_cols if name not in self._state.feature_index
        ]
        view = DiskFeatureView(
            self._state,
            positions,
            feature_columns=feature_columns,
            metadata_columns=metadata_columns,
        )
        from agents.code.modeling.utils.model_data import ModelDataBatch

        return ModelDataBatch(
            X=view,
            y=self.manifest[self.target_col].iloc[positions],
            era=self.manifest[self.era_col].iloc[positions],
            id=(
                self.manifest[self.id_col].iloc[positions]
                if self.id_col is not None
                else None
            ),
        )

    def close(self) -> None:
        self._state.close()
        self._cleanup_retired_generation()

    def _cleanup_retired_generation(self) -> None:
        """Best-effort removal of this exact generation after it is retired."""
        metadata_path = self.directory / FEATURE_STORE_METADATA_FILENAME
        try:
            with metadata_path.open("r", encoding="utf-8") as stream:
                current = json.load(stream)
            if not isinstance(current, dict):
                return
            if current.get("format") != FEATURE_STORE_FORMAT:
                return
            if current.get("format_version") != FEATURE_STORE_FORMAT_VERSION:
                return
            if current.get("complete") is not True:
                return
            current_generation = _canonical_generation_id(
                current.get("generation_id")
            )
            current_features = current.get("features")
            current_manifest = current.get("manifest")
            if not isinstance(current_features, dict) or not isinstance(
                current_manifest, dict
            ):
                return
            if current_features.get("filename") != (
                f"features-{current_generation}.int8.bin"
            ):
                return
            if current_manifest.get("filename") != (
                f"manifest-{current_generation}.parquet"
            ):
                return
        except (OSError, json.JSONDecodeError, ValueError):
            return
        if current_generation == self._state.metadata["generation_id"]:
            return
        for artifact_path in (self.feature_path, self.manifest_path):
            try:
                artifact_path.unlink(missing_ok=True)
            except OSError:
                # Another process may still hold the same retired generation.
                pass
