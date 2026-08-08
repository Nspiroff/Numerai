from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import stat
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
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class _ReadOnlyFileLease:
    """Keep a verified store file open without permitting write/delete sharing."""

    def __init__(self, path: Path, label: str) -> None:
        self.path = path
        self.label = label
        self.stream = None
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            import msvcrt

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            create_file = kernel32.CreateFileW
            create_file.argtypes = (
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            )
            create_file.restype = wintypes.HANDLE
            handle = create_file(
                str(path),
                0x80000000,  # GENERIC_READ
                0x00000001,  # FILE_SHARE_READ only
                None,
                3,  # OPEN_EXISTING
                0x00000080,  # FILE_ATTRIBUTE_NORMAL
                None,
            )
            invalid_handle = ctypes.c_void_p(-1).value
            if handle == invalid_handle:
                error = ctypes.WinError(ctypes.get_last_error())
                raise ValueError(
                    f"Cannot acquire immutable feature-store {label} lease: {path}"
                ) from error
            try:
                descriptor = msvcrt.open_osfhandle(
                    int(handle), os.O_RDONLY | getattr(os, "O_BINARY", 0)
                )
            except BaseException:
                kernel32.CloseHandle(handle)
                raise
            self.stream = os.fdopen(descriptor, "rb", buffering=0)
        else:
            try:
                stream = path.open("rb", buffering=0)
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except (OSError, BlockingIOError) as error:
                try:
                    stream.close()
                except (UnboundLocalError, OSError):
                    pass
                raise ValueError(
                    f"Cannot acquire immutable feature-store {label} lease: {path}"
                ) from error
            self.stream = stream

    def read_bytes(self) -> bytes:
        if self.stream is None:
            raise RuntimeError(f"Feature-store {self.label} lease is closed.")
        self.stream.seek(0)
        value = self.stream.read()
        self.stream.seek(0)
        return value

    def fileno(self) -> int:
        if self.stream is None:
            raise RuntimeError(f"Feature-store {self.label} lease is closed.")
        return self.stream.fileno()

    def size_bytes(self) -> int:
        return int(os.fstat(self.fileno()).st_size)

    def sha256(self, *, chunk_size: int = 8 * 1024 * 1024) -> str:
        if self.stream is None:
            raise RuntimeError(f"Feature-store {self.label} lease is closed.")
        digest = hashlib.sha256()
        self.stream.seek(0)
        while chunk := self.stream.read(chunk_size):
            digest.update(chunk)
        self.stream.seek(0)
        return digest.hexdigest()

    def close(self) -> None:
        stream = self.stream
        self.stream = None
        if stream is not None:
            stream.close()


def _require_plain_directory_chain(path: Path) -> None:
    """Reject missing, symbolic, or reparse-point directories in a store path."""

    absolute = Path(os.path.abspath(path))
    chain = [absolute, *absolute.parents]
    for directory in reversed(chain):
        if directory == directory.parent:
            continue
        try:
            inspected = directory.lstat()
        except OSError as error:
            raise ValueError(
                f"Cannot inspect feature-store directory: {directory}"
            ) from error
        attributes = getattr(inspected, "st_file_attributes", 0)
        if directory.is_symlink() or bool(
            attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise ValueError(
                f"Feature-store directory may not be a link or reparse point: {directory}"
            )
        if not stat.S_ISDIR(inspected.st_mode):
            raise ValueError(f"Feature-store path is not a directory: {directory}")


def _require_plain_file(path: Path, label: str) -> None:
    """Reject symbolic, reparse-point, non-regular, or hard-linked artifacts."""

    try:
        inspected = path.lstat()
    except OSError as error:
        raise ValueError(f"Cannot inspect feature-store {label}: {path}") from error
    attributes = getattr(inspected, "st_file_attributes", 0)
    if path.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ):
        raise ValueError(
            f"Feature-store {label} may not be a link or reparse point: {path}"
        )
    if not stat.S_ISREG(inspected.st_mode):
        raise ValueError(f"Feature-store {label} is not a regular file: {path}")
    if inspected.st_nlink != 1:
        raise ValueError(f"Feature-store {label} may not be hard linked: {path}")


def _validate_expected_store_receipt(
    value: Mapping[str, object],
    *,
    receipt_root: Path,
    directory: Path,
) -> dict[str, object]:
    expected_keys = {
        "generation_id",
        "row_count",
        "feature_count",
        "feature_order_sha256",
        "target_column",
        "metadata",
        "manifest",
        "features",
    }
    if set(value) != expected_keys:
        raise ValueError("Expected feature-store receipt keys are malformed.")
    receipt = dict(value)
    for name in ("metadata", "manifest", "features"):
        item = receipt.get(name)
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"Expected feature-store {name} receipt is malformed.")
        relative = Path(str(item.get("path")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Expected feature-store {name} path is not repo-relative.")
        if not _is_sha256(item.get("sha256")):
            raise ValueError(f"Expected feature-store {name} hash is malformed.")
        if type(item.get("size_bytes")) is not int or item["size_bytes"] < 0:
            raise ValueError(f"Expected feature-store {name} size is malformed.")
        expected_path = Path(os.path.abspath(receipt_root / relative))
        if expected_path.parent != directory:
            raise ValueError(f"Expected feature-store {name} path targets another store.")
    return receipt


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
        metadata_sha256: str,
        inventory_identity: Mapping[str, str] | None,
        file_leases: Sequence[_ReadOnlyFileLease],
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
        self.metadata_sha256 = metadata_sha256
        self.inventory_identity = (
            dict(inventory_identity) if inventory_identity is not None else None
        )
        self._file_leases = tuple(file_leases)
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
        for lease in reversed(self._file_leases):
            lease.close()
        self._file_leases = ()


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
        expected_store_receipt: Mapping[str, object] | None = None,
        expected_receipt_root: str | Path | None = None,
        expected_inventory_identity: Mapping[str, str] | None = None,
    ) -> None:
        lexical_directory = Path(directory).expanduser()
        if ".." in lexical_directory.parts:
            raise ValueError("Feature-store directory may not contain parent traversal.")
        self.directory = Path(os.path.abspath(lexical_directory))
        _require_plain_directory_chain(self.directory)
        if expected_store_receipt is None:
            if expected_receipt_root is not None or expected_inventory_identity is not None:
                raise ValueError("Expected store provenance is incomplete.")
            self._expected_store_receipt = None
            self._expected_inventory_identity = None
            self._expected_receipt_root = None
        else:
            if expected_receipt_root is None or expected_inventory_identity is None:
                raise ValueError("Expected store provenance is incomplete.")
            receipt_root = Path(os.path.abspath(Path(expected_receipt_root)))
            self._expected_store_receipt = _validate_expected_store_receipt(
                expected_store_receipt,
                receipt_root=receipt_root,
                directory=self.directory,
            )
            self._expected_receipt_root = receipt_root
            inventory = dict(expected_inventory_identity)
            if set(inventory) != {"path", "git_blob_id", "checkpoint_commit"}:
                raise ValueError("Expected inventory identity is malformed.")
            if not isinstance(inventory["path"], str) or not inventory["path"]:
                raise ValueError("Expected inventory path is malformed.")
            inventory_path = Path(inventory["path"])
            if inventory_path.is_absolute() or ".." in inventory_path.parts:
                raise ValueError("Expected inventory path is not repo-relative.")
            if (
                not isinstance(inventory["git_blob_id"], str)
                or len(inventory["git_blob_id"]) != 40
                or not all(
                    character in "0123456789abcdef"
                    for character in inventory["git_blob_id"]
                )
            ):
                raise ValueError("Expected inventory Git blob is malformed.")
            if (
                not isinstance(inventory["checkpoint_commit"], str)
                or len(inventory["checkpoint_commit"]) != 40
                or not all(
                    character in "0123456789abcdef"
                    for character in inventory["checkpoint_commit"]
                )
            ):
                raise ValueError("Expected inventory checkpoint is malformed.")
            self._expected_inventory_identity = inventory
        self.era_col = str(era_col)
        self.target_col = str(target_col)
        self.id_col = str(id_col) if id_col else None
        self.benchmark_col = str(benchmark_col)
        self._x_cols: tuple[str, ...] | None = None
        self._era_positions: dict[str, np.ndarray] | None = None
        self._loading_generation_id: str | None = None
        self._loading_file_leases: list[_ReadOnlyFileLease] = []
        for attempt in range(3):
            try:
                self._state = self._load_and_validate()
                break
            except Exception:
                self._close_loading_file_leases()
                if attempt == 2 or not self._metadata_generation_changed():
                    raise

    def _lease_file(self, path: Path, label: str) -> _ReadOnlyFileLease:
        lease = _ReadOnlyFileLease(path, label)
        self._loading_file_leases.append(lease)
        return lease

    def _close_loading_file_leases(self) -> None:
        for lease in reversed(self._loading_file_leases):
            lease.close()
        self._loading_file_leases.clear()

    def _release_loading_file_lease(self, lease: _ReadOnlyFileLease) -> None:
        lease.close()
        self._loading_file_leases.remove(lease)

    def _metadata_generation_changed(self) -> bool:
        previous = self._loading_generation_id
        if previous is None:
            return False
        metadata_path = self.directory / FEATURE_STORE_METADATA_FILENAME
        try:
            _require_plain_file(metadata_path, "metadata")
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
            _require_plain_directory_chain(self.directory)
            _require_plain_file(metadata_path, "metadata")
            metadata_lease = self._lease_file(metadata_path, "metadata")
            metadata_sha256 = metadata_lease.sha256()
            if self._expected_store_receipt is not None:
                expected_metadata = self._expected_store_receipt["metadata"]
                assert isinstance(expected_metadata, Mapping)
                if metadata_lease.size_bytes() != expected_metadata["size_bytes"]:
                    raise ValueError("Feature-store metadata size differs from inventory.")
                if metadata_sha256 != expected_metadata["sha256"]:
                    raise ValueError("Feature-store metadata SHA-256 differs from inventory.")
            metadata = json.loads(metadata_lease.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
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
        if self._expected_store_receipt is not None:
            for key in (
                "generation_id",
                "row_count",
                "feature_count",
                "feature_order_sha256",
                "target_column",
            ):
                if metadata.get(key) != self._expected_store_receipt[key]:
                    raise ValueError(
                        f"Feature-store metadata.{key} differs from committed inventory."
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
        _require_plain_file(feature_path, "payload")
        _require_plain_file(manifest_path, "manifest")
        feature_lease = self._lease_file(feature_path, "payload")
        manifest_lease = self._lease_file(manifest_path, "manifest")
        if self._expected_store_receipt is not None:
            for name, path in (("features", feature_path), ("manifest", manifest_path)):
                expected_item = self._expected_store_receipt[name]
                assert isinstance(expected_item, Mapping)
                expected_path = Path(
                    os.path.abspath(
                        self._expected_receipt_root / str(expected_item["path"])
                    )
                )
                if expected_path != path:
                    raise ValueError(
                        f"Feature-store {name} path differs from committed inventory."
                    )
        expected_feature_bytes = row_count * feature_count * np.dtype(np.int8).itemsize
        if feature_meta.get("size_bytes") != expected_feature_bytes:
            raise ValueError("Feature-store payload size metadata is inconsistent.")
        if feature_lease.size_bytes() != expected_feature_bytes:
            raise ValueError("Feature-store payload byte size does not match its shape.")
        if manifest_meta.get("size_bytes") != manifest_lease.size_bytes():
            raise ValueError("Feature-store manifest byte size does not match metadata.")
        feature_sha256 = feature_lease.sha256()
        if feature_sha256 != feature_meta["sha256"]:
            raise ValueError("Feature-store payload SHA-256 does not match metadata.")
        if self._expected_store_receipt is not None:
            expected_features = self._expected_store_receipt["features"]
            expected_manifest = self._expected_store_receipt["manifest"]
            assert isinstance(expected_features, Mapping)
            assert isinstance(expected_manifest, Mapping)
            if (
                feature_lease.size_bytes() != expected_features["size_bytes"]
                or feature_sha256 != expected_features["sha256"]
            ):
                raise ValueError("Feature-store payload differs from committed inventory.")
            if (
                manifest_lease.size_bytes() != expected_manifest["size_bytes"]
                or manifest_meta["sha256"] != expected_manifest["sha256"]
            ):
                raise ValueError("Feature-store manifest differs from committed inventory.")
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

        # Metadata is fully copied and externally bound at this point. Release
        # only that lease so an atomic next-generation commit can replace it;
        # retain the exact feature and manifest leases through OOF fitting.
        self._release_loading_file_lease(metadata_lease)

        # The payload and manifest leases were acquired before hashing. Keep
        # them through OOF fitting; the memmap reserves address space without
        # loading the feature payload into RAM.
        assert feature_lease.stream is not None
        try:
            feature_memmap = np.memmap(
                feature_lease.stream,
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
                manifest_lease,
            )
            file_leases = tuple(self._loading_file_leases)
            state = _DiskFeatureStoreState(
                directory=self.directory,
                feature_path=feature_path,
                manifest_path=manifest_path,
                metadata_path=metadata_path,
                metadata=metadata,
                manifest=manifest,
                feature_columns=feature_columns,
                row_offsets=offsets,
                feature_memmap=feature_memmap,
                metadata_sha256=metadata_sha256,
                inventory_identity=self._expected_inventory_identity,
                file_leases=file_leases,
            )
            self._loading_file_leases.clear()
            return state
        except BaseException:
            feature_memmap._mmap.close()
            raise

    def _load_validated_manifest(
        self,
        manifest_path: Path,
        manifest_meta: dict,
        expected_manifest_columns: list[str],
        row_count: int,
        manifest_lease: _ReadOnlyFileLease,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        if manifest_lease.sha256() != manifest_meta["sha256"]:
            raise ValueError("Feature-store manifest SHA-256 does not match metadata.")

        assert manifest_lease.stream is not None
        manifest_lease.stream.seek(0)
        parquet = pq.ParquetFile(manifest_lease.stream)
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

        manifest_lease.stream.seek(0)
        manifest = pd.read_parquet(
            manifest_lease.stream,
            columns=expected_manifest_columns,
        )
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
            "metadata_sha256": self._state.metadata_sha256,
            "feature_sha256": metadata["features"]["sha256"],
            "manifest_sha256": metadata["manifest"]["sha256"],
            "committed_inventory": self._state.inventory_identity,
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
