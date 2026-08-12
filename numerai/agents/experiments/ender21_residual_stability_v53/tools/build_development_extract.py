"""Build Ender21 development Parquets without reading protected row groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _select_row_groups(source: Path, cutoff: int) -> tuple[list[int], list[dict]]:
    parquet = pq.ParquetFile(source)
    era_index = parquet.schema_arrow.names.index("era")
    selected: list[int] = []
    receipts: list[dict] = []
    stopped = False
    for index in range(parquet.num_row_groups):
        group = parquet.metadata.row_group(index)
        statistics = group.column(era_index).statistics
        if statistics is None or not statistics.has_min_max:
            raise ValueError(f"Row group {index} has no era min/max statistics: {source}")
        first = str(statistics.min)
        last = str(statistics.max)
        if int(last) <= cutoff:
            if stopped:
                raise ValueError("Eligible row groups are not one contiguous prefix.")
            selected.append(index)
            receipts.append(
                {
                    "index": index,
                    "rows": int(group.num_rows),
                    "first_era": first,
                    "last_era": last,
                }
            )
        else:
            stopped = True
    if not selected:
        raise ValueError(f"No row groups end at or before era {cutoff:04d}: {source}")
    return selected, receipts


def _copy_prefix(source: Path, destination: Path, cutoff: int) -> dict:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite Ender21 extract: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(f"Incomplete Ender21 extract already exists: {partial}")

    parquet = pq.ParquetFile(source)
    selected, groups = _select_row_groups(source, cutoff)
    with partial.open("xb") as raw:
        with pq.ParquetWriter(raw, parquet.schema_arrow, compression="zstd") as writer:
            for index in selected:
                table = parquet.read_row_group(index)
                eras = table.column("era").to_pylist()
                if not eras or max(map(int, eras)) > cutoff:
                    raise ValueError(f"Protected era reached in row group {index}: {source}")
                writer.write_table(table, row_group_size=table.num_rows)
        raw.flush()
        os.fsync(raw.fileno())
    os.rename(partial, destination)

    output = pq.ParquetFile(destination)
    era_index = output.schema_arrow.names.index("era")
    max_era = max(
        int(output.metadata.row_group(index).column(era_index).statistics.max)
        for index in range(output.num_row_groups)
    )
    if max_era > cutoff or output.metadata.num_rows != sum(x["rows"] for x in groups):
        raise ValueError(f"Extract validation failed: {destination}")
    return {
        "source": source.as_posix(),
        "destination": destination.as_posix(),
        "selected_source_row_groups": groups,
        "row_count": int(output.metadata.num_rows),
        "row_group_count": int(output.num_row_groups),
        "last_era": f"{max_era:04d}",
        "size_bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numerai-dir", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=861)
    args = parser.parse_args()
    if args.cutoff != 861:
        raise ValueError("This Round-1 discovery builder requires cutoff era 0861.")
    numerai = args.numerai_dir.resolve()
    version = numerai / "v5.3"
    pairs = (
        (
            version / "downsampled_full.parquet",
            version / f"ender21_discovery_full_through_{args.cutoff:04d}.parquet",
        ),
        (
            version / "downsampled_full_benchmark_models.parquet",
            version
            / f"ender21_discovery_benchmark_models_through_{args.cutoff:04d}.parquet",
        ),
    )
    receipts = [_copy_prefix(source, destination, args.cutoff) for source, destination in pairs]
    # The feature file legitimately includes pre-benchmark rows. The modeling
    # pipeline joins by exact id/era and then requires benchmark coverage, so
    # equality of raw row counts would be an invalid requirement here.
    print(json.dumps({"cutoff": args.cutoff, "artifacts": receipts}, indent=2))


if __name__ == "__main__":
    main()
