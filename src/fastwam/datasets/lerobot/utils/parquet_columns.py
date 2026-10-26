"""Column-selective parquet reads used by the dataset-statistics pass.

Computing normalization stats only needs the action / state columns, but the
regular per-episode read path materializes *every* non-video column of an
episode (camera intrinsics, extrinsics, force/torque, ...).  Datasets that carry
such columns pay for them twice: once in per-row Arrow -> numpy -> tensor
conversions, and once more per random read on a network filesystem.

`iter_bulk_episodes` reads only the requested columns in one sequential pass and
yields per-episode slices into them, which is what
`BaseLerobotDataset.get_dataset_stats` needs.

A dataset class opts in by providing:

    bulk_data_paths() -> list[Path]   # data parquet files, in row order
    bulk_episode_ids() -> list[int]   # episode id of every nested episode index

Anything that cannot be read this way raises `UnsupportedBulkRead`, and the
caller falls back to the regular per-episode path.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np


class UnsupportedBulkRead(RuntimeError):
    """The dataset layout cannot be read with the bulk (column-selective) path."""


def _list_width(column) -> int:
    """Number of elements per row of a list column; all rows must agree."""
    import pyarrow as pa

    if isinstance(column, pa.FixedSizeListArray):
        return int(column.type.list_size)
    widths = np.diff(column.offsets.to_numpy())
    if widths.size == 0:
        return 0
    width = int(widths[0])
    if np.any(widths != width):
        raise UnsupportedBulkRead("a list column stores rows of different lengths")
    return width


def _column_to_numpy(column) -> np.ndarray:
    """Arrow column -> numpy, keeping the per-row vector dimension when there is one."""
    import pyarrow as pa

    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if isinstance(column, pa.ExtensionArray):
        column = column.storage
    # v2 layouts use a variable-size `list<float>`, v3 a `fixed_size_list<float, D>`.
    if isinstance(column, (pa.FixedSizeListArray, pa.ListArray, pa.LargeListArray)):
        width = _list_width(column)
        values = column.flatten().to_numpy(zero_copy_only=False)
        return values.reshape(len(column), width)
    return column.to_numpy(zero_copy_only=False)


def _episode_slices(episode_index: np.ndarray) -> Dict[int, Tuple[int, int]]:
    """Row ranges of every episode, assuming each episode is one contiguous block."""
    ids = np.asarray(episode_index).reshape(-1)
    if ids.size == 0:
        return {}

    boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [ids.size]))
    slices = {int(ids[start]): (int(start), int(end)) for start, end in zip(starts, ends)}
    if len(slices) != len(np.unique(ids)):
        raise UnsupportedBulkRead("episode rows are not stored in contiguous blocks")
    return slices


def check_bulk_columns(dataset, keys: Iterable[str]) -> None:
    """Raise `UnsupportedBulkRead` unless `keys` can be bulk-read from `dataset`.

    The whole layout is validated here - columns present, every episode stored as
    one contiguous block - so `iter_bulk_episodes` cannot fail halfway through
    and leave the caller with partially accumulated statistics.
    """
    import pyarrow.parquet as pq

    keys = list(dict.fromkeys(keys))
    paths = dataset.bulk_data_paths()
    names = set(pq.read_schema(str(paths[0])).names)
    missing = [key for key in keys if key not in names]
    if missing:
        raise UnsupportedBulkRead(f"data parquet has no columns {missing}")
    if "episode_index" not in names:
        raise UnsupportedBulkRead("data parquet has no `episode_index` column")

    # reading the index column alone costs ~0.3 s for a few hundred thousand rows
    # and warms the file footers the real pass needs anyway
    ids = _column_to_numpy(pq.read_table([str(path) for path in paths], columns=["episode_index"])["episode_index"])
    slices = _episode_slices(ids)
    absent = [int(episode) for episode in dataset.bulk_episode_ids() if int(episode) not in slices]
    if absent:
        raise UnsupportedBulkRead(f"episodes missing from the data parquet: {absent[:3]}")


def iter_bulk_episodes(dataset, keys: Iterable[str]) -> Iterator[Tuple[int, Dict[str, np.ndarray]]]:
    """Yield `(nested_episode_index, {key: rows})` for every episode of one dataset."""
    import pyarrow.parquet as pq

    keys = list(dict.fromkeys(keys))
    episode_ids = dataset.bulk_episode_ids()
    if len(episode_ids) != len(set(episode_ids)):
        raise UnsupportedBulkRead("duplicate episode ids")

    table = pq.read_table(
        [str(path) for path in dataset.bulk_data_paths()],
        columns=[*keys, "episode_index"],
    )
    columns = {key: _column_to_numpy(table[key]) for key in keys}
    slices = _episode_slices(_column_to_numpy(table["episode_index"]))

    for nested, episode_id in enumerate(episode_ids):
        bounds = slices.get(int(episode_id))
        if bounds is None:
            raise UnsupportedBulkRead(f"episode {episode_id} is missing from the data parquet")
        start, end = bounds
        yield nested, {key: values[start:end] for key, values in columns.items()}


def iter_multi_dataset_episodes(
    datasets: Sequence, keys: Iterable[str]
) -> Iterator[Tuple[int, Dict[str, np.ndarray]]]:
    """Yield `(global_episode_index, {key: rows})` across concatenated datasets."""
    keys = list(dict.fromkeys(keys))
    offset = 0
    for dataset in datasets:
        for _, columns in iter_bulk_episodes(dataset, keys):
            yield offset, columns
            offset += 1


__all__: List[str] = [
    "UnsupportedBulkRead",
    "check_bulk_columns",
    "iter_bulk_episodes",
    "iter_multi_dataset_episodes",
]
