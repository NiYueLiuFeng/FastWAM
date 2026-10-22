#!/usr/bin/env python3
"""调整 LeRobot 数据集，使其满足 FastWAM 训练 / 多数据集混合训练的要求。

两种操作可以单独或组合使用:

  1. 重命名特征列: --rename OLD=NEW（可重复）
     例如把动作列 actions 改成 FastWAM 认识的标准名 action。
  2. 调整 fps: --fps N
     按比例抽帧（只支持降低 fps），使多个数据集的 fps 一致。

输出写入 --dst（新建目录），**不修改 --src**。meta/ 与 data/ 会复制并重写，
videos/ 默认用相对路径符号链接（--copy-videos 可改为复制，会占用同样大的空间）。

用法:

    # 统一列名
    python scripts/adapt_dataset_for_training.py \\
        --src /workspace/data/ds_b --dst /workspace/data/ds_b_adapted \\
        --rename actions=action

    # 统一 fps（30 -> 15），同时改列名
    python scripts/adapt_dataset_for_training.py \\
        --src /workspace/data/ds_b --dst /workspace/data/ds_b_15fps \\
        --fps 15 --rename actions=action

    # 只看计划，不写文件
    python scripts/adapt_dataset_for_training.py --src ... --dst ... --fps 15 --dry-run

实现说明:

  * 重命名会同步修改 info.json(features)、stats.json、data/*.parquet 列名、
    meta/episodes/*.parquet 中 `stats/<key>/...` 与 `videos/<key>/...` 列，
    以及 videos/ 下对应的目录（符号链接或复制）。
  * fps 抽帧保留原始 `timestamp` 列：被保留帧的时间戳恰好等于新 fps 下的
    时间戳，读取器按时间戳取视频帧，因此视频文件**无需重新编码**。
  * 3.0 会同步重算 meta/episodes 的 length、dataset_from/to_index 以及
    stats/<feature>/{count,min,max,mean,std}；q01/q99 等其它统计量保持原值。
  * 支持 LeRobot 2.0 / 2.1 / 3.0；2.x 需要 meta/episodes.jsonl。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Sequence

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:  # pragma: no cover - environment issue
    print(f"error: 需要 numpy 与 pyarrow（{exc}）。请使用容器内的 python。", file=sys.stderr)
    raise SystemExit(2)

MARKER = ".adapt_dataset_for_training.json"


def log(message: str) -> None:
    print(message, file=sys.stderr)


def die(message: str, code: int = 2) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


# ---------------------------------------------------------------------------
# metadata helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        die(f"文件不存在: {path}")
    except json.JSONDecodeError as exc:
        die(f"无法解析 JSON {path}: {exc}")
    raise AssertionError("unreachable")


def dump_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4, ensure_ascii=False)
        handle.write("\n")


def classify_version(info: dict[str, Any], root: Path) -> str:
    raw = str(info.get("codebase_version", "")).strip()
    if not raw:
        die(f"{root}/meta/info.json 缺少 codebase_version")
    cleaned = raw[1:] if raw.lower().startswith("v") else raw
    match = re.match(r"^(\d+)\.(\d+)", cleaned)
    if not match:
        die(f"无法识别的 codebase_version: {raw!r} ({root})")
    major, minor = int(match.group(1)), int(match.group(2))
    if (major, minor) in {(2, 0), (2, 1), (3, 0)}:
        return f"{major}.{minor}"
    die(f"不支持的 LeRobot codebase_version {raw!r}；仅支持 2.0 / 2.1 / 3.0")
    raise AssertionError("unreachable")


def discover_data_files(root: Path) -> list[Path]:
    files = sorted((root / "data").glob("*/*.parquet"))
    if not files:
        files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        die(f"{root}/data 下没有 parquet 文件")
    return files


# ---------------------------------------------------------------------------
# fps resampling helpers
# ---------------------------------------------------------------------------


def resample_indices(num_frames: int, old_fps: int, new_fps: int) -> np.ndarray:
    """Nearest-frame decimation, always keeping the episode's last frame."""
    if num_frames <= 0:
        return np.empty(0, dtype=np.int64)
    step = old_fps / new_fps
    indices = np.unique(np.round(np.arange(0, num_frames, step)).astype(np.int64))
    indices = indices[(indices >= 0) & (indices < num_frames)]
    if indices.size == 0:
        indices = np.array([0], dtype=np.int64)
    if int(indices[-1]) != num_frames - 1:
        indices = np.append(indices, num_frames - 1)
    return indices.astype(np.int64)


def episode_segments(episode_index: np.ndarray) -> list[tuple[int, int, int]]:
    """Return [(episode_id, start, end)] for contiguous rows."""
    if episode_index.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(episode_index)) + 1
    bounds = np.concatenate(([0], cuts, [episode_index.size]))
    return [
        (int(episode_index[start]), int(start), int(end))
        for start, end in zip(bounds[:-1], bounds[1:])
    ]


def column_array(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table.column(name).to_pylist(), dtype=np.float64)


def _stats_from_array(arr: np.ndarray, wanted: set[str]) -> dict[str, Any] | None:
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    result: dict[str, Any] = {}
    if "count" in wanted:
        result["count"] = [int(data.shape[0])]
    if data.shape[0] == 0:
        return result
    if "min" in wanted:
        result["min"] = np.nanmin(data, axis=0).tolist()
    if "max" in wanted:
        result["max"] = np.nanmax(data, axis=0).tolist()
    if "mean" in wanted:
        result["mean"] = np.nanmean(data, axis=0).tolist()
    if "std" in wanted:
        result["std"] = np.nanstd(data, axis=0).tolist()
    if "q01" in wanted:
        result["q01"] = np.nanquantile(data, 0.01, axis=0).tolist()
    if "q99" in wanted:
        result["q99"] = np.nanquantile(data, 0.99, axis=0).tolist()
    return result


@dataclass
class Accumulator:
    count: int
    total: np.ndarray
    sumsq: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray


def accumulate(acc: Accumulator | None, arr: np.ndarray) -> Accumulator | None:
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[0] == 0:
        return acc
    if acc is None:
        return Accumulator(
            count=int(data.shape[0]),
            total=np.nansum(data, axis=0),
            sumsq=np.nansum(data * data, axis=0),
            minimum=np.nanmin(data, axis=0),
            maximum=np.nanmax(data, axis=0),
        )
    acc.count += int(data.shape[0])
    acc.total += np.nansum(data, axis=0)
    acc.sumsq += np.nansum(data * data, axis=0)
    acc.minimum = np.minimum(acc.minimum, np.nanmin(data, axis=0))
    acc.maximum = np.maximum(acc.maximum, np.nanmax(data, axis=0))
    return acc


def acc_stat(acc: Accumulator, stat: str) -> list[float]:
    if stat == "count":
        return [acc.count]
    if stat == "min":
        return acc.minimum.tolist()
    if stat == "max":
        return acc.maximum.tolist()
    if stat == "mean":
        return (acc.total / acc.count).tolist()
    if stat == "std":
        variance = acc.sumsq / acc.count - (acc.total / acc.count) ** 2
        return np.sqrt(np.clip(variance, 0.0, None)).tolist()
    raise KeyError(stat)


def set_values(table: pa.Table, name: str, values: Sequence[Any]) -> pa.Table:
    """Replace a column, matching the list/scalar convention of the original."""
    index = table.schema.get_field_index(name)
    if index < 0:
        return table
    field = table.schema.field(index)
    sample = table.column(index)[0].as_py() if table.num_rows else None
    if not isinstance(sample, (list, tuple)):
        values = [
            value[0] if isinstance(value, (list, tuple)) and len(value) == 1 else value
            for value in values
        ]
    return table.set_column(index, field, pa.array(values, type=field.type))


def rename_meta_column(name: str, renames: dict[str, str]) -> str:
    for old, new in renames.items():
        for prefix in ("stats/", "videos/"):
            if name.startswith(f"{prefix}{old}/"):
                return f"{prefix}{new}/" + name[len(f"{prefix}{old}/"):]
    return name


# ---------------------------------------------------------------------------
# data processing (copy + rename + resample in one pass)
# ---------------------------------------------------------------------------


def process_data(
    src: Path,
    dst: Path,
    renames: dict[str, str],
    old_fps: int,
    new_fps: int | None,
    features_needed: set[str],
    stats_needed: dict[str, set[str]],
    dry_run: bool,
) -> tuple[int, dict[int, int], dict[int, dict[str, dict[str, Any]]], dict[str, Accumulator]]:
    """Rewrite data/*.parquet into dst.

    Returns (total_frames, episode_lengths, per_episode_stats, global_accumulators).
    """
    change_fps = new_fps is not None and new_fps != old_fps
    lengths: dict[int, int] = {}
    episode_stats: dict[int, dict[str, dict[str, Any]]] = {}
    accumulators: dict[str, Accumulator] = {}

    total_frames = 0
    offset = 0
    last_episode = -1

    for src_file in discover_data_files(src):
        rel = src_file.relative_to(src)
        dst_file = dst / rel
        table = pq.read_table(src_file)
        if renames:
            table = table.rename_columns(
                [renames.get(name, name) for name in table.schema.names]
            )

        if not change_fps:
            total_frames += table.num_rows
            if not dry_run:
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, dst_file)
            continue

        if "episode_index" not in table.schema.names:
            die(f"{src_file} 缺少 episode_index 列，无法按 episode 抽帧")
        episode_index = table.column("episode_index").to_numpy(zero_copy_only=False)
        segments = episode_segments(episode_index)

        row_parts: list[np.ndarray] = []
        frame_parts: list[np.ndarray] = []
        seg_meta: list[tuple[int, int, int, np.ndarray]] = []
        for episode_id, start, end in segments:
            if episode_id <= last_episode:
                die(
                    f"{src_file} 中 episode_index 非严格递增（{last_episode} -> {episode_id}）。"
                    "该布局（episode 跨文件或乱序）暂不支持按 fps 抽帧"
                )
            last_episode = episode_id
            keep = resample_indices(end - start, old_fps, new_fps)
            row_parts.append(start + keep)
            frame_parts.append(np.arange(keep.size, dtype=np.int64))
            seg_meta.append((episode_id, start, end, keep))

        if not row_parts:
            continue
        rows = np.concatenate(row_parts)
        frame_index = np.concatenate(frame_parts)
        seg_lengths = np.concatenate(
            [np.full(keep.size, keep.size, dtype=np.int64) for _, _, _, keep in seg_meta]
        )

        new_table = table.take(pa.array(rows))
        new_table = set_values(new_table, "frame_index", frame_index)
        new_table = set_values(
            new_table, "index", np.arange(offset, offset + rows.size, dtype=np.int64)
        )
        new_table = set_values(new_table, "is_first", frame_index == 0)
        new_table = set_values(new_table, "is_last", frame_index == seg_lengths - 1)

        for episode_id, start, end, keep in seg_meta:
            subset = table.slice(start, end - start).take(pa.array(keep))
            lengths[episode_id] = int(keep.size)
            episode_stats[episode_id] = {}
            for feature in features_needed:
                if feature not in subset.schema.names:
                    continue
                arr = column_array(subset, feature)
                wanted = stats_needed.get(feature) or set()
                episode_stats[episode_id][feature] = _stats_from_array(arr, wanted) or {}
                accumulators[feature] = accumulate(accumulators.get(feature), arr)

        total_frames += int(rows.size)
        offset += int(rows.size)

        if not dry_run:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(new_table, dst_file)

    return total_frames, lengths, episode_stats, accumulators


def process_data_v2(
    src: Path,
    dst: Path,
    renames: dict[str, str],
    old_fps: int,
    new_fps: int | None,
    features_needed: set[str],
    dry_run: bool,
) -> tuple[int, dict[int, int], dict[str, Accumulator]]:
    """2.x layout: one episode per parquet file."""
    change_fps = new_fps is not None and new_fps != old_fps
    lengths: dict[int, int] = {}
    accumulators: dict[str, Accumulator] = {}
    total_frames = 0
    offset = 0

    for src_file in discover_data_files(src):
        table = pq.read_table(src_file)
        if renames:
            table = table.rename_columns(
                [renames.get(name, name) for name in table.schema.names]
            )

        match = re.search(r"episode_(\d+)", src_file.name)
        if "episode_index" in table.schema.names and table.num_rows:
            episode_id = int(table.column("episode_index")[0].as_py())
        elif match:
            episode_id = int(match.group(1))
        else:
            episode_id = len(lengths)

        if change_fps:
            keep = resample_indices(table.num_rows, old_fps, new_fps)
            new_table = table.take(pa.array(keep))
            new_table = set_values(
                new_table, "frame_index", np.arange(keep.size, dtype=np.int64)
            )
            new_table = set_values(
                new_table, "index", np.arange(offset, offset + keep.size, dtype=np.int64)
            )
            new_table = set_values(new_table, "is_first", np.arange(keep.size) == 0)
            new_table = set_values(
                new_table, "is_last", np.arange(keep.size) == keep.size - 1
            )
            lengths[episode_id] = int(keep.size)
            for feature in features_needed:
                if feature not in new_table.schema.names:
                    continue
                arr = column_array(new_table, feature)
                accumulators[feature] = accumulate(accumulators.get(feature), arr)
            total_frames += int(keep.size)
            offset += int(keep.size)
        else:
            new_table = table
            total_frames += table.num_rows
            lengths[episode_id] = table.num_rows

        if not dry_run:
            dst_file = dst / src_file.relative_to(src)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(new_table, dst_file)

    return total_frames, lengths, accumulators


# ---------------------------------------------------------------------------
# meta processing
# ---------------------------------------------------------------------------


def copy_meta(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        return
    src_meta = src / "meta"
    dst_meta = dst / "meta"
    for path in src_meta.rglob("*"):
        if path.is_dir():
            continue
        target = dst_meta / path.relative_to(src_meta)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def collect_stats_needs(
    src: Path,
    version: str,
    renames: dict[str, str],
) -> tuple[set[str], dict[str, set[str]]]:
    """Features/stats that must be recomputed after resampling."""
    needed: dict[str, set[str]] = {}

    stats_path = src / "meta" / "stats.json"
    if stats_path.exists():
        try:
            stats = load_json(stats_path)
        except SystemExit:
            stats = {}
        for feature, entry in stats.items():
            if isinstance(entry, dict):
                needed.setdefault(renames.get(feature, feature), set()).update(entry.keys())

    if version == "3.0":
        episodes_dir = src / "meta" / "episodes"
        if episodes_dir.exists():
            for path in sorted(episodes_dir.rglob("*.parquet")):
                for name in pq.ParquetFile(path).schema_arrow.names:
                    if not name.startswith("stats/"):
                        continue
                    feature, _, stat = name[len("stats/"):].rpartition("/")
                    needed.setdefault(renames.get(feature, feature), set()).add(stat)

    wanted = {stat for stats in needed.values() for stat in stats}
    wanted &= {"count", "min", "max", "mean", "std", "q01", "q99"}
    return set(needed), {feature: (stats & wanted) for feature, stats in needed.items()}


def update_info_json(
    dst: Path,
    renames: dict[str, str],
    new_fps: int | None,
    total_frames: int | None,
    dry_run: bool,
) -> dict[str, Any]:
    path = dst / "meta" / "info.json"
    info = load_json(path)
    if renames:
        info["features"] = {
            renames.get(key, key): feature for key, feature in info.get("features", {}).items()
        }
    if new_fps is not None:
        info["fps"] = int(new_fps)
        if total_frames is not None:
            info["total_frames"] = int(total_frames)
        # episode 数量不变，splits（episode 区间）保持原样
    if not dry_run:
        dump_json(path, info)
    return info


def update_stats_json(
    dst: Path,
    renames: dict[str, str],
    video_keys: set[str],
    total_frames: int,
    accumulators: dict[str, Accumulator],
    change_fps: bool,
    dry_run: bool,
) -> None:
    path = dst / "meta" / "stats.json"
    if not path.exists():
        return
    stats = load_json(path)
    if renames:
        stats = {renames.get(key, key): value for key, value in stats.items()}
    if change_fps:
        for feature, entry in stats.items():
            if not isinstance(entry, dict):
                continue
            if feature in video_keys:
                if "count" in entry:
                    entry["count"] = [int(total_frames)]
                continue
            acc = accumulators.get(feature)
            if acc is None or acc.count == 0:
                continue
            for stat in list(entry.keys()):
                if stat in {"count", "min", "max", "mean", "std"}:
                    entry[stat] = acc_stat(acc, stat)
    if not dry_run:
        dump_json(path, stats)


def update_episodes_v3(
    dst: Path,
    renames: dict[str, str],
    video_keys: set[str],
    lengths: dict[int, int],
    episode_stats: dict[int, dict[str, dict[str, Any]]],
    change_fps: bool,
    dry_run: bool,
) -> None:
    episodes_dir = dst / "meta" / "episodes"
    if not episodes_dir.exists():
        return

    starts: dict[int, int] = {}
    cursor = 0
    for episode_id in sorted(lengths):
        starts[episode_id] = cursor
        cursor += lengths[episode_id]

    for path in sorted(episodes_dir.rglob("*.parquet")):
        table = pq.read_table(path)
        if renames:
            table = table.rename_columns(
                [rename_meta_column(name, renames) for name in table.schema.names]
            )

        if change_fps:
            episode_ids = [int(value) for value in table.column("episode_index").to_pylist()]
            table = set_values(
                table, "length", [lengths.get(episode_id, 0) for episode_id in episode_ids]
            )
            table = set_values(
                table,
                "dataset_from_index",
                [starts.get(episode_id, 0) for episode_id in episode_ids],
            )
            table = set_values(
                table,
                "dataset_to_index",
                [starts.get(episode_id, 0) + lengths.get(episode_id, 0)
                 for episode_id in episode_ids],
            )

            for name in list(table.schema.names):
                if not name.startswith("stats/"):
                    continue
                feature, _, stat = name[len("stats/"):].rpartition("/")
                values: list[Any] = []
                for episode_id in episode_ids:
                    if feature in video_keys:
                        values.append([lengths.get(episode_id, 0)] if stat == "count" else None)
                    else:
                        per_feature = episode_stats.get(episode_id, {}).get(feature)
                        values.append(per_feature.get(stat) if per_feature else None)
                if any(value is None for value in values):
                    # q01/q99 等无法合并统计的量保持原值
                    continue
                table = set_values(table, name, values)

        if not dry_run:
            pq.write_table(table, path)


def update_episodes_v2(
    dst: Path,
    lengths: dict[int, int],
    dry_run: bool,
) -> None:
    path = dst / "meta" / "episodes.jsonl"
    if not path.exists() or not lengths:
        return
    lines = []
    updated = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            episode_id = record.get("episode_index")
            if episode_id is not None and int(episode_id) in lengths:
                record["length"] = lengths[int(episode_id)]
                updated += 1
            lines.append(json.dumps(record, ensure_ascii=False))
    if not dry_run:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[meta] meta/episodes.jsonl: 更新 {updated} 条 length")


def populate_videos(
    src: Path,
    dst: Path,
    video_renames: dict[str, str],
    copy_videos: bool,
    dry_run: bool,
) -> str:
    src_videos = src / "videos"
    if not src_videos.is_dir():
        return "none"
    if dry_run:
        return "copied" if copy_videos else "symlink"

    dst_videos = dst / "videos"
    if copy_videos:
        shutil.copytree(src_videos, dst_videos)
        for old, new in video_renames.items():
            for path in sorted(dst_videos.rglob(old)):
                if path.is_dir():
                    path.rename(path.with_name(new))
        return "copied"

    # 用相对路径的符号链接：数据集在容器内外挂载点不同（例如宿主机
    # /mnt/.../train_data 对应容器 /workspace/data）时依然可用。
    if not video_renames:
        os.symlink(os.path.relpath(src_videos, dst), dst_videos)
        return "symlink"

    for path in src_videos.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src_videos)
        new_rel = Path(*[video_renames.get(part, part) for part in rel.parts])
        target = dst_videos / new_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.relpath(path, target.parent), target)
    return "symlink"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_renames(items: Sequence[str]) -> dict[str, str]:
    renames: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            die(f"--rename 需要 OLD=NEW 形式，实际为 {item!r}")
        old, new = (part.strip() for part in item.split("=", 1))
        if not old or not new:
            die(f"--rename 的 OLD/NEW 不能为空: {item!r}")
        if old in renames and renames[old] != new:
            die(f"特征 {old!r} 被重命名到多个目标: {renames[old]!r} 与 {new!r}")
        renames[old] = new
    targets = list(renames.values())
    if len(set(targets)) != len(targets):
        die(f"--rename 的目标名重复: {targets}")
    return renames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="重命名 LeRobot 特征列 / 调整 fps，输出到新目录（不修改原数据）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--src", required=True, metavar="PATH", help="源数据集目录")
    parser.add_argument("--dst", required=True, metavar="PATH", help="输出目录（必须不存在）")
    parser.add_argument("--rename", action="append", default=[], metavar="OLD=NEW",
                        help="重命名特征列，可重复，例如 --rename actions=action")
    parser.add_argument("--fps", type=int, default=None,
                        help="目标 fps（只支持降低，按比例抽帧）")
    parser.add_argument("--copy-videos", action="store_true",
                        help="复制 videos/（默认用符号链接，不占额外空间）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写文件")
    parser.add_argument("--force", action="store_true",
                        help="删除上一次本工具生成的 --dst 后重新生成")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    info_path = src / "meta" / "info.json"
    if not info_path.exists():
        die(f"{src} 不是 LeRobot 数据集（缺少 meta/info.json）")
    if src == dst:
        die("--src 与 --dst 不能相同（本工具不修改原数据）")

    info = load_json(info_path)
    version = classify_version(info, src)
    old_fps = int(info.get("fps", 0) or 0)
    if old_fps <= 0:
        die("info.json 的 fps 无效")

    renames = parse_renames(args.rename)
    features = info.get("features") or {}
    for old, new in renames.items():
        if old not in features:
            die(f"info.json 中找不到特征 {old!r}；现有特征: {sorted(features)}")
        if new in features and new not in renames:
            die(f"目标特征名 {new!r} 已存在，无法重命名 {old!r} -> {new!r}")

    new_fps = args.fps
    if new_fps is not None:
        if new_fps <= 0:
            die("--fps 必须是正整数")
        if new_fps > old_fps:
            die(f"只支持降低 fps（当前 {old_fps}，目标 {new_fps}）；"
                "提高 fps 需要重新编码视频，本工具不做")

    change_fps = new_fps is not None and new_fps != old_fps
    if not renames and not change_fps:
        die("没有任何要做的修改（--rename / --fps 都没生效）")

    if dst.exists():
        marker = dst / MARKER
        if not (args.force and marker.exists()):
            die(f"{dst} 已存在；换一个 --dst，或确认是上次本工具的输出后加 --force")
        shutil.rmtree(dst)

    old_video_keys = {
        key for key, feature in features.items()
        if str(feature.get("dtype", "")) in {"image", "video"}
    }
    video_keys = {renames.get(key, key) for key in old_video_keys}
    video_renames = {old: new for old, new in renames.items() if old in old_video_keys}

    features_needed, stats_needed = collect_stats_needs(src, version, renames)

    log(f"[plan] {src} -> {dst}")
    log(f"[plan] version={version} fps={old_fps}"
        + (f" -> {new_fps}" if change_fps else "")
        + (f" renames={renames}" if renames else ""))
    log(f"[plan] data files: {len(discover_data_files(src))}")

    if args.dry_run:
        if change_fps:
            for path in discover_data_files(src):
                if version != "3.0":
                    break
                table = pq.read_table(path, columns=["episode_index"])
                episodes = len(set(table.column("episode_index").to_pylist()))
                log(f"[dry-run] {path.relative_to(src)}: {table.num_rows} 行 / {episodes} episodes")
        log("[dry-run] 未写入任何文件")
        return 0

    dst.mkdir(parents=True)

    if version == "3.0":
        total_frames, lengths, episode_stats, accumulators = process_data(
            src, dst, renames, old_fps, new_fps if change_fps else None,
            features_needed, stats_needed, args.dry_run,
        )
    else:
        total_frames, lengths, accumulators = process_data_v2(
            src, dst, renames, old_fps, new_fps if change_fps else None,
            features_needed, args.dry_run,
        )
        episode_stats = {}

    copy_meta(src, dst, args.dry_run)
    update_info_json(
        dst, renames, new_fps if change_fps else None,
        total_frames if change_fps else None, args.dry_run,
    )
    update_stats_json(
        dst, renames, video_keys, total_frames, accumulators, change_fps, args.dry_run,
    )
    if version == "3.0":
        update_episodes_v3(
            dst, renames, video_keys, lengths, episode_stats, change_fps, args.dry_run,
        )
    else:
        update_episodes_v2(dst, lengths if change_fps else {}, args.dry_run)

    video_mode = populate_videos(
        src, dst, video_renames, args.copy_videos, args.dry_run,
    )

    marker_data = {
        "tool": "scripts/adapt_dataset_for_training.py",
        "src": str(src),
        "version": version,
        "old_fps": old_fps,
        "new_fps": new_fps if change_fps else old_fps,
        "renames": renames,
        "videos": video_mode,
    }
    dump_json(dst / MARKER, marker_data)

    log("")
    log(f"[done] {dst}")
    log(f"  version : {version}")
    if change_fps:
        log(f"  fps     : {old_fps} -> {new_fps}"
            f"（{info.get('total_frames', '?')} -> {total_frames} 帧）")
    if renames:
        log(f"  rename  : {renames}")
    log(f"  videos  : {video_mode}")
    log("")
    log("下一步：用 check_data_merge_train.py 重新校验，例如")
    log(f"  python scripts/check_data_merge_train.py --dataset {src} --dataset {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
