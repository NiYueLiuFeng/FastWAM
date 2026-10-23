#!/usr/bin/env python3
"""检查多个 LeRobot 数据集能否一起混合训练。

用法:

    python scripts/check_data_merge_train.py \\
        --dataset /workspace/data/ds_a \\
        --dataset /workspace/data/ds_b \\
        --action-feature actions

退出码:
    0  可以混合训练
    1  存在冲突（逐条列出原因与修复建议）
    2  参数或数据集路径错误

检查规则与 `prepare_custom_training.py generate --dataset A --dataset B`
完全一致（见该脚本的 merge_probes）:

    codebase_version / fps 必须一致
    action、state 的列名、子特征与维度必须一致
    相机集合必须一致（或用 --camera / --intersect-cameras 取交集）
    采样窗口 (num_frames-1)*stride+1 不能超过最短的“最长 episode”

除规则检查外，还会核对 info.json 与 data parquet 的实际列是否对得上，
避免出现“只改了 meta、没改 parquet”的半成品数据集。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn, Sequence

DEFAULT_NUM_FRAMES = 33
DEFAULT_FREQ_RATIO = 4
DEFAULT_STRIDE = 1

READER_V2 = "fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset"
READER_V3 = "fastwam.datasets.lerobot3.robot_video_dataset.RobotVideoDataset"

OK, WARN, FAIL = "OK", "WARN", "FAIL"


def emit(text: str = "") -> None:
    print(text)


def die(message: str, code: int = 2) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


# ---------------------------------------------------------------------------
# metadata helpers
# ---------------------------------------------------------------------------


@dataclass
class VectorSpec:
    key: str  # "default" 或 base.suffix 的 suffix
    dim: int
    lerobot_key: str  # info.json 中真实的特征名


@dataclass
class DatasetSummary:
    path: Path
    version: str
    reader: str
    fps: int
    features: dict[str, Any]
    cameras: dict[str, tuple[int, int, int]]
    action: list[VectorSpec]
    state: list[VectorSpec]
    total_frames: int | None = None
    total_episodes: int | None = None
    max_episode_length: int | None = None
    task_count: int | None = None
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        die(f"文件不存在: {path}")
    except json.JSONDecodeError as exc:
        die(f"无法解析 JSON {path}: {exc}")
    raise AssertionError("unreachable")


def classify_version(info: dict[str, Any], path: Path) -> tuple[str, str]:
    raw = str(info.get("codebase_version", "")).strip()
    if not raw:
        die(f"{path}/meta/info.json 缺少 codebase_version")
    cleaned = raw[1:] if raw.lower().startswith("v") else raw
    match = re.match(r"^(\d+)\.(\d+)", cleaned)
    if not match:
        die(f"无法识别的 codebase_version: {raw!r} ({path})")
    major, minor = int(match.group(1)), int(match.group(2))
    if (major, minor) in {(2, 0), (2, 1)}:
        return f"{major}.{minor}", READER_V2
    if (major, minor) == (3, 0):
        return "3.0", READER_V3
    die(f"不支持的 LeRobot codebase_version {raw!r} ({path})；仅支持 2.0 / 2.1 / 3.0")
    raise AssertionError("unreachable")


def parse_camera_shape(feature: dict[str, Any], key: str) -> tuple[int, int, int]:
    """Return raw shape as CHW regardless of the metadata's HWC/CHW convention."""
    shape = feature.get("shape")
    names = feature.get("names")
    if shape is None:
        info = feature.get("info") or {}
        height = info.get("video.height")
        width = info.get("video.width")
        channels = info.get("video.channels", 3)
        if height is None or width is None:
            raise ValueError(f"相机特征 {key!r} 缺少 shape 信息")
        shape = [height, width, channels]
        names = ["height", "width", "channels"]

    shape = [int(v) for v in shape]
    if len(shape) != 3:
        raise ValueError(f"相机特征 {key!r} 的 shape 必须是 3 维 (HWC 或 CHW)，实际为 {shape}")

    if names:
        lowered = [str(n).lower() for n in names]
        channel = "channels" if "channels" in lowered else ("channel" if "channel" in lowered else None)
        if "height" in lowered and "width" in lowered and channel is not None:
            return (
                shape[lowered.index(channel)],
                shape[lowered.index("height")],
                shape[lowered.index("width")],
            )

    if shape[-1] in (1, 3, 4) and shape[0] not in (1, 3, 4):
        height, width, channels = shape
    elif shape[0] in (1, 3, 4):
        channels, height, width = shape
    else:
        raise ValueError(
            f"无法判断相机特征 {key!r} 的布局: shape={shape}, names={names}"
        )
    return channels, height, width


def _vector_dim(feature: dict[str, Any], key: str) -> int:
    shape = feature.get("shape")
    if shape is None:
        return 1
    shape = [int(v) for v in shape]
    if len(shape) == 0:
        return 1
    if len(shape) > 1:
        raise ValueError(f"特征 {key!r} 必须是 1 维向量，实际 shape={shape}")
    return shape[0]


def _vector_feature_names(features: dict[str, Any]) -> list[str]:
    names = []
    for key, feature in features.items():
        shape = feature.get("shape")
        if shape is not None and len(shape) == 1:
            names.append(key)
    return sorted(names)


def parse_vector_specs(
    features: dict[str, Any],
    base_key: str,
    label: str,
    flag: str,
) -> list[VectorSpec]:
    if base_key in features:
        return [VectorSpec("default", _vector_dim(features[base_key], base_key), base_key)]

    prefix = base_key + "."
    specs = [
        VectorSpec(key[len(prefix):], _vector_dim(feature, key), key)
        for key, feature in features.items()
        if key.startswith(prefix)
    ]
    if not specs:
        raise ValueError(
            f"找不到 {label} 特征 {base_key!r}；"
            f"若列名不是标准命名，请用 {flag} 指定（当前 1 维特征: "
            f"{_vector_feature_names(features)}）"
        )
    return specs


def _fmt_specs(specs: Sequence[VectorSpec]) -> str:
    parts = []
    for spec in specs:
        parts.append(f"{spec.lerobot_key}[{spec.dim}]" if spec.key == "default"
                     else f"{spec.lerobot_key}(key={spec.key})[{spec.dim}]")
    return ", ".join(parts) if parts else "-"


def _specs_tuple(specs: Sequence[VectorSpec]) -> list[tuple[str, int, str]]:
    return [(spec.key, spec.dim, spec.lerobot_key) for spec in specs]


# ---------------------------------------------------------------------------
# per-dataset probes (pyarrow / pandas are optional)
# ---------------------------------------------------------------------------


def _read_episode_lengths(root: Path, version: str) -> tuple[list[int] | None, str | None]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        return None, "未安装 pyarrow，跳过 episode 长度检查"

    try:
        if version.startswith("2"):
            path = root / "meta" / "episodes.jsonl"
            if not path.exists():
                return None, "缺少 meta/episodes.jsonl，跳过 episode 长度检查"
            lengths = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if "length" in record:
                        lengths.append(int(record["length"]))
            return lengths, None

        episodes_dir = root / "meta" / "episodes"
        if not episodes_dir.exists():
            return None, "缺少 meta/episodes/，跳过 episode 长度检查"
        lengths = []
        for path in sorted(episodes_dir.rglob("*.parquet")):
            table = pq.read_table(path, columns=["length"])
            lengths.extend(int(v) for v in table.column("length").to_pylist())
        return lengths, None
    except Exception as exc:  # best effort
        return None, f"读取 episode 长度失败: {exc}"


def _read_task_count(root: Path) -> tuple[int | None, str | None]:
    try:
        import pyarrow.parquet as pq  # noqa: F401  (pandas needs it for parquet)
        import pandas as pd
    except ModuleNotFoundError:
        return None, "未安装 pandas/pyarrow，跳过任务文本检查"

    try:
        jsonl = root / "meta" / "tasks.jsonl"
        if jsonl.exists():
            count = sum(1 for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip())
            return count, None
        parquet = root / "meta" / "tasks.parquet"
        if parquet.exists():
            return int(len(pd.read_parquet(parquet))), None
        return None, "缺少 meta/tasks.jsonl 与 meta/tasks.parquet，训练时拿不到任务文本"
    except Exception as exc:
        return None, f"读取任务文本失败: {exc}"


def _read_first_data_columns(root: Path) -> tuple[set[str] | None, str | None]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        return None, "未安装 pyarrow，跳过 parquet 列名核对"

    try:
        files = sorted((root / "data").glob("*/*.parquet"))
        if not files:
            files = sorted((root / "data").rglob("*.parquet"))
        if not files:
            return None, "data/ 下没有 parquet 文件"
        schema = pq.ParquetFile(files[0]).schema_arrow
        return set(schema.names), None
    except Exception as exc:
        return None, f"读取 data parquet 列名失败: {exc}"


def probe_dataset(path: Path, args: argparse.Namespace) -> DatasetSummary:
    info_path = path / "meta" / "info.json"
    if not info_path.exists():
        die(f"{path} 不是 LeRobot 数据集（缺少 meta/info.json）")

    info = _read_json(info_path)
    version, reader = classify_version(info, path)
    features = info.get("features") or {}

    summary = DatasetSummary(
        path=path,
        version=version,
        reader=reader,
        fps=int(info.get("fps", 0) or 0),
        features=features,
        cameras={},
        action=[],
        state=[],
        total_frames=info.get("total_frames"),
        total_episodes=info.get("total_episodes"),
    )

    if summary.fps <= 0:
        summary.problems.append("info.json 的 fps 必须为正整数")

    for key, feature in features.items():
        dtype = str(feature.get("dtype", ""))
        if dtype not in {"image", "video"}:
            continue
        if not key.startswith("observation.images."):
            summary.warnings.append(f"跳过非 observation.images.* 相机特征 {key!r}")
            continue
        try:
            raw_shape = parse_camera_shape(feature, key)
        except ValueError as exc:
            summary.problems.append(str(exc))
            continue
        if raw_shape[0] != 3:
            summary.warnings.append(f"跳过非 RGB 相机特征 {key!r}（channels={raw_shape[0]}）")
            continue
        summary.cameras[key[len("observation.images."):]] = raw_shape

    if not summary.cameras:
        summary.problems.append("数据集中没有任何 RGB 相机（observation.images.*）")

    for label, base, flag in (
        ("action", args.action_feature, "--action-feature"),
        ("state", args.state_feature, "--state-feature"),
    ):
        try:
            specs = parse_vector_specs(features, base, label, flag)
        except ValueError as exc:
            summary.problems.append(str(exc))
            specs = []
        if label == "action":
            summary.action = specs
        else:
            summary.state = specs

    lengths, length_warning = _read_episode_lengths(path, version)
    if length_warning:
        summary.warnings.append(length_warning)
    if lengths:
        summary.max_episode_length = max(lengths)

    task_count, task_warning = _read_task_count(path)
    if task_warning:
        summary.warnings.append(task_warning)
    summary.task_count = task_count
    if task_count == 0:
        summary.problems.append("任务文本为空，无法生成文本嵌入")

    # Cross-check metadata against the actual parquet columns.
    columns, column_warning = _read_first_data_columns(path)
    if column_warning:
        summary.warnings.append(column_warning)
    elif columns is not None:
        for label, specs in (("action", summary.action), ("state", summary.state)):
            for spec in specs:
                if spec.lerobot_key not in columns:
                    summary.problems.append(
                        f"info.json 声明了 {label} 特征 {spec.lerobot_key!r}，"
                        f"但 data parquet 里没有这一列（现有: {sorted(columns)}）。"
                        "数据集可能只改了 meta，请用 "
                        "scripts/adapt_dataset_for_training.py 重新生成"
                    )
        has_video = any(str(f.get("dtype")) == "video" for f in features.values())
        if has_video and "timestamp" not in columns:
            summary.problems.append(
                "有视频特征但 data parquet 缺少 `timestamp` 列，"
                "读取器无法按时间戳取视频帧"
            )
        if "episode_index" not in columns:
            summary.problems.append("data parquet 缺少 `episode_index` 列")

        # 训练读取器按 info.json 的 features 选列，缺任何一列都会在
        # Dataset.from_parquet 里抛 ArrowInvalid。timestamp/episode_index
        # 上面已有更具体的提示，这里只看其余的列。
        declared = {
            key for key, feature in features.items()
            if str(feature.get("dtype")) != "video"
        }
        missing = sorted(declared - columns - {"timestamp", "episode_index"})
        if missing:
            hint = ""
            if "frame_index" in missing and "frame_idx" in columns:
                hint = (
                    "（meta 里叫 frame_index、parquet 里叫 frame_idx；"
                    "用 scripts/fix_dataset_columns.py --dataset ... --write 原地改名）"
                )
            summary.problems.append(
                f"info.json 声明的列在 data parquet 中不存在: {missing}{hint}"
            )

    return summary


def print_dataset(summary: DatasetSummary, index: int, total: int) -> None:
    emit(f"== 数据集 {index}/{total}: {summary.path}")
    emit(f"   version : {summary.version}  ({summary.reader})")
    emit(f"   fps     : {summary.fps}")
    size = []
    if summary.total_frames is not None:
        size.append(f"{summary.total_frames} 帧")
    if summary.total_episodes is not None:
        size.append(f"{summary.total_episodes} episodes")
    if summary.max_episode_length is not None:
        size.append(f"最长 episode {summary.max_episode_length} 帧")
    emit(f"   数据量  : {', '.join(size) if size else '未知'}")
    cameras = ", ".join(
        f"{key} {channels}x{height}x{width}"
        for key, (channels, height, width) in summary.cameras.items()
    )
    emit(f"   相机    : {cameras or '-'}")
    emit(f"   action  : {_fmt_specs(summary.action)}")
    emit(f"   state   : {_fmt_specs(summary.state)}")
    if summary.task_count is not None:
        emit(f"   任务文本: {summary.task_count} 条")
    for message in summary.problems:
        emit(f"   [FAIL] {message}")
    for message in summary.warnings:
        emit(f"   [WARN] {message}")
    emit()


# ---------------------------------------------------------------------------
# cross-dataset checks
# ---------------------------------------------------------------------------


def _adapt_hint(src: Path, *options: str) -> str:
    # 输出目录名带上具体操作（fps / 改名），避免多条建议互相覆盖；
    # 需要同时改 fps 和列名时，可把两条命令的选项合并成一条。
    tags = []
    for option in options:
        value = option.split()[-1]
        tags.append(re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_"))
    tag = "_".join(tag for tag in tags if tag) or "adapted"
    dst = src.with_name(f"{src.name}_{tag}")
    parts = ["python", "scripts/adapt_dataset_for_training.py",
             "--src", str(src), "--dst", str(dst), *options]
    return " ".join(parts)


def _fmt_camera_sets(summaries: Sequence[DatasetSummary]) -> str:
    return "; ".join(f"{s.name}={sorted(s.cameras)}" for s in summaries)


def cross_check(
    summaries: Sequence[DatasetSummary],
    args: argparse.Namespace,
) -> list[tuple[str, str, list[str]]]:
    """Return entries of (level, message, suggestions)."""
    entries: list[tuple[str, str, list[str]]] = []
    first = summaries[0]

    # dataset identity
    resolved = [str(s.path.resolve()) for s in summaries]
    if len(set(resolved)) != len(resolved):
        entries.append((FAIL, "同一个数据集目录被重复传入", []))

    names = [s.name for s in summaries]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        entries.append((
            WARN,
            f"数据集目录名重复: {duplicated}；生成的配置名会冲突",
            ["生成配置时用 --name 显式指定任务名"],
        ))

    # version
    versions = sorted({s.version for s in summaries})
    if len(versions) > 1:
        detail = "; ".join(f"{s.name}=v{s.version}" for s in summaries)
        entries.append((FAIL, f"codebase_version 不一致: {detail}",
                        ["LeRobot 2.x 与 3.0 不能混合；需先统一转换数据版本"]))

    # fps
    fps_values = sorted({s.fps for s in summaries})
    if len(fps_values) > 1:
        target = first.fps
        detail = "; ".join(f"{s.name}={s.fps}" for s in summaries)
        hints = []
        for s in summaries:
            if s.fps != target:
                if s.fps > target:
                    hints.append(_adapt_hint(s.path, f"--fps {target}"))
                else:
                    hints.append(
                        f"{s.path.name} 的 fps 更低（{s.fps} < {target}），"
                        f"请以 {s.fps} 为目标调整其它数据集: --fps {s.fps}"
                    )
        entries.append((FAIL, f"fps 不一致: {detail}", hints))

    # action / state specs
    for label, key in (("action", "action"), ("state", "state")):
        reference = getattr(first, key)
        mismatched = [s for s in summaries[1:] if _specs_tuple(getattr(s, key)) != _specs_tuple(reference)]
        if not mismatched:
            continue
        detail = "; ".join(f"{s.name}={_fmt_specs(getattr(s, key))}" for s in summaries)
        hints = []
        for s in mismatched:
            specs = getattr(s, key)
            if (len(reference) == 1 and len(specs) == 1
                    and reference[0].key == specs[0].key
                    and reference[0].dim == specs[0].dim
                    and reference[0].lerobot_key != specs[0].lerobot_key):
                hints.append(_adapt_hint(
                    s.path, f"--rename {specs[0].lerobot_key}={reference[0].lerobot_key}"))
            else:
                hints.append(
                    f"{s.path.name} 的 {label} 子特征/维度与基准不同，"
                    "需要先对齐列结构（子特征拆分方式或维度）"
                )
        entries.append((FAIL, f"{label} 列名/维度不一致: {detail}", hints))

    # cameras
    if args.camera:
        missing = [
            f"{s.name}: {[c for c in args.camera if c not in s.cameras]}"
            for s in summaries
            if any(c not in s.cameras for c in args.camera)
        ]
        if missing:
            entries.append((FAIL, f"--camera 指定的相机并非每个数据集都有: {'; '.join(missing)}",
                            ["用 --intersect-cameras 取交集，或先重命名/补充相机列"]))
        selected = [key for key in args.camera if all(key in s.cameras for s in summaries)]
    elif args.intersect_cameras:
        selected = [key for key in first.cameras if all(key in s.cameras for s in summaries)]
        dropped = sorted(set(first.cameras) - set(selected))
        if not selected:
            entries.append((FAIL, "取交集后没有任何公共相机",
                            [f"各数据集的相机: {_fmt_camera_sets(summaries)}"]))
        elif dropped:
            entries.append((WARN, f"--intersect-cameras 会丢弃相机: {dropped}", []))
    else:
        same = all(set(s.cameras) == set(first.cameras) for s in summaries)
        selected = list(first.cameras)
        if not same:
            entries.append((
                FAIL,
                f"相机集合不一致: {_fmt_camera_sets(summaries)}",
                ["用 --intersect-cameras 取交集，或统一相机列名/补齐缺失相机"],
            ))

    if selected:
        for key in selected:
            shapes = {
                s.cameras[key]: s.name for s in summaries if key in s.cameras
            }
            if len(shapes) > 1:
                detail = ", ".join(f"{name}={shape}" for shape, name in shapes.items())
                entries.append((
                    WARN,
                    f"相机 {key!r} 分辨率不同（{detail}）；训练前统一 resize，通常无影响",
                    [],
                ))

    # sampling window
    window = (args.num_frames - 1) * args.global_sample_stride + 1
    if args.num_frames < 2:
        entries.append((FAIL, f"--num-frames 必须 >= 2（当前 {args.num_frames}）", []))
    elif (args.num_frames - 1) % args.action_video_freq_ratio != 0:
        entries.append((
            FAIL,
            f"(num_frames-1)={args.num_frames - 1} 不能被 "
            f"action_video_freq_ratio={args.action_video_freq_ratio} 整除",
            [],
        ))
    else:
        video_frames = (args.num_frames - 1) // args.action_video_freq_ratio
        if video_frames % 4 != 0:
            entries.append((FAIL, f"视频帧数 {video_frames} 必须是 4 的倍数（VAE 约束）", []))

    lengths = [(s.name, s.max_episode_length) for s in summaries if s.max_episode_length is not None]
    if not lengths:
        entries.append((WARN, "无法读取 episode 长度（缺 pyarrow 或 meta），跳过采样窗口检查", []))
    else:
        shortest = min(lengths, key=lambda item: item[1])
        if shortest[1] < window:
            entries.append((
                FAIL,
                f"数据集 {shortest[0]} 的最长 episode 只有 {shortest[1]} 帧，"
                f"小于采样窗口 {window} 帧",
                ["调小 --num-frames / --global-sample-stride，或去掉该数据集"],
            ))
        elif shortest[1] == window:
            entries.append((
                WARN,
                f"数据集 {shortest[0]} 的最长 episode 恰好等于采样窗口 {window} 帧，没有余量",
                [],
            ))

    return entries


def print_cross_report(entries: Sequence[tuple[str, str, list[str]]]) -> tuple[int, int]:
    emit("== 交叉检查")
    if not entries:
        emit("   [OK] 各项规则一致")
    fails = warns = 0
    for level, message, hints in entries:
        if level == FAIL:
            fails += 1
        elif level == WARN:
            warns += 1
        emit(f"   [{level}] {message}")
        for hint in hints:
            emit(f"          -> {hint}")
    emit()
    return fails, warns


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="检查多个 LeRobot 数据集能否一起混合训练（规则与 prepare_custom_training.py 一致）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", action="append", required=True, metavar="PATH",
                        help="LeRobot dataset root (repeatable)")
    parser.add_argument("--action-feature", default="action", metavar="NAME",
                        help="LeRobot action feature/column name (default: action)")
    parser.add_argument("--state-feature", default="observation.state", metavar="NAME",
                        help="LeRobot state feature/column name (default: observation.state)")
    parser.add_argument("--camera", action="append", default=[], metavar="KEY",
                        help="只使用这些相机（去掉 observation.images. 前缀，可重复）")
    parser.add_argument("--intersect-cameras", action="store_true",
                        help="取所有数据集共有的相机（默认要求相机集合完全一致）")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--action-video-freq-ratio", type=int, default=DEFAULT_FREQ_RATIO)
    parser.add_argument("--global-sample-stride", type=int, default=DEFAULT_STRIDE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    paths: list[Path] = []
    for item in args.dataset:
        path = Path(item).expanduser().resolve()
        if not path.is_dir():
            die(f"数据集目录不存在: {path}")
        paths.append(path)
    if len(paths) < 1:
        die("至少需要一个 --dataset")

    summaries = []
    for index, path in enumerate(paths, start=1):
        summary = probe_dataset(path, args)
        summaries.append(summary)
        print_dataset(summary, index, len(paths))

    dataset_failures = sum(1 for s in summaries if s.problems)
    entries = cross_check(summaries, args)
    cross_fails, cross_warns = print_cross_report(entries)

    total_fails = dataset_failures + cross_fails
    if total_fails:
        emit(f"结果: 不可混合训练（{total_fails} 项 FAIL，{cross_warns} 项 WARN）")
        return 1

    emit(f"结果: 可以混合训练（{cross_warns} 项 WARN）")
    emit("建议命令:")
    dataset_args = " ".join(f"--dataset {path}" for path in paths)
    emit(f"  python scripts/prepare_custom_training.py generate {dataset_args} \\")
    emit(f"      --action-feature {args.action_feature} --print-name")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
