#!/usr/bin/env python3
"""修复 LeRobot 数据集 meta 与 data parquet 之间的列名不一致（原地改 parquet）。

典型症状（训练启动前报错，落在 data.train）::

    ArrowInvalid: No match for FieldRef.Name(frame_index) in observation.state: ...

原因：meta/info.json、meta/stats.json、meta/episodes 里都叫 `frame_index`，
但 data/*.parquet 里这一列叫 `frame_idx`（不同版本的转换脚本写出来的）。
v3 读取器按 info.json 的 features 选列（`columns=list(features)`），
列名对不上就直接抛 ArrowInvalid，训练在构建数据集阶段就挂掉。

本脚本只改 data/*.parquet 的列名，不动 videos/ 与 meta/：

  * `frame_idx` -> `frame_index`（内置别名，仅在 info.json 声明了 frame_index、
    parquet 里只有 frame_idx 时自动生效）；
  * 其它不一致用 --rename OLD=NEW 显式指定（可重复）。

默认只打印计划，加 --write 才真正改写（先写 .tmp 再原子替换，压缩方式沿用原文件）。

用法::

    # 只检查
    python scripts/fix_dataset_columns.py --dataset /workspace/data/ds_a

    # 真正修复（可一次传多个数据集）
    python scripts/fix_dataset_columns.py \\
        --dataset /workspace/data/ds_a --dataset /workspace/data/ds_b --write

    # 其它列名（例如 action 与 actions 在不同文件里混用）
    python scripts/fix_dataset_columns.py --dataset /workspace/data/ds_a \\
        --rename actions=action --write

退出码：0 = 一致或已修复，1 = 仍有声明了却补不出来的列，2 = 参数/读写错误。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence

# old name in parquet -> new name declared by meta/info.json
ALIASES: dict[str, str] = {"frame_idx": "frame_index"}


def log(message: str = "") -> None:
    print(message)


def die(message: str, code: int = 2) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def data_files(root: Path) -> list[Path]:
    """v3 是 data/chunk-*/file-*.parquet，v2.x 是 data/chunk-*/episode_*.parquet。"""
    files = sorted((root / "data").glob("*/*.parquet"))
    if not files:
        files = sorted((root / "data").rglob("*.parquet"))
    return files


def parquet_columns(path: Path, pq: Any) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def declared_columns(features: dict[str, Any]) -> set[str]:
    """读取器实际会从 parquet 里选的列：info.json 里除 video 之外的全部特征。"""
    return {key for key, feature in features.items() if str(feature.get("dtype")) != "video"}


def parse_renames(items: Sequence[str]) -> dict[str, str]:
    renames: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            die(f"--rename 需要 OLD=NEW 形式，实际为 {item!r}")
        old, new = item.split("=", 1)
        old, new = old.strip(), new.strip()
        if not old or not new:
            die(f"--rename 需要 OLD=NEW 形式，实际为 {item!r}")
        if old == new:
            die(f"--rename {item!r} 的新旧名字相同")
        renames[old] = new
    return renames


def plan_renames(
    columns: set[str],
    declared: set[str],
    explicit: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    """返回 (要执行的改名, 无法自动补齐的缺失列)。

    只在“parquet 里有 OLD、没有 NEW，且 info.json 声明了 NEW”时改名，
    其余情况一律不猜。
    """
    mapping: dict[str, str] = {}
    for old, new in {**ALIASES, **explicit}.items():
        if old in columns and old not in declared and new in declared and new not in columns:
            mapping[old] = new
    remaining = sorted(declared - columns - set(mapping.values()))
    return mapping, remaining


def rename_pandas_metadata(schema: Any, mapping: dict[str, str]) -> Any:
    """同步改写 schema 里的 pandas 元数据，避免新旧列名对不上。"""
    metadata = dict(schema.metadata or {})
    raw = metadata.get(b"pandas")
    if not raw:
        return schema
    try:
        info = json.loads(raw)
    except Exception:
        return schema
    for column in info.get("columns", []):
        if column.get("name") in mapping:
            column["name"] = mapping[column["name"]]
        if column.get("field_name") in mapping:
            column["field_name"] = mapping[column["field_name"]]
    info["index_columns"] = [
        mapping.get(name, name) if isinstance(name, str) else name
        for name in info.get("index_columns", [])
    ]
    metadata[b"pandas"] = json.dumps(info).encode("utf-8")
    return schema.with_metadata(metadata)


def rewrite_parquet(path: Path, mapping: dict[str, str], pq: Any) -> None:
    parquet_file = pq.ParquetFile(path)
    table = parquet_file.read()
    compression = "snappy"
    try:
        compression = (parquet_file.metadata.row_group(0).column(0).compression or "snappy").lower()
    except Exception:  # 空文件或列组异形：退回默认压缩
        pass
    if compression == "uncompressed":
        compression = "none"

    table = table.rename_columns([mapping.get(name, name) for name in table.column_names])
    table = table.replace_schema_metadata(rename_pandas_metadata(table.schema, mapping).metadata)
    tmp_path = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp_path, compression=compression)
    os.replace(tmp_path, path)


def fix_dataset(root: Path, explicit: dict[str, str], write: bool, pq: Any) -> bool:
    """处理单个数据集，返回 True 表示最终一致。"""
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        die(f"{root} 不是 LeRobot 数据集（缺少 {info_path}）")
    info = load_json(info_path)
    features = info.get("features") or {}
    declared = declared_columns(features)

    files = data_files(root)
    log(f"== {root}")
    if not files:
        log("   跳过：data/ 下没有 parquet 文件")
        return True
    log(f"   数据文件: {len(files)} 个")

    columns = set(parquet_columns(files[0], pq))
    mapping, remaining = plan_renames(columns, declared, explicit)

    if not mapping and not remaining:
        log("   OK：info.json 声明的列在 parquet 里都有")
        return True

    for old, new in mapping.items():
        hits = [path for path in files if old in parquet_columns(path, pq)]
        log(f"   改名: {old} -> {new}（{len(hits)} 个文件）")
        if write:
            for path in hits:
                rewrite_parquet(path, {old: new}, pq)
                log(f"     rewritten: {path.relative_to(root)}")

    if remaining:
        log(f"   缺失且无法自动补齐: {remaining}")
        log(f"   parquet 现有列: {sorted(columns)}")

    if not write and mapping:
        log("   仅检查模式，未改动任何文件（加 --write 执行）")
        return True

    if mapping and write:
        columns = set(parquet_columns(files[0], pq))
        left = sorted(declared - columns)
        if left:
            log(f"   改完后仍缺少: {left}")
            return False
        log("   完成：info.json 与 parquet 列名已一致")

    return not remaining


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="修复 LeRobot 数据集 meta 与 data parquet 的列名不一致（原地修改 parquet）",
    )
    parser.add_argument("--dataset", action="append", default=[], metavar="PATH",
                        help="数据集目录（可重复）")
    parser.add_argument("--rename", action="append", default=[], metavar="OLD=NEW",
                        help="额外的列名映射（可重复）")
    parser.add_argument("--write", action="store_true",
                        help="真正改写 parquet（默认只检查）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dataset:
        die("至少需要一个 --dataset PATH")

    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        die("需要 pyarrow：pip install pyarrow（容器镜像里已预装）")

    explicit = parse_renames(args.rename)

    ok = True
    for item in args.dataset:
        root = Path(item)
        if not root.is_dir():
            die(f"数据集目录不存在: {root}")
        ok = fix_dataset(root, explicit, args.write, pq) and ok

    if not ok:
        log()
        log("仍有无法自动修复的列名问题；如果 parquet 缺的是整列数据，"
            "需要用 scripts/adapt_dataset_for_training.py 从源数据重新生成。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
