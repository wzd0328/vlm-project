#!/usr/bin/env python3
"""按 README 下载 MiniMind-V 基础权重和数据集。

脚本优先调用官方 README 使用的 `modelscope download`，避免手写大量文件清单。
如果当前环境没有安装 ModelScope CLI，脚本会给出可复制的安装与下载命令。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class DownloadTask:
    name: str
    command: list[str]
    target: Path
    required: bool = True


def build_tasks(repo_root: Path, dataset: str, released_weights: bool) -> list[DownloadTask]:
    tasks = [
        DownloadTask(
            name="SigLIP2 视觉编码器",
            command=[
                "modelscope",
                "download",
                "--model",
                "gongjy/siglip2-base-p32-256-ve",
                "--local_dir",
                str(repo_root / "model" / "siglip2-base-p32-256-ve"),
            ],
            target=repo_root / "model" / "siglip2-base-p32-256-ve",
        ),
        DownloadTask(
            name="MiniMind 语言模型基座 llm_768.pth",
            command=[
                "modelscope",
                "download",
                "--model",
                "gongjy/minimind-3v-pytorch",
                "llm_768.pth",
                "--local_dir",
                str(repo_root / "out"),
            ],
            target=repo_root / "out" / "llm_768.pth",
        ),
    ]
    if released_weights:
        tasks.append(
            DownloadTask(
                name="MiniMind-V 发布权重（完整 out 目录）",
                command=[
                    "modelscope",
                    "download",
                    "--model",
                    "gongjy/minimind-3v-pytorch",
                    "--local_dir",
                    str(repo_root / "out"),
                ],
                target=repo_root / "out",
                required=False,
            )
        )
    if dataset in {"sft", "all"}:
        tasks.append(
            DownloadTask(
                name="SFT 数据集 sft_i2t.parquet",
                command=[
                    "modelscope",
                    "download",
                    "--dataset",
                    "gongjy/minimind-v_dataset",
                    "sft_i2t.parquet",
                    "--local_dir",
                    str(repo_root / "dataset"),
                ],
                target=repo_root / "dataset" / "sft_i2t.parquet",
            )
        )
    if dataset in {"pretrain", "all"}:
        tasks.append(
            DownloadTask(
                name="Pretrain 数据集 pretrain_i2t.parquet",
                command=[
                    "modelscope",
                    "download",
                    "--dataset",
                    "gongjy/minimind-v_dataset",
                    "pretrain_i2t.parquet",
                    "--local_dir",
                    str(repo_root / "dataset"),
                ],
                target=repo_root / "dataset" / "pretrain_i2t.parquet",
                required=False,
            )
        )
    return tasks


def write_manifest(repo_root: Path, tasks: list[DownloadTask]) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "记录 README 推荐的 MiniMind-V 基础资源下载任务；大文件本身不纳入 git。",
        "tasks": [
            {"name": task.name, "target": str(task.target.relative_to(repo_root)), "command": task.command}
            for task in tasks
        ],
    }
    (repo_root / "download_manifest.minimindv.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 MiniMind-V README 中列出的基础权重和数据集")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", choices=("none", "sft", "pretrain", "all"), default="sft")
    parser.add_argument("--released-weights", action="store_true", help="额外下载完整发布权重到 out")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令并写 manifest，不实际下载")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    tasks = build_tasks(repo_root, args.dataset, args.released_weights)
    write_manifest(repo_root, tasks)

    print("将执行以下下载任务：")
    for task in tasks:
        print(f"- {task.name}: {' '.join(task.command)}")

    if args.dry_run:
        print("dry-run 模式：未实际下载。")
        return 0

    if shutil.which("modelscope") is None:
        print("\n未找到 modelscope CLI。请先安装后重试，例如：")
        print("python -m pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple")
        print("也可以直接复制上方命令手动执行。")
        return 2

    for task in tasks:
        task.target.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n开始下载：{task.name}")
        subprocess.run(task.command, check=True)
    print("\n下载完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
