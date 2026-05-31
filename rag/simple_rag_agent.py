#!/usr/bin/env python3
"""一个零依赖的 MiniMind-V RAG + Agent 演示。

该示例不调用外部大模型，目的是把 RAG 的核心链路（切块、检索、拼接上下文、生成回答）
和 Agent 的工具编排思想用最小代码跑通，便于在下载真实 MiniMind-V 权重前先理解流程。
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


@dataclass
class DocumentChunk:
    """RAG 检索的最小单元。"""

    source: str
    text: str
    score: float = 0.0


class TinyVectorIndex:
    """基于词频余弦相似度的极简向量索引。

    中文注释：生产环境通常会把文本送入 embedding 模型得到稠密向量，并写入 FAISS、Milvus
    等向量库。为了让 demo 在没有 GPU、没有额外依赖的环境中也能运行，这里使用“分词 + 词频”
    模拟向量化；接口保持与真实向量检索类似，后续替换 embedding 模型时只需要改本类。
    """

    def __init__(self, chunks: Iterable[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self.vectors = [self._vectorize(chunk.text) for chunk in self.chunks]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # 中文注释：同时保留英文单词、数字和连续中文字符，满足 README 中中英混排文本检索。
        return [token.lower() for token in TOKEN_RE.findall(text)]

    @classmethod
    def _vectorize(cls, text: str) -> dict[str, float]:
        vector: dict[str, float] = {}
        for token in cls._tokenize(text):
            vector[token] = vector.get(token, 0.0) + 1.0
        return vector

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        dot = sum(value * right.get(token, 0.0) for token, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def search(self, query: str, top_k: int = 3) -> list[DocumentChunk]:
        # 中文注释：把问题向量化后与所有 chunk 比相似度，取分数最高的片段作为上下文。
        query_vector = self._vectorize(query)
        ranked: list[DocumentChunk] = []
        for chunk, vector in zip(self.chunks, self.vectors):
            ranked.append(DocumentChunk(source=chunk.source, text=chunk.text, score=self._cosine(query_vector, vector)))
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:top_k]


class MiniMindRAG:
    """负责把项目文档切块、建立索引并生成带引用的回答。"""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.index = TinyVectorIndex(self._load_chunks())

    def _load_chunks(self) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for relative_path in ("README.md", "README_en.md"):
            path = self.repo_root / relative_path
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            # 中文注释：按标题和空行做轻量切块；真实项目可按 token 数设置滑动窗口避免上下文过长。
            blocks = [block.strip() for block in re.split(r"\n(?=#{1,6} )|\n\s*\n", text) if block.strip()]
            for index, block in enumerate(blocks):
                if len(block) < 40:
                    continue
                chunks.append(DocumentChunk(source=f"{relative_path}#chunk-{index}", text=block[:1200]))
        return chunks

    def answer(self, question: str, top_k: int = 3) -> str:
        retrieved = self.index.search(question, top_k=top_k)
        context = "\n\n".join(f"[{i}] {chunk.source}\n{chunk.text}" for i, chunk in enumerate(retrieved, 1))
        # 中文注释：此处用模板“生成”答案；接入 MiniMind-V/其他 LLM 时，把 question + context
        # 作为 prompt 输入模型，即可得到自然语言回答。RAG 的关键是让模型先看检索到的私有资料。
        answer_lines = [
            "【RAG回答】我先从项目 README 中检索相关片段，再基于这些片段回答：",
            self._synthesize(question, retrieved),
            "\n【检索上下文】",
            context,
        ]
        return "\n".join(answer_lines)

    @staticmethod
    def _synthesize(question: str, chunks: list[DocumentChunk]) -> str:
        lowered = question.lower()
        if "下载" in question or "download" in lowered or "权重" in question:
            return (
                "基础资源包括 SigLIP2 视觉编码器、MiniMind 语言模型权重以及训练/演示数据。"
                "README 推荐用 modelscope download 分别下载视觉编码器到 model/siglip2-base-p32-256-ve，"
                "下载 llm_768.pth 到 out，并把 sft_i2t.parquet 放到 dataset。"
            )
        if "数据" in question or "dataset" in lowered:
            return (
                "项目数据集采用 Parquet 图文一体格式，核心文件是 sft_i2t.parquet；"
                "可选 pretrain_i2t.parquet 只包含 caption 子集。SFT 文件已合并 Pretrain 子集，"
                "快速复现时可以直接使用 SFT 数据。"
            )
        return "最相关的 README 片段如下，可据此继续接入真实 LLM 生成更自然的回答。"


class MiniMindAgent:
    """一个极简 Agent：根据用户意图选择工具，然后组织最终回复。

    中文注释：Agent 与普通 RAG 的区别在于“先决定做什么”。这里注册两个工具：
    1. rag_search：检索 README 并回答问题；
    2. list_assets：检查本地是否已有 README 要求的权重/数据文件。
    真实 Agent 可以继续增加图片理解、联网下载、训练启动等工具。
    """

    def __init__(self, rag: MiniMindRAG) -> None:
        self.rag = rag
        self.tools: dict[str, Callable[[str], str]] = {
            "rag_search": self._rag_search,
            "list_assets": self._list_assets,
        }

    def run(self, question: str) -> str:
        tool_name = self._plan(question)
        # 中文注释：规划阶段输出工具名，执行阶段调用工具；这就是最小可运行的 ReAct/Tool-use 思想。
        tool_result = self.tools[tool_name](question)
        return f"【Agent计划】选择工具：{tool_name}\n\n{tool_result}"

    @staticmethod
    def _plan(question: str) -> str:
        if any(keyword in question for keyword in ("本地", "文件", "是否下载", "下载完成")):
            return "list_assets"
        return "rag_search"

    def _rag_search(self, question: str) -> str:
        return self.rag.answer(question)

    def _list_assets(self, _: str) -> str:
        expected = [
            self.rag.repo_root / "model" / "siglip2-base-p32-256-ve",
            self.rag.repo_root / "out" / "llm_768.pth",
            self.rag.repo_root / "dataset" / "sft_i2t.parquet",
            self.rag.repo_root / "dataset" / "pretrain_i2t.parquet",
        ]
        lines = ["【本地资源检查】"]
        for path in expected:
            status = "存在" if path.exists() else "缺失"
            lines.append(f"- {path.relative_to(self.rag.repo_root)}：{status}")
        lines.append("\n缺失文件可运行：python scripts/download_minimindv_assets.py --dataset sft")
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMind-V RAG + Agent 零依赖演示")
    parser.add_argument("--question", default="MiniMind-V 需要下载哪些基础资源和数据集？")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    agent = MiniMindAgent(MiniMindRAG(args.repo_root))
    print(agent.run(args.question))


if __name__ == "__main__":
    main()
