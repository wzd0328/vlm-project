# MiniMind-V-RAG-Agent

本项目在 MiniMind-V 的基础上增加一个轻量级多模态 RAG demo：先从图文知识库中检索 caption/OCR/文档片段，再把检索结果注入 VLM prompt，让模型结合当前图片和外部资料回答问题。

## 1. 数据集调研结论

优先建议使用 Hugging Face 上的小型或可切片图文数据集构建本地知识库：

| 数据集 | 规模/特点 | 推荐用途 |
| --- | --- | --- |
| `nlphuji/flickr30k` | Hugging Face 数据集页显示约 31k 行、每图 5 条 caption、含 image/caption/split/img_id/filename 字段 | 通用图片 caption 检索、image-to-text RAG 入门 |
| `AhmedSSabir/Textual-Image-Caption-Dataset` | 数据集页展示 COCO 文件名、关键词/标签和 caption 字段 | caption + tags 混合检索 |
| `cborg/coco-small` | Hugging Face 上的 COCO 小型子集 | 快速构建 COCO 风格 caption 知识库 |
| Kaggle `RSICD Image Caption Dataset` | 遥感图像 caption 数据，适合做领域 RAG | 遥感/卫星图像问答 |

为了让仓库在无网络环境下也能跑通，我同时实现了 `--source synthetic`，会自动生成 4 条离线图文/OCR 样例，用于 smoke test 和展示 RAG 流程。

## 2. 新增模块

```text
rag/
  __init__.py
  retriever.py        # TF-IDF + cosine 的轻量文本检索器
  prompt_builder.py   # 将检索结果格式化为 VLM prompt
  mm_rag_agent.py     # RAG 调度器：retrieve -> build_prompt
  build_index.py      # 从 synthetic/jsonl/Hugging Face 构建索引
scripts/
  eval_rag_vlm.py     # RAG 版 MiniMind-V 推理入口
```

## 3. 快速开始

### 3.1 离线构建一个最小 RAG 索引

```bash
python rag/build_index.py --source synthetic --output_dir rag_index
```

该命令会：

1. 在 `rag_index/synthetic_images/` 生成 4 张简单示例图；
2. 构建 caption/OCR 文本索引；
3. 保存 `text_rag_index.pkl` 和 `documents.jsonl`；
4. 打印一次示例检索结果。

### 3.2 只查看 RAG 检索和 Prompt，不加载 VLM

```bash
python scripts/eval_rag_vlm.py \
  --index_dir rag_index \
  --image_path rag_index/synthetic_images/attention_diagram.ppm \
  --question "这张图和 Transformer attention 有什么关系？" \
  --min_score 0.000001 \
  --dry_run 1
```

### 3.3 加载 MiniMind-V 权重进行 RAG 问答

请先按主 README 下载视觉编码器和 VLM 权重，然后运行：

```bash
python scripts/eval_rag_vlm.py \
  --index_dir rag_index \
  --image_path rag_index/synthetic_images/attention_diagram.ppm \
  --question "这张图展示了什么模块？请结合检索资料解释。" \
  --load_from model \
  --weight sft_vlm
```

## 4. 使用 Hugging Face 数据集构建索引

以 Flickr30k 的前 200 条为例：

```bash
python rag/build_index.py \
  --source hf \
  --hf_dataset nlphuji/flickr30k \
  --hf_split 'test[:200]' \
  --image_column image \
  --caption_column caption \
  --id_column img_id \
  --output_dir rag_index_flickr30k
```

如果你希望使用本地 JSONL，可以准备如下格式：

```json
{"doc_id":"doc_001","caption":"A transformer attention diagram.","ocr":"Q K V Softmax","source":"paper_page_1","image_path":"images/attention.png","metadata":{"tags":"attention,transformer"}}
```

然后运行：

```bash
python rag/build_index.py --source jsonl --jsonl_path data/rag_docs.jsonl --output_dir rag_index_custom
```

## 5. 工作流程

1. **数据准备**：从 Hugging Face/Kaggle/本地网页资料收集图片、caption、OCR 文本和来源字段。
2. **索引构建**：`rag/build_index.py` 将样本转为 `RAGDocument`，使用 `TextRAGRetriever` 构建 TF-IDF 向量和 cosine 近邻索引。
3. **检索增强**：`MiniMindRAGAgent` 根据用户问题检索 top-k 资料。
4. **Prompt 注入**：`prompt_builder.py` 将来源、score、caption、OCR 组织成 RAG prompt。
5. **VLM 生成**：`scripts/eval_rag_vlm.py` 复用 MiniMind-V 的图片预处理和生成逻辑，把 `<image>` 替换为项目原有图像占位 token 后回答。

## 6. 后续可扩展方向

- 将 `TextRAGRetriever` 替换为 SigLIP/CLIP dense embedding + FAISS，实现真正的 image-to-image 检索。
- 接入 PaddleOCR，把截图/论文图中的文字自动写入 `ocr` 字段。
- 给 `MiniMindRAGAgent` 增加工具选择策略，例如 OCR、裁剪、图表解析、网页搜索。
- 构造 OCR/Chart 小型 SFT 数据，使用 `trainer/train_sft_vlm.py` 对 RAG 风格回答进行指令微调。
