from typing import Dict, Iterable, List


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def format_retrieval_context(results: Iterable[Dict], max_chars_per_item: int = 360) -> str:
    """Render retrieved documents as compact, source-aware context."""

    lines: List[str] = []
    for rank, item in enumerate(results, start=1):
        caption = _truncate(item.get("caption", ""), max_chars_per_item)
        ocr = _truncate(item.get("ocr", ""), max_chars_per_item)
        source = item.get("source") or item.get("image_path") or item.get("doc_id", "unknown")
        score = item.get("score")
        score_text = f", score={score:.3f}" if isinstance(score, float) else ""
        lines.append(f"[资料{rank}] 来源: {source}{score_text}")
        if caption:
            lines.append(f"Caption: {caption}")
        if ocr:
            lines.append(f"OCR/文本: {ocr}")
        metadata = item.get("metadata") or {}
        if metadata.get("tags"):
            lines.append(f"Tags: {_truncate(str(metadata['tags']), 160)}")
    return "\n".join(lines).strip()


def build_rag_prompt(
    question: str,
    retrieval_results: Iterable[Dict],
    image_token: str = "<image>",
    language: str = "zh",
) -> str:
    """Build a VLM prompt that injects retrieved multimodal knowledge."""

    context = format_retrieval_context(retrieval_results)
    if language == "en":
        instruction = (
            "Answer the user's question based on the image and the retrieved context. "
            "If the context is insufficient, say so clearly. Cite the source ids when useful."
        )
        context_title = "Retrieved context"
        question_title = "Question"
    else:
        instruction = "请结合当前图片和检索资料回答用户问题；如果资料不足，请明确说明，并尽量引用资料来源。"
        context_title = "检索资料"
        question_title = "用户问题"
    context = context or "无相关检索结果。"
    return f"{image_token}\n{instruction}\n\n【{context_title}】\n{context}\n\n【{question_title}】\n{question}"
