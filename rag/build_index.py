import argparse
import json
import os
import sys
from io import BytesIO
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.retriever import RAGDocument, TextRAGRetriever, load_jsonl_documents


def _caption_to_text(value) -> str:
    if isinstance(value, list):
        return " | ".join(str(v) for v in value[:5])
    return str(value or "")


def _write_ppm(path: str, width: int, height: int, background=(245, 247, 250), rectangles=None):
    rectangles = rectangles or []
    pixels = [[background for _ in range(width)] for _ in range(height)]
    for x1, y1, x2, y2, color in rectangles:
        for y in range(max(0, y1), min(height, y2)):
            for x in range(max(0, x1), min(width, x2)):
                pixels[y][x] = color
    with open(path, "w", encoding="ascii") as f:
        f.write(f"P3\n{width} {height}\n255\n")
        for row in pixels:
            f.write(" ".join(f"{r} {g} {b}" for r, g, b in row) + "\n")


def build_synthetic_documents(output_dir: str) -> List[RAGDocument]:
    """Create a tiny offline image-caption-OCR corpus for smoke tests/demo.

    The images are written as plain PPM files so this smoke-test path does not
    require Pillow. Real VLM inference still uses Pillow through the base
    project requirements.
    """

    image_dir = os.path.join(output_dir, "synthetic_images")
    os.makedirs(image_dir, exist_ok=True)
    specs = [
        {
            "name": "red_bar_chart.ppm",
            "caption": "A simple red bar chart comparing sales in Q1, Q2, and Q3.",
            "ocr": "Quarterly sales: Q1 12, Q2 18, Q3 25. The tallest bar is Q3.",
            "tags": "chart, sales, red bars, quarterly report",
            "rectangles": [(80, 170, 140, 280, (220, 60, 60)), (190, 125, 250, 280, (220, 60, 60)), (300, 70, 360, 280, (220, 60, 60))],
        },
        {
            "name": "blue_login_ui.ppm",
            "caption": "A blue login interface with username, password, and submit button.",
            "ocr": "Username Password Sign in Forgot password",
            "tags": "screenshot, login, UI, form",
            "rectangles": [(120, 105, 390, 150, (220, 235, 255)), (120, 170, 390, 215, (220, 235, 255)), (190, 240, 320, 285, (40, 120, 220))],
        },
        {
            "name": "green_scene.ppm",
            "caption": "A green outdoor scene with a tree, sun, and small house.",
            "ocr": "",
            "tags": "outdoor, house, tree, sun",
            "rectangles": [(0, 220, 512, 320, (120, 190, 95)), (185, 175, 335, 280, (140, 90, 55)), (390, 45, 460, 115, (255, 210, 70))],
        },
        {
            "name": "attention_diagram.ppm",
            "caption": "A transformer attention diagram showing query, key, value, and softmax.",
            "ocr": "Q K V MatMul Scale Softmax MatMul Attention Output",
            "tags": "paper figure, transformer, attention, deep learning",
            "rectangles": [(70, 145, 140, 215, (210, 220, 255)), (180, 145, 250, 215, (210, 220, 255)), (290, 145, 360, 215, (210, 220, 255)), (385, 130, 480, 230, (230, 220, 255))],
        },
    ]
    docs = []
    for idx, spec in enumerate(specs):
        path = os.path.join(image_dir, spec["name"])
        _write_ppm(path, 512, 320, rectangles=spec["rectangles"])
        docs.append(
            RAGDocument(
                doc_id=f"synthetic_{idx}",
                caption=spec["caption"],
                ocr=spec["ocr"],
                source=f"synthetic:{spec['name']}",
                image_path=path,
                metadata={"tags": spec["tags"]},
            )
        )
    return docs

def build_hf_documents(args) -> List[RAGDocument]:
    from datasets import load_dataset
    from PIL import Image

    ds = load_dataset(args.hf_dataset, split=args.hf_split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    docs: List[RAGDocument] = []
    for idx, row in enumerate(ds):
        caption = _caption_to_text(row.get(args.caption_column))
        ocr = str(row.get(args.ocr_column, "")) if args.ocr_column else ""
        doc_id = str(row.get(args.id_column, idx)) if args.id_column else str(idx)
        image_path = ""
        image_value = row.get(args.image_column) if args.image_column else None
        if image_value is not None:
            image_path = os.path.join(image_dir, f"{doc_id}.jpg")
            if isinstance(image_value, Image.Image):
                image_value.convert("RGB").save(image_path)
            elif isinstance(image_value, dict) and image_value.get("bytes"):
                Image.open(BytesIO(image_value["bytes"])).convert("RGB").save(image_path)
            elif isinstance(image_value, str) and os.path.exists(image_value):
                image_path = image_value
            else:
                image_path = ""
        docs.append(
            RAGDocument(
                doc_id=doc_id,
                caption=caption,
                ocr=ocr,
                source=f"{args.hf_dataset}:{args.hf_split}:{doc_id}",
                image_path=image_path,
                metadata={"split": args.hf_split},
            )
        )
    return docs


def write_preview_jsonl(documents: List[RAGDocument], output_dir: str):
    with open(os.path.join(output_dir, "preview_documents.jsonl"), "w", encoding="utf-8") as f:
        for doc in documents[:20]:
            f.write(json.dumps(doc.__dict__, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build a lightweight text RAG index for MiniMind-V.")
    parser.add_argument("--source", choices=["synthetic", "jsonl", "hf"], default="synthetic")
    parser.add_argument("--output_dir", default="rag_index", help="Directory to save the built index.")
    parser.add_argument("--jsonl_path", default="", help="Local JSONL corpus for --source jsonl.")
    parser.add_argument("--hf_dataset", default="nlphuji/flickr30k", help="HF dataset name for --source hf.")
    parser.add_argument("--hf_split", default="test[:200]", help="HF split expression, e.g. test[:200].")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit after loading the split.")
    parser.add_argument("--image_column", default="image")
    parser.add_argument("--caption_column", default="caption")
    parser.add_argument("--ocr_column", default="")
    parser.add_argument("--id_column", default="img_id")
    parser.add_argument("--top_k_test", type=int, default=3)
    parser.add_argument("--test_query", default="transformer attention diagram or chart")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.source == "synthetic":
        documents = build_synthetic_documents(args.output_dir)
    elif args.source == "jsonl":
        if not args.jsonl_path:
            raise ValueError("--jsonl_path is required when --source jsonl")
        documents = load_jsonl_documents(args.jsonl_path)
    else:
        documents = build_hf_documents(args)

    retriever = TextRAGRetriever().fit(documents)
    retriever.save(args.output_dir)
    write_preview_jsonl(documents, args.output_dir)
    print(f"Built RAG index with {len(documents)} documents at {args.output_dir}")
    print("Sample retrieval:")
    for result in retriever.search(args.test_query, top_k=args.top_k_test):
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
