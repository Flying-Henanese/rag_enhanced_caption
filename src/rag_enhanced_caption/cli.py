import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

from dotenv import load_dotenv
from loguru import logger

from rag_enhanced_caption.enhancer.processor import MarkdownMultimodalProcessor
from rag_enhanced_caption.enhancer.vlm_client import create_default_vlm_client
from rag_enhanced_caption.chunker.dispatcher import (
    chunk_markdown as semantic_chunk_with_metadata,
)

# --- 环境配置 ---
# 优先加载当前运行目录下的 .env，其次加载脚本所在目录的 .env
load_dotenv(dotenv_path=Path.cwd() / ".env")
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# 文本类元素类型(其余如 Image / Table / Table KV / html_block 视为多模态)
_TEXT_ELEMENT_TYPES = {"text", "", None}


def _build_record(
    chunk: Dict[str, Any], source_file: Path
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """将一个已增强的语义块转换为 (index 记录, docstore 记录)。

    直接采用 chunker + VLM 增强后写入的 ``text_for_embedding``:
    - 文本块:即其纯文本内容
    - 图片 / 表格等多模态块:即 VLM 生成的语义摘要(+ 关键词)
    这样多模态语义才能真正进入向量索引,而不是只剩一个图片路径。

    Args:
        chunk: chunker 产出、经 ``MarkdownMultimodalProcessor`` 增强后的块。
        source_file: 源 Markdown 文件路径,用于回填 source。

    Returns:
        一个二元组 ``(index_record, docstore_record)``。
    """
    content = chunk.get("content", "")
    base_id = chunk.get("id") or hashlib.md5(content.encode()).hexdigest()[:12]
    meta = chunk.get("metadata", {}) or {}
    element_type = meta.get("element_type", "text")

    is_multimodal = element_type not in _TEXT_ELEMENT_TYPES
    rec_type = "child" if is_multimodal else "parent"
    rec_id = f"{'c_' if is_multimodal else 'p_'}{base_id}"

    # 关键:使用增强阶段写好的 text_for_embedding(多模态=VLM 摘要),
    # 而不是从 content 里重新抠;为空时退回原文,保证不丢内容。
    text_for_embedding = chunk.get("text_for_embedding") or content

    image_url = None
    if is_multimodal:
        url_match = re.search(r"!\[.*?\]\((.*?)\)", content)
        if url_match:
            image_url = url_match.group(1)

    # 图片的 content 只是 ![](路径),回表后没有文字可用(检索/重排会把图片误杀)。
    # 故图片的 full_content 拼上 VLM 摘要,让回表有文字表示(图片的唯一文字就是 caption)。
    full_content = content
    if element_type == "Image" and text_for_embedding and text_for_embedding != content:
        full_content = f"{content}\n\n{text_for_embedding}"

    common_metadata = {
        "source": source_file.name,
        "file_id": chunk.get("file_id"),
        "filename": chunk.get("filename"),
    }

    index_record = {
        "id": rec_id,
        "text_for_embedding": text_for_embedding,
        "metadata": {
            "type": rec_type,
            "source": source_file.name,
            "element_type": element_type,
        },
    }

    # parent_id 命名空间修复: chunker 存的 parent_id 是父文本块的序号(chunk_index),
    # 与记录 id(p_{file_id}_chunk_{n}) 不是同一命名空间,直接落盘会导致 child.parent_id
    # 命中不了任何父记录。这里转成父记录的真实 id —— 父块一定是文本块,记录类型为
    # parent,前缀固定 p_ —— 使 docstore 自洽、可直接回表 / AutoMerging。
    raw_parent_id = meta.get("parent_id")
    file_id = chunk.get("file_id")
    resolved_parent_id = (
        f"p_{file_id}_chunk_{raw_parent_id}" if raw_parent_id is not None else None
    )

    docstore_record = {
        "id": rec_id,
        "type": rec_type,
        "parent_id": resolved_parent_id,
        "text_for_embedding": text_for_embedding,
        "full_content": full_content,
        "header_path": meta.get("header_path", []),
        "element_type": element_type,
        "entities": meta.get("entities", []),
        "metadata": {
            **common_metadata,
            "chunk_type": "multimodal" if is_multimodal else "text",
            "element_type": element_type,
            "image_url": image_url,
            # 保留语义块序号,作为 parent_id 之外的桥接备份(可按需用它反查父块)。
            "chunk_index": meta.get("chunk_index"),
        },
    }

    return index_record, docstore_record


async def process_document(file_path: Path, output_dir: Path):
    """
    Parent-Child 模式文档处理流水线：
    1. 语义分块。
    2. VLM 增强描述。
    3. 拆分父子记录，输出 Index 和 DocStore 两层数据。
    """
    start_time = time.time()

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # --- 阶段 1: 语义分块 ---
    logger.info("Phase 1: Semantic Chunking")
    chunks = semantic_chunk_with_metadata(
        markdown_content=md_content,
        file_id=file_path.stem,
        filename=file_path.name,
        parser_config={"chunk_token_num": 512},
    )

    # --- 阶段 2: 多模态增强 ---
    logger.info("Phase 2: VLM Enhancement")
    vlm_client = create_default_vlm_client()
    processor = MarkdownMultimodalProcessor(vlm_func=vlm_client, max_concurrency=2)
    # 图片解析基础路径仍然是 Markdown 文件的相对目录
    base_dir = file_path.parent.absolute()
    chunks = await processor.enrich_chunks(chunks=chunks, base_dir=base_dir)

    # --- 阶段 3: 构建层级记录并保存 ---
    logger.info("Phase 3: Building Parent-Child Hierarchy & Persistence")

    index_path = output_dir / f"{file_path.stem}_index.jsonl"
    docstore_path = output_dir / f"{file_path.stem}_docstore.jsonl"
    enhanced_md_path = output_dir / f"{file_path.stem}_enhanced.md"

    with (
        open(index_path, "w", encoding="utf-8") as f_idx,
        open(docstore_path, "w", encoding="utf-8") as f_doc,
        open(enhanced_md_path, "w", encoding="utf-8") as f_md,
    ):
        for idx, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {idx + 1}/{len(chunks)}...")

            # 增强阶段已写入 text_for_embedding，content 保留原始 Markdown
            enriched_text = chunk["content"]

            # 每个语义块直接落一条记录(多模态块携带 VLM 摘要)
            index_record, docstore_record = _build_record(chunk, file_path)

            # 写入增强后的完整 Markdown (预览用)
            f_md.write(enriched_text + "\n\n---\n\n")

            f_idx.write(json.dumps(index_record, ensure_ascii=False) + "\n")
            f_doc.write(json.dumps(docstore_record, ensure_ascii=False) + "\n")

    logger.info(f"Workflow complete! Files saved to: {output_dir}")
    logger.info("Generated:")
    logger.info(f"  - {enhanced_md_path.name}")
    logger.info(f"  - {index_path.name}")
    logger.info(f"  - {docstore_path.name}")
    logger.info(f"Total time: {time.time() - start_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="RAG Enhanced Caption & Chunking CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python cli.py ./docs/input.md ./output
        """,
    )

    parser.add_argument("input_file", type=str, help="Path to the input Markdown file.")

    parser.add_argument(
        "output_dir", type=str, help="Directory where the output files will be saved."
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_dir)

    try:
        asyncio.run(process_document(input_path, output_path))
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
