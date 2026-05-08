import asyncio
import json
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# --- 环境配置 ---
root_dir = Path(__file__).resolve().parent.parent
env_path = root_dir / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loaded configuration from: {env_path}")
else:
    logger.warning(f"No .env file found at {env_path}, relying on system environment variables.")

sys.path.insert(0, str(root_dir))

from rag_enhanced_caption.enhancer.processor import MarkdownMultimodalProcessor  # noqa: E402
from rag_enhanced_caption.enhancer.vlm_client import create_default_vlm_client  # noqa: E402
from rag_enhanced_caption.chunker.dispatcher import chunk_markdown  # noqa: E402
from rag_enhanced_caption.enhancer.cleaning_utils import compress_and_relink_chunks  # noqa: E402

async def process_document(file_path: str):
    """
    Parent-Child 模式文档处理流水线（结构化新版）：
    1. 语义分块（输出干净的元数据，不使用丑陋的正则）。
    2. 清洗与过滤（提早剔除空块，重塑上下文关联）。
    3. VLM 增强描述（多模态隔离存储）。
    4. 模拟落地存储格式。
    """
    start_time = time.time()
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return

    logger.info(f"Reading file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # --- 阶段 1: 语义分块 ---
    logger.info("Phase 1: Semantic Chunking (Structured)")
    # 返回的是结构化的 dict，内部包含 header_path, element_type 等
    raw_chunks = chunk_markdown(
        markdown_content=md_content,
        file_id=file_path.stem,
        filename=file_path.name,
        parser_config={"chunk_token_num": 512}
    )

    # --- 阶段 2: 语义清洗与压缩 ---
    logger.info("Phase 2: Semantic Cleaning and Relinking")
    # 提前清洗，过滤空块，并修复断裂的 parent_id 关联
    chunks = compress_and_relink_chunks(raw_chunks)

    # --- 阶段 3: 多模态增强 ---
    logger.info("Phase 3: VLM Enhancement (Multi-Vector Processing)")
    vlm_client = create_default_vlm_client()
    processor = MarkdownMultimodalProcessor(vlm_func=vlm_client, max_concurrency=2)
    base_dir = file_path.parent.absolute()
    
    # 批量增强，VLM 只对 Table/Image 生成摘要，写入 text_for_embedding
    # 此时发给 VLM 的已经是脱水后的纯净 Markdown
    enriched_chunks = await processor.enrich_chunks(chunks, base_dir=str(base_dir))
    
    # --- 阶段 4: 构建持久化数据展示 ---
    logger.info("Phase 4: Persistence Demonstration")
    
    output_dir = root_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = output_dir / f"{file_path.stem}_index_new.jsonl"
    docstore_path = output_dir / f"{file_path.stem}_docstore_new.jsonl"

    with open(index_path, "w", encoding="utf-8") as f_idx, \
         open(docstore_path, "w", encoding="utf-8") as f_doc:
        
        for chunk in enriched_chunks:
            # Index 仅仅存储专供向量搜索的高浓度纯文本或 VLM 摘要
            # 因为已经提前清洗过，这里的 text_for_embedding 不会再受 HTML 干扰
            f_idx.write(json.dumps({
                "id": chunk["id"],
                "text_for_embedding": chunk["text_for_embedding"],
                "metadata": {"element_type": chunk["metadata"]["element_type"]}
            }, ensure_ascii=False) + "\n")
            
            # Docstore 存储原汁原味的、带前后文的复杂 Markdown 代码 (已经过了脱水)
            f_doc.write(json.dumps({
                "id": chunk["id"],
                "full_content": chunk.get("full_content", chunk["content"]),
                "parent_id": chunk["metadata"].get("parent_id"),
                "header_path": chunk["metadata"].get("header_path", []),
                "element_type": chunk["metadata"].get("element_type", "text"),
                "entities": chunk["metadata"].get("entities", [])
            }, ensure_ascii=False) + "\n")

    logger.info(f"Workflow complete! Files saved to {output_dir}")
    logger.info(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    TARGET_MD_FILE = "test_resource/paddleocr.md"
    asyncio.run(process_document(TARGET_MD_FILE))