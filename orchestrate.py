import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from loguru import logger

# --- 环境配置 ---
root_dir = Path(__file__).resolve().parent
env_path = root_dir / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loaded configuration from: {env_path}")
else:
    logger.warning(f"No .env file found at {env_path}, relying on system environment variables.")

sys.path.insert(0, str(root_dir / "src"))

from rag_enhanced_caption import (
    MarkdownMultimodalProcessor,
    create_default_vlm_client,
)
from semantic_chunking_standalone.ragflow_like import (
    semantic_chunk_with_metadata,
)

# --- 正则匹配模式 ---
_IMAGE_ANALYSIS_RE = re.compile(r"<image_analysis>(.*?)</image_analysis>", re.DOTALL)
_ENTITY_RE = re.compile(r"-\s+\*\*关键实体\*\*:\s*(.*)")
_SUMMARY_RE = re.compile(r"-\s+\*\*简短摘要\*\*:\s*(.*)")
_CORE_RE = re.compile(r"-\s+\*\*核心总结\*\*:\s*(.*)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

def _extract_metadata_from_analysis(analysis_body: str) -> Dict[str, Any]:
    """从 AI 分析块中提取结构化元数据"""
    entities = []
    ent_match = _ENTITY_RE.search(analysis_body)
    if ent_match:
        entities = [e.strip() for e in ent_match.group(1).split(",") if e.strip()]
        
    sum_match = _SUMMARY_RE.search(analysis_body)
    summary = sum_match.group(1).strip() if sum_match else ""
    
    core_match = _CORE_RE.search(analysis_body)
    ai_core_summary = core_match.group(1).strip() if core_match else ""
    
    return {
        "entities": entities,
        "summary": summary,
        "ai_core_summary": ai_core_summary
    }

def _build_hierarchical_records(chunk_record: Dict[str, Any], source_file: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    将一个语义块拆分为一个父块(Parent)和多个多模态子块(Children)。
    """
    content = chunk_record.get("content", "")
    base_id = chunk_record.get("id", hashlib.md5(content.encode()).hexdigest()[:12])
    
    # 1. 提取所有 AI 分析块
    analysis_blocks = _IMAGE_ANALYSIS_RE.findall(content)
    # 2. 提取所有图片 URL (用于关联)
    image_urls = re.findall(r"!\[.*?\]\((.*?)\)", content)
    
    # 生成 Parent 记录
    content_clean = _IMAGE_ANALYSIS_RE.sub("", content)
    content_clean = re.sub(r"<details>.*?</details>", "", content_clean, flags=re.DOTALL).strip()
    
    parent_id = f"p_{base_id}"
    common_metadata = {
        "source": source_file.name,
        "file_id": chunk_record.get("file_id"),
        "filename": chunk_record.get("filename"),
    }
    
    parent_record = {
        "id": parent_id,
        "type": "parent",
        "parent_id": None,
        "text_for_embedding": content_clean,  # 仅基于原文文本
        "full_content": content,              # 包含完整 AI 注释的原文
        "metadata": {**common_metadata, "chunk_type": "text"}
    }
    
    # 生成 Children 记录
    child_records = []
    for i, analysis_body in enumerate(analysis_blocks):
        child_id = f"c_{base_id}_{i}"
        meta = _extract_metadata_from_analysis(analysis_body)
        
        # 关联图片 URL (如果能对上的话)
        url = image_urls[i] if i < len(image_urls) else "unknown"
        
        child_records.append({
            "id": child_id,
            "parent_id": parent_id,           # 核心关联：指向父块
            "type": "child",
            "text_for_embedding": meta["summary"], # 仅基于 AI 摘要，避免噪声
            "full_content": f"![image]({url})\n\n{analysis_body}", 
            "metadata": {
                **common_metadata,
                "chunk_type": "multimodal",
                "image_url": url,
                "entities": meta["entities"],
                "ai_summary": meta["summary"],
                "ai_core_summary": meta["ai_core_summary"]
            }
        })
        
    return parent_record, child_records

async def process_document(file_path: str):
    """
    Parent-Child 模式文档处理流水线：
    1. 语义分块。
    2. VLM 增强描述。
    3. 拆分父子记录，输出 Index 和 DocStore 两层数据。
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
    logger.info("Phase 1: Semantic Chunking")
    chunks = semantic_chunk_with_metadata(
        markdown_content=md_content,
        file_id=file_path.stem,
        filename=file_path.name,
        parser_config={"chunk_token_num": 512}
    )

    # --- 阶段 2: 多模态增强 ---
    logger.info("Phase 2: VLM Enhancement")
    vlm_client = create_default_vlm_client()
    processor = MarkdownMultimodalProcessor(vlm_func=vlm_client, max_concurrency=2)
    base_dir = file_path.parent.absolute()
    
    # --- 阶段 3: 构建层级记录并保存 ---
    logger.info("Phase 3: Building Parent-Child Hierarchy & Persistence")
    
    index_path = file_path.parent / f"{file_path.stem}_index.jsonl"
    docstore_path = file_path.parent / f"{file_path.stem}_docstore.jsonl"
    enhanced_md_path = file_path.parent / f"{file_path.stem}_enhanced.md"

    with open(index_path, "w", encoding="utf-8") as f_idx, \
         open(docstore_path, "w", encoding="utf-8") as f_doc, \
         open(enhanced_md_path, "w", encoding="utf-8") as f_md:
        
        for idx, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {idx + 1}/{len(chunks)}...")
            
            # 运行 VLM 增强
            enriched_text = await processor.enrich_markdown(
                md_content=chunk["content"], 
                base_dir=str(base_dir)
            )
            chunk["content"] = enriched_text
            
            # 拆分父子记录
            parent, children = _build_hierarchical_records(chunk, file_path)
            
            # 写入增强后的完整 Markdown (预览用)
            f_md.write(enriched_text + "\n\n---\n\n")
            
            # 处理 Parent
            f_idx.write(json.dumps({
                "id": parent["id"],
                "text_for_embedding": parent["text_for_embedding"],
                "metadata": {"type": "parent", "source": parent["metadata"]["source"]}
            }, ensure_ascii=False) + "\n")
            
            f_doc.write(json.dumps(parent, ensure_ascii=False) + "\n")
            
            # 处理 Children
            for child in children:
                f_idx.write(json.dumps({
                    "id": child["id"],
                    "parent_id": child["parent_id"],
                    "text_for_embedding": child["text_for_embedding"],
                    "metadata": {"type": "child", "source": child["metadata"]["source"]}
                }, ensure_ascii=False) + "\n")
                
                f_doc.write(json.dumps(child, ensure_ascii=False) + "\n")

    logger.info(f"Workflow complete! Files saved to {file_path.parent}")
    logger.info(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    TARGET_MD_FILE = "test_resource/高性能文档解析方案 2e2848cda67f8020abf0d58252a28708.md"
    asyncio.run(process_document(TARGET_MD_FILE))
