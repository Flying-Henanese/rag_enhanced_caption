import asyncio
import os
import re
import sys
import json
import hashlib
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# --- 动态加载环境配置 ---
# 直接定位到项目根目录下的 .env
root_dir = Path(__file__).resolve().parent
env_path = root_dir / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loaded configuration from: {env_path}")
else:
    logger.warning(f"No .env file found at {env_path}, using system environment variables.")

# 将 src 目录加入 sys.path，以便能够导入项目内部的包
sys.path.insert(0, str(root_dir / "src"))

from semantic_chunking_standalone.ragflow_like import semantic_chunk_with_metadata
from rag_enhanced_caption import MarkdownMultimodalProcessor, create_default_vlm_client

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_IMAGE_ANALYSIS_RE = re.compile(r"<image_analysis>(.*?)</image_analysis>", re.DOTALL)
_ENTITY_RE = re.compile(r"-\s+\*\*关键实体\*\*:\s*(.*)")
_SUMMARY_RE = re.compile(r"-\s+\*\*简短摘要\*\*:\s*(.*)")
_CORE_RE = re.compile(r"-\s+\*\*核心总结\*\*:\s*(.*)")


def _extract_title_path(chunk_text: str) -> list[str]:
    for line in chunk_text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            return [p.strip() for p in m.group(2).split("|") if p.strip()]
    return []


def _infer_chunk_type(chunk_text: str) -> str:
    if "<image_analysis>" in chunk_text:
        return "multimodal_enhanced"
    if "![" in chunk_text:
        return "image_related"
    if "| --- |" in chunk_text:
        return "table"
    if "```" in chunk_text:
        return "code_or_json"
    return "text"


def _build_sidecar_record(chunk_record: dict, source_file: Path) -> dict:
    content = chunk_record.get("content", "")

    # 提取 AI 增强信息
    analysis_match = _IMAGE_ANALYSIS_RE.search(content)
    entities = []
    summary = ""
    ai_core_summary = ""

    if analysis_match:
        analysis_body = analysis_match.group(1)
        # 提取实体
        ent_match = _ENTITY_RE.search(analysis_body)
        if ent_match:
            entities = [e.strip() for e in ent_match.group(1).split(",") if e.strip()]
        # 提取摘要
        sum_match = _SUMMARY_RE.search(analysis_body)
        if sum_match:
            summary = sum_match.group(1).strip()
        # 提取核心总结
        core_match = _CORE_RE.search(analysis_body)
        if core_match:
            ai_core_summary = core_match.group(1).strip()

    # 生成清理后的正文 (移除 XML 标签和折叠块)
    content_clean = _IMAGE_ANALYSIS_RE.sub("", content)
    content_clean = re.sub(r"<details>.*?</details>", "", content_clean, flags=re.DOTALL)
    content_clean = content_clean.strip()

    # 计算 Hash
    content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()

    return {
        "id": chunk_record.get("id"),
        "chunk_id": chunk_record.get("chunk_id"),
        "chunk_index": chunk_record.get("chunk_index"),
        "file_id": chunk_record.get("file_id"),
        "filename": chunk_record.get("filename"),
        "source": chunk_record.get("source", source_file.name),
        "title_path": _extract_title_path(content),
        "chunk_type": _infer_chunk_type(content),
        "has_image": "![" in content,
        "has_table": "| --- |" in content,
        "has_ai_analysis": analysis_match is not None,
        "entities": entities,
        "summary": summary,
        "ai_core_summary": ai_core_summary,
        "content": content,
        "content_clean": content_clean,
        "content_hash": content_hash,
        "dedup_key": f"{chunk_record.get('file_id')}_{content_hash[:8]}"
    }


async def process_document(file_path: str):
    """
    协调脚本:
    1. 读取 Markdown 文件。
    2. 使用 semantic_chunking_standalone 进行语义化分块。
    3. 遍历分块，使用 rag_enhanced_caption 为其中的图片/表格添加 AI 增强注释。
    """
    import time
    start_time = time.time()
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return

    logger.info(f"Reading file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # 阶段 1: 语义分块
    logger.info("Phase 1: Semantic Chunking")
    # 我们使用 `semantic_chunk_with_metadata` 获取带元数据的字典块。
    # 默认情况下，它会自动从环境变量加载 Embedding 客户端 (SiliconFlow 等)。
    # parser_config 可以用来调节最大 token 数量
    parser_config = {"chunk_token_num": 512}
    
    chunks = semantic_chunk_with_metadata(
        markdown_content=md_content,
        file_id=file_path.stem,
        filename=file_path.name,
        parser_config=parser_config
    )
    
    logger.info(f"Semantic chunking completed. Generated {len(chunks)} chunks.")

    # 阶段 2: 遍历 Chunk 并做增强注释 (图片/表格 Caption)
    logger.info("Phase 2: RAG Enhanced Captioning for Multimodal Elements")
    
    # 初始化增强注释的 VLM 客户端与处理器
    vlm_client = create_default_vlm_client()
    # 默认最大并发 max_concurrency 为 5，可根据你的 VLM API 限流情况调节
    processor = MarkdownMultimodalProcessor(vlm_func=vlm_client, max_concurrency=2)
    
    base_dir = file_path.parent.absolute()
    
    enhanced_chunks = []
    
    # 逐块进行处理
    # 考虑到有些块可能非常多，这里按顺序逐块 (也可考虑 `asyncio.gather` 并发多块，但内部已对元素并发)
    for idx, chunk_record in enumerate(chunks):
        logger.info(f"Enhancing chunk {idx + 1}/{len(chunks)} (ID: {chunk_record['id']})...")
        raw_text = chunk_record["content"]
        
        # 运行增强处理逻辑
        enriched_text = await processor.enrich_markdown(
            md_content=raw_text,
            base_dir=str(base_dir)
        )
        
        # 将增强后的文本保存回去
        chunk_record["content"] = enriched_text
        enhanced_chunks.append(chunk_record)

    # 阶段 3: 输出/保存结果
    logger.info("Phase 3: Output Generation")
    
    # 作为演示，我们将增强后的文本块重新拼装，或者保存到新的文件
    output_path = file_path.parent / f"{file_path.stem}_enhanced.md"
    chunk_jsonl_path = file_path.parent / f"{file_path.stem}_chunks.jsonl"

    with open(output_path, "w", encoding="utf-8") as md_file, open(chunk_jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for chunk in enhanced_chunks:
            # 保持人类可读 Markdown 输出
            md_file.write(chunk["content"])
            md_file.write("\n\n---\n\n")

            # 旁路输出：每个 chunk 一行 JSON
            sidecar = _build_sidecar_record(chunk, file_path)
            jsonl_file.write(json.dumps(sidecar, ensure_ascii=False) + "\n")

    logger.info(f"Workflow complete! Enhanced markdown saved to: {output_path}")
    logger.info(f"Chunk sidecar JSONL saved to: {chunk_jsonl_path}")
    end_time = time.time()
    logger.info(f"总处理时间: {end_time - start_time:.2f} seconds")
    
    # (可选) 此时 enhanced_chunks 列表中的数据，已经可以直接灌入向量数据库了！
    # 例如： vector_db.insert(enhanced_chunks)
    
if __name__ == "__main__":
    # 这里替换为你想要处理的实际 Markdown 文件的路径
    # 请确保该文件里的图片路径(相对路径)是正确的，或者图片是可访问的外部链接/Base64
    TARGET_MD_FILE = "test_resource/高性能文档解析方案 2e2848cda67f8020abf0d58252a28708.md"
    
    asyncio.run(process_document(TARGET_MD_FILE))
