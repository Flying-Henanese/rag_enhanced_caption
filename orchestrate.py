import asyncio
import os
import sys
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

async def process_document(file_path: str):
    """
    协调脚本:
    1. 读取 Markdown 文件。
    2. 使用 semantic_chunking_standalone 进行语义化分块。
    3. 遍历分块，使用 rag_enhanced_caption 为其中的图片/表格添加 AI 增强注释。
    """
    
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
    processor = MarkdownMultimodalProcessor(vlm_func=vlm_client)
    
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
    
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in enhanced_chunks:
            # 使用分块器本身添加的标题或者块标识分隔 (原本已经包含了标题级联)
            f.write(chunk["content"])
            f.write("\n\n---\n\n")  # 人为添加一个块分隔符便于肉眼观察
            
    logger.info(f"Workflow complete! Enhanced markdown saved to: {output_path}")
    
    # (可选) 此时 enhanced_chunks 列表中的数据，已经可以直接灌入向量数据库了！
    # 例如： vector_db.insert(enhanced_chunks)
    
if __name__ == "__main__":
    # 这里替换为你想要处理的实际 Markdown 文件的路径
    # 请确保该文件里的图片路径(相对路径)是正确的，或者图片是可访问的外部链接/Base64
    TARGET_MD_FILE = "test_resource/高性能文档解析方案 2e2848cda67f8020abf0d58252a28708.md"
    
    asyncio.run(process_document(TARGET_MD_FILE))
