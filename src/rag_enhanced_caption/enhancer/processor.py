import base64
import asyncio
from typing import Dict, Any, Optional, Callable, Awaitable, List
from pathlib import Path
from loguru import logger
import re

# 内部引用
from .json_utils import robust_json_parse
from .image_utils import create_image_resolver
from .vlm_client import get_mime_type
from . import prompts

class MarkdownMultimodalProcessor:
    """
    Markdown 多模态处理器 (Multi-Vector 架构版)
    负责对已被底层语义解析器分离出的 Table, Image 等独立 Chunk 进行批量 VLM 分析，
    并将生成的纯净自然语言摘要注入到 Chunk 的 text_for_embedding 字段中。
    """

    def __init__(
        self, 
        vlm_func: Callable[[str, str, Optional[str]], Awaitable[str]],
        max_concurrency: int = 5
    ):
        self.vlm_func = vlm_func
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def enrich_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        image_resolver: Optional[Callable[[str], Awaitable[Optional[bytes]]]] = None,
        base_dir: str | Path = "."
    ) -> List[Dict[str, Any]]:
        """
        批量增强语义块。仅对 element_type 为 Table 或 Image 的子节点执行 VLM 分析。
        """
        if image_resolver is None:
            image_resolver = create_image_resolver(base_dir)

        # 构建 ID 到 Chunk 的映射，方便查找 Parent 上下文
        chunk_map = {chunk["id"]: chunk for chunk in chunks}
        tasks = []

        for chunk in chunks:
            element_type = chunk["metadata"].get("element_type", "text")
            
            # 如果是文本块，它的 text_for_embedding 就是它自己的纯文本内容
            if element_type == "text":
                chunk["text_for_embedding"] = chunk["content"]
                continue

            # 获取父节点上下文 (Context)
            parent_context = ""
            parent_id = chunk["metadata"].get("parent_id")
            if parent_id and parent_id in chunk_map:
                parent_context = chunk_map[parent_id]["content"]

            if element_type == "Table" or element_type == "html_block" or element_type == "Table KV":
                tasks.append(self._process_table_chunk(chunk, parent_context))
            
            elif element_type == "Image":
                tasks.append(self._process_image_chunk(chunk, parent_context, image_resolver))
            
            elif element_type == "Math Block":
                # 公式可能不需要 VLM 增强，或者用特定的小模型。此处暂时保持原文。
                chunk["text_for_embedding"] = chunk["content"]

        # 并发执行所有 VLM 请求
        if tasks:
            await asyncio.gather(*tasks)

        return chunks

    async def _analyze_element(
        self, 
        user_prompt: str, 
        system_prompt: str, 
        image_bytes: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """统一的 VLM 分析与解析逻辑（受信号量控制并发）"""
        image_base64 = base64.b64encode(image_bytes).decode("utf-8") if image_bytes else None
        try:
            async with self.semaphore:
                raw_response = await self.vlm_func(user_prompt, system_prompt, image_base64, image_bytes)
            
            result = robust_json_parse(raw_response)
            return {
                "summary": result.get("summary", ""),
                "entities": result.get("entities", []),
                "success": True
            }
        except Exception as e:
            logger.exception(f"VLM analysis failed: {e}")
            return {"success": False, "error": str(e), "summary": "", "entities": []}

    async def _process_image_chunk(self, chunk: Dict[str, Any], context: str, image_resolver: Callable) -> None:
        """处理独立图像 Chunk"""
        # 从内容中提取 URL (例如 ![alt](url))
        url_match = re.search(r"!\[.*?\]\((.*?)\)", chunk["content"])
        if not url_match:
            chunk["text_for_embedding"] = chunk["content"]
            return
            
        image_url = url_match.group(1)

        # 1. 优先从 URL 后缀进行初步判断，避免不必要的网络下载。
        # TODO: 目前暂不支持 SVG 格式图片的 VLM 分析（需后续攻克：转码为 PNG 或解析 XML 文本）。
        if image_url.lower().split('?')[0].endswith(".svg"):
            logger.warning(f"Skipping SVG image (detected via URL): {image_url}")
            chunk["text_for_embedding"] = f"Image (SVG format): {image_url}"
            return

        img_bytes = await image_resolver(image_url)
        if not img_bytes:
            chunk["text_for_embedding"] = f"Image placeholder: {image_url}"
            return
            
        # 2. 兜底检查 MIME 类型。大多数 VLM 不支持 SVG，直接发送会导致 400 错误。
        # TODO: 目前暂不支持 SVG 格式图片的 VLM 分析（需后续攻克：转码为 PNG 或解析 XML 文本）。
        mime_type = get_mime_type(img_bytes)
        if mime_type == "image/svg+xml":
            logger.warning(f"Skipping VLM analysis for SVG image: {image_url}. Using fallback text.")
            chunk["text_for_embedding"] = f"Image (SVG format): {image_url}"
            return

        if context:
            user_prompt = prompts.VISION_PROMPT_WITH_CONTEXT.format(context=context, image_path=image_url)
        else:
            user_prompt = prompts.VISION_PROMPT.format(image_path=image_url)

        res = await self._analyze_element(user_prompt, prompts.IMAGE_ANALYSIS_SYSTEM, img_bytes)
        
        # 将高浓度摘要赋值给专门用来做向量的字段，彻底实现信噪分离
        if res["success"] and res["summary"]:
            chunk["text_for_embedding"] = res["summary"]
            if res["entities"]:
                chunk["metadata"]["entities"] = res["entities"]
                chunk["text_for_embedding"] += f"\n[Keywords: {', '.join(res['entities'])}]"
        else:
            chunk["text_for_embedding"] = f"Image content: {image_url}"

    async def _process_table_chunk(self, chunk: Dict[str, Any], context: str) -> None:
        """处理独立表格 Chunk"""
        table_markdown = chunk["content"]

        if context:
            user_prompt = prompts.TABLE_PROMPT_WITH_CONTEXT.format(context=context, table_body=table_markdown)
        else:
            user_prompt = prompts.TABLE_PROMPT.format(table_body=table_markdown)
        
        res = await self._analyze_element(user_prompt, prompts.TABLE_ANALYSIS_SYSTEM)

        if res["success"] and res["summary"]:
            chunk["text_for_embedding"] = res["summary"]
            if res["entities"]:
                chunk["metadata"]["entities"] = res["entities"]
                chunk["text_for_embedding"] += f"\n[Keywords: {', '.join(res['entities'])}]"
        else:
            # 失败兜底：使用截断的表格原文
            chunk["text_for_embedding"] = table_markdown[:500]
