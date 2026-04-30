import base64
import asyncio
from typing import Dict, Any, Optional, Callable, Awaitable, List
from pathlib import Path
from loguru import logger

# 内部引用
from .context_extractor import MarkdownContextExtractor
from .json_utils import robust_json_parse
from .image_utils import create_image_resolver
from .markdown_utils import format_as_collapsible_block
from . import prompts

class MarkdownMultimodalProcessor:
    """
    Markdown 多模态处理器 (功能完备版)
    能够自动扫描并处理 Markdown 文档中的所有多模态元素，支持并发处理以提升速度。
    """

    def __init__(
        self, 
        vlm_func: Callable[[str, str, Optional[str]], Awaitable[str]],
        context_extractor: Optional[MarkdownContextExtractor] = None,
        caption_mode: str = "detailed",  # 支持 "detailed" 或 "concise"
        max_concurrency: int = 5         # 默认最大并发数为 5
    ):
        self.vlm_func = vlm_func
        self.extractor = context_extractor or MarkdownContextExtractor()
        self.caption_mode = caption_mode
        # 使用信号量控制同时发往 VLM 的请求数量，防止触发 429 Rate Limit
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def _task_image(self, md_content: str, img_url: str, image_resolver: Callable, target_line: int):
        """包装处理单张图片的异步任务"""
        img_bytes = await image_resolver(img_url)
        if not img_bytes:
            return None
            
        logger.info(f"Processing image: {img_url}")
        res = await self.process_image(md_content, img_url, img_bytes)
        block = format_as_collapsible_block(res)
        if block:
            return (target_line, block)
        return None

    async def _task_table(self, md_content: str, table_md: str, target_idx: int, target_line: int):
        """包装处理单个表格的异步任务"""
        logger.info(f"Processing table at line {target_line}")
        res = await self.process_table(md_content, table_md, target_idx=target_idx)
        block = format_as_collapsible_block(res)
        if block:
            return (target_line, block)
        return None

    async def enrich_markdown(
        self, 
        md_content: str, 
        image_resolver: Optional[Callable[[str], Awaitable[Optional[bytes]]]] = None,
        base_dir: str | Path = "."
    ) -> str:
        """
        在 Markdown 原文的图片/表格下方直接注入 AI 解析折叠块 (并发版本)。
        """
        if image_resolver is None:
            image_resolver = create_image_resolver(base_dir)

        tokens = self.extractor.md.parse(md_content)
        lines = md_content.splitlines()
        
        # 收集所有需要并发执行的任务
        tasks = []

        for i, token in enumerate(tokens):
            # 1. 扫描图片
            if token.type == "inline" and token.children:
                for child in token.children:
                    if child.type == "image":
                        img_url = child.attrGet("src")
                        target_line = -1
                        
                        if token.map:
                            # 方案一：在块元素的行范围内寻找实际包含该图片的行
                            for line_offset in range(token.map[0], token.map[1]):
                                if img_url in lines[line_offset]:
                                    # 将目标行设置为图片所在行的下一行，即放在图片下方
                                    target_line = line_offset + 1
                                    break
                            
                            # 如果没找到 (罕见情况，例如 URL 被编码/转义导致不完全匹配)，回退到块元素的结束行
                            if target_line == -1:
                                target_line = token.map[1]

                        if target_line != -1:
                            tasks.append(self._task_image(md_content, img_url, image_resolver, target_line))

            # 2. 扫描表格
            if token.type == "table_open" and token.map:
                target_line = token.map[0]
                table_md = "\n".join(lines[token.map[0]:token.map[1]])
                tasks.append(self._task_table(md_content, table_md, target_idx=i, target_line=target_line))

        # 并发执行所有解析任务
        results = await asyncio.gather(*tasks)
        
        # 过滤出成功的结果并组装 insertions
        insertions = [res for res in results if res is not None]

        # 按行号倒序插入，防止修改前面的行导致后面的行号错乱
        insertions.sort(key=lambda x: x[0], reverse=True)
        enriched_lines = list(lines)
        for line_idx, block in insertions:
            content = block + "\n"
            # 插入到 target_line 的位置，相当于在这行原来内容的正上方
            if line_idx < len(enriched_lines):
                enriched_lines.insert(line_idx, content)
            else:
                enriched_lines.append(content)

        return "\n".join(enriched_lines)

    async def _analyze_element(
        self, 
        user_prompt: str, 
        system_prompt: str, 
        image_bytes: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """统一的 VLM 分析与解析逻辑（受信号量控制并发）"""
        image_base64 = base64.b64encode(image_bytes).decode("utf-8") if image_bytes else None
        try:
            # 只有这部分实际发送网络请求的代码受 Semaphore 限制
            async with self.semaphore:
                # 传递 image_bytes 给 vlm_func，让其可以动态检测 MIME Type
                raw_response = await self.vlm_func(user_prompt, system_prompt, image_base64, image_bytes)
                
            result = robust_json_parse(raw_response)
            return {
                "enhanced_caption": result.get("detailed_description", ""),
                "entity_info": result.get("entity_info", {}),
                "success": True
            }
        except Exception as e:
            logger.exception(f"VLM analysis failed due to an unhandled exception: {e}")
            return {"success": False, "error": str(e)}

    async def process_image(
        self, 
        md_content: str, 
        image_url: str, 
        image_bytes: bytes,
        entity_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """处理单张图片"""
        context = self.extractor.extract_context(md_content, target_image_url=image_url)
        
        # 选择 Prompt 模板
        is_concise = self.caption_mode == "concise"
        if context:
            tpl = prompts.VISION_PROMPT_CONCISE_WITH_CONTEXT if is_concise else prompts.VISION_PROMPT_WITH_CONTEXT
            user_prompt = tpl.format(context=context, entity_name=entity_name or Path(image_url).stem, image_path=image_url)
        else:
            tpl = prompts.VISION_PROMPT_CONCISE if is_concise else prompts.VISION_PROMPT
            user_prompt = tpl.format(entity_name=entity_name or Path(image_url).stem, image_path=image_url)

        res = await self._analyze_element(user_prompt, prompts.IMAGE_ANALYSIS_SYSTEM, image_bytes)
        res["url"] = image_url
        res["context_used"] = context
        return res

    async def process_table(
        self,
        md_content: str,
        table_markdown: str,
        entity_name: Optional[str] = None,
        target_idx: int = -1
    ) -> Dict[str, Any]:
        """处理单个表格"""
        # 尝试为表格提取语境
        context = ""
        if target_idx != -1:
            context = self.extractor.extract_context(md_content, target_idx=target_idx)

        if context:
            user_prompt = prompts.TABLE_PROMPT_WITH_CONTEXT.format(
                context=context,
                entity_name=entity_name or "table_entity",
                table_body=table_markdown
            )
        else:
            user_prompt = prompts.TABLE_PROMPT.format(
                entity_name=entity_name or "table_entity",
                table_body=table_markdown
            )
        
        return await self._analyze_element(user_prompt, prompts.TABLE_ANALYSIS_SYSTEM)
