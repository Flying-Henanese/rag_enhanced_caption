"""
核心的语义化 Markdown 解析器。

利用 `markdown-it-py` 对文档进行结构化遍历，根据标题层级建立树状上下文路径。
将长段落结合 Embedding 模型进行语义聚类切分，保留图像题注、表格和数学公式的完整性。
目标是提供高精度、不易割裂上下文（Context）的 RAG 知识片段。
"""
from __future__ import annotations

import re
from typing import Any

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin

import logging
from ..nlp import count_tokens
from ..clients import get_default_embedding_client

logger = logging.getLogger(__name__)

from ..utils.md_parser_utils import (
    extract_table_block,
    get_title_path,
    split_text_by_length_and_newline,
)
from ..utils.table_utils import html_table_to_key_value


def _flush_content(
    result: list,
    current_content: list,
    title_stack: list,
    max_length: int,
    embed_fn: Any,
    special_element: str = None,
    allow_split: bool = False,
) -> None:
    """
    刷新并输出积累的文本内容。

    每当解析到新的标题级结构，或者特殊块（表格、图表、列表）完毕时调用。
    它负责：
    1. 给积累的内容打上"标题路径标签"（例如 `# 一级|二级|段落`）
    2. 若文本超长，则触发文本拆分 (`split_text_by_length_and_newline`)，避免超过模型最大 token
    3. 清空 `current_content`
    """
    if not current_content:
        return

    content = "\n".join(current_content).strip()
    if not content:
        current_content.clear()
        return

    # 推断当前属于几级标题（反向遍历，找到第一个非空的标题层级）
    level = next((i + 1 for i in range(5, -1, -1) if title_stack[i]), 1)
    title_path = get_title_path(title_stack)

    # 针对不适合截断的特殊元素（如提取后的表格整体，或者图表公式块）
    if special_element and not allow_split:
        header = f"{'#' * level} {title_path}|{special_element}" if title_path else f"{'#' * level} {special_element}"
        result.extend([header, content, "-" * 10])
    else:
        # 如果当前积攒的内容量超标，必须进行拆分
        if count_tokens(content) > max_length:
            # 引入语义截断模块，进行安全拆分
            chunks = split_text_by_length_and_newline(
                content, max_length, embed_fn=embed_fn, token_count_fn=count_tokens
            )
            for idx, chunk in enumerate(chunks, 1):
                base_header = f"{'#' * level} {title_path}" if title_path else f"{'#' * level}"
                if special_element:
                    header = f"{base_header}|{special_element}|Part {idx}"
                else:
                    header = f"{base_header}|Part {idx}"
                # 将 Header、分块内容、以及块边界分隔符(`----------`)加入结果
                result.extend([header, chunk, "-" * 10])
        else:
            # 未超标，直接作为一个完整的片段输出
            base_header = f"{'#' * level} {title_path}" if title_path else f"{'#' * level}"
            if special_element:
                header = f"{base_header}|{special_element}"
            else:
                header = base_header

            if header:
                result.append(header)
                result.append("")
            result.extend([content, "-" * 10])

    # 刷新完成，清空暂存
    current_content.clear()


def _handle_image_caption(tokens, i, result, current_content, title_stack, max_length, embed_fn):
    """
    专门处理 Markdown 中的图片以及跟随在其后的题注(Caption)。

    尝试探测诸如 `![alt](url)\n图1 示意图` 这样的组合，将图和图名绑定在同一个切片中，
    避免在向量化时题注与图片发生语义断裂。
    
    返回:
        (是否被处理为图片题注对, 下一个 token 索引位置)
    """
    token = tokens[i]
    if token.type != "paragraph_open":
        return False, i

    inline_token = tokens[i + 1]
    if inline_token.type != "inline":
        return False, i

    content = inline_token.content.strip()
    image_pattern = r"^!\[.*?\]\(.*?\)\s*$"
    caption_pattern = r"^(?:Figure|图|Fig\.|表|Table)\s*[\d\w\.]+"

    # 1. 图文混在同一个段落里
    img_match = re.search(r"^(!\[.*?\]\(.*?\))", content)
    if img_match:
        rest = content[img_match.end() :].strip()
        if rest and re.match(caption_pattern, rest, re.IGNORECASE):
            # 将之前积攒的内容刷走，保证图文对独占一个块
            _flush_content(result, current_content, title_stack, max_length, embed_fn)
            current_content.append(content)
            caption_title = rest.split("\n")[0].strip()
            _flush_content(result, current_content, title_stack, max_length, embed_fn, special_element=caption_title)
            return True, i + 3

    # 2. 图片单独成段，题注在下一个段落
    if re.match(image_pattern, content):
        next_p_idx = i + 3
        if next_p_idx + 1 < len(tokens) and tokens[next_p_idx].type == "paragraph_open":
            next_inline = tokens[next_p_idx + 1]
            if next_inline.type == "inline":
                next_content = next_inline.content.strip()
                if re.match(caption_pattern, next_content, re.IGNORECASE):
                    _flush_content(result, current_content, title_stack, max_length, embed_fn)
                    current_content.append(content)
                    current_content.append(next_content)
                    _flush_content(
                        result, current_content, title_stack, max_length, embed_fn, special_element=next_content
                    )
                    return True, i + 6

    # 3. 题注在前一个段落积攒的内容里（罕见，为了兼容处理）
    if current_content and re.match(caption_pattern, content, re.IGNORECASE):
        last_item = current_content[-1].strip()
        if re.match(image_pattern, last_item):
            image_tag = current_content.pop()
            _flush_content(result, current_content, title_stack, max_length, embed_fn)
            current_content.append(image_tag)
            current_content.append(content)
            _flush_content(result, current_content, title_stack, max_length, embed_fn, special_element=content)
            return True, i + 3

    return False, i


def chunk_markdown(
    markdown_content: str, parser_config: dict[str, Any] | None = None, embed_fn: Any | None = None
) -> list[str]:
    """
    基于 AST 的结构化语义切分 Markdown 内容。

    解析 Markdown 抽象语法树（AST），依据树状结构合并标题路径，避免因为块切分导致内容失去上级语境。

    Args:
        markdown_content: 待切分的 Markdown 文本。
        parser_config: 切分参数，例如 max_length (chunk_token_num)。
        embed_fn: 可选。用于句子向量相似度聚类的 Embedding 客户端函数。若为 None 则尝试加载默认环境变量模型。
        
    Returns:
        包含 Markdown 头信息、带完整上下文的切分片段列表。
    """
    parser_config = parser_config or {}
    max_length = int(parser_config.get("chunk_token_num", 512))
    logger.info(f"语义切分开始: max_length={max_length}, content_length={len(markdown_content)}")

    # 延迟加载重型向量化资源，方便在不同环境中优雅降级
    if embed_fn is None:
        logger.info("未提供 embed_fn，尝试从 .env 环境初始化默认客户端...")
        embed_fn = get_default_embedding_client()
        if embed_fn is None:
            logger.warning("默认 Embedding 客户端初始化失败，退化为简单的纯文本字数切分。")

    # 配置并启动 markdown-it，启用常用扩展插件（表格、数学公式）
    md = MarkdownIt("commonmark").enable("table")
    md.use(dollarmath_plugin, allow_space=True, allow_digits=True)

    # 获得 Markdown 的一维 tokens 流
    tokens: list = md.parse(markdown_content)
    original_lines: list = markdown_content.split("\n")

    result: list = []
    current_content: list = []
    # 维护 H1 到 H6 的父子级联标题栈，用于后续附加在每个文本块前面
    title_stack: list = [""] * 6

    i = 0
    # 开始遍历语法树 Token
    while i < len(tokens):
        token = tokens[i]
        
        # 1. 遇到标题，压入标题栈
        if token.type == "heading_open":
            _flush_content(result, current_content, title_stack, max_length, embed_fn)
            level = int(token.tag[1:]) if token.tag and len(token.tag) > 1 else 1
            inline_token = tokens[i + 1]
            if inline_token.type == "inline":
                full_title = inline_token.content.strip()
                title_stack[level - 1] = full_title
                # 压入某级标题时，清除其下级的全部标题记录
                for j in range(level, 6):
                    title_stack[j] = ""
            i += 3
            continue
            
        # 2. 遇到表格，保持表格结构的完整性
        elif token.type == "table_open":
            _flush_content(result, current_content, title_stack, max_length, embed_fn)
            j, table_content = extract_table_block(tokens, i, original_lines)
            current_content.append(table_content)
            _flush_content(result, current_content, title_stack, max_length, embed_fn, special_element="Table")
            i = j + 1 if j < len(tokens) else len(tokens)
            continue
            
        # 3. 遇到普通段落
        elif token.type == "paragraph_open":
            # 先尝试探测是否属于图片与题注的配对关系
            handled, new_i = _handle_image_caption(
                tokens, i, result, current_content, title_stack, max_length, embed_fn
            )
            if handled:
                i = new_i
                continue
            # 若不是图文配对，则作为正常段落内容积攒
            inline_token = tokens[i + 1]
            if inline_token.type == "inline":
                current_content.append(inline_token.content.strip())
            i += 3
            continue
            
        # 4. 遇到代码块
        elif token.type == "fence":
            current_content.append(f"```\n{token.content}\n```")
            i += 1
            continue
            
        # 5. 遇到有序列表
        elif token.type == "ordered_list_open":
            list_content = []
            j = i + 1
            list_item_counter = 1
            # 完整解析直到遇到 close 闭合标签
            while j < len(tokens) and tokens[j].type != "ordered_list_close":
                if tokens[j].type == "list_item_open":
                    k = j + 1
                    while k < len(tokens) and tokens[k].type != "list_item_close":
                        if (
                            tokens[k].type == "paragraph_open"
                            and k + 1 < len(tokens)
                            and tokens[k + 1].type == "inline"
                        ):
                            list_content.append(f"{list_item_counter}. {tokens[k + 1].content.strip()}")
                            list_item_counter += 1
                        k += 1
                j += 1
            if list_content:
                current_content.extend(list_content)
                _flush_content(result, current_content, title_stack, max_length, embed_fn, special_element=token.type)
            i = j + 1
            continue
            
        # 6. 遇到无序列表
        elif token.type == "bullet_list_open":
            list_content = []
            j = i + 1
            while j < len(tokens) and tokens[j].type != "bullet_list_close":
                if tokens[j].type == "list_item_open":
                    k = j + 1
                    while k < len(tokens) and tokens[k].type != "list_item_close":
                        if (
                            tokens[k].type == "paragraph_open"
                            and k + 1 < len(tokens)
                            and tokens[k + 1].type == "inline"
                        ):
                            list_content.append(f"- {tokens[k + 1].content.strip()}")
                        k += 1
                j += 1
            if list_content:
                current_content.extend(list_content)
                _flush_content(result, current_content, title_stack, max_length, embed_fn, special_element=token.type)
            i = j + 1
            continue
            
        # 7. 遇到 HTML 代码块（部分老旧工具转化文档会产生 HTML Table）
        elif token.type == "html_block":
            _flush_content(result, current_content, title_stack, max_length, embed_fn)
            content = token.content.strip()
            is_converted_table = False
            # 对于难以通过 Markdown 支持合并单元格的 HTML 表格，将其转化为键值对形式 (Key: Value)
            if "<table" in content.lower():
                try:
                    kv_list = html_table_to_key_value(content)
                    if kv_list:
                        content = "\n".join([f"- {item}" for item in kv_list])
                        is_converted_table = True
                except Exception as e:
                    logger.warning(f"HTML表格转KV失败: {e}")

            current_content.append(content)
            # 如果转换为了键值对（单行很长），则允许它被 split_text_by_length_and_newline 打碎
            if is_converted_table:
                _flush_content(
                    result,
                    current_content,
                    title_stack,
                    max_length,
                    embed_fn,
                    special_element="Table KV",
                    allow_split=True,
                )
            else:
                _flush_content(result, current_content, title_stack, max_length, embed_fn, special_element=token.type)
            i += 1
            continue
            
        # 8. 忽略结构闭合标签，游标前进即可
        elif token.type in ["list_item_close", "ordered_list_close", "bullet_list_close", "list_item_open"]:
            i += 1
            continue
            
        # 9. 遇到公式块
        elif token.type == "math_block":
            current_content.append(f"$ {token.content} $")
            _flush_content(result, current_content, title_stack, max_length, embed_fn, special_element="Math Block")
            i += 1
            continue
            
        # 未命中处理的 Token
        else:
            i += 1

    # 遍历结束后，将残余暂存的文本一并刷出
    _flush_content(result, current_content, title_stack, max_length, embed_fn)

    # 通过分隔符 `----------` 将所有平铺的内容列表真正拆分成一段段 string
    chunks = []
    current_chunk_parts = []
    for item in result:
        if item == "-" * 10:
            if current_chunk_parts:
                chunks.append("\n".join(current_chunk_parts).strip())
                current_chunk_parts = []
        else:
            current_chunk_parts.append(item)

    if current_chunk_parts:
        chunks.append("\n".join(current_chunk_parts).strip())

    logger.info(f"语义切分完成: 共提取出 {len(chunks)} 个块。")
    return chunks
