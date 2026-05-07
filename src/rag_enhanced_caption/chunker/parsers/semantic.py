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
from ..embed_client import get_default_embedding_client
from ..schema import SemanticChunk

logger = logging.getLogger(__name__)

_CONTROL_SPECIAL_ELEMENTS = {
    "bullet_list_open",
    "ordered_list_open",
}

def _normalize_special_element(special_element: str | None) -> str:
    """过滤解析过程中的控制标签，避免其进入最终分类。"""
    if not special_element:
        return "text"
    special = special_element.strip()
    if not special or special in _CONTROL_SPECIAL_ELEMENTS:
        return "text"
    return special

from ..utils.md_parser_utils import (
    extract_table_block,
    get_title_path,
    split_text_by_length_and_newline,
)
from ..utils.table_utils import html_table_to_key_value


def _flush_content(
    result: list[SemanticChunk],
    current_content: list[str],
    title_stack: list[str],
    max_length: int,
    embed_fn: Any,
    special_element: str | None = None,
    allow_split: bool = False,
    state: dict | None = None,
) -> None:
    if not current_content:
        return

    if state is None:
        state = {"last_text_idx": -1}

    content = "\n".join(current_content).strip()
    if not content:
        current_content.clear()
        return

    path_list = [t for t in title_stack if t]
    element_type = _normalize_special_element(special_element)

    # Determine parent_id for non-text elements
    parent_id = str(state["last_text_idx"]) if element_type != "text" and state["last_text_idx"] != -1 else None

    if element_type != "text" and not allow_split:
        chunk = SemanticChunk(
            content=content,
            full_content=content,
            header_path=path_list.copy(),
            element_type=element_type,
            parent_id=parent_id
        )
        result.append(chunk)
    else:
        if count_tokens(content) > max_length:
            chunks = split_text_by_length_and_newline(
                content, max_length, embed_fn=embed_fn, token_count_fn=count_tokens
            )
            for idx, text_chunk in enumerate(chunks, 1):
                chunk = SemanticChunk(
                    content=text_chunk,
                    full_content=text_chunk,
                    header_path=path_list.copy() + [f"Part {idx}"],
                    element_type=element_type,
                    parent_id=parent_id if element_type != "text" else None
                )
                result.append(chunk)
                if element_type == "text":
                    state["last_text_idx"] = len(result) - 1
        else:
            chunk = SemanticChunk(
                content=content,
                full_content=content,
                header_path=path_list.copy(),
                element_type=element_type,
                parent_id=parent_id if element_type != "text" else None
            )
            result.append(chunk)
            if element_type == "text":
                state["last_text_idx"] = len(result) - 1

    current_content.clear()


def _handle_image_caption(tokens, i, result, current_content, title_stack, max_length, embed_fn, state):
    token = tokens[i]
    if token.type != "paragraph_open":
        return False, i

    inline_token = tokens[i + 1]
    if inline_token.type != "inline":
        return False, i

    content = inline_token.content.strip()
    image_pattern = r"^!\[.*?\]\(.*?\)\s*$"
    caption_pattern = r"^(?:Figure|图|Fig\.|表|Table)\s*[\d\w\.]+"

    img_match = re.search(r"^(!\[.*?\]\(.*?\))", content)
    if img_match:
        rest = content[img_match.end() :].strip()
        if rest and re.match(caption_pattern, rest, re.IGNORECASE):
            _flush_content(result, current_content, title_stack, max_length, embed_fn, state=state)
            current_content.append(content)
            caption_title = rest.split("\n")[0].strip()
            _flush_content(result, current_content, title_stack, max_length, embed_fn, special_element="Image", state=state)
            return True, i + 3

    if re.match(image_pattern, content):
        next_p_idx = i + 3
        if next_p_idx + 1 < len(tokens) and tokens[next_p_idx].type == "paragraph_open":
            next_inline = tokens[next_p_idx + 1]
            if next_inline.type == "inline":
                next_content = next_inline.content.strip()
                if re.match(caption_pattern, next_content, re.IGNORECASE):
                    _flush_content(result, current_content, title_stack, max_length, embed_fn, state=state)
                    current_content.append(content + "\n" + next_content)
                    _flush_content(
                        result, current_content, title_stack, max_length, embed_fn, special_element="Image", state=state
                    )
                    return True, i + 6

    if current_content and re.match(caption_pattern, content, re.IGNORECASE):
        last_item = current_content[-1].strip()
        if re.match(image_pattern, last_item):
            image_tag = current_content.pop()
            _flush_content(result, current_content, title_stack, max_length, embed_fn, state=state)
            current_content.append(image_tag + "\n" + content)
            _flush_content(result, current_content, title_stack, max_length, embed_fn, special_element="Image", state=state)
            return True, i + 3

    # If it's just a raw image without caption, emit it as an Image chunk
    if re.match(image_pattern, content):
        _flush_content(result, current_content, title_stack, max_length, embed_fn, state=state)
        current_content.append(content)
        _flush_content(result, current_content, title_stack, max_length, embed_fn, special_element="Image", state=state)
        return True, i + 3

    return False, i


def chunk_markdown(
    markdown_content: str, parser_config: dict[str, Any] | None = None, embed_fn: Any | None = None
) -> list[SemanticChunk]:
    parser_config = parser_config or {}
    max_length = int(parser_config.get("chunk_token_num", 512))
    logger.info(f"语义切分开始: max_length={max_length}, content_length={len(markdown_content)}")

    if embed_fn is None:
        logger.info("未提供 embed_fn，尝试从 .env 环境初始化默认客户端...")
        embed_fn = get_default_embedding_client()
        if embed_fn is None:
            logger.warning("默认 Embedding 客户端初始化失败，退化为简单的纯文本字数切分。")

    md = MarkdownIt("commonmark").enable("table")
    md.use(dollarmath_plugin, allow_space=True, allow_digits=True)

    tokens: list = md.parse(markdown_content)
    original_lines: list = markdown_content.split("\n")

    result: list[SemanticChunk] = []
    current_content: list[str] = []
    title_stack: list[str] = [""] * 6
    state = {"last_text_idx": -1}
    last_heading_idx = 0

    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        if token.type == "heading_open":
            _flush_content(result, current_content, title_stack, max_length, embed_fn, state=state)
            
            # 断开跨章节的父子关联：遇到新标题时，重置上一个文本锚点
            state["last_text_idx"] = -1
            
            level = int(token.tag[1:]) if token.tag and len(token.tag) > 1 else 1
            last_heading_idx = level - 1
            inline_token = tokens[i + 1]
            if inline_token.type == "inline":
                full_title = inline_token.content.strip()
                title_stack[last_heading_idx] = full_title
                for j in range(last_heading_idx + 1, 6):
                    title_stack[j] = ""
            i += 3
            continue
            
        elif token.type == "table_open":
            _flush_content(result, current_content, title_stack, max_length, embed_fn, state=state)
            j, table_content = extract_table_block(tokens, i, original_lines)
            current_content.append(table_content)
            _flush_content(result, current_content, title_stack, max_length, embed_fn, special_element="Table", state=state)
            i = j + 1 if j < len(tokens) else len(tokens)
            continue
            
        elif token.type == "paragraph_open":
            handled, new_i = _handle_image_caption(
                tokens, i, result, current_content, title_stack, max_length, embed_fn, state
            )
            if handled:
                i = new_i
                continue
            inline_token = tokens[i + 1]
            if inline_token.type == "inline":
                # For inline images inside text, we can optionally split them out.
                # But for simplicity, if it wasn't handled by _handle_image_caption, we keep it as text.
                current_content.append(inline_token.content.strip())
            i += 3
            continue
            
        elif token.type == "fence":
            current_content.append(f"```\n{token.content}\n```")
            i += 1
            continue
            
        elif token.type in ["ordered_list_open", "bullet_list_open"]:
            _flush_content(result, current_content, title_stack, max_length, embed_fn, state=state)
            
            if token.map:
                start_line, end_line = token.map
                list_text = "\n".join(original_lines[start_line:end_line])
                current_content.append(list_text)
            
            target_level = token.level
            close_type = token.type.replace("_open", "_close")
            j = i + 1
            while j < len(tokens):
                if tokens[j].type == close_type and tokens[j].level == target_level:
                    break
                j += 1
                
            _flush_content(result, current_content, title_stack, max_length, embed_fn, special_element=token.type, state=state)
            i = j + 1
            continue
            
        elif token.type == "html_block":
            _flush_content(result, current_content, title_stack, max_length, embed_fn, state=state)
            content = token.content.strip()
            
            # 解决 markdown-it 将带空行的 HTML 表格切碎的问题
            if "<table" in content.lower() and "</table>" not in content.lower():
                table_start = token.map[0] if token.map else -1
                if table_start != -1:
                    table_end = table_start + 1
                    for line_idx in range(table_start, len(original_lines)):
                        if "</table>" in original_lines[line_idx].lower():
                            table_end = line_idx + 1
                            break
                    content = "\n".join(original_lines[table_start:table_end])
                    j = i + 1
                    while j < len(tokens):
                        next_token = tokens[j]
                        if next_token.map:
                            if next_token.map[0] >= table_end:
                                break
                        j += 1
                    i = j - 1

            # 处理 <summary> 作为虚拟标题
            summary_match = re.search(r'<summary>(.*?)</summary>', content, re.IGNORECASE | re.DOTALL)
            if summary_match:
                summary_text = summary_match.group(1).strip()
                summary_text = re.sub(r'<[^>]+>', '', summary_text).strip()
                if summary_text:
                    target_idx = last_heading_idx + 1
                    if target_idx < 6:
                        title_stack[target_idx] = summary_text
                        for d in range(target_idx + 1, 6):
                            title_stack[d] = ""

            is_converted_table = False
            if "<table" in content.lower():
                try:
                    kv_list = html_table_to_key_value(content)
                    if kv_list:
                        content = "\n".join([f"- {item}" for item in kv_list])
                        is_converted_table = True
                except Exception as e:
                    logger.warning(f"HTML表格转KV失败: {e}")

            current_content.append(content)
            if is_converted_table:
                _flush_content(
                    result,
                    current_content,
                    title_stack,
                    max_length,
                    embed_fn,
                    special_element="Table KV",
                    allow_split=True,
                    state=state
                )
            else:
                _flush_content(result, current_content, title_stack, max_length, embed_fn, special_element="html_block", state=state)
            i += 1
            continue
            
        elif token.type in ["list_item_close", "ordered_list_close", "bullet_list_close", "list_item_open"]:
            i += 1
            continue
            
        elif token.type == "math_block":
            _flush_content(result, current_content, title_stack, max_length, embed_fn, state=state)
            current_content.append(f"$ {token.content} $")
            _flush_content(result, current_content, title_stack, max_length, embed_fn, special_element="Math Block", state=state)
            i += 1
            continue
            
        else:
            i += 1

    _flush_content(result, current_content, title_stack, max_length, embed_fn, state=state)

    logger.info(f"语义切分完成: 共提取出 {len(result)} 个结构化块。")
    return result
