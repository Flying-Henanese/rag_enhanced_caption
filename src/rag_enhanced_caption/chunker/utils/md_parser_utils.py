"""
Markdown 解析辅助工具模块。

提供标题层级推断、标题路径拼接、表格块提取，以及超长文本的安全截断逻辑。
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .semantic_utils import semantic_chunking_with_auto_clusters


def infer_heading_level(title: str) -> int:
    """
    根据标题文本推断其层级级别（1-6级）。

    逻辑说明：
    1. 数字序号推断：
       - 匹配如 "1.", "1.1", "1.2.3" 等格式。
       - 根据点号分隔的数量确定层级，例如 "1.1" 为 2 级，"1.2.3" 为 3 级。
       - 层级限制在 1-6 之间。
    2. 中文序号推断：
       - 匹配如 "一、", "二." 等中文数字序号。
       - 统一归类为 1 级标题。
    3. 默认处理：
       - 若不匹配以上规则，默认返回 1 级。
       
    Args:
        title: 标题字符串。
        
    Returns:
        推断的标题层级（整数 1-6）。
    """
    m = re.match(r"^\s*(\d+(?:\.\d+)*)[.)、]?\s*", title)
    if m:
        return max(1, min(len(m.group(1).split(".")), 6))
    m_zh = re.match(r"^\s*[一二三四五六七八九十百千]+[、.]\s*", title)
    if m_zh:
        return 1
    return 1


def get_title_path(stack: list[str]) -> str:
    """
    根据标题栈生成标题路径，用"|"分隔。
    
    例如: stack 为 ["一级标题", "二级标题", "", "", "", ""]
    返回: "一级标题|二级标题"
    """
    return "|".join([t for t in stack if t])


def extract_table_block(tokens: list[Any], i: int, original_lines: list[str]) -> tuple[int, str]:
    """
    从 token 流和原始文本中提取完整的 Markdown 表格块。

    逻辑说明：
    1. 定位起始：通过当前 token (i) 的 `map` 属性获取表格在原始行中的起始行号 `table_start`。
    2. 查找结束 token：遍历后续 tokens 直到找到 `table_close`。
    3. 确定结束行号 (`table_end`)：
       - 优先使用 `table_close` token 的 `map` 属性。
       - 若不存在，则尝试查找下一个带有 `map` 信息的 token 的起始行作为当前表格的结束。
       - 若上述均失败（如文件末尾或解析异常），则回退到基于文本内容的启发式扫描：
         从 `table_start` 开始向下扫描，直到遇到不符合 Markdown 表格特征（不以 '|' 开头且不含 '|'）的行为止。
    4. 返回结果：返回 `table_close` 的索引 `j` 以及拼接后的表格原始字符串。
    
    Args:
        tokens: markdown-it 解析出的 token 列表。
        i: 当前 `table_open` token 的索引。
        original_lines: 原始 Markdown 文本按行切分的列表。
        
    Returns:
        (表格结束时的 token 索引, 提取出的表格纯文本)
    """
    token = tokens[i]
    table_start = token.map[0] if token.map else 0
    j = i + 1
    # 找到匹配的 table_close
    while j < len(tokens) and tokens[j].type != "table_close":
        j += 1
        
    if j < len(tokens):
        end_token = tokens[j]
        if end_token.map and end_token.map[1] is not None:
            table_end = end_token.map[1]
        else:
            # 找不到明确结束，尝试从后续 token 的 map[0] 借用边界
            table_end = None
            for k in range(j + 1, len(tokens)):
                if tokens[k].map and tokens[k].map[0] is not None:
                    table_end = tokens[k].map[0]
                    break
            # 最差情况，启发式向下扫描
            if table_end is None:
                table_end = table_start + 1
                for line_idx in range(table_start, len(original_lines)):
                    line = original_lines[line_idx].strip()
                    if not line or not (line.startswith("|") or "|" in line):
                        table_end = line_idx
                        break
    else:
        # 没有闭合标签（残缺的 token 流），启发式向下扫描
        table_end = table_start + 1
        for line_idx in range(table_start, len(original_lines)):
            line = original_lines[line_idx].strip()
            if not line or not (line.startswith("|") or "|" in line):
                table_end = line_idx
                break
                
    return j, "\n".join(original_lines[table_start:table_end])


def split_text_by_length_and_newline(
    text: str, max_length: int, embed_fn: Callable[[list[str]], Any] | None, token_count_fn: Callable[[str], int]
) -> list[str]:
    """
    层次化的长文本安全切分策略。
    
    当遇到极其长（超过 max_length 限制）且没有明显标题层级的一大段文本时，调用此方法。
    处理逻辑：
    1. 优先尝试按双换行符（即段落 `\n\n`）拆分。
    2. 如果单段依然超长，则尝试按单换行符（行 `\n`）拆分，累加直到接近最大长度。
    3. 如果哪怕是单行也超长（例如某些极长且没有换行的句子），则使用基于 Embedding 向量聚类的
       `semantic_chunking_with_auto_clusters` 策略进行句子级切分。
       
    Args:
        text: 超长文本内容。
        max_length: 允许的最大 token 数。
        embed_fn: 向量化函数。
        token_count_fn: 计算 token 的函数。
        
    Returns:
        拆分后的文本片段列表，每个片段理论上不超过 max_length。
    """
    chunks = []

    # 第一级拆分：按段落
    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_token_count = token_count_fn(paragraph)

        # 如果当前段落长度未超过最大 Token 数量，直接作为独立分块放入chunks
        # 否则继续尝试按行切分
        if paragraph_token_count <= max_length:
            chunks.append(paragraph)
            continue

        # 第二级拆分：把超长段落进一步使用换行符进行切分为行
        lines = paragraph.split("\n")
        current_chunk_lines = []
        current_chunk_tokens = 0

        for line in lines:
            line = line.strip()
            if not line:  # 跳过空行
                continue

            line_token_count = token_count_fn(line)  # 计算当前行的 Token 数量
            # 为了考虑行之间的空格/换行，需要在计算 Token 数量时加 1（如果当前行不是第一行，需要添加一个换行符的Token数量）
            added_tokens = line_token_count + (1 if current_chunk_lines else 0)
            
            # 第三级拆分（保底）：如果当前这一行的 Token 数量就已经超过最大 Token 数量
            if line_token_count > max_length:
                # 先把之前攒的行存起来
                if current_chunk_lines:
                    chunks.append("\n".join(current_chunk_lines))
                    current_chunk_lines = []
                    current_chunk_tokens = 0

                # 使用基于向量语义的精细聚类拆分这一超长行
                sub_chunks = semantic_chunking_with_auto_clusters(
                    line, embed_fn=embed_fn, token_count_fn=token_count_fn, max_chunk_size=max_length
                )
                chunks.extend(sub_chunks)
                
            # 如果当前行的 Token 数量与当前分块的 Token 数量合并后超过最大 Token 数量，当前块收尾
            elif current_chunk_tokens + added_tokens > max_length:
                # 把之前的分块内容放入chunks
                chunks.append("\n".join(current_chunk_lines))
                # 重置当前分块为当前行的内容
                current_chunk_lines = [line]
                # 更新当前分块的 Token 数量
                current_chunk_tokens = line_token_count
                
            # 如果当前行的内容加入当前分块后不会超过最大 Token 数量，直接加入当前分块
            else:
                current_chunk_lines.append(line)
                current_chunk_tokens += added_tokens  # 更新当前分块的 Token 数量
                
        # 最后的收尾，把当前处理块中剩余的内容放入chunks
        if current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines))

    return chunks
