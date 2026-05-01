"""
自然语言处理 (NLP) 基础工具模块。

提供轻量级的文本处理函数，例如 token 数量估算，
以避免在简单场景下引入过于庞大的分词库 (如 tiktoken) 增加打包负担。
"""
from __future__ import annotations

import re


def count_tokens(text: str) -> int:
    """
    近似计算给定文本的 Token 数量。
    
    这种方法通过正则表达式简单统计：
    1. 英文单词、数字、下划线组合
    2. 中日韩 (CJK) 单字符
    
    它比 tiktoken 等模型专用分词器快廉价，非常适合用于估算长度限制并进行文本截断操作。
    但在对长度精确要求极高的场景（例如严格卡模型的上下文上限）可能存在一定误差。
    
    Args:
        text: 需要估算 Token 数量的原始文本。
        
    Returns:
        整数，表示近似的 token 数量。至少返回 1（如果文本不为空）。
    """
    if not text:
        return 0
        
    # 正则规则解释：
    # [A-Za-z0-9_]+ 匹配连续的英文字母、数字和下划线，将其作为一个 token
    # [\u4e00-\u9fff] 匹配独立的中文汉字，将其作为一个 token
    parts = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text)
    
    # 只要文本 strip() 后不为空，即使正则没匹配到也至少算 1 个 Token
    return max(1, len(parts)) if text.strip() else 0
