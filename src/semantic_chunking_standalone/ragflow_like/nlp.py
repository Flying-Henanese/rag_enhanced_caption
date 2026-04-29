from __future__ import annotations

import re


def count_tokens(text: str) -> int:
    """近似 token 计数，避免引入额外依赖。"""
    if not text:
        return 0
    # 英文单词 + 数字 + CJK 单字
    parts = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text)
    return max(1, len(parts)) if text.strip() else 0
