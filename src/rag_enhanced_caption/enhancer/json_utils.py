import json
import re
from typing import Any

from loguru import logger


def try_parse_json(json_str: str) -> dict[str, Any] | None:
    """尝试解析 JSON 字符串。

    Args:
        json_str: 待解析的 JSON 字符串。

    Returns:
        解析后的字典；输入为空或解析失败时返回 ``None``。
    """
    if not json_str or not json_str.strip():
        return None
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None


def basic_json_cleanup(json_str: str) -> str:
    """处理常见的不规范 JSON 格式。

    Args:
        json_str: 待清理的 JSON 字符串。

    Returns:
        修正智能引号和尾随逗号后的字符串。
    """
    json_str = json_str.strip()
    # 修复中文引号或智能引号
    json_str = json_str.replace("“", '"').replace("”", '"')
    json_str = json_str.replace("‘", "'").replace("’", "'")
    # 修复多余的尾随逗号 (如 {"a": 1,} -> {"a": 1})
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
    return json_str


def progressive_quote_fix(json_str: str) -> str:
    """处理 JSON 字符串中的转义字符和引号问题。

    Args:
        json_str: 待修复的 JSON 字符串。

    Returns:
        修正转义字符后的字符串。
    """
    # 仅转义引号前未转义的反斜杠
    json_str = re.sub(r'(?<!\\)\\(?=")', r"\\\\", json_str)

    def fix_string_content(match: re.Match[str]) -> str:
        content = match.group(1)
        # 修复字符串值中未转义的反斜杠 (例如 \alpha -> \\alpha)
        content = re.sub(r"\\(?=[a-zA-Z])", r"\\\\", content)
        return f'"{content}"'

    json_str = re.sub(r'"([^"]*(?:\\.[^"]*)*)"', fix_string_content, json_str)
    return json_str


def extract_all_json_candidates(response: str) -> list[str]:
    """从模型长文本回复中提取所有可能的 JSON 候选字符串。

    Args:
        response: 模型返回的原始文本。

    Returns:
        按提取策略收集的 JSON 候选字符串。
    """
    candidates = []

    # 预处理：移除推理模型（如 DeepSeek-R1, Qwen-math）产生的思考过程标签
    cleaned_response = re.sub(
        r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE
    )
    cleaned_response = re.sub(
        r"<thinking>.*?</thinking>",
        "",
        cleaned_response,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 策略 1：匹配 Markdown 代码块中的 JSON (```json ... ```)
    json_blocks = re.findall(
        r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_response, re.DOTALL
    )
    candidates.extend(json_blocks)

    # 策略 2：利用大括号平衡匹配提取纯 JSON 对象
    brace_count = 0
    start_pos = -1
    for i, char in enumerate(cleaned_response):
        if char == "{":
            if brace_count == 0:
                start_pos = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_pos != -1:
                candidates.append(cleaned_response[start_pos : i + 1])

    # 策略 3：简单的正则兜底提取
    simple_match = re.search(r"\{.*\}", cleaned_response, re.DOTALL)
    if simple_match:
        candidates.append(simple_match.group(0))

    return candidates


def extract_fields_with_regex(response: str) -> dict[str, Any]:
    """在 JSON 完全损坏时用正则提取关键字段。

    Args:
        response: 模型返回的原始文本。

    Returns:
        包含描述和实体信息的兜底结果。
    """
    logger.warning("JSON 解析完全失败，正在使用正则表达式回退提取字段")

    # 预处理：移除推理模型产生的思考过程标签，防止正则提取到思考内容
    cleaned_response = re.sub(
        r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE
    )
    cleaned_response = re.sub(
        r"<thinking>.*?</thinking>",
        "",
        cleaned_response,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 提取 detailed_description
    desc_match = re.search(
        r'"detailed_description":\s*"([^"]*(?:\\.[^"]*)*)"', cleaned_response, re.DOTALL
    )
    description = desc_match.group(1) if desc_match else ""

    # 提取 entity_name
    name_match = re.search(r'"entity_name":\s*"([^"]*(?:\\.[^"]*)*)"', cleaned_response)
    entity_name = name_match.group(1) if name_match else "unknown_entity"

    # 提取 entity_type
    type_match = re.search(r'"entity_type":\s*"([^"]*(?:\\.[^"]*)*)"', cleaned_response)
    entity_type = type_match.group(1) if type_match else "unknown"

    # 提取 summary
    summary_match = re.search(
        r'"summary":\s*"([^"]*(?:\\.[^"]*)*)"', cleaned_response, re.DOTALL
    )
    summary = summary_match.group(1) if summary_match else description[:100]

    return {
        "detailed_description": description,
        "entity_info": {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "summary": summary,
        },
    }


def robust_json_parse(response: str) -> dict[str, Any]:
    """解析可能包含 Markdown 或轻微语法错误的模型 JSON 输出。

    Args:
        response: 模型返回的原始文本。

    Returns:
        解析后的 JSON 字典，或正则提取得到的兜底字段。
    """
    candidates = extract_all_json_candidates(response)

    # 尝试 1：直接解析
    for json_candidate in candidates:
        result = try_parse_json(json_candidate)
        if result:
            return result

    # 尝试 2：基础清理后解析
    for json_candidate in candidates:
        cleaned = basic_json_cleanup(json_candidate)
        result = try_parse_json(cleaned)
        if result:
            return result

    # 尝试 3：进阶转义修复后解析
    for json_candidate in candidates:
        fixed = progressive_quote_fix(json_candidate)
        result = try_parse_json(fixed)
        if result:
            return result

    # 尝试 4：所有 JSON 解析手段均失败，正则暴力提取
    return extract_fields_with_regex(response)
