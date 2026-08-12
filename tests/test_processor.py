"""
测试模块：验证多模态处理器 (Processor) 的核心流程
主要逻辑：
1. 异步处理：验证异步 VLM 调用和信号量并发控制。
2. 语义增强：针对 Markdown 表格生成用于向量检索的摘要 (text_for_embedding)。
3. 上下文注入：验证 VLM 是否能利用表格上方的文字背景生成更精准的描述。
"""

import asyncio
import pytest
import os

from dotenv import load_dotenv

load_dotenv()

from rag_enhanced_caption.enhancer.processor import MarkdownMultimodalProcessor  # noqa: E402
from rag_enhanced_caption.enhancer.vlm_client import create_default_vlm_client  # noqa: E402
from rag_enhanced_caption.chunker.dispatcher import chunk_markdown  # noqa: E402


def test_short_parent_id_injects_parent_context_into_table_prompt() -> None:
    """A persisted short parent ID should resolve to its complete chunk ID."""
    captured_prompts: list[str] = []

    async def fake_vlm(user_prompt: str, *args: object) -> str:
        captured_prompts.append(user_prompt)
        return '{"summary": "表格摘要", "entities": []}'

    chunks = [
        {
            "id": "sample_chunk_5",
            "content": "这是表格对应的正文上下文。",
            "metadata": {"element_type": "text", "parent_id": None},
        },
        {
            "id": "sample_chunk_6",
            "content": "| 名称 | 值 |\n| --- | --- |\n| 测试 | 1 |",
            "metadata": {"element_type": "Table", "parent_id": "5"},
        },
    ]

    processor = MarkdownMultimodalProcessor(vlm_func=fake_vlm, max_concurrency=1)
    enriched_chunks = asyncio.run(processor.enrich_chunks(chunks))

    assert captured_prompts
    assert "这是表格对应的正文上下文。" in captured_prompts[0]
    assert enriched_chunks[1]["text_for_embedding"] == "表格摘要"


def test_vlm_table_summarization() -> None:
    """
    Test using REAL VLM to generate high-density summaries for tables.
    Verifies that the prompt successfully extracts entities and doesn't output filenames or noise.
    """

    async def run_test() -> None:
        if not os.getenv("VLM_API_KEY"):
            pytest.skip("No VLM_API_KEY configured.")

        vlm_client = create_default_vlm_client()
        processor = MarkdownMultimodalProcessor(vlm_func=vlm_client, max_concurrency=1)

        test_md = """
# Q3 Financial Report

## Revenue Breakdown

Here is the revenue breakdown for Q3 2023 across different regions. We saw a significant increase in the APAC region due to new product launches.

| Region | Q3 2023 Revenue (M) | YoY Growth |
|---|---|---|
| North America | $120.5 | +5% |
| EMEA | $85.2 | -2% |
| APAC | $150.8 | +25% |

The data clearly shows APAC taking the lead.
"""

        chunks = chunk_markdown(
            markdown_content=test_md,
            file_id="test_financial",
            filename="report.md",
            parser_config={"chunk_token_num": 512},
        )

        enriched_chunks = await processor.enrich_chunks(chunks)

        # Find the table chunk
        table_chunk = next(
            (c for c in enriched_chunks if c["metadata"]["element_type"] == "Table"),
            None,
        )

        assert table_chunk is not None, "Failed to extract table chunk"
        assert "text_for_embedding" in table_chunk

        # Asserting prompt effectiveness
        summary = table_chunk["text_for_embedding"]
        print("\n=======================================================")
        print("🤖 [VLM Table Summary Check]")
        print("=======================================================")
        print(f"Raw Table Data:\n{table_chunk['full_content']}")
        print("-------------------------------------------------------")
        print(f"VLM Generated Summary (Used for Vector Search):\n{summary}")

        # Check entities
        entities = table_chunk["metadata"].get("entities", [])
        print("-------------------------------------------------------")
        print(f"VLM Extracted Entities:\n{entities}")
        print("=======================================================\n")

        # The summary should be a JSON parsed result, so it shouldn't have raw JSON syntax like "{"
        assert "{" not in summary and "}" not in summary, (
            "VLM failed to parse JSON properly and leaked syntax"
        )

        # It should mention APAC or Revenue since it's asked to extract entities/insights
        assert "APAC" in summary.upper() or "REVENUE" in summary.upper(), (
            "VLM failed to extract key table semantics"
        )

        # The metadata should contain extracted entities
        assert len(entities) > 0, "VLM failed to extract any entities"

    asyncio.run(run_test())
