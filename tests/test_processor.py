import asyncio
import os
import sys
from pathlib import Path
import pytest
from loguru import logger

from dotenv import load_dotenv
load_dotenv()

from rag_enhanced_caption.enhancer.processor import MarkdownMultimodalProcessor
from rag_enhanced_caption.enhancer.vlm_client import create_default_vlm_client
from rag_enhanced_caption.chunker.dispatcher import chunk_markdown

def test_vlm_table_summarization():
    """
    Test using REAL VLM to generate high-density summaries for tables.
    Verifies that the prompt successfully extracts entities and doesn't output filenames or noise.
    """
    async def run_test():
        vlm_client = create_default_vlm_client()
        if not vlm_client:
            pytest.skip("No VLM client configured.")

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
            parser_config={"chunk_token_num": 512}
        )

        enriched_chunks = await processor.enrich_chunks(chunks)

        # Find the table chunk
        table_chunk = next((c for c in enriched_chunks if c["metadata"]["element_type"] == "Table"), None)
        
        assert table_chunk is not None, "Failed to extract table chunk"
        assert "text_for_embedding" in table_chunk
        
        # Asserting prompt effectiveness
        summary = table_chunk["text_for_embedding"]
        print(f"\n=======================================================")
        print(f"🤖 [VLM Table Summary Check]")
        print(f"=======================================================")
        print(f"Raw Table Data:\n{table_chunk['full_content']}")
        print(f"-------------------------------------------------------")
        print(f"VLM Generated Summary (Used for Vector Search):\n{summary}")
        
        # Check entities
        entities = table_chunk["metadata"].get("entities", [])
        print(f"-------------------------------------------------------")
        print(f"VLM Extracted Entities:\n{entities}")
        print(f"=======================================================\n")
        
        # The summary should be a JSON parsed result, so it shouldn't have raw JSON syntax like "{"
        assert "{" not in summary and "}" not in summary, "VLM failed to parse JSON properly and leaked syntax"
        
        # It should mention APAC or Revenue since it's asked to extract entities/insights
        assert "APAC" in summary.upper() or "REVENUE" in summary.upper(), "VLM failed to extract key table semantics"
        
        # The metadata should contain extracted entities
        assert len(entities) > 0, "VLM failed to extract any entities"

    asyncio.run(run_test())
