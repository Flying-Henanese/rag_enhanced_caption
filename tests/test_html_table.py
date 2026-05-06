import pytest
import asyncio
import os
import sys
from pathlib import Path
from loguru import logger

from dotenv import load_dotenv
load_dotenv()

from rag_enhanced_caption.enhancer.processor import MarkdownMultimodalProcessor
from rag_enhanced_caption.enhancer.vlm_client import create_default_vlm_client
from rag_enhanced_caption.chunker.dispatcher import chunk_markdown

def test_html_table_parsing_and_enrichment():
    async def run_test():
        vlm_client = create_default_vlm_client()
        if not vlm_client:
            pytest.skip("No VLM client configured.")
            
        processor = MarkdownMultimodalProcessor(vlm_func=vlm_client, max_concurrency=1)
        
        test_md_content = """# Performance Overview

This document outlines the performance.

## Complex HTML Table

Here is a complex table layout that standard markdown cannot easily represent.

<div align="center">
    <table border="1">
        <tr>
            <th>Feature</th>
            <th>Support</th>
        </tr>
        <tr>
            <td>HTML Parsing</td>
            <td>Yes</td>
        </tr>
    </table>
</div>

End of document.
"""

        chunks = chunk_markdown(markdown_content=test_md_content)
        enriched_chunks = await processor.enrich_chunks(chunks)
        
        html_chunk = next((c for c in enriched_chunks if c["metadata"]["element_type"] == "Table KV" or c["metadata"]["element_type"] == "html_block"), None)
        
        assert html_chunk is not None, "Failed to identify HTML table chunk"
        
        summary = html_chunk["text_for_embedding"]
        
        print(f"\n=======================================================")
        print(f"🤖 [VLM HTML Table Summary Check]")
        print(f"=======================================================")
        print(f"Raw HTML Table Data:\n{html_chunk['full_content']}")
        print(f"-------------------------------------------------------")
        print(f"VLM Generated Summary (Used for Vector Search):\n{summary}")
        
        entities = html_chunk["metadata"].get("entities", [])
        print(f"-------------------------------------------------------")
        print(f"VLM Extracted Entities:\n{entities}")
        print(f"=======================================================\n")
        
        assert "{" not in summary and "}" not in summary, "VLM failed to parse JSON properly and leaked syntax"
        assert len(entities) > 0, "VLM failed to extract entities from HTML table"

    asyncio.run(run_test())
    
def test_html_table_regex_logic():
    html_content = "<div align=\"center\">\n    <table border=\"1\">\n        <tr>\n            <th>Feature</th>\n        </tr>\n    </table>\n</div>".lower()
    assert "<table" in html_content
    
    plain_div = "<div class='note'>This is just a note</div>".lower()
    assert "<table" not in plain_div
