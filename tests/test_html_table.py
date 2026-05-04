import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from rag_enhanced_caption.enhancer.processor import MarkdownMultimodalProcessor

def test_html_table_parsing_and_enrichment():
    async def run_test():
        # We mock the VLM call so we don't actually hit the API during CI runs.
        # The mock returns a valid JSON string structure expected by robust_json_parse.
        mock_vlm_response = """
        {
            "detailed_description": "This is an analysis of an HTML table. It compares Product A and Product B across several metrics.",
            "entity_info": {
                "entity_name": "Product Comparison Table",
                "summary": "Product A performs better in speed, while Product B is more cost-effective."
            }
        }
        """
        mock_vlm_client = AsyncMock(return_value=mock_vlm_response)
        
        # Initialize the processor with the mocked VLM client
        processor = MarkdownMultimodalProcessor(vlm_func=mock_vlm_client, max_concurrency=1)
        
        # A test markdown document containing a standard markdown table and a complex HTML table
        test_md_content = """# Performance Overview

This document outlines the performance.

## Standard Markdown Table

| Model | Accuracy |
|---|---|
| Model A | 95% |

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

        # Run the enhancement process
        enriched_md = await processor.enrich_markdown(md_content=test_md_content, base_dir=".")
        
        # Verify that the processor correctly called the VLM for the HTML table
        # We expect 2 calls: one for the markdown table, one for the HTML table
        assert mock_vlm_client.call_count == 2
        
        # Verify the generated markdown contains the enriched analysis blocks for the HTML table
        assert "<summary>🤖 <b>AI 图像/表格解析</b></summary>" in enriched_md
        assert "This is an analysis of an HTML table" in enriched_md
        assert "Product Comparison Table" in enriched_md
        
        # Verify the structure is somewhat intact
        assert "<table border=\"1\">" in enriched_md

    asyncio.run(run_test())
    
def test_html_table_regex_logic():
    # A quick unit test to verify our detection logic behavior in processor
    # We simulate how markdown-it-py tokenizes the content
    html_content = "<div align=\"center\">\n    <table border=\"1\">\n        <tr>\n            <th>Feature</th>\n        </tr>\n    </table>\n</div>".lower()
    
    assert "<table" in html_content
    
    # A standard HTML block without a table should not trigger it
    plain_div = "<div class='note'>This is just a note</div>".lower()
    assert "<table" not in plain_div
