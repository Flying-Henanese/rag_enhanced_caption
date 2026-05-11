# RAG Enhanced Caption

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[**English**](README.md) | [**中文**](README_zh.md)

`rag-enhanced-caption` is a lightweight Python toolkit for the last-mile processing of parsed Markdown before RAG ingestion.

It focuses on three things:
- extracting local context for images and tables from Markdown structure
- generating VLM-based semantic summaries for multimodal elements
- producing structure-aware chunks for embedding and retrieval

The project assumes your upstream pipeline has already converted source documents into Markdown. It is not a PDF parser or OCR engine.

## What It Produces

Given one Markdown file, the CLI generates three artifacts:
- `<name>_enhanced.md`: Markdown preview with multimodal analysis injected
- `<name>_index.jsonl`: lightweight records intended for embedding / vector indexing
- `<name>_docstore.jsonl`: full parent-child records intended for a docstore

This split helps keep noisy raw Markdown, large tables, and image references out of the embedding text while preserving full content for downstream retrieval.

## Core Capabilities

- Structure-aware Markdown chunking based on `markdown-it-py`
- Heading-path preservation through `header_path` metadata
- Image and caption binding to avoid semantic fragmentation
- Table / image enrichment with an OpenAI-compatible VLM endpoint
- Parent-child output layout for multi-vector retrieval workflows
- Optional remote or local embedding clients for semantic splitting

## Current Project Scope

The current repository is centered on:
- the `rag-caption` CLI for end-to-end processing
- reusable Python modules under `src/rag_enhanced_caption/`
- tests for chunking, HTML cleaning, VLM enrichment, and retrieval-related behaviors

The README below reflects the repository as it exists now. Older references to `backend/` or `examples/` are intentionally removed because those directories are not part of the current tree.

## Installation

### Install from PyPI-compatible source

```bash
pip install .
```

### Development setup with `uv`

```bash
git clone https://github.com/Flying-Henanese/rag_enhanced_caption.git
cd rag_enhanced_caption
uv sync
```

## Environment Variables

Create a local `.env` in the project root or working directory.

### VLM settings

Required when you want image / table semantic enhancement:

```env
VLM_API_KEY=your_api_key
VLM_ENDPOINT=https://api.siliconflow.cn/v1/chat/completions
VLM_MODEL_NAME=Qwen/Qwen2.5-VL-72B-Instruct
```

Notes:
- `VLM_ENDPOINT` must be OpenAI-compatible.
- If you point to a local compatible server, the same client can still be used.

### Embedding settings

Used by semantic chunk splitting when remote embeddings are enabled:

```env
EMBEDDING_API_KEY=your_api_key
EMBEDDING_ENDPOINT=https://api.siliconflow.cn/v1/embeddings
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_TIMEOUT=60
```

Optional local embedding model:

```env
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

Notes:
- If no remote embedding client is available, the parser degrades to simpler text splitting.
- Local embeddings require `sentence-transformers`, which is not installed by default.

## CLI Usage

After installation, use:

```bash
rag-caption INPUT_MARKDOWN OUTPUT_DIR
```

Example:

```bash
rag-caption ./test_resource/paddleocr.md ./output
```

Expected output files:

```text
output/
├── paddleocr_enhanced.md
├── paddleocr_index.jsonl
└── paddleocr_docstore.jsonl
```

## JSONL Output Shape

`_index.jsonl` contains compact embedding-facing records:

```json
{"id": "p_xxx", "text_for_embedding": "...", "metadata": {"type": "parent", "source": "demo.md"}}
{"id": "c_xxx_0", "parent_id": "p_xxx", "text_for_embedding": "...", "metadata": {"type": "child", "source": "demo.md"}}
```

`_docstore.jsonl` contains full records for retrieval-time expansion:

```json
{
  "id": "c_xxx_0",
  "parent_id": "p_xxx",
  "type": "child",
  "text_for_embedding": "semantic summary",
  "full_content": "![image](path)\n\n...analysis...",
  "metadata": {
    "chunk_type": "multimodal",
    "image_url": "path",
    "entities": ["entity_a", "entity_b"]
  }
}
```

## Python Usage

### 1. Chunk Markdown with metadata

```python
from rag_enhanced_caption import semantic_chunk_with_metadata

markdown = """
# Demo

Some text.

| Name | Value |
| --- | --- |
| A | 1 |
"""

records = semantic_chunk_with_metadata(
    markdown_content=markdown,
    file_id="demo",
    filename="demo.md",
    parser_config={"chunk_token_num": 512},
)

for record in records:
    print(record["id"], record["metadata"]["element_type"])
```

### 2. Enrich multimodal chunks with a VLM

```python
import asyncio

from rag_enhanced_caption import (
    MarkdownMultimodalProcessor,
    create_default_vlm_client,
    semantic_chunk_with_metadata,
)

markdown = """
# Demo

![chart](./chart.png)
Figure 1. Revenue growth by region.
"""


async def main() -> None:
    chunks = semantic_chunk_with_metadata(
        markdown_content=markdown,
        file_id="demo",
        filename="demo.md",
    )

    processor = MarkdownMultimodalProcessor(
        vlm_func=create_default_vlm_client(),
        max_concurrency=2,
    )

    enriched = await processor.enrich_chunks(chunks, base_dir=".")
    for chunk in enriched:
        print(chunk["id"], chunk.get("text_for_embedding", ""))


asyncio.run(main())
```

## Package Layout

```text
src/rag_enhanced_caption/
├── cli.py
├── chunker/
│   ├── dispatcher.py
│   ├── embed_client.py
│   ├── parsers/semantic.py
│   └── utils/
└── enhancer/
    ├── cleaning_utils.py
    ├── context_extractor.py
    ├── processor.py
    ├── prompts.py
    └── vlm_client.py
```

## Test Coverage

The repository currently includes tests for:
- semantic chunking and parser edge cases
- HTML table cleaning / extraction
- VLM enrichment behavior
- integration-oriented retrieval and rerank flows

Typical commands:

```bash
uv run pytest tests/test_semantic_parser_fixes.py
uv run pytest tests/test_html_table.py
uv run pytest tests/test_processor.py
```

Some tests require external API credentials and will skip if the corresponding environment variables are missing.

## Notes and Limitations

- Input must already be Markdown.
- VLM enhancement depends on network access to a compatible endpoint.
- SVG images are currently skipped during VLM analysis.
- Some retrieval-related tests reference optional integration code paths and external services.

## License

Apache-2.0.
