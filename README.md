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
- advanced ingestion and retrieval workflows under `examples/`
- tests for chunking, HTML cleaning, VLM enrichment, and retrieval-related behaviors

The README below reflects the repository as it exists now. Older references to a
standalone `backend/` application are intentionally omitted because it is not
part of the current tree.

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

## Hybrid BM25 + Embedding Retrieval

The advanced examples provide a second retrieval path for exact keywords that
may be weakened or omitted by semantic summaries. This is useful for table
headers and rows, dates, email addresses, product names, identifiers, and other
literal values.

This path is additive and does not change the `rag-caption` CLI output contract.
`examples/data_ingestion_pipeline.py` builds an additional
`<name>_sparse.jsonl` artifact alongside its vector index and docstore files.
Each record is a backend-neutral searchable object:

```json
{
  "id": "stable-field-id",
  "owner_node_id": "rag-anything_chunk_89",
  "searchable_text": "Project: MiniRAG; Description: ...",
  "field_type": "table_row",
  "metadata": {"row_index": 2},
  "schema_version": 1
}
```

Searchable objects are extracted deterministically without an LLM. The current
extractor recognizes Markdown and HTML table headers/rows, dates, and email
addresses. Its interface is intentionally pluggable so additional fixed-format
fields or storage backends can be added later. JSONL is the current persistence
backend; the schema is independent of BM25 and can later be mapped to
Elasticsearch or MongoDB.

At query time, `examples/llama_index_advanced_rag.py` uses this flow:

```text
embedding top-k + BM25 top-k
              ↓
 reciprocal rank fusion (RRF)
              ↓
 RecursiveRetriever → reranker → short-context expansion → AutoMerge
```

RRF combines rank positions rather than comparing incompatible vector and BM25
scores directly. Both paths resolve to the same LlamaIndex node IDs, so the
existing docstore remains the source of full content. If the sparse JSONL file
is missing or empty, the example automatically falls back to vector-only
retrieval.

Build the example artifacts for `rag-anything.md`, then run the advanced
retrieval example:

```bash
uv run python -c "import asyncio; from examples.data_ingestion_pipeline import process_document; asyncio.run(process_document('test_resource/rag-anything.md'))"
uv run python examples/llama_index_advanced_rag.py
```

The ingestion command creates:

```text
output/
├── rag-anything_index_new.jsonl
├── rag-anything_docstore_new.jsonl
└── rag-anything_sparse.jsonl
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
├── enhancer/
│   ├── cleaning_utils.py
│   ├── context_extractor.py
│   ├── processor.py
│   ├── prompts.py
│   └── vlm_client.py
├── lexical_search/
│   ├── bm25.py
│   ├── builder.py
│   ├── extractors.py
│   ├── fusion.py
│   ├── repository.py
│   └── schema.py
└── integrations/llama_index/
    ├── hybrid_retriever.py
    └── retrievers.py
```

## Test Coverage

The repository currently includes tests for:
- semantic chunking and parser edge cases
- HTML table cleaning / extraction
- VLM enrichment behavior
- searchable-object extraction, JSONL persistence, BM25 ranking, and RRF fusion
- vector and BM25 candidate integration with LlamaIndex
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

## Roadmap

- Improve chunk-size hard limits for extreme cases where a single very long sentence can still exceed the configured token threshold after sentence-level splitting.

## License

Apache-2.0.
