# RAG Enhanced Caption & Chunking Toolkit

A comprehensive multi-modal RAG (Retrieval-Augmented Generation) enhancement toolkit. This project provides surgical Markdown context extraction, VLM-powered image/table captioning, and structure-aware semantic chunking to build high-performance RAG pipelines.

Inspired by [RAG-Anything](https://github.com/HKUDS/RAG-Anything).

## ✨ Core Capabilities

- **VLM-Powered Enrichment**: Automatically analyzes images and tables within Markdown documents, generating descriptive captions and structured metadata using Vision Language Models (VLMs).
- **Structure-Aware Semantic Chunking**: An AST-based chunking engine that preserves document hierarchy (headers, breadcrumbs), keeps tables/lists intact, and uses embedding-based semantic splitting for long paragraphs.
- **Parent-Child RAG Strategy**: Demonstrates a sophisticated workflow that splits documents into searchable "Child" chunks (AI summaries) and contextual "Parent" blocks for optimal retrieval performance.
- **Surgical Context Extraction**: Uses `markdown-it-py` to precisely locate multi-modal elements and gather surrounding textual context for the VLM.

## 🔄 Dual-Track Processing System (双轨制)

The toolkit operates on two parallel tracks to satisfy both human readability and machine retrieval needs:

1.  **Track 1: Markdown Slicing & Preview**
    *   **Full Markdown Export**: Generates an enriched `.md` file where chunks are delimited by `---` (hr tokens), ideal for documentation portals or human review.
    *   **Flexible Granularity**: The parser can return a joined string for display or a `list[str]` of individual slices for custom processing.

2.  **Track 2: Parent-Child Hierarchical Indexing**
    *   **Two-Step Retrieval**: Generates two structured JSONL files (`index` for embeddings and `docstore` for full content).
    *   **Child Chunks**: High-precision, smaller fragments (e.g., AI-generated image summaries) optimized for vector search "hits".
    *   **Parent Chunks**: Larger, context-rich blocks (the original semantic section) that are retrieved once a child is matched, providing the LLM with the full "big picture" for more accurate answering.

## 🚀 Quick Start

### Installation

We recommend using [uv](https://github.com/astral-sh/uv) for fast dependency management.

```bash
# Clone the repository
git clone https://github.com/your-repo/rag-enhanced-caption.git
cd rag-enhanced-caption

# Install core dependencies
uv sync

# (Optional) Install local embedding support for semantic chunking
uv sync --extra local
```

### 1. Multi-modal Enrichment (VLM)

Analyze images and tables in your Markdown and inject descriptions as collapsible `<details>` blocks.

```python
import asyncio
from rag_enhanced_caption import MarkdownMultimodalProcessor, create_default_vlm_client

async def main():
    # Compatible with OpenAI, vLLM, SiliconFlow, etc.
    # Reads VLM_API_KEY, VLM_ENDPOINT from .env by default
    vlm_client = create_default_vlm_client()

    processor = MarkdownMultimodalProcessor(vlm_func=vlm_client)
    
    with open("document.md", "r", encoding="utf-8") as f:
        md_content = f.read()
    
    enriched_md = await processor.enrich_markdown(md_content, base_dir="./")
    print(enriched_md)

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Semantic Chunking

Split Markdown while preserving structural context and header breadcrumbs.

```python
from semantic_chunking_standalone.ragflow_like import semantic_chunk_with_metadata

chunks = semantic_chunk_with_metadata(
    markdown_content="# Section\n## Sub-section\nContent here...",
    file_id="doc_001",
    filename="guide.md",
    parser_config={"chunk_token_num": 512}
)

for chunk in chunks:
    print(f"Header: {chunk['title']}")
    print(f"Content: {chunk['content'][:100]}...")
```

### 3. Advanced Workflow: Parent-Child RAG

See `orchestrate.py` for a full implementation of:
1.  **Chunking**: Semantic splitting of the document.
2.  **Enrichment**: VLM analysis for each chunk.
3.  **Hierarchy**: Generating `index.jsonl` (for embeddings) and `docstore.jsonl` (for storage) with Parent-Child relationships.

## 🛠️ Project Structure

```text
rag_enhanced_caption/
├── orchestrate.py           # End-to-end Parent-Child RAG pipeline
├── src/
│   ├── rag_enhanced_caption/ # Core VLM & Extraction Logic
│   │   ├── clients.py       # VLM client factory (OpenAI-compatible)
│   │   ├── processor.py     # Main multi-modal processing engine
│   │   ├── context_extractor.py # Markdown AST traversal
│   │   └── json_utils.py    # Robust JSON parsing for AI outputs
│   └── semantic_chunking_standalone/
│       └── ragflow_like/    # Semantic chunking implementation
│           ├── parsers/     # AST-based Markdown parser
│           └── dispatcher.py # Chunking entry point with metadata
└── test_resource/           # Sample documents and images
```

## 🧠 Model Recommendations

- **Enrichment**: `qwen3-vl-8b-instruct` (via SiliconFlow or local vLLM) offers the best balance of accuracy and cost.
- **Embedding**: For semantic chunking, `bge-m3` or `text-embedding-3-small` are recommended.

## 📄 License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
