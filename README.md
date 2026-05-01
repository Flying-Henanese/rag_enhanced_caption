# RAG Enhanced Caption & Chunking Toolkit

A modular multi-modal RAG (Retrieval-Augmented Generation) enhancement toolkit. This project provides decoupled modules for surgical Markdown context extraction, VLM-powered image/table captioning, and structure-aware semantic chunking.

Inspired by [RAG-Anything](https://github.com/HKUDS/RAG-Anything).

## ✨ Core Capabilities

- **VLM-Powered Enrichment (`enhancer`)**: Automatically analyzes images and tables within Markdown documents, generating descriptive captions and structured metadata.
- **Structure-Aware Semantic Chunking (`chunker`)**: An AST-based chunking engine that preserves document hierarchy (headers, breadcrumbs) and uses embedding-based semantic splitting.
- **Parent-Child RAG Strategy**: A built-in workflow to split documents into searchable "Child" chunks (AI summaries) and contextual "Parent" blocks.
- **Fully Decoupled**: Use the chunker or the enhancer independently or together as a suite.

## 🔄 Dual-Track Processing System (双轨制)

The toolkit operates on two parallel tracks to satisfy both human readability and machine retrieval needs:

1.  **Track 1: Markdown Slicing & Preview**
    *   **Full Markdown Export**: Generates an enriched `.md` file where chunks are delimited by `---` (hr tokens), ideal for human review.
    *   **Flexible Granularity**: The parser can return a joined string or a `list[str]` of individual slices.

2.  **Track 2: Parent-Child Hierarchical Indexing**
    *   **Two-Step Retrieval**: Generates structured JSONL files (`index` for embeddings and `docstore` for full content).
    *   **Child Chunks**: High-precision fragments optimized for vector search "hits".
    *   **Parent Chunks**: Context-rich blocks retrieved to provide the LLM with the full "big picture".

## 🚀 Quick Start

### Installation

We recommend using [uv](https://github.com/astral-sh/uv) for fast dependency management.

```bash
# Clone the repository
git clone https://github.com/Flying-Henanese/rag_enhanced_caption.git
cd rag_enhanced_caption

# Install core dependencies
uv sync

# (Optional) Install local embedding support for semantic chunking
uv sync --extra local
```

### 1. Multi-modal Enrichment (VLM)

```python
import asyncio
from rag_enhanced_caption.enhancer.processor import MarkdownMultimodalProcessor
from rag_enhanced_caption.enhancer.vlm_client import create_default_vlm_client

async def main():
    vlm_client = create_default_vlm_client() # Reads VLM_API_KEY from .env
    processor = MarkdownMultimodalProcessor(vlm_func=vlm_client)
    
    with open("document.md", "r", encoding="utf-8") as f:
        md_content = f.read()
    
    enriched_md = await processor.enrich_markdown(md_content, base_dir="./")
    print(enriched_md)

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Semantic Chunking

```python
from rag_enhanced_caption.chunker.dispatcher import chunk_markdown as semantic_chunk_with_metadata

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

See `src/example_orchestrator.py` for a full implementation combining chunking and VLM enrichment.

## 🛠️ Project Structure

```text
rag_enhanced_caption/
├── src/
│   ├── example_orchestrator.py      # End-to-end Parent-Child RAG pipeline
│   └── rag_enhanced_caption/
│       ├── enhancer/                # VLM Enrichment Module
│       │   ├── processor.py         # Main engine
│       │   ├── context_extractor.py # Context logic
│       │   └── vlm_client.py        # OpenAI-compatible client
│       └── chunker/                 # Semantic Chunking Module
│           ├── dispatcher.py        # Entry point with metadata
│           ├── embed_client.py      # Embedding client
│           └── parsers/             # AST-based Markdown parser
└── tests/                           # Unified test suite
```

## 🧠 Model Recommendations

- **Enrichment**: `qwen3-vl-8b-instruct` (via SiliconFlow or local vLLM).
- **Embedding**: `bge-m3` or `text-embedding-3-small`.

## 📄 License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
