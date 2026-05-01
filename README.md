# RAG Enhanced Caption & Chunking Toolkit

[**English**](README.md) | [**中文**](README_zh.md)

A modular multi-modal RAG (Retrieval-Augmented Generation) enhancement toolkit. This project provides decoupled modules for surgical Markdown context extraction, VLM-powered image/table captioning, and structure-aware semantic chunking.

Inspired by [RAG-Anything](https://github.com/HKUDS/RAG-Anything).

## 💡 Our Philosophy: Purpose-Driven vs. OCR-Driven

Unlike standard multimodal RAG tools that simply OCR and transcribe image text, this toolkit focuses on **illustrative intent**:
- **Context-Aware**: The VLM knows *where* the image/table is located and *what* the surrounding text says.
- **Intent-Focused**: It explains **why** an element exists (e.g., "This chart proves the high-concurrency capability mentioned in Section 2.1") rather than just listing numbers.
- **No Attention Hijacking**: By generating concise purpose-driven summaries, we prevent the LLM from getting "distracted" by irrelevant raw data inside images.

## 🎯 Scope & Prerequisites: The "Last Mile" of Document Processing

Please note that **this toolkit is designed to process *already parsed* Markdown documents.** It is not a PDF parser or an OCR engine (you can use tools like MinerU, Docling, or Marker for that upstream step). 

Our toolkit serves as the critical **"last mile"** processing step—refining, enriching, and structuring the Markdown data—right before it is ingested into a Vector Database for retrieval or used to construct a Knowledge Graph.

## ✨ Core Capabilities

- **VLM-Powered Enrichment (`enhancer`)**: Automatically analyzes images and tables. Analysis results are injected as collapsible `<details>` blocks **directly below** the element for optimal cognitive flow.
- **Surgical Context Extraction**: Uses `markdown-it-py` to gather deep situational awareness (breadcrumbs, parent headers, and neighboring paragraphs) for the VLM.
- **Structure-Aware Semantic Chunking (`chunker`)**: Unlike traditional naive character-count chunking, our engine uses AST (Abstract Syntax Tree) parsing to preserve Markdown document hierarchy and leverages Embedding models for true semantic splitting.
- **Parent-Child RAG Strategy**: A built-in workflow to split documents into searchable high-density "Child" chunks (AI intent summaries) and context-rich "Parent" blocks (original content).
- **Fully Decoupled**: Use the chunker or the enhancer independently or together.

## 🔄 Dual-Track Processing System

1.  **Track 1: Markdown Slicing & Preview**
    *   **Enriched Markdown**: Generates a `.md` file with delimited chunks (`---`), ideal for human review or documentation portals.
2.  **Track 2: Parent-Child Hierarchical Indexing**
    *   **Small-to-Big Retrieval**: Generates `_index.jsonl` (pure semantic hooks for vector search) and `_docstore.jsonl` (full content retrieval with metadata).

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Flying-Henanese/rag_enhanced_caption.git
cd rag_enhanced_caption

# Install dependencies using uv
uv sync
uv sync --extra local # Optional: for local embedding support
```

### ⚙️ Configuration
Create a `.env` file in the root directory:
```env
VLM_API_KEY=your_key_here
VLM_ENDPOINT=https://api.siliconflow.cn/v1/chat/completions
VLM_MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
# If using semantic chunking with remote embeddings:
EMBEDDING_API_KEY=your_key_here
EMBEDDING_ENDPOINT=https://api.siliconflow.cn/v1/embeddings
EMBEDDING_MODEL_NAME=BAAI/bge-m3
```

### 1. Command Line Interface (CLI) - Recommended

Run the included CLI tool to process a document through the full pipeline (Chunking -> VLM Enrichment -> Parent-Child Split):

```bash
# Usage: python cli.py <input_markdown_file> <output_directory>
python cli.py ./docs/input.md ./output_folder
```
This will generate three files in the specified output directory:
- `*_enhanced.md`: The enriched markdown with AI annotations.
- `*_index.jsonl`: The lightweight index file for vector search.
- `*_docstore.jsonl`: The full payload document store file.

### 2. Multi-modal Enrichment (Standalone)

```python
import asyncio
from rag_enhanced_caption import MarkdownMultimodalProcessor, create_default_vlm_client

async def main():
    vlm_client = create_default_vlm_client()
    processor = MarkdownMultimodalProcessor(vlm_func=vlm_client)
    
    with open("doc.md", "r") as f:
        enriched_md = await processor.enrich_markdown(f.read(), base_dir="./")
    print(enriched_md)

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Semantic Chunking (Standalone)

```python
from semantic_chunking_standalone.ragflow_like import semantic_chunk_with_metadata

chunks = semantic_chunk_with_metadata(
    markdown_content="# Header\nContent...",
    file_id="doc_001",
    filename="doc.md",
    parser_config={"chunk_token_num": 512}
)
```

## 🛠️ Project Structure

```text
rag_enhanced_caption/
├── orchestrate.py                   # End-to-end Parent-Child RAG pipeline script
├── src/
│   ├── rag_enhanced_caption/        # Top-level package
│   │   ├── chunker/                 # Semantic Chunking (Structure-Aware)
│   │   └── enhancer/                # VLM Enrichment (Visual-to-Text)
│   └── semantic_chunking_standalone/# Independent chunking logic
└── tests/                           # Unified test suite
```

## 📄 License
Apache-2.0 License.