# RAG Enhanced Caption

Analyze and generate captions for images and tables in Markdown documents, inspired by the enhanced caption mechanism in [RAG-Anything](https://github.com/HKUDS/RAG-Anything).

This tiny project provides a surgical way to extract context around multimodal elements (images/tables) in Markdown files and uses Vision Language Models (VLMs) to generate descriptive captions that are injected back into the document as collapsible blocks.

## ✨ Features

- **Surgical Context Extraction**: Uses `markdown-it-py` to precisely locate images/tables and extract surrounding text (headers, previous/next paragraphs).
- **Collapsible AI Analysis**: Injects analysis results into Markdown using `<details>` blocks to keep the document readable while providing rich metadata for RAG.
- **Flexible Image Resolving**: Supports local files, remote URLs, and Base64 inline data.
- **VLM Agnostic**: Easily plug in any VLM (GPT-4o, Claude 3.5 Sonnet, DeepSeek-VL, etc.) via a simple async function.
- **Standardized Structure**: Modern Python project layout managed by `uv` and `hatchling`.

## 💡 Best Practices for Better Captions

This tool is highly sensitive to the hierarchy of your document. For the best results, especially when dealing with pre-processed or chunked Markdown:

- **Breadcrumb Headers**: If your Markdown headers include hierarchical paths (e.g., `# Document | Section | Subsection`), the context extractor will automatically parse these to provide the VLM with a precise "location path".
- **Hierarchical Chunking**: Using tools that preserve breadcrumbs in headers significantly improves the VLM's understanding of where an image or table sits within a complex document.

## 🚀 Quick Start

### Installation

We recommend using [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

```bash
# Clone the repository
git clone https://github.com/your-repo/rag-enhanced-caption.git
cd rag-enhanced-caption

# Create a virtual environment and install dependencies
uv venv
uv pip install -e .
```

### Basic Usage

```python
import asyncio
from rag_enhanced_caption import MarkdownMultimodalProcessor

# Define your VLM caller
async def my_vlm_func(user_prompt: str, system_prompt: str, image_base64: str = None) -> str:
    # Call your preferred LLM API here (e.g., OpenAI, Anthropic)
    # Expected to return a JSON-like string with "detailed_description" and "entity_info"
    return '{"detailed_description": "...", "entity_info": {...}}'

async def main():
    processor = MarkdownMultimodalProcessor(vlm_func=my_vlm_func)
    
    with open("document.md", "r", encoding="utf-8") as f:
        md_content = f.read()
    
    enriched_md = await processor.enrich_markdown(md_content, base_dir="./")
    
    with open("document_enriched.md", "w", encoding="utf-8") as f:
        f.write(enriched_md)

if __name__ == "__main__":
    asyncio.run(main())
```

## 🛠️ Project Structure

```text
rag_enhanced_caption/
├── pyproject.toml           # Project metadata and dependencies
├── src/
│   └── rag_enhanced_caption/
│       ├── __init__.py      # Exports main classes
│       ├── processor.py     # Core multimodal processing logic
│       ├── context_extractor.py # Markdown context extraction logic
│       └── image_utils.py   # Image loading and resolving
└── tests/                   # Test scripts and suites
```

## 📄 License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
