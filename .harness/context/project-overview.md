# Project Overview

Sources: `README.md`, `README_zh.md`, `AGENT.md`

`rag-enhanced-caption` is a lightweight Python toolkit for last-mile processing
of parsed Markdown before RAG ingestion. It assumes an upstream system has
already converted PDFs, Office files, images, or other sources into Markdown.
It is not a PDF parser or OCR engine.

## Core Purpose

The project focuses on three responsibilities:

- Extract local context for images and tables from Markdown structure.
- Generate VLM-based semantic summaries for multimodal elements.
- Produce structure-aware chunks for embedding, retrieval, and docstore
  expansion.

## Current Repository Scope

- `rag-caption` CLI for end-to-end Markdown processing.
- Reusable Python modules under `src/rag_enhanced_caption/`.
- Advanced ingestion and retrieval workflows under `examples/`.
- Tests for chunking, HTML cleaning, VLM enrichment, lexical search, and
  retrieval-related behavior.

Historical references to a standalone `backend/` application are out of scope
for the current tree.

## Core Package Structure

```text
src/rag_enhanced_caption/
+-- cli.py
+-- chunker/
|   +-- dispatcher.py
|   +-- embed_client.py
|   +-- parsers/semantic.py
|   +-- utils/
+-- enhancer/
|   +-- cleaning_utils.py
|   +-- context_extractor.py
|   +-- processor.py
|   +-- prompts.py
|   +-- vlm_client.py
+-- lexical_search/
|   +-- bm25.py
|   +-- builder.py
|   +-- extractors.py
|   +-- fusion.py
|   +-- repository.py
|   +-- schema.py
+-- integrations/llama_index/
    +-- hybrid_retriever.py
    +-- retrievers.py
```

## Module Responsibilities

- `src/rag_enhanced_caption/cli.py`: production CLI entry point. It reads one
  Markdown file, runs semantic chunking and multimodal enrichment, and writes
  the base CLI artifacts.
- `src/rag_enhanced_caption/chunker/`: Markdown-to-chunk layer. This owns
  structure-aware parsing, heading metadata, token/sentence splitting, and
  parser dispatch.
- `src/rag_enhanced_caption/enhancer/`: enrichment layer. This owns local
  context extraction, image path resolution, VLM calls, response parsing,
  Markdown cleanup, and multimodal `text_for_embedding` generation.
- `src/rag_enhanced_caption/lexical_search/`: backend-neutral lexical retrieval
  layer. This owns searchable-object schemas, deterministic extraction, JSONL
  persistence, BM25 ranking, and rank fusion.
- `src/rag_enhanced_caption/integrations/llama_index/`: optional LlamaIndex
  adapter layer. This owns hybrid retrieval composition and retriever helpers
  such as short-context expansion.
- `examples/`: demonstration orchestration. Example scripts can combine core
  library pieces with optional dependencies, but they should not redefine the
  core package contract.
- `tests/`: regression boundaries. Tests are the best starting point for
  confirming whether a behavior is intended, especially around parsing,
  artifacts, lexical search, and retrieval.

## Development Entrypoints

- CLI output or base artifact behavior:
  - Read `src/rag_enhanced_caption/cli.py`.
  - Check `tests/test_cli_process_document.py`.
  - Read `artifact-contracts.md` before changing file names or fields.
- Markdown parsing or heading/chunk behavior:
  - Read `src/rag_enhanced_caption/chunker/dispatcher.py`.
  - Read `src/rag_enhanced_caption/chunker/parsers/semantic.py`.
  - Check `tests/test_semantic_chunking.py`,
    `tests/test_semantic_parser_fixes.py`, and
    `tests/test_heading_level_inference.py`.
- Table cleanup or HTML table handling:
  - Read `src/rag_enhanced_caption/enhancer/cleaning_utils.py`.
  - Read `src/rag_enhanced_caption/chunker/utils/table_utils.py`.
  - Check `tests/test_html_table.py`.
- VLM prompt or multimodal enrichment behavior:
  - Read `src/rag_enhanced_caption/enhancer/prompts.py`.
  - Read `src/rag_enhanced_caption/enhancer/processor.py`.
  - Read `src/rag_enhanced_caption/enhancer/vlm_client.py`.
  - Check `tests/test_processor.py` and `tests/test_cli_process_document.py`.
- Sparse/BM25/searchable-object behavior:
  - Read `src/rag_enhanced_caption/lexical_search/`.
  - Read `examples/data_ingestion_pipeline.py`.
  - Check `tests/test_lexical_search.py`,
    `tests/test_searchable_object_builder.py`, and
    `tests/test_example_lexical_pipeline.py`.
- LlamaIndex retrieval, hybrid retrieval, or context expansion:
  - Read `src/rag_enhanced_caption/integrations/llama_index/`.
  - Read `examples/llama_index_advanced_rag.py`.
  - Read `retrieval-pipeline.md`.
  - Check `tests/test_hybrid_retriever.py`,
    `tests/test_context_expanding_retriever.py`, and
    `tests/test_retrieval_components.py`.

## Non-Goals and Boundaries

- The project is not an OCR engine, PDF parser, Office parser, backend service,
  or complete question-answering application.
- The CLI contract is intentionally smaller than the advanced example flow.
  `rag-caption` writes enhanced Markdown, index JSONL, and docstore JSONL; the
  sparse JSONL artifact currently belongs to the example ingestion pipeline.
- Optional integrations and demos should not force heavy dependencies into the
  core package dependency list.
- Retrieval examples may compose LlamaIndex, rerankers, and BM25, but the core
  library should remain useful without those optional integrations.

## Engineering Constraints

- Use `uv` for dependency management and command execution.
- Build backend is `hatchling`.
- Markdown parsing is based on `markdown-it-py`.
- VLM calls and processing pipelines should remain async.
- Source layout follows `src/rag_enhanced_caption/`.
- Use `loguru` for logging.
- Use explicit type hints and Python 3.10+ typing style.
- Use Google-style docstrings for public classes and methods.
- Keep the library lightweight. Heavy demo or integration dependencies belong
  in optional dependency groups, not core dependencies.

## Common Commands

```bash
uv sync
uv run pytest tests/test_semantic_parser_fixes.py
uv run pytest tests/test_html_table.py
uv run pytest tests/test_processor.py
uv run ruff check .
uv run ruff format .
```
