# Project Instructions: rag-enhanced-caption

This document provides foundational guidance and architectural mandates for the `rag-enhanced-caption` project.

## 🏗 Core Architecture & Tech Stack

- **Package Management**: Use `uv` for all dependency management and environment orchestration.
- **Build Backend**: `hatchling` (configured in `pyproject.toml`).
- **Markdown Parsing**: `markdown-it-py` is the core engine for token-based surgical extraction.
- **Asynchronous Pattern**: All VLM calls and processing pipelines MUST be `async`.
- **Project Structure**: Follows the standard `src` layout (`src/rag_enhanced_caption/`).

## 🛠 Coding Conventions

- **Naming**: 
  - The main package name is `rag_enhanced_caption`.
  - Loggers must be named following the pattern `rag_enhanced_caption.<module_name>`.
- **Type Safety**: All functions and methods MUST have explicit type hints.
- **Documentation**: Use Google-style docstrings for all classes and public methods.
- **Error Handling**: Use the `robust_json_parse` utility for all VLM output parsing to handle potential markdown formatting in AI responses.

## 🔄 Workflows

- **Testing**:
  - Run the functional test script: `python tests/test_processor.py`.
  - Ensure a `.env` file exists in the root or `src` directory for VLM configuration.
- **Adding Dependencies**: Use `uv add <package>` to update `pyproject.toml`.

## 🚫 Anti-patterns & Constraints

- **No Synchronous IO**: Avoid `requests` or synchronous `open()` in performance-critical paths; prefer async alternatives where applicable (though local file reads in this project are currently simple).
- **Minimal Dependencies**: Do not add heavy frameworks (like LangChain or LlamaIndex) as core dependencies; keep this as a lightweight utility library.
- **Clean Markdown**: Ensure injected `<details>` blocks are correctly closed and don't break existing Markdown rendering.
