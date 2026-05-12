# Project Instructions: rag-enhanced-caption

This document provides foundational guidance and architectural mandates for the `rag-enhanced-caption` project.

## 🏗 Core Architecture & Tech Stack

- **Package Management**: Use `uv` for all dependency management and environment orchestration.
- **Build Backend**: `hatchling` (configured in `pyproject.toml`).
- **Markdown Parsing**: `markdown-it-py` is the core engine for token-based surgical extraction.
- **Asynchronous Pattern**: All VLM calls and processing pipelines MUST be `async`.
- **Project Structure**: Follows the standard `src` layout (`src/rag_enhanced_caption/`).
- **Public API Stability**: The following modules are primary public-facing entry points. Avoid making breaking changes to their public class/function signatures:
  - `src/rag_enhanced_caption/chunker/dispatcher.py` (`MarkdownSemanticDispatcher`)
  - `src/rag_enhanced_caption/chunker/embed_client.py` (Embedding client factories)
  - `src/rag_enhanced_caption/enhancer/processor.py` (`MarkdownMultimodalProcessor`)
  - `src/rag_enhanced_caption/enhancer/context_extractor.py` (`MarkdownContextExtractor`)

## 🛠 Coding Conventions

- **Linting & Formatting**: Default to using `ruff`. Run `ruff check .` and `ruff format .` after making significant code changes.
- **Naming & Logging**: 
  - The main package name is `rag_enhanced_caption`.
  - Use `loguru` for all logging (`from loguru import logger`). Do not use the standard `logging` library.
- **Type Safety**: All functions and methods MUST have explicit type hints.
  - **Modern Typing (Python 3.10+)**: Use built-in collection types as generics (e.g., `list[str]`, `dict[str, Any]` instead of `typing.List` or `typing.Dict`). Use the `|` operator instead of `Union` or `Optional` (e.g., `str | None`).
- **Documentation**: Use Google-style docstrings for all classes and public methods.
- **Error Handling**: Use the `robust_json_parse` utility for all VLM output parsing to handle potential markdown formatting in AI responses.

## 🔄 Workflows

- **Testing**:
  - Always run Python commands through `uv run` (for example, `uv run python ...`) to ensure the correct project environment is used.
  - Run the functional test script: `uv run python tests/test_processor.py`.
  - Ensure a `.env` file exists in the root or `src` directory for VLM configuration.
- **Dependency Management**:
  - Use `uv add <package>` to add production core dependencies to `pyproject.toml`.
  - Use `uv add --dev <package>` for development and test dependencies (e.g., linters, testing frameworks).
  - **CRITICAL**: Do NOT add dependencies required only for demos, examples, integrations, or non-core features (e.g., `fastapi`, `llama-index-core`) to the main `dependencies` array. Use optional groups instead (e.g., `uv add --optional demo <package>`).
  - NEVER manually edit `pyproject.toml` dependency lists or `uv.lock`. All changes must go through `uv`.

## 🚫 Anti-patterns & Constraints

- **No Synchronous IO**: Avoid `requests` or synchronous `open()` in performance-critical paths; prefer async alternatives where applicable (though local file reads in this project are currently simple).
- **Minimal Dependencies**: Do not add heavy frameworks (like LangChain, LlamaIndex) or web frameworks (like FastAPI) as core dependencies; keep this as a lightweight utility library. If a dependency is only needed for an example, a demo, or a specific integration, add it to the corresponding optional dependency group (e.g., `[project.optional-dependencies] demo = [...]`) rather than the main dependencies list.
- **Clean Markdown**: Ensure injected `<details>` blocks are correctly closed and don't break existing Markdown rendering.

## 🤖 Agent Instructions (For AI Assistants)

- **Refactoring Scope**: Unless explicitly requested by the user, do not perform unrelated refactoring or "cleanups". Maintain surgical precision in your edits.
- **Docstring Updates**: When modifying a function or method signature (e.g., adding parameters, changing return types), you MUST synchronously update its Google-style docstring.
- **Error Handling in API Calls**: When handling LLM/VLM API calls, account for network exceptions and timeouts. Use `try...except` blocks and log errors with full tracebacks using `logger.exception`.
