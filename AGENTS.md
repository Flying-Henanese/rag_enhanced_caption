# Project Instructions: rag-enhanced-caption

This document provides foundational guidance and architectural mandates for the `rag-enhanced-caption` project.

## 🏗 Core Architecture & Tech Stack

- **Package Management**: Use `uv` for all dependency management and environment orchestration.
- **Build Backend**: `hatchling` (configured in `pyproject.toml`).
- **Markdown Parsing**: `markdown-it-py` is the core engine for token-based surgical extraction.
- **Asynchronous Pattern**: All VLM calls and processing pipelines MUST be `async`.
- **Project Structure**: Follows the standard `src` layout (`src/rag_enhanced_caption/`).

## 🛠 Coding Conventions

- **Linting & Formatting**: Default to using `ruff`. Run `ruff check .` and `ruff format .` after making significant code changes. Ruff enforces explicit function annotations and modern typing syntax for the core package; examples and tests remain gradual-adoption areas for these typing rules. The standard-library `logging` import ban applies project-wide.
- **Naming & Logging**: 
  - The main package name is `rag_enhanced_caption`.
  - Use `loguru` for all logging (`from loguru import logger`). Do not use the standard `logging` library.
- **Type Safety**: All functions and methods MUST have explicit type hints.
  - **Modern Typing (Python 3.10+)**: Use built-in collection types as generics (e.g., `list[str]`, `dict[str, Any]` instead of `typing.List` or `typing.Dict`). Use the `|` operator instead of `Union` or `Optional` (e.g., `str | None`).
- **Pythonic Structure**:
  - Prefer small helper functions over complex inline expressions or deeply nested comprehension logic.
  - Use `dataclass`, `TypedDict`, or dedicated result objects for stable structured data; avoid requiring callers to unpack more than three return values.
  - Avoid mutable default arguments. Use `None` plus explicit initialization and docstring notes for dynamic defaults.
  - Avoid deeply nested dictionaries/lists for core data contracts; promote stable shapes into typed objects or documented schemas.
- **Documentation**: Use Google-style docstrings for all classes and public methods.
- **Error Handling**: Use the `robust_json_parse` utility for all VLM output parsing to handle potential markdown formatting in AI responses.

## 🔄 Workflows

- **Harness Context Routing**:
  - Before broad project work, read `.harness/README.md` and the relevant files in `.harness/context/`.
  - Read `.harness/context/project-overview.md` when the task touches project structure, module ownership, package layout, dependencies, or where to start reading.
  - Read `.harness/context/artifact-contracts.md` before changing CLI outputs, example outputs, JSONL schemas, `text_for_embedding`, docstore fields, or sparse searchable objects.
  - Read `.harness/context/retrieval-pipeline.md` before changing BM25, hybrid retrieval, RRF, rerank, short-context expansion, AutoMerge, or LlamaIndex example wiring.
  - After changing source docs such as `README.md`, `README_zh.md`, `examples/README_zh.md`, or this file, update the matching harness context file if the project facts changed.
- **Testing**:
  - Always run Python commands through `uv run` (for example, `uv run python ...`) to ensure the correct project environment is used.
  - Run the focused processor tests: `uv run pytest tests/test_processor.py`.
  - Ensure a `.env` file exists in the project root or current working directory for VLM configuration.
- **Dependency Management**:
  - Use `uv add <package>` to add production core dependencies to `pyproject.toml`.
  - Use `uv add --dev <package>` for development and test dependencies (e.g., linters, testing frameworks).
  - **CRITICAL**: Do NOT add dependencies required only for demos, examples, integrations, or non-core features (e.g., `fastapi`, `llama-index-core`) to the main `dependencies` array. Use optional groups instead (e.g., `uv add --optional demo <package>`).
  - NEVER manually edit `pyproject.toml` dependency lists or `uv.lock`. All changes must go through `uv`.

## 🚫 Anti-patterns & Constraints

- **No Synchronous IO**: Avoid `requests` or synchronous `open()` in performance-critical paths; prefer async alternatives where applicable (though local file reads in this project are currently simple).
- **Minimal Dependencies**: Do not add heavy frameworks (like LangChain, LlamaIndex) or web frameworks (like FastAPI) as core dependencies; keep this as a lightweight utility library. If a dependency is only needed for an example, a demo, or a specific integration, add it to the corresponding optional dependency group (e.g., `[project.optional-dependencies] demo = [...]`) rather than the main dependencies list.
- **Clean Markdown**: Ensure injected `<details>` blocks are correctly closed and don't break existing Markdown rendering.
- **Performance Discipline**: Profile before optimizing. Avoid adding caching, concurrency, batching, or complex algorithms without measured bottlenecks or clear scale requirements; prefer streaming or iterator-based processing for large Markdown, JSONL, docstore, and retrieval-candidate flows when practical.

## 🤖 Agent Instructions (For AI Assistants)

- **Refactoring Scope**: Unless explicitly requested by the user, do not perform unrelated refactoring or "cleanups". Maintain surgical precision in your edits.
- **Docstring Updates**: When modifying a function or method signature (e.g., adding parameters, changing return types), you MUST synchronously update its Google-style docstring.
- **Error Handling in API Calls**: When handling LLM/VLM API calls, account for network exceptions and timeouts. Use narrow `try` blocks, prefer explicit exceptions over `None` for failure paths unless the optional result is documented, and log errors with full tracebacks using `logger.exception`.
