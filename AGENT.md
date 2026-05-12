# AGENT.md — Codex Execution Guide for `rag-enhanced-caption`

This file defines how Codex should operate in this repository. Follow these rules for all code changes unless the user explicitly asks otherwise.

## 1. Core Architecture & Tech Stack

- Package/dependency management: use `uv`.
- Build backend: `hatchling` (configured in `pyproject.toml`).
- Markdown parsing core: `markdown-it-py`.
- VLM pipeline design: keep calls and processing flows asynchronous.
- Project layout: standard `src` structure under `src/rag_enhanced_caption/`.

## 2. Coding Conventions

- Lint/format: use `ruff`.
  - Run `ruff check .` and `ruff format .` after meaningful changes.
- Naming/logging:
  - Main package name is `rag_enhanced_caption`.
  - Use `loguru` logging (`from loguru import logger`), not `logging`.
- Type safety:
  - Add explicit type hints to all functions/methods.
  - Use modern Python typing (3.10+): `list[str]`, `dict[str, Any]`, `X | None`.
- Documentation:
  - Use Google-style docstrings for classes and public methods.
- VLM output parsing:
  - Use `robust_json_parse` for VLM/LLM structured output parsing.
- Public API Stability:
  - The following modules are primary exposed interfaces. Avoid changing or breaking their public API signatures (class constructors and main methods) to maintain compatibility for external users:
    - `src/rag_enhanced_caption/chunker/dispatcher.py` (`MarkdownSemanticDispatcher`)
    - `src/rag_enhanced_caption/chunker/embed_client.py` (Embedding client factories)
    - `src/rag_enhanced_caption/enhancer/processor.py` (`MarkdownMultimodalProcessor`)
    - `src/rag_enhanced_caption/enhancer/context_extractor.py` (`MarkdownContextExtractor`)

## 3. Workflow Requirements

- Testing:
  - Always run Python commands via `uv run` (for example, `uv run python ...`), rather than bare `python`, to ensure the project environment is used consistently.
  - Minimum validation after code changes:
    - Run `ruff check .`.
    - Run `uv run python tests/test_processor.py` when the change can affect processing logic, parsing behavior, or VLM flows.
  - If tests are skipped (e.g., missing external credentials, non-functional doc-only change, or environment constraints), explicitly state what was skipped and why in the final response.
  - Run `uv run python tests/test_processor.py` for functional validation when relevant.
  - Ensure `.env` exists at repo root or `src/` for VLM config.
- Dependency management (strict):
  - Add runtime deps via `uv add <package>`.
  - Add dev/test deps via `uv add --dev <package>`.
  - For demo/example/integration-only deps, use optional groups, e.g. `uv add --optional demo <package>`.
  - Do not place non-core dependencies (e.g., web frameworks, heavyweight integration stacks) into main `dependencies`.
  - Never manually edit dependency lists in `pyproject.toml` or `uv.lock`.

## 4. Constraints & Anti-Patterns

- Avoid synchronous I/O in performance-critical paths where async alternatives are appropriate.
- Keep the library lightweight: avoid introducing heavy frameworks as core dependencies.
- Preserve Markdown correctness:
  - Ensure injected `<details>` blocks are valid and properly closed.
  - Do not break existing Markdown rendering.

## 5. Codex Editing Behavior

- Keep edits surgical. Do not perform unrelated refactors/cleanup unless requested.
- If a function/method signature changes, update related Google-style docstrings in the same change.
- For LLM/VLM API calls:
  - Handle network errors and timeouts with `try...except`.
  - Use `logger.exception(...)` when logging exceptions that need full traceback context.

## 6. Priority & Conflict Handling

- Direct user instructions override this file.
- If rules conflict, prefer minimal, safe changes that preserve existing architecture.
- If a required change would violate these standards, explicitly call out the tradeoff in the final response.

## 7. Configuration & Secrets Safety

- Never commit secrets (API keys, tokens, credentials) or real `.env` values into the repository.
- Keep `.env` local-only and out of version control.
- When introducing a new required environment variable:
  - Add it to `.env.template` with a safe placeholder.
  - Update corresponding setup documentation (README/README_zh if applicable).
- In logs, examples, and test artifacts, redact sensitive values and avoid printing raw secrets.
