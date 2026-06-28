# Agent Guidelines

Source: `AGENT.md`

This file distills project instructions for coding agents. Treat `AGENT.md` as
the authoritative source when details conflict.

## Development Environment

- Use `uv` for all dependency management and environment orchestration.
- Run Python commands through `uv run`.
- Use `ruff` for linting and formatting.
- Keep demo and integration dependencies out of core dependencies. Use optional
  groups for packages such as `llama-index-core`.

## Code Style

- Package name is `rag_enhanced_caption`.
- Use `loguru` for logging.
- Do not introduce standard-library `logging` in project code.
- Use explicit type hints for functions and methods.
- Prefer Python 3.10+ built-in generics: `list[str]`, `dict[str, Any]`.
- Prefer `str | None` over `Optional[str]`.
- Public classes and methods should use Google-style docstrings.

## Async and API Behavior

- VLM calls and processing pipelines should remain async.
- Account for network exceptions and timeouts when handling LLM/VLM API calls.
- Log API call errors with `logger.exception` when a traceback is useful.
- Use `robust_json_parse` for VLM output parsing.

## Scope Control

- Do not perform unrelated refactors or broad cleanups.
- Keep edits surgical and tied to the user request.
- When changing a function or method signature, update the matching docstring in
  the same change.
- Ensure injected Markdown blocks, especially `<details>`, are correctly closed
  and do not break rendering.

