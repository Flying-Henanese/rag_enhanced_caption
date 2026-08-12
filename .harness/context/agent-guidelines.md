# Agent Guidelines

Source: `AGENTS.md`

This file distills project instructions for coding agents. Treat `AGENTS.md` as
the authoritative source when details conflict.

## Development Environment

- Use `uv` for all dependency management and environment orchestration.
- Run Python commands through `uv run`.
- Use `ruff` for linting and formatting.
- Ruff enforces explicit function annotations and modern typing syntax for the
  core package. Examples and tests remain gradual-adoption areas for these
  typing rules. The standard-library `logging` import ban applies project-wide.
- Keep demo and integration dependencies out of core dependencies. Use optional
  groups for packages such as `llama-index-core`.

## Code Style

- Package name is `rag_enhanced_caption`.
- Use `loguru` for logging.
- Do not introduce standard-library `logging` in project code.
- Use explicit type hints for functions and methods.
- Prefer Python 3.10+ built-in generics: `list[str]`, `dict[str, Any]`.
- Prefer `str | None` over `Optional[str]`.
- Prefer small helper functions over complex inline expressions or deeply nested
  comprehension logic.
- Use `dataclass`, `TypedDict`, or dedicated result objects for stable structured
  data; avoid requiring callers to unpack more than three return values.
- Avoid mutable default arguments. Use `None` plus explicit initialization and
  docstring notes for dynamic defaults.
- Avoid deeply nested dictionaries/lists for core data contracts; promote stable
  shapes into typed objects or documented schemas.
- Public classes and methods should use Google-style docstrings.

## Async and API Behavior

- VLM calls and processing pipelines should remain async.
- Account for network exceptions and timeouts when handling LLM/VLM API calls.
- Keep `try` blocks narrow and prefer explicit exceptions over `None` for failure
  paths unless the optional result is documented.
- Log API call errors with `logger.exception` when a traceback is useful.
- Use `robust_json_parse` for VLM output parsing.

## Performance

- Profile before optimizing.
- Avoid caching, concurrency, batching, or complex algorithms without measured
  bottlenecks or clear scale requirements.
- Prefer streaming or iterator-based processing for large Markdown, JSONL,
  docstore, and retrieval-candidate flows when practical.

## Scope Control

- Do not perform unrelated refactors or broad cleanups.
- Keep edits surgical and tied to the user request.
- When changing a function or method signature, update the matching docstring in
  the same change.
- Ensure injected Markdown blocks, especially `<details>`, are correctly closed
  and do not break rendering.
