# Code Change Checklist

Sources: `AGENT.md`, `README_zh.md`

Use this before and after source changes.

## Before Editing

- Confirm the requested behavior and scope.
- Check current `git status --short`.
- Read the nearest source file, test, and documentation section before changing
  code.
- Identify whether the change affects CLI output, docstore schema, sparse
  search schema, retrieval behavior, or prompt behavior.
- If dependencies are needed, use `uv add` or optional dependency groups. Do not
  manually edit dependency lists or `uv.lock`.

## During Editing

- Keep changes localized.
- Preserve async behavior in VLM and processing pipelines.
- Use `loguru`, not standard `logging`.
- Add or update explicit type hints.
- Update Google-style docstrings when changing public signatures.
- Keep Markdown injection balanced and renderable.

## Verification

Run the narrowest meaningful tests first, then expand when risk is higher.

```bash
uv run pytest tests/test_semantic_parser_fixes.py
uv run pytest tests/test_html_table.py
uv run pytest tests/test_processor.py
uv run pytest tests/test_lexical_search.py
uv run pytest tests/test_hybrid_retriever.py
uv run pytest tests/test_context_expanding_retriever.py
```

For broad changes:

```bash
uv run pytest
uv run ruff check .
```

