# Code Change Checklist

Sources: `AGENTS.md`, `README_zh.md`

Use this as the execution checklist for source changes. `AGENTS.md` remains the
authoritative rule source; this file focuses on when to read context, what to
verify, and what to report before finishing.

## Before Editing

- Confirm the requested behavior and scope.
- Check current `git status --short`.
- Read the nearest source file, test, and documentation section before changing
  code.
- If dependencies are needed, use `uv add` or optional dependency groups. Do not
  manually edit dependency lists or `uv.lock`.

## Context Triggers

- If the change touches project structure, module ownership, package layout, or
  dependency placement, read `.harness/context/project-overview.md`.
- If the change touches CLI output, docstore schema, sparse search schema,
  `text_for_embedding`, JSONL fields, or output filenames, read
  `.harness/context/artifact-contracts.md`.
- If the change touches BM25, hybrid retrieval, RRF, rerank, context expansion,
  AutoMerge, or LlamaIndex example wiring, read
  `.harness/context/retrieval-pipeline.md`.
- If the change updates coding conventions, dependency rules, async behavior,
  logging, typing, or docstring expectations, update
  `.harness/context/agent-guidelines.md` together with `AGENTS.md`.

## During Editing

- Keep changes localized and tied to the requested behavior.
- Preserve project constraints from `AGENTS.md`: async VLM/pipelines, `loguru`,
  explicit type hints, Google-style docstrings, and dependency hygiene.
- Update docstrings and tests when changing public signatures or behavior.
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

## Completion

- Run `git diff --check`.
- Review the final diff for unrelated edits.
- Report the tests/checks run and any checks that were skipped.
