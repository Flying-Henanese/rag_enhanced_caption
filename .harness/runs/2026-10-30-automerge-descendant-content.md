# AutoMerge descendant-content preservation

## Problem and reproduction

The supplied historical QA report shows Q1 losing its four-person audience
list after short-context expansion and AutoMerge. Current code already treats
bullet/ordered lists as ordinary text, but the same loss still reproduced when
full image/table content was attached below a text chunk.

The deterministic test uses the real example tree builder, recursive retriever,
short-context expander, and asynchronous AutoMerge. Only vector candidate
selection is replaced to avoid an external embedding API. The fake index also
writes embedding-facing leaves into the docstore, matching the overwrite
boundary that the fix must handle.

Before the fix:

```text
uv run pytest tests/test_retrieval_components.py -k preserves_expanded_descendant_content -q --tb=short
4 failed, 2 passed
```

Both table/image cases lost the audience list when merged to a section. When
other section siblings prevented section merging, the intermediate text parent
returned only its embedding summary, losing both the original lead and list.

## Change

`_build_parent_context_nodes` walks the actual content tree, collecting each
parent's original content and all descendants in child order. It creates full
context copies of all parents, including non-section text parents. These copies
are installed in the docstore after vector-index construction; input leaves
retain their embedding-facing text. Persisted JSONL schemas are unchanged.

## Validation

The regression covers bullet lists, ordered lists, tables, images, nested
sections, and merging that stops at an intermediate content parent. It checks
that all four audience categories survive exactly once and embedding-facing
text remains unchanged.

```text
uv run pytest tests/test_retrieval_components.py -k preserves_expanded_descendant_content -q --tb=short
6 passed

uv run pytest tests/test_retrieval_components.py tests/test_context_expanding_retriever.py tests/test_hybrid_retriever.py tests/test_rerank_components.py tests/test_processor.py -k 'not real_components and not real_api and not vlm_table_summarization' -q --tb=short
26 passed, 3 deselected

uv run ruff check .
All checks passed!

uv run ruff format --check .
55 files already formatted

git diff --check
(no errors)
```

The initial five-file pytest run without deselection had 26 passes and 3 failures
in live-service tests. Embedding and VLM calls returned HTTP 401 with an invalid
token; the real rerank test did not return its expected ranking. These tests
were not rerun after configuration correction. A subsequent real CLI run on the
small Markdown sample successfully generated a VLM table summary; the earlier
401 therefore no longer describes that sample's latest VLM result. See
[`2026-10-30-markdown-artifact-validation.md`](2026-10-30-markdown-artifact-validation.md)
for artifacts, validation modes, the successful retry, and outstanding summary
faithfulness/context overlap observations. The historical 828D 10-question
benchmark was not rerun, and no new answer-accuracy claim is made.
