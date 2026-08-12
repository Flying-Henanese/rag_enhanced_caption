# 2026-06-28 Doc Alignment and Harness Validation

## Scope

Checked whether project documentation, current source layout, data-flow
documentation, and `.harness/` agree with the current repository behavior.

## Sources Inspected

- `README.md`
- `README_zh.md`
- `examples/README_zh.md`
- `AGENTS.md`
- `.harness/`
- `examples/data_ingestion_pipeline.py`
- `examples/llama_index_advanced_rag.py`
- `src/rag_enhanced_caption/`
- retrieval-related tests under `tests/`

## Drift Found

- `examples/README_zh.md` still described `data_ingestion_pipeline.py` as
  producing only two JSONL files. The current code writes
  `*_index_new.jsonl`, `*_docstore_new.jsonl`, and `*_sparse.jsonl`.
- `examples/README_zh.md` did not fully describe the BM25 sparse artifact,
  RRF, short-context expansion, or `SPARSE_PATH`.
- `README.md` and `README_zh.md` package layout omitted
  `src/rag_enhanced_caption/integrations/llama_index/retrievers.py`.

## Updates Made

- Updated `examples/README_zh.md` to describe the three example artifacts,
  sparse searchable objects, hybrid retrieval flow, and `SPARSE_PATH`.
- Updated `README.md` and `README_zh.md` package layout to include
  `integrations/llama_index/retrievers.py`.
- Kept `.harness/context/` aligned with the current source-backed flow:
  vector artifacts, docstore artifacts, sparse lexical artifacts, hybrid
  retrieval, short-context expansion, and AutoMerge.

## Harness-Guided Verification

The verification commands were selected from `.harness/checklists/` and focused
on retrieval/data-flow behavior.

### Documentation Term Check

```bash
rg -n "_sparse\\.jsonl|SPARSE_PATH|short-context|短上下文|retrievers\\.py|vector top-k \\+ BM25" README.md README_zh.md examples/README_zh.md .harness
```

Result: exit 0. The updated docs and harness all reference the sparse artifact,
short-context expansion, and `retrievers.py` where expected.

### Placeholder Check

```bash
LC_ALL=C rg -n "TO[D]O|TB[D]|FIX[M]E|fill[ ]in|implement[ ]later" .harness README.md README_zh.md examples/README_zh.md
```

Result: exit 1. No placeholder matches were found.

### Retrieval Tests

```bash
uv run pytest tests/test_retrieval_components.py tests/test_example_lexical_pipeline.py tests/test_hybrid_retriever.py tests/test_context_expanding_retriever.py
```

Result: 16 passed in 4.95s.

## Decision

The harness is usable as a development guide for retrieval-related changes:

- `context/artifact-contracts.md` describes the JSONL contracts.
- `context/retrieval-pipeline.md` describes the current retrieval flow.
- `checklists/retrieval-validation.md` points to tests that exercise sparse
  fallback, hybrid retrieval, short-context expansion, and advanced example
  wiring.
