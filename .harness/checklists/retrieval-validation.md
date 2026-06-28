# Retrieval Validation Checklist

Sources: `README_zh.md`, `examples/README_zh.md`

Use this before claiming retrieval quality, recall, or context expansion has
improved.

## Artifact Checks

- Confirm the expected input Markdown file.
- Confirm `*_index_new.jsonl` exists and contains embedding-facing text.
- Confirm `*_docstore_new.jsonl` exists and contains full content.
- Confirm `*_sparse.jsonl` exists when testing BM25 behavior.
- Check that sparse `owner_node_id` values map back to docstore/vector node IDs.

## Pipeline Checks

- Verify vector top-k results.
- Verify BM25 top-k results.
- Verify RRF output ordering and deduplication.
- Verify `RecursiveRetriever` maps index nodes to full nodes.
- Verify rerank behavior when rerank credentials are configured.
- Verify short-context expansion adds only structurally valid neighbors.
- Verify `AutoMergingRetriever` still handles parent-child aggregation.
- Verify sparse-missing fallback degrades to vector-only retrieval.

## Recommended Full Flow

```bash
uv run python -c "import asyncio; from examples.data_ingestion_pipeline import process_document; asyncio.run(process_document('test_resource/rag-anything.md'))"
uv run python examples/llama_index_advanced_rag.py
```

## Focused Tests

```bash
uv run pytest tests/test_lexical_search.py
uv run pytest tests/test_hybrid_retriever.py
uv run pytest tests/test_context_expanding_retriever.py
uv run pytest tests/test_retrieval_components.py
```

