# Retrieval Validation Checklist

Sources: `README_zh.md`, `examples/README_zh.md`

Use this before claiming retrieval quality, recall, or context expansion has
improved.

## Artifact Checks

- Confirm the expected input Markdown file.
- Confirm `*_index_new.jsonl` exists and every row contains non-empty
  embedding-facing text, including bullet and ordered list chunks.
- Confirm `*_docstore_new.jsonl` exists and contains full content.
- Confirm index and docstore IDs have a one-to-one mapping.
- Confirm `*_sparse.jsonl` exists when testing BM25 behavior.
- Check that sparse `owner_node_id` values map back to docstore/vector node IDs.
- Resolve stored parent IDs to actual records; advanced-example short IDs and
  base-CLI fully qualified IDs use different namespaces.
- Check VLM logs and summary text separately from file existence. Record
  authentication errors, raw-table fallback, and image placeholders explicitly.
- Compare generated summaries with source `full_content` for unsupported claims;
  successful authentication and JSON parsing are not faithfulness checks.

## Pipeline Checks

- Verify vector top-k results.
- Verify BM25 top-k results.
- Verify RRF output ordering and deduplication.
- Verify `RecursiveRetriever` maps index nodes to full nodes.
- Verify rerank behavior when rerank credentials are configured.
- Verify short-context expansion adds only structurally valid neighbors.
- Verify `AutoMergingRetriever` still handles parent-child aggregation.
- Verify every replacing parent contains its own original content and all
  descendant evidence, even when merging stops at an intermediate text parent.
- Check final context for overlapping ancestor/descendant results and repeated
  passages, separately from checking for evidence loss.
- Verify sparse-missing fallback degrades to vector-only retrieval.

## Evidence and Claim Boundaries

- Separate persisted JSONL content from runtime full-parent context snapshots.
- Record whether VLM, embeddings, and reranking are live, disabled, or mocked.
- Attribute improvements to the first stage where the relevant evidence changes.
  Do not credit short-context expansion for evidence already present before it.
- Check exact alarm identifiers as well as answer fields; matching a field value
  from a similar alarm does not establish a correct source match.
- Report structural validation separately from semantic recall and answer
  accuracy. Rerun the actual benchmark before updating accuracy claims.

## Recommended Full Flow

```bash
uv run python -c "import asyncio; from examples.data_ingestion_pipeline import process_document; asyncio.run(process_document('test_resource/rag-anything.md'))"
uv run python examples/llama_index_advanced_rag.py
```

## Focused Tests

```bash
uv run pytest tests/test_lexical_search.py
uv run pytest tests/test_example_lexical_pipeline.py
uv run pytest tests/test_hybrid_retriever.py
uv run pytest tests/test_context_expanding_retriever.py
uv run pytest tests/test_retrieval_components.py
uv run pytest tests/test_rerank_components.py
```
