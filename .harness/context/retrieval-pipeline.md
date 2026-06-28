# Retrieval Pipeline

Sources: `README_zh.md`, `examples/README_zh.md`

The advanced retrieval examples combine semantic vector recall with lexical
BM25 recall. Both paths map back to the same LlamaIndex node IDs so the
docstore remains the source of full content.

## When To Read This File

Read this file before changing or answering questions about:

- BM25 or sparse searchable-object recall
- vector top-k retrieval
- reciprocal rank fusion (RRF)
- rerank placement or rerank fallback behavior
- `RecursiveRetriever`
- short-context expansion
- `AutoMergingRetriever`
- LlamaIndex node conversion
- retrieval quality, missing nearby content, or recall debugging

## High-Level Flow

```text
Embedding top-k + BM25 top-k
               |
        Reciprocal Rank Fusion
               |
RecursiveRetriever -> reranker -> short-context expansion -> AutoMerge
```

## Recall Mechanisms

The current example retrieval stack has several distinct recall and expansion
mechanisms. Keep their boundaries clear:

- Vector recall: retrieves semantically similar `TextNode` or `IndexNode`
  candidates from `VectorStoreIndex` using `text_for_embedding`.
- BM25 recall: retrieves literal matches from `*_sparse.jsonl` searchable
  objects. This is useful for table rows, headers, IDs, dates, emails, product
  names, and exact terms that semantic summaries may weaken.
- RRF fusion: combines vector and BM25 rankings by rank position. It does not
  compare raw vector scores with raw BM25 scores.
- Recursive retrieval: follows `IndexNode.index_id` from embedding-facing
  multimodal nodes to full content nodes.
- Rerank: optionally reorders candidate nodes before broader context expansion.
  Missing rerank credentials should not break retrieval.
- Short-context expansion: adds nearby short nodes after rerank when structural
  relationships and section constraints allow it.
- AutoMerge: uses parent-child relationships to merge a sufficient set of child
  hits into broader parent context.

The safest mental model is:

```text
vector index = semantic recall view
sparse index = lexical recall view
docstore = authoritative full-content view
retrievers = mapping, ranking, and expansion logic
```

## Component Boundaries

- `src/rag_enhanced_caption/lexical_search/`: core lexical-search layer,
  including schema, extractors, builder, repository, BM25 backend, and fusion.
- `src/rag_enhanced_caption/integrations/llama_index/hybrid_retriever.py`:
  LlamaIndex adapter for hybrid retrieval.
- `src/rag_enhanced_caption/integrations/llama_index/retrievers.py`:
  retrieval helpers such as short-context expansion.
- `examples/data_ingestion_pipeline.py`: example ingestion orchestration that
  builds vector/docstore artifacts and sparse searchable objects.
- `examples/llama_index_advanced_rag.py`: example retrieval orchestration that
  reads artifacts, builds LlamaIndex nodes, runs retrieval, and prints context.

## Query-Time Decision Points

- If exact terms, identifiers, table content, or dates are missing, inspect
  sparse searchable objects and BM25 results first.
- If semantically related prose is missing, inspect `text_for_embedding`, vector
  top-k, and embedding configuration first.
- If a multimodal hit returns only a summary instead of full content, inspect
  `IndexNode` to full node mapping and `RecursiveRetriever`.
- If nearby short content is missing after the correct anchor is retrieved,
  inspect `ShortContextExpandingRetriever` relationships, section matching, and
  expansion limits.
- If the result is too narrow despite several child hits under the same parent,
  inspect AutoMerge parent-child relationships and merge threshold.
- If retrieval only works when `*_sparse.jsonl` exists, verify vector-only
  fallback behavior.

## LlamaIndex Mapping

Normal text chunks become `TextNode` instances:

```python
TextNode(
    id_=chunk_id,
    text=text_for_embedding,
    metadata={"type": "chunk", "full_content": full_content},
)
```

Images, tables, and other multimodal elements use an index node plus a full
content node:

```python
element_node = TextNode(
    id_=f"{chunk_id}_full",
    text=full_content,
    metadata={"type": "element"},
)
```

```python
index_node = IndexNode(
    id_=chunk_id,
    text=text_for_embedding,
    index_id=element_node.id_,
)
```

The index node participates in vector retrieval. `RecursiveRetriever` can then
jump to the complete content node.

## Relationship Signals

Retrieval-time expansion depends on relationship metadata:

- `header_path` from docstore records is converted into section/path nodes.
- `parent_id` and LlamaIndex parent-child relationships support AutoMerge.
- previous/next relationships between content nodes support short-context
  expansion.
- sparse `owner_node_id` maps BM25 hits back to the same node universe used by
  vector retrieval.

When changing chunking or artifact schemas, verify these relationships still
exist before evaluating retrieval quality.

## Fallback Behavior

If the sparse JSONL file does not exist or contains no usable searchable
objects, the advanced example should degrade to pure vector retrieval instead
of failing the whole retrieval flow.

## Validation Dataset

Use `test_resource/rag-anything.md` when validating the full hybrid retrieval
flow because the advanced example expects these artifacts:

```text
output/rag-anything_index_new.jsonl
output/rag-anything_docstore_new.jsonl
output/rag-anything_sparse.jsonl
```

Recommended commands:

```bash
uv run python -c "import asyncio; from examples.data_ingestion_pipeline import process_document; asyncio.run(process_document('test_resource/rag-anything.md'))"
uv run python examples/llama_index_advanced_rag.py
```
