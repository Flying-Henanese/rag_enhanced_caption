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
- Textual structural chunks such as Markdown bullet and ordered lists remain
  ordinary text nodes. Their `text_for_embedding` must fall back to their
  cleaned content and must never be persisted as an empty string.
- BM25 recall: retrieves literal matches from `*_sparse.jsonl` searchable
  objects. This is useful for table rows, headers, IDs, dates, emails, product
  names, and exact terms that semantic summaries may weaken.
- RRF fusion: combines vector and BM25 rankings by rank position. It does not
  compare raw vector scores with raw BM25 scores.
- Cleaning and relinking before indexing: `examples/data_ingestion_pipeline.py`
  runs `compress_and_relink_chunks` after semantic chunking. This removes
  cleaned-empty chunks, keeps cleaned content, repairs `parent_id` links through
  deleted chunks, and corrects pure image blocks. It does not physically merge
  short chunks with neighbors.
- Recursive retrieval: follows `IndexNode.index_id` from embedding-facing
  multimodal nodes to full content nodes.
- Rerank: optionally reorders candidate nodes before broader context expansion.
  Missing rerank credentials should not break retrieval. Remote reranking uses
  the asynchronous postprocessor path and `httpx.AsyncClient`.
- Short-context expansion: query-time expansion only. After rerank,
  `ShortContextExpandingRetriever` expands short eligible chunk hits by
  following same-section `PREVIOUS` and `NEXT` relationships. The example
  expands up to one previous and one next node, with `max_added_nodes=2` and
  `max_expansion_tokens=512`.
- AutoMerge: uses parent-child relationships to merge a sufficient set of child
  hits into broader parent context. The advanced example builds full-content
  copies of all parent nodes, including content parents with image/table
  descendants, before allowing them to replace children. Descendant content is
  collected in child order; embedding-facing leaf text stays unchanged. Parent
  copies are restored to the docstore after vector-index construction, which
  can otherwise overwrite them with embedding-facing leaf nodes.

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
- If expanded evidence disappears after AutoMerge, inspect the replacement
  parent's actual text and full-content metadata, including non-section
  intermediate parents. Structural ancestry alone does not prove content
  coverage.
- If final context repeats the same passage, check whether an ancestor and
  its descendant are both returned. Complete parent content does not by itself
  guarantee non-overlapping final results.
- If retrieval only works when `*_sparse.jsonl` exists, verify vector-only
  fallback behavior.

## LlamaIndex Mapping

Normal text chunks map directly to `TextNode` records: `text_for_embedding`
drives vector recall, while `full_content` stays available in metadata/docstore
for answer context.

Multimodal chunks such as images and tables use two nodes: an `IndexNode` with
`text_for_embedding` for vector recall, and a full `TextNode` containing
`full_content`. `IndexNode.index_id` points to the full node, and
`RecursiveRetriever` follows that link after recall.

When a hit returns only a summary, inspect the `IndexNode.index_id` mapping, the
docstore full node, and `RecursiveRetriever` wiring.

## Relationship Signals

Retrieval-time expansion depends on relationship metadata:

- `header_path` from docstore records is converted into section/path nodes.
- `parent_id` and LlamaIndex parent-child relationships support AutoMerge.
- previous/next relationships between content nodes support short-context
  expansion.
- sparse `owner_node_id` maps BM25 hits back to the same node universe used by
  vector retrieval.

When changing chunking or artifact schemas, verify these relationships still
exist before evaluating retrieval quality. Also verify that every vector index
record has non-empty `text_for_embedding` and a matching docstore ID.

## Fallback Behavior

If the sparse JSONL file does not exist or contains no usable searchable
objects, the advanced example should degrade to pure vector retrieval instead
of failing the whole retrieval flow.

The example executes the query pipeline with `aretrieve`; vector retrieval,
recursive resolution, reranking, short-context expansion, and AutoMerge all
provide an asynchronous path.

## Validation Dataset

For the 2026-10-30 Markdown/artifact checks, successful live VLM retry, and
remaining summary-faithfulness and context-overlap observations, see
[`../runs/2026-10-30-markdown-artifact-validation.md`](../runs/2026-10-30-markdown-artifact-validation.md).
This record distinguishes local structural checks from live service and
answer-quality evaluation; it is not a rerun of the historical 828D benchmark.

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
