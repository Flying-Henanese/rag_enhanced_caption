# Artifact Contracts

Sources: `README.md`, `README_zh.md`, `examples/README_zh.md`

This project uses JSONL artifacts to separate embedding-facing text from full
retrieval-time content.

## When To Read This File

Read this file before changing any code or documentation that touches:

- output file names or output directories
- `*_index.jsonl` or `*_index_new.jsonl`
- `*_docstore.jsonl` or `*_docstore_new.jsonl`
- `*_sparse.jsonl`
- `text_for_embedding`
- `full_content`, `parent_id`, `header_path`, `element_type`, or `entities`
- `owner_node_id`, `searchable_text`, `field_type`, or sparse schema versions
- mappings between vector hits, BM25 hits, and docstore nodes

## CLI Outputs

For an input Markdown file named `<name>.md`, the CLI produces:

```text
output/
+-- <name>_enhanced.md
+-- <name>_index.jsonl
+-- <name>_docstore.jsonl
```

- `<name>_enhanced.md`: chunk-by-chunk Markdown preview separated by horizontal
  rules. VLM summaries remain in `text_for_embedding` and are not injected into
  this preview.
- `<name>_index.jsonl`: compact records intended for embedding and vector
  indexing.
- `<name>_docstore.jsonl`: full parent-child records intended for retrieval-time
  expansion.

Writers and readers:

- Written by `src/rag_enhanced_caption/cli.py`.
- Covered by `tests/test_cli_process_document.py`.
- Intended as the stable package-level output contract.

## Advanced Example Outputs

`examples/data_ingestion_pipeline.py` writes example-oriented artifacts:

```text
output/
+-- <name>_index_new.jsonl
+-- <name>_docstore_new.jsonl
+-- <name>_sparse.jsonl
```

`*_sparse.jsonl` is the lexical-search artifact used by the BM25 path. It is
additive and should not change the base CLI output contract unless a design
explicitly says so.

Writers and readers:

- Written by `examples/data_ingestion_pipeline.py`.
- Read by `examples/llama_index_advanced_rag.py`.
- Persisted through `JsonlSearchableObjectRepository`.
- Covered by `tests/test_example_lexical_pipeline.py`,
  `tests/test_lexical_search.py`, and `tests/test_retrieval_components.py`.

## Base CLI Index JSONL

`*_index.jsonl` contains compact embedding-facing records. Text records use a
`p_` ID prefix and multimodal records use a `c_` prefix:

```json
{"id": "p_chunk_id", "parent_id": null, "text_for_embedding": "embedding text", "metadata": {"type": "parent", "source": "document.md", "element_type": "text"}}
```

## Advanced Example Index JSONL

`*_index_new.jsonl` uses the semantic chunk IDs directly and has a smaller
metadata shape:

```json
{"id": "chunk_id", "text_for_embedding": "embedding text", "metadata": {"element_type": "text"}}
```

The text should be clean semantic content. Noisy raw Markdown, large tables, and
image references should be avoided when they would degrade embedding quality.

Index invariants:

- `id` must remain stable enough to map back to docstore records.
- `text_for_embedding` is the retrieval-facing summary or text, not necessarily
  the full original Markdown.
- Metadata should contain enough type information for downstream conversion,
  but full content belongs in docstore artifacts.

## Base CLI Docstore JSONL

`*_docstore.jsonl` preserves full content, embedding text, and structure:

```json
{
  "id": "p_chunk_id",
  "type": "parent",
  "parent_id": null,
  "text_for_embedding": "semantic summary",
  "full_content": "complete Markdown content",
  "header_path": ["Heading", "Subheading"],
  "element_type": "text",
  "entities": []
}
```

## Advanced Example Docstore JSONL

`*_docstore_new.jsonl` keeps embedding text exclusively in the matching
`*_index_new.jsonl` record. Its docstore row contains the cleaned content and
structure used by the example retrieval pipeline:

```json
{
  "id": "chunk_id",
  "full_content": "cleaned Markdown content",
  "parent_id": null,
  "header_path": ["Heading", "Subheading"],
  "element_type": "text",
  "entities": []
}
```

The docstore is authoritative for full content, parent-child structure, and
retrieval-time expansion.

Docstore invariants:

- Every retrieval-facing index record should have a corresponding full-content
  docstore record when expansion is expected.
- `full_content` should preserve the content needed after recall, including
  richer Markdown that may be inappropriate for embedding.
- `parent_id` and `header_path` are structural signals used by downstream
  retrievers and AutoMerge-style aggregation.
- If a chunk is enriched by a VLM, the embedding-facing summary and full content
  must not be confused: `text_for_embedding` is for recall, `full_content` is
  for answer context and inspection. In the advanced example these fields live
  in separate index and docstore rows; the base CLI also repeats
  `text_for_embedding` in its docstore row.

## Sparse JSONL

`*_sparse.jsonl` contains backend-independent searchable objects:

```json
{
  "id": "stable-field-id",
  "owner_node_id": "rag-anything_chunk_89",
  "searchable_text": "Project: MiniRAG; Description: ...",
  "field_type": "table_row",
  "metadata": {"row_index": 2},
  "schema_version": 1
}
```

Important constraints:

- `owner_node_id` maps lexical hits back to the same node IDs used by vector
  retrieval and docstore expansion.
- Searchable-object extraction is deterministic and should not depend on LLM
  output.
- Current extractors cover Markdown/HTML table headers and rows, plus fixed
  formats such as dates and emails.
- JSONL persistence is decoupled from BM25 and can later map to Elasticsearch,
  MongoDB, or another backend.

Sparse invariants:

- `owner_node_id` must refer to a node ID that the retrieval example can map
  back to a LlamaIndex node and docstore content.
- `searchable_text` should contain literal, keyword-relevant text such as table
  headers, table rows, dates, email addresses, product names, and identifiers.
- Searchable objects are a parallel recall view over the same content, not a
  replacement for vector index records or docstore records.
- VLM summaries may improve vector recall, but sparse objects should remain
  deterministic and inspectable.

## Data-Flow Summary

```text
Markdown input
  -> semantic chunks
  -> optional VLM enrichment
  -> index JSONL for embedding/vector recall
  -> docstore JSONL for full-content expansion
  -> sparse JSONL for lexical/BM25 recall in examples
```

The index and sparse artifacts are recall views. The docstore artifact is the
full-content source used after recall.
