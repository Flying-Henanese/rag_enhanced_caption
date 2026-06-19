# Lexical Search Design

## Goal

Add a BM25-based lexical retrieval route alongside the existing embedding route. The first implementation persists backend-neutral searchable objects as JSONL and builds an in-memory BM25 index at query startup. Storage and search interfaces remain replaceable by Elasticsearch or MongoDB-backed implementations.

## Data flow

1. The ingestion example enriches semantic chunks as it does today.
2. Deterministic extractors produce high-value searchable objects from table rows and fixed-format values such as dates and email addresses.
3. A JSONL repository writes `<document>_sparse.jsonl` beside the existing index and docstore files.
4. The advanced LlamaIndex example loads these objects into an in-memory BM25 backend.
5. A LlamaIndex adapter retrieves vector and lexical candidates, fuses their rankings with reciprocal-rank fusion, and passes the resulting nodes into the existing recursive, rerank, and AutoMerge stages.

## Boundaries

- `lexical_search` contains no LlamaIndex dependency.
- Searchable objects store canonical text, not backend-specific tokenization or postings.
- `owner_node_id` links every searchable object to the existing vector/docstore node ID.
- The JSONL repository is the current persistence implementation; Elasticsearch and MongoDB can implement the same repository/search protocols later.
- The current parent/child CLI JSONL schema is out of scope. The first implementation targets `examples/data_ingestion_pipeline.py` and its `_new.jsonl` outputs.

## Searchable object schema

Each object contains `id`, `owner_node_id`, `searchable_text`, `field_type`, `metadata`, and `schema_version`. IDs are deterministic so rebuilding the same document produces stable records.

## Error handling

- Empty or malformed searchable records are rejected when decoded.
- Missing sparse files in the advanced example disable lexical retrieval without breaking vector retrieval.
- Empty queries and empty corpora return no lexical results.

## Verification

Unit tests cover deterministic extraction, JSONL round-tripping, BM25 ranking, RRF fusion, and LlamaIndex node mapping. Existing tests verify that the original ingestion and retrieval paths remain intact.
