# Retrieval Debugging Prompt

Sources: `README_zh.md`, `examples/README_zh.md`

Use this prompt pattern when asking an assistant to debug retrieval behavior in
this repository.

```text
You are debugging retrieval in rag-enhanced-caption.

Use the current repository source as ground truth. First identify which artifact
or retrieval stage is involved:

1. Markdown parsing and chunking
2. VLM enrichment and text_for_embedding
3. *_index.jsonl / *_index_new.jsonl
4. *_docstore.jsonl / *_docstore_new.jsonl
5. *_sparse.jsonl searchable objects
6. vector top-k
7. BM25 top-k
8. RRF fusion
9. RecursiveRetriever
10. reranker
11. short-context expansion
12. AutoMerge

Then show the concrete file path, function/class, artifact row, or test that
supports the diagnosis. Avoid generic RAG advice unless it maps to this exact
pipeline.
```

## Expected Evidence

- Relevant source files under `src/rag_enhanced_caption/`.
- Relevant orchestration files under `examples/`.
- Artifact rows from `output/` or `test_resource/`.
- Focused tests under `tests/`.

