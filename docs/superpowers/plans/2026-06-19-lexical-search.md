# Lexical Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add JSONL-persisted searchable objects and an in-memory BM25 route fused with the existing embedding retriever.

**Architecture:** Deterministic extractors build backend-neutral searchable objects. A repository persists them, an in-memory BM25 backend ranks owner node IDs, and a thin optional LlamaIndex adapter fuses lexical and vector rankings before recursive retrieval and AutoMerge.

**Tech Stack:** Python 3.10+, dataclasses, markdown-it/BeautifulSoup helpers, LlamaIndex optional integration, pytest.

---

### Task 1: Searchable object extraction

**Files:**
- Create: `src/rag_enhanced_caption/lexical_search/schema.py`
- Create: `src/rag_enhanced_caption/lexical_search/extractors.py`
- Create: `src/rag_enhanced_caption/lexical_search/builder.py`
- Test: `tests/test_searchable_object_builder.py`

- [ ] Write failing tests for table rows, dates, emails, stable IDs, and duplicate removal.
- [ ] Run the focused test and confirm it fails because the package does not exist.
- [ ] Implement the smallest extractor and builder API that satisfies the tests.
- [ ] Run the focused test and confirm it passes.

### Task 2: Persistence and BM25 ranking

**Files:**
- Create: `src/rag_enhanced_caption/lexical_search/repository.py`
- Create: `src/rag_enhanced_caption/lexical_search/bm25.py`
- Create: `src/rag_enhanced_caption/lexical_search/fusion.py`
- Test: `tests/test_lexical_search.py`

- [ ] Write failing tests for JSONL round-tripping, BM25 exact-term ranking, and RRF fusion.
- [ ] Run the focused test and confirm the intended missing APIs fail.
- [ ] Implement repository protocols, JSONL persistence, in-memory BM25, and RRF.
- [ ] Run the focused test and confirm it passes.

### Task 3: LlamaIndex hybrid adapter

**Files:**
- Create: `src/rag_enhanced_caption/integrations/llama_index/hybrid_retriever.py`
- Test: `tests/test_hybrid_retriever.py`

- [ ] Write a failing test that maps BM25 owner IDs to the same leaf nodes used by vector retrieval.
- [ ] Confirm the adapter import or behavior fails.
- [ ] Implement vector plus lexical RRF retrieval as a `BaseRetriever`.
- [ ] Confirm the focused test passes.

### Task 4: Example pipeline integration

**Files:**
- Modify: `examples/data_ingestion_pipeline.py`
- Modify: `examples/llama_index_advanced_rag.py`
- Modify: `tests/test_retrieval_components.py`

- [ ] Write failing integration tests for sparse JSONL output and hybrid component construction.
- [ ] Confirm the tests fail on missing output/wiring.
- [ ] Write `_sparse.jsonl` during ingestion and wire hybrid retrieval before recursive retrieval.
- [ ] Run focused and complete tests, then run Ruff checks.
