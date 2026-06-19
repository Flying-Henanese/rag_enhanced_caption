# Short Context Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add post-rerank expansion of short paragraph hits through explicit document-order relationships while retaining LlamaIndex parent/child AutoMerge.

**Architecture:** A reusable optional-integration retriever wraps the reranked retriever, follows `PREVIOUS` and `NEXT` relationships from eligible short text nodes, applies section and budget constraints, and passes the expanded set to `AutoMergingRetriever`. The advanced example constructs the relationships on actual content nodes in JSONL order; deterministic unit tests cover the behavior without network access before an optional `.env`-backed integration run.

**Tech Stack:** Python 3.10+, LlamaIndex Core, pytest, uv, ruff

---

## File Map

- Create `src/rag_enhanced_caption/integrations/__init__.py`: optional integration namespace.
- Create `src/rag_enhanced_caption/integrations/llama_index/__init__.py`: exports the retriever.
- Create `src/rag_enhanced_caption/integrations/llama_index/retrievers.py`: short-context expansion implementation.
- Create `tests/test_context_expanding_retriever.py`: deterministic unit and AutoMerge composition coverage.
- Modify `examples/llama_index_advanced_rag.py`: content ordering, relationships, and pipeline wiring.
- Modify `tests/test_retrieval_components.py`: verify example component construction and optional live retrieval.

### Task 1: Define short-anchor expansion behavior

**Files:**
- Create: `tests/test_context_expanding_retriever.py`
- Create: `src/rag_enhanced_caption/integrations/__init__.py`
- Create: `src/rag_enhanced_caption/integrations/llama_index/__init__.py`
- Create: `src/rag_enhanced_caption/integrations/llama_index/retrievers.py`

- [ ] **Step 1: Write failing tests for short and long anchors**

Create three linked `TextNode` instances in a `SimpleDocumentStore`, mark them with `metadata={"type": "chunk", "section_path": ["A"]}`, and use a static `BaseRetriever`. Assert that a short middle anchor adds both neighbors with decayed scores, while an anchor at or above the threshold remains unchanged.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest tests/test_context_expanding_retriever.py -k "short_anchor or long_anchor" -v`

Expected: collection fails because `rag_enhanced_caption.integrations.llama_index.retrievers` does not exist.

- [ ] **Step 3: Implement the minimal wrapper**

Implement `ShortContextExpandingRetriever(BaseRetriever)` with constructor arguments:

```python
base_retriever: BaseRetriever
docstore: BaseDocumentStore
token_count_fn: Callable[[str], int] | None = None
short_node_token_threshold: int = 100
previous_nodes: int = 1
next_nodes: int = 1
max_added_nodes: int = 2
max_expansion_tokens: int = 512
score_decay: float = 0.85
same_section: bool = True
section_metadata_key: str = "section_path"
eligible_node_types: set[str] | None = None
```

Use LlamaIndex `get_tokenizer()` when no counter is injected. `_retrieve()` must preserve anchors, expand only metadata type `chunk`, follow docstore relationships, deduplicate by node ID, and apply `anchor_score * score_decay**hop` to added nodes.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `uv run pytest tests/test_context_expanding_retriever.py -k "short_anchor or long_anchor" -v`

Expected: both tests pass.

### Task 2: Enforce traversal safety and budgets

**Files:**
- Modify: `tests/test_context_expanding_retriever.py`
- Modify: `src/rag_enhanced_caption/integrations/llama_index/retrievers.py`

- [ ] **Step 1: Write failing boundary tests**

Add tests proving that expansion does not cross a different `section_path`, overlapping anchors are deduplicated, a missing docstore target is ignored, cycles terminate, and `max_added_nodes` plus `max_expansion_tokens` cap expansion.

- [ ] **Step 2: Run boundary tests and verify RED**

Run: `uv run pytest tests/test_context_expanding_retriever.py -k "section or deduplicate or missing or cycle or budget" -v`

Expected: at least one assertion fails because the minimal traversal does not yet implement all guards.

- [ ] **Step 3: Add minimal guards**

Track visited node IDs per traversal direction, compare exact section metadata, catch missing-document lookup errors, stop at configured hop counts, and apply global added-node and expansion-token budgets without deleting original anchors.

- [ ] **Step 4: Run all retriever unit tests and verify GREEN**

Run: `uv run pytest tests/test_context_expanding_retriever.py -v`

Expected: all tests pass without API credentials.

### Task 3: Compose expansion with AutoMerge

**Files:**
- Modify: `tests/test_context_expanding_retriever.py`
- Modify: `src/rag_enhanced_caption/integrations/llama_index/retrievers.py`

- [ ] **Step 1: Write a failing composition test**

Build a parent with two child nodes, make one short child the reranked anchor, link the other as its `NEXT` neighbor, wrap the static retriever with `ShortContextExpandingRetriever`, then wrap that with `AutoMergingRetriever(simple_ratio_thresh=0.5)`. Assert that the final result contains the parent and not the children.

- [ ] **Step 2: Run the composition test and verify RED**

Run: `uv run pytest tests/test_context_expanding_retriever.py -k automerge -v`

Expected: fail until the expanded nodes preserve relationships and scores in the shape AutoMerge consumes.

- [ ] **Step 3: Correct only the interoperability defect**

Ensure added `NodeWithScore` instances contain the original docstore nodes unchanged and numeric scores, allowing `AutoMergingRetriever` to read their parent relationships.

- [ ] **Step 4: Run the composition test and verify GREEN**

Run: `uv run pytest tests/test_context_expanding_retriever.py -k automerge -v`

Expected: the parent replaces both children.

### Task 4: Build document-order relationships in the advanced example

**Files:**
- Modify: `tests/test_retrieval_components.py`
- Modify: `examples/llama_index_advanced_rag.py`

- [ ] **Step 1: Write a failing relationship-construction test**

Extract a helper named `_link_content_nodes_in_order(nodes)` and test it with text, table, and cross-section nodes. Assert reciprocal `NEXT`/`PREVIOUS` links inside one section and no link across sections.

- [ ] **Step 2: Run the helper test and verify RED**

Run: `uv run pytest tests/test_retrieval_components.py -k link_content_nodes -v`

Expected: fail because the helper does not exist.

- [ ] **Step 3: Implement relationship construction**

While reading docstore records, append each actual text node or resolved element node to a content-order list and store its `section_path` metadata. After node construction, call `_link_content_nodes_in_order()` to add reciprocal relationships only when consecutive nodes share the exact section path. Do not link path aggregation nodes or `IndexNode` pointers.

- [ ] **Step 4: Run the helper test and verify GREEN**

Run: `uv run pytest tests/test_retrieval_components.py -k link_content_nodes -v`

Expected: pass without network access.

### Task 5: Wire the post-rerank expansion pipeline

**Files:**
- Modify: `tests/test_retrieval_components.py`
- Modify: `examples/llama_index_advanced_rag.py`

- [ ] **Step 1: Write a failing construction test**

Patch the example's vector index with deterministic nodes and assert that the retriever passed into `AutoMergingRetriever` is a `ShortContextExpandingRetriever` whose wrapped retriever is `RerankedRetriever`.

- [ ] **Step 2: Run the construction test and verify RED**

Run: `uv run pytest tests/test_retrieval_components.py -k context_expansion_pipeline -v`

Expected: fail because AutoMerge still wraps `RerankedRetriever` directly.

- [ ] **Step 3: Wire the new retriever**

Import `ShortContextExpandingRetriever`, wrap `reranked_retriever` with the design defaults, and pass the wrapper into `AutoMergingRetriever`. Keep the public return tuple compatible with the existing example and tests.

- [ ] **Step 4: Run retrieval component tests and verify GREEN**

Run: `uv run pytest tests/test_retrieval_components.py -v`

Expected: offline construction tests pass; the existing live smoke test either passes with configured credentials or skips when credentials/resources are absent.

### Task 6: Full verification and optional live flow

**Files:**
- Verify only; no source changes expected.

- [ ] **Step 1: Run focused tests**

Run: `uv run pytest tests/test_context_expanding_retriever.py tests/test_retrieval_components.py -v`

Expected: all offline tests pass; credential-gated live tests pass or report an explicit skip.

- [ ] **Step 2: Run relevant regression suite**

Run: `uv run pytest tests/test_rerank_components.py tests/test_cli_process_document.py tests/test_semantic_parent_binding.py -v`

Expected: all available tests pass, with only documented credential skips.

- [ ] **Step 3: Run lint and formatting checks**

Run: `uv run ruff check src/rag_enhanced_caption/integrations examples/llama_index_advanced_rag.py tests/test_context_expanding_retriever.py tests/test_retrieval_components.py`

Run: `uv run ruff format --check src/rag_enhanced_caption/integrations examples/llama_index_advanced_rag.py tests/test_context_expanding_retriever.py tests/test_retrieval_components.py`

Expected: both commands exit zero.

- [ ] **Step 4: Run the `.env`-backed live retrieval smoke test**

Run: `uv run pytest tests/test_retrieval_components.py::test_advanced_retrieval_pipeline_real_components -v -s`

Expected: pass when `.env` contains working embedding credentials and the configured JSONL resources exist; otherwise report the exact external blocker without exposing secrets.

- [ ] **Step 5: Review the final diff**

Run: `git diff --check` and `git status --short`

Expected: no whitespace errors; only scoped implementation files plus the user's pre-existing changes are present.
