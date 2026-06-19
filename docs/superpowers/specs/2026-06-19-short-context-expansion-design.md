# Short Context Expansion Design

## Goal

Extend short, highly relevant paragraph hits with their immediate document-order
neighbors after reranking, while preserving the existing LlamaIndex parent/child
AutoMerge behavior.

## Scope

This change is limited to the optional LlamaIndex integration and its advanced
example. It does not change the CLI persistence schema, core chunking APIs, or
required project dependencies.

## Retrieval Pipeline

The advanced pipeline will run in this order:

1. Vector retrieval and recursive resolution produce candidate content nodes.
2. Reranking selects relevant anchor nodes.
3. Short-context expansion follows explicit `PREVIOUS` and `NEXT` relationships.
4. `AutoMergingRetriever` performs its existing parent/child promotion.
5. Results remain score-sorted for downstream context budgeting.

Expansion runs after reranking so neighboring context is not independently
penalized for having low query similarity. It runs before AutoMerge so expanded
siblings can participate in structural aggregation.

## Components

### Optional LlamaIndex integration

Add `src/rag_enhanced_caption/integrations/llama_index/retrievers.py` containing
`ShortContextExpandingRetriever`, a `BaseRetriever` wrapper.

The retriever will:

- delegate initial retrieval to a wrapped retriever;
- measure anchor length with an injected token-count function;
- expand only eligible content leaf nodes below a configurable token threshold;
- follow a configurable number of `PREVIOUS` and `NEXT` links through the
  docstore;
- require neighbors to share the anchor's section identity;
- enforce a maximum added-node count and total expansion-token budget;
- deduplicate anchors and neighbors by node ID;
- assign neighbor scores using configurable distance decay;
- preserve the original anchor scores.

Path aggregation nodes and other non-leaf nodes are not eligible anchors.
Tables and images remain valid document-order neighbors so a short paragraph
can recover an adjacent structured element, but they do not themselves trigger
short-paragraph expansion.

### Advanced example

Update `examples/llama_index_advanced_rag.py` to:

- retain content nodes in original JSONL order;
- add reciprocal `PREVIOUS` and `NEXT` relationships between consecutive content
  nodes within the same section;
- wrap the reranked recursive retriever with
  `ShortContextExpandingRetriever`;
- pass that wrapper into `AutoMergingRetriever`;
- expose conservative expansion defaults in `get_advanced_components()`.

Section path nodes will not receive document-order links. Links must not cross
section boundaries.

## Configuration Defaults

Initial defaults will be conservative:

- short-anchor threshold: 100 tokens;
- previous neighbors: 1;
- next neighbors: 1;
- maximum added nodes: 2 per retrieval;
- score decay: 0.85 per hop;
- same-section constraint: enabled.

The implementation will keep these values constructor-configurable so they can
be tuned from retrieval evaluation rather than hard-coded into the algorithm.

## Failure Handling

Missing docstore entries, incomplete relationships, unsupported node metadata,
and exhausted token budgets result in skipping that expansion path. They do not
fail the original retrieval request. Relationship traversal must stop on cycles.

## Tests

Add `tests/test_context_expanding_retriever.py` with local, deterministic nodes
and no external API calls. Cover:

- short anchors add both eligible neighbors;
- long anchors do not expand;
- expansion does not cross section boundaries;
- first and last nodes are handled;
- overlapping expansions are deduplicated;
- score decay and token budgets are enforced;
- relationship cycles terminate safely;
- expanded children can still be consumed by `AutoMergingRetriever`.

Existing retrieval tests remain integration smoke tests and should not require
network credentials for the new unit coverage.

## Non-goals

- Replacing `AutoMergingRetriever`.
- Implementing SentenceWindowNodeParser.
- Changing reranker models or globally selecting a new `top_n` value.
- Unifying the current CLI schema with the advanced example's `_new.jsonl`
  schema.
- Adding BM25 retrieval in this change.
