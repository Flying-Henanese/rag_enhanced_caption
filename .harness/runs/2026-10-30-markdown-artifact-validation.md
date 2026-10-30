# Markdown artifacts and live VLM retry

## Scope and reproducible evidence

This session processed the existing industrial Markdown sample
`test_resource/test4.md` (wind-turbine tower document), plus a small inspection
fixture containing an audience list, an alarm reaction/acknowledgment table,
and cable inspection steps.

Generated artifacts, the fixture, and the verification script are in the
git-ignored directory `output/2026-10-30-artifact-validation/`. This run record
preserves the findings even if those local outputs are later removed.

```bash
uv run rag-caption output/2026-10-30-artifact-validation/industrial-check.md output/2026-10-30-artifact-validation/cli
uv run python output/2026-10-30-artifact-validation/validate_artifacts.py
```

## Structural validation

| Flow | Index/docstore rows | Valid parent references | Runtime parents checked |
| --- | --- | --- | --- |
| Small fixture, base CLI | 7 / 7 | 1 / 1 | Not built in CLI run |
| Small fixture, advanced example | 7 / 7 | 1 / 1 | 6 |
| Existing test4, advanced example | 338 / 338 | 104 / 104 | 287 |

All pairs had unique, matching IDs, non-empty embedding text, non-empty original
content, and resolvable parent references. The large sample contained 230 text
chunks, 4 list chunks, 43 math blocks, 43 Table KV chunks, and 18 image chunks.
Every checked runtime parent's text covered the text of each immediate child,
including already-complete child parents.

The small fixture retained all four audience categories after real AutoMerge,
with each category appearing exactly once in final context. Lists remained
ordinary text records; the table remained a multimodal child with a valid
parent reference.

Validation modes matter:

- Initial base CLI run: real configured VLM call returned HTTP 401 and the table
  used source-text fallback.
- Advanced-example ingestion: table summaries and image loading were explicitly
  disabled, using empty-summary and missing-image callbacks. The actual parser,
  cleaner, processor fallback, and artifact writers ran. These outputs do not
  establish successful VLM image/table enhancement.
- Runtime tree checks: real LlamaIndex index construction and AutoMerge ran with
  local `MockEmbedding`. They validate content coverage, not semantic ranking.

Persisted docstore rows are individual chunks. Complete parent-context copies
are constructed in memory at retrieval startup, after vector-index construction,
and do not rewrite the JSONL. Snapshots are in
`industrial-check_runtime_parents.jsonl` and `test4_runtime_parents.jsonl`.

## Successful live VLM retry after user configuration correction

The same small fixture was processed again through the real CLI processing
function with the corrected configuration. The successful output directory is
`output/2026-10-30-artifact-validation/cli-vlm-fixed/`; the log is
`output/2026-10-30-artifact-validation/vlm-fixed.log`.

There were no VLM analysis/authentication errors in this successful run. It
generated seven index/docstore pairs with valid IDs and parent references. The
table record `c_industrial-check_chunk_3` points to
`p_industrial-check_chunk_2` and has a generated summary plus keywords `OFF2`
and `立即（上电）`, rather than the source-table fallback. Docstore preserved:

```markdown
| 字段 | 内容 |
| --- | --- |
| 默认反应 | OFF2 |
| 应答方式 | 立即（上电） |
```

The generated summary was:

> 该表描述了核验样例中默认反应状态为'OFF2'，且应答方式为'立即（上电）'，表明系统在上电后立即响应，且默认处于OFF2状态。

The second clause interprets the fields beyond what the source table states.
Authentication, summary generation, and persistence succeeded; summary
faithfulness is a separate unresolved concern. No prompt or parser change was
made to address it in this session.

## Other observations and limits

- The local small-fixture retrieval returned both a complete ancestor section
  and an alarm child section. `OFF2` occurred twice in final context. Evidence
  retention passed, but ancestor/descendant overlap remains unaddressed.
- The later CLI success does not establish that Embedding or Rerank credentials
  work: their live-service tests were not rerun after configuration correction.
- The full test4 document was not rerun with live VLM enhancement.
- The historical 828D ten-question benchmark was not rerun. In its trace, Q1
  loses an expanded list during merging; Q6/Q7 already have relevant fields in
  the new pipeline before short-context expansion. Q6 also cites similar alarm
  identifiers. Do not attribute those two historical successes solely to the
  expansion step or claim an exact Q6 source match from matching field values.

Related automated verification:

```text
uv run pytest tests/test_cli_process_document.py tests/test_example_lexical_pipeline.py tests/test_parent_id_linkage.py tests/test_semantic_parent_binding.py tests/test_semantic_parser_fixes.py tests/test_retrieval_components.py -k 'not real_components' -q --tb=short
26 passed, 1 deselected
```

The deselected test requires the real embedding service. This local suite and
the successful VLM sample run do not establish new retrieval or answer accuracy.
