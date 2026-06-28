# Harness

This directory keeps project context, reusable prompts, checklists, designs,
plans, and run records for `rag-enhanced-caption`.

The harness is intentionally lightweight. It does not replace source code,
tests, `README.md`, `README_zh.md`, `examples/README_zh.md`, or `AGENT.md`.
Instead, it provides a stable working layer for agents and humans who need to
make changes without rediscovering the same project facts.

## Directory Map

- `context/`: source-backed project facts and architecture notes.
- `prompts/`: reusable prompts for VLM, retrieval debugging, and review work.
- `checklists/`: pre-change, validation, and review checklists.
- `designs/`: proposed or accepted designs before implementation.
- `plans/`: executable implementation plans.
- `runs/`: dated records for experiments, validations, and debugging sessions.

## Source Documents

The initial harness content is distilled from:

- `AGENT.md`: project-level engineering conventions and constraints.
- `README.md`: English project overview, usage, outputs, and retrieval notes.
- `README_zh.md`: Chinese project overview, package structure, and testing notes.
- `examples/README_zh.md`: end-to-end example workflow and LlamaIndex mapping.

When these source documents change materially, update the matching files under
`context/` or `checklists/` instead of letting harness notes drift.

## Recommended Workflow

1. Read `context/project-overview.md` before broad project work.
2. Read `context/artifact-contracts.md` before changing CLI output or JSONL
   schema behavior.
3. Read `context/retrieval-pipeline.md` before changing retrieval examples,
   BM25 behavior, rerank, short-context expansion, or AutoMerge wiring.
4. Use `checklists/code-change.md` before editing source code.
5. Use `checklists/retrieval-validation.md` before claiming retrieval behavior
   is fixed or improved.
6. Save new design notes in `designs/`.
7. Save executable implementation plans in `plans/`.
8. Save real validation logs and decisions in `runs/`.

## Context Recall Rules

Use these rules whenever an agent starts from `AGENT.md`, this README, or a
fresh task description:

- For "where is this implemented?", "what owns this?", package layout,
  dependency placement, or a new feature area, read
  `context/project-overview.md`.
- For CLI output, example output, JSONL fields, node IDs, `text_for_embedding`,
  docstore records, or sparse searchable objects, read
  `context/artifact-contracts.md`.
- For BM25, vector retrieval, RRF, rerank, LlamaIndex, short-context expansion,
  AutoMerge, or recall-quality questions, read
  `context/retrieval-pipeline.md`.
- For coding style, dependency-management rules, logging, typing, async behavior,
  and docstring expectations, read `context/agent-guidelines.md`.

When more than one rule matches, read all matching context files before editing.
Do not rely on the harness alone: verify the current source and tests before
making claims or code changes.
