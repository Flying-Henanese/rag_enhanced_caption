# VLM Captioning Prompt Notes

Sources: `README.md`, `README_zh.md`, `AGENTS.md`

Use this file when adjusting or reviewing prompts for multimodal enrichment.
The live prompt implementation is in `src/rag_enhanced_caption/enhancer/prompts.py`.

## Prompt Goals

- Produce retrieval-friendly semantic summaries for images and tables.
- Preserve factual details such as labels, row names, dates, quantities,
  entities, and visible relationships.
- Avoid over-describing decorative or irrelevant visual details.
- Return output that downstream parsing can handle robustly.

## Project Constraints

- VLM endpoint must be OpenAI-compatible.
- VLM calls should remain async in processing pipelines.
- SVG images are currently skipped for VLM analysis.
- VLM parsing should use `robust_json_parse` rather than fragile direct JSON
  parsing.

## Review Questions

- Does the prompt produce text useful for `text_for_embedding`?
- Does it preserve enough complete detail for table or image retrieval?
- Does it avoid hallucinating unseen values?
- Is the response format compatible with the parser?
- Does the enriched output keep Markdown rendering valid?
