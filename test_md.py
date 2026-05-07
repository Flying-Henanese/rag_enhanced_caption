import markdown_it

md = markdown_it.MarkdownIt("commonmark")
text = """- **发布PaddleOCR-VL**：
    - **模型介绍**：
        - **PaddleOCR-VL** 是一款..."""

tokens = md.parse(text)
for t in tokens:
    print(f"{t.type}: {t.map}, {t.content}")
