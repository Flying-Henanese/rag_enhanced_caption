"""
测试模块：验证特殊元素（Image/Table）与最近文本段落的 parent_id 绑定关系。
"""

from rag_enhanced_caption.chunker.parsers import semantic


def test_image_and_table_bind_to_nearest_text_parent():
    markdown_content = """
## 章节A
这是第一段正文，用于作为图片和表格的父上下文。

![架构图](attachment:arch.png)

| 列1 | 列2 |
| --- | --- |
| A   | B   |

这是第二段正文，用于作为下一张图片的父上下文。

![流程图](attachment:flow.png)
"""

    chunks = semantic.chunk_markdown(
        markdown_content,
        parser_config={"chunk_token_num": 1024},
        embed_fn=lambda texts: [[0.0] * 8 for _ in texts],
    )

    first_text_idx = next(
        i for i, c in enumerate(chunks) if "第一段正文" in c.content and c.element_type == "text"
    )
    second_text_idx = next(
        i for i, c in enumerate(chunks) if "第二段正文" in c.content and c.element_type == "text"
    )

    arch_image_chunk = next(
        c for c in chunks if c.element_type == "Image" and "arch.png" in c.content
    )
    table_chunk = next(c for c in chunks if c.element_type == "Table")
    flow_image_chunk = next(
        c for c in chunks if c.element_type == "Image" and "flow.png" in c.content
    )

    assert arch_image_chunk.parent_id == str(first_text_idx), (
        f"首张图片 parent_id 应该绑定第一段文本索引 {first_text_idx}，"
        f"实际为 {arch_image_chunk.parent_id}"
    )
    assert table_chunk.parent_id == str(first_text_idx), (
        f"表格 parent_id 应该绑定第一段文本索引 {first_text_idx}，"
        f"实际为 {table_chunk.parent_id}"
    )
    assert flow_image_chunk.parent_id == str(second_text_idx), (
        f"第二张图片 parent_id 应该绑定第二段文本索引 {second_text_idx}，"
        f"实际为 {flow_image_chunk.parent_id}"
    )

