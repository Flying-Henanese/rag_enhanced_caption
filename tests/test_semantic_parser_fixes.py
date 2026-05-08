"""
测试模块：验证 Semantic Parser 针对特定 Bug 的修复（嵌套列表重复、HTML Block 上下文断裂）。
"""

from rag_enhanced_caption.chunker.parsers import semantic


def test_nested_list_duplication_fix():
    """
    测试点：嵌套列表在解析时不再发生内容翻倍、解析错位或生成多余重复块的问题。
    原先暴力的 while 循环会导致 "发布PaddleOCR-VL" 这样的内容被多次 append 到同一个 Chunk，
    进而因为超长触发截断产生 Part 1, Part 2 这样的重复冗余块。
    """
    markdown_content = """
## 特性列表
- **发布PaddleOCR-VL**：
    - **模型介绍**：
        - 这是一个强大的视觉语言模型。
        - 它的性能达到了 SOTA。
    - **多语言支持**：
        - 支持 109 种语言。
- **发布PP-OCRv5**：
    - 提升了小语种识别准度。
"""
    # Mock 一个空的 embed_fn，避免触发真实的远端调用，这里主要测的是解析和组装逻辑
    chunks = semantic.chunk_markdown(
        markdown_content, embed_fn=lambda x: [0.0] * len(x)
    )

    # 筛选包含列表内容的 Chunk
    list_chunks = [c for c in chunks if "发布PaddleOCR-VL" in c.content]
    assert len(list_chunks) == 1, (
        "嵌套列表应该被完整提取为一个 Chunk，而不是被错误切分成多个或产生冗余分段。"
    )

    chunk = list_chunks[0]
    assert chunk.element_type in ["bullet_list_open", "ordered_list_open"], (
        "元素类型应该被识别为列表。"
    )

    # 验证所有层级的列表内容都被完整保留
    assert "这是一个强大的视觉语言模型。" in chunk.content
    assert "支持 109 种语言。" in chunk.content
    assert "提升了小语种识别准度。" in chunk.content

    # 验证内容没有内部翻倍重复
    assert chunk.content.count("发布PaddleOCR-VL") == 1, (
        "列表内容发生内部翻倍重复（这是修护前的典型 Bug）！"
    )


def test_html_block_summary_context_fix():
    """
    测试点：html_block 中的 <summary> 能够被正确提取为虚拟标题，
    从而保障 details 标签内部包裹的 Markdown 元素不会丢失上下文 (parent_id/header_path)。
    """
    markdown_content = """
## 版本记录

<details>
<summary><strong>2025.10.16: PaddleOCR 3.3.0 发布</strong></summary>

- **特性一**：SOTA性能
- **特性二**：多语言支持

</details>
"""
    chunks = semantic.chunk_markdown(
        markdown_content, embed_fn=lambda x: [0.0] * len(x)
    )

    # html_block (<details> 标签本身) 应该作为一个 Chunk 被输出
    html_chunks = [c for c in chunks if c.element_type == "html_block"]
    assert len(html_chunks) >= 1, "未能提取出 html_block。"

    # <details> 内部的 Markdown 列表应该单独成块，并且继承 <summary> 提取出的上下文
    list_chunks = [c for c in chunks if "特性一" in c.content]
    assert len(list_chunks) == 1, "细节块内部的 Markdown 列表未被正确切分提取。"

    list_chunk = list_chunks[0]

    # 验证虚拟标题是否被成功注入到 header_path (层级路径) 中
    has_virtual_heading = any(
        "PaddleOCR 3.3.0 发布" in header for header in list_chunk.header_path
    )
    assert has_virtual_heading, (
        "Markdown 列表的 header_path 未能继承 <summary> 定义的虚拟标题上下文！"
    )

    # 验证之前的常规标题上下文依然存在
    has_normal_heading = any("版本记录" in header for header in list_chunk.header_path)
    assert has_normal_heading, "原有的外层标题上下文 (版本记录) 丢失了。"
