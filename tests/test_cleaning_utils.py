"""
测试模块：验证语义化清理工具 (Cleaning Utils)
主要逻辑：
1. 验证 HTML 清洗：剔除装饰性标签 (div, span) 和样式属性 (style, class)，同时保留语义结构 (table, tr, td) 及其核心属性 (colspan, rowspan)。
2. 验证 Markdown 清洗：移除 <style> 块，并将嵌入的 HTML 标签平铺为纯文本内容。
3. 验证空块过滤与重链接：确认清洗后变为空的块被剔除，并且依赖它们的子块的 parent_id 能够向上跳跃，正确指向最近的有效祖先。
目的：确保注入到 Docstore 的内容在保留结构信息的同时，最大限度地减少 Token 占用和 LLM 干扰，并且保持关联图谱不断链。
"""
from rag_enhanced_caption.enhancer.cleaning_utils import (
    clean_html_for_llm, 
    clean_markdown_styles,
    compress_and_relink_chunks
)

def test_clean_html_for_llm():
    # 测试 1: 带有样式的表格
    html_table = """
    <div style="background: red;">
        <center>
            <table style="width: 100%; border: 1px solid black;" class="my-table">
                <tr style="height: 20px;">
                    <th colspan="2" style="color: blue;">Header</th>
                </tr>
                <tr>
                    <td id="cell-1">Content 1</td>
                    <td rowspan="2">Content 2</td>
                </tr>
            </table>
        </center>
    </div>
    """
    cleaned = clean_html_for_llm(html_table)
    assert "colspan=\"2\"" in cleaned
    assert "rowspan=\"2\"" in cleaned
    assert "style=" not in cleaned
    assert "class=" not in cleaned
    assert "id=" not in cleaned
    assert "div" not in cleaned
    assert "center" not in cleaned

def test_clean_markdown_styles():
    # 测试 2: 带有 style 块的 Markdown
    md_with_style = """
    <style>
    .heavy { font-weight: bold; }
    </style>
    # Title
    <div class="heavy">Hello World</div>
    """
    cleaned_md = clean_markdown_styles(md_with_style)
    assert "<style>" not in cleaned_md
    assert "Hello World" in cleaned_md
    assert "class=" not in cleaned_md

def test_compress_and_relink_chunks():
    # 测试 3: 空块过滤与指针跳跃继承
    raw_chunks = [
        {
            "id": "fileA_chunk_0",
            "file_id": "fileA",
            "content": "Valid Text 1",
            "parent_id": None
        },
        {
            "id": "fileA_chunk_1",
            "file_id": "fileA",
            "content": "<div align='center'>\n</div>", # 清洗后为空
            "parent_id": "0"
        },
        {
            "id": "fileA_chunk_2",
            "file_id": "fileA",
            "content": "![Image](img.png)",
            "parent_id": "1" # 原本指向空块
        },
        {
            "id": "fileA_chunk_3",
            "file_id": "fileA",
            "content": "<span style='color:red'></span>", # 清洗后为空
            "parent_id": "2"
        },
        {
            "id": "fileA_chunk_4",
            "file_id": "fileA",
            "content": "| Table |",
            "parent_id": "3" # 指向空块，空块又指向 2
        }
    ]

    compressed = compress_and_relink_chunks(raw_chunks)

    # 验证数量：应该剔除了 1 和 3
    assert len(compressed) == 3
    ids = [c["id"] for c in compressed]
    assert "fileA_chunk_1" not in ids
    assert "fileA_chunk_3" not in ids

    # 验证内容更新
    assert "div" not in compressed[0].get("full_content", compressed[0]["content"])

    # 验证指针重塑
    chunk_2 = next(c for c in compressed if c["id"] == "fileA_chunk_2")
    assert chunk_2["parent_id"] == "0", f"Expected parent 0, got {chunk_2['parent_id']}"

    chunk_4 = next(c for c in compressed if c["id"] == "fileA_chunk_4")
    assert chunk_4["parent_id"] == "2", f"Expected parent 2, got {chunk_4['parent_id']}"
