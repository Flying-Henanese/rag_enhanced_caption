"""
测试模块：验证语义化清理工具 (Cleaning Utils)
主要逻辑：
1. 验证 HTML 清洗：剔除装饰性标签 (div, span) 和样式属性 (style, class)，同时保留语义结构 (table, tr, td) 及其核心属性 (colspan, rowspan)。
2. 验证 Markdown 清洗：移除 <style> 块，并将嵌入的 HTML 标签平铺为纯文本内容。
目的：确保注入到 Docstore 的内容在保留结构信息的同时，最大限度地减少 Token 占用和 LLM 干扰。
"""
from rag_enhanced_caption.enhancer.cleaning_utils import clean_html_for_llm, clean_markdown_styles

def test_cleaning():
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
    print("--- Cleaned HTML Table ---")
    print(cleaned)
    assert "colspan=\"2\"" in cleaned
    assert "rowspan=\"2\"" in cleaned
    assert "style=" not in cleaned
    assert "class=" not in cleaned
    assert "id=" not in cleaned
    assert "div" not in cleaned
    assert "center" not in cleaned

    # 测试 2: 带有 style 块的 Markdown
    md_with_style = """
    <style>
    .heavy { font-weight: bold; }
    </style>
    # Title
    <div class="heavy">Hello World</div>
    """
    cleaned_md = clean_markdown_styles(md_with_style)
    print("\n--- Cleaned Markdown ---")
    print(cleaned_md)
    assert "<style>" not in cleaned_md
    assert "Hello World" in cleaned_md
    assert "class=" not in cleaned_md

    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_cleaning()
