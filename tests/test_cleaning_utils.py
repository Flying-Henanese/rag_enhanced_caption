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
