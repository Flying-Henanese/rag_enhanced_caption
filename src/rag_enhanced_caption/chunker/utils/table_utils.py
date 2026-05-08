"""
表格处理与格式转换工具模块。

由于源文档在不同工具之间转换时，有些包含复杂合并单元格（colspan/rowspan）的表格
无法被表达为标准的 Markdown 语法。本模块使用 BeautifulSoup 将那些以 HTML 形式嵌入的
复杂表格进行降维或扁平化处理，便于 RAG 模型准确理解表格数据。
"""

from __future__ import annotations

from bs4 import BeautifulSoup


def html_table_to_markdown(html: str) -> str:
    """
    将复杂的 HTML 表格转换为扁平的 Markdown 表格格式。

    会解析 colspan 和 rowspan，将跨行跨列的单元格值平铺展开到每个实际占据的网格中。
    但此方案在长表格上容易超限，故实际流程中可能更倾向于用键值对形式。

    Args:
        html: 包含 <table> 标签的 html 字符串。

    Returns:
        拼装好的 Markdown 表格字符串。
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return ""

    rows = table.find_all("tr")
    if not rows:
        return ""

    grid = []

    # 遍历 DOM 构建二维数组网格
    for r_idx, row in enumerate(rows):
        while len(grid) <= r_idx:
            grid.append([])

        cells = row.find_all(["td", "th"])
        c_idx = 0

        for cell in cells:
            # 找到下一个空闲位置
            while c_idx < len(grid[r_idx]) and grid[r_idx][c_idx] is not None:
                c_idx += 1

            text = cell.get_text(strip=True)
            text = text.replace("\n", " ")

            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))

            # 填充二维数组
            for r in range(rowspan):
                target_r = r_idx + r
                while len(grid) <= target_r:
                    grid.append([])

                for c in range(colspan):
                    target_c = c_idx + c
                    while len(grid[target_r]) <= target_c:
                        grid[target_r].append(None)

                    grid[target_r][target_c] = text

            c_idx += colspan

    if not grid:
        return ""

    markdown_lines = []
    max_cols = max(len(r) for r in grid)
    # 补齐尾部
    for r in grid:
        while len(r) < max_cols:
            r.append("")

    # 表头处理
    header = grid[0]
    header = [h if h is not None else "" for h in header]
    markdown_lines.append("| " + " | ".join(header) + " |")
    markdown_lines.append("|" + "|".join([" --- " for _ in range(max_cols)]) + "|")

    # 数据行处理
    for row in grid[1:]:
        row_clean = [cell if cell is not None else "" for cell in row]
        line = "| " + " | ".join(row_clean) + " |"
        markdown_lines.append(line)

    return "\n".join(markdown_lines)


def html_table_to_key_value(html: str) -> list[str]:
    """
    将 HTML 表格转换为高度易懂的键值对（Key-Value）列表。

    这是 RAG 领域的常用技巧：将表格行列关系转为自然语义。
    例如将：
    | 表头1 | 表头2 |
    | 值1   | 值2   |
    转换为：
    "表头1：值1；表头2：值2；"

    优势：非常抗截断。若遇到超长表格被截断，这种格式不会导致截断后的下半段内容因丢失表头而失去语义。

    Args:
        html: 包含 <table> 标签的 html 字符串。

    Returns:
        字符串列表，列表每项对应原表的一行数据（已被转化为 KV 串）。
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return []

    rows = table.find_all("tr")
    if not rows:
        return []

    grid = []

    # 填充二维网格（逻辑同上）
    for r_idx, row in enumerate(rows):
        while len(grid) <= r_idx:
            grid.append([])

        cells = row.find_all(["td", "th"])
        c_idx = 0

        for cell in cells:
            while c_idx < len(grid[r_idx]) and grid[r_idx][c_idx] is not None:
                c_idx += 1

            text = cell.get_text(strip=True)
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))

            for r in range(rowspan):
                target_r = r_idx + r
                while len(grid) <= target_r:
                    grid.append([])

                for c in range(colspan):
                    target_c = c_idx + c
                    while len(grid[target_r]) <= target_c:
                        grid[target_r].append(None)
                    grid[target_r][target_c] = text
            c_idx += colspan

    if not grid:
        return []

    headers = grid[0]
    headers = [h if h is not None else "" for h in headers]

    kv_lines = []

    # 如果表格只有一行，把它自己当成数据行处理
    if len(grid) == 1:
        data_rows = grid
    else:
        data_rows = grid[1:]

    # 从数据行开始映射 KV
    for row_values in data_rows:
        min_len = min(len(headers), len(row_values))
        row_parts = []
        for i in range(min_len):
            key = headers[i]
            val = row_values[i] if row_values[i] is not None else ""
            if key and key != val:  # 当只有一行时，key 和 val 是一样的，避免重复 "A：A"
                row_parts.append(f"{key}：{val}")
            elif val:
                row_parts.append(val)
        if row_parts:
            # 单行数据使用分号拼接成一整句
            kv_lines.append("；".join(row_parts) + "；")

    return kv_lines
