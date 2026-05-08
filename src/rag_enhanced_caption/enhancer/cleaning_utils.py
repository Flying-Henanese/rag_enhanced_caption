import re
from bs4 import BeautifulSoup, Comment
from loguru import logger
from typing import List, Dict, Any


def clean_html_for_llm(html_content: str) -> str:
    """
    语义化清洗 HTML 内容，专为 LLM 优化。

    保留语义相关的标签（如 table, tr, td）及其关键属性（如 colspan, rowspan），
    剔除装饰性标签（如 div, span, center）和所有样式属性（如 style, class, id）。
    将 h1-h6 转换为 Markdown 标题格式。
    移除 HTML 注释。

    Args:
        html_content: 包含 HTML 标签的原始字符串。

    Returns:
        清洗后的字符串。
    """
    if not html_content or "<" not in html_content:
        return html_content

    try:
        # 使用 html.parser 以减少依赖
        soup = BeautifulSoup(html_content, "html.parser")

        # 0. 移除所有 HTML 注释
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()

        # 1. 定义需要完整保留结构的语义标签及其核心属性
        SEMANTIC_TAGS = {
            "table": [],
            "thead": [],
            "tbody": [],
            "tfoot": [],
            "tr": [],
            "th": ["colspan", "rowspan"],
            "td": ["colspan", "rowspan"],
            "caption": [],
            "b": [],
            "strong": [],
            "i": [],
            "em": [],
            "code": [],
            "pre": [],
        }

        # 2. 定义需要剔除标签但保留其内部文字的装饰标签
        DECORATIVE_TAGS = [
            "div",
            "span",
            "center",
            "p",
            "a",
            "picture",
            "source",
            "font",
            "header",
            "footer",
            "section",
            "article",
        ]

        # 3. 递归清洗
        for tag in soup.find_all(True):
            if tag.name in SEMANTIC_TAGS:
                # 保留语义标签，但只留下允许的属性
                allowed_attrs = SEMANTIC_TAGS[tag.name]
                new_attrs = {}
                for attr, value in tag.attrs.items():
                    if attr.lower() in allowed_attrs:
                        new_attrs[attr] = value
                tag.attrs = new_attrs
            elif re.match(r"^h[1-6]$", tag.name):
                # 将 h1-h6 转换为 Markdown 标题
                level = int(tag.name[1])
                prefix = "#" * level + " "
                tag.insert_before(prefix)
                tag.unwrap()
            elif tag.name in DECORATIVE_TAGS:
                # 剔除装饰标签，保留内容
                tag.unwrap()
            else:
                # 对于不在列表中的其他标签（如 img），默认保留文字但移除标签
                # 如果是 img 且有 alt 属性，可以考虑保留 alt
                if tag.name == "img":
                    alt_text = tag.get("alt", "")
                    src = tag.get("src", "")
                    if alt_text or src:
                        tag.replace_with(f"![{alt_text}]({src})")
                    else:
                        tag.decompose()
                else:
                    tag.unwrap()

        # 获取处理后的 HTML 字符串
        cleaned_html = str(soup)

        # 4. 进一步清理多余的空白符和换行
        cleaned_html = re.sub(r"\s{2,}", " ", cleaned_html)  # 压缩空格
        cleaned_html = re.sub(r"\n\s*\n", "\n\n", cleaned_html)  # 压缩空行

        return cleaned_html.strip()

    except Exception as e:
        logger.warning(
            f"HTML cleaning failed, fallback to original content. Error: {e}"
        )
        return html_content


def clean_markdown_styles(md_content: str) -> str:
    """
    清理 Markdown 中的 HTML 样式块和冗余的编码数据。
    例如移除 <style> 块、替换 Base64 图片编码为占位符等。
    """
    # 移除 <style>...</style> 块
    md_content = re.sub(r"<style.*?>.*?</style>", "", md_content, flags=re.DOTALL)

    # 替换 Base64 数据为占位符，防止撑爆 Token 限制
    # 匹配 data:image/xxx;base64, 后面的长字符串
    md_content = re.sub(
        r"data:image/[a-zA-Z0-9+-\.]+;base64,[A-Za-z0-9+/=]+",
        "[BASE64_IMAGE_OMITTED]",
        md_content,
    )

    # 使用 clean_html_for_llm 处理剩余内容
    return clean_html_for_llm(md_content)


def compress_and_relink_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    清理无意义的空块，并安全地重塑断开的 parent_id 关联。

    工作流：
    1. 对每个 chunk 进行脱水清洗。
    2. 找出清洗后为空的废弃块。
    3. 识别纯图片块，纠正其 element_type。
    4. 为每个废弃块寻找它最近的有效祖先节点（跳跃继承）。
    5. 重建列表，移除废弃块，并更新幸存块的 parent_id。
    """

    # 阶段 1 & 2: 清洗并识别待删除块
    cleaned_chunks_map = {}
    to_delete_ids = set()

    # 用于匹配纯图片的正则 (支持多张图片连在一起)
    image_pattern = re.compile(r"^(!\[.*?\]\(.*?\)\s*)+$")

    for chunk in chunks:
        raw_content = chunk.get("full_content", chunk["content"])
        cleaned = clean_markdown_styles(raw_content)

        # 将清洗后的内容暂存在 chunk 中，避免后续重复清洗
        chunk["_cleaned_content"] = cleaned

        chunk_id = chunk["id"]
        # 我们用 chunk["metadata"]["parent_id"] 或者 chunk["parent_id"] 取决于你的数据结构
        # semantic.py 里生成的结构，全局和 metadata 里都有 parent_id，这里我们以全局为准但兼顾 metadata
        # 注意 semantic.py 里返回的 parent_id 是整数型的 string，如 "5"，而最终生成的块 id 可能是 "filename_chunk_5"
        # dispatcher.py 在构建 records 时：
        # "id": f"{file_id}_chunk_{idx}", "parent_id": chunk.parent_id

        cleaned_chunks_map[chunk_id] = chunk

        if not cleaned.strip():
            # 这里我们要极度小心：只有真的是空的，而且不是为了某些特殊逻辑故意留的哨兵，才删除
            to_delete_ids.add(chunk_id)
        else:
            # 阶段 3: 类型纠正
            # 如果清洗后发现它其实就是一张或多张图片，修正类型为 Image
            # 这样原本被 div 包裹的图片就能被 VLM 正常处理了
            if (
                chunk.get("metadata", {}).get("element_type") == "html_block"
                or chunk.get("element_type") == "html_block"
            ):
                if image_pattern.match(cleaned.strip()):
                    if "metadata" in chunk:
                        chunk["metadata"]["element_type"] = "Image"
                    chunk["element_type"] = "Image"

    if not to_delete_ids:
        # 如果没有任何块被过滤，直接返回清洗后的块，清理临时字段
        for c in chunks:
            c["full_content"] = c.pop("_cleaned_content")
            c["content"] = c["full_content"]
        return chunks

    logger.info(f"Filtering out {len(to_delete_ids)} empty chunks after cleaning.")

    # 辅助函数：根据 parent_id 字符串 (e.g. "5") 解析出完整的真实 ID (e.g. "file_chunk_5")
    def _resolve_full_id(base_file_id: str, raw_parent_id: str) -> str:
        if not raw_parent_id:
            return None
        # 如果已经是完整 ID 就直接返回
        if "_chunk_" in str(raw_parent_id):
            return raw_parent_id
        return f"{base_file_id}_chunk_{raw_parent_id}"

    # 阶段 4: 追踪有效祖先 (路径压缩)
    # parent_map 记录真正的最终指向
    resolved_parent_map = {}

    file_id = chunks[0]["file_id"] if chunks else ""

    for chunk in chunks:
        chunk_id = chunk["id"]
        raw_parent = chunk.get("parent_id")

        full_parent_id = _resolve_full_id(file_id, raw_parent)

        # 沿着 parent 链一直往上找，直到找到一个【不在待删除列表】里的祖先，或者找到 None
        curr_parent = full_parent_id
        visited = set()  # 防止死循环
        while curr_parent in to_delete_ids and curr_parent not in visited:
            visited.add(curr_parent)
            # 找到要被删除的那个父节点
            dead_parent_chunk = cleaned_chunks_map.get(curr_parent)
            if dead_parent_chunk:
                # 获取废弃父节点的爷爷节点
                grand_parent = dead_parent_chunk.get("parent_id")
                curr_parent = _resolve_full_id(file_id, grand_parent)
            else:
                curr_parent = None
                break

        resolved_parent_map[chunk_id] = curr_parent

    # 阶段 5: 重建有效列表
    final_chunks = []
    for chunk in chunks:
        chunk_id = chunk["id"]
        if chunk_id in to_delete_ids:
            continue

        # 应用清洗后的内容
        chunk["full_content"] = chunk.pop("_cleaned_content")
        chunk["content"] = chunk["full_content"]

        # 应用修正后的 parent_id
        # 因为我们存入 JSONL 的 parent_id 期望是原始的 "5" 这种格式，所以需要从 full_id 中提取出来
        new_full_parent = resolved_parent_map.get(chunk_id)
        if new_full_parent and "_chunk_" in new_full_parent:
            new_short_parent = new_full_parent.split("_chunk_")[-1]
            chunk["parent_id"] = new_short_parent
            if "metadata" in chunk:
                chunk["metadata"]["parent_id"] = new_short_parent
        else:
            chunk["parent_id"] = None
            if "metadata" in chunk:
                chunk["metadata"]["parent_id"] = None

        final_chunks.append(chunk)

    return final_chunks
