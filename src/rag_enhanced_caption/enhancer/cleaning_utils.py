import re
from bs4 import BeautifulSoup, Tag
from loguru import logger

def clean_html_for_llm(html_content: str) -> str:
    """
    语义化清洗 HTML 内容，专为 LLM 优化。
    
    保留语义相关的标签（如 table, tr, td）及其关键属性（如 colspan, rowspan），
    剔除装饰性标签（如 div, span, center）和所有样式属性（如 style, class, id）。
    
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
            "h1": [], "h2": [], "h3": [], "h4": [], "h5": [], "h6": []
        }

        # 2. 定义需要剔除标签但保留其内部文字的装饰标签
        DECORATIVE_TAGS = ["div", "span", "center", "p", "a", "picture", "source", "font", "header", "footer", "section", "article"]

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
        cleaned_html = re.sub(r'\s{2,}', ' ', cleaned_html) # 压缩空格
        cleaned_html = re.sub(r'\n\s*\n', '\n\n', cleaned_html) # 压缩空行
        
        return cleaned_html.strip()

    except Exception as e:
        logger.warning(f"HTML cleaning failed, fallback to original content. Error: {e}")
        return html_content

def clean_markdown_styles(md_content: str) -> str:
    """
    清理 Markdown 中的 HTML 样式块。
    例如移除 <style> 块或包含大量样式的 <div>。
    """
    # 移除 <style>...</style> 块
    md_content = re.sub(r'<style.*?>.*?</style>', '', md_content, flags=re.DOTALL)
    
    # 使用 clean_html_for_llm 处理剩余内容
    return clean_html_for_llm(md_content)
