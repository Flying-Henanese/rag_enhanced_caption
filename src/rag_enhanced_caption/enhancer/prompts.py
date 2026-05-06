"""
Prompts for Markdown-based Multi-modal Enhanced Captioning
"""

IMAGE_ANALYSIS_SYSTEM = (
    "You are an expert data analyst and document parser. "
    "Your goal is to extract pure semantic meaning and business entities from images. "
    "CRITICAL RULE: DO NOT output any file names, extensions, or metadata like 'This is an image named image.png'. "
    "Output ONLY valid JSON."
)

TABLE_ANALYSIS_SYSTEM = (
    "You are an expert data analyst and document parser. "
    "Your goal is to extract pure semantic meaning and business entities from tables. "
    "Output ONLY valid JSON."
)

# 带语境的图片分析
VISION_PROMPT_WITH_CONTEXT = """Please examine this image in conjunction with its surrounding context from the document.

CRITICAL INSTRUCTION:
1. DO NOT transcribe all text from the image.
2. DO NOT mention the filename or extension.
3. Use the provided [章节主题] (Section Topic) and [语境信息] (Context Information) to explain what the image demonstrates or proves.
4. Extract specific technical terms, numbers, system components, or names as entities.

Provide a JSON response strictly matching this schema:
{{
    "summary": "A high-density 1-3 sentence summary explaining the core business or technical meaning of the image in this context.",
    "entities": ["entity1", "entity2", "entity3"]
}}

Document Context:
{context}

Image Source (for your reference only, DO NOT output this): {image_path}
"""

# 基础图片分析
VISION_PROMPT = """Please analyze this image.

CRITICAL INSTRUCTION:
1. DO NOT mention the filename or extension.
2. Extract specific technical terms, numbers, system components, or names as entities.

Provide a JSON response strictly matching this schema:
{{
    "summary": "A high-density 1-3 sentence summary explaining the core business or technical meaning of the image.",
    "entities": ["entity1", "entity2", "entity3"]
}}

Image Source (for your reference only, DO NOT output this): {image_path}
"""

# 表格分析提示词
TABLE_PROMPT_WITH_CONTEXT = """Please examine this table in conjunction with its surrounding context from the document.

CRITICAL INSTRUCTION:
1. Explain what data or relationship the table illustrates in relation to the context.
2. Extract specific values, metrics, product names, or key findings as entities.

Provide a JSON response strictly matching this schema:
{{
    "summary": "A high-density summary explaining the core message of the table, incorporating the context.",
    "entities": ["entity1", "entity2", "entity3"]
}}

Document Context:
{context}

Table Data:
{table_body}
"""

TABLE_PROMPT = """Please analyze this table.

CRITICAL INSTRUCTION:
1. Explain the core trends, structure, or key data points.
2. Extract specific values, metrics, or key findings as entities.

Provide a JSON response strictly matching this schema:
{{
    "summary": "A high-density summary of the table's findings.",
    "entities": ["entity1", "entity2", "entity3"]
}}

Table Data:
{table_body}
"""

# ---------------------------------------------------------
# 以下为适配传统 RAG 的精简版 (Concise) 提示词
# ---------------------------------------------------------

VISION_PROMPT_CONCISE_WITH_CONTEXT = """Please briefly summarize this image in conjunction with its surrounding context.

CRITICAL INSTRUCTION: DO NOT mention the filename. Extract core entities.

Provide a concise JSON response:
{{
    "summary": "A 1-sentence summary of the image's core illustrative purpose.",
    "entities": ["entity1", "entity2"]
}}

Document Context:
{context}

Image Source (DO NOT output this): {image_path}
"""

VISION_PROMPT_CONCISE = """Please briefly summarize this image. 

CRITICAL INSTRUCTION: DO NOT mention the filename.

Provide a concise JSON response:
{{
    "summary": "A 1-sentence summary.",
    "entities": ["entity1", "entity2"]
}}

Image Source (DO NOT output this): {image_path}
"""
