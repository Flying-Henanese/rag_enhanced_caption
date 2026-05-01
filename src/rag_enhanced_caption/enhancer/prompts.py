"""
Prompts for Markdown-based Multi-modal Enhanced Captioning
"""

IMAGE_ANALYSIS_SYSTEM = "You are an expert at analyzing documents. Provide detailed, accurate descriptions and output ONLY valid JSON."
TABLE_ANALYSIS_SYSTEM = "You are an expert at analyzing documents. Provide detailed table analysis with specific insights and output ONLY valid JSON."

# 带语境的图片分析
VISION_PROMPT_WITH_CONTEXT = """Please examine this image in conjunction with its surrounding context from the document.

CRITICAL INSTRUCTION TO AVOID OCR ATTACK:
Often, images (especially screenshots, scanned pages, or examples) contain a large amount of text. Your task is NOT to transcribe or summarize the text inside the image. 
Instead, you MUST use the provided [章节主题] (Section Topic) and [语境信息] (Context Information) to understand WHY this image is placed here.
Explain what the image demonstrates, proves, or illustrates in relation to the context.

Provide a JSON response:
{{
    "detailed_description": "A clear description explaining the image's purpose and how it relates to the context. Focus on its structural or illustrative role rather than transcribing its text content.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "1-2 sentence summary of what the image illustrates in this context"
    }}
}}

Document Context:
{context}

Image Source: {image_path}
"""

# 基础图片分析
VISION_PROMPT = """Please analyze this image and provide a JSON response:

{{
    "detailed_description": "Detailed visual description of composition, objects, and text.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "concise summary"
    }}
}}

Image Source: {image_path}
"""

# 表格分析提示词
TABLE_PROMPT_WITH_CONTEXT = """Please examine this table in conjunction with its surrounding context from the document.

Your task is to understand the purpose of this table based on the provided [章节主题] (Section Topic) and [语境信息] (Context Information). 
Explain what data or relationship the table illustrates in relation to the context, rather than just repeating the raw data.

Provide a JSON response:
{{
    "detailed_description": "Analyze the table's structure and core message, heavily incorporating the context to explain WHAT the data actually represents and WHY it is significant here.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "1-2 sentence summary of what the table illustrates in this context"
    }}
}}

Document Context:
{context}

Table Data:
{table_body}
"""

TABLE_PROMPT = """Please analyze this table and provide a JSON response:

{{
    "detailed_description": "Analyze table structure, trends, and key data points.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "concise summary of findings"
    }}
}}
"""

# ---------------------------------------------------------
# 以下为适配传统 RAG 的精简版 (Concise) 提示词
# ---------------------------------------------------------

VISION_PROMPT_CONCISE_WITH_CONTEXT = """Please briefly summarize this image in conjunction with its surrounding context.

CRITICAL INSTRUCTION: Avoid merely transcribing text from the image. Use the [章节主题] and [语境信息] to briefly explain what the image demonstrates or illustrates.

Provide a concise JSON response:
{{
    "detailed_description": "A very brief summary (under 50 words) of the image's core illustrative purpose in relation to the context.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "1-sentence summary"
    }}
}}

Document Context:
{context}

Image Source: {image_path}
"""

VISION_PROMPT_CONCISE = """Please briefly summarize this image. Provide a concise JSON response:

{{
    "detailed_description": "A very brief summary (under 50 words) of the core content of the image.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "1-sentence summary"
    }}
}}

Image Source: {image_path}
"""
