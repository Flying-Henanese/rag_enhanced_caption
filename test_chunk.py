import asyncio
from pathlib import Path
from rag_enhanced_caption.chunker.dispatcher import chunk_markdown
with open("test_resource/rag-anything.md", "r", encoding="utf-8") as f:
    md_content = f.read()
chunks = chunk_markdown(md_content, file_id="test", filename="test.md", parser_config={"chunk_token_num": 512})
for c in chunks:
    if "Table" in c["metadata"].get("element_type", "") or "html" in c["metadata"].get("element_type", "") or "table" in c["metadata"].get("element_type", ""):
        print("---- CHUNK ----")
        print("Type:", c["metadata"]["element_type"])
        print(c["content"][:200])
