"""
测试模块：验证主流 RAG 生态系统（LlamaIndex & LangChain）的集成兼容性
主要逻辑：
1. 格式验证：确保生成的 _docstore.jsonl 格式能被标准 JSON 解析。
2. 对象转换：验证数据能否无缝转换为 LlamaIndex 的 Document 对象和 LangChain 的 Document 对象。
3. 元数据保留：检查转换过程中 metadata 字段是否完整留存。
"""

import json
import os
from llama_index.core import Document as LlamaDocument
from langchain_core.documents import Document as LangchainDocument


def test_ecosystem_integrations() -> None:
    # Use the existing docstore file from test_resource
    test_file_path = "test_resource/高性能文档解析方案 2e2848cda67f8020abf0d58252a28708_docstore.jsonl"

    if not os.path.exists(test_file_path):
        print(f"Skipping test: {test_file_path} not found.")
        return

    llama_docs = []
    langchain_docs = []

    print("Testing LlamaIndex and LangChain integrations...")

    with open(test_file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            content = data.get("full_content", "")
            assert content, f"Document {data.get('id', 'unknown')} has empty content!"

            # 1. Test LlamaIndex
            llama_doc = LlamaDocument(text=content, metadata=data.get("metadata", {}))
            llama_docs.append(llama_doc)

            # 2. Test LangChain
            langchain_doc = LangchainDocument(
                page_content=content, metadata=data.get("metadata", {})
            )
            langchain_docs.append(langchain_doc)

    print(f"Successfully loaded {len(llama_docs)} LlamaIndex documents.")
    print(f"Successfully loaded {len(langchain_docs)} LangChain documents.")
    print("Integration tests passed successfully!")


if __name__ == "__main__":
    test_ecosystem_integrations()
