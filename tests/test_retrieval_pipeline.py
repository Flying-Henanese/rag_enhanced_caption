"""
测试模块：验证后端多向量检索引擎（Multi-Vector Retrieval Engine）
包括：向量初筛 -> Recursive 回表 -> AutoMerging 上下文合并的完整工作流。
"""
import pytest
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional
from loguru import logger

# Add project root to sys.path so we can import from backend
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_index.core import VectorStoreIndex, StorageContext
from backend.rag_pipeline import build_advanced_rag_index, retrieve_advanced
from rag_enhanced_caption.chunker.schema import SemanticChunk

# 开启 llama_index 的调试日志，方便观察合并过程
import logging
logging.getLogger("llama_index.core.retrievers.auto_merging_retriever").setLevel(logging.DEBUG)

# Mock 依赖：提供假的 Embedding 和 VLM，确保测试可以在无网络、无 API Key 的环境下秒级运行
class MockEmbedding:
    def __call__(self, texts: list[str]) -> list[list[float]]:
        # 返回全 0 向量，因为在测试流程中我们只是看组装和连通性
        return [[0.0] * 384 for _ in texts]

async def mock_vlm_func(user_prompt: str, system_prompt: str, image_base64: Optional[str] = None, image_bytes: Optional[bytes] = None) -> str:
    # 模拟 VLM 输出合规的 JSON
    if "table" in user_prompt.lower() or "table" in system_prompt.lower():
        return '{"summary": "这是一个模拟的表格高浓度摘要", "entities": ["Mock_Table_Entity"]}'
    return '{"summary": "这是一个模拟的图片高浓度摘要", "entities": ["Mock_Image_Entity"]}'

@pytest.fixture
def mock_markdown_content():
    return """
# 第一章 测试架构

## 1.1 模块 A

这是第一章第一小节的第一段文本内容，它负责介绍我们的核心模块 A。我们需要它作为一个普通的 Chunk。

这是第一章第一小节的第二段文本内容。这部分描述了一些背景信息。

| 模块名称 | 功能描述 |
| --- | --- |
| 模块 A1 | 处理输入 |
| 模块 A2 | 处理输出 |

这是第一章第一小节的第三段文本内容。它在表格之后，用来总结表格。

## 1.2 模块 B

这里是第二小节的内容。它与上面是相互独立的。
"""

def test_multi_vector_retrieval_pipeline(mock_markdown_content, monkeypatch):
    """
    测试：验证三级检索引擎是否正常工作，并打印详细的节点流转日志。
    """
    # 1. 替换环境变量和底层函数调用，使用 Mock 服务
    monkeypatch.setattr("backend.rag_pipeline.get_default_embedding_client", lambda: MockEmbedding())
    monkeypatch.setattr("backend.rag_pipeline.create_default_vlm_client", lambda: mock_vlm_func)

    # 2. 构建高级索引（模拟 backend/main.py 中的流程）
    print("\n\n" + "="*60)
    print("🚀 [Step 1] 开始构建多向量索引 (Multi-Vector Index)")
    print("="*60)
    
    index, storage_context, ui_results = asyncio.run(build_advanced_rag_index(
        md_content=mock_markdown_content,
        session_id="test_session_001",
        md_filename="test_doc.md",
        base_dir="."
    ))
    
    assert index is not None
    assert storage_context is not None
    assert len(ui_results) > 0

    # 打印建库结果，让开发者看清结构
    print(f"\n✅ 索引构建完成！")
    print(f"  - 向量库(Index)包含 {len(index.docstore.docs)} 个搜索节点 (Leaf Nodes)。")
    print(f"  - 文档库(Docstore)包含 {len(storage_context.docstore.docs)} 个全量节点 (Chunks, Elements, Path Nodes)。")
    
    print("\n🔍 抽样检查子节点(IndexNode)是否正确指向父节点：")
    for node_id, node in storage_context.docstore.docs.items():
        if hasattr(node, "index_id") and node.index_id:  # 这是一个 IndexNode（指针）
            print(f"   👉 [指针节点] ID: {node.id_} | 摘要: {node.text[:20]}... | 指向回表ID: {node.index_id}")

    # 3. 模拟一次查询检索
    print("\n\n" + "="*60)
    print("🚀 [Step 2] 开始多级火箭检索测试 (Vector -> Recursive -> AutoMerging)")
    print("="*60)
    
    test_query = "请解释一下表格里面提到的模块功能"
    
    # 我们拦截并临时将 auto_merging_retriever 的 verbose 设为 True 来强制输出日志
    # 为了直接使用我们封装的 retrieve_advanced，我们可以在里面稍微侵入一下，但这里作为外部测试，
    # 我们重新组装一次检索器，以便捕获和展示每一步的中间结果。
    
    # ---------------- 模拟 retrieve_advanced 的内部拆解 ----------------
    vector_retriever = index.as_retriever(similarity_top_k=5)
    
    # [火箭 1] 纯向量初筛：看看命中了什么
    initial_nodes = vector_retriever.retrieve(test_query)
    print("\n🎯 [Phase 1: Vector Search] 纯向量初筛命中的最相关节点：")
    for i, n in enumerate(initial_nodes):
        node_type = type(n.node).__name__
        print(f"   [{i+1}] 类型: {node_type} | 文本: {n.node.text[:40].replace(chr(10), '')}...")

    # [火箭 2] Recursive Retriever：回表
    from llama_index.core.retrievers import RecursiveRetriever
    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever},
        node_dict=storage_context.docstore.docs
    )
    recursive_nodes = recursive_retriever.retrieve(test_query)
    print("\n🔄 [Phase 2: Recursive Retrieval] 触发指针回表后，捞出的实际节点：")
    for i, n in enumerate(recursive_nodes):
        # 观察文本是否从“摘要”变成了“原始 Markdown 表格”
        print(f"   [{i+1}] 回表后内容: {n.node.text[:40].replace(chr(10), '')}...")

    # [火箭 3] AutoMerging Retriever：上下文合并
    from llama_index.core.retrievers import AutoMergingRetriever
    auto_merging_retriever = AutoMergingRetriever(
        recursive_retriever,
        storage_context,
        verbose=True,  # 【关键】这里开启 True，控制台就会打印合并过程
        simple_ratio_thresh=0.3 # 测试环境下调低阈值，确保必能触发合并
    )
    
    print("\n🌳 [Phase 3: Auto-Merging] 触发上下文自动合并（请观察下方 LlamaIndex 的内部合并日志）：")
    print("-" * 50)
    final_merged_nodes = auto_merging_retriever.retrieve(test_query)
    print("-" * 50)
    
    print("\n🎉 [Final Output] 最终提交给 LLM 的豪华上下文内容：")
    for i, n in enumerate(final_merged_nodes):
        content = n.node.text.replace('\n', ' ')
        is_merged = "【章节聚合：" in content
        badge = "🟩 [合并大章节]" if is_merged else "🟦 [单一独立块]"
        print(f"   {badge} {content[:80]}...")

    assert len(final_merged_nodes) > 0, "检索结果不应为空"
