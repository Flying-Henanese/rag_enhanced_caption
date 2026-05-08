"""
测试模块：验证后端多向量检索引擎（Multi-Vector Retrieval Engine）
包括：向量初筛 -> Recursive 回表 -> AutoMerging 上下文合并的完整工作流。
"""
import pytest
import asyncio
import os
import sys
from pathlib import Path
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Add project root to sys.path so we can import from backend
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.rag_pipeline import build_advanced_rag_index, retrieve_advanced

# 开启 llama_index 的调试日志，方便观察合并过程
import logging
logging.getLogger("llama_index.core.retrievers.auto_merging_retriever").setLevel(logging.DEBUG)

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

def test_multi_vector_retrieval_pipeline(mock_markdown_content):
    """
    测试：验证四级检索引擎是否正常工作，并打印详细的节点流转日志。
    """
    if not os.getenv("EMBEDDING_API_KEY"):
        pytest.skip("跳过测试：未检测到 EMBEDDING_API_KEY 环境变量。")

    # 1. 构建高级索引（模拟 backend/main.py 中的流程，使用真实服务）
    print("\n\n" + "="*60)
    print("🚀 [Step 1] 开始构建多向量索引 (Multi-Vector Index) [使用真实模型]")
    print("="*60)
    
    index, storage_context, ui_results = asyncio.run(build_advanced_rag_index(
        md_content=mock_markdown_content,
        session_id="test_session_real_001",
        md_filename="test_real_doc.md",
        base_dir="."
    ))
    
    assert index is not None
    assert storage_context is not None
    assert len(ui_results) > 0

    # 打印建库结果，让开发者看清结构
    print("\n✅ 索引构建完成！")
    print(f"  - 向量库(Index)包含 {len(index.docstore.docs)} 个搜索节点 (Leaf Nodes)。")
    print(f"  - 文档库(Docstore)包含 {len(storage_context.docstore.docs)} 个全量节点 (Chunks, Elements, Path Nodes)。")
    
    print("\n🔍 抽样检查子节点(IndexNode)是否正确指向父节点：")
    for node_id, node in storage_context.docstore.docs.items():
        if hasattr(node, "index_id") and node.index_id:  # 这是一个 IndexNode（指针）
            print(f"   👉 [指针节点] ID: {node.id_} | 摘要: {node.text[:20]}... | 指向回表ID: {node.index_id}")

    # 2. 模拟一次查询检索
    print("\n\n" + "="*60)
    print("🚀 [Step 2] 开始四级火箭检索测试 (Vector -> Recursive -> Rerank -> AutoMerging)")
    print("="*60)
    
    test_query = "请解释一下表格里面提到的模块功能"
    
    # ---------------- 模拟 retrieve_advanced 的调用 ----------------
    # 注意这里我们直接调用你刚才写的 retrieve_advanced，
    # 这样就能直接测试到 RerankedRetriever 和 AutoMerging 的真实组合效果
    
    final_merged_nodes = retrieve_advanced(
        query=test_query,
        index=index,
        storage_context=storage_context,
        top_k=15,
        rerank_top_n=5
    )
    
    print("\n🎉 [Final Output] 最终提交给 LLM 的豪华上下文内容：")
    for i, n in enumerate(final_merged_nodes):
        content = n["content"].replace('\n', ' ')
        is_merged = n["is_merged_section"]
        badge = "🟩 [合并大章节]" if is_merged else "🟦 [单一独立块]"
        print(f"   [{i+1}] {badge} Score: {n['score']:.4f} | {content[:80]}...")

    assert len(final_merged_nodes) > 0, "检索结果不应为空"
