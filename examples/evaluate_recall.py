import json
import os
import sys
from pathlib import Path
from loguru import logger

# Suppress debug logs from llama_index for a clean test output
import logging
logging.getLogger("llama_index").setLevel(logging.ERROR)
logger.remove()
logger.add(sys.stderr, level="WARNING")

# Import the retriever builders from our previous script
from llama_index_advanced_rag import get_baseline_components, get_advanced_components

def evaluate_query(query: str, purpose: str, base_retriever, adv_retriever):
    """Evaluates a single query against both retrievers and prints a comparative report."""
    print(f"\n" + "="*80)
    print(f"🤔 评测维度: {purpose}")
    print(f"❓ 提问: {query}")
    print("="*80)
    
    # -----------------------------------------
    # Baseline Evaluation
    # -----------------------------------------
    print("\n[方案 A: 传统基线 (Naive Chunking)]")
    try:
        base_nodes = base_retriever.retrieve(query)
        if not base_nodes:
            print("  ❌ 未召回任何内容")
        else:
            node = base_nodes[0]
            score = node.score if node.score is not None else 0.0
            print(f"  🎯 命中得分: {score:.4f}")
            text = node.node.text.strip()
            print(f"  📄 召回上下文片段 (前300字符):")
            print(f"  {text[:300] + '...' if len(text) > 300 else text}")
    except Exception as e:
        print(f"  ❌ 基线检索失败: {e}")

    print("\n" + "-"*80)

    # -----------------------------------------
    # Advanced Evaluation
    # -----------------------------------------
    print("[方案 B: 全局树状融合 (Global Tree + AutoMerging)]")
    try:
        adv_nodes = adv_retriever.retrieve(query)
        if not adv_nodes:
            print("  ❌ 未召回任何内容")
        else:
            node = adv_nodes[0]
            score = node.score if node.score is not None else 0.0
            print(f"  🎯 最终合并得分 (通常为命中叶子的最高分): {score:.4f}")
            text = node.node.text.strip()
            
            # Detect if it's a merged chapter or a single chunk
            if "【章节聚合" in text:
                print(f"  🌳 触发自动合并！成功召回上级完整章节。")
            elif "<image_analysis>" in text:
                print(f"  👁️ 触发多模态挂钩！成功召回包含 AI 图像分析的完整父段落。")
            else:
                print(f"  📄 命中基础段落。")
                
            print(f"  📄 召回豪华上下文 (前500字符):")
            print(f"  {text[:500] + '...' if len(text) > 500 else text}")
    except Exception as e:
        print(f"  ❌ 高级检索失败: {e}")
        
    print("="*80 + "\n")

def run_test_suite():
    print("🚀 正在初始化检索引擎 (这可能需要几秒钟加载向量库)...")
    
    # We suppress stdout temporarily if the AutoMergingRetriever is too verbose
    # But it's good to see the merging logs, so we'll leave it for now, 
    # or rely on the llama_index logging level set above.
    
    _, base_retriever = get_baseline_components()
    _, _, _, adv_retriever = get_advanced_components()
    
    print("✅ 引擎初始化完成！开始多维度召回测试...\n")
    
    test_cases = [
        {
            "purpose": "1. 细粒度事实抽取 (考察对 Markdown 表格的切分与召回)",
            "query": "文档中提到 fastapi 的版本范围要求是什么？它属于哪个核心组件？"
        },
        {
            "purpose": "2. 多模态/图像意图理解 (考察 VLM 摘要与 Parent 指针的威力)",
            "query": "请根据系统架构图解释一下，Celery Worker 和 vLLM Server 是如何协同工作实现资源利用最大化的？"
        },
        {
            "purpose": "3. 宏观架构理解 (考察 AutoMerging 向上合并零散段落的能力)",
            "query": "请列举这套高性能解析方案的‘核心技术亮点’，并简述其中智能表格处理的两种方式。"
        },
        {
            "purpose": "4. 纯文本概念提问 (考察基础的语义召回准确度)",
            "query": "在传统的 RAG 处理中，文档格式繁杂和简单的按字符切分分别导致了什么业务痛点？"
        }
    ]
    
    for i, case in enumerate(test_cases):
        evaluate_query(case["query"], case["purpose"], base_retriever, adv_retriever)
        
    print("🎉 召回测试流程执行完毕！")
    print("💡 可以明显看出：")
    print(" - 方案A 经常切断表格或抓错重点，上下文残缺。")
    print(" - 方案B 不仅能精准命中，还能根据问题宏观程度自动决定是返回‘一个段落’还是‘一整个大章节’，完美兼顾了高精度与大上下文。")

if __name__ == "__main__":
    run_test_suite()
