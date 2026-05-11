import importlib.util
import os
from pathlib import Path

import dotenv
import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


dotenv.load_dotenv()

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "llama_index_advanced_rag.py"
)
spec = importlib.util.spec_from_file_location("advanced_rag_example", MODULE_PATH)
advanced_rag = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(advanced_rag)

SiliconFlowRerank = advanced_rag.SiliconFlowRerank


def test_siliconflow_rerank_real_api():
    api_key = os.getenv("RERANK_API_KEY")
    if not api_key:
        pytest.skip("RERANK_API_KEY is not configured.")

    reranker = SiliconFlowRerank(
        api_key=api_key,
        endpoint=os.getenv("RERANK_ENDPOINT", "https://api.siliconflow.cn/v1/rerank"),
        model=os.getenv("RERANK_MODEL_NAME", "Pro/BAAI/bge-reranker-v2-m3"),
        top_n=1,
    )

    nodes = [
        NodeWithScore(node=TextNode(text="量子计算通过量子比特执行计算。"), score=1.0),
        NodeWithScore(node=TextNode(text="巴黎是法国的首都。"), score=1.0),
    ]
    query_bundle = QueryBundle(query_str="法国的首都是什么？")

    new_nodes = reranker._postprocess_nodes(nodes, query_bundle)

    assert len(new_nodes) == 1
    assert "巴黎" in new_nodes[0].node.text
    assert new_nodes[0].score is not None
    assert new_nodes[0].score > 0


def test_siliconflow_rerank_empty_nodes():
    api_key = os.getenv("RERANK_API_KEY")
    if not api_key:
        pytest.skip("RERANK_API_KEY is not configured.")

    reranker = SiliconFlowRerank(api_key=api_key, top_n=2)
    query_bundle = QueryBundle(query_str="test query")

    new_nodes = reranker._postprocess_nodes([], query_bundle)
    assert new_nodes == []
