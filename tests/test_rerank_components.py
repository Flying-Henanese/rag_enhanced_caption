import asyncio
import importlib.util
import os
from pathlib import Path
from typing import Any, NoReturn

import dotenv
import httpx
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


def test_siliconflow_rerank_logs_and_falls_back_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exceptions: list[str] = []

    class TimeoutClient:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["timeout"] == 30.0

        async def __aenter__(self) -> "TimeoutClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, *args: object, **kwargs: Any) -> NoReturn:
            raise httpx.TimeoutException("request timed out")

    monkeypatch.setattr(advanced_rag.httpx, "AsyncClient", TimeoutClient)
    monkeypatch.setattr(advanced_rag.logger, "exception", exceptions.append)
    reranker = SiliconFlowRerank(api_key="test-key", top_n=1)
    nodes = [
        NodeWithScore(node=TextNode(text="first"), score=1.0),
        NodeWithScore(node=TextNode(text="second"), score=0.5),
    ]

    result = asyncio.run(
        reranker._apostprocess_nodes(nodes, QueryBundle(query_str="query"))
    )

    assert result == nodes[:1]
    assert exceptions == ["Rerank request timed out after 30.0 seconds"]


def test_siliconflow_rerank_sync_path_falls_back_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(advanced_rag.logger, "warning", warnings.append)
    reranker = SiliconFlowRerank(api_key="test-key", top_n=1)
    nodes = [
        NodeWithScore(node=TextNode(text="first"), score=1.0),
        NodeWithScore(node=TextNode(text="second"), score=0.5),
    ]

    result = reranker._postprocess_nodes(nodes, QueryBundle(query_str="query"))

    assert result == nodes[:1]
    assert warnings == [
        "Remote reranking requires apostprocess_nodes(); returning "
        "unreranked candidates on the synchronous path."
    ]


def test_siliconflow_rerank_real_api() -> None:
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

    new_nodes = asyncio.run(reranker._apostprocess_nodes(nodes, query_bundle))

    assert len(new_nodes) == 1
    assert "巴黎" in new_nodes[0].node.text
    assert new_nodes[0].score is not None
    assert new_nodes[0].score > 0


def test_siliconflow_rerank_empty_nodes() -> None:
    api_key = os.getenv("RERANK_API_KEY")
    if not api_key:
        pytest.skip("RERANK_API_KEY is not configured.")

    reranker = SiliconFlowRerank(api_key=api_key, top_n=2)
    query_bundle = QueryBundle(query_str="test query")

    new_nodes = asyncio.run(reranker._apostprocess_nodes([], query_bundle))
    assert new_nodes == []
