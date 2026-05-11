import importlib.util
import os
from pathlib import Path

import dotenv
import pytest


dotenv.load_dotenv()

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "llama_index_advanced_rag.py"
)
spec = importlib.util.spec_from_file_location("advanced_rag_example", MODULE_PATH)
advanced_rag = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(advanced_rag)


def test_advanced_retrieval_pipeline_real_components():
    if not os.getenv("EMBEDDING_API_KEY"):
        pytest.skip("EMBEDDING_API_KEY is not configured.")

    if not advanced_rag.DOCSTORE_PATH.exists() or not advanced_rag.INDEX_PATH.exists():
        pytest.skip("Required JSONL resources for retrieval are missing.")

    _, auto_merging_retriever, _ = advanced_rag.get_advanced_components()
    query = "RAG-Anything 的架构图展示了哪些主要处理阶段？"

    final_results = auto_merging_retriever.retrieve(query)

    assert len(final_results) > 0
    assert final_results[0].node is not None
    assert final_results[0].node.text
