import pytest
import sys
import os
from pathlib import Path

# Add project root to sys.path so we can import from backend
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from backend.rag_pipeline import SiliconFlowRerank
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle

def test_siliconflow_rerank_real_api():
    """
    Test using REAL SiliconFlow Rerank API.
    Verifies that the postprocessing actually sorts and filters the nodes.
    """
    api_key = os.getenv("RERANK_API_KEY")
    if not api_key:
        pytest.skip("RERANK_API_KEY is not configured.")

    reranker = SiliconFlowRerank(
        api_key=api_key,
        endpoint=os.getenv("RERANK_ENDPOINT", "https://api.siliconflow.cn/v1/rerank"),
        model=os.getenv("RERANK_MODEL_NAME", "Qwen/Qwen3-Reranker-0.6B"),
        top_n=1
    )
    
    # Create dummy nodes. One is highly relevant to the query, one is noise.
    nodes = [
        NodeWithScore(node=TextNode(text="Paris is the capital of France."), score=1.0),
        NodeWithScore(node=TextNode(text="Quantum computing utilizes qubits for computation."), score=1.0)
    ]
    query_bundle = QueryBundle(query_str="What is the capital of France?")
    
    # Call the real API
    new_nodes = reranker._postprocess_nodes(nodes, query_bundle)
    
    # Verify the reranker filtered and sorted the nodes correctly
    assert len(new_nodes) == 1, "Reranker failed to filter down to top_n=1"
    assert "Paris" in new_nodes[0].node.text, "Reranker failed to pick the relevant context"
    assert new_nodes[0].score > 0, "Reranker failed to assign a positive score"
    print(f"\n[Real Rerank Passed] Kept relevant node with score: {new_nodes[0].score:.4f}")

def test_siliconflow_rerank_empty_nodes():
    api_key = os.getenv("RERANK_API_KEY")
    if not api_key:
        pytest.skip("RERANK_API_KEY is not configured.")

    reranker = SiliconFlowRerank(api_key=api_key, top_n=2)
    query_bundle = QueryBundle(query_str="Test query")
    
    # Should handle empty lists gracefully without throwing an error
    new_nodes = reranker._postprocess_nodes([], query_bundle)
    assert new_nodes == []
