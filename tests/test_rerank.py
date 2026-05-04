import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# We dynamically import the SiliconFlowRerank class from the examples folder
# to ensure it functions as expected during CI testing.
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from llama_index_advanced_rag import SiliconFlowRerank
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle

def test_siliconflow_rerank_postprocess():
    # Initialize the reranker with a dummy key and top_n=1 to test filtering
    reranker = SiliconFlowRerank(api_key="test_key", top_n=1)
    
    # Create some dummy nodes
    nodes = [
        NodeWithScore(node=TextNode(text="This is a noisy context that should be ranked lower."), score=0.5),
        NodeWithScore(node=TextNode(text="This is the highly relevant context containing the exact answer."), score=0.5)
    ]
    query_bundle = QueryBundle(query_str="What is the highly relevant context?")
    
    # Mock the SiliconFlow API response to avoid actual network calls during testing
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.1},
            {"index": 1, "relevance_score": 0.95}
        ]
    }
    
    with patch("requests.post", return_value=mock_response) as mock_post:
        new_nodes = reranker._postprocess_nodes(nodes, query_bundle)
        
        # Verify that the post request was made with the correct arguments
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test_key"
        assert kwargs["json"]["query"] == query_bundle.query_str
        assert "documents" in kwargs["json"] # Verify the correct payload field is used
        assert len(kwargs["json"]["documents"]) == 2
        
        # Verify the reranker filtered and sorted the nodes correctly
        assert len(new_nodes) == 1
        assert new_nodes[0].node.text == "This is the highly relevant context containing the exact answer."
        assert new_nodes[0].score == 0.95

def test_siliconflow_rerank_empty_nodes():
    reranker = SiliconFlowRerank(api_key="test_key", top_n=2)
    query_bundle = QueryBundle(query_str="Test query")
    
    # Should handle empty lists gracefully without calling the API
    with patch("requests.post") as mock_post:
        new_nodes = reranker._postprocess_nodes([], query_bundle)
        mock_post.assert_not_called()
        assert new_nodes == []
