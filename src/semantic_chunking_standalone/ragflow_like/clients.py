"""
Embedding 客户端工具模块。

提供获取本地和远程 Embedding 客户端的方法，用于语义分块过程中的文本向量化计算，以实现智能聚类和分段。
支持通过环境变量灵活配置本地 SentenceTransformer 模型或远程 SiliconFlow/OpenAI 兼容 API。
"""
import os
import httpx
from typing import List, Any, Optional, Callable
from loguru import logger
from dotenv import load_dotenv

# 加载环境变量 (主要是获取 API KEY、ENDPOINT 等信息)
load_dotenv()

def get_remote_embedding_client(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    model_name: Optional[str] = None
) -> Optional[Callable[[List[str]], Any]]:
    """
    获取远程 API 驱动的 Embedding 客户端 (例如 OpenAI 或 SiliconFlow)。

    如果入参为空，将自动从环境变量 (EMBEDDING_API_KEY 等) 获取配置。
    
    返回:
        一个接收 List[str] 文本并返回对应 List[List[float]] 向量的函数。
        如果未配置 api_key，则返回 None，代表不启用远程客户端。
    """
    api_key = api_key or os.getenv("EMBEDDING_API_KEY", "")
    endpoint = endpoint or os.getenv("EMBEDDING_ENDPOINT", "https://api.siliconflow.cn/v1/embeddings")
    model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

    if not api_key:
        logger.warning("EMBEDDING_API_KEY 未找到。远程 Embedding 客户端将不会初始化。")
        return None

    def embed_fn(texts: List[str]) -> Any:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"input": texts, "model": model_name}
        try:
            logger.info(f"正在调用远程 Embedding API: {endpoint} (模型: {model_name}, 批量大小: {len(texts)})")
            with httpx.Client(timeout=60.0) as client:
                response = client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                # 提取并返回所有句子的向量表示
                return [item["embedding"] for item in data["data"]]
        except Exception as e:
            logger.error(f"远程 Embedding API 调用失败: {e}")
            return None

    return embed_fn

def get_local_embedding_client(model_id: Optional[str] = None) -> Optional[Callable[[List[str]], Any]]:
    """
    获取本地模型驱动的 Embedding 客户端。
    
    采用延迟加载 `sentence-transformers` 策略，避免在未安装该库的环境下报错。
    通过下载并加载模型到本地执行推理，适合对数据隐私要求高、或不希望依赖外部网络的场景。
    
    返回:
        一个接收 List[str] 并返回向量的函数。
        如果无法加载包或模型，返回 None。
    """
    model_id = model_id or os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"正在加载本地 Embedding 模型: {model_id}")
        model = SentenceTransformer(model_id)
        
        def embed_fn(texts: List[str]) -> Any:
            # 使用本地模型进行编码
            return model.encode(texts)
            
        return embed_fn
    except ImportError:
        logger.error("未安装 sentence-transformers，无法使用本地 Embedding 功能。")
        return None
    except Exception as e:
        logger.error(f"加载本地模型失败 {model_id}: {e}")
        return None

def get_default_embedding_client():
    """
    获取默认的 Embedding 客户端策略。
    
    逻辑：
    优先尝试从环境变量配置加载远程 API（成本低、无需下载重型模型）。
    如果用户没有配置远程 API KEY，则返回 None。此时调用者 (parsers)
    可根据业务逻辑决定是否降级或跳过语义切分过程。
    """
    return get_remote_embedding_client()
