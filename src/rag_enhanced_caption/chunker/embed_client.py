"""
Embedding 客户端工具模块。

提供获取本地和远程 Embedding 客户端的方法，用于语义分块过程中的文本向量化计算，以实现智能聚类和分段。
支持通过环境变量灵活配置本地 SentenceTransformer 模型或远程 SiliconFlow/OpenAI 兼容 API。
"""

from collections.abc import Callable
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from loguru import logger

# 加载环境变量 (确保配置加载的优先级)
load_dotenv(override=False)


def get_remote_embedding_client(
    api_key: str | None = None,
    endpoint: str | None = None,
    model_name: str | None = None,
) -> Callable[[list[str]], list[list[float]]] | None:
    """获取通用的 Embedding API 客户端。

    支持 OpenAI 兼容协议，适配云端服务 (SiliconFlow/OpenAI) 或本地私有化模型 (vLLM/Ollama)。

    Args:
        api_key: API 密钥。默认读取 ``EMBEDDING_API_KEY``。
        endpoint: Embedding 端点。默认读取 ``EMBEDDING_ENDPOINT``。
        model_name: 模型名称。默认读取 ``EMBEDDING_MODEL_NAME``。

    Returns:
        同步 Embedding 回调；默认云端服务缺少密钥时返回 ``None``。
    """
    default_endpoint = "https://api.siliconflow.cn/v1/embeddings"

    api_key = api_key or os.getenv("EMBEDDING_API_KEY", "")
    endpoint = endpoint or os.getenv("EMBEDDING_ENDPOINT", default_endpoint)
    model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    timeout = float(os.getenv("EMBEDDING_TIMEOUT", "60.0"))

    # 逻辑：如果使用默认云端服务则强制要求 API KEY；如果是自定义 Endpoint 则允许匿名访问
    if not api_key and endpoint == default_endpoint:
        logger.warning(
            "未配置 EMBEDDING_API_KEY，且正在使用默认服务地址。客户端将不会初始化。"
        )
        return None

    def embed_fn(texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {"input": texts, "model": model_name}
        try:
            logger.info(
                f"发送 Embedding 请求: {endpoint} (模型: {model_name}, 批量: {len(texts)})"
            )
            # trust_env=False 确保在有代理的环境下，本地/内网请求不被拦截
            with httpx.Client(timeout=timeout, trust_env=False, verify=False) as client:
                response = client.post(endpoint, headers=headers, json=payload)
                if response.status_code != 200:
                    logger.error(
                        f"Embedding API 返回错误 (状态码 {response.status_code}): {response.text}"
                    )
                    return []

                data = response.json()
                # 兼容性处理 A: OpenAI 标准结构
                if (
                    isinstance(data, dict)
                    and "data" in data
                    and isinstance(data["data"], list)
                ):
                    return [item["embedding"] for item in data["data"]]
                # 兼容性处理 B: 某些本地服务直接返回向量列表
                if isinstance(data, list):
                    return data

                logger.error(f"无法解析 API 响应格式: {data}")
                return []
        except Exception as e:
            logger.error(f"Embedding API 连接异常: {e}")
            return []

    return embed_fn


def get_local_embedding_client(
    model_id: str | None = None,
) -> Callable[[list[str]], Any] | None:
    """获取本地模型驱动的 Embedding 客户端。

    采用延迟加载 `sentence-transformers` 策略，避免在未安装该库的环境下报错。
    通过下载并加载模型到本地执行推理，适合对数据隐私要求高、或不希望依赖外部网络的场景。

    Args:
        model_id: 本地模型标识。默认读取 ``LOCAL_EMBEDDING_MODEL``。

    Returns:
        接收 ``list[str]`` 并返回向量的函数；无法加载包或模型时返回
        ``None``。
    """
    model_id = model_id or os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

    try:
        from sentence_transformers import SentenceTransformer

        logger.info(f"正在加载本地 Embedding 模型: {model_id}")
        model = SentenceTransformer(model_id)

        def embed_fn(texts: list[str]) -> Any:
            # 使用本地模型进行编码
            return model.encode(texts)

        return embed_fn
    except ImportError:
        logger.error("未安装 sentence-transformers，无法使用本地 Embedding 功能。")
        return None
    except Exception as e:
        logger.error(f"加载本地模型失败 {model_id}: {e}")
        return None


def get_default_embedding_client() -> Callable[[list[str]], list[list[float]]] | None:
    """获取默认的 Embedding 客户端策略。

    逻辑：
    优先尝试从环境变量配置加载远程 API（成本低、无需下载重型模型）。
    如果用户没有配置远程 API KEY，则返回 None。此时调用者 (parsers)
    可根据业务逻辑决定是否降级或跳过语义切分过程。

    Returns:
        远程 Embedding 回调；无法初始化时返回 ``None``。
    """
    return get_remote_embedding_client()
