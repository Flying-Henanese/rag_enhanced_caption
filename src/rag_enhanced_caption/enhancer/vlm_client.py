from collections.abc import Awaitable, Callable
import os
from typing import Any

import httpx
from loguru import logger


VlmCall = Callable[[str, str, str | None, bytes | None], Awaitable[str]]


def get_mime_type(image_bytes: bytes) -> str:
    """根据图片的 Magic Numbers 动态判断 MIME 类型。

    支持常见的 JPEG, PNG, GIF, WEBP。如果无法识别，默认回退到 image/jpeg。

    Args:
        image_bytes: 图片的原始字节。

    Returns:
        检测出的 MIME 类型。
    """
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    elif image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    elif image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return "image/gif"
    elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    elif image_bytes.lstrip().startswith(b"<svg") or image_bytes.lstrip().startswith(
        b"<?xml"
    ):
        return "image/svg+xml"
    else:
        logger.warning("Unrecognized image format, defaulting to image/jpeg")
        return "image/jpeg"


async def default_vlm_call(
    user_prompt: str,
    system_prompt: str,
    image_base64: str | None = None,
    image_bytes: bytes | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
    model_name: str | None = None,
    temperature: float = 0.2,
    timeout: float = 60.0,
) -> str:
    """Call an OpenAI-compatible VLM endpoint.

    This works out-of-the-box with services like vLLM, SiliconFlow, Together AI, OpenAI, etc.

    If parameters are not provided, it attempts to load them from environment variables:
    - VLM_API_KEY
    - VLM_ENDPOINT (default: "http://127.0.0.1:8000/v1/chat/completions")
    - VLM_MODEL_NAME (default: "default-model")

    Args:
        user_prompt: User message sent to the model.
        system_prompt: System instruction sent to the model.
        image_base64: Optional Base64-encoded image payload.
        image_bytes: Optional raw image bytes used for MIME detection.
        api_key: API key. Defaults to ``VLM_API_KEY``.
        endpoint: Chat-completions endpoint. Defaults to ``VLM_ENDPOINT``.
        model_name: Model identifier. Defaults to ``VLM_MODEL_NAME``.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.

    Returns:
        Text content from the first response choice.

    Raises:
        httpx.HTTPStatusError: If the endpoint returns a non-success status.
        httpx.HTTPError: If the request fails.
        KeyError: If the response does not contain the expected choice payload.
    """

    api_key = api_key or os.getenv("VLM_API_KEY", "")
    endpoint = endpoint or os.getenv(
        "VLM_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions"
    )
    model_name = model_name or os.getenv("VLM_MODEL_NAME", "default-model")

    logger.info(f"Using default VLM client. Endpoint: {endpoint}, Model: {model_name}")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    messages = [{"role": "system", "content": system_prompt}]

    user_content = [{"type": "text", "text": user_prompt}]
    if image_base64:
        # 动态检测 MIME Type，避免硬编码 image/jpeg 导致的潜在兼容性或透明通道丢失问题。
        mime_type = "image/jpeg"
        if image_bytes:
            mime_type = get_mime_type(image_bytes)

        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
            }
        )
        messages.append({"role": "user", "content": user_content})
    else:
        # 如果没有图片，部分 OpenAI 兼容服务器更偏好 content 为字符串而不是列表
        messages.append({"role": "user", "content": user_prompt})

    payload = {"model": model_name, "messages": messages, "temperature": temperature}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"VLM Server Error {response.status_code}: {error_detail}")
                raise httpx.HTTPStatusError(
                    f"Client error '{response.status_code} {response.reason_phrase}' for url '{response.url}'\nResponse: {error_detail}",
                    request=response.request,
                    response=response,
                )

            result = response.json()
            return result["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError:
        # 已经在上面处理并抛出了带详细信息的异常
        raise
    except Exception as e:
        logger.exception(f"VLM Call failed with exception: {str(e)}")
        raise


def create_default_vlm_client(**kwargs: Any) -> VlmCall:
    """Create a VLM caller with preconfigured settings.

    Useful for passing into MarkdownMultimodalProcessor.

    Args:
        **kwargs: Keyword arguments forwarded to ``default_vlm_call``.

    Returns:
        An asynchronous VLM callback with the processor-compatible signature.
    """

    async def configured_vlm_call(
        user_prompt: str,
        system_prompt: str,
        image_base64: str | None = None,
        image_bytes: bytes | None = None,
    ) -> str:
        return await default_vlm_call(
            user_prompt, system_prompt, image_base64, image_bytes, **kwargs
        )

    return configured_vlm_call
