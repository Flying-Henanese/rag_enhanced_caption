import os
import httpx
from typing import Optional
from loguru import logger

def get_mime_type(image_bytes: bytes) -> str:
    """
    根据图片的 Magic Numbers (文件头魔数) 动态判断 MIME Type。
    支持常见的 JPEG, PNG, GIF, WEBP。如果无法识别，默认回退到 image/jpeg。
    """
    if image_bytes.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    elif image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    elif image_bytes.startswith(b'GIF87a') or image_bytes.startswith(b'GIF89a'):
        return 'image/gif'
    elif image_bytes.startswith(b'RIFF') and image_bytes[8:12] == b'WEBP':
        return 'image/webp'
    elif image_bytes.lstrip().startswith(b'<svg') or image_bytes.lstrip().startswith(b'<?xml'):
        return 'image/svg+xml'
    else:
        logger.warning("Unrecognized image format, defaulting to image/jpeg")
        return 'image/jpeg'

async def default_vlm_call(
    user_prompt: str, 
    system_prompt: str, 
    image_base64: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    timeout: float = 60.0
) -> str:
    """
    A default, highly-versatile VLM caller that assumes an OpenAI-compatible HTTP endpoint.
    This works out-of-the-box with services like vLLM, SiliconFlow, Together AI, OpenAI, etc.
    
    If parameters are not provided, it attempts to load them from environment variables:
    - VLM_API_KEY
    - VLM_ENDPOINT (default: "http://127.0.0.1:8000/v1/chat/completions")
    - VLM_MODEL_NAME (default: "default-model")
    """
    
    api_key = api_key or os.getenv("VLM_API_KEY", "")
    endpoint = endpoint or os.getenv("VLM_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
    model_name = model_name or os.getenv("VLM_MODEL_NAME", "default-model")

    logger.info(f"Using default VLM client. Endpoint: {endpoint}, Model: {model_name}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    user_content = [{"type": "text", "text": user_prompt}]
    if image_base64:
        # 动态检测 MIME Type，避免硬编码 image/jpeg 导致的潜在兼容性或透明通道丢失问题。
        mime_type = "image/jpeg"
        if image_bytes:
            mime_type = get_mime_type(image_bytes)
        
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
        })
        messages.append({"role": "user", "content": user_content})
    else:
        # 如果没有图片，部分 OpenAI 兼容服务器更偏好 content 为字符串而不是列表
        messages.append({"role": "user", "content": user_prompt})
    
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature
    }
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"VLM Server Error {response.status_code}: {error_detail}")
                raise httpx.HTTPStatusError(
                    f"Client error '{response.status_code} {response.reason_phrase}' for url '{response.url}'\nResponse: {error_detail}",
                    request=response.request,
                    response=response
                )
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        # 已经在上面处理并抛出了带详细信息的异常
        raise
    except Exception as e:
        logger.exception(f"VLM Call failed with exception: {str(e)}")
        raise

def create_default_vlm_client(**kwargs):
    """
    Factory function to create a VLM caller with pre-configured settings.
    Useful for passing into MarkdownMultimodalProcessor.
    """
    async def configured_vlm_call(user_prompt: str, system_prompt: str, image_base64: Optional[str] = None, image_bytes: Optional[bytes] = None) -> str:
        return await default_vlm_call(
            user_prompt, 
            system_prompt, 
            image_base64,
            image_bytes,
            **kwargs
        )
    return configured_vlm_call
