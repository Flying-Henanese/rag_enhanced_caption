import urllib.parse
import base64
import asyncio
import httpx
from pathlib import Path
from loguru import logger


def create_image_resolver(base_dir: str | Path = "."):
    """
    创建一个支持 本地/网络/Base64 的异步图像解析器。

    Args:
        base_dir: Markdown 文件所在的物理目录。用于拼接相对路径。
                  例如，如果 MD 文件在 /doc/readme.md，这里应传入 "/doc"

    Returns:
        一个接收 image_url 并返回 bytes 的异步回调函数。
    """
    base_path = Path(base_dir).resolve()

    async def resolver(image_url: str) -> bytes | None:
        try:
            # 场景 1: 处理网络图片 (HTTP/HTTPS)
            parsed_url = urllib.parse.urlparse(image_url)
            if parsed_url.scheme in ("http", "https"):
                logger.info(f"Downloading remote image: {image_url}")
                # 伪装 User-Agent，防止部分图床防盗链拦截
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    response = await client.get(
                        image_url,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                        },
                        timeout=15.0,
                    )
                    response.raise_for_status()
                    return response.content

            # 场景 2: 处理 Base64 内联图片数据 (Data URI)
            if image_url.startswith("data:image"):
                logger.info("Decoding inline Base64 image...")
                # 格式通常是: data:image/png;base64,iVBORw0KGgo...
                _, encoded = image_url.split(",", 1)
                return base64.b64decode(encoded)

            # 场景 3: 处理本地文件路径
            # 解码 URL 编码的字符 (例如: "my%20image.png" -> "my image.png")
            local_path_str = urllib.parse.unquote(image_url)
            img_path = Path(local_path_str)

            # 如果是相对路径，则基于传入的 base_dir 进行拼接
            if not img_path.is_absolute():
                img_path = base_path / img_path

            # 规范化路径 (消除 ../ 等符号)
            img_path = img_path.resolve()

            if img_path.exists() and img_path.is_file():
                logger.debug(f"Reading local image: {img_path}")
                # 使用 to_thread 防止读取大文件阻塞异步事件循环
                return await asyncio.to_thread(img_path.read_bytes)
            else:
                logger.warning(f"Local image file not found: {img_path}")
                return None

        except Exception as e:
            logger.error(f"Failed to resolve image {image_url}: {str(e)}")
            return None

    return resolver
