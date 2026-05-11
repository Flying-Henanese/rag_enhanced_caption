from .enhancer.context_extractor import MarkdownContextExtractor
from .enhancer.processor import MarkdownMultimodalProcessor
from .enhancer.image_utils import create_image_resolver
from .enhancer.vlm_client import create_default_vlm_client
from .chunker.dispatcher import chunk_markdown as semantic_chunk_with_metadata
from .chunker.parsers.semantic import chunk_markdown as semantic_chunk_raw

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

__all__ = [
    "MarkdownContextExtractor",
    "MarkdownMultimodalProcessor",
    "create_image_resolver",
    "create_default_vlm_client",
    "semantic_chunk_with_metadata",
    "semantic_chunk_raw",
]

try:
    __version__ = version("rag-enhanced-caption")
except PackageNotFoundError:
    # Fallback for source-only execution before package metadata is installed.
    __version__ = "0.1.2"
