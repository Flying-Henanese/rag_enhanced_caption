"""Asynchronously convert PDF pages to Markdown with SiliconFlow PaddleOCR-VL."""

import argparse
import asyncio
import base64
from collections.abc import Mapping
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
import httpx
from dotenv import load_dotenv
from loguru import logger


DEFAULT_ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_MODEL_NAME = "PaddlePaddle/PaddleOCR-VL-1.5"
DEFAULT_PROMPT = (
    "Convert this document page to clean Markdown. Preserve headings, paragraphs, "
    "tables, formulas, and figure captions. Return only Markdown."
)
_LOCATION_TOKEN_PATTERN = re.compile(r"<\|LOC_\d+\|>")
_PAGE_NUMBER_PATTERN = re.compile(r"-(\d+)\.png$")


@dataclass(frozen=True)
class PaddleOcrConfig:
    """Configuration for SiliconFlow PaddleOCR-VL requests.

    Attributes:
        api_key: SiliconFlow API key.
        endpoint: OpenAI-compatible chat completions endpoint.
        model_name: PaddleOCR model identifier.
        timeout: Per-page request timeout in seconds.
        max_concurrency: Maximum number of pages sent concurrently.
    """

    api_key: str
    endpoint: str = DEFAULT_ENDPOINT
    model_name: str = DEFAULT_MODEL_NAME
    timeout: float = 120.0
    max_concurrency: int = 2

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "PaddleOcrConfig":
        """Build configuration from ``OCR_*`` environment variables.

        Args:
            environ: Environment mapping to read. Defaults to ``os.environ``.

        Returns:
            A validated PaddleOCR configuration.

        Raises:
            ValueError: If the API key is missing or concurrency is not positive.
        """
        values = os.environ if environ is None else environ
        api_key = values.get("OCR_API_KEY", "")
        if api_key == "${VLM_API_KEY}":
            api_key = values.get("VLM_API_KEY", "")
        max_concurrency = int(values.get("OCR_MAX_CONCURRENCY", "2"))
        if not api_key:
            raise ValueError("OCR_API_KEY must be configured before using PDF OCR.")
        if max_concurrency < 1:
            raise ValueError("OCR_MAX_CONCURRENCY must be at least 1.")
        return cls(
            api_key=api_key,
            endpoint=values.get("OCR_ENDPOINT", DEFAULT_ENDPOINT),
            model_name=values.get("OCR_MODEL_NAME", DEFAULT_MODEL_NAME),
            timeout=float(values.get("OCR_TIMEOUT", "120")),
            max_concurrency=max_concurrency,
        )


def strip_location_tokens(content: str) -> str:
    """Remove PaddleOCR coordinate tokens from a Markdown response.

    Args:
        content: Raw model response content.

    Returns:
        Markdown text without inline ``<|LOC_...|>`` tokens.
    """
    return _LOCATION_TOKEN_PATTERN.sub("", content).strip()


def _page_number(page_path: Path) -> int:
    match = _PAGE_NUMBER_PATTERN.search(page_path.name)
    if match is None:
        raise ValueError(f"Could not determine page number from {page_path.name}.")
    return int(match.group(1))


def _copy_page_images(page_paths: list[Path], asset_dir: Path) -> dict[int, Path]:
    """Copy rendered pages into a stable Markdown asset directory."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[int, Path] = {}
    for page_path in page_paths:
        destination = asset_dir / page_path.name
        shutil.copy2(page_path, destination)
        assets[_page_number(page_path)] = destination
    return assets


async def _render_pdf_pages(pdf_path: Path, page_dir: Path, dpi: int) -> list[Path]:
    """Render a PDF into PNG pages with Poppler's ``pdftoppm``."""
    if shutil.which("pdftoppm") is None:
        raise RuntimeError(
            "pdftoppm is required for PaddleOCR PDF input. Install Poppler first."
        )

    page_prefix = page_dir / "page"
    process = await asyncio.create_subprocess_exec(
        "pdftoppm",
        "-png",
        "-r",
        str(dpi),
        str(pdf_path),
        str(page_prefix),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"PDF rendering failed: {stderr.decode().strip()}")

    pages = sorted(page_dir.glob("page-*.png"), key=_page_number)
    if not pages:
        raise RuntimeError(f"No pages were rendered from {pdf_path}.")
    return pages


async def _ocr_page(
    client: httpx.AsyncClient,
    page_path: Path,
    config: PaddleOcrConfig,
    semaphore: asyncio.Semaphore,
) -> tuple[int, str]:
    """Submit one rendered PDF page to PaddleOCR-VL."""
    async with semaphore:
        image_base64 = base64.b64encode(page_path.read_bytes()).decode("ascii")
        payload = {
            "model": config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": DEFAULT_PROMPT},
                    ],
                }
            ],
            "temperature": 0,
        }
        try:
            response = await client.post(
                config.endpoint,
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            logger.exception("PaddleOCR request failed for page {}.", page_path.name)
            raise

    if not isinstance(content, str):
        raise RuntimeError(f"PaddleOCR returned non-text content for {page_path.name}.")
    cleaned_content = strip_location_tokens(content)
    if not cleaned_content:
        raise RuntimeError(f"PaddleOCR returned no content for {page_path.name}.")
    return _page_number(page_path), cleaned_content


def _combine_pages(
    page_contents: list[tuple[int, str]],
    asset_dir_name: str,
    include_page_images: bool,
) -> str:
    """Combine page OCR results while preserving page boundaries."""
    return (
        "\n\n".join(
            _format_page_markdown(
                page_number, content, asset_dir_name, include_page_images
            )
            for page_number, content in sorted(page_contents)
        )
        + "\n"
    )


def _format_page_markdown(
    page_number: int,
    content: str,
    asset_dir_name: str,
    include_page_images: bool,
) -> str:
    """Format one OCR page with an optional relative image link."""
    page_header = f"<!-- OCR page {page_number} -->"
    if not include_page_images:
        return f"{page_header}\n\n{content}"
    image_name = f"page-{page_number:03d}.png"
    image_link = f"![OCR source page {page_number}]({asset_dir_name}/{image_name})"
    return f"{page_header}\n\n{image_link}\n\n{content}"


async def ocr_pdf_to_markdown(
    pdf_path: Path,
    output_dir: Path,
    config: PaddleOcrConfig | None = None,
    dpi: int = 144,
    include_page_images: bool = False,
) -> Path:
    """Render a PDF and asynchronously convert every page to Markdown.

    Args:
        pdf_path: Source PDF to OCR.
        output_dir: Directory for the generated Markdown file.
        config: Request configuration. Defaults to ``OCR_*`` environment values.
        dpi: Rendering resolution for each input page.
        include_page_images: Add relative page image links to the Markdown. The
            rendered pages are always copied into a sibling asset directory.

    Returns:
        Path to ``<pdf-stem>.md`` in ``output_dir``.

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
        ValueError: If the source is not a PDF or the DPI is invalid.
        RuntimeError: If rendering fails or OCR returns invalid content.
        httpx.HTTPError: If an OCR request fails.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")
    if dpi < 1:
        raise ValueError("DPI must be positive.")

    resolved_config = config or PaddleOcrConfig.from_environment()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pdf_path.stem}.md"
    asset_dir = output_dir / pdf_path.stem

    with tempfile.TemporaryDirectory(prefix="paddleocr-pages-") as temporary_dir:
        page_paths = await _render_pdf_pages(pdf_path, Path(temporary_dir), dpi)
        logger.info("Rendered {} PDF pages for OCR.", len(page_paths))
        _copy_page_images(page_paths, asset_dir)
        semaphore = asyncio.Semaphore(resolved_config.max_concurrency)
        timeout = httpx.Timeout(resolved_config.timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            page_contents = await asyncio.gather(
                *[
                    _ocr_page(client, page_path, resolved_config, semaphore)
                    for page_path in page_paths
                ]
            )

    output_path.write_text(
        _combine_pages(page_contents, asset_dir.name, include_page_images),
        encoding="utf-8",
    )
    logger.info("Saved PaddleOCR Markdown to {}.", output_path)
    return output_path


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the standalone PaddleOCR example command parser."""
    parser = argparse.ArgumentParser(
        description="Convert a PDF to Markdown with SiliconFlow PaddleOCR-VL."
    )
    parser.add_argument("input_pdf", type=Path, help="Source PDF file.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/ocr"), help="Output directory."
    )
    parser.add_argument("--dpi", type=int, default=144, help="Page rendering DPI.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent OCR calls.",
    )
    parser.add_argument(
        "--include-page-images",
        action="store_true",
        help="Add generated page image links to the Markdown output.",
    )
    return parser


async def _main() -> None:
    """Run the standalone OCR example."""
    root_dir = Path(__file__).resolve().parent.parent
    load_dotenv(root_dir / ".env")
    args = _build_argument_parser().parse_args()
    config = PaddleOcrConfig.from_environment()
    if args.max_concurrency is not None:
        config = replace(config, max_concurrency=args.max_concurrency)
    await ocr_pdf_to_markdown(
        args.input_pdf,
        args.output_dir,
        config,
        args.dpi,
        args.include_page_images,
    )


if __name__ == "__main__":
    asyncio.run(_main())
