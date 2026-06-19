"""Deterministic field extraction for lexical search."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from bs4 import BeautifulSoup

from rag_enhanced_caption.chunker.utils.table_utils import html_table_to_key_value

_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_DATE_RE = re.compile(
    r"(?<!\d)(?:\d{4}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01]))(?!\d)"
)
_TABLE_TYPES = {"Table", "Table KV", "html_block"}


@dataclass(frozen=True)
class ExtractedSearchField:
    """A lexical field extracted before it receives a stable ID."""

    field_type: str
    text: str
    metadata: dict[str, Any]


class SearchFieldExtractor(Protocol):
    """Interface for pluggable deterministic search-field extraction."""

    def extract(self, chunk: dict[str, Any]) -> list[ExtractedSearchField]:
        """Extract high-value lexical fields from a chunk."""


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", stripped)]


def _is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _markdown_table_fields(content: str) -> list[ExtractedSearchField]:
    rows = [_split_markdown_row(line) for line in content.splitlines() if "|" in line]
    if len(rows) < 2 or not _is_separator_row(rows[1]):
        return []

    headers = rows[0]
    fields = [
        ExtractedSearchField(
            field_type="table_header",
            text=" ".join(header for header in headers if header),
            metadata={},
        )
    ]
    for row_index, values in enumerate(rows[2:]):
        parts = [
            f"{header}：{value}"
            for header, value in zip(headers, values)
            if header and value
        ]
        if parts:
            fields.append(
                ExtractedSearchField(
                    field_type="table_row",
                    text="；".join(parts),
                    metadata={"row_index": row_index},
                )
            )
    return fields


def _html_table_fields(content: str) -> list[ExtractedSearchField]:
    if "<table" not in content.lower():
        return []
    soup = BeautifulSoup(content, "html.parser")
    first_row = soup.find("tr")
    fields: list[ExtractedSearchField] = []
    if first_row is not None:
        headers = [
            cell.get_text(" ", strip=True) for cell in first_row.find_all(["th", "td"])
        ]
        header_text = " ".join(header for header in headers if header)
        if header_text:
            fields.append(ExtractedSearchField("table_header", header_text, {}))
    for row_index, text in enumerate(html_table_to_key_value(content)):
        fields.append(ExtractedSearchField("table_row", text, {"row_index": row_index}))
    return fields


class DeterministicSearchFieldExtractor:
    """Extract tables and fixed-format values without an LLM."""

    def extract(self, chunk: dict[str, Any]) -> list[ExtractedSearchField]:
        """Extract high-value fields from one enriched chunk."""
        content = str(chunk.get("content") or chunk.get("full_content") or "")
        metadata = chunk.get("metadata") or {}
        element_type = str(
            metadata.get("element_type") or chunk.get("element_type") or "text"
        )

        fields: list[ExtractedSearchField] = []
        if element_type in _TABLE_TYPES:
            fields.extend(_html_table_fields(content))
            if not fields:
                fields.extend(_markdown_table_fields(content))
            if not fields and content.strip():
                fields.append(ExtractedSearchField("table_row", content.strip(), {}))

        fields.extend(
            ExtractedSearchField("date", match.group(0), {})
            for match in _DATE_RE.finditer(content)
        )
        fields.extend(
            ExtractedSearchField("email", match.group(0), {})
            for match in _EMAIL_RE.finditer(content)
        )
        return fields
