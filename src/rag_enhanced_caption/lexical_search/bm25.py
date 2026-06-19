"""Small in-memory BM25 backend for local lexical retrieval."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, Sequence

from .schema import SearchableObject

_TOKEN_RE = re.compile(
    r"[\w.+-]+@[\w.-]+\.[a-z]{2,}|"
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|"
    r"[a-z][a-z0-9_.:/+-]*|"
    r"\d+(?:\.\d+)*|"
    r"[\u3400-\u9fff]+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SparseSearchResult:
    """A lexical match mapped back to an owning retrieval node."""

    owner_node_id: str
    object_id: str
    score: float


class SparseSearchBackend(Protocol):
    """Search boundary for local BM25 and future remote backends."""

    def search(self, query: str, top_k: int = 10) -> list[SparseSearchResult]:
        """Return ranked owner nodes for a lexical query."""


def tokenize_lexical_text(text: str) -> list[str]:
    """Tokenize identifiers and Chinese text without model dependencies."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(normalized):
        token = match.group(0)
        if re.fullmatch(r"[\u3400-\u9fff]+", token):
            tokens.extend(token)
            tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
        else:
            tokens.append(token)
    return tokens


class InMemoryBM25Backend:
    """BM25 Okapi index built from persisted searchable objects."""

    def __init__(
        self,
        objects: Sequence[SearchableObject],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._objects = list(objects)
        self._k1 = k1
        self._b = b
        self._term_frequencies = [
            Counter(tokenize_lexical_text(obj.searchable_text)) for obj in self._objects
        ]
        self._document_lengths = [sum(freq.values()) for freq in self._term_frequencies]
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            document_frequency.update(frequencies.keys())
        document_count = len(self._objects)
        self._idf = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[SparseSearchResult]:
        """Rank objects with BM25 and deduplicate them by owner node."""
        if top_k <= 0 or not self._objects:
            return []
        query_terms = tokenize_lexical_text(query)
        if not query_terms:
            return []

        best_by_owner: dict[str, SparseSearchResult] = {}
        average_length = self._average_document_length or 1.0
        for obj, frequencies, document_length in zip(
            self._objects, self._term_frequencies, self._document_lengths
        ):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * document_length / average_length
                )
                score += self._idf.get(term, 0.0) * (
                    frequency * (self._k1 + 1) / denominator
                )
            if score <= 0:
                continue
            result = SparseSearchResult(obj.owner_node_id, obj.id, score)
            previous = best_by_owner.get(obj.owner_node_id)
            if previous is None or result.score > previous.score:
                best_by_owner[obj.owner_node_id] = result

        return sorted(
            best_by_owner.values(), key=lambda item: item.score, reverse=True
        )[:top_k]
