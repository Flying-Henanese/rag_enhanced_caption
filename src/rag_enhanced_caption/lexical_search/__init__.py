"""Backend-neutral lexical search primitives."""

from .builder import build_searchable_objects
from .schema import SearchableObject

__all__ = ["SearchableObject", "build_searchable_objects"]
