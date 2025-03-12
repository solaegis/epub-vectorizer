"""EPUB Vectorizer - Process EPUB books and store vectorized content in Supabase."""

from epub_vectorizer.__about__ import __version__
from epub_vectorizer.core import (
    AnthropicEmbeddings,
    BookVectorizer,
    EPUBProcessor,
    SupabaseVectorStore,
)

__all__ = [
    "__version__",
    "BookVectorizer",
    "AnthropicEmbeddings",
    "EPUBProcessor",
    "SupabaseVectorStore",
]
