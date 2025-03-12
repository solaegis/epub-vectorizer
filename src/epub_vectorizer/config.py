"""Configuration settings for EPUB Vectorizer."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration settings loaded from environment variables."""

    supabase_url: str
    supabase_key: str
    anthropic_api_key: str
    google_books_api_key: Optional[str]
    embedding_model: str
    vector_dimension: int
    table_name: str
    chunk_size: int
    chunk_overlap: int

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables.

        Returns:
            Config: Configuration object with settings loaded from environment

        Raises:
            ValueError: If required environment variables are missing
        """
        load_dotenv()

        # Required settings
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        # Validate required settings
        missing = []
        if not supabase_url:
            missing.append("SUPABASE_URL")
        if not supabase_key:
            missing.append("SUPABASE_KEY")
        if not anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")

        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        # Optional settings with defaults
        google_books_api_key = os.getenv("GOOGLE_BOOKS_API_KEY")
        embedding_model = os.getenv("EMBEDDING_MODEL", "claude-3-7-sonnet-20250219")
        vector_dimension = int(os.getenv("VECTOR_DIMENSION", "3072"))
        table_name = os.getenv("TABLE_NAME", "book_chunks")
        chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

        return cls(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            anthropic_api_key=anthropic_api_key,
            google_books_api_key=google_books_api_key,
            embedding_model=embedding_model,
            vector_dimension=vector_dimension,
            table_name=table_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
