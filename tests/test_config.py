"""Tests for epub_vectorizer.config module."""

import os
from unittest import mock

import pytest

from epub_vectorizer.config import Config


class TestConfig:
    """Tests for Config class."""

    def test_from_env_with_required_variables(self):
        """Test loading config from environment with required variables."""
        env_vars = {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_KEY": "test_key",
            "ANTHROPIC_API_KEY": "test_anthropic_key",
        }

        with mock.patch.dict(os.environ, env_vars):
            config = Config.from_env()

            assert config.supabase_url == "https://test.supabase.co"
            assert config.supabase_key == "test_key"
            assert config.anthropic_api_key == "test_anthropic_key"
            assert config.embedding_model == "claude-3-7-sonnet-20250219"  # default
            assert config.vector_dimension == 3072  # default
            assert config.table_name == "book_chunks"  # default
            assert config.chunk_size == 1000  # default
            assert config.chunk_overlap == 200  # default

    def test_from_env_with_all_variables(self):
        """Test loading config from environment with all variables."""
        env_vars = {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_KEY": "test_key",
            "ANTHROPIC_API_KEY": "test_anthropic_key",
            "GOOGLE_BOOKS_API_KEY": "test_google_key",
            "EMBEDDING_MODEL": "custom-model",
            "VECTOR_DIMENSION": "1536",
            "TABLE_NAME": "custom_table",
            "CHUNK_SIZE": "500",
            "CHUNK_OVERLAP": "100",
        }

        with mock.patch.dict(os.environ, env_vars):
            config = Config.from_env()

            assert config.supabase_url == "https://test.supabase.co"
            assert config.supabase_key == "test_key"
            assert config.anthropic_api_key == "test_anthropic_key"
            assert config.google_books_api_key == "test_google_key"
            assert config.embedding_model == "custom-model"
            assert config.vector_dimension == 1536
            assert config.table_name == "custom_table"
            assert config.chunk_size == 500
            assert config.chunk_overlap == 100

    def test_from_env_missing_required_variables(self):
        """Test loading config with missing required variables."""
        # Empty environment
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                Config.from_env()

            # Check that it mentions all the missing variables
            error_msg = str(exc_info.value)
            assert "SUPABASE_URL" in error_msg
            assert "SUPABASE_KEY" in error_msg
            assert "ANTHROPIC_API_KEY" in error_msg

        # Partial environment
        with mock.patch.dict(os.environ, {"SUPABASE_URL": "test"}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                Config.from_env()

            # Check that it mentions only the missing variables
            error_msg = str(exc_info.value)
            assert "SUPABASE_URL" not in error_msg
            assert "SUPABASE_KEY" in error_msg
            assert "ANTHROPIC_API_KEY" in error_msg
