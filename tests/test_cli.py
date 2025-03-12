"""Tests for epub_vectorizer.cli module."""

import io
import sys
from unittest import mock

import pytest

from epub_vectorizer.cli import display_book_metadata, main


@pytest.fixture
def mock_vectorizer():
    """Create a mock BookVectorizer."""
    with mock.patch("epub_vectorizer.cli.BookVectorizer") as mock_vectorizer_cls:
        mock_instance = mock_vectorizer_cls.return_value
        yield mock_instance


@pytest.fixture
def mock_stdout():
    """Redirect stdout to a StringIO object."""
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    yield sys.stdout
    sys.stdout = original_stdout


class TestCLI:
    """Tests for CLI module."""

    def test_display_book_metadata(self, mock_stdout):
        """Test displaying book metadata."""
        metadata = {
            "book_title": "Test Book",
            "book_author": "Test Author",
            "publisher": "Test Publisher",
            "publication_date": "2023-01-01",
            "isbn": "1234567890123",
            "genre": ["Fiction", "Thriller"],
            "series": "Test Series",
            "series_position": 1,
            "language": "en",
            "page_count": 300,
            "cover_url": "http://example.com/cover.jpg",
            "description": "This is a test book.",
            "total_chunks": 50,
        }

        display_book_metadata(metadata)

        output = mock_stdout.getvalue()
        assert "Test Book" in output
        assert "Test Author" in output
        assert "Test Publisher" in output
        assert "2023-01-01" in output
        assert "1234567890123" in output
        assert "Fiction, Thriller" in output
        assert "Test Series #1" in output
        assert "300" in output

    def test_command_vectorize(self, mock_vectorizer, tmp_path):
        """Test vectorize command."""
        # Create a test file
        test_file = tmp_path / "test.epub"
        test_file.touch()

        # Set up mocks
        mock_vectorizer.vectorize_epub.return_value = "test-book-id"
        mock_vectorizer.get_book_info.return_value = {
            "book_title": "Test Book",
            "book_author": "Test Author",
        }

        # Call CLI
        result = main(["vectorize", str(test_file)])

        # Verify
        assert result == 0
        mock_vectorizer.vectorize_epub.assert_called_once_with(test_file)
        mock_vectorizer.get_book_info.assert_called_once_with("test-book-id")

    def test_command_vectorize_nonexistent_file(self):
        """Test vectorize command with nonexistent file."""
        result = main(["vectorize", "/nonexistent/path.epub"])
        assert result == 1

    def test_command_search(self, mock_vectorizer):
        """Test search command."""
        # Set up mocks
        mock_vectorizer.search.return_value = [
            {
                "book_id": "test-book-id",
                "book_title": "Test Book",
                "book_author": "Test Author",
                "chunk_index": 1,
                "content": "This is test content.",
            }
        ]

        # Call CLI
        result = main(["search", "test query"])

        # Verify
        assert result == 0
        mock_vectorizer.search.assert_called_once_with("test query", None, 5)

    def test_command_search_with_options(self, mock_vectorizer):
        """Test search command with options."""
        # Set up mocks
        mock_vectorizer.search.return_value = [
            {
                "book_id": "test-book-id",
                "book_title": "Test Book",
                "book_author": "Test Author",
                "chunk_index": 1,
                "content": "This is test content.",
            }
        ]

        # Call CLI
        result = main([
            "search",
            "test query",
            "--book-id", "test-book-id",
            "--limit", "10",
            "--metadata-only"
        ])

        # Verify
        assert result == 0
        mock_vectorizer.search.assert_called_once_with("test query", "test-book-id", 10)

    def test_command_info(self, mock_vectorizer):
        """Test info command."""
        # Set up mocks
        mock_vectorizer.get_book_info.return_value = {
            "book_title": "Test Book",
            "book_author": "Test Author",
        }

        # Call CLI
        result = main(["info", "test-book-id"])

        # Verify
        assert result == 0
        mock_vectorizer.get_book_info.assert_called_once_with("test-book-id")

    def test_command_info_no_book(self, mock_vectorizer):
        """Test info command with nonexistent book."""
        # Set up mocks
        mock_vectorizer.get_book_info.return_value = {}

        # Call CLI
        result = main(["info", "nonexistent-id"])

        # Verify
        assert result == 0
        mock_vectorizer.get_book_info.assert_called_once_with("nonexistent-id")

    def test_no_command(self):
        """Test CLI with no command."""
        result = main([])
        assert result == 1

    def test_unknown_command(self):
        """Test CLI with unknown command."""
        result = main(["unknown"])
        assert result == 1

    def test_config_error(self):
        """Test configuration error handling."""
        with mock.patch("epub_vectorizer.cli.Config.from_env", side_effect=ValueError("Test error")):
            result = main(["search", "test"])
            assert result == 1
