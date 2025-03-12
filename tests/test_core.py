"""Tests for epub_vectorizer.core module."""

from unittest import mock

import pytest

from epub_vectorizer.config import Config
from epub_vectorizer.core import AnthropicEmbeddings, BookVectorizer, EPUBProcessor


@pytest.fixture
def mock_config():
    """Return a mock configuration."""
    return mock.MagicMock(spec=Config)


@pytest.fixture
def mock_anthropic_client():
    """Return a mock Anthropic client."""
    with mock.patch("anthropic.Anthropic") as mock_client:
        yield mock_client


@pytest.fixture
def mock_supabase_client():
    """Return a mock Supabase client."""
    with mock.patch("supabase.create_client") as mock_client:
        yield mock_client


class TestEPUBProcessor:
    """Tests for EPUBProcessor class."""

    @pytest.fixture
    def processor(self, mock_config):
        """Return an EPUBProcessor instance with mocked config."""
        mock_config.chunk_size = 1000
        mock_config.chunk_overlap = 200
        return EPUBProcessor(mock_config)

    def test_html_to_text(self):
        """Test HTML to text conversion."""
        html = "<html><body><h1>Title</h1><p>This is a paragraph.</p></body></html>"
        text = EPUBProcessor.html_to_text(html)
        assert text == "Title This is a paragraph."

    def test_split_text_into_chunks(self, processor):
        """Test text splitting into chunks."""
        # Create a text with 2500 characters
        text = "a" * 2500
        chunks = processor.split_text_into_chunks(text)

        # We expect 3 chunks with sizes 1000, 1000, and 500
        assert len(chunks) == 3
        assert len(chunks[0]) == 1000
        assert len(chunks[1]) == 1000
        assert len(chunks[2]) == 500

        # Check overlap
        assert chunks[0][-200:] == chunks[1][:200]
        assert chunks[1][-200:] == chunks[2][:200]


class TestAnthropicEmbeddings:
    """Tests for AnthropicEmbeddings class."""

    @pytest.fixture
    def embeddings_client(self, mock_config, mock_anthropic_client):
        """Return an AnthropicEmbeddings instance with mocked dependencies."""
        mock_config.anthropic_api_key = "test_key"
        mock_config.embedding_model = "claude-3-7-sonnet-20250219"
        return AnthropicEmbeddings(mock_config)

    def test_get_embedding(self, embeddings_client, mock_anthropic_client):
        """Test getting embeddings from Anthropic API."""
        # Set up mock response
        mock_instance = mock_anthropic_client.return_value
        mock_instance.embeddings.create.return_value.embedding = [0.1, 0.2, 0.3]

        # Call method to test
        result = embeddings_client.get_embedding("test text")

        # Verify results
        assert result == [0.1, 0.2, 0.3]
        mock_instance.embeddings.create.assert_called_once_with(
            model="claude-3-7-sonnet-20250219", input="test text"
        )

        # Test caching - second call should use cached result
        mock_instance.embeddings.create.reset_mock()
        result2 = embeddings_client.get_embedding("test text")
        assert result2 == [0.1, 0.2, 0.3]
        mock_instance.embeddings.create.assert_not_called()


class TestBookVectorizer:
    """Tests for BookVectorizer class."""

    @pytest.fixture
    def vectorizer(self):
        """Return a BookVectorizer with mocked dependencies."""
        with mock.patch("epub_vectorizer.core.Config"), \
             mock.patch("epub_vectorizer.core.EPUBProcessor"), \
             mock.patch("epub_vectorizer.core.AnthropicEmbeddings"), \
             mock.patch("epub_vectorizer.core.SupabaseVectorStore"):
            vectorizer = BookVectorizer()
            yield vectorizer

    def test_vectorize_epub(self, vectorizer):
        """Test vectorizing an EPUB file."""
        # Set up mocks
        text = "This is the book text"
        metadata = {"title": "Test Book", "author": "Test Author"}
        chunks = ["This is", "the book", "text"]

        vectorizer.epub_processor.extract_text_from_epub.return_value = (text, metadata)
        vectorizer.epub_processor.split_text_into_chunks.return_value = chunks

        # Mock uuid.uuid4 to return a predictable value
        with mock.patch("uuid.uuid4", return_value=mock.MagicMock(return_value="test-uuid")):
            book_id = vectorizer.vectorize_epub("test.epub")

        # Verify calls
        vectorizer.epub_processor.extract_text_from_epub.assert_called_once()
        vectorizer.epub_processor.split_text_into_chunks.assert_called_once_with(text)
        vectorizer.vector_store.store_chunks.assert_called_once()

        # Verify result
        assert isinstance(book_id, str)

    def test_search(self, vectorizer):
        """Test searching for similar text."""
        # Set up mock
        expected_results = [{"id": "1", "content": "test content"}]
        vectorizer.vector_store.search_similar_text.return_value = expected_results

        # Call method to test
        results = vectorizer.search("test query", book_id="test-id", limit=10)

        # Verify calls and results
        vectorizer.vector_store.search_similar_text.assert_called_once_with(
            "test query", vectorizer.embedding_client, "test-id", 10
        )
        assert results == expected_results

    def test_get_book_info(self, vectorizer):
        """Test getting book information."""
        # Set up mock
        expected_info = {"book_title": "Test Book", "book_author": "Test Author"}
        vectorizer.vector_store.get_book_info.return_value = expected_info

        # Call method to test
        info = vectorizer.get_book_info("test-id")

        # Verify calls and results
        vectorizer.vector_store.get_book_info.assert_called_once_with("test-id")
        assert info == expected_info
