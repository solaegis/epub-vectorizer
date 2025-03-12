"""Core components of the EPUB Vectorizer."""

import json
import logging
import re
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote

import anthropic
import ebooklib
import requests
from bs4 import BeautifulSoup
from ebooklib import epub
from supabase import create_client

from epub_vectorizer.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EPUBProcessor:
    """Process EPUB files to extract text and metadata."""

    def __init__(self, config: Config) -> None:
        """Initialize the EPUB processor.

        Args:
            config: Application configuration
        """
        self.config = config
        self.google_books_api_key = config.google_books_api_key

    @staticmethod
    def html_to_text(html_content: str) -> str:
        """Convert HTML content to plain text.

        Args:
            html_content: HTML content to convert

        Returns:
            Plain text extracted from HTML
        """
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        # Clean up extra whitespaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_epub_metadata(self, book: ebooklib.epub.EpubBook) -> Dict[str, Any]:
        """Extract all available metadata from an EPUB book.

        Args:
            book: EPUB book object

        Returns:
            Dictionary of metadata extracted from the book
        """
        metadata: Dict[str, Any] = {}

        # Extract Dublin Core metadata (common in EPUBs)
        dc_fields = [
            ("title", "title"),
            ("creator", "author"),
            ("publisher", "publisher"),
            ("date", "publication_date"),
            ("identifier", "isbn"),  # May include ISBN, but could be other identifiers
            ("language", "language"),
            ("description", "description"),
            ("subject", "genre"),  # Subject often contains genre information
        ]

        for dc_field, meta_field in dc_fields:
            values = book.get_metadata("DC", dc_field)
            if values:
                # For most fields, just take the first value
                if meta_field != "genre":
                    if meta_field == "isbn":
                        # Check if it's actually an ISBN
                        identifier = values[0][0]
                        if re.search(
                            r"(?:ISBN[- ])?((?:978[- ])?[0-9][- 0-9]{10}[- 0-9Xx])",
                            identifier,
                        ):
                            metadata[meta_field] = identifier
                    else:
                        metadata[meta_field] = values[0][0]
                # For genre/subject, collect all values
                else:
                    metadata[meta_field] = [v[0] for v in values]

        # Look for series information in calibre metadata or opf:series
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content().decode("utf-8", errors="replace")
                if "calibre:series" in content or "opf:series" in content:
                    soup = BeautifulSoup(content, "html.parser")
                    series = soup.find("meta", {"name": "calibre:series"}) or soup.find(
                        "meta", {"name": "opf:series"}
                    )
                    if series and series.get("content"):
                        metadata["series"] = series.get("content")

                    series_index = soup.find(
                        "meta", {"name": "calibre:series_index"}
                    ) or soup.find("meta", {"name": "opf:series_index"})
                    if series_index and series_index.get("content"):
                        try:
                            metadata["series_position"] = float(
                                series_index.get("content")
                            )
                        except ValueError:
                            metadata["series_position"] = 0

            # Try to extract cover image URL
            if item.get_type() == ebooklib.ITEM_COVER:
                # This is simplistic - in a real-world scenario, you'd save the cover image
                # to your server and generate a proper URL
                book_title = metadata.get("title", "unknown")
                metadata["cover_url"] = f"internal://covers/{book_title}_cover"

        # If we have an identifier but couldn't detect ISBN, try harder to find it
        if "isbn" not in metadata:
            for item in book.get_metadata("DC", "identifier"):
                identifier = item[0]
                # Look for ISBN pattern
                isbn_match = re.search(
                    r"(?:ISBN[- ])?((?:978[- ])?[0-9][- 0-9]{10}[- 0-9Xx])",
                    identifier,
                )
                if isbn_match:
                    metadata["isbn"] = isbn_match.group(1)
                    break

        return metadata

    def fetch_metadata_from_google_books(
        self, title: str, author: str, isbn: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch book metadata from Google Books API.

        Args:
            title: Book title
            author: Book author
            isbn: Book ISBN if available

        Returns:
            Dictionary of metadata from Google Books
        """
        if not self.google_books_api_key:
            logger.warning("GOOGLE_BOOKS_API_KEY not set. Skipping metadata enrichment.")
            return {}

        try:
            # Create search query
            query_parts = []
            if title:
                query_parts.append(f"intitle:{quote(title)}")
            if author:
                query_parts.append(f"inauthor:{quote(author)}")
            if isbn:
                # Clean ISBN to remove dashes, etc.
                clean_isbn = re.sub(r"[^0-9Xx]", "", isbn)
                if clean_isbn:
                    query_parts.append(f"isbn:{clean_isbn}")

            if not query_parts:
                logger.warning("Insufficient metadata for Google Books lookup.")
                return {}

            # Construct the API URL
            query = "+".join(query_parts)
            url = f"https://www.googleapis.com/books/v1/volumes?q={query}&key={self.google_books_api_key}"

            # Make the request
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if "items" in data and len(data["items"]) > 0:
                    # Get the first (most relevant) book result
                    book_data = data["items"][0]["volumeInfo"]

                    # Extract the relevant metadata
                    metadata = {
                        "title": book_data.get("title", title),
                        "author": book_data.get("authors", [author])[0]
                        if book_data.get("authors")
                        else author,
                        "publisher": book_data.get("publisher", ""),
                        "publication_date": book_data.get("publishedDate", ""),
                        "isbn": None,
                        "genre": book_data.get("categories", []),
                        "description": book_data.get("description", ""),
                        "cover_url": book_data.get("imageLinks", {}).get("thumbnail", ""),
                        "page_count": book_data.get("pageCount", 0),
                        "language": book_data.get("language", ""),
                        # Google Books doesn't specifically track series info, but sometimes it's in the subtitle
                        "series": "",
                        "series_position": 0,
                    }

                    # Extract ISBN from industry identifiers
                    identifiers = book_data.get("industryIdentifiers", [])
                    for identifier in identifiers:
                        if identifier.get("type") in ("ISBN_13", "ISBN_10"):
                            metadata["isbn"] = identifier.get("identifier", "")
                            break

                    # Try to extract series info from subtitle or title
                    subtitle = book_data.get("subtitle", "")
                    if subtitle and ("series" in subtitle.lower() or "book" in subtitle.lower()):
                        # Try to parse series info like "Book 1 of X Series"
                        series_match = re.search(r"(book|volume|part)\s+(\d+)", subtitle.lower())
                        if series_match:
                            # Look for series name after "of" or in the whole subtitle
                            series_name_match = re.search(
                                r"of\s+(the\s+)?(.*?)(\s+series)?$", subtitle.lower()
                            )
                            if series_name_match:
                                metadata["series"] = series_name_match.group(2).strip().title()
                            else:
                                metadata["series"] = subtitle.split("Book")[0].strip()
                            metadata["series_position"] = int(series_match.group(2))

                    logger.info(f"Metadata enriched from Google Books for: {title} by {author}")
                    return metadata
                else:
                    logger.info(f"No matching books found in Google Books for: {title} by {author}")
                    return {}
            else:
                logger.error(
                    f"Error fetching metadata from Google Books: {response.status_code} - {response.text}"
                )
                return {}

        except Exception as e:
            logger.exception(f"Exception while fetching metadata from Google Books: {e}")
            return {}

    def extract_text_from_epub(
        self, epub_path: Union[str, Path]
    ) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from an EPUB file.

        Args:
            epub_path: Path to the EPUB file

        Returns:
            Tuple containing (extracted text, metadata dictionary)

        Raises:
            FileNotFoundError: If the EPUB file does not exist
        """
        # Ensure epub_path is a Path object
        epub_path = Path(epub_path)
        if not epub_path.exists():
            raise FileNotFoundError(f"EPUB file not found: {epub_path}")

        book = epub.read_epub(str(epub_path))

        # Extract basic metadata from EPUB
        epub_metadata = self.extract_epub_metadata(book)

        # Get essential metadata (title, author, and ISBN) for Google Books lookup
        title = epub_metadata.get("title", "Unknown")
        author = epub_metadata.get("author", "Unknown")
        isbn = epub_metadata.get("isbn", "")
        logger.info(f"Processing book: {title} by {author}")

        # Fetch additional metadata from Google Books
        google_metadata = self.fetch_metadata_from_google_books(title, author, isbn)

        # Merge metadata, with EPUB data taking precedence over Google Books data
        # (only fill in missing fields from Google Books)
        merged_metadata = {**epub_metadata}
        for key, value in google_metadata.items():
            if key not in merged_metadata or not merged_metadata[key]:
                merged_metadata[key] = value

        # Ensure title and author are always present
        if "title" not in merged_metadata or not merged_metadata["title"]:
            merged_metadata["title"] = title
        if "author" not in merged_metadata or not merged_metadata["author"]:
            merged_metadata["author"] = author

        # Extract text from all items
        all_text = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content().decode("utf-8", errors="replace")
                text = self.html_to_text(content)
                all_text.append(text)

        return " ".join(all_text), merged_metadata

    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks.

        Args:
            text: Text to split into chunks

        Returns:
            List of text chunks
        """
        chunks = []
        for i in range(0, len(text), self.config.chunk_size - self.config.chunk_overlap):
            chunk = text[i : i + self.config.chunk_size]
            if len(chunk) > 20:  # Only keep chunks with meaningful content
                chunks.append(chunk)
        return chunks


class AnthropicEmbeddings:
    """Generate embeddings using Anthropic Claude API."""

    def __init__(self, config: Config) -> None:
        """Initialize the Anthropic embeddings client.

        Args:
            config: Application configuration
        """
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self.model = config.embedding_model

    @lru_cache(maxsize=100)
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text using Anthropic API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception: If embedding generation fails after multiple retries
        """
        # Implement exponential backoff for rate limiting
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Truncate text if needed (Claude has input limits)
                truncated_text = text[:25000]  # Appropriate truncation for Claude

                response = self.client.embeddings.create(
                    model=self.model, input=truncated_text
                )
                return response.embedding
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.exception(f"Failed to get embedding after {max_retries} attempts: {e}")
                    raise
                wait_time = 2**attempt  # Exponential backoff
                logger.warning(f"API error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)


class SupabaseVectorStore:
    """Store and retrieve vectorized data in Supabase."""

    def __init__(self, config: Config) -> None:
        """Initialize the Supabase vector store.

        Args:
            config: Application configuration
        """
        self.config = config
        self.supabase = create_client(config.supabase_url, config.supabase_key)
        self.table_name = config.table_name
        self.vector_dimension = config.vector_dimension

    def create_table_if_not_exists(self) -> None:
        """Create the vector table in Supabase if it doesn't exist."""
        # Check if the table exists
        response = self.supabase.table(self.table_name).select("count(*)", count="exact").execute()

        # If we got an error, the table likely doesn't exist
        if response.get("error"):
            logger.info(f"Creating table {self.table_name}...")

            # SQL for creating the table with vector extension
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                book_id UUID NOT NULL,
                book_title TEXT NOT NULL,
                book_author TEXT NOT NULL,
                publisher TEXT,
                publication_date TEXT,
                isbn TEXT,
                genre JSONB,
                description TEXT,
                cover_url TEXT,
                page_count INTEGER,
                language TEXT,
                series TEXT,
                series_position NUMERIC,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR({self.vector_dimension})
            );
            
            CREATE INDEX IF NOT EXISTS book_chunks_book_id_idx ON {self.table_name} (book_id);
            CREATE INDEX IF NOT EXISTS book_chunks_isbn_idx ON {self.table_name} (isbn) WHERE isbn IS NOT NULL;
            CREATE INDEX IF NOT EXISTS book_chunks_series_idx ON {self.table_name} (series) WHERE series IS NOT NULL;
            """

            # Execute the SQL
            self.supabase.query(create_table_sql).execute()

            # Create a vector index for similarity search
            create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS book_chunks_embedding_idx ON {self.table_name} 
            USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
            """

            self.supabase.query(create_index_sql).execute()
            logger.info(f"Table {self.table_name} created successfully.")

    def store_chunks(
        self,
        book_id: str,
        chunks: List[str],
        metadata: Dict[str, Any],
        embedding_client: AnthropicEmbeddings,
    ) -> None:
        """Store book chunks with embeddings in Supabase.

        Args:
            book_id: Unique identifier for the book
            chunks: List of text chunks
            metadata: Book metadata
            embedding_client: Client for generating embeddings
        """
        # Ensure the table exists
        self.create_table_if_not_exists()

        # Process chunks and store in Supabase
        batch_size = 50  # Process in batches to avoid overwhelming the API
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}...")

            # Get embeddings for all chunks in the batch
            batch_data = []
            for chunk_idx, chunk in enumerate(batch_chunks, start=i):
                embedding = embedding_client.get_embedding(chunk)

                # Prepare data for insertion with all available metadata
                chunk_data = {
                    "book_id": book_id,
                    "book_title": metadata.get("title", "Unknown"),
                    "book_author": metadata.get("author", "Unknown"),
                    "publisher": metadata.get("publisher"),
                    "publication_date": metadata.get("publication_date"),
                    "isbn": metadata.get("isbn"),
                    "genre": json.dumps(metadata.get("genre", []))
                    if isinstance(metadata.get("genre", []), list)
                    else None,
                    "description": metadata.get("description"),
                    "cover_url": metadata.get("cover_url"),
                    "page_count": metadata.get("page_count"),
                    "language": metadata.get("language"),
                    "series": metadata.get("series"),
                    "series_position": metadata.get("series_position"),
                    "chunk_index": chunk_idx,
                    "content": chunk,
                    "embedding": embedding,
                }
                batch_data.append(chunk_data)

            # Insert batch into Supabase
            response = self.supabase.table(self.table_name).insert(batch_data).execute()
            if response.get("error"):
                logger.error(f"Error inserting batch: {response['error']}")
            else:
                logger.info(f"Inserted {len(batch_data)} chunks successfully")

    def search_similar_text(
        self,
        query: str,
        embedding_client: AnthropicEmbeddings,
        book_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for text chunks similar to the query.

        Args:
            query: Query text to search for
            embedding_client: Client for generating embeddings
            book_id: Optional book ID to filter results
            limit: Maximum number of results to return

        Returns:
            List of similar text chunks with metadata
        """
        # Get embedding for the query
        query_embedding = embedding_client.get_embedding(query)

        # Build the query with all metadata fields
        similarity_query = self.supabase.table(self.table_name).select(
            "book_title, book_author, publisher, publication_date, isbn, "
            + "genre, description, cover_url, page_count, language, "
            + "series, series_position, content, chunk_index, id, book_id"
        ).order(f"embedding <-> '{str(query_embedding)}'::vector").limit(limit)

        # Add book_id filter if specified
        if book_id:
            similarity_query = similarity_query.eq("book_id", book_id)

        # Execute the query
        result = similarity_query.execute()

        # Parse JSON genre field
        for item in result.data:
            if item.get("genre") and isinstance(item["genre"], str):
                try:
                    item["genre"] = json.loads(item["genre"])
                except json.JSONDecodeError:
                    item["genre"] = []

        return result.data

    def get_book_info(self, book_id: str) -> Dict[str, Any]:
        """Get book information by ID.

        Args:
            book_id: Book ID to retrieve

        Returns:
            Book metadata dictionary
        """
        # Get book metadata
        result = self.supabase.table(self.table_name).select(
            "book_title, book_author, publisher, publication_date, isbn, "
            + "genre, description, cover_url, page_count, language, "
            + "series, series_position"
        ).eq("book_id", book_id).limit(1).execute()

        if not result.data:
            return {}

        book_info = result.data[0]

        # Parse JSON genre field
        if book_info.get("genre") and isinstance(book_info["genre"], str):
            try:
                book_info["genre"] = json.loads(book_info["genre"])
            except json.JSONDecodeError:
                book_info["genre"] = []

        # Get chunk count
        count_result = self.supabase.table(self.table_name).select(
            "count(*)", count="exact"
        ).eq("book_id", book_id).execute()

        if hasattr(count_result, "count"):
            book_info["total_chunks"] = count_result.count

        return book_info


class BookVectorizer:
    """Main class for vectorizing books and searching."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize the book vectorizer.

        Args:
            config: Optional custom configuration. If None, loads from environment.
        """
        self.config = config or Config.from_env()
        self.epub_processor = EPUBProcessor(self.config)
        self.embedding_client = AnthropicEmbeddings(self.config)
        self.vector_store = SupabaseVectorStore(self.config)

    def vectorize_epub(self, epub_path: Union[str, Path]) -> str:
        """Process EPUB book, vectorize its content, and store in Supabase.

        Args:
            epub_path: Path to the EPUB file

        Returns:
            Book ID of the vectorized book
        """
        # Extract text and metadata from EPUB
        text, metadata = self.epub_processor.extract_text_from_epub(epub_path)

        # Split text into chunks
        chunks = self.epub_processor.split_text_into_chunks(text)
        logger.info(f"Book split into {len(chunks)} chunks")

        # Generate a unique ID for this book
        book_id = str(uuid.uuid4())

        # Store chunks in Supabase
        self.vector_store.store_chunks(book_id, chunks, metadata, self.embedding_client)

        return book_id

    def search(
        self, query: str, book_id: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar text.

        Args:
            query: Query text to search for
            book_id: Optional book ID to filter results
            limit: Maximum number of results to return

        Returns:
            List of similar text chunks with metadata
        """
        return self.vector_store.search_similar_text(
            query, self.embedding_client, book_id, limit
        )

    def get_book_info(self, book_id: str) -> Dict[str, Any]:
        """Get book information.

        Args:
            book_id: Book ID to get information for

        Returns:
            Book metadata dictionary
        """
        return self.vector_store.get_book_info(book_id)
