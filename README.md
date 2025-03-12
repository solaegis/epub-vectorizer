# EPUB Vectorizer

[![PyPI - Version](https://img.shields.io/pypi/v/epub-vectorizer.svg)](https://pypi.org/project/epub-vectorizer)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/epub-vectorizer.svg)](https://pypi.org/project/epub-vectorizer)
[![License - MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

A tool to extract text from EPUB books, generate embeddings using Anthropic's Claude 3.7 Sonnet model, and store them in a Supabase vector database for semantic search. The tool also enriches book metadata using the Google Books API.

## Features

- Extract text and metadata from EPUB files
- Generate embeddings using Anthropic's Claude 3.7 Sonnet model
- Enrich book metadata using Google Books API
- Store vectorized content in Supabase for semantic search
- CLI interface for vectorizing, searching, and retrieving book information

## Installation

```bash
pip install epub-vectorizer
```

## Configuration

Set the following environment variables or create a `.env` file:

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_BOOKS_API_KEY=your_google_books_api_key  # Optional
EMBEDDING_MODEL=claude-3-7-sonnet-20250219
VECTOR_DIMENSION=3072
TABLE_NAME=book_chunks
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Usage

### Command Line Interface

#### Vectorize a Book

```bash
epub-vectorizer vectorize path/to/book.epub
```

This will:
1. Extract text and metadata from the EPUB file
2. Split the text into overlapping chunks
3. Generate embeddings for each chunk using Anthropic
4. Store the chunks and embeddings in Supabase
5. Display the book's metadata and ID

#### Search for Content

```bash
epub-vectorizer search "query text"
```

Search options:
- `--book-id BOOK_ID`: Limit search to a specific book
- `--limit N`: Return at most N results (default: 5)
- `--metadata-only`: Display only book metadata, not content chunks

#### Get Book Information

```bash
epub-vectorizer info BOOK_ID
```

This will display comprehensive metadata about the book, including title, author, publisher, genres, series information, etc.

### Python API

```python
from epub_vectorizer import BookVectorizer

# Initialize the vectorizer
vectorizer = BookVectorizer()

# Vectorize a book
book_id = vectorizer.vectorize_epub("path/to/book.epub")

# Search for content
results = vectorizer.search("query text", book_id=None, limit=5)

# Get book information
book_info = vectorizer.get_book_info(book_id)
```

## Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/epub-vectorizer.git
   cd epub-vectorizer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

5. Run the tests:
   ```bash
   pytest
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
