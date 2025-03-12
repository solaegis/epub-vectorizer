"""Command-line interface for EPUB Vectorizer."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from epub_vectorizer.config import Config
from epub_vectorizer.core import BookVectorizer


def display_book_metadata(metadata: Dict[str, Any]) -> None:
    """Display book metadata in a formatted way.

    Args:
        metadata: Book metadata to display
    """
    print("\nBook Metadata:")
    print(f"  Title: {metadata.get('book_title', 'Unknown')}")
    print(f"  Author: {metadata.get('book_author', 'Unknown')}")

    if metadata.get("publisher"):
        print(f"  Publisher: {metadata['publisher']}")

    if metadata.get("publication_date"):
        print(f"  Publication Date: {metadata['publication_date']}")

    if metadata.get("isbn"):
        print(f"  ISBN: {metadata['isbn']}")

    if metadata.get("genre") and (
        isinstance(metadata["genre"], list) or isinstance(metadata["genre"], str)
    ):
        genres = metadata["genre"]
        if isinstance(genres, str):
            try:
                genres = json.loads(genres)
            except json.JSONDecodeError:
                genres = [genres]

        if genres:
            print(f"  Genre(s): {', '.join(genres)}")

    if metadata.get("series"):
        series_info = f"{metadata['series']}"
        if metadata.get("series_position"):
            series_info += f" #{metadata['series_position']}"
        print(f"  Series: {series_info}")

    if metadata.get("language"):
        print(f"  Language: {metadata['language']}")

    if metadata.get("page_count"):
        print(f"  Pages: {metadata['page_count']}")

    if metadata.get("cover_url"):
        print(f"  Cover URL: {metadata['cover_url']}")

    if metadata.get("description"):
        print(f"\n  Description: {metadata['description'][:200]}...")

    if metadata.get("total_chunks"):
        print(f"\n  Total chunks: {metadata['total_chunks']}")


def command_vectorize(args: argparse.Namespace, vectorizer: BookVectorizer) -> int:
    """Handle 'vectorize' command.

    Args:
        args: Command-line arguments
        vectorizer: BookVectorizer instance

    Returns:
        Exit code (0 on success)
    """
    try:
        # Make sure EPUB file exists
        epub_path = Path(args.epub_path)
        if not epub_path.exists():
            print(f"Error: EPUB file not found: {epub_path}")
            return 1

        # Vectorize the book
        book_id = vectorizer.vectorize_epub(epub_path)
        print(f"Book vectorized successfully. Book ID: {book_id}")

        # Display the book's metadata
        book_info = vectorizer.get_book_info(book_id)
        if book_info:
            display_book_metadata(book_info)

        return 0
    except Exception as e:
        print(f"Error vectorizing book: {e}")
        return 1


def command_search(args: argparse.Namespace, vectorizer: BookVectorizer) -> int:
    """Handle 'search' command.

    Args:
        args: Command-line arguments
        vectorizer: BookVectorizer instance

    Returns:
        Exit code (0 on success)
    """
    try:
        results = vectorizer.search(args.query, args.book_id, args.limit)
        print(f"Found {len(results)} similar chunks:")

        # Group results by book
        books = {}
        for result in results:
            book_id = result["book_id"]
            if book_id not in books:
                books[book_id] = {"metadata": result, "chunks": []}
            books[book_id]["chunks"].append(result)

        # Display results
        for book_id, book_data in books.items():
            metadata = book_data["metadata"]
            display_book_metadata(metadata)

            if not args.metadata_only:
                print("\nMatching Chunks:")
                for chunk in book_data["chunks"]:
                    print(f"\n--- Chunk {chunk['chunk_index']} ---")
                    print(f"Content: {chunk['content'][:200]}...")

        return 0
    except Exception as e:
        print(f"Error searching: {e}")
        return 1


def command_info(args: argparse.Namespace, vectorizer: BookVectorizer) -> int:
    """Handle 'info' command.

    Args:
        args: Command-line arguments
        vectorizer: BookVectorizer instance

    Returns:
        Exit code (0 on success)
    """
    try:
        book_info = vectorizer.get_book_info(args.book_id)

        if book_info:
            display_book_metadata(book_info)
        else:
            print(f"No book found with ID: {args.book_id}")

        return 0
    except Exception as e:
        print(f"Error retrieving book info: {e}")
        return 1


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: Command-line arguments to parse (default: sys.argv[1:])

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Vectorize EPUB books and store in Supabase."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Vectorize command
    vectorize_parser = subparsers.add_parser("vectorize", help="Vectorize an EPUB book")
    vectorize_parser.add_argument("epub_path", help="Path to the EPUB file")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar text")
    search_parser.add_argument("query", help="Query text to search for")
    search_parser.add_argument("--book-id", help="Limit search to a specific book ID")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results"
    )
    search_parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only show book metadata, not content chunks",
    )

    # Info command - get book information
    info_parser = subparsers.add_parser(
        "info", help="Get information about a book in the database"
    )
    info_parser.add_argument("book_id", help="Book ID to get information for")

    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the command-line interface.

    Args:
        args: Command-line arguments (default: sys.argv[1:])

    Returns:
        Exit code (0 on success)
    """
    parsed_args = parse_args(args)

    if not parsed_args.command:
        print("No command specified. Use -h or --help for usage information.")
        return 1

    # Load configuration and initialize vectorizer
    try:
        config = Config.from_env()
        vectorizer = BookVectorizer(config)
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1

    # Execute the appropriate command
    if parsed_args.command == "vectorize":
        return command_vectorize(parsed_args, vectorizer)
    elif parsed_args.command == "search":
        return command_search(parsed_args, vectorizer)
    elif parsed_args.command == "info":
        return command_info(parsed_args, vectorizer)
    else:
        print(f"Unknown command: {parsed_args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
