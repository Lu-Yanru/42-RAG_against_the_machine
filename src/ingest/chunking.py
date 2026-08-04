from dataclasses import dataclass

from src.ingest.loader import File


@dataclass
class Chunk:
    """Represents a single chunk text."""
    file_path: str
    first_character_index: int
    last_character_index: int
    content: str


class Chunker:
    """Basic chunker class."""
    def __init__(self, files: list[File],
                 max_chunk_size: int = 2000) -> None:
        self.files = files
        self.max_chunk_size = max_chunk_size


class TextChunker(Chunker):
    def __init__(self, files: list[File],
                 max_chunk_size: int = 2000) -> None:
        super().__init__(files, max_chunk_size)


class MarkdownChunker(TextChunker):
    def __init__(self, files: list[File],
                 max_chunk_size: int = 2000) -> None:
        super().__init__(files, max_chunk_size)


class PythonChunker(Chunker):
    def __init__(self, files: list[File],
                 max_chunk_size: int = 2000) -> None:
        super().__init__(files, max_chunk_size)
