import bm25s
import json
from pathlib import Path
import sys

from src.indexing.tokenizer import build_tokenizer
from src.ingest.chunking import ChunkError
from src.ingest.chunking_text import TextChunker, MarkdownChunker
from src.ingest.chunking_python import PythonChunker
from src.ingest.loader import Loader, LoadError
from src.models import MinimalSource


class IndexingError(Exception):
    """Error related to indexing."""
    pass


class Indexer:
    def __init__(self, save_dir: str = "data/processed/") -> None:
        self.save_dir = Path(save_dir)
        self.save_path = self.save_dir / "chunk_metadata.json"

    def build(self, raw_data: str = "data/raw",
              max_chunk_size: int = 2000) -> None:
        """
        Create indexes from the raw data.
        """
        try:
            self.loader = Loader(raw_data)
        except LoadError as e:
            print(e, file=sys.stderr)
            exit(1)

        self.txt_chunks = None
        self.md_chunks = None
        self.py_chunks = None

        try:
            txt_chunker = TextChunker(self.loader.txt_files,
                                      max_chunk_size, "txt")
            self.txt_chunks = txt_chunker.chunk()
        except ChunkError as e:
            print(f"Text {e}", file=sys.stderr)

        try:
            md_chunker = MarkdownChunker(self.loader.md_files,
                                         max_chunk_size, "md")
            self.md_chunks = md_chunker.chunk()
        except ChunkError as e:
            print(f"Markdown {e}", file=sys.stderr)

        try:
            py_chunker = PythonChunker(self.loader.py_files,
                                       max_chunk_size, "py")
            self.py_chunks = py_chunker.chunk()
        except ChunkError as e:
            print(f"Python {e}", file=sys.stderr)

        if self.txt_chunks is None and self.md_chunks is None \
                and self.py_chunks is None:
            print("Fail to chunk files. Exiting...", file=sys.stderr)
            exit(1)

        self.chunks = []
        for chunks in [self.txt_chunks, self.md_chunks, self.py_chunks]:
            if chunks is not None:
                self.chunks.extend(chunks)

        self.texts = [c.content for c in self.chunks]
        self.metadata = [MinimalSource(
            file_path=c.file_path,
            first_character_index=c.first_character_index,
            last_character_index=c.last_character_index
        ) for c in self.chunks]

        self.tokenizer = build_tokenizer()
        self.corpus_tokenized = self.tokenizer.tokenize(self.texts,
                                                        return_as="tuple")
        self.retriever = bm25s.BM25()
        self.retriever.index(self.corpus_tokenized)

    def save(self) -> None:
        """Save indexes created from raw data."""
        self.retriever.save(str(self.save_dir))
        self.tokenizer.save_vocab(str(self.save_dir))
        try:
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump([m.model_dump() for m in self.metadata], f, indent=4)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error: Failed to save indexes: {e} Exiting...",
                  file=sys.stderr)
            exit(1)

    def load(self) -> None:
        """
        Reload a previously persisted index for querying. Raises
        BuildError if nothing has been indexed yet.
        """
        if not self.save_dir.exists() or not self.save_path.exists():
            raise IndexingError(
                "IndexingError: No persisted index found under "
                f"'{self.save_dir}'. Run the 'index' command first.")

        try:
            self.retriever = bm25s.BM25.load(str(self.save_dir),
                                             load_corpus=False,
                                             load_vocab=True)
            self.tokenizer = build_tokenizer()
            self.tokenizer.load_vocab(str(self.save_dir))

            with open(self.save_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.metadata = [MinimalSource.model_validate(m)
                             for m in raw]
        except FileNotFoundError:
            raise IndexingError(
                "IndexingError: No persisted index found under "
                f"'{self.save_dir}'. Run the 'index' command first.")
        except (json.JSONDecodeError, OSError) as e:
            raise IndexingError("IndexingError: Failed to load "
                                f"persisted index: {e}")
