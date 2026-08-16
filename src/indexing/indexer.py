import bm25s
import json
from pathlib import Path
from pydantic import ValidationError
import sys

from src.config import INDEX_DIR, RAW_DATA, MAX_CHUNK_SIZE, HASH_PATH
from src.indexing import hash
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
    def __init__(self, save_dir: str = INDEX_DIR) -> None:
        self.save_dir = Path(save_dir)
        self.save_path = self.save_dir / "chunk_metadata.json"

    def load_chunks(self, raw_data: str = RAW_DATA,
                    max_chunk_size: int = MAX_CHUNK_SIZE) -> None:
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

    def load_chunks_incremental(self, raw_data: str = RAW_DATA,
                                max_chunk_size: int = MAX_CHUNK_SIZE,
                                hash_path: str = HASH_PATH) -> None:
        """
        Like load_chunks(), but skips re-chunking any file whose
        content hash matches the last persisted hash, reusing its
        previously computed chunk offsets instead.

        Falls back to chunking every file -- identical output to
        load_chunks() -- when there is no prior hash, or when
        max_chunk_size differs from the one the hash was built
        with, since chunk boundaries cut under a different cap can't
        be reused.
        """
        try:
            self.loader = Loader(raw_data)
        except LoadError as e:
            print(e, file=sys.stderr)
            exit(1)

        all_files = (self.loader.py_files + self.loader.md_files
                     + self.loader.txt_files)
        current_hashes = {f.file_path: hash.compute_hash(f.content)
                          for f in all_files}

        old_hash = hash.load_hash(Path(hash_path))
        needs_rechunk, unchanged = hash.diff(
            old_hash, current_hashes, max_chunk_size)

        old_by_path = self._old_metadata_by_path()
        if old_by_path is None:
            # chunk_metadata.json is missing or unreadable.
            needs_rechunk |= unchanged
            unchanged = set()
            old_by_path = {}

        files_by_path = {f.file_path: f for f in all_files}
        kept_metadata: list[MinimalSource] = []
        kept_texts: list[str] = []
        for path in unchanged:
            content = files_by_path[path].content
            for m in old_by_path.get(path, []):
                kept_metadata.append(m)
                kept_texts.append(content[m.first_character_index:
                                          m.last_character_index])

        changed_py = [f for f in self.loader.py_files
                      if f.file_path in needs_rechunk]
        changed_md = [f for f in self.loader.md_files
                      if f.file_path in needs_rechunk]
        changed_txt = [f for f in self.loader.txt_files
                       if f.file_path in needs_rechunk]

        new_chunks = []
        for chunker_cls, files, label in (
                (PythonChunker, changed_py, "py"),
                (MarkdownChunker, changed_md, "md"),
                (TextChunker, changed_txt, "txt")):
            if not files:
                continue
            try:
                chunker = chunker_cls(files, max_chunk_size, label)
                new_chunks.extend(chunker.chunk())
            except ChunkError as e:
                print(f"{label} {e}", file=sys.stderr)

        if not kept_metadata and not new_chunks:
            print("Fail to chunk files. Exiting...", file=sys.stderr)
            exit(1)

        self.chunks = new_chunks  # only newly (re)chunked files
        self.metadata = kept_metadata + [MinimalSource(
            file_path=c.file_path,
            first_character_index=c.first_character_index,
            last_character_index=c.last_character_index)
            for c in new_chunks]
        self.texts = kept_texts + [c.content for c in new_chunks]

        hash.save_hash(Path(hash_path), max_chunk_size,
                       current_hashes)

    def _old_metadata_by_path(self) -> dict[str, list[MinimalSource]] | None:
        """
        Best-effort read of previously persisted chunk metadata,
        grouped by file_path. Returns None if
        chunk_metadata.json doesn't exist or can't be parsed.
        """
        if not self.save_path.exists():
            return None
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            by_path: dict[str, list[MinimalSource]] = {}
            for entry in raw:
                m = MinimalSource.model_validate(entry)
                by_path.setdefault(m.file_path, []).append(m)
            return by_path
        except (json.JSONDecodeError, OSError, TypeError, KeyError,
                ValidationError):
            return None

    def build(self) -> None:
        """
        Create indexes from the raw data.
        """
        if not hasattr(self, "texts") or not self.texts:
            raise IndexingError(
                "IndexingError: the given Indexer has no chunks. "
                "Call indexer.load_chunks(...) "
                "before Indexer.build(...).")

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
            raise IndexingError(f"Error: Failed to save indexes: {e}")

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
