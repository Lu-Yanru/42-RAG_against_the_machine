"""Build and persist a semantic index."""


import json
import numpy as np
from pathlib import Path
from pydantic import ValidationError

from src.config import SEMANTIC_INDEX_DIR
from src.indexing.hash import chunk_key
from src.indexing.indexer import Indexer
from src.indexing.semantic_encoder import SemanticEncoder
from src.models import MinimalSource


class SemanticIndexingError(Exception):
    """Error related to semantic indexing."""
    pass


class SemanticIndexer:
    """
    Builds and persists a dense-embedding sibling index alongside the
    BM25 index. Reuses an already-built `Indexer`'s `.chunks`/
    `.texts` instead of re-running Loader/Chunker, so chunk
    boundaries stay identical to the BM25 index.
    """
    EMBEDDING_FILENAME = "semantic_embeddings.npy"
    METADATA_FILENAME = "semantic_metadata.json"

    def __init__(self, indexer: Indexer,
                 save_dir: str = SEMANTIC_INDEX_DIR) -> None:
        self.indexer = indexer
        self.save_dir = Path(save_dir)
        self.embeddings_path = self.save_dir / self.EMBEDDING_FILENAME
        self.metadata_path = self.save_dir / self.METADATA_FILENAME

    def build(self) -> None:
        if not hasattr(self.indexer, "chunks") or not self.indexer.chunks:
            raise SemanticIndexingError(
                "SemanticIndexingError: the given Indexer has no chunks. "
                "Call indexer.load_chunks(...) "
                "before SemanticIndexer.build(...).")

        self.encoder = SemanticEncoder()
        self.metadata = [MinimalSource(
            file_path=c.file_path,
            first_character_index=c.first_character_index,
            last_character_index=c.last_character_index
        ) for c in self.indexer.chunks]

        self.embeddings = self.encoder.encode(self.indexer.texts)

    def build_incremental(self) -> None:
        """
        Like build(), but only encodes chunks not already present in
        the previously persisted semantic index. Chunk identity is
        matched by (file_path, start, end) against the old semantic
        metadata, so a chunk that was never
        actually embedded before is correctly (re-)encoded
        even though its source file
        looks "unchanged" to the lexical incremental pass.
        """
        if self.embeddings_path.exists() \
                and self.metadata_path.exists():
            if not hasattr(self.indexer, "chunks") \
                    or not self.indexer.chunks:
                raise SemanticIndexingError(
                    "No new chunks to be indexed.")

        self.encoder = SemanticEncoder()
        old_metadata, old_embeddings = self._load_old()
        dim_ok = (old_embeddings.ndim == 2
                  and old_embeddings.shape[0] == len(old_metadata)
                  and old_embeddings.shape[1] == self.encoder.dim)
        old_by_key = ({chunk_key(m): row
                       for row, m in enumerate(old_metadata)}
                      if dim_ok else {})

        kept_meta: list[MinimalSource] = []
        kept_rows: list[int] = []
        for m in self.indexer.metadata:
            row = old_by_key.get(chunk_key(m))
            if row is not None:
                kept_meta.append(m)
                kept_rows.append(row)

        kept_keys = {chunk_key(m) for m in kept_meta}
        to_encode = [(m, t) for m, t in zip(self.indexer.metadata,
                                            self.indexer.texts)
                     if chunk_key(m) not in kept_keys]
        to_encode_meta = [m for m, _ in to_encode]
        to_encode_texts = [t for _, t in to_encode]

        new_embeddings = self.encoder.encode(to_encode_texts)

        if kept_rows:
            self.embeddings = np.vstack(
                [old_embeddings[kept_rows], new_embeddings]
            )
        else:
            self.embeddings = new_embeddings
        self.metadata = kept_meta + to_encode_meta

    def _load_old(self) -> tuple[list[MinimalSource], np.ndarray]:
        """
        Best-effort read of the previously persisted semantic index.
        Returns ([], an empty array) if nothing has been embedded
        before, or the persisted files are unreadable. Every current
        chunk is then encoded, same as build()'s full path.
        """
        if not self.embeddings_path.exists() \
                or not self.metadata_path.exists():
            return [], np.empty((0, 0), dtype=np.float32)

        try:
            embeddings = np.load(self.embeddings_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            metadata = [MinimalSource.model_validate(m) for m in raw]
            return metadata, embeddings
        except (json.JSONDecodeError, OSError, ValueError, TypeError,
                ValidationError):
            return [], np.empty((0, 0), dtype=np.float32)

    def save(self) -> None:
        """Save the semantic embeddings and chunk metadata."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        try:
            np.save(self.embeddings_path, self.embeddings)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump([m.model_dump() for m in self.metadata], f, indent=4)
        except (json.JSONDecodeError, OSError) as e:
            raise SemanticIndexingError(
                f"Error: Failed to save semantic indexes: {e}")

    def load(self) -> None:
        """
        Reload a previously persisted semantic index for querying.
        Raises SemanticIndexingError if nothing has been indexed yet.
        """
        if not self.save_dir.exists() \
                or not self.embeddings_path.exists() \
                or not self.metadata_path.exists():
            raise SemanticIndexingError(
                "SemanticIndexingError: No persisted semantic index found "
                f"under '{self.save_dir}'. Run 'index --method semantic' "
                "or 'index --method hybrid' first.")

        try:
            self.embeddings = np.load(self.embeddings_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.metadata = [MinimalSource.model_validate(m) for m in raw]
        except (json.JSONDecodeError, OSError, ValueError,
                TypeError, ValidationError) as e:
            raise SemanticIndexingError(
                "SemanticIndexingError: Failed to load persisted semantic "
                f"index: {e}")
