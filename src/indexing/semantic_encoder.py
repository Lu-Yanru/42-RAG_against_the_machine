"""
Semantic-embedding encoder wrapper for semantic indexing.
"""


import numpy as np
import torch
from typing import Any, cast
from sentence_transformers import SentenceTransformer

from src.config import SEMANTIC_BATCH_SIZE, SEMANTIC_MODEL_NAME


class SemanticEncoder:
    """
    Wraps a SentenceTransformer model. Device selection
    mirrors src.generation.generator.Generator's
    mps > cuda > cpu order.

    Parameters
    ----------
    model_name: str, default=SEMANTIC_MODEL_NAME
    device: str | None, default=None
        Auto-selected (mps > cuda > cpu) when None.
    batch_size: int, default=SEMANTIC_BATCH_SIZE
    """
    def __init__(self,
                 model_name: str = SEMANTIC_MODEL_NAME,
                 device: str | None = None,
                 batch_size: int = SEMANTIC_BATCH_SIZE) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        # Auto-select device with prority mps > cuda > cpu
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device

        self._model: Any = SentenceTransformer(self.model_name,
                                               device=self._device)

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts into L2-normalized float32 embeddings, shape
        (len(texts), dim). Normalizing here means query-time
        similarity is a plain dot product, not a separate cosine
        computation.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return cast(np.ndarray, embeddings).astype(np.float32)

    @property
    def dim(self) -> int:
        """
        Returns the number of dimensions in the output of
        SentenceTransformer.encode().
        """
        return cast(int, self._model.get_embedding_dimension())
