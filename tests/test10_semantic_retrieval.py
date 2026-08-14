"""Tests for SemanticRetriever.

Deliberately does NOT go through a real Indexer/mini_repo chunking
pass or real persistence -- that's already covered end-to-end in
test09. Instead, SemanticIndexer is patched wholesale to a fake
object exposing only .metadata/.embeddings, isolating retrieve()'s
own logic (top-k ranking, batching, the degenerate-query guard) from
everything underneath it. This also sidesteps having to hand-compute
the real mini_repo chunk count to build matching fake embeddings.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models import MinimalSource
from src.retrieval.semantic_retriever import SemanticRetriever


def _source(name: str) -> MinimalSource:
    return MinimalSource(file_path=name, first_character_index=0,
                         last_character_index=10)


def _fake_semantic_index(metadata: list[MinimalSource],
                         embeddings: np.ndarray) -> MagicMock:
    fake = MagicMock()
    fake.metadata = metadata
    fake.embeddings = embeddings
    return fake


@pytest.fixture
def make_semantic_retriever():
    """Returns make(queries, k, metadata, embeddings, query_vectors)
    -- constructs a SemanticRetriever against a fully hand-built,
    in-memory fake index. query_vectors maps each query string to the
    vector the mocked encoder should return for it."""
    def _make(queries: list[str], k: int,
             metadata: list[MinimalSource], embeddings: np.ndarray,
             query_vectors: dict[str, np.ndarray]) -> SemanticRetriever:
        fake_index = _fake_semantic_index(metadata, embeddings)
        query_encoder = MagicMock()
        query_encoder.encode.side_effect = (
            lambda texts: np.stack([query_vectors[t] for t in texts]))

        with patch("src.retrieval.semantic_retriever.Indexer"), \
             patch("src.retrieval.semantic_retriever.SemanticIndexer",
                  return_value=fake_index), \
             patch("src.retrieval.semantic_retriever.SemanticEncoder",
                  return_value=query_encoder):
            return SemanticRetriever(queries, k)
    return _make


class TestSemanticRetrieverDegenerateQueries:
    def test_empty_queries_list_returns_empty_list(self, make_semantic_retriever):
        retriever = make_semantic_retriever(
            [], k=5, metadata=[], embeddings=np.zeros((0, 4), dtype=np.float32),
            query_vectors={})
        assert retriever.retrieve() == []

    def test_k_zero_returns_empty_list_for_the_query(self, make_semantic_retriever):
        metadata = [_source("a.py")]
        embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        retriever = make_semantic_retriever(
            ["anything"], k=0, metadata=metadata, embeddings=embeddings,
            query_vectors={"anything": embeddings[0]})
        assert retriever.retrieve() == [[]]

    def test_empty_string_query_never_reaches_the_encoder(self, make_semantic_retriever):
        metadata = [_source("a.py")]
        embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        retriever = make_semantic_retriever(
            [""], k=5, metadata=metadata, embeddings=embeddings, query_vectors={})
        assert retriever.retrieve() == [[]]
        retriever.encoder.encode.assert_not_called()

    def test_whitespace_only_query_never_reaches_the_encoder(self, make_semantic_retriever):
        metadata = [_source("a.py")]
        embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        retriever = make_semantic_retriever(
            ["   "], k=5, metadata=metadata, embeddings=embeddings, query_vectors={})
        assert retriever.retrieve() == [[]]
        retriever.encoder.encode.assert_not_called()


class TestSemanticRetrieverRanking:
    def test_ranks_the_most_similar_chunk_first(self, make_semantic_retriever):
        metadata = [_source("a.py"), _source("b.py"), _source("c.py")]
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        query = "find b"
        retriever = make_semantic_retriever(
            [query], k=2, metadata=metadata, embeddings=embeddings,
            query_vectors={query: embeddings[1]})
        results = retriever.retrieve()
        assert results[0][0] == metadata[1]
        assert len(results[0]) == 2

    def test_no_more_than_k_results_returned(self, make_semantic_retriever):
        metadata = [_source(f"{i}.py") for i in range(5)]
        embeddings = np.eye(5, dtype=np.float32)[:, :4]
        query = "q"
        retriever = make_semantic_retriever(
            [query], k=2, metadata=metadata, embeddings=embeddings,
            query_vectors={query: embeddings[0]})
        assert len(retriever.retrieve()[0]) == 2

    def test_k_larger_than_corpus_size_is_clamped_with_a_warning(
            self, capsys, make_semantic_retriever):
        metadata = [_source(f"{i}.py") for i in range(3)]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                              dtype=np.float32)
        query = "q"
        retriever = make_semantic_retriever(
            [query], k=1000, metadata=metadata, embeddings=embeddings,
            query_vectors={query: embeddings[0]})
        results = retriever.retrieve()
        assert len(results[0]) == 3
        assert "Warning" in capsys.readouterr().out

    def test_batch_preserves_query_order(self, make_semantic_retriever):
        metadata = [_source("a.py"), _source("b.py"), _source("c.py")]
        embeddings = np.eye(3, dtype=np.float32)
        q1, q2 = "first", "second"
        retriever = make_semantic_retriever(
            [q1, q2], k=1, metadata=metadata, embeddings=embeddings,
            query_vectors={q1: embeddings[0], q2: embeddings[2]})
        results = retriever.retrieve()
        assert results[0][0] == metadata[0]
        assert results[1][0] == metadata[2]

    def test_degenerate_query_in_a_batch_does_not_contaminate_other_rows(
            self, make_semantic_retriever):
        metadata = [_source("a.py"), _source("b.py")]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        q = "real query"
        retriever = make_semantic_retriever(
            ["   ", q], k=1, metadata=metadata, embeddings=embeddings,
            query_vectors={q: embeddings[1]})
        results = retriever.retrieve()
        assert results[0] == []
        assert results[1][0] == metadata[1]
