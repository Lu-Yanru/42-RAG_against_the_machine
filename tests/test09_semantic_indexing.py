"""Tests for SemanticEncoder and SemanticIndexer.

Patches sentence_transformers.SentenceTransformer at its lowest
layer (mocked_st), so SemanticEncoder + SemanticIndexer are tested
together as a real integration, without downloading real weights.
SemanticRetriever's own logic gets isolated separately in
test10 -- see the note there for why the mocking strategy differs.
"""
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.indexing.indexer import Indexer
from src.indexing.semantic_encoder import SemanticEncoder
from src.indexing.semantic_indexer import SemanticIndexer, SemanticIndexingError

FIXTURE_ROOT = "tests/fixtures/mini_repo"


@pytest.fixture
def mocked_st():
    """Stands in for SentenceTransformer so SemanticEncoder() never
    downloads real weights. get_sentence_embedding_dimension is
    pre-wired for when the bug below gets fixed."""
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 4
    with patch("src.indexing.semantic_encoder.SentenceTransformer") as st_cls:
        st_cls.return_value = model
        yield st_cls, model


def _mini_repo_indexer() -> Indexer:
    """Loads chunks only -- SemanticIndexer.build() only reads
    indexer.chunks/.texts, not a fitted BM25 retriever. Calling
    indexer.build() here would fit BM25 in memory for nothing."""
    indexer = Indexer(save_dir="unused")
    indexer.load_chunks(raw_data=FIXTURE_ROOT)
    return indexer


class TestSemanticEncoderDimBug:
    """SemanticEncoder.dim calls self._model.get_sentence_embedding
    .dimension() -- should be get_sentence_embedding_dimension().
    Documented as xfail(strict=True) rather than papered over; remove
    the marker once the property is fixed."""

    def test_dim_returns_the_model_dimension(self, mocked_st):
        encoder = SemanticEncoder()
        assert encoder.dim == 4

    def test_encode_empty_list_returns_correctly_shaped_array(self, mocked_st):
        encoder = SemanticEncoder()
        result = encoder.encode([])
        assert result.shape == (0, 4)


class TestSemanticEncoderEncode:
    def test_normalizes_and_returns_numpy(self, mocked_st):
        _, model = mocked_st
        model.encode.return_value = np.zeros((2, 4), dtype=np.float64)
        encoder = SemanticEncoder()
        result = encoder.encode(["a", "b"])
        _, kwargs = model.encode.call_args
        assert kwargs["normalize_embeddings"] is True
        assert kwargs["convert_to_numpy"] is True
        assert result.dtype == np.float32

    def test_batch_size_is_forwarded(self, mocked_st):
        _, model = mocked_st
        model.encode.return_value = np.zeros((1, 4), dtype=np.float32)
        encoder = SemanticEncoder(batch_size=7)
        encoder.encode(["a"])
        _, kwargs = model.encode.call_args
        assert kwargs["batch_size"] == 7


class TestSemanticIndexerBuild:
    def test_raises_before_chunks_are_loaded(self, mocked_st):
        indexer = Indexer(save_dir="unused")  # load_chunks() never called
        sem_indexer = SemanticIndexer(indexer, save_dir="unused")
        with pytest.raises(SemanticIndexingError):
            sem_indexer.build()

    def test_produces_one_metadata_entry_per_chunk(self, mocked_st):
        _, model = mocked_st
        indexer = _mini_repo_indexer()
        num_chunks = len(indexer.chunks)
        model.encode.return_value = np.zeros((num_chunks, 4), dtype=np.float32)

        sem_indexer = SemanticIndexer(indexer, save_dir="unused")
        sem_indexer.build()

        assert len(sem_indexer.metadata) == num_chunks
        assert sem_indexer.embeddings.shape == (num_chunks, 4)

    def test_metadata_order_matches_chunk_order(self, mocked_st):
        _, model = mocked_st
        indexer = _mini_repo_indexer()
        model.encode.return_value = np.zeros(
            (len(indexer.chunks), 4), dtype=np.float32)

        sem_indexer = SemanticIndexer(indexer, save_dir="unused")
        sem_indexer.build()

        for chunk, source in zip(indexer.chunks, sem_indexer.metadata):
            assert source.file_path == chunk.file_path
            assert source.first_character_index == chunk.first_character_index
            assert source.last_character_index == chunk.last_character_index

    def test_encoder_receives_chunk_texts_not_raw_files(self, mocked_st):
        _, model = mocked_st
        indexer = _mini_repo_indexer()
        model.encode.return_value = np.zeros(
            (len(indexer.chunks), 4), dtype=np.float32)

        sem_indexer = SemanticIndexer(indexer, save_dir="unused")
        sem_indexer.build()

        args, _ = model.encode.call_args
        assert args[0] == indexer.texts


class TestSemanticIndexerPersistRoundTrip:
    def test_persisted_metadata_has_no_chunk_text(self, tmp_path, mocked_st):
        _, model = mocked_st
        indexer = _mini_repo_indexer()
        num_chunks = len(indexer.chunks)
        model.encode.return_value = np.zeros((num_chunks, 4), dtype=np.float32)

        sem_indexer = SemanticIndexer(indexer, save_dir=str(tmp_path))
        sem_indexer.build()
        sem_indexer.save()

        with open(tmp_path / "semantic_metadata.json", encoding="utf-8") as f:
            raw = json.load(f)
        assert len(raw) == num_chunks
        for entry in raw:
            assert set(entry.keys()) == {
                "file_path", "first_character_index", "last_character_index"
            }

    def test_reloaded_embeddings_match_the_built_ones(self, tmp_path, mocked_st):
        _, model = mocked_st
        indexer = _mini_repo_indexer()
        num_chunks = len(indexer.chunks)
        fake_embeddings = np.arange(
            num_chunks * 4, dtype=np.float32).reshape(num_chunks, 4)
        model.encode.return_value = fake_embeddings

        sem_indexer = SemanticIndexer(indexer, save_dir=str(tmp_path))
        sem_indexer.build()
        sem_indexer.save()

        # Fresh instance simulates a new process loading from disk.
        reloaded = SemanticIndexer(indexer, save_dir=str(tmp_path))
        reloaded.load()

        assert np.array_equal(reloaded.embeddings, fake_embeddings)
        assert reloaded.metadata == sem_indexer.metadata

    def test_load_raises_before_anything_has_been_indexed(self, tmp_path):
        indexer = _mini_repo_indexer()
        sem_indexer = SemanticIndexer(indexer, save_dir=str(tmp_path))
        with pytest.raises(SemanticIndexingError):
            sem_indexer.load()

    def test_save_failure_is_wrapped_as_semantic_indexing_error(
            self, tmp_path, mocked_st, monkeypatch):
        _, model = mocked_st
        indexer = _mini_repo_indexer()
        model.encode.return_value = np.zeros(
            (len(indexer.chunks), 4), dtype=np.float32)

        sem_indexer = SemanticIndexer(indexer, save_dir=str(tmp_path))
        sem_indexer.build()

        def _raise(*args, **kwargs):
            raise OSError("disk full")
        monkeypatch.setattr(np, "save", _raise)

        with pytest.raises(SemanticIndexingError):
            sem_indexer.save()

    def test_corrupted_metadata_json_raises_semantic_indexing_error(
            self, tmp_path, mocked_st):
        _, model = mocked_st
        indexer = _mini_repo_indexer()
        model.encode.return_value = np.zeros(
            (len(indexer.chunks), 4), dtype=np.float32)
        sem_indexer = SemanticIndexer(indexer, save_dir=str(tmp_path))
        sem_indexer.build()
        sem_indexer.save()

        (tmp_path / "semantic_metadata.json").write_text(
            "not json", encoding="utf-8")

        reloaded = SemanticIndexer(indexer, save_dir=str(tmp_path))
        with pytest.raises(SemanticIndexingError):
            reloaded.load()

    def test_reindexing_overwrites_the_previous_index(self, tmp_path, mocked_st):
        _, model = mocked_st
        indexer = _mini_repo_indexer()
        model.encode.return_value = np.zeros(
            (len(indexer.chunks), 4), dtype=np.float32)

        sem_indexer = SemanticIndexer(indexer, save_dir=str(tmp_path))
        sem_indexer.build()
        sem_indexer.save()

        indexer.load_chunks(raw_data=FIXTURE_ROOT, max_chunk_size=50)
        num_chunks_b = len(indexer.chunks)
        model.encode.return_value = np.zeros((num_chunks_b, 4), dtype=np.float32)
        sem_indexer.build()
        sem_indexer.save()

        sem_indexer.load()
        assert len(sem_indexer.metadata) == num_chunks_b
