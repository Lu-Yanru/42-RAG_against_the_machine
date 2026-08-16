import ast
import shutil
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.indexing import hash
from src.indexing.indexer import Indexer
from src.indexing.semantic_indexer import SemanticIndexer
from src.ingest.chunking_python import PythonChunker

FIXTURE_ROOT = "tests/fixtures/mini_repo"


def _copy_fixture(tmp_path) -> str:
    dest = tmp_path / "raw"
    shutil.copytree(FIXTURE_ROOT, dest)
    return str(dest)


class TesthashDiff:
    def test_no_prior_hash_means_everything_needs_rechunk(self):
        needs, unchanged = hash.diff(None, {"a.py": "h1"}, 2000)
        assert needs == {"a.py"}
        assert unchanged == set()

    def test_matching_hash_is_unchanged(self):
        old = hash.FileHash(max_chunk_size=2000, hashes={"a.py": "h1"})
        needs, unchanged = hash.diff(old, {"a.py": "h1"}, 2000)
        assert needs == set()
        assert unchanged == {"a.py"}

    def test_differing_hash_needs_rechunk(self):
        old = hash.FileHash(max_chunk_size=2000, hashes={"a.py": "h1"})
        needs, unchanged = hash.diff(old, {"a.py": "h2"}, 2000)
        assert needs == {"a.py"}
        assert unchanged == set()

    def test_deleted_file_is_absent_from_both_sets(self):
        old = hash.FileHash(max_chunk_size=2000,
                                hashes={"a.py": "h1", "gone.py": "h9"})
        needs, unchanged = hash.diff(old, {"a.py": "h1"}, 2000)
        assert "gone.py" not in needs
        assert "gone.py" not in unchanged

    def test_max_chunk_size_mismatch_forces_full_rebuild(self):
        old = hash.FileHash(max_chunk_size=2000, hashes={"a.py": "h1"})
        needs, unchanged = hash.diff(old, {"a.py": "h1"}, 500)
        assert needs == {"a.py"}
        assert unchanged == set()


class TestIndexerLoadChunksIncremental:
    def test_first_run_matches_a_full_load_chunks(self, tmp_path):
        raw = _copy_fixture(tmp_path)
        full = Indexer(save_dir=str(tmp_path / "unused"))
        full.load_chunks(raw_data=raw)

        inc = Indexer(save_dir=str(tmp_path / "lexical"))
        inc.load_chunks_incremental(raw_data=raw)

        assert len(inc.metadata) == len(full.metadata)
        assert set((m.file_path, m.first_character_index,
                   m.last_character_index) for m in inc.metadata) == \
               set((m.file_path, m.first_character_index,
                   m.last_character_index) for m in full.metadata)

    def test_unchanged_run_reuses_chunks_without_rechunking(self, tmp_path):
        raw = _copy_fixture(tmp_path)
        indexer = Indexer(save_dir=str(tmp_path / "lexical"))
        indexer.load_chunks_incremental(raw_data=raw)
        indexer.build()
        indexer.save()
        first_metadata = indexer.metadata

        indexer2 = Indexer(save_dir=str(tmp_path / "lexical"))
        with patch.object(PythonChunker, "chunk") as mock_chunk:
            indexer2.load_chunks_incremental(raw_data=raw)
            mock_chunk.assert_not_called()

        assert len(indexer2.metadata) == len(first_metadata)
        assert {(m.file_path, m.first_character_index,
                m.last_character_index) for m in indexer2.metadata} == \
               {(m.file_path, m.first_character_index,
                m.last_character_index) for m in first_metadata}

    def test_changed_file_gets_rechunked(self, tmp_path):
        raw = _copy_fixture(tmp_path)
        indexer = Indexer(save_dir=str(tmp_path / "lexical"))
        indexer.load_chunks_incremental(raw_data=raw)
        indexer.build()
        indexer.save()

        sample_py = tmp_path / "raw" / "sample.py"
        sample_py.write_text(
            '"""Rewritten module."""\n\n'
            'def brand_new_function():\n    return 42\n',
            encoding="utf-8")
        ast.parse(sample_py.read_text(encoding="utf-8"))  # sanity check

        indexer2 = Indexer(save_dir=str(tmp_path / "lexical"))
        indexer2.load_chunks_incremental(raw_data=raw)

        texts = " ".join(indexer2.texts)
        assert "brand_new_function" in texts
        assert "class Greeter" not in texts  # old content is gone

    def test_new_file_is_picked_up(self, tmp_path):
        raw = _copy_fixture(tmp_path)
        indexer = Indexer(save_dir=str(tmp_path / "lexical"))
        indexer.load_chunks_incremental(raw_data=raw)
        indexer.build()
        indexer.save()
        before = len(indexer.metadata)

        (tmp_path / "raw" / "extra.txt").write_text(
            "A brand new plain text file.\n", encoding="utf-8")

        indexer2 = Indexer(save_dir=str(tmp_path / "lexical"))
        indexer2.load_chunks_incremental(raw_data=raw)
        assert len(indexer2.metadata) == before + 1
        assert any(m.file_path.endswith("extra.txt")
                  for m in indexer2.metadata)

    def test_deleted_file_is_removed_from_the_index(self, tmp_path):
        raw = _copy_fixture(tmp_path)
        indexer = Indexer(save_dir=str(tmp_path / "lexical"))
        indexer.load_chunks_incremental(raw_data=raw)
        indexer.build()
        indexer.save()

        (tmp_path / "raw" / "sample.txt").unlink()

        indexer2 = Indexer(save_dir=str(tmp_path / "lexical"))
        indexer2.load_chunks_incremental(raw_data=raw)
        assert not any(m.file_path.endswith("sample.txt")
                      for m in indexer2.metadata)

    def test_max_chunk_size_change_triggers_full_rebuild(self, tmp_path):
        raw = _copy_fixture(tmp_path)
        indexer = Indexer(save_dir=str(tmp_path / "lexical"))
        indexer.load_chunks_incremental(raw_data=raw, max_chunk_size=2000)
        indexer.build()
        indexer.save()

        full = Indexer(save_dir=str(tmp_path / "unused2"))
        full.load_chunks(raw_data=raw, max_chunk_size=50)

        indexer2 = Indexer(save_dir=str(tmp_path / "lexical"))
        indexer2.load_chunks_incremental(raw_data=raw, max_chunk_size=50)
        assert len(indexer2.metadata) == len(full.metadata)


@pytest.fixture
def mocked_st():
    model = MagicMock()
    model.get_embedding_dimension.return_value = 4
    with patch("src.indexing.semantic_encoder.SentenceTransformer") as st_cls:
        st_cls.return_value = model
        yield st_cls, model


class TestSemanticIndexerBuildIncremental:
    def test_first_semantic_run_encodes_everything(self, tmp_path, mocked_st):
        _, model = mocked_st
        raw = _copy_fixture(tmp_path)
        indexer = Indexer(save_dir=str(tmp_path / "unused"))
        indexer.load_chunks_incremental(raw_data=raw)
        model.encode.return_value = np.zeros(
            (len(indexer.metadata), 4), dtype=np.float32)

        sem = SemanticIndexer(indexer, save_dir=str(tmp_path / "semantic"))
        sem.build_incremental()
        assert len(sem.metadata) == len(indexer.metadata)
        args, _ = model.encode.call_args
        assert len(args[0]) == len(indexer.texts)

    def test_second_run_only_encodes_the_changed_file(self, tmp_path, mocked_st):
        _, model = mocked_st
        raw = _copy_fixture(tmp_path)

        indexer = Indexer(save_dir=str(tmp_path / "unused"))
        indexer.load_chunks_incremental(raw_data=raw)
        model.encode.return_value = np.zeros(
            (len(indexer.metadata), 4), dtype=np.float32)
        sem = SemanticIndexer(indexer, save_dir=str(tmp_path / "semantic"))
        sem.build_incremental()
        sem.save()

        sample_py = tmp_path / "raw" / "sample.py"
        sample_py.write_text(
            '"""Rewritten module."""\n\n'
            'def brand_new_function():\n    return 42\n',
            encoding="utf-8")

        indexer2 = Indexer(save_dir=str(tmp_path / "unused"))
        indexer2.load_chunks_incremental(raw_data=raw)
        sem2 = SemanticIndexer(indexer2, save_dir=str(tmp_path / "semantic"))

        new_chunk_count = sum(
            1 for m in indexer2.metadata
            if m.file_path == str(sample_py))
        model.encode.return_value = np.ones(
            (new_chunk_count, 4), dtype=np.float32)

        sem2.build_incremental()

        encode_args, _ = model.encode.call_args
        assert len(encode_args[0]) == new_chunk_count
        assert len(sem2.metadata) == len(indexer2.metadata)

    def test_dimension_mismatch_falls_back_to_full_encode(self, tmp_path, mocked_st):
        _, model = mocked_st
        raw = _copy_fixture(tmp_path)
        indexer = Indexer(save_dir=str(tmp_path / "unused"))
        indexer.load_chunks_incremental(raw_data=raw)

        # Persist a "stale" semantic index with the wrong dimension.
        sem_dir = tmp_path / "semantic"
        sem_dir.mkdir()
        np.save(sem_dir / "semantic_embeddings.npy",
               np.zeros((len(indexer.metadata), 9), dtype=np.float32))
        import json
        with open(sem_dir / "semantic_metadata.json", "w") as f:
            json.dump([m.model_dump() for m in indexer.metadata], f)

        model.encode.return_value = np.zeros(
            (len(indexer.metadata), 4), dtype=np.float32)
        sem = SemanticIndexer(indexer, save_dir=str(sem_dir))
        sem.build_incremental()  # must not raise
        assert sem.embeddings.shape == (len(indexer.metadata), 4)
