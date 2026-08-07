import json

import pytest

from src.indexing.indexer import IndexingError, Indexer
from src.indexing.tokenizer import build_tokenizer
from src.ingest.chunking import Chunk
from src.ingest.chunking_python import PythonChunker
from src.ingest.chunking_text import MarkdownChunker, TextChunker
from src.ingest.loader import Loader
from src.models import MinimalSource


class TestTokenizer:
    def test_lowercases_tokens(self):
        tok = build_tokenizer()
        result = tok.tokenize(["Hello WORLD"], return_as="string",
                              show_progress=False)
        assert result == [["hello", "world"]]

    def test_keeps_snake_case_identifier_whole(self):
        # Regression guard for the "start simple, don't pre-split
        # identifiers" decision -- if this ever starts failing because
        # someone added camelCase/snake_case splitting, that's a real
        # design change, not an accidental regression, so update this
        # test deliberately rather than "fixing" the tokenizer.
        tok = build_tokenizer()
        result = tok.tokenize(["What does VLLM_RPC_TIMEOUT default to?"],
                              return_as="string", show_progress=False)
        assert "vllm_rpc_timeout" in result[0]
        assert "vllm" not in result[0]
        assert "timeout" not in result[0]

    def test_does_not_stem(self):
        tok = build_tokenizer()
        result = tok.tokenize(["running runs"], return_as="string",
                              show_progress=False)
        assert "running" in result[0]
        assert "runs" in result[0]
        assert "run" not in result[0]

    def test_two_tokenizer_instances_produce_the_same_ids_for_the_same_word(self):
        # This is the property retrieval correctness depends on: the
        # index-time tokenizer and a freshly-constructed query-time
        # tokenizer must assign identical word->id mappings once vocab
        # is loaded from disk. We test the pre-persistence half here;
        # the full disk round trip is covered in TestIndexer.
        tok_a = build_tokenizer()
        tok_b = build_tokenizer()
        ids_a = tok_a.tokenize(["Greeter"], return_as="ids",
                               show_progress=False)
        ids_b = tok_b.tokenize(["Greeter"], return_as="ids",
                               show_progress=False)
        assert ids_a == ids_b


def _all_mini_repo_chunks(fixture_root: str, max_chunk_size: int = 2000) \
        -> list[Chunk]:
    loader = Loader(raw_path=fixture_root)
    chunks: list[Chunk] = []
    chunks += TextChunker(loader.txt_files, max_chunk_size).chunk()
    chunks += MarkdownChunker(loader.md_files, max_chunk_size).chunk()
    chunks += PythonChunker(loader.py_files, max_chunk_size).chunk()
    return chunks


FIXTURE_ROOT = "tests/fixtures/mini_repo"


class TestIndexerBuild:
    def test_build_produces_one_metadata_entry_per_chunk(self):
        chunks = _all_mini_repo_chunks(FIXTURE_ROOT)
        builder = Indexer(save_dir="unused")
        builder.build(raw_data=FIXTURE_ROOT)
        assert len(builder.metadata) == len(chunks)
        assert all(isinstance(m, MinimalSource) for m in builder.metadata)

    def test_metadata_order_matches_chunk_order(self):
        chunks = _all_mini_repo_chunks(FIXTURE_ROOT)
        builder = Indexer(save_dir="unused")
        builder.build(raw_data=FIXTURE_ROOT)
        for chunk, source in zip(chunks, builder.metadata):
            assert source.file_path == chunk.file_path
            assert source.first_character_index == chunk.first_character_index
            assert source.last_character_index == chunk.last_character_index


class TestIndexerPersistRoundTrip:
    def test_persisted_metadata_has_no_chunk_text(self, tmp_path):
        # Enforces the "store only offsets, never chunk text" decision.
        chunks = _all_mini_repo_chunks(FIXTURE_ROOT)
        builder = Indexer(save_dir=str(tmp_path))
        builder.build(raw_data=FIXTURE_ROOT)
        builder.save()

        with open(tmp_path / "chunk_metadata.json", encoding="utf-8") as f:
            raw = json.load(f)

        assert len(raw) == len(chunks)
        for entry in raw:
            assert set(entry.keys()) == {
                "file_path", "first_character_index", "last_character_index"
            }

    def test_reloaded_index_returns_the_relevant_chunk(self, tmp_path):
        chunks = _all_mini_repo_chunks(FIXTURE_ROOT)
        builder = Indexer(save_dir=str(tmp_path))
        builder.build(raw_data=FIXTURE_ROOT)
        builder.save()

        # Fresh Indexer instance simulates a new process loading
        # from disk -- must not depend on any in-memory state from build().
        reloaded_builder = Indexer(save_dir=str(tmp_path))
        reloaded_builder.load()

        assert len(reloaded_builder.metadata) == len(chunks)

        query_tokens = reloaded_builder.tokenizer.tokenize(
            ["What does the Greeter class's greet method return?"],
            return_as="tuple", update_vocab=False, show_progress=False)
        results, _scores = reloaded_builder.retriever.retrieve(query_tokens, k=3,
                                              show_progress=False)
        top_indices = results[0].tolist()
        top_sources = [reloaded_builder.metadata[i] for i in top_indices]
        assert any(s.file_path.endswith("sample.py") for s in top_sources)

    def test_load_raises_before_anything_has_been_indexed(self, tmp_path):
        builder = Indexer(save_dir=str(tmp_path))
        with pytest.raises(IndexingError):
            builder.load()

    def test_reindexing_overwrites_the_previous_index(self, tmp_path):
        # Regression guard: a second `index` run must fully replace the
        # first, not append to or corrupt it.
        chunks_a = _all_mini_repo_chunks(FIXTURE_ROOT)
        builder = Indexer(save_dir=str(tmp_path))
        builder.build(raw_data=FIXTURE_ROOT)
        builder.save()

        chunks_b = _all_mini_repo_chunks(FIXTURE_ROOT, max_chunk_size=50)
        builder.build(FIXTURE_ROOT, max_chunk_size=50)
        builder.save()

        builder.load()
        assert len(builder.metadata) == len(chunks_b)