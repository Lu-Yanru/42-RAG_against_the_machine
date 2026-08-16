"""Tests for HybridRetriever.

Split into three tiers by how much of the real object each test
needs:
  - Key/fuse math: HybridRetriever.__new__ bypasses __init__ entirely
    (fuse/key only read self.k, never touch self.lexical/self.semantic),
    so RRF math is tested with zero mocking of either sub-retriever.
  - retrieve()-level orchestration: same __new__ bypass, but
    self.lexical/self.semantic are MagicMocks with a scripted
    .retrieve() return value -- tests the blank-query short circuit
    and result ordering without needing real indices.
  - Construction: __init__'s real wiring is checked by patching
    Retriever/SemanticRetriever as imported into hybrid_retriever.py.
"""
from unittest.mock import MagicMock, patch

from src.indexing.hash import chunk_key
from src.models import MinimalSource
from src.retrieval.hybrid_retriever import HybridRetriever


def _source(name: str) -> MinimalSource:
    return MinimalSource(file_path=name, first_character_index=0,
                         last_character_index=10)


def _hybrid_for_fuse(k: int) -> HybridRetriever:
    instance = HybridRetriever.__new__(HybridRetriever)
    instance.k = k
    return instance


def _hybrid_with_fakes(queries: list[str], k: int,
                       lexical_results: list[list[MinimalSource]],
                       semantic_results: list[list[MinimalSource]]) \
        -> HybridRetriever:
    instance = HybridRetriever.__new__(HybridRetriever)
    instance.queries = queries
    instance.k = k
    instance.lexical = MagicMock()
    instance.lexical.retrieve.return_value = lexical_results
    instance.semantic = MagicMock()
    instance.semantic.retrieve.return_value = semantic_results
    return instance


class TestKey:
    def test_two_equal_but_distinct_sources_produce_the_same_key(self):
        a = _source("x.py")
        b = _source("x.py")
        assert a is not b
        assert chunk_key(a) == chunk_key(b)

    def test_different_offsets_produce_different_keys(self):
        a = MinimalSource(file_path="x.py", first_character_index=0,
                          last_character_index=10)
        b = MinimalSource(file_path="x.py", first_character_index=5,
                          last_character_index=15)
        assert chunk_key(a) != chunk_key(b)


class TestFuseRRFMath:
    def test_doc_only_in_lexical_gets_a_score(self):
        hybrid = _hybrid_for_fuse(k=5)
        a = _source("a.py")
        assert hybrid.fuse([a], []) == [a]

    def test_doc_only_in_semantic_gets_a_score(self):
        hybrid = _hybrid_for_fuse(k=5)
        a = _source("a.py")
        assert hybrid.fuse([], [a]) == [a]

    def test_doc_in_both_lists_outranks_a_doc_in_only_one(self):
        hybrid = _hybrid_for_fuse(k=5)
        a, b = _source("a.py"), _source("b.py")
        # a: rank 0 in both lists (1/61 + 1/61). b: rank 1 lexical only (1/62).
        result = hybrid.fuse([a, b], [a])
        assert result == [a, b]

    def test_same_chunk_from_two_independently_built_metadata_lists_fuses_into_one_entry(self):
        # Regression guard: two objects, identical field values,
        # different identity -- exactly what BM25's and semantic's
        # independently persisted metadata files produce for the
        # same underlying chunk. Must count as one document, not two.
        hybrid = _hybrid_for_fuse(k=5)
        lex_copy = _source("shared.py")
        sem_copy = _source("shared.py")
        assert lex_copy is not sem_copy

        result = hybrid.fuse([lex_copy], [sem_copy])
        assert len(result) == 1

    def test_result_truncated_to_k(self):
        hybrid = _hybrid_for_fuse(k=2)
        sources = [_source(f"{i}.py") for i in range(5)]
        assert len(hybrid.fuse(sources, [])) == 2

    def test_empty_both_lists_returns_empty(self):
        hybrid = _hybrid_for_fuse(k=5)
        assert hybrid.fuse([], []) == []


class TestHybridRetrieve:
    def test_empty_queries_list_returns_empty_list(self):
        hybrid = _hybrid_with_fakes([], k=5, lexical_results=[],
                                    semantic_results=[])
        assert hybrid.retrieve() == []

    def test_blank_query_short_circuits_without_breaking_alignment(self):
        a = _source("a.py")
        hybrid = _hybrid_with_fakes(
            ["   ", "real query"], k=5,
            lexical_results=[[a], [a]], semantic_results=[[a], [a]])
        results = hybrid.retrieve()
        assert results[0] == []
        assert results[1] == [a]

    def test_all_stopword_lexical_result_does_not_force_hybrid_to_empty(self):
        # BM25 returning [] for a non-blank query (e.g. all-stopword)
        # must not zero out hybrid entirely -- only a blank query
        # string itself short-circuits. Semantic's ranking still
        # applies via RRF over a single non-empty list.
        a = _source("a.py")
        hybrid = _hybrid_with_fakes(
            ["the a is of and"], k=5,
            lexical_results=[[]], semantic_results=[[a]])
        assert hybrid.retrieve()[0] == [a]

    def test_result_order_matches_query_order(self):
        a, b = _source("a.py"), _source("b.py")
        hybrid = _hybrid_with_fakes(
            ["q1", "q2"], k=5,
            lexical_results=[[a], [b]], semantic_results=[[], []])
        results = hybrid.retrieve()
        assert results[0] == [a]
        assert results[1] == [b]


class TestHybridRetrieverConstruction:
    def test_constructs_both_sub_retrievers_with_the_same_queries_and_k(self):
        queries = ["q1", "q2"]
        with patch("src.retrieval.hybrid_retriever.Retriever") as retriever_cls, \
             patch("src.retrieval.hybrid_retriever.SemanticRetriever") as sem_cls:
            HybridRetriever(queries, k=7)
        retriever_cls.assert_called_once_with(queries, 7)
        sem_cls.assert_called_once_with(queries, 7)
