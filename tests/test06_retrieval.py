from src.indexing.indexer import Indexer
from src.retrieval.retriever import Retriever

FIXTURE_ROOT = "tests/fixtures/mini_repo"


def _built_retriever(queries: list[str], k: int, max_chunk_size: int = 2000) -> Retriever:
    indexer = Indexer(save_dir="unused")
    indexer.build(raw_data=FIXTURE_ROOT, max_chunk_size=max_chunk_size)
    return Retriever(queries, k)


class TestRetrieveSingleQuery:
    def test_targeted_query_ranks_the_known_chunk_first(self):
        retriever = _built_retriever(["What does the Greeter class's greet method return?"], k=3)
        sources = retriever.retrieve()
        assert len(sources[0]) > 0
        assert sources[0][0].file_path.endswith("utils.py")

    def test_k_zero_returns_empty_list(self):
        retriever = _built_retriever(["Greeter"], k=0)
        sources = retriever.retrieve()
        assert sources[0] == []

    def test_empty_string_query_returns_empty_list(self):
        retriever = _built_retriever([""], k=5)
        sources = retriever.retrieve()
        assert sources[0] == []

    def test_whitespace_only_query_returns_empty_list(self):
        retriever = _built_retriever(["   "], k=5)
        sources = retriever.retrieve()
        assert sources[0] == []

    def test_all_stopword_query_returns_empty_list(self):
        # Regression guard: bm25s tokenizes an all-stopword query down to
        # a single id-0 ("" vocab entry) rather than an empty token list.
        # A naive `len(ids) == 0` check would miss this and let retrieve()
        # return arbitrary zero-score chunks instead of [].
        retriever = _built_retriever(["the a is of and"], k=5)
        sources = retriever.retrieve()
        assert sources[0] == []

    def test_k_larger_than_corpus_size_does_not_crash(self):
        retriever = _built_retriever(["Greeter helper"], k=1000000)
        corpus_size = len(retriever.indexer.metadata)
        sources = retriever.retrieve()
        assert len(sources[0]) == corpus_size

    def test_nonsensical_query_does_not_crash(self):
        retriever = _built_retriever(["zzxxqqjjvvbbwwyyy asdkjfh"], k=5)
        sources = retriever.retrieve()
        assert isinstance(sources[0], list)


class TestRetrieveBatch:

    def test_batch_preserves_query_order(self):
        queries = [
            "What does the Greeter class's greet method return?",
            "How does the sample markdown doc describe usage?",
        ]
        retriever = _built_retriever(queries, k=3)
        results = retriever.retrieve()
        assert len(results) == 2
        assert results[0][0].file_path.endswith(".py")
        assert results[1][0].file_path.endswith(".txt")

    def test_degenerate_query_in_a_batch_does_not_contaminate_other_rows(self):
        queries = [
            "the a is of and",  # degenerate: should resolve to []
            "What does the Greeter class's greet method return?",
        ]
        retriever = _built_retriever(queries, k=3)
        results = retriever.retrieve()
        assert results[0] == []
        assert len(results[1]) > 0
        assert results[1][0].file_path.endswith("utils.py")

    def test_empty_queries_list_returns_empty_list(self):
        retriever = _built_retriever([], k=5)
        assert retriever.retrieve() == []

    def test_k_zero_returns_empty_lists_for_every_query(self):
        retriever = _built_retriever(["Greeter", "helper"], k=0)
        results = retriever.retrieve()
        assert results == [[], []]


class TestRetrieveResultShape:
    def test_results_are_minimal_source_instances_with_valid_offsets(self):
        retriever = _built_retriever("Greeter helper function", k=5)
        sources = retriever.retrieve()
        for s in sources[0]:
            assert s.first_character_index < s.last_character_index

    def test_no_more_than_k_results_returned(self):
        retriever = _built_retriever("Greeter helper function", k=2)
        sources = retriever.retrieve()
        assert len(sources[0]) <= 2
