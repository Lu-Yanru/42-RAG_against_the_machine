"""
Reciprocal Rank Fusion of BM25 and semantic retrieval.
score(doc) = sum(1 / (RRF_K + rank + 1))
"""


from src.config import RRF_K
from src.models import MinimalSource
from src.retrieval.retriever import Retriever
from src.retrieval.semantic_retriever import SemanticRetriever


class HybridRetriever:
    def __init__(self, queries: list[str], k: int = 5) -> None:
        self.queries = queries
        self.k = k
        self.lexical = Retriever(queries, k)
        self.semantic = SemanticRetriever(queries, k)

    def retrieve(self) -> list[list[MinimalSource]]:
        if not self.queries:
            return []

        lexical_res = self.lexical.retrieve()
        semantic_res = self.semantic.retrieve()

        output: list[list[MinimalSource]] = []
        for query, lex_list, sem_list in zip(
                self.queries,  lexical_res, semantic_res):
            if not query.strip():
                output.append([])
                continue
            output.append(self.fuse(lex_list, sem_list))
        return output

    def fuse(self, lex_list: list[MinimalSource],
             sem_list: list[MinimalSource]) -> list[MinimalSource]:
        """Fuse two retrieved list of sources by rank."""
        scores: dict[tuple[str, int, int], float] = {}
        by_key: dict[tuple[str, int, int], MinimalSource] = {}

        for ranked_list in (lex_list, sem_list):
            for rank, source in enumerate(ranked_list):
                key = HybridRetriever.key(source)
                scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
                by_key[key] = source

        ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [by_key[k] for k in ranked[:self.k]]

    @staticmethod
    def key(source: MinimalSource) -> tuple[str, int, int]:
        """Create dict key from a minimal source."""
        return (source.file_path, source.first_character_index,
                source.last_character_index)
