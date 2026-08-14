import numpy as np
import sys


from src.indexing.indexer import Indexer
from src.indexing.semantic_encoder import SemanticEncoder
from src.indexing.semantic_indexer import (SemanticIndexer,
                                           SemanticIndexingError)
from src.retrieval.retriever import Retriever
from src.models import MinimalSource


class SemanticRetriever:
    def __init__(self, queries: list[str], k: int = 5) -> None:
        indexer = Indexer()
        self.sem_indexer = SemanticIndexer(indexer)
        try:
            self.sem_indexer.load()
            print("Successfully loaded semantic indexes.")
        except SemanticIndexingError as e:
            print(e, file=sys.stderr)
            exit(1)

        self.encoder = SemanticEncoder()
        self.queries = queries

        num_metadata = len(self.sem_indexer.metadata)
        if k > num_metadata:
            print("Warning: k is larger than the number of available scores, "
                  f"which is {num_metadata}. "
                  f"Retrieving only {num_metadata} sources.")
            self.k = num_metadata
        else:
            self.k = k

    def retrieve(self) -> list[list[MinimalSource]]:
        """
        Retrieve the top-k semantically-nearest sources per query.
        Encoding is batched across every non-degenerate query in one
        call instead of looping per-question.
        """
        if not self.queries:
            return []

        output: list[list[MinimalSource]] = [[] for _ in self.queries]
        # keep only non-empty queries
        live_idx = [i for i, q in enumerate(self.queries) if q.strip()]
        live_queries = [self.queries[i] for i in live_idx]

        if live_queries:
            # numpy array of shape (num_live_queries, dimensions_of_vector)
            query_vecs = self.encoder.encode(live_queries)
            # Transpose to (score, num_chunks)
            # Both sides are L2-normalized, so a dot product is
            # cosine similarity.
            scores = query_vecs @ self.sem_indexer.embeddings.T

            for local_i, global_i in enumerate(live_idx):
                # returns the indices that would sort the array,
                # ascending by default -> [::-1] to reverse
                # [:self.k] take the top k indices
                top_idx = np.argsort(scores[local_i])[::-1][:self.k]
                output[global_i] = [self.sem_indexer.metadata[j]
                                    for j in top_idx]

        if Retriever.output_is_empty(output):
            print("No relevant source found.")

        return output
