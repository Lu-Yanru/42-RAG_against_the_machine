import sys

from src.indexing.indexer import Indexer, IndexingError
from src.models import MinimalSource


class Retriever:
    def __init__(self, queries: list[str], k: int = 5) -> None:
        self.indexer = Indexer()
        try:
            self.indexer.load()
            print("Successfully loaded indexes.")
        except IndexingError as e:
            print(e, file=sys.stderr)
            exit(1)

        self.queries = queries

        num_metadata = len(self.indexer.metadata)
        if k > num_metadata:
            print("Warning: k is larger than the number of available scores, "
                  f"which is {num_metadata}. "
                  f"Retrieving only {num_metadata} sources.")
            self.k = num_metadata
        else:
            self.k = k

    def retrieve(self) -> list[list[MinimalSource]]:
        """
        Retrieve the top-k sources for each query in `queries`.
        Batches tokenization and scoring into one bm25s call instead of
        looping per-question.
        """
        if not self.queries:
            return []

        tokenized = self.indexer.tokenizer.tokenize(
            self.queries,
            return_as="tuple",
        )

        # bm25s pads an all-stopword/out-of-vocabulary/empty query down to
        # a single id 0 (the reserved "" vocab entry) instead of an empty
        # token list. Left unchecked, retrieve() still runs on that query
        # and returns arbitrary documents with score 0.0 for every one of
        # them -- not the [] the CLI's edge-case handling requires. Detect
        # that degenerate case explicitly per query.
        is_degenerate = [all(token_id == 0 for token_id in ids)
                         for ids in tokenized.ids]

        results, scores = self.indexer.retriever.retrieve(
            tokenized, k=self.k
        )

        output: list[list[MinimalSource]] = []
        for row, doc_indices in enumerate(results):
            if is_degenerate[row]:
                output.append([])
            else:
                output.append([self.indexer.metadata[i] for i in doc_indices])

        if Retriever.output_is_empty(output):
            print("No relevant source found.")

        return output

    @staticmethod
    def output_is_empty(output: list[list[MinimalSource]]) -> bool:
        if not output:
            return True
        for lst in output:
            if lst:
                return False
        return True
