import sys
from timeit import default_timer as timer
import torch
from tqdm import tqdm

from src.config import (INDEX_DIR, DATASET_PATH, SEARCH_RESULTS_PATH,
                        SEARCH_RESULTS_SAVE_DIR, ANSWER_SAVE_DIR,
                        GROUND_TRUTH_PATH, GENERATION_FAILED_ANSWER,
                        VALID_METHODS, SEMANTIC_INDEX_DIR)
from src.evaluation.evaluator import Evaluator
from src.generation.generator import Generator
from src.indexing.indexer import Indexer, IndexingError
from src.indexing.semantic_indexer import (SemanticIndexer,
                                           SemanticIndexingError)
from src.models import (MinimalSearchResults, MinimalAnswer, MinimalSource,
                        StudentSearchResults, StudentSearchResultsAndAnswer)
from src.retrieval.query_cache import QueryCache
from src.retrieval.retriever import Retriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.json_io import (load_dataset_unanswered, save_search_result,
                               load_search_results)


class CLI:
    """Handles the CLI."""

    @staticmethod
    def index(max_chunk_size: int = 2000,
              method: str = "lexical") -> None:
        """Index the corpus."""
        if not isinstance(max_chunk_size, int):
            print("Error: max_chunk_size must be an integer.",
                  file=sys.stderr)
            exit(1)
        if max_chunk_size < 1 or max_chunk_size > 2000:
            print("Error: "
                  "max_chunk_size must be between 1 and 2000.",
                  file=sys.stderr)
            exit(1)
        if not check_method_validity(method):
            exit(1)

        start = timer()
        indexer = Indexer(INDEX_DIR)
        indexer.load_chunks_incremental(max_chunk_size=max_chunk_size)
        if method.lower() == "semantic" or method.lower() == "hybrid":
            sem_indexer = SemanticIndexer(indexer, SEMANTIC_INDEX_DIR)
            if not indexer.has_changes \
                    and sem_indexer.embeddings_path.exists():
                print("Semantic index unchanged. Skipping rebuild.")
            else:
                try:
                    sem_indexer.build_incremental()
                    sem_indexer.save()
                    print("Ingestion complete! "
                          f"Semantically indexed {len(sem_indexer.metadata)} "
                          f"chunks under {SEMANTIC_INDEX_DIR}")
                except SemanticIndexingError as e:
                    print(e, file=sys.stderr)
                    exit(0)

        if method.lower() == "lexical" or method.lower() == "hybrid":
            if not indexer.has_changes and indexer.save_path.exists():
                print("Lexical index unchanged. Skipping rebuild.")
            else:
                try:
                    indexer.build()
                    indexer.save()
                    print("Ingestion complete! "
                          f"Lexically indexed {len(indexer.metadata)} chunks "
                          f"under {INDEX_DIR}")
                except IndexingError as e:
                    print(e, file=sys.stderr)
                    exit(1)
        end = timer()
        print(f"Processing time: {end - start:.2f}s")

    @staticmethod
    def search(query: str, k: int = 5,
               method: str = "lexical") -> None:
        """Search sources for a single query."""
        if not check_k_validity(k):
            exit(1)
        if not isinstance(query, str):
            print("Error: Query must be a valid string.",
                  file=sys.stderr)
            exit(1)
        if not check_method_validity(method):
            exit(1)

        start = timer()
        cache = QueryCache()
        results = cache.get(method, k, query) if cache is not None else None
        if results is None:
            retriever = make_retriever([query], k, method)
            results = retriever.retrieve()[0]
            if cache is not None:
                cache.put(method, k, query, results)
                cache.save()

        for res in results:
            print(res.file_path + " [" + str(res.first_character_index)
                  + ":" + str(res.last_character_index) + "]")
        end = timer()
        processing_time = end - start
        print(f"Processing time: {processing_time:.2f}s.")

    @staticmethod
    def search_dataset(dataset_path: str = DATASET_PATH,
                       k: int = 5,
                       save_directory: str = SEARCH_RESULTS_SAVE_DIR,
                       method: str = "lexical") -> None:
        """
        Search sources for a whole dataset
        and save the results in save_directory.
        """
        if not check_k_validity(k):
            exit(1)
        if not check_method_validity(method):
            exit(1)

        start = timer()
        question_sets = load_dataset_unanswered(dataset_path)
        question_num = len(question_sets)
        queries = [q.question for q in question_sets]

        cache = QueryCache()
        hits: dict[int, list[MinimalSource]] = {}
        miss_indices: list[int] = []
        miss_queries: list[str] = []
        for i, q in enumerate(tqdm(queries, desc="Loading from cache")):
            hit = cache.get(method, k, q) if cache is not None else None
            if hit is not None:
                hits[i] = hit
            else:
                miss_indices.append(i)
                miss_queries.append(q)

        if len(miss_queries) > 0:
            retriever = make_retriever(queries, k, method)
            for idx, q, res in tqdm(zip(miss_indices, miss_queries,
                                    retriever.retrieve()),
                                    total=len(miss_queries),
                                    desc="Searching"):
                hits[idx] = res
                if cache is not None:
                    cache.put(method, k, q, res)
            if cache is not None:
                cache.save()

        if cache is not None:
            hit_count = question_num - len(miss_queries)
            print(f"Query cache: {hit_count}/{question_num} hit")

        results = [hits[i] for i in range(question_num)]
        result_sets = []
        for question, sources in tqdm(zip(question_sets, results),
                                      desc="Saving search results",
                                      total=len(results)):
            result_sets.append(MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=sources
            ))
        student_results = StudentSearchResults(
            search_results=result_sets,
            k=k
        )

        save_search_result(student_results, dataset_path, save_directory)
        end = timer()
        processing_time = end - start
        print(f"Processed {question_num} questions in "
              f"{processing_time:.2f}s. "
              "Retireval throughput: "
              f"{processing_time / question_num * 200:.2f}s"
              "/200 questions")

    @staticmethod
    def answer(query: str, k: int = 5, method: str = "lexical") -> None:
        """Answer a single query using the retrieved context."""
        if not check_k_validity(k):
            exit(1)
        if not isinstance(query, str):
            print("Error: Query must be a valid string.",
                  file=sys.stderr)
            exit(1)
        if not check_method_validity(method):
            exit(1)

        start = timer()
        retriever = make_retriever([query], k, method)
        sources = retriever.retrieve()[0]
        print("\nSources:")
        for res in sources:
            print(res.file_path + " [" + str(res.first_character_index)
                  + ":" + str(res.last_character_index) + "]")
        generator = Generator()
        print("\nAnswer:\n", generator.generate_answer(query, sources))
        end = timer()
        print(f"\nAnswer generation time: {end - start:.2f}s.")

    @staticmethod
    def answer_dataset(student_search_results_path: str = SEARCH_RESULTS_PATH,
                       save_directory: str = ANSWER_SAVE_DIR) -> None:
        """
        Answer a whole dataset using the sources found.
        Save the results under save_directory.
        """
        search_res = load_search_results(student_search_results_path)
        generator = Generator()
        answers: list[MinimalAnswer] = []

        for res in tqdm(search_res.search_results, desc="Generating answers"):
            torch.cuda.empty_cache()
            try:
                answer = generator.generate_answer(res.question,
                                                   res.retrieved_sources)
            except RuntimeError as e:
                print("Warning: Generation failed for question "
                      f"'{res.question_id}': {e}", file=sys.stderr)
                answer = GENERATION_FAILED_ANSWER
            answers.append(MinimalAnswer(
                question_id=res.question_id,
                question=res.question,
                retrieved_sources=res.retrieved_sources,
                answer=answer
            ))

        out = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=search_res.k
        )

        save_search_result(out, student_search_results_path, save_directory)

    @staticmethod
    def evaluate(student_search_results_path: str = SEARCH_RESULTS_PATH,
                 dataset_path: str = GROUND_TRUTH_PATH) -> None:
        """
        Evaluate search quality using recall@k
        against the ground-truth dataset.
        """
        print(f"Evaluating '{student_search_results_path}' "
              f"against '{dataset_path}'...")
        evaluator = Evaluator(student_search_results_path, dataset_path)
        print(f"Recall@{evaluator.k}: {evaluator.mean_recall:.3f} "
              f"for {len(evaluator.matched_res)} questions.")


def check_k_validity(k: int) -> bool:
    if not isinstance(k, int):
        print("Error: k must be an integer.",
              file=sys.stderr)
        return False
    if k < 1:
        print("Error: "
              "k must be at least 1.",
              file=sys.stderr)
        return False
    return True


def check_method_validity(method: str) -> bool:
    if not isinstance(method, str) \
            or method.lower() not in VALID_METHODS:
        print("Error: method must be one of "
              f"{sorted(VALID_METHODS)}.",
              file=sys.stderr)
        return False
    return True


def make_retriever(queries: list[str], k: int = 5,
                   method: str = "lexical") \
            -> Retriever | SemanticRetriever | HybridRetriever:
    if method.lower() == "semantic":
        return SemanticRetriever(queries, k)
    if method.lower() == "hybrid":
        return HybridRetriever(queries, k)
    return Retriever(queries, k)
