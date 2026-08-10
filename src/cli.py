from dotenv import load_dotenv
import os
import sys
from timeit import default_timer as timer
from tqdm import tqdm

from src.config import (INDEX_DIR, DATASET_PATH, SEARCH_RESULTS_PATH,
                        SEARCH_RESULTS_SAVE_DIR, ANSWER_SAVE_DIR,
                        GROUND_TRUTH_PATH, GENERATION_FAILED_ANSWER)
from src.evaluation.evaluator import Evaluator
from src.generation.generator import Generator
from src.indexing.indexer import Indexer
from src.models import (MinimalSearchResults, MinimalAnswer,
                        StudentSearchResults, StudentSearchResultsAndAnswer)
from src.retrieval.retriever import Retriever
from src.utils.json_io import (load_dataset_unanswered, save_search_result,
                               load_search_results)


class CLI:
    """Handles the CLI."""

    @staticmethod
    def index(max_chunk_size: int = 2000) -> None:
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

        start = timer()
        indexer = Indexer(INDEX_DIR)
        indexer.build(max_chunk_size=max_chunk_size)
        indexer.save()
        print("Ingestion complete! "
              f"Indexed {len(indexer.metadata)} chunks under {INDEX_DIR}")
        end = timer()
        print(f"Processing time: {end - start:.2f}s")

    @staticmethod
    def search(query: str, k: int = 5) -> None:
        """Search sources for a single query."""
        if not check_k_validity(k):
            exit(1)

        retriever = Retriever([query], k)
        results = retriever.retrieve()[0]
        for res in results:
            print(res.file_path + " [" + str(res.first_character_index)
                  + ":" + str(res.last_character_index) + "]")

    @staticmethod
    def search_dataset(dataset_path: str = DATASET_PATH,
                       k: int = 5,
                       save_directory: str = SEARCH_RESULTS_SAVE_DIR) -> None:
        """
        Search sources for a whole dataset
        and save the results in save_directory.
        """
        if not check_k_validity(k):
            exit(1)

        start = timer()
        question_sets = load_dataset_unanswered(dataset_path)
        question_num = len(question_sets)
        queries = [q.question for q in question_sets]

        retriever = Retriever(queries, k)
        results = retriever.retrieve()
        result_sets = []
        for question, sources in zip(question_sets, results):
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
    def answer(query: str, k: int = 5) -> None:
        """Answer a single query using the retrieved context."""
        if not check_k_validity(k):
            exit(1)

        # Load HF token
        if os.path.exists(".env"):
            load_dotenv()

        start = timer()
        retriever = Retriever([query], k)
        sources = retriever.retrieve()[0]
        generator = Generator()
        print(generator.generate_answer(query, sources))
        end = timer()
        print(f"Answer generation time: {end - start:.2f}s.")

    @staticmethod
    def answer_dataset(student_search_results_path: str = SEARCH_RESULTS_PATH,
                       save_directory: str = ANSWER_SAVE_DIR) -> None:
        """
        Answer a whole dataset using the sources found.
        Save the results under save_directory.
        """
        # Load HF token
        if os.path.exists(".env"):
            load_dotenv()

        start = timer()
        search_res = load_search_results(student_search_results_path)
        generator = Generator()
        answers: list[MinimalAnswer] = []

        for res in tqdm(search_res.search_results, desc="Generating answers"):
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
        end = timer()
        print(f"Answer generation time: {end - start:.2f}s.")

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
        print(f"Recall@{evaluator.k}: {evaluator.mean_recall:.2f} "
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
