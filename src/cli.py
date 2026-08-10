import json
from pathlib import Path
from pydantic import ValidationError
import sys
from timeit import default_timer as timer

from src.evaluation.evaluator import Evaluator
from src.indexing.indexer import Indexer
from src.models import (MinimalSearchResults,
                        StudentSearchResults, UnansweredQuestion)
from src.retrieval.retriever import Retriever


INDEX_DIR = "data/processed"
DATASET_PATH = "data/datasets/UnansweredQuestions/dataset_docs_public.json"
SEARCH_RESULTS_SAVE_DIR = "data/output/search_results"
SEARCH_RESULTS_PATH = ("data/output/search_results/dataset_docs_public.json")
ANSWER_SAVE_DIR = "data/output/search_results_and_answer"
GROUND_TRUTH_PATH = "data/datasets/AnsweredQuestions/dataset_docs_public.json"


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
        if not isinstance(k, int):
            print("Error: k must be an integer.",
                  file=sys.stderr)
            exit(1)
        if k < 1:
            print("Error: "
                  "k must be at least 1.",
                  file=sys.stderr)
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
        if not isinstance(k, int):
            print("Error: k must be an integer.",
                  file=sys.stderr)
            exit(1)
        if k < 1:
            print("Error: "
                  "k must be at least 1.",
                  file=sys.stderr)
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
              f"{question_num/200 * processing_time:.2f}s"
              "/200 questions")

    @staticmethod
    def answer(query: str, k: int = 5) -> None:
        """Answer a single query using the retrieved context."""
        print(f"query: {query}")
        print(f"k: {k}")
        print("Answer result...")

    @staticmethod
    def answer_dataset(student_search_results_path: str = SEARCH_RESULTS_PATH,
                       save_directory: str = ANSWER_SAVE_DIR) -> None:
        """
        Answer a whole dataset using the sources found.
        Save the results under save_directory.
        """
        print(f"student_search_results_path: {student_search_results_path}")
        print(f"Saved student_search_results_and_answer to {save_directory}")

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


def load_dataset_unanswered(dataset_path: str = DATASET_PATH) \
        -> list[UnansweredQuestion]:
    """Load unanswered questions."""
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"Error loading dataset '{dataset_path}'. "
              "Exiting...",
              file=sys.stderr)
        exit(1)

    try:
        queries = [UnansweredQuestion.model_validate(q)
                   for q in raw["rag_questions"]]
        return queries
    except (ValidationError, KeyError, TypeError):
        print(f"Error loading dataset '{dataset_path}'. "
              "Exiting...",
              file=sys.stderr)
        exit(1)


def save_search_result(res: StudentSearchResults,
                       dataset_path: str = DATASET_PATH,
                       save_dir: str = SEARCH_RESULTS_SAVE_DIR) -> None:
    """Save search results to a json file."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(dataset_path).name

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res.model_dump(), f, indent=4)
        print(f"Saved student_search_results to {out_path}")
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error: Failed to save search results: {e} "
              "Exiting...",
              file=sys.stderr)
        exit(1)
