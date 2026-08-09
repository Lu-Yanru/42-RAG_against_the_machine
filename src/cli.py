import sys
from timeit import default_timer as timer

from src.indexing.indexer import Indexer
from src.retrieval.retriever import Retriever


INDEX_DIR = "data/processed"
DATASET_PATH = "data/datasets/UnansweredQuestions/dataset_docs_public.json"
SEARCH_RESULTS_SAVE_DIR = "data/output/search_results/UnansweredQuestions"
SEARCH_RESULTS_PATH = ("data/output/search_results/UnansweredQuestions/"
                       "dataset_docs_public.json")
ANSWER_SAVE_DIR = "data/output/search_results_and_answer/UnansweredQuestions"


class CLI:
    """Handles the CLI."""

    @staticmethod
    def index(max_chunk_size: int = 2000) -> None:
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
        print(f"Ingestion complete! Indices saved under {INDEX_DIR}")
        end = timer()
        print(f"Processing time: {end - start:.2f}s")

    @staticmethod
    def search(query: str, k: int = 5) -> None:
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
        print(f"dataset_path: {dataset_path}")
        print(f"k: {k}")
        print(f"Saved student_search_results to {save_directory}")

    @staticmethod
    def answer(query: str, k: int = 5) -> None:
        print(f"query: {query}")
        print(f"k: {k}")
        print("Answer result...")

    @staticmethod
    def answer_dataset(student_search_results_path: str = SEARCH_RESULTS_PATH,
                       save_directory: str = ANSWER_SAVE_DIR) -> None:
        print(f"student_search_results_path: {student_search_results_path}")
        print(f"Saved student_search_results_and_answer to {save_directory}")

    @staticmethod
    def evaluate(student_search_results_path: str = SEARCH_RESULTS_PATH,
                 dataset_path: str = DATASET_PATH,
                 k: int = 5) -> None:
        print("Evaluating...")
        print(f"student_search_results_path: {student_search_results_path}")
        print(f"dataset_path {dataset_path}")
