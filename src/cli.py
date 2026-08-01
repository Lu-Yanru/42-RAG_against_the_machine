class CLI:
    """Handles the CLI."""

    @staticmethod
    def index(max_chunk_size: int = 2000) -> None:
        print(f"max_chunk_size: {max_chunk_size}")
        print("Ingestion complete! Indices saved under data/processed/")

    @staticmethod
    def search(query: str, k: int = 5) -> None:
        print(f"query: {query}")
        print(f"k: {k}")
        print("Search result...")

    @staticmethod
    def search_dataset(dataset_path: str = "data/datasets/UnansweredQuestions"
                                           "/dataset_docs_public.json",
                       k: int = 5,
                       save_directory: str = "data/output/search_results"
                                             "/UnansweredQuestions") -> None:
        print(f"dataset_path: {dataset_path}")
        print(f"k: {k}")
        print(f"Saved student_search_results to {save_directory}")

    @staticmethod
    def answer(query: str, k: int = 5) -> None:
        print(f"query: {query}")
        print(f"k: {k}")
        print("Answer result...")

    @staticmethod
    def answer_dataset(student_search_results_path: str =
                       "data/output/search_results"
                       "/UnansweredQuestions/dataset_docs_public.json",
                       save_directory: str =
                       "data/output/search_results_and_answer"
                       "/UnansweredQuestions") -> None:
        print(f"student_search_results_path: {student_search_results_path}")
        print(f"Saved student_search_results_and_answer to {save_directory}")

    @staticmethod
    def evaluate(student_search_results_path: str =
                 "data/output/search_results"
                 "/UnansweredQuestions/dataset_docs_public.json",
                 dataset_path: str =
                 "data/datasets/UnansweredQuestions"
                 "/dataset_docs_public.json") -> None:
        print("Evaluating...")
        print(f"student_search_results_path: {student_search_results_path}")
        print(f"dataset_path {dataset_path}")
