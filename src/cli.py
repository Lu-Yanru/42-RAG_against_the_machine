import sys

from src.ingest.chunking import ChunkError, TextChunker
from src.ingest.loader import Loader, LoadError


class CLI:
    """Handles the CLI."""

    @staticmethod
    def index(max_chunk_size: int = 2000) -> None:
        print(f"max_chunk_size: {max_chunk_size}")
        try:
            loader = Loader()
            print(loader.txt_files[0].file_path)
            print(loader.txt_files[0].line_offsets[0])
        except LoadError as e:
            print(e, file=sys.stderr)
            exit(1)

        try:
            text_chunker = TextChunker(loader.txt_files, max_chunk_size, "txt")
            text_chunks = text_chunker.chunk()
            print(text_chunks[0].content)
            print(text_chunks[0].first_character_index)
            print(text_chunks[0].last_character_index)
            print("Ingestion complete! Indices saved under data/processed/")
        except ChunkError as e:
            print(e, file=sys.stderr)

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
