import sys
from timeit import default_timer as timer

from src.ingest.chunking import ChunkError
from src.ingest.chunking_text import TextChunker, MarkdownChunker
from src.ingest.chunking_python import PythonChunker
from src.ingest.loader import Loader, LoadError


class CLI:
    """Handles the CLI."""

    @staticmethod
    def index(max_chunk_size: int = 2000) -> None:
        print(f"max_chunk_size: {max_chunk_size}")
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
        try:
            loader = Loader()
            print(loader.py_files[0].file_path)
        except LoadError as e:
            print(e, file=sys.stderr)
            exit(1)

        text_chunks = None
        md_chunks = None
        py_chunks = None

        try:
            text_chunker = TextChunker(loader.txt_files, max_chunk_size, "txt")
            text_chunks = text_chunker.chunk()
        except ChunkError as e:
            print(f"Text {e}", file=sys.stderr)

        try:
            md_chunker = MarkdownChunker(loader.md_files, max_chunk_size, "md")
            md_chunks = md_chunker.chunk()
        except ChunkError as e:
            print(f"Markdown {e}", file=sys.stderr)

        try:
            py_chunker = PythonChunker(loader.py_files, max_chunk_size, "py")
            py_chunks = py_chunker.chunk()
            print(py_chunks[2].content)
        except ChunkError as e:
            print(f"Python {e}", file=sys.stderr)

        if text_chunks is None and md_chunks is None and py_chunks is None:
            print("Fail to chunk files. Exiting...", file=sys.stderr)
            exit(1)

        if text_chunks:
            print(f"text_chunks: {len(text_chunks)}")
        if md_chunks:
            print(f"md_chunks: {len(md_chunks)}")
        if py_chunks:
            print(f"py_chunks: {len(py_chunks)}")
        print("Ingestion complete! Indices saved under data/processed/")
        end = timer()
        print(f"processing time: {end - start:.2f}s")

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
