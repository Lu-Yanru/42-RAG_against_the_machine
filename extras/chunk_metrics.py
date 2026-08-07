from statistics import median
import sys

from src.ingest.chunking import Chunk, ChunkError
from src.ingest.chunking_text import TextChunker, MarkdownChunker
from src.ingest.chunking_python import PythonChunker
from src.ingest.loader import Loader, LoadError


def chunk_metrics(max_chunk_size: int = 2000) -> None:
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

    try:
        loader = Loader()
        # print(loader.py_files[0].file_path)
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
        # print(py_chunks[2].content)
    except ChunkError as e:
        print(f"Python {e}", file=sys.stderr)

    if text_chunks is None and md_chunks is None and py_chunks is None:
        print("Fail to chunk files. Exiting...", file=sys.stderr)
        exit(1)

    if text_chunks:
        print(f"\nnum text_chunks: {len(text_chunks)}")
        print(f"num text files: {len(loader.txt_files)}")
        print(f"avg chunks per file: {len(text_chunks) / len(loader.txt_files)}")
        text_chunk_lens = get_all_chunk_len(text_chunks)
        print(f"min chunks size: {min(text_chunk_lens)}")
        print(f"max chunks size: {max(text_chunk_lens)}")
        print(f"avg chunks size: {sum(text_chunk_lens)/len(text_chunks)}")
        print(f"median chunks size: {median(text_chunk_lens)}")
    if md_chunks:
        print(f"\nnum md_chunks: {len(md_chunks)}")
        print(f"num md files: {len(loader.md_files)}")
        print(f"avg chunks per file: {len(md_chunks) / len(loader.md_files)}")
        md_chunk_lens = get_all_chunk_len(md_chunks)
        print(f"min chunks size: {min(md_chunk_lens)}")
        print(f"max chunks size: {max(md_chunk_lens)}")
        print(f"avg chunks size: {sum(md_chunk_lens)/len(md_chunks)}")
        print(f"median chunks size: {median(md_chunk_lens)}")
    if py_chunks:
        print(f"\nnum py_chunks: {len(py_chunks)}")
        print(f"num py files: {len(loader.py_files)}")
        print(f"avg chunks per file: {len(py_chunks) / len(loader.py_files)}")
        py_chunk_lens = get_all_chunk_len(py_chunks)
        print(f"min chunks size: {min(py_chunk_lens)}")
        print(f"max chunks size: {max(py_chunk_lens)}")
        print(f"avg chunks size: {sum(py_chunk_lens)/len(py_chunks)}")
        print(f"median chunks size: {median(py_chunk_lens)}")


def get_all_chunk_len(chunks: list[Chunk]) -> list[int]:
    res = []
    for chunk in chunks:
        res.append(chunk.last_character_index - chunk.first_character_index)
    return res


chunk_metrics()