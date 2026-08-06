import ast
import sys

from src.ingest.chunking import Chunk, Chunker
from src.ingest.loader import File


class PythonChunker(Chunker):
    def __init__(self, files: list[File],
                 max_chunk_size: int = 2000,
                 file_type: str = "py") -> None:
        super().__init__(files, max_chunk_size, file_type)

    def chunk(self) -> list[Chunk]:
        chunks = []
        for file in self.files:
            spans = self.chunk_py(file.content, file_path=file.file_path)

            for start, end in spans:
                chunks.append(Chunk(
                    file_path=file.file_path,
                    first_character_index=start,
                    last_character_index=end,
                    content=file.content[start:end],
                ))
        return chunks

    def chunk_py(self, content: str, file_path: str) -> list[tuple[int, int]]:
        chunks = []

        try:
            tree = ast.parse(content)
            ast.dump(tree)
        except SyntaxError as e:
            # Fall back to simple text chunking if ast parsing failed.
            print(f"Failed to parse file '{file_path}' AST: {e} "
                  "Using simple text chunking instead.",
                  file=sys.stderr)
            chunks = self.chunk_span(content, file_path=file_path)

        return chunks
