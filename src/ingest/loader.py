from dataclasses import dataclass
from pathlib import Path
import sys
from tqdm import tqdm


class LoadError(Exception):
    """Error when loading the files."""
    pass


@dataclass
class File:
    """Represents a file with a file path and its content."""
    file_path: str
    content: str
    line_offsets: list[int]


class Loader:
    """Loads all md, txt, py files in a file path."""
    def __init__(self, raw_path: str = "data/raw") -> None:
        self.raw_path = Path(raw_path)
        self.md_file_paths = sorted(list(self.raw_path.rglob("*.md")))
        self.txt_file_paths = sorted(list(self.raw_path.rglob("*.txt")))
        self.py_file_paths = sorted(list(self.raw_path.rglob("*.py")))

        self.md_files = self.load_files(self.md_file_paths,
                                        "Loading md files")
        self.txt_files = self.load_files(self.txt_file_paths,
                                         "Loading txt files")
        self.py_files = self.load_files(self.py_file_paths,
                                        "Loading py files")

        total = len(self.md_files) + len(self.txt_files) + len(self.py_files)
        if total == 0:
            raise LoadError(f"Warning: no .md/.txt/.py files found under "
                            f"'{self.raw_path}'.")

    def load_files(self, files: list[Path], desc: str) -> list[File]:
        res = []
        for file_path in tqdm(files, desc=desc):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except (OSError, UnicodeDecodeError) as e:
                print(f"Error reading file '{file_path}': {e} Skipping...",
                      file=sys.stderr)
                continue

            offsets = self.line_offsets(content)
            file = File(file_path=file_path.as_posix(), content=content,
                        line_offsets=offsets)
            res.append(file)

        return res

    @staticmethod
    def line_offsets(content: str) -> list[int]:
        """Character offset where each line starts.

        Returns a list where index i (0-indexed) is the absolute character
        offset of the start of line i+1 (1-indexed) in `content`. This
        1-indexed convention matches Python's `ast` module line numbering
        (`node.lineno`, `node.end_lineno`), so a chunker can later do
        `offsets[node.lineno - 1]` to get the absolute start offset of a
        function/class def. Blank lines are included as their own entries,
        since `ast` counts them too.
        """
        lines = content.splitlines(keepends=True)
        res = [0] * len(lines)
        cursor = 0
        for i, line in enumerate(lines):
            res[i] = cursor
            cursor += len(line)
        return res
