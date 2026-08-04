from dataclasses import dataclass
from pathlib import Path
import sys
from tqdm import tqdm


@dataclass
class File:
    """Represents a file with a file path and its content."""
    file_path: str
    content: str
    lines: list[tuple[int, str]]


class Loader:
    """Loads all md, txt, py files in a file path."""
    def __init__(self) -> None:
        self.raw_path = Path("data/raw")
        self.md_file_paths = sorted(list(self.raw_path.rglob("*.md")))
        self.txt_file_paths = sorted(list(self.raw_path.rglob("*.txt")))
        self.py_file_paths = sorted(list(self.raw_path.rglob("*.py")))

        self.md_files = self.load_files(self.md_file_paths,
                                        "Loading md files")
        self.txt_files = self.load_files(self.txt_file_paths,
                                         "Loading txt files")
        self.py_files = self.load_files(self.py_file_paths,
                                        "Loading py files")

    def load_files(self, files: list[Path], desc: str) -> list[File]:
        res = []
        for file_path in tqdm(files, desc=desc):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = self.line_offsets(content)
                    file = File(file_path=str(file_path), content=content,
                                lines=lines)
                    res.append(file)
            except (OSError, UnicodeDecodeError) as e:
                print(f"Error reading file '{file_path}': {e} Skipping...",
                      file=sys.stderr)
        return res

    @staticmethod
    def line_offsets(file: str) -> list[tuple[int, str]]:
        """Compute the character offset where each line starts in a file."""
        res = []
        for line in file.split("\n"):
            if not line.strip():
                continue
            count = 0
            for char in line:
                if char not in [" ", "\t"]:
                    break
                count += 1
            res.append((count, line))
        return res
