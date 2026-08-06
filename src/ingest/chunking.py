from dataclasses import dataclass
import sys
from typing import Callable

from src.ingest.loader import File


class ChunkError(Exception):
    """Error when chunking."""
    pass


@dataclass
class Chunk:
    """Represents a single chunk text."""
    file_path: str
    first_character_index: int
    last_character_index: int
    content: str


class Chunker:
    """Basic chunker class."""
    def __init__(self, files: list[File],
                 max_chunk_size: int = 2000,
                 file_type: str = "txt") -> None:
        if len(files) == 0:
            raise ChunkError("ChunkError: "
                             f"No {file_type} file to chunk.")

        self.files = files
        self.max_chunk_size = max_chunk_size

    def chunk_span(self, content: str, start: int = 0,
                   end: int | None = None,
                   overlap: int | None = None,
                   file_path: str = "",
                   paragraph_fn: Callable[[str, int, int | None],
                                          list[tuple[int, int]]] | None = None
                   ) -> list[tuple[int, int]]:
        """
        Chunk content[start:end] using paragraph -> line -> sliding-window
        cascading fallback.
        Returns a list of (first_character_index,
        last_character_index) spans, each <= max_chunk_size.
        """
        if end is None:
            end = len(content)
        if overlap is None:
            overlap = max(int(self.max_chunk_size * 0.1), 0)
        if paragraph_fn is None:
            paragraph_fn = self.paragraph_span

        res = []
        paragraphs = paragraph_fn(content, start, end)
        packed_paragraphs = self.greedy_pack(paragraphs)

        for p_start, p_end in packed_paragraphs:
            if p_end - p_start <= self.max_chunk_size:
                res.append((p_start, p_end))
                continue

            # If a single paragraph exceeds max_chunk_size,
            # split by line and greedy pack again
            lines = Chunker.line_span(content, p_start, p_end)
            packed_lines = self.greedy_pack(lines)

            for l_start, l_end in packed_lines:
                if l_end - l_start <= self.max_chunk_size:
                    res.append((l_start, l_end))
                    continue

                # If a single line exceeds max_chunk_size,
                # use a sliding window and log a warning.
                print(f"Warning: line of {l_end - l_start} chars exceeds "
                      f"max_chunk_size={self.max_chunk_size}"
                      f"{' in ' + file_path if file_path else ''}; "
                      f"falling back to a raw sliding window.",
                      file=sys.stderr)
                res.extend(self.sliding_window(l_start, l_end, overlap))

        return res

    def greedy_pack(self, spans: list[tuple[int, int]]) \
            -> list[tuple[int, int]]:
        """
        Merge consecutive spans into chunks up to max_chunk_size.
        If a single span exceeds max_chunk_size, it is returned
        as its own over-cap span.
        """
        packed = []
        cur_start = None
        cur_end = None

        for span_start, span_end in spans:
            if cur_start is None:
                cur_start, cur_end = span_start, span_end
                continue
            if span_end - cur_start <= self.max_chunk_size:
                cur_end = span_end
            else:
                if cur_end is not None:
                    packed.append((cur_start, cur_end))
                cur_start, cur_end = span_start, span_end

        if cur_start is not None and cur_end is not None:
            packed.append((cur_start, cur_end))

        return packed

    @staticmethod
    def paragraph_span(content: str, start: int = 0,
                       end: int | None = None) -> list[tuple[int, int]]:
        """
        Group consecutive non-blank lines in content[start:end) into
        paragraph spans, dropping blank-line separators.
        """
        spans = []
        para_start = None
        para_end = None

        for line_start, line_end in Chunker.line_span(content, start, end):
            # If there is a blank line, it is the end of a paragraph
            # Add the start and end points to spans
            is_blank = content[line_start:line_end].strip() == ""
            if is_blank:
                if para_start is not None and para_end is not None:
                    spans.append((para_start, para_end))
                    para_start = None
                continue
            # Start of the paragraph the the start of the first line
            if para_start is None:
                para_start = line_start
            # If no blank line, extend the end of the paragraph
            # to the end of the current line
            para_end = line_end

        # Add the last paragraph when no more lines
        if para_start is not None and para_end is not None:
            spans.append((para_start, para_end))

        return spans

    @staticmethod
    def line_span(content: str, start: int = 0,
                  end: int | None = None) -> list[tuple[int, int]]:
        """
        Returns a list of (line_start, line_end) tuples
        for each line in content[start:end],
        excluding the line terminator from the span.
        """
        if end is None:
            end = len(content)
        spans = []
        cursor = start
        text = content[start:end]
        for line in text.splitlines(keepends=True):
            terminator_len = len(line) - len(line.strip("\r\n"))
            line_end = cursor + len(line) - terminator_len
            spans.append((cursor, line_end))
            cursor += len(line)

        return spans

    def sliding_window(self, start: int, end: int,
                       overlap: int) -> list[tuple[int, int]]:
        """
        Raw character sliding window over [start, end],
        each window <= max_chunk_size,
        consecutive windows overlapping by `overlap` characters.
        """
        spans = []
        step = max(self.max_chunk_size - overlap, 1)
        pos = start
        while pos < end:
            window_end = min(pos + self.max_chunk_size, end)
            spans.append((pos, window_end))
            if window_end >= end:
                break
            pos += step

        return spans
