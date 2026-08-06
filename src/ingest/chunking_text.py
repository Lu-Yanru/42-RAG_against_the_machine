import re

from src.ingest.chunking import Chunk, Chunker
from src.ingest.loader import File


CODE_BLOCK_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
HEADER_RE = re.compile(r"^ {0,3}#{1,6}(\s|$)")


class TextChunker(Chunker):
    def __init__(self, files: list[File],
                 max_chunk_size: int = 2000,
                 file_type: str = "txt") -> None:
        super().__init__(files, max_chunk_size, file_type)

    def chunk(self) -> list[Chunk]:
        chunks = []
        for file in self.files:
            spans = self.chunk_span(file.content, file_path=file.file_path)

            for start, end in spans:
                chunks.append(Chunk(
                    file_path=file.file_path,
                    first_character_index=start,
                    last_character_index=end,
                    content=file.content[start:end],
                ))
        return chunks


class MarkdownChunker(TextChunker):
    def __init__(self, files: list[File],
                 max_chunk_size: int = 2000,
                 file_type: str = "md") -> None:
        super().__init__(files, max_chunk_size, file_type)

    def chunk(self) -> list[Chunk]:
        chunks = []
        for file in self.files:
            # First chunk in sections
            sections = MarkdownChunker.md_section_span(file.content)
            packed_secs = self.greedy_pack(sections)

            for p_start, p_end in packed_secs:
                if p_end - p_start <= self.max_chunk_size:
                    spans = [(p_start, p_end)]
                # If a single section is too big, use chunk_span
                # but for markdown with code block detection
                else:
                    spans = self.chunk_span(
                        file.content, start=p_start, end=p_end,
                        file_path=file.file_path,
                        paragraph_fn=MarkdownChunker.md_paragraph_span
                    )
                for start, end in spans:
                    chunks.append(Chunk(
                        file_path=file.file_path,
                        first_character_index=start,
                        last_character_index=end,
                        content=file.content[start:end]
                    ))
        return chunks

    @staticmethod
    def code_block_open(line: str) -> tuple[str, int] | None:
        """
        If `line` opens a code block (<=3 leading spaces, then 3+ of
        the same backtick/tilde char), return (fence_char, fence_length).
        Otherwise None. Trailing content is allowed on an opening line.
        """
        matched = CODE_BLOCK_RE.match(line)
        if not matched:
            return None
        marker = matched.group(1)
        return (marker[0], len(marker))

    @staticmethod
    def code_block_close(line: str, fence_char: str,
                         fence_length: int) -> bool:
        """
        Check if there is a code block closing string.
        It must use the same fence_char, at least as long as
        the opening fence and has nothing but whitspace after it.
        """
        matched = CODE_BLOCK_RE.match(line)
        if not matched:
            return False
        marker = matched.group(1)
        if marker[0] != fence_char or len(marker) < fence_length:
            return False
        return line[matched.end():].strip() == ""

    @staticmethod
    def md_paragraph_span(content: str, start: int = 0,
                          end: int | None = None) -> list[tuple[int, int]]:
        """
        Paragraph span but fenced code block is treated as one.
        """
        spans: list[tuple[int, int]] = []
        para_start = None
        para_end = None
        in_code_block = False
        fence_char = ""
        fence_length = 0

        def flush() -> None:
            # Helper function for adding span to list.
            nonlocal para_start
            if para_start is not None and para_end is not None:
                spans.append((para_start, para_end))
                para_start = None

        for line_start, line_end in Chunker.line_span(content, start, end):
            line_text = content[line_start:line_end]

            # If the text is in a code block, keep adding lines
            # to the current paragraph until the closing string is found.
            if in_code_block:
                para_end = line_end
                if MarkdownChunker.code_block_close(line_text, fence_char,
                                                    fence_length):
                    in_code_block = False
                    flush()
                continue

            # If a code block opening string is found,
            # close the paragraph before the code block first
            # then open a new paragraph
            opening = MarkdownChunker.code_block_open(line_text)
            if opening:
                flush()
                in_code_block = True
                fence_char, fence_length = opening
                para_start, para_end = line_start, line_end
                continue

            # If there is a blank newline,
            # this is the end of a paragraph
            if line_text.strip() == "":
                flush()
                continue

            if para_start is None:
                para_start = line_start
            para_end = line_end

        # Add the last paragraph
        flush()

        return spans

    @staticmethod
    def md_header_offsets(content: str, start: int = 0,
                          end: int | None = None) -> list[int]:
        """
        Get a list of the absolute positions of all headers in a md file.
        A header can have <= 3 leading spaces followed by 1 - 6 `#`.
        """
        offsets = []
        in_code_block = False
        fence_char = ""
        fence_len = 0

        for line_start, line_end in Chunker.line_span(content, start, end):
            line_text = content[line_start:line_end]

            if in_code_block:
                if MarkdownChunker.code_block_close(line_text, fence_char,
                                                    fence_len):
                    in_code_block = False
                continue

            opening = MarkdownChunker.code_block_open(line_text)
            if opening:
                in_code_block = True
                fence_char, fence_len = opening
                continue

            if HEADER_RE.match(line_text):
                offsets.append(line_start)

        return offsets

    @staticmethod
    def md_section_span(content: str, start: int = 0,
                        end: int | None = None) -> list[tuple[int, int]]:
        """
        Section = from one header line (any level) through the character
        right before the next header line, or EOF. Content before the first
        header (or a file with no headers at all) becomes its own section.
        """
        if end is None:
            end = len(content)
        header_offsets = MarkdownChunker.md_header_offsets(content, start, end)
        boundaries = sorted(set([start] + header_offsets))
        spans = []

        for i, boundary in enumerate(boundaries):
            sec_end = boundaries[i + 1] if i + 1 < len(boundaries) else end
            if sec_end > boundary:
                spans.append((boundary, sec_end))

        return spans
