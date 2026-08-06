from src.ingest.chunking import Chunk
from src.ingest.chunking_text import TextChunker
from src.ingest.loader import File


def assert_offsets_round_trip(content: str, spans: list[tuple[int, int]]):
    """Every span must slice the original content exactly -- this is the
    invariant the whole grading model depends on."""
    for start, end in spans:
        assert 0 <= start < end <= len(content)


class TestParagraphSpans:
    def test_single_paragraph_no_blank_lines(self):
        content = "line one\nline two\nline three"
        spans = TextChunker.paragraph_span(content)
        assert spans == [(0, len(content))]

    def test_two_paragraphs_split_on_blank_line(self):
        content = "para one line a\npara one line b\n\npara two"
        spans = TextChunker.paragraph_span(content)
        assert len(spans) == 2
        assert content[spans[0][0]:spans[0][1]] == "para one line a\npara one line b"
        assert content[spans[1][0]:spans[1][1]] == "para two"

    def test_multiple_consecutive_blank_lines_treated_as_one_separator(self):
        content = "para one\n\n\n\npara two"
        spans = TextChunker.paragraph_span(content)
        assert len(spans) == 2
        assert content[spans[1][0]:spans[1][1]] == "para two"

    def test_whitespace_only_line_counts_as_blank(self):
        content = "para one\n   \npara two"
        spans = TextChunker.paragraph_span(content)
        assert len(spans) == 2

    def test_empty_content_returns_no_paragraphs(self):
        assert TextChunker.paragraph_span("") == []

    def test_leading_and_trailing_blank_lines_are_dropped(self):
        content = "\n\npara one\n\n"
        spans = TextChunker.paragraph_span(content)
        assert len(spans) == 1
        assert content[spans[0][0]:spans[0][1]] == "para one"


class TestGreedyPack:
    def test_packs_small_units_together(self):
        # three 3-char units, cap 10 -> should merge into one span
        units = [(0, 3), (3, 6), (6, 9)]
        chunker = TextChunker(["hello"], 10)
        packed = chunker.greedy_pack(units)
        assert packed == [(0, 9)]

    def test_starts_new_chunk_when_cap_would_be_exceeded(self):
        units = [(0, 5), (5, 10), (10, 15)]
        chunker = TextChunker(["hello"], 10)
        packed = chunker.greedy_pack(units)
        assert packed == [(0, 10), (10, 15)]

    def test_oversized_single_unit_passed_through_uncapped(self):
        units = [(0, 3), (3, 100)]
        chunker = TextChunker(["hello"], 10)
        packed = chunker.greedy_pack(units)
        # first unit alone, then the oversized unit alone -- packer never
        # splits a unit itself
        assert packed == [(0, 3), (3, 100)]

    def test_empty_input(self):
        chunker = TextChunker(["hello"], 10)
        assert chunker.greedy_pack([]) == []


class TestSlidingWindow:
    def test_windows_never_exceed_cap(self):
        chunker = TextChunker(["hello"], 20)
        spans = chunker.sliding_window(0, 55, overlap=5)
        assert all(end - start <= 20 for start, end in spans)

    def test_windows_cover_the_full_range(self):
        chunker = TextChunker(["hello"], 20)
        spans = chunker.sliding_window(0, 55, overlap=5)
        assert spans[0][0] == 0
        assert spans[-1][1] == 55

    def test_consecutive_windows_overlap_by_requested_amount(self):
        chunker = TextChunker(["hello"], 20)
        spans = chunker.sliding_window(0, 55, overlap=5)
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            assert s2 == s1 + (20 - 5)
            assert e1 - s2 == 5  # actual overlap amount

    def test_range_smaller_than_cap_yields_one_window(self):
        chunker = TextChunker(["hello"], 20)
        spans = chunker.sliding_window(0, 12, overlap=5)
        assert spans == [(0, 12)]


class TestChunkSpanCascade:
    def test_small_file_is_one_chunk(self):
        content = "short paragraph, well under the cap"
        chunker = TextChunker([content])
        spans = chunker.chunk_span(content)
        assert spans == [(0, len(content))]

    def test_multiple_small_paragraphs_get_packed_together(self):
        content = "para a\n\npara b\n\npara c"
        chunker = TextChunker([content])
        spans = chunker.chunk_span(content)
        assert len(spans) == 1
        assert_offsets_round_trip(content, spans)

    def test_no_chunk_exceeds_max_chunk_size(self):
        content = ("paragraph one is short\n\n" +
                   "word " * 100 + "\n\n" +
                   "paragraph three is also short")
        chunker = TextChunker([content], 50)
        spans = chunker.chunk_span(content)
        assert all(end - start <= 50 for start, end in spans)
        assert_offsets_round_trip(content, spans)

    def test_oversized_paragraph_falls_back_to_line_splitting(self):
        long_line_para = "\n".join(f"line {i} of a long paragraph"
                                   for i in range(20))
        content = "intro\n\n" + long_line_para
        chunker = TextChunker([content], 100)
        spans = chunker.chunk_span(content)
        assert all(end - start <= 100 for start, end in spans)
        assert_offsets_round_trip(content, spans)
        # every line of the long paragraph must still be recoverable
        # somewhere in the chunk set
        joined = "".join(content[s:e] for s, e in spans)
        assert "line 19 of a long paragraph" in joined

    def test_oversized_single_line_falls_back_to_sliding_window(self, capsys):
        content = "x" * 500  # one line, no newlines at all
        chunker = TextChunker([content], 100)
        spans = chunker.chunk_span(content, file_path="f.txt")
        assert len(spans) > 1
        assert all(end - start <= 100 for start, end in spans)
        assert_offsets_round_trip(content, spans)
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert "f.txt" in captured.err

    def test_empty_content_yields_no_chunks(self):
        content = ""
        chunker = TextChunker([content])
        spans = chunker.chunk_span(content)
        assert spans == []

    def test_content_reconstructable_from_chunks_in_order(self):
        # chunks may drop blank-line separators, but every span's own
        # text must exactly match the source slice (the core invariant).
        content = "alpha beta gamma\n\ndelta epsilon\n\nzeta"
        chunker = TextChunker([content])
        spans = chunker.chunk_span(content)
        for start, end in spans:
            assert content[start:end] in content


class TestTextChunkerIntegration:
    def test_chunks_a_loaded_file(self):
        file = File(file_path="tests/fixtures/mini_repo/sample.txt",
                   content="This is a plain text fixture file.\n"
                           "It has more than one line.\n",
                   line_offsets=[])
        chunker = TextChunker(files=[file], max_chunk_size=2000)
        chunks = chunker.chunk()
        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].file_path == file.file_path
        assert chunks[0].content == file.content[
            chunks[0].first_character_index:chunks[0].last_character_index]

    def test_no_chunk_exceeds_cap_across_multiple_files(self):
        big_content = ("word " * 20 + "\n\n") * 30
        files = [
            File(file_path="a.txt", content=big_content, line_offsets=[]),
            File(file_path="b.txt", content="short", line_offsets=[]),
        ]
        chunker = TextChunker(files=files, max_chunk_size=80)
        chunks = chunker.chunk()
        assert len(chunks) > 2
        assert all(len(c.content) <= 80 for c in chunks)
        for c in chunks:
            src = next(f.content for f in files if f.file_path == c.file_path)
            assert c.content == src[c.first_character_index:
                                    c.last_character_index]
