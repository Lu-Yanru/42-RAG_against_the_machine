from src.ingest.chunking import Chunk
from src.ingest.chunking_text import MarkdownChunker
from src.ingest.loader import File


def assert_offsets_round_trip(content: str, spans: list[tuple[int, int]]):
    for start, end in spans:
        assert 0 <= start < end <= len(content)


class TestFenceMatching:
    def test_backtick_fence_open(self):
        assert MarkdownChunker.code_block_open("```python") == ("`", 3)

    def test_tilde_fence_open(self):
        assert MarkdownChunker.code_block_open("~~~") == ("~", 3)

    def test_indented_fence_up_to_three_spaces_still_opens(self):
        assert MarkdownChunker.code_block_open("   ```") == ("`", 3)

    def test_four_space_indent_does_not_open_a_fence(self):
        # >=4 spaces makes it an indented code block, not a fence, per
        # CommonMark -- the fence characters are just code content there.
        assert MarkdownChunker.code_block_open("    ```") is None

    def test_two_backticks_is_not_a_fence(self):
        assert MarkdownChunker.code_block_open("``") is None

    def test_close_requires_same_char(self):
        assert MarkdownChunker.code_block_close("~~~", "`", 3) is False

    def test_close_requires_at_least_opening_length(self):
        assert MarkdownChunker.code_block_close("``", "`", 3) is False
        assert MarkdownChunker.code_block_close("````", "`", 3) is True

    def test_close_rejects_trailing_content(self):
        # a line with the fence chars plus other text doesn't close it
        assert MarkdownChunker.code_block_close("``` still text", "`", 3) is False

    def test_close_allows_trailing_whitespace(self):
        assert MarkdownChunker.code_block_close("```   ", "`", 3) is True


class TestHeaderOffsets:
    def test_finds_atx_headers(self):
        content = "# H1\n\ntext\n\n## H2\n"
        offsets = MarkdownChunker.md_header_offsets(content)
        assert offsets == [0, content.index("## H2")]

    def test_ignores_hash_without_leading_space_requirement_violation(self):
        # "#5" has no space after the hashes -- not a header
        content = "#5 is a number\n"
        assert MarkdownChunker.md_header_offsets(content) == []

    def test_ignores_header_inside_fence(self):
        content = "# Real header\n\n```bash\n# not a header, a comment\necho hi\n```\n\nmore text\n"
        offsets = MarkdownChunker.md_header_offsets(content)
        assert offsets == [0]

    def test_ignores_header_look_alike_in_four_space_indented_code(self):
        # 4-space indentation makes this an indented code block; the
        # header regex's own <=3-space rule is what excludes it -- no
        # separate indented-code-block tracking needed.
        content = "Intro paragraph.\n\n    # this is a shell comment\n    echo hi\n\nMore text.\n"
        assert MarkdownChunker.md_header_offsets(content) == []

    def test_tab_indented_line_is_not_a_header(self):
        content = "Intro.\n\n\t# tab-indented comment, not a header\n\nMore.\n"
        assert MarkdownChunker.md_header_offsets(content) == []

    def test_three_space_indent_header_still_counts(self):
        content = "   # still a header (<=3 spaces)\n"
        assert MarkdownChunker.md_header_offsets(content) == [0]

    def test_bare_hash_with_no_text_is_a_header(self):
        content = "###\n"
        assert MarkdownChunker.md_header_offsets(content) == [0]


class TestMarkdownParagraphSpans:
    def test_fence_with_internal_blank_line_stays_one_unit(self):
        content = ("intro\n\n"
                   "```python\n"
                   "def f():\n"
                   "\n"
                   "    return 1\n"
                   "```\n\n"
                   "outro\n")
        spans = MarkdownChunker.md_paragraph_span(content)
        fence_start = content.index("```python")
        fence_end = content.index("```\n\n") + len("```")
        assert (fence_start, fence_end) in spans
        # the blank line *inside* the fence must not have split it
        assert len(spans) == 3  # intro, whole fence, outro

    def test_blank_line_outside_fence_still_splits_normally(self):
        content = "para a\n\npara b\n"
        spans = MarkdownChunker.md_paragraph_span(content)
        assert len(spans) == 2

    def test_unterminated_fence_at_eof_does_not_crash(self):
        content = "intro\n\n```python\ndef f():\n    pass\n"
        spans = MarkdownChunker.md_paragraph_span(content)
        assert_offsets_round_trip(content, spans)
        # last line's span excludes its own trailing newline, consistent
        # with _line_spans everywhere else in this module
        assert spans[-1][1] == len(content) - 1
        assert content[spans[-1][0]:spans[-1][1]].endswith("pass")


class TestMarkdownSectionSpans:
    def test_content_before_first_header_is_its_own_section(self):
        content = "intro text\n\n# H1\n\nbody\n"
        spans = MarkdownChunker.md_section_span(content)
        assert content[spans[0][0]:spans[0][1]] == "intro text\n\n"
        assert content[spans[1][0]:spans[1][1]].startswith("# H1")

    def test_no_headers_yields_one_section(self):
        content = "just prose, no headers at all\n"
        spans = MarkdownChunker.md_section_span(content)
        assert spans == [(0, len(content))]

    def test_header_at_position_zero_has_no_duplicate_empty_section(self):
        content = "# H1\n\nbody\n"
        spans = MarkdownChunker.md_section_span(content)
        assert spans[0][0] == 0
        assert len(spans) == 1

    def test_section_keeps_header_glued_to_its_body(self):
        content = "# H1\n\nbody one\n\n## H2\n\nbody two\n"
        spans = MarkdownChunker.md_section_span(content)
        assert content[spans[0][0]:spans[0][1]] == "# H1\n\nbody one\n\n"
        assert content[spans[1][0]:spans[1][1]] == "## H2\n\nbody two\n"

    def test_hash_inside_fence_does_not_create_a_spurious_section(self):
        content = ("# Real\n\n"
                   "```bash\n"
                   "# looks like a header, isn't one\n"
                   "```\n\n"
                   "still under Real\n")
        spans = MarkdownChunker.md_section_span(content)
        assert len(spans) == 1


class TestMarkdownChunkerIntegration:
    def test_small_sections_are_merged(self):
        content = "# H1\n\nbody one\n\n## H2\n\nbody two\n"
        file = File(file_path="doc.md", content=content, line_offsets=[])
        chunker = MarkdownChunker(files=[file], max_chunk_size=2000)
        chunks = chunker.chunk()
        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        for c in chunks:
            assert c.content == content[c.first_character_index:
                                        c.last_character_index]

    def test_oversized_section_falls_back_but_keeps_fence_intact(self):
        filler = "padding text to blow past the cap. " * 10
        content = ("# H1\n\n" + filler + "\n\n"
                   "```python\n"
                   "def f():\n"
                   "\n"
                   "    return 1\n"
                   "```\n\n" + filler + "\n")
        file = File(file_path="doc.md", content=content, line_offsets=[])
        chunker = MarkdownChunker(files=[file], max_chunk_size=120)
        chunks = chunker.chunk()
        assert all(len(c.content) <= 120 for c in chunks)
        fence_text = content[content.index("```python"):
                             content.index("```\n\n") + len("```")]
        # the fence, blank line and all, must appear intact inside a
        # single chunk somewhere -- not split across two chunks
        assert any(fence_text in c.content for c in chunks)
        for c in chunks:
            assert c.content == content[c.first_character_index:
                                        c.last_character_index]

    def test_single_oversized_fence_falls_through_to_line_splitting(self):
        # a fence so large that even the whole-fence unit exceeds the cap
        body_lines = "\n".join(f"line_{i} = {i}" for i in range(60))
        content = f"# H1\n\n```python\n{body_lines}\n```\n"
        file = File(file_path="doc.md", content=content, line_offsets=[])
        chunker = MarkdownChunker(files=[file], max_chunk_size=100)
        chunks = chunker.chunk()
        assert len(chunks) > 1
        assert all(len(c.content) <= 100 for c in chunks)
        for c in chunks:
            assert c.content == content[c.first_character_index:
                                        c.last_character_index]
