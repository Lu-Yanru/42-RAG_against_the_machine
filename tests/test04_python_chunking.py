import ast

from src.ingest.chunking import Chunk
from src.ingest.chunking_python import PythonChunker
from src.ingest.loader import File, Loader


def assert_offsets_round_trip(content: str, spans: list[tuple[int, int]]):
    for start, end in spans:
        assert 0 <= start < end <= len(content)


def offsets_of(content: str) -> list[int]:
    return Loader.line_offsets(content)


class TestDefStartEnd:
    def test_plain_function_start_is_the_def_keyword(self):
        content = "def foo():\n    pass\n"
        tree = ast.parse(content)
        node = tree.body[0]
        start = PythonChunker._def_start(node, offsets_of(content))
        assert content[start:].startswith("def foo")

    def test_decorated_function_start_includes_the_at_symbol(self):
        content = "@my_decorator\ndef foo():\n    pass\n"
        tree = ast.parse(content)
        node = tree.body[0]
        start = PythonChunker._def_start(node, offsets_of(content))
        assert content[start] == "@"
        assert content[start:].startswith("@my_decorator\ndef foo")

    def test_multiple_decorators_use_the_first_one(self):
        content = "@a\n@b.c(1)\ndef foo():\n    pass\n"
        tree = ast.parse(content)
        node = tree.body[0]
        start = PythonChunker._def_start(node, offsets_of(content))
        assert content[start:].startswith("@a\n@b.c(1)\ndef foo")

    def test_def_end_matches_end_of_body(self):
        content = "def foo():\n    return 1\nX = 2\n"
        tree = ast.parse(content)
        node = tree.body[0]
        line_offsets = offsets_of(content)
        start = PythonChunker._def_start(node, line_offsets)
        end = PythonChunker._def_end(node, line_offsets)
        assert content[start:end] == "def foo():\n    return 1"

    def test_class_start_includes_its_decorator(self):
        content = "@dataclass\nclass Foo:\n    x: int\n"
        tree = ast.parse(content)
        node = tree.body[0]
        start = PythonChunker._def_start(node, offsets_of(content))
        assert content[start:].startswith("@dataclass\nclass Foo")


class TestMemberUnitSpans:
    def test_module_preamble_before_first_def_is_one_unit(self):
        content = "import os\n\nX = 1\nY = 2\n\ndef foo():\n    pass\n"
        tree = ast.parse(content)
        line_offsets = offsets_of(content)
        units = PythonChunker._member_unit_span(tree.body, 0, len(content), line_offsets)
        assert_offsets_round_trip(content, units)
        preamble = content[units[0][0]:units[0][1]]
        assert "import os" in preamble
        assert "X = 1" in preamble
        assert "Y = 2" in preamble
        assert "def foo" not in preamble

    def test_top_level_constant_pair_stays_in_one_unit(self):
        # regression case for the FP8_MIN / FP8_MAX style question: both
        # module-level constants must land in the same retrievable chunk
        content = "FP8_MIN = -448.0\nFP8_MAX = 448.0\n"
        tree = ast.parse(content)
        line_offsets = offsets_of(content)
        units = PythonChunker._member_unit_span(tree.body, 0, len(content), line_offsets)
        assert len(units) == 1
        # the leftover region legitimately runs to region_end (=len(content)),
        # trailing newline included -- that's correct, not something to trim
        assert content[units[0][0]:units[0][1]] == content

    def test_function_is_its_own_unit_separate_from_preamble(self):
        content = "X = 1\n\ndef foo():\n    pass\n\nY = 2\n"
        tree = ast.parse(content)
        line_offsets = offsets_of(content)
        units = PythonChunker._member_unit_span(tree.body, 0, len(content), line_offsets)
        assert len(units) == 3
        assert "X = 1" in content[units[0][0]:units[0][1]]
        assert content[units[1][0]:units[1][1]].startswith("def foo")
        assert "Y = 2" in content[units[2][0]:units[2][1]]

    def test_trailing_content_after_last_def_is_captured(self):
        content = "def foo():\n    pass\n\n# trailing comment\nZ = 3\n"
        tree = ast.parse(content)
        line_offsets = offsets_of(content)
        units = PythonChunker._member_unit_span(tree.body, 0, len(content), line_offsets)
        assert_offsets_round_trip(content, units)
        assert "Z = 3" in content[units[-1][0]:units[-1][1]]

    def test_empty_module_body_yields_a_single_leftover_unit(self):
        content = '"""just a docstring, no statements after it"""\n'
        tree = ast.parse(content)
        line_offsets = offsets_of(content)
        # a module docstring IS a body statement (an Expr), so this
        # exercises the "no defs/classes at all" path
        units = PythonChunker._member_unit_span(tree.body, 0, len(content), line_offsets)
        assert len(units) == 1
        assert units[0] == (0, len(content))


class TestClassUnitSpans:
    def test_class_with_no_methods_is_one_header_unit(self):
        # covers the Enum / plain-constants-class case
        content = "class Color(Enum):\n    RED = 1\n    BLUE = 2\n"
        tree = ast.parse(content)
        node = tree.body[0]
        assert isinstance(node, ast.ClassDef)
        units = PythonChunker._class_unit_span(node, offsets_of(content))
        assert len(units) == 1
        assert content[units[0][0]:units[0][1]] == content.rstrip("\n")

    def test_class_header_and_methods_split_into_separate_units(self):
        content = ('class Greeter:\n'
                   '    """docstring"""\n'
                   '    DEFAULT = "world"\n\n'
                   '    def greet(self):\n'
                   '        return self.DEFAULT\n')
        tree = ast.parse(content)
        node = tree.body[0]
        assert isinstance(node, ast.ClassDef)
        units = PythonChunker._class_unit_span(node, offsets_of(content))
        assert len(units) == 2
        header = content[units[0][0]:units[0][1]]
        assert "docstring" in header
        assert "DEFAULT" in header
        assert "def greet" not in header
        method = content[units[1][0]:units[1][1]]
        assert method.startswith("def greet")

    def test_statement_between_two_methods_is_captured(self):
        content = ('class Foo:\n'
                   '    def a(self):\n'
                   '        pass\n\n'
                   '    MID_CONST = 1\n\n'
                   '    def b(self):\n'
                   '        pass\n')
        tree = ast.parse(content)
        node = tree.body[0]
        assert isinstance(node, ast.ClassDef)
        units = PythonChunker._class_unit_span(node, offsets_of(content))
        assert_offsets_round_trip(content, units)
        joined_texts = [content[s:e] for s, e in units]
        assert any("MID_CONST" in t for t in joined_texts)
        assert any(t.startswith("def a") for t in joined_texts)
        assert any(t.startswith("def b") for t in joined_texts)

    def test_nested_class_recurses_at_any_depth(self):
        content = ('class Outer:\n'
                   '    class Inner:\n'
                   '        def deep_method(self):\n'
                   '            pass\n')
        tree = ast.parse(content)
        node = tree.body[0]
        assert isinstance(node, ast.ClassDef)
        units = PythonChunker._class_unit_span(node, offsets_of(content))
        assert_offsets_round_trip(content, units)
        assert any(content[s:e].strip().startswith("def deep_method")
                   for s, e in units)


class TestPythonChunkerIntegration:
    def _file(self, content: str, path: str = "sample.py") -> File:
        return File(file_path=path, content=content,
                    line_offsets=Loader.line_offsets(content))

    def test_chunks_a_small_file_correctly(self):
        content = ('"""Sample module."""\n\n'
                   'import os\n\n'
                   'FP8_MIN = -448.0\n'
                   'FP8_MAX = 448.0\n\n\n'
                   'def helper(a, b):\n'
                   '    return a + b\n\n\n'
                   'class Greeter:\n'
                   '    DEFAULT_NAME = "world"\n\n'
                   '    def greet(self, name=DEFAULT_NAME):\n'
                   '        return f"Hello, {name}! cwd={os.getcwd()}"\n')
        chunker = PythonChunker(files=[self._file(content)],
                                max_chunk_size=2000)
        chunks = chunker.chunk()
        assert all(isinstance(c, Chunk) for c in chunks)
        for c in chunks:
            assert c.content == content[c.first_character_index:
                                        c.last_character_index]
        joined = [c.content for c in chunks]
        assert any("FP8_MIN" in t and "FP8_MAX" in t for t in joined)
        assert any(t.startswith("def helper") for t in joined)
        assert any(t.startswith("def greet") for t in joined)

    def test_oversized_method_falls_back_to_line_splitting(self):
        body_lines = "\n".join(f"        x{i} = {i}" for i in range(80))
        content = f"class Foo:\n    def big(self):\n{body_lines}\n"
        # sanity-check the fixture itself is valid Python before blaming
        # the chunker for anything
        ast.parse(content)
        chunker = PythonChunker(files=[self._file(content)],
                                max_chunk_size=100)
        chunks = chunker.chunk()
        assert len(chunks) > 2
        assert all(len(c.content) <= 100 for c in chunks)
        for c in chunks:
            assert c.content == content[c.first_character_index:
                                        c.last_character_index]

    def test_syntax_error_file_is_skipped_not_crashed(self, capsys):
        good = self._file("def ok():\n    pass\n", path="good.py")
        bad = self._file("def broken(:\n    pass\n", path="bad.py")
        chunker = PythonChunker(files=[good, bad], max_chunk_size=2000)
        chunks = chunker.chunk()  # must not raise
        assert all(c.file_path == "good.py" for c in chunks)
        captured = capsys.readouterr()
        assert "Error parsing" in captured.err
        assert "bad.py" in captured.err

    def test_no_chunk_exceeds_cap_across_multiple_files(self):
        content_a = "X = 1\n\n\ndef f():\n    return 1\n"
        content_b = "Y = 2\n\n\ndef g():\n    return 2\n"
        files = [self._file(content_a, "a.py"),
                 self._file(content_b, "b.py")]
        chunker = PythonChunker(files=files, max_chunk_size=15)
        chunks = chunker.chunk()
        assert all(len(c.content) <= 15 for c in chunks)
        for c in chunks:
            src = content_a if c.file_path == "a.py" else content_b
            assert c.content == src[c.first_character_index:
                                    c.last_character_index]
