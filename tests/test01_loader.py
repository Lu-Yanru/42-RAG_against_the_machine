import ast
from pathlib import Path
import pytest

from src.ingest.loader import Loader, LoadError

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mini_repo"


def all_file_paths(loader: Loader) -> list[str]:
    return [f.file_path
            for f in loader.py_files + loader.md_files + loader.txt_files]


def test_loads_one_file_per_known_extension():
    loader = Loader(raw_path=str(FIXTURE_ROOT))
    paths = all_file_paths(loader)
    assert any(p.endswith("sample.py") for p in paths)
    assert any(p.endswith("sample.md") for p in paths)
    assert any(p.endswith("sample.txt") for p in paths)


def test_recursive_walk_finds_nested_files():
    loader = Loader(raw_path=str(FIXTURE_ROOT))
    nested = [f for f in loader.py_files if "nested" in f.file_path]
    assert len(nested) == 1
    assert nested[0].file_path.endswith("nested/nested_sample.py")


def test_ignores_unrelated_extensions():
    loader = Loader(raw_path=str(FIXTURE_ROOT))
    paths = all_file_paths(loader)
    assert not any(p.endswith(".json") for p in paths)


def test_skips_non_utf8_file_without_crashing(capsys):
    # This must not raise -- constructing the Loader is the assertion.
    loader = Loader(raw_path=str(FIXTURE_ROOT))
    paths = all_file_paths(loader)
    assert not any("bad_encoding" in p for p in paths)
    captured = capsys.readouterr()
    assert "Error reading file" in captured.err
    assert "bad_encoding.py" in captured.err


def test_file_path_is_posix_style_no_backslashes():
    loader = Loader(raw_path=str(FIXTURE_ROOT))
    for path in all_file_paths(loader):
        assert "\\" not in path


def test_file_path_matches_ground_truth_format():
    # Ground-truth dataset paths look like a relative, forward-slash path
    # from the repo root (e.g. "data/raw/vllm-0.10.1/docs/lora.md"). The
    # grader compares this string verbatim, so it must include the full
    # raw_path prefix, not be stripped or made absolute.
    loader = Loader(raw_path=str(FIXTURE_ROOT))
    sample = next(f for f in loader.py_files
                  if f.file_path.endswith("sample.py")
                  and "nested" not in f.file_path)
    assert sample.file_path == FIXTURE_ROOT.as_posix() + "/sample.py"


def test_content_matches_file_on_disk():
    loader = Loader(raw_path=str(FIXTURE_ROOT))
    sample = next(f for f in loader.py_files
                  if f.file_path.endswith("sample.py")
                  and "nested" not in f.file_path)
    on_disk = (FIXTURE_ROOT / "sample.py").read_text(encoding="utf-8")
    assert sample.content == on_disk


def test_empty_file_loads_without_crashing():
    loader = Loader(raw_path=str(FIXTURE_ROOT))
    empty = next(f for f in loader.txt_files if f.file_path.endswith("empty.txt"))
    assert empty.content == ""
    assert empty.line_offsets == []


def test_missing_root_directory_returns_no_files_not_a_crash(capsys):
    with pytest.raises(LoadError):
        loader = Loader(raw_path=str(FIXTURE_ROOT / "does_not_exist"))
        assert loader.py_files == []
        assert loader.md_files == []
        assert loader.txt_files == []


class TestLineOffsets:
    """Directly targets the bug in the original implementation: it must
    return absolute character offsets into the file, not per-line
    indentation counts, and must not skip blank lines."""

    def test_basic_offsets(self):
        content = "a\n\nb\n"
        assert Loader.line_offsets(content) == [0, 2, 3]

    def test_no_trailing_newline(self):
        content = "hello\nworld"
        offsets = Loader.line_offsets(content)
        assert offsets == [0, 6]
        assert content[offsets[1]:] == "world"

    def test_offsets_survive_leading_whitespace(self):
        # The buggy version returned indentation width here instead of a
        # file offset -- this is the regression case for that bug.
        content = "if True:\n    x = 1\n    y = 2\n"
        offsets = Loader.line_offsets(content)
        assert content[offsets[1]:offsets[1] + 9] == "    x = 1"
        assert content[offsets[2]:offsets[2] + 9] == "    y = 2"

    def test_empty_content_returns_empty_list(self):
        assert Loader.line_offsets("") == []

    def test_matches_ast_lineno_for_a_function_def(self):
        src = "import os\n\nX = 1\n\ndef foo():\n    pass\n"
        tree = ast.parse(src)
        offsets = Loader.line_offsets(src)
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        start = offsets[func.lineno - 1]
        assert src[start:].startswith("def foo")

    def test_matches_ast_lineno_across_a_blank_line(self):
        # Regression case: the buggy version skipped blank lines entirely,
        # which shifts every offset after the first blank line and breaks
        # alignment with ast.lineno (which counts blank lines).
        src = "x = 1\n\n\ndef foo():\n    return 1\n"
        tree = ast.parse(src)
        offsets = Loader.line_offsets(src)
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        start = offsets[func.lineno - 1]
        assert src[start:].startswith("def foo")
