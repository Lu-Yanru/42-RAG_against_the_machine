"""Tests for src.generation.generator.Generator.

Generator.__init__ loads real HF weights unconditionally -- there is no
lazy-load path to exploit for fast tests. Every test below that
constructs a Generator() uses the `mocked_hf` fixture to stand in for
AutoTokenizer.from_pretrained / AutoModelForCausalLM.from_pretrained, so
the fast suite never touches the network or downloads weights. Only
TestGeneratorRealModel (marked slow) loads the real model.
"""
import re
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.config import MAX_SOURCE_CHARS, NO_CONTEXT_ANSWER
from src.generation.generator import Generator
from src.models import MinimalSource


def _write(tmp_path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.fixture
def mocked_hf():
    """
    Stands in for AutoTokenizer.from_pretrained / AutoModelForCausalLM
    .from_pretrained so Generator() can be constructed in the fast suite
    without a real download. The fake model's .to() returns itself, same
    as a real nn.Module, so self._model ends up pointing at the same
    mock that .eval() gets called on.
    """
    tokenizer = MagicMock()
    model = MagicMock()
    model.to.return_value = model

    with patch("src.generation.generator.AutoTokenizer") as tok_cls, \
         patch("src.generation.generator.AutoModelForCausalLM") as model_cls:
        tok_cls.from_pretrained.return_value = tokenizer
        model_cls.from_pretrained.return_value = model
        yield tok_cls.from_pretrained, model_cls.from_pretrained, tokenizer, model


class TestDeviceAndDtypeSelection:
    def test_prefers_mps_when_available(self, monkeypatch, mocked_hf):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        generator = Generator()
        assert generator._device == "mps"

    def test_mps_uses_float16(self, monkeypatch, mocked_hf):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        generator = Generator()
        assert generator._dtype == torch.float16

    def test_falls_back_to_cuda_when_mps_unavailable(self, monkeypatch, mocked_hf):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        generator = Generator()
        assert generator._device == "cuda"

    def test_cuda_uses_float16(self, monkeypatch, mocked_hf):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        generator = Generator()
        assert generator._dtype == torch.float16

    def test_falls_back_to_cpu_when_neither_available(self, monkeypatch, mocked_hf):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        generator = Generator()
        assert generator._device == "cpu"

    def test_cpu_uses_float32(self, monkeypatch, mocked_hf):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        generator = Generator()
        assert generator._dtype == torch.float32

    def test_explicit_device_and_dtype_override_auto_selection(self, mocked_hf):
        generator = Generator(device="cpu", dtype=torch.float32)
        assert generator._device == "cpu"
        assert generator._dtype == torch.float32

    def test_model_moved_to_selected_device_and_set_to_eval(self, monkeypatch, mocked_hf):
        _, _, _, model = mocked_hf
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        Generator()
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once()


class TestModelLoadingParameters:
    def test_trust_remote_code_defaults_true_and_reaches_both_loaders(self, mocked_hf):
        tok_from_pretrained, model_from_pretrained, _, _ = mocked_hf
        Generator()
        assert tok_from_pretrained.call_args.kwargs["trust_remote_code"] is True
        assert model_from_pretrained.call_args.kwargs["trust_remote_code"] is True

    def test_trust_remote_code_can_be_disabled(self, mocked_hf):
        tok_from_pretrained, model_from_pretrained, _, _ = mocked_hf
        Generator(trust_remote_code=False)
        assert tok_from_pretrained.call_args.kwargs["trust_remote_code"] is False
        assert model_from_pretrained.call_args.kwargs["trust_remote_code"] is False


class TestReadSourceText:
    def test_reads_the_exact_slice(self, tmp_path):
        path = _write(tmp_path, "sample.py", "hello world, this is a test")
        source = MinimalSource(file_path=path, first_character_index=6,
                               last_character_index=11)
        assert Generator.read_source_text(source) == "world"

    def test_missing_file_returns_none(self, capsys, tmp_path):
        source = MinimalSource(file_path=str(tmp_path / "missing.py"),
                               first_character_index=0,
                               last_character_index=5)
        assert Generator.read_source_text(source) is None
        captured = capsys.readouterr()
        assert "could not read source" in captured.err

    def test_offsets_beyond_file_length_return_none(self, capsys, tmp_path):
        path = _write(tmp_path, "short.py", "short")
        source = MinimalSource(file_path=path, first_character_index=0,
                               last_character_index=999)
        assert Generator.read_source_text(source) is None
        captured = capsys.readouterr()
        assert "no longer fit" in captured.err


class TestBuildPrompt:
    def test_returns_system_and_user_messages(self, tmp_path):
        path = _write(tmp_path, "a.py", "def foo(): pass")
        source = MinimalSource(file_path=path, first_character_index=0,
                               last_character_index=15)
        messages = Generator.build_prompt("What does foo do?", [source])
        assert [m["role"] for m in messages] == ["system", "user"]

    def test_user_message_includes_question_and_labeled_source(self, tmp_path):
        path = _write(tmp_path, "a.py", "def foo(): pass")
        source = MinimalSource(file_path=path, first_character_index=0,
                               last_character_index=15)
        messages = Generator.build_prompt("What does foo do?", [source])
        user_content = messages[1]["content"]
        assert "What does foo do?" in user_content
        assert f"### Source: {path}" in user_content
        assert "def foo(): pass" in user_content

    def test_no_sources_omits_the_context_section_entirely(self):
        # Current implementation: `f"Question: {question}"` with no
        # "Context:" section at all when there's nothing to include --
        # not a placeholder string. If that behavior is intentional,
        # fine; if not, this is the test to change alongside the code.
        messages = Generator.build_prompt("A question with nothing retrieved", [])
        assert messages[1]["content"] == "Question: A question with nothing retrieved"

    def test_unreadable_source_is_skipped_not_crashed(self, tmp_path, capsys):
        missing = MinimalSource(file_path=str(tmp_path / "gone.py"),
                                first_character_index=0,
                                last_character_index=5)
        messages = Generator.build_prompt("question", [missing])
        assert messages[1]["content"] == "Question: question"
        captured = capsys.readouterr()
        assert "could not read source" in captured.err

    def test_single_oversized_source_is_included_in_full_not_truncated(self, tmp_path):
        # MAX_SOURCE_CHARS only gates whether *further* sources get
        # added -- build_prompt never slices the text of the source that
        # exhausts the budget. This relies entirely on the chunker's own
        # max_chunk_size cap to keep any single source reasonably sized;
        # it is not a second line of defense on its own.
        long_content = "x" * (MAX_SOURCE_CHARS + 500)
        path = _write(tmp_path, "big.py", long_content)
        source = MinimalSource(file_path=path, first_character_index=0,
                               last_character_index=len(long_content))
        messages = Generator.build_prompt("question", [source])
        assert long_content in messages[1]["content"]

    def test_second_source_dropped_once_first_exhausts_the_budget(self, tmp_path):
        first = _write(tmp_path, "first.py", "x" * MAX_SOURCE_CHARS)
        second = _write(tmp_path, "second.py", "unique_marker_text")
        sources = [
            MinimalSource(file_path=first, first_character_index=0,
                         last_character_index=MAX_SOURCE_CHARS),
            MinimalSource(file_path=second, first_character_index=0,
                         last_character_index=len("unique_marker_text")),
        ]
        messages = Generator.build_prompt("question", sources)
        assert "unique_marker_text" not in messages[1]["content"]


class TestStripThinking:
    def test_removes_a_think_block(self):
        text = "<think>reasoning here</think>The actual answer."
        assert Generator.strip_thinking(text) == "The actual answer."

    def test_removes_a_multiline_think_block(self):
        text = "<think>\nstep one\nstep two\n</think>\nFinal answer."
        assert Generator.strip_thinking(text) == "Final answer."

    def test_no_think_block_is_left_unchanged(self):
        text = "Just a plain answer, no thinking tags at all."
        assert Generator.strip_thinking(text) == text

    def test_strips_surrounding_whitespace(self):
        text = "<think>x</think>   Answer with leading whitespace.   "
        assert Generator.strip_thinking(text) == "Answer with leading whitespace."

    def test_joins_list_input_before_stripping(self):
        assert Generator.strip_thinking(
            ["<think>x</think>", "Answer."]) == "Answer."


class TestGenerateAnswerEmptySources:
    def test_returns_no_context_answer(self, mocked_hf):
        # Reaches the `if not sources` branch without calling .generate()
        # at all, so the mock model is never exercised past construction.
        generator = Generator()
        assert generator.generate_answer("unanswerable question", []) == \
            NO_CONTEXT_ANSWER


@pytest.mark.slow
class TestGeneratorRealModel:
    """Loads actual Qwen3-0.6B weights -- excluded from the default test
    run. Run explicitly with: uv run pytest -m slow"""

    def test_generates_a_grounded_answer_without_thinking_tags(self, tmp_path):
        content = 'def greet(name):\n    return f"Hello, {name}!"\n'
        path = _write(tmp_path, "greeter.py", content)
        source = MinimalSource(file_path=path, first_character_index=0,
                               last_character_index=len(content))
        generator = Generator()
        answer = generator.generate_answer(
            "What does the greet function return?", [source])
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert "<think>" not in answer
        assert "</think>" not in answer