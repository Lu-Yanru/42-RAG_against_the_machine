from pathlib import Path
import re
import sys
from typing import Any, cast
# import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (MODEL_NAME, NO_CONTEXT_ANSWER,
                        MAX_NEW_TOKENS, MAX_SOURCE_CHARS)
from src.models import MinimalSource


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer the user's question using "
    "only the information in the provided sources. Be concise and factual. "
    "If the sources do not contain the answer, say so explicitly instead "
    "of guessing."
)


class Generator:
    """
    Utility class wrapping a lightweight Hugging Face causal-LM for fast,
    low-memory experimentation.

    Parameters
    ----------
    model_name: str, default="Qwen/Qwen3-0.6B"
        Identifier of the model on the HF Hub.
    device: str | None, default=None
        Computation device. If *None* we automatically select ``mps``
        when available on macOS,
        ``cuda`` when available, otherwise we fall back to ``cpu``.
    dtype: torch.dtype | None, default=None
        Numerical precision. When using a GPU or MPS we default to ``float16``
        to keep memory usage reasonable;
        on CPU we keep ``float32`` for maximum compatibility.
    """
    def __init__(self,
                 model_name: str = MODEL_NAME,
                 device: str | None = None,
                 dtype: torch.dtype | None = None,
                 trust_remote_code: bool = True,
                 ) -> None:
        self.model_name = model_name
        self._tokenizer: Any = None
        self._model: Any = None

        # Auto-select device with prority mps > cuda > cpu
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device

        if dtype is None:
            dtype = torch.float16 if self._device in ["cuda", "mps"] \
                else torch.float32
        self._dtype = dtype

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=trust_remote_code)
        self._tokenizer = tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self._dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation="sdpa",
        )
        self._model = cast(Any, model).to(self._device)
        # Set model in eval mode (as opposed to training mode)
        if self._model is not None:
            self._model.eval()

    def generate_answer(self, question: str,
                        sources: list[MinimalSource]) -> str:
        """
        Generate a plain text answer based on the provided sources.
        """
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            print("Model not loaded. Load model first.")
            return ""

        if not sources:
            return NO_CONTEXT_ANSWER

        prompt = Generator.build_prompt(question, sources)
        prompt_text = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,  # Do not tokenize the output to get a str
            # Add template token to indicate the start of a response
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt_text,
                           return_tensors="pt").to(self._device)

        # start = time.perf_counter()
        # Greedy decoding to stay close to the context,
        # no creative generation
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        # elapsed = time.perf_counter() - start

        input_length = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_length:]

        # generated_len = output_ids.shape[1] - inputs["input_ids"].shape[1]
        # print(f"device={self._device} "
        #       f"prompt_tokens={inputs['input_ids'].shape[1]} "
        #       f"generated_tokens={generated_len} time={elapsed:.2f}s "
        #       f"tok/s={generated_len/elapsed:.1f}", file=sys.stderr)

        raw_text = tokenizer.decode(generated, skip_special_tokens=True)
        return Generator.strip_thinking(raw_text)

    @staticmethod
    def strip_thinking(text: str | list[str]) -> str:
        """Remove <think>...</think> block."""
        if isinstance(text, list):
            text = "".join(text)
        return THINK_BLOCK_RE.sub("", text).strip()

    @staticmethod
    def build_prompt(question: str,
                     sources: list[MinimalSource]) -> list[dict[str, str]]:
        """Build prompt from a question and its retrieved sources."""
        blocks = []
        budget = MAX_SOURCE_CHARS

        for source in sources:
            if budget <= 0:
                break

            text = Generator.read_source_text(source)
            if text is None:
                continue
            budget -= len(text)
            blocks.append(f"### Source: {source.file_path}\n{text}")

        context = "\n\n".join(blocks)
        user_content = (f"Question: {question}\n\nContext:\n{context}"
                        if context else
                        f"Question: {question}")

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def read_source_text(source: MinimalSource) -> str | None:
        """
        Re-slice source text from the raw file by offset.
        Returns None if the file cannot be read or the offsets
        no longer fit the file on disk.
        """
        file_path = Path(source.file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError) as e:
            print(f"Warning: could not read source '{source.file_path}': {e} "
                  "Skipping this source.", file=sys.stderr)
            return None

        if source.last_character_index > len(content):
            print(f"Warning: offsets for '{source.file_path}' no longer fit "
                  "the file on disk. Skipping this source.", file=sys.stderr)
            return None

        return content[source.first_character_index:
                       source.last_character_index]
