from bm25s.tokenization import Tokenizer


DEFAULT_SPLITTER = r"\w+"


def build_tokenizer() -> Tokenizer:
    return Tokenizer(
        lower=True,
        splitter=DEFAULT_SPLITTER,
        stopwords="english",
        stemmer=None,
    )
