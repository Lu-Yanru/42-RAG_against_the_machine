"""
Incremental indexing support: content-hash helper functions that records
which files an index build has already seen, so a later run can skip
re-chunking anything unchanged.
"""


import hashlib
import json
from pathlib import Path
from typing import NamedTuple

from src.models import MinimalSource


class HashError(Exception):
    """Error when hashing the files."""
    pass


class FileHash(NamedTuple):
    """
    What the last index run saw:
    the max_chunk_size it used and each file's hash.
    """
    max_chunk_size: int
    hashes: dict[str, str]


def compute_hash(content: str) -> str:
    """SHA-256 of a file's decoded content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_hash(path: Path) -> FileHash | None:
    """
    Load a previously persisted manifest.
    Returns None if none exists yet,
    or if the file is missing/corrupted.
    """
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return FileHash(max_chunk_size=raw["max_chunk_size"],
                        hashes=raw["hashes"])
    except (json.JSONDecodeError, OSError, KeyError, TypeError):
        return None


def save_hash(path: Path, max_chunk_size: int,
              hashes: dict[str, str]) -> None:
    """Save hashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"max_chunk_size": max_chunk_size,
                       "hashes": hashes},
                      f, indent=4)
    except (json.JSONDecodeError, OSError) as e:
        raise HashError(f"Error saving the hashes: {e}")


def diff(old: FileHash | None, new: dict[str, str],
         max_chunk_size: int) -> tuple[set[str], set[str]]:
    """
    Compare current file hashes against the last persisted hashes.

    Returns (needs_rechunk, unchanged). Deleted files are implicitly
    excluded from both: they're simply absent from current_hashes, so
    nothing downstream ever looks for them again.

    If there is no prior hashes, or max_chunk_size differs from the
    one the manifest was built with, every path is returned in
    needs_rechunk.
    """
    if old is None or old.max_chunk_size != max_chunk_size:
        return set(new), set()

    needs_rechunk = {path for path, h in new.items()
                     if old.hashes.get(path) != h}
    unchanged = set(new) - needs_rechunk
    return needs_rechunk, unchanged


def chunk_key(source: MinimalSource) -> tuple[str, int, int]:
    """Hashable identity for a chunk."""
    return (source.file_path,
            source.first_character_index,
            source.last_character_index)
