import hashlib
import json
from pathlib import Path
from pydantic import ValidationError
import sys

from src.config import QUERY_CACHE_PATH, HASH_PATH
from src.models import MinimalSource


class QueryCache:
    def __init__(self, cache_path: str = QUERY_CACHE_PATH,
                 hash_path: str = HASH_PATH) -> None:
        self.cache_path = Path(cache_path)
        self.index_version = self._compute_version(Path(hash_path))
        self.entries: dict[str, list[MinimalSource]] = {}
        self._load()

    @staticmethod
    def _compute_version(hash_path: Path) -> str | None:
        if not hash_path.exists():
            return None
        return hashlib.sha256(hash_path.read_bytes()).hexdigest()

    def _load(self) -> None:
        if self.index_version is None or not self.cache_path.exists():
            return
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if raw.get("index_version") != self.index_version:
            return
        try:
            self.entries = {k: [MinimalSource.model_validate(s) for s in v]
                            for k, v in raw.get("entries", {}).items()}
        except ValidationError:
            return

    @staticmethod
    def _key(method: str, k: int, query: str) -> str:
        """Build a hash out of method, k and the query."""
        return hashlib.sha256(f"{method.lower()}|{k}|"
                              f"{query}".encode()).hexdigest()

    def get(self, method: str, k: int,
            query: str) -> list[MinimalSource] | None:
        return self.entries.get(self._key(method, k, query))

    def put(self, method: str, k: int,
            query: str, sources: list[MinimalSource]) -> None:
        self.entries[self._key(method, k, query)] = sources

    def save(self) -> None:
        if self.index_version is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"index_version": self.index_version,
                   "entries": {k: [s.model_dump() for s in v]
                               for k, v in self.entries.items()}}

        try:
            self.cache_path.write_text(json.dumps(payload, indent=4),
                                       encoding="utf-8")
        except OSError as e:
            print(f"Error: Failed to save query cache: {e}", file=sys.stderr)
            return

        print(f"Queries cached in {self.cache_path}")
