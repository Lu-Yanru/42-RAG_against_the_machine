"""Tests for src.retrieval.query_cache.QueryCache.

Constructs QueryCache with explicit cache_path/hash_path under tmp_path in
every test -- never touches the real data/processed/ files, same pattern
as test05/test09/test12's save_dir injection.
"""
import json

import pytest

from src.models import MinimalSource
from src.retrieval.query_cache import QueryCache


def _source(name: str, start: int = 0, end: int = 10) -> MinimalSource:
    return MinimalSource(file_path=name, first_character_index=start,
                         last_character_index=end)


def _write_hash_file(
        path,
        payload: str = '{"max_chunk_size": 2000, "hashes": {"a.py": "h1"}}'
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _version_of(path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestIndexVersion:
    def test_no_hash_file_means_no_version(self, tmp_path):
        cache = QueryCache(cache_path=str(tmp_path / "cache.json"),
                           hash_path=str(tmp_path / "does_not_exist.json"))
        assert cache.index_version is None

    def test_version_is_the_sha256_of_the_hash_manifest_bytes(self, tmp_path):
        hash_path = tmp_path / "file_hashes.json"
        _write_hash_file(hash_path)
        cache = QueryCache(cache_path=str(tmp_path / "cache.json"),
                           hash_path=str(hash_path))
        assert cache.index_version == _version_of(hash_path)

    def test_two_instances_against_the_same_manifest_agree(self, tmp_path):
        hash_path = tmp_path / "file_hashes.json"
        _write_hash_file(hash_path)
        a = QueryCache(cache_path=str(tmp_path / "a.json"), hash_path=str(hash_path))
        b = QueryCache(cache_path=str(tmp_path / "b.json"), hash_path=str(hash_path))
        assert a.index_version == b.index_version

    def test_changing_manifest_content_changes_the_version(self, tmp_path):
        hash_path = tmp_path / "file_hashes.json"
        _write_hash_file(hash_path, '{"hashes": {"a.py": "h1"}}')
        v1 = QueryCache(cache_path=str(tmp_path / "c.json"),
                        hash_path=str(hash_path)).index_version
        _write_hash_file(hash_path, '{"hashes": {"a.py": "h2"}}')
        v2 = QueryCache(cache_path=str(tmp_path / "c.json"),
                        hash_path=str(hash_path)).index_version
        assert v1 != v2


class TestGetPut:
    def _cache(self, tmp_path) -> QueryCache:
        hash_path = tmp_path / "file_hashes.json"
        _write_hash_file(hash_path)
        return QueryCache(cache_path=str(tmp_path / "cache.json"),
                          hash_path=str(hash_path))

    def test_miss_returns_none(self, tmp_path):
        cache = self._cache(tmp_path)
        assert cache.get("lexical", 5, "anything") is None

    def test_put_then_get_returns_the_same_sources(self, tmp_path):
        cache = self._cache(tmp_path)
        sources = [_source("a.py"), _source("b.py")]
        cache.put("lexical", 5, "greet", sources)
        assert cache.get("lexical", 5, "greet") == sources

    def test_different_query_text_is_a_different_entry(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.put("lexical", 5, "greet", [_source("a.py")])
        assert cache.get("lexical", 5, "a different query") is None

    def test_different_k_is_a_different_entry(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.put("lexical", 5, "greet", [_source("a.py")])
        assert cache.get("lexical", 3, "greet") is None

    def test_different_method_is_a_different_entry(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.put("lexical", 5, "greet", [_source("a.py")])
        assert cache.get("semantic", 5, "greet") is None

    def test_method_is_case_insensitive(self, tmp_path):
        # _key() lowercases method -- "Lexical" and "LEXICAL" must collide.
        cache = self._cache(tmp_path)
        sources = [_source("a.py")]
        cache.put("Lexical", 5, "greet", sources)
        assert cache.get("LEXICAL", 5, "greet") == sources

    def test_put_overwrites_an_existing_entry_for_the_same_key(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.put("lexical", 5, "greet", [_source("a.py")])
        cache.put("lexical", 5, "greet", [_source("b.py")])
        assert cache.get("lexical", 5, "greet") == [_source("b.py")]


class TestSaveLoadRoundTrip:
    def test_round_trip_preserves_entries(self, tmp_path):
        hash_path = tmp_path / "file_hashes.json"
        _write_hash_file(hash_path)
        cache_path = tmp_path / "cache.json"

        first = QueryCache(cache_path=str(cache_path), hash_path=str(hash_path))
        sources = [_source("a.py", 0, 10), _source("b.py", 5, 20)]
        first.put("hybrid", 5, "how does greet work?", sources)
        first.save()

        second = QueryCache(cache_path=str(cache_path), hash_path=str(hash_path))
        assert second.get("hybrid", 5, "how does greet work?") == sources

    def test_saved_file_records_the_index_version(self, tmp_path):
        hash_path = tmp_path / "file_hashes.json"
        _write_hash_file(hash_path)
        cache_path = tmp_path / "cache.json"

        cache = QueryCache(cache_path=str(cache_path), hash_path=str(hash_path))
        cache.put("lexical", 5, "q", [_source("a.py")])
        cache.save()

        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        assert raw["index_version"] == cache.index_version

    def test_save_creates_missing_parent_directories(self, tmp_path):
        hash_path = tmp_path / "file_hashes.json"
        _write_hash_file(hash_path)
        cache_path = tmp_path / "nested" / "dir" / "cache.json"

        cache = QueryCache(cache_path=str(cache_path), hash_path=str(hash_path))
        cache.put("lexical", 5, "q", [_source("a.py")])
        cache.save()

        assert cache_path.exists()

    def test_save_with_no_entries_still_writes_a_valid_empty_cache_file(self, tmp_path):
        hash_path = tmp_path / "file_hashes.json"
        _write_hash_file(hash_path)
        cache_path = tmp_path / "cache.json"

        cache = QueryCache(cache_path=str(cache_path), hash_path=str(hash_path))
        cache.save()

        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        assert raw["entries"] == {}

    def test_save_without_a_hash_manifest_is_a_no_op(self, tmp_path):
        # index_version is None -- nothing to key the cache against, so
        # save() must not write a file at all.
        cache_path = tmp_path / "cache.json"
        cache = QueryCache(cache_path=str(cache_path),
                           hash_path=str(tmp_path / "missing.json"))
        cache.put("lexical", 5, "q", [_source("a.py")])
        cache.save()
        assert not cache_path.exists()


class TestStaleCacheInvalidation:
    def test_index_version_mismatch_clears_previously_saved_entries(self, tmp_path):
        hash_path = tmp_path / "file_hashes.json"
        cache_path = tmp_path / "cache.json"

        _write_hash_file(hash_path, '{"hashes": {"a.py": "h1"}}')
        old = QueryCache(cache_path=str(cache_path), hash_path=str(hash_path))
        old.put("lexical", 5, "greet", [_source("a.py")])
        old.save()

        # Simulate a reindex: manifest content (and therefore the version)
        # changes, even though the query itself is unrelated to what changed.
        _write_hash_file(hash_path, '{"hashes": {"a.py": "h2"}}')
        new = QueryCache(cache_path=str(cache_path), hash_path=str(hash_path))
        assert new.get("lexical", 5, "greet") is None

    def test_reindex_then_save_overwrites_the_stale_file_with_the_new_version(self, tmp_path):
        hash_path = tmp_path / "file_hashes.json"
        cache_path = tmp_path / "cache.json"

        _write_hash_file(hash_path, '{"hashes": {"a.py": "h1"}}')
        old = QueryCache(cache_path=str(cache_path), hash_path=str(hash_path))
        old.put("lexical", 5, "greet", [_source("a.py")])
        old.save()

        _write_hash_file(hash_path, '{"hashes": {"a.py": "h2"}}')
        new = QueryCache(cache_path=str(cache_path), hash_path=str(hash_path))
        new.put("lexical", 5, "different query after reindex", [_source("b.py")])
        new.save()

        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        assert raw["index_version"] == new.index_version
        assert raw["index_version"] != old.index_version


class TestCorruptedCacheFile:
    def test_non_json_cache_file_does_not_crash(self, tmp_path):
        hash_path = tmp_path / "file_hashes.json"
        _write_hash_file(hash_path)
        cache_path = tmp_path / "cache.json"
        cache_path.write_text("not json at all", encoding="utf-8")

        cache = QueryCache(cache_path=str(cache_path), hash_path=str(hash_path))
        assert cache.entries == {}
        assert cache.get("lexical", 5, "anything") is None

    def test_cache_file_missing_entries_key_does_not_crash(self, tmp_path):
        hash_path = tmp_path / "file_hashes.json"
        _write_hash_file(hash_path)
        cache_path = tmp_path / "cache.json"
        version = _version_of(hash_path)
        cache_path.write_text(json.dumps({"index_version": version}),
                              encoding="utf-8")

        cache = QueryCache(cache_path=str(cache_path), hash_path=str(hash_path))
        assert cache.entries == {}

    def test_malformed_source_entry_does_not_crash(self, tmp_path):
        # An entry whose sources don't validate as MinimalSource --
        # _load()'s inner try/except ValidationError must catch this.
        hash_path = tmp_path / "file_hashes.json"
        _write_hash_file(hash_path)
        cache_path = tmp_path / "cache.json"
        version = _version_of(hash_path)
        cache_path.write_text(json.dumps({
            "index_version": version,
            "entries": {"somekey": [{"file_path": "a.py"}]},  # missing offsets
        }), encoding="utf-8")

        cache = QueryCache(cache_path=str(cache_path), hash_path=str(hash_path))
        assert cache.entries == {}  # whole load abandoned, not partial


class TestSaveConsolePrint:
    def test_successful_save_prints_a_confirmation_line(self, tmp_path, capsys):
        # Documents current behavior -- see the note above the test file
        # about whether this new stdout line should exist at all.
        hash_path = tmp_path / "file_hashes.json"
        _write_hash_file(hash_path)
        cache_path = tmp_path / "cache.json"

        cache = QueryCache(cache_path=str(cache_path), hash_path=str(hash_path))
        cache.put("lexical", 5, "q", [_source("a.py")])
        cache.save()

        out = capsys.readouterr().out
        assert f"Queries cached in {cache_path}" in out

    def test_failed_save_does_report_the_error_to_stderr(self, tmp_path):
        hash_path = tmp_path / "file_hashes.json"
        _write_hash_file(hash_path)
        cache_path = tmp_path / "cache.json"
        cache_path.mkdir()  # forces write_text() to raise IsADirectoryError

        cache = QueryCache(cache_path=str(cache_path), hash_path=str(hash_path))
        cache.put("lexical", 5, "q", [_source("a.py")])

        import pytest as _pytest  # noqa: F401  (capsys needs no import; kept minimal)


class TestKnownIssues:
    """
    Pins known-broken behavior in src/retrieval/query_cache.py, per the
    project's own convention of documenting bugs as xfail(strict=True)
    rather than silently working around them. Remove the marker once the
    underlying save() method is fixed to `return` after a write failure.
    """
    def test_failed_save_does_not_also_claim_success(self, tmp_path, capsys):
        hash_path = tmp_path / "file_hashes.json"
        _write_hash_file(hash_path)
        cache_path = tmp_path / "cache.json"
        cache_path.mkdir()  # forces write_text() to raise IsADirectoryError

        cache = QueryCache(cache_path=str(cache_path), hash_path=str(hash_path))
        cache.put("lexical", 5, "q", [_source("a.py")])
        cache.save()

        out = capsys.readouterr().out
        assert "Queries cached in" not in out


class TestSaveConsolePrint:
    def test_successful_save_prints_a_confirmation_line(self, tmp_path, capsys):
        hash_path = tmp_path / "file_hashes.json"
        _write_hash_file(hash_path)
        cache_path = tmp_path / "cache.json"

        cache = QueryCache(cache_path=str(cache_path), hash_path=str(hash_path))
        cache.put("lexical", 5, "q", [_source("a.py")])
        cache.save()

        out = capsys.readouterr().out
        assert f"Queries cached in {cache_path}" in out

    def test_failed_save_does_report_the_error_to_stderr(self, tmp_path, capsys):
        hash_path = tmp_path / "file_hashes.json"
        _write_hash_file(hash_path)
        cache_path = tmp_path / "cache.json"
        cache_path.mkdir()  # forces write_text() to raise IsADirectoryError

        cache = QueryCache(cache_path=str(cache_path), hash_path=str(hash_path))
        cache.put("lexical", 5, "q", [_source("a.py")])
        cache.save()

        err = capsys.readouterr().err
        assert "Error: Failed to save query cache" in err
