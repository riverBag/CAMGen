"""Select only commits from the query repository's strictly earlier history."""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


def _required_text(item: Mapping[str, Any], field: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid {field} for commit {item.get('commit_sha')!r}")
    return value


def _metadata(item: Mapping[str, Any]):
    repository = (_required_text(item, "owner"), _required_text(item, "repo"))
    commit_sha = _required_text(item, "commit_sha")
    date = _required_text(item, "date")
    try:
        timestamp = datetime.fromisoformat(date.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid ISO 8601 date for commit {commit_sha!r}: {date!r}") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"Commit {commit_sha!r} date must include a timezone: {date!r}")
    return repository, commit_sha, timestamp.astimezone(timezone.utc)


class HistoricalCandidateIndex:
    """Keep corpus positions aligned with BM25 and dense embedding indexes.

    Both indexed records and queries use ApacheCM's owner/repo/commit_sha/date
    fields. Dates must be timezone-aware ISO 8601 strings. Invalid metadata
    raises rather than allowing a candidate with unknown provenance or time.
    """

    def __init__(self, items: Sequence[Mapping[str, Any]]):
        self._repositories = defaultdict(list)
        for position, item in enumerate(items):
            try:
                repository, commit_sha, timestamp = _metadata(item)
            except ValueError as exc:
                raise ValueError(f"Invalid retrieval corpus record at index {position}: {exc}") from exc
            self._repositories[repository].append((position, commit_sha, timestamp))

    def candidate_indices(self, query: Mapping[str, Any]) -> list[int]:
        """Return original corpus indices before scoring or normalization."""
        repository, query_sha, query_timestamp = _metadata(query)
        return [
            position
            for position, commit_sha, timestamp in self._repositories.get(repository, ())
            if commit_sha != query_sha and timestamp < query_timestamp
        ]
