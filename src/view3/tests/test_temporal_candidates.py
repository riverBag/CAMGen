"""Temporal and repository boundaries for historical exemplar retrieval."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retriever.candidate_filter import HistoricalCandidateIndex


def commit(sha, date="2024-01-01T00:00:00Z", *, owner="apache", repo="arrow"):
    return {"owner": owner, "repo": repo, "commit_sha": sha, "date": date}


class HistoricalCandidateIndexTests(unittest.TestCase):
    def test_only_strictly_earlier_commits_are_eligible(self):
        items = [
            commit("earlier", "2024-01-01T23:59:59.999999Z"),
            commit("equal", "2024-01-02T00:00:00Z"),
            commit("later", "2024-01-02T00:00:00.000001Z"),
        ]
        index = HistoricalCandidateIndex(items)

        self.assertEqual(
            index.candidate_indices(commit("query", "2024-01-02T00:00:00Z")),
            [0],
        )

    def test_offsets_are_compared_as_utc_instants(self):
        items = [
            commit("earlier", "2024-01-02T07:59:59+08:00"),
            commit("equal", "2024-01-01T19:00:00-05:00"),
            commit("later", "2024-01-01T20:00:00-05:00"),
        ]
        index = HistoricalCandidateIndex(items)

        self.assertEqual(
            index.candidate_indices(commit("query", "2024-01-02T01:00:00+01:00")),
            [0],
        )

    def test_owner_and_repo_both_define_repository(self):
        items = [
            commit("same-repository"),
            commit("other-owner", owner="another-org"),
            commit("other-repo", repo="airflow"),
        ]
        index = HistoricalCandidateIndex(items)

        self.assertEqual(
            index.candidate_indices(commit("query", "2024-01-02T00:00:00Z")),
            [0],
        )

    def test_self_is_excluded_even_if_its_stored_date_is_earlier(self):
        items = [commit("query"), commit("historical")]
        index = HistoricalCandidateIndex(items)

        self.assertEqual(
            index.candidate_indices(commit("query", "2024-01-02T00:00:00Z")),
            [1],
        )

    def test_candidate_indices_preserve_corpus_order_not_date_order(self):
        items = [
            commit("recent", "2024-01-04T00:00:00Z"),
            commit("future", "2024-01-06T00:00:00Z"),
            commit("oldest", "2024-01-01T00:00:00Z"),
            commit("equal", "2024-01-05T00:00:00Z"),
            commit("middle", "2024-01-03T00:00:00Z"),
        ]
        index = HistoricalCandidateIndex(items)

        self.assertEqual(
            index.candidate_indices(commit("query", "2024-01-05T00:00:00Z")),
            [0, 2, 4],
        )

    def test_no_eligible_history_returns_empty_list(self):
        index = HistoricalCandidateIndex([
            commit("equal", "2024-01-02T00:00:00Z"),
            commit("future", "2024-01-03T00:00:00Z"),
            commit("different-repo", repo="airflow"),
        ])

        self.assertEqual(
            index.candidate_indices(commit("query", "2024-01-02T00:00:00Z")),
            [],
        )
        self.assertEqual(
            index.candidate_indices(commit("query", "2024-01-02T00:00:00Z", repo="unknown")),
            [],
        )
        self.assertEqual(HistoricalCandidateIndex([]).candidate_indices(commit("query")), [])

    def test_same_query_sha_in_different_repositories_does_not_share_candidates(self):
        index = HistoricalCandidateIndex([
            commit("arrow-history", repo="arrow"),
            commit("airflow-history", repo="airflow"),
        ])

        self.assertEqual(
            index.candidate_indices(commit("shared-sha", "2024-01-02T00:00:00Z", repo="arrow")),
            [0],
        )
        self.assertEqual(
            index.candidate_indices(commit("shared-sha", "2024-01-02T00:00:00Z", repo="airflow")),
            [1],
        )

    def test_query_timestamp_changes_the_candidate_boundary(self):
        index = HistoricalCandidateIndex([commit("history", "2024-01-02T00:00:00Z")])

        self.assertEqual(
            index.candidate_indices(commit("query", "2024-01-03T00:00:00Z")), [0]
        )
        self.assertEqual(
            index.candidate_indices(commit("query", "2024-01-01T00:00:00Z")), []
        )

    def test_invalid_candidate_metadata_raises_during_index_construction(self):
        for item in self.invalid_records():
            with self.subTest(item=item):
                with self.assertRaises(ValueError):
                    HistoricalCandidateIndex([item])

    def test_invalid_query_metadata_raises_during_filtering(self):
        index = HistoricalCandidateIndex([commit("history")])
        for query in self.invalid_records():
            with self.subTest(query=query):
                with self.assertRaises(ValueError):
                    index.candidate_indices(query)

    @staticmethod
    def invalid_records():
        for field in ("owner", "repo", "commit_sha", "date"):
            item = commit("query")
            del item[field]
            yield item

        for field in ("owner", "repo", "commit_sha"):
            for value in (None, "", "   ", 123, True, []):
                item = commit("query")
                item[field] = value
                yield item

        for value in (
            None,
            "",
            "   ",
            "not-a-date",
            "2024-02-30T00:00:00Z",
            "2024-01-01",
            "2024-01-01T00:00:00",
            1704067200,
            True,
        ):
            item = commit("query")
            item["date"] = value
            yield item


if __name__ == "__main__":
    unittest.main()
