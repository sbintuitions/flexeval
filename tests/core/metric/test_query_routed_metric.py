from __future__ import annotations

import pytest

from flexeval.core.metric import ExactMatch, Metric, MetricResult, MetricRoute, QueryRoutedMetric

OUTPUTS = ["a", "b", "c", "d"]
REFS = [["a"], ["x"], ["c"], ["d"]]
INFOS = [
    {"task": "qa", "meta": {"lang": "ja"}, "domain": "x", "score": 1},
    {"task": "qa", "meta": {"lang": "en"}, "domain": "y", "score": 5},
    {"task": "mt", "meta": {"lang": "ja"}, "domain": "x", "score": 3},
    {"task": "other", "score": 3},
]


def test_basic_routing_and_namespacing() -> None:
    metric = QueryRoutedMetric(
        routes=[
            MetricRoute(name="qa", where={"task": "qa"}, metrics=[ExactMatch()]),
            MetricRoute(name="all", where=None, metrics=[ExactMatch()]),
        ]
    )
    result = metric.evaluate(OUTPUTS, REFS, INFOS)
    assert result.summary["qa/exact_match"] == 0.5
    assert result.summary["all/exact_match"] == 0.75
    assert "qa/exact_match" not in result.instance_details[2]
    assert result.instance_details[1]["qa/exact_match"] is False
    assert len(result.instance_details) == 4


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        ({"task": ["qa", "mt"]}, 3),
        ({"meta.lang": "ja"}, 2),
        ({"task": ["qa", "mt"], "meta.lang": "ja"}, 2),
        ({"task": {"ne": "qa"}}, 2),
        ({"task": {"not_in": ["qa", "mt"]}}, 1),
        ({"task": {"regex": "^(qa|other)$"}}, 3),
        ({"domain": {"exists": False}}, 1),
        ({"meta.lang": {"exists": True}}, 3),
        ({"score": {"gt": 3}}, 1),
        ({"score": {"gte": 3}}, 3),
        ({"score": {"lt": 3}}, 1),
        ({"score": {"lte": 3}}, 3),
        ({"score": {"gte": 3, "lt": 5}}, 2),
        ({"missing": {"gt": 0}}, 0),
        ({"task": {"==": "qa"}}, 2),
        ({"task": {"!=": "qa"}}, 2),
        ({"score": {">": 3}}, 1),
        ({"score": {">=": 3, "<": 5}}, 2),
        ({"score": {"<=": 3}}, 3),
        ({"score": {"gt": 1, ">": 3}}, 1),
    ],
)
def test_where_operators(where: dict, expected: int) -> None:
    route = MetricRoute(name="r", where=where, metrics=[ExactMatch()])
    assert sum(route.matches(info) for info in INFOS) == expected


def test_group_by() -> None:
    metric = QueryRoutedMetric(
        routes=[MetricRoute(name="r", where={"task": ["qa", "mt"]}, group_by="domain", metrics=[ExactMatch()])]
    )
    summary = metric.evaluate(OUTPUTS, REFS, INFOS).summary
    assert summary["r/x/exact_match"] == 1.0
    assert summary["r/y/exact_match"] == 0.0


def test_nesting() -> None:
    inner = QueryRoutedMetric(routes=[MetricRoute(name="ja", where={"meta.lang": "ja"}, metrics=[ExactMatch()])])
    outer = QueryRoutedMetric(routes=[MetricRoute(name="qa", where={"task": "qa"}, metrics=[inner])])
    assert outer.evaluate(OUTPUTS, REFS, INFOS).summary == {"qa/ja/exact_match": 1.0}


def test_empty_subset_is_skipped() -> None:
    metric = QueryRoutedMetric(routes=[MetricRoute(name="r", where={"task": "none"}, metrics=[ExactMatch()])])
    result = metric.evaluate(OUTPUTS, REFS, INFOS)
    assert result.summary == {}


def test_errors() -> None:
    r = MetricRoute(name="r", where={"task": "qa"}, metrics=[ExactMatch()])
    with pytest.raises(ValueError, match="unique"):
        QueryRoutedMetric(routes=[r, r])
    with pytest.raises(ValueError, match="matched no route"):
        QueryRoutedMetric(routes=[r], on_unmatched="error").evaluate(OUTPUTS, REFS, INFOS)
    with pytest.raises(KeyError):
        QueryRoutedMetric(
            routes=[MetricRoute(name="r", where={"domain": "x"}, metrics=[ExactMatch()])], on_missing_key="error"
        ).evaluate(OUTPUTS, REFS, INFOS)
    with pytest.raises(ValueError, match="extra_info"):
        QueryRoutedMetric(routes=[r]).evaluate(OUTPUTS, REFS)
    with pytest.raises(ValueError, match="Unknown operator"):
        MetricRoute(name="r", where={"task": {"foo": 1}}, metrics=[ExactMatch()])


def test_cleanup_propagates() -> None:
    class Dummy(Metric):
        cleaned = False

        def evaluate(self, lm_outputs, references_list, extra_info_list=None) -> MetricResult:  # noqa: ANN001
            return MetricResult({})

        def cleanup_resources(self) -> None:
            self.cleaned = True

    dummy = Dummy()
    QueryRoutedMetric(routes=[MetricRoute(name="d", metrics=[dummy])]).cleanup_resources()
    assert dummy.cleaned


def test_multiple_metrics_in_route() -> None:
    metric = QueryRoutedMetric(
        routes=[
            MetricRoute(
                name="qa",
                where={"task": "qa"},
                group_by="domain",
                metrics=[ExactMatch(), ExactMatch(metric_key="em2")],
            )
        ]
    )
    result = metric.evaluate(OUTPUTS, REFS, INFOS)
    assert result.summary["qa/exact_match"] == 0.5
    assert result.summary["qa/em2"] == 0.5
    assert result.summary["qa/x/em2"] == 1.0
    assert result.instance_details[0]["qa/em2"] is True


def test_same_key_metrics_collide_and_empty_metrics_rejected() -> None:
    route = MetricRoute(name="qa", where={"task": "qa"}, metrics=[ExactMatch(), ExactMatch()])
    with pytest.raises(ValueError, match="collision"):
        QueryRoutedMetric(routes=[route]).evaluate(OUTPUTS, REFS, INFOS)
    with pytest.raises(ValueError, match="at least one"):
        MetricRoute(name="r", metrics=[])
