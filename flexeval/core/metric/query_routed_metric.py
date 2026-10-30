from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger

from flexeval.core.language_model.base import LMOutput

from .base import Metric, MetricResult
from .utils import validate_inputs

_MISSING = object()
_OPERATORS = {
    "eq",
    "ne",
    "in",
    "not_in",
    "regex",
    "exists",
    "gt",
    "gte",
    "lt",
    "lte",
    "==",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "elem_match",
}


def _resolve(current: Any, parts: list[str]) -> list[Any]:  # noqa: ANN401
    """Resolve path parts against `current`. A ``*`` part fans out over all list items."""
    if not parts:
        return [current]
    part, rest = parts[0], parts[1:]
    if isinstance(current, dict):
        return _resolve(current[part], rest) if part in current else []
    if isinstance(current, list):
        if part == "*":
            return [v for item in current for v in _resolve(item, rest)]
        try:
            return _resolve(current[int(part)], rest)
        except (ValueError, IndexError):
            return []
    return []


def _lookup(extra_info: dict[str, Any], key: str) -> Any:  # noqa: ANN401
    """
    Resolve a (possibly dotted) key in extra_info. Returns `_MISSING` if absent.

    List items can be addressed by index (``messages.0.role``, ``messages.-1.content``).
    ``*`` fans out over all list items (``messages.*.role``) and returns a `_Multi` of the matched values.
    """
    if key in extra_info:
        return extra_info[key]
    parts = key.split(".")
    values = _resolve(extra_info, parts)
    if "*" in parts:
        return _Multi(values) if values else _MISSING
    return values[0] if values else _MISSING


class _Multi(list):
    """Values collected by a ``*`` path. Positive operators match if any value matches."""


def _normalize_condition(condition: Any) -> dict[str, Any]:  # noqa: ANN401
    """Normalize a condition into `{operator: operand}` form."""
    if isinstance(condition, dict):
        unknown = set(condition) - _OPERATORS
        if unknown:
            msg = f"Unknown operator(s) {sorted(unknown)} in where clause. Supported: {sorted(_OPERATORS)}."
            raise ValueError(msg)
        if "elem_match" in condition:
            condition = {**condition, "elem_match": _normalize_elem_query(condition["elem_match"])}
        return condition
    if isinstance(condition, list):
        return {"in": condition}
    return {"eq": condition}


def _normalize_where(where: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {key: _normalize_condition(cond) for key, cond in (where or {}).items()}


def _normalize_elem_query(query: Any) -> tuple[str, Any]:  # noqa: ANN401
    """
    Normalize the operand of `elem_match`.

    A dict with keys other than operators is a `where` query on each (dict) item: ``("where", conditions)``.
    Anything else is a condition on each (scalar) item itself: ``("cond", {operator: operand})``.
    """
    if isinstance(query, dict) and not set(query) <= _OPERATORS:
        return "where", _normalize_where(query)
    return "cond", _normalize_condition(query)


def _where_matches(conditions: dict[str, dict[str, Any]], info: dict[str, Any]) -> bool:
    return all(
        all(_check_operator(op, operand, _lookup(info, key)) for op, operand in ops.items())
        for key, ops in conditions.items()
    )


def _elem_matches(operand: tuple[str, Any], value: Any) -> bool:  # noqa: ANN401
    """Whether at least one item of the list `value` satisfies the normalized `elem_match` operand."""
    if not isinstance(value, list):
        return False
    kind, query = operand
    if kind == "where":
        return any(isinstance(item, dict) and _where_matches(query, item) for item in value)
    return any(all(_check_operator(op, o, item) for op, o in query.items()) for item in value)


def _check_operator(op: str, operand: Any, value: Any) -> bool:  # noqa: ANN401, C901, PLR0911, PLR0912
    if op == "exists":
        return (value is not _MISSING) == bool(operand)
    if value is _MISSING:
        return False
    if isinstance(value, _Multi):
        # negative operators must hold for every value, the others for at least one
        aggregate = all if op in {"ne", "!=", "not_in"} else any
        return aggregate(_check_operator(op, operand, v) for v in value)
    if op == "elem_match":
        return _elem_matches(operand, value)
    if op in {"eq", "=="}:
        return value == operand
    if op in {"ne", "!="}:
        return value != operand
    if op == "in":
        return value in operand
    if op == "not_in":
        return value not in operand
    if op == "regex":
        return isinstance(value, str) and re.search(operand, value) is not None
    if op in {"gt", ">"}:
        return value > operand
    if op in {"gte", ">="}:
        return value >= operand
    if op in {"lt", "<"}:
        return value < operand
    if op in {"lte", "<="}:
        return value <= operand
    msg = f"Unknown operator: {op}"
    raise ValueError(msg)


@dataclass
class MetricRoute:
    """
    A group of metrics and a query that selects the instances the metrics are applied to.

    Args:
        name: Name of the subset selected by this route (not of a metric).
            Used as the namespace of the keys in the summary and the instance details,
            e.g. ``{name}/{metric_key}``. Must be unique within a `QueryRoutedMetric`.
        metrics: The metrics applied to the selected instances. The selection is done once per route.
            Metrics emitting the same key cause a collision error; distinguish them with e.g. `metric_key`.
        where: Query on `extra_info`. ``None`` selects all instances.
            Keys are AND-ed. Each key may use dotted notation for nested dicts (e.g. ``"meta.lang"``).
            List items are addressed by index (``"messages.0.role"``, ``"messages.-1.content"``),
            or by ``*`` for all items (``"messages.*.role"``): positive operators match if any item matches,
            ``ne`` / ``not_in`` only if all items do.
            Each key is evaluated independently, so conditions on different keys
            may be satisfied by different list items; use ``elem_match`` to constrain a single item.
            A value is one of:

            - a scalar: equality (``{"task": "qa"}``)
            - a list: membership (``{"task": ["qa", "mt"]}``)
            - a dict of operators: ``eq``, ``ne``, ``in``, ``not_in``, ``regex`` (``re.search``), ``exists``,
              ``gt``, ``gte``, ``lt``, ``lte``, ``elem_match``.
              ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=`` are accepted as aliases of the named operators.
              Multiple operators are AND-ed (``{"title": {"regex": "^A", "ne": "AB"}}``,
              ``{"score": {"gte": 1, "lt": 5}}``).
              Operands and values are not type-checked; the operators are applied as Python does.
              Note that ``in`` / ``not_in`` with a string operand do substring matching,
              that comparisons between strings are lexicographic (``"10" < "9"``),
              and that comparing incompatible types (e.g. ``int`` and ``str``) raises ``TypeError``.
            - ``elem_match``: matches a list if at least one item satisfies all the given conditions.
              For dict items, give a `where`-style query (``{"messages": {"elem_match": {"role": "user",
              "content": {"regex": "^h"}}}}``); for scalar items, give a condition on the item itself
              (``{"tags": {"elem_match": {"regex": "^a"}}}``). A key missing in an item just makes it not match.

            Instances lacking a queried key do not match (except for ``exists: false``).
        group_by: If given, additionally reports the mean of each numeric instance-level value
            of every metric per value of this `extra_info` key,
            as ``{name}/{group_value}/{key}``.
    """

    name: str
    metrics: list[Metric]
    where: dict[str, Any] | None = None
    group_by: str | None = None

    def __post_init__(self) -> None:
        if not self.metrics:
            msg = f"Route '{self.name}' must have at least one metric."
            raise ValueError(msg)
        if self.group_by is not None and "*" in self.group_by.split("."):
            msg = f"Route '{self.name}': group_by does not support '*' (got '{self.group_by}'). Use an index instead."
            raise ValueError(msg)
        self._conditions = _normalize_where(self.where)

    def matches(self, extra_info: dict[str, Any] | None, on_missing_key: str = "skip") -> bool:
        if not self._conditions:
            return True
        if extra_info is None:
            msg = f"Route '{self.name}' has a `where` query but extra_info is not available."
            raise ValueError(msg)
        for key, ops in self._conditions.items():
            value = _lookup(extra_info, key)
            if value is _MISSING and on_missing_key == "error" and "exists" not in ops:
                msg = f"Key '{key}' used in route '{self.name}' is missing in extra_info: {extra_info}"
                raise KeyError(msg)
            if not all(_check_operator(op, operand, value) for op, operand in ops.items()):
                return False
        return True


class QueryRoutedMetric(Metric):
    """
    Apply different metrics to different subsets of the data, selected by queries on `extra_info`.

    Each route selects instances with its `where` query and evaluates its metrics on them only.
    An instance may match several routes, or none.
    Summary keys and instance-detail keys are namespaced as ``{route.name}/{key}``.
    Since this class is itself a `Metric`, it can be nested to combine multiple keys.

    Args:
        routes: List of `MetricRoute`.
        on_unmatched: What to do with instances matching no route. ``"skip"`` ignores them, ``"error"`` raises.
        on_missing_key: What to do when a queried key is absent from an instance's extra_info.
            ``"skip"`` treats the instance as not matching, ``"error"`` raises `KeyError`.

    Examples:
        >>> from flexeval import ExactMatch, OutputLengthStats, MetricRoute, QueryRoutedMetric
        >>> metric = QueryRoutedMetric(
        ...     routes=[
        ...         MetricRoute(name="qa", where={"task": "qa"}, metrics=[ExactMatch()]),
        ...         MetricRoute(name="length", where=None, metrics=[OutputLengthStats()]),
        ...     ]
        ... )
        >>> result = metric.evaluate(
        ...     ["A", "B"], [["A"], ["C"]], [{"task": "qa"}, {"task": "mt"}]
        ... )
        >>> result.summary["qa/exact_match"]
        1.0
    """

    def __init__(
        self,
        routes: list[MetricRoute],
        on_unmatched: Literal["skip", "error"] = "skip",
        on_missing_key: Literal["skip", "error"] = "skip",
    ) -> None:
        names = [route.name for route in routes]
        duplicated = {n for n in names if names.count(n) > 1}
        if duplicated:
            msg = f"Route names must be unique, but got duplicates: {sorted(duplicated)}"
            raise ValueError(msg)
        self.routes = routes
        self.on_unmatched = on_unmatched
        self.on_missing_key = on_missing_key

    def evaluate(
        self,
        lm_outputs: list[str | LMOutput],
        references_list: list[list[str]],
        extra_info_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        validate_inputs(lm_outputs, references_list, extra_info_list)
        n = len(lm_outputs)
        infos: list[dict[str, Any] | None] = list(extra_info_list) if extra_info_list is not None else [None] * n

        matched_any = [False] * n
        summary: dict[str, Any] = {}
        instance_details: list[dict[str, Any]] = [{} for _ in range(n)]
        has_details = False

        for route in self.routes:
            indices = [i for i in range(n) if route.matches(infos[i], self.on_missing_key)]
            logger.info(f"Route '{route.name}': matched {len(indices)}/{n} instances")
            for i in indices:
                matched_any[i] = True
            if not indices:
                continue

            for metric in route.metrics:
                result = metric.evaluate(
                    [lm_outputs[i] for i in indices],
                    [references_list[i] for i in indices],
                    [infos[i] for i in indices] if extra_info_list is not None else None,
                )
                has_details |= self._add_result(summary, instance_details, route, indices, infos, result)

        if self.on_unmatched == "error" and not all(matched_any):
            unmatched = [i for i, m in enumerate(matched_any) if not m]
            msg = f"{len(unmatched)} instance(s) matched no route (e.g. index {unmatched[:5]})."
            raise ValueError(msg)

        return MetricResult(summary, instance_details=instance_details if has_details else None)

    def _add_result(
        self,
        summary: dict[str, Any],
        instance_details: list[dict[str, Any]],
        route: MetricRoute,
        indices: list[int],
        infos: list[dict[str, Any] | None],
        result: MetricResult,
    ) -> bool:
        """Merge a metric result into the outputs. Returns whether the result had instance details."""
        for key, value in result.summary.items():
            self._set_summary(summary, f"{route.name}/{key}", value)

        if result.instance_details is None:
            if route.group_by is not None:
                logger.warning(f"Route '{route.name}': group_by is ignored because a metric returns no details.")
            return False

        self._merge_details(instance_details, route, indices, result.instance_details)
        if route.group_by is not None:
            self._add_group_summary(summary, route, indices, infos, result.instance_details)
        return True

    @staticmethod
    def _merge_details(
        instance_details: list[dict[str, Any]],
        route: MetricRoute,
        indices: list[int],
        details: list[dict[str, Any]],
    ) -> None:
        for i, detail in zip(indices, details):
            for key, value in detail.items():
                full_key = f"{route.name}/{key}"
                if full_key in instance_details[i]:
                    msg = f"Instance detail key collision: '{full_key}'. Use distinct route names or metric keys."
                    raise ValueError(msg)
                instance_details[i][full_key] = value

    @staticmethod
    def _set_summary(summary: dict[str, Any], key: str, value: Any) -> None:  # noqa: ANN401
        if key in summary:
            msg = f"Summary key collision: '{key}'. Use distinct route names or metric keys."
            raise ValueError(msg)
        summary[key] = value

    def _add_group_summary(
        self,
        summary: dict[str, Any],
        route: MetricRoute,
        indices: list[int],
        infos: list[dict[str, Any] | None],
        details: list[dict[str, Any]],
    ) -> None:
        if any(infos[i] is None for i in indices):
            msg = "group_by requires extra_info."
            raise ValueError(msg)
        scores: dict[tuple[Any, str], list[float]] = defaultdict(list)
        for i, detail in zip(indices, details):
            group = _lookup(infos[i], route.group_by)
            if group is _MISSING:
                if self.on_missing_key == "error":
                    msg = f"group_by key '{route.group_by}' is missing in extra_info: {infos[i]}"
                    raise KeyError(msg)
                continue
            if isinstance(group, (list, dict)):
                msg = (
                    f"group_by key '{route.group_by}' must resolve to a scalar value, "
                    f"but got {type(group).__name__}: {group!r}"
                )
                raise TypeError(msg)
            for key, value in detail.items():
                if isinstance(value, (bool, int, float)):
                    scores[(group, key)].append(value)
        for (group, key), values in scores.items():
            self._set_summary(summary, f"{route.name}/{group}/{key}", sum(values) / len(values))

    def cleanup_resources(self) -> None:
        for route in self.routes:
            for metric in route.metrics:
                metric.cleanup_resources()
