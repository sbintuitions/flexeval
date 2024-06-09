import pytest

from flexeval.core.utils.data_util import batch_iter


def test_batch_iter_normal_case() -> None:
    items = range(10)
    batches = list(batch_iter(items, 3))
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_batch_iter_empty_input() -> None:
    items = []
    batches = list(batch_iter(items, 3))
    assert batches == []


def test_batch_iter_single_item_batches() -> None:
    items = range(5)
    batches = list(batch_iter(items, 1))
    assert batches == [[0], [1], [2], [3], [4]]


def test_batch_iter_large_batch_size() -> None:
    items = range(5)
    batches = list(batch_iter(items, 10))
    assert batches == [[0, 1, 2, 3, 4]]


def test_batch_iter_invalid_batch_size() -> None:
    with pytest.raises(ValueError):
        list(batch_iter(range(5), 0))
