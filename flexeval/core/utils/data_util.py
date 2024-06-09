from __future__ import annotations

from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def batch_iter(iterable: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """
    Yields batches of items from an input iterable, with each batch being of a specified size.

    Args:
        iterable (Iterable[T]): The iterable from which to retrieve the items.
        batch_size (int): The maximum number of items per batch. Must be greater than 0.

    Yields:
        Iterator[list[T]]: An iterator over batches, where each batch is a list of items.

    Raises:
        ValueError: If the batch_size is less than 1.

    Examples:
        >>> list(batch_iter(range(10), 3))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """

    if batch_size < 1:
        msg = "batch_size must be at least 1"
        raise ValueError(msg)

    batch: list[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch
