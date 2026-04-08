"""Shared parallel utilities for estimation helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal, Optional, TypeVar


T = TypeVar("T")
U = TypeVar("U")

ParallelBackend = Literal["threads", "processes"]


def parallel_map(
    items: Iterable[T],
    worker_fn: Callable[[T], U],
    n_jobs: Optional[int] = 1,
    backend: ParallelBackend = "threads",
) -> list[U]:
    """Map ``worker_fn`` over ``items`` with an opt-in parallel backend."""
    if backend not in ("threads", "processes"):
        raise ValueError(
            'backend must be either "threads" or "processes", '
            f"got {backend!r}."
        )

    item_list = list(items)
    if not item_list:
        return []

    if n_jobs is None or n_jobs == 1:
        return [worker_fn(item) for item in item_list]

    try:
        from joblib import Parallel, delayed
    except ImportError:
        return [worker_fn(item) for item in item_list]

    return list(
        Parallel(n_jobs=n_jobs, prefer=backend)(
            delayed(worker_fn)(item) for item in item_list
        )
    )
