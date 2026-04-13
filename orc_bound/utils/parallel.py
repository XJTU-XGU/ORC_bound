"""
Parallel utilities for ORC-Bound.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

__all__ = ["ThreadPool", "_cpu_count"]


def _cpu_count(n_jobs: int | None) -> int:
    """
    Resolve the number of workers for parallel execution.

    Parameters
    ----------
    n_jobs : int or None
        - ``None`` or ``0``: use all CPUs.
        - Positive int: use exactly that many workers.
        - ``-1``: use all CPUs minus one.

    Returns
    -------
    int
        Number of workers to use (always at least 1).
    """
    physical_cores = os.cpu_count() or 1

    if n_jobs is None or n_jobs == 0:
        return max(1, physical_cores)
    if n_jobs == -1:
        return max(1, physical_cores - 1)
    return max(1, n_jobs)


class ThreadPool:
    """
    Thin wrapper around ThreadPoolExecutor for use within orc_bound.

    Parameters
    ----------
    n_workers : int or None
        Number of worker threads. Passed through to :func:`_cpu_count`.
    func : callable
        The function to apply to each task.
    iterable : iterable
        Iterable of tasks. Each task is unpacked as ``func(*task)``.
    """

    def __init__(
        self,
        n_workers: int | None = None,
    ):
        self.n_workers = _cpu_count(n_workers)

    def map(self, func, iterable):
        """
        Apply ``func`` to each item in ``iterable`` in parallel.

        Yields results in completion order (not submission order).

        Parameters
        ----------
        func : callable
            Function to execute. Should be a top-level or module-level
            function for pickling compatibility.
        iterable : iterable
            Each element is passed as a single argument to ``func``.

        Yields
        ------
        Results from ``func`` calls as they complete.
        """
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(func, item) for item in iterable]
            for f in as_completed(futures):
                yield f.result()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# Re-export as_completed for convenience
from concurrent.futures import as_completed

__all__ += ["as_completed"]
