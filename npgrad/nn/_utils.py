from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter


def pair(x: int | tuple[int, int]) -> tuple[int, int]:
    return x if isinstance(x, tuple) else (x, x)


@contextmanager
def timed(s: str | None = None):
    start = perf_counter()
    yield
    duration = perf_counter() - start
    prefix = f"{s} " if s else ""
    print(f"{prefix}time: {duration}")
