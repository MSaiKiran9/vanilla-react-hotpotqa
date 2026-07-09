"""Small timing helpers."""

from __future__ import annotations

from time import perf_counter


class Timer:
    """Context manager that records elapsed wall-clock seconds."""

    def __enter__(self):
        self.start = perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = perf_counter() - self.start
        return False
