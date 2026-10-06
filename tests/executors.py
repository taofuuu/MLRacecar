"""An executor that runs work immediately, for testing code that normally uses a thread."""

from collections.abc import Callable
from concurrent.futures import Executor, Future


class InlineExecutor(Executor):
    """Runs each task as soon as it's submitted, on the caller's thread: results are ready at
    once and tests stay deterministic."""

    def __init__(self) -> None:
        self.submitted = 0

    def submit[**P, T](self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Future[T]:
        self.submitted += 1
        future: Future[T] = Future()
        future.set_result(fn(*args, **kwargs))
        return future
