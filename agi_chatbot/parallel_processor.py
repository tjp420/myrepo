from typing import Callable, Iterable, List, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class ParallelProcessor:
    def __init__(self, max_workers: int = 1) -> None:
        self.max_workers = max_workers

    def map(self, func: Callable[[T], R], inputs: Iterable[T]) -> List[R]:
        return [func(x) for x in inputs]
