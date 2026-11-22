"""Small TTL + LRU cache, used for SPARQL results and for API answers."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

V = TypeVar("V")


class TTLCache(Generic[V]):
    def __init__(self, maxsize: int = 512, ttl_s: float = 600.0, clock: Callable[[], float] = time.monotonic):
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, V]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> V | None:
        item = self._data.get(key)
        if item is None:
            self.misses += 1
            return None
        stamp, value = item
        if self._clock() - stamp > self.ttl_s:
            del self._data[key]
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: str, value: V) -> None:
        self._data[key] = (self._clock(), value)
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)
