"""Minimal dynamic-batching helper (PLAN.md §4, G2/G3): collect concurrent
requests up to max_batch or wait_s, whichever comes first, then run all of
them through `fn` in one call. Framework-agnostic (no Ray dependency), so
it's unit-testable without a GPU -- infra/policy_worker.py wraps this
around the real Pi0.5 forward pass on TC1.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, List


@dataclass
class _Pending:
    item: Any
    result: List[Any] = field(default_factory=lambda: [None])
    event: threading.Event = field(default_factory=threading.Event)


class DynamicBatcher:
    def __init__(self, fn: Callable[[List[Any]], List[Any]], max_batch: int = 16, wait_s: float = 0.01):
        self.fn = fn
        self.max_batch = max_batch
        self.wait_s = wait_s
        self._lock = threading.Lock()
        self._pending: List[_Pending] = []

    def submit(self, item: Any) -> Any:
        p = _Pending(item=item)
        with self._lock:
            self._pending.append(p)
            is_first = len(self._pending) == 1
            full = len(self._pending) >= self.max_batch
        if full:
            self._flush()
        elif is_first:
            threading.Timer(self.wait_s, self._flush).start()
        p.event.wait()
        return p.result[0]

    def _flush(self) -> None:
        with self._lock:
            batch, self._pending = self._pending, []
        if not batch:
            return
        results = self.fn([p.item for p in batch])
        for p, r in zip(batch, results):
            p.result[0] = r
            p.event.set()
