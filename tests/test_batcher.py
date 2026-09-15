import threading
import time
from concurrent.futures import ThreadPoolExecutor

from infra.batcher import DynamicBatcher


def test_single_item_is_processed_via_wait_timeout():
    calls = []

    def fn(batch):
        calls.append(list(batch))
        return [x * 2 for x in batch]

    batcher = DynamicBatcher(fn, max_batch=16, wait_s=0.01)
    result = batcher.submit(21)
    assert result == 42
    assert calls == [[21]]


def test_full_batch_flushes_immediately_without_waiting():
    calls = []

    def fn(batch):
        calls.append(list(batch))
        return [x for x in batch]

    batcher = DynamicBatcher(fn, max_batch=3, wait_s=5.0)  # long wait -- must flush on max_batch instead
    results = [None, None, None]

    def submit(i):
        results[i] = batcher.submit(i)

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(3)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)
    elapsed = time.perf_counter() - start

    assert sorted(results) == [0, 1, 2]
    assert elapsed < 1.0  # proves it didn't wait for the 5s timer
    assert sum(len(c) for c in calls) == 3


def test_concurrent_submits_are_batched_together():
    calls = []
    lock = threading.Lock()

    def fn(batch):
        with lock:
            calls.append(len(batch))
        return list(batch)

    batcher = DynamicBatcher(fn, max_batch=8, wait_s=0.05)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(batcher.submit, range(8)))

    assert sorted(results) == list(range(8))
    assert sum(calls) == 8
    # thread scheduling isn't deterministic, but at least one flush should
    # have caught more than one item -- that's the actual thing under test
    assert max(calls) > 1
