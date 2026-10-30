import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .._utilities._config import get_dgcv_settings_registry

_state = threading.local()
_pool = None
_pool_size = 0
_pool_lock = threading.Lock()
_gil_warned = [False]


def _mark_worker():
    _state.worker = True


def thread_count():
    n = get_dgcv_settings_registry().get("parallel_threads")
    try:
        n = int(n) if n is not None else 0
    except (TypeError, ValueError):
        return 0
    return n if n > 1 else 0


def _site_enabled(site):
    sites = get_dgcv_settings_registry().get("_parallel_sites")
    return sites is None or site in sites


def _active(site, count, min_items):
    if getattr(_state, "worker", False):
        return 0
    n = thread_count()
    if not n or count < min_items:
        return 0
    if not _site_enabled(site):
        return 0
    from ._engine import engine_kind

    if engine_kind() != "builtin":
        return 0
    if not _gil_warned[0]:
        _gil_warned[0] = True
        if getattr(sys, "_is_gil_enabled", lambda: True)():
            from .._utilities._config import dgcv_warning

            dgcv_warning(
                "parallel_threads is set, but this Python build keeps the global interpreter lock, so threads will not run at the same time.",
                stacklevel=3,
            )
    return n


def _get_pool(n):
    global _pool, _pool_size
    with _pool_lock:
        if _pool is None or _pool_size != n:
            if _pool is not None:
                _pool.shutdown(wait=False)
            _pool = ThreadPoolExecutor(max_workers=n, initializer=_mark_worker)
            _pool_size = n
        return _pool


def drop_pool():
    global _pool, _pool_size
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False)
        _pool = None
        _pool_size = 0


def _chunks(items, n, chunk):
    if chunk is None:
        chunk = max(1, -(-len(items) // (8 * n)))
    return [items[i : i + chunk] for i in range(0, len(items), chunk)]


def _run_block(fn, block):
    return [fn(x) for x in block]


def _cooperative(fn, items, chunk):
    pool = _pool
    n = _pool_size
    if pool is None or not n:
        return [fn(x) for x in items]
    blocks = _chunks(items, n, chunk)
    futures = [pool.submit(_run_block, fn, block) for block in blocks]
    parts = [None] * len(blocks)
    for k in range(len(blocks) - 1, -1, -1):
        if futures[k].cancel():
            parts[k] = _run_block(fn, blocks[k])
    out = []
    for k, fut in enumerate(futures):
        out.extend(parts[k] if parts[k] is not None else fut.result())
    return out


def parallel_map(site, fn, items, min_items=8, chunk=None, on_start=None, on_result=None):
    items = list(items)
    if (
        getattr(_state, "worker", False)
        and on_start is None
        and on_result is None
        and len(items) >= min_items
        and _site_enabled(site)
    ):
        return _cooperative(fn, items, chunk)
    n = _active(site, len(items), min_items)
    if not n:
        out = []
        for x in items:
            if on_start is not None:
                on_start(x)
            r = fn(x)
            if on_result is not None:
                on_result(x, r)
            out.append(r)
        return out
    pool = _get_pool(n)
    blocks = _chunks(items, n, chunk)
    if on_start is not None:
        for x in items:
            on_start(x)
    if on_result is None:
        out = []
        for part in pool.map(lambda block: [fn(x) for x in block], blocks):
            out.extend(part)
        return out
    futures = {pool.submit(lambda block: [fn(x) for x in block], block): k for k, block in enumerate(blocks)}
    parts = [None] * len(blocks)
    for fut in as_completed(futures):
        k = futures[fut]
        parts[k] = fut.result()
        for x, r in zip(blocks[k], parts[k]):
            on_result(x, r)
    out = []
    for part in parts:
        out.extend(part)
    return out


def parallel_map_until(site, fn, items, stop, min_items=8, chunk=None):
    items = list(items)
    n = _active(site, len(items), min_items)
    if not n:
        out = []
        for x in items:
            r = fn(x)
            out.append(r)
            if stop(r):
                break
        return out
    pool = _get_pool(n)
    out = []
    start = 0
    size = max(1, min_items // n) if chunk is None else chunk
    while start < len(items):
        blocks = [items[i : i + size] for i in range(start, min(len(items), start + n * size), size)]
        for part in pool.map(lambda block: [fn(x) for x in block], blocks):
            for r in part:
                out.append(r)
                if stop(r):
                    return out
        start += n * size
        if chunk is None:
            size = min(2 * size, max(1, -(-len(items) // (8 * n))))
    return out
