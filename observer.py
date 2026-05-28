"""
immune/observer.py — The "nervous system" of the immune library.

Hooks into CPython's sys.settrace() to intercept every function call
in the monitored application, measure its properties, and forward
observations to the detector via a non-blocking queue.

Design goals:
  - Zero-copy: we never copy argument values, only measure their size.
  - Low overhead: sampling kicks in above max_sample_rate.
  - Thread-safe: all state is either thread-local or protected by a lock.
  - Non-intrusive: restores the original trace function on deactivation.
"""

from __future__ import annotations

import sys
import time
import queue
import logging
import threading
import random
from typing import Callable

from .config import ImmuneConfig
from .models import CallRecord

logger = logging.getLogger("immune.observer")


def _sizeof(obj) -> int:
    """
    Fast, approximate size in bytes of a Python object.
    We deliberately avoid sys.getsizeof recursion for performance —
    this is a heuristic, not an exact measurement.
    """
    try:
        if isinstance(obj, (str, bytes, bytearray)):
            return len(obj)
        if isinstance(obj, (list, tuple, set, frozenset)):
            return sum(_sizeof(x) for x in obj) + 64
        if isinstance(obj, dict):
            return sum(_sizeof(k) + _sizeof(v) for k, v in obj.items()) + 64
        return sys.getsizeof(obj)
    except Exception:
        return 0


class Observer:
    """
    Installs a sys.settrace hook that records every function call
    and pushes CallRecord objects onto an output queue.

    The queue is consumed by the Detector on a separate thread so the
    trace function itself is as fast as possible.
    """

    def __init__(self, config: ImmuneConfig, output_queue: queue.Queue):
        self._config = config
        self._queue = output_queue
        self._active = False
        self._lock = threading.Lock()

        # Per-function call timestamps for RPS tracking (last N seconds).
        self._call_times: dict[str, list[float]] = {}
        self._call_times_lock = threading.Lock()

        # Track in-progress calls so we can measure duration.
        # Key: (thread_id, frame_id) → (func_name, module, start_time, arg_size)
        self._in_flight: dict[tuple, tuple] = {}
        self._in_flight_lock = threading.Lock()

        # Previous trace function so we can restore it on deactivation.
        self._previous_trace: Callable | None = None

        # Sampling counter per function.
        self._sample_counters: dict[str, int] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def activate(self) -> None:
        """Install the trace hook."""
        with self._lock:
            if self._active:
                return
            self._previous_trace = sys.gettrace()
            sys.settrace(self._trace)
            threading.settrace(self._trace)
            self._active = True
            logger.info("Observer activated (sys.settrace installed).")

    def deactivate(self) -> None:
        """Remove the trace hook and restore the previous one."""
        with self._lock:
            if not self._active:
                return
            sys.settrace(self._previous_trace)
            threading.settrace(self._previous_trace)
            self._active = False
            logger.info("Observer deactivated.")

    @property
    def is_active(self) -> bool:
        return self._active

    # ── Trace implementation ───────────────────────────────────────────────────

    def _trace(self, frame, event: str, arg):
        """
        Called by CPython for every 'call', 'return', and 'exception' event.
        Must be as fast as possible — it runs on the application's thread.
        """
        if event == "call":
            return self._on_call(frame)
        if event in ("return", "exception"):
            self._on_return(frame, event, arg)
        return self._trace

    def _on_call(self, frame):
        func_name = frame.f_code.co_name
        module    = frame.f_globals.get("__name__", "")

        # Fast-path: ignore internal / stdlib functions.
        if self._should_ignore(func_name, module):
            return None  # returning None stops tracing for this scope

        # Sampling above max_sample_rate.
        if not self._should_sample(func_name):
            return None

        # Measure argument size without copying values.
        arg_size = 0
        try:
            locals_ = frame.f_locals
            for val in locals_.values():
                arg_size += _sizeof(val)
        except Exception:
            pass

        key = (threading.get_ident(), id(frame))
        with self._in_flight_lock:
            self._in_flight[key] = (func_name, module, time.perf_counter(), arg_size)

        return self._trace

    def _on_return(self, frame, event: str, arg):
        key = (threading.get_ident(), id(frame))
        with self._in_flight_lock:
            entry = self._in_flight.pop(key, None)

        if entry is None:
            return

        func_name, module, start, arg_size = entry
        duration_ms = (time.perf_counter() - start) * 1000
        raised = (event == "exception")
        exc_type = ""
        if raised and isinstance(arg, tuple) and len(arg) >= 1:
            exc_type = getattr(arg[0], "__name__", "")

        record = CallRecord(
            func_name=func_name,
            module=module,
            timestamp=time.perf_counter(),
            duration_ms=duration_ms,
            arg_size_bytes=arg_size,
            raised_exception=raised,
            exception_type=exc_type,
        )

        try:
            self._queue.put_nowait(record)
        except queue.Full:
            pass  # drop record rather than slowing the app

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _should_ignore(self, func_name: str, module: str) -> bool:
        cfg = self._config
        if func_name in cfg.ignored_functions:
            return True
        for prefix in cfg.ignored_modules:
            if module == prefix or module.startswith(prefix + "."):
                return True
        return False

    def _should_sample(self, func_name: str) -> bool:
        """
        Track calls-per-second per function. If above max_sample_rate,
        sample only a fraction of calls to keep overhead low.
        """
        now = time.perf_counter()
        cfg = self._config

        with self._call_times_lock:
            times = self._call_times.setdefault(func_name, [])
            times.append(now)
            # Keep only last 1 second.
            cutoff = now - 1.0
            # Trim from the front.
            while times and times[0] < cutoff:
                times.pop(0)
            rps = len(times)

        if rps > cfg.max_sample_rate:
            counter = self._sample_counters.get(func_name, 0) + 1
            self._sample_counters[func_name] = counter
            return random.random() < cfg.sample_fraction

        return True
