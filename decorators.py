"""
immune/decorators.py — The @protect decorator.

Provides direct, reliable protection for a specific function without
relying on sys.modules lookup. Wraps the function at definition time,
so it works on methods, closures, and nested functions too.

Usage:
    from immune import protect

    @protect
    def process_payment(amount):
        ...

    @protect(threshold=0.90, patch=PatchType.CIRCUIT_BREAKER)
    def get_user(user_id):
        ...
"""

from __future__ import annotations

import functools
import logging
import queue
import time
import threading
from typing import Callable, Optional

from .config import ImmuneConfig
from .models import AnomalyType, PatchType, CallRecord

logger = logging.getLogger("immune.decorators")


def protect(
    func: Optional[Callable] = None,
    *,
    threshold: float = 0.80,
    patch: Optional[PatchType] = None,
    max_rps: Optional[int] = None,
    max_latency_ms: Optional[float] = None,
    max_arg_bytes: Optional[int] = None,
    fallback=None,
):
    """
    Decorator that attaches a self-contained mini-immune-system to a
    single function. Works standalone (no need to call immune.activate()).

    Args:
        threshold:       Anomaly score above which a patch is applied.
        patch:           Force a specific PatchType instead of auto-selecting.
        max_rps:         Hard RPS cap; overrides auto-detection.
        max_latency_ms:  Hard latency cap in ms; overrides auto-detection.
        max_arg_bytes:   Hard input size cap; overrides auto-detection.
        fallback:        Value to return when function is quarantined.

    Can be used with or without arguments:
        @protect
        @protect()
        @protect(threshold=0.9, max_rps=100)
    """
    def decorator(fn: Callable) -> Callable:
        return _ProtectedFunction(
            fn,
            threshold=threshold,
            forced_patch=patch,
            max_rps=max_rps,
            max_latency_ms=max_latency_ms,
            max_arg_bytes=max_arg_bytes,
            fallback=fallback,
        )

    if func is not None:
        # Called as @protect (no parentheses).
        return decorator(func)

    # Called as @protect(...).
    return decorator


class _ProtectedFunction:
    """
    Wraps a callable with per-call monitoring, anomaly scoring,
    and automatic patching — all self-contained in this object.
    """

    BASELINE_SAMPLES = 50  # observations before model activates

    def __init__(
        self,
        func: Callable,
        threshold: float,
        forced_patch: Optional[PatchType],
        max_rps: Optional[int],
        max_latency_ms: Optional[float],
        max_arg_bytes: Optional[int],
        fallback,
    ):
        self._func = func
        self._threshold = threshold
        self._forced_patch = forced_patch
        self._max_rps = max_rps
        self._max_latency_ms = max_latency_ms
        self._max_arg_bytes = max_arg_bytes
        self._fallback = fallback

        # Copy function metadata so it looks identical to the original.
        functools.update_wrapper(self, func)

        # Observations.
        self._durations: list[float] = []
        self._arg_sizes: list[int]   = []
        self._errors: list[bool]     = []
        self._call_times: list[float] = []
        self._lock = threading.Lock()

        # Derived baseline stats (set after training).
        self._mean_dur = 0.0
        self._std_dur  = 1.0
        self._mean_size = 0.0
        self._std_size  = 1.0
        self._trained  = False

        # Anomaly confirmation buffer.
        self._score_buffer: list[float] = []
        self._CONFIRM = 3  # consecutive high scores needed

        # Patch state.
        self._patched = False
        self._patch_type: Optional[PatchType] = None
        self._patch_applied_at: Optional[float] = None
        self._rate_limiter_last: float = 0.0

    def __call__(self, *args, **kwargs):
        # ── Hard limit checks (instant, no ML needed) ──────────────────────────
        now = time.perf_counter()

        if self._max_rps is not None:
            with self._lock:
                self._call_times.append(now)
                cutoff = now - 1.0
                while self._call_times and self._call_times[0] < cutoff:
                    self._call_times.pop(0)
                rps = len(self._call_times)
            if rps > self._max_rps:
                logger.warning("[protect] %r: RPS %d > max %d. Blocking call.", self.__name__, rps, self._max_rps)
                if self._fallback is not None:
                    return self._fallback
                raise RuntimeError(f"[immune] {self.__name__!r} rate limit exceeded.")

        if self._max_arg_bytes is not None:
            total = sum(len(str(a).encode()) for a in args)
            if total > self._max_arg_bytes:
                logger.warning("[protect] %r: arg size %d > max %d bytes. Rejecting.", self.__name__, total, self._max_arg_bytes)
                if self._fallback is not None:
                    return self._fallback
                raise ValueError(f"[immune] {self.__name__!r} input too large.")

        # ── Patch enforcement ──────────────────────────────────────────────────
        if self._patched:
            if self._patch_type == PatchType.FALLBACK:
                logger.error("[protect] %r quarantined. Returning fallback.", self.__name__)
                return self._fallback

            if self._patch_type == PatchType.RATE_LIMITER:
                with self._lock:
                    elapsed = now - self._rate_limiter_last
                    if elapsed < 0.05:
                        time.sleep(0.05 - elapsed)
                    self._rate_limiter_last = time.perf_counter()

        # ── Execute ────────────────────────────────────────────────────────────
        start = time.perf_counter()
        raised = False
        try:
            result = self._func(*args, **kwargs)
        except Exception:
            raised = True
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000

            # Hard latency check.
            if self._max_latency_ms and duration_ms > self._max_latency_ms:
                logger.warning("[protect] %r: latency %.0fms > max %.0fms.", self.__name__, duration_ms, self._max_latency_ms)

            arg_size = sum(len(str(a).encode()) for a in args)

            self._observe(duration_ms, arg_size, raised)

        return result

    def _observe(self, duration_ms: float, arg_size: int, raised: bool) -> None:
        """Record an observation and check for anomalies."""
        with self._lock:
            self._durations.append(duration_ms)
            self._arg_sizes.append(arg_size)
            self._errors.append(raised)

            if not self._trained and len(self._durations) >= self.BASELINE_SAMPLES:
                self._train()
                return

            if self._trained:
                score = self._score(duration_ms, arg_size, raised)
                self._score_buffer.append(score)
                if len(self._score_buffer) > self._CONFIRM:
                    self._score_buffer.pop(0)

                if (
                    len(self._score_buffer) == self._CONFIRM
                    and all(s >= self._threshold for s in self._score_buffer)
                    and not self._patched
                ):
                    self._activate_patch()
                    self._score_buffer.clear()

    def _train(self) -> None:
        import statistics
        self._mean_dur  = statistics.mean(self._durations)
        self._std_dur   = statistics.stdev(self._durations) or 1.0
        self._mean_size = statistics.mean(self._arg_sizes)
        self._std_size  = statistics.stdev(self._arg_sizes) or 1.0
        self._trained   = True
        logger.info(
            "[protect] %r baseline ready: dur=%.1f±%.1fms, size=%.0f±%.0fB",
            self.__name__, self._mean_dur, self._std_dur, self._mean_size, self._std_size,
        )

    def _score(self, duration_ms: float, arg_size: int, raised: bool) -> float:
        dur_z  = abs(duration_ms - self._mean_dur)  / self._std_dur
        size_z = abs(arg_size    - self._mean_size) / self._std_size
        err_z  = 5.0 if raised else 0.0
        raw = max(dur_z, size_z, err_z)
        return min(raw / 8.0, 1.0)

    def _activate_patch(self) -> None:
        patch = self._forced_patch

        if patch is None:
            # Auto-select based on recent symptoms.
            recent_errors = sum(self._errors[-20:])
            recent_sizes  = self._arg_sizes[-20:]
            recent_durs   = self._durations[-20:]

            if recent_errors > 10:
                patch = PatchType.FALLBACK
            elif max(recent_sizes, default=0) > self._mean_size * 5:
                patch = PatchType.INPUT_SANITIZER
            elif max(recent_durs, default=0) > self._mean_dur * 5:
                patch = PatchType.RATE_LIMITER
            else:
                patch = PatchType.RATE_LIMITER

        self._patched = True
        self._patch_type = patch
        self._patch_applied_at = time.perf_counter()
        logger.warning(
            "[protect] Anomaly confirmed in %r. Applying %s.",
            self.__name__, patch.value,
        )

    @property
    def immune_status(self) -> dict:
        """Introspect the protection state of this function."""
        return {
            "function": self.__name__,
            "trained": self._trained,
            "patched": self._patched,
            "patch_type": self._patch_type.value if self._patch_type else None,
            "observations": len(self._durations),
            "mean_duration_ms": round(self._mean_dur, 2),
            "mean_arg_size_bytes": round(self._mean_size, 2),
        }
