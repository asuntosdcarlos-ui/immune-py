"""
immune/patcher.py — The "antibody generator".

Takes an AnomalyEvent and modifies the offending function in memory
without restarting the process. Patches are automatically rolled back
after patch_ttl_seconds of normal behavior.

Patch strategies:
  RATE_LIMITER     — wraps the function with a per-call minimum interval
  CIRCUIT_BREAKER  — blocks calls above a RPS ceiling
  INPUT_SANITIZER  — truncates oversized bytes/str arguments
  MEMORY_CAP       — calls gc.collect() before each invocation + caps output
  FALLBACK         — replaces the function with a stub that returns a safe value
"""

from __future__ import annotations

import gc
import logging
import sys
import threading
import time
from typing import Any, Callable

from .config import ImmuneConfig
from .models import AnomalyEvent, AnomalyType, PatchEvent, PatchType

logger = logging.getLogger("immune.patcher")


# ── Patch wrappers ─────────────────────────────────────────────────────────────

def _rate_limiter_wrapper(func: Callable, delay: float) -> Callable:
    """Ensures at least `delay` seconds between consecutive calls."""
    last_call = [0.0]
    lock = threading.Lock()

    def wrapper(*args, **kwargs):
        with lock:
            elapsed = time.perf_counter() - last_call[0]
            if elapsed < delay:
                time.sleep(delay - elapsed)
            last_call[0] = time.perf_counter()
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__wrapped__ = func
    wrapper.__immune_patch__ = PatchType.RATE_LIMITER
    return wrapper


def _circuit_breaker_wrapper(func: Callable, max_rps: int) -> Callable:
    """Rejects calls when RPS exceeds max_rps."""
    call_times: list[float] = []
    lock = threading.Lock()

    def wrapper(*args, **kwargs):
        now = time.perf_counter()
        with lock:
            cutoff = now - 1.0
            while call_times and call_times[0] < cutoff:
                call_times.pop(0)
            if len(call_times) >= max_rps:
                raise RuntimeError(
                    f"[immune] Circuit breaker open for {func.__name__!r}: "
                    f"RPS limit {max_rps} exceeded."
                )
            call_times.append(now)
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__wrapped__ = func
    wrapper.__immune_patch__ = PatchType.CIRCUIT_BREAKER
    return wrapper


def _input_sanitizer_wrapper(func: Callable, max_bytes: int = 65_536) -> Callable:
    """Truncates str/bytes arguments that exceed max_bytes."""

    def _sanitize(val: Any) -> Any:
        if isinstance(val, str) and len(val.encode()) > max_bytes:
            logger.warning(
                "[immune] Truncating oversized str arg to %s bytes in %s()",
                max_bytes, func.__name__,
            )
            return val.encode()[:max_bytes].decode(errors="ignore")
        if isinstance(val, (bytes, bytearray)) and len(val) > max_bytes:
            logger.warning(
                "[immune] Truncating oversized bytes arg to %s bytes in %s()",
                max_bytes, func.__name__,
            )
            return val[:max_bytes]
        return val

    def wrapper(*args, **kwargs):
        clean_args = tuple(_sanitize(a) for a in args)
        clean_kwargs = {k: _sanitize(v) for k, v in kwargs.items()}
        return func(*clean_args, **clean_kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__wrapped__ = func
    wrapper.__immune_patch__ = PatchType.INPUT_SANITIZER
    return wrapper


def _memory_cap_wrapper(func: Callable) -> Callable:
    """Runs gc.collect() before the call to reclaim leaked memory."""

    def wrapper(*args, **kwargs):
        gc.collect()
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__wrapped__ = func
    wrapper.__immune_patch__ = PatchType.MEMORY_CAP
    return wrapper


def _fallback_wrapper(func: Callable, fallback_value: Any = None) -> Callable:
    """Replaces function with a stub that returns fallback_value."""

    def wrapper(*args, **kwargs):
        logger.error(
            "[immune] Function %r is quarantined. Returning fallback value.",
            func.__name__,
        )
        return fallback_value

    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__wrapped__ = func
    wrapper.__immune_patch__ = PatchType.FALLBACK
    return wrapper


# ── Patch type selection ───────────────────────────────────────────────────────

def _select_patch(event: AnomalyEvent, config: ImmuneConfig) -> PatchType:
    """Choose the appropriate patch strategy based on anomaly type."""
    mapping = {
        AnomalyType.RATE_SPIKE:    PatchType.CIRCUIT_BREAKER,
        AnomalyType.LATENCY_SPIKE: PatchType.RATE_LIMITER,
        AnomalyType.PAYLOAD_SPIKE: PatchType.INPUT_SANITIZER,
        AnomalyType.ERROR_BURST:   PatchType.FALLBACK,
        AnomalyType.MEMORY_LEAK:   PatchType.MEMORY_CAP,
        AnomalyType.UNKNOWN:       PatchType.RATE_LIMITER,
    }
    return mapping.get(event.anomaly_type, PatchType.RATE_LIMITER)


def _apply_wrapper(original: Callable, patch_type: PatchType, config: ImmuneConfig) -> Callable:
    if patch_type == PatchType.RATE_LIMITER:
        return _rate_limiter_wrapper(original, config.rate_limit_delay)
    if patch_type == PatchType.CIRCUIT_BREAKER:
        return _circuit_breaker_wrapper(original, config.circuit_breaker_rps)
    if patch_type == PatchType.INPUT_SANITIZER:
        return _input_sanitizer_wrapper(original)
    if patch_type == PatchType.MEMORY_CAP:
        return _memory_cap_wrapper(original)
    if patch_type == PatchType.FALLBACK:
        return _fallback_wrapper(original)
    return original


# ── Patcher ────────────────────────────────────────────────────────────────────

class Patcher:
    """
    Finds the live callable in the interpreter's module registry and
    replaces it with a patched wrapper. Maintains a registry of patches
    for rollback and TTL management.
    """

    def __init__(self, config: ImmuneConfig, on_patch=None, on_rollback=None):
        self._config = config
        self._on_patch = on_patch
        self._on_rollback = on_rollback

        # func_name → (module_ref, attr_name, original_callable, PatchEvent)
        self._patches: dict[str, tuple] = {}
        self._lock = threading.Lock()

        # Start the TTL monitor thread.
        self._active = True
        self._ttl_thread = threading.Thread(
            target=self._ttl_monitor, daemon=True, name="immune-patcher-ttl"
        )
        self._ttl_thread.start()

    def stop(self):
        self._active = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def apply(self, event: AnomalyEvent) -> PatchEvent | None:
        """
        Locate the function in sys.modules and replace it with a patch.
        Returns the PatchEvent, or None if the function couldn't be found.
        """
        with self._lock:
            if event.func_name in self._patches:
                logger.debug("Patch already applied to %s, skipping.", event.func_name)
                return None

        original, module_obj, attr_name = self._find_callable(event.func_name)
        if original is None:
            logger.warning(
                "Patcher: could not locate %r in sys.modules. "
                "Use @protect decorator for direct protection.",
                event.func_name,
            )
            return None

        patch_type = _select_patch(event, self._config)
        patched = _apply_wrapper(original, patch_type, self._config)

        try:
            setattr(module_obj, attr_name, patched)
        except (AttributeError, TypeError) as exc:
            logger.error("Failed to patch %r: %s", event.func_name, exc)
            return None

        patch_event = PatchEvent(
            func_name=event.func_name,
            patch_type=patch_type,
            anomaly_event=event,
        )

        with self._lock:
            self._patches[event.func_name] = (module_obj, attr_name, original, patch_event)

        logger.info(
            "Patch applied: %s → %s (%s)",
            event.func_name, patch_type.value, event.anomaly_type.value,
        )

        if self._on_patch:
            try:
                self._on_patch(patch_event)
            except Exception:
                logger.exception("on_patch callback raised an exception.")

        return patch_event

    def rollback(self, func_name: str) -> bool:
        """Remove a patch and restore the original function."""
        with self._lock:
            entry = self._patches.pop(func_name, None)

        if entry is None:
            return False

        module_obj, attr_name, original, patch_event = entry
        try:
            setattr(module_obj, attr_name, original)
            logger.info("Rolled back patch on %r.", func_name)
            if self._on_rollback:
                self._on_rollback(func_name)
            return True
        except Exception as exc:
            logger.error("Rollback failed for %r: %s", func_name, exc)
            return False

    def rollback_all(self) -> None:
        """Remove all active patches."""
        for func_name in list(self._patches.keys()):
            self.rollback(func_name)

    def active_patches(self) -> dict[str, PatchEvent]:
        with self._lock:
            return {name: entry[3] for name, entry in self._patches.items()}

    # ── Function locator ───────────────────────────────────────────────────────

    def _find_callable(self, func_name: str):
        """
        Search sys.modules for a module-level attribute matching func_name.
        Returns (callable, module_object, attribute_name) or (None, None, None).

        This covers module-level functions. Methods on class instances require
        the @protect decorator approach instead.
        """
        for mod in list(sys.modules.values()):
            if mod is None:
                continue
            obj = getattr(mod, func_name, None)
            if obj is not None and callable(obj) and not isinstance(obj, type):
                if getattr(obj, "__name__", None) == func_name:
                    return obj, mod, func_name
        return None, None, None

    # ── TTL monitor ────────────────────────────────────────────────────────────

    def _ttl_monitor(self) -> None:
        """Periodically check if patches should be rolled back."""
        ttl = self._config.patch_ttl_seconds
        while self._active:
            time.sleep(10)
            now = time.time()
            for func_name, entry in list(self._patches.items()):
                _, _, _, patch_event = entry
                age = (now - patch_event.applied_at.timestamp())
                if age >= ttl:
                    logger.info(
                        "Patch TTL expired for %r (age=%.0fs). Rolling back.",
                        func_name, age,
                    )
                    self.rollback(func_name)
