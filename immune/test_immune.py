"""
tests/test_immune.py — Integration + unit tests for immune-py.

Run with:
    pytest tests/ -v
"""

import gc
import queue
import threading
import time

import pytest

from immune.config import ImmuneConfig
from immune.decorators import protect
from immune.detector import Detector
from immune.memory import ImmuneMemory
from immune.models import AnomalyEvent, AnomalyType, PatchType
from immune.observer import Observer
from immune.patcher import Patcher, _circuit_breaker_wrapper, _rate_limiter_wrapper


# ──────────────────────────────────────────────────────────────────────────────
# Observer
# ──────────────────────────────────────────────────────────────────────────────

class TestObserver:

    def _make(self, **kwargs):
        cfg = ImmuneConfig(**kwargs)
        q = queue.Queue()
        return Observer(cfg, q), q

    def test_activate_deactivate(self):
        obs, _ = self._make()
        assert not obs.is_active
        obs.activate()
        assert obs.is_active
        obs.deactivate()
        assert not obs.is_active

    def test_records_calls(self):
        obs, q = self._make()
        obs.activate()

        def target():
            return 42

        target()
        obs.deactivate()

        # At least one record should be in the queue.
        assert not q.empty()
        record = q.get_nowait()
        assert record.func_name == "target"
        assert record.duration_ms >= 0

    def test_ignores_own_module(self):
        obs, q = self._make()
        obs.activate()

        # Drain queue, then check nothing from immune itself leaked in.
        time.sleep(0.05)
        obs.deactivate()

        while not q.empty():
            record = q.get_nowait()
            assert not record.module.startswith("immune"), (
                f"Observer should never record calls from the immune module: {record}"
            )

    def test_does_not_observe_ignored_functions(self):
        cfg = ImmuneConfig(ignored_functions=["ignored_fn"])
        q = queue.Queue()
        obs = Observer(cfg, q)
        obs.activate()

        def ignored_fn():
            return 1

        ignored_fn()
        obs.deactivate()
        names = [q.get_nowait().func_name for _ in range(q.qsize())]
        assert "ignored_fn" not in names


# ──────────────────────────────────────────────────────────────────────────────
# Detector
# ──────────────────────────────────────────────────────────────────────────────

class TestDetector:

    def _push_normal(self, q, func_name="my_func", n=110):
        """Push n normal-looking CallRecords into the queue."""
        from immune.models import CallRecord
        for _ in range(n):
            q.put(CallRecord(
                func_name=func_name,
                module="myapp",
                timestamp=time.perf_counter(),
                duration_ms=50.0 + (time.perf_counter() % 5),
                arg_size_bytes=100,
            ))

    def test_trains_baseline(self):
        cfg = ImmuneConfig(baseline_min_samples=50, baseline_timeout_seconds=999)
        q = queue.Queue()
        events = []
        det = Detector(cfg, q, on_anomaly=events.append)
        det.start()

        self._push_normal(q, n=60)
        time.sleep(0.5)  # give detector thread time to process

        det.stop()
        baseline = det.get_baseline("my_func")
        assert baseline is not None
        assert baseline.trained

    def test_detects_latency_spike(self):
        cfg = ImmuneConfig(
            baseline_min_samples=30,
            baseline_timeout_seconds=999,
            anomaly_threshold=0.60,
            confirmation_window=2,
        )
        q = queue.Queue()
        events = []
        det = Detector(cfg, q, on_anomaly=events.append)
        det.start()

        from immune.models import CallRecord

        # Push baseline
        for _ in range(35):
            q.put(CallRecord("fn", "m", time.perf_counter(), 40.0, 100))
        time.sleep(0.4)

        # Push spikes
        for _ in range(5):
            q.put(CallRecord("fn", "m", time.perf_counter(), 9000.0, 100))
        time.sleep(0.4)

        det.stop()
        assert len(events) >= 1
        assert events[0].func_name == "fn"


# ──────────────────────────────────────────────────────────────────────────────
# Patcher
# ──────────────────────────────────────────────────────────────────────────────

class TestPatcher:

    def test_rate_limiter(self):
        call_times = []

        def fn():
            call_times.append(time.perf_counter())

        wrapped = _rate_limiter_wrapper(fn, delay=0.05)
        for _ in range(3):
            wrapped()

        # Each call should be at least 50ms after the previous.
        gaps = [b - a for a, b in zip(call_times, call_times[1:])]
        assert all(g >= 0.04 for g in gaps), f"Gaps too small: {gaps}"

    def test_circuit_breaker_blocks_at_limit(self):
        def fn():
            return 1

        wrapped = _circuit_breaker_wrapper(fn, max_rps=3)
        # First 3 calls should succeed.
        for _ in range(3):
            wrapped()
        # 4th call within the same second should raise.
        with pytest.raises(RuntimeError, match="Circuit breaker"):
            wrapped()

    def test_rollback_restores_original(self):
        import sys
        import types

        # Create a fake module with a known function.
        mod = types.ModuleType("fake_app_module")
        original_calls = []

        def fake_func():
            original_calls.append(1)

        mod.fake_func = fake_func
        sys.modules["fake_app_module"] = mod

        cfg = ImmuneConfig(rate_limit_delay=0.0, patch_ttl_seconds=999)
        patcher = Patcher(cfg)

        event = AnomalyEvent(
            func_name="fake_func",
            anomaly_type=AnomalyType.RATE_SPIKE,
            score=0.9,
            description="test",
            raw_metrics={},
        )
        patch_event = patcher.apply(event)
        assert patch_event is not None
        assert "fake_func" in patcher.active_patches()

        patcher.rollback("fake_func")
        assert "fake_func" not in patcher.active_patches()

        # Original should be restored.
        mod.fake_func()
        assert len(original_calls) == 1

        patcher.stop()
        del sys.modules["fake_app_module"]


# ──────────────────────────────────────────────────────────────────────────────
# ImmuneMemory
# ──────────────────────────────────────────────────────────────────────────────

class TestImmuneMemory:

    def test_remember_and_recall(self):
        mem = ImmuneMemory(db_path=None)  # in-memory
        event = AnomalyEvent(
            func_name="pay", anomaly_type=AnomalyType.RATE_SPIKE,
            score=0.9, description="test", raw_metrics={},
        )
        assert mem.recall(event) is None

        mem.remember(event, PatchType.CIRCUIT_BREAKER)
        entry = mem.recall(event)
        assert entry is not None
        assert entry.func_name == "pay"
        assert entry.patch_type == PatchType.CIRCUIT_BREAKER
        assert entry.occurrences == 1

        mem.remember(event, PatchType.CIRCUIT_BREAKER)
        entry = mem.recall(event)
        assert entry.occurrences == 2

        mem.close()

    def test_forget(self):
        mem = ImmuneMemory(db_path=None)
        event = AnomalyEvent(
            func_name="fn", anomaly_type=AnomalyType.ERROR_BURST,
            score=0.8, description="x", raw_metrics={},
        )
        mem.remember(event, PatchType.FALLBACK)
        deleted = mem.forget("fn")
        assert deleted == 1
        assert mem.recall(event) is None
        mem.close()


# ──────────────────────────────────────────────────────────────────────────────
# @protect decorator
# ──────────────────────────────────────────────────────────────────────────────

class TestProtectDecorator:

    def test_basic_call_works(self):
        @protect
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_max_rps_blocks(self):
        @protect(max_rps=2, fallback=-1)
        def fast():
            return 1

        results = [fast() for _ in range(5)]
        # At least one call should have returned the fallback.
        assert -1 in results

    def test_max_arg_bytes_blocks(self):
        @protect(max_arg_bytes=10, fallback="blocked")
        def process(data):
            return data

        assert process("hi") == "hi"
        assert process("x" * 100) == "blocked"

    def test_immune_status(self):
        @protect
        def fn():
            pass

        # Trigger baseline collection.
        for _ in range(55):
            fn()

        status = fn.immune_status
        assert status["trained"] is True
        assert status["observations"] >= 55

    def test_preserves_function_name(self):
        @protect
        def my_special_function():
            pass

        assert my_special_function.__name__ == "my_special_function"


# ──────────────────────────────────────────────────────────────────────────────
# Full integration
# ──────────────────────────────────────────────────────────────────────────────

class TestImmuneSystem:

    def test_activate_deactivate(self):
        import immune
        system = immune.activate(ImmuneConfig(db_path=None))
        assert system.is_active
        s = immune.status()
        assert s["active"] is True
        immune.deactivate()
        assert not system.is_active

    def test_status_keys(self):
        import immune
        system = immune.activate(ImmuneConfig(db_path=None))
        s = immune.status()
        assert "observer" in s
        assert "detector" in s
        assert "patcher" in s
        assert "memory" in s
        immune.deactivate()
