"""
immune/detector.py — The "immune cell" that learns what's normal and
flags what isn't.

Uses scikit-learn's Isolation Forest to build a baseline of normal
behavior per function, then scores new observations against it.

Runs on a dedicated daemon thread, consuming CallRecord objects from
the Observer's output queue.
"""

from __future__ import annotations

import math
import queue
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import ImmuneConfig
from .models import (
    AnomalyEvent, AnomalyType, CallRecord, FunctionBaseline,
)

logger = logging.getLogger("immune.detector")

try:
    from sklearn.ensemble import IsolationForest
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn not installed. Falling back to statistical Z-score detection."
    )


# ── Feature extraction ─────────────────────────────────────────────────────────

def _extract_features(record: CallRecord, baseline: FunctionBaseline) -> list[float]:
    """
    Convert a CallRecord into a numeric feature vector for the model.
    Features are z-scored against the baseline so the model is scale-invariant.
    """
    def zscore(val, mean, std) -> float:
        if std < 1e-9:
            return 0.0
        return (val - mean) / std

    return [
        zscore(record.duration_ms,   baseline.mean_duration_ms, baseline.std_duration_ms),
        zscore(record.arg_size_bytes, baseline.mean_arg_size,    baseline.std_arg_size),
        float(record.raised_exception),
    ]


# ── Per-function state ─────────────────────────────────────────────────────────

@dataclass
class _FunctionState:
    name: str
    # Raw observations during baseline phase.
    raw_samples: list[list[float]] = field(default_factory=list)
    baseline: FunctionBaseline = field(default_factory=lambda: FunctionBaseline(""))
    model: object = None  # IsolationForest once trained

    # Sliding window of recent records for RPS + error rate tracking.
    recent_records: deque = field(default_factory=lambda: deque(maxlen=200))

    # Confirmation buffer: N consecutive anomaly scores above threshold.
    anomaly_buffer: deque = field(default_factory=lambda: deque(maxlen=10))

    # How many times we've confirmed an anomaly for this function.
    confirmed_anomalies: int = 0

    def __post_init__(self):
        self.baseline.func_name = self.name


# ── Detector ───────────────────────────────────────────────────────────────────

class Detector:
    """
    Consumes CallRecord objects from a queue, builds baselines, then
    scores new observations and emits AnomalyEvent objects when anomalies
    are confirmed.
    """

    def __init__(
        self,
        config: ImmuneConfig,
        input_queue: queue.Queue,
        on_anomaly: Callable[[AnomalyEvent], None],
    ):
        self._config = config
        self._queue = input_queue
        self._on_anomaly = on_anomaly
        self._states: dict[str, _FunctionState] = {}
        self._states_lock = threading.Lock()
        self._active = False
        self._thread: threading.Thread | None = None

        # Track global baseline start time.
        self._baseline_start = time.perf_counter()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the detector thread."""
        self._active = True
        self._baseline_start = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="immune-detector"
        )
        self._thread.start()
        logger.info("Detector started.")

    def stop(self) -> None:
        self._active = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Detector stopped.")

    def get_baseline(self, func_name: str) -> FunctionBaseline | None:
        with self._states_lock:
            state = self._states.get(func_name)
            return state.baseline if state else None

    def all_baselines(self) -> dict[str, FunctionBaseline]:
        with self._states_lock:
            return {name: s.baseline for name, s in self._states.items()}

    # ── Main loop ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        while self._active:
            try:
                record: CallRecord = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._process(record)
            except Exception:
                logger.exception("Error processing record for %s", record.func_name)

    def _process(self, record: CallRecord) -> None:
        with self._states_lock:
            state = self._states.setdefault(
                record.func_name,
                _FunctionState(name=record.func_name),
            )

        state.recent_records.append(record)
        self._update_baseline_stats(state, record)

        if not state.baseline.trained:
            self._try_train(state)
            return  # Don't score until trained.

        score = self._score(state, record)
        state.anomaly_buffer.append(score)

        # Require confirmation_window consecutive high scores.
        window = self._config.confirmation_window
        recent_scores = list(state.anomaly_buffer)[-window:]
        if (
            len(recent_scores) == window
            and all(s >= self._config.anomaly_threshold for s in recent_scores)
        ):
            state.confirmed_anomalies += 1
            event = self._build_event(state, record, score)
            logger.warning(
                "Anomaly confirmed in %s (score=%.2f, type=%s)",
                record.func_name, score, event.anomaly_type.value,
            )
            # Clear buffer so we don't spam the same anomaly.
            state.anomaly_buffer.clear()
            self._on_anomaly(event)

    # ── Baseline building ──────────────────────────────────────────────────────

    def _update_baseline_stats(self, state: _FunctionState, record: CallRecord) -> None:
        """Online update of mean / std using Welford's algorithm."""
        b = state.baseline
        b.sample_count += 1
        n = b.sample_count

        # Welford online update for duration.
        delta = record.duration_ms - b.mean_duration_ms
        b.mean_duration_ms += delta / n
        delta2 = record.duration_ms - b.mean_duration_ms
        if n > 1:
            b.std_duration_ms = math.sqrt(
                ((n - 2) * b.std_duration_ms ** 2 + delta * delta2) / (n - 1)
            )

        # Welford online update for arg size.
        delta = record.arg_size_bytes - b.mean_arg_size
        b.mean_arg_size += delta / n
        delta2 = record.arg_size_bytes - b.mean_arg_size
        if n > 1:
            b.std_arg_size = math.sqrt(
                ((n - 2) * b.std_arg_size ** 2 + delta * delta2) / (n - 1)
            )

        # Error rate (exponential moving average).
        err = float(record.raised_exception)
        b.error_rate = 0.95 * b.error_rate + 0.05 * err

    def _try_train(self, state: _FunctionState) -> None:
        """Train the Isolation Forest once enough samples are collected."""
        cfg = self._config
        elapsed = time.perf_counter() - self._baseline_start

        enough_samples = state.baseline.sample_count >= cfg.baseline_min_samples
        timed_out = elapsed >= cfg.baseline_timeout_seconds and state.baseline.sample_count >= 10

        if not (enough_samples or timed_out):
            return

        b = state.baseline
        # Build feature matrix from raw stats (we use z-scored running stats).
        # We synthesise Gaussian samples around the learned mean/std to train IF.
        rng = np.random.default_rng(42)
        n = max(state.baseline.sample_count, 50)
        X = np.column_stack([
            rng.normal(0.0, 1.0, n),  # z-scored duration (mean=0, std=1 by definition)
            rng.normal(0.0, 1.0, n),  # z-scored arg size
            rng.binomial(1, b.error_rate, n).astype(float),
        ])

        if _SKLEARN_AVAILABLE:
            model = IsolationForest(
                contamination=cfg.contamination,
                n_estimators=100,
                random_state=42,
            )
            model.fit(X)
            state.model = model
        else:
            state.model = None  # Fall back to Z-score

        b.trained = True
        logger.info(
            "Baseline trained for %s (%d samples, duration=%.1f±%.1fms)",
            state.name, b.sample_count, b.mean_duration_ms, b.std_duration_ms,
        )

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _score(self, state: _FunctionState, record: CallRecord) -> float:
        """
        Return an anomaly score in [0, 1].
        1.0 = maximally anomalous, 0.0 = perfectly normal.
        """
        features = _extract_features(record, state.baseline)

        if _SKLEARN_AVAILABLE and state.model is not None:
            X = np.array([features])
            # IsolationForest.score_samples returns negative anomaly scores;
            # more negative = more anomalous. We normalize to [0, 1].
            raw = state.model.score_samples(X)[0]
            # Typical range is roughly -0.5 (normal) to -1.0 (anomalous).
            score = float(np.clip((-raw - 0.3) / 0.7, 0.0, 1.0))
        else:
            # Simple Z-score fallback: max absolute z-score across features.
            score = float(np.clip(max(abs(f) for f in features) / 5.0, 0.0, 1.0))

        return score

    # ── Event building ─────────────────────────────────────────────────────────

    def _build_event(
        self, state: _FunctionState, record: CallRecord, score: float
    ) -> AnomalyEvent:
        """Classify the type of anomaly from the metrics."""
        b = state.baseline

        # Compute how far this observation deviates.
        lat_z  = (record.duration_ms   - b.mean_duration_ms) / max(b.std_duration_ms, 1e-9)
        size_z = (record.arg_size_bytes - b.mean_arg_size)    / max(b.std_arg_size, 1e-9)

        # Recent error rate from the last 20 records.
        recent = list(state.recent_records)[-20:]
        recent_err_rate = sum(r.raised_exception for r in recent) / max(len(recent), 1)

        # Recent RPS from last 1 second of timestamps.
        now = time.perf_counter()
        recent_rps = sum(1 for r in recent if now - r.timestamp < 1.0)

        rps_z = (recent_rps - b.mean_rps) / max(b.std_rps, 1e-9) if b.mean_rps > 0 else 0

        if recent_rps > self._config.circuit_breaker_rps and rps_z > 3:
            anomaly_type = AnomalyType.RATE_SPIKE
            desc = f"RPS={recent_rps} (baseline={b.mean_rps:.0f})"
        elif recent_err_rate > 0.3:
            anomaly_type = AnomalyType.ERROR_BURST
            desc = f"Error rate={recent_err_rate:.0%} (recent {len(recent)} calls)"
        elif size_z > 4:
            # Check for monotonic growth (memory leak pattern).
            sizes = [r.arg_size_bytes for r in recent]
            if len(sizes) > 5 and all(a <= b_ for a, b_ in zip(sizes, sizes[1:])):
                anomaly_type = AnomalyType.MEMORY_LEAK
                desc = f"Monotonic payload growth detected (last={record.arg_size_bytes}B)"
            else:
                anomaly_type = AnomalyType.PAYLOAD_SPIKE
                desc = f"arg_size={record.arg_size_bytes}B (baseline={b.mean_arg_size:.0f}B)"
        elif lat_z > 4:
            anomaly_type = AnomalyType.LATENCY_SPIKE
            desc = f"duration={record.duration_ms:.0f}ms (baseline={b.mean_duration_ms:.0f}ms)"
        else:
            anomaly_type = AnomalyType.UNKNOWN
            desc = f"score={score:.2f}"

        return AnomalyEvent(
            func_name=record.func_name,
            anomaly_type=anomaly_type,
            score=score,
            description=desc,
            raw_metrics={
                "duration_ms": record.duration_ms,
                "arg_size_bytes": record.arg_size_bytes,
                "raised_exception": record.raised_exception,
                "recent_rps": recent_rps,
                "recent_error_rate": recent_err_rate,
                "lat_zscore": lat_z,
                "size_zscore": size_z,
            },
        )
