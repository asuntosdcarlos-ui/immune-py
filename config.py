"""
immune/config.py — Configuration for the immune system.
All thresholds and tunables live here so users can customize behavior
without touching internals.
"""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ImmuneConfig:
    # ── Baseline ───────────────────────────────────────────────────────────────
    # How many observations to collect before the model is considered trained.
    baseline_min_samples: int = 100

    # How long to wait (seconds) before forcing baseline completion even if
    # we haven't reached min_samples. Useful for low-traffic apps.
    baseline_timeout_seconds: float = 30.0

    # ── Detection ──────────────────────────────────────────────────────────────
    # Isolation Forest contamination parameter: expected fraction of anomalies
    # in normal traffic. 0.05 = expect 5% of observations to be anomalous.
    contamination: float = 0.05

    # Score above which a function is considered anomalous (0.0 – 1.0).
    anomaly_threshold: float = 0.75

    # How many consecutive anomalous observations before triggering a response.
    # Prevents false positives from a single spike.
    confirmation_window: int = 3

    # ── Observer ───────────────────────────────────────────────────────────────
    # Maximum calls per second the observer will record. Sampling above this
    # keeps overhead low on high-traffic endpoints.
    max_sample_rate: int = 1000

    # Fraction of calls to sample when above max_sample_rate. 0.1 = 10%.
    sample_fraction: float = 0.1

    # ── Patcher ────────────────────────────────────────────────────────────────
    # Rate limit injected into anomalous functions (seconds between calls).
    rate_limit_delay: float = 0.05

    # Circuit breaker: max allowed calls per second before blocking.
    circuit_breaker_rps: int = 50

    # After this many seconds of normal behavior, remove the patch.
    patch_ttl_seconds: float = 60.0

    # ── Functions to ignore ────────────────────────────────────────────────────
    # The observer ignores calls to these function names (internal + stdlib).
    ignored_functions: list[str] = field(default_factory=lambda: [
        "trace_calls", "<module>", "<listcomp>", "<dictcomp>",
        "<setcomp>", "<genexpr>", "__repr__", "__str__",
    ])

    # Ignore all functions from these module prefixes.
    ignored_modules: list[str] = field(default_factory=lambda: [
        "immune",       # never watch ourselves
        "threading",
        "asyncio",
        "logging",
        "importlib",
        "_pytest",
        "pytest",
    ])

    # ── Callbacks ──────────────────────────────────────────────────────────────
    # Called when an anomaly is detected. Receives the AnomalyEvent.
    on_anomaly: Callable | None = None

    # Called when a patch is applied. Receives the PatchEvent.
    on_patch: Callable | None = None

    # Called when a patch is rolled back. Receives the function name.
    on_rollback: Callable | None = None

    # ── Storage ────────────────────────────────────────────────────────────────
    # Path to SQLite database for immune memory. None = in-memory only.
    db_path: str | None = "immune_memory.db"

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "[immune] %(levelname)s %(message)s"
