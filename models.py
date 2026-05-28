"""
immune/models.py — Shared data structures used across all modules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AnomalyType(str, Enum):
    RATE_SPIKE      = "rate_spike"       # Calls-per-second far above baseline
    LATENCY_SPIKE   = "latency_spike"    # Execution time far above baseline
    PAYLOAD_SPIKE   = "payload_spike"    # Argument size far above baseline
    ERROR_BURST     = "error_burst"      # Repeated exceptions from one function
    MEMORY_LEAK     = "memory_leak"      # Monotonic growth in payload/object size
    UNKNOWN         = "unknown"


class PatchType(str, Enum):
    RATE_LIMITER     = "rate_limiter"    # Inject a sleep() into the function
    CIRCUIT_BREAKER  = "circuit_breaker" # Block calls above RPS threshold
    INPUT_SANITIZER  = "input_sanitizer" # Truncate / reject oversized inputs
    MEMORY_CAP       = "memory_cap"      # Force GC + cap result object size
    FALLBACK         = "fallback"        # Replace function with a safe stub


@dataclass
class CallRecord:
    """One observation of a single function call."""
    func_name:       str
    module:          str
    timestamp:       float          # time.perf_counter() at call entry
    duration_ms:     float          # wall-clock ms for the call
    arg_size_bytes:  int            # rough size of *args + **kwargs
    raised_exception: bool = False
    exception_type:  str  = ""


@dataclass
class FunctionBaseline:
    """Learned normal behavior for one function."""
    func_name:          str
    sample_count:       int   = 0
    mean_duration_ms:   float = 0.0
    std_duration_ms:    float = 0.0
    mean_arg_size:      float = 0.0
    std_arg_size:       float = 0.0
    mean_rps:           float = 0.0
    std_rps:            float = 0.0
    error_rate:         float = 0.0   # fraction 0–1
    trained:            bool  = False


@dataclass
class AnomalyEvent:
    """Emitted when the detector confirms an anomaly."""
    func_name:      str
    anomaly_type:   AnomalyType
    score:          float           # 0.0 – 1.0
    description:    str
    raw_metrics:    dict[str, Any]
    timestamp:      datetime = field(default_factory=datetime.utcnow)


@dataclass
class PatchEvent:
    """Emitted when a patch is applied to a function."""
    func_name:      str
    patch_type:     PatchType
    anomaly_event:  AnomalyEvent
    applied_at:     datetime = field(default_factory=datetime.utcnow)
    rolled_back_at: datetime | None = None
    effective:      bool = False    # set to True after TTL check passes


@dataclass
class ImmuneMemoryEntry:
    """Persisted record of a past attack + successful response."""
    signature:      str             # hash of (func_name, anomaly_type)
    func_name:      str
    anomaly_type:   AnomalyType
    patch_type:     PatchType
    occurrences:    int   = 1
    last_seen:      datetime = field(default_factory=datetime.utcnow)
    avg_score:      float = 0.0
