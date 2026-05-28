# immune-py

**Self-healing immune system for Python applications.**

immune-py monitors your app in production, learns what "normal" looks like,
detects anomalies in real time, and patches affected functions **in memory —
without restarting the server**.

```
Your app → Observer → Detector → Patcher → back to Your app
              ↓           ↓          ↓
           metrics    anomaly    patch in
           queue      score      memory
```

---

## Install

```bash
pip install immune-py
```

---

## Usage

### Option 1 — One import (watches everything)

```python
import immune
immune.activate()

# Your existing app runs below, completely unchanged.
from flask import Flask
app = Flask(__name__)
```

### Option 2 — Decorator (protect specific functions)

```python
from immune import protect

@protect
def process_payment(amount):
    ...

# With explicit limits:
@protect(max_rps=100, max_arg_bytes=65536, fallback={"error": "blocked"})
def get_user_data(query):
    ...
```

### Option 3 — Middleware (Flask / Django / FastAPI)

```python
# Flask / Django (WSGI)
from immune import ImmuneMiddleware
app.wsgi_app = ImmuneMiddleware(app.wsgi_app)

# FastAPI / Starlette (ASGI)
from immune.middleware import ImmuneASGIMiddleware
app.add_middleware(ImmuneASGIMiddleware)
```

---

## What it detects & fixes automatically

| Anomaly | What triggers it | Patch applied |
|---|---|---|
| **Rate spike** (DDoS) | RPS far above baseline | Circuit breaker |
| **Latency spike** | Execution time far above baseline | Rate limiter |
| **Payload spike** (injection) | Argument size far above baseline | Input sanitizer |
| **Error burst** | Repeated exceptions from one function | Fallback stub |
| **Memory leak** | Monotonic growth in payload size | GC + memory cap |

---

## Configuration

```python
from immune import ImmuneConfig, activate

cfg = ImmuneConfig(
    # Baseline
    baseline_min_samples=100,      # observations before model trains
    baseline_timeout_seconds=30.0, # train anyway after this many seconds

    # Detection
    contamination=0.05,            # expected anomaly fraction
    anomaly_threshold=0.75,        # score [0-1] above which = anomaly
    confirmation_window=3,         # consecutive hits before triggering

    # Response
    rate_limit_delay=0.05,         # seconds between calls when throttled
    circuit_breaker_rps=50,        # max RPS before circuit opens
    patch_ttl_seconds=60.0,        # roll back patch after this many seconds

    # Callbacks
    on_anomaly=my_alert_fn,        # called when anomaly confirmed
    on_patch=my_pagerduty_fn,      # called when patch applied
    on_rollback=my_log_fn,         # called when patch rolled back

    # Storage
    db_path="immune_memory.db",    # SQLite path; None = in-memory only
)

system = activate(cfg)
```

---

## Status / health endpoint

```python
import immune

status = immune.status()
# {
#   "active": True,
#   "detector": {"functions_monitored": 12, "anomalies_detected": 2},
#   "patcher":  {"active_patches": 1, "patches": {"process_payment": {...}}},
#   "memory":   {"entries": 3},
# }
```

---

## How it works

```
sys.settrace()          Intercepts every Python function call.
                        Overhead: ~5-15% CPU; reduced via sampling above
                        max_sample_rate.

IsolationForest         Learns normal behavior per function (RPS, latency,
                        payload size, error rate). Scores new observations.
                        Falls back to Z-score if scikit-learn is absent.

Patcher                 Wraps the anomalous function with the appropriate
                        strategy. Uses setattr() on the live module object —
                        no restart needed. Auto-rollback after patch_ttl.

ImmuneMemory            SQLite store of past (attack → patch) pairs.
                        Known attack signatures trigger immediate response
                        on the next occurrence.
```

---

## Run tests

```bash
pip install pytest scikit-learn numpy
pytest tests/ -v
```

---

## License

**Business Source License 1.1 (BSL-1.1)**

Free for personal use, small teams (<10 employees), and non-commercial projects.
Commercial license required for production use in revenue-generating organizations.
Automatically becomes Apache 2.0 on 2029-01-01.

See [LICENSE](LICENSE) for full terms or contact for commercial licensing.
