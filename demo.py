"""
examples/demo.py — Shows all three ways to use immune-py.

Run with:
    pip install scikit-learn numpy
    python examples/demo.py
"""

import time
import random
import threading
import immune
from immune import protect, ImmuneConfig
from immune.models import AnomalyEvent, PatchEvent


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1 — Global activation (watches everything automatically)
# ─────────────────────────────────────────────────────────────────────────────

def example_global():
    print("\n── Example 1: Global activation ──")

    anomalies_seen = []

    def alert(event: AnomalyEvent):
        print(f"  🚨 Anomaly: {event.func_name} | {event.anomaly_type.value} | score={event.score:.2f}")
        anomalies_seen.append(event)

    def patched(event: PatchEvent):
        print(f"  🩹 Patch applied: {event.func_name} → {event.patch_type.value}")

    cfg = ImmuneConfig(
        baseline_min_samples=30,
        baseline_timeout_seconds=5,
        anomaly_threshold=0.70,
        confirmation_window=2,
        on_anomaly=alert,
        on_patch=patched,
        db_path=None,  # in-memory for this demo
    )

    system = immune.activate(cfg)
    print("  System active. Building baseline...")

    # Simulate a normal function running for a while.
    def normal_api_call(user_id: int) -> dict:
        time.sleep(random.uniform(0.01, 0.03))  # ~20ms, stable
        return {"user_id": user_id, "status": "ok"}

    # Build baseline (30 normal calls).
    for i in range(35):
        normal_api_call(i)
    print("  Baseline ready.")

    # Simulate a sudden latency spike (attack / bug).
    print("  Injecting latency spike...")
    for _ in range(5):
        time.sleep(2.0)  # 2000ms — way above baseline
        normal_api_call(999)

    print(f"  Anomalies detected: {len(anomalies_seen)}")
    print(f"  Status: {immune.status()['patcher']['active_patches']} active patches")

    immune.deactivate()


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 2 — @protect decorator (per-function, no global activation needed)
# ─────────────────────────────────────────────────────────────────────────────

def example_decorator():
    print("\n── Example 2: @protect decorator ──")

    @protect(max_rps=5, fallback={"error": "rate limited"})
    def process_payment(amount: float) -> dict:
        time.sleep(0.01)
        return {"status": "ok", "amount": amount}

    @protect(max_arg_bytes=50, fallback=None)
    def search_database(query: str) -> list:
        return [f"result for {query}"]

    # Normal usage — should work fine.
    print(f"  Payment: {process_payment(99.99)}")
    print(f"  Search: {search_database('hello')}")

    # Simulate DDoS — 10 rapid calls, only 5 allowed per second.
    results = [process_payment(1.0) for _ in range(10)]
    blocked = sum(1 for r in results if r.get("error"))
    print(f"  DDoS simulation: {blocked}/10 calls blocked by rate limiter")

    # Simulate SQL injection — huge payload.
    malicious = "A" * 1000
    result = search_database(malicious)
    print(f"  Injection attempt (1000 bytes): blocked={result is None}")

    print(f"  Payment status: {process_payment.immune_status}")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 3 — Flask middleware (requires flask installed)
# ─────────────────────────────────────────────────────────────────────────────

def example_flask():
    print("\n── Example 3: Flask middleware ──")
    try:
        from flask import Flask, jsonify
        from immune import ImmuneMiddleware
    except ImportError:
        print("  Flask not installed. Skipping. (pip install flask)")
        return

    app = Flask(__name__)
    app.wsgi_app = ImmuneMiddleware(app.wsgi_app)

    @app.route("/pay", methods=["GET"])
    def pay():
        return jsonify({"status": "ok"})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(immune.status())

    print("  Flask app with ImmuneMiddleware created.")
    print("  Routes: /pay, /health")
    print("  (Not starting server in demo mode — use app.run() in your code)")
    immune.deactivate()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("immune-py demo")
    print("=" * 40)
    example_decorator()   # No deps beyond immune itself
    example_flask()
    # example_global()    # Uncomment to run (takes ~1 minute)
    print("\n✓ Demo complete.")
