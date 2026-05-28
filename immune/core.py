"""
immune/core.py — The ImmuneSystem orchestrator.

Wires together Observer → Detector → Patcher → Memory and exposes
a clean status API. This is the object returned by immune.activate().
"""

from __future__ import annotations

import logging
import queue
from datetime import datetime
from typing import Optional

from .config import ImmuneConfig
from .detector import Detector
from .memory import ImmuneMemory
from .models import AnomalyEvent, PatchEvent
from .observer import Observer
from .patcher import Patcher

logger = logging.getLogger("immune")


class ImmuneSystem:
    """
    Top-level facade. Owns all subsystems and manages their lifecycle.

    Typical call sequence:
        system = ImmuneSystem(config)
        system.activate()
        # ... app runs normally ...
        system.deactivate()
    """

    def __init__(self, config: ImmuneConfig):
        self._config = config
        self._active = False

        # Configure logging once.
        logging.basicConfig(level=config.log_level, format=config.log_format)

        # Shared queue between Observer (producer) and Detector (consumer).
        # maxsize prevents unbounded memory growth under extreme load.
        self._record_queue: queue.Queue = queue.Queue(maxsize=10_000)

        # Subsystems.
        self._observer = Observer(config, self._record_queue)
        self._detector = Detector(config, self._record_queue, self._on_anomaly)
        self._patcher  = Patcher(
            config,
            on_patch=self._on_patch,
            on_rollback=self._on_rollback,
        )
        self._memory   = ImmuneMemory(config.db_path)

        # Counters for the status report.
        self._anomaly_count = 0
        self._patch_count   = 0
        self._started_at: Optional[datetime] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def activate(self) -> None:
        """Start all subsystems."""
        if self._active:
            logger.warning("ImmuneSystem is already active.")
            return
        self._started_at = datetime.utcnow()
        self._detector.start()
        self._observer.activate()
        self._active = True
        logger.info("ImmuneSystem active. Collecting baseline...")

    def deactivate(self) -> None:
        """Stop all subsystems and roll back any active patches."""
        if not self._active:
            return
        self._observer.deactivate()
        self._detector.stop()
        self._patcher.stop()
        self._patcher.rollback_all()
        self._memory.close()
        self._active = False
        logger.info("ImmuneSystem deactivated. All patches rolled back.")

    @property
    def is_active(self) -> bool:
        return self._active

    # ── Event handlers ─────────────────────────────────────────────────────────

    def _on_anomaly(self, event: AnomalyEvent) -> None:
        """Called by Detector when an anomaly is confirmed."""
        self._anomaly_count += 1

        # Check immune memory first — fast path for known attacks.
        memory_entry = self._memory.recall(event)
        if memory_entry:
            logger.info(
                "Known attack pattern recalled from memory: %s → %s "
                "(seen %d times before, immediate response).",
                event.func_name, memory_entry.patch_type.value, memory_entry.occurrences,
            )

        # Fire user callback if set.
        if self._config.on_anomaly:
            try:
                self._config.on_anomaly(event)
            except Exception:
                logger.exception("on_anomaly callback raised an exception.")

        # Apply patch.
        patch_event = self._patcher.apply(event)
        if patch_event and memory_entry:
            self._memory.remember(event, patch_event.patch_type)

    def _on_patch(self, event: PatchEvent) -> None:
        """Called by Patcher when a patch is successfully applied."""
        self._patch_count += 1
        # Persist to immune memory.
        self._memory.remember(event.anomaly_event, event.patch_type)

        if self._config.on_patch:
            try:
                self._config.on_patch(event)
            except Exception:
                logger.exception("on_patch callback raised an exception.")

    def _on_rollback(self, func_name: str) -> None:
        """Called by Patcher when a patch is rolled back."""
        if self._config.on_rollback:
            try:
                self._config.on_rollback(func_name)
            except Exception:
                logger.exception("on_rollback callback raised an exception.")

    # ── Status ─────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """
        Return a serializable snapshot of the current system state.
        Useful for health endpoints or dashboards.
        """
        baselines = self._detector.all_baselines()
        trained = sum(1 for b in baselines.values() if b.trained)
        patches = self._patcher.active_patches()

        return {
            "active": self._active,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "observer": {
                "active": self._observer.is_active,
                "queue_size": self._record_queue.qsize(),
            },
            "detector": {
                "functions_monitored": len(baselines),
                "functions_trained": trained,
                "anomalies_detected": self._anomaly_count,
            },
            "patcher": {
                "active_patches": len(patches),
                "total_patches_applied": self._patch_count,
                "patches": {
                    name: {
                        "type": pe.patch_type.value,
                        "applied_at": pe.applied_at.isoformat(),
                        "anomaly_type": pe.anomaly_event.anomaly_type.value,
                        "score": round(pe.anomaly_event.score, 3),
                    }
                    for name, pe in patches.items()
                },
            },
            "memory": {
                "entries": len(self._memory.all_entries()),
            },
        }

    # ── Direct access ──────────────────────────────────────────────────────────

    @property
    def observer(self) -> Observer:
        return self._observer

    @property
    def detector(self) -> Detector:
        return self._detector

    @property
    def patcher(self) -> Patcher:
        return self._patcher

    @property
    def memory(self) -> ImmuneMemory:
        return self._memory
