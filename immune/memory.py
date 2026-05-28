"""
immune/memory.py — Persistent immune memory.

Stores signatures of past attacks and the patches that resolved them,
so future occurrences of the same anomaly trigger an immediate response
instead of going through the full detection cycle again.

Backed by SQLite so the memory survives process restarts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime
from typing import Optional

from .models import AnomalyEvent, AnomalyType, ImmuneMemoryEntry, PatchType

logger = logging.getLogger("immune.memory")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS immune_memory (
    signature    TEXT PRIMARY KEY,
    func_name    TEXT NOT NULL,
    anomaly_type TEXT NOT NULL,
    patch_type   TEXT NOT NULL,
    occurrences  INTEGER DEFAULT 1,
    last_seen    TEXT NOT NULL,
    avg_score    REAL DEFAULT 0.0
);
"""


def _signature(func_name: str, anomaly_type: AnomalyType) -> str:
    """Stable hash that identifies a (function, anomaly_type) pair."""
    raw = f"{func_name}:{anomaly_type.value}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ImmuneMemory:
    """
    Read/write interface to the immune memory store.

    If db_path is None, uses an in-memory SQLite database (lost on exit).
    """

    def __init__(self, db_path: Optional[str] = "immune_memory.db"):
        self._db_path = db_path or ":memory:"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        logger.info("Immune memory initialized (db=%s).", self._db_path)

    # ── Public API ─────────────────────────────────────────────────────────────

    def recall(self, event: AnomalyEvent) -> Optional[ImmuneMemoryEntry]:
        """
        Look up whether we've seen this (function, anomaly_type) before.
        Returns an ImmuneMemoryEntry if found, or None.
        """
        sig = _signature(event.func_name, event.anomaly_type)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM immune_memory WHERE signature = ?", (sig,)
            ).fetchone()

        if row is None:
            return None

        return ImmuneMemoryEntry(
            signature=row["signature"],
            func_name=row["func_name"],
            anomaly_type=AnomalyType(row["anomaly_type"]),
            patch_type=PatchType(row["patch_type"]),
            occurrences=row["occurrences"],
            last_seen=datetime.fromisoformat(row["last_seen"]),
            avg_score=row["avg_score"],
        )

    def remember(self, event: AnomalyEvent, patch_type: PatchType) -> None:
        """
        Store or update a memory entry after a successful patch.
        If the entry already exists, increment its occurrence counter.
        """
        sig = _signature(event.func_name, event.anomaly_type)
        now = datetime.utcnow().isoformat()

        with self._lock:
            existing = self._conn.execute(
                "SELECT occurrences, avg_score FROM immune_memory WHERE signature = ?",
                (sig,),
            ).fetchone()

            if existing:
                n = existing["occurrences"] + 1
                new_avg = (existing["avg_score"] * existing["occurrences"] + event.score) / n
                self._conn.execute(
                    """
                    UPDATE immune_memory
                    SET occurrences = ?, last_seen = ?, avg_score = ?
                    WHERE signature = ?
                    """,
                    (n, now, new_avg, sig),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO immune_memory
                        (signature, func_name, anomaly_type, patch_type, occurrences, last_seen, avg_score)
                    VALUES (?, ?, ?, ?, 1, ?, ?)
                    """,
                    (
                        sig,
                        event.func_name,
                        event.anomaly_type.value,
                        patch_type.value,
                        now,
                        event.score,
                    ),
                )
            self._conn.commit()

        logger.debug(
            "Memory updated: %s → %s (sig=%s)",
            event.func_name, patch_type.value, sig,
        )

    def all_entries(self) -> list[ImmuneMemoryEntry]:
        """Return all stored memory entries."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM immune_memory").fetchall()

        return [
            ImmuneMemoryEntry(
                signature=r["signature"],
                func_name=r["func_name"],
                anomaly_type=AnomalyType(r["anomaly_type"]),
                patch_type=PatchType(r["patch_type"]),
                occurrences=r["occurrences"],
                last_seen=datetime.fromisoformat(r["last_seen"]),
                avg_score=r["avg_score"],
            )
            for r in rows
        ]

    def forget(self, func_name: str) -> int:
        """Remove all memory entries for a given function. Returns rows deleted."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM immune_memory WHERE func_name = ?", (func_name,)
            )
            self._conn.commit()
        return cursor.rowcount

    def clear(self) -> None:
        """Wipe all immune memory."""
        with self._lock:
            self._conn.execute("DELETE FROM immune_memory")
            self._conn.commit()
        logger.warning("Immune memory cleared.")

    def close(self) -> None:
        self._conn.close()
