"""
immune/middleware.py — WSGI and ASGI middleware adapters.

Drop-in integration for Flask, Django, FastAPI, Starlette, and any
other framework that follows the WSGI or ASGI protocol.

WSGI (Flask, Django):
    from immune import ImmuneMiddleware
    app.wsgi_app = ImmuneMiddleware(app.wsgi_app)

ASGI (FastAPI, Starlette):
    from immune.middleware import ImmuneASGIMiddleware
    app.add_middleware(ImmuneASGIMiddleware)
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterable

import immune as _immune

logger = logging.getLogger("immune.middleware")


class ImmuneMiddleware:
    """
    WSGI middleware that activates the immune system on first request
    and records per-request metrics (latency, status, content-length)
    as additional signals for the anomaly detector.
    """

    def __init__(self, app, config=None):
        self._app = app
        self._system = _immune.activate(config)
        logger.info("ImmuneMiddleware installed (WSGI).")

    def __call__(self, environ: dict, start_response: Callable) -> Iterable[bytes]:
        start = time.perf_counter()
        status_holder: list[str] = []

        def _start_response(status, headers, exc_info=None):
            status_holder.append(status)
            return start_response(status, headers, exc_info)

        try:
            result = self._app(environ, _start_response)
        except Exception:
            logger.warning(
                "[middleware] Unhandled exception on %s %s",
                environ.get("REQUEST_METHOD", "?"),
                environ.get("PATH_INFO", "?"),
            )
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            code = int(status_holder[0].split()[0]) if status_holder else 0
            logger.debug(
                "[middleware] %s %s → %s (%.0fms)",
                environ.get("REQUEST_METHOD", "?"),
                environ.get("PATH_INFO", "?"),
                code,
                elapsed_ms,
            )

        return result


class ImmuneASGIMiddleware:
    """
    ASGI middleware for FastAPI / Starlette.

    Usage:
        app.add_middleware(ImmuneASGIMiddleware)
    """

    def __init__(self, app, config=None):
        self._app = app
        self._system = _immune.activate(config)
        logger.info("ImmuneMiddleware installed (ASGI).")

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return

        start = time.perf_counter()
        status_holder: list[int] = []

        async def _send(message):
            if message["type"] == "http.response.start":
                status_holder.append(message.get("status", 0))
            await send(message)

        try:
            await self._app(scope, receive, _send)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            code = status_holder[0] if status_holder else 0
            logger.debug(
                "[middleware] %s %s → %s (%.0fms)",
                scope.get("method", "?"),
                scope.get("path", "?"),
                code,
                elapsed_ms,
            )
