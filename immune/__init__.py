"""
immune-py — Self-healing immune system for Python applications.

Usage:
    # Option 1: protect the whole app
    import immune
    immune.activate()

    # Option 2: protect specific functions
    from immune import protect

    @protect
    def my_function():
        ...

    # Option 3: WSGI/ASGI middleware
    from immune import ImmuneMiddleware
    app.wsgi_app = ImmuneMiddleware(app.wsgi_app)
"""

from .core import ImmuneSystem
from .decorators import protect
from .middleware import ImmuneMiddleware
from .config import ImmuneConfig

__version__ = "0.1.0"
__author__ = "immune-py contributors"

# Global singleton
_system: ImmuneSystem | None = None


def activate(config: ImmuneConfig | None = None) -> ImmuneSystem:
    """
    Activate the immune system globally.
    Hooks into sys.settrace() and starts monitoring all function calls.

    Args:
        config: Optional ImmuneConfig to customize behavior.

    Returns:
        The active ImmuneSystem instance.
    """
    global _system
    if _system is not None and _system.is_active:
        return _system
    _system = ImmuneSystem(config or ImmuneConfig())
    _system.activate()
    return _system


def deactivate() -> None:
    """Deactivate the immune system and restore original function states."""
    global _system
    if _system is not None:
        _system.deactivate()
        _system = None


def get_system() -> ImmuneSystem | None:
    """Return the active ImmuneSystem instance, or None if not activated."""
    return _system


def status() -> dict:
    """Return a snapshot of the current immune system status."""
    if _system is None:
        return {"active": False}
    return _system.status()


__all__ = [
    "activate",
    "deactivate",
    "get_system",
    "status",
    "protect",
    "ImmuneMiddleware",
    "ImmuneConfig",
    "ImmuneSystem",
]
