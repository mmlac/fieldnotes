"""Daemon lifecycle management for fieldnotes.

Delegates to platform-specific backends in :mod:`worker.service`.

.. deprecated::
    The ``fieldnotes daemon`` CLI is kept for backwards compatibility.
    Prefer ``fieldnotes service``.
"""

from __future__ import annotations

# Re-export the public API from the service package so existing callers
# (including ``fieldnotes daemon …``) continue to work unchanged.
from worker.service import install, platform_backend, start, status, stop, uninstall

__all__ = [
    "install",
    "uninstall",
    "status",
    "start",
    "stop",
    "platform_backend",
]
