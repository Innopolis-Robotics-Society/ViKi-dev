"""viki.streamlit_app.settings
------------------------------
Environment-driven backend URLs and helpers.

Two base URLs are kept separate even though they default the same, because
compose uses host networking (so both resolve to ``localhost:8000`` from the
Streamlit container AND the user's browser), but they may diverge in other
deployments:

- ``BACKEND_URL``          -- used by server-side ``requests`` (``api.py``).
- ``BROWSER_BACKEND_URL``  -- used by embedded ``<img>`` MJPEG streams and the
                              skeleton WebSocket, i.e. what the *browser* hits.
"""

from __future__ import annotations

import os

BACKEND_URL: str = os.environ.get("VIKI_BACKEND_URL", "http://localhost:8000").rstrip("/")
BROWSER_BACKEND_URL: str = os.environ.get(
    "VIKI_BROWSER_BACKEND_URL", "http://localhost:8000"
).rstrip("/")

# Request timeout (seconds) for all server-side calls.
REQUEST_TIMEOUT: float = float(os.environ.get("VIKI_REQUEST_TIMEOUT", "8"))


def browser_url(path: str) -> str:
    """Absolute HTTP(S) URL the browser loads directly (e.g. an MJPEG stream)."""
    return f"{BROWSER_BACKEND_URL}{path}"


def ws_url(path: str) -> str:
    """Derive a ``ws://``/``wss://`` URL from :data:`BROWSER_BACKEND_URL`."""
    base = BROWSER_BACKEND_URL
    if base.startswith("https://"):
        return "wss://" + base[len("https://"):] + path
    if base.startswith("http://"):
        return "ws://" + base[len("http://"):] + path
    # No scheme -> assume ws.
    return "ws://" + base + path
