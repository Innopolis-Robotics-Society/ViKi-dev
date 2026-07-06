"""Recordings tab -- browse recorded RGB-D sessions and download files.

Sessions are written by :class:`viki.capture.recorder.RGBDRecorder` under
``data/videos/rec_<YYYYmmdd_HHMMSS>/`` (see the module docstring in
``recorder.py`` for the on-disk layout). This tab lists those sessions, shows the
per-session file tree grouped by camera, and offers a direct browser download
link for each file.

Download mechanism mirrors the MJPEG ``<img>`` pattern: recordings (especially
the ``.mp4`` videos) can be large, so bytes are *never* pulled through Python /
Streamlit. Instead each row renders a :func:`st.link_button` pointing at
``browser_url("/api/recordings/{session}/download?path=...")`` so the user's
browser downloads straight from FastAPI (a ``FileResponse`` with
``Content-Disposition: attachment``).

The three backend endpoints this tab targets are DEFINED (but not yet
implemented) in ``docs/recordings_api_contract.md``. Until the backend ships
them, :meth:`ViKiApi.list_recordings` raises :class:`ViKiApiError` (HTTP 404);
:func:`render` catches that and shows a friendly ``st.info`` pointing at the
contract doc rather than crashing the page.
"""

from __future__ import annotations

from urllib.parse import quote

import streamlit as st

from ..api import ViKiApiError
from ..settings import browser_url
from ..state import get_api

_CONTRACT_DOC = "docs/recordings_api_contract.md"

# Human labels for the ``kind`` field of the file-tree contract.
_KIND_LABELS: dict[str, str] = {
    "color": "Colour video",
    "depth_viz": "Depth (colorised) video",
    "depth_raw": "Raw depth (.npy)",
    "intrinsics": "Intrinsics",
    "meta": "Metadata",
}


def _fmt_size(num_bytes: int | None) -> str:
    """Render a byte count as a compact human-readable string."""
    if not num_bytes:
        return "—"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def _download_url(session: str, rel_path: str) -> str:
    """Direct browser URL for downloading one file from a session.

    The ``path`` query parameter is URL-encoded; the backend must reject any
    path that escapes the session directory (see the contract doc).
    """
    return browser_url(
        f"/api/recordings/{quote(session)}/download?path={quote(rel_path, safe='')}"
    )


@st.fragment(run_every="10s")
def _session_list() -> None:
    """Poll ``/api/recordings`` and render the session picker + file tree.

    Isolated in a fragment so refreshing the list does not rerun the whole page.
    """
    api = get_api()
    try:
        sessions = api.list_recordings()
    except ViKiApiError as exc:
        if exc.status == 404:
            st.info(
                "The **Recordings** backend is not implemented yet. The frontend "
                "is already wired to the API contract in "
                f"`{_CONTRACT_DOC}` — sessions and downloads will appear here once "
                "the backend serves those endpoints."
            )
        else:
            st.error(f"Could not load recordings: {exc}")
        return

    if not sessions:
        st.info("No recordings yet. Use **🔴 Record RGB-D** above to capture a session.")
        return

    # Newest first (fall back to name, which is timestamp-sortable).
    sessions = sorted(
        sessions,
        key=lambda s: (s.get("created_at") or s.get("name") or ""),
        reverse=True,
    )

    labels = {}
    for s in sessions:
        name = s.get("name", "?")
        created = s.get("created_at", "")
        size = _fmt_size(s.get("size_bytes"))
        cams = ", ".join(s.get("cameras", [])) or "no cameras"
        labels[name] = f"{name}  ·  {created}  ·  {size}  ·  {cams}"

    names = [s.get("name", "?") for s in sessions]
    selected = st.selectbox(
        "Session",
        names,
        format_func=lambda n: labels.get(n, n),
        key="recordings_selected_session",
    )
    if selected:
        _file_tree(selected)


def _file_tree(session: str) -> None:
    """Fetch and render the per-session file tree, grouped by camera."""
    api = get_api()
    try:
        tree = api.get_recording(session)
    except ViKiApiError as exc:
        if exc.status == 404:
            st.warning(f"Session `{session}` is no longer available.")
        else:
            st.error(f"Could not load session `{session}`: {exc}")
        return

    files = tree.get("files", [])
    if not files:
        st.caption("This session contains no files.")
        return

    # Group by device_id; None (top-level metadata) goes into a "Session" bucket.
    groups: dict[str | None, list[dict]] = {}
    for f in files:
        groups.setdefault(f.get("device_id"), []).append(f)

    # Cameras first (sorted), then session-level metadata last.
    device_ids = sorted(d for d in groups if d is not None)
    ordered = [*device_ids, None] if None in groups else device_ids

    for dev in ordered:
        header = dev if dev is not None else "Session-level files"
        st.markdown(f"**{header}**")
        with st.container(border=True):
            for f in sorted(groups[dev], key=lambda x: x.get("path", "")):
                rel_path = f.get("path", "")
                kind = _KIND_LABELS.get(f.get("kind", ""), f.get("kind", "file"))
                size = _fmt_size(f.get("size_bytes"))
                cols = st.columns([4, 2, 2])
                with cols[0]:
                    st.markdown(f"`{rel_path}`")
                with cols[1]:
                    st.caption(f"{kind} · {size}")
                with cols[2]:
                    st.link_button(
                        "⬇ Download",
                        _download_url(session, rel_path),
                        use_container_width=True,
                    )


def render() -> None:
    st.subheader("Recordings")
    st.caption(
        "Browse recorded RGB-D sessions and download individual files. Downloads "
        "stream directly from the backend to your browser."
    )
    _session_list()
