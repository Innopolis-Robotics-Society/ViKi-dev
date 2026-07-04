"""viki.streamlit_app -- Streamlit frontend for the ViKi capture server.

A thin Python UI that talks to the existing FastAPI backend over HTTP
(``requests``) for control/config/calibration/skeleton actions, and embeds the
MJPEG camera streams and skeleton WebSocket canvas directly in the browser via
``st.components.v1.html``. See ``app.py`` for the entry point.
"""
