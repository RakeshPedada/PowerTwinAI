import os
import json
import streamlit as st
from streamlit_autorefresh import st_autorefresh


def render_progress_panel(log_file, status_file):
    """
    Display live reconstruction logs and monitor reconstruction status.

    Returns
    -------
    bool
        True if reconstruction has completed, otherwise False.
    """

    if not st.session_state.reconstruction_running:
        return False

    st_autorefresh(interval=7000, key="refresh")

    logs = []

    if os.path.exists(log_file):

        with open(
            log_file,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            logs = f.readlines()

    st.markdown(
        "<div class='section-title'>📜 Live Reconstruction Logs</div>",
        unsafe_allow_html=True
    )

    st.text_area(
        "Reconstruction Progress",
        "".join(logs[-25:]),
        height=250,
        disabled=True
    )

    if os.path.exists(status_file):

        with open(status_file, "r") as f:
            status_data = json.load(f)

        if status_data.get("status") == "COMPLETED":

            st.session_state.reconstruction_running = False
            st.session_state.reconstruction_done = True

            return True

    return False