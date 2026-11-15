import os
import json
import streamlit as st
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


def render_progress_panel(log_file, status_file):
    """
    Display live reconstruction logs and monitor reconstruction status.

    Returns
    -------
    bool
        True if reconstruction has completed, otherwise False.
    """

    # Check active status dynamically
    is_active = st.session_state.get("reconstruction_running", False)
    if not is_active and os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                if json.load(f).get("status") == "RUNNING":
                    is_active = True
                    st.session_state.reconstruction_running = True
        except Exception:
            pass

    if not is_active:
        return False

    if st_autorefresh is not None:
        st_autorefresh(interval=3500, key="refresh_recon_progress")

    status_data = {}
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                status_data = json.load(f)
        except Exception:
            pass

    current_step = status_data.get("step", "COLMAP SIFT extraction, multi-view feature matching & ML surface meshing in progress...")
    current_progress = float(status_data.get("progress", 0.15))

    st.progress(min(1.0, max(0.0, current_progress)), text=f"⚡ {current_step}")

    # Structured Live Execution Telemetry Cards
    stage_name = status_data.get("stage")
    pairs_total = status_data.get("pairs_total")
    elapsed_val = status_data.get("elapsed")
    last_op = status_data.get("last_op")

    if stage_name or pairs_total or elapsed_val or last_op:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Current Stage", stage_name or "Reconstruction")
        with m2:
            if pairs_total:
                done = status_data.get("pairs_done", 0)
                st.metric("Pairs Processed", f"{done} / {pairs_total}")
            else:
                st.metric("Pipeline Status", status_data.get("status", "RUNNING"))
        with m3:
            if elapsed_val is not None:
                mins, secs = divmod(int(elapsed_val), 60)
                st.metric("Elapsed Time", f"{mins:02d}m {secs:02d}s")
            else:
                st.metric("Execution Mode", "Active")
        with m4:
            st.metric("Last Operation", (last_op[:25] + "...") if last_op and len(last_op) > 25 else (last_op or "In progress"))

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
        "".join(logs[-30:]),
        height=260,
        disabled=True
    )

    if os.path.exists(status_file):

        with open(status_file, "r") as f:
            status_data = json.load(f)

        status = status_data.get("status")
        if status == "COMPLETED":

            st.session_state.reconstruction_running = False
            st.session_state.reconstruction_done = True

            return True

        elif status == "FAILED":

            st.session_state.reconstruction_running = False
            st.session_state.reconstruction_done = False

            st.error("❌ Reconstruction pipeline encountered an error. Please inspect the logs above.")
            return False

    return False