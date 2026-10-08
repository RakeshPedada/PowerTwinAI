"""
Session state manager for PowerTwinAI.
"""

import streamlit as st


def initialize_session():
    """
    Initialize all required Streamlit session variables.
    """

    import os
    import json
    from config.paths import RESULT_FILE, STATUS_FILE

    is_running = False
    already_done = False

    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                s = json.load(f).get("status")
                if s == "RUNNING":
                    is_running = True
                elif s == "COMPLETED" and os.path.exists(RESULT_FILE):
                    already_done = True
        except Exception:
            pass

    # Check if runner PID or background process is actively executing
    runner_pid_file = os.path.join(os.path.dirname(STATUS_FILE), "runner.pid")
    if not is_running and not already_done:
        if os.path.exists(runner_pid_file):
            try:
                with open(runner_pid_file, "r") as pf:
                    r_pid = int(pf.read().strip())
                if os.name == "nt":
                    import subprocess
                    chk = subprocess.run(["tasklist", "/FI", f"PID eq {r_pid}"], capture_output=True, text=True)
                    if str(r_pid) in chk.stdout:
                        is_running = True
            except Exception:
                pass

        if not is_running and os.name == "nt":
            try:
                import subprocess
                out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq colmap.exe"], capture_output=True, text=True)
                if "colmap.exe" in out.stdout:
                    is_running = True
            except Exception:
                pass

    # Ensure st.session_state reflects the live background state
    st.session_state["reconstruction_running"] = is_running
    if already_done and not is_running:
        st.session_state["reconstruction_done"] = True
    elif "reconstruction_done" not in st.session_state:
        st.session_state["reconstruction_done"] = False