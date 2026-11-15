"""
core/session_manager.py
-----------------------
Robust session state manager for PowerTwinAI enterprise workstation.
Initializes and maintains reactive navigation, persistent dataset selections,
background job tracking, and active visualizer states.
"""

import os
import json
import streamlit as st
from config.paths import RESULT_FILE, STATUS_FILE, TEMP_DIR


def initialize_session():
    """
    Initialize all required Streamlit session variables with reactive background tracking.
    """
    is_running = False
    already_done = False

    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                s = json.load(f).get("status")
                if s == "RUNNING":
                    is_running = True
                elif s == "COMPLETED":
                    already_done = True
        except Exception:
            pass

    # Check if runner PID or background process is actively executing
    runner_pid_file = os.path.join(os.path.dirname(STATUS_FILE), "runner.pid")
    if not is_running and not already_done:
        if os.path.exists(runner_pid_file):
            try:
                with open(runner_pid_file, "r", encoding="utf-8") as pf:
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

    # Navigation state
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Overview"

    # Active reconstruction backend choice
    if "active_backend" not in st.session_state:
        st.session_state["active_backend"] = "colmap"

    # Dataset persistence
    from core.dataset_manager import get_shared_dataset
    get_shared_dataset()

    if "dataset_mode" not in st.session_state:
        st.session_state["dataset_mode"] = "folder"
    if "dataset_folder" not in st.session_state:
        default_folder = os.path.abspath(os.path.join(os.path.dirname(TEMP_DIR), "colmap_workspace", "images"))
        if not os.path.exists(default_folder):
            default_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(TEMP_DIR)), "colmap_workspace", "images"))
        st.session_state["dataset_folder"] = default_folder if os.path.exists(default_folder) else ""
    if "dataset_files" not in st.session_state:
        st.session_state["dataset_files"] = []

    # Viewer state
    if "viewer_backend" not in st.session_state:
        st.session_state["viewer_backend"] = "COLMAP"
    if "viewer_color_mode" not in st.session_state:
        st.session_state["viewer_color_mode"] = "Photorealistic RGB"

    # Standalone backend studio states
    if "vggt_running" not in st.session_state:
        st.session_state["vggt_running"] = False
    if "vggt_results" not in st.session_state:
        st.session_state["vggt_results"] = None
    if "mast3r_latest_result" not in st.session_state:
        st.session_state["mast3r_latest_result"] = None
    if "input_mode" not in st.session_state:
        st.session_state["input_mode"] = "upload"
    if "reconstruction_colors" not in st.session_state:
        st.session_state["reconstruction_colors"] = None
    if "mesh_updated" not in st.session_state:
        st.session_state["mesh_updated"] = False

    # Active recent activity or notification message
    if "status_notification" not in st.session_state:
        st.session_state["status_notification"] = None
