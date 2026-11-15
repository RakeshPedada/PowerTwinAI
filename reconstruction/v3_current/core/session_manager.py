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


def _is_active_process_alive(pid: int) -> bool:
    """Accurately check whether the given PID is actively running Python or COLMAP."""
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        try:
            res = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, timeout=2)
            if "No tasks" in res.stdout or "INFO:" in res.stdout:
                return False
            lines = [l for l in res.stdout.strip().split("\n") if l.strip()]
            for line in lines[2:]:
                if str(pid) in line and ("python" in line.lower() or "colmap" in line.lower()):
                    return True
            return False
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _is_colmap_process_alive() -> bool:
    """Check if colmap.exe is currently running on the system."""
    if os.name == "nt":
        try:
            out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq colmap.exe"], capture_output=True, text=True, timeout=2)
            return "colmap.exe" in out.stdout.lower()
        except Exception:
            return False
    return False


def initialize_session():
    """
    Initialize all required Streamlit session variables with reactive background tracking.
    Never allows stale status files or dead processes to falsely lock the application.
    """
    import subprocess
    is_running = False
    already_done = False

    status_state = "IDLE"
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                status_state = json.load(f).get("status", "IDLE")
        except Exception:
            pass

    # Process alive check
    runner_pid_file = os.path.join(os.path.dirname(STATUS_FILE), "runner.pid")
    pid_alive = False
    if os.path.exists(runner_pid_file):
        try:
            with open(runner_pid_file, "r", encoding="utf-8") as pf:
                r_pid = int(pf.read().strip())
            pid_alive = _is_active_process_alive(r_pid)
            if not pid_alive:
                try:
                    os.remove(runner_pid_file)
                except Exception:
                    pass
        except Exception:
            pass

    colmap_alive = _is_colmap_process_alive()
    actual_process_alive = pid_alive or colmap_alive

    if actual_process_alive:
        is_running = True
    elif status_state == "RUNNING" and not actual_process_alive:
        # Stale RUNNING state from aborted or terminated job — auto-heal to IDLE!
        is_running = False
        try:
            with open(STATUS_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "status": "IDLE",
                    "step": "Pipeline ready for reconstruction.",
                    "progress": 0.0
                }, f, indent=2)
        except Exception:
            pass
    elif status_state == "COMPLETED":
        already_done = True
        is_running = False
    else:
        is_running = False

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
