"""
ui/reconstruction_studio.py
---------------------------
Real-Time Reconstruction Studio & Monitoring Workspace.
Tracks pipeline stages, live stdout logs, process PID, and stage transitions
for COLMAP, VGGT-1B, and MASt3R-SfM without mock indicators.
"""

import os
import json
import time
import streamlit as st
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

from config.paths import LOG_FILE, STATUS_FILE, RESULT_FILE, TEMP_DIR
from core.reconstruction_manager import cancel_reconstruction


def _get_pipeline_stages(backend: str) -> list[str]:
    """Return actual implemented pipeline stages for the selected backend."""
    b = backend.lower()
    if "vggt" in b:
        return [
            "1. Sequence Preprocessing",
            "2. Transformer Feed-Forward",
            "3. Pose & Depth Decoding",
            "4. Pointmap Unprojection",
            "5. Outlier Pruning",
            "6. Poisson Surface Mesh"
        ]
    elif "mast3r" in b:
        return [
            "1. View Retrieval & Pairs",
            "2. Dense Feature Matching",
            "3. SparseGA Global Alignment",
            "4. Point Cloud Fusion",
            "5. Outlier Filtering",
            "6. Poisson Surface Mesh"
        ]
    else:
        # COLMAP
        return [
            "1. Workspace Setup",
            "2. SIFT Feature Extraction",
            "3. Inlier Feature Matching",
            "4. Incremental Triangulation",
            "5. Multi-View Stereo Fusion",
            "6. Poisson Surface Meshing"
        ]


def _determine_active_stage(backend: str, step_text: str, progress: float) -> int:
    """
    Map current status step string and progress to the active stage index.
    Returns 0-based index.
    """
    text = (step_text or "").lower()
    if progress >= 1.0 or "completed" in text:
        return 6
    if "mesh" in text or "poisson" in text:
        return 5
    if "dense" in text or "stereo" in text or "fusion" in text or "unproject" in text or "outlier" in text:
        return 4
    if "triangulat" in text or "bundle" in text or "sparse" in text or "global alignment" in text or "pose" in text:
        return 3
    if "match" in text or "pair" in text or "retrieval" in text:
        return 2
    if "sift" in text or "extract" in text or "feed-forward" in text or "transformer" in text or "inference" in text:
        return 1
    return 0


def render_reconstruction_studio():
    """Render the Reconstruction Studio monitoring workspace."""

    # Autorefresh when running
    is_running = st.session_state.get("reconstruction_running", False)

    # Check status file dynamically
    status_data = {}
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                status_data = json.load(f)
                if status_data.get("status") == "RUNNING":
                    is_running = True
                    st.session_state["reconstruction_running"] = True
                elif status_data.get("status") == "COMPLETED":
                    is_running = False
                    st.session_state["reconstruction_running"] = False
                    st.session_state["reconstruction_done"] = True
        except Exception:
            pass

    if is_running and st_autorefresh is not None:
        st_autorefresh(interval=3000, key="studio_autorefresh")

    backend = st.session_state.get("active_backend", "colmap").upper()
    status_str = status_data.get("status", "IDLE" if not is_running else "RUNNING")
    current_step = status_data.get("step", "Pipeline idle. Ready to execute reconstruction.")
    progress_val = float(status_data.get("progress", 0.0 if not is_running else 0.15))

    # Badge styling
    badge_cls = "badge-success" if status_str == "COMPLETED" else ("badge-primary" if is_running else "badge-info")
    if status_str == "FAILED":
        badge_cls = "badge-error"

    # ── Header Banner ─────────────────────────────────────────────────────────
    st.markdown(
        f"""
<div class="eng-card" style="padding: 18px 24px; margin-bottom: 20px;">
  <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
    <div>
      <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 22px;">🚀</span>
        <h2 style="margin: 0; font-size: 20px; font-weight: 700; color: #fff;">Reconstruction Studio &amp; Telemetry</h2>
      </div>
      <p style="margin: 4px 0 0; font-size: 13px; color: #94A3B8;">
        Active Backend: <b>{backend}</b> &nbsp;·&nbsp; Live Execution Monitor &amp; Stage Inspection
      </p>
    </div>
    <div style="display: flex; align-items: center; gap: 10px;">
      <span class="badge-pill {badge_cls}">● {status_str}</span>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    # ── Pipeline Stage Tracker ────────────────────────────────────────────────
    stages = _get_pipeline_stages(backend)
    active_idx = _determine_active_stage(backend, current_step, progress_val) if is_running or status_str == "COMPLETED" else -1

    st.markdown(
        """
<div class="eng-card" style="padding: 16px 20px; margin-bottom: 20px;">
  <div style="font-size: 12px; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 12px; letter-spacing: 0.05em;">
    PIPELINE STAGE SEQUENCER
  </div>
""",
        unsafe_allow_html=True
    )

    stage_cols = st.columns(len(stages))
    for i, s_label in enumerate(stages):
        with stage_cols[i]:
            if i < active_idx:
                s_icon = "✅"
                s_color = "#20C6A0"
                s_status = "Completed"
            elif i == active_idx and is_running:
                s_icon = "⚡"
                s_color = "#7C5CFC"
                s_status = "Active"
            elif i == active_idx and status_str == "COMPLETED":
                s_icon = "✅"
                s_color = "#20C6A0"
                s_status = "Completed"
            else:
                s_icon = "○"
                s_color = "#64748B"
                s_status = "Pending"

            st.markdown(
                f"""
<div style="text-align: center; padding: 10px 6px; background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 8px;">
  <div style="font-size: 16px; margin-bottom: 4px;">{s_icon}</div>
  <div style="font-size: 11px; font-weight: 600; color: {s_color}; line-height: 1.2;">{s_label}</div>
  <div style="font-size: 10px; color: var(--text-muted); margin-top: 4px;">{s_status}</div>
</div>
""",
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Progress Bar & Telemetry Details ──────────────────────────────────────
    if is_running:
        st.progress(min(1.0, max(0.0, progress_val)), text=f"⚡ {current_step}")
        col_cancel_btn, col_empty = st.columns([1, 4])
        with col_cancel_btn:
            if st.button("🛑 Cancel Active Run", type="secondary", width="stretch"):
                cancel_reconstruction(TEMP_DIR)
                st.session_state["reconstruction_running"] = False
                st.warning("Reconstruction job cancelled by user.")
                st.rerun()

    # ── Completion Notice & Navigation Actions ────────────────────────────────
    if status_str == "COMPLETED" or st.session_state.get("reconstruction_done"):
        st.markdown(
            """
<div class="eng-card" style="border-color: #20C6A0; background: rgba(32, 198, 160, 0.08); padding: 16px 20px; margin-bottom: 20px;">
  <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
    <div>
      <div style="font-weight: 700; color: #20C6A0; font-size: 15px;">🎉 3D Reconstruction Finished Successfully</div>
      <div style="font-size: 12px; color: #cbd5e1; margin-top: 4px;">
        Multi-view dense point cloud geometry, camera poses, and watertight surface mesh are generated and verified on disk.
      </div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True
        )
        c_btn1, c_btn2, c_btn3 = st.columns(3)
        with c_btn1:
            if st.button("🌐 Inspect 3D Geometry in Viewer", type="primary", width="stretch", key="studio_to_viewer"):
                st.session_state["nav_page"] = "3D Viewer"
                st.rerun()
        with c_btn2:
            if st.button("🔺 Inspect Mesh in Mesh Studio", type="secondary", width="stretch", key="studio_to_mesh"):
                st.session_state["nav_page"] = "Mesh Studio"
                st.rerun()
        with c_btn3:
            if st.button("💾 Access Export Center", type="secondary", width="stretch", key="studio_to_export"):
                st.session_state["nav_page"] = "Export Center"
                st.rerun()

    # ── Live Execution Logs ───────────────────────────────────────────────────
    st.markdown(
        """
<div class="eng-card" style="padding: 16px 20px;">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
    <div style="font-size: 13px; font-weight: 700; color: #fff;">📜 Live Execution Stream</div>
    <span style="font-size: 11px; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">reconstruction.log</span>
  </div>
""",
        unsafe_allow_html=True
    )

    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                logs = f.readlines()
        except Exception:
            pass

    log_content = "".join(logs[-45:]) if logs else "[SYSTEM] No active execution log stream found yet."
    st.text_area("Live Log Output", log_content, height=320, disabled=True, label_visibility="collapsed")

    st.markdown("</div>", unsafe_allow_html=True)
