"""
ui/sidebar.py
-------------
Persistent navigation sidebar for the PowerTwinAI workstation.
Renders branding, page links with active indicators, and live hardware telemetry.
"""

import streamlit as st
from core.gpu_detector import get_gpu_info


NAV_ITEMS = [
    ("📊 Overview", "Overview"),
    ("➕ New Reconstruction", "New Reconstruction"),
    ("🚀 Reconstruction Studio", "Reconstruction Studio"),
    ("🌐 3D Viewer", "3D Viewer"),
    ("🧠 AI Diagnostics", "AI Diagnostics"),
    ("⚖️ Backend Comparison", "Backend Comparison"),
    ("🔺 Mesh Studio", "Mesh Studio"),
    ("💾 Export Center", "Export Center"),
    ("⚙️ Settings", "Settings"),
]


def render_sidebar() -> str:
    """
    Render persistent sidebar navigation.
    Returns the currently active page name.
    """
    with st.sidebar:
        # 1. Branding Header
        st.markdown(
            """
<div class="sidebar-brand-container">
  <div class="brand-icon-box">⚡</div>
  <div class="brand-text-col">
    <div class="brand-app-title">PowerTwin<span>AI</span></div>
    <div class="brand-app-sub">3D Reconstruction Studio</div>
  </div>
</div>
""",
            unsafe_allow_html=True
        )

        st.markdown('<div class="sidebar-nav-header">Workstation Navigation</div>', unsafe_allow_html=True)

        current_page = st.session_state.get("nav_page", "Overview")

        # Map display labels to canonical names
        labels = [item[0] for item in NAV_ITEMS]
        keys = [item[1] for item in NAV_ITEMS]

        # Determine index
        default_idx = 0
        if current_page in keys:
            default_idx = keys.index(current_page)

        # Render styled radio navigation
        selected_label = st.radio(
            "Navigation",
            labels,
            index=default_idx,
            label_visibility="collapsed",
            key="sidebar_navigation_radio"
        )

        # Match label back to key
        selected_page = keys[labels.index(selected_label)]
        if selected_page != current_page:
            st.session_state["nav_page"] = selected_page
            st.rerun()

        # 2. Live Hardware & Pipeline Telemetry Footer
        gpu = get_gpu_info()
        cuda_badge_color = "#20C6A0" if gpu.get("cuda_available") else "#F5B942"
        cuda_label = "CUDA (GPU)" if gpu.get("cuda_available") else "CPU Mode"
        is_running = st.session_state.get("reconstruction_running", False)
        status_color = "#7C5CFC" if is_running else "#20C6A0"
        status_text = "Reconstruction Active" if is_running else "Idle · Ready"

        st.markdown(
            f"""
<div class="sidebar-footer-card">
  <div class="sidebar-footer-row">
    <span>Hardware</span>
    <span class="sidebar-footer-val" style="color: {cuda_badge_color};">● {cuda_label}</span>
  </div>
  <div class="sidebar-footer-row">
    <span>GPU / Processor</span>
    <span class="sidebar-footer-val" title="{gpu.get('name', 'CPU')}">{gpu.get('name', 'CPU')[:16]}</span>
  </div>
  <div class="sidebar-footer-row">
    <span>Engine State</span>
    <span class="sidebar-footer-val" style="color: {status_color};">● {status_text}</span>
  </div>
  <div style="border-top: 1px solid var(--border); margin: 8px 0; padding-top: 6px;">
    <div class="sidebar-footer-row" style="font-size: 11px; color: var(--text-muted);">
      <span>PowerTwinAI</span>
      <span>v3.2 · Enterprise</span>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True
        )

        return selected_page
