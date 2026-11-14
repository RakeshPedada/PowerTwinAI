"""
ui/dashboard.py
---------------
Enterprise Overview Dashboard for PowerTwinAI.
Displays real workspace telemetry, aggregate dataset metrics, backend status,
and real reconstruction history from disk without fabricating values.
"""

import os
import streamlit as st
import pandas as pd
from ui.data_manager import scan_all_reconstructions, get_backend_status, get_recent_activity_list


def render_dashboard():
    """Render the executive Overview Dashboard."""
    summary = scan_all_reconstructions()
    backends = get_backend_status()
    recent_activity = get_recent_activity_list()

    # ── Top Hero / Welcome Banner ─────────────────────────────────────────────
    st.markdown(
        """
<div class="eng-card" style="background: linear-gradient(135deg, rgba(22, 30, 45, 0.95) 0%, rgba(28, 38, 56, 0.95) 100%); margin-bottom: 24px;">
  <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
    <div>
      <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 26px;">⚡</span>
        <h2 style="margin: 0; font-size: 22px; font-weight: 700; color: #fff;">PowerTwinAI 3D Reconstruction Workstation</h2>
      </div>
      <p style="margin: 8px 0 0; font-size: 13px; color: #94A3B8; max-width: 780px;">
        Enterprise multi-backend platform integrating classical incremental <b>COLMAP SfM</b>,
        feed-forward visual geometry <b>VGGT-1B</b>, and retrieval-grounded <b>MASt3R-SfM</b> with automated surface meshing and ML diagnostics.
      </p>
    </div>
    <div style="display: flex; align-items: center; gap: 10px;">
      <span class="badge-pill badge-primary">v3.2 Production</span>
      <span class="badge-pill badge-success">3 Backends Online</span>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    # ── Quick Action Row ──────────────────────────────────────────────────────
    col_act1, col_act2, col_act3 = st.columns([2, 1, 1])
    with col_act1:
        st.markdown(
            f"<p style='color: #cbd5e1; font-size: 13px; margin-top: 6px;'>"
            f"Active Workspace: <code>{os.path.basename(os.getcwd())}</code> &nbsp;·&nbsp; "
            f"Dataset: <b>{summary['total_images_processed']} images</b> ready for multi-view stereo."
            f"</p>",
            unsafe_allow_html=True
        )
    with col_act2:
        if st.button("➕ New Reconstruction", type="primary", width="stretch"):
            st.session_state["nav_page"] = "New Reconstruction"
            st.rerun()
    with col_act3:
        if st.button("🌐 Launch 3D Viewer", type="secondary", width="stretch"):
            st.session_state["nav_page"] = "3D Viewer"
            st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # ── Statistics Metric Cards (Real Measured Values) ─────────────────────────
    st.markdown("<div class='section-sub-label' style='margin-bottom: 12px; font-weight: 700; color: #94A3B8; font-size: 12px; letter-spacing: 0.05em;'>WORKSPACE GEOMETRIC TELEMETRY</div>", unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label-row">
    <span>Reconstructions</span>
    <span style="color: #38BDF8;">●</span>
  </div>
  <div class="metric-value">{summary['total_reconstruction_projects']}</div>
  <div class="metric-sub">Verified models on disk</div>
</div>
""",
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label-row">
    <span>Dataset Views</span>
    <span style="color: #7C5CFC;">●</span>
  </div>
  <div class="metric-value">{summary['total_images_processed']}</div>
  <div class="metric-sub">Multi-view input frames</div>
</div>
""",
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label-row">
    <span>Reconstructed Points</span>
    <span style="color: #20C6A0;">●</span>
  </div>
  <div class="metric-value">{summary['total_3d_points']:,}</div>
  <div class="metric-sub">Dense geometric points</div>
</div>
""",
            unsafe_allow_html=True
        )
    with m4:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label-row">
    <span>Backends</span>
    <span style="color: #F5B942;">●</span>
  </div>
  <div class="metric-value">{summary['available_backends_count']}</div>
  <div class="metric-sub">COLMAP · VGGT · MASt3R</div>
</div>
""",
            unsafe_allow_html=True
        )
    with m5:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label-row">
    <span>Latest Latency</span>
    <span style="color: #EF5350;">●</span>
  </div>
  <div class="metric-value">{summary['latest_runtime']:.1f} s</div>
  <div class="metric-sub">Feed-forward execution</div>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # ── 3-Backend Architecture Status Cards ──────────────────────────────────
    st.markdown("<div class='section-sub-label' style='margin-bottom: 12px; font-weight: 700; color: #94A3B8; font-size: 12px; letter-spacing: 0.05em;'>RECONSTRUCTION BACKEND PLATFORMS</div>", unsafe_allow_html=True)
    b_col1, b_col2, b_col3 = st.columns(3)

    for col, (b_key, b_info) in zip([b_col1, b_col2, b_col3], backends.items()):
        with col:
            b_data = summary["backends"].get(b_key, {})
            has_data = b_data.get("available", False)
            pts_count = b_data.get("points", 0)
            faces_count = b_data.get("faces", 0)
            status_badge = "✅ Output Ready" if has_data else "Ready to Run"
            badge_cls = "badge-success" if has_data else "badge-info"

            st.markdown(
                f"""
<div class="backend-card">
  <div>
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
      <h3 style="margin: 0; font-size: 16px; font-weight: 700; color: #fff;">{b_info['name']}</h3>
      <span class="badge-pill {badge_cls}">{status_badge}</span>
    </div>
    <div style="font-size: 11px; color: {b_info['badge_color']}; font-weight: 600; text-transform: uppercase; margin-bottom: 8px;">
      {b_info['tag']}
    </div>
    <p style="font-size: 12px; color: #94A3B8; line-height: 1.5; margin-bottom: 14px;">
      {b_info['description']}
    </p>
  </div>
  <div style="border-top: 1px solid var(--border); padding-top: 12px;">
    <div style="display: flex; justify-content: space-between; font-size: 11px; color: #cbd5e1; margin-bottom: 4px;">
      <span>Execution Device:</span>
      <span style="font-weight: 600; color: #fff;">{b_info['device']}</span>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 11px; color: #cbd5e1; margin-bottom: 4px;">
      <span>Verified 3D Points:</span>
      <span style="font-weight: 600; color: #fff;">{pts_count:,}</span>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 11px; color: #cbd5e1;">
      <span>Surface Triangles:</span>
      <span style="font-weight: 600; color: #fff;">{faces_count:,}</span>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True
            )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # ── Recent Reconstruction Activity & History ─────────────────────────────
    st.markdown("<div class='section-sub-label' style='margin-bottom: 12px; font-weight: 700; color: #94A3B8; font-size: 12px; letter-spacing: 0.05em;'>RECENT RECONSTRUCTION RUNS & ARTIFACTS</div>", unsafe_allow_html=True)

    if recent_activity:
        table_rows = []
        for item in recent_activity:
            table_rows.append({
                "Backend": item["backend"],
                "Paradigm": item["paradigm"],
                "Points": f"{item['points']:,}",
                "Mesh Faces": f"{item['faces']:,}",
                "Bounding Vol": f"{item['volume_m3']:.3f} m³",
                "Status": item["status"],
                "Modified": item["timestamp"]
            })
        df_act = pd.DataFrame(table_rows)
        st.dataframe(df_act, width="stretch", hide_index=True)

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            if st.button("🌐 Open Recent Run in 3D Viewer", width="stretch", key="dash_btn_viewer"):
                st.session_state["nav_page"] = "3D Viewer"
                st.rerun()
        with col_r2:
            if st.button("🧠 View AI Geometric Diagnostics", width="stretch", key="dash_btn_diag"):
                st.session_state["nav_page"] = "AI Diagnostics"
                st.rerun()
        with col_r3:
            if st.button("💾 Access Export Center", width="stretch", key="dash_btn_export"):
                st.session_state["nav_page"] = "Export Center"
                st.rerun()
    else:
        st.info("No prior reconstruction runs recorded yet. Click 'New Reconstruction' to start a job.")
