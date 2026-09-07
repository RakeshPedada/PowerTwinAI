import streamlit as st


def render_reconstruction_summary(
    analytics,
    uploaded_files,
    cameras,
    points,
    processing_time,
    colmap_time,
    reconstruction_time,
    total_pipeline_time
):
    """
    Render final reconstruction statistics.
    """

    analytics_data = analytics if isinstance(analytics, dict) else (
        analytics.item() if hasattr(analytics, "item") else {}
    )

    registered_images = int(analytics_data.get("registered_images", len(cameras)))
    input_images = int(
        analytics_data.get(
            "total_input_images",
            len(uploaded_files) if uploaded_files else registered_images
        )
    )
    registration_ratio = float(
        analytics_data.get(
            "registration_ratio",
            (registered_images / max(input_images, 1)) * 100.0
        )
    )
    sparse_points = int(analytics_data.get("total_sparse_points", 0))
    dense_points = int(analytics_data.get("total_dense_points", len(points)))
    health_score = float(analytics_data.get("health_score", 100.0))

    st.markdown(
        """
<div class="figma-card">
  <div class="card-header-bar">
    <div class="card-header-icon">📋</div>
    <div class="card-header-titles">
      <h3>Reconstruction Analytics &amp; Performance</h3>
      <p>Comprehensive telemetry, quality metrics, and registration health</p>
    </div>
  </div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
<div class="glass-stat-grid">
  <div class="glass-stat-card">
    <div class="glass-stat-label">Registered Images</div>
    <div class="glass-stat-value">{registered_images:,} / {input_images:,}</div>
    <div class="glass-stat-sub">Registration Ratio: <b>{registration_ratio:.1f}%</b></div>
  </div>
  <div class="glass-stat-card">
    <div class="glass-stat-label">Cleaned 3D Points</div>
    <div class="glass-stat-value">{len(points):,}</div>
    <div class="glass-stat-sub">Dense points: {dense_points:,}</div>
  </div>
  <div class="glass-stat-card">
    <div class="glass-stat-label">Camera Poses</div>
    <div class="glass-stat-value">{len(cameras):,}</div>
    <div class="glass-stat-sub">6-DOF estimated trajectories</div>
  </div>
  <div class="glass-stat-card">
    <div class="glass-stat-label">Health Score</div>
    <div class="glass-stat-value">{health_score:.1f}%</div>
    <div class="glass-stat-sub">Pipeline stability &amp; bundle adjustment</div>
  </div>
</div>

<div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.06);">
  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; font-size: 13px; color: #cbd5e1;">
    <div>🔹 Sparse Points: <b style="color:#d8b4fe;">{sparse_points:,}</b></div>
    <div>🔹 Dense Points Generated: <b style="color:#f472b6;">{dense_points:,}</b></div>
    <div>🔹 Reconstruction Stage Time: <b style="color:#67e8f9;">{processing_time:.2f}s</b></div>
    <div>🔹 COLMAP Backend Time: <b style="color:#67e8f9;">{colmap_time:.2f}s</b></div>
    <div>🔹 Dense + Mesh Time: <b style="color:#67e8f9;">{reconstruction_time:.2f}s</b></div>
    <div>🔹 Total Pipeline Time: <b style="color:#4ade80;">{total_pipeline_time:.2f}s</b></div>
  </div>
</div>
</div>
""",
        unsafe_allow_html=True
    )