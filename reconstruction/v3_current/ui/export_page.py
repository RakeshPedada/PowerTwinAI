"""
ui/export_page.py
-----------------
Enterprise Export Center & Artifact Repository.
Provides download capabilities for all verified reconstruction outputs on disk:
Point Clouds (PLY), Watertight Meshes (PLY, OBJ, STL), Camera Models (COLMAP/JSON),
and Quantitative Diagnostic Reports.
"""

import os
import streamlit as st
from ui.data_manager import find_artifact


def _fmt_mb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 * 1024):.2f} MB"


def _render_download_card(title: str, rel_path: str, mime: str, download_name: str, desc: str, icon: str = "💾"):
    """Render an individual artifact download row with real file status verification."""
    full_path = find_artifact(rel_path)
    exists = full_path is not None and os.path.exists(full_path)

    if exists:
        sz = os.path.getsize(full_path)
        sz_str = _fmt_mb(sz)
        badge = '<span class="badge-pill badge-success">Available</span>'
    else:
        sz_str = "Not Generated"
        badge = '<span class="badge-pill badge-error">Missing</span>'

    st.markdown(
        f"""
<div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 18px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
  <div style="display: flex; align-items: center; gap: 12px;">
    <div style="font-size: 22px;">{icon}</div>
    <div>
      <div style="font-size: 14px; font-weight: 700; color: #fff;">{title} <span style="font-size: 11px; color: #94A3B8; font-weight: normal;">({download_name})</span></div>
      <div style="font-size: 12px; color: #94A3B8; margin-top: 2px;">{desc} &nbsp;·&nbsp; Size: <b style="color: #cbd5e1;">{sz_str}</b></div>
    </div>
  </div>
  <div style="display: flex; align-items: center; gap: 10px;">
    {badge}
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    if exists:
        with open(full_path, "rb") as f:
            st.download_button(
                label=f"⬇️ Download {download_name} ({sz_str})",
                data=f.read(),
                file_name=download_name,
                mime=mime,
                width="stretch",
                key=f"dl_btn_{download_name.replace('.', '_')}"
            )


def render_export_page():
    """Render the centralized Export Center."""

    st.markdown(
        """
<div class="eng-card" style="padding: 16px 22px; margin-bottom: 20px;">
  <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
    <div>
      <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 22px;">💾</span>
        <h2 style="margin: 0; font-size: 20px; font-weight: 700; color: #fff;">Export Center &amp; Artifact Repository</h2>
      </div>
      <p style="margin: 4px 0 0; font-size: 13px; color: #94A3B8;">
        Export point clouds, watertight surface meshes, calibrated camera models, and scientific diagnostic reports.
      </p>
    </div>
    <span class="badge-pill badge-primary">Structured Exports</span>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    tab_mesh, tab_pcd, tab_cam, tab_reports = st.tabs([
        "🔺 3D Surface Meshes",
        "☁️ 3D Point Clouds",
        "📷 Camera Poses & Models",
        "📋 Diagnostic Reports & Logs"
    ])

    # ── TAB 1: 3D Surface Meshes ──────────────────────────────────────────────
    with tab_mesh:
        st.markdown("<div class='section-sub-label' style='margin-bottom: 12px; font-weight: 700; color: #94A3B8;'>WATERTIGHT TRIANGULAR MESH ARTIFACTS</div>", unsafe_allow_html=True)
        _render_download_card("COLMAP Final Mesh (.ply)", "output/mesh.ply", "application/octet-stream", "colmap_final_mesh.ply", "Watertight Poisson surface mesh with photorealistic RGB vertex colors", "🔺")
        _render_download_card("VGGT Surface Mesh (.ply)", "outputs/vggt/vggt_surface_mesh.ply", "application/octet-stream", "vggt_surface_mesh.ply", "Screened Poisson 3D surface mesh generated from feed-forward VGGT points", "🔺")
        _render_download_card("MASt3R Surface Mesh (.ply)", "outputs/mast3r/mesh/mesh.ply", "application/octet-stream", "mast3r_surface_mesh.ply", "Watertight triangular surface mesh generated from MASt3R SparseGA geometry", "🔺")
        _render_download_card("COLMAP Surface Mesh (.obj)", "output/mesh.obj", "text/plain", "mesh.obj", "Standard Wavefront OBJ triangular polygon mesh", "📦")
        _render_download_card("COLMAP Surface Mesh (.stl)", "output/mesh.stl", "application/octet-stream", "mesh.stl", "Binary Stereolithography STL format for CAD and 3D printing", "📦")

    # ── TAB 2: 3D Point Clouds ────────────────────────────────────────────────
    with tab_pcd:
        st.markdown("<div class='section-sub-label' style='margin-bottom: 12px; font-weight: 700; color: #94A3B8;'>DENSE &amp; SPARSE 3D POINT CLOUDS</div>", unsafe_allow_html=True)
        _render_download_card("VGGT Dense Point Cloud (.ply)", "outputs/vggt/vggt_point_cloud.ply", "application/octet-stream", "vggt_point_cloud.ply", "Dense multi-view unprojected 3D point cloud from VGGT transformer", "☁️")
        _render_download_card("MASt3R-SfM Point Cloud (.ply)", "outputs/mast3r/point_cloud.ply", "application/octet-stream", "mast3r_point_cloud.ply", "Dense 3D point cloud from MASt3R SparseGA global alignment", "☁️")
        _render_download_card("COLMAP Cleaned Point Cloud (.ply)", "output/reconstruction.ply", "application/octet-stream", "colmap_cleaned_pointcloud.ply", "Filtered multi-view stereo dense point cloud with statistical outlier removal", "✨")
        _render_download_card("COLMAP Raw Dense Point Cloud (.ply)", "outputs/dense/raw_pointcloud.ply", "application/octet-stream", "raw_pointcloud.ply", "Unfiltered raw multi-view stereo point cloud prior to cleaning", "☁️")

    # ── TAB 3: Camera Poses & Models ──────────────────────────────────────────
    with tab_cam:
        st.markdown("<div class='section-sub-label' style='margin-bottom: 12px; font-weight: 700; color: #94A3B8;'>CALIBRATED CAMERA INTRINSICS &amp; 6-DOF POSES</div>", unsafe_allow_html=True)
        _render_download_card("VGGT Cameras (.json)", "outputs/vggt/cameras/cameras.json", "application/json", "vggt_cameras.json", "Extrinsics, scaled intrinsics, focal length, and 6-DOF camera positions", "📷")
        _render_download_card("MASt3R Cameras (.json)", "outputs/mast3r/cameras/cameras.json", "application/json", "mast3r_cameras.json", "Camera poses and focal parameters from MASt3R SparseGA", "📷")
        _render_download_card("COLMAP Cameras Text (.txt)", "colmap_data/cameras.txt", "text/plain", "cameras.txt", "Standard COLMAP camera intrinsic parameters file", "📄")
        _render_download_card("COLMAP Images Text (.txt)", "colmap_data/images.txt", "text/plain", "images.txt", "Standard COLMAP 6-DOF image extrinsics and feature observations", "📄")

    # ── TAB 4: Diagnostic Reports & Logs ──────────────────────────────────────
    with tab_reports:
        st.markdown("<div class='section-sub-label' style='margin-bottom: 12px; font-weight: 700; color: #94A3B8;'>RECONSTRUCTION TELEMETRY &amp; SCIENTIFIC AUDIT LOGS</div>", unsafe_allow_html=True)
        _render_download_card("20-Metric Diagnostic Suite (.json)", "temp_session/diagnostics.json", "application/json", "reconstruction_diagnostics.json", "Comprehensive 20-metric quantitative diagnostic assessment report", "📊")
        _render_download_card("VGGT Reconstruction Metadata (.json)", "outputs/vggt/metadata.json", "application/json", "vggt_metadata.json", "Complete telemetry, runtime latency, bounding volume, and density metrics", "📋")
        _render_download_card("ML Candidate Pairs Telemetry (.json)", "temp_session/ml_pair_telemetry.json", "application/json", "ml_pair_telemetry.json", "Visual embeddings, spatial pyramid moments, and candidate graph stats", "🧠")
        _render_download_card("COLMAP Pair Matching Logs (.csv)", "output/pair_logs.csv", "text/csv", "pair_logs.csv", "CSV log of candidate pair image matches and verified inliers", "📋")
        _render_download_card("Pipeline Execution Stream (.log)", "temp_session/reconstruction.log", "text/plain", "reconstruction.log", "Full stdout and stderr logs from reconstruction runners", "📜")
