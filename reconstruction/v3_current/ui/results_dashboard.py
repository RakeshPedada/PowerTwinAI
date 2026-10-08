"""
ui/results_dashboard.py
-----------------------
Render reconstruction summary, structured file downloads (Point Cloud, Surface Mesh,
Raw Dense, OBJ/STL), and pass multi-stage point clouds to the 5-tab visualizer.
"""

import os
import numpy as np
import streamlit as st
from mesh_generator import is_mesh_ply


def render_results_dashboard(result_file):
    """
    Render reconstruction dashboard and extract all point cloud stages.

    Returns
    -------
    dict | None
        Dictionary containing all reconstruction datasets for 5-tab visualization.
    """
    if (
        not st.session_state.reconstruction_done
        or not os.path.exists(result_file)
    ):
        return None

    data = np.load(
        result_file,
        allow_pickle=True
    )

    points = data["points"]
    cameras = data["cameras"]
    colors = data["colors"] if "colors" in data.files else None
    st.session_state["reconstruction_colors"] = colors

    sparse_points = data["sparse_points"] if "sparse_points" in data.files else np.zeros((0, 3))
    raw_points = data["raw_points"] if "raw_points" in data.files else np.zeros((0, 3))

    analytics = data["analytics"].item() if hasattr(data["analytics"], "item") else data["analytics"]

    processing_time = float(data["processing_time"])
    colmap_time = float(data["colmap_time"]) if "colmap_time" in data.files else 0.0
    reconstruction_time = float(data["reconstruction_time"]) if "reconstruction_time" in data.files else processing_time
    total_pipeline_time = float(data["total_pipeline_time"]) if "total_pipeline_time" in data.files else (colmap_time + reconstruction_time)

    ply_path = str(data["ply_path"].item()) if "ply_path" in data.files else os.path.join("output", "reconstruction.ply")
    mesh_path = str(data["mesh_path"].item()) if "mesh_path" in data.files else os.path.join("output", "mesh.ply")
    raw_ply_path = str(data["raw_ply_path"].item()) if "raw_ply_path" in data.files else os.path.join("outputs", "dense", "raw_pointcloud.ply")

    st.markdown(
        """
        <div class="status-box-success">
          <span>✅</span> &nbsp;<b>Reconstruction Completed Successfully</b> — Multi-view stereo geometry, dense point clouds &amp; surface mesh ready.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='section-title'><span>⬇️</span> Download Reconstruction Artifacts</div>",
        unsafe_allow_html=True
    )

    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        if ply_path and os.path.exists(ply_path):
            with open(ply_path, "rb") as ply_file:
                st.download_button(
                    label="⬇️ Cleaned Point Cloud (.ply)",
                    data=ply_file.read(),
                    file_name="filtered_pointcloud.ply",
                    mime="application/octet-stream",
                    use_container_width=True,
                    key="dl_cleaned_pcd_btn"
                )
        else:
            st.warning(f"Point cloud file not found: {ply_path}")

    with dl_col2:
        if mesh_path and os.path.exists(mesh_path):
            with open(mesh_path, "rb") as mesh_file:
                st.download_button(
                    label="⬇️ 3D Surface Mesh (.ply with faces)",
                    data=mesh_file.read(),
                    file_name="final_mesh.ply",
                    mime="application/octet-stream",
                    use_container_width=True,
                    key="dl_surface_mesh_btn"
                )
        else:
            st.info("Surface mesh not yet generated.")

    with dl_col3:
        if raw_ply_path and os.path.exists(raw_ply_path):
            with open(raw_ply_path, "rb") as raw_file:
                st.download_button(
                    label="⬇️ Raw Dense Point Cloud (.ply)",
                    data=raw_file.read(),
                    file_name="raw_pointcloud.ply",
                    mime="application/octet-stream",
                    use_container_width=True,
                    key="dl_raw_dense_btn"
                )
        else:
            mesh_obj = os.path.join("output", "mesh.obj")
            if os.path.exists(mesh_obj):
                with open(mesh_obj, "rb") as obj_file:
                    st.download_button(
                        label="⬇️ 3D Surface Mesh (.obj)",
                        data=obj_file.read(),
                        file_name="surface_mesh.obj",
                        mime="application/octet-stream",
                        use_container_width=True,
                        key="dl_mesh_obj_btn"
                    )

    st.markdown(
        "<div class='section-title'><span>⏱️</span> Processing Time Breakdown</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
<div class="glass-stat-grid">
  <div class="glass-stat-card">
    <div class="glass-stat-label">COLMAP Stage</div>
    <div class="glass-stat-value">{colmap_time:.2f}s</div>
    <div class="glass-stat-sub">Feature matching &amp; sparse SfM</div>
  </div>
  <div class="glass-stat-card">
    <div class="glass-stat-label">Dense &amp; Mesh</div>
    <div class="glass-stat-value">{reconstruction_time:.2f}s</div>
    <div class="glass-stat-sub">Stereo fusion &amp; filtering</div>
  </div>
  <div class="glass-stat-card">
    <div class="glass-stat-label">Total Pipeline</div>
    <div class="glass-stat-value">{total_pipeline_time:.2f}s</div>
    <div class="glass-stat-sub">End-to-end execution time</div>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    return {
        "points": points,
        "colors": colors,
        "cameras": cameras,
        "sparse_points": sparse_points,
        "raw_points": raw_points,
        "analytics": analytics,
        "processing_time": processing_time,
        "colmap_time": colmap_time,
        "reconstruction_time": reconstruction_time,
        "total_pipeline_time": total_pipeline_time,
        "ply_path": ply_path,
        "mesh_path": mesh_path
    }