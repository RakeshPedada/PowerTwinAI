"""
ui/mesh_studio_page.py
----------------------
Dedicated 3D Surface Mesh Studio & Topology Inspector.
Provides interactive 3D mesh rendering, structural manifold diagnostics,
photorealistic color transfers, and Screened Poisson mesh generation.
"""

import os
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import open3d as o3d

from ui.data_manager import find_artifact, load_cached_mesh, load_cached_point_cloud
from mesh_generator import generate_mesh, is_mesh_ply, get_ply_statistics


def render_mesh_studio_page():
    """Render the Mesh Studio page."""

    st.markdown(
        """
<div class="eng-card" style="padding: 16px 22px; margin-bottom: 20px;">
  <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
    <div>
      <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 22px;">🔺</span>
        <h2 style="margin: 0; font-size: 20px; font-weight: 700; color: #fff;">3D Mesh Studio &amp; Topology Inspector</h2>
      </div>
      <p style="margin: 4px 0 0; font-size: 13px; color: #94A3B8;">
        Inspect triangular surface meshes, manifold properties, and run Screened Poisson surface reconstruction.
      </p>
    </div>
    <span class="badge-pill badge-primary">Watertight Meshing</span>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    # ── Mesh Candidate Detection ──────────────────────────────────────────────
    mesh_candidates = {
        "COLMAP Final Mesh": find_artifact("output/mesh.ply") or find_artifact("outputs/mesh/final_mesh.ply"),
        "VGGT-1B Screened Poisson Mesh": find_artifact("outputs/vggt/vggt_surface_mesh.ply"),
        "MASt3R-SfM Surface Mesh": find_artifact("outputs/mast3r/mesh/mesh.ply"),
    }
    available_meshes = {k: v for k, v in mesh_candidates.items() if v and os.path.exists(v)}

    # Control bar
    c1, c2, c3 = st.columns([2, 1.5, 1.5])
    with c1:
        if available_meshes:
            selected_mesh_name = st.selectbox("Active Mesh Model:", list(available_meshes.keys()), index=0)
            selected_mesh_path = available_meshes[selected_mesh_name]
        else:
            selected_mesh_name = "None"
            selected_mesh_path = None
            st.info("No surface mesh files detected on disk yet. You can generate one below.")

    with c2:
        shading_mode = st.selectbox(
            "Surface Material Shading:",
            ["Photorealistic RGB Colors", "Engineering Lavender", "Z-Elevation Turbo"],
            index=0
        )
    with c3:
        mesh_opacity = st.slider("Mesh Opacity:", 0.3, 1.0, 0.95, 0.05)

    # ── Render 3D Mesh Viewer ─────────────────────────────────────────────────
    if selected_mesh_path and os.path.exists(selected_mesh_path):
        mesh_data = load_cached_mesh(selected_mesh_path, max_faces=60000)
        ply_stats = get_ply_statistics(selected_mesh_path)

        # Structural Telemetry Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Mesh Vertices", f"{ply_stats.get('vertex_count', 0):,}")
        m2.metric("Triangular Faces", f"{ply_stats.get('face_count', 0):,}")
        m3.metric("Verified Faces", "✅ element face" if ply_stats.get("has_faces") else "❌ Point Cloud")
        m4.metric("Vertex Normals", "✅ Present" if ply_stats.get("has_normals") else "❌ None")
        m5.metric("File Size", f"{ply_stats.get('file_size_mb', 0.0)} MB")

        if mesh_data:
            verts = mesh_data["vertices"]
            tris = mesh_data["triangles"]
            cols = mesh_data["colors"]

            fig = go.Figure()

            mesh_kwargs = {}
            if "RGB" in shading_mode and mesh_data["has_colors"] and cols is not None:
                mesh_kwargs["vertexcolor"] = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in cols]
            elif "Height" in shading_mode or "Turbo" in shading_mode:
                mesh_kwargs["intensity"] = verts[:, 2]
                mesh_kwargs["colorscale"] = "Turbo"
            else:
                mesh_kwargs["color"] = "#a855f7"

            fig.add_trace(
                go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=tris[:, 0],
                    j=tris[:, 1],
                    k=tris[:, 2],
                    opacity=mesh_opacity,
                    name=f"{selected_mesh_name}",
                    lighting=dict(ambient=0.45, diffuse=0.6, roughness=0.3, specular=0.2),
                    **mesh_kwargs
                )
            )

            axis_cfg = dict(showbackground=True, backgroundcolor="#0F1420", gridcolor="#2A3548", titlefont=dict(color="#94A3B8"))
            fig.update_layout(
                paper_bgcolor="#0B0F19",
                plot_bgcolor="#0B0F19",
                height=650,
                margin=dict(l=0, r=0, t=10, b=0),
                scene=dict(
                    xaxis=dict(title="X (m)", **axis_cfg),
                    yaxis=dict(title="Y (m)", **axis_cfg),
                    zaxis=dict(title="Z (m)", **axis_cfg),
                    aspectmode="data"
                )
            )
            st.plotly_chart(fig, width="stretch")

    # ── Interactive Poisson Surface Reconstruction Generator ──────────────────
    with st.expander("🛠️ Interactive Poisson Surface Reconstruction Generator", expanded=False):
        st.markdown(
            "<p style='color: #94A3B8; font-size: 13px;'>"
            "Reconstruct a clean, watertight 3D triangular surface mesh from a point cloud using Screened Poisson Surface Reconstruction."
            "</p>",
            unsafe_allow_html=True
        )

        pcd_source_options = {
            "VGGT Dense Point Cloud": find_artifact("outputs/vggt/vggt_point_cloud.ply"),
            "COLMAP Cleaned Point Cloud": find_artifact("output/reconstruction.ply"),
            "MASt3R-SfM Point Cloud": find_artifact("outputs/mast3r/point_cloud.ply")
        }
        valid_sources = {k: v for k, v in pcd_source_options.items() if v and os.path.exists(v)}

        if valid_sources:
            src_choice = st.selectbox("Source Point Cloud:", list(valid_sources.keys()))
            selected_src_file = valid_sources[src_choice]

            gc1, gc2, gc3, gc4 = st.columns(4)
            with gc1:
                p_depth = st.slider("Poisson Depth (Octree Resolution):", 6, 11, 8, 1, key="ms_poisson_depth")
            with gc2:
                p_density = st.slider("Density Pruning Trim (%):", 0.0, 25.0, 5.0, 1.0, key="ms_density_trim")
            with gc3:
                p_outlier = st.slider("Outlier Filter (std dev):", 1.0, 3.0, 1.5, 0.1, key="ms_outlier_std")
            with gc4:
                p_smooth = st.slider("Taubin Smoothing Steps:", 0, 10, 3, 1, key="ms_taubin_steps")

            if st.button("🚀 Run Surface Meshing", type="primary", width="stretch", key="ms_run_mesh_btn"):
                with st.spinner("Reconstructing triangular surface mesh with Open3D..."):
                    try:
                        res = generate_mesh(
                            selected_src_file,
                            output_dir="output",
                            output_name="mesh",
                            poisson_depth=p_depth,
                            density_quantile=p_density / 100.0,
                            outlier_std_ratio=p_outlier,
                            smooth_iterations=p_smooth
                        )
                        st.success(f"🎉 Generated 3D Mesh: {res['vertices']:,} vertices, {res['triangles']:,} triangular faces!")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Mesh generation error: {e}")
        else:
            st.info("No source point clouds available to mesh.")
