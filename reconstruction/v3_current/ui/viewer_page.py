"""
ui/viewer_page.py
-----------------
Professional 3D Viewport & Multi-Backend Geometry Inspection Canvas.
Renders dense 3D point clouds, triangular surface meshes, and camera trajectories
with high-performance Plotly WebGL, dark canvas background, and rich display controls.
"""

import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ui.data_manager import (
    find_artifact,
    load_cached_point_cloud,
    load_cached_mesh,
    scan_all_reconstructions
)


def _build_bounding_box_lines(min_b: list, max_b: list) -> go.Scatter3d:
    """Generate wireframe lines for 3D bounding box."""
    x0, y0, z0 = min_b
    x1, y1, z1 = max_b

    # 12 edges of a box
    lines_x = [
        x0, x1, x1, x0, x0, None,
        x0, x1, x1, x0, x0, None,
        x0, x0, None, x1, x1, None,
        x1, x1, None, x0, x0, None
    ]
    lines_y = [
        y0, y0, y1, y1, y0, None,
        y0, y0, y1, y1, y0, None,
        y0, y0, None, y0, y0, None,
        y1, y1, None, y1, y1, None
    ]
    lines_z = [
        z0, z0, z0, z0, z0, None,
        z1, z1, z1, z1, z1, None,
        z0, z1, None, z0, z1, None,
        z0, z1, None, z0, z1, None
    ]

    return go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode="lines",
        line=dict(color="#38BDF8", width=2, dash="dash"),
        name="Bounding Box",
        hoverinfo="none"
    )


def render_viewer_page():
    """Render the full-screen 3D Viewer page."""

    summary = scan_all_reconstructions()

    st.markdown(
        """
<div class="eng-card" style="padding: 16px 22px; margin-bottom: 18px;">
  <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
    <div>
      <h2 style="margin: 0; font-size: 20px; font-weight: 700; color: #fff;">🌐 High-Fidelity 3D Viewport</h2>
      <p style="margin: 4px 0 0; font-size: 13px; color: #94A3B8;">
        Interactive 3D geometry engine · Point cloud inspection, surface mesh topology, and camera poses
      </p>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
      <span class="badge-pill badge-primary">WebGL Accelerated</span>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    # ── Top Control Bar ───────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1.5, 1.2, 1.2])

    with ctrl_col1:
        backend_choice = st.selectbox(
            "Reconstruction Backend:",
            ["VGGT-1B (Feed-Forward)", "COLMAP SfM (Incremental)", "MASt3R-SfM (SparseGA)"],
            index=0,
            key="v_backend_select"
        )
    with ctrl_col2:
        layer_mode = st.selectbox(
            "Geometry Layer:",
            ["3D Point Cloud", "3D Surface Mesh", "Combined (Mesh + Points)"],
            index=0,
            key="v_layer_select"
        )
    with ctrl_col3:
        color_palette = st.selectbox(
            "Color Palette:",
            ["Photorealistic RGB", "Height Map Elevation (Turbo)", "Solid Engineering Violet"],
            index=0,
            key="v_palette_select"
        )
    with ctrl_col4:
        point_sz = st.slider("Point Size:", min_value=1, max_value=8, value=2, step=1, key="v_pt_size")

    # Expandable advanced display toggles
    with st.expander("⚙️ Viewport & Camera Frustum Controls", expanded=False):
        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            show_bbox = st.checkbox("Show Bounding Box Frame", value=True, key="v_chk_bbox")
        with ec2:
            show_axes = st.checkbox("Show Coordinate Grid Axes", value=True, key="v_chk_axes")
        with ec3:
            pt_opacity = st.slider("Point Opacity:", 0.2, 1.0, 0.9, 0.05, key="v_pt_opacity")
        with ec4:
            max_points_lim = st.select_slider(
                "Max Display Points:",
                options=[20000, 50000, 80000, 120000, 200000],
                value=80000,
                key="v_max_pts"
            )

    # ── Resolve Model Files Based on Selected Backend ─────────────────────────
    pcd_path = None
    mesh_path = None

    if "VGGT" in backend_choice:
        pcd_path = find_artifact("outputs/vggt/vggt_point_cloud.ply") or find_artifact("outputs/vggt/point_cloud.ply")
        mesh_path = find_artifact("outputs/vggt/vggt_surface_mesh.ply")
        backend_name = "VGGT-1B"
    elif "MASt3R" in backend_choice:
        pcd_path = find_artifact("outputs/mast3r/point_cloud.ply")
        mesh_path = find_artifact("outputs/mast3r/mesh/mesh.ply")
        backend_name = "MASt3R-SfM"
    else:
        pcd_path = find_artifact("output/reconstruction.ply") or find_artifact("outputs/dense/filtered_pointcloud.ply")
        mesh_path = find_artifact("output/mesh.ply") or find_artifact("outputs/mesh/final_mesh.ply")
        backend_name = "COLMAP SfM"

    # ── Check Data Availability ───────────────────────────────────────────────
    if not pcd_path or not os.path.exists(pcd_path):
        st.info(f"ℹ️ No reconstruction artifact found for **{backend_choice}** yet. You can run this backend in **New Reconstruction**.")
        return

    # Load cached point cloud data
    pcd_data = load_cached_point_cloud(pcd_path, max_points=max_points_lim)
    mesh_data = load_cached_mesh(mesh_path, max_faces=50000) if (mesh_path and os.path.exists(mesh_path)) else None

    if not pcd_data:
        st.error(f"❌ Failed to parse 3D point cloud: {pcd_path}")
        return

    # ── Build Plotly 3D Figure ────────────────────────────────────────────────
    fig = go.Figure()
    traces_added = 0

    pts = pcd_data["points"]
    colors = pcd_data["colors"]

    # 1. Point Cloud Trace
    if "Point Cloud" in layer_mode or "Combined" in layer_mode:
        if "RGB" in color_palette and colors is not None and len(colors) == len(pts):
            c_vals = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in colors]
            c_kwargs = dict(color=c_vals)
        elif "Height" in color_palette:
            c_kwargs = dict(
                color=pts[:, 2],
                colorscale="Turbo",
                colorbar=dict(title=dict(text="Z Elevation (m)", font=dict(color="#94A3B8")), tickfont=dict(color="#94A3B8"), len=0.6, thickness=12)
            )
        else:
            c_kwargs = dict(color="#7C5CFC")

        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(
                    size=point_sz,
                    opacity=pt_opacity,
                    **c_kwargs
                ),
                name=f"{backend_name} Points ({pcd_data['display_count']:,})",
                hoverinfo="skip"
            )
        )
        traces_added += 1

    # 2. Surface Mesh Trace
    if ("Surface Mesh" in layer_mode or "Combined" in layer_mode) and mesh_data:
        m_verts = mesh_data["vertices"]
        m_tris = mesh_data["triangles"]
        m_cols = mesh_data["colors"]

        mesh_kwargs = dict()
        if "RGB" in color_palette and m_cols is not None and len(m_cols) == len(m_verts):
            mesh_kwargs["vertexcolor"] = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in m_cols]
        elif "Height" in color_palette:
            mesh_kwargs["intensity"] = m_verts[:, 2]
            mesh_kwargs["colorscale"] = "Turbo"
        else:
            mesh_kwargs["color"] = "#a855f7"

        fig.add_trace(
            go.Mesh3d(
                x=m_verts[:, 0],
                y=m_verts[:, 1],
                z=m_verts[:, 2],
                i=m_tris[:, 0],
                j=m_tris[:, 1],
                k=m_tris[:, 2],
                opacity=0.92,
                name=f"{backend_name} Mesh ({mesh_data['display_faces']:,} faces)",
                lighting=dict(ambient=0.45, diffuse=0.6, roughness=0.3, specular=0.2),
                **mesh_kwargs
            )
        )
        traces_added += 1

    # 3. Bounding Box Wireframe
    if show_bbox and pcd_data:
        fig.add_trace(_build_bounding_box_lines(pcd_data["min_bound"], pcd_data["max_bound"]))

    # ── Scene Layout Styling (Dark Professional Engineering Theme) ────────────
    axis_cfg = dict(
        showbackground=True,
        backgroundcolor="#0F1420" if show_axes else "#0B0F19",
        gridcolor="#2A3548" if show_axes else "rgba(0,0,0,0)",
        showgrid=show_axes,
        zerolinecolor="#3E4D67",
        titlefont=dict(color="#94A3B8", size=11),
        tickfont=dict(color="#64748B", size=10),
        visible=show_axes
    )

    fig.update_layout(
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19",
        height=740,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            bgcolor="rgba(16, 21, 34, 0.85)",
            bordercolor="#2A3548",
            borderwidth=1,
            font=dict(color="#F1F5F9", size=11),
            x=0.02,
            y=0.98
        ),
        scene=dict(
            xaxis=dict(title="X (m)", **axis_cfg),
            yaxis=dict(title="Y (m)", **axis_cfg),
            zaxis=dict(title="Z (m)", **axis_cfg),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.35, y=1.35, z=1.2),
                up=dict(x=0, y=0, z=1)
            )
        )
    )

    # Render Plotly canvas
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": True, "scrollZoom": True})

    # ── Geometry Metadata Telemetry Bar ───────────────────────────────────────
    st.markdown(
        f"""
<div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 18px; margin-top: 10px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; font-size: 12px; color: #94A3B8;">
  <div>
    Total 3D Points: <b style="color: #fff;">{pcd_data['raw_count']:,}</b> &nbsp;·&nbsp;
    Rendering: <b style="color: #20C6A0;">{pcd_data['display_count']:,}</b> &nbsp;·&nbsp;
    Mesh Faces: <b style="color: #fff;">{mesh_data['raw_faces'] if mesh_data else 0:,}</b>
  </div>
  <div>
    Extents [X, Y, Z]: <code style="color: #38BDF8;">[{pcd_data['extent'][0]:.2f}, {pcd_data['extent'][1]:.2f}, {pcd_data['extent'][2]:.2f}] m</code> &nbsp;·&nbsp;
    Format: <span class="badge-pill badge-primary">PLY ASCII</span>
  </div>
</div>
""",
        unsafe_allow_html=True
    )
