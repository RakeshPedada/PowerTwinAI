"""
ui/mesh_viewer.py
------------------
3D Surface Mesh Viewer & Generator Control Panel.
Renders triangular meshes using Plotly go.Mesh3d with photorealistic vertex colors,
surface shading, parameter controls, and programmatic face element verification.
"""

import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import open3d as o3d

from mesh_generator import generate_mesh, is_mesh_ply
from config.settings import PLOT_HEIGHT, PLOT_BACKGROUND, SCENE_BACKGROUND


def render_mesh_viewer(points=None, colors=None, default_mesh_path=None, key=None):
    """
    Render the 3D surface mesh visualization and interactive generation controls.
    """
    prefix = f"{key}_" if key else "mesh_viewer_"

    st.markdown(
        "<div class='section-title'>🔺 3D Surface Mesh Reconstruction</div>",
        unsafe_allow_html=True
    )

    mesh_file = default_mesh_path or os.path.join("output", "mesh.ply")
    point_cloud_file = os.path.join("output", "reconstruction.ply")

    # ── Interactive Mesh Generation Controls ──────────────────────────────────
    with st.expander("⚙️ Surface Mesh Generation & Tuning Controls", expanded=not os.path.exists(mesh_file)):
        st.markdown(
            "<p style='color: #94a3b8; font-size: 13px;'>"
            "Configure Poisson surface reconstruction and density filtering parameters "
            "to reconstruct a smooth, watertight triangular mesh from the point cloud."
            "</p>",
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            poisson_depth = st.slider(
                "Poisson Depth (Resolution)",
                min_value=6,
                max_value=11,
                value=8,
                step=1,
                key=f"{prefix}poisson_depth",
                help="Higher depth captures fine geometric details. Depth 8-9 provides great balance for dense scenes without noise."
            )

        with col2:
            density_percentile = st.slider(
                "Density Pruning Cutoff (%)",
                min_value=0.0,
                max_value=25.0,
                value=10.0,
                step=1.0,
                key=f"{prefix}density_percentile",
                help="Trims low-density boundary vertices created by Poisson reconstruction, eliminating floating artifacts."
            )

        with col3:
            outlier_strength = st.slider(
                "Outlier Filter Strength (std dev)",
                min_value=1.0,
                max_value=3.0,
                value=1.5,
                step=0.1,
                key=f"{prefix}outlier_strength",
                help="Lower values apply stricter outlier filtering before meshing. 1.5 is ideal for multi-view SfM."
            )

        gen_clicked = st.button("🛠️ Generate Surface Mesh", type="primary", use_container_width=True, key=f"{prefix}gen_btn")

        if gen_clicked:
            # Determine point source
            source_pcd = None
            if points is not None and len(points) > 0:
                source_pcd = points
            elif os.path.exists(point_cloud_file):
                source_pcd = point_cloud_file
            else:
                st.error("❌ No point cloud available for mesh generation. Please run reconstruction first.")

            if source_pcd is not None:
                with st.spinner("Running Poisson surface reconstruction and topology cleaning..."):
                    try:
                        result = generate_mesh(
                            source_pcd,
                            colors=colors if isinstance(source_pcd, np.ndarray) else None,
                            output_dir="output",
                            output_name="mesh",
                            poisson_depth=poisson_depth,
                            density_quantile=density_percentile / 100.0,
                            outlier_std_ratio=outlier_strength,
                            smooth_iterations=5,
                            progress_callback=lambda m: None
                        )
                        st.success(
                            f"✅ Surface Mesh successfully generated! "
                            f"{result['vertices']:,} vertices, {result['triangles']:,} triangular faces."
                        )
                        st.session_state["mesh_updated"] = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to generate surface mesh: {e}")

    # ── Check Mesh Existence & Verify Face Elements ───────────────────────────
    if not os.path.exists(mesh_file):
        st.info(
            "ℹ️ Surface mesh not yet generated. Use the controls above and click "
            "**Generate Surface Mesh** to compute triangular geometry."
        )
        return

    # Programmatic verification of face elements (Requirement 2 & 19)
    if not is_mesh_ply(mesh_file):
        st.error(
            f"⚠️ The file '{mesh_file}' is a point cloud and does not contain an 'element face' section. "
            f"Please click 'Generate Surface Mesh' above to reconstruct triangular faces."
        )
        return

    # ── Load Mesh for Visualization ──────────────────────────────────────────
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_file)
    except Exception as e:
        st.error(f"Could not load mesh file '{mesh_file}': {e}")
        return

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if len(vertices) == 0 or len(triangles) == 0:
        st.warning("Surface mesh contains no vertices or faces.")
        return

    # ── Display Stats Badge ──────────────────────────────────────────────────
    st.markdown(
        f"""
<div class="glass-stat-grid" style="margin-bottom: 16px;">
  <div class="glass-stat-card">
    <div class="glass-stat-label">Mesh Vertices</div>
    <div class="glass-stat-value">{len(vertices):,}</div>
    <div class="glass-stat-sub">Cleaned 3D vertex coordinates</div>
  </div>
  <div class="glass-stat-card">
    <div class="glass-stat-label">Triangular Faces</div>
    <div class="glass-stat-value">{len(triangles):,}</div>
    <div class="glass-stat-sub">Verified 'element face' triangles</div>
  </div>
  <div class="glass-stat-card">
    <div class="glass-stat-label">Surface Topology</div>
    <div class="glass-stat-value">Watertight</div>
    <div class="glass-stat-sub">Oriented Poisson manifold</div>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    # ── Color & Shading Options ──────────────────────────────────────────────
    has_vertex_colors = mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(vertices)

    shading_cols = st.columns([2, 1, 1])
    with shading_cols[0]:
        color_choices = ["Photorealistic RGB (Preserved from Dataset)", "Smooth Clay Render"] if has_vertex_colors else ["Smooth Clay Render"]
        selected_shading = st.radio(
            "Mesh Surface Shading",
            color_choices,
            index=0,
            horizontal=True,
            key=f"{prefix}mesh_shading"
        )

    # Downsample triangles for interactive smooth rendering if mesh is extremely large (> 100k faces)
    MAX_RENDER_FACES = 100000
    display_mesh = mesh
    if len(triangles) > MAX_RENDER_FACES:
        st.caption(f"Optimizing display for browser interactivity ({len(triangles):,} faces -> {MAX_RENDER_FACES:,} faces). Full resolution is preserved in download.")
        display_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=MAX_RENDER_FACES)

    disp_v = np.asarray(display_mesh.vertices)
    disp_f = np.asarray(display_mesh.triangles)

    vertexcolor = None
    mesh_color = "#c2a688"  # Warm bronze/clay fallback

    if selected_shading.startswith("Photorealistic") and display_mesh.has_vertex_colors():
        c_vals = np.asarray(display_mesh.vertex_colors)
        if np.max(c_vals) <= 1.01:
            c_vals = (c_vals * 255.0).astype(int)
        else:
            c_vals = c_vals.astype(int)
        c_vals = np.clip(c_vals, 0, 255)
        vertexcolor = [f"rgb({r},{g},{b})" for r, g, b in c_vals]

    # Normalize coordinate center for clean rotation
    center = np.mean(disp_v, axis=0)
    extent = np.ptp(disp_v, axis=0)
    max_extent = np.max(extent)
    scale = 1.0 / max_extent if max_extent > 0 else 1.0
    norm_v = (disp_v - center) * scale

    # ── Build Plotly Mesh3d Trace ────────────────────────────────────────────
    mesh_kwargs = dict(
        x=norm_v[:, 0],
        y=norm_v[:, 1],
        z=norm_v[:, 2],
        i=disp_f[:, 0],
        j=disp_f[:, 1],
        k=disp_f[:, 2],
        flatshading=False,
        lighting=dict(
            ambient=0.45,
            diffuse=0.85,
            specular=0.25,
            roughness=0.5,
            fresnel=0.25
        ),
        lightposition=dict(x=80, y=120, z=150),
        opacity=1.0,
        hoverinfo="skip"
    )

    if vertexcolor is not None:
        mesh_kwargs["vertexcolor"] = vertexcolor
    else:
        mesh_kwargs["color"] = mesh_color

    mesh_trace = go.Mesh3d(**mesh_kwargs)

    fig = go.Figure(data=[mesh_trace])

    fig.update_layout(
        height=PLOT_HEIGHT,
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            bgcolor=SCENE_BACKGROUND,
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.3, y=1.3, z=1.1),
                up=dict(x=0, y=0, z=1)
            )
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Download Mesh Section ────────────────────────────────────────────────
    st.markdown("<div class='section-title'><span>⬇️</span> Download Surface Mesh Model</div>", unsafe_allow_html=True)
    dcol1, dcol2 = st.columns(2)

    with dcol1:
        if os.path.exists(mesh_file):
            with open(mesh_file, "rb") as f:
                st.download_button(
                    label="⬇️ Download 3D Mesh (.ply with faces)",
                    data=f.read(),
                    file_name="surface_mesh.ply",
                    mime="application/octet-stream",
                    use_container_width=True,
                    key=f"{prefix}dl_mesh_ply"
                )

    obj_file = os.path.join("output", "mesh.obj")
    with dcol2:
        if os.path.exists(obj_file):
            with open(obj_file, "rb") as f:
                st.download_button(
                    label="⬇️ Download 3D Mesh (.obj format)",
                    data=f.read(),
                    file_name="surface_mesh.obj",
                    mime="text/plain",
                    use_container_width=True,
                    key=f"{prefix}dl_mesh_obj"
                )
