"""
reconstruction/v3_current/ui/vggt_section.py
---------------------------------------------
Streamlit UI section for the independent VGGT-based 3D reconstruction pipeline.
Supports hardware acceleration telemetry, configurable inference parameters,
interactive 3D point cloud and camera visualizers, and artifact downloads.
"""

import os
import glob
import time
import json
import streamlit as st
import numpy as np
import open3d as o3d

from vggt_pipeline import (
    run_vggt_reconstruction,
    detect_vggt_hardware,
    is_vggt_weights_cached,
    create_vggt_point_cloud_figure,
    create_vggt_mesh_figure,
    create_vggt_camera_trajectory_figure,
    load_vggt_summary,
    run_dataset_diagnostics
)
from ui.point_cloud_viewer import render_point_cloud


def _prepare_vggt_image_inputs(uploaded_files, mode) -> list[str]:
    """
    Extract or save image paths for VGGT without modifying original images.
    """
    if not uploaded_files:
        return []

    # If folder mode, uploaded_files is already a list of file paths
    if mode == "folder":
        if isinstance(uploaded_files, list) and all(isinstance(p, str) for p in uploaded_files):
            return [os.path.abspath(p) for p in uploaded_files if os.path.exists(p)]

    # If upload mode, save to a dedicated temporary folder for VGGT
    temp_dir = os.path.abspath("temp_session_vggt")
    os.makedirs(temp_dir, exist_ok=True)

    saved_paths = []
    for file in uploaded_files:
        filename = getattr(file, "name", str(file))
        save_path = os.path.join(temp_dir, filename)
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
        saved_paths.append(save_path)

    return sorted(saved_paths)


def render_vggt_section(uploaded_files, mode):
    """
    Render the complete VGGT 3D reconstruction controls and results panel.
    """
    st.markdown(
        """
<div style="background: linear-gradient(135deg, rgba(168, 85, 247, 0.12) 0%, rgba(59, 130, 246, 0.08) 100%);
            border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 12px; padding: 20px 24px; margin-bottom: 20px;">
  <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
    <div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 24px;">🧠</span>
        <h3 style="margin: 0; color: #fff; font-size: 20px; font-weight: 700;">VGGT-1B 3D Reconstruction Engine</h3>
      </div>
      <p style="margin: 6px 0 0; color: #94a3b8; font-size: 13px;">
        Feed-forward 3D reconstruction powered by Meta Research's <b>Visual Geometry Grounded Transformer</b>.
        Predicts dense camera poses, depth maps, and 3D point geometry in a unified forward pass.
      </p>
    </div>
    <div style="background: rgba(168, 85, 247, 0.2); border: 1px solid rgba(168, 85, 247, 0.4); border-radius: 20px; padding: 6px 14px; font-size: 12px; color: #d8b4fe; font-weight: 600;">
      Model: facebook/VGGT-1B
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    # ── 1. Hardware & Acceleration Status Card ──────────────────────────────
    hw = detect_vggt_hardware()
    cuda_badge_color = "#4ade80" if hw["cuda_available"] else "#facc15"
    cuda_text = "CUDA Accelerated (GPU)" if hw["cuda_available"] else "CPU Fallback Mode (CUDA Unavailable)"
    is_cached = is_vggt_weights_cached("facebook/VGGT-1B")
    cache_badge_color = "#4ade80" if is_cached else "#60a5fa"
    cache_text = "Weights Cached Locally (~4.68 GB)" if is_cached else "First Run: Will Download (~4.68 GB)"

    col_hw1, col_hw2 = st.columns([1, 1])
    with col_hw1:
        st.markdown(
            f"""
<div class="figma-card" style="padding: 14px 18px; margin-bottom: 16px;">
  <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Hardware Acceleration</div>
  <div style="font-size: 15px; font-weight: 700; color: #fff; margin-top: 4px;">🖥️ {hw['device_name']}</div>
  <div style="margin-top: 6px; font-size: 12px; color: #cbd5e1;">
    <span style="color: {cuda_badge_color};">●</span> &nbsp;<b>{cuda_text}</b> &nbsp;·&nbsp; Precision: <b>{hw['dtype_name']}</b>
  </div>
</div>
""",
            unsafe_allow_html=True
        )
    with col_hw2:
        st.markdown(
            f"""
<div class="figma-card" style="padding: 14px 18px; margin-bottom: 16px;">
  <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Model Weights Status</div>
  <div style="font-size: 15px; font-weight: 700; color: #fff; margin-top: 4px;">📦 facebook/VGGT-1B (1 Billion Parameters)</div>
  <div style="margin-top: 6px; font-size: 12px; color: #cbd5e1;">
    <span style="color: {cache_badge_color};">●</span> &nbsp;<b>{cache_text}</b>
  </div>
</div>
""",
            unsafe_allow_html=True
        )

    # ── 2. Configuration & Parameter Controls ───────────────────────────────
    with st.expander("⚙️ VGGT Reconstruction Parameters & Frame Controls", expanded=True):
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            max_frames = st.slider(
                "Max Sequence Frames to Process:",
                min_value=2,
                max_value=64,
                value=12 if not hw["cuda_available"] else 24,
                step=2,
                help="Number of multi-view frames passed coherently into VGGT transformer."
            )
            recon_mode = st.selectbox(
                "3D Reconstruction Branch:",
                options=["Depth-Map Unprojection (Recommended)", "Direct Point-Head Output"],
                help="Official VGGT docs show depth-map unprojection produces higher geometric precision."
            )
            reconstruction_mode_key = "depth" if "Depth" in recon_mode else "point_head"

            coherent_order = st.checkbox(
                "Overlap-Guided Coherent Sequencing",
                value=True,
                help="Orders views by visual overlap to maintain continuous orbital trajectory for transformer cross-view attention."
            )

        with col_c2:
            conf_threshold_val = st.slider(
                "Depth Confidence Cutoff Threshold:",
                min_value=1.0,
                max_value=10.0,
                value=1.8,
                step=0.1,
                help="Absolute confidence cutoff score (expp1). Higher values prune background, sky, and low-texture noise."
            )
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                remove_outliers = st.checkbox("Statistical Outlier Filter", value=True, help="Removes floating noise artifacts via Open3D.")
                use_ba = st.checkbox("Bundle Adjustment (VGGT+BA)", value=False, help="Refines camera poses and points with pycolmap/COLMAP BA.")
                generate_mesh_flag = st.checkbox("Generate 3D Surface Mesh (Poisson)", value=True, key="vggt_gen_mesh_chk", help="Applies Screened Poisson Surface Reconstruction to VGGT point cloud.")
            with col_b2:
                export_colmap = st.checkbox("Export COLMAP Format (.bin & .txt)", value=True, help="Exports cameras.bin, images.bin, and points3D.bin.")
                poisson_depth = st.slider("Poisson Octree Depth", min_value=6, max_value=11, value=8, step=1, key="vggt_poisson_depth_sld", help="Controls surface mesh detail. Depth 8 is balanced; 9-10 captures finer geometry.")

    # ── 3. Execution Trigger ────────────────────────────────────────────────
    image_paths = _prepare_vggt_image_inputs(uploaded_files, mode)
    if not image_paths:
        st.info("ℹ️ Please upload images or specify a valid local dataset directory above to run VGGT reconstruction.")
        return

    st.markdown(
        f"<p style='color: #cbd5e1; font-size: 13px;'>📷 Ready to process <b>{len(image_paths)}</b> input images with VGGT-1B.</p>",
        unsafe_allow_html=True
    )

    vggt_output_dir = os.path.abspath(os.path.join("outputs", "vggt"))

    if "vggt_running" not in st.session_state:
        st.session_state.vggt_running = False
    if "vggt_results" not in st.session_state:
        st.session_state.vggt_results = None

    if st.button("🚀 Run VGGT Reconstruction", type="primary", width="stretch", disabled=st.session_state.vggt_running):
        st.session_state.vggt_running = True
        progress_bar = st.progress(0.0)
        status_box = st.empty()
        log_lines = []

        def update_log(msg):
            log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            status_box.markdown(
                f"""
<div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(168, 85, 247, 0.4); border-radius: 8px; padding: 12px 16px; font-family: monospace; font-size: 12px; color: #a5f3fc; max-height: 180px; overflow-y: auto;">
  {'<br>'.join(log_lines[-6:])}
</div>
""",
                unsafe_allow_html=True
            )

        try:
            progress_bar.progress(0.08)
            update_log("Validating input image files...")
            time.sleep(0.1)

            progress_bar.progress(0.20)
            update_log("Analyzing dataset image quality and visual overlap...")
            diag = run_dataset_diagnostics(image_paths)
            update_log(diag.get("summary", "Diagnostics complete."))

            progress_bar.progress(0.35)
            update_log("Initializing VGGT-1B feed-forward engine...")

            result = run_vggt_reconstruction(
                image_paths=image_paths,
                output_dir=vggt_output_dir,
                reconstruction_mode=reconstruction_mode_key,
                conf_threshold_val=float(conf_threshold_val),
                max_frames=int(max_frames),
                stride=1,
                coherent_order=bool(coherent_order),
                remove_outliers=bool(remove_outliers),
                export_colmap=bool(export_colmap),
                use_ba=bool(use_ba),
                generate_mesh_flag=bool(generate_mesh_flag),
                poisson_depth=int(poisson_depth),
                progress_callback=update_log
            )

            progress_bar.progress(1.0)
            st.session_state.vggt_results = result
            st.session_state.vggt_running = False
            st.success(f"🎉 VGGT 3D Reconstruction completed successfully in {result['runtime']:.2f} seconds!")
            time.sleep(0.5)
            st.rerun()

        except Exception as e:
            st.session_state.vggt_running = False
            progress_bar.empty()
            st.error(f"❌ VGGT Reconstruction failed: {e}")
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                st.warning("💡 **Memory Hint**: Try reducing the 'Max Sequence Frames' slider to fit within your system RAM/VRAM.")

    # ── 4. Display Existing or Newly Generated VGGT Results ──────────────────
    existing_vggt = load_vggt_summary(vggt_output_dir)
    if existing_vggt is not None and len(existing_vggt["points"]) > 0:
        meta = existing_vggt.get("metadata", {})
        cam_val = meta.get("camera_validation", {})

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 30px 0 20px;'>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='section-title'>📊 VGGT Reconstruction Analytics & 3D Visualization</div>",
            unsafe_allow_html=True
        )

        mesh_meta = meta.get("mesh", {})
        mesh_verts = mesh_meta.get("vertex_count", 0)
        mesh_faces = mesh_meta.get("face_count", 0)

        col_m1, col_m2, col_m3, col_m4, col_m5, col_m6, col_m7 = st.columns(7)
        col_m1.metric("3D Points", f"{existing_vggt['num_points']:,}")
        col_m2.metric("Mesh Vertices", f"{mesh_verts:,}" if mesh_verts > 0 else "N/A")
        col_m3.metric("Mesh Faces", f"{mesh_faces:,}" if mesh_faces > 0 else "N/A")
        col_m4.metric("Camera Views", f"{existing_vggt['registered_images']}")
        col_m5.metric("Total Runtime", f"{existing_vggt['runtime_seconds']:.1f} s")
        col_m6.metric("Bounding Volume", f"{meta.get('bounding_box_volume', 0.0):.4f} m³")
        col_m7.metric("Point Density", f"{meta.get('point_density_per_m3', 0):,} pts/m³")

        # Camera pose validation telemetry bar
        if cam_val:
            is_valid = cam_val.get("is_camera_pose_valid", True)
            val_icon = "✅" if is_valid else "⚠️"
            val_text = "Camera Poses Validated (Continuous Orbit · No Interior/Outlier Cameras)" if is_valid else "Camera Warnings Detected"
            st.caption(
                f"{val_icon} **{val_text}** | Mean Camera Step: `{cam_val.get('mean_camera_step_meters', 0.0)} m` | "
                f"Orbit Radius: `{cam_val.get('median_orbit_radius_meters', 0.0)} m` | "
                f"Interior Cams: `{cam_val.get('cameras_inside_object', 0)}` | Outlier Cams: `{cam_val.get('outlier_cameras', 0)}`"
            )

        # Tabs for Visualizers
        tab_vggt_pc, tab_vggt_mesh, tab_vggt_cam, tab_depth, tab_diag, tab_download = st.tabs([
            "🌐 3D Point Cloud",
            "📐 3D Surface Mesh",
            "📷 Cameras & Trajectory",
            "🗺️ Depth & Confidence Maps",
            "🔍 Dataset Diagnostics",
            "💾 Download Artifacts"
        ])

        with tab_vggt_pc:
            col_v1, col_v2 = st.columns([3, 1])
            with col_v2:
                color_mode = st.radio("Color Palette:", ["RGB Colors", "Height Elevation (Turbo)"], key="vggt_color_mode_radio")
                c_mode = "rgb" if "RGB" in color_mode else "turbo"
                st.info(f"**Points**: {existing_vggt['num_points']:,}\n\n**Format**: PLY (Point Cloud)")
            with col_v1:
                fig_pc = create_vggt_point_cloud_figure(
                    existing_vggt["points"],
                    existing_vggt["colors"],
                    title="🌐 VGGT Dense 3D Point Cloud",
                    color_mode=c_mode
                )
                st.plotly_chart(fig_pc, width="stretch")

        with tab_vggt_mesh:
            mesh_candidates = [
                os.path.join(vggt_output_dir, "vggt_surface_mesh.ply"),
                os.path.join(vggt_output_dir, "mesh", "vggt_surface_mesh.ply"),
                os.path.join(vggt_output_dir, "mesh", "mesh.ply"),
            ]
            mesh_ply_file = next((p for p in mesh_candidates if os.path.exists(p)), None)

            if mesh_ply_file:
                col_vm1, col_vm2 = st.columns([3, 1])
                with col_vm2:
                    mesh_color_mode = st.radio("Mesh Material:", ["RGB Vertex Colors", "Solid Lavender"], key="vggt_mesh_mat_radio")
                    mc_mode = "rgb" if "RGB" in mesh_color_mode else "solid"
                    st.info(
                        f"**Vertices**: {mesh_verts:,}\n\n"
                        f"**Faces**: {mesh_faces:,}\n\n"
                        f"**Algorithm**: Screened Poisson\n\n"
                        f"**Format**: PLY Triangular Mesh"
                    )
                with col_vm1:
                    fig_mesh = create_vggt_mesh_figure(
                        mesh_ply_file,
                        title="📐 VGGT 3D Surface Mesh (Screened Poisson)",
                        color_mode=mc_mode,
                        default_color="#a855f7"
                    )
                    st.plotly_chart(fig_mesh, width="stretch")
            else:
                st.info("ℹ️ No surface mesh found for this reconstruction yet. Re-run reconstruction with 'Generate 3D Surface Mesh' checked.")

        with tab_vggt_cam:
            fig_cam = create_vggt_camera_trajectory_figure(
                existing_vggt["cameras"],
                existing_vggt["points"],
                existing_vggt["colors"],
                title="📷 VGGT Predicted Camera Poses & Point Cloud (Unified Metric Frame)"
            )
            st.plotly_chart(fig_cam, width="stretch")

        with tab_depth:
            st.markdown("<p style='color: #94a3b8; font-size: 13px;'>Predicted dense multi-view depth maps and confidence maps (Inferno & Viridis colormaps).</p>", unsafe_allow_html=True)
            preview_files = sorted(glob.glob(os.path.join(vggt_output_dir, "depth", "depth_preview_*.png")))
            conf_files = sorted(glob.glob(os.path.join(vggt_output_dir, "debug_visuals", "confidence_map_*.png")))
            if preview_files:
                cols = st.columns(len(preview_files))
                for idx, p in enumerate(preview_files):
                    with cols[idx]:
                        st.image(p, caption=f"Depth Map {idx+1}", width="stretch")
            if conf_files:
                cols_c = st.columns(len(conf_files))
                for idx, p in enumerate(conf_files):
                    with cols_c[idx]:
                        st.image(p, caption=f"Confidence Map {idx+1}", width="stretch")

        with tab_diag:
            st.markdown("<p style='color: #94a3b8; font-size: 13px;'>Dataset diagnostic analysis across input views.</p>", unsafe_allow_html=True)
            montage_file = os.path.join(vggt_output_dir, "debug_visuals", "input_montage.png")
            if os.path.exists(montage_file):
                st.image(montage_file, caption="Coherent View Sequence Montage", width="stretch")
            st.json({
                "camera_validation": meta.get("camera_validation", {}),
                "depth_vs_point_head_comparison": meta.get("depth_unprojection_vs_point_head", {}),
                "bundle_adjustment_report": meta.get("bundle_adjustment_report", {})
            })

        with tab_download:
            st.markdown("<p style='color: #94a3b8; font-size: 13px;'>Exported 3D point clouds, surface meshes, and camera models.</p>", unsafe_allow_html=True)
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                # Point Cloud PLY
                pc_candidates = [
                    os.path.join(vggt_output_dir, "vggt_point_cloud.ply"),
                    os.path.join(vggt_output_dir, "point_cloud.ply"),
                ]
                pc_file = next((p for p in pc_candidates if os.path.exists(p)), None)
                if pc_file:
                    with open(pc_file, "rb") as f:
                        st.download_button(
                            label="⬇️ Download VGGT Point Cloud (vggt_point_cloud.ply)",
                            data=f.read(),
                            file_name="vggt_point_cloud.ply",
                            mime="application/octet-stream",
                            width="stretch"
                        )
                
                # Metadata
                meta_f = os.path.join(vggt_output_dir, "metadata.json")
                if os.path.exists(meta_f):
                    with open(meta_f, "rb") as f:
                        st.download_button(
                            label="⬇️ Download VGGT Metadata (vggt_metadata.json)",
                            data=f.read(),
                            file_name="vggt_metadata.json",
                            mime="application/json",
                            width="stretch"
                        )

            with col_d2:
                # Surface Mesh PLY
                mesh_candidates = [
                    os.path.join(vggt_output_dir, "vggt_surface_mesh.ply"),
                    os.path.join(vggt_output_dir, "mesh", "vggt_surface_mesh.ply"),
                    os.path.join(vggt_output_dir, "mesh", "mesh.ply"),
                ]
                mesh_file = next((p for p in mesh_candidates if os.path.exists(p)), None)
                if mesh_file:
                    with open(mesh_file, "rb") as f:
                        st.download_button(
                            label="⬇️ Download VGGT Surface Mesh (vggt_surface_mesh.ply)",
                            data=f.read(),
                            file_name="vggt_surface_mesh.ply",
                            mime="application/octet-stream",
                            width="stretch"
                        )
                # OBJ mesh if available
                obj_file = os.path.join(vggt_output_dir, "mesh", "vggt_surface_mesh.obj")
                if not os.path.exists(obj_file):
                    obj_file = os.path.join(vggt_output_dir, "mesh", "mesh.obj")
                if os.path.exists(obj_file):
                    with open(obj_file, "rb") as f:
                        st.download_button(
                            label="⬇️ Download VGGT Surface Mesh (vggt_surface_mesh.obj)",
                            data=f.read(),
                            file_name="vggt_surface_mesh.obj",
                            mime="application/octet-stream",
                            width="stretch"
                        )
