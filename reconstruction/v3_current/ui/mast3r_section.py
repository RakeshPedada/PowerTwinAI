"""
reconstruction/v3_current/ui/mast3r_section.py
----------------------------------------------
Streamlit UI section for official Naver MASt3R-SfM 3D reconstruction backend.
Provides controls for:
- Image resolution (512, 384, 256)
- Pairing strategies: Mode A (MASt3R Retrieval), Mode B (ML Overlap), Mode C (Hybrid)
- Minimum pair matches cutoff and confidence thresholds
- Outlier filtering & voxel downsampling
- Optional COLMAP Bundle Adjustment (MASt3R-SfM + BA)
- Rich telemetry metrics cards, Plotly point cloud, and camera trajectory visualizers
"""

import os
import glob
import json
import time
import numpy as np
import open3d as o3d
import streamlit as st

from reconstruction.mast3r import (
    run_mast3r_reconstruction,
    detect_mast3r_hardware,
    is_mast3r_weights_cached,
    create_mast3r_point_cloud_figure,
    create_mast3r_mesh_figure,
    create_mast3r_camera_pointcloud_combined_figure,
    DEFAULT_MAST3R_MODEL
)


def _collect_image_paths(uploaded_files, mode) -> list:
    """Resolve input image paths from upload or dataset folder."""
    if mode == "Dataset Folder":
        colmap_img_dir = os.path.join("colmap_workspace", "images")
        if os.path.exists(colmap_img_dir):
            patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
            paths = []
            for pat in patterns:
                paths.extend(glob.glob(os.path.join(colmap_img_dir, pat)))
            if paths:
                return sorted(list(set(paths)))

        temp_orig = os.path.join("temp_session_c861f0ac", "original")
        if os.path.exists(temp_orig):
            patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
            paths = []
            for pat in patterns:
                paths.extend(glob.glob(os.path.join(temp_orig, pat)))
            if paths:
                return sorted(list(set(paths)))

    if not uploaded_files:
        return []

    temp_save_dir = os.path.join("temp_session", "mast3r_inputs")
    os.makedirs(temp_save_dir, exist_ok=True)
    saved_paths = []
    for file in uploaded_files:
        save_path = os.path.join(temp_save_dir, file.name)
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
        saved_paths.append(save_path)

    return sorted(saved_paths)


def render_mast3r_section(uploaded_files, mode):
    """
    Render MASt3R-SfM 3D reconstruction controls, execution panel, and results.
    """
    st.markdown(
        """
<div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(59, 130, 246, 0.08) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 20px 24px; margin-bottom: 20px;">
  <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
    <div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 24px;">🔬</span>
        <h3 style="margin: 0; color: #fff; font-size: 20px; font-weight: 700;">MASt3R-SfM 3D Reconstruction Engine</h3>
      </div>
      <p style="margin: 6px 0 0; color: #94a3b8; font-size: 13px;">
        Structure-from-Motion powered by Naver's <b>Grounding Image Matching in 3D with MASt3R</b>.
        Integrates retrieval pair selection, dense local feature matching, and sparse global alignment (SparseGA).
      </p>
    </div>
    <div style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 20px; padding: 6px 14px; font-size: 12px; color: #6ee7b7; font-weight: 600;">
      Model: naver/MASt3R_ViTLarge_512
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    # 1. Hardware Status Card
    hw = detect_mast3r_hardware()
    cuda_badge_color = "#4ade80" if hw["cuda_available"] else "#facc15"
    cuda_text = "CUDA Accelerated (GPU)" if hw["cuda_available"] else "CPU Fallback Mode (CUDA Unavailable)"
    is_cached = is_mast3r_weights_cached(DEFAULT_MAST3R_MODEL)
    cache_badge_color = "#4ade80" if is_cached else "#60a5fa"

    st.markdown(
        f"""
<div style="background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 14px 18px; margin-bottom: 20px;">
  <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
    <div>
      <span style="font-size: 13px; color: #94a3b8;">Compute Device:</span>
      <span style="font-size: 13px; font-weight: 600; color: {cuda_badge_color}; margin-left: 6px;">● {cuda_text}</span>
      <span style="color: #64748b; font-size: 12px; margin-left: 8px;">({hw['device_name']}{f' · VRAM: {hw["gpu_memory_gb"]} GB' if hw['cuda_available'] else ''})</span>
    </div>
    <div>
      <span style="font-size: 13px; color: #94a3b8;">Weights Cache:</span>
      <span style="font-size: 13px; font-weight: 600; color: {cache_badge_color}; margin-left: 6px;">{'✅ Cached Locally' if is_cached else '⬇️ Auto-Download'}</span>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    if hw["hardware_warning"]:
        st.warning(f"⚠️ **Performance Notice**: {hw['hardware_warning']}")

    # 2. Configuration & Parameter Controls
    with st.expander("⚙️ MASt3R-SfM Parameters & Pairing Controls", expanded=True):
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            image_size = st.select_slider(
                "Image Resolution:",
                options=[256, 384, 512],
                value=hw["recommended_image_size"],
                help="Resolution passed to MASt3R ViT encoder. 512 is official full-resolution. 384/256 saves memory on CPU."
            )

            pairing_strategy = st.selectbox(
                "Pairing Strategy:",
                options=[
                    "Mode C: Hybrid (MASt3R Retrieval + ML Overlap)",
                    "Mode A: Official MASt3R Retrieval (ASMK)",
                    "Mode B: Existing PowerTwinAI ML Overlap"
                ],
                index=0,
                help="Mode A uses ASMK retrieval. Mode B uses existing PowerTwinAI ML cosine embeddings. Mode C combines both."
            )

            max_frames = st.slider(
                "Max Images to Reconstruct:",
                min_value=4,
                max_value=67,
                value=8 if not hw["cuda_available"] else 30,
                step=2,
                help="Subset of images for MASt3R reconstruction."
            )

        with col_c2:
            min_matches = st.slider(
                "Minimum Pair Matches Threshold:",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                help="Candidate pairs with fewer reciprocal matches are pruned before global alignment."
            )

            conf_threshold_val = st.slider(
                "Point Confidence Cutoff:",
                min_value=0.5,
                max_value=10.0,
                value=1.5,
                step=0.25,
                help="Minimum confidence required to retain a 3D point in the global point cloud."
            )

            voxel_size = st.slider(
                "Voxel Downsampling Size (m):",
                min_value=0.0,
                max_value=0.05,
                value=0.0,
                step=0.005,
                help="Voxel downsampling grid size in meters (0.0 = disabled)."
            )

            poisson_depth = st.slider(
                "Poisson Octree Depth:",
                min_value=6,
                max_value=11,
                value=8,
                step=1,
                help="Spatial resolution of Poisson surface reconstruction (default: 8)."
            )

            density_quantile = st.slider(
                "Poisson Density Threshold:",
                min_value=0.00,
                max_value=0.10,
                value=0.01,
                step=0.005,
                help="Removes low-support boundary extrapolation vertices."
            )

            normal_radius = st.slider(
                "Normal Estimation Radius (m):",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01,
                help="Neighborhood radius for normal estimation."
            )

            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                remove_outliers = st.checkbox("Statistical Outlier Filter", value=True, help="Removes floating noise artifacts via Open3D.")
            with col_b2:
                generate_mesh_cb = st.checkbox("Generate Real Triangle Mesh", value=True, help="Computes Poisson 3D triangle surface mesh.")
            with col_b3:
                use_ba = st.checkbox("Bundle Adjustment (MASt3R+BA)", value=False, help="Refines camera poses and points with PyCOLMAP BA.")

    # 3. Execution Trigger
    image_paths = _collect_image_paths(uploaded_files, mode)
    num_found = len(image_paths)

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        run_clicked = st.button("🚀 Run MASt3R-SfM Reconstruction", type="primary", width="stretch")

    with col_info:
        st.caption(f"Input Dataset: **{num_found} images** detected. Ready for multi-stage MASt3R-SfM pipeline.")

    output_dir = os.path.join("outputs", "mast3r")

    if run_clicked:
        if num_found < 2:
            st.error("Please provide at least 2 images to run MASt3R-SfM.")
            return

        progress_container = st.empty()
        log_lines = []

        def update_progress(msg: str):
            log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            progress_container.info(f"**MASt3R-SfM Progress**: {msg}")

        pairing_mode_clean = "Mode C (Hybrid)" if "Mode C" in pairing_strategy else ("Mode A (Retrieval)" if "Mode A" in pairing_strategy else "Mode B (ML Overlap)")

        with st.spinner("Executing official MASt3R-SfM pipeline..."):
            try:
                res = run_mast3r_reconstruction(
                    image_paths=image_paths,
                    output_dir=output_dir,
                    pairing_mode=pairing_mode_clean,
                    image_size=image_size,
                    min_matches=min_matches,
                    min_conf_thr=conf_threshold_val,
                    voxel_size=voxel_size if voxel_size > 0.0 else None,
                    remove_outliers=remove_outliers,
                    use_ba=use_ba,
                    generate_mesh_flag=generate_mesh_cb,
                    poisson_depth=poisson_depth,
                    density_quantile=density_quantile,
                    max_frames=max_frames,
                    progress_callback=update_progress
                )
                progress_container.success("✅ MASt3R-SfM Reconstruction Completed Successfully!")
                st.session_state["mast3r_latest_result"] = res
            except Exception as e:
                progress_container.error(f"❌ MASt3R-SfM failed: {e}")
                logger.exception("MASt3R reconstruction failure")
                return

    # 4. Display Results if available
    meta_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            mesh_meta = meta.get("mesh", {})
            has_mesh = mesh_meta.get("has_faces", False) and mesh_meta.get("triangles", 0) > 0

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            st.markdown("### 📊 MASt3R-SfM Reconstruction Telemetry")

            # Metrics row 1: Alignment & Registration
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Registered Images", f"{meta['registered_images']} / {meta['total_input_images']}")
            m2.metric("Candidate Pairs", f"{meta['candidate_pairs_count']:,}", f"-{meta['pair_reduction_percentage']}%")
            m3.metric("Accepted Pairs", f"{meta['accepted_pairs_count']:,}")
            m4.metric("Reconstructed Points", f"{meta['filtered_points']:,}")

            # Metrics row 2: Mesh & Hardware Telemetry (Requirement 12)
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Mesh Vertices", f"{mesh_meta.get('vertices', 0):,}")
            s2.metric("Mesh Triangles", f"{mesh_meta.get('triangles', 0):,}")
            s3.metric("Total Runtime", f"{meta['timing']['total_pipeline_time_seconds']}s")
            s4.metric("Device / Memory", f"{meta.get('device', 'cpu').upper()} · {hw['gpu_memory_gb']} GB")

            # Metrics row 3: Geometry & Trajectory
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Bounding Box Volume", f"{meta['bounding_box_volume_m3']:.3f} m³")
            g2.metric("Point Density", f"{meta['point_density_per_m3']:,} pts/m³")
            g3.metric("Camera Trajectory", meta["camera_trajectory"]["overall_camera_validity"])
            g4.metric("Bundle Adjustment", "Applied" if meta["bundle_adjustment"]["enabled"] else "Disabled")

            # Direct Download Buttons (Requirement 12)
            ply_file = meta["artifacts"]["ply_path"]
            mesh_file = meta["artifacts"].get("mesh_ply", os.path.join(output_dir, "mesh", "mesh.ply"))
            if not os.path.exists(mesh_file):
                mesh_file = os.path.join(output_dir, "mesh.ply")

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                if os.path.exists(ply_file):
                    with open(ply_file, "rb") as pf:
                        st.download_button(
                            "📥 Download Point Cloud (.ply)",
                            data=pf.read(),
                            file_name="mast3r_pointcloud.ply",
                            mime="application/octet-stream",
                            width="stretch"
                        )
            with col_d2:
                if os.path.exists(mesh_file):
                    with open(mesh_file, "rb") as mf:
                        st.download_button(
                            "📥 Download Triangle Mesh (.ply)",
                            data=mf.read(),
                            file_name="mast3r_mesh.ply",
                            mime="application/octet-stream",
                            width="stretch"
                        )

            # Interactive Visualizations (Requirement 13)
            tab_pcd, tab_mesh, tab_previews, tab_pairs = st.tabs([
                "🌐 3D Point Cloud & Trajectory",
                "📐 Real 3D Triangular Mesh",
                "🔬 Depth & Confidence Maps",
                "📋 Pair Matching Diagnostics"
            ])

            with tab_pcd:
                cameras_file = meta["artifacts"]["cameras_json"]
                if os.path.exists(ply_file) and os.path.exists(cameras_file):
                    pcd = o3d.io.read_point_cloud(ply_file)
                    pts = np.asarray(pcd.points)
                    cols = np.asarray(pcd.colors) * 255.0 if pcd.has_colors() else None
                    with open(cameras_file, "r", encoding="utf-8") as f:
                        cams = json.load(f)

                    fig_pcd = create_mast3r_camera_pointcloud_combined_figure(
                        points=pts,
                        colors=cols,
                        cameras_list=cams,
                        title=f"📷 MASt3R-SfM 3D Scene ({len(pts):,} Points · {len(cams)} Cameras)"
                    )
                    st.plotly_chart(fig_pcd, width="stretch")

            with tab_mesh:
                if os.path.exists(mesh_file):
                    fig_mesh = create_mast3r_mesh_figure(
                        mesh_file,
                        title=f"📐 MASt3R Real Triangle Surface Mesh ({mesh_meta.get('triangles', 0):,} Faces · Poisson Depth {mesh_meta.get('method', 'Poisson')})"
                    )
                    st.plotly_chart(fig_mesh, width="stretch")
                else:
                    st.info("Triangle mesh not yet generated. Check 'Generate Real Triangle Mesh' and re-run.")

            with tab_previews:
                st.markdown("#### 🔬 Metric Depth and Local Descriptor Confidence")
                debug_dir = meta["artifacts"]["debug_visuals_dir"]
                c_p1, c_p2 = st.columns(2)
                with c_p1:
                    depth_preview = os.path.join(debug_dir, "depth_map_00.png")
                    if os.path.exists(depth_preview):
                        st.image(depth_preview, caption="MASt3R Estimated Metric Depth Map (Inferno)", width="stretch")
                with c_p2:
                    conf_preview = os.path.join(debug_dir, "confidence_map_00.png")
                    if os.path.exists(conf_preview):
                        st.image(conf_preview, caption="MASt3R Feature Matching Confidence Map (Viridis)", width="stretch")

            with tab_pairs:
                st.markdown("#### 📋 Candidate and Accepted Pair Matching Telemetry")
                matches_file = os.path.join(output_dir, "matches", "matching_reports.json")
                if os.path.exists(matches_file):
                    with open(matches_file, "r", encoding="utf-8") as f:
                        reports = json.load(f)
                    st.dataframe(reports, width="stretch")

        except Exception as e:
            st.info(f"Could not load previous MASt3R telemetry: {e}")
