"""
ui/reconstruction_controls.py
-----------------------------
Launch and execution controls with Quality Presets, GPU telemetry,
and expandable Advanced Settings.
"""

import os
import json
import streamlit as st

from core.reconstruction_manager import (
    prepare_workspace,
    save_uploaded_images,
    launch_reconstruction,
    cancel_reconstruction
)
from core.gpu_detector import get_gpu_info
from backends.neuralangelo_backend import check_neuralangelo_status
from config.settings import QUALITY_PRESETS, get_active_config
from config.paths import CONFIG_FILE


def render_reconstruction_controls(
    uploaded_files,
    mode,
    temp_dir,
    log_file,
    status_file,
    result_file,
    image_paths_file
):
    if not uploaded_files:
        return

    if st.session_state.reconstruction_running:
        col_banner, col_cancel = st.columns([4, 1])
        with col_banner:
            st.markdown(
                """
<div class="status-box-success" style="border-color: rgba(168, 85, 247, 0.4); color: #d8b4fe; padding: 12px 18px; margin-bottom: 8px;">
  ⚡ &nbsp; <b>Reconstruction in progress...</b> COLMAP feature extractor, dense stereo, and ML mesh engine active.
</div>
""",
                unsafe_allow_html=True,
            )
        with col_cancel:
            if st.button("🛑 Cancel Run", type="secondary", use_container_width=True, help="Stop background reconstruction"):
                cancel_reconstruction(temp_dir)
                st.session_state.reconstruction_running = False
                st.warning("Reconstruction cancelled.")
                st.rerun()
        return

    # ── 1. Hardware & GPU Detection ──────────────────────────────────────────
    gpu = get_gpu_info()
    cuda_badge_color = "#4ade80" if gpu["cuda_available"] else "#facc15"
    cuda_text = "CUDA Available" if gpu["cuda_available"] else "CUDA Unavailable (CPU Mode)"

    st.markdown(
        f"""
<div class="figma-card" style="padding: 16px 24px; margin-bottom: 16px;">
  <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
    <div style="display: flex; align-items: center; gap: 10px;">
      <div style="font-size: 20px;">🖥️</div>
      <div>
        <div style="font-size: 14px; font-weight: 700; color: #fff;">Hardware Acceleration: {gpu['name']}</div>
        <div style="font-size: 12px; color: #94a3b8;">VRAM: <b>{gpu['vram_gb']} GB</b> &nbsp;·&nbsp; Driver: <b>{gpu['driver_version']}</b></div>
      </div>
    </div>
    <div class="nav-badge-pill" style="border: 1px solid rgba(255,255,255,0.1); background: rgba(255,255,255,0.03);">
      <span style="color: {cuda_badge_color};">●</span> &nbsp;{cuda_text}
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    # ── 2. Quality Preset Selector ───────────────────────────────────────────
    st.markdown(
        """
<div class="figma-card" style="padding: 20px 28px;">
  <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
    <div>
      <h4 style="margin: 0; font-size: 16px; font-weight: 700; color: #fff;">Reconstruction Quality & Pipeline Configuration</h4>
      <p style="margin: 4px 0 0; font-size: 12px; color: #94a3b8;">Select reconstruction density mode or configure fine-grained SfM parameters</p>
    </div>
  </div>
  <div style="margin-top: 16px;">
""",
        unsafe_allow_html=True,
    )

    preset_options = ["Standard", "High Density", "Fast", "Neuralangelo High Fidelity"]
    selected_preset = st.radio(
        "Reconstruction Quality Preset:",
        preset_options,
        index=1,  # Default to High Density
        horizontal=True,
        help="High Density mode uses full image resolution, dense SIFT extraction, and relaxed fusion to maximize valid geometry without fabricating points."
    )

    # Check Neuralangelo status if selected
    if selected_preset == "Neuralangelo High Fidelity":
        na_status = check_neuralangelo_status()
        if not na_status["cuda_ready"] or not na_status["available"]:
            st.warning(f"⚠️ {na_status['message']}")
            st.caption("COLMAP High Density mode will run as the primary SfM engine while preparing Neuralangelo pose exports.")

    base_config = get_active_config(selected_preset)

    # ── 3. Expandable Advanced Settings ──────────────────────────────────────
    with st.expander("⚙️ Advanced Reconstruction Settings (Fine-Tuning)", expanded=False):
        st.markdown(
            "<p style='color: #94a3b8; font-size: 12px;'>Adjust low-level COLMAP feature extraction, PatchMatch stereo, fusion tolerances, and meshing parameters.</p>",
            unsafe_allow_html=True
        )

        col_a, col_b = st.columns(2)
        with col_a:
            img_res = st.slider(
                "Image Resolution (max px)",
                min_value=1200,
                max_value=4096,
                value=int(base_config.get("image_max_size", 4096)),
                step=200,
                help="Preserve native resolution for maximum fine-detail feature extraction."
            )
            pm_radius = st.slider(
                "PatchMatch Window Radius",
                min_value=3,
                max_value=15,
                value=int(base_config.get("pm_window_radius", 9)),
                step=2,
                help="Larger window handles low-contrast and weakly textured surfaces."
            )
            geom_consistency = st.checkbox(
                "Enforce Geometric Consistency",
                value=bool(base_config.get("pm_geom_consistency", True)),
                help="Requires depth estimates to be mutually consistent across multiple camera views."
            )
            poisson_depth = st.slider(
                "Poisson Meshing Depth",
                min_value=6,
                max_value=11,
                value=int(base_config.get("poisson_depth", 9)),
                step=1,
                help="Higher depth captures sharper surface features."
            )

        with col_b:
            fusion_min_px = st.slider(
                "Stereo Fusion Minimum Consistency (images)",
                min_value=1,
                max_value=6,
                value=int(base_config.get("fusion_min_num_pixels", 2)),
                step=1,
                help="Lower value (2) retains thin structures and fringe points; higher value filters more aggressively."
            )
            reproj_thresh = st.slider(
                "Max Reprojection Error (pixels)",
                min_value=1.0,
                max_value=5.0,
                value=float(base_config.get("fusion_max_reproj_error", 2.5)),
                step=0.5,
                help="Maximum allowable reprojection discrepancy between overlapping views."
            )
            outlier_strength = st.slider(
                "Outlier Filter Strength (std dev)",
                min_value=1.0,
                max_value=3.0,
                value=float(base_config.get("outlier_std_ratio", 1.5)),
                step=0.1,
                help="Standard deviation threshold for statistical noise pruning."
            )
            mesh_trim = st.slider(
                "Mesh Density Trim (%)",
                min_value=0.0,
                max_value=25.0,
                value=float(base_config.get("mesh_density_trim", 0.08) * 100.0),
                step=1.0,
                help="Prunes outer low-density manifold boundary triangles."
            )
            sift_features = st.slider(
                "SIFT Max Features (per image)",
                min_value=4096,
                max_value=16384,
                value=int(base_config.get("sift_max_features", 10240)),
                step=1024,
                help="Optimized feature count (8192-12288) prevents CPU RAM exhaustion while preserving keypoint coverage."
            )
            matcher_choice = st.selectbox(
                "Feature Matcher Strategy",
                [
                    "ML-Ranked Candidate Pairs (Smart & Fast ~2-3 min)",
                    "Exhaustive (All 2,211 Pairs ~15-20 min on CPU)",
                    "Sequential (Adjacent Frames with Quadratic Overlap)"
                ],
                index=0 if base_config.get("matcher", "ml_ranked") == "ml_ranked" else (2 if base_config.get("matcher") == "sequential" else 1),
                help="ML-Ranked uses visual embeddings and graph connectivity to replace O(N^2) all-to-all matching with top-K candidate pairs."
            )

            ml_top_k = 8
            ml_min_pairs = 4
            ml_threshold = 0.0
            if "ML-Ranked" in matcher_choice:
                c_ml1, c_ml2 = st.columns(2)
                with c_ml1:
                    ml_top_k = st.slider(
                        "ML Top-K (neighbors per image)",
                        min_value=4,
                        max_value=20,
                        value=int(base_config.get("ml_top_k", 8)),
                        step=1,
                        help="Number of most visually similar candidate images to select for each view."
                    )
                with c_ml2:
                    ml_min_pairs = st.slider(
                        "ML Min Pairs per Image",
                        min_value=2,
                        max_value=8,
                        value=int(base_config.get("ml_min_pairs_per_image", 4)),
                        step=1,
                        help="Guaranteed minimum candidate edges per view to prevent isolated images."
                    )

    # ── 4. Launch Button & Config Persistence ────────────────────────────────
    start_clicked = st.button("🚀 Start 3D Reconstruction Pipeline", type="primary", use_container_width=True)

    if start_clicked:
        # Build combined active configuration
        active_config = dict(base_config)
        active_config["name"] = selected_preset
        active_config["image_max_size"] = img_res
        active_config["pm_max_image_size"] = img_res
        active_config["pm_window_radius"] = pm_radius
        active_config["pm_geom_consistency"] = geom_consistency
        active_config["fusion_min_num_pixels"] = fusion_min_px
        active_config["fusion_max_reproj_error"] = reproj_thresh
        active_config["outlier_std_ratio"] = outlier_strength
        active_config["poisson_depth"] = poisson_depth
        active_config["mesh_density_trim"] = mesh_trim / 100.0
        active_config["sift_max_features"] = sift_features
        if "ML-Ranked" in matcher_choice:
            active_config["matcher"] = "ml_ranked"
            active_config["ml_pair_ranking"] = True
            active_config["ml_top_k"] = ml_top_k
            active_config["ml_min_pairs_per_image"] = ml_min_pairs
            active_config["ml_similarity_threshold"] = ml_threshold
        elif "Sequential" in matcher_choice:
            active_config["matcher"] = "sequential"
            active_config["ml_pair_ranking"] = False
        else:
            active_config["matcher"] = "exhaustive"
            active_config["ml_pair_ranking"] = False
        active_config["use_gpu"] = gpu.get("cuda_available", False)

        # Ensure temp_dir exists and save config
        os.makedirs(temp_dir, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as cf:
            json.dump(active_config, cf, indent=2)

        prepare_workspace(
            temp_dir,
            log_file,
            status_file,
            result_file
        )

        save_uploaded_images(
            uploaded_files,
            mode,
            temp_dir,
            image_paths_file
        )

        launch_reconstruction(log_file=log_file)

        st.session_state.reconstruction_running = True
        st.session_state.reconstruction_done = False
        st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)