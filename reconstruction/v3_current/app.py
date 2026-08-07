import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd 
import subprocess
import json
import os
import glob
from streamlit_autorefresh import st_autorefresh
from dense_reconstruction import DenseReconstructor
from ui.styles import load_styles
from ui.header import show_header
from ui.input_section import render_input_section
from ui.dataset_preview import render_dataset_preview
from ui.reconstruction_controls import render_reconstruction_controls
from ui.progress_panel import render_progress_panel
from ui.results_dashboard import render_results_dashboard
from core.session_manager import initialize_session
from ui.point_cloud_viewer import render_point_cloud


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Advanced SfM Reconstruction Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# LOAD UI STYLES
# =========================================================

load_styles ()

# =========================================================
# PATHS
# =========================================================

from config.paths import *
from config.settings import *

# =========================================================
# SESSION STATE
# =========================================================

initialize_session()



# =========================================================
# HEADER
# =========================================================

show_header()


# =========================================================
# UPLOAD
# =========================================================

uploaded_files, mode = render_input_section()


# =========================================================
# IMAGE PREVIEW
# =========================================================

render_dataset_preview(uploaded_files, mode)


# =========================================================
# START BUTTON
# =========================================================

render_reconstruction_controls(
    uploaded_files,
    mode,
    TEMP_DIR,
    LOG_FILE,
    STATUS_FILE,
    RESULT_FILE,
    IMAGE_PATHS_FILE
)
# =========================================================
# LIVE LOGS
# =========================================================

if render_progress_panel(
    LOG_FILE,
    STATUS_FILE
):
    st.rerun()

# =========================================================
# RESULTS
# =========================================================
results = render_results_dashboard(
    RESULT_FILE
)

if results:

    (
        points,
        cameras,
        analytics,
        processing_time,
        colmap_time,
        reconstruction_time,
        total_pipeline_time
    ) = results
    # =====================================================
    # POINT CLOUD VISUALIZATION
    # =====================================================

    render_point_cloud(points)

    # =====================================================
    # CAMERA TRAJECTORY
    # =====================================================

    st.markdown(
        "<div class='section-title'>📷 Camera Trajectory</div>",
        unsafe_allow_html=True
    )

    cameras = np.asarray(
        cameras,
        dtype=np.float64
    )

    if (
        cameras.ndim == 2
        and cameras.shape[1] == 3
        and len(cameras) > 0
    ):

        fig_cam = go.Figure()

        fig_cam.add_trace(
            go.Scatter3d(

                x=cameras[:, 0],
                y=cameras[:, 1],
                z=cameras[:, 2],

                mode="lines+markers",

                marker=dict(
                    size=6,
                    color="#00ff99"
                ),

                line=dict(
                    width=4,
                    color="#00ffaa"
                )
            )
        )

        fig_cam.update_layout(

            height=650,

            paper_bgcolor="#020617",

            scene=dict(
                bgcolor="#010409",
                aspectmode="data"
            ),

            font=dict(
                family="Inter",
                size=16,
                color="white"
            ),

            margin=dict(
                l=0,
                r=0,
                t=20,
                b=0
            )
        )

        st.plotly_chart(
            fig_cam,
            use_container_width=True
        )

    else:

        st.warning(
            "No valid camera trajectory available."
        )

    # =====================================================
    # PAIR STATISTICS
    # =====================================================

    pair_log_path = os.path.join(
        "output",
        "pair_logs.csv"
    )

    st.markdown(
        "<div class='section-title'>📋 Pair Statistics</div>",
        unsafe_allow_html=True
    )

    if os.path.exists(pair_log_path):

        try:

            pair_logs = pd.read_csv(
                pair_log_path
            )

        except pd.errors.EmptyDataError:

            pair_logs = pd.DataFrame()

    else:

        pair_logs = pd.DataFrame()

    if not pair_logs.empty:

        st.dataframe(
            pair_logs,
            use_container_width=True
        )

        # ================================================
        # CONFIDENCE HISTOGRAM
        # ================================================

        if (
            "confidence" in pair_logs.columns
            and pair_logs["confidence"].notna().any()
        ):

            st.markdown(
                "<div class='section-title'>📊 Confidence Distribution</div>",
                unsafe_allow_html=True
            )

            fig_hist = px.histogram(
                pair_logs,
                x="confidence",
                nbins=20,
                title="Confidence Histogram"
            )

            fig_hist.update_layout(
                paper_bgcolor="#020617",
                plot_bgcolor="#020617",
                font=dict(
                    family="Inter",
                    size=16,
                    color="white"
                )
            )

            st.plotly_chart(
                fig_hist,
                use_container_width=True
            )

        # ================================================
        # STATUS PIE
        # ================================================

        if (
            "status" in pair_logs.columns
            and pair_logs["status"].notna().any()
        ):

            st.markdown(
                "<div class='section-title'>📈 Pair Status Distribution</div>",
                unsafe_allow_html=True
            )

            status_counts = (
                pair_logs["status"]
                .value_counts()
                .reset_index()
            )

            status_counts.columns = [
                "status",
                "count"
            ]

            fig_pie = px.pie(
                status_counts,
                names="status",
                values="count",
                hole=0.45
            )

            fig_pie.update_layout(
                paper_bgcolor="#020617",
                plot_bgcolor="#020617",
                font=dict(
                    family="Inter",
                    size=16,
                    color="white"
                )
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True
            )

    else:

        st.info(
            "Detailed pair statistics are not "
            "available in the current Stage-1 pipeline."
        )

    # =====================================================
    # RECONSTRUCTION SUMMARY
    # =====================================================

    st.markdown(
        "<div class='section-title'>📋 Reconstruction Summary</div>",
        unsafe_allow_html=True
    )

    input_images = int(
        analytics.get(
            "total_input_images",
            len(uploaded_files)
        )
    )

    registered_images = int(
        analytics.get(
            "registered_images",
            len(cameras)
        )
    )

    registration_ratio = float(
        analytics.get(
            "registration_ratio",
            (
                registered_images
                / max(input_images, 1)
                * 100.0
            )
        )
    )

    sparse_points = int(
        analytics.get(
            "total_sparse_points",
            0
        )
    )

    dense_points = int(
        analytics.get(
            "total_dense_points",
            0
        )
    )

    health_score = float(
        analytics.get(
            "health_score",
            registration_ratio
        )
    )

    st.markdown(
        f"""
### Final Reconstruction Statistics

- Input Images: **{input_images:,}**
- COLMAP Registered Images: **{registered_images:,}**
- Registration Ratio: **{registration_ratio:.2f}%**
- Sparse Points: **{sparse_points:,}**
- Dense Points Generated: **{dense_points:,}**
- Final Cleaned Points: **{len(points):,}**
- Camera Positions: **{len(cameras):,}**
- Registration Health Score: **{health_score:.2f}%**
- Reconstruction Stage Time: **{processing_time:.2f} sec**
- COLMAP Time: **{colmap_time:.2f} sec**
- Dense + Cleaning + Mesh Time: **{reconstruction_time:.2f} sec**
- **Total Pipeline Time: {total_pipeline_time:.2f} sec**
- Sparse Reconstruction: **COLMAP SfM + Bundle Adjustment**
- Dense Reconstruction: **Stage-1 StereoSGBM Prototype**
"""
    )