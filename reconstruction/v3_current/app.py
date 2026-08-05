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

TEMP_DIR = "temp_session"

LOG_FILE = os.path.join(TEMP_DIR, "logs.txt")
STATUS_FILE = os.path.join(TEMP_DIR, "status.json")
RESULT_FILE = os.path.join(TEMP_DIR, "result_data.npz")
IMAGE_PATHS_FILE = os.path.join(TEMP_DIR, "image_paths.json")

# =========================================================
# SESSION STATE
# =========================================================

if "reconstruction_running" not in st.session_state:
    st.session_state.reconstruction_running = False

if "reconstruction_done" not in st.session_state:
    st.session_state.reconstruction_done = False



# =========================================================
# HEADER
# =========================================================

show_header()
# =========================================================
# UPLOAD
# =========================================================

mode = st.radio(
    "Input Method",
    [
        "Upload Images",
        "Dataset Folder"
    ]
)
uploaded_files = []

dataset_path = ""

if mode == "Upload Images":

    uploaded_files = st.file_uploader(
        "📂 Upload Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

else:

    dataset_path = st.text_input(
        "Dataset Folder",
        r"E:\owl_dataset"
    )
    if mode == "Dataset Folder":

        if os.path.exists(dataset_path):

            uploaded_files = glob.glob(
                os.path.join(dataset_path, "*.jpg")
            )

            uploaded_files += glob.glob(
                os.path.join(dataset_path, "*.jpeg")
            )

            uploaded_files += glob.glob(
                os.path.join(dataset_path, "*.png")
            )    
st.write(f"Images Loaded: {len(uploaded_files)}")

# =========================================================
# IMAGE PREVIEW
# =========================================================

if uploaded_files:

    st.markdown(
        "<div class='section-title'>📸 Uploaded Dataset</div>",
        unsafe_allow_html=True
    )

    cols = st.columns(4)

    total_size = 0

    for idx, file in enumerate(uploaded_files):

        if mode == "Upload Images":

            size_mb = len(file.getvalue()) / (1024 * 1024)

        else:

            size_mb = os.path.getsize(file) / (1024 * 1024)

        total_size += size_mb
        if idx < 8:

            if mode == "Upload Images":

                cols[idx % 4].image(
                    file,
                    caption=file.name,
                    use_container_width=True
                )

            else:

                cols[idx % 4].image(
                    file,
                    caption=os.path.basename(file),
                    use_container_width=True
                )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Images</div>
            <div class="metric-value">{len(uploaded_files)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Dataset Size</div>
            <div class="metric-value">{total_size:.2f} MB</div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# START BUTTON
# =========================================================

if uploaded_files:

    if not st.session_state.reconstruction_running:

        if st.button("🚀 Start Reconstruction"):

            os.makedirs(TEMP_DIR, exist_ok=True)

            with open(
                LOG_FILE,
                "w",
                encoding="utf-8",
                errors="ignore"
            ) as f:
                f.write("")

            if os.path.exists(STATUS_FILE):
                os.remove(STATUS_FILE)

            if os.path.exists(RESULT_FILE):
                os.remove(RESULT_FILE)

            
            saved_paths = []

            if mode == "Upload Images":

                for file in uploaded_files:

                    save_path = os.path.join(
                        TEMP_DIR,
                        file.name
                    )

                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())

                    saved_paths.append(save_path)

            else:

                saved_paths = uploaded_files


            with open(IMAGE_PATHS_FILE, "w") as f:
                json.dump(saved_paths, f)

            subprocess.Popen(
                ["python", "reconstruction_runner.py"],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )

            st.session_state.reconstruction_running = True
            st.session_state.reconstruction_done = False
# =========================================================
# LIVE LOGS
# =========================================================

if st.session_state.reconstruction_running:

    st_autorefresh(interval=7000, key="refresh")

    logs = []

    if os.path.exists(LOG_FILE):

        with open(
            LOG_FILE,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            logs = f.readlines()

    st.markdown(
        "<div class='section-title'>📜 Live Reconstruction Logs</div>",
        unsafe_allow_html=True
    )

    st.text_area(
        "Reconstruction Progress",
        "".join(logs[-25:]),
        height=250,
        disabled=True
    )

    if os.path.exists(STATUS_FILE):

        with open(STATUS_FILE, "r") as f:
            status_data = json.load(f)

        if status_data.get("status") == "COMPLETED":

            st.session_state.reconstruction_running = False
            st.session_state.reconstruction_done = True

            st.rerun()

# =========================================================
# RESULTS
# =========================================================

if (
    st.session_state.reconstruction_done
    and os.path.exists(RESULT_FILE)
):

    data = np.load(
        RESULT_FILE,
        allow_pickle=True
    )

    points = data["points"]
    cameras = data["cameras"]

    analytics = data["analytics"].item()

    # =====================================================
    # TIMINGS
    # =====================================================

    processing_time = float(
        data["processing_time"]
    )

    colmap_time = float(
        data["colmap_time"]
    ) if "colmap_time" in data.files else 0.0

    reconstruction_time = float(
        data["reconstruction_time"]
    ) if "reconstruction_time" in data.files else processing_time

    total_pipeline_time = float(
        data["total_pipeline_time"]
    ) if "total_pipeline_time" in data.files else (
        colmap_time + reconstruction_time
    )

    # =====================================================
    # PLY PATH
    # =====================================================

    ply_path = ""

    if "ply_path" in data.files:
        ply_path = str(
            data["ply_path"].item()
        )

    # =====================================================
    # SUCCESS
    # =====================================================

    st.markdown(
        """
        <div class="success-box">
        ✅ Reconstruction Completed Successfully
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # DOWNLOAD
    # =====================================================

    st.markdown(
        "<div class='section-title'>⬇️ Download Reconstruction</div>",
        unsafe_allow_html=True
    )

    if (
        ply_path
        and os.path.exists(ply_path)
    ):

        with open(
            ply_path,
            "rb"
        ) as ply_file:

            st.download_button(
                label="⬇️ Download 3D Point Cloud (.ply)",
                data=ply_file.read(),
                file_name=os.path.basename(
                    ply_path
                ),
                mime="application/octet-stream"
            )

    else:

        st.warning(
            f"PLY output file not found: {ply_path}"
        )

    # =====================================================
    # PROCESSING TIME BREAKDOWN
    # =====================================================

    st.markdown(
        "<div class='section-title'>⏱️ Processing Time</div>",
        unsafe_allow_html=True
    )

    t1, t2, t3 = st.columns(3)

    with t1:

        st.metric(
            "COLMAP",
            f"{colmap_time:.2f} sec"
        )

    with t2:

        st.metric(
            "Dense + Cleaning + Mesh",
            f"{reconstruction_time:.2f} sec"
        )

    with t3:

        st.metric(
            "Total Pipeline Time",
            f"{total_pipeline_time:.2f} sec"
        )

    # =====================================================
    # POINT CLOUD VISUALIZATION
    # =====================================================

    st.markdown(
        "<div class='section-title'>🌐 3D Point Cloud Reconstruction</div>",
        unsafe_allow_html=True
    )

    points = np.asarray(
        points,
        dtype=np.float64
    )

    points = np.nan_to_num(
        points,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    mask = (
        np.linalg.norm(
            points,
            axis=1
        ) > 0
    )

    points = points[mask]

    st.write(
        f"Valid reconstruction points: "
        f"{len(points):,}"
    )

    if len(points) == 0:

        st.error(
            "No valid reconstruction points found."
        )

        st.stop()

    # =====================================================
    # DISPLAY-ONLY OUTLIER REMOVAL
    # =====================================================

    center = np.mean(
        points,
        axis=0
    )

    distances = np.linalg.norm(
        points - center,
        axis=1
    )

    threshold = np.percentile(
        distances,
        95
    )

    display_points = points[
        distances < threshold
    ]

    st.write(
        f"Display points after outlier filtering: "
        f"{len(display_points):,}"
    )

    # =====================================================
    # NORMALIZE FOR DISPLAY ONLY
    # =====================================================

    max_val = np.max(
        np.abs(
            display_points
        )
    )

    if max_val > 0:

        display_points = (
            display_points
            / max_val
        )

    # =====================================================
    # DOWNSAMPLE FOR BROWSER
    # =====================================================

    if len(display_points) > 100000:

        rng = np.random.default_rng(42)

        idx = rng.choice(
            len(display_points),
            100000,
            replace=False
        )

        display_points = (
            display_points[idx]
        )

    # =====================================================
    # POINT CLOUD FIGURE
    # =====================================================

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(

            x=display_points[:, 0],
            y=display_points[:, 1],
            z=display_points[:, 2],

            mode="markers",

            marker=dict(
                size=2,
                color=display_points[:, 2],
                colorscale="Turbo",
                opacity=0.8
            )
        )
    )

    fig.update_layout(

        height=900,

        paper_bgcolor="#020617",

        plot_bgcolor="#020617",

        scene=dict(
            bgcolor="#010409",
            aspectmode="data"
        ),

        margin=dict(
            l=0,
            r=0,
            t=20,
            b=0
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

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