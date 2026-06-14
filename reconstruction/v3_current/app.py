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

# =========================================================
# PATHS
# =========================================================

TEMP_DIR = "temp_session"

LOG_FILE = os.path.join(TEMP_DIR, "logs.txt")
STATUS_FILE = os.path.join(TEMP_DIR, "status.json")
RESULT_FILE = os.path.join(TEMP_DIR, "result_data.npz")
IMAGE_PATHS_FILE = os.path.join(TEMP_DIR, "image_paths.json")

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
# SESSION STATE
# =========================================================

if "reconstruction_running" not in st.session_state:
    st.session_state.reconstruction_running = False

if "reconstruction_done" not in st.session_state:
    st.session_state.reconstruction_done = False

# =========================================================
# PREMIUM CSS
# =========================================================

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">

<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #020617;
    color: white;
}

.stApp {

    background:
        radial-gradient(circle at top left,
        rgba(0,255,200,0.08),
        transparent 25%),

        radial-gradient(circle at bottom right,
        rgba(0,140,255,0.08),
        transparent 25%),

        linear-gradient(
            135deg,
            #020617 0%,
            #06111f 40%,
            #020617 100%
        );
}

/* TITLE */

.main-title {

    font-size: 72px;
    font-weight: 900;

    text-align: center;

    margin-top: 30px;
    margin-bottom: 10px;

    background: linear-gradient(
        90deg,
        #00e5ff,
        #00ff99,
        #00c3ff
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    text-shadow:
        0 0 25px rgba(0,255,255,0.25);
}

.sub-title {

    text-align: center;

    font-size: 24px;

    color: #94a3b8;

    margin-bottom: 50px;
}

/* METRIC CARDS */

.metric-card {

    background: rgba(255,255,255,0.04);

    border: 1px solid rgba(0,255,170,0.15);

    backdrop-filter: blur(14px);

    border-radius: 24px;

    padding: 30px;

    box-shadow:
        0 0 25px rgba(0,255,170,0.08);

    transition: 0.3s ease;
}

.metric-card:hover {

    transform: translateY(-5px);

    box-shadow:
        0 0 40px rgba(0,255,170,0.18);
}

.metric-title {

    font-size: 18px;
    color: #94a3b8;
    font-weight: 500;
}

.metric-value {

    font-size: 48px;
    font-weight: 800;
    color: white;
}

/* BUTTON */

.stButton>button {

    background: linear-gradient(
        90deg,
        #00c6ff,
        #00ff99
    );

    color: black;

    border: none;

    border-radius: 16px;

    padding: 16px 34px;

    font-size: 20px;

    font-weight: 800;

    box-shadow:
        0 0 25px rgba(0,255,170,0.25);

    transition: 0.3s ease;
}

.stButton>button:hover {

    transform: scale(1.04);

    box-shadow:
        0 0 45px rgba(0,255,170,0.45);
}

/* SECTION HEADERS */

.section-title {

    font-size: 42px;
    font-weight: 800;

    margin-top: 50px;
    margin-bottom: 25px;

    background: linear-gradient(
        90deg,
        white,
        #9cecff
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* SUCCESS BOX */

.success-box {

    background: linear-gradient(
        90deg,
        rgba(0,255,140,0.18),
        rgba(0,180,255,0.12)
    );

    border-radius: 24px;

    padding: 24px;

    border: 1px solid rgba(0,255,170,0.22);

    font-size: 28px;

    font-weight: 700;

    color: #d1fae5;

    text-align: center;

    box-shadow:
        0 0 30px rgba(0,255,170,0.12);
}

/* HEALTH BOX */

.health-box {

    background: linear-gradient(
        90deg,
        rgba(0,255,120,0.15),
        rgba(0,180,255,0.15)
    );

    border-radius: 26px;

    padding: 34px;

    text-align: center;

    font-size: 34px;

    font-weight: 800;

    border: 1px solid rgba(0,255,170,0.18);

    box-shadow:
        0 0 35px rgba(0,255,170,0.14);

    margin-top: 20px;
    margin-bottom: 30px;
}

/* PLOTS */

.js-plotly-plot {

    border-radius: 24px !important;

    overflow: hidden !important;

    border: 1px solid rgba(0,255,170,0.12);

    box-shadow:
        0 0 25px rgba(0,255,170,0.08);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown("""
<div class='main-title'>
🚀 Advanced SfM Reconstruction Engine
</div>

<div class='sub-title'>
Global Structure-from-Motion + Bundle Adjustment + AI Analytics Pipeline
</div>
""", unsafe_allow_html=True)

st.markdown("---")

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
    and
    os.path.exists(RESULT_FILE)
):

    data = np.load(
        RESULT_FILE,
        allow_pickle=True
    )

    points = data["points"]

    cameras = data["cameras"]

    processing_time = float(
        data["processing_time"]
    )

    analytics = data["analytics"].item()

    st.markdown("""
    <div class="success-box">
    ✅ Reconstruction Completed Successfully
    </div>
    """, unsafe_allow_html=True)

 #   st.balloons()

    # =====================================================
    # 3D MESH RECONSTRUCTION
    # =====================================================

    st.markdown(
        "<div class='section-title'>🌐 3D Mesh Reconstruction</div>",
        unsafe_allow_html=True
    )
    # Remove invalid points
    # =====================================================
    # SAFE POINT CLEANING
    # =====================================================

    points = np.nan_to_num(
        points,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    mask = np.linalg.norm(
        points,
        axis=1
    ) > 0

    points = points[mask]

    st.write(f"Valid sparse points: {len(points)}")

    # =====================================================
    # SAFE EMPTY CHECK
    # =====================================================

    if len(points) == 0:

        st.error("No valid reconstruction points found.")

        st.stop()

    # =====================================================
    # REMOVE LARGE OUTLIERS
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
        99
    )

    points = points[
        distances < threshold
    ]

    st.write(f"Filtered sparse points: {len(points)}")
    
    st.write(f"Filtered sparse points: {len(points)}")

    # Normalize
    max_val = np.max(np.abs(points))

    if max_val > 0:

        points = points / max_val

    # Downsample
    if len(points) > 100000:

        idx = np.random.choice(
            len(points),
            100000,
            replace=False
        )

        points = points[idx]

    # Create figure
    fig = go.Figure()
    fig.add_trace(
    go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],

        mode='markers',

        marker=dict(
            size=2,
            color=points[:, 2],
            colorscale='Turbo',
            opacity=0.8
        )
    )
)
# dense_reconstructor = DenseReconstructor()
# dense_reconstructor.dense_points = points
# dense_reconstructor.dense_colors = np.ones_like(points)
# mesh = dense_reconstructor.create_mesh_from_cloud()
#
# if mesh is not None:
#
#     vertices = np.asarray(mesh.vertices)
#
#     triangles = np.asarray(mesh.triangles)
#
#     fig.add_trace(
#         go.Mesh3d(
#             x=vertices[:, 0],
#             y=vertices[:, 1],
#             z=vertices[:, 2],
#
#             i=triangles[:, 0],
#             j=triangles[:, 1],
#             k=triangles[:, 2],
#
#             opacity=1.0,
#
#             color='lightblue',
#
#             flatshading=False
#         )
#     )

    fig.update_layout(

        height=900,

        paper_bgcolor="#020617",

        plot_bgcolor="#020617",

        scene=dict(
            bgcolor="#010409",
            aspectmode='cube'
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

    fig_cam = go.Figure()

    fig_cam.add_trace(
        go.Scatter3d(
            x=cameras[:,0],
            y=cameras[:,1],
            z=cameras[:,2],

            mode='lines+markers',

            marker=dict(
                size=8,
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
            bgcolor="#010409"
        ),

        font=dict(
            family="Inter",
            size=16,
            color="white"
        )
    )

    st.plotly_chart(
        fig_cam,
        use_container_width=True
    )

    # =====================================================
    # CONFIDENCE HISTOGRAM
    # =====================================================

    pair_logs = pd.read_csv(
        "output/pair_logs.csv"
    )

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

    fig_hist.update_traces(
        marker_color="#00d4ff"
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

    # =====================================================
    # PIE CHART
    # =====================================================

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

    # =====================================================
    # TABLE
    # =====================================================

    st.markdown(
        "<div class='section-title'>📋 Pair Statistics</div>",
        unsafe_allow_html=True
    )

    st.dataframe(
        pair_logs,
        use_container_width=True
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    st.markdown(
        "<div class='section-title'>📋 Reconstruction Summary</div>",
        unsafe_allow_html=True
    )

    st.markdown(f"""
### Final Reconstruction Statistics

- Uploaded Images: **{len(uploaded_files)}**
- Generated 3D Points: **{len(points):,}**
- Camera Positions: **{len(cameras)}**
- Successful Pairs: **{analytics["successful_pairs"]}**
- Failed Pairs: **{analytics["failed_pairs"]}**
- Average Confidence: **{analytics.get("avg_confidence", 100.0):.2f}%**
- Average Inliers: **{analytics["avg_inliers"]}**
- Average Points: **{analytics["avg_points"]}**
- Health Score: **{analytics["health_score"]:.2f}%**
- Processing Time: **{processing_time:.2f} sec**
- Reconstruction Type: **Global SfM + Bundle Adjustment**
- Feature Detector: **SIFT**
- Matcher: **FLANN**
- Parallel Processing: **Enabled**
""")
