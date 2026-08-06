import os
import numpy as np
import streamlit as st


def render_results_dashboard(result_file):
    """
    Render reconstruction summary dashboard.

    Returns
    -------
    tuple
        (points, cameras, analytics, processing_time,
         colmap_time, reconstruction_time,
         total_pipeline_time)
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

    analytics = data["analytics"].item()

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

    ply_path = ""

    if "ply_path" in data.files:
        ply_path = str(
            data["ply_path"].item()
        )

    st.markdown(
        """
        <div class="success-box">
        ✅ Reconstruction Completed Successfully
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='section-title'>⬇️ Download Reconstruction</div>",
        unsafe_allow_html=True
    )

    if ply_path and os.path.exists(ply_path):

        with open(
            ply_path,
            "rb"
        ) as ply_file:

            st.download_button(
                label="⬇️ Download 3D Point Cloud (.ply)",
                data=ply_file.read(),
                file_name=os.path.basename(ply_path),
                mime="application/octet-stream"
            )

    else:

        st.warning(
            f"PLY output file not found: {ply_path}"
        )

    st.markdown(
        "<div class='section-title'>⏱️ Processing Time</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "COLMAP",
        f"{colmap_time:.2f} sec"
    )

    c2.metric(
        "Dense + Cleaning + Mesh",
        f"{reconstruction_time:.2f} sec"
    )

    c3.metric(
        "Total Pipeline Time",
        f"{total_pipeline_time:.2f} sec"
    )

    return (
        points,
        cameras,
        analytics,
        processing_time,
        colmap_time,
        reconstruction_time,
        total_pipeline_time
    )