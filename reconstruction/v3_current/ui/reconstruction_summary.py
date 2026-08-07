import streamlit as st


def render_reconstruction_summary(
    analytics,
    uploaded_files,
    cameras,
    points,
    processing_time,
    colmap_time,
    reconstruction_time,
    total_pipeline_time
):
    """
    Render final reconstruction statistics.
    """

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