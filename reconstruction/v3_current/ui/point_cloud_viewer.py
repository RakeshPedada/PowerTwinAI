import numpy as np
import plotly.graph_objects as go
import streamlit as st

from config.settings import (
    MAX_DISPLAY_POINTS,
    DISPLAY_OUTLIER_PERCENTILE,
    POINT_MARKER_SIZE,
    POINT_OPACITY,
    PLOT_HEIGHT,
    PLOT_BACKGROUND,
    SCENE_BACKGROUND,
)


def render_point_cloud(points):
    """
    Render the reconstructed point cloud.
    """

    st.markdown(
        "<div class='section-title'>🌐 3D Point Cloud Reconstruction</div>",
        unsafe_allow_html=True
    )

    points = np.asarray(points, dtype=np.float64)

    points = np.nan_to_num(
        points,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    mask = np.linalg.norm(points, axis=1) > 0
    points = points[mask]

    st.write(f"Valid reconstruction points: {len(points):,}")

    if len(points) == 0:
        st.error("No valid reconstruction points found.")
        st.stop()

    center = np.mean(points, axis=0)

    distances = np.linalg.norm(
        points - center,
        axis=1
    )

    threshold = np.percentile(
        distances,
        DISPLAY_OUTLIER_PERCENTILE
    )

    display_points = points[
        distances < threshold
    ]

    st.write(
        f"Display points after outlier filtering: {len(display_points):,}"
    )

    max_val = np.max(np.abs(display_points))

    if max_val > 0:
        display_points = display_points / max_val

    if len(display_points) > MAX_DISPLAY_POINTS:

        rng = np.random.default_rng(42)

        idx = rng.choice(
            len(display_points),
            MAX_DISPLAY_POINTS,
            replace=False
        )

        display_points = display_points[idx]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=display_points[:, 0],
            y=display_points[:, 1],
            z=display_points[:, 2],
            mode="markers",
            marker=dict(
                size=POINT_MARKER_SIZE,
                color=display_points[:, 2],
                colorscale="Turbo",
                opacity=POINT_OPACITY
            )
        )
    )

    fig.update_layout(
        height=PLOT_HEIGHT,
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        scene=dict(
            bgcolor=SCENE_BACKGROUND,
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