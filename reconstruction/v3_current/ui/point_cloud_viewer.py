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


def render_point_cloud(points, colors=None, title="🌐 3D Point Cloud Reconstruction", key=None):
    """
    Render the reconstructed point cloud with support for photorealistic
    RGB colors from the original dataset.
    """
    import hashlib
    radio_key = key or f"pc_color_mode_{hashlib.md5(title.encode('utf-8')).hexdigest()[:8]}"

    st.markdown(
        f"<div class='section-title'>{title}</div>",
        unsafe_allow_html=True
    )

    if colors is None:
        colors = st.session_state.get("reconstruction_colors")

    points = np.asarray(points, dtype=np.float64)
    has_colors = False

    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)
        if len(colors) == len(points):
            has_colors = True

    points = np.nan_to_num(
        points,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    mask = np.linalg.norm(points, axis=1) > 0
    points = points[mask]
    if has_colors:
        colors = colors[mask]

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

    keep_mask = distances < threshold
    display_points = points[keep_mask]
    display_colors = colors[keep_mask] if has_colors else None

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
        if display_colors is not None:
            display_colors = display_colors[idx]

    # Color scheme selector: photorealistic RGB vs height elevation
    color_options = (
        ["Photorealistic RGB (Original Dataset)", "Height Map Elevation (Turbo)"]
        if has_colors
        else ["Height Map Elevation (Turbo)"]
    )

    selected_mode = st.radio(
        "Point Cloud Color Scheme",
        color_options,
        index=0,
        horizontal=True,
        key=radio_key,
        help="Switch between true photographic colors from your dataset and pseudo-color height mapping."
    )

    if selected_mode.startswith("Photorealistic") and display_colors is not None:
        c_vals = display_colors.copy()
        if np.max(c_vals) <= 1.01:
            c_vals = (c_vals * 255.0).astype(int)
        else:
            c_vals = c_vals.astype(int)
        c_vals = np.clip(c_vals, 0, 255)
        marker_color = [f"rgb({r},{g},{b})" for r, g, b in c_vals]

        marker_dict = dict(
            size=POINT_MARKER_SIZE,
            color=marker_color,
            opacity=POINT_OPACITY
        )
    else:
        marker_dict = dict(
            size=POINT_MARKER_SIZE,
            color=display_points[:, 2],
            colorscale="Turbo",
            opacity=POINT_OPACITY
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=display_points[:, 0],
            y=display_points[:, 1],
            z=display_points[:, 2],
            mode="markers",
            marker=marker_dict
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
        width='stretch'
    )