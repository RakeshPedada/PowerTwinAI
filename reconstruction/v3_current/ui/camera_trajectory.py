import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_camera_trajectory(cameras):
    """
    Render reconstructed camera trajectory.
    """

    st.markdown(
        "<div class='section-title'>📷 Camera Trajectory</div>",
        unsafe_allow_html=True
    )

    cameras = np.asarray(
        cameras,
        dtype=np.float64
    )

    if (
        cameras.ndim != 2
        or cameras.shape[1] != 3
        or len(cameras) == 0
    ):

        st.warning(
            "No valid camera trajectory available."
        )

        return

    fig = go.Figure()

    fig.add_trace(
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

    fig.update_layout(

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
        fig,
        use_container_width=True
    )