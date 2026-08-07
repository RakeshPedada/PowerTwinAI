import os
import pandas as pd
import plotly.express as px
import streamlit as st


def render_pair_statistics():
    """
    Render pair matching statistics.
    """

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

    if pair_logs.empty:

        st.info(
            "Detailed pair statistics are not available in the current Stage-1 pipeline."
        )

        return

    st.dataframe(
        pair_logs,
        use_container_width=True
    )

    # =====================================================
    # CONFIDENCE HISTOGRAM
    # =====================================================

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

    # =====================================================
    # STATUS PIE
    # =====================================================

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