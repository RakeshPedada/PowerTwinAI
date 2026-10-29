import os
import json
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

    # ── ML Pair Selection Telemetry Card ─────────────────────────────────────
    ml_telemetry_path = os.path.join("output", "ml_pair_telemetry.json")
    if not os.path.exists(ml_telemetry_path):
        ml_telemetry_path = os.path.join("temp_session", "ml_pair_telemetry.json")

    if os.path.exists(ml_telemetry_path):
        try:
            with open(ml_telemetry_path, "r", encoding="utf-8") as tf:
                ml_telem = json.load(tf)
            if ml_telem and "candidate_pairs" in ml_telem:
                st.markdown(
                    """
<div class="glass-stat-card" style="margin-bottom: 16px; border-color: rgba(56, 189, 248, 0.3);">
  <div class="glass-stat-label" style="color: #38bdf8;">🧠 ML Image-Pair Candidate Selection Telemetry</div>
  <p style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
    Visual embeddings &amp; graph connectivity replaced exhaustive all-to-all matching with top-K candidate pairs.
  </p>
</div>
""",
                    unsafe_allow_html=True
                )
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                m_c1.metric(
                    "Candidate Pairs",
                    f"{ml_telem.get('candidate_pairs', 0):,}",
                    delta=f"-{ml_telem.get('pair_reduction_pct', 0)}% vs exhaustive",
                    delta_color="inverse"
                )
                m_c2.metric("Exhaustive Baseline", f"{ml_telem.get('exhaustive_pairs', 0):,} pairs")
                m_c3.metric(
                    "Graph Connectivity",
                    "✅ 1 Component" if ml_telem.get("connected_components") == 1 else f"⚠️ {ml_telem.get('connected_components')} components",
                    help="Ensures no isolated images or split clusters exist in candidate graph."
                )
                m_c4.metric(
                    "Degree (Min / Avg)",
                    f"{ml_telem.get('min_degree', 0)} / {ml_telem.get('avg_degree', 0.0):.1f}"
                )
        except Exception:
            pass

    pairs_txt_path = os.path.join("outputs", "ml_candidate_pairs.txt")
    if not os.path.exists(pairs_txt_path):
        pairs_txt_path = os.path.join("output", "ml_candidate_pairs.txt")
    if os.path.exists(pairs_txt_path):
        with st.expander("📄 View ML Candidate Pair List (`ml_candidate_pairs.txt`)", expanded=False):
            with open(pairs_txt_path, "r", encoding="utf-8") as pf:
                pairs_lines = pf.readlines()
            st.caption(f"Showing {len(pairs_lines):,} candidate pairs selected by visual embedding ranker:")
            st.text_area("Candidate Pairs", "".join(pairs_lines[:50]) + (f"\n... and {len(pairs_lines)-50} more pairs" if len(pairs_lines) > 50 else ""), height=150, disabled=True)

    if pair_logs.empty:
        st.info(
           "Detailed pair statistics are not available for the current COLMAP backend output."
        )
        return

    st.dataframe(
        pair_logs,
        width='stretch'
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
            paper_bgcolor="#0a0614",
            plot_bgcolor="#0a0614",
            font=dict(
                family="Inter",
                size=14,
                color="white"
            )
        )

        st.plotly_chart(
            fig_hist,
            width='stretch'
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
            paper_bgcolor="#0a0614",
            plot_bgcolor="#0a0614",
            font=dict(
                family="Inter",
                size=14,
                color="white"
            )
        )

        st.plotly_chart(
            fig_pie,
            width='stretch'
        )