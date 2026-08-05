"""
Header UI components for the PowerTwinAI Streamlit application.

Responsibilities:
- Display application title
- Display application subtitle
"""

import streamlit as st


def show_header():
    st.markdown("""
    <div class='main-title'>
    🚀 Advanced SfM Reconstruction Engine
    </div>

    <div class='sub-title'>
    Global Structure-from-Motion + Bundle Adjustment + AI Analytics Pipeline
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")