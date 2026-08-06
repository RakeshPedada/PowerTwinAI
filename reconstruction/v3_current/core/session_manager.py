"""
Session state manager for PowerTwinAI.
"""

import streamlit as st


def initialize_session():
    """
    Initialize all required Streamlit session variables.
    """

    defaults = {
        "reconstruction_running": False,
        "reconstruction_done": False,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value