"""
ui/header.py
------------
Top navigation & centered hero header matching the Figma design for the
Advanced SfM Reconstruction Engine.
"""

import streamlit as st


def show_header() -> None:
    """Render the top navbar and glowing hero header matching the Figma design."""

    st.markdown(
        """
<!-- Top Navbar -->
<div class="top-navbar">
  <div class="nav-brand">
    <div class="nav-logo-box">🚀</div>
    <div class="nav-brand-title">SfM <span>Studio AI</span></div>
  </div>
  <div class="nav-links">
    <span class="nav-link-item">Documentation</span>
    <span class="nav-link-item">Examples</span>
    <span class="nav-link-item">API</span>
    <div class="nav-badge-pill">
      <span style="color:#4ade80;">●</span> v3.0 · GPU Active
    </div>
  </div>
</div>

<!-- Centered Hero Header -->
<div class="hero-container">
  <div class="hero-glow-sphere"></div>
  <div class="hero-content">
    <div class="hero-icon-wrapper">
      🚀
    </div>
    <br/>
    <div class="hero-pill-tag">
      ✨ Next-Gen 3D Reconstruction
    </div>
    <h1 class="hero-title">
      <span class="gradient-text">Advanced SfM</span><br/>
      <span class="white-text">Reconstruction Engine</span>
    </h1>
    <p class="hero-subtitle">
      Structure-from-Motion · COLMAP Pipeline · High-Fidelity 3D Point Clouds &amp; Camera Poses
    </p>
    <div class="hero-neon-divider"></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)