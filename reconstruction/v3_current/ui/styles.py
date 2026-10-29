"""
ui/styles.py
------------
Luxury Cyberpunk & Dark Glassmorphic Design System matching the Figma UI for
Advanced SfM Reconstruction Engine.
"""

import streamlit as st


def load_styles() -> None:
    """Inject all custom CSS into the Streamlit app matching the Figma design."""
    st.markdown(
        """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global Theme Colors ── */
:root {
    --bg-main: #07050d;
    --bg-surface: #0e0a1a;
    --bg-card: rgba(18, 13, 33, 0.85);
    --border-card: rgba(168, 85, 247, 0.22);
    --border-card-hover: rgba(236, 72, 153, 0.45);
    --primary-purple: #a855f7;
    --primary-pink: #ec4899;
    --primary-magenta: #d946ef;
    --primary-violet: #8b5cf6;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
}

html, body, [data-testid="stApp"] {
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
    background-color: #07050d !important;
    background-image: 
        radial-gradient(at 15% 10%, rgba(147, 51, 234, 0.22) 0px, transparent 50%),
        radial-gradient(at 85% 15%, rgba(236, 72, 153, 0.18) 0px, transparent 50%),
        radial-gradient(at 50% 60%, rgba(126, 34, 206, 0.12) 0px, transparent 65%),
        radial-gradient(at 80% 85%, rgba(219, 39, 119, 0.14) 0px, transparent 50%) !important;
    background-attachment: fixed !important;
    color: var(--text-main) !important;
    letter-spacing: -0.01em;
}

/* Hide default Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Main Container ── */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* ====================================================
   TOP NAVBAR (Figma header bar)
   ==================================================== */
.top-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    margin: 0 -1.5rem 24px -1.5rem;
    background: rgba(10, 7, 20, 0.9);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(168, 85, 247, 0.14);
}

.nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.nav-logo-box {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: linear-gradient(135deg, #9333ea, #db2777);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    box-shadow: 0 0 16px rgba(219, 39, 119, 0.45);
}

.nav-brand-title {
    font-size: 16px;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.02em;
}

.nav-brand-title span {
    background: linear-gradient(135deg, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.nav-links {
    display: flex;
    align-items: center;
    gap: 20px;
}

.nav-link-item {
    font-size: 13px;
    font-weight: 500;
    color: #94a3b8;
    text-decoration: none;
    transition: color 0.2s ease;
}

.nav-link-item:hover {
    color: #f472b6;
}

.nav-badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    background: rgba(168, 85, 247, 0.12);
    border: 1px solid rgba(168, 85, 247, 0.35);
    border-radius: 9999px;
    color: #d8b4fe;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    box-shadow: 0 0 12px rgba(168, 85, 247, 0.2);
}

/* ====================================================
   HERO HEADER SECTION (Figma Hero)
   ==================================================== */
.hero-container {
    text-align: center;
    padding: 30px 20px 20px;
    position: relative;
    overflow: hidden;
}

.hero-glow-sphere {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 380px;
    height: 380px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.16) 0%, rgba(236, 72, 153, 0.08) 40%, transparent 70%);
    filter: blur(50px);
    pointer-events: none;
    z-index: 0;
}

.hero-content {
    position: relative;
    z-index: 1;
}

.hero-icon-wrapper {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 64px;
    height: 64px;
    background: linear-gradient(135deg, rgba(147, 51, 234, 0.35), rgba(219, 39, 119, 0.35));
    border: 1px solid rgba(236, 72, 153, 0.35);
    border-radius: 18px;
    font-size: 30px;
    box-shadow: 0 0 32px rgba(168, 85, 247, 0.3), inset 0 0 12px rgba(255, 255, 255, 0.15);
    margin-bottom: 16px;
}

.hero-pill-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 16px;
    background: rgba(168, 85, 247, 0.08);
    border: 1px solid rgba(236, 72, 153, 0.3);
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #f472b6;
    margin-bottom: 14px;
    box-shadow: 0 0 16px rgba(236, 72, 153, 0.15);
}

.hero-title {
    font-size: 44px;
    font-weight: 800;
    line-height: 1.15;
    margin: 0 0 12px 0;
    letter-spacing: -0.03em;
}

.hero-title .gradient-text {
    background: linear-gradient(135deg, #c084fc 0%, #f472b6 60%, #fb7185 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-title .white-text {
    color: #ffffff;
}

.hero-subtitle {
    font-size: 15px;
    color: #94a3b8;
    max-width: 600px;
    margin: 0 auto 20px;
    line-height: 1.6;
    font-weight: 400;
}

.hero-neon-divider {
    height: 2px;
    width: 100%;
    max-width: 900px;
    margin: 18px auto 0;
    background: linear-gradient(90deg, transparent 0%, rgba(168, 85, 247, 0.8) 30%, rgba(236, 72, 153, 0.9) 70%, transparent 100%);
    box-shadow: 0 0 12px rgba(236, 72, 153, 0.5);
    border-radius: 999px;
}

/* ====================================================
   FIGMA MAIN CARD CONTAINER
   ==================================================== */
.figma-card {
    background: rgba(15, 10, 26, 0.82);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(168, 85, 247, 0.22);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    position: relative;
    overflow: hidden;
}

.figma-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(236, 72, 153, 0.6), rgba(168, 85, 247, 0.6), transparent);
}

.card-header-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.card-header-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(147, 51, 234, 0.35), rgba(219, 39, 119, 0.35));
    border: 1px solid rgba(168, 85, 247, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 16px rgba(168, 85, 247, 0.25);
}

.card-header-titles h3 {
    margin: 0;
    font-size: 18px;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.01em;
}

.card-header-titles p {
    margin: 3px 0 0;
    font-size: 11px;
    font-weight: 600;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* ====================================================
   INTERACTIVE MODE CARDS (Upload Images vs Existing Data Folder)
   ==================================================== */
.section-sub-label {
    font-size: 11px;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
}

.mode-select-card {
    border-radius: 16px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 14px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    user-select: none;
    min-height: 80px;
}

.mode-card-active {
    background: linear-gradient(135deg, rgba(147, 51, 234, 0.32) 0%, rgba(219, 39, 119, 0.22) 100%);
    border: 1.5px solid rgba(236, 72, 153, 0.65);
    box-shadow: 0 0 24px rgba(168, 85, 247, 0.28), inset 0 0 12px rgba(236, 72, 153, 0.15);
}

.mode-card-inactive {
    background: rgba(18, 12, 33, 0.6);
    border: 1px solid rgba(168, 85, 247, 0.16);
}

.mode-card-inactive:hover {
    border-color: rgba(236, 72, 153, 0.35);
    background: rgba(24, 16, 44, 0.75);
}

.mode-radio-circle {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.radio-active {
    border: 2px solid #ec4899;
    background: rgba(236, 72, 153, 0.2);
    box-shadow: 0 0 10px rgba(236, 72, 153, 0.5);
}

.radio-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #ec4899;
    box-shadow: 0 0 6px #f472b6;
}

.radio-inactive {
    border: 2px solid rgba(148, 163, 184, 0.4);
    background: transparent;
}

.mode-card-content h4 {
    margin: 0;
    font-size: 15px;
    font-weight: 700;
    color: #ffffff;
}

.mode-card-content p {
    margin: 3px 0 0;
    font-size: 12px;
    color: #94a3b8;
}

/* Yellow bullet tag */
.input-zone-tag {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 700;
    color: #f8fafc;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 18px 0 12px 0;
}

.yellow-bullet {
    width: 10px;
    height: 10px;
    background: #eab308;
    border-radius: 2px;
    box-shadow: 0 0 8px rgba(234, 179, 8, 0.6);
}

/* Mode toggle buttons inside columns */
div[data-testid="column"] .stButton > button {
    background: rgba(26, 17, 46, 0.75) !important;
    border: 1px solid rgba(168, 85, 247, 0.25) !important;
    border-radius: 10px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 8px 14px !important;
    margin-top: 6px !important;
    box-shadow: none !important;
    color: #cbd5e1 !important;
    width: 100% !important;
}

div[data-testid="column"] .stButton > button:hover {
    background: linear-gradient(135deg, rgba(147, 51, 234, 0.6), rgba(219, 39, 119, 0.6)) !important;
    border-color: #ec4899 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(236, 72, 153, 0.3) !important;
}

/* ====================================================
   FILE UPLOADER / DROPZONE
   ==================================================== */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(22, 14, 42, 0.5) !important;
    border: 2px dashed rgba(192, 132, 252, 0.38) !important;
    border-radius: 16px !important;
    padding: 28px 20px !important;
    transition: all 0.25s ease !important;
    box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.3) !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(244, 114, 182, 0.75) !important;
    background: rgba(32, 18, 60, 0.65) !important;
    box-shadow: 0 0 24px rgba(236, 72, 153, 0.25), inset 0 2px 10px rgba(0, 0, 0, 0.3) !important;
}

[data-testid="stFileUploader"] label {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* ── Text Input ── */
[data-testid="stTextInput"] input {
    background: rgba(20, 13, 38, 0.85) !important;
    border: 1px solid rgba(168, 85, 247, 0.25) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    padding: 12px 16px !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: #ec4899 !important;
    box-shadow: 0 0 0 3px rgba(236, 72, 153, 0.2) !important;
}

/* ====================================================
   BUTTONS (Figma Glowing Gradient)
   ==================================================== */
.stButton > button {
    background: linear-gradient(135deg, #9333ea 0%, #c026d3 50%, #db2777 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 14px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    padding: 14px 32px !important;
    box-shadow: 0 8px 24px rgba(192, 38, 211, 0.4), 0 0 20px rgba(147, 51, 234, 0.25) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    letter-spacing: 0.02em !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 30px rgba(219, 39, 119, 0.55), 0 0 28px rgba(168, 85, 247, 0.4) !important;
    border-color: rgba(255, 255, 255, 0.4) !important;
    opacity: 0.98 !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.35) !important;
}

/* ====================================================
   METRIC CARDS & GLASS STATS
   ==================================================== */
.glass-stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin: 20px 0;
}

.glass-stat-card {
    background: rgba(18, 12, 33, 0.7);
    border: 1px solid rgba(168, 85, 247, 0.18);
    border-radius: 16px;
    padding: 18px 20px;
    backdrop-filter: blur(16px);
    transition: all 0.2s ease;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

.glass-stat-card:hover {
    border-color: rgba(236, 72, 153, 0.35);
    transform: translateY(-2px);
}

.glass-stat-label {
    font-size: 11px;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

.glass-stat-value {
    font-size: 24px;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.glass-stat-sub {
    font-size: 11px;
    color: #64748b;
    margin-top: 4px;
}

/* ====================================================
   SECTION HEADINGS & BADGES
   ==================================================== */
.section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
    margin: 28px 0 16px 0;
    letter-spacing: -0.01em;
}

.badge-counter {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 18px;
    background: rgba(168, 85, 247, 0.12);
    border: 1px solid rgba(168, 85, 247, 0.3);
    border-radius: 999px;
    color: #d8b4fe;
    font-size: 13px;
    font-weight: 600;
    margin: 12px 0;
}

/* Requirements checks */
.req-item {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 12px;
    color: #94a3b8;
    margin-bottom: 6px;
}

.dot-green {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px #10b981;
    flex-shrink: 0;
}

.dot-red {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ef4444;
    box-shadow: 0 0 8px #ef4444;
    flex-shrink: 0;
}

.dot-yellow {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #f59e0b;
    box-shadow: 0 0 8px #f59e0b;
    flex-shrink: 0;
}

/* Status banner */
.status-box-success {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 14px;
    padding: 16px 20px;
    color: #34d399;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
    margin: 16px 0;
}

.status-box-error {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: 14px;
    padding: 16px 20px;
    color: #f87171;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.15);
    margin: 16px 0;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #07050d;
}
::-webkit-scrollbar-thumb {
    background: rgba(168, 85, 247, 0.3);
    border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(236, 72, 153, 0.5);
}
</style>
""",
        unsafe_allow_html=True,
    )