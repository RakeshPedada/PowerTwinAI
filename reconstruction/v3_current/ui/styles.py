"""
ui/styles.py
------------
Premium Engineering & AI Workstation Design System for PowerTwinAI.
Precision scientific visualization theme with refined dark surfaces,
subtle borders, clean typography, and high-contrast telemetry.
"""

import streamlit as st


def load_styles() -> None:
    """Inject the professional engineering workstation CSS."""
    st.markdown(
        """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global Color System (User Specification) ── */
:root {
    --bg-main: #0B1120;
    --bg-sidebar: #10192B;
    --bg-card: #151F33;
    --bg-secondary: #1B2942;
    --bg-elevated: #1B2942;
    --primary-accent: #249BFF;
    --secondary-accent: #8057FF;
    --primary-hover: #1D82D6;
    --secondary-hover: #6D46E6;
    --border: #263653;
    --border-hover: #364B73;
    --text-main: #F1F5FC;
    --text-secondary: #A5B4CC;
    --text-muted: #64748B;
    --success: #22C987;
    --warning: #F5B942;
    --error: #F05D73;
    --info-accent: #249BFF;
}

html, body, [data-testid="stApp"] {
    font-family: 'Inter', 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    background-color: var(--bg-main) !important;
    background-image: 
        radial-gradient(at 10% 10%, rgba(36, 155, 255, 0.06) 0px, transparent 50%),
        radial-gradient(at 90% 15%, rgba(128, 87, 255, 0.05) 0px, transparent 50%),
        radial-gradient(at 50% 90%, rgba(34, 201, 135, 0.03) 0px, transparent 50%) !important;
    background-attachment: fixed !important;
    color: var(--text-main) !important;
    letter-spacing: -0.01em;
}

/* Hide default Streamlit decoration */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Main Container Spacing ── */
.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
}

/* ── Custom Scrollbars ── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-main);
}
::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--border-hover);
}

/* ── Persistent Sidebar Design ── */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 2px 0 12px rgba(0, 0, 0, 0.25) !important;
}

[data-testid="stSidebar"] .block-container {
    padding: 1.25rem 1rem !important;
}

/* Brand Header in Sidebar */
.sidebar-brand-container {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    margin-bottom: 18px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
}

.brand-icon-box {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    background: linear-gradient(135deg, #7C5CFC, #20C6A0);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    box-shadow: 0 0 12px rgba(124, 92, 252, 0.35);
}

.brand-text-col {
    display: flex;
    flex-direction: column;
}

.brand-app-title {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-main);
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.brand-app-title span {
    color: var(--primary-accent);
}

.brand-app-sub {
    font-size: 11px;
    color: var(--text-secondary);
    font-weight: 500;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

/* Navigation items in Sidebar */
.sidebar-nav-header {
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 10px 14px 6px;
    margin-top: 6px;
}

/* Hide default radio circle completely in sidebar */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
    display: flex !important;
    align-items: center !important;
    padding: 9px 14px !important;
    margin-bottom: 4px !important;
    border-radius: 8px !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-left: 3px solid transparent !important;
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    background: var(--bg-card) !important;
    color: var(--text-main) !important;
    border-color: var(--border) !important;
}

/* Active navigation item with primary indicator */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
    background: rgba(36, 155, 255, 0.12) !important;
    border: 1px solid rgba(36, 155, 255, 0.3) !important;
    border-left: 3px solid var(--primary-accent) !important;
    color: var(--text-main) !important;
    font-weight: 600 !important;
}

/* Telemetry Footer in Sidebar */
.sidebar-footer-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    margin-top: 20px;
}

.sidebar-footer-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 6px;
}

.sidebar-footer-row:last-child {
    margin-bottom: 0;
}

.sidebar-footer-val {
    font-weight: 600;
    color: var(--text-main);
}

/* ── Top Header Bar ── */
.app-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 22px;
    margin-bottom: 24px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
}

.topbar-left {
    display: flex;
    align-items: center;
    gap: 14px;
}

.topbar-page-icon {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}

.topbar-title-wrap h2 {
    margin: 0 !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    color: var(--text-main) !important;
    letter-spacing: -0.02em;
}

.topbar-title-wrap p {
    margin: 2px 0 0 !important;
    font-size: 12px !important;
    color: var(--text-secondary) !important;
}

.topbar-right-telemetry {
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── Professional Card & Panel System ── */
.eng-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

.eng-card:hover {
    border-color: var(--border-hover);
}

.eng-card-elevated {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 16px;
}

.card-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}

.card-header-left {
    display: flex;
    align-items: center;
    gap: 10px;
}

.card-header-left h3 {
    margin: 0 !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    color: var(--text-main) !important;
}

.card-header-left p {
    margin: 2px 0 0 !important;
    font-size: 12px !important;
    color: var(--text-secondary) !important;
}

/* ── Stat Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 16px;
    margin-bottom: 20px;
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 105px;
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.metric-card:hover {
    border-color: var(--border-hover);
    transform: translateY(-1px);
}

.metric-label-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.metric-value {
    font-size: 24px;
    font-weight: 700;
    color: var(--text-main);
    letter-spacing: -0.02em;
    margin: 6px 0;
    font-family: 'JetBrains Mono', monospace;
}

.metric-sub {
    font-size: 11px;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Pill Badges ── */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.badge-primary {
    background: rgba(36, 155, 255, 0.15);
    color: #93c5fd;
    border: 1px solid rgba(36, 155, 255, 0.35);
}

.badge-secondary {
    background: rgba(128, 87, 255, 0.15);
    color: #c4b5fd;
    border: 1px solid rgba(128, 87, 255, 0.35);
}

.badge-success {
    background: rgba(34, 201, 135, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(34, 201, 135, 0.35);
}

.badge-info {
    background: rgba(36, 155, 255, 0.15);
    color: #7dd3fc;
    border: 1px solid rgba(36, 155, 255, 0.35);
}

.badge-warning {
    background: rgba(245, 185, 66, 0.15);
    color: #fde047;
    border: 1px solid rgba(245, 185, 66, 0.35);
}

.badge-error {
    background: rgba(240, 93, 115, 0.15);
    color: #fca5a5;
    border: 1px solid rgba(240, 93, 115, 0.35);
}

/* ── Interactive Workflow Steps ── */
.workflow-step-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
}

.step-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}

.step-number {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--primary-accent);
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 700;
}

.step-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-main);
    letter-spacing: -0.01em;
}

.step-desc {
    font-size: 12px;
    color: var(--text-secondary);
    margin-left: 40px;
    margin-top: -8px;
    margin-bottom: 16px;
}

/* ── Backend Selection Cards ── */
.backend-card-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 20px;
}

.backend-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.15s ease;
}

.backend-card:hover {
    border-color: var(--border-hover);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

.backend-card-active {
    border-color: var(--primary-accent) !important;
    background: var(--bg-elevated) !important;
    box-shadow: 0 0 0 1px var(--primary-accent), 0 8px 24px rgba(124, 92, 252, 0.2) !important;
}

/* ── Pipeline Stage Visualization ── */
.pipeline-tracker {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 20px;
    overflow-x: auto;
}

.pipeline-stage {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    position: relative;
    flex: 1;
    min-width: 120px;
}

.pipeline-dot {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--bg-elevated);
    border: 2px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 700;
    color: var(--text-secondary);
    z-index: 2;
    margin-bottom: 6px;
}

.stage-active .pipeline-dot {
    background: var(--primary-accent);
    border-color: var(--primary-accent);
    color: #fff;
    box-shadow: 0 0 10px rgba(124, 92, 252, 0.5);
}

.stage-done .pipeline-dot {
    background: var(--secondary-accent);
    border-color: var(--secondary-accent);
    color: #0B0F19;
}

.pipeline-stage-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
}

.stage-active .pipeline-stage-label {
    color: var(--text-main);
}

.stage-done .pipeline-stage-label {
    color: var(--secondary-accent);
}

/* ── Streamlit Element Overrides ── */
/* Streamlit Buttons */
button[kind="primary"] {
    background-color: var(--primary-accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    transition: background-color 0.15s ease, transform 0.1s ease !important;
}

button[kind="primary"]:hover {
    background-color: var(--primary-hover) !important;
    transform: translateY(-1px) !important;
}

button[kind="secondary"] {
    background-color: var(--bg-elevated) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}

button[kind="secondary"]:hover {
    background-color: var(--border) !important;
    border-color: var(--border-hover) !important;
}

/* Streamlit Inputs & Textareas */
input, textarea, [data-baseweb="select"] {
    background-color: var(--bg-card) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

input:focus, textarea:focus {
    border-color: var(--primary-accent) !important;
    box-shadow: 0 0 0 1px var(--primary-accent) !important;
}

/* Streamlit Tabs */
[data-baseweb="tab-list"] {
    background-color: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 4px !important;
    gap: 4px !important;
    margin-bottom: 20px !important;
}

[data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    border: none !important;
    background: transparent !important;
}

[aria-selected="true"] {
    background-color: var(--bg-elevated) !important;
    color: var(--text-main) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
}

/* Streamlit Dataframe Dark Theme (Never White!) */
[data-testid="stDataFrame"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Streamlit Expanders */
[data-testid="stExpander"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 16px !important;
}

[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: var(--text-main) !important;
}

/* Code & Log Blocks */
pre, code {
    font-family: 'JetBrains Mono', monospace !important;
    background-color: #070A11 !important;
    color: #A5F3FC !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
</style>
""",
        unsafe_allow_html=True,
    )
