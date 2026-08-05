import streamlit as st

def load_styles():

    st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">

<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #020617;
    color: white;
}

.stApp {

    background:
        radial-gradient(circle at top left,
        rgba(0,255,200,0.08),
        transparent 25%),

        radial-gradient(circle at bottom right,
        rgba(0,140,255,0.08),
        transparent 25%),

        linear-gradient(
            135deg,
            #020617 0%,
            #06111f 40%,
            #020617 100%
        );
}

/* TITLE */

.main-title {

    font-size: 72px;
    font-weight: 900;

    text-align: center;

    margin-top: 30px;
    margin-bottom: 10px;

    background: linear-gradient(
        90deg,
        #00e5ff,
        #00ff99,
        #00c3ff
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    text-shadow:
        0 0 25px rgba(0,255,255,0.25);
}

.sub-title {

    text-align: center;

    font-size: 24px;

    color: #94a3b8;

    margin-bottom: 50px;
}

/* METRIC CARDS */

.metric-card {

    background: rgba(255,255,255,0.04);

    border: 1px solid rgba(0,255,170,0.15);

    backdrop-filter: blur(14px);

    border-radius: 24px;

    padding: 30px;

    box-shadow:
        0 0 25px rgba(0,255,170,0.08);

    transition: 0.3s ease;
}

.metric-card:hover {

    transform: translateY(-5px);

    box-shadow:
        0 0 40px rgba(0,255,170,0.18);
}

.metric-title {

    font-size: 18px;
    color: #94a3b8;
    font-weight: 500;
}

.metric-value {

    font-size: 48px;
    font-weight: 800;
    color: white;
}

/* BUTTON */

.stButton>button {

    background: linear-gradient(
        90deg,
        #00c6ff,
        #00ff99
    );

    color: black;

    border: none;

    border-radius: 16px;

    padding: 16px 34px;

    font-size: 20px;

    font-weight: 800;

    box-shadow:
        0 0 25px rgba(0,255,170,0.25);

    transition: 0.3s ease;
}

.stButton>button:hover {

    transform: scale(1.04);

    box-shadow:
        0 0 45px rgba(0,255,170,0.45);
}

/* SECTION HEADERS */

.section-title {

    font-size: 42px;
    font-weight: 800;

    margin-top: 50px;
    margin-bottom: 25px;

    background: linear-gradient(
        90deg,
        white,
        #9cecff
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* SUCCESS BOX */

.success-box {

    background: linear-gradient(
        90deg,
        rgba(0,255,140,0.18),
        rgba(0,180,255,0.12)
    );

    border-radius: 24px;

    padding: 24px;

    border: 1px solid rgba(0,255,170,0.22);

    font-size: 28px;

    font-weight: 700;

    color: #d1fae5;

    text-align: center;

    box-shadow:
        0 0 30px rgba(0,255,170,0.12);
}

/* HEALTH BOX */

.health-box {

    background: linear-gradient(
        90deg,
        rgba(0,255,120,0.15),
        rgba(0,180,255,0.15)
    );

    border-radius: 26px;

    padding: 34px;

    text-align: center;

    font-size: 34px;

    font-weight: 800;

    border: 1px solid rgba(0,255,170,0.18);

    box-shadow:
        0 0 35px rgba(0,255,170,0.14);

    margin-top: 20px;
    margin-bottom: 30px;
}

/* PLOTS */

.js-plotly-plot {

    border-radius: 24px !important;

    overflow: hidden !important;

    border: 1px solid rgba(0,255,170,0.12);

    box-shadow:
        0 0 25px rgba(0,255,170,0.08);
}

</style>
""", unsafe_allow_html=True)
