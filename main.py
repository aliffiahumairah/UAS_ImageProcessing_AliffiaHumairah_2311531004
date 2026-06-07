import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="GarbageAI Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stSidebarNav"] {display: none !important;}
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
    .stDeployButton {display: none;}
    .block-container {padding-top: 2rem;}

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #2d5a8e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    div[data-testid="stRadio"] label {
        color: white !important;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.5rem 0 1rem 0;">
        <div style="font-size:3.5rem;">♻️</div>
        <h2 style="margin:0; color:white; font-size:1.5rem;">GarbageAI</h2>
        <p style="font-size:0.85rem; opacity:0.75; margin:0;">Garbage Classifier</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigasi",
        ["🏠  Beranda", "🔍  Prediksi", "📊  Dashboard"],
        label_visibility="hidden"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.8rem; opacity:0.7; text-align:center; padding:0.5rem 0;">
        <p style="margin:2px 0;">DIF60202 – Image Processing</p>
        <p style="margin:2px 0;">Semester Genap 2025/2026</p>
        <p style="margin:2px 0;">Universitas Andalas</p>
    </div>
    """, unsafe_allow_html=True)

if "Beranda" in page:
    from views.home import show_home
    show_home()
elif "Prediksi" in page:
    from views.predict import show_predict
    show_predict()
elif "Dashboard" in page:
    from views.dashboard import show_dashboard
    show_dashboard()