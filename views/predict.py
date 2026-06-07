import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_model, predict, CLASS_EMOJI, RECYCLE_INFO, CATEGORY_MAP

def show_predict():
    st.markdown("## 🔍 Halaman Prediksi")
    st.write("Upload gambar sampah atau gunakan kamera untuk diklasifikasikan oleh AI.")

    model, class_names = load_model()

    tab1, tab2 = st.tabs(["📁 Upload Gambar", "📷 Kamera"])

    with tab1:
        uploaded = st.file_uploader(
            "Pilih gambar (JPG / PNG / JPEG)",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded:
            img = Image.open(uploaded)
            _run_prediction(img, model, class_names)

    with tab2:
        st.info("📷 Arahkan kamera ke objek sampah lalu klik **Take Photo**")
        camera = st.camera_input("Ambil Foto")
        if camera:
            img = Image.open(camera)
            _run_prediction(img, model, class_names)


def _run_prediction(img, model, class_names):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📷 Gambar Input**")
        st.image(img, use_column_width=True)

    with st.spinner("🤖 AI sedang menganalisis gambar..."):
        result = predict(model, class_names, img)

    pred_class = result['class']
    confidence = result['confidence']
    time_ms    = result['time_ms']
    emoji      = CLASS_EMOJI.get(pred_class, '🗑️')
    info       = RECYCLE_INFO.get(pred_class, '-')

    if confidence >= 0.85:
        color, status = "#28a745", "Tinggi ✅"
    elif confidence >= 0.60:
        color, status = "#ffc107", "Sedang ⚠️"
    else:
        color, status = "#dc3545", "Rendah ❌"

    cat_name, cat_emoji, cat_color, cat_desc = CATEGORY_MAP.get(
        pred_class, ('Tidak Diketahui', '❓', '#999999', '-')
    )

    with col2:
        st.markdown("**🎯 Hasil Deteksi**")

        st.markdown(f"""
        <div style="background:#f0f4ff; border-left:5px solid {color};
                    border-radius:10px; padding:1.5rem; margin-bottom:1rem;">
            <div style="font-size:3rem; text-align:center;">{emoji}</div>
            <h2 style="text-align:center; color:#1e3a5f; margin:0.5rem 0;">
                {pred_class.upper()}
            </h2>
            <div style="text-align:center; margin-bottom:0.5rem;">
                <span style="background:{color}; color:white; padding:4px 14px;
                             border-radius:20px; font-size:0.95rem; font-weight:bold;">
                    {confidence:.1%} — {status}
                </span>
            </div>
            <p style="text-align:center; color:#777; font-size:0.85rem; margin:0;">
                ⏱️ Waktu inferensi: <b>{time_ms:.1f} ms</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="border-radius:12px; padding:1.2rem;
                    background:linear-gradient(135deg, {cat_color}22, {cat_color}44);
                    border:2px solid {cat_color}; text-align:center;
                    margin-bottom:1rem;">
            <div style="font-size:2rem;">{cat_emoji}</div>
            <h3 style="color:{cat_color}; margin:0.3rem 0; font-size:1.2rem;">
                {cat_name.upper()}
            </h3>
            <p style="color:#555; font-size:0.85rem; margin:0;">{cat_desc}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#e8f5e9; border-radius:10px;
                    padding:1rem; border-left:4px solid #28a745;">
            <b>♻️ Saran Pengelolaan Sampah:</b><br>
            <span style="color:#2e7d32; font-size:0.95rem;">{info}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📊 Confidence Score Semua Kelas**")

    probs_df = pd.DataFrame({
        'Kelas':      list(result['all_probs'].keys()),
        'Confidence': [v * 100 for v in result['all_probs'].values()]
    }).sort_values('Confidence', ascending=True)

    bar_colors = ['#667eea' if k == pred_class else '#dde3ff'
                  for k in probs_df['Kelas']]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(probs_df['Kelas'], probs_df['Confidence'],
                   color=bar_colors, edgecolor='white', height=0.6)
    ax.set_xlabel('Confidence (%)', fontsize=11)
    ax.set_title('Probability per Class', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 110)
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, probs_df['Confidence']):
        ax.text(bar.get_width() + 0.8,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("**📋 Ringkasan Hasil**")
    col_a, col_b, col_c = st.columns(3)

    col_a.markdown(f"""
    <div style="background:#f0f4ff; border-radius:10px; padding:1rem; text-align:center;">
        <div style="font-size:1.5rem;">{emoji}</div>
        <div style="font-weight:bold; color:#1e3a5f;">Jenis Sampah</div>
        <div style="color:#667eea; font-size:1.1rem;">{pred_class}</div>
    </div>
    """, unsafe_allow_html=True)

    col_b.markdown(f"""
    <div style="background:#f0f4ff; border-radius:10px; padding:1rem; text-align:center;">
        <div style="font-size:1.5rem;">{cat_emoji}</div>
        <div style="font-weight:bold; color:#1e3a5f;">Kategori</div>
        <div style="color:{cat_color}; font-size:1.1rem;">{cat_name}</div>
    </div>
    """, unsafe_allow_html=True)

    col_c.markdown(f"""
    <div style="background:#f0f4ff; border-radius:10px; padding:1rem; text-align:center;">
        <div style="font-size:1.5rem;">🎯</div>
        <div style="font-weight:bold; color:#1e3a5f;">Confidence</div>
        <div style="color:{color}; font-size:1.1rem;">{confidence:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        'Jenis Sampah': f"{emoji} {pred_class}",
        'Kategori':     f"{cat_emoji} {cat_name}",
        'Confidence':   f"{confidence:.1%}",
        'Status':       status,
        'Waktu (ms)':   f"{time_ms:.1f}",
    })