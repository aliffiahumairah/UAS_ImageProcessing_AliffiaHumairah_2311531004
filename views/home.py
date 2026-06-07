import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import CLASS_EMOJI, RECYCLE_INFO

def show_home():
    st.markdown("""
    <div style="text-align:center; padding:2rem 0 1rem 0;">
        <h1 style="font-size:3rem; font-weight:800; color:#1e3a5f; margin:0;">
            ♻️ GarbageAI Classifier
        </h1>
        <p style="font-size:1.2rem; color:#666; margin-top:0.5rem;">
            Klasifikasi Sampah Otomatis Berbasis Deep Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("🎯", "91.81%",      "Akurasi Model"),
        ("🗂️", "12",          "Kelas Sampah"),
        ("🖼️", "15.515",      "Total Dataset"),
        ("⚡", "MobileNetV2", "Arsitektur"),
    ]
    for col, (icon, val, label) in zip([c1, c2, c3, c4], stats):
        col.markdown(f"""
        <div style="background:linear-gradient(135deg,#667eea,#764ba2);
                    border-radius:12px; padding:1.2rem;
                    text-align:center; color:white; margin-bottom:1rem;">
            <div style="font-size:1.8rem;">{icon}</div>
            <div style="font-size:1.4rem; font-weight:bold;">{val}</div>
            <div style="font-size:0.85rem; opacity:0.9;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### 📌 Tentang Proyek")
        st.write("""
        **GarbageAI Classifier** adalah aplikasi kecerdasan buatan yang mampu
        mengklasifikasikan jenis sampah secara otomatis dari gambar menggunakan
        model **Convolutional Neural Network (CNN)** berbasis **MobileNetV2**
        dengan teknik *Transfer Learning*.

        Aplikasi ini diharapkan dapat membantu masyarakat dalam memilah sampah
        dengan benar sehingga mendukung program daur ulang dan pengelolaan
        sampah yang lebih baik.
        """)

        st.markdown("### 🎯 Fitur Aplikasi")
        fitur = [
            ("📁", "Upload gambar dari perangkat"),
            ("📷", "Deteksi langsung via kamera"),
            ("📊", "Confidence score semua kelas"),
            ("♻️", "Saran pengelolaan sampah"),
            ("🏷️", "Kategori organik/anorganik"),
            ("📈", "Dashboard statistik performa model"),
            ("🗂️", "Riwayat prediksi lengkap"),
        ]
        for icon, text in fitur:
            st.markdown(f"- {icon} {text}")

    with col2:
        st.markdown("### 👤 Identitas Mahasiswa")
        st.markdown("""
        <div style="background:#f8f9ff; border-radius:12px; padding:1.5rem;
                    border-left:4px solid #667eea;">
            <table style="width:100%; font-size:0.95rem; border-collapse:collapse;">
                <tr><td style="padding:4px 0;"><b>Nama</b></td>
                    <td>: Aliffia Humairah</td></tr>
                <tr><td style="padding:4px 0;"><b>NIM</b></td>
                    <td>: 2311531004</td></tr>
                <tr><td style="padding:4px 0;"><b>Prodi</b></td>
                    <td>: Informatika</td></tr>
                <tr><td style="padding:4px 0;"><b>Matkul</b></td>
                    <td>: DIF60202 Image Processing</td></tr>
                <tr><td style="padding:4px 0;"><b>Dosen</b></td>
                    <td>: Derisma, MT</td></tr>
                <tr><td style="padding:4px 0;"><b>Semester</b></td>
                    <td>: Genap 2025/2026</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🛠️ Teknologi")
        techs = ["Python 3.11", "TensorFlow 2.x", "MobileNetV2",
                 "Streamlit", "OpenCV", "Scikit-learn"]
        cols  = st.columns(2)
        for i, tech in enumerate(techs):
            cols[i % 2].markdown(
                f'<div style="background:#667eea; color:white; padding:4px 10px;'
                f'border-radius:20px; text-align:center; margin:3px; font-size:0.8rem;">'
                f'{tech}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗑️ Kelas Sampah yang Dapat Dideteksi")
    cols = st.columns(4)
    for i, (cls, emoji) in enumerate(CLASS_EMOJI.items()):
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:#f0f4ff; border-radius:10px; padding:0.8rem;
                        text-align:center; margin:4px; border:1px solid #dde3ff;">
                <div style="font-size:1.8rem;">{emoji}</div>
                <div style="font-weight:bold; color:#1e3a5f; font-size:0.9rem;">{cls}</div>
            </div>
            """, unsafe_allow_html=True)