import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_history

def show_dashboard():
    st.markdown("## 📊 Dashboard Statistik")

    hist      = load_history()
    best_acc  = max(hist['val_accuracy']) if hist else 0.9181
    best_loss = min(hist['val_loss'])     if hist else 0.2688
    history   = st.session_state.get('history', [])

    st.markdown("### 📌 Ringkasan Model")
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("🧪", len(history),        "Gambar Diuji"),
        ("🎯", f"{best_acc:.2%}",   "Akurasi Terbaik"),
        ("📉", f"{best_loss:.4f}",  "Val Loss Terbaik"),
        ("🏗️", "MobileNetV2",       "Arsitektur"),
    ]
    for col, (icon, val, label) in zip([c1, c2, c3, c4], metrics):
        col.markdown(f"""
        <div style="background:linear-gradient(135deg,#667eea,#764ba2);
                    border-radius:12px; padding:1.2rem; text-align:center;
                    color:white; margin-bottom:1rem;">
            <div style="font-size:1.5rem;">{icon}</div>
            <div style="font-size:1.3rem; font-weight:bold;">{val}</div>
            <div style="font-size:0.8rem; opacity:0.9;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    if hist:
        st.markdown("### 📈 Kurva Training")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        axes[0].plot(hist['accuracy'],     label='Train',      linewidth=2, color='#667eea')
        axes[0].plot(hist['val_accuracy'], label='Validation', linewidth=2, color='#f093fb')
        axes[0].fill_between(range(len(hist['accuracy'])),
                             hist['accuracy'], hist['val_accuracy'],
                             alpha=0.1, color='#667eea')
        axes[0].set_title('Model Accuracy', fontweight='bold', fontsize=13)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0.7, 1.0])

        axes[1].plot(hist['loss'],     label='Train',      linewidth=2, color='#667eea')
        axes[1].plot(hist['val_loss'], label='Validation', linewidth=2, color='#f093fb')
        axes[1].set_title('Model Loss', fontweight='bold', fontsize=13)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Training History — MobileNetV2 Garbage Classifier',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("### 🏗️ Informasi Model")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Parameter Training**")
        st.markdown("""
        | Parameter | Nilai |
        |---|---|
        | Arsitektur | MobileNetV2 |
        | Input Size | 224 × 224 px |
        | Optimizer | Adam |
        | Loss Function | Categorical Crossentropy |
        | Total Epochs | 19 (Early Stop) |
        | Batch Size | 32 |
        | Dropout | 0.5 |
        """)
    with col2:
        st.markdown("**Informasi Dataset**")
        st.markdown("""
        | Keterangan | Nilai |
        |---|---|
        | Nama Dataset | Garbage Classification |
        | Total Gambar | 15.515 |
        | Jumlah Kelas | 12 |
        | Train Split | 80% |
        | Val Split | 20% |
        | Akurasi Terbaik | 91.81% |
        | Best Epoch | 14 |
        """)

    st.markdown("### 🗂️ Riwayat Prediksi Sesi Ini")
    if history:
        df = pd.DataFrame(history)
        st.dataframe(df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Distribusi Kelas Terprediksi**")
            dist = df['Jenis Sampah'].value_counts()
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            dist.plot(kind='bar', ax=ax2, color='#667eea', edgecolor='white')
            ax2.set_xlabel('Kelas')
            ax2.set_ylabel('Jumlah')
            ax2.set_title('Distribusi Prediksi', fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        with col2:
            st.markdown("**Status Confidence**")
            status_dist = df['Status'].value_counts()
            colors_pie  = ['#28a745', '#ffc107', '#dc3545'][:len(status_dist)]
            fig3, ax3   = plt.subplots(figsize=(6, 4))
            ax3.pie(status_dist.values, labels=status_dist.index,
                    autopct='%1.1f%%', colors=colors_pie,
                    startangle=90, wedgeprops={'edgecolor': 'white'})
            ax3.set_title('Status Confidence', fontweight='bold')
            st.pyplot(fig3)
            plt.close()

        if st.button("🗑️ Reset Riwayat Prediksi"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("💡 Belum ada prediksi. Coba upload gambar di halaman **Prediksi** terlebih dahulu!")