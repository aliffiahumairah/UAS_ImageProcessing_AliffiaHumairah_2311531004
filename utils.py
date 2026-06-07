import numpy as np
from PIL import Image
import json, time, os
import streamlit as st

# Ganti tensorflow dengan keras langsung
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    import keras

# ── Path ──────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR        = os.path.join(BASE_DIR, 'model')
MODEL_PATH       = os.path.join(MODEL_DIR, 'classifier_model.h5')
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, 'class_names.json')
HISTORY_PATH     = os.path.join(MODEL_DIR, 'history.json')

# ── Emoji per kelas ───────────────────────────────────
CLASS_EMOJI = {
    'battery':     '🔋',
    'biological':  '🍃',
    'brown-glass': '🟤',
    'cardboard':   '📦',
    'clothes':     '👕',
    'green-glass': '🟢',
    'metal':       '🔩',
    'paper':       '📄',
    'plastic':     '🧴',
    'shoes':       '👟',
    'trash':       '🗑️',
    'white-glass': '⚪',
}

RECYCLE_INFO = {
    'battery':     '⚠️ Bawa ke tempat daur ulang baterai khusus. Jangan buang ke tempat sampah biasa!',
    'biological':  '🌱 Bisa dijadikan kompos. Buang ke tempat sampah organik.',
    'brown-glass': '♻️ Masukkan ke bank sampah kaca. Bisa didaur ulang.',
    'cardboard':   '📦 Lipat dan masukkan ke tempat sampah kertas/kardus.',
    'clothes':     '👗 Donasikan jika masih layak pakai, atau bawa ke bank sampah tekstil.',
    'green-glass': '♻️ Masukkan ke bank sampah kaca. Bisa didaur ulang.',
    'metal':       '🔧 Bawa ke pengepul logam atau bank sampah.',
    'paper':       '📄 Masukkan ke tempat sampah kertas. Bisa didaur ulang.',
    'plastic':     '🧴 Cuci bersih lalu masukkan ke tempat sampah plastik.',
    'shoes':       '👟 Donasikan jika masih layak, atau bawa ke bank sampah.',
    'trash':       '🗑️ Masukkan ke tempat sampah residu (tidak dapat didaur ulang).',
    'white-glass': '♻️ Masukkan ke bank sampah kaca. Bisa didaur ulang.',
}

CATEGORY_MAP = {
    'battery':     ('Anorganik', '♻️', '#2196F3', 'Sampah anorganik yang dapat didaur ulang secara khusus.'),
    'biological':  ('Organik',   '🌱', '#4CAF50', 'Sampah organik yang dapat terurai secara alami.'),
    'brown-glass': ('Anorganik', '♻️', '#2196F3', 'Sampah anorganik yang dapat didaur ulang.'),
    'cardboard':   ('Anorganik', '♻️', '#2196F3', 'Sampah anorganik yang dapat didaur ulang.'),
    'clothes':     ('Anorganik', '♻️', '#2196F3', 'Sampah anorganik, donasikan jika masih layak.'),
    'green-glass': ('Anorganik', '♻️', '#2196F3', 'Sampah anorganik yang dapat didaur ulang.'),
    'metal':       ('Anorganik', '♻️', '#2196F3', 'Sampah anorganik bernilai ekonomis tinggi.'),
    'paper':       ('Anorganik', '♻️', '#2196F3', 'Sampah anorganik yang dapat didaur ulang.'),
    'plastic':     ('Anorganik', '♻️', '#2196F3', 'Sampah anorganik, kurangi penggunaannya.'),
    'shoes':       ('Anorganik', '♻️', '#2196F3', 'Sampah anorganik, donasikan jika masih layak.'),
    'trash':       ('Residu',    '🗑️', '#F44336', 'Sampah residu, tidak dapat didaur ulang.'),
    'white-glass': ('Anorganik', '♻️', '#2196F3', 'Sampah anorganik yang dapat didaur ulang.'),
}

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    return model, class_names

def load_history():
    try:
        with open(HISTORY_PATH) as f:
            return json.load(f)
    except:
        return None

def preprocess_image(img: Image.Image):
    img = img.resize((224, 224)).convert('RGB')
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(model, class_names, img: Image.Image):
    processed = preprocess_image(img)
    start     = time.time()
    preds     = model.predict(processed, verbose=0)[0]
    elapsed   = (time.time() - start) * 1000
    pred_idx  = int(np.argmax(preds))
    return {
        'class':      class_names[pred_idx],
        'confidence': float(preds[pred_idx]),
        'all_probs':  {class_names[i]: float(preds[i]) for i in range(len(class_names))},
        'time_ms':    elapsed
    }