import streamlit as st
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model
from PIL import Image

# Setup Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Penyakit Selada", layout="centered")

@st.cache_resource
def load_all_models():
    
    extractor = load_model('models/efficientnet_extractor.keras')
    
    scaler_model = joblib.load('models/scaler.pkl') 
    classifier = joblib.load('models/svm_model.pkl') 
    
    return extractor, scaler_model, classifier

extractor, scaler, svm = load_all_models()
CLASSES = ['Sehat (Healthy)', 'Sakit (Sick)']

st.title("🥬 Klasifikasi Penyakit Daun Selada")
st.write("Aplikasi ini menggunakan Hybrid AI (EfficientNet-B3 + SVM) untuk mendeteksi apakah daun selada hidroponik sehat atau terjangkit penyakit.")

# Komponen Upload File
uploaded_file = st.file_uploader("Unggah gambar daun selada...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    st.write("Menganalisis gambar...")
    
    # Preprocessing
    img_array = np.array(image)
    if img_array.shape[2] == 4: # Jika punya alpha channel (PNG)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    img_resized = cv2.resize(img_array, (224, 224))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0) # Ubah ke (1, 224, 224, 3)
    
    # Ekstraksi dan Klasifikasi
    features = extractor.predict(img_batch, verbose=0)
    features_scaled = scaler.transform(features)
    
    prediction = svm.predict(features_scaled)[0]
    probabilities = svm.predict_proba(features_scaled)[0]
    
    # Hasil
    confidence = np.max(probabilities) * 100
    hasil_kelas = CLASSES[prediction]
    
    st.subheader(f"Hasil Analisis: **{hasil_kelas}**")
    st.progress(int(confidence))
    st.write(f"Tingkat Keyakinan: {confidence:.2f}%")