## Lettuce Disease Severity Estimation & Classification using Hybrid Machine Learning

Mata Kuliah: Pembelajaran Mesin 1  
Pengembang: Harry Pratama Yunus  
NIM: 2310817210010  
Institusi: Fakultas Teknik, Universitas Lambung Mangkurat (FT ULM)

---

## 📌 Deskripsi Proyek
Proyek ini mengimplementasikan pendekatan **Hybrid Machine Learning** untuk mendeteksi dan mengklasifikasikan penyakit mata kodok (*Cercospora leaf spot*) pada tanaman selada (*Lactuca sativa*) hidroponik. Sistem ini menggabungkan kekuatan **Deep Learning** (Arsitektur *pre-trained* **EfficientNet-B3**) untuk ekstraksi fitur spasial otomatis secara mendalam, dengan keandalan algoritma **Machine Learning Klasik** (Support Vector Machine, Random Forest, dan K-Nearest Neighbors) sebagai pengambil keputusan akhir.

Untuk memastikan transparansi model (*explainable AI*), proyek ini mengintegrasikan modul **LIME (Local Interpretable Model-agnostic Explanations)** guna memvisualisasikan segmen piksel daun yang mendasari keputusan prediksi model, serta dideploy menjadi aplikasi web interaktif menggunakan **Streamlit**.

---

## 📂 Struktur Direktori Proyek
Berikut adalah susunan folder proyek yang rapi dan terstruktur:

```text
lettuce-disease-severity-estimation-ml/
├── data/
│   ├── raw/                             # Dataset asli (100 Sehat, 100 Sakit)
│   └── augmented/                       # Dataset hasil augmentasi (700 Sehat, 700 Sakit)
├── models/
│   ├── efficientnet_extractor.keras     # Bobot model pengekstraksi fitur CNN
│   ├── scaler_tuned.pkl                 # Objek standardisasi fitur StandardScaler
│   └── svm_model_tuned.pkl              # Model akhir SVM terbaik hasil GridSearchCV
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb   # Eksplorasi & resizing citra awal ke 224x224
│   ├── 02_hybrid_model_training.ipynb   # Pelatihan baseline menggunakan data asli
│   ├── 03_data_augmentation.ipynb       # Skrip augmentasi citra fotometris & geometris
│   ├── 04_hybrid_model_training_augmented.ipynb # Pelatihan model menggunakan data augmentasi
│   └── 05_hyperparameter_tuning.ipynb   # Optimasi parameter GridSearchCV & analisis overfitting
├── app.py                               # Skrip utama aplikasi web Streamlit
├── requirements.txt                     # Daftar dependensi library Python
└── README.md                            # Dokumentasi repositori
```

## 🔧 Alur Preprocessing & Ekstraksi Fitur

- **Image Resizing & Color Conversion**: Mengonversi ruang warna citra dari BGR bawaan OpenCV ke RGB standar, lalu menyeragamkan ukuran menjadi 224 × 224 piksel dengan normalisasi nilai piksel ke rentang [0, 1].
- **Data Augmentation**: Mengatasi keterbatasan data asli melalui 6 teknik manipulasi geometris dan fotometris (flip horizontal, flip vertical, rotasi 90°, rotasi 180°, brightness adjustment, dan contrast adjustment). Total dataset meluas dari 200 gambar menjadi 1.400 gambar seimbang (700 Healthy, 700 Sick).
- **Deep Feature Extraction**: Citra dilewatkan pada lapisan konvolusi pre-trained EfficientNet-B3 dengan lapisan ujung klasifikasi yang dipotong (`include_top=False`) dan dihubungkan pada Global Average Pooling. Proses ini memampatkan gambar menjadi vektor fitur 1.536 dimensi.
- **Feature Scaling**: Menyetarakan distribusi fitur menggunakan `StandardScaler` agar memiliki nilai rata-rata 0 dan standar deviasi 1 sebelum diumpankan ke model ML.

## 📊 Hasil Evaluasi & Perbandingan Model (UAS - Tuned)

Evaluasi performa model dilakukan pada data uji steril sebanyak 280 sampel (140 Healthy, 140 Sick) menggunakan validasi silang 5-Fold dan GridSearchCV.

### 1. Support Vector Machine (SVM) - MODEL TERBAIK

- Kombinasi Parameter Terbaik: `{'C': 100, 'gamma': 'auto', 'kernel': 'rbf'}`
- Akurasi Akhir: **88.93%**
- Macro F1-Score: **0.89**

Classification Report:

```text
              precision    recall  f1-score   support

     Healthy       0.88      0.90      0.89       140
        Sick       0.90      0.88      0.89       140

    accuracy                           0.89       280
   macro avg       0.89      0.89      0.89       280
weighted avg       0.89      0.89      0.89       280
```

### 2. Random Forest

- Kombinasi Parameter Terbaik: `{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}`
- Akurasi Akhir: **84.29%**
- Macro F1-Score: **0.84**

### 3. K-Nearest Neighbors (KNN)

- Kombinasi Parameter Terbaik: `{'n_neighbors': 7, 'weights': 'uniform'}`
- Akurasi Akhir: **77.86%**
- Macro F1-Score: **0.78**

## 💻 Panduan Menjalankan Proyek

### 1. Kloning Repositori & Instalasi Dependensi

Pastikan Anda menjalankan proyek ini di lingkungan Linux (native Linux atau WSL2) untuk optimalisasi kinerja komputasi.

```bash
git clone https://github.com/Harry154git/lettuce-disease-severity-estimation-ml.git
cd lettuce-disease-severity-estimation-ml
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 2. Menjalankan Pipeline Eksperimen secara Berurutan

Eksekusi file notebook di folder `notebooks/` dengan urutan berikut untuk mereproduksi hasil eksperimen:

1. `01_eda_and_preprocessing.ipynb`: Mengolah data mentah awal.
2. `02_hybrid_model_training.ipynb`: Pengujian performa data asli (baseline accuracy ~86.36%).
3. `03_data_augmentation.ipynb`: Memproduksi gambar data augmentasi.
4. `04_hybrid_model_training_augmented.ipynb`: Pelatihan model dengan data augmentasi.
5. `05_hyperparameter_tuning.ipynb`: Menjalankan GridSearchCV untuk model akhir terbaik `svm_model_tuned.pkl`.

### 3. Menjalankan Aplikasi Web Streamlit

```bash
streamlit run app.py
```

Buka URL lokal yang muncul (biasanya `http://localhost:8501`) pada browser Anda untuk menguji deteksi kesehatan daun selada dengan file citra baru.

## 💡 Fitur Utama Aplikasi Web

- **Drag-and-Drop Image Uploader**: Memudahkan pengguna mengunggah citra daun selada (`.jpg`, `.png`, `.jpeg`).
- **Confidence Progress Bar**: Menampilkan persentase tingkat kepastian prediksi model SVM.
- **LIME Interpretability Visualizer**: Menampilkan segmentasi superpixel untuk memperlihatkan area daun yang paling berpengaruh pada prediksi penyakit.