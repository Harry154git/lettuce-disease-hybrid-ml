# Lettuce Disease Severity Estimation & Classification using Hybrid Machine Learning

Mata Kuliah: Pembelajaran Mesin 1  
Pengembang: Harry Pratama Yunus  
NIM: 2310817210010  
Institusi: Fakultas Teknik, Universitas Lambung Mangkurat (FT ULM)

---

## 📌 Deskripsi Proyek
Proyek ini mengimplementasikan pendekatan **Hybrid Machine Learning** untuk mendeteksi dan mengklasifikasikan penyakit mata kodok (*Cercospora leaf spot*) pada tanaman selada (*Lactuca sativa*) hidroponik. Sistem ini menggabungkan kekuatan **Deep Learning** (Arsitektur *pre-trained* **EfficientNet-B3**) untuk ekstraksi fitur spasial otomatis secara mendalam, dengan keandalan algoritma **Machine Learning Klasik** (Support Vector Machine, Random Forest, dan K-Nearest Neighbors) sebagai pengambil keputusan akhir.

Untuk memastikan transparansi model (*explainable AI*), proyek ini mengintegrasikan modul **LIME (Local Interpretable Model-agnostic Explanations)** guna memvisualisasikan segmen piksel daun yang mendasari keputusan prediksi model, serta dideploy menjadi aplikasi web interaktif menggunakan **Streamlit**.

---

## 📂 Struktur Direktori Proyek & File

```
lettuce-disease-hybrid-ml/
├── data/                                # Dataset dan preprocessed features
│   ├── raw/                             # Dataset original (100 Sehat, 100 Sakit)
│   │   ├── healthy/
│   │   └── sick/
│   ├── split/                           # Train-test split 80-20
│   │   ├── train/
│   │   │   ├── healthy/
│   │   │   └── sick/
│   │   └── test/
│   │       ├── healthy/
│   │       └── sick/
│   ├── augmented/                       # Dataset hasil augmentasi (1.400 images)
│   │   ├── healthy/
│   │   ├── sick/
│   │   ├── train/
│   │   │   ├── healthy/
│   │   │   └── sick/
│   │   └── test/
│   │       ├── healthy/
│   │       └── sick/
│   └── processed/                       # Fitur numerik yang diekstraksi
│       ├── X_data.npy
│       └── y_data.npy
├── models/                              # Pre-trained models dan scaler
│   ├── efficientnet_extractor.keras     # Feature extractor CNN
│   ├── scaler.pkl                       # StandardScaler baseline
│   ├── scaler_tuned.pkl                 # StandardScaler optimized
│   ├── svm_model.pkl                    # SVM baseline
│   └── svm_model_tuned.pkl              # SVM optimized (BEST)
├── notebooks/                           # Jupyter notebooks untuk experiment pipeline
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_hybrid_model_training.ipynb
│   ├── 03_data_augmentation.ipynb
│   ├── 04_hybrid_model_training_augmented.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   └── 06_interpretation.ipynb
├── app.py                               # Streamlit web application
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
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
- Akurasi Akhir: **95.00%**
- Macro F1-Score: **0.95**

Classification Report:

```text
              precision    recall  f1-score   support

     Healthy       0.95      0.95      0.95        20
        Sick       0.95      0.95      0.95        20

    accuracy                           0.95        40
   macro avg       0.95      0.95      0.95        40
weighted avg       0.95      0.95      0.95        40
```

### 2. Random Forest

- Kombinasi Parameter Terbaik: `{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}`
- Akurasi Akhir: **90%**
- Macro F1-Score: **0.8997**

Classification Report:

```text
              precision    recall  f1-score   support

     Healthy       0.86      0.95      0.90        20
        Sick       0.94      0.85      0.89        20

    accuracy                           0.90        40
   macro avg       0.90      0.90      0.90        40
weighted avg       0.90      0.90      0.90        40
```

### 3. K-Nearest Neighbors (KNN)

- Kombinasi Parameter Terbaik: `{'n_neighbors': 7, 'weights': 'uniform'}`
- Akurasi Akhir: **80%**
- Macro F1-Score: **0.80**

Classification Report:

```text
========================================
Memulai Hyperparameter Tuning untuk KNN...
Fitting 5 folds for each of 8 candidates, totalling 40 fits
Kombinasi Parameter Terbaik: {'n_neighbors': 5, 'weights': 'distance'}
Akurasi (Tuned): 0.8000
Macro F1-Score (Tuned): 0.8000

Classification Report:
              precision    recall  f1-score   support

     Healthy       0.80      0.80      0.80        20
        Sick       0.80      0.80      0.80        20

    accuracy                           0.80        40
   macro avg       0.80      0.80      0.80        40
weighted avg       0.80      0.80      0.80        40
```

## 💻 Panduan Menjalankan Proyek

### 1. Setup Environment & Installation

**System Requirements**:
- Python 3.8+
- Preferably Linux (native atau WSL2)
- GPU recommended untuk feature extraction (optional but faster)

**Instalasi Langkah-Langkah**:

```bash

git clone https://github.com/Harry154git/lettuce-disease-hybrid-ml.git

# Navigate to project directory
cd lettuce-disease-hybrid-ml

# Create virtual environment
python3 -m venv env

# Activate virtual environment
source env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 2. Dependencies Overview

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow[and-cuda]` | Latest | Deep learning & EfficientNet-B3 |
| `scikit-learn` | Latest | SVM, Random Forest, KNN, GridSearchCV |
| `opencv-python` | Latest | Image processing & augmentation |
| `pandas` | Latest | Data manipulation |
| `numpy` | Latest | Numerical operations |
| `matplotlib` | Latest | Data visualization |
| `seaborn` | Latest | Statistical visualization |
| `jupyter` | Latest | Interactive notebooks |
| `streamlit` | Latest | Web application framework |
| `lime` | Latest | Model interpretability |
| `joblib` | Latest | Model serialization |
| `pillow` | Latest | Image I/O |

### 3. Struktur Dataset

Pastikan dataset sudah tersimpan dengan struktur berikut:

```
data/
├── raw/                    # Dataset original (100 Sehat, 100 Sakit)
│   ├── healthy/            # 100 gambar daun sehat
│   └── sick/               # 100 gambar daun sakit
├── split/                  # Train-test split (80-20)
│   ├── train/
│   │   ├── healthy/        # 80 training healthy
│   │   └── sick/           # 80 training sick
│   └── test/
│       ├── healthy/        # 20 test healthy
│       └── sick/           # 20 test sick
├── augmented/              # Dataset hasil augmentasi (1.400 images)
│   ├── healthy/            # 700 augmented healthy
│   ├── sick/               # 700 augmented sick
│   ├── train/
│   │   ├── healthy/        # 560 training augmented healthy
│   │   └── sick/           # 560 training augmented sick
│   └── test/
│       ├── healthy/        # 140 test augmented healthy
│       └── sick/           # 140 test augmented sick
└── processed/              # Extracted features (NumPy arrays)
    ├── X_data.npy          # Feature vectors
    └── y_data.npy          # Labels
```

### 4. Menjalankan Pipeline Eksperimen Secara Berurutan

Jalankan jupyter notebooks dalam urutan berikut untuk mereproduksi keseluruhan experiment:

#### **Tahap 1: Persiapan Data**

1. **`01_eda_and_preprocessing.ipynb`**
   - Exploratory Data Analysis (EDA) pada dataset mentah
   - Resizing gambar ke ukuran standar 224×224 piksel (sesuai input EfficientNet-B3)
   - Normalisasi nilai piksel ke rentang [0, 1]
   - Train-test split 80-20 ke folder `data/split/`
   - Output: 160 gambar training dan 40 gambar testing per kelas

2. **`02_hybrid_model_training.ipynb`**
   - Ekstraksi fitur dari dataset original menggunakan EfficientNet-B3 pre-trained
   - Pelatihan dan evaluasi baseline model (SVM, Random Forest, KNN)
   - Analisis performa tanpa augmentasi data
   - Hasil: Baseline accuracy ~85-88% (lihat tabel hasil di atas)

#### **Tahap 2: Augmentasi dan Optimisasi**

3. **`03_data_augmentation.ipynb`**
   - Implementasi 6 teknik augmentasi citra:
     - Flip horizontal & vertikal
     - Rotasi 90° dan 180°
     - Brightness adjustment
     - Contrast adjustment
   - Memperluas dataset dari 200 menjadi 1.400 gambar seimbang (700 per kelas)
   - Output: Augmented images di `data/augmented/`

4. **`04_hybrid_model_training_augmented.ipynb`**
   - Ekstraksi fitur dari augmented dataset
   - Pelatihan kembali model hybrid dengan data yang lebih besar
   - Perbandingan performa pre-augmentasi vs post-augmentasi
   - Observasi: Peningkatan akurasi dan generalisasi model

#### **Tahap 3: Fine-tuning dan Interpretabilitas**

5. **`05_hyperparameter_tuning.ipynb`**
   - GridSearchCV untuk optimasi parameter SVM, Random Forest, dan KNN
   - Validasi silang 5-fold untuk estimasi performa yang robust
   - Deteksi dan analisis overfitting
   - Pemilihan model terbaik berdasarkan F1-score macro
   - **Output Final**: Model SVM terbaik dengan akurasi **88.93%**

6. **`06_interpretation.ipynb`**
   - Implementasi LIME (Local Interpretable Model-agnostic Explanations)
   - Visualisasi segmen piksel yang berkontribusi pada prediksi
   - Explainable AI untuk meningkatkan transparansi model
   - Analisis fitur penting yang digunakan model untuk klasifikasi

### 4. Menjalankan Aplikasi Web Streamlit

Setelah model dilatih dan tersimpan, jalankan aplikasi web interaktif:

```bash
streamlit run app.py
```

Buka URL lokal yang muncul (biasanya `http://localhost:8501`) pada browser Anda untuk menguji deteksi kesehatan daun selada dengan file citra baru.

## 💡 Fitur Utama Aplikasi Web

- **Drag-and-Drop Image Uploader**: Memudahkan pengguna mengunggah citra daun selada (`.jpg`, `.png`, `.jpeg`).
- **Confidence Progress Bar**: Menampilkan persentase tingkat kepastian prediksi model SVM.
- **LIME Interpretability Visualizer**: Menampilkan segmentasi superpixel untuk memperlihatkan area daun yang paling berpengaruh pada prediksi penyakit.

---

## 📋 Model Architecture Pipeline

```
Input Image (224×224×3)
        ↓
[Preprocessing]
  - Resize
  - Normalize
        ↓
[EfficientNet-B3 Feature Extractor]
  - Pre-trained CNN
  - Global Average Pooling
  - Output: 1.536-dim vector
        ↓
[StandardScaler]
  - Standardize features
  - Mean = 0, Std = 1
        ↓
[SVM Classifier]
  - Kernel: RBF
  - C = 100, gamma = auto
  - Output: [Healthy / Sick]
        ↓
[Confidence Score]
  - Probability output
```

---

## 🎯 Key Insights & Results

### Data Augmentation Impact
- **Pre-augmentation**: 200 images → ~86% accuracy
- **Post-augmentation**: 1,400 images → ~89% accuracy
- **Improvement**: +3% accuracy with better generalization

### Model Performance Ranking
1. **SVM (RBF)**: 88.93% ✅ SELECTED
2. **Random Forest**: 84.29%
3. **KNN**: 77.86%

### Balanced Classification
- Healthy detection: 90% recall, 88% precision
- Sick detection: 88% recall, 90% precision
- **No significant class bias**

---

## 📚 Additional Resources

### References
- EfficientNet Paper: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- LIME Explanation: [Ribeiro et al., 2016](https://arxiv.org/abs/1602.04938)
- Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)

### Project Tools
- **Deep Learning**: TensorFlow/Keras
- **ML Algorithms**: Scikit-learn
- **Image Processing**: OpenCV, PIL
- **Visualization**: Matplotlib, Seaborn, Streamlit
- **Interpretability**: LIME

---

## 📝 Notes

- Model files are pre-trained and saved in `models/` directory
- Raw dataset expected in `data/raw/` (directory structure as shown above)
- For GPU acceleration, ensure CUDA and cuDNN are properly installed
- Streamlit app requires all pre-trained models to be available in `models/`

---