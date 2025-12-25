# <h1 align="center"> HEART DISEASE CLASSIFICATION </h1>

<div align="center">
  
![Banner](https://github.com/user-attachments/assets/f963b763-14ed-4678-a3d0-1a497670eefe)
  <p><i>Sistem deteksi dini risiko kesehatan jantung berbasis Deep Learning & Transfer Learning.</i></p>
</div>

---

### 📝 Deskripsi Project
Project ini dikembangkan sebagai solusi cerdas dalam bidang teknologi kesehatan (*Health-Tech*) untuk mengklasifikasikan risiko penyakit jantung pada individu. Dengan memanfaatkan data tabular medis, sistem ini mampu memberikan prediksi probabilitas risiko yang dapat digunakan sebagai alat pendukung keputusan awal. Fokus utama riset ini adalah membandingkan performa arsitektur Neural Network murni dengan metode Transfer Learning pada data tabular guna menangani kompleksitas data medis.

---

### 📂 Dataset & Preprocessing
Dataset yang digunakan berisi **445,132 entri** dengan **40 kolom fitur** yang mencakup profil demografi, kondisi kesehatan kronis, dan kebiasaan gaya hidup.

**Tahapan Preprocessing:**
1. **Handling Missing Values:** Melakukan imputasi nilai median untuk data numerik dan mode untuk data kategorikal.
2. **Label Encoding:** Mengonversi data kategori menjadi format numerik yang dapat diproses oleh Neural Network.
3. **Feature Scaling:** Menggunakan *StandardScaler* untuk menyeragamkan rentang data numerik agar proses *gradient descent* lebih stabil.
4. **Handling Class Imbalance:** Mengimplementasikan **Class Weights (Balanced)** pada fungsi *loss* untuk memastikan model tetap sensitif terhadap kelas minoritas (positif serangan jantung).

---

### 🧠 Arsitektur Model
Terdapat tiga model berbeda yang diimplementasikan dalam proyek ini:

1. **Model 1: Multilayer Perceptron (MLP) - Base Model**
   Arsitektur *Feedforward Neural Network* murni yang dibangun dari awal dengan optimasi *Class Weights* untuk menangani ketidakseimbangan data.
2. **Model 2: TabNet - Pretrained (Transfer Learning 1)**
   Arsitektur *attention-based* khusus data tabular yang menggunakan *Self-Supervised Pretraining* untuk menangkap pola fitur secara otomatis sebelum tahap klasifikasi.
3. **Model 3: pretrained embedding + Neural Network - Pretrained (Transfer Learning 2)**
   Model berbasis ekstraksi fitur mendalam (Feature Extractor) dengan arsitektur yang dioptimalkan untuk performa stabil pada klasifikasi biner medis.

---

### 📊 Exploratory Data Analysis (EDA)
Analisis awal dilakukan untuk memahami distribusi data dan faktor risiko utama.

| Distribusi Target | Pengaruh BMI |
| :---: | :---: |
| ![Target](Media/distribusi_target.png) | ![BMI](Media/korelasi.png) |
| *Visualisasi ketidakseimbangan data target.* | *Korelasi antara indeks massa tubuh dengan risiko.* |

---

### 📈 Grafik Performa (Loss & Accuracy)
Visualisasi proses pembelajaran setiap model selama tahap pelatihan dan validasi.

| Grafik Performa Model 1 | Grafik Performa Model 2 | Grafik Performa Model 3 |
| :---: | :---: | :---: |
| ![LossAcc1](Media/accuracy_MLP.png) | ![LossAcc2](Media/accuracy_Tabnet.png) | ![LossAcc3](Media/accuracy_NN.png) |

---

### 🧩 Confusion Matrix
Evaluasi detail untuk melihat presisi prediksi pada tiap kelas:

| Model 1: Base MLP | Model 2: TabNet | Model 3: pretrained embedding + Neural Network |
| :---: | :---: | :---: |
| ![CM1](Media/confusion_matrix_MLP.png) | ![CM2](Media/confusion_matrix_Tabnet.png) | ![CM3](Media/confusion_matrix_Style_NN.png) |

---

### 📉 Perbandingan Hasil Evaluasi
Ringkasan performa akhir berdasarkan data uji (*test set*) menggunakan metrik klasifikasi lengkap:

| Arsitektur Model | Accuracy | Loss | Prec. | Rec. | F1 | Hasil Analisis |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model 1 (Base MLP)** | 78% | **0.4139** | 0.18 | **0.80** | 0.30 | Memiliki Recall tertinggi, sangat krusial untuk menjaring pasien sakit agar tidak terlewat (False Negative rendah). |
| **Model 2 (TabNet)** | **83%** | 0.4191 | **0.21** | 0.76 | **0.33** | Unggul dalam akurasi dan presisi sistem, paling baik dalam meminimalisir kesalahan prediksi pada orang sehat. |
| **Model 3 (pretrained embedding + Neural Network)** | 80% | 0.4231 | 0.19 | 0.78 | 0.31 | Menunjukkan performa yang seimbang antara kemampuan deteksi (Recall) dan akurasi sistem secara keseluruhan. |
---
### 🔍 Analisis Perbandingan & Kesimpulan
Setiap arsitektur model menunjukkan karakteristik performa yang berbeda sesuai dengan metode pendekatannya terhadap data medis:

*   **Model 1 (Base MLP):** Menitikberatkan pada aspek **Medical Safety** dengan nilai **Recall tertinggi (0.80)** dan **Loss terendah (0.4139)**. Kemampuannya dalam menekan angka *False Negative* sangat krusial dalam dunia medis agar pasien yang benar-benar sakit tidak terlewat saat diagnosis awal.
*   **Model 2 (TabNet):** Unggul sebagai model yang paling **Presisi dan Akurat (83% Acc)**. model ini paling cerdas dalam mengenali profil fitur tabular secara mendalam, sehingga sangat baik dalam meminimalisir kesalahan prediksi pada individu yang sehat (*False Positive*).
*   **Model 3 (pretrained embedding + Neural Network):** Memberikan performa yang **Stabil dan Seimbang**. Model ini merupakan "jalan tengah" yang solid, menawarkan tingkat akurasi dan kemampuan deteksi yang konsisten, menjadikannya model yang paling reliabel untuk generalisasi data di berbagai kondisi.

**Kesimpulan Akhir:**
Pemilihan model terbaik bergantung pada prioritas klinis. Dalam konteks klasifikasi penyakit jantung, **Model 1** dianggap paling optimal karena sensitivitasnya yang tinggi (Recall) dalam menjaring pasien berisiko, sementara **Model 2** menjadi pilihan terbaik jika tujuannya adalah efisiensi sistem dengan tingkat akurasi tertinggi.

---

### 💻 Panduan Menjalankan Secara Lokal
Ikuti langkah berikut untuk menjalankan sistem website di perangkat Anda:

1. **Persiapan Folder:**
   Pastikan file model (`.pth`, `.zip`) dan scaler (`.pkl`) berada di dalam folder `src/models/`.
2. **Install Dependensi:**
   Gunakan PDM atau Pip untuk menginstal library:
   ```bash
   pip install streamlit pandas numpy torch pytorch-tabnet joblib scikit-learn
   ```
3. **Jalankan Aplikasi:**
   Eksekusi perintah berikut melalui terminal di root direktori proyek:
   ```bash
   streamlit run src/app.py
   ```

---

### 🛠️ Teknologi yang Digunakan
*   **Framework:** PyTorch, PyTorch-TabNet, Streamlit
*   **Library:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Joblib
*   **Tools:** Google Colab, VS Code, PDM

---

### 👤 Profil Pengembang

**Ahmad Naufal Luthfan Marzuqi**  
🆔 **NIM:** 202210370311072  
🎓 **Program Studi:** Teknik Informatika  
🏛️ **Universitas Muhammadiyah Malang**

---

### 🔗 Live Demo
Coba aplikasi deteksi jantung secara langsung di sini:  
**[Klasifikasi Risiko Penyakit Jantung App](https://klasifikasi-resiko-penyakit-jantung.streamlit.app/)**

