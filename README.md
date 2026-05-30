# Laporan Proyek Machine Learning - Muhammad Faris Akbar

## Domain Proyek
Dalam era digital modern, penggunaan perangkat *Generative AI* telah menjadi alat bantu utama bagi mahasiswa dalam menunjang aktivitas akademik mereka. Berbagai macam *tools* AI digunakan untuk *copywriting*, merangkum bacaan, pencarian ide, hingga *debugging* kode. Namun, pemanfaatan AI yang tidak terkendali atau tingkat ketergantungan yang terlalu tinggi dapat memicu dampak negatif, salah satunya adalah kelelahan mental atau *burnout* pada mahasiswa. 

Proyek ini bertujuan untuk menganalisis dan memprediksi tingkat risiko *burnout* mahasiswa (*Burnout Risk Level*) berdasarkan pola penggunaan AI, kinerja akademik, serta kebiasaan belajar mereka. Menyelesaikan masalah ini menjadi sangat krusial bagi institusi pendidikan agar dapat merumuskan kebijakan penggunaan AI yang tepat dan memberikan intervensi dini bagi mahasiswa yang rentan mengalami *burnout*. 

**Referensi Riset Terkait:**
- A. . Wafiq, A. Syawal, I. ., and Z. A. Ahmad, “Burnout Akademik di Era Digital dan Persepsi Mahasiswa terhadap Peluang Pemanfaatan Artificial Intelligence untuk Deteksi Dini”, JSIT, vol. 5, no. 3, pp. 426–434, Nov. 2025.
- https://www.yarsi.ac.id/ketergantungan-ai-ancam-daya-pikir-siswa-akademisi-ingatkan-risiko-cognitive-debt

## Business Understanding
### Problem Statements
- **Pernyataan Masalah 1:** Faktor-faktor apa saja dari pola penggunaan AI dan kebiasaan belajar yang memiliki korelasi paling signifikan terhadap tingkat risiko *burnout* mahasiswa?
- **Pernyataan Masalah 2:** Bagaimana cara membangun model *machine learning* yang akurat untuk mengklasifikasikan *Burnout Risk Level* (Low, Medium, High) menggunakan data mahasiswa tersebut?

### Goals
- **Jawaban Pernyataan Masalah 1:** Melakukan *Exploratory Data Analysis* (EDA) dan analisis korelasi Spearman untuk menyeleksi variabel prediktor terbaik yang memengaruhi risiko *burnout*.
- **Jawaban Pernyataan Masalah 2:** Membangun, melatih, dan mengevaluasi model klasifikasi *machine learning* dengan membandingkan performa model dasar (*baseline*) dan model hasil *hyperparameter tuning*.

### Solution Statements
1. **Pendekatan Beberapa Algoritma:** Menggunakan dua algoritma *Machine Learning* yang terbukti tangguh untuk data tabular, yaitu **Random Forest Classifier** dan **XGBoost Classifier**. 
2. **Penanganan Imbalanced Data:** Melakukan proses *Upsampling* menggunakan `sklearn.utils.resample` karena distribusi target *Burnout Risk Level* didominasi oleh kelas *Medium*.
3. **Improvement Model:** Menggunakan `GridSearchCV` pada kedua model untuk mencari kombinasi parameter terbaik. Selanjutnya, membandingkan performa antara model *baseline* (tanpa tuning) dan model *tuned* untuk memilih solusi akhir yang paling optimal.

## Data Understanding
Data yang digunakan diambil dari *Kaggle* dengan judul dataset **AI Impact on Students** oleh Lavesh Jadon. 
**Sumber Dataset:** [Kaggle - AI Impact on Students](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students/data)

### Variabel-variabel pada dataset:
- **Major_Category**, **Year_of_Study**, **Pre_Semester_GPA**, **Post_Semester_GPA**: Data akademik mahasiswa.
- **Weekly_GenAI_Hours**, **Primary_Use_Case**, **Prompt_Engineering_Skill**, **Tool_Diversity**, **Paid_Subscription**: Data pola penggunaan AI.
- **Traditional_Study_Hours**, **Perceived_AI_Dependency**, **Anxiety_Level_During_Exams**, **Skill_Retention_Score**: Data psikologis dan kebiasaan belajar.
- **Burnout_Risk_Level**: Tingkat risiko *burnout* (Low, Medium, High). Ini adalah **variabel target**.

### Exploratory Data Analysis (EDA)
- **Cek Nilai Kosong & Duplikat:** Tidak ditemukan nilai kosong (Missing Values) maupun duplikasi data.
- **Cek Outlier:** Berdasarkan analisis rentang IQR dan *Boxplot*, nilai pencilan (*outlier*) yang ada masih berada pada batas empiris yang wajar dan logis, sehingga diputuskan untuk **dipertahankan**.
- **Analisis Tren:** Melalui *Point Plot*, ditemukan bahwa tingginya jam penggunaan AI, tingginya kecemasan ujian, jumlah variasi alat AI, dan ketergantungan pada AI berkorelasi linier dengan tingginya tingkat *burnout*. Sebaliknya, tingginya jam belajar konvensional berkorelasi dengan rendahnya tingkat *burnout*.

## Data Preparation
Teknik penyiapan data yang diterapkan adalah:
1. **Menghapus Kolom Identifier**: Menghapus fitur `Student_ID`.
2. **Encoding Data Kategorikal**:
   - *Label Encoding* pada target `Burnout_Risk_Level`.
   - *Ordinal Mapping* pada `Prompt_Engineering_Skill` dan `Year_of_Study`.
   - *One-Hot Encoding* (`pd.get_dummies()`) pada kolom kategorikal nominal lainnya dengan `drop_first=False` untuk memaksimalkan pembelajaran pada *Tree-based model*.
3. **Feature Engineering**: Menciptakan dua fitur turunan, yaitu `gpa_difference` (Selisih GPA) dan `hours_per_week` (Total jam belajar AI + Tradisional).
4. **Feature Selection (Korelasi Spearman)**: Ditemukan multikolinearitas ekstrem pada `Pre_Semester_GPA`, sehingga fitur tersebut **dihapus**. Setelah itu, dipilih **Top 15 fitur** dengan korelasi tertinggi terhadap variabel target.
5. **Resampling (Upsampling)**: Kelas minoritas (*Low* dan *High*) diperbanyak (duplikasi acak) agar jumlah datanya setara dengan kelas *Medium*.
6. **Train-Test Split**: Membagi dataset menjadi data latih (80%) dan data uji (20%).
7. **Standarisasi (Scaling)**: Menggunakan `StandardScaler` untuk menyamakan rentang skala pada data numerik.

## Modeling

Proyek ini menggunakan dua algoritma berbasis pohon (*tree-based*):
1. **Random Forest Classifier**: Algoritma ansambel berbasis *bagging* yang membangun banyak pohon keputusan secara acak. Sangat tangguh dan stabil, tetapi bisa lambat pada dataset masif.
2. **XGBoost Classifier**: Algoritma ansambel berbasis *boosting* yang memperbaiki *error* pohon sebelumnya. Sangat cepat, namun rentan *overfitting* jika parameter tidak tepat.

### Proses Improvement & Pemilihan Model Terbaik
Proses eksperimen dibagi menjadi dua tahap, yakni pelatihan model dasar (*Baseline*) dan pelatihan dengan *Hyperparameter Tuning* menggunakan `GridSearchCV`.

* **XGBoost:** Akurasi *baseline* berada di **51.77%**. Setelah di-*tuning* dengan parameter terbaik (`learning_rate`: 0.2, `max_depth`: 7, `n_estimators`: 200), performanya meningkat menjadi **60.14%**.
* **Random Forest:** Akurasi *baseline* (dengan nilai *default* `max_depth=None`) sangat tinggi di **70.75%**. Namun, setelah di-*tuning* dengan parameter grid terbatas (`max_depth`: 7, `min_samples_split`: 5, `n_estimators`: 100), akurasinya turun tajam ke **42.54%**. Hal ini membuktikan bahwa pembatasan kedalaman pohon membuat Random Forest kehilangan informasi vital dari data (*underfitting*).

**Kesimpulan Pemilihan Model:** Berdasarkan eksperimen di atas, model **Random Forest (Baseline)** dipilih sebagai solusi akhir terbaik karena memberikan metrik evaluasi (akurasi) tertinggi dan paling mampu memetakan pola data dengan benar dibandingkan semua iterasi model lainnya.

## Evaluation

Evaluasi dilakukan untuk klasifikasi multi-kelas menggunakan matriks evaluasi **Accuracy, Precision, Recall, dan F1-Score**. Formula dari masing-masing metrik adalah:
- **Accuracy**: $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$ (Rasio prediksi benar keseluruhan).
- **Precision**: $Precision = \frac{TP}{TP + FP}$ (Rasio ketepatan kelas positif).
- **Recall**: $Recall = \frac{TP}{TP + FN}$ (Sensitivitas model menangkap data kelas positif).
- **F1-Score**: $F1-Score = 2 \times \frac{Precision \times Recall}{Precision + Recall}$ (Rata-rata harmonis *precision* dan *recall*).

### Hasil Evaluasi Model Terbaik (Random Forest Baseline):
Model **Random Forest Baseline** memperoleh performa terbaik pada data uji dengan ringkasan sebagai berikut:
- **Akurasi Akhir:** **70.75%**

**Detail Klasifikasi:**
- **Kelas High (Risiko Burnout Tinggi):** Memiliki *Recall* yang sangat baik, yaitu **0.80 (80%)** dan *Precision* **0.76**. Nilai *F1-Score* mencapai **0.78**. Hal ini sangat memuaskan, mengingat mendeteksi mahasiswa dengan risiko *burnout* tinggi adalah prioritas utama proyek.
- **Kelas Low (Risiko Burnout Rendah):** *Precision* **0.74** dan *Recall* **0.73**, dengan nilai *F1-Score* **0.73**.
- **Kelas Medium (Risiko Burnout Menengah):** Merupakan kelas yang paling sulit ditebak oleh model (dibandingkan dua kelas lainnya) dengan *F1-Score* **0.60**.

**Kesimpulan Akhir:** Proyek ini sukses mendemonstrasikan bahwa algoritma *Random Forest* tanpa pembatasan kedalaman pohon mampu mengungguli *XGBoost* pada dataset ini. Model telah siap memberikan prediksi yang dapat diandalkan, khususnya untuk memitigasi mahasiswa yang diprediksi masuk ke dalam kategori *High Burnout Risk*.
