# Laporan Proyek Machine Learning - Muhammad Faris Akbar

## Domain Proyek

Dalam era digital modern, penggunaan perangkat *Generative AI* telah menjadi alat bantu utama bagi masyarakat termasuk mahasiswa dalam menunjang aktivitas akademik mereka. Berbagai macam *tools* AI digunakan untuk pencarian ide hingga *debugging* kode. Namun, pemanfaatan AI yang tidak terkendali atau tingkat ketergantungan yang terlalu tinggi dapat memicu dampak negatif, salah satunya adalah kelelahan mental atau *burnout* pada mahasiswa [1]. 

Proyek ini bertujuan untuk menganalisis dan memprediksi tingkat risiko *burnout* mahasiswa (*Burnout Risk Level*) berdasarkan pola penggunaan AI, kinerja akademik, serta kebiasaan belajar mereka. Menyelesaikan masalah ini sangat krusial karena berpotensi mempengaruhi kualitas pembelajaran dan mental mahasiswa terutama yang aktif dalam menggunakan AI.

## Business Understanding

### Problem Statements
- **Pernyataan Masalah 1:** Faktor-faktor apa saja dari pola penggunaan AI dan kebiasaan belajar yang memiliki korelasi paling signifikan terhadap tingkat risiko *burnout* mahasiswa?
- **Pernyataan Masalah 2:** Bagaimana cara membangun model *machine learning* yang akurat untuk mengklasifikasikan *Burnout Risk Level* (Low, Medium, High) menggunakan data perilaku mahasiswa tersebut?

### Goals
- **Jawaban Pernyataan Masalah 1:** Melakukan *Exploratory Data Analysis* (EDA) dan analisis korelasi Spearman untuk menyeleksi variabel prediktor terbaik yang memengaruhi risiko *burnout*.
- **Jawaban Pernyataan Masalah 2:** Membangun, melatih, dan mengevaluasi model klasifikasi *machine learning* dengan membandingkan performa model dasar (*baseline*) dan model hasil *hyperparameter tuning* untuk memilih model dengan metrik terbaik.

### Solution Statements
1. **Pendekatan Beberapa Algoritma:** Menggunakan dua algoritma *Machine Learning* yaitu **Random Forest Classifier** dan **XGBoost Classifier**. 
2. **Penanganan Imbalanced Data:** Melakukan proses *Upsampling* menggunakan `sklearn.utils.resample` karena distribusi target *Burnout Risk Level* didominasi oleh kelas *Medium*.
3. **Improvement Model:** Menggunakan `GridSearchCV` pada kedua model untuk mencari kombinasi parameter terbaik. Selanjutnya, membandingkan performa seluruh skema model untuk mendapatkan solusi paling optimal.

## Data Understanding

Data yang digunakan diambil dari *Kaggle* dengan judul dataset **AI Impact on Students**. Dataset ini merekam kebiasaan penggunaan AI dan dampaknya terhadap mahasiswa.
**Tautan Sumber Data:** [Kaggle - AI Impact on Students](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students/data)

### Jumlah Baris dan Kolom
Dataset asli memiliki jumlah baris sebanyak **50.000** baris dan memiliki **16** kolom/fitur. (Catatan: Setelah tahap *upsampling*, jumlah baris berubah untuk menyeimbangkan kelas target menjadi **63.432** baris data).

### Kondisi Data
- **Missing Value:** Setelah diperiksa menggunakan `isna().sum()`, tidak ditemukan adanya nilai kosong (*missing value*) pada seluruh kolom.
- **Data Duplikat:** Pemeriksaan menggunakan metode `duplicated()` menunjukkan tidak ada baris data yang terduplikasi.
- **Outlier:** Berdasarkan analisis rentang IQR (Interquartile Range) dan visualisasi *Boxplot* + *Violinplot*, ditemukan beberapa nilai pencilan (*outlier*). Namun, nilai-nilai tersebut masih berada pada batas empiris yang logis dan wajar untuk data perilaku manusia, sehingga diputuskan untuk **dipertahankan** (tidak di-*drop* atau dipangkas).

### Uraian Seluruh Fitur pada Data
Berikut adalah penjelasan detail untuk masing-masing kolom pada dataset:
1. **Student_ID:** Nomor identifikasi unik (integer) untuk setiap mahasiswa.
2. **Major_Category:** Bidang studi mahasiswa secara umum (kategorikal: STEM, Business, Humanities, Medical, Arts).
3. **Year_of_Study:** Tingkatan tahun akademik mahasiswa (kategorikal: Freshman, Sophomore, Junior, Senior, Graduate).
4. **Pre_Semester_GPA:** Nilai IPK mahasiswa pada awal semester (float).
5. **Post_Semester_GPA:** Nilai IPK mahasiswa pada akhir semester (float).
6. **Weekly_GenAI_Hours:** Rata-rata waktu (dalam jam) yang dihabiskan mahasiswa per minggu untuk menggunakan AI Generatif (float).
7. **Primary_Use_Case:** Tujuan atau fungsi utama mahasiswa dalam memanfaatkan AI, misalnya untuk perumusan draf atau pencarian jawaban langsung (kategorikal).
8. **Prompt_Engineering_Skill:** Tingkat kemahiran mahasiswa dalam merancang perintah atau *prompt* AI (kategorikal: Beginner, Intermediate, Advanced).
9. **Tool_Diversity:** Jumlah jenis/alat AI yang berbeda yang digunakan oleh mahasiswa (integer skala 1-5).
10. **Paid_Subscription:** Menunjukkan apakah mahasiswa berlangganan layanan AI berbayar (boolean: True/False).
11. **Traditional_Study_Hours:** Rata-rata jam per minggu yang didedikasikan untuk belajar secara konvensional tanpa AI (float).
12. **Perceived_AI_Dependency:** Penilaian mandiri mahasiswa terhadap tingkat ketergantungan mereka pada AI (integer skala 1-10).
13. **Institutional_Policy:** Kebijakan resmi dari institusi kampus terkait penggunaan AI (kategorikal: Allowed, Banned, dll).
14. **Anxiety_Level_During_Exams:** Tingkat kecemasan mahasiswa ketika menghadapi ujian (integer skala 1-10).
15. **Skill_Retention_Score:** Skor kemampuan mahasiswa dalam mengingat dan mempertahankan materi pelajaran (float skala 0-100).
16. **Burnout_Risk_Level:** Tingkat risiko kelelahan mental mahasiswa (kategorikal: Low, Medium, High). Ini bertindak sebagai **variabel target (y)** pada proyek ini.

## Data Preparation

Tahapan persiapan data (*Data Preparation*) dilakukan setelah data dipastikan bersih dari nilai Missing, Duplikat, dan Outlier, namun seperti yang telah dijabarkan sebelumnya, data yang digunakan ini bersih dari nilai Missing, Duplikat, dan Outlier nya pun dipertahankan, maka pada tahapan data preparation ini tidak akan dilakukan pre-processing untuk menangani 3 hal tersebut, namun langsung dilakukan tahapan preparation secara berurutan pada proyek ini adalah:

1. **Menghapus Kolom (Drop):** Menghapus fitur `Student_ID` karena hanya berperan sebagai pengidentifikasi baris dan tidak memiliki bobot informasi untuk dipelajari oleh model pembelajaran mesin.
2. **Encoding Data Kategorikal:**
   - *Label Encoding* pada target `Burnout_Risk_Level` agar berubah dari teks menjadi numerik (0, 1, 2).
   - *Ordinal Mapping* pada fitur yang memiliki tingkatan seperti `Prompt_Engineering_Skill` (0=Beginner, 1=Intermediate, 2=Advanced) dan `Year_of_Study` (1=Freshman, dst).
   - Mengubah tipe data *Boolean* (`Paid_Subscription` dkk) menjadi numerik `int` (1 dan 0).
   - Melakukan *One-Hot Encoding* menggunakan fungsi pandas `get_dummies()` pada sisa kolom kategorikal nominal untuk mengubahnya menjadi matriks biner.
3. **Feature Engineering:** Menciptakan fitur turunan baru yang logis untuk menambah variasi informasi, yaitu:
   - `gpa_difference`: Mengurangkan nilai IPK Akhir dengan IPK Awal untuk melihat progres nilai.
   - `hours_per_week`: Menjumlahkan jam belajar menggunakan AI dengan jam belajar tradisional untuk melihat total beban belajar mingguan.
4. **Analisis Korelasi & Seleksi Fitur:** Menghitung korelasi dengan metode *Spearman*. 
   - Ditemukan bahwa kolom `Pre_Semester_GPA` memiliki korelasi sangat ekstrem (mendekati 1.0) dengan `Post_Semester_GPA`, yang berpotensi menyebabkan redudansi (multikolinearitas), sehingga kolom `Pre_Semester_GPA` **dihapus**. 
   - Selanjutnya, 15 kolom dengan korelasi tertinggi terhadap target diseleksi untuk masuk ke tahap pemodelan.
5. **Pemilihan Fitur (X) dan Target (y):** - Variabel Independen **(X)** ditetapkan sebagai seluruh kolom dari dataset hasil seleksi *kecuali* kolom `Burnout_Risk_Level`. Variabel X memuat fitur perilaku dan akademik.
   - Variabel Dependen **(y)** ditetapkan murni pada kolom `Burnout_Risk_Level` yang merupakan label kelas yang harus diprediksi oleh model.
6. **Resampling Data (Upsampling):** Distribusi target menunjukkan dominasi di kelas *Medium*. Untuk mencegah model menjadi bias ke kelas mayoritas, dilakukan *upsampling* secara acak (menggandakan data minoritas) pada kelas *High* dan *Low* sehingga jumlah ketiga kelas menjadi sama persis.
7. **Train-Test Split:** Membagi keseluruhan dataset yang telah seimbang menjadi data latih (*training data*) sebesar 80% dan data uji (*testing data*) sebesar 20% menggunakan `train_test_split`.
8. **Standarisasi (Scaling):** Menggunakan `StandardScaler` pada data X_train dan X_test. Langkah ini sangat penting agar fitur yang memiliki rentang nilai berbeda (seperti IPK skala 4 dengan skor retensi skala 100) memiliki skala yang seragam (rata-rata 0, deviasi standar 1), sehingga algoritma model tidak terdistorsi.

## Modeling

Proyek ini menggunakan dua algoritma berbasis pohon (*tree-based algorithm*) untuk menyelesaikan masalah klasifikasi.

### Model 1: Random Forest Classifier
- **Pembahasan Cara Kerja:** Random Forest adalah algoritma ansambel (*ensemble*) yang menggunakan teknik *Bagging* (Bootstrap Aggregating). Algoritma ini membangun banyak pohon keputusan (*Decision Trees*) secara independen menggunakan sampel acak dari data. Saat melakukan klasifikasi, setiap pohon akan memberikan prediksi (seperti melakukan voting), dan kelas dengan suara terbanyak (*majority vote*) akan dipilih sebagai hasil prediksi akhir.
- **Pembahasan Parameter:** - **Parameter Baseline (Inisialisasi Awal):** Menggunakan `n_estimators = 100` (jumlah pohon) dan `random_state = 42`. Parameter `max_depth` (kedalaman pohon) dibiarkan pada setelan **default** (None), yang berarti pohon akan tumbuh terus hingga semua daun (*leaves*) menjadi murni (pure).
  - **Parameter Terbaik (Tuned):** Setelah dilakukan *GridSearchCV*, didapatkan parameter terbaik: `{'max_depth': 7, 'min_samples_split': 5, 'n_estimators': 100}`.
- **Kelebihan/Kekurangan:** Kelebihannya adalah sangat tangguh terhadap *outlier* dan mampu mengurangi varians (*overfitting*) berkat sistem *voting* dari banyak pohon. Kekurangannya adalah proses pelatihan membutuhkan waktu komputasi yang lebih lama dibandingkan algoritma pohon tunggal.

### Model 2: XGBoost Classifier (Extreme Gradient Boosting)
- **Pembahasan Cara Kerja:** XGBoost adalah algoritma ansambel yang bekerja dengan prinsip *Boosting*. Berbeda dengan Random Forest yang membangun pohon secara paralel dan mandiri, XGBoost membangun pohon keputusannya secara sekuensial (berurutan). Pohon kedua dibangun khusus untuk memperbaiki kesalahan (*residual error*) yang dibuat oleh pohon pertama, dan seterusnya. Pembelajaran ini dioptimasi menggunakan metode *Gradient Descent*.
- **Pembahasan Parameter:**
  - **Parameter Baseline (Inisialisasi Awal):** Menggunakan `eval_metric = 'mlogloss'` (metrik evaluasi standar untuk klasifikasi multi-kelas pada XGBoost) dan `random_state = 42`. Parameter *learning rate* dan *depth* menggunakan nilai **default** bawaan *library*.
  - **Parameter Terbaik (Tuned):** Setelah dilakukan *GridSearchCV*, didapatkan parameter terbaik: `{'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200}`.
- **Kelebihan/Kekurangan:** Kelebihannya memiliki performa akurasi komputasi yang sangat cepat serta fungsi penalti regulerisasi bawaan. Kekurangannya adalah sangat sensitif terhadap *outlier* dan rentan mengalami *overfitting* jika parameter *learning rate* tidak diatur dengan cermat.

## Evaluation

Tahap ini mengevaluasi kinerja seluruh skema pemodelan (Baseline dan Tuned) menggunakan metrik untuk masalah klasifikasi, yaitu **Accuracy** (rasio tebakan benar secara keseluruhan), **Precision** (akurasi prediksi kelas positif), **Recall** (kemampuan model mendeteksi kelas positif sebenarnya), dan **F1-Score** (rata-rata harmonis presisi dan *recall*).

### Paparan Hasil Evaluasi
Untuk komparasi, nilai *Precision*, *Recall*, dan *F1-Score* yang ditampilkan di tabel bawah ini diambil berdasarkan perhitungan rata-rata murni per kelas (*Macro Avg*) yang didapat dari fungsi `classification_report`.

| Model Pemodelan | Skema Parameter | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | Baseline (Default `max_depth`) | **71% (0.71)** | **0.71** | **0.71** | **0.71** |
| **XGBoost** | Baseline | 52% (0.52) | 0.51 | 0.52 | 0.51 |
| **Random Forest** | Tuned (`max_depth=7`) | 43% (0.43) | 0.41 | 0.42 | 0.40 |
| **XGBoost** | Tuned (`learning_rate=0.2, max_depth=7`) | 60% (0.60) | 0.60 | 0.60 | 0.60 |

**Analisis Perbandingan Skema Model:**
Dari tabel di atas, **Random Forest (Baseline)** secara mutlak merupakan model terbaik. Menariknya, ketika dilakukan *Hyperparameter Tuning* dengan membatasi kedalaman maksimal pohon menjadi 7 (`max_depth=7`), akurasi Random Forest justru anjlok secara drastis dari 71% menjadi 43%. Ini menandakan bahwa dataset sangat kompleks, sehingga ketika pohon tidak diizinkan tumbuh secara maksimal (*default*), model mengalami *underfitting* (gagal menangkap informasi esensial dari fitur-fitur yang ada).

### Hubungan dengan Business Understanding
Analisis akhir untuk menyelaraskan evaluasi model terhadap rumusan bisnis proyek:

- **Apakah sudah menjawab setiap Problem Statement?** Sudah terjawab. Melalui tahap persiapan dan seleksi fitur korelasi, kita berhasil mengidentifikasi bahwa jam penggunaan AI, *Anxiety Level*, dan keragaman *tools* merupakan metrik kunci (Pernyataan 1). Kita juga telah sukses merancang perbandingan algoritma ansambel guna mendapatkan model dengan klasifikasi yang optimal (Pernyataan 2).
- **Apakah berhasil mencapai setiap Goals yang diharapkan?**
  Berhasil. Analisis EDA dan korelasi memberikan wawasan (*insights*) jernih terkait pemicu *burnout*. Proyek ini juga sukses mengevaluasi dua algoritma dalam dua skema parameter (baseline dan tuned) untuk mendapatkan arsitektur model dengan performa paling maksimal (Mencapai *Accuracy*, *Precision*, *Recall*, dan *F1-Score* rata-rata sebesar 71%).
- **Apakah setiap Solution Statement yang direncakan berdampak?**
  Sangat berdampak. Penggunaan teknik *Upsampling* secara fundamental berhasil mengeliminasi sifat bias pada model; terbukti dari nilai *Precision* dan *Recall* yang seimbang (71%) pada laporan klasifikasi Model Terbaik. Jika solusi ini tidak diterapkan, model mungkin hanya akan pandai mengenali kelas "Medium". Selain itu, penerapan eksperimen *baseline* vs *tuning* berdampak sangat vital, karena mampu menyelamatkan proyek dari memilih model yang mengalami *underfitting* pasca-*tuning*. Oleh karena itu, *Random Forest Baseline* resmi ditetapkan sebagai solusi akhir bisnis untuk mendeteksi profil kelelahan mental mahasiswa di institusi pendidikan.

## Referensi
[1] A. . Wafiq, A. Syawal, I. ., and Z. A. Ahmad, “Burnout Akademik di Era Digital dan Persepsi Mahasiswa terhadap Peluang Pemanfaatan Artificial Intelligence untuk Deteksi Dini”, JSIT, vol. 5, no. 3, pp. 426–434, Nov. 2025.
https://www.yarsi.ac.id/ketergantungan-ai-ancam-daya-pikir-siswa-akademisi-ingatkan-risiko-cognitive-debt

[2] S. Nagi, "AI Impact on Students Dataset," *Kaggle*, 2026. [Online]. Available: https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students
