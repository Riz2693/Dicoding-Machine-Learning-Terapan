# Laporan Proyek Machine Learning - Muhammad Faris Akbar

## Project Overview
Dalam era perkembangan ilmu pengetahuan modern, jumlah publikasi ilmiah dan jurnal bertambah secara eksponensial setiap harinya [1]. 
Repositori literatur terbuka seperti arXiv memproses ribuan karya ilmiah baru setiap bulannya dan kini secara akumulatif telah menembus angka jutaan dokumen [2]. 
Bagi akademisi, peneliti, maupun mahasiswa, menemukan artikel ilmiah yang relevan dengan topik atau ide spesifik mereka sering kali menjadi tantangan berat akibat fenomena information overload (kelebihan informasi) [3].
Sistem pencarian literatur tradisional umumnya masih mengandalkan pencocokan kata kunci eksak (Lexical/Keyword Matching). Sistem ini memiliki kelemahan fundamental yang dikenal sebagai Vocabulary Mismatch dan Semantic Gap [4], [5]. 
Sebagai contoh, jika seorang pengguna memasukkan kueri penelusuran "mathematical theory of deep learning", sistem leksikal tradisional mungkin akan melewatkan artikel berkualitas yang menggunakan frasa sinonim seperti "neural networks calculus", karena ketiadaan kecocokan kata secara langsung.

Untuk menyelesaikan persoalan ini, proyek ini diinisiasi untuk membangun Sistem Rekomendasi berbasis **Content-Based Filtering (CBF)**. 
Proyek ini membandingkan pendekatan klasik (TF-IDF) dengan pendekatan menggunakan representasi bahasa **Deep Learning Embeddings (Sentence Transformers)**. 
Model berbasis *Transformer* diimplementasikan untuk mengekstrak makna semantik dan gagasan dari narasi abstrak jurnal, sehingga sistem mampu merekomendasikan literatur yang linier secara konteks ilmiah meskipun penulis jurnal menggunakan ragam kosakata yang sama sekali berbeda.

**Referensi:**

[1] L. Bornmann, R. Haunschild, and R. Mutz, "Growth rates of modern science: A latent piecewise regression analysis of global publication data," Scientometrics, vol. 126, no. 11, pp. 8915–8921, Nov. 2021.

[2] M. Krenn et al., "Forecasting the future of artificial intelligence with machine learning-based link prediction in an exponentially growing knowledge network," Nature Machine Intelligence, vol. 4, no. 12, pp. 1098–1106, Dec. 2022.

[3] S. Shariq, "Causes, consequences, and strategies to deal with information overload: A scoping review," International Journal of Information Management, vol. 74, p. 102713, Feb. 2024.

[4] Y. Kim, "Improving Query Representations for Dense Retrieval with Pseudo Relevance Feedback," IEEE Access, vol. 12, pp. 45120–45131, Mar. 2024.

[5] J. Lin and X. Wang, "Generative Retrieval Overcomes Limitations of Dense Retrieval in Scientific Contexts," in Proc. IEEE International Conference on Big Data and Information Analytics, 2025, pp. 312–319.


## Business Understanding
Proses klarifikasi masalah dan tujuan dalam pengembangan sistem rekomendasi ini didefinisikan sebagai berikut:

### Problem Statements

1. **Pernyataan Masalah 1:** Data mentah jurnal ilmiah (seperti arXiv) sangat kotor, dipenuhi dengan sintaks format matematika LaTeX, kode *backslash*, serta karakter *Unicode* asing yang berpotensi merusak ekstraksi fitur NLP jika diproses secara mentah. Bagaimana teknik prapemrosesan (*preprocessing*) yang optimal untuk menangani teks tersebut?
2. **Pernyataan Masalah 2:** Bagaimana cara membangun arsitektur sistem rekomendasi yang tidak hanya mencocokkan kata kunci eksak, tetapi mampu "memahami" makna semantik kueri untuk menemukan artikel yang relevan?
3. **Pernyataan Masalah 3:** Mengingat dataset arXiv tidak memiliki data historis preferensi atau *rating* dari pembaca, bagaimana cara mengevaluasi performa sistem rekomendasi ini secara objektif?

### Goals

1. **Jawaban Pernyataan Masalah 1:** Mengimplementasikan teknik pembersihan data (*Data Preparation*) berjenjang memanfaatkan kombinasi *Regular Expression* (Regex) tingkat lanjut dan normalisasi *Unidecode* untuk menstandarisasi sintaks LaTeX menjadi teks natural.
2. **Jawaban Pernyataan Masalah 2:** Membangun dan membandingkan dua pendekatan model *Content-Based Filtering*: model berbasis frekuensi (TF-IDF) dan model berbasis semantik (*Sentence Transformers* dengan K-Nearest Neighbors).
3. **Jawaban Pernyataan Masalah 3:** Merancang skema evaluasi *offline* melalui pendekatan *Dual-Metric* kuantitatif, menggunakan metrik *Cosine Similarity* (untuk mengukur kedekatan makna/semantik) dan *Jaccard Similarity* (untuk mengukur keberagaman variasi kosakata).

### Solution Approach

Untuk meraih objektif tersebut, diajukan 2 *solution approach* (algoritma sistem rekomendasi):

1. **Algoritma 1 (Baseline): TF-IDF + Cosine Similarity**
Pendekatan ini menggunakan representasi matriks statistik berbasis frekuensi (*Term Frequency - Inverse Document Frequency*). Teks diubah menjadi ruang vektor diskrit berdasarkan kemunculan kosakata. Rekomendasi diambil berdasarkan perkalian matriks *Cosine Similarity*.
2. **Algoritma 2 (Main Model): Deep Learning Embeddings (Sentence Transformers) + K-Nearest Neighbors (KNN)**
Pendekatan ini mendayagunakan *Pre-trained Model* arsitektur Transformer (`all-MiniLM-L6-v2`) untuk mengompresi teks dokumen ke dalam ruang vektor numerik berdimensi 384. Algoritma *K-Nearest Neighbors* difungsikan untuk menghitung jarak sudut (*Cosine*) kedekatan antar-dokumen.


## Data Understanding

Data yang digunakan merupakan kumpulan metadata karya ilmiah terbuka dari [Cornell University arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv/data) dengan total hingga 1.7 juta artikel yang diunduh langsung melalui repositori Kaggle. Mengingat *file* asli berupa JSON berukuran sangat besar (*Big Data*), ekstraksi data dilakukan menggunakan pola iterasi *Generator* (`yield`), dengan secara spesifik menyaring (*filtering*) data publikasi dari tahun **$\ge$ 2026** agar sistem difokuskan pada pemrosesan literatur sains termutakhir.

### Detail dan Penjelasan Seluruh Kolom/Fitur Dataset

Berdasarkan struktur JSON yang diunduh, didapatkan 237.226 total blok entitas artikel dengan setiap satu blok entitas artikel karya ilmiah memuat 14 variabel metadata. Berikut adalah penjabaran keseluruhan variabel yang direkam dalam dataset aslinya:

1. **`id`**: Identifier unik (kombinasi angka) untuk setiap artikel di arXiv. *String* ini sering digunakan sebagai ujung URL untuk mengakses makalah asli (contoh: *arxiv.org/abs/0704.0001*).
2. **`submitter`**: Nama pihak atau perwakilan penulis yang bertugas mengunggah dan mengirimkan dokumen ke repositori arXiv.
3. **`authors`**: Rangkaian nama lengkap para penulis artikel (dalam bentuk string tunggal mentah).
4. **`title`**: Judul resmi publikasi jurnal/penelitian. Fitur ini sangat kaya informasi namun sering disisipi *inline syntax* matematika/LaTeX.
5. **`comments`**: Catatan tambahan informal dari penulis, biasanya mendeskripsikan jumlah halaman, jumlah gambar, catatan *conference*, atau rilis versi.
6. **`journal-ref`**: Rujukan referensi jurnal fisik (jika makalah ini juga secara paralel diterbitkan di majalah penerbit eksternal/jurnal komersial).
7. **`doi`**: *Digital Object Identifier*, tautan permanen standar internasional untuk mengidentifikasi keberadaan dokumen di jagat maya.
8. **`report-no`**: Nomor laporan teknis institusi (*Institution Report Number*). Kolom ini kebanyakan terisi pada publikasi ilmu fisika energi tinggi atau riset nasional.
9. **`categories`**: Kode klasifikasi yang merujuk pada domain sub-ilmu pengetahuan spesifik (contoh: `hep-ph` untuk *High Energy Physics*, `cs.AI` untuk *Artificial Intelligence*).
10. **`license`**: Detail legal perizinan atau hak cipta akses publikasi (*Creative Commons* atau non-lisensi).
11. **`abstract`**: Teks paragraf ringkasan yang menjabarkan latar belakang, metodologi, dan kesimpulan penelitian. Ini adalah nyawa utama dari sistem NLP dalam proyek ini.
12. **`versions`**: Data *array* berbentuk JSON yang menyimpan histori rilis versi dokumen (seperti kapan *v1* dikirim, dan kapan revisi *v2* diterbitkan).
13. **`update_date`**: Tanggal terakhir kalinya repositori arXiv memperbarui atau mensinkronisasi data makalah ini di sistem (*YYYY-MM-DD*).
14. **`authors_parsed`**: Struktur data bentuk *list-of-lists* hasil pemrosesan internal Kaggle yang secara otomatis membedah kolom `authors` menjadi nama awalan, nama akhiran, dan afiliasi.

### Pemilihan Fitur (*Feature Selection*) dan Exploratory Data Analysis

Mengingat pendekatan pemodelan berfokus pada metode *Content-Based Filtering* yang menitikberatkan pada analisis semantik tulisan, pemuatan seluruh 14 kolom ke dalam *Pandas DataFrame* akan membuang memori *RAM* secara sia-sia. Oleh karena itu, fitur-fitur administratif seperti `id`, `submitter`, `comments`, `journal-ref`, `doi`, `report-no`, `license`, `versions`, dan `authors_parsed` dieliminasi/ *didrop* dari awal.

Dataset final dibatasi dan hanya **mempertahankan 5 fitur krusial**: `authors`, `title`, `categories`, `abstract`, dan `update_date`.

Dari kelima fitur tersebut, dilakukan **Exploratory Data Analysis (EDA)** dengan temuan berikut:

1. **Pemeriksaan *Missing Value* & Duplikasi:** Tidak ditemukan adanya nilai kosong (*NaN*). Namun, sistem deteksi menemukan adanya data ganda akibat histori `update_date` yang tumpang-tindih. Data duplikat ini didrop secara permanen.
2. **Distribusi Top-10 Kategori:** Hasil analisis visualisasi membuktikan bahwa bidang ilmu *Computer Science* dan fisika tingkat tinggi (*High Energy Physics*) mendominasi pasokan jurnal global pada filter tahun $>2026$.
<img width="868" height="547" alt="image" src="https://github.com/user-attachments/assets/1b01acde-2fb4-4f22-8638-9ca3cf09e630" />

3. **Ekstraksi Fitur Waktu:** Kolom `update_date` diformat menjadi atribut diskrit (`year` dan `month`) untuk melihat produktivitas submission jurnal bulanan para peneliti, yang menunjukkan adanya *seasonality* produktivitas riset.
<img width="868" height="547" alt="image" src="https://github.com/user-attachments/assets/694e21ac-acf5-4e90-9903-fc8c4c4a9af6" />



## Data Preparation

Tahapan persiapan data dilakukan secara berurutan (*pipeline*) untuk mentransformasi teks akademis yang sarat akan *noise* ke dalam bentuk teks bahasa natural yang dipahami komputer:

1. **Pembatasan Skala via Generator:** Memuat baris JSON menggunakan teknik Python `yield` dipadukan pemfilteran tahun. Ini krusial agar memori komputasi tidak mengalami *Crash* saat harus menangani data orisinal seberat puluhan Gigabyte.
2. **Case Folding:** Menurunkan seluruh karakter alfabet pada setiap kolom (`.str.lower()`) untuk menjaga stabilitas sensitivitas kapital model NLP.
3. **Menghapus Kolom yang Tidak Diperlukan (update_date):** Menghapus kolom update_date secara permanen menggunakan perintah drop(). Hal ini dilakukan karena informasi waktu dari kolom tersebut telah diekstrak dan diwakili secara sempurna oleh fitur turunan year dan month.
4. **Penghapusan Data Duplikat & Reset Index:** Menghapus 3 pasangan data duplikat yang terdeteksi secara aktual dari peninjauan subset kolom (authors, title, categories, abstract). Setelah duplikat dihapus, dilakukan pemanggilan fungsi reset_index(drop=True) agar penomoran urutan data kembali berurutan dari angka 0.

5. **Pembersihan Regex per-Kolom (Domain-Specific Cleaning):**
    * *`bersihkan_title`*: Menghapus tanda penanda *subscript* (`_`), *dollar* (`$`), dan aksen kurung kurawal (`{}`) yang merupakan formula *inline-math* LaTeX. (Alasan: Model bahasa akan bingung dan terdistraksi jika menemui variabel komputasi matematika di tengah judul teks).
    * *`bersihkan_abstract`*: Menghapus operator komparasi (`:=`) dan *backslash* (`\`).
    * *`bersihkan_categories`*: Mengganti spasi antar-kode keilmuan dengan koma `,`. (Alasan: Koma memberikan demarkasi entitas bagi *Self-Attention mechanism* pada *Transformer* agar memahami batasan subjek disiplin ilmu).
    * *`bersihkan_author`*: Membuang catatan kaki berupa angka afiliasi kampus di dalam tanda kurung.

6. **Pembersihan Dasar (Unidecode & Whitespace):** Menormalkan karakter alfabet asing menjadi ASCII (contoh `Balázs` $\rightarrow$ `Balazs`) serta meluruhkan *multiple-whitespace* menjadi *single-space*.
7. **Text Concatenation (Feature Engineering):** Menyatukan kolom dominan menjadi *string* padat tunggal bertajuk `combined_text` dengan format `"title: [judul]. abstract: [abstrak]"`. (Alasan strategis: Jika fitur *author* dan kategori diikutkan ke penggabungan, ide pokok penelitian akan mengalami apa yang disebut *Semantic Dilution*/Pelemahan Vektor. Sistem didesain murni menilai ide, bukan mengutamakan *author* terkenal).
8. **Tokenisasi & Lemmatisasi (Eksklusif TF-IDF):** Menggunakan pustaka *NLTK* untuk *word_tokenize* dan *WordNetLemmatizer* guna menyeragamkan berbagai *suffix* kata ke akar dasarnya. (Alasan: TF-IDF berbasis statistik penghitungan frekuensi (Bag-of-Words). Tanpa *Lemmatization*, kata kerja bentuk lampau dan bentuk berjalan akan terhitung sebagai kolom fitur yang sepenuhnya terpisah, membuat dimensi model bengkak).


## Modeling

Sistem ini dimodelkan di atas konsep arsitektur *Content-Based Filtering*. Berikut adalah penjabaran dua solusi algoritma yang diteliti kinerjanya:

### Pendekatan 1: TF-IDF & Cosine Similarity (Baseline Model)

Sistem memproses fitur teks termodifikasi ke dalam matriks frekuensi komparatif renggang (*Sparse Matrix*) memanfaatkan fasilitas `TfidfVectorizer` dari *Scikit-Learn*.
    
  * **Parameter Unik:** Dilakukan optimasi efisiensi RAM berupa `min_df=5` (mengeliminasi kata langka/*typo* yang muncul di < 5 jurnal) dan `max_df=0.8` (membuang kata repetitif yang mendominasi lebih dari 80% jurnal keseluruhan). Batas kosa kata (*Vocabulary*) dijaga pada 10.000 bobot kata tertinggi (*max_features*).
  * **Kelebihan:** Sangat ringan dan memakan waktu komputasi sepersekian detik, akurasi tinggi pada pencarian jurnal menggunakan *exact keyword matching* spesifik.
  * **Kekurangan:** Tidak sanggup memahami kedekatan konteks. Sebuah kueri berjudul "deep learning" tidak akan merekomendasikan artikel bersubjek "neural networks" karena perbedaan kombinasi susunan huruf.

### Pendekatan 2: Sentence Transformers & KNN (Model Utama)

Sistem memanfaatkan arsitektur Deep Learning bernama *Transformers* (*Pre-trained Model* `all-MiniLM-L6-v2` via API Hugging Face). Model akan memampatkan narasi dokumen panjang menjadi garis vektor matematis (*Dense Matrix*) dimensi 384 di dalam sebuah ruang representasi. Kemudian algoritma `NearestNeighbors` bertugas mengukur dan menarik dokumen-dokumen yang jarak sudurnya bertetangga.
  * **Parameter Unik**: Menggunakan konfigurasi algorithm=`auto` dipadukan dengan penghitungan sudut metric=`cosine` untuk mengakumulasi jarak ketetanggaan antar-dokumen di ruang multi-dimensi.
  * **Kelebihan**: Mampu menerobos hambatan *Semantic Gap*. Walaupun menggunakan frasa kata yang diparafrase total secara kosakatanya, model ini tetap mendeteksi bahwa gagasan atau topik utama kedua makalah adalah sama.
  * **Kekurangan**: Tahap *encoding* yang merender kumpulan teks mentah ke dalam bentuk representasi embeddings memerlukan memori operasional (RAM/GPU) yang terlampau besar.

## Output Sistem Rekomendasi (Top-5 Recommendation):

Berdasarkan input kueri aktual dari pengguna, yaitu abstrak yang memuat konteks "mathematical theory of deep learning", berikut adalah hasil daftar rekomendasi faktual yang ditarik oleh kedua model:

   1. Model Berbasis TF-IDF & Cosine Similarity

| Index | Title | Authors | Categories | Abstrach | Year | Cosine Score |
| --- | --- | --- | --- | --- | --- | --- |
| 37514 | mathematical theory of deep learning | philipp petersen and jakob zech | cs.lg, math.ho | this book provides an introduction to the math... | 2026 | 0.624958 |
| 158704 | mathematical foundations of deep learning | xiaojing ye | cs.lg, math.oc | this draft book offers a comprehensive and rig... | 2026 | 0.545734 |
| 177603 | weaves, wires, and morphisms: formalizing and ... | vincent abbott, gioele zardini | cs.lg, math.ct | despite deep learning models running well-defi... | 2026 | 0.443212 |
| 192052 | there will be a scientific theory of deep lear... | jamie simon, daniel kunin, alexander atanasov,... | stat.ml, cs.lg | in this paper, we make the case that a scienti... | 2026 | 0.424222 |
| 192296 | math takes two: a test for emergent mathematic... | michael cooper and samuel cooper | cs.ai, cs.lg | although language models demonstrate remarkabl... | 2026 | 0.382550 |

   2. Model Berbasis Sentence Transformers & KNN

| Index | Title | Authors | Categories | Abstrach | Year |
| --- | --- | --- | --- | --- | --- | 
| 37514 | mathematical theory of deep learning | philipp petersen and jakob zech | cs.lg, math.ho | this book provides an introduction to the math... | 2026 | 
| 158704 | mathematical foundations of deep learning | xiaojing ye | cs.lg, math.oc | this draft book offers a comprehensive and rig... | 2026 | 
| 20495 | deep learning: an introduction for applied mat... | catherine f. higham and desmond j. higham | math.ho, cs.lg, cs.na, math.na, stat.ml | multilayered artificial neural networks are be... | 2026 | 
| 192052 | there will be a scientific theory of deep lear... | jamie simon, daniel kunin, alexander atanasov,... | stat.ml, cs.lg | in this paper, we make the case that a scienti... | 2026 | 
| 19226 | a representer theorem for deep kernel learning | bastian bohn, michael griebel, christian rieger | cs.lg, cs.na, math.na | in this paper we provide a finite-sample and a... | 2026 | 


## Evaluation

Evaluasi sistem rekomendasi dilakukan dengan tujuan mendapakan nilai keakuratan dari output yang dihasilkan sistem rekomendasi, terdapat beberapa metode yang dapat digunakan untuk sistem rekomendasi terutama untuk sistem rekomendasi *Content-Based Filtering* sebagai contoh diantaranya adalah metrik **Precision@K** dan **NDCG@K**. Berikut merupakan tahapan evaluasi dari sistem rekomendasi menggunakan metrik **Precision@K** dan **NDCG@K**.

### Penetapan Ground Truth (Nilai Relevansi)
Sistem rekomendasi biasanya membutuhkan informasi rekaman interaksi pengguna (seperti *Rating* atau histori Klik) untuk dapat mengevaluasi hasil rekomendasi, namun dikarenakan dataset jurnal arXiv tidak memuat data rekaman interaksi pengguna tersebut, nilai relevansi (*Ground Truth*) diukur menggunakan keselarasan atribut fitur **Categories (Kategori Ilmu)**.

* **Definisi Relevan (Nilai 1):** Artikel yang direkomendasikan memiliki minimal satu (1) irisan kategori ilmiah (`categories`) yang sama dengan artikel yang dikuerikan oleh pengguna.
* **Definisi Tidak Relevan (Nilai 0):** Tidak ada satupun kategori ilmu yang beririsan.

### 1. Metrik Precision@K

*Precision@K* berfungsi mengkalkulasi rasio kemunculan item relevan (Kategori sama) yang berhasil terdeteksi dalam daftar *Top-K* rekomendasi (K=5).
**Formula:**

$$Precision@K = \frac{\text{Jumlah rekomendasi relevan di daftar K}}{K}$$

### 2. Metrik NDCG@K (Normalized Discounted Cumulative Gain)

Sementara Precision hanya menghitung *jumlah*, **NDCG@K** mengevaluasi *kualitas peringkat*. NDCG memberikan bobot penalti (potongan nilai logaritmik) apabila artikel yang relevan justru ditempatkan oleh sistem di urutan peringkat yang paling bawah.
**Formula:**

$$NDCG@K = \frac{DCG@K}{IDCG@K}$$

*(Dimana DCG adalah bobot diskon kumulatif aktual, dan IDCG adalah susunan diskon kumulatif paling ideal/sempurna).*

### Hasil Evaluasi (A/B Testing Eksperimen)

Pengujian matriks pemeringkatan dilakukan dengan mengambil 100 sampel indeks artikel kueri secara acak dari dataset (*offline evaluation*). Berikut adalah skor evaluasi aktual keluaran *notebook*:

**1. Baseline Model (Algoritma TF-IDF)**

* **Rata-rata Precision@5 : 63.80%**
* **Rata-rata NDCG@5      : 81.19%**
* *Analisis:* Performa model klasikal membuahkan hasil pemeringkatan yang cukup moderat. Hal ini terjadi karena TF-IDF bertindak buta berdasarkan probabilitas jumlah kata secara kaku. Jika kueri mengandung teks tertentu, TF-IDF akan merespon dengan menarik artikel sembarang yang sering mengulang terminologi yang sama, meskipun pada akhirnya artikel tersebut tidak selaras secara kategori ilmu aslinya.

**2. Main Model (Algoritma Sentence Transformers)**

* **Rata-rata Precision@5 : 83.60%**
* **Rata-rata NDCG@5      : 94.03%**
* *Analisis:* Terjadi lompatan kualitas nilai pemeringkatan yang teramat masif. Model sukses memastikan nyaris 4 hingga 5 artikel yang direkomendasikan dalam daftar Top-5 merupakan disiplin ilmu (Categories) yang benar-benar relevan dengan pencarian user (Precision > 83%). Di sisi lain, skor NDCG yang turut membubung hingga melampaui 94% memvalidasi bahwa mesin pemeringkat Transformers bukan hanya berhasil menarik entitas relevan, melainkan sukses menyajikannya di urutan rekomendasi paling atas/prioritas.


## Konklusi Proyek

Arsitektur penelusuran semantik berlandaskan *Sentence Transformers Embeddings* dibuktikan mutlak lebih superior dalam menangani fenomena ledakan informasi pustaka dibandingkan metode TF-IDF klasik. Penerapan instrumen metrik yang relevan (**Precision@K** dan **NDCG@K**) mengonfirmasi secara valid dan faktual bahwa sistem *Machine Learning* yang direkayasa berhasil mengatasi hambatan Semantic Gap, menghadirkan rekomendasi yang presisi, kontekstual secara pemahaman ide, dan sangat optimal secara pemeringkatannya.
