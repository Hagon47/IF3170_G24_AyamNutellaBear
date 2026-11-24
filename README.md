# Tugas Besar 2 IF3170 Inteligensi Artifisial

**Kelompok 24 – AyamNutellaBear**

Repository ini berisi implementasi *from scratch* lima algoritma *supervised learning* untuk memprediksi pembatalan reservasi hotel (`is_canceled`) pada kompetisi Kaggle **T2-IF3170-2026-2027**. Metrik evaluasinya Macro F1.

Algoritma yang diimplementasikan (hanya memakai NumPy untuk bagian model):

| Algoritma | Berkas | Keterangan |
|---|---|---|
| K-Nearest Neighbors | `knn.py`, `knn_tuning.py` | `KNN` dasar (Minkowski, voting mayoritas) dan `KNNTuning` (turunan `KNN`: penalti atribut berbeda untuk pemesanan "kerabat", voting berbobot jarak, threshold) |
| Decision Tree Learning (C4.5) | `DTL_main.py`, `DTL_alt.py` | `DTL_Main` (gain ratio, multi-way split, PEP/REP pruning) dan `DTL_Alt` (class weight, MSD split, kriteria L'Hospital, threshold) |
| Logistic Regression | `log_regression.py`, `log_regression_visualization.py` | Regularisasi L1, L2, Elastic Net dengan SGD; visualisasi lintasan training (bonus) |
| Support Vector Machine | `SVM.py`, `AdaBoostSVM.py` | Kernel linear, polynomial, RBF dengan solver QP (SLSQP); AdaBoost-SVM (bonus) |
| Naive Bayes | `naive.py` | Frekuensi, Laplace smoothing +1, MAP; graf Bayesian Network (bonus) |

Model terbaik berdasarkan Macro F1 K-Fold adalah **KNN kerabat** (`KNNTuning`), dengan Macro F1 K-Fold 0,8499 dan holdout 0,8603.

## Struktur Repository

```
.
├── src/
│   ├── algorithm/        # implementasi from scratch setiap algoritma
│   ├── data/             # train.csv, test.csv, sample_submission.csv
│   └── notebook/
│       └── notebook.ipynb
├── output/               # dibuat otomatis: submission, model, dan gambar
└── README.md
```

## Setup

1. Pastikan Python 3.10 atau lebih baru sudah terpasang.
2. Clone repository lalu masuk ke folder repository:
   ```bash
   git clone <url-repository>
   cd <nama-repository>
   ```
3. (Disarankan) buat dan aktifkan virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux / macOS
   source .venv/bin/activate
   ```
4. Pasang library yang dibutuhkan:
   ```bash
   pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
   ```
   Video lintasan training Logistic Regression (bonus) membutuhkan **FFmpeg**. Tanpa FFmpeg, sel tersebut hanya mencetak pesan dan notebook tetap berjalan.
5. Unduh data kompetisi dari Kaggle (`train.csv`, `test.csv`, `sample_submission.csv`) dan letakkan di `src/data/`.

## Menjalankan Program

1. Buka `src/notebook/notebook.ipynb` di VS Code (ekstensi Jupyter) atau Jupyter Notebook.
2. Pastikan notebook dijalankan dari folder `src/notebook/`. Sel Settings memakai `Path('..')`, sehingga folder `src` harus berada satu tingkat di atas folder kerja notebook.
3. Pilih kernel dari virtual environment, lalu jalankan **Run All**.

Seluruh notebook membutuhkan waktu sekitar 30–40 menit, tergantung spesifikasi komputer. Bagian terlama adalah tuning KNN kerabat, tuning Decision Tree, dan tuning SVM.

Hasil yang tersimpan di folder `output/`:

| Berkas | Isi |
|---|---|
| `submission.csv` | Prediksi data test dari model terbaik (dipilih otomatis berdasarkan Macro F1 K-Fold) |
| `submission_knn.csv`, `submission_dtl.csv` | Prediksi KNN kerabat dan `DTL_Alt` final, untuk dibandingkan di Kaggle |
| `models/` | Model yang sudah dilatih (`.npz` / `.pkl`), dapat dimuat ulang dengan method `load` |
| `*.png`, `*.mp4` | Visualisasi pohon keputusan, graf Naive Bayes, bidang pemisah SVM, dan lintasan training Logistic Regression |

## Pembagian Tugas

| NIM | Nama | Tugas |
|---|---|---|
| 13524005 | Geraldo Artemius | Logistic Regression, Laporan |
| 13524051 | Mikhael Yonatan | Decision Tree Learning, Laporan |
| 13524055 | Junior Natra Situmorang | Support Vector Machine, Laporan |
| 13524103 | Reynard Nathanael | Naive Bayes (implementasi, seleksi fitur, graf Bayesian Network), Naive Bayes, Laporan |
| 13524105 | Nicholas Luis Chandra | KNN, Laporan |
