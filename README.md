# UTS Machine Learning: Linear Regression dan Logistic Regression

**Nama:** Agustien Eka Suci Calista  
**NIM:** 241011402953  
**Kelas:** 05TPLM005

# UTS Machine Learning: Linear Regression dan Logistic Regression

## Deskripsi Proyek

Proyek ini merupakan tugas Ujian Tengah Semester (UTS) yang membahas implementasi algoritma Machine Learning menggunakan Python. Algoritma yang digunakan adalah **Linear Regression** untuk memprediksi nilai numerik dan **Logistic Regression** untuk mengklasifikasikan email spam dan bukan spam.

Implementasi model dilakukan menggunakan perhitungan manual dengan NumPy untuk memahami proses pelatihan, prediksi, dan evaluasi model.

## Tujuan

1. Memahami proses pembagian dataset menjadi data training dan testing.
2. Mengimplementasikan Linear Regression menggunakan Normal Equation.
3. Mengevaluasi model regresi menggunakan MAE, MSE, RMSE, dan R².
4. Memahami Gradient Descent dan pengaruh learning rate terhadap proses optimasi.
5. Mengimplementasikan Logistic Regression untuk klasifikasi email.
6. Mengevaluasi hasil klasifikasi menggunakan confusion matrix, accuracy, precision, recall, dan F1-score.
7. Menganalisis pengaruh threshold terhadap hasil klasifikasi.
8. Membandingkan implementasi manual dengan scikit-learn pada bagian bonus.

## Teknologi yang Digunakan

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn (bagian bonus)

## Struktur dan Pembahasan

### Bagian A — Linear Regression

Linear Regression digunakan untuk memodelkan hubungan antara biaya iklan dan penjualan.

Dataset yang digunakan adalah `Ads_Data.csv`, dengan:
- **BiayaIklan** sebagai variabel input (X).
- **Penjualan** sebagai variabel target (y).

Tahapan yang dilakukan:

1. **A1 — Load dan Visualisasi Data:** Membaca dataset, memisahkan fitur dan target, serta menampilkan hubungan biaya iklan dan penjualan melalui scatter plot.
2. **A2 — Split Data:** Membagi data menjadi 80% training dan 20% testing.
3. **A3 — Normal Equation:** Melatih model regresi dan memperoleh nilai intercept serta koefisien.
4. **A4 — Evaluasi Model:** Mengukur kesalahan prediksi menggunakan MAE, MSE, RMSE, dan R².
5. **A5 — Gradient Descent:** Membandingkan learning rate 0.001, 0.02, dan 0.1 untuk mengamati perubahan loss selama proses training.
6. **A6 — Visualisasi dan Prediksi:** Menampilkan garis regresi dan memprediksi penjualan untuk biaya iklan sebesar Rp4,5 juta, Rp7 juta, dan Rp12 juta.

Evaluasi dilakukan menggunakan data testing untuk mengetahui kemampuan model dalam memprediksi data yang tidak digunakan selama pelatihan.

### Bagian B — Logistic Regression

Logistic Regression digunakan untuk mengklasifikasikan email ke dalam dua kategori, yaitu spam dan bukan spam.

Dataset yang digunakan adalah `Email_Data.csv`, dengan:
- **JumlahLink** sebagai fitur pertama.
- **KataPromo** sebagai fitur kedua.
- **Spam** sebagai variabel target, dengan nilai 0 untuk bukan spam dan 1 untuk spam.

Tahapan yang dilakukan:

1. **B1 — Load dan Visualisasi Data:** Membaca dataset, memeriksa distribusi kelas, dan memvisualisasikan hubungan fitur dengan label email.
2. **B2 — Training Logistic Regression:** Membagi dataset menjadi data training dan testing, kemudian melatih model menggunakan Gradient Descent dan fungsi sigmoid.
3. **B3 — Evaluasi Model:** Menghitung confusion matrix, accuracy, precision, recall, dan F1-score.
4. **B4 — Decision Boundary:** Memvisualisasikan batas keputusan model dan mengidentifikasi data yang salah diklasifikasikan.
5. **B5 — Eksperimen Threshold:** Membandingkan hasil klasifikasi pada threshold 0.3, 0.5, dan 0.7.
6. **B6 — Prediksi Email Baru:** Menggunakan model untuk memprediksi probabilitas spam pada email baru berdasarkan jumlah link dan kata promosi.

#### Hasil Evaluasi Logistic Regression

Pada pengujian dengan 15 data testing, diperoleh hasil berikut:

| Metrik | Hasil |
|---|---:|
| Accuracy | 93,33% |
| Precision | 85,71% |
| Recall | 100% |
| F1-score | 92,31% |

Confusion matrix yang diperoleh adalah:

```text
[[8, 1],
 [0, 6]]
```

Hasil tersebut menunjukkan bahwa 8 email bukan spam dan 6 email spam berhasil diklasifikasikan dengan benar. Terdapat 1 email bukan spam yang salah diklasifikasikan sebagai spam, sedangkan tidak ada email spam yang salah diklasifikasikan sebagai bukan spam.

Dengan hasil ini, model menunjukkan performa yang baik pada data testing yang digunakan. Namun, jumlah data testing yang relatif kecil perlu dipertimbangkan saat menilai kemampuan model pada data baru.

#### Analisis Threshold

Pengujian dilakukan menggunakan tiga nilai threshold:

| Threshold | TN | FP | FN | TP | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 6 | 3 | 0 | 6 | 66,67% | 100% |
| 0.5 | 8 | 1 | 0 | 6 | 85,71% | 100% |
| 0.7 | 9 | 0 | 1 | 5 | 100% | 83,33% |

Pada threshold 0.3, model berhasil mendeteksi seluruh email spam, tetapi menghasilkan lebih banyak false positive. Pada threshold 0.5, jumlah false positive menurun menjadi satu tanpa kehilangan deteksi email spam. Sementara itu, threshold 0.7 menghasilkan precision sebesar 100%, tetapi terdapat satu email spam yang tidak terdeteksi.

Berdasarkan hasil pengujian tersebut, threshold 0.5 dipilih karena memberikan keseimbangan yang baik antara precision dan recall serta tetap mendeteksi seluruh email spam pada data testing.

### Bagian C — Pemahaman Konsep

Bagian ini membahas konsep dasar Machine Learning, meliputi:

1. Peran fungsi sigmoid dalam Logistic Regression.
2. Perbedaan Normal Equation dan Gradient Descent.
3. Pentingnya data testing dan pengertian overfitting.
4. Perbedaan penggunaan MSE dan log loss.
5. Penyebab perbedaan hasil model akibat pembagian data dan parameter pelatihan.

Bagian ini bertujuan memperkuat pemahaman mengenai cara kerja algoritma serta alasan pemilihan metode evaluasi yang sesuai.

### Bonus — Perbandingan dengan scikit-learn

Bagian bonus membandingkan implementasi Linear Regression dan Logistic Regression secara manual dengan model dari scikit-learn.

Perbandingan dilakukan menggunakan data training dan testing yang sama, dengan melihat koefisien, intercept, dan metrik evaluasi. Hasil dapat berbeda karena metode optimasi, pengaturan regularisasi, dan kriteria konvergensi yang digunakan.

## Kesimpulan

Melalui proyek ini, Linear Regression digunakan untuk memprediksi nilai penjualan berdasarkan biaya iklan, sedangkan Logistic Regression digunakan untuk mengklasifikasikan email spam dan bukan spam.

Evaluasi Logistic Regression menghasilkan accuracy 93,33%, precision 85,71%, recall 100%, dan F1-score 92,31% pada data testing. Eksperimen threshold menunjukkan bahwa pemilihan threshold memengaruhi keseimbangan antara mendeteksi spam dan menghindari kesalahan dalam menandai email bukan spam.

Secara keseluruhan, proyek ini memberikan pemahaman mengenai implementasi algoritma Machine Learning, proses pelatihan, evaluasi, visualisasi, dan analisis hasil prediksi.



## Catatan

Hasil evaluasi yang ditampilkan pada README ini berasal dari hasil pengujian Logistic Regression yang diperoleh pada notebook. Hasil Linear Regression dan bagian bonus perlu mengacu pada output aktual setelah seluruh sel dijalankan.

