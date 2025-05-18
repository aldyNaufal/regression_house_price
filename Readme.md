# 🏠 House Price Prediction in JABODETABEK Using Ensemble Regression

**Ensemble Regression and Feature Engineering for Urban Housing Price Estimation**

---

![neighbourhood](images/7573720.jpg)

## 📌 1. Domain Proyek: Real Estate & Urban Socioeconomics

Harga rumah di wilayah JABODETABEK (Jakarta, Bogor, Depok, Tangerang, Bekasi) mengalami kenaikan signifikan akibat urbanisasi dan keterbatasan lahan. Masalah backlog perumahan (12 juta unit menurut BPS 2023) dan disparitas harga menjadi tantangan nyata bagi masyarakat serta pengembang perumahan.

Prediksi harga rumah secara akurat sangat penting bagi:

* Pembeli untuk merencanakan anggaran
* Developer dalam menentukan harga yang kompetitif
* Pemerintah untuk perencanaan tata kota dan kebijakan subsidi
* Investor properti dalam pengambilan keputusan investasi

Penelitian ini bertujuan **membangun model prediksi harga rumah** berbasis **machine learning** dengan teknik **ensemble regression** yang dikombinasikan dengan *feature engineering* yang kuat.

---

## 🎯 2. Business Understanding

### 🔍 Problem Statements

* Bagaimana memprediksi harga rumah di JABODETABEK berdasarkan fitur properti dan lokasi?
* Algoritma machine learning mana yang paling efektif untuk regresi harga rumah di kawasan urban Indonesia?

### 🎯 Objectives

* Membangun model prediksi harga rumah yang presisi.
* Membandingkan performa model: **Linear Regression**, **MLP Regressor**, dan **XGBoost**.
* Memberikan insight dari fitur-fitur yang paling memengaruhi harga rumah.

### 💡 Solusi

1. **Linear Regression** sebagai baseline model untuk interpretabilitas.
2. **MLP Regressor** untuk mengakomodasi hubungan non-linear antar fitur.
3. **XGBoost Regressor** sebagai model ensemble utama dengan **hyperparameter tuning** menggunakan `RandomizedSearchCV`.

---

## 📁 3. Dataset Overview

* **Sumber**: Kaggle - [Daftar Harga Rumah JABODETABEK](https://www.kaggle.com/datasets/nafisbarizki/daftar-harga-rumah-jabodetabek)
* **Jumlah entri**: 3.553 baris
* **Jumlah fitur**: 27 kolom
* **Penulis dataset**: Nafis Barizki

---

## 📋 4. Fitur Dataset

| Fitur                   | Tipe    | Deskripsi                              |
| ----------------------- | ------- | -------------------------------------- |
| `price_in_rp`           | float64 | Harga rumah (dalam rupiah) - target    |
| `building_size_m2`      | float64 | Luas bangunan                          |
| `land_size_m2`          | float64 | Luas tanah                             |
| `bedrooms`, `bathrooms` | float64 | Jumlah kamar tidur dan mandi           |
| `floors`                | float64 | Jumlah lantai                          |
| `city`, `district`      | object  | Lokasi geografis                       |
| `property_type`         | object  | Jenis properti (rumah, apartemen, dsb) |
| `furnishing`            | object  | Tingkat perabotan                      |
| `building_age`          | float64 | Umur bangunan (dalam tahun)            |
| `year_built`            | float64 | Tahun dibangun                         |
| ...                     | ...     | dan fitur lainnya                      |

---

## 🔍 5. Data Understanding

### Statistik Umum:

* Data harga sangat **skewed ke kanan**, menunjukkan adanya outlier harga rumah mewah.
* Korelasi tertinggi dengan `price_in_rp`:

  * `building_size_m2`, `land_size_m2`
  * `bedrooms`, `bathrooms`
  * `property_type` dan `furnishing` juga memberikan insight penting.

### Visualisasi:

* 📊 Histogram distribusi harga
* 🔥 Heatmap korelasi antar fitur numerik
* 🏷️ Perbandingan `furnishing` dan `property_condition`

---

## 🧹 6. Data Preparation

### Langkah Preprocessing:

| Langkah             | Penjelasan                                                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Handling Missing    | Imputasi menggunakan **median** untuk numerik dan **mode** untuk kategorikal                                        |
| Encoding            | - **Label Encoding** untuk `city`, `district`  <br> - **One-Hot Encoding** untuk `property_type`, `furnishing`, dll |
| Scaling             | **StandardScaler** untuk semua kolom numerik (termasuk target)                                                      |
| Feature Engineering | - Transformasi `year_built` ke `building_age` jika hilang <br> - Ekstraksi informasi dari `facilities`              |
| Drop Columns        | Kolom seperti `url`, `ads_id`, `title`, `address` tidak digunakan karena tidak relevan dalam regresi harga          |
| Train-Test Split    | Proporsi 80% data training dan 20% data testing                                                                     |

---

## ⚙️ 7. Modeling

### 🔹 Model 1: Linear Regression

* Sebagai baseline model.
* Mudah diinterpretasikan, cocok untuk melihat signifikansi fitur.
* **Kelemahan**: tidak mampu menangkap hubungan non-linear.

### 🔹 Model 2: MLP Regressor

* Multi-layer Perceptron dengan arsitektur `[100, 50]`.
* Aktivasi: ReLU, Optimizer: Adam.
* Perlu feature scaling agar hasil stabil.
* **Kelebihan**: mampu belajar pola non-linear kompleks.

### 🔹 Model 3: XGBoost Regressor ✅ (Model Terbaik)

* Gradient Boosting Trees dengan regularisasi.
* Tuning menggunakan **RandomizedSearchCV** untuk:

  * `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
* **Kelebihan**: tangguh terhadap missing data, interpretasi fitur lewat feature importance.

---

## 📏 8. Evaluation (Dikaitkan dengan Business Understanding)

### 🔍 Tujuan Evaluasi:

Sejalan dengan *business understanding*, tujuan dari evaluasi ini adalah untuk mengetahui seberapa **akurat dan andal** model dalam memprediksi harga rumah, sehingga dapat digunakan untuk **pengambilan keputusan strategis oleh pembeli, pengembang, investor, dan regulator**.

---

### 📐 Metrik Evaluasi & Rumus

| Metrik       | Rumus                                                               | Keterangan                                                                                                           |   |                                                                                                                 |
| ------------ | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | - | --------------------------------------------------------------------------------------------------------------- |
| **MSE**      | $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$       | Rata-rata dari kuadrat error. Sensitif terhadap outlier.                                                             |   |                                                                                                                 |
| **RMSE**     | $\text{RMSE} = \sqrt{\text{MSE}}$                                   | Mengakar nilai MSE agar berada dalam satuan yang sama dengan data asli (misal: juta rupiah).                         |   |                                                                                                                 |
| **MAE**      | ( \text{MAE} = \frac{1}{n} \sum\_{i=1}^{n}                          | y\_i - \hat{y}\_i                                                                                                    | ) | Rata-rata error absolut. Mudah dimengerti dan cukup stabil terhadap outlier.                                    |
| **MAPE**     | ( \text{MAPE} = \frac{100%}{n} \sum\_{i=1}^{n} \left                | \frac{y\_i - \hat{y}\_i}{y\_i} \right                                                                                | ) | Mengukur rata-rata error dalam bentuk persentase, cocok untuk membandingkan across wilayah atau kategori rumah. |
| **R² Score** | $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Mengukur seberapa besar variasi target yang bisa dijelaskan oleh model. Nilai mendekati 1 menunjukkan performa baik. |   |                                                                                                                 |

---

### 📊 Hasil Evaluasi Model:

#### 🔹 **Linear Regression**

| Metrik   | Nilai  |
| -------- | ------ |
| MSE      | 0.1000 |
| RMSE     | 0.3162 |
| MAE      | 0.2190 |
| MAPE     | 11.76% |
| R² Score | 0.8646 |

---

#### 🔹 **Multi-Layer Perceptron (MLP) Regressor**

| Metrik   | Nilai  |
| -------- | ------ |
| MSE      | 0.0722 |
| RMSE     | 0.2688 |
| MAE      | 0.1721 |
| MAPE     | 7.08%  |
| R² Score | 0.9022 |

---

#### 🔹 **XGBoost Regressor**

| Metrik   | Nilai  |
| -------- | ------ |
| MSE      | 0.0507 |
| RMSE     | 0.2252 |
| MAE      | 0.1152 |
| MAPE     | 8.36%  |
| R² Score | 0.9313 |

> 🎯 Berdasarkan hasil di atas, **XGBoost memiliki performa terbaik**, ditunjukkan oleh nilai **R² paling tinggi (0.9313)** dan **MAE terendah**, menjadikannya kandidat paling tepat untuk deployment.

---

### 📌 Implikasi Bisnis dari Hasil Model:

| Stakeholder       | Dampak dari Model (XGBoost)                                                         |
| ----------------- | ----------------------------------------------------------------------------------- |
| **Calon Pembeli** | Estimasi harga realistis untuk membuat keputusan pembelian lebih terinformasi.      |
| **Developer**     | Menentukan harga rumah berdasarkan data pasar dan spesifikasi rumah.                |
| **Investor**      | Analisis kelayakan investasi properti berdasarkan prediksi nilai dan tren harga.    |
| **Pemerintah**    | Menyusun kebijakan perumahan berdasarkan estimasi harga aktual di berbagai wilayah. |

---

### ✅ Kesimpulan

Dengan **R² = 0.9313** dan **MAPE = 8.36%**, model XGBoost terbukti mampu menjawab permasalahan dalam *business understanding*:

> **"Bagaimana memprediksi harga rumah secara akurat untuk membantu stakeholder dalam pengambilan keputusan properti?"**

Model ini memberikan estimasi yang dapat diandalkan dan **mudah diterjemahkan ke dalam aksi nyata**, seperti penetapan harga jual, analisis kompetitor, serta evaluasi program subsidi perumahan.



---

## 📌 9. Feature Importance

Visualisasi menggunakan `.feature_importances_` dari XGBoost menunjukkan bahwa:

* `building_size_m2` dan `land_size_m2` adalah kontributor utama harga.
* `property_type`, `furnishing`, dan `location (district)` juga signifikan.

---

## 🚀 10. Deployment (Opsional)

> Untuk keperluan praktis, model dapat di-*pickle* dan disajikan via API dengan **Flask** atau **Streamlit** sebagai antarmuka interaktif prediksi harga rumah berdasarkan input pengguna.

---

## 📚 11. Referensi

* El Mouna, L., et al. (2023). *A Comparative Study of Urban House Price Prediction using Machine Learning Algorithms*. E3S Web of Conferences.
* Al Maula, S. F., et al. (2025). *Modeling House Selling Prices in Jakarta and South Tangerang Using Machine Learning*. BAREKENG.
* Badan Pusat Statistik (2023). *Statistik Perumahan Nasional*.

---

## ✍️ 12. Penutup

Prediksi harga rumah di wilayah urban seperti JABODETABEK menjadi semakin penting di tengah ketimpangan permintaan dan ketersediaan lahan. Pendekatan machine learning dengan ensemble regression seperti XGBoost terbukti memberikan prediksi yang sangat akurat. Hasil ini tidak hanya berkontribusi pada pengambilan keputusan ekonomi yang lebih baik, namun juga berpotensi menjadi dasar sistem rekomendasi harga rumah berbasis data di masa depan.

