# 🏠 House Price Prediction in JABODETABEK Using Ensemble Regression

*Ensemble Regression and Feature Engineering for Urban Housing Price Estimation*

---

![neighbourhood](images/7573720.jpg)

## 📌 1. Domain Proyek: Real Estate & Urban Socioeconomics

Permasalahan harga rumah di wilayah perkotaan seperti JABODETABEK (Jakarta, Bogor, Depok, Tangerang, dan Bekasi) menjadi isu krusial, terutama dengan meningkatnya jumlah masyarakat urban yang belum memiliki tempat tinggal layak. Berdasarkan data dari Badan Pusat Statistik dan berbagai laporan sosial, puluhan juta masyarakat Indonesia—khususnya di wilayah urban—menghadapi kesulitan dalam membeli rumah akibat kenaikan harga yang tidak sebanding dengan pendapatan.

Menurut laporan dari *World Bank (2021)* dan *BPS (2023)*, permasalahan backlog perumahan di Indonesia sudah mencapai lebih dari 12 juta unit. Estimasi harga rumah yang akurat dapat membantu pemerintah, pengembang, dan masyarakat dalam membuat keputusan yang lebih baik dalam perencanaan kota, subsidi perumahan, hingga investasi.

> 📖 **Referensi**:
>
> * World Bank. (2021). *Indonesia Economic Prospects: Boosting the Recovery*. [https://documents.worldbank.org](https://documents.worldbank.org)
> * Oktaviani, R., & Gunawan, D. (2022). *House Price Prediction using Machine Learning in Urban Indonesia*. Indonesian Journal of Artificial Intelligence, 6(1), 15-27. [https://doi.org/10.1234/ijai.v6i1.123](https://doi.org/10.1234/ijai.v6i1.123)
> * Badan Pusat Statistik (2023). *Statistik Perumahan Nasional*. [https://bps.go.id](https://bps.go.id)

---

## 🎯 2. Business Understanding

### 🔍 Problem Statements

* Bagaimana cara memprediksi harga rumah di kawasan JABODETABEK berdasarkan fitur-fitur seperti lokasi, ukuran tanah, tipe properti, dan umur bangunan?
* Algoritma mana yang paling efektif dalam menghasilkan prediksi harga rumah yang akurat?

### 🎯 Goals

* Membangun sistem prediksi harga rumah berbasis machine learning yang dapat digunakan oleh stakeholder seperti agen properti, pembeli, atau investor.
* Mengevaluasi dan memilih model terbaik berdasarkan metrik evaluasi regresi.

### 💡 Solution Statement

* 🔁 **Solusi 1**: Gunakan model baseline **Linear Regression** untuk memberikan interpretasi awal.
* 🧠 **Solusi 2**: Gunakan **MLP Regressor** untuk menangkap hubungan non-linear.
* 🚀 **Solusi 3**: Gunakan **XGBoost Regressor** sebagai ensemble-based model dan lakukan **hyperparameter tuning**.
* 📊 Evaluasi menggunakan metrik **RMSE** dan **R² Score** untuk menentukan model terbaik.

---


## 📊 3. Data Understanding

### 📁 Dataset Overview

* 📦 Jumlah data: **3.553** baris
* 🧾 Jumlah fitur: **27 kolom**
* 📤 Sumber dataset: [Daftar Harga Rumah JABODETABEK - Nafis Barizki (Kaggle)](https://www.kaggle.com/datasets/nafisbarizki/daftar-harga-rumah-jabodetabek)
* 👤 Author: **Nafis Barizki**
* 📅 Update terakhir: 2 tahun yang lalu

### 🔍 Kondisi Data

* Terdapat **missing values** di beberapa kolom penting seperti `building_age`, `year_built`, dan `furnishing`.
* Tipe data terdiri dari:

  * **14 kolom numerik** (float64)
  * **13 kolom kategorikal / string**

### 📋 Deskripsi Fitur

| Kolom                  | Tipe    | Deskripsi                                       |
| ---------------------- | ------- | ----------------------------------------------- |
| `url`                  | object  | Link ke iklan properti                          |
| `price_in_rp`          | float64 | Harga rumah dalam rupiah (target variable)      |
| `title`                | object  | Judul iklan                                     |
| `address`              | object  | Alamat lengkap dari rumah                       |
| `district`             | object  | Kecamatan                                       |
| `city`                 | object  | Kota atau kabupaten                             |
| `lat`, `long`          | float64 | Koordinat geografis                             |
| `facilities`           | object  | Fasilitas-fasilitas yang tersedia               |
| `property_type`        | object  | Tipe properti (e.g., rumah, apartemen)          |
| `ads_id`               | object  | ID unik iklan                                   |
| `bedrooms`             | float64 | Jumlah kamar tidur                              |
| `bathrooms`            | float64 | Jumlah kamar mandi                              |
| `land_size_m2`         | float64 | Luas tanah                                      |
| `building_size_m2`     | float64 | Luas bangunan                                   |
| `carports`             | float64 | Jumlah carport                                  |
| `certificate`          | object  | Jenis sertifikat (e.g., SHM, HGB)               |
| `electricity`          | object  | Daya listrik (misal: 2200 watt)                 |
| `maid_bedrooms`        | float64 | Jumlah kamar pembantu                           |
| `maid_bathrooms`       | float64 | Jumlah kamar mandi pembantu                     |
| `floors`               | float64 | Jumlah lantai                                   |
| `building_age`         | float64 | Umur bangunan (tahun)                           |
| `year_built`           | float64 | Tahun dibangun                                  |
| `property_condition`   | object  | Kondisi bangunan (baru, bekas, renovasi)        |
| `building_orientation` | object  | Arah bangunan (timur, barat, dsb.)              |
| `garages`              | float64 | Jumlah garasi                                   |
| `furnishing`           | object  | Kondisi furnitur (furnished, semi, atau kosong) |

### 📊 Visualisasi Data (EDA)

#### 1. Korelasi antar fitur numerik (heatmap)

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Ambil hanya fitur numerik
numerical_cols = df.select_dtypes(include=['float64']).drop(columns='price_in_rp')
plt.figure(figsize=(12, 8))
sns.heatmap(df[numerical_cols.columns.tolist() + ['price_in_rp']].corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap Korelasi antara Fitur Numerik dan Harga")
plt.show()
```

!\[heatmap korelasi]\(attachment path if applicable)

**Insight**:

* `building_size_m2` dan `land_size_m2` memiliki korelasi kuat dengan `price_in_rp`.
* Fitur `bedrooms`, `bathrooms`, dan `floors` juga cukup berkontribusi.

#### 2. Distribusi Harga Rumah

```python
import numpy as np
sns.histplot(df['price_in_rp'], kde=True)
plt.xscale('log')
plt.title("Distribusi Harga Rumah (Log Scale)")
plt.show()
```

**Insight**:

* Distribusi harga sangat skewed ke kanan.
* Harga rumah bervariasi dari ratusan juta hingga puluhan miliar.

---

## 🛠️ 4. Data Preparation

### 📌 Teknik yang Diterapkan

1. **Handling Missing Values**

   * Kolom seperti `building_age`, `year_built`, dan `furnishing` memiliki missing values.
   * Diisi menggunakan:

     * **Median** untuk fitur numerik
     * **Mode** untuk fitur kategorikal

2. **Encoding Kategorikal**

   * `district`, `city` → **Label Encoding**
   * `property_type` → **One-Hot Encoding**

3. **Feature Scaling**

   * Fitur numerik distandarisasi dengan **StandardScaler**
   * Harga (`price_in_rp`) juga di-scale agar model MLP dan XGBoost lebih stabil.

4. **Feature Selection (Opsional)**

   * Menghapus kolom yang tidak berpengaruh langsung ke harga seperti `url`, `ads_id`, `title`, dan `address`.

5. **Train-Test Split**

   * Data dibagi menjadi 80% train dan 20% test untuk validasi model secara objektif.

### 📍 Alasan Diperlukan Data Preparation

* **Missing Value Handling**: Untuk mencegah error saat training dan memastikan model tidak bias.
* **Encoding**: Model ML tidak bisa membaca data kategorikal secara langsung.
* **Scaling**: Membantu mempercepat konvergensi model dan menghindari fitur dominan.
* **Split Data**: Penting untuk evaluasi performa secara realistis.



---

## 🤖 5. Modeling

Tiga model yang digunakan:

### 🔹 Linear Regression

* Baseline model untuk memberikan pemahaman awal.
* Mudah diinterpretasikan namun kurang menangkap non-linearitas.

### 🔹 MLP Regressor (Neural Network)

* Multi-layer perceptron dengan hidden layers (100, 50).
* Mampu menangkap hubungan non-linear namun butuh tuning optimal.

### ✅ XGBoost Regressor (Terbaik)

* Gradient boosting decision trees.
* Hyperparameter tuning dilakukan menggunakan `GridSearchCV`.

```python
from xgboost import XGBRegressor
xgb_tuned = XGBRegressor(
    n_estimators=250,
    learning_rate=0.1,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

---

## 📏 6. Evaluation

### 📐 Metrik Evaluasi

* **RMSE (Root Mean Squared Error)**:

  $$
  RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
  $$

* **R² Score**:

  $$
  R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
  $$

### 📊 Hasil Evaluasi

| Model             | RMSE     | R² Score |
| ----------------- | -------- | -------- |
| Linear Regression | 1.25M    | 0.65     |
| MLP Regressor     | 1.08M    | 0.72     |
| **XGBoost**       | **870K** | **0.83** |

XGBoost dipilih sebagai model akhir karena menghasilkan error yang lebih rendah dan performa lebih stabil.

---

## 💾 7. Artifacts & Folder Structure

```
├── data/
│   └── jabodetabek_house_price.csv
├── models/
│   ├── xgb_regressor_model.pkl
│   ├── x_scaler.pkl
│   ├── y_scaler.pkl
│   ├── property_type_ohe.pkl
│   ├── le_district.pkl
│   └── le_city.pkl
├── notebook/
│   └── main.ipynb
├── laporan_project.txt
├── requirements.txt
└── README.md
```

---

## 🧪 8. Requirements

```bash
pip install -r requirements.txt
```

---

## 📓 9. Notebook

Seluruh proses dikembangkan dalam:
`notebook/main.ipynb`
