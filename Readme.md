# 🏠 House Price Prediction in JABODETABEK Using Ensemble Regression

*Ensemble Regression and Feature Engineering for Urban Housing Price Estimation*

---

![neighbourhood](images/7573720.jpg)

## 📌 1. Domain Proyek: Real Estate & Urban Socioeconomics

Permasalahan harga rumah di wilayah perkotaan seperti JABODETABEK (Jakarta, Bogor, Depok, Tangerang, dan Bekasi) menjadi isu krusial, terutama dengan meningkatnya jumlah masyarakat urban yang belum memiliki tempat tinggal layak. Masifnya urbanisasi di kota besar seperti Jakarta mendorong lonjakan permintaan akan hunian, namun keterbatasan lahan dan tingginya harga properti menyebabkan banyak kelompok masyarakat kesulitan memperoleh rumah yang sesuai dengan kemampuan ekonomi mereka. Hal ini diperparah oleh backlog perumahan yang mencapai lebih dari 12 juta unit, sebagaimana dilaporkan oleh Badan Pusat Statistik pada tahun 2023.

Kondisi ini menimbulkan kebutuhan mendesak akan solusi yang dapat membantu pemangku kebijakan, pengembang, serta masyarakat umum untuk memahami dan memprediksi harga rumah secara lebih akurat. Prediksi harga rumah yang tepat sangat penting, tidak hanya untuk menentukan nilai jual beli properti, tetapi juga untuk mendukung keputusan investasi dan perencanaan tata kota. Seperti dijelaskan oleh Al Maula et al. (2025), tren hunian kini mulai bergeser ke daerah pinggiran yang lebih terjangkau dan mendukung kualitas hidup, namun tetap menimbulkan tantangan dalam distribusi dan perencanaan kawasan perumahan.

Studi sebelumnya juga menunjukkan bahwa harga rumah dipengaruhi oleh banyak faktor, seperti ukuran bangunan, lokasi, jumlah kamar, akses terhadap fasilitas umum, dan usia bangunan (El Mouna et al., 2023). Dengan menggunakan algoritma machine learning seperti *Linear Regression*, *Random Forest*, dan *Gradient Boosting*, prediksi harga rumah kini dapat dilakukan secara lebih akurat dengan memanfaatkan data dalam jumlah besar. Pendekatan ini menawarkan solusi yang lebih adaptif dan efisien dibandingkan metode tradisional, serta menjadi alat penting bagi perencana kota dan investor untuk mengambil keputusan berbasis data.

> 📖 **Referensi**:
>
> * El Mouna, L., Silkan, H., Haynf, Y., & Nann, M. F. (2023). *A Comparative Study of Urban House Price Prediction using Machine Learning Algorithms*. E3S Web of Conferences 418, 03001. [https://doi.org/10.1051/e3sconf/202341803001](https://doi.org/10.1051/e3sconf/202341803001)
>
> * Al Maula, S. F., Setiawan, N. A. D., Pusporani, E., & Jannah, S. Z. (2025). *Modeling House Selling Prices in Jakarta and South Tangerang Using Machine Learning Prediction Analysis*. **BAREKENG: Journal of Mathematics and Its Applications**, 19(1), 0107–0118. [https://ojs3.unpatti.ac.id/index.php/barekeng/article/download/12906/9511](https://ojs3.unpatti.ac.id/index.php/barekeng/article/download/12906/9511)
>
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

  * **14 kolom numerik** (`float64`)
  * **13 kolom kategorikal / string**

---

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

---

### 📈 Visualisasi Data (EDA)

#### 1. Distribusi Harga Rumah

Distribusi harga rumah menunjukkan adanya **skewness** ke kanan, menandakan bahwa sebagian besar properti berada di kisaran harga yang lebih rendah, sementara sebagian kecil lainnya memiliki harga yang sangat tinggi.

![Distribusi Harga Rumah](images/distribusi_harga.png)

---

#### 2. Korelasi Antar Fitur Numerik

Visualisasi ini menunjukkan hubungan antar fitur numerik dalam dataset. Beberapa fitur seperti `building_size_m2`, `land_size_m2`, `bathrooms`, dan `bedrooms` memiliki korelasi positif yang cukup tinggi dengan `price_in_rp`.

![Heatmap Korelasi](images/heatmap_korelasi.png)

---

#### 3. Hubungan antara Furnishing dan Kondisi Properti

Heatmap berikut menunjukkan hubungan antara kondisi properti dan jenis furnishing-nya. Misalnya, properti **unfurnished** lebih sering ditemukan dalam kondisi **butuh renovasi**, sementara **furnished** cenderung berada dalam kondisi **bagus**.

![Hubungan Furnishing dan Kondisi Properti](images/furnishing_vs_condition.png)

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

Pada tahap ini, dilakukan evaluasi terhadap tiga model regresi yang telah dilatih: **Linear Regression**, **MLP Regressor**, dan **XGBoost Regressor**. Evaluasi dilakukan menggunakan metrik regresi yang relevan:

* **MSE** (Mean Squared Error)
* **RMSE** (Root Mean Squared Error)
* **MAE** (Mean Absolute Error)
* **MAPE** (Mean Absolute Percentage Error)
* **R² Score**

Berikut adalah hasil evaluasi masing-masing model:

---

### 🔹 **Linear Regression**

| Metric   | Value   |
| -------- | ------- |
| MSE      | 0.1000  |
| RMSE     | 0.3162  |
| MAE      | 0.2190  |
| MAPE     | 11.76 T |
| R² Score | 0.8646  |

---

### 🔹 **Multi-Layer Perceptron (MLP) Regressor**

| Metric   | Value  |
| -------- | ------ |
| MSE      | 0.0722 |
| RMSE     | 0.2688 |
| MAE      | 0.1721 |
| MAPE     | 7.08 T |
| R² Score | 0.9022 |

---

### 🔹 **XGBoost Regressor**

| Metric   | Value  |
| -------- | ------ |
| MSE      | 0.0507 |
| RMSE     | 0.2252 |
| MAE      | 0.1152 |
| MAPE     | 8.36 T |
| R² Score | 0.9313 |

> 💡 *Catatan:* Nilai MAPE sangat besar kemungkinan karena banyak nilai harga rumah sangat tinggi dan menyebabkan distorsi. Ini perlu dipertimbangkan dengan pembobotan atau penghapusan outlier ekstrem di preprocessing lanjutan.

---

## ✅ **📌 Conclusion / Summary**

Berdasarkan hasil evaluasi terhadap ketiga model regresi:

* **Linear Regression** menunjukkan performa paling rendah, walaupun cukup baik secara R².
* **MLP Regressor** menunjukkan peningkatan signifikan terutama pada MSE, MAE, dan R².
* **XGBoost Regressor** memberikan hasil terbaik pada hampir semua metrik, terutama **R² Score sebesar 0.9313**, dan **RMSE terendah sebesar 0.2252**.

✅ Oleh karena itu, **XGBoost Regressor** dipilih sebagai **model akhir yang digunakan untuk prediksi harga rumah di wilayah JABODETABEK** karena memiliki kinerja paling optimal dan stabil berdasarkan evaluasi.

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





