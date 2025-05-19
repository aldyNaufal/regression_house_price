# 🏠 House Price Prediction in JABODETABEK Using Ensemble Regression

**Ensemble Regression and Feature Engineering for Urban Housing Price Estimation**

---

![neighbourhood](images/7573720.jpg)

## 📌 1. Domain Proyek: Real Estate & Urban Socioeconomics

Permasalahan harga rumah di wilayah JABODETABEK (Jakarta, Bogor, Depok, Tangerang, dan Bekasi) terus menjadi isu strategis di tengah pertumbuhan populasi perkotaan yang pesat. Urbanisasi masif yang terjadi di kawasan ini telah mendorong peningkatan permintaan terhadap hunian, namun tidak diimbangi oleh ketersediaan lahan dan pembangunan yang memadai. Hal ini menyebabkan harga properti melonjak dan menjadikan rumah layak huni semakin sulit dijangkau oleh sebagian besar masyarakat. Data Badan Pusat Statistik (2023) mencatat adanya backlog perumahan yang mencapai lebih dari 12 juta unit, mencerminkan krisis kepemilikan rumah yang kian serius di Indonesia.

Dalam situasi ini, kemampuan untuk memprediksi harga rumah secara akurat menjadi sangat penting. Prediksi harga tidak hanya membantu calon pembeli dalam menyusun anggaran, tetapi juga mendukung pengembang dalam merancang strategi harga yang kompetitif, serta menjadi dasar bagi pengambilan keputusan investasi oleh pelaku pasar properti. Selain itu, pemerintah juga memerlukan estimasi harga yang presisi untuk merancang kebijakan subsidi dan perencanaan tata kota yang inklusif.

Penelitian dari Al Maula et al. (2025) menunjukkan bahwa tren permukiman telah bergeser ke daerah-daerah pinggiran seperti Tangerang Selatan, yang menawarkan harga lebih terjangkau namun tetap menghadirkan tantangan dalam aspek infrastruktur dan pemerataan pembangunan. Di sisi lain, studi oleh El Mouna et al. (2023) menegaskan bahwa harga rumah sangat dipengaruhi oleh berbagai faktor—termasuk ukuran, lokasi, akses terhadap fasilitas umum, dan usia bangunan—dan bahwa pendekatan *machine learning* mampu memberikan hasil prediksi yang lebih adaptif dibanding metode konvensional.

Oleh karena itu, proyek ini bertujuan untuk membangun model prediksi harga rumah di kawasan JABODETABEK dengan memanfaatkan algoritma *machine learning*, khususnya teknik *ensemble regression* seperti *Random Forest* dan *Gradient Boosting*. Model ini akan diperkuat dengan proses *feature engineering* yang bertujuan mengekstraksi informasi penting dari data historis perumahan. Pendekatan ini diharapkan mampu memberikan kontribusi nyata dalam pemetaan harga properti, pengambilan keputusan berbasis data, dan perencanaan kota yang lebih berkelanjutan.



---


## 🎯 2. Business Understanding

### 🔍 Problem Statements

* Bagaimana memprediksi harga rumah di wilayah JABODETABEK secara akurat dengan mempertimbangkan variabel seperti ukuran bangunan, lokasi geografis, jumlah kamar, akses terhadap fasilitas umum, dan usia bangunan?
* Algoritma *machine learning* mana yang paling efektif dalam menghasilkan prediksi harga rumah di kawasan urban Indonesia yang kompleks dan heterogen, seperti JABODETABEK?

### 🎯 Objectives

* Membangun model prediksi harga rumah yang akurat dan dapat diandalkan untuk mendukung pengambilan keputusan oleh pembeli, pengembang, investor, dan perencana kota.
* Membandingkan performa beberapa algoritma *machine learning*, yaitu **Linear Regression**, **MLP Regressor**, dan **XGBoost Regressor**, dalam memodelkan harga rumah berdasarkan data properti.
* Mengidentifikasi fitur-fitur yang paling berpengaruh terhadap harga rumah sebagai dasar untuk kebijakan tata ruang, subsidi perumahan, dan strategi investasi.

### 💡 Solusi

1. Menggunakan **Linear Regression** sebagai baseline model untuk memberikan interpretasi sederhana terhadap hubungan antar fitur dan harga rumah.
2. Menerapkan **MLP Regressor** (Multi-Layer Perceptron) untuk menangkap hubungan non-linear yang mungkin terjadi antar variabel input, seperti interaksi kompleks antara lokasi dan fasilitas.
3. Mengimplementasikan **XGBoost Regressor** sebagai model utama berbasis ensemble dengan kapabilitas pembelajaran adaptif, dilengkapi proses **hyperparameter tuning** menggunakan `RandomizedSearchCV` untuk mengoptimalkan kinerja prediksi.

Pendekatan ini dirancang untuk menjawab tantangan pasar properti di JABODETABEK, yang ditandai oleh tekanan urbanisasi, disparitas harga antar wilayah, dan kebutuhan akan sistem pengambilan keputusan berbasis data sebagaimana disoroti dalam studi Al Maula et al. (2025) dan El Mouna et al. (2023).


---

## 📁 3. Dataset Overview

* **Sumber**: Kaggle - [Daftar Harga Rumah JABODETABEK](https://www.kaggle.com/datasets/nafisbarizki/daftar-harga-rumah-jabodetabek)
* **Jumlah entri**: 3.553 baris
* **Jumlah fitur**: 27 kolom
* **Penulis dataset**: Nafis Barizki

---

## 📋 4. Fitur Dataset

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

| Langkah               | Penjelasan                                                                                                                                                                                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Handling Missing**  | - Menghapus kolom dengan jumlah **missing value lebih dari 50 baris** <br> - Mengisi nilai hilang pada kolom **numerik** dengan **median** dan kolom **kategorikal** dengan **mode**                                                                                |
| **Outlier Treatment** | Menggunakan metode **Winsorization** berbasis **IQR (Interquartile Range)** untuk memangkas nilai ekstrem dari data numerik, sehingga distribusi menjadi lebih stabil tanpa menghilangkan data asli.                                                                |
| **Encoding**          | - **Label Encoding** untuk kolom `district` dan `city` (karena memiliki banyak kategori dan bersifat ordinal geografis) <br> - **One-Hot Encoding** (dengan `drop='first'`) untuk `property_type` agar menghindari dummy trap                                       |
| **Ekstraksi Data**    | - Menyaring nilai numerik dari kolom `electricity` <br> - Melakukan `.str.strip()` pada `city` untuk menghindari spasi yang tidak diperlukan                                                                                                                        |
| **Drop Columns**      | Menghapus kolom yang tidak relevan untuk prediksi harga seperti: `url`, `title`, `ads_id`, `address`, `facilities`, `lat`, `long`                                                                                                                                   |
| **Scaling**           | - Menggunakan **RobustScaler** untuk kolom-kolom numerik dan target `price_in_rp`, karena **harga properti cenderung memiliki distribusi skewed dan banyak outlier** <br> - RobustScaler tahan terhadap outlier karena menggunakan median dan IQR untuk normalisasi |
| **Train-Test Split**  | Membagi dataset menjadi **80% data training** dan **20% data testing** menggunakan `train_test_split` dari Scikit-learn dengan `random_state=42` untuk reproducibility                                                                                              |

---

### 🧭 Alasan Penggunaan RobustScaler

Pada kasus prediksi **harga rumah**, data sangat rentan terhadap **outlier** karena:

* Adanya properti mewah yang jauh lebih mahal dari harga rata-rata
* Variasi harga antar lokasi yang sangat signifikan

Penggunaan **`RobustScaler`** sangat tepat dalam konteks ini karena:

* Tidak menggunakan **mean** dan **standar deviasi** (yang sangat sensitif terhadap outlier)
* Berdasarkan **median** dan **interquartile range (IQR)**, sehingga dapat menstabilkan skala fitur numerik tanpa terdampak oleh nilai ekstrem

Hal ini membantu model belajar dengan **lebih stabil** dan **menghindari bias ekstrim** akibat nilai harga yang sangat tinggi atau rendah.


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


Sebagai bagian integral dari *business understanding*, evaluasi ini bertujuan untuk menilai seberapa **akurat dan andal** model prediksi harga rumah yang dibangun. Akurasi prediksi sangat penting agar hasil dari model dapat digunakan secara nyata oleh berbagai pemangku kepentingan seperti **calon pembeli**, **pengembang properti**, **investor**, dan **pemerintah**, dalam rangka mengambil **keputusan strategis** terkait properti di wilayah JABODETABEK.

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

#### **1. Bagaimana memprediksi harga rumah di JABODETABEK berdasarkan fitur properti dan lokasi?**

Model dikembangkan dengan menggunakan data yang mencakup berbagai fitur penting: ukuran bangunan, lokasi, jumlah kamar, usia bangunan, dan akses terhadap fasilitas umum. Ketiga model regresi—**Linear Regression**, **MLP Regressor**, dan **XGBoost Regressor**—dilatih dan dievaluasi berdasarkan performa prediksi harga rumah. Model XGBoost terbukti paling efektif dalam menangkap kompleksitas hubungan antar fitur.

#### **2. Algoritma machine learning mana yang paling efektif untuk regresi harga rumah di kawasan urban Indonesia?**

Hasil evaluasi menunjukkan bahwa:

* **Linear Regression** memiliki performa baseline yang cukup baik (*R² = 0.8646*), namun masih terbatas dalam menangkap hubungan non-linear.
* **MLP Regressor** memperbaiki performa tersebut secara signifikan (*R² = 0.9022*), menangani kompleksitas yang lebih tinggi.
* **XGBoost Regressor** menghasilkan performa **terbaik** dengan nilai *R² = 0.9313* dan *MAE = 0.1152*, menjadikannya model paling efektif dan siap digunakan untuk deployment.

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

Melalui proses evaluasi yang sistematis dan berorientasi pada kebutuhan bisnis, dapat disimpulkan bahwa:

> 🔎 **Model XGBoost Regressor** merupakan model terbaik dengan *R² = 0.9313* dan *MAPE = 8.36%*, menunjukkan kemampuan sangat baik dalam memodelkan variasi harga rumah di wilayah JABODETABEK.

Model ini secara efektif menjawab permasalahan utama dalam domain proyek: **"Bagaimana memprediksi harga rumah secara akurat untuk mendukung keputusan strategis berbagai pihak?"** Dengan performa tinggi dan interpretabilitas yang kuat, model ini layak untuk diimplementasikan dalam sistem rekomendasi harga properti atau sebagai dasar pertimbangan kebijakan pembangunan kawasan urban.





---

## 📌 10. Feature Importance

Visualisasi menggunakan `.feature_importances_` dari XGBoost menunjukkan bahwa:

* `building_size_m2` dan `land_size_m2` adalah kontributor utama harga.
* `property_type`, `furnishing`, dan `location (district)` juga signifikan.

---

## 🚀 11. Deployment (Opsional)

> Untuk keperluan praktis, model dapat di-*pickle* dan disajikan via API dengan **Flask** atau **Streamlit** sebagai antarmuka interaktif prediksi harga rumah berdasarkan input pengguna.

---

## 📚 12. Referensi

* El Mouna, L., et al. (2023). *A Comparative Study of Urban House Price Prediction using Machine Learning Algorithms*. E3S Web of Conferences.
* Al Maula, S. F., et al. (2025). *Modeling House Selling Prices in Jakarta and South Tangerang Using Machine Learning*. BAREKENG.
* Badan Pusat Statistik (2023). *Statistik Perumahan Nasional*.

---

## ✍️ 13. Penutup

Prediksi harga rumah di wilayah urban seperti JABODETABEK menjadi semakin penting di tengah ketimpangan permintaan dan ketersediaan lahan. Pendekatan machine learning dengan ensemble regression seperti XGBoost terbukti memberikan prediksi yang sangat akurat. Hasil ini tidak hanya berkontribusi pada pengambilan keputusan ekonomi yang lebih baik, namun juga berpotensi menjadi dasar sistem rekomendasi harga rumah berbasis data di masa depan.

