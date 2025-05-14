import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
import joblib
import os


def load_data(path='data/jabodetabek_house_price.csv'):
    data = pd.read_csv(path)
    print(data.info())
    print(data.describe())
    print(data.head())
    print(data.isnull().sum())
    return data


def check_duplicates(data):
    duplicate_rows = data[data.duplicated(keep=False)]
    print("Baris duplikat:")
    print(duplicate_rows)
    print(f"Jumlah baris duplikat (selain baris pertama): {data.duplicated().sum()}")


def visualize_distributions(data):
    os.makedirs("images", exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.histplot(data['price_in_rp'], kde=True)
    plt.xscale('log')
    plt.title("Distribusi Harga Rumah (Log Scale)")
    plt.xlabel("Harga Rumah (Rp)")
    plt.ylabel("Frekuensi")
    plt.tight_layout()
    plt.savefig("images/distribusi_harga.png")
    plt.close()

    top_cities = data['city'].value_counts().nlargest(10).index
    filtered = data[data['city'].isin(top_cities)]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='building_size_m2', y='price_in_rp', hue='city', data=filtered, palette='tab10')
    plt.yscale('log')
    plt.title("Ukuran Bangunan vs Harga Rumah")
    plt.xlabel("Luas Bangunan (m2)")
    plt.ylabel("Harga Rumah (Rp)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("images/luas_bangunan_vs_harga.png")
    plt.close()

    ct = pd.crosstab(data['furnishing'], data['property_condition'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu')
    plt.title("Hubungan Furnishing dan Kondisi Properti")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("images/furnishing_vs_condition.png")
    plt.close()

    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Heatmap Korelasi Antar Fitur Numerik')
    plt.tight_layout()
    plt.savefig("images/heatmap_korelasi.png")
    plt.close()


def clean_data(data):
    data_original = data.copy()
    null_counts = data.isnull().sum()
    cols_to_drop = null_counts[null_counts > 50].index
    data_cleaned = data_original.drop(columns=cols_to_drop)

    for col in data_cleaned.columns:
        if data_cleaned[col].isnull().sum() > 0:
            if data_cleaned[col].dtype in ['float64', 'int64']:
                data_cleaned[col] = data_cleaned[col].fillna(data_cleaned[col].median())
            else:
                data_cleaned[col] = data_cleaned[col].fillna(data_cleaned[col].mode()[0])

    return data_cleaned


def treat_outliers(data):
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

    for feature in numeric_cols:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data[feature] = np.where(data[feature] < lower, lower, data[feature])
        data[feature] = np.where(data[feature] > upper, upper, data[feature])

    return data


def drop_unused_columns(data):
    drop_cols = ['url', 'title', 'ads_id', 'address', 'facilities', 'lat', 'long']
    return data.drop(columns=drop_cols)


def encode_and_split_data(data, target_column='price_in_rp'):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(X_encoded.columns.tolist(), 'models/feature_columns.pkl')

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_and_save_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/xgb_model.pkl')
    print("Model berhasil dilatih dan disimpan.")
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("Evaluasi Model:")
    print(f"MAE  : {mean_absolute_error(y_test, predictions):,.2f}")
    print(f"MSE  : {mean_squared_error(y_test, predictions):,.2f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, predictions)):.2f}")
    print(f"R2   : {r2_score(y_test, predictions):.4f}")
    print(f"MAPE : {mean_absolute_percentage_error(y_test, predictions):.4f}")


def inference_test(model_path='models/xgb_model.pkl', scaler_path='models/scaler.pkl', feature_path='models/feature_columns.pkl'):
    sample_input = {
        'building_size_m2': 120,
        'land_size_m2': 150,
        'bedrooms': 3,
        'bathrooms': 2,
        'floors': 2,
        'carports': 1,
        'electricity': 2200,
        'certificate': 'SHM',
        'furnishing': 'Unfurnished',
        'property_condition': 'New',
        'city': 'Jakarta Selatan',
        'type': 'Secondary'
    }

    # Load model dan scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(feature_path)

    # Convert input ke DataFrame
    input_df = pd.DataFrame([sample_input])

    # OneHot Encoding sesuai feature training
    input_encoded = pd.get_dummies(input_df)
    for col in feature_columns:
        if col not in input_encoded:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_columns]

    # Scaling
    input_scaled = scaler.transform(input_encoded)

    # Prediksi
    predicted_price = model.predict(input_scaled)[0]
    print(f"\nHasil Prediksi Harga Rumah: Rp {predicted_price:,.0f}")


def main():
    data = load_data()
    check_duplicates(data)
    visualize_distributions(data)
    cleaned_data = clean_data(data)
    cleaned_data = treat_outliers(cleaned_data)
    cleaned_data = drop_unused_columns(cleaned_data)

    print("\nInfo data setelah pembersihan dan treatment:")
    print(cleaned_data.info())

    X_train, X_test, y_train, y_test = encode_and_split_data(cleaned_data)
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    model = train_and_save_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    inference_test()


if __name__ == "__main__":
    main()
