import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2

def proses_keempat():
    # Load Dataset
    try:
        df = pd.read_csv('heart.csv')
    except FileNotFoundError:
        print("File heart.csv tidak ditemukan.")
        return

    # --- 1. Cek Missing Value ---
    print("=== 1. Cek Missing Value ===")
    missing = df.isnull().sum()
    print(missing)
    if missing.sum() == 0:
        print("\n[INFO] Tidak ada missing value dalam dataset.")
    else:
        print("\n[INFO] Terdapat missing value.")

    # Persiapan Data (Pisahkan Fitur X dan Target y)
    X = df.drop('target', axis=1)
    y = df['target']

    # --- 2. Transformasi --> MinMaxScaler ---
    print("\n=== 2. Transformasi (MinMaxScaler) ===")
    scaler = MinMaxScaler()
    # Fit dan transform fitur
    X_scaled = scaler.fit_transform(X)
    # Ubah kembali ke DataFrame agar nama kolom tetap ada
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    print("Data setelah normalisasi (5 baris pertama):")
    print(X_scaled_df.head())

    # --- 4. Metode Spesifikasi Sebanyak 3 (Seleksi Fitur) ---
    # Interpretasi: Memilih 3 fitur terbaik (SelectKBest dengan k=3)
    print("\n=== 4. Metode Spesifikasi Sebanyak 3 (SelectKBest k=3) ===")
    # Menggunakan Chi-Square (chi2) karena data sudah non-negatif (MinMax)
    selector = SelectKBest(score_func=chi2, k=3)
    X_selected = selector.fit_transform(X_scaled_df, y)
    
    # Mendapatkan nama kolom yang terpilih
    selected_features = X.columns[selector.get_support()]
    
    print(f"3 Fitur Terpilih: {selected_features.tolist()}")
    print("\nData dengan 3 fitur spesifik (5 baris pertama):")
    X_final = pd.DataFrame(X_selected, columns=selected_features)
    print(X_final.head())

if __name__ == "__main__":
    proses_keempat()