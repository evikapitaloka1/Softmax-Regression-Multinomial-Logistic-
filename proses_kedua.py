import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
import time

# ======= Load Dataset =======
df = pd.read_csv("heart.csv")

# ======= 1. Cek Missing Value =======
print("=== Pengecekan Missing Value ===")
missing_values = df.isnull().sum()
print(missing_values)

# ======= Pisahkan fitur & label =======
X = df.drop(columns=['target'])
y = df['target']

# ======= 2. Transformasi: MinMaxScaler =======
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ======= Split Data =======
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ======= 3. Softmax Regression =======
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# --- Waktu Training ---
start_train = time.time()
model.fit(X_train, y_train)
end_train = time.time()

# --- Waktu Testing ---
start_test = time.time()
y_pred = model.predict(X_test)
end_test = time.time()

# ======= Evaluasi =======
print("\n=== Evaluasi Softmax Regression (Proses Kedua: Tanpa LDA & SMOTE) ===")

# Metrik Dasar
print(f"Akurasi        : {accuracy_score(y_test, y_pred):.4f}")
print(f"Presisi (Macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall (Macro) : {recall_score(y_test, y_pred, average='macro'):.4f}")

# Waktu Komputasi
print(f"Waktu Training : {end_train - start_train:.6f} detik")
print(f"Waktu Testing  : {end_test - start_test:.6f} detik")

# AUC Score
try:
    # Mengambil probabilitas kelas positif (1)
    y_proba = model.predict_proba(X_test)
    auc_score = roc_auc_score(y_test, y_proba[:, 1])
    print(f"AUC            : {auc_score:.4f}")
except Exception as e:
    print("AUC tidak dapat dihitung:", e)

# Laporan Klasifikasi 
print("\nClassification Report:")
print(classification_report(y_test, y_pred))