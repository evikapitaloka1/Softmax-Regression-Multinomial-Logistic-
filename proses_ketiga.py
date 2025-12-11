# proses_ketiga.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import time

# ======= Load Dataset =======
df = pd.read_csv("heart.csv")

# ======= 1. Cek Missing Value =======
print("=== Pengecekan Missing Value ===")
print(df.isnull().sum())

# Pisahkan fitur & label
X = df.drop(columns=['target'])
y = df['target']

# ======= 2. Transformasi: MinMaxScaler =======
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ======= 3. Ekstraksi Fitur (LDA) =======
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X_scaled, y)

# ======= Split Data =======
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)

# ======= 4. Klasifikasi (Softmax Regression) =======
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

start_train = time.time()
model.fit(X_train, y_train)
end_train = time.time()

start_test = time.time()
y_pred = model.predict(X_test)
end_test = time.time()

# ======= Evaluasi =======
print("\n=== Evaluasi Softmax Regression (Proses Ketiga: LDA tanpa SMOTE) ===")
print(f"Akurasi        : {accuracy_score(y_test, y_pred):.4f}")
print(f"Presisi (Macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall (Macro) : {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"Waktu Training : {end_train - start_train:.6f} detik")
print(f"Waktu Testing  : {end_test - start_test:.6f} detik")

# Hitung AUC
try:
    y_proba = model.predict_proba(X_test)
    auc_score = roc_auc_score(y_test, y_proba[:, 1])
    print(f"AUC            : {auc_score:.4f}")
except:
    print("AUC tidak dapat dihitung.")
