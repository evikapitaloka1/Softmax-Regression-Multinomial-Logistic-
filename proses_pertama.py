# proses_pertama.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import time

# ======= Load Dataset =======
df = pd.read_csv("heart.csv")   # Upload di Colab lalu sesuaikan nama file

# ======= 1. Cek Missing Value =======
print("Missing Value per Kolom:")
print(df.isnull().sum())

# ======= Pisahkan fitur & label =======
X = df.drop(columns=['target'])
y = df['target']

# ======= 2. Transformasi: MinMaxScaler =======
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ======= 3. Ekstraksi Fitur (LDA) =======
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X_scaled, y)

# ======= 4. Imbalanced Data (SMOTE) =======
sm = SMOTE()
X_res, y_res = sm.fit_resample(X_lda, y)

# ======= Split =======
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)

# ======= 5. Softmax Regression =======
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)

start_train = time.time()
model.fit(X_train, y_train)
end_train = time.time()

start_test = time.time()
y_pred = model.predict(X_test)
end_test = time.time()

# ======= Evaluasi =======
print("\n=== Evaluasi Softmax Regression (Proses Pertama) ===")
print("Akurasi :", accuracy_score(y_test, y_pred))
print("Presisi :", precision_score(y_test, y_pred, average='macro'))
print("Recall  :", recall_score(y_test, y_pred, average='macro'))
print("Waktu Training :", end_train - start_train, "detik")
print("Waktu Testing  :", end_test - start_test, "detik")

try:
    y_proba = model.predict_proba(X_test)
    print("AUC      :", roc_auc_score(y_test, y_proba[:,1]))
except:
    print("AUC tidak dapat dihitung untuk dataset ini")
