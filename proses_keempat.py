# proses_keempat.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import time

def proses_keempat():
    # ===== Load Dataset =====
    df = pd.read_csv("heart.csv")

    # ===== 1. Cek Missing Value =====
    print("=== 1. Cek Missing Value ===")
    missing = df.isnull().sum()
    print(missing)

    # Pisahkan fitur dan target
    X = df.drop(columns=['target'])
    y = df['target']

    # ===== 2. Transformasi MinMaxScaler =====
    print("\n=== 2. Transformasi: MinMaxScaler ===")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # ===== 3. Imbalanced Data: SMOTE =====
    print("\n=== 3. Imbalanced Data: SMOTE ===")
    sm = SMOTE()
    X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

    print(f"Jumlah sebelum SMOTE : {len(y)}")
    print(f"Jumlah sesudah SMOTE : {len(y_resampled)}")

    # ===== Simpan Grafik Distribusi SMOTE =====
    plt.figure()
    before = y.value_counts()
    after = y_resampled.value_counts()
    plt.bar(["Class 0 (Before)", "Class 1 (Before)", 
             "Class 0 (After)", "Class 1 (After)"],
            [before[0], before[1], after[0], after[1]])
    plt.title("Distribusi Data Sebelum & Sesudah SMOTE")
    plt.savefig("smote_distribution_proses4.png")
    plt.close()

    # ===== Split Data =====
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    # ===== 4. Metode Klasifikasi =====
    print("\n=== 4. Training Model (Softmax Regression) ===")
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

    # Waktu training
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()

    # Waktu testing
    start_test = time.time()
    y_pred = model.predict(X_test)
    end_test = time.time()

    # ===== 5. Evaluasi =====
    print("\n=== Evaluasi Softmax Regression (Proses Keempat) ===")
    print(f"Akurasi        : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Presisi (Macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall (Macro) : {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Waktu Training : {end_train - start_train:.6f} detik")
    print(f"Waktu Testing  : {end_test - start_test:.6f} detik")

    # ===== AUC Score + Simpan ROC Curve =====
    try:
        y_proba = model.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, y_proba[:, 1])
        print(f"AUC            : {auc_score:.4f}")

        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title("ROC Curve - Proses Keempat")
        plt.savefig("roc_proses4.png")
        plt.close()

    except:
        print("AUC tidak dapat dihitung.")

    # ===== Confusion Matrix =====
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix - Proses Keempat")
    plt.savefig("confusion_matrix_proses4.png")
    plt.close()


if __name__ == "__main__":
    proses_keempat()
