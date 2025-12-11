import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from imblearn.over_sampling import SMOTE

st.title("Aplikasi Analisis Heart Disease (4 Proses)")

st.write("""
Aplikasi ini menjalankan 4 proses analisis sesuai diagram:
1. Normalisasi + LDA + SMOTE  
2. Normalisasi  
3. Normalisasi + LDA  
4. Normalisasi + SMOTE  
""")

# =============================
# Upload Dataset
# =============================
uploaded = st.file_uploader("Upload file heart.csv", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("Dataset berhasil dimuat!")
    st.dataframe(df.head())

    proses = st.selectbox(
        "Pilih Proses:",
        ["Proses 1", "Proses 2", "Proses 3", "Proses 4"]
    )

    X = df.drop(columns=['target'])
    y = df['target']

    st.subheader("1. Cek Missing Value")
    st.write(df.isnull().sum())

    # ====== NORMALISASI ======
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    st.subheader("2. Normalisasi Data")
    st.write(pd.DataFrame(X_scaled, columns=X.columns).head())

    # =========================
    # PROSES 1
    # =========================
    if proses == "Proses 1":
        st.header("PROSES 1: MinMax + LDA + SMOTE")

        lda = LinearDiscriminantAnalysis()
        X_trans = lda.fit_transform(X_scaled, y)

        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X_trans, y)

    # =========================
    # PROSES 2
    # =========================
    elif proses == "Proses 2":
        st.header("PROSES 2: MinMax (tanpa LDA/SMOTE)")
        X_res, y_res = X_scaled, y

    # =========================
    # PROSES 3
    # =========================
    elif proses == "Proses 3":
        st.header("PROSES 3: MinMax + LDA (tanpa SMOTE)")
        lda = LinearDiscriminantAnalysis()
        X_res, y_res = lda.fit_transform(X_scaled, y), y

    # =========================
    # PROSES 4
    # =========================
    elif proses == "Proses 4":
        st.header("PROSES 4: MinMax + SMOTE")
        
        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X_scaled, y)

        # Grafik distribusi SMOTE
        st.subheader("Distribusi Data Setelah SMOTE")
        fig, ax = plt.subplots()
        ax.bar(["Before 0", "Before 1", "After 0", "After 1"],
               [y.value_counts()[0], y.value_counts()[1],
                y_res.value_counts()[0], y_res.value_counts()[1]])
        st.pyplot(fig)

    # =========================
    # TRAIN TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    # =========================
    # TRAIN MODEL
    # =========================
    model = LogisticRegression(max_iter=2000, multi_class="multinomial")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # =========================
    # EVALUASI
    # =========================
    st.subheader("Evaluasi Model")

    st.write("**Akurasi:**", accuracy_score(y_test, y_pred))
    st.write("**Presisi:**", precision_score(y_test, y_pred, average="macro"))
    st.write("**Recall:**", recall_score(y_test, y_pred, average="macro"))

    # AUC  
    try:
        y_proba = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba[:, 1])
        st.write("**AUC:**", auc)
    except:
        st.warning("AUC tidak bisa dihitung.")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax_cm)
    st.pyplot(fig_cm)

    # ROC Curve
    st.subheader("ROC Curve")
    try:
        fig_roc, ax_roc = plt.subplots()
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc)
        st.pyplot(fig_roc)
    except:
        st.warning("ROC Curve tidak dapat ditampilkan.")

else:
    st.info("Silakan upload file heart.csv untuk memulai analisis.")
