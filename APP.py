import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_iris

# Load model dari file model_saya.pkl
model = joblib.load("naive_bayes_model.pkl")

# Load dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# DataFrame untuk eksplorasi dan visualisasi
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
df["target_name"] = df["target"].apply(lambda i: target_names[i])

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Iris", layout="centered")

# Sidebar navigasi
page = st.sidebar.radio("Navigasi", ["Deskripsi Data", "Prediksi", "Visualisasi"])

# Halaman 1: Deskripsi
if page == "Deskripsi Data":
    st.title("📄 Deskripsi Dataset Iris")
    st.write("""
        Dataset Iris terdiri dari 150 data bunga dari 3 spesies:
        - Setosa
        - Versicolor
        - Virginica
    """)
    st.dataframe(df.head())

    st.markdown("### Fitur:")
    for f in feature_names:
        st.markdown(f"- {f.capitalize()}")

# Halaman 2: Prediksi
elif page == "Prediksi":
    st.title("🔍 Prediksi Spesies Iris")
    st.markdown("Masukkan nilai-nilai fitur bunga:")

    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

  if st.button("Prediksi"):
        pred = model.predict(input_data)[0]
        st.success(f"🌸 Prediksi: **{target_names[pred]}**")


# Halaman 3: Visualisasi
elif page == "Visualisasi":
    st.title("📊 Visualisasi Dataset Iris")
    st.markdown("Analisis visual dari fitur-fitur dalam dataset.")

    st.subheader("Pairplot")
    fig = sns.pairplot(df, hue="target_name")
    st.pyplot(fig)

    st.subheader("Heatmap Korelasi")
    fig2, ax = plt.subplots()
    sns.heatmap(df.iloc[:, :4].corr(), annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig2)
