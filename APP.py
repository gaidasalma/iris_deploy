import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris.target_names

# Load model
@st.cache_resource
def load_model():
    return joblib.load('naive_bayes_model.pkl')

# Muat data dan model
df, target_names = load_data()
model = load_model()

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Menu", ["Deskripsi Data", "Prediksi", "Visualisasi"])

# Halaman 1: Deskripsi Data
if page == "Deskripsi Data":
    st.title("📊 Deskripsi Dataset Iris")
    st.write("""
        Dataset Iris terdiri dari 150 bunga dari tiga spesies:
        - Setosa
        - Versicolor
        - Virginica

        Masing-masing memiliki 4 fitur:
        - Sepal length (cm)
        - Sepal width (cm)
        - Petal length (cm)
        - Petal width (cm)
    """)
    st.subheader("Contoh Data")
    st.dataframe(df.head())

    st.subheader("Statistik Ringkasan")
    st.write(df.describe())

    st.subheader("Distribusi Spesies")
    st.bar_chart(df['species'].value_counts())

# Halaman 2: Prediksi
elif page == "Prediksi":
    st.title("🔍 Prediksi Spesies Iris")

    # Ambil range nilai dari data
    def col_min(col): return float(df[col].min())
    def col_max(col): return float(df[col].max())
    def col_mean(col): return float(df[col].mean())

    sepal_length = st.slider("Sepal length (cm)", col_min('sepal length (cm)'), col_max('sepal length (cm)'), col_mean('sepal length (cm)'))
    sepal_width = st.slider("Sepal width (cm)", col_min('sepal width (cm)'), col_max('sepal width (cm)'), col_mean('sepal width (cm)'))
    petal_length = st.slider("Petal length (cm)", col_min('petal length (cm)'), col_max('petal length (cm)'), col_mean('petal length (cm)'))
    petal_width = st.slider("Petal width (cm)", col_min('petal width (cm)'), col_max('petal width (cm)'), col_mean('petal width (cm)'))

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if st.button("Prediksi"):
        pred = model.predict(input_data)[0]
       st.success(f"🌸 Spesies yang Diprediksi: **{pred.capitalize()}**")

# Halaman 3: Visualisasi
elif page == "Visualisasi":
    st.title("📈 Visualisasi Dataset Iris")

    st.subheader("Scatterplot Sepal Length vs Sepal Width")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species', ax=ax1)
    st.pyplot(fig1)

    st.subheader("Boxplot Petal Length per Spesies")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='species', y='petal length (cm)', data=df, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Pairplot Semua Fitur")
    st.write("Harap tunggu sejenak...")
    pairplot_fig = sns.pairplot(df, hue='species')
    st.pyplot(pairplot_fig.fig)
