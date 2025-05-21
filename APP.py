import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

# Load and prepare data
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Train model (built-in, avoids loading external files)
model = GaussianNB()
model.fit(X, y)

# Create DataFrame for visualization
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
df["target_name"] = df["target"].apply(lambda i: target_names[i])

# Streamlit config
st.set_page_config(page_title="Iris Classifier", layout="centered")

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["Data Description", "Prediction", "Visualization"])

# Page 1: Data Description
if page == "Data Description":
    st.title("🌼 Iris Dataset Overview")
    st.markdown("This dataset includes measurements of iris flowers from three species:")
    st.markdown("- Setosa\n- Versicolor\n- Virginica")
    st.write("Preview of the dataset:")
    st.dataframe(df.head())

    st.markdown("### Features")
    for f in feature_names:
        st.markdown(f"- {f.capitalize()}")

# Page 2: Prediction
elif page == "Prediction":
    st.title("🔎 Predict Iris Species")

    st.markdown("Enter flower measurements:")

    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if st.button("Predict"):
        pred = model.predict(input_data)[0]
        pred_name = target_names[pred]
        st.success(f"The predicted species is **{pred_name}** 🌸")

# Page 3: Visualization
elif page == "Visualization":
    st.title("📊 Data Visualization")

    st.markdown("Here are some visual insights from the Iris dataset.")

    st.subheader("Pairplot (by species)")
    fig = sns.pairplot(df, hue="target_name")
    st.pyplot(fig)

    st.subheader("Feature Correlation Heatmap")
    corr = df.iloc[:, :-2].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
