# heart_disease_app.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------- Load Dataset ------------------- #
@st.cache_data
def load_data():
    return pd.read_csv("Heart Disease/dataset.csv", encoding="ISO-8859-1")

df = load_data()

# ------------------- Page Title ------------------- #
st.title("❤️ Heart Disease Detection App")
st.markdown("Predicts the likelihood of heart disease based on patient data.")

# ------------------- Exploratory Data Analysis ------------------- #
with st.expander("🔍 Show Data Preview & Visualizations"):
    st.subheader("Raw Dataset")
    st.write(df.head())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Chest Pain Type vs Heart Disease")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x="chest pain type", hue="target", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Age Distribution by Heart Disease Status")
    fig3, ax3 = plt.subplots()
    sns.histplot(data=df, x="age", hue="target", kde=True, bins=30, ax=ax3)
    st.pyplot(fig3)

# ------------------- Model Training ------------------- #
@st.cache_resource
def train_model(data):
    X = data.drop("target", axis=1)
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

model, accuracy = train_model(df)
st.success(f"Model trained with **{accuracy * 100:.2f}%** accuracy.")

# ------------------- User Inputs ------------------- #
st.header("🩺 Enter Patient Details")

age = st.slider("Age", 20, 90, 50)
sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
ecg = st.selectbox("Resting ECG Result", [0, 1, 2])
max_hr = st.slider("Maximum Heart Rate Achieved", 70, 210, 150)
ex_angina = st.radio("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("ST depression induced by exercise", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST segment", [1, 2, 3])

# ------------------- Prediction ------------------- #
if st.button("🧠 Predict Heart Disease"):
    input_data = pd.DataFrame([[age, sex, cp, bp, chol, fbs, ecg, max_hr, ex_angina, oldpeak, slope]],
                              columns=['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
                                       'fasting blood sugar', 'resting ecg', 'max heart rate',
                                       'exercise angina', 'oldpeak', 'ST slope'])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High likelihood of Heart Disease.")
    else:
        st.success("✅ Low likelihood of Heart Disease.")

