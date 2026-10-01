import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")

st.title("🍷 Red Wine Quality Prediction")
st.write("Enter the chemical properties of the wine and click Predict.")

try:
    model = joblib.load("best_wine_model.pkl")
except FileNotFoundError:
    st.error("best_wine_model.pkl not found. Run the GridSearchCV notebook first.")
    st.stop()

st.subheader("Wine Properties")

fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", value=0.70)
citric_acid = st.number_input("Citric Acid", value=0.00)
residual_sugar = st.number_input("Residual Sugar", value=1.9)
chlorides = st.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
density = st.number_input("Density", value=0.9978, format="%.4f")
ph = st.number_input("pH", value=3.51)
sulphates = st.number_input("Sulphates", value=0.56)
alcohol = st.number_input("Alcohol", value=9.4)

if st.button("Predict Wine Quality"):
    input_data = pd.DataFrame({
        "fixed acidity": [fixed_acidity],
        "volatile acidity": [volatile_acidity],
        "citric acid": [citric_acid],
        "residual sugar": [residual_sugar],
        "chlorides": [chlorides],
        "free sulfur dioxide": [free_sulfur_dioxide],
        "total sulfur dioxide": [total_sulfur_dioxide],
        "density": [density],
        "pH": [ph],
        "sulphates": [sulphates],
        "alcohol": [alcohol]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Wine Quality: {prediction}")

    if prediction >= 7:
        st.info("The model predicts a high quality wine.")
    elif prediction >= 5:
        st.info("The model predicts a medium quality wine.")
    else:
        st.info("The model predicts a lower quality wine.")

st.caption("Model: Random Forest tuned using GridSearchCV")
