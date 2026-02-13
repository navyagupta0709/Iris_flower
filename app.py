import streamlit as st
import pickle
import numpy as np
import os

st.title("Navya's Iris Prediction App 🌸")

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = pickle.load(open("model.pkl", "rb"))

sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Class: {prediction[0]}")
