import streamlit as st
import numpy as np
import pandas as pd
import pickle
#import mlflow.pyfunc

# -----------------------------
# 🎯 Load Scaler (local pickle)
# -----------------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# 🎯 Load Model from MLflow Registry
# -----------------------------
# Make sure your MLflow tracking URI is configured
#mlflow.set_tracking_uri("http://127.0.0.1:5000")
# and model is registered as "IrisModel" with stage "Production"

# model = mlflow.pyfunc.load_model("models:/RandomForest/1")


with open("RandomForest_model.pkl",'rb') as f:
    model = pickle.load(f)

# -----------------------------
# 🎨 Streamlit UI
# -----------------------------
st.title("🌸 Iris Flower Classifier (MLflow Production Model)")

st.write("Enter flower measurements to predict the Iris species:")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width  = st.number_input("Sepal Width (cm)",  min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width  = st.number_input("Petal Width (cm)",  min_value=0.0, max_value=10.0, value=0.2)

# Prepare input
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
scaled_input = scaler.transform(input_data)
input_df = pd.DataFrame(scaled_input, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

# Predict
if st.button("Predict Iris Species"):
    prediction = model.predict(input_df)
    species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    result = species_map[int(prediction[0])] if int(prediction[0]) in species_map else str(prediction[0])

    st.success(f"🌼 Predicted Species: **{result}**")

    # optional: show input
    st.write("Scaled Input:", input_df)

st.markdown("---")
st.caption("Model served from MLflow Registry (Production stage)")
