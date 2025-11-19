import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

import shap
from shap_explain import explain_single_prediction, explainer, feature_names
import matplotlib.pyplot as plt


# load model and data
model = joblib.load('svm_model.joblib')
scaler = joblib.load('scaler.joblib')
data = load_breast_cancer()

st.title("Breast Cancer Prediction App")

# user input form
st.write("Enter tumor feature values:")
inputs = []
for feature in data.feature_names:
    val = st.number_input(feature, value=float(np.mean(data.data[:, list(data.feature_names).index(feature)])))
    inputs.append(val)

if st.button("Predict"):
    scaled = scaler.transform([inputs])
    pred = model.predict(scaled)
    st.subheader("Prediction: " + ("Benign" if pred[0] == 1 else "Malignant"))

if st.button("Explain Prediction"):
    st.subheader("Why this prediction was made:")

    # Prepare SHAP values
    scaled_single = scaler.transform([inputs])
    shap_values = explain_single_prediction(scaled_single)

    # Show SHAP bar chart
    st.write("### Feature impact on prediction")
    shap_fig = shap.plots.bar(shap_values, show=False)
    st.pyplot(shap_fig)

    # Force plot (HTML)
    st.write("### SHAP Force Plot")
    force_plot_html = shap.force_plot(
        explainer.expected_value[1],
        shap_values,
        scaled_single,
        feature_names=feature_names,
        matplotlib=False
    )

    # Display in Streamlit
    st.components.v1.html(shap.getjshtml() + force_plot_html.html())

