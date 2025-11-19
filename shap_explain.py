import shap
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load dataset (for background data)
data = load_breast_cancer()
X = data.data
feature_names = data.feature_names

# Load model and scaler
model = joblib.load("svm_model.joblib")
scaler = joblib.load("scaler.joblib")

# Scale dataset
X_scaled = scaler.transform(X)

# Create SHAP explainer using 50 samples (faster)
background = shap.kmeans(X_scaled, 50)
explainer = shap.KernelExplainer(model.predict_proba, background)

def explain_single_prediction(input_data):
    """
    input_data: 1 sample (scaled)
    returns: SHAP values for malignant (class 1)
    """
    shap_values = explainer.shap_values(input_data)
    return shap_values[1]   # class 1 (benign)

if st.button("Explain Prediction"):
    st.subheader("Why this prediction was made:")

    # Prepare SHAP values
    scaled_single = scaler.transform([inputs])
    shap_values = explain_single_prediction(scaled_single)

    # ---------------------------
    # BAR CHART (FIXED VERSION)
    # ---------------------------
    st.write("### Feature impact on prediction")

    fig, ax = plt.subplots(figsize=(8, 10))
    shap_values_flat = shap_values.flatten()

    # Create bar chart
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, shap_values_flat)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()  # biggest value at top
    ax.set_xlabel("SHAP value")
    ax.set_title("Feature Impact (SHAP Values)")

    st.pyplot(fig)

st.write("### Global Feature Importance (Summary Plot)")
summary_fig = plt.figure()
shap.summary_plot(shap_values, scaled_single, feature_names=feature_names, show=False)
st.pyplot(summary_fig)
