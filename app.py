import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import numpy as np



model_options = {
    "Logistic Regression": "logistic_regression",
    "LSTM": "lstm",
    "Random Forest": "random_forest",
    "SVM": "svm",
    "XGBoost": "xgboost"
}


# Load the trained model


@st.cache_resource
def load_trained_model(model_name):
    if model_name == "logistic_regression":
        domain_model = joblib.load("models/LogisticRegression/domain_model.pkl")
        sub_domain_model = joblib.load("models/LogisticRegression/sub_domain_model.pkl")
        return domain_model, sub_domain_model
    # elif model_name == "lstm":
    #     domain_model = load_model("models/LSTM/simple_lstm_domain.h5")
    #     sub_domain_model = load_model("models/LSTM/simple_lstm_sub_domain.h5")
    #     return domain_model, sub_domain_model
    # elif model_name == "random_forest":
    #     domain_model = joblib.load("models/RDF/optimized_random_forest_domain_1.pkl")
    #     sub_domain_model = joblib.load("models/RDF/optimized_random_forest_sub_domain_1.pkl")
    #     return domain_model, sub_domain_model
    elif model_name == "svm":
        domain_model = joblib.load("models/SVM/domain_model.pkl")
        sub_domain_model = joblib.load("models/SVM/sub_domain_model.pkl")
        return domain_model, sub_domain_model
    # elif model_name == "xgboost":
    #     domain_model = joblib.load("models/XGBoost/xgboost_domain.pkl")
    #     sub_domain_model = joblib.load("models/XGBoost/xgboost_sub_domain.pkl")
    #     return domain_model, sub_domain_model
    else:
        return None, None




# Load Label Encoders


@st.cache_data
def load_label_encoders():
    return joblib.load("models/domain_label_encoder.pkl"), joblib.load("models/sub_domain_label_encoder.pkl")


# Load Vectorizer


@st.cache_data
def load_vectorizer():
    return joblib.load("models/tfidf_vectorizer.pkl")


# Load model, tokenizer, and encoders

# Streamlit UI
st.title("🔍 Multi-Label Text Classification")
st.subheader("Classify text into Domain & Sub-Domain")

# User input
description = st.text_area("Enter a description:", "")


# Dropdown menu for model selection
selected_model_name = st.selectbox(
    "Select a Model", list(model_options.keys()))
selected_model = model_options[selected_model_name]


model_domain, model_sub_domain = load_trained_model(selected_model)

vectorizer = load_vectorizer()
domain_encoder, sub_domain_encoder = load_label_encoders()

# Predict function


def predict_description(description):
    if description.strip():
        # Convert text to numerical sequence
        vectorized_text = vectorizer.transform([description])
        # Get predictions
        domain_pred, sub_domain_pred = model_domain.predict(vectorized_text), model_sub_domain.predict(vectorized_text)

        
        # Decode Predictions
        domain_label = domain_encoder.inverse_transform(
            domain_pred.astype(int))
        sub_domain_label = sub_domain_encoder.inverse_transform(
            sub_domain_pred.astype(int))
        
        print(domain_label, sub_domain_label)

        return domain_label[0], sub_domain_label[0]
    return None, None



# Button for prediction
if st.button("Classify"):
    domain_result, sub_domain_result = predict_description(description)

    if domain_result and sub_domain_result:
        st.success(f"**Predicted Domain:** {domain_result}")
        st.info(f"**Predicted Sub-Domain:** {sub_domain_result}")
    elif domain_result:
        st.success(f"**Predicted Domain:** {domain_result}")
    elif sub_domain_result:
        st.info(f"**Predicted Sub-Domain:** {sub_domain_result}")
    else:
        st.warning("Please enter a valid description!")
