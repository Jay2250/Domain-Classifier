import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the trained model


@st.cache_resource
def load_trained_model():
    return load_model("models/neural_network/multi_label_model.h5")

# Load Tokenizer


@st.cache_data
def load_tokenizer():
    with open("models/neural_network/tokenizer.pkl", "rb") as f:
        return pickle.load(f)

# Load Label Encoders


@st.cache_data
def load_label_encoders():
    with open("models/neural_network/domain_encoder.pkl", "rb") as f:
        domain_encoder = pickle.load(f)
    with open("models/neural_network/sub_domain_encoder.pkl", "rb") as f:
        sub_domain_encoder = pickle.load(f)
    return domain_encoder, sub_domain_encoder


# Load model, tokenizer, and encoders
model = load_trained_model()
tokenizer = load_tokenizer()
domain_encoder, sub_domain_encoder = load_label_encoders()

# Streamlit UI
st.title("🔍 Multi-Label Text Classification")
st.subheader("Classify text into Domain & Sub-Domain")

# User input
description = st.text_area("Enter a description:", "")

# Predict function


def predict_description(description):
    if description.strip():
        # Convert text to numerical sequence
        seq = tokenizer.texts_to_sequences([description])
        # Ensure it matches the model input
        padded_seq = pad_sequences(seq, padding='post', maxlen=100)

        # Get predictions
        domain_pred, sub_domain_pred = model.predict(padded_seq)

        # Decode Predictions
        domain_label = domain_encoder.inverse_transform(
            [np.argmax(domain_pred)])
        sub_domain_label = sub_domain_encoder.inverse_transform(
            [np.argmax(sub_domain_pred)])

        return domain_label[0], sub_domain_label[0]
    return None, None



# Button for prediction
if st.button("Classify"):
    domain_result, sub_domain_result = predict_description(description)

    if domain_result and sub_domain_result:
        st.success(f"**Predicted Domain:** {domain_result}")
        st.info(f"**Predicted Sub-Domain:** {sub_domain_result}")
    else:
        st.warning("Please enter a valid description!")
