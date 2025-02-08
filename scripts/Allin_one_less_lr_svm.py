import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, hamming_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# Streamlit UI Setup
st.title("🚀 Multi-Label Text Classification")
st.subheader("Classify text into Domains and Sub-Domains")
st.info("Processing data and training models, please wait...")

# Load dataset
st.write("### Loading Dataset...")
df = pd.read_csv("final_dataset.csv")
st.success("Dataset Loaded Successfully!")

# Text Preprocessing
st.write("### Preprocessing Text...")
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df['cleaned_text'] = df['description'].apply(preprocess_text)

# Save Preprocessing Model
with open("text_preprocessing.pkl", "wb") as f:
    pickle.dump(preprocess_text, f)
st.success("Text Preprocessing Completed!")

# TF-IDF Vectorization
st.write("### Applying TF-IDF Vectorization...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['cleaned_text'])

# Save TF-IDF Vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)
st.success("TF-IDF Vectorization Done!")

# Label Encoding
st.write("### Encoding Labels...")
domain_encoder = LabelEncoder()
sub_domain_encoder = LabelEncoder()
df['domain_encoded'] = domain_encoder.fit_transform(df['domain'])
df['sub_domain_encoded'] = sub_domain_encoder.fit_transform(df['sub_domain'])

# Save Label Encoders
with open("domain_label_encoder.pkl", "wb") as f:
    pickle.dump(domain_encoder, f)
with open("sub_domain_label_encoder.pkl", "wb") as f:
    pickle.dump(sub_domain_encoder, f)
st.success("Label Encoding Completed!")

# Train-Test Split
st.write("### Splitting Data into Training and Testing Sets...")
X_train, X_test, y_train_domain, y_test_domain = train_test_split(X, df['domain_encoded'], test_size=0.2, random_state=42)
X_train, X_test, y_train_sub, y_test_sub = train_test_split(X, df['sub_domain_encoded'], test_size=0.2, random_state=42)

# Save Train-Test Data
with open("train_test_data.pkl", "wb") as f:
    pickle.dump((X_train, X_test, y_train_domain, y_test_domain, y_train_sub, y_test_sub), f)
st.success("Data Split Successful!")

# Model Training & Evaluation
st.write("### Training Models...")

# Create a list to store training history
training_history = []

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name, label):
    with st.spinner(f"Training {model_name} for {label}..."):
        time.sleep(1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute Evaluation Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        hamming = hamming_loss(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Save model
        model_filename = f"{label}_{model_name}.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)

        # Store training results
        training_history.append({
            "Model": model_name,
            "Label": label,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Hamming Loss": hamming
        })

        return model, report, model_filename

models = {
    "Logistic": LogisticRegression(),
    "SVM": SVC(),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "NaiveBayes": MultinomialNB(),
    "KNN": KNeighborsClassifier()
}

eval_reports = {}
saved_models = {}

for name, model in models.items():
    domain_model, report_domain, domain_model_filename = train_and_evaluate(model, X_train, y_train_domain, X_test, y_test_domain, name, "domain")
    sub_model, report_sub, sub_model_filename = train_and_evaluate(model, X_train, y_train_sub, X_test, y_test_sub, name, "sub_domain")
    
    eval_reports[name] = {"Domain": report_domain, "Sub-Domain": report_sub}
    saved_models[name] = {"Domain": domain_model_filename, "Sub-Domain": sub_model_filename}

# Save training history to CSV
training_history_df = pd.DataFrame(training_history)
training_history_df.to_csv("training_history.csv", index=False)

# Save evaluation reports
with open("model_evaluation.pkl", "wb") as f:
    pickle.dump(eval_reports, f)

st.success("All Models Trained and Saved Successfully!")
st.success("Training history saved as 'training_history.csv'!")

# Streamlit Prediction Interface
st.write("### Make Predictions")
input_text = st.text_area("Enter text for classification:")
if st.button("Predict"):
    with st.spinner("Processing Input..."):
        time.sleep(1)
        processed_text = preprocess_text(input_text)
        vectorized_text = tfidf_vectorizer.transform([processed_text])

        predictions = {}
        for name, paths in saved_models.items():
            domain_model = pickle.load(open(paths["Domain"], "rb"))
            sub_model = pickle.load(open(paths["Sub-Domain"], "rb"))

            domain_pred = domain_encoder.inverse_transform(domain_model.predict(vectorized_text))[0]
            sub_pred = sub_domain_encoder.inverse_transform(sub_model.predict(vectorized_text))[0]

            predictions[name] = {"Domain": domain_pred, "Sub-Domain": sub_pred}

        st.success("Prediction Completed!")
        st.write("### Predictions")
        st.json(predictions)

# Display Model Performance
st.write("### Model Performance")
for model_name, reports in eval_reports.items():
    st.write(f"#### {model_name} Model")
    st.write("**Domain Classification:**")
    st.json(reports["Domain"])
    st.write("**Sub-Domain Classification:**")
    st.json(reports["Sub-Domain"])
