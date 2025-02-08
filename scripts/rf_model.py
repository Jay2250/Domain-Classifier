from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import warnings


# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load dataset
# df = pd.read_csv("../Dataset/processed_data/final_dataset.csv")
df = pd.read_csv("final_dataset.csv")
print("Dataset loaded successfully.")


# Text Preprocessing


def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word)
             for word in words if word not in stop_words]
    return " ".join(words)


# Apply preprocessing
df['clean_description'] = df['description'].astype(str).apply(preprocess_text)
print("Text preprocessing completed.")


# Label Encoding for Target Variables
domain_encoder = joblib.load('../models/domain_label_encoder.pkl')
sub_domain_encoder = joblib.load('../models/sub_domain_label_encoder.pkl')
df['domain_encoded'] = domain_encoder.transform(df['domain'])
df['sub_domain_encoded'] = sub_domain_encoder.transform(df['sub_domain'])
print("Label encoding completed.")


# TF-IDF Vectorization
tfidf = joblib.load('../models/tfidf_vectorizer.pkl')
X = tfidf.transform(df['clean_description'])
print("TF-IDF vectorization completed.")


y_domain = df['domain_encoded']
y_sub_domain = df['sub_domain_encoded']


print("Data split for domain prediction...")
# DOMAIN
# Split data
X_train, X_test, y_train_domain, y_test_domain = train_test_split(
    X, y_domain, test_size=0.2, random_state=42)
print("Data split for domain prediction completed.")

# Define parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200, 300],  # Number of trees
    'max_depth': [None, 10, 20, 30],  # Depth of trees
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split
    'min_samples_leaf': [1, 2, 4],  # Minimum samples at leaf node
    'bootstrap': [True, False]  # Use bootstrap sampling
}

# Initialize Random Forest Classifier
rf = RandomForestClassifier(random_state=42)


print("Random Forest Domain Model Training...")
# Perform Grid Search
grid_search_domain = GridSearchCV(
    rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search_domain.fit(X_train, y_train_domain)

# Best model
best_domain_model = grid_search_domain.best_estimator_

# Evaluate
y_pred_domain = best_domain_model.predict(X_test)
print("Domain Prediction Accuracy (Random Forest):",
      accuracy_score(y_test_domain, y_pred_domain))

print("Random Forest Domain Model Training Completed.")


# Save model
joblib.dump(best_domain_model, "../models/RFC/domain_rf_model.pkl")
print("Random Forest Domain Model Saved.")

# -------------------------------------------------------------------------
# SUB DOMAIN
print("Data split for sub-domain prediction...")
# Split data
X_train, X_test, y_train_sub, y_test_sub = train_test_split(
    X, y_sub_domain, test_size=0.2, random_state=42)
print("Data split for sub-domain prediction completed.")

print("Random Forest Sub-Domain Model Training...")
# Perform Grid Search
grid_search_sub_domain = GridSearchCV(
    rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search_sub_domain.fit(X_train, y_train_sub)

# Best model
best_sub_domain_model = grid_search_sub_domain.best_estimator_

# Evaluate
y_pred_sub = best_sub_domain_model.predict(X_test)
print("Sub-Domain Prediction Accuracy (Random Forest):",
      accuracy_score(y_test_sub, y_pred_sub))

print("Random Forest Sub-Domain Model Training Completed.")

# Save model
joblib.dump(best_sub_domain_model, "../models/RFC/sub_domain_rf_model.pkl")
print("Random Forest Sub-Domain Model Saved.")