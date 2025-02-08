from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import warnings

from sklearn.svm import SVC
warnings.filterwarnings("ignore")

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
domain_encoder = LabelEncoder()
sub_domain_encoder = LabelEncoder()
df['domain_encoded'] = domain_encoder.fit_transform(df['domain'])
df['sub_domain_encoded'] = sub_domain_encoder.fit_transform(df['sub_domain'])
print("Label encoding completed.")


# saving label encoder
joblib.dump(domain_encoder, "../models/domain_label_encoder.pkl")
joblib.dump(sub_domain_encoder, "../models/sub_domain_label_encoder.pkl")
print("Label encoder saved successfully.)")

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(df['clean_description'])
print("TF-IDF vectorization completed.")

# Save vectorizer
joblib.dump(tfidf, "../models/tfidf_vectorizer.pkl")
print("Vectorizer saved successfully.)")

y_domain = df['domain_encoded']
y_sub_domain = df['sub_domain_encoded']



print("Data split for domain prediction...")
# DOMAIN
# Split data for domain prediction
X_train, X_test, y_train_domain, y_test_domain = train_test_split(
    X, y_domain, test_size=0.2, random_state=42)
print("Data split for domain prediction completed.")

print("Logistic Regression Domain Model Training...")
# LOGISTIC REGRESSION
# Train Logistic Regression for domain prediction
domain_model_lr = LogisticRegression(max_iter=500, solver='lbfgs')
domain_model_lr.fit(X_train, y_train_domain)
joblib.dump(domain_model_lr, "../models/LogisticRegression/domain_model.pkl")


print("SVM Domain Model Training...)")
# Train SVM for domain prediction
domain_model_svm = SVC(kernel='linear', probability=True)
domain_model_svm.fit(X_train, y_train_domain)
joblib.dump(domain_model_svm, "../models/SVM/domain_model.pkl")


print("Evaluation")
# Evaluate
y_pred_domain_lr = domain_model_lr.predict(X_test)
print("Domain Prediction Accuracy For Logistic Regression:",
      accuracy_score(y_test_domain, y_pred_domain_lr))

# Evaluate
y_pred_domain = domain_model_svm.predict(X_test)
print("Domain Prediction Accuracy (SVM):",
      accuracy_score(y_test_domain, y_pred_domain))


# ---------------------------------------------------------------------------------------

print("Data split for sub domain prediction...")
# SUB DOMAIN
# Split data for sub-domain prediction
X_train, X_test, y_train_sub, y_test_sub = train_test_split(
    X, y_sub_domain, test_size=0.2, random_state=42)
print("Data split for sub domain prediction completed.")


print("Logistic Regression Sub Domain Model Training...)")
# LOGISTIC REGRESSION
# Train Logistic Regression for sub-domain prediction
sub_domain_model_lr = LogisticRegression(max_iter=500, solver='lbfgs')
sub_domain_model_lr.fit(X_train, y_train_sub)
joblib.dump(sub_domain_model_lr, "../models/LogisticRegression/sub_domain_model.pkl")


print("SVM Sub Domain Model Training...")
# Train SVM for sub-domain prediction
sub_domain_model_svm = SVC(kernel='linear', probability=True)
sub_domain_model_svm.fit(X_train, y_train_sub)
joblib.dump(sub_domain_model_svm, "../models/SVM/sub_domain_model.pkl")


print("Evaluation")
# Evaluate
y_pred_sub = sub_domain_model_lr.predict(X_test)
print("Sub-Domain Prediction Accuracy For Logistic Regression:",
      accuracy_score(y_test_sub, y_pred_sub))

# Evaluate
y_pred_sub = sub_domain_model_svm.predict(X_test)
print("Sub-Domain Prediction Accuracy (SVM):",
      accuracy_score(y_test_sub, y_pred_sub))





print(" Model saved successfully!")
