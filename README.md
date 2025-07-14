# Step 1: Import libraries
import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import csv

# Step 2: Download stopwords (run once)
import nltk
nltk.download('stopwords')

# Step 3: Load datasets with error handling (skip bad lines)
df_fake = pd.read_csv('/content/Fake.csv', on_bad_lines='skip')
df_true = pd.read_csv('/content/True.csv', on_bad_lines='skip')

# Step 4: Add labels (0 = Fake, 1 = Real)
df_fake['label'] = 0
df_true['label'] = 1

# Step 5: Combine datasets and shuffle
df = pd.concat([df_fake, df_true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 6: Inspect columns and check for text column
print("Columns in dataset:", df.columns)
# Make sure text column exists, here assuming 'text'
if 'text' not in df.columns:
    raise ValueError("Expected a 'text' column in dataset")

# Step 7: Preprocess text data
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()  # lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # remove punctuation
    tokens = text.split()  # tokenize
    tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return ' '.join(tokens)

df['text_clean'] = df['text'].apply(clean_text)

# Step 8: Prepare features and labels
X = df['text_clean']
y = df['label']

# Step 9: Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Step 10: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Step 11: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 12: Make predictions and evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

