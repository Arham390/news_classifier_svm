import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_json("News_Category_Dataset_v3.json", lines=True)

# Keep necessary columns
df = df[['headline', 'short_description', 'category']]
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Preprocess text
def pretext(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['headline'] = df['headline'].apply(pretext)
df['short_description'] = df['short_description'].apply(pretext)

# Combine text fields
df['text'] = df['headline'] + ' ' + df['short_description']

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['category'])

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])

# Labels
y = df['label'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Show class distribution (optional)
print("\nClass distribution:")
print(df['label'].value_counts())

# === Model selection ===
print("\nWhich model would you like to train?")
print("1 - Support Vector Machine (SVM)")
print("2 - Random Forest")

choice = input("Enter your choice (1 or 2): ")

if choice == "1":
    print("\nTraining Linear SVM with class_weight='balanced'...")
    model = LinearSVC(class_weight='balanced')
elif choice == "2":
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    raise ValueError("Invalid choice. Enter 1 or 2.")

# Train the model
model.fit(X_train, y_train)
print("\nModel trained successfully!")

# Predict and evaluate
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
