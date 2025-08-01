import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

# Load dataset
df = pd.read_json("News_Category_Dataset_v3.json", lines=True)
df = df[['headline', 'short_description', 'category']].dropna().drop_duplicates()

# Preprocess text
def pretext(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['headline'] = df['headline'].apply(pretext)
df['short_description'] = df['short_description'].apply(pretext)

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['category'])

# Combine text fields
df['combined_text'] = df['headline'] + " " + df['short_description']

# === Vectorization Choice ===
print("Select vectorization method:")
print("1 - TF-IDF")
print("2 - Bag of Words (BoW)")

vec_choice = input("Enter your choice (1 or 2): ")

if vec_choice == "1":
    print("\nUsing TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000)
elif vec_choice == "2":
    print("\nUsing Bag of Words (CountVectorizer)...")
    vectorizer = CountVectorizer(max_features=5000)
else:
    raise ValueError("Invalid choice. Enter 1 or 2.")

X = vectorizer.fit_transform(df['combined_text']).toarray()
y = df['label'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Model Choice ===
print("\nSelect classifier:")
print("1 - Support Vector Machine (SVM)")
print("2 - Random Forest")

model_choice = input("Enter your choice (1 or 2): ")

if model_choice == "1":
    print("\nTraining Linear SVM with class_weight='balanced'...")
    model = LinearSVC(class_weight='balanced')
elif model_choice == "2":
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    raise ValueError("Invalid choice. Enter 1 or 2.")

# Train the model
model.fit(X_train, y_train)
print("\nModel trained successfully!")

# Evaluation
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
