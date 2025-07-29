import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack

# Load data
df = pd.read_json("News_Category_Dataset_v3.json", lines=True)

# Preprocessing
df = df[['headline', 'short_description', 'category']]
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

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

# Split data
Xh_train, Xh_test, Xd_train, Xd_test, y_train, y_test = train_test_split(
    df['headline'], df['short_description'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
tfidf_headline = TfidfVectorizer(max_features=5000)
tfidf_desc = TfidfVectorizer(max_features=5000)

Xh_train_vec = tfidf_headline.fit_transform(Xh_train)
Xh_test_vec = tfidf_headline.transform(Xh_test)

Xd_train_vec = tfidf_desc.fit_transform(Xd_train)
Xd_test_vec = tfidf_desc.transform(Xd_test)

# Combine headline + short description features
X_train_combined = hstack([Xh_train_vec, Xd_train_vec])
X_test_combined = hstack([Xh_test_vec, Xd_test_vec])

# === USER CHOICE ===
print("Which model would you like to train?")
print("1 - Support Vector Machine (SVM)")
print("2 - Random Forest")
choice = input("Enter 1 or 2: ").strip()

if choice == "1":
    print("\nTraining Linear SVM...")
    model = LinearSVC()
elif choice == "2":
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    raise ValueError("Invalid choice. Enter 1 or 2.")

# Train selected model
model.fit(X_train_combined, y_train)

# Predict & evaluate
y_pred = model.predict(X_test_combined)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
