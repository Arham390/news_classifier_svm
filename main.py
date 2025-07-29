import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# Load the JSON file into a DataFrame
df = pd.read_json("News_Category_Dataset_v3.json", lines=True)

# Display the first few rows
print(df.head())

print(df.columns)
print(df.shape)
df.sample(5)

# Select relevant columns
df = df[['headline', 'short_description', 'category']]

# Drop missing and duplicate values
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Preprocess text
def pretext(text):
    # Remove HTML tags and clean text
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

# Split data into training and testing sets
Xh_train, Xh_test, Xd_train, Xd_test, y_train, y_test = train_test_split(
    df['headline'], df['short_description'], df['label'], test_size=0.2, random_state=42
)

# Vectorize text data
tfidf_headline = TfidfVectorizer(max_features=5000)
Xh_train_vec = tfidf_headline.fit_transform(Xh_train)
Xh_test_vec = tfidf_headline.transform(Xh_test)

tfidf_desc = TfidfVectorizer(max_features=5000)
Xd_train_vec = tfidf_desc.fit_transform(Xd_train)
Xd_test_vec = tfidf_desc.transform(Xd_test)

# Combine headline and short_description features
X_train_combined = hstack([Xh_train_vec, Xd_train_vec])
X_test_combined = hstack([Xh_test_vec, Xd_test_vec])

# Train SVM model
svm = LinearSVC()
svm.fit(X_train_combined, y_train)

# Evaluate the model
y_pred = svm.predict(X_test_combined)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
