"""
Movie Genre Classification using Linear Support Vector Machine
"""

# 1. Import Required Libraries

import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load Dataset

print("Loading dataset...")

df = pd.read_csv(
    "train_data.txt",
    sep=" ::: ",
    engine="python",
    header=None
)

df.columns = ["ID", "Title", "Genre", "Description"]

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)

# 3. Text Preprocessing

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    words = [word for word in words if len(word) > 2]
    return " ".join(words)

df["Description"] = df["Description"].apply(clean_text)

print("Text preprocessing completed.")

# 4. Encode Target Labels

label_encoder = LabelEncoder()
df["Genre_encoded"] = label_encoder.fit_transform(df["Genre"])

print("Genre encoding completed.")
print("Total Genres:", len(label_encoder.classes_))

# 5. Train-Test Split (Stratified)

X = df["Description"]
y = df["Genre_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-Test split completed.")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 6. TF-IDF Feature Extraction

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),      # unigram + bigram
    max_df=0.9,
    min_df=3,
    max_features=60000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF vectorization completed.")
print("Feature matrix shape:", X_train_tfidf.shape)

# 7. Train Linear SVM Model

model = LinearSVC(
    class_weight="balanced",
    max_iter=5000
)

model.fit(X_train_tfidf, y_train)

print("Model training completed using Linear SVM.")

# 8. Model Evaluation

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n===== Model Evaluation =====")
print("Accuracy:", round(accuracy, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 9. Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("Confusion Matrix - Linear SVM")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.colorbar()
plt.tight_layout()
plt.show()

print("Confusion matrix displayed.")

# 10. Interactive Prediction System

print("\n===== Movie Genre Predictor Ready =====")

while True:
    user_input = input("Enter movie plot (type 'exit' to stop): ")

    if user_input.lower() == "exit":
        print("Program terminated successfully.")
        break

    processed_input = clean_text(user_input)
    vector_input = vectorizer.transform([processed_input])

    prediction = model.predict(vector_input)
    predicted_genre = label_encoder.inverse_transform(prediction)[0]

    print("Predicted Genre:", predicted_genre)
    print("-" * 50)