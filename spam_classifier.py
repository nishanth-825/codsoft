# SMS SPAM CLASSIFICATION - LOGISTIC REGRESSION VERSION

import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# 1. DATA LOADING

df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 2. TEXT PREPROCESSING

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['message'] = df['message'].apply(clean_text)

# 3. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'],
    test_size=0.2,
    random_state=42
)

# 4. TF-IDF WITH BIGRAMS

vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    max_df=0.95,
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. LOGISTIC REGRESSION WITH HYPERPARAMETER TUNING

param_grid = {
    'C': [0.1, 1, 10]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight='balanced'),
    param_grid,
    cv=5,
    scoring='f1'
)

grid.fit(X_train_tfidf, y_train)

best_model = grid.best_estimator_

print("\nBest Regularization Parameter C:", grid.best_params_)

# 6. MODEL EVALUATION

y_pred = best_model.predict(X_test_tfidf)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 7. CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# 8. ROC CURVE

y_probs = best_model.predict_proba(X_test_tfidf)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUC = {roc_auc:.4f})")
plt.show()

print("AUC Score:", round(roc_auc, 4))

# 9. FEATURE IMPORTANCE ANALYSIS

feature_names = vectorizer.get_feature_names_out()
coefficients = best_model.coef_[0]

top_spam_indices = np.argsort(coefficients)[-10:]
top_spam_words = feature_names[top_spam_indices]

print("\nTop Words Indicating Spam:")
for word in reversed(top_spam_words):
    print(word)

# 10. INTERACTIVE DETECTOR

print("\n=========== INTERACTIVE SPAM DETECTOR ===========\n")

while True:
    user_input = input("Enter a message (type 'exit' to stop): ")

    if user_input.lower() == "exit":
        print("Program ended.")
        break

    cleaned_input = clean_text(user_input)
    user_tfidf = vectorizer.transform([cleaned_input])
    prediction = best_model.predict(user_tfidf)

    if prediction[0] == 1:
        print("Result: SPAM MESSAGE\n")
    else:
        print("Result: NOT A SPAM MESSAGE\n")
