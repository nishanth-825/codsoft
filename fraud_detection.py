import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample

print("Loading dataset...")

# Load dataset

df = pd.read_csv("fraudTrain.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Select useful features

features = ['amt','lat','long','city_pop','unix_time','merch_lat','merch_long']
X = df[features]
y = df['is_fraud']

print("\nSelected Features:")
print(features)

# Balance the dataset

fraud = df[df.is_fraud == 1]
legit = df[df.is_fraud == 0]

legit_downsampled = resample(
    legit,
    replace=False,
    n_samples=len(fraud),
    random_state=42
)

df_balanced = pd.concat([fraud, legit_downsampled])

X = df_balanced[features]
y = df_balanced['is_fraud']

print("\nBalanced Dataset Shape:", df_balanced.shape)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain-Test split completed")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Train Random Forest Model

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed!")

# Model Evaluation

y_pred = model.predict(X_test)

print("\nModel Evaluation Results")

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance

importance = pd.DataFrame({
    'Feature':features,
    'Importance':model.feature_importances_
}).sort_values(by='Importance',ascending=False)

print("\nFeature Importance:\n")
print(importance)

print("\nFraud Detection Model Completed Successfully.")

# Manual Fraud Check

while True:

    check = input("\nDo you want to test a transaction? (yes/no): ")

    if check.lower() != "yes":
        break

    print("\nEnter transaction details:")

    amt = float(input("Transaction Amount: "))
    lat = float(input("Customer Latitude: "))
    long = float(input("Customer Longitude: "))
    city_pop = int(input("City Population: "))
    unix_time = int(input("Unix Time: "))
    merch_lat = float(input("Merchant Latitude: "))
    merch_long = float(input("Merchant Longitude: "))

    input_data = pd.DataFrame(
        [[amt,lat,long,city_pop,unix_time,merch_lat,merch_long]],
        columns=features
    )

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        print("\n🚨 Fraudulent Transaction")
    else:
        print("\n✓ Legitimate Transaction")