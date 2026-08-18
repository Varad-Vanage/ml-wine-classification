# -*- coding: utf-8 -*-
"""
ML Assignment 2 - Model Training & Evaluation Pipeline
Dataset: UCI Red Wine Quality
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# 1. Create directory to store trained model files
os.makedirs('model', exist_ok=True)
print("Directory 'model/' checked/created successfully.")

# 2. Fetch Red Wine Quality dataset directly from UCI Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# 3. Convert quality target into Binary Classification problem: 1 for Quality >= 6, 0 for Quality < 6
df['target'] = (df['quality'] >= 6).astype(int)
df = df.drop(columns=['quality'])

print(f"Dataset Shape: {df.shape}")
print("Target Class Distribution:")
print(df['target'].value_counts())

# 4. Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# 5. 80-20 Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the fitted scaler artifact
joblib.dump(scaler, 'model/scaler.pkl')

# 7. Save Scaled Test Data to CSV for Streamlit testing (saved in root project folder)
test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
test_df['target'] = y_test.values
test_df.to_csv('test_data.csv', index=False)

print("Scaler ('model/scaler.pkl') and test set ('test_data.csv') saved successfully!")

# 8. Define the 5 classification algorithms
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

# 9. Train, Save, and Evaluate All Models on Scaled Features
for name, model in models.items():
    # Fit model on scaled training features
    model.fit(X_train_scaled, y_train)
    
    # Predict on scaled test features
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    # Save trained model artifact inside model/ directory
    model_filename = f"model/{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, model_filename)

    # Calculate evaluation metrics
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)

    results.append({
        'ML Model Name': name,
        'Accuracy': round(acc, 4),
        'AUC': round(auc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4)
    })

print("All 5 models trained and saved to 'model/' directory successfully!")

# 10. Display Summary Table
results_df = pd.DataFrame(results)

print("\n=== METRICS SUMMARY TABLE ===")
print(results_df.to_string(index=False))

# Print raw markdown table for README.md
print("\n--- COPY MARKDOWN TABLE BELOW FOR YOUR README.MD --- \n")
print(results_df.to_markdown(index=False))